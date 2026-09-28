"""Unit tests for `analysis/build_video_spx_schedules.py` (video SPX, stage V1).

The schedule set is frozen once and then executed 400-odd times, so what has to
hold is structural rather than numerical: every row spends exactly its budget,
no row caches a step the payload it will be paired with must run full, the
Hamming ladder really sits where it says it does, and a second run of the
builder reproduces the committed files byte for byte.

Packaged schedule definitions are read-only. Rebuilding gate-derived and
geometry-derived schedules requires experiment outputs, so those integration
tests are skipped until the required input files have been generated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis import build_video_spx_schedules as B

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "resources/video_spx_schedules"
EXPERIMENT_INPUTS = (*B.GATE_PATH_COUNTS.values(), *B.DENSITY_FORM.values())
MISSING_EXPERIMENT_INPUTS = [str(path.relative_to(REPO)) for path in EXPERIMENT_INPUTS
                             if not path.is_file()]
requires_experiment_outputs = pytest.mark.skipif(
    bool(MISSING_EXPERIMENT_INPUTS),
    reason="Regenerate gate-census and trajectory-density outputs first: "
           + ", ".join(MISSING_EXPERIMENT_INPUTS),
)


@pytest.fixture(scope="module")
def rows() -> list[B.Row]:
    """Load only the lightweight schedule definitions included in this repository."""
    result = []
    for backbone in B.BACKBONES:
        for path in sorted((ROOT / backbone).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            result.append(B.Row(
                backbone=backbone, row=payload["row"], group=payload["group"],
                budget=payload["nominal_k"], bits=payload["bits"],
                source=payload["source"], off_budget=payload["off_budget"],
                jvp_spans={int(k): int(v) for k, v in payload["jvp_spans"].items()},
                provenance=payload["provenance"],
                payload_columns=tuple(payload["payload_columns"]),
            ))
    assert result, "Packaged schedule definitions are missing"
    return result


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, str]]:
    with (ROOT / "manifest.tsv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_fourteen_rows_per_partition(rows: list[B.Row]) -> None:
    """Plan section 2 plus owner decision 3: 14 rows per (backbone, K), the
    three off-budget rows replacing the gate rows that have no exactly-K path
    rather than adding to them. Supplement S1 adds the first-step-preserving
    ladder on top, up to three rungs more."""
    ladder_f = {f"ham{distance}{B.LADDER_F_SUFFIX}" for distance, _ in B.LADDER_F_DRAWS}
    per_partition: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        per_partition.setdefault((row.backbone, row.budget), []).append(row.row)
    assert len(per_partition) == len(B.BACKBONES) * len(B.KS)
    for key, names in per_partition.items():
        base = [name for name in names if name not in ladder_f]
        assert len(base) == 14, (key, sorted(base))
        assert len(set(names)) == len(names)
    assert sum(1 for row in rows if row.off_budget) == 3


def test_popcount_matches_budget(rows: list[B.Row]) -> None:
    for row in rows:
        if row.off_budget:
            # the row runs at the gate's own realized count, which is recorded
            assert row.popcount == row.provenance["realized_cache_count"]
            assert row.popcount != row.budget
        else:
            assert row.popcount == row.budget, row.schedule_id


def test_boundary_steps_are_full(rows: list[B.Row]) -> None:
    for row in rows:
        assert row.bits[0] == "0", row.schedule_id
        assert row.bits[-1] == "0", row.schedule_id


def test_feasibility_matches_the_forbidden_sets(rows: list[B.Row]) -> None:
    for row in rows:
        cached = set(row.cache_steps)
        flags = B.feasibility(row.backbone, row.bits)
        for payload, forbidden in B.PAYLOAD_FORBIDDEN[row.backbone].items():
            assert flags[payload] == (not cached & forbidden)


def test_only_gate_rows_are_ever_infeasible(rows: list[B.Row]) -> None:
    """A control or a frozen table that could not run a payload would be a
    construction error; a gate path that cannot is data."""
    for row in rows:
        if all(B.feasibility(row.backbone, row.bits).values()):
            continue
        assert row.group == "gate", row.schedule_id


def test_controls_keep_the_shared_warmup_full(rows: list[B.Row]) -> None:
    for row in rows:
        if row.group != "control":
            continue
        assert not set(row.cache_steps) & set(B.CONTROL_FORCED_FULL), row.schedule_id


# ---------------------------------------------------------------------------
# the Hamming ladder
# ---------------------------------------------------------------------------


def test_ladder_distance_and_gap_cap(rows: list[B.Row]) -> None:
    anchors = {(row.backbone, row.budget): row.bits
               for row in rows if row.row == "meancache"}
    for row in rows:
        if not row.row.startswith("ham"):
            continue
        target = int(row.row[3:].rstrip(B.LADDER_F_SUFFIX))
        anchor = anchors[(row.backbone, row.budget)]
        assert B.hamming(row.bits, anchor) == target, row.schedule_id
        assert row.popcount == anchor.count("1")
        assert max(B.full_step_gaps(row.bits)) <= B.MAX_GAP, row.schedule_id


def test_ladder_leaves_the_warmup_alone(rows: list[B.Row]) -> None:
    anchors = {(row.backbone, row.budget): row.bits
               for row in rows if row.row == "meancache"}
    for row in rows:
        if not row.row.startswith("ham"):
            continue
        anchor = anchors[(row.backbone, row.budget)]
        assert row.bits[: B.SWAP_LOW] == anchor[: B.SWAP_LOW], row.schedule_id
        assert row.bits[B.SWAP_HIGH + 1:] == anchor[B.SWAP_HIGH + 1:], row.schedule_id


def test_ladder_rejects_an_odd_distance() -> None:
    with pytest.raises(SystemExit):
        B.hamming_ladder_bits("0" * 50, 3, "hunyuan_video", 29, 11)


# ---------------------------------------------------------------------------
# the DP control
# ---------------------------------------------------------------------------


def test_force_full_makes_the_dp_visit_the_step() -> None:
    """`force_full` is what turns the plan's "forced full {0,1,2,49}" into a
    constraint the solver -- which only forces the first and the last step --
    actually obeys."""
    rho2 = [1.0] * B.NUM_STEPS
    sigmas = [1.0 - index / B.NUM_STEPS for index in range(B.NUM_STEPS)]
    bits, _cost = B.dp_rho2_bits(rho2, sigmas, 29)
    assert not set(B.steps_of(bits)) & set(B.CONTROL_FORCED_FULL)
    assert bits.count("1") == 29


def test_dp_rows_respect_the_gap_cap(rows: list[B.Row]) -> None:
    for row in rows:
        if row.row != "dp_rho2":
            continue
        assert max(B.full_step_gaps(row.bits)) <= B.MAX_GAP, row.schedule_id


# ---------------------------------------------------------------------------
# provenance and determinism
# ---------------------------------------------------------------------------


@requires_experiment_outputs
def test_gate_rows_carry_their_census_counts() -> None:
    rows, _, _ = B.build_all()
    for row in rows:
        if row.group != "gate":
            continue
        assert row.provenance["count"] > 0
        assert 0.0 < row.provenance["share_of_exact_k_pool"] <= 1.0
        assert row.provenance["n_tied_at_top"] >= 1
        assert set(row.provenance["per_dataset_counts"]) == set(B.DATASETS)


def test_random_and_ladder_rows_carry_their_rng_recipe(rows: list[B.Row]) -> None:
    for row in rows:
        if row.row.startswith("rand_") or row.row.startswith("ham"):
            assert "default_rng([" in row.provenance["rng"], row.schedule_id


@requires_experiment_outputs
def test_build_is_deterministic() -> None:
    first, _, first_skipped = B.build_all()
    second, _, second_skipped = B.build_all()
    assert [row.bits for row in first] == [row.bits for row in second]
    assert first_skipped == second_skipped


@requires_experiment_outputs
def test_fresh_build_matches_packaged_schedule_values(tmp_path: Path) -> None:
    """Compare execution inputs; publication removes result-derived provenance."""
    rows, unavailable, skipped = B.build_all()
    B.write_outputs(rows, unavailable, tmp_path, skipped)
    fields = ("bits", "cache_steps", "cache_count", "nominal_k", "off_budget",
              "jvp_spans", "jvp_spans_are_per_edge", "jvp_span_global", "payload_columns")
    for row in rows:
        fresh = tmp_path / row.backbone / f"{row.schedule_id}.json"
        stored = ROOT / row.backbone / f"{row.schedule_id}.json"
        new = json.loads(fresh.read_text(encoding="utf-8"))
        old = json.loads(stored.read_text(encoding="utf-8"))
        assert {k: new[k] for k in fields} == {k: old[k] for k in fields}, row.schedule_id
        if row.row not in B.SPAN_VARIANT_ROWS:
            continue
        name = f"{B.SPAN_VARIANT_DIR}/{row.schedule_id}.json"
        new = json.loads((tmp_path / row.backbone / name).read_text(encoding="utf-8"))
        old = json.loads((ROOT / row.backbone / name).read_text(encoding="utf-8"))
        assert {k: new[k] for k in fields} == {k: old[k] for k in fields}, name


# ---------------------------------------------------------------------------
# the committed artifacts
# ---------------------------------------------------------------------------


def test_manifest_covers_every_schedule_json(manifest: list[dict[str, str]]) -> None:
    ids = {row["schedule_id"] for row in manifest}
    on_disk = {path.stem for backbone in B.BACKBONES
               for path in (ROOT / backbone).glob("*.json")}
    assert ids == on_disk


def test_schedule_json_agrees_with_its_bits(manifest: list[dict[str, str]]) -> None:
    for row in manifest:
        payload = json.loads(
            (ROOT / row["backbone"] / f"{row['schedule_id']}.json").read_text(encoding="utf-8"))
        assert payload["bits"] == row["bits"]
        assert payload["cache_steps"] == B.steps_of(row["bits"])
        assert payload["cache_count"] == int(row["cache_count"])
        assert payload["schema"] == B.SCHEMA


def test_only_meancache_carries_per_edge_spans(manifest: list[dict[str, str]]) -> None:
    """Plan section 3: the offline search solved per-edge spans for MeanCache's
    own table only; every other row runs the global fallback span."""
    for row in manifest:
        payload = json.loads(
            (ROOT / row["backbone"] / f"{row['schedule_id']}.json").read_text(encoding="utf-8"))
        if row["row"] == "meancache":
            assert payload["jvp_spans_are_per_edge"]
            assert sorted(int(step) for step in payload["jvp_spans"]) == payload["cache_steps"]
        else:
            assert payload["jvp_spans"] == {}
            assert payload["jvp_span_global"] == B.GLOBAL_JVP_SPAN


def test_off_budget_rows_only_take_the_reuse_column(manifest: list[dict[str, str]]) -> None:
    for row in manifest:
        payload = json.loads(
            (ROOT / row["backbone"] / f"{row['schedule_id']}.json").read_text(encoding="utf-8"))
        if int(row["off_budget"]) or row["row"].endswith(B.LADDER_F_SUFFIX):
            assert payload["payload_columns"] == ["reuse"]
        else:
            assert payload["payload_columns"] == list(B.PAYLOADS)


# ---------------------------------------------------------------------------
# supplement S1: the first-step-preserving ladder
# ---------------------------------------------------------------------------


def test_f_ladder_keeps_the_anchor_first_cache_step(rows: list[B.Row]) -> None:
    """The point of the rung: `first_cache_step` is MeanCache's, so a paired
    difference against MeanCache measures Hamming distance and not "the caching
    starts earlier" (audit finding 3)."""
    anchors = {(row.backbone, row.budget): row.bits
               for row in rows if row.row == "meancache"}
    seen = 0
    for row in rows:
        if not row.row.endswith(B.LADDER_F_SUFFIX) or not row.row.startswith("ham"):
            continue
        seen += 1
        anchor = anchors[(row.backbone, row.budget)]
        assert min(row.cache_steps) == min(B.steps_of(anchor)), row.schedule_id
        assert row.provenance["first_step_preserving"] is True
        low = min(B.steps_of(anchor)) + 1
        assert row.provenance["swap_window"] == [low, B.SWAP_HIGH]
        # nothing below the window moved
        assert row.bits[:low] == anchor[:low], row.schedule_id
    assert seen == 16


def test_f_ladder_runs_the_reuse_column_only(rows: list[B.Row]) -> None:
    for row in rows:
        if row.row.startswith("ham") and row.row.endswith(B.LADDER_F_SUFFIX):
            assert B.columns_of(row) == ("reuse",), row.schedule_id
        elif not row.off_budget:
            assert B.columns_of(row) == tuple(B.PAYLOADS), row.schedule_id


def test_k41_has_no_ham8f_rung(rows: list[B.Row]) -> None:
    """Refused, not repaired: the K41 MeanCache tables leave only three full
    steps above their first cached step and `ham8f` needs four to swap in."""
    anchors = [row for row in rows if row.row == "meancache" and row.budget == 41]
    assert {row.backbone for row in anchors} == set(B.BACKBONES)
    for row in anchors:
        bits, recipe = B.hamming_ladder_bits(
            row.bits, 8, row.backbone, 41, 23,
            swap_low=min(row.cache_steps) + 1, strict=False,
        )
        assert bits is None
        assert recipe["swappable_full"] < recipe["swaps"]


def test_ladder_helper_reports_instead_of_dying_when_asked() -> None:
    anchor = "0" * 40 + "1" * 9 + "0"
    bits, recipe = B.hamming_ladder_bits(anchor, 8, "hunyuan_video", 29, 21,
                                         swap_low=41, strict=False)
    assert bits is None and "unavailable" in recipe
    with pytest.raises(SystemExit):
        B.hamming_ladder_bits(anchor, 8, "hunyuan_video", 29, 21, swap_low=41)


# ---------------------------------------------------------------------------
# supplement S2: the global-span transport
# ---------------------------------------------------------------------------


def test_span_variant_is_the_same_table_without_spans(rows: list[B.Row]) -> None:
    seen = 0
    for row in rows:
        if row.row not in B.SPAN_VARIANT_ROWS:
            continue
        seen += 1
        stored = json.loads(
            (ROOT / row.backbone / B.SPAN_VARIANT_DIR / f"{row.schedule_id}.json")
            .read_text(encoding="utf-8"))
        base = json.loads(
            (ROOT / row.backbone / f"{row.schedule_id}.json").read_text(encoding="utf-8"))
        assert stored["bits"] == base["bits"]
        assert stored["cache_steps"] == base["cache_steps"]
        assert stored["jvp_spans"] == {}
        assert stored["jvp_spans_are_per_edge"] is False
        assert stored["jvp_span_global"] == B.GLOBAL_JVP_SPAN
        assert stored["payload_columns"] == [B.SPAN_VARIANT_PAYLOAD]
        assert stored["variant_of"] == row.schedule_id
        # the searched spans are recorded, so what was dropped is on the record
        assert stored["provenance"]["jvp_spans_stripped"] == base["jvp_spans"]
        assert base["jvp_spans"] != {}
    assert seen == len(B.BACKBONES) * len(B.KS)


def test_span_variant_files_do_not_shadow_a_row(manifest: list[dict[str, str]]) -> None:
    """The variant is a transport file, not a fifteenth row: it must not appear
    in the manifest and must not sit beside the rows."""
    ids = {row["schedule_id"] for row in manifest}
    for backbone in B.BACKBONES:
        variant_dir = ROOT / backbone / B.SPAN_VARIANT_DIR
        names = {path.stem for path in variant_dir.glob("*.json")}
        assert names == {f"meancache_K{k}" for k in B.KS}
        assert names <= ids
        # a schedule row is a `<backbone>/*.json`; the variants sit one level
        # down, so nothing that enumerates rows can pick them up
        assert not any(path.parent.name == B.SPAN_VARIANT_DIR
                       for path in (ROOT / backbone).glob("*.json"))
