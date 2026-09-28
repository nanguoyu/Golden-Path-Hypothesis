"""Unit tests for analysis/video_trajectory/cached_vs_reference.py (P6, plan 3.9.1).

Everything is synthetic and CPU-only. The store is built with the SAME writer
the probes use (`analysis.trajectory_math.trajectory_metrics`) so the arrays are
genuine and self-consistent, and the cached paths are PLANTED so each check has
a known right answer:

* a cached run's states 0..k0 are copied bit-for-bit from its reference, so the
  local quantities' prefixes must be exactly equal and `d_perp` (which is
  measured from each run's own chord) must NOT be;
* the cache schedules are chosen so the realised step counts and current
  documented-K exceptions are exercised;
* the event-alignment offsets are counted by hand from the bit strings.

P1 floor files are synthetic fixtures written under pytest's temporary directory.
"""

import json

import numpy as np
import pytest

from analysis.trajectory_math import trajectory_metrics
from analysis.video_trajectory import cached_vs_reference as CVR

D = 48
N_STEPS = CVR.NUM_STEPS
N_STATES = CVR.N_STATES
DATASETS = ("penguin599", "vbench944")
BASE_SEED = 54
N_PROMPTS = 4

SIGMAS = [float(v) for v in np.linspace(1.0, 0.0, N_STATES)]

# schedules with the shapes the checks need: a frozen table (identical on every
# row), and gates whose realised counts sit on / beside the documented numbers
FIXED_BITS = {29: None, 37: None, 41: None}


def bits_with(k: int, *, first: int = 3) -> str:
    """A 50-bit schedule caching `k` steps, none of them step 0 or the last."""
    steps = list(range(first, first + k))
    assert steps[-1] < N_STEPS - 1, (k, first)
    return "".join("1" if n in set(steps) else "0" for n in range(N_STEPS))


for _k in FIXED_BITS:
    FIXED_BITS[_k] = bits_with(_k)


def reference_path(rng: np.random.Generator) -> np.ndarray:
    """A smooth, curved path — nothing here depends on its shape."""
    t = np.linspace(0.0, 1.0, N_STATES)[:, None]
    basis = np.linalg.qr(rng.standard_normal((D, 3)))[0].T
    Z = (20.0 * t * basis[0] + 3.0 * np.sin(np.pi * t) * basis[1]
         + 0.7 * np.sin(2.4 * np.pi * t) * basis[2])
    return Z + 0.02 * rng.standard_normal((N_STATES, D))


def cached_path(Z_ref: np.ndarray, bits: str) -> np.ndarray:
    """States 0..k0 copied EXACTLY; from k0 on, a cached step reuses the previous
    displacement (zero-order residual reuse) and a full step takes the
    reference's own displacement."""
    k0 = bits.index("1")
    Z = Z_ref.copy()
    prev = Z_ref[k0] - Z_ref[k0 - 1] if k0 > 0 else Z_ref[1] - Z_ref[0]
    for n in range(k0, N_STEPS):
        step = prev if bits[n] == "1" else (Z_ref[n + 1] - Z_ref[n])
        Z[n + 1] = Z[n] + step
        prev = step
    return Z


def row(Z: np.ndarray, *, kind: str, dataset: str, prompt_idx: int, source_dir: str,
        mode: str = "original", mode_raw: str | None = None, budget=None,
        bits: str | None = None, ref_source_dir: str | None = None,
        ref_z_T_match=None, n_cached: int | None = None) -> dict:
    rec = trajectory_metrics(Z, SIGMAS)
    rec.update({
        "schema": "hunyuan_video.trajectory_geometry.v1",
        "model": "test", "kind": kind, "mode": mode, "mode_raw": mode_raw or mode,
        "dataset": dataset, "dataset_raw": None if kind == "reference" else dataset,
        "budget": budget, "base_seed": BASE_SEED, "seed": BASE_SEED + prompt_idx,
        "prompt_idx": prompt_idx, "prompt_id": f"p{prompt_idx}",
        "num_steps": N_STEPS, "d": D, "latent_shape": [1, 1, 1, 1, D],
        "z_T_dtype": "float32", "path_dtype": "float32",
        "z_T_sha256": f"{dataset}-{BASE_SEED}-{prompt_idx}",
        "sigmas": SIGMAS, "source_dir": source_dir, "frame_files": {},
        "actions": bits, "n_cached": n_cached if n_cached is not None else (
            bits.count("1") if bits else None),
        "ref_source_dir": ref_source_dir, "ref_z_T_match": ref_z_T_match,
    })
    return rec


def build_store(tmp_path, *, cells, bad_ref_join=False, t3_copy=True,
                one_ref_mismatch=True, mismatch_idx=None):
    """`cells` = list of (mode, budget K, bits) applied to every dataset.

    `bits` is either one 50-character string used for every prompt, or a list of
    them indexed by prompt (so one cell can realise different step counts per
    video, the way a gate does)."""
    root = tmp_path / "trajectory"
    root.mkdir(parents=True, exist_ok=True)
    merged = root / "t1_merged.jsonl"
    index = {"n_sigma_grids": 1, "sigma_grids": [SIGMAS], "dirs": {}}
    lines: list[str] = []
    refs: dict[tuple[str, int], np.ndarray] = {}

    for ds in DATASETS:
        ref_dir = f"references/{ds}_s{BASE_SEED}"
        index["dirs"][ref_dir] = {"kind": "reference", "n_t1": N_PROMPTS}
        for idx in range(N_PROMPTS):
            rng = np.random.default_rng(abs(hash((ds, idx))) % (2 ** 31))
            Z = reference_path(rng)
            refs[(ds, idx)] = Z
            lines.append(json.dumps(row(Z, kind="reference", dataset=ds, prompt_idx=idx,
                                        source_dir=ref_dir)))

    bad_idx = N_PROMPTS - 1 if mismatch_idx is None else mismatch_idx
    for mode, k, bits in cells:
        for ds in DATASETS:
            cell_dir = f"cells/{mode}_{ds}_K{k}_s{BASE_SEED}"
            index["dirs"][cell_dir] = {"kind": "cell", "n_t1": N_PROMPTS}
            for idx in range(N_PROMPTS):
                b = bits if isinstance(bits, str) else bits[idx % len(bits)]
                Z = cached_path(refs[(ds, idx)], b)
                join = f"references/{ds}_s{BASE_SEED}"
                match = True
                if one_ref_mismatch and idx == bad_idx and ds == DATASETS[0]:
                    match = False          # excluded and counted, never averaged
                if bad_ref_join and idx == 0 and ds == DATASETS[0]:
                    join = "references/somewhere_else_s99"
                lines.append(json.dumps(row(
                    Z, kind="cell", dataset=ds, prompt_idx=idx, source_dir=cell_dir,
                    mode=mode, mode_raw=f"{mode}_exact", budget=f"K{k}", bits=b,
                    ref_source_dir=join, ref_z_T_match=match)))
            if t3_copy:
                # the P3/P7 re-runs: same mode/dataset/budget/seed, MUST be dropped
                t3_dir = f"cells_t3/{mode}_{ds}_K{k}_s{BASE_SEED}"
                index["dirs"][t3_dir] = {"kind": "cell", "n_t1": 1}
                bits0 = bits if isinstance(bits, str) else bits[0]
                Z = cached_path(refs[(ds, 0)], bits0)
                lines.append(json.dumps(row(
                    Z, kind="cell", dataset=ds, prompt_idx=0, source_dir=t3_dir,
                    mode=mode, mode_raw=f"{mode}_exact", budget=f"K{k}", bits=bits0,
                    ref_source_dir=f"references/{ds}_s{BASE_SEED}", ref_z_T_match=True)))

    merged.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "t1_index.json").write_text(json.dumps(index), encoding="utf-8")
    return merged, root / "t1_index.json"


def build_floor(tmp_path):
    """Small valid P1 files with a readable float32 one-step turning signal."""
    floor_dir = tmp_path / "floor"
    floor_dir.mkdir(parents=True, exist_ok=True)
    for dtype, scale in (("float32", 1e-7), ("bfloat16", 1e-3)):
        dump = {
            "n_trajectories": 4,
            "windows": {
                str(w): {
                    "centers": list(range(w, N_STATES - w)),
                    "ratio": [6.0] * (N_STATES - 2 * w),
                }
                for w in (1, 5, 7)
            },
            "deviation": {
                "measured_med": [0.0] + [1.0] * (N_STATES - 2) + [0.0],
                "floor_med": [0.0] + [scale] * (N_STATES - 2) + [0.0],
            },
            "spacing_rel_med": [scale] * N_STEPS,
            "magnitude_rel_med": [scale] * N_STATES,
            "plane_share": {"measured_med": 0.9, "planar_floor_med": 1.0 - scale},
            "chord_rel_med": scale,
            "path_len_rel_med": scale,
            "straightness_rel_med": scale,
        }
        (floor_dir / f"p1_floor_{dtype}.json").write_text(json.dumps(dump))
    return floor_dir


def run(tmp_path, cells, *, backbone="hunyuan_video", extra=(), **kw):
    merged, index = build_store(tmp_path, cells=cells, **kw)
    tables = tmp_path / "tables"
    figs = tmp_path / "figs"
    floor_dir = build_floor(tmp_path)
    CVR.main(["--backbone", backbone, "--merged", str(merged), "--index", str(index),
              "--out_tables", str(tables), "--out_figs", str(figs),
              "--p1_floor_dir", str(floor_dir), *extra])
    report = json.loads((tables / f"cached_profiles_{backbone}.json").read_text())
    return report, tables, figs


DEFAULT_CELLS = [
    ("budcache", 29, FIXED_BITS[29]),          # fixed table, realises exactly 29
    ("sencache", 37, bits_with(36)),           # gate, documented cap at 36
    ("sencache", 41, bits_with(36)),           # same synthetic count, not a documented duplicate
    ("seacache", 41, bits_with(38)),           # gate, no documented number
]


def group_of(report, method, budget, dataset="__pooled__"):
    for g in report["groups"]:
        if g["method"] == method and g["budget"] == budget and g["dataset"] == dataset:
            return g
    raise AssertionError(f"no group {method} K{budget} {dataset}")


# ---------------------------------------------------------------------------
# 1. pairing
# ---------------------------------------------------------------------------


def test_pairing_denominators_and_cells_t3_drop(tmp_path):
    report, tables, _ = run(tmp_path, DEFAULT_CELLS, extra=["--no_figures"])
    n_cells = len(DEFAULT_CELLS)
    # one row per (cell, dataset, prompt) minus the planted ref_z_T_match=False
    per_cell = len(DATASETS) * N_PROMPTS - 1
    assert report["denominators"]["n_pairs_total"] == n_cells * per_cell
    assert report["denominators"]["n_excluded_ref_mismatch"] == n_cells
    # every cells_t3/ row was dropped: one per (cell, dataset)
    assert report["denominators"]["n_dropped_not_cells_prefix"] == n_cells * len(DATASETS)
    assert report["denominators"]["n_references"] == len(DATASETS) * N_PROMPTS

    g = group_of(report, "budcache", 29)
    assert g["n_pairs"] == per_cell
    assert g["n_excluded_ref_mismatch"] == 1
    # the pooled row is exactly the two dataset rows
    halves = [group_of(report, "budcache", 29, ds) for ds in DATASETS]
    assert sum(h["n_pairs"] for h in halves) == g["n_pairs"]
    # and the drop is visible per group in the JSON, keyed by (method, budget, dataset)
    assert report["source"]["cell_scan"]["excluded_ref_z_T_mismatch"][
        f"budcache_K29_{DATASETS[0]}"] == 1


def test_pairing_join_disagreement_is_a_hard_stop(tmp_path):
    with pytest.raises(SystemExit, match="joined this cell to"):
        run(tmp_path, DEFAULT_CELLS[:1], bad_ref_join=True, extra=["--no_figures"])


def test_paired_run_is_identical_up_to_its_only_cache_step(tmp_path):
    """A schedule that caches only the last step follows its reference exactly
    until then: the displacement ratio is 1 on steps 0..48 and departs from 1
    at step 49, which is where the endpoint (and therefore the chord) moves."""
    bits = "0" * 49 + "1"
    report, _, _ = run(tmp_path, [("seacache", 29, bits)], extra=["--no_figures"])
    g = group_of(report, "seacache", 29)
    med = np.asarray(g["profiles"]["disp_ratio"]["median"], dtype=float)
    assert np.allclose(med[:49], 1.0, atol=0.0, rtol=0.0)
    assert med[49] != 1.0
    chord = g["scalars"]["chord_ratio"]["median"]
    assert chord != 1.0 and chord == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# 2. prefix-identity classification
# ---------------------------------------------------------------------------


def test_prefix_length_rules():
    k0 = 12
    assert CVR.prefix_length("spacing", k0) == 12          # steps 0..11
    assert CVR.prefix_length("velocity_norm", k0) == 12
    assert CVR.prefix_length("magnitude", k0) == 13        # states 0..12
    assert CVR.prefix_length("turn_angle_deg", k0) == 11   # junctions 0..10
    assert CVR.prefix_length("second_diff_norm", k0) == 11
    # w=5: centres 5..7 -> array indices 0..2 (c + 5 <= 12)
    assert CVR.prefix_length("turn_angle_w5_deg", k0) == 3
    # w=7: c + 7 <= 12 has no centre >= 7
    assert CVR.prefix_length("turn_angle_w7_deg", k0) == 0
    # a schedule that caches step 0 has an EMPTY prefix, never a negative one —
    # except `magnitude`, whose state 0 is the shared z_T and is still comparable
    for field in CVR.PREFIX_FIELDS:
        assert CVR.prefix_length(field, 0) == (1 if field == "magnitude" else 0)
    # and nothing may exceed the array it indexes
    for field, length in CVR.PROFILE_LEN.items():
        if field in CVR.PREFIX_FIELDS:
            assert CVR.prefix_length(field, 50) <= length


def test_prefix_identity_is_zero_for_the_local_fields(tmp_path):
    report, tables, _ = run(tmp_path, DEFAULT_CELLS, extra=["--no_figures"])
    for method, budget, _ in DEFAULT_CELLS:
        g = group_of(report, method, budget)
        for field in CVR.PREFIX_FIELDS:
            entry = g["prefix_identity"][field]
            if entry["n_values_compared"] == 0:
                # nothing comparable at this k0 -> NOTHING is reported, never a 0
                assert entry["max_abs_diff"] is None, (method, budget, field, entry)
                continue
            assert entry["max_abs_diff"] == 0.0, (method, budget, field, entry)
            if entry["n_pairs_with_a_prefix"]:
                assert entry["n_pairs_exactly_zero"] == entry["n_pairs_with_a_prefix"]
        # the prefix really was compared, not silently empty
        assert g["prefix_identity"]["spacing"]["n_values_compared"] > 0
    # the TSV carries every group x field
    lines = (tables / "cached_prefix_identity_hunyuan_video.tsv").read_text().splitlines()
    assert len(lines) - 1 == len(report["groups"]) * len(CVR.PREFIX_FIELDS)


def test_an_all_zero_prefix_is_reported_as_zero_not_as_a_floor(tmp_path):
    """The defect this pins: a page that calls whatever is non-zero "the T1
    run-to-run reproducibility floor" reads the tail of the statistic as its
    headline. With every pair bit-identical the page must say so, name no pair,
    and make no floor claim."""
    report, tables, _ = run(tmp_path, DEFAULT_CELLS, extra=["--no_figures"])
    for g in report["groups"]:
        for entry in g["prefix_identity"].values():
            if entry["n_values_compared"] == 0:
                assert entry["n_pairs_not_exactly_zero"] is None
                continue
            assert entry["n_pairs_not_exactly_zero"] == 0
            # an all-zero row names no pair: every pair is equally "the maximum"
            assert entry["max_abs_diff_pair"] is None

    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "Every pooled row is bit-identical in every pair" in md
    assert "**The floor is an exact zero, not the maximum in the table below.**" in md
    # the overstating sentence must be gone from the page and from the JSON note
    assert "Whatever is non-zero **is** the run-to-run numerical floor" not in md
    assert "whatever is non-zero IS the T1 run-to-run reproducibility floor" \
        not in report["prefix_identity_note"]
    assert "n_pairs_exactly_zero" in report["prefix_identity_note"]
    # the fraction is on the page, not only in the TSV
    assert "pairs exactly zero / pairs with a prefix" in md
    head = (tables / "cached_prefix_identity_hunyuan_video.tsv").read_text().splitlines()[0]
    assert "n_pairs_not_exactly_zero" in head.split("\t")
    assert "max_abs_diff_pair" in head.split("\t")


def test_one_anomalous_generation_is_named_and_not_generalised(tmp_path):
    """One cached row perturbed after the fact = one generation that ran on
    different numerics. Its group must report exactly ONE non-zero pair, name
    that pair by cell and prompt, and keep every other pair at exact zero."""
    merged, index = build_store(tmp_path, cells=DEFAULT_CELLS[:1], one_ref_mismatch=False,
                                t3_copy=False)
    target_dir = f"cells/budcache_{DATASETS[0]}_K29_s{BASE_SEED}"
    lines = merged.read_text().splitlines()
    n_hit = 0
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("source_dir") == target_dir and rec.get("prompt_idx") == 2:
            rec["spacing"] = [v * (1.0 + 2.0e-4) for v in rec["spacing"]]
            lines[i] = json.dumps(rec)
            n_hit += 1
    assert n_hit == 1
    merged.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tables = tmp_path / "tables"
    floor_dir = build_floor(tmp_path)
    CVR.main(["--backbone", "hunyuan_video", "--merged", str(merged), "--index", str(index),
              "--out_tables", str(tables), "--out_figs", str(tmp_path / "figs"),
              "--p1_floor_dir", str(floor_dir), "--no_figures"])
    report = json.loads((tables / "cached_profiles_hunyuan_video.json").read_text())

    entry = group_of(report, "budcache", 29)["prefix_identity"]["spacing"]
    assert entry["n_pairs_not_exactly_zero"] == 1
    assert entry["n_pairs_exactly_zero"] == entry["n_pairs_with_a_prefix"] - 1
    assert entry["max_abs_diff"] > 0.0
    assert entry["max_abs_diff_pair"].startswith(f"{target_dir} idx 2 ")
    # velocity_norm shares the same array, magnitude does not: only the fields
    # the perturbation touches go non-zero, so the count is per field
    assert group_of(report, "budcache", 29)["prefix_identity"]["magnitude"][
        "n_pairs_not_exactly_zero"] == 0

    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "**Every pooled row that is not bit-identical**" in md
    assert f"{target_dir} idx 2 " in md
    assert "one generation set the row's maximum by itself" in md


def test_fields_that_must_not_be_checked_are_named_and_are_really_different(tmp_path):
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    named = {item["field"] for item in report["not_prefix_identical"]}
    assert {"d_perp", "max_dev_ratio", "straightness", "pca_evr",
            "update_chord_share"} <= named
    # none of them is in the identity check
    assert not (named & set(CVR.PREFIX_FIELDS))
    for g in report["groups"]:
        assert set(g["prefix_identity"]) == set(CVR.PREFIX_FIELDS)

    # and they are genuinely NOT equal before k0 — the whole reason they are
    # excluded. Rebuild one pair and compare its d_perp prefix directly.
    rng = np.random.default_rng(abs(hash((DATASETS[0], 1))) % (2 ** 31))
    Z_ref = reference_path(rng)
    bits = DEFAULT_CELLS[0][2]
    k0 = bits.index("1")
    m_ref = trajectory_metrics(Z_ref, SIGMAS)
    m_cached = trajectory_metrics(cached_path(Z_ref, bits), SIGMAS)
    assert np.allclose(m_ref["spacing"][:k0], m_cached["spacing"][:k0], atol=0, rtol=0)
    dev_ref = np.asarray(m_ref["d_perp"][1:k0], dtype=float)
    dev_cached = np.asarray(m_cached["d_perp"][1:k0], dtype=float)
    assert np.max(np.abs(dev_ref - dev_cached)) > 0.0
    assert m_ref["max_dev_ratio"] != m_cached["max_dev_ratio"]


# ---------------------------------------------------------------------------
# 3. event alignment
# ---------------------------------------------------------------------------


def test_event_samples_offsets_and_stratification():
    bits = list("0" * N_STEPS)
    for n in (5, 6, 20):
        bits[n] = "1"
    bits = "".join(bits)
    act = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
    parts = {"disp_ratio": np.arange(N_STEPS, dtype=float),
             "dev_diff": np.arange(N_STATES, dtype=float) * 100.0}

    out = CVR.event_samples(act, parts, 0)
    # j = 0 is by construction always the cache stratum
    assert set(out) == {"cache"}
    assert sorted(out["cache"]["disp_ratio"]) == [5.0, 6.0, 20.0]
    # dev_diff is read at the SAME index, the shifted variant one state later
    assert sorted(out["cache"]["dev_diff"]) == [500.0, 600.0, 2000.0]
    assert sorted(out["cache"]["dev_diff_state_shifted"]) == [600.0, 700.0, 2100.0]

    out = CVR.event_samples(act, parts, 1)
    # k=5 -> step 6 is cached; k=6 -> step 7 is full; k=20 -> step 21 is full
    assert sorted(out["cache"]["disp_ratio"]) == [6.0]
    assert sorted(out["full"]["disp_ratio"]) == [7.0, 21.0]

    out = CVR.event_samples(act, parts, -1)
    assert sorted(out["cache"]["disp_ratio"]) == [5.0]      # k=6 -> step 5 cached
    assert sorted(out["full"]["disp_ratio"]) == [4.0, 19.0]

    # offsets that leave the 0..49 step axis are dropped, not clipped
    edge = np.zeros(N_STEPS, dtype=np.uint8)
    edge[0] = edge[N_STEPS - 1] = 1
    out = CVR.event_samples(edge, parts, -2)
    assert sorted(out["full"]["disp_ratio"]) == [47.0]      # only k=49 survives
    out = CVR.event_samples(edge, parts, 5)
    assert sorted(out["full"]["disp_ratio"]) == [5.0]       # only k=0 survives


def test_event_alignment_sample_counts_in_the_report(tmp_path):
    bits = FIXED_BITS[29]
    report, tables, _ = run(tmp_path, [("budcache", 29, bits)], extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    n_pairs = g["n_pairs"]
    act = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
    for offset in CVR.EVENT_OFFSETS:
        expected = CVR.event_samples(act, {
            "disp_ratio": np.zeros(N_STEPS), "dev_diff": np.zeros(N_STATES)}, offset)
        for stratum in CVR.EVENT_STRATA:
            got = g["events"][str(offset)][stratum]["disp_ratio"]["n_samples"]
            want = expected.get(stratum, {}).get("disp_ratio", np.zeros(0)).size * n_pairs
            assert got == want, (offset, stratum)
    # median primary, mean alongside (the plan's word for this row is "average")
    entry = g["events"]["0"]["cache"]["disp_ratio"]
    assert entry["median"] is not None and entry["mean"] is not None
    lines = (tables / "cached_event_alignment_hunyuan_video.tsv").read_text().splitlines()
    assert len(lines) - 1 == (len(report["groups"]) * len(CVR.EVENT_OFFSETS)
                              * len(CVR.EVENT_STRATA) * len(CVR.EVENT_QUANTITIES))


# ---------------------------------------------------------------------------
# 4. realised K
# ---------------------------------------------------------------------------


def test_realized_k_fixed_gate_and_documented_counts(tmp_path):
    report, _, _ = run(tmp_path, DEFAULT_CELLS, extra=["--no_figures"])

    fixed = group_of(report, "budcache", 29)["k"]
    assert fixed["family"] == "fixed-table"
    assert (fixed["k_realized_min"], fixed["k_realized_max"]) == (29, 29)
    assert fixed["n_rows_k_not_nominal"] == 0
    assert fixed["k_documented"] is None and fixed["duplicate_config"] is False

    # HYV SenCache K37 has a documented realised count of 36.  K41 no longer
    # shares that capped configuration after the second frozen knob, so it has
    # no documented count and neither tier is marked as a duplicate.
    k37 = group_of(report, "sencache", 37)["k"]
    k41 = group_of(report, "sencache", 41)["k"]
    assert k37["family"] == "dynamic-gate"
    assert k37["k_documented"] == 36
    assert k41["k_documented"] is None
    assert k37["k_realized_mean"] == pytest.approx(36.0)
    assert k37["k_documented_mismatch"] is False
    assert k41["k_documented_mismatch"] is None
    assert k37["duplicate_config"] is False and k41["duplicate_config"] is False
    assert k37["duplicate_group"] is None and k41["duplicate_group"] is None
    assert k37["counted_instance"] is None and k41["counted_instance"] is None

    # a gate with no documented number reports the realised spread and no verdict
    sea = group_of(report, "seacache", 41)["k"]
    assert sea["k_documented"] is None and sea["k_documented_mismatch"] is None
    assert sea["k_realized_min"] == sea["k_realized_max"] == 38


def test_a_fixed_table_row_off_its_budget_is_flagged_not_dropped(tmp_path):
    report, _, _ = run(tmp_path, [("budcache", 29, bits_with(28))],
                       extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    assert g["n_pairs"] == len(DATASETS) * N_PROMPTS - 1       # nothing dropped
    assert g["k"]["n_rows_k_not_nominal"] == g["n_pairs"]
    assert g["k"]["k_realized_mean"] == pytest.approx(28.0)


def test_documented_k_mismatch_is_flagged(tmp_path):
    """A configuration whose realised count lands beside the documented number is
    a flagged row, never a dead run."""
    report, _, _ = run(tmp_path, [("sencache", 37, bits_with(30))],
                       extra=["--no_figures"])
    k = group_of(report, "sencache", 37)["k"]
    assert k["k_documented"] == 36
    assert k["k_documented_mismatch"] is True
    assert group_of(report, "sencache", 37)["n_pairs"] > 0

    # Wan carries a different pair of documented numbers, read from the same map
    assert CVR.DOCUMENTED_K["wan21"][("teacache", 37)] == 36
    assert CVR.DOCUMENTED_K["wan21"][("seacache", 41)] == 40


# ---------------------------------------------------------------------------
# 5. the section 3.9.2 boundaries and the rest of the contract
# ---------------------------------------------------------------------------


def test_cannot_measure_block_is_emitted_everywhere(tmp_path):
    report, tables, figs = run(tmp_path, DEFAULT_CELLS[:2])
    assert len(report["cannot_measure"]) == 5
    assert "P7" in report["cannot_measure_owner"]
    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    for item in report["cannot_measure"]:
        assert item["quantity"] in md
    # the boundaries stand near the head of the page, before any table
    assert md.index("CANNOT measure") < md.index("## 3.")
    # and in every figure caption block
    for item in report["cannot_measure"][:2]:
        assert item["quantity"] in md.split("## 6. Figures")[1]


def test_figures_and_floor_citation(tmp_path):
    report, tables, figs = run(tmp_path, DEFAULT_CELLS[:2])
    for method, _, _ in DEFAULT_CELLS[:2]:
        assert CVR.output_complete(figs / f"cached_profiles_{method}_hunyuan_video.png")
    assert CVR.output_complete(figs / "cached_event_alignment_hunyuan_video.png")
    # float32 throughout; the bf16 row is carried but not used for a verdict
    assert "float32" in report["floor_rows_cited"]["every reading in this file"]
    assert report["floor"]["source"]["float32"].endswith("p1_floor_float32.json")
    assert report["turn_w1"]["readable_float32"] is True
    g = group_of(report, "budcache", 29)
    assert set(g["turn_w1_at_cache_step"]) == {"-1", "0", "1"}


def test_turn_w1_readability_is_read_from_the_floor_not_assumed(tmp_path):
    """With a float32 floor that says w=1 is unreadable, the w=1 block is null
    with its reason — the readability is never hardcoded either way."""
    floor_dir = build_floor(tmp_path)
    for dtype in ("float32", "bfloat16"):
        dump = json.loads((floor_dir / f"p1_floor_{dtype}.json").read_text())
        if dtype == "float32":
            w1 = dump["windows"]["1"]
            w1["ratio"] = [1.0] * len(w1["ratio"])     # every centre below SNR 3
        (floor_dir / f"p1_floor_{dtype}.json").write_text(json.dumps(dump))

    merged, index = build_store(tmp_path, cells=DEFAULT_CELLS[:1])
    CVR.main(["--backbone", "hunyuan_video", "--merged", str(merged), "--index", str(index),
              "--out_tables", str(tmp_path / "tables"), "--out_figs", str(tmp_path / "figs"),
              "--p1_floor_dir", str(floor_dir), "--no_figures"])
    report = json.loads(
        (tmp_path / "tables" / "cached_profiles_hunyuan_video.json").read_text())
    assert report["turn_w1"]["readable_float32"] is False
    block = group_of(report, "budcache", 29)["turn_w1_at_cache_step"]
    assert block == {"value": None, "reason": "w=1 not readable at the float32 floor"}
    assert report["source"]["cell_scan"]["turn_w1_accumulated"] is False


def test_velocity_ratio_is_the_displacement_ratio(tmp_path):
    """Same |dsigma_n| on both sides: the two ratios are the same number, so the
    velocity ratio is a consistency check and not independent evidence."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    assert g["velocity_vs_displacement_ratio_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)


def test_reuse_refuses_to_overwrite_a_bigger_run(tmp_path):
    run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    with pytest.raises(SystemExit, match="different run parameters"):
        run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures", "--limit", "2"])


def test_limit_caps_cells_but_never_the_references(tmp_path):
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures", "--limit", "2"])
    assert report["denominators"]["n_references"] == len(DATASETS) * N_PROMPTS
    assert report["denominators"]["n_pairs_total"] <= 2 * len(DATASETS)


# ---------------------------------------------------------------------------
# 6. the audit fixes
# ---------------------------------------------------------------------------


def test_empty_prefix_reports_nothing_not_a_fabricated_zero(tmp_path):
    """A2. With k0 = 3, `turn_angle_w5_deg` (needs k0 >= 10) and
    `turn_angle_w7_deg` (needs k0 >= 14) have no comparable entry at all. The
    completion-criterion-4 table must say so, not print 0.0."""
    report, tables, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    for field in ("turn_angle_w5_deg", "turn_angle_w7_deg"):
        entry = g["prefix_identity"][field]
        assert entry["n_values_compared"] == 0
        assert entry["no_comparable_prefix"] is True
        assert entry["max_abs_diff"] is None and entry["max_rel_diff"] is None
        assert entry["max_abs_diff_at_index"] is None
        assert entry["n_pairs_exactly_zero"] is None
        assert "NOT a measured zero" in entry["note"]
        # the denominator is still honest: the pairs were looked at
        assert entry["n_pairs"] == g["n_pairs"]
    # ... and the TSV cell is empty, never "0"
    head, *body = (tables / "cached_prefix_identity_hunyuan_video.tsv"
                   ).read_text().splitlines()
    cols = head.split("\t")
    for line in body:
        cells = dict(zip(cols, line.split("\t")))
        if cells["field"] == "turn_angle_w7_deg":
            assert cells["max_abs_diff"] == "" and cells["n_pairs_exactly_zero"] == ""
            assert cells["no_comparable_prefix"] == "true"
    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "no comparable prefix" in md.lower()
    assert "NOT a measured zero" in md


def test_rows_off_nominal_is_null_for_gates_and_counted_for_fixed(tmp_path):
    """A3. The column would otherwise claim "0 videos deviate from the budget"
    for a gate, where every video does."""
    report, tables, _ = run(tmp_path, DEFAULT_CELLS, extra=["--no_figures"])
    assert group_of(report, "budcache", 29)["k"]["n_rows_k_not_nominal"] == 0
    for method, budget in (("sencache", 37), ("seacache", 41)):
        k = group_of(report, method, budget)["k"]
        assert k["family"] == "dynamic-gate"
        assert k["n_rows_k_not_nominal"] is None
        assert "gate" in k["n_rows_k_not_nominal_scope"]
    # the documented-K rows carry their own per-row counter
    assert group_of(report, "sencache", 37)["k"]["n_rows_k_not_documented"] == 0
    assert group_of(report, "seacache", 41)["k"]["n_rows_k_not_documented"] is None
    head, *body = (tables / "cached_scalar_diffs_hunyuan_video.tsv"
                   ).read_text().splitlines()
    cols = head.split("\t")
    rows = [dict(zip(cols, line.split("\t"))) for line in body]
    gate = [r for r in rows if r["method"] == "seacache"][0]
    assert gate["n_rows_k_not_nominal"] == ""


def test_documented_k_bracketing_is_not_agreement(tmp_path):
    """A4. Counts of 34 and 38 bracket the documented 36 while NO video realises
    it: that is a flagged row, and the bracket test is reported separately."""
    bits = [bits_with(34), bits_with(38)]
    report, _, _ = run(tmp_path, [("sencache", 37, bits)],
                       extra=["--no_figures"], one_ref_mismatch=False)
    k = group_of(report, "sencache", 37)["k"]
    assert k["k_documented"] == 36
    assert (k["k_realized_min"], k["k_realized_max"]) == (34, 38)
    assert k["k_documented_bracketed"] is True      # the old, weaker test
    assert k["k_documented_mismatch"] is True       # the per-row verdict
    assert k["n_rows_k_not_documented"] == group_of(report, "sencache", 37)["n_pairs"]


def test_a_row_that_caches_nothing_is_counted_and_skipped_not_fatal(tmp_path):
    """A7 / B3. Every other data anomaly here is count-and-continue; this one
    used to kill an sbatch job after the whole file had been parsed."""
    merged, index = build_store(tmp_path, cells=DEFAULT_CELLS[:1],
                                one_ref_mismatch=False)
    lines = merged.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("kind") == "cell" and rec["source_dir"].startswith("cells/"):
            rec["actions"] = "0" * N_STEPS
            rec["n_cached"] = 0
            lines[i] = json.dumps(rec)
            break
    merged.write_text("\n".join(lines) + "\n")
    CVR.main(["--backbone", "hunyuan_video", "--merged", str(merged),
              "--index", str(index), "--out_tables", str(tmp_path / "t"),
              "--out_figs", str(tmp_path / "f"), "--no_figures",
              "--p1_floor_dir", str(build_floor(tmp_path))])
    report = json.loads((tmp_path / "t" / "cached_profiles_hunyuan_video.json").read_text())
    assert report["denominators"]["n_rows_no_cache_step"] == 1
    g = group_of(report, "budcache", 29)
    assert g["n_rows_no_cache_step"] == 1
    assert g["n_pairs"] == len(DATASETS) * N_PROMPTS - 1     # only that row is gone
    assert len(g["n_rows_no_cache_step_keys"]) == 1
    assert "idx" in g["n_rows_no_cache_step_keys"][0]


def test_non_finite_entries_do_not_split_statistic_from_denominator():
    """A8. `n` must be the denominator the median actually used."""
    mat = np.array([[1.0], [2.0], [np.inf], [np.nan]], dtype=np.float32)
    out = CVR.summarise_profile(mat)
    assert out["n"] == [2] and out["n_pairs"] == 4
    assert out["median"] == [pytest.approx(1.5)]


def test_limit_spends_its_quota_on_pairs_not_on_excluded_rows(tmp_path):
    """A9 / B9. The excluded row is prompt 0, so an old row-counting cap would
    return one pair for that dataset while the JSON says limit=2."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures", "--limit", "2"],
                       mismatch_idx=0)
    assert group_of(report, "budcache", 29, DATASETS[0])["n_pairs"] == 2
    assert group_of(report, "budcache", 29, DATASETS[1])["n_pairs"] == 2
    assert report["source"]["cell_scan"]["n_rows_skipped_over_limit"] > 0
    assert "pairs, not rows" in report["source"]["cell_scan"]["limit_counts"]


def test_peak_and_k0_readings_reach_the_tsv(tmp_path):
    """A10 / B5. Quantity (3)'s per-side peaks and the one same-state ratio
    reading must not be JSON-only, and `mode_raw` must be somewhere."""
    report, tables, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    head = (tables / "cached_scalar_diffs_hunyuan_video.tsv"
            ).read_text().splitlines()[0].split("\t")
    for name in CVR.EXTRA_SCALARS:
        assert f"{name}_med" in head, name
    assert "mode_raw" in head
    body = (tables / "cached_scalar_diffs_hunyuan_video.tsv").read_text().splitlines()[1]
    assert "budcache_exact" in body
    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "disp_ratio_at_k0" in md.replace(" ", "_") or "disp ratio at k0" in md
    assert group_of(report, "budcache", 29)["mode_raw"] == ["budcache_exact"]


def test_disp_ratio_at_k0_is_read_at_each_rows_own_k0(tmp_path):
    """B5. With two different k0 in one group, the reading is per row, not the
    profile sampled at the group's median k0."""
    bits = ["0" * 5 + "1" + "0" * 44, "0" * 20 + "1" + "0" * 29]
    report, _, _ = run(tmp_path, [("seacache", 29, bits)], extra=["--no_figures"],
                       one_ref_mismatch=False)
    g = group_of(report, "seacache", 29)
    # each row's own k0 is the only index where the two runs share a state, and
    # the planted cache reuses the previous displacement there
    assert g["scalars"]["k0"]["median"] is not None
    assert g["scalars"]["disp_ratio_at_k0"]["median"] is not None
    med = g["profiles"]["disp_ratio"]["median"]
    # the two k0 are 5 and 20; a median-k0 lookup would land at neither row's k0
    assert g["scalars"]["disp_ratio_at_k0"]["n_samples"] == g["n_pairs"]
    assert med[5] is not None and med[20] is not None


def test_short_cell_scan_is_refused(tmp_path):
    """B1. Deleting cell rows must not shrink a median's denominator silently."""
    merged, index = build_store(tmp_path, cells=DEFAULT_CELLS[:1])
    lines = merged.read_text().splitlines()
    kept = []
    dropped = 0
    for line in lines:
        rec = json.loads(line)
        if (rec.get("kind") == "cell" and rec["source_dir"].startswith("cells/")
                and dropped < 2):
            dropped += 1
            continue
        kept.append(line)
    merged.write_text("\n".join(kept) + "\n")
    with pytest.raises(SystemExit, match="disagrees with t1_index.json"):
        CVR.main(["--backbone", "hunyuan_video", "--merged", str(merged),
                  "--index", str(index), "--out_tables", str(tmp_path / "t"),
                  "--out_figs", str(tmp_path / "f"), "--no_figures",
                  "--p1_floor_dir", str(build_floor(tmp_path))])


def test_duplicated_cell_row_is_refused(tmp_path):
    """B1. A duplicated line would double-count one prompt in every median."""
    merged, index = build_store(tmp_path, cells=DEFAULT_CELLS[:1])
    lines = merged.read_text().splitlines()
    dup = next(line for line in lines
               if json.loads(line).get("source_dir", "").startswith("cells/"))
    merged.write_text("\n".join(lines + [dup]) + "\n")
    with pytest.raises(SystemExit, match="disagrees with t1_index.json"):
        CVR.main(["--backbone", "hunyuan_video", "--merged", str(merged),
                  "--index", str(index), "--out_tables", str(tmp_path / "t"),
                  "--out_figs", str(tmp_path / "f"), "--no_figures",
                  "--p1_floor_dir", str(build_floor(tmp_path))])


def test_pooled_rows_state_their_weighting_and_the_dataset_spread(tmp_path):
    """B2. A pooled median is pair-weighted; the page and the JSON must say so
    and must carry the per-dataset denominators and min-max."""
    report, tables, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    pool = g["pooling"]
    assert "pair-weighted" in pool["weighting"]
    assert set(pool["n_pairs_per_dataset"]) == set(DATASETS)
    assert sum(pool["n_pairs_per_dataset"].values()) == g["n_pairs"]
    spread = pool["scalar_median_across_datasets"]["chord_ratio"]
    assert set(spread["per_dataset"]) == set(DATASETS)
    assert spread["min"] <= spread["mean_of_dataset_medians"] <= spread["max"]
    md = (tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "PAIR-weighted" in md
    assert "pairs per dataset" in md
    # a per-dataset row says it is not pooled
    solo = group_of(report, "budcache", 29, DATASETS[0])
    assert solo["pooling"]["weighting"].startswith("single dataset")


def test_wan_rider_travels_with_the_wan_figures_only(tmp_path):
    """A1 / B9. The ratio caveat is a HunyuanVideo sentence; on wan21 every
    figure caption must carry the section 2.2 rider next to it, and the HYV page
    must not print the Wan-only rider at all."""
    hyv, hyv_tables, _ = run(tmp_path / "hyv", DEFAULT_CELLS[:1])
    wan, wan_tables, _ = run(tmp_path / "wan", DEFAULT_CELLS[:1], backbone="wan21")

    assert CVR.CAV_WAN_RIDER in wan["caveats"]
    assert CVR.CAV_WAN_RIDER not in hyv["caveats"]
    assert wan["figure_caveats"] == [CVR.CAV_DISPLACEMENT, CVR.CAV_WAN_RIDER]
    assert hyv["figure_caveats"] == [CVR.CAV_DISPLACEMENT]

    wan_md = (wan_tables / "cached_profiles_wan21.md").read_text()
    figs = wan_md.split("## 6. Figures")[1]
    assert figs.count("Wan2.1 rider") >= 2          # every caption, both figures
    for caption in ("profiles", "alignment"):
        assert "Wan2.1 rider" in CVR.figure_caption(wan, caption, "budcache")
        assert "Wan2.1 rider" not in CVR.figure_caption(hyv, caption, "budcache")
    hyv_md = (hyv_tables / "cached_profiles_hunyuan_video.md").read_text()
    assert "Wan2.1 rider" not in hyv_md


def test_figure_captions_describe_what_is_actually_drawn(tmp_path):
    """A6 / B4 / B7. The markers are the per-dataset data-derived modal path, and
    the alignment figure draws the literal same-index dev_diff."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    prof = CVR.figure_caption(report, "profiles", "budcache")
    assert "frozen table" not in prof and "pooled modal path" not in prof
    assert "modal path" in prof and "not read from the frozen" in prof
    align = CVR.figure_caption(report, "alignment", "hunyuan_video")
    assert "LITERAL same index" in align
    assert "dev_diff_state_shifted" in align


def test_two_passes_are_declared(tmp_path):
    """A5. The file is read twice; the docstring and the JSON say so."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    assert report["source"]["passes"] == 2
    assert "two streaming passes" in report["source"]["passes_note"]
    assert "TWO streaming passes" in CVR.__doc__


def test_index_reconciliation_is_recorded(tmp_path):
    """B1. The clean case records the counts it matched, so a later reader can
    see the check ran rather than assume it."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    recon = report["source"]["cell_scan"]["index_reconciliation"]
    assert recon["reconciled"] is True
    assert recon["n_rows_seen_under_cells_prefix"] == recon["n_rows_expected_by_index"]
    assert recon["n_cell_dirs_seen"] == recon["n_cell_dirs_in_index"] == len(DATASETS)


def test_a_null_prefix_index_means_only_that_nothing_was_comparable(tmp_path):
    """A2 follow-through: an exactly-identical prefix still reports the index it
    measured (with a 0 difference), so a null index is unambiguous."""
    report, _, _ = run(tmp_path, DEFAULT_CELLS[:1], extra=["--no_figures"])
    g = group_of(report, "budcache", 29)
    measured = g["prefix_identity"]["spacing"]
    assert measured["max_abs_diff"] == 0.0
    assert measured["max_abs_diff_at_index"] is not None
    empty = g["prefix_identity"]["turn_angle_w7_deg"]
    assert empty["max_abs_diff"] is None and empty["max_abs_diff_at_index"] is None
