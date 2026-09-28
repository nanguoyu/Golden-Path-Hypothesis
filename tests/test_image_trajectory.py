"""Unit tests for the image cache-bend analysis layer.

`docs/image_cached_trajectory_plan_zh.md` sections 7.2 and 7.3. The pair-level
arithmetic itself is the video pipeline's and is tested there
(`tests/test_latent_paths_cached.py`); what is asserted here is the image side's
own: the frozen inputs, the three decisions schemas, the prefix-identity and
early-offset definitions, and the statistics the Q2 protocol pins down before
any number is read.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.image_trajectory import build_inputs as BI  # noqa: E402
from analysis.image_trajectory import cached_paths as CP  # noqa: E402
from analysis.image_trajectory import early_quality_link as QL  # noqa: E402


# ---------------------------------------------------------------------------
# frozen inputs
# ---------------------------------------------------------------------------


def test_sample_builder_is_deterministic_with_synthetic_inputs(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("".join(f"Synthetic prompt {i}\n" for i in range(1632)))
    roles = {"discovery": list(range(544)), "validation": list(range(544, 1088)),
             "test": list(range(1088, 1632))}
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"prompt_count": 1632, "roles": roles}))
    monkeypatch.setattr(BI, "_ROOT", tmp_path)
    monkeypatch.setattr(BI, "PROMPT_FILE", prompts)
    monkeypatch.setattr(BI, "SPLITS", splits)
    sample = BI.build_sample()
    assert json.dumps(BI.build_sample(), sort_keys=True) == json.dumps(sample, sort_keys=True)
    assert sample["prompt_file_sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest()
    assert len(sample["prompt_indices"]) == len(set(sample["prompt_indices"])) == 50
    for idx, role in zip(sample["prompt_indices"], sample["roles"]):
        assert idx in roles[role]
    monkeypatch.setattr(BI, "SAMPLE_RNG_SEED", [20260825])
    assert BI.build_sample()["prompt_indices"] != sample["prompt_indices"]
    assert (BI.OUT_DIR / "zero_schedule_50.txt").read_text(
        encoding="utf-8") == "0" * 50 + "\n"


def test_option_a_is_256_cells_and_2_reference_streams() -> None:
    rows = CP.load_manifest()
    cells = [r for r in rows if r["schedule"] != "refnone"]
    refs = [r for r in rows if r["schedule"] == "refnone"]
    assert len(cells) == 256
    assert len(refs) == 2
    assert sum(1 for r in cells if r["payload"] == "reuse") == 151
    assert len({r["cell_id"] for r in rows}) == len(rows)


def test_every_cell_bitstring_matches_its_declared_budget() -> None:
    for row in CP.load_manifest():
        bits = CP.schedule_bits(row["schedule_file"])
        assert len(bits) == 50
        assert set(bits) <= {"0", "1"}
        assert bits.count("1") == row["k_realized"]
        # only the off-budget gate rows are allowed to differ from the label
        if row["k_realized"] != row["k"]:
            assert row["schedule"].endswith("_off")


def test_sample_is_50_shared_indices_with_recorded_roles() -> None:
    sample = CP.load_sample()
    assert len(sample["prompt_indices"]) == 50
    assert sorted(sample["prompt_indices"]) == sample["prompt_indices"]
    assert len(set(sample["prompt_indices"])) == 50
    assert len(sample["roles"]) == 50
    assert (len(sample["discovery_indices"]) + len(sample["held_out_indices"])
            == 50)
    assert all(i < sample["prompt_count"] for i in sample["prompt_indices"])


# ---------------------------------------------------------------------------
# the three image decisions schemas
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "decisions_00000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_actions_read_from_each_schema(tmp_path: Path) -> None:
    steps = [{"step": i, "action": "cache" if i in (3, 7) else "full"}
             for i in range(50)]
    want = "".join("1" if i in (3, 7) else "0" for i in range(50))
    for key in ("per_step", "steps", "records"):
        path = _write(tmp_path, {key: steps, "cache_steps": [3, 7]})
        assert CP.actions_from_decisions(path) == want


def test_actions_from_cache_steps_alone(tmp_path: Path) -> None:
    path = _write(tmp_path, {"cache_steps": [0, 49]})
    bits = CP.actions_from_decisions(path)
    assert bits[0] == "1" and bits[49] == "1" and bits.count("1") == 2


def test_action_sources_must_agree(tmp_path: Path) -> None:
    steps = [{"step": i, "action": "cache" if i == 3 else "full"} for i in range(50)]
    path = _write(tmp_path, {"per_step": steps, "cache_steps": [4]})
    with pytest.raises(SystemExit):
        CP.actions_from_decisions(path)


def test_a_short_step_list_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, {"per_step": [{"step": 0, "action": "full"}]})
    with pytest.raises(SystemExit):
        CP.actions_from_decisions(path)


# ---------------------------------------------------------------------------
# one pair
# ---------------------------------------------------------------------------


def _reference_path(seed: int = 0, d: int = 64) -> np.ndarray:
    """A bending 51-state path: a straight run plus two independent curvature
    directions, so the chord-orthogonal residual has the rank 2 that a
    `[chord, PC1, PC2]` frame needs."""
    rng = np.random.default_rng(seed)
    a, b, c, e = (rng.normal(size=d) for _ in range(4))
    t = np.linspace(0.0, 1.0, CP.N_STATES)[:, None]
    return (a + t * b * 10.0 + np.sin(np.pi * t) * c * 2.0
            + np.sin(2.0 * np.pi * t) * e * 1.0).astype(np.float32)


def _frame(Zr: np.ndarray) -> np.ndarray:
    frame = CP.plane_frame(Zr.astype(np.float64))
    assert frame is not None, "the synthetic reference has no bend plane"
    return np.asarray(frame)


def _cached_from(Zr: np.ndarray, k0: int, seed: int = 1) -> np.ndarray:
    """Identical to the reference up to state k0, then pushed off it."""
    rng = np.random.default_rng(seed)
    Zc = Zr.copy()
    kick = rng.normal(size=Zr.shape[1]).astype(np.float32)
    for n in range(k0 + 1, CP.N_STATES):
        Zc[n] = Zr[n] + kick * 0.02 * (n - k0)
    return Zc


def _actions(k0: int, extra: tuple[int, ...] = ()) -> str:
    hits = {k0, *extra}
    return "".join("1" if i in hits else "0" for i in range(50))


def test_prefix_before_the_first_cache_step_is_exactly_zero() -> None:
    Zr = _reference_path()
    Zc = _cached_from(Zr, k0=12)
    Fr = _frame(Zr)
    rec = CP.pair_record(prompt_idx=0, role="test", Zc=Zc, Zr=Zr, Fr=Fr,
                         actions=_actions(12), sigmas=list(np.linspace(1, 0, 51)),
                         ref_scalars=CP.path_scalars(Zr))
    assert rec["k0"] == 12
    assert rec["prefix"]["n_states_compared"] == 13
    assert rec["prefix"]["exactly_zero"] is True
    assert rec["prefix"]["max_abs_D"] == 0.0
    assert rec["endpoint"]["D50_over_chord_ref"] > 0.0


def test_direction_shares_sum_to_one_at_every_measured_state() -> None:
    Zr = _reference_path(seed=3)
    Zc = _cached_from(Zr, k0=5, seed=4)
    Fr = _frame(Zr)
    D, _, shares = CP.cached_pair_readings(Zc, Zr, Fr)
    measured = shares[np.isfinite(shares).all(axis=1)]
    assert measured.shape[0] >= 40
    assert np.allclose(measured.sum(axis=1), 1.0, atol=1e-9)
    assert (measured >= -1e-12).all()
    assert np.all(np.diff(D[5:]) > 0)


def test_early_offset_is_structurally_zero_for_a_late_first_jump() -> None:
    Zr = _reference_path(seed=7)
    Fr = _frame(Zr)
    late = CP.pair_record(prompt_idx=1, role="test",
                          Zc=_cached_from(Zr, k0=30), Zr=Zr, Fr=Fr,
                          actions=_actions(30), sigmas=list(np.linspace(1, 0, 51)),
                          ref_scalars=CP.path_scalars(Zr))
    assert late["early"]["structurally_zero_early"] is True
    assert late["early"]["D10_over_chord_ref"] == 0.0
    assert late["early"]["early_auc_over_chord_ref"] == 0.0

    early = CP.pair_record(prompt_idx=2, role="test",
                           Zc=_cached_from(Zr, k0=2), Zr=Zr, Fr=Fr,
                           actions=_actions(2), sigmas=list(np.linspace(1, 0, 51)),
                           ref_scalars=CP.path_scalars(Zr))
    assert early["early"]["structurally_zero_early"] is False
    assert early["early"]["D10_over_chord_ref"] > 0.0
    assert (early["early"]["early_auc_over_chord_ref"]
            > early["early"]["D10_over_chord_ref"])


def test_first_jump_reads_the_step_the_cache_writes() -> None:
    Zr = _reference_path(seed=11)
    Fr = _frame(Zr)
    sigmas = list(np.linspace(1.0, 0.0, 51))
    rec = CP.pair_record(prompt_idx=3, role="test", Zc=_cached_from(Zr, k0=8),
                         Zr=Zr, Fr=Fr, actions=_actions(8), sigmas=sigmas,
                         ref_scalars=CP.path_scalars(Zr))
    jump = rec["first_jump"]
    assert jump["delta_D_k0"] > 0.0
    assert jump["slope_k0_plus_5"] > 0.0
    assert jump["sigma_k0"] == pytest.approx(sigmas[8])
    assert jump["sigma_step_k0"] == pytest.approx(sigmas[8] - sigmas[9])


def test_a_row_that_never_caches_has_no_first_jump() -> None:
    Zr = _reference_path(seed=13)
    Fr = _frame(Zr)
    rec = CP.pair_record(prompt_idx=4, role="test", Zc=Zr.copy(), Zr=Zr, Fr=Fr,
                         actions="0" * 50, sigmas=list(np.linspace(1, 0, 51)),
                         ref_scalars=CP.path_scalars(Zr))
    assert rec["k0"] is None
    assert rec["first_jump"]["delta_D_k0"] is None
    assert rec["prefix"]["has_cache_step"] is False
    assert rec["scalar_diff"]["chord_ratio"] == pytest.approx(1.0)


def test_scalar_diffs_are_zero_against_the_path_itself() -> None:
    Zr = _reference_path(seed=17)
    scalars = CP.path_scalars(Zr)
    Fr = _frame(Zr)
    rec = CP.pair_record(prompt_idx=5, role="test", Zc=Zr.copy(), Zr=Zr, Fr=Fr,
                         actions="0" * 50, sigmas=list(np.linspace(1, 0, 51)),
                         ref_scalars=scalars)
    diff = rec["scalar_diff"]
    assert diff["straightness_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diff["max_dev_ratio_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diff["pca_evr_top2_diff"] == pytest.approx(0.0, abs=1e-12)
    assert rec["endpoint"]["chord_angle_deg"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# floors
# ---------------------------------------------------------------------------


def test_bf16_floor_is_small_positive_and_two_routed() -> None:
    Z = _reference_path(seed=23)
    floor = CP.bf16_floor(Z)
    assert floor["displacement_route"] > 0.0
    assert floor["deviation_route"] >= 0.0
    assert floor["floor"] == max(floor["displacement_route"],
                                 floor["deviation_route"])
    assert floor["floor"] < 0.05


def test_a_bf16_exact_path_is_recognised_as_such() -> None:
    """The production case: both samplers step in bf16, so a float32 store of
    their states is bit-identical to a bf16 one and the floor is exactly zero
    rather than small."""
    import torch

    Z = _reference_path(seed=31)
    bf16_exact = torch.from_numpy(Z).to(torch.bfloat16).to(torch.float32).numpy()
    assert np.array_equal(CP._round_trip_bf16(bf16_exact), bf16_exact)
    assert not np.array_equal(CP._round_trip_bf16(Z), Z)


def test_bf16_floor_of_an_already_rounded_path_is_zero() -> None:
    """Why the floor is measured on the fp32 subset: rounding the production
    store to bf16 again is the identity, and would report no floor at all."""
    import torch

    Z = _reference_path(seed=29)
    already = torch.from_numpy(Z).to(torch.bfloat16).to(torch.float32).numpy()
    assert CP.bf16_floor(already)["floor"] == 0.0
    assert CP.bf16_floor(Z)["floor"] > 0.0


# ---------------------------------------------------------------------------
# the Q2 statistics
# ---------------------------------------------------------------------------


def test_spearman_reports_n_and_refuses_a_degenerate_column() -> None:
    assert QL.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])["rho"] == pytest.approx(1.0)
    assert QL.spearman([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])["rho"] == pytest.approx(-1.0)
    assert QL.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])["rho"] is None
    assert QL.spearman([1, None, 3, 4, 5], [1, 2, None, 4, 5])["n"] == 3


def test_ols_recovers_a_planted_coefficient() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 2.0 * x1 - 0.5 * x2 + rng.normal(size=200) * 0.01
    fit = QL.ols(y, np.column_stack([x1, x2]))
    assert fit["coef"][0] == pytest.approx(2.0, abs=0.02)
    assert fit["coef"][1] == pytest.approx(-0.5, abs=0.02)
    assert fit["r2"] > 0.99
    assert fit["dof"] == 197


def test_mediation_shows_full_absorption_when_early_carries_k0() -> None:
    """A synthetic row table where quality is a function of the early offset
    alone and k0 only predicts it through that: the k0 coefficient must
    collapse when the early offset joins the regression."""
    rng = np.random.default_rng(5)
    rows: dict[tuple, dict] = {}
    quality: dict[str, dict] = {"flux": {}, "qwen": {}}
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            for i in range(20):
                k0 = int(rng.integers(1, 13))
                early = 1.0 / k0 + rng.normal() * 1e-3
                schedule, payload = f"s{i}", "reuse"
                rows[(model, k, schedule, payload)] = {
                    "k0": k0, "D10_over_chord_ref": early,
                    "early_auc_over_chord_ref": early * 5,
                    "D50_over_chord_ref": early * 3,
                }
                quality[model][(schedule, payload, k)] = -40.0 * early
    med = QL.q2_mediation(rows, quality, "D10_over_chord_ref")
    assert med["quality_on_k0"]["r2"] > 0.5
    assert abs(med["quality_on_both"]["coef"][1]) > 0.9
    assert med["k0_absorbed_fraction"] > 0.9


def test_row_table_medians_and_the_held_out_slice() -> None:
    bend = []
    for role, value in (("discovery", 10.0), ("validation", 1.0), ("test", 3.0)):
        bend.append({
            "model": "flux", "k": 29, "schedule": "uniform", "payload": "reuse",
            "prompt_idx": len(bend), "role": role, "k0": 4, "n_cached": 29,
            "D10_over_chord_ref": value, "early_auc_over_chord_ref": value,
            "D50_over_chord_ref": value, "delta_D_k0": value,
            "slope_k0_plus_5": value, "chord_ratio": 1.0,
            "straightness_diff": 0.0, "max_dev_ratio_diff": 0.0,
            "pca_evr_top2_diff": 0.0, "chord_angle_deg": 0.0,
            "plane_angle1_deg": 0.0, "share_chord_50": 0.1,
            "share_in_plane_50": 0.2, "share_off_plane_50": 0.7,
            "share_off_plane_k0p1": 0.6,
        })
    key = ("flux", 29, "uniform", "reuse")
    assert QL.row_table(bend, roles="all")[key]["D10_over_chord_ref"] == 3.0
    held = QL.row_table(bend, roles="held_out")[key]
    assert held["D10_over_chord_ref"] == 2.0
    assert held["n_pairs"] == 2


def test_restricting_the_mediation_removes_the_structural_zero_flip() -> None:
    """The audit finding, planted: rows whose k0 >= 10 have an early offset of
    exactly 0 by definition. On the full population those rows drag the joint
    k0 coefficient negative; on the k0 < 10 population the coefficient
    collapses to zero (full mediation) instead."""
    rng = np.random.default_rng(11)
    rows: dict[tuple, dict] = {}
    quality: dict[str, dict] = {"flux": {}, "qwen": {}}
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            for i in range(40):
                k0 = int(rng.integers(1, 14))
                early = 0.0 if k0 >= 10 else 1.0 / k0 + abs(rng.normal()) * 1e-3
                schedule, payload = f"s{i}", "reuse"
                rows[(model, k, schedule, payload)] = {
                    "k0": k0, "D10_over_chord_ref": early,
                    "early_auc_over_chord_ref": early * 5,
                    "D50_over_chord_ref": early * 3 + 0.05 / k0,
                }
                # quality depends on the true early bend, which for a late-k0
                # row is small but NOT zero -- the structural zero hides it
                true_early = 1.0 / k0
                quality[model][(schedule, payload, k)] = -40.0 * true_early
    full = QL.q2_mediation(rows, quality, "D10_over_chord_ref")
    restricted = QL.q2_mediation(rows, quality, "D10_over_chord_ref",
                                 restrict_k0_lt=10)
    assert full["restricted_to_k0_lt"] is None
    assert restricted["restricted_to_k0_lt"] == 10
    assert restricted["n_rows"] < full["n_rows"]
    # full population: joint k0 coefficient pulled below the restricted one
    assert full["quality_on_both"]["coef"][0] < restricted["quality_on_both"]["coef"][0]
    # restricted population: the early offset carries k0 entirely
    assert abs(restricted["quality_on_both"]["coef"][0]) < 0.1
    assert (restricted["quality_on_both"]["r2"]
            - restricted["quality_on_early"]["r2"]) < 0.01


def test_within_cell_partial_correlation_vanishes_when_control_carries_it() -> None:
    """Plant a per-prompt quality that is a function of D[50] alone, with
    D[10] correlated with D[50] but carrying no extra information: the raw
    within-cell rho is strong, the partial given D[50] is near zero. The
    p < 0.05 share must be over the cells whose rho exists."""
    rng = np.random.default_rng(7)
    bend = []
    perprompt: dict[str, dict] = {"flux": {}, "qwen": {}}
    for c in range(6):
        schedule = f"s{c}"
        for i in range(40):
            d50 = float(abs(rng.normal()) + 0.1)
            d10 = d50 * 0.1 + abs(rng.normal()) * 0.05
            bend.append({"model": "flux", "k": 29, "schedule": schedule,
                         "payload": "reuse", "prompt_idx": i, "role": "test",
                         "D10_over_chord_ref": d10,
                         "D50_over_chord_ref": d50})
            perprompt["flux"][(schedule, "reuse", 29, i)] = (
                -10.0 * d50 + float(rng.normal()) * 0.2)
    # one degenerate cell: D10 constant, no rho, still counted in n_cells
    for i in range(40):
        bend.append({"model": "flux", "k": 29, "schedule": "zeros",
                     "payload": "reuse", "prompt_idx": i, "role": "test",
                     "D10_over_chord_ref": 0.0,
                     "D50_over_chord_ref": float(i + 1)})
        perprompt["flux"][("zeros", "reuse", 29, i)] = -float(i + 1)
    out = QL.q2_within_cell(bend, perprompt, roles="all",
                            field="D10_over_chord_ref")
    assert out["n_cells"] == 7
    assert out["n_cells_with_rho"] == 6
    assert out["share_p_below_05"] == 1.0  # over the 6 cells with a rho
    assert out["control_rho_median"] < -0.95
    assert abs(out["partial_rho_median"]) < abs(out["rho_median"])
    assert abs(out["partial_rho_median"]) < 0.3
