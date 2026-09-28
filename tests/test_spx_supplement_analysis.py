"""The image SPX supplement analysis, on synthetic staged tables.

The point of these is the *shape* of the statistics, not the science: a fixture
where the truth is known by construction is the only way to check that the
off-modal split really removes the deterministic zeros, that the formal P1 fit
is the balanced sub-grid and not the ragged panel, and that a paired band is a
paired band.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import analysis.spx_supplement as sup

METRICS = ("psnr", "ssim", "lpips", "image_reward", "clip")
N_PROMPTS = 40


def write_tsv_gz(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell) for cell in row) + "\n")


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """Two models x one budget, a handful of rows, with a known structure."""

    rng = np.random.default_rng(20260823)
    schedules = {}
    rows_spx = []
    rows_native = []
    modal_bits = {}

    for model, base in (("flux", 30.0), ("qwen", 26.0)):
        for budget_k in (29,):
            # deterministic, budget-respecting bitstrings
            for index, name in enumerate(
                ["budcache", "dpcache", "uniform", "dicache_top1", "meancache",
                 "seacache_top1", "rand_1", "rand_2", "ham2f", "ham4f", "dp_rho2"]
            ):
                bits = ["0"] * 50
                picks = rng.permutation(np.arange(3, 48))[:budget_k]
                for step in picks:
                    bits[int(step)] = "1"
                schedules[(model, budget_k, name)] = "".join(bits)
            modal_bits[(model, budget_k, "seacache")] = schedules[
                (model, budget_k, "seacache_top1")
            ]

            payloads = list(sup.PAYLOADS)
            row_effect = {name: rng.normal(0, 1.5) for name in
                          ["budcache", "dpcache", "uniform", "dicache_top1"]}
            payload_effect = {p: rng.normal(0, 0.8) for p in payloads}
            for seed in (41, 42, 43):
                for name in ["budcache", "dpcache", "uniform", "dicache_top1"]:
                    for payload in payloads:
                        for prompt in range(N_PROMPTS):
                            value = (base + row_effect[name] + payload_effect[payload]
                                     + rng.normal(0, 0.2))
                            rows_spx.append([
                                model, name, payload, budget_k, seed, prompt,
                                value, 0.9, 0.05, 0.5, 30.0,
                            ])
                # single- and two-column rows
                for name, payload in (
                    ("meancache", "reuse"), ("meancache", "mean_avg_vel"),
                    ("seacache_top1", "reuse"), ("rand_1", "reuse"), ("rand_2", "reuse"),
                    ("ham2f", "reuse"), ("ham4f", "reuse"), ("dp_rho2", "reuse"),
                ):
                    bump = {"meancache": 1.2, "ham2f": 1.0, "ham4f": 0.6,
                            "dp_rho2": 0.1, "rand_1": 0.0, "rand_2": -0.1,
                            "seacache_top1": -2.0}[name]
                    for prompt in range(N_PROMPTS):
                        rows_spx.append([
                            model, name, payload, budget_k, seed, prompt,
                            base + bump + rng.normal(0, 0.2), 0.9, 0.05, 0.5, 30.0,
                        ])

            # native gate rows: half the prompts walk the modal path (and then
            # carry byte-identical values), half walk something else.
            for seed in (41, 42, 43):
                fixed = {
                    (seed, prompt): value
                    for (_m, name, payload, _k, s, prompt, value, *_rest) in
                    [tuple(r) for r in rows_spx]
                    if name == "seacache_top1" and payload == "reuse"
                    and s == seed and _m == model
                    for value in [value]
                }
                other = "1" * 29 + "0" * 21
                for prompt in range(N_PROMPTS):
                    on_modal = prompt % 2 == 0
                    value = fixed[(seed, prompt)] if on_modal else fixed[(seed, prompt)] + 0.5
                    rows_native.append([
                        "seacache", budget_k, seed, prompt, value, 0.9, 0.05, 0.5, 30.0,
                        modal_bits[(model, budget_k, "seacache")] if on_modal else other,
                    ])

        write_tsv_gz(
            tmp_path / f"perprompt_spx_{model}.tsv.gz",
            ["schedule", "payload", "k", "seed", "prompt_idx", *METRICS],
            [row[1:] for row in rows_spx if row[0] == model],
        )
        write_tsv_gz(
            tmp_path / f"perprompt_native_{model}.tsv.gz",
            ["method", "k", "seed", "prompt_idx", *METRICS, "path"],
            rows_native,
        )
        rows_native = []

    monkeypatch.setattr(sup, "SPX_DIR", tmp_path)
    return {"dir": tmp_path, "schedules": schedules, "modal": modal_bits}


# ----- loaders ---------------------------------------------------------------


def test_staged_tables_round_trip(staged):
    spx = sup.load_spx("flux")
    assert ("budcache", "reuse", 29, 41) in spx
    assert len(spx[("budcache", "reuse", 29, 41)]) == N_PROMPTS
    native = sup.load_native("flux")
    assert ("seacache", 29, 41) in native
    entry = native[("seacache", 29, 41)][0]
    assert entry["path"] and len(entry["path"]) == 50


def test_frozen_schedules_and_manifests_load_from_the_repo():
    schedules = sup.load_schedules()
    for model in sup.MODELS:
        for budget_k in sup.BUDGETS:
            assert (model, budget_k, "meancache") in schedules
            assert (model, budget_k, "rand_1") in schedules
            assert (model, budget_k, "dp_rho2") in schedules
    manifest = sup.load_manifest_records()
    assert ("flux", 29, "rand_1") in manifest
    assert ("flux", 29, "budcache") in manifest


# ----- P1 --------------------------------------------------------------------


def test_formal_p1_uses_the_balanced_grid_only(staged):
    spx = sup.load_spx("flux")
    table = {}
    for schedule in sup.W1:
        for payload in sup.PAYLOADS:
            values = sup.cell_series(
                spx, schedule=schedule, payload=payload, budget_k=29,
                metric="psnr", seeds=(41, 42, 43), prompts=None,
            )
            table[(schedule, payload)] = sup.summarise(values)["value"]
    assert len(table) == 20
    fit = sup.variance_shares(table)
    assert fit["n_obs"] == 20
    assert fit["residual_dof"] == 20 - (1 + 3 + 4)
    total = sum(v for v in fit["share"].values())
    assert abs(total - 1.0) < 1e-9
    diagonal = sup.diagonal_gammas(fit, sup.W1)
    # budcache/reuse, dpcache/hermite, uniform/{taylor,hermite}, dicache/di
    assert len(diagonal) == 5


def test_a_single_column_row_cannot_carry_an_identified_gamma(staged):
    """A row observed in one column has its advantage absorbed by beta."""
    spx = sup.load_spx("flux")
    table = {}
    for schedule in sup.W1:
        for payload in sup.PAYLOADS:
            values = sup.cell_series(spx, schedule=schedule, payload=payload,
                                     budget_k=29, metric="psnr",
                                     seeds=(41, 42, 43), prompts=None)
            table[(schedule, payload)] = sup.summarise(values)["value"]
    values = sup.cell_series(spx, schedule="seacache_top1", payload="reuse",
                             budget_k=29, metric="psnr", seeds=(41, 42, 43), prompts=None)
    ragged = dict(table)
    ragged[("seacache_top1", "reuse")] = sup.summarise(values)["value"]
    fit = sup.variance_shares(ragged)
    assert abs(fit["gamma_by_cell"]["seacache_top1xreuse"]) < 1e-9


# ----- P4 --------------------------------------------------------------------


def test_modal_pairs_are_a_deterministic_zero_and_leave_the_offmodal_mean(staged):
    spx = sup.load_spx("flux")
    native = sup.load_native("flux")
    fixed = sup.cell_series(spx, schedule="seacache_top1", payload="reuse",
                            budget_k=29, metric="psnr", seeds=(41, 42, 43), prompts=None)
    result = sup.p4_gate(
        gate="seacache",
        fixed_series=fixed,
        native=native,
        modal_bits=staged["modal"][("flux", 29, "seacache")],
        metric="psnr",
        seeds=(41, 42, 43),
        native_by_seed={seed: native[("seacache", 29, seed)] for seed in (41, 42, 43)},
        prompts=None,
    )
    assert result["n_modal"] + result["n_offmodal"] == result["n_pairs"]
    assert result["modal_max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert result["modal"]["mean"] == pytest.approx(0.0, abs=1e-12)
    # The fixture makes the gate 0.5 dB better off-modal, so the fixed path is
    # 0.5 dB worse there -- and the all-prompt mean is that, halved.
    assert result["offmodal"]["mean"] == pytest.approx(-0.5, abs=1e-9)
    assert result["all"]["mean"] == pytest.approx(-0.25, abs=1e-9)
    assert result["modal_mass"] == pytest.approx(0.5, abs=1e-9)
    # The fixture's off-modal offset is a constant, so the paired SD -- and
    # therefore the band -- is exactly zero. That is the right answer: a paired
    # band measures spread of the *difference*, not of either run.
    assert result["offmodal"]["sd"] == pytest.approx(0.0, abs=1e-9)
    assert result["offmodal"]["ci_low"] == pytest.approx(result["offmodal"]["mean"])


def test_paired_band_is_two_paired_standard_errors(staged):
    left = {(41, i): float(i) for i in range(100)}
    right = {(41, i): float(i) - 1.0 for i in range(100)}
    entry = sup.paired_difference(left, right)
    assert entry["mean"] == pytest.approx(1.0)
    assert entry["sd"] == pytest.approx(0.0, abs=1e-12)
    assert entry["n_pairs"] == 100


# ----- geometry --------------------------------------------------------------


@pytest.fixture
def synthetic_geometry():
    return {"sigmas": np.linspace(1.0, 0.02, 50),
            "rho2": np.linspace(0.5, 2.0, 50), "n_rows": N_PROMPTS, "window": 5}


@pytest.fixture
def synthetic_discovery(staged):
    return {
        (model, 29, gate): [
            {"schedule": staged["schedules"][(model, 29, name)],
             "cache_count": 29, "mass": mass, "count": count}
            for name, mass, count in (("seacache_top1", 0.6, 24), ("budcache", 0.4, 16))
        ]
        for model in sup.MODELS for gate in sup.GATES
    }


def test_geometry_predictors_are_all_computable_on_a_synthetic_row(
    staged, synthetic_geometry, synthetic_discovery
):
    schedules = staged["schedules"]
    geometry = synthetic_geometry
    anchors = {
        "meancache": schedules[("flux", 29, "meancache")],
        "budcache": schedules[("flux", 29, "budcache")],
    }
    discovery = synthetic_discovery
    distributions = {}
    universe = []
    for gate in sup.GATES:
        entries = discovery[("flux", 29, gate)]
        total = sum(e["mass"] for e in entries)
        distributions[gate] = {e["schedule"]: e["mass"] / total for e in entries}
        universe.extend(e["schedule"] for e in entries)
    out = sup.geometry_predictors(
        schedules[("flux", 29, "dp_rho2")],
        geometry=geometry,
        anchors=anchors,
        gate_distributions=distributions,
        universe=sorted(set(universe)),
    )
    for key in sup.PREDICTOR_KEYS:
        assert key in out, key
        assert out[key] is not None, key
    assert out["cache_count"] == 29
    assert out["hamming_to_meancache"] == 2 * out["transpositions_to_meancache"]


def test_pooled_geometry_ranks_within_partitions():
    """A predictor that orders rows the same way inside every partition should
    pool to a strong positive rho even when the partitions sit at different
    absolute levels."""
    partitions = []
    for offset in (0.0, 100.0, -50.0):
        rows = [
            {"row": f"r{i}", "quality": offset + i, "first_cache_step": i}
            for i in range(6)
        ]
        partitions.append({"rows": rows})
    pooled = sup.pooled_geometry(partitions)
    assert pooled["first_cache_step"]["rho"] == pytest.approx(1.0, abs=1e-9)


def test_off_budget_rows_stay_out_of_the_geometry_regression(
    staged, synthetic_geometry, synthetic_discovery
):
    """A 30-step row on a 29-step quality axis is not a schedule comparison."""
    schedules = dict(staged["schedules"])
    off = ["1"] * 30 + ["0"] * 20
    schedules[("flux", 29, "sencache_top1_off")] = "".join(off)
    panel = {"cells": {"psnr": {
        "budcachexreuse": {"value": 30.0},
        "sencache_top1_offxreuse": {"value": 20.0},
    }}}
    block = sup.partition_geometry(
        model="flux", budget_k=29, panel=panel, schedules=schedules,
        geometry=synthetic_geometry, discovery=synthetic_discovery,
        metric="psnr",
    )
    assert [row["row"] for row in block["rows"]] == ["budcache"]


# ----- densification: a rung is a distribution, not a point ------------------


def _entry(row: str, mean: float) -> dict:
    return {"row": row,
            "paired_vs_meancache": {"mean": mean, "ci_low": mean - 0.02,
                                    "ci_high": mean + 0.02, "n_pairs": 1632}}


def test_a_rung_reports_the_median_draw_and_the_range_over_draws():
    by_row = {name: _entry(name, mean) for name, mean in (
        ("ham2f", -0.46), ("ham2f_d2", -1.30), ("ham2f_d3", -0.80),
        ("ham4f", -1.07),
    )}
    ladder = sup.build_ladder(by_row, {2: ("ham2f", "ham2f_d2", "ham2f_d3"),
                                       4: ("ham4f", "ham4f_d2", "ham4f_d3")})
    rungs = {entry["hamming"]: entry for entry in ladder}
    assert rungs[0]["row"] == sup.ANCHOR_ROW
    near = rungs[2]
    assert near["n_draws"] == 3
    assert near["row"] == "ham2f_d3"          # the median draw, not the frozen one
    assert near["delta"] == pytest.approx(-0.80)
    assert (near["delta_min"], near["delta_max"]) == pytest.approx((-1.30, -0.46))
    assert near["delta_range"] == pytest.approx(0.84)
    # The paired interval belongs to the median draw only: the two kinds of
    # uncertainty must not be blended into one number.
    assert near["ci_low"] == pytest.approx(-0.82)
    # A rung with one draw reads exactly as it did before the densification.
    far = rungs[4]
    assert far["n_draws"] == 1
    assert far["delta"] == far["delta_min"] == far["delta_max"] == pytest.approx(-1.07)


def test_a_family_with_no_rows_is_absent_rather_than_a_lone_anchor():
    """The free ladder exists only where it was densified."""

    assert sup.build_ladder({}, sup.LADDER_RUNGS["free"]) == []


def test_every_ladder_row_belongs_to_exactly_one_rung_of_one_family():
    seen = [row for family in sup.LADDER_RUNGS.values()
            for rows in family.values() for row in rows]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(sup.LADDER_ROWS)
    # The two families must not share a row, or the two dose curves would be
    # partly the same measurement.
    preserving = {r for rows in sup.LADDER_RUNGS["preserving"].values() for r in rows}
    free = {r for rows in sup.LADDER_RUNGS["free"].values() for r in rows}
    assert preserving.isdisjoint(free)
    assert all(name.endswith("f") or "f_d" in name for name in preserving)


def test_the_random_reference_carries_its_own_spread(
    staged, synthetic_geometry, synthetic_discovery, monkeypatch
):
    # All measurements in this test are generated by the staged fixture.
    monkeypatch.setattr(sup, "BUDGETS", (29,))
    monkeypatch.setattr(sup, "load_schedules", lambda: staged["schedules"])
    monkeypatch.setattr(sup, "load_geometry", lambda model: synthetic_geometry)
    monkeypatch.setattr(sup, "load_discovery_counts", lambda: synthetic_discovery)
    monkeypatch.setattr(sup, "load_matrix_cells", lambda: {})
    report = sup.build_report(["all"])
    assert len(report["splits"]["all"]["random_reference"]) == len(sup.MODELS)
    for entry in report["splits"]["all"]["random_reference"]:
        assert entry["n_random"] >= 1
        assert entry["random_min"] <= entry["random_mean"] <= entry["random_max"]
        assert entry["random_spread"] == pytest.approx(
            entry["random_max"] - entry["random_min"])
        # The conservative reading of the gain can only be smaller than the
        # mean reading, never larger.
        assert entry["design_gain_vs_best_random"] <= entry["design_gain_over_random"]
        assert entry["design_gain_vs_worst_random"] >= entry["design_gain_over_random"]
