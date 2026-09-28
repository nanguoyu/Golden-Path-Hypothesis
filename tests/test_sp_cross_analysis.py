from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import sp_cross


def _write_cell(
    root: Path,
    *,
    model: str,
    budget_k: int,
    schedule: str,
    payload: str,
    seed: int,
    metric: str,
    value: float,
    n_pairs: int = 1632,
) -> Path:
    cell = root / model / f"k{budget_k}" / f"{schedule}x{payload}_s{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "metrics.json").write_text(
        json.dumps(
            {
                "n_pairs": n_pairs,
                "summary": {
                    metric: {"mean": value, "std": 0.5, "n": n_pairs},
                },
            }
        ),
        encoding="utf-8",
    )
    return cell


def test_parse_cell_name_roundtrips_every_axis_combination() -> None:
    for schedule in sp_cross.HOMOLOGOUS:
        for payload in sp_cross.PAYLOADS:
            name = f"{schedule}x{payload}_s100042"
            assert sp_cross.parse_cell_name(name) == (schedule, payload, 100042)


@pytest.mark.parametrize(
    "name",
    ["budcachexnope_s42", "budcache_s42", "budcachexreuse", "budcachexreuse_sxx", "xreuse_s42"],
)
def test_parse_cell_name_rejects_foreign_directories(name: str) -> None:
    assert sp_cross.parse_cell_name(name) is None


def test_discover_cells_reads_the_matrix_metrics_layout(tmp_path: Path) -> None:
    _write_cell(
        tmp_path,
        model="flux",
        budget_k=41,
        schedule="dpcache",
        payload="hermite_o2",
        seed=42,
        metric="psnr",
        value=27.5,
    )
    _write_cell(
        tmp_path,
        model="qwen",
        budget_k=29,
        schedule="uniform",
        payload="reuse",
        seed=100042,
        metric="psnr",
        value=31.25,
    )
    (tmp_path / "flux" / "k41" / "not_a_cell").mkdir(parents=True)
    (tmp_path / "flux" / "notes").mkdir(parents=True)

    cells = sp_cross.discover_cells(tmp_path, metric="psnr")
    assert [(c["model"], c["budget_k"], c["schedule"], c["payload"], c["seed"], c["value"])
            for c in cells] == [
        ("flux", 41, "dpcache", "hermite_o2", 42, 27.5),
        ("qwen", 29, "uniform", "reuse", 100042, 31.25),
    ]
    assert cells[0]["n_pairs"] == 1632


def test_lpips_cells_are_oriented_so_larger_is_better() -> None:
    assert sp_cross.orient(0.25, "lpips") == -0.25
    assert sp_cross.orient(0.25, "psnr") == 0.25
    cells = [
        {
            "model": "flux",
            "budget_k": 41,
            "schedule": "budcache",
            "payload": "reuse",
            "seed": 42,
            "value": 0.25,
        }
    ]
    grouped = sp_cross.group_cells(cells, metric="lpips")
    assert grouped[("flux", 41)][("budcache", "reuse")]["value"] == -0.25


def test_group_cells_averages_seeds_and_reports_spread() -> None:
    cells = [
        {
            "model": "flux",
            "budget_k": 37,
            "schedule": "budcache",
            "payload": "reuse",
            "seed": seed,
            "value": value,
        }
        for seed, value in ((41, 30.0), (42, 31.0), (43, 32.0))
    ]
    entry = sp_cross.group_cells(cells, metric="psnr")[("flux", 37)][("budcache", "reuse")]
    assert entry["value"] == pytest.approx(31.0)
    assert entry["n_seeds"] == 3
    assert entry["seed_sd"] == pytest.approx(1.0)


def _planted_grid() -> tuple[dict[tuple[str, str], float], dict, dict, float]:
    mu = 30.0
    alpha = {
        "budcache": 0.5,
        "meancache": -0.25,
        "dpcache": 0.75,
        "uniform": -1.0,
    }
    beta = {"reuse": 0.4, "taylor_o1": -0.1, "hermite_o2": -0.3}
    gamma = {
        ("budcache", "reuse"): 0.2,
        ("budcache", "taylor_o1"): -0.1,
        ("budcache", "hermite_o2"): -0.1,
        ("meancache", "reuse"): -0.1,
        ("meancache", "taylor_o1"): 0.2,
        ("meancache", "hermite_o2"): -0.1,
        ("dpcache", "reuse"): -0.1,
        ("dpcache", "taylor_o1"): -0.1,
        ("dpcache", "hermite_o2"): 0.2,
        ("uniform", "reuse"): 0.0,
        ("uniform", "taylor_o1"): 0.0,
        ("uniform", "hermite_o2"): 0.0,
    }
    matrix = {
        (schedule, payload): mu + alpha[schedule] + beta[payload] + gamma[(schedule, payload)]
        for schedule in alpha
        for payload in beta
    }
    return matrix, alpha, beta, mu


def test_decompose_recovers_planted_effects_on_a_complete_grid() -> None:
    matrix, alpha, beta, mu = _planted_grid()
    result = sp_cross.decompose(matrix)
    assert result["mu"] == pytest.approx(mu)
    for name, value in alpha.items():
        assert result["alpha"][name] == pytest.approx(value)
    for name, value in beta.items():
        assert result["beta"][name] == pytest.approx(value)
    assert result["n_obs"] == 12
    assert result["n_params"] == 1 + 3 + 2
    assert result["residual_dof"] == 6
    assert sum(result["alpha"].values()) == pytest.approx(0.0)
    assert sum(result["beta"].values()) == pytest.approx(0.0)


def test_decompose_matches_the_classical_means_decomposition() -> None:
    matrix = {
        (schedule, payload): float(3 * s_index + p_index) ** 1.7
        for s_index, schedule in enumerate(("a", "b", "c"))
        for p_index, payload in enumerate(("x", "y", "z", "w"))
    }
    result = sp_cross.decompose(matrix)
    grand = sum(matrix.values()) / len(matrix)
    for schedule in ("a", "b", "c"):
        row = [value for (s, _), value in matrix.items() if s == schedule]
        assert result["alpha"][schedule] == pytest.approx(sum(row) / len(row) - grand)
    for payload in ("x", "y", "z", "w"):
        column = [value for (_, p), value in matrix.items() if p == payload]
        assert result["beta"][payload] == pytest.approx(sum(column) / len(column) - grand)
    for (schedule, payload), value in matrix.items():
        row = [v for (s, _), v in matrix.items() if s == schedule]
        column = [v for (_, p), v in matrix.items() if p == payload]
        expected = value - sum(row) / len(row) - sum(column) / len(column) + grand
        assert result["gamma_pairs"][(schedule, payload)] == pytest.approx(expected)


def test_decompose_rejects_a_rank_deficient_design() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        sp_cross.decompose({("a", "x"): 1.0, ("b", "y"): 2.0})


@pytest.mark.parametrize(
    ("n_pos", "n_total", "expected"),
    [(10, 10, 2 / 1024), (9, 9, 2 / 512), (5, 10, 1.0), (0, 0, None)],
)
def test_two_sided_sign_p_is_the_exact_binomial(n_pos, n_total, expected) -> None:
    result = sp_cross.two_sided_sign_p(n_pos, n_total)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_p1_reads_gamma_on_the_homologous_cells() -> None:
    matrix, _alpha, _beta, _mu = _planted_grid()
    result = sp_cross.p1_diagonal(sp_cross.decompose(matrix))
    diagonal = {(row["schedule"], row["payload"]) for row in result["cells"]}
    assert diagonal == {
        ("budcache", "reuse"),
        ("meancache", "mean_avg_vel"),
        ("dpcache", "hermite_o2"),
        ("uniform", "taylor_o1"),
        ("uniform", "hermite_o2"),
    } - {("meancache", "mean_avg_vel")}
    assert result["n_diagonal"] == 4
    assert result["n_positive"] == 2
    # the two structurally-absorbed cells come back as ~1e-14 solver noise;
    # the tolerance keeps them out of the trial count, so the sign test sees
    # 2 positives in 2 trials -- BLAS-independent, unlike the old expectation
    # of p = 1.0 which relied on the noise signs landing non-positive
    assert result["n_nonzero"] == 2
    assert result["mean_gamma"] == pytest.approx((0.2 + 0.2 + 0.0 + 0.0) / 4)
    assert result["sign_test_p_two_sided"] == pytest.approx(0.5)


def test_gpf_cells_sit_on_the_diagonal() -> None:
    """Plan section 8.2b freezes gpf members constructed FOR one payload order;
    section 2's diagonal definition covers them, and the fix this pins: they
    were classified off-diagonal, polluting the off-diagonal mean with cells
    the plan calls homologous."""
    schedules = ("budcache", "gpf_reuse_e05_1", "gpf_o1_e15_1", "gpf_o1_e20_1")
    payloads = ("reuse", "taylor_o1", "hermite_o2")
    matrix = {(s_, p_): 30.0 + (1.0 if p_ in sp_cross.HOMOLOGOUS[s_] else 0.0)
              for s_ in schedules for p_ in payloads}
    result = sp_cross.p1_diagonal(sp_cross.decompose(matrix))
    diagonal = {(row["schedule"], row["payload"]) for row in result["cells"]}
    assert ("gpf_reuse_e05_1", "reuse") in diagonal
    assert ("gpf_o1_e15_1", "taylor_o1") in diagonal
    assert ("gpf_o1_e20_1", "taylor_o1") in diagonal


def test_p1_detects_a_uniformly_positive_diagonal() -> None:
    schedules = ("budcache", "meancache", "dpcache", "uniform", "dicache_top1")
    payloads = ("reuse", "mean_avg_vel", "hermite_o2", "taylor_o1", "di_two_anchor")
    matrix = {}
    for schedule in schedules:
        for payload in payloads:
            bonus = 1.0 if payload in sp_cross.HOMOLOGOUS[schedule] else 0.0
            matrix[(schedule, payload)] = 30.0 + bonus
    result = sp_cross.p1_diagonal(sp_cross.decompose(matrix))
    assert result["n_diagonal"] == 6
    assert result["n_positive"] == 6
    assert result["mean_gamma"] > 0
    assert result["mean_offdiagonal_gamma"] < 0
    assert result["sign_test_p_two_sided"] == pytest.approx(2 / 64)


def test_p2_checks_the_pre_registered_order_and_row_argmax() -> None:
    matrix = {
        ("dpcache", "hermite_o2"): 31.0,
        ("dpcache", "reuse"): 30.0,
        ("dicache_top1", "di_two_anchor"): 29.0,
        ("dicache_top1", "taylor_o1"): 30.0,
        ("dicache_top1", "reuse"): 28.0,
        ("budcache", "reuse"): 33.0,
        ("budcache", "taylor_o1"): 32.0,
    }
    result = sp_cross.p2_order(matrix)
    holds = {
        (row["schedule"], row["better"], row["worse"]): (row["holds"], row["delta"])
        for row in result["ordered_pairs"]
    }
    assert holds[("dpcache", "hermite_o2", "reuse")] == (True, pytest.approx(1.0))
    assert holds[("dicache_top1", "di_two_anchor", "taylor_o1")] == (False, pytest.approx(-1.0))
    assert holds[("dicache_top1", "taylor_o1", "reuse")] == (True, pytest.approx(2.0))
    assert result["row_argmax"] == [
        {
            "schedule": "budcache",
            "expected_argmax": "reuse",
            "observed_argmax": "reuse",
            "holds": True,
            "row": {"reuse": 33.0, "taylor_o1": 32.0},
        }
    ]


def test_p3_ranks_zero_order_first_at_high_compression() -> None:
    matrix = {}
    for schedule in ("budcache", "seacache_top1", "uniform"):
        for payload, value in (("reuse", 30.0), ("taylor_o1", 29.0), ("hermite_o2", 28.0)):
            matrix[(schedule, payload)] = value
    matrix[("uniform", "hermite_o2")] = 31.0  # one row breaks the pattern
    result = sp_cross.p3_high_compression(matrix)
    assert result["median_rank"] == {"reuse": 1.0, "taylor_o1": 2.0, "hermite_o2": 3.0}
    assert result["median_order_holds"] is True
    assert len(result["rows"]) == 3


def test_p4_compares_modal_cells_against_a_native_reference(tmp_path: Path) -> None:
    reference_path = tmp_path / "native.tsv"
    reference_path.write_text(
        "model\tbudget_k\tmethod\tmetric\tvalue\n"
        "flux\t41\tseacache\tpsnr\t30.0\n"
        "flux\t41\tdicache\tpsnr\t28.0\n",
        encoding="utf-8",
    )
    table = sp_cross.read_reference(reference_path)
    assert table[("flux", "seacache", 41, "psnr")] == 30.0

    entries = {
        ("seacache_top1", "reuse"): {"value": 30.05, "n_seeds": 3, "seed_sd": 0.1},
        ("dicache_top1", "di_two_anchor"): {"value": 26.0, "n_seeds": 3, "seed_sd": 0.1},
        ("dicache_top1", "reuse"): {"value": 27.9, "n_seeds": 3, "seed_sd": 0.1},
    }
    band = sp_cross.pooled_seed_band(entries)
    assert band == pytest.approx(0.2)
    result = sp_cross.p4_modal_capacity(
        entries,
        metric="psnr",
        reference={
            (method, metric): value
            for (model, method, budget_k, metric), value in table.items()
            if model == "flux" and budget_k == 41
        },
        pooled_band=band,
    )
    rows = {row["schedule"]: row for row in result["rows"]}
    assert rows["seacache_top1"]["native"]["within_band"] is True
    assert rows["dicache_top1"]["native_payload_name"] == "di_two_anchor"
    assert rows["dicache_top1"]["native"]["within_band"] is False
    assert rows["dicache_top1"]["reuse"]["within_band"] is True
    assert rows["teacache_top1"]["native"] is None


def test_analyse_end_to_end_on_synthetic_cells(tmp_path: Path) -> None:
    schedules = ("budcache", "meancache", "dpcache", "uniform")
    payloads = ("reuse", "mean_avg_vel", "hermite_o2")
    for schedule in schedules:
        for payload in payloads:
            for seed, offset in ((41, -0.1), (42, 0.0), (43, 0.1)):
                bonus = 0.5 if payload in sp_cross.HOMOLOGOUS[schedule] else 0.0
                _write_cell(
                    tmp_path,
                    model="flux",
                    budget_k=41,
                    schedule=schedule,
                    payload=payload,
                    seed=seed,
                    metric="psnr",
                    value=30.0 + bonus + offset,
                )
    cells = sp_cross.discover_cells(tmp_path, metric="psnr")
    assert len(cells) == 36
    report = sp_cross.analyse(cells, metric="psnr", high_k=41, reference_table={})
    assert len(report["panels"]) == 1
    panel = report["panels"][0]
    assert panel["model"] == "flux"
    assert panel["budget_k"] == 41
    assert panel["n_cells"] == 12
    assert panel["decomposition"]["residual_dof"] == 6
    assert panel["pooled_seed_band"] == pytest.approx(2 * 0.1)
    # budcache x reuse, meancache x mean_avg_vel, dpcache x hermite_o2 and the
    # uniform x hermite_o2 half of the shared-predictor triplet.
    assert panel["P1_diagonal"]["n_positive"] == panel["P1_diagonal"]["n_diagonal"] == 4
    assert panel["P1_diagonal"]["mean_gamma"] > 0
    assert "P3_high_compression" in panel
    assert panel["cells"]["budcachexreuse"]["n_seeds"] == 3
    sp_cross.print_report(report)


W1_ERA_SCHEDULES = (
    "budcache", "meancache", "dpcache", "uniform",
    "seacache_top1", "teacache_top1", "sencache_top1", "dicache_top1",
)


def test_p1_excludes_confounded_cells_of_a_w1_shaped_panel() -> None:
    """The REPUDIATED W1 shape (all reuse cells + each homologous diagonal,
    method schedules only): each lone non-reuse payload appears at its own
    diagonal cell alone, so the planted bonus is absorbed. Those CELLS are
    excluded and listed; the panel is still scored on the identifiable rest
    (dpcache x hermite_o2, uniform x hermite_o2; budcache's row is reuse alone)."""

    matrix = {}
    for schedule in W1_ERA_SCHEDULES:
        matrix[(schedule, "reuse")] = 30.0 + 0.1 * W1_ERA_SCHEDULES.index(schedule)
        for payload in sp_cross.HOMOLOGOUS[schedule]:
            # A large planted diagonal bonus that the fit cannot attribute.
            matrix[(schedule, payload)] = matrix[(schedule, "reuse")] + 1.0
    result = sp_cross.p1_diagonal(sp_cross.decompose(matrix))
    assert result["structurally_identifiable"] is True
    assert result["n_diagonal"] == 9
    assert result["n_identifiable"] == 2
    assert "dpcachexhermite_o2" not in result["confounded_cells"]  # column has 2
    assert set(result["confounded_cells"]) == {
        "budcachexreuse",
        "meancachexmean_avg_vel",
        "uniformxtaylor_o1",
        "seacache_top1xreuse",
        "teacache_top1xreuse",
        "sencache_top1xreuse",
        "dicache_top1xdi_two_anchor",
    }
    by_cell = {(c["schedule"], c["payload"]): c for c in result["cells"]}
    assert by_cell[("seacache_top1", "reuse")]["row_residual_dof"] == 0
    assert by_cell[("seacache_top1", "reuse")]["identifiable"] is False
    assert by_cell[("uniform", "taylor_o1")]["column_residual_dof"] == 0
    assert by_cell[("dpcache", "hermite_o2")]["row_residual_dof"] == 1
    assert by_cell[("dpcache", "hermite_o2")]["column_residual_dof"] == 1
    assert by_cell[("dpcache", "hermite_o2")]["identifiable"] is True


def test_p1_scores_the_identifiable_subset_of_a_w1_plus_w1b_grid() -> None:
    """The real SPX shape per (model, K): W1 = 4 schedules x 5 payloads,
    dicache_top1 x 5, W1b = meancache x {reuse, mean_avg_vel} + sea/tea/sen_top1
    x reuse. Diagonal = 9 cells; the three single-cell rows are excluded
    (row_residual_dof 0), so 6 identifiable -- the qwen count; flux adds two
    5-payload gpf rows (8 identifiable). Both are pinned here."""

    def build(with_gpf: bool) -> dict[tuple[str, str], float]:
        full_rows = ["budcache", "dpcache", "uniform", "dicache_top1"]
        if with_gpf:
            full_rows += ["gpf_reuse_e05_1", "gpf_o1_e15_1"]
        matrix = {}
        for schedule in full_rows:
            for payload in sp_cross.PAYLOADS:
                bonus = 0.5 if payload in sp_cross.HOMOLOGOUS[schedule] else 0.0
                matrix[(schedule, payload)] = 30.0 + bonus + 0.01 * len(schedule)
        matrix[("meancache", "reuse")] = 30.1
        matrix[("meancache", "mean_avg_vel")] = 30.4
        for gate in ("seacache_top1", "teacache_top1", "sencache_top1"):
            matrix[(gate, "reuse")] = 33.0  # large, but structurally absorbed
        return matrix

    for with_gpf, n_ident in ((False, 6), (True, 8)):
        result = sp_cross.p1_diagonal(sp_cross.decompose(build(with_gpf)))
        assert result["n_diagonal"] == n_ident + 3
        assert result["n_identifiable"] == n_ident
        assert result["structurally_identifiable"] is True
        assert set(result["confounded_cells"]) == {
            "seacache_top1xreuse", "teacache_top1xreuse", "sencache_top1xreuse"
        }
        # the sign test never sees the absorbed cells: n_nonzero <= n_identifiable
        assert result["n_nonzero"] <= n_ident
        assert result["n_positive"] <= result["n_nonzero"]
        cells = {(c["schedule"], c["payload"]): c for c in result["cells"]}
        mean = cells[("meancache", "mean_avg_vel")]
        assert mean["identifiable"] is True
        assert mean["row_residual_dof"] == 1  # 2-cell row: pinned to -gamma[mean x reuse]
        assert mean["column_residual_dof"] >= 4
        pinned = sp_cross.decompose(build(with_gpf))["gamma_pairs"]
        assert mean["gamma"] == pytest.approx(-pinned[("meancache", "reuse")])
        # mean_gamma is over the identifiable subset only
        identifiable = [c["gamma"] for c in result["cells"] if c["identifiable"]]
        assert result["mean_gamma"] == pytest.approx(sum(identifiable) / n_ident)


def test_p1_is_identifiable_on_a_complete_grid() -> None:
    matrix = {}
    for schedule in sp_cross.HOMOLOGOUS:
        for payload in sp_cross.PAYLOADS:
            bonus = 0.5 if payload in sp_cross.HOMOLOGOUS[schedule] else 0.0
            matrix[(schedule, payload)] = 30.0 + bonus
    result = sp_cross.p1_diagonal(sp_cross.decompose(matrix))
    assert result["structurally_identifiable"] is True
    assert result["confounded_cells"] == []
    assert result["n_identifiable"] == result["n_diagonal"]
    assert result["n_positive"] == result["n_diagonal"]


def test_p3_reports_strict_and_tied_order_separately() -> None:
    matrix = {}
    for schedule in ("budcache", "uniform"):
        matrix[(schedule, "reuse")] = 30.0
        matrix[(schedule, "taylor_o1")] = 29.0
        matrix[(schedule, "hermite_o2")] = 28.0
    strict = sp_cross.p3_high_compression(matrix)
    assert strict["median_order_holds"] is True
    assert strict["strict_order_holds"] is True
    # swap taylor/hermite on one row: medians tie at 2.5, so <= holds, < does not
    matrix[("uniform", "taylor_o1")], matrix[("uniform", "hermite_o2")] = 28.0, 29.0
    tied = sp_cross.p3_high_compression(matrix)
    assert tied["median_rank"] == {"reuse": 1.0, "taylor_o1": 2.5, "hermite_o2": 2.5}
    assert tied["median_order_holds"] is True
    assert tied["strict_order_holds"] is False


def test_p4_emits_the_reuse_native_cell_once() -> None:
    entries = {
        ("seacache_top1", "reuse"): {"value": 30.05, "n_seeds": 3, "seed_sd": 0.1},
        ("dicache_top1", "di_two_anchor"): {"value": 28.1, "n_seeds": 3, "seed_sd": 0.1},
        ("dicache_top1", "reuse"): {"value": 26.9, "n_seeds": 3, "seed_sd": 0.1},
    }
    result = sp_cross.p4_modal_capacity(
        entries,
        metric="psnr",
        reference={("seacache", "psnr"): 30.0, ("dicache", "psnr"): 28.0},
        pooled_band=0.2,
    )
    rows = {row["schedule"]: row for row in result["rows"]}
    assert rows["seacache_top1"]["native"]["payload"] == "reuse"
    assert rows["seacache_top1"]["reuse"] is None  # not printed twice
    assert rows["dicache_top1"]["native"]["payload"] == "di_two_anchor"
    assert rows["dicache_top1"]["reuse"]["payload"] == "reuse"
    assert rows["dicache_top1"]["reuse"]["within_band"] is False


def _write_split_cell(
    root: Path, *, schedule: str, payload: str, seed: int, values: list[float]
) -> None:
    cell = root / "flux" / "k41" / f"{schedule}x{payload}_s{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "metrics.json").write_text(
        json.dumps(
            {
                "n_pairs": len(values),
                "indices": list(range(len(values))),
                "per_image": {"psnr": values},
                "summary": {"psnr": {"mean": sum(values) / len(values), "n": len(values)}},
            }
        ),
        encoding="utf-8",
    )


def test_split_restriction_equals_manual_masking(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "roles": {
                    "discovery": [0, 3, 6],
                    "validation": [1, 4],
                    "test": [2, 5, 7],
                }
            }
        ),
        encoding="utf-8",
    )
    assert sp_cross.load_split(splits_path, "all") is None
    assert sp_cross.load_split(splits_path, "validation") == frozenset({1, 4})
    assert sp_cross.load_split(splits_path, "heldout") == frozenset({1, 2, 4, 5, 7})

    root = tmp_path / "root"
    raw = {}
    for schedule in ("budcache", "uniform"):
        for payload in ("reuse", "taylor_o1"):
            values = [
                20.0 + 3.0 * i + (1.0 if payload == "reuse" else 0.0) + len(schedule) * 0.1
                for i in range(8)
            ]
            raw[(schedule, payload)] = values
            _write_split_cell(root, schedule=schedule, payload=payload, seed=42, values=values)

    all_cells = sp_cross.discover_cells(root, metric="psnr")
    heldout = sp_cross.load_split(splits_path, "heldout")
    held_cells = sp_cross.discover_cells(root, metric="psnr", prompt_indices=heldout)
    assert len(all_cells) == len(held_cells) == 4
    for cell in held_cells:
        values = raw[(cell["schedule"], cell["payload"])]
        expected = [values[i] for i in sorted(heldout)]
        assert cell["value"] == pytest.approx(sum(expected) / len(expected))
        assert cell["n_pairs"] == 5
    for cell in all_cells:
        values = raw[(cell["schedule"], cell["payload"])]
        assert cell["value"] == pytest.approx(sum(values) / len(values))
        assert cell["n_pairs"] == 8
    # the split changes the numbers, so a panel fitted on it differs from all
    report_all = sp_cross.analyse(all_cells, metric="psnr", high_k=41, reference_table={})
    report_held = sp_cross.analyse(held_cells, metric="psnr", high_k=41, reference_table={})
    mu_all = report_all["panels"][0]["decomposition"]["mu"]
    mu_held = report_held["panels"][0]["decomposition"]["mu"]
    manual_all = sum(sum(v) / 8 for v in raw.values()) / 4
    manual_held = sum(sum(v[i] for i in sorted(heldout)) / 5 for v in raw.values()) / 4
    assert mu_all == pytest.approx(manual_all)
    assert mu_held == pytest.approx(manual_held)


# ----- the supplement's two conventions ---------------------------------------


def _full_grid(tmp_path: Path, *, budget_k: int = 41) -> None:
    """W1's four rows x all five payloads, plus a one-column extra row."""
    for schedule in sp_cross.W1_ROWS:
        for payload in sp_cross.PAYLOADS:
            for seed, offset in ((41, -0.1), (42, 0.0), (43, 0.1)):
                bonus = 0.5 if payload in sp_cross.HOMOLOGOUS[schedule] else 0.0
                _write_cell(
                    tmp_path, model="flux", budget_k=budget_k, schedule=schedule,
                    payload=payload, seed=seed, metric="psnr",
                    value=30.0 + bonus + offset,
                )
    for seed, offset in ((41, -0.1), (42, 0.0), (43, 0.1)):
        _write_cell(
            tmp_path, model="flux", budget_k=budget_k, schedule="seacache_top1",
            payload="reuse", seed=seed, metric="psnr", value=27.0 + offset,
        )


def test_formal_p1_is_the_balanced_grid_and_the_ragged_fit_is_kept_beside_it(
    tmp_path: Path,
) -> None:
    _full_grid(tmp_path)
    cells = sp_cross.discover_cells(tmp_path, metric="psnr")
    panel = sp_cross.analyse(cells, metric="psnr", high_k=41, reference_table={})["panels"][0]
    assert panel["w1_cells_present"] == 20
    formal = panel["P1_diagonal_w1"]
    assert formal is not None
    # budcache/reuse, dpcache/hermite, uniform/{taylor, hermite}, dicache/di
    assert formal["n_diagonal"] == 5
    assert formal["n_positive"] == 5
    # The ragged panel carries the extra single-column row, so it is a
    # different fit on a different cell set -- which is the point of keeping
    # both: the numbers are not interchangeable.
    assert panel["P1_diagonal"]["n_diagonal"] >= formal["n_diagonal"]
    assert panel["n_cells"] == 21
    sp_cross.print_report(panel and {"metric": "psnr", "high_k": 41, "split": "all",
                                     "panels": [panel]})


def test_a_panel_without_the_full_balanced_grid_has_no_formal_p1(tmp_path: Path) -> None:
    for schedule in ("budcache", "dpcache"):
        for payload in ("reuse", "hermite_o2"):
            for seed in (41, 42, 43):
                _write_cell(
                    tmp_path, model="flux", budget_k=41, schedule=schedule,
                    payload=payload, seed=seed, metric="psnr", value=30.0,
                )
    cells = sp_cross.discover_cells(tmp_path, metric="psnr")
    panel = sp_cross.analyse(cells, metric="psnr", high_k=41, reference_table={})["panels"][0]
    assert panel["P1_diagonal_w1"] is None
    assert panel["w1_cells_present"] < 20


def test_p4_refuses_a_pooled_reference_on_a_prompt_subset(tmp_path: Path) -> None:
    """The shipped reference is a mean over all 1,632 prompts.

    Subtracting it from a held-out subset's mean compares two different prompt
    populations, which is a bias of the same order as the effect. On a split,
    P4 is not computed here at all.
    """
    _full_grid(tmp_path)
    cells = sp_cross.discover_cells(tmp_path, metric="psnr")
    reference = {("flux", "seacache", 41, "psnr"): 27.0}
    full = sp_cross.analyse(cells, metric="psnr", high_k=41,
                            reference_table=reference, split="all")["panels"][0]
    assert full["P4_modal_capacity"]["rows"]
    assert "skipped_reason" not in full["P4_modal_capacity"]
    held = sp_cross.analyse(cells, metric="psnr", high_k=41,
                            reference_table=reference, split="heldout")["panels"][0]
    assert held["P4_modal_capacity"]["rows"] == []
    assert "spx_supplement" in held["P4_modal_capacity"]["skipped_reason"]
