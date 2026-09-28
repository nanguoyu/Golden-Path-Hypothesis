"""Selection over sweep families, and the cell tables built from it.

The bug this guards: an n-ablation family was swept at its own run limit and
needs its own `threshold_main` to hold K29. Selecting per budget rather than per
family silently gave those cells the canonical family's pair, so the ablation
would have measured nothing.

Plan: docs/sencache_recalibration_plan_zh.md sections 9.1 and 9.1.1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    # family: (budget, max_skip, switch_ratio, {main: mean K})
    "k29": (29, 10, 0.2, {0.30: 24.1, 0.60: 29.05, 1.05: 33.2}),
    "k37": (37, 39, 0.2, {0.60: 30.4, 3.60: 36.90, 12.0: 38.8}),
    "k41": (41, 43, 0.12, {0.60: 33.1, 9.40: 41.12, 22.0: 42.9}),
    # each ablation rung needs a different main to land on the same budget
    "k29_n3": (29, 3, 0.2, {0.60: 26.2, 1.40: 29.18, 3.60: 29.9}),
    "k29_n20": (29, 20, 0.2, {0.30: 25.0, 0.45: 28.88, 1.05: 34.7}),
    "k29_n39": (29, 39, 0.2, {0.30: 26.6, 0.40: 29.10, 1.05: 35.9}),
}
GROUPS = ("flux", "qwen")


def _frontier_tsv(path: Path) -> None:
    header = ["group", "family", "budget", "max_skip", "switch_ratio", "start",
              "main", "admissible", "n_prompts", "k_mean", "k_std", "k_min",
              "k_max", "exact_29", "exact_37", "exact_41", "longest_cached_run",
              "strict_window", "strict_cache_mean",
              "first10_cache_mean", "first10_unique_patterns",
              "first10_modal_pattern", "first10_modal_mass"]
    lines = ["\t".join(header)]
    for group in GROUPS:
        for family, (budget, max_skip, switch, curve) in FAMILIES.items():
            for start in (0.005, 0.01):
                for main, mean in curve.items():
                    # the stricter start is admissible, so selection must take it
                    shifted = mean if start == 0.005 else mean + 1.5
                    lines.append("\t".join((
                        group, family, str(budget), str(max_skip), f"{switch:g}",
                        f"{start:g}", f"{main:g}", "1", "128", f"{shifted:.4f}",
                        "0.5000", "20", "40", "0", "0", "0", str(max_skip),
                        str(int(round(50 * switch))), "0.0000",
                        "0.0000", "1", "0" * 10, "1.0000")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def selection(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("sel")
    _frontier_tsv(tmp / "frontier.tsv")
    subprocess.run(
        [sys.executable, "analysis/sencache_frontier.py", "select",
         "--frontier", str(tmp / "frontier.tsv"),
         "--out_json", str(tmp / "selection.json")],
        cwd=_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads((tmp / "selection.json").read_text())
    payload["_dir"] = str(tmp)
    return payload


def test_every_family_resolves(selection: dict) -> None:
    for group in GROUPS:
        for family in FAMILIES:
            entry = selection["groups"][group]["budgets"][family]
            assert entry["frozen"] is not None, f"{group} {family}"


def test_each_family_keeps_its_own_pair(selection: dict) -> None:
    """The ablation rungs must not inherit the canonical K29 threshold."""
    for group in GROUPS:
        budgets = selection["groups"][group]["budgets"]
        canonical = budgets["k29"]["frozen"]["main"]
        for family in ("k29_n3", "k29_n20", "k29_n39"):
            assert budgets[family]["frozen"]["main"] != canonical, family


def test_each_family_keeps_its_own_run_limit(selection: dict) -> None:
    for group in GROUPS:
        budgets = selection["groups"][group]["budgets"]
        assert [budgets[f]["max_skip"] for f in
                ("k29", "k29_n3", "k29_n20", "k29_n39")] == [10, 3, 20, 39]


def test_the_strictest_admissible_start_is_taken(selection: dict) -> None:
    for group in GROUPS:
        for family in FAMILIES:
            assert selection["groups"][group]["budgets"][family]["frozen"]["start"] == 0.005


def test_realized_k_lands_inside_the_tolerance(selection: dict) -> None:
    for group in GROUPS:
        for family, (budget, *_rest) in FAMILIES.items():
            entry = selection["groups"][group]["budgets"][family]
            assert abs(entry["frozen"]["k_mean"] - budget) <= selection["tolerance"]


def test_canonical_families_are_reachable_by_budget_key(selection: dict) -> None:
    """The Wan config refreeze looks cells up as K29/K37/K41."""
    for group in GROUPS:
        budgets = selection["groups"][group]["budgets"]
        for key, family in (("K29", "k29"), ("K37", "k37"), ("K41", "k41")):
            assert budgets[key] == budgets[family]


def test_the_cell_table_carries_each_family_pair(selection: dict) -> None:
    tmp = Path(selection["_dir"])
    subprocess.run(
        [sys.executable, "analysis/build_sencache_recal_cells.py",
         "--selection", str(tmp / "selection.json"),
         "--out_tsv", str(tmp / "cells.tsv")],
        cwd=_ROOT, check=True, capture_output=True, text=True)
    rows = [line.split("\t") for line in
            (tmp / "cells.tsv").read_text().splitlines()[1:]]
    header_main, header_n = 6, 7  # main, max_skip in the cell table
    by_family: dict[str, set] = {}
    for row in rows:
        by_family.setdefault(row[1], set()).add((row[header_main], row[header_n]))
    # one pair per family, and no two families share one
    assert all(len(v) == 1 for v in by_family.values()), by_family
    assert len(set().union(*by_family.values())) == len(by_family)
    main_cells = sum(1 for row in rows if row[1] in ("k29", "k37", "k41"))
    assert main_cells == 72          # 2 models x 4 datasets x 3 budgets x 3 seeds
    assert len(rows) - main_cells == 18   # 2 models x 3 rungs x 3 seeds
