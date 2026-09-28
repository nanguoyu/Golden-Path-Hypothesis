#!/usr/bin/env python3
"""Density-form test of the geometric cost model against the SPX schedule axis.

Produces the table and the figure behind `docs/full_trajectory_results.md`
section 7. Both artefacts used to come from a throwaway session script, which
is how a row computed from a schedule file that was later edited
(`flux_k41_gpf_o1_e15_1.txt`, commit dc96b95) survived unnoticed in the stored
table; re-running this module rebuilds them from the files that are on disk now.

What the test asks
------------------
The geometric cost model `sum rho * (dsigma)^p` does NOT predict that full
steps land on high-rho steps. It predicts **cost equalisation**: a gap that
covers a high-rho stretch should be short in sigma, a gap over a low-rho
stretch should be long. So, per schedule, for every maximal run of cached
steps (a "gap") between two full steps:

    x = mean rho2 inside the gap
    y = sigma span from the anchor (the full step before the gap) to the
        full step that ends it
    c = the DP segment cost of that gap, sum_n rho2[n] * |sigma_anchor - sigma_n|

and report `spearman(x, y)` (cost equalisation predicts negative) plus the
coefficient of variation of `c` (a DP optimum of this cost should be low).

rho2 comes from `build_golden_path_family.risk_profiles`, i.e. the same
population profile the geometry-built schedules were constructed from. That is
why those schedules are a *positive control for the statistic only*: they are
DP optima of this very profile, so their behaviour says the test can see cost
equalisation when it is there, and says nothing about whether rho2 is the right
geometry.

Usage:
    python analysis/density_form_test.py                       # print only
    python analysis/density_form_test.py --write               # + tsv + figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.build_golden_path_family import read_population, risk_profiles

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "resources/full_trajectory/tables_jsonl/full_traj_flux_parti_full.jsonl"
SCHEDULES = REPO / "resources/sp_cross_schedules"
OUT_TSV = REPO / "resources/full_trajectory_analysis/a2_density_table.tsv"
OUT_FIG = REPO / "resources/full_trajectory_analysis/figures/flux_density_form_test.png"

NUM_STEPS = 50
KS = (29, 37, 41)
MIN_GAPS = 4  # below this a rank correlation over gaps is not worth reporting

# The rho2 profile carries one sharp feature, the state-18 kink reported in
# section 3.1 of the results doc. Every correlation below is a rank statistic
# over a handful of gaps, so a single feature can carry the ranking on its own.
# Flattening it and re-running is the check that says how much of the table is
# that one feature; the default span is the three steps the feature occupies.
SPIKE_SPAN = (17, 19)


def flatten_span(rho: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """`rho` with steps `lo..hi` replaced by the straight line through its two
    neighbours — the feature removed, the profile either side untouched."""
    if not 0 < lo <= hi < len(rho) - 1:
        raise SystemExit(f"--flatten-spike {lo} {hi} must leave a neighbour on each side "
                         f"of a profile of length {len(rho)}")
    out = rho.copy()
    out[lo:hi + 1] = np.interp(np.arange(lo, hi + 1), [lo - 1, hi + 1], [rho[lo - 1], rho[hi + 1]])
    return out

# (row label, file stem, family). The four gate rows are the rank-1 modal path
# of that gate on parti_full; see resources/sp_cross_schedules/README.md.
SCHEDULES_UNDER_TEST = [
    ("gpf_reuse_e05_1", "gpf_reuse_e05_1", "geometry-built"),
    ("gpf_o1_e15_1", "gpf_o1_e15_1", "geometry-built"),
    ("uniform", "uniform", "uniform-null"),
    ("meancache", "meancache", "searched"),
    ("budcache", "budcache", "searched"),
    ("dpcache", "dpcache", "searched"),
    ("seacache", "seacache_top1", "gate modal"),
    ("sencache", "sencache_top1", "gate modal"),
    ("teacache", "teacache_top1", "gate modal"),
    ("dicache", "dicache_top1", "gate modal"),
]


def gaps_of(bits: str) -> list[tuple[int, int]]:
    """Maximal runs of cached steps, as inclusive `(first, last)` step indices."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(bits)
    while i < n:
        if bits[i] == "1":
            j = i
            while j + 1 < n and bits[j + 1] == "1":
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def measure(bits: str, rho: np.ndarray, sigmas: np.ndarray) -> dict[str, float]:
    gaps = gaps_of(bits)
    local, span, cost = [], [], []
    for first, last in gaps:
        idx = np.arange(first, last + 1)
        anchor = sigmas[max(first - 1, 0)]
        local.append(float(rho[idx].mean()))
        span.append(abs(float(anchor - sigmas[last + 1])))
        cost.append(float(np.sum(rho[idx] * np.abs(anchor - sigmas[idx]))))
    local, span, cost = map(np.asarray, (local, span, cost))
    if len(gaps) >= 3:
        result = stats.spearmanr(local, span)
        rho_density, p = float(result.statistic), float(result.pvalue)
    else:
        rho_density = p = float("nan")
    return {
        "n_gaps": len(gaps),
        "rho_density": rho_density,
        "p": p,
        "cost_cv": float(np.std(cost) / np.mean(cost)),
        "local": local,
        "span": span,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the stored tsv and figure")
    ap.add_argument("--flatten-spike", nargs="?", const=f"{SPIKE_SPAN[0]}:{SPIKE_SPAN[1]}",
                    metavar="LO:HI",
                    help="linearly interpolate rho2 across these steps before measuring, to "
                         f"see how much of the table rests on that one feature "
                         f"(default span {SPIKE_SPAN[0]}:{SPIKE_SPAN[1]})")
    args = ap.parse_args()

    population = read_population(TABLE, num_steps=NUM_STEPS)
    profiles = risk_profiles(population)
    rho, sigmas = profiles.rho2, population.sigmas
    print(f"rho2 from {population.model}/{population.dataset}, "
          f"{population.n_rows} trajectories, window {profiles.rho2_window}")
    if args.flatten_spike:
        if args.write:
            # the archived table is the measurement, not the sensitivity check;
            # letting one overwrite the other is how a session artefact ends up
            # in the store with nothing to say which it is
            raise SystemExit("--flatten-spike is a diagnostic; it cannot be combined with --write")
        lo, hi = (int(v) for v in args.flatten_spike.split(":"))
        rho = flatten_span(rho, lo, hi)
        print(f"rho2 steps {lo}-{hi} flattened; this run is the sensitivity check, "
              f"not the archived table")

    rows: list[dict[str, object]] = []
    points: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    print(f"\n{'schedule':16s} {'K':>3s} {'gaps':>5s} {'spearman':>9s} {'p':>7s} {'cost cv':>8s}  family")
    for label, stem, family in SCHEDULES_UNDER_TEST:
        for k in KS:
            path = SCHEDULES / f"flux_k{k}_{stem}.txt"
            if not path.exists():
                continue
            bits = path.read_text().strip()
            if len(bits) != NUM_STEPS or set(bits) - {"0", "1"}:
                raise SystemExit(f"{path}: not a {NUM_STEPS}-character 0/1 bitstring")
            m = measure(bits, rho, sigmas)
            points[(label, k)] = {"local": m["local"], "span": m["span"], "family": family}
            if m["n_gaps"] < MIN_GAPS:
                print(f"{label:16s} {k:3d} {m['n_gaps']:5d}       -- too few gaps to rank --")
                continue
            rows.append({"schedule": label, "K": k, "n_gaps": m["n_gaps"],
                         "rho_density": m["rho_density"], "p": m["p"],
                         "cost_cv": m["cost_cv"], "family": family})
            print(f"{label:16s} {k:3d} {m['n_gaps']:5d} {m['rho_density']:9.3f} "
                  f"{m['p']:7.4f} {m['cost_cv']:8.3f}  {family}")

    if not args.write:
        print("\n(dry run; pass --write to update the tsv and the figure)")
        return

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t",
                                fieldnames=["schedule", "K", "n_gaps", "rho_density",
                                            "p", "cost_cv", "family"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten: {OUT_TSV}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"searched": "#d62728", "gate modal": "#1f77b4",
               "uniform-null": "#7f7f7f", "geometry-built": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    axes[0].plot(np.arange(NUM_STEPS), rho[:NUM_STEPS], color="black", lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("solver step n (0..49)")
    axes[0].set_ylabel(r"$\rho_2$  (derivatives taken in $\sigma$)")
    axes[0].set_title(r"$\rho_2$ risk profile (reuse order)" "\n"
                      f"{population.n_rows:,} Parti trajectories "
                      "(1,632 prompts x 3 seeds)", fontsize=10)
    axes[0].grid(alpha=0.3)
    for ax, k in zip(axes[1:], (29, 41)):
        seen = set()
        for (label, kk), pt in points.items():
            if kk != k:
                continue
            fam = pt["family"]
            ax.scatter(pt["local"], pt["span"], s=26, alpha=0.85,
                       color=colours[fam], label=fam if fam not in seen else None)
            seen.add(fam)
        ax.set_xscale("log")
        ax.set_xlabel(r"local $\rho_2$ inside the gap")
        ax.set_ylabel(r"gap length $\Delta\sigma$")
        ax.set_title(f"K={k} cached steps ({NUM_STEPS - k} really computed):\n"
                     "cost equalisation predicts a downward trend", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("FLUX: density-form test of the geometric cost model - "
                 "each point is one gap between two full steps of one schedule",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"written: {OUT_FIG}")


if __name__ == "__main__":
    main()
