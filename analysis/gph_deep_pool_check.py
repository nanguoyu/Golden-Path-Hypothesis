#!/usr/bin/env python3
"""Deep-pool robustness check for the oracle-relative coverage table.

Paper Table~\\ref{tab:oracle-coverage} (section 2.4) reports how often one
fixed prompt-independent path stays within a margin of the best path any pool
member reaches on that prompt.  There the pool holds 9 to 17 evaluated paths,
so the natural objection is that the per-prompt best is easy to stay near
simply because the pool is shallow.

The exhaustive K41 experiment answers that objection directly.  Its held-out
family wave scores a panel of 337 schedules -- roughly thirty times the depth
of the Table 3 pools -- on every held-out prompt-seed pair of three prompt
sets and three seeds.  This script recomputes the Table 3 statistic on that
panel, with the same conventions:

  * the unit is a prompt, not a prompt-seed pair,
  * PSNR is averaged over the three seeds before anything else,
  * b(x) = max over the pool on prompt x is the per-prompt oracle,
  * for a pool member u and margin eps, the share of prompts with
    q(u;x) >= b(x) - eps, plus its one-sided exact binomial
    (Clopper-Pearson) 95% lower bound.

The four PartiPrompts indices the exhaustive search selected on are dropped,
so that every prompt reported here is held out from that selection, as in the
image rows of the paper table.

It reports, per prompt set and pooled over the three sets:

  (a) the pool member with the largest share at 0.25 dB (ties by lower rank),
  (b) the schedule the paper selected (rank 164762 by default), and
  (c) how many of the 337 members clear a 0.5 dB lower bound above one half.

Inputs (read-only) are the pair-level JSON files written by
``flux/exhaustive_k41_family_runner.py``:

    <data_root>/<prompt_set>_s<seed>/pairs/pair_XXXXX.json

Outputs are ``<out>.json`` (everything printed, machine readable) and
``<out>.tsv`` (one row per partition and pool member).

Usage:

    python analysis/gph_deep_pool_check.py \\
        --data_root outputs/exhaustive_k41/heldout_family_v1 \\
        --out $DATA/exhaustive_k41/deep_pool_check

The whole wave is a few hundred megabytes of JSON, so run it through
``RUN/slurm_gph_deep_pool.sh`` rather than on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAIR_SCHEMA = "flux_exhaustive_k41_family_pair.v1"
DEFAULT_DATA_ROOT = Path("outputs/exhaustive_k41/heldout_family_v1")
DEFAULT_CANDIDATES = 337
#: the schedule the paper selected: highest mean PSNR on the four selection
#: pairs of the exhaustive search
DEFAULT_SELECTED_RANK = 164762
#: The exhaustive search picked that schedule on PartiPrompts indices 5, 8, 9
#: and 15 of resources/prompts/partiprompts_full_eval1632_seed42.txt
#: (resources/exhaustive_k41/formal_results/summary.json, experiment
#: .prompt_indices), which is the prompt file the parti environments of this
#: wave use.  Those four prompts are therefore in sample for the selected
#: schedule and are dropped, so that every reported prompt is held out, as in
#: the image rows of the paper table.  The wave already omits them at base
#: seed 42, the base seed of the selection experiment, so the seed-coverage
#: rule below would drop them in any case; naming them states the reason.

DEFAULT_EXCLUDE = {"parti": (5, 8, 9, 15)}
MARGINS = (0.25, 0.5, 1.0)
ALPHA = 0.05
POOLED = "pooled"

ENV_RE = re.compile(r"^(?P<dataset>[a-z0-9]+)_s(?P<seed>\d+)$")


# --------------------------------------------------------------- statistics


try:  # the cluster env has scipy; the fallback below keeps this stdlib-only
    from scipy.stats import beta as _beta_dist
except Exception:  # pragma: no cover - exercised only without scipy
    _beta_dist = None


_LOG_BINOM: dict[int, list[float]] = {}


def _log_binom(n: int) -> list[float]:
    """log C(n, i) for i = 0 .. n."""
    cached = _LOG_BINOM.get(n)
    if cached is None:
        lg = math.lgamma
        top = lg(n + 1)
        cached = [top - lg(i + 1) - lg(n - i + 1) for i in range(n + 1)]
        _LOG_BINOM[n] = cached
    return cached


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    lb = _log_binom(n)
    terms = [lb[i] + i * log_p + (n - i) * log_q for i in range(k, n + 1)]
    top = max(terms)
    return math.exp(top) * sum(math.exp(t - top) for t in terms)


def _cp_lower_stdlib(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson lower bound by bisection on the exact binomial tail.

    The bound is the p at which P(X >= k) equals alpha; the tail is increasing
    in p, so plain bisection converges.
    """
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_tail(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_lower(k: int, n: int, alpha: float = ALPHA, backend: str = "auto") -> float:
    """One-sided exact binomial (Clopper-Pearson) lower bound for k/n."""
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    if backend != "stdlib" and _beta_dist is not None:
        return float(_beta_dist.ppf(alpha, k, n - k + 1))
    return _cp_lower_stdlib(k, n, alpha)


def cp_backend_name(backend: str) -> str:
    if backend != "stdlib" and _beta_dist is not None:
        return "scipy.stats.beta.ppf"
    return "stdlib bisection on the exact binomial tail"


# ------------------------------------------------------------------ loading


def parse_exclude(values) -> dict[str, set[int]]:
    """Parse --exclude 'parti=5,8,9,15' arguments; 'none' clears the default."""
    if values is None:
        return {k: set(v) for k, v in DEFAULT_EXCLUDE.items()}
    if len(values) == 1 and values[0].lower() == "none":
        return {}
    parsed: dict[str, set[int]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--exclude wants DATASET=idx,idx, got {value!r}")
        dataset, raw = value.split("=", 1)
        parsed.setdefault(dataset.strip(), set()).update(
            int(piece) for piece in raw.split(",") if piece.strip()
        )
    return parsed


def load_wave(data_root: Path, expect_candidates: int,
              exclude: dict[str, set[int]] | None = None) -> dict:
    """Read every pair file under data_root and seed-average the panel.

    Returns the per-prompt seed-averaged PSNR vectors keyed by prompt set,
    the shared rank order, and the inventory of what was read.
    """
    env_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and ENV_RE.match(p.name))
    if not env_dirs:
        raise SystemExit(f"no <dataset>_s<seed> directories under {data_root}")

    ranks: list[int] | None = None
    # dataset -> prompt_idx -> [psnr sum over seeds]
    totals: dict[str, dict[int, list[float]]] = defaultdict(dict)
    seen_seeds: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    seeds_per_dataset: dict[str, set[int]] = defaultdict(set)
    env_counts: list[dict] = []
    problems: list[str] = []

    for env_dir in env_dirs:
        match = ENV_RE.match(env_dir.name)
        assert match is not None
        dataset = match.group("dataset")
        base_seed = int(match.group("seed"))
        seeds_per_dataset[dataset].add(base_seed)
        pair_files = sorted((env_dir / "pairs").glob("pair_*.json"))
        identities: set[int] = set()

        for pair_file in pair_files:
            payload = json.loads(pair_file.read_text(encoding="utf-8"))
            if payload.get("schema") != PAIR_SCHEMA:
                problems.append(f"{pair_file}: schema {payload.get('schema')!r}")
                continue
            prompt_idx = int(payload["prompt_idx"])
            if prompt_idx in identities:
                problems.append(f"{pair_file}: duplicate prompt_idx {prompt_idx}")
                continue
            identities.add(prompt_idx)

            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != expect_candidates:
                problems.append(
                    f"{pair_file}: {0 if candidates is None else len(candidates)} "
                    f"candidates, expected {expect_candidates}"
                )
                continue
            pair_ranks = [int(row["rank"]) for row in candidates]
            if ranks is None:
                ranks = pair_ranks
                if len(set(ranks)) != len(ranks):
                    raise SystemExit(f"{pair_file}: repeated ranks in the panel")
            elif pair_ranks != ranks:
                if sorted(pair_ranks) == sorted(ranks):
                    problems.append(f"{pair_file}: candidates reordered")
                else:
                    problems.append(f"{pair_file}: different rank set")
                continue

            values = [float(row["psnr_db"]) for row in candidates]
            row = totals[dataset].get(prompt_idx)
            if row is None:
                totals[dataset][prompt_idx] = values
            else:
                for i, value in enumerate(values):
                    row[i] += value
            seen_seeds[dataset][prompt_idx].add(base_seed)

        env_counts.append(
            {"env": env_dir.name, "dataset": dataset, "base_seed": base_seed,
             "pair_files": len(pair_files), "pairs_used": len(identities)}
        )

    if ranks is None:
        raise SystemExit(f"no usable pair files under {data_root}")

    # keep prompts covered by every seed of their own prompt set, minus the
    # prompts the selected schedule was chosen on
    exclude = exclude or {}
    scores: dict[str, dict[int, list[float]]] = {}
    coverage: list[dict] = []
    for dataset, table in sorted(totals.items()):
        wanted = seeds_per_dataset[dataset]
        drop_idx = exclude.get(dataset, set())
        kept, dropped, in_sample = {}, [], []
        for prompt_idx, sums in table.items():
            if prompt_idx in drop_idx:
                in_sample.append(prompt_idx)
                continue
            have = seen_seeds[dataset][prompt_idx]
            if have == wanted:
                kept[prompt_idx] = [value / len(wanted) for value in sums]
            else:
                dropped.append(prompt_idx)
        scores[dataset] = kept
        coverage.append(
            {"dataset": dataset, "seeds": sorted(wanted),
             "prompts_seen": len(table), "prompts_kept": len(kept),
             "prompts_dropped": len(dropped),
             "dropped_prompt_idx": sorted(dropped)[:20],
             "selection_prompts_excluded": sorted(in_sample)}
        )

    return {
        "ranks": ranks,
        "scores": scores,
        "env_counts": env_counts,
        "coverage": coverage,
        "problems": problems,
    }


# ----------------------------------------------------------------- analysis


def partition_units(scores: dict[str, dict[int, list[float]]], datasets: list[str]):
    """Ordered (dataset, prompt_idx) units, each prompt weighted equally."""
    units = []
    for dataset in datasets:
        for prompt_idx in sorted(scores[dataset]):
            units.append((dataset, prompt_idx, scores[dataset][prompt_idx]))
    return units


def analyze_partition(name, units, ranks, margins, alpha, selected_rank, backend):
    """Shares and lower bounds for every pool member of one partition."""
    n = len(units)
    n_cand = len(ranks)
    if n == 0:
        return {"partition": name, "n_prompts": 0, "pool_size": n_cand,
                "members": [], "best_at_tightest": None, "selected": None,
                "n_members_lb_above_half": 0, "oracle_argmax": None}

    best = [max(values) for _, _, values in units]
    hits = [[0] * len(margins) for _ in range(n_cand)]
    argmax_counts = [0] * n_cand
    gaps_to_best = [[] for _ in range(n_cand)]

    for row, (_, _, values) in enumerate(units):
        top = best[row]
        argmax_counts[values.index(top)] += 1
        for j, value in enumerate(values):
            gap = top - value
            gaps_to_best[j].append(gap)
            row_hits = hits[j]
            for m, eps in enumerate(margins):
                if gap <= eps:
                    row_hits[m] += 1

    members = []
    for j, rank in enumerate(ranks):
        shares = [hits[j][m] / n for m in range(len(margins))]
        lowers = [cp_lower(hits[j][m], n, alpha, backend) for m in range(len(margins))]
        gaps = gaps_to_best[j]
        members.append({
            "rank": rank,
            "hits": list(hits[j]),
            "share": shares,
            "lower": lowers,
            "mean_gap_db": statistics.fmean(gaps),
            "median_gap_db": statistics.median(gaps),
            "max_gap_db": max(gaps),
            "prompts_best": argmax_counts[j],
        })

    tightest = 0
    best_member = min(members, key=lambda m: (-m["share"][tightest], m["rank"]))
    selected = next((m for m in members if m["rank"] == selected_rank), None)
    half_index = margins.index(0.5) if 0.5 in margins else min(1, len(margins) - 1)
    n_above_half = sum(1 for m in members if m["lower"][half_index] > 0.5)
    winners = sorted(
        ({"rank": m["rank"], "prompts_best": m["prompts_best"]}
         for m in members if m["prompts_best"] > 0),
        key=lambda r: (-r["prompts_best"], r["rank"]),
    )

    return {
        "partition": name,
        "n_prompts": n,
        "pool_size": n_cand,
        "members": members,
        "best_at_tightest": best_member,
        "selected": selected,
        "half_margin": margins[half_index],
        "n_members_lb_above_half": n_above_half,
        "oracle_argmax": {
            "distinct_winners": len(winners),
            "top": winners[:5],
        },
    }


# -------------------------------------------------------------------- report


def fmt_row(label, result, margins):
    cells = []
    for m in range(len(margins)):
        cells.append(f"{result['share'][m]:.3f} [{result['lower'][m]:.3f}]")
    return f"  {label:<26}{result['rank']:>8}  " + "  ".join(f"{c:>15}" for c in cells)


def print_report(report, margins):
    print("=" * 88)
    print("Deep-pool robustness check for the oracle-relative coverage table")
    print("=" * 88)
    print(f"data root      : {report['data_root']}")
    print(f"pool size      : {report['pool_size']} schedules "
          f"(paper Table 3 pools hold 9 to 17)")
    print(f"selected rank  : {report['selected_rank']}")
    print(f"binomial bound : one-sided {1 - report['alpha']:.0%}, "
          f"{report['cp_backend']}")
    excluded = report["excluded_selection_prompts"]
    print("in sample      : " + (
        ", ".join(f"{k} {v}" for k, v in sorted(excluded.items())) if excluded
        else "nothing excluded"))
    print()

    print("Pairs read per environment")
    total_files = total_used = 0
    for row in report["env_counts"]:
        total_files += row["pair_files"]
        total_used += row["pairs_used"]
        print(f"  {row['env']:<16}{row['pair_files']:>7} pair files"
              f"{row['pairs_used']:>7} used")
    print(f"  {'total':<16}{total_files:>7} pair files{total_used:>7} used")
    print()

    print("Prompt coverage (a prompt is kept when every seed of its set has it)")
    for row in report["coverage"]:
        seeds = ",".join(str(s) for s in row["seeds"])
        print(f"  {row['dataset']:<16}seeds {seeds:<12}"
              f"seen {row['prompts_seen']:>5}  kept {row['prompts_kept']:>5}"
              f"  dropped {row['prompts_dropped']:>3}"
              + (f"  {row['dropped_prompt_idx']}" if row["prompts_dropped"] else ""))
        if row["selection_prompts_excluded"]:
            print(f"  {'':<16}selection prompts excluded as in sample: "
                  f"{row['selection_prompts_excluded']}")
    print()

    if report["problems"]:
        print(f"Data problems ({len(report['problems'])}, first 20 shown)")
        for line in report["problems"][:20]:
            print(f"  {line}")
        print()
    else:
        print("Data check    : every pair file carries the same "
              f"{report['pool_size']} ranks in the same order")
        print()

    header = f"  {'member':<26}{'rank':>8}  " + "  ".join(
        f"{f'{eps} dB':>15}" for eps in margins)
    for result in report["partitions"]:
        print(f"{result['partition']}  (n = {result['n_prompts']} prompts, "
              f"pool = {result['pool_size']})")
        if result["n_prompts"] == 0:
            print("  no prompt has full seed coverage")
            print()
            continue
        print(header)
        print(fmt_row("best at 0.25 dB", result["best_at_tightest"], margins))
        if result["selected"] is not None:
            print(fmt_row("selected schedule", result["selected"], margins))
        else:
            print(f"  selected rank {report['selected_rank']} is not in the panel")
        sel = result["selected"]
        if sel is not None:
            print(f"  selected schedule gap to the per-prompt best: "
                  f"mean {sel['mean_gap_db']:.3f} dB, "
                  f"median {sel['median_gap_db']:.3f} dB, "
                  f"max {sel['max_gap_db']:.3f} dB")
        print(f"  members with a {result['half_margin']} dB lower bound above 0.5: "
              f"{result['n_members_lb_above_half']} of {result['pool_size']}")
        argmax = result["oracle_argmax"]
        top = ", ".join(f"{r['rank']} ({r['prompts_best']})" for r in argmax["top"])
        print(f"  schedules that are the per-prompt best at least once: "
              f"{argmax['distinct_winners']}; most often {top}")
        print()


# --------------------------------------------------------------------- io


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, report: dict, margins) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    header = ["partition", "n_prompts", "pool_size", "rank", "is_selected",
              "prompts_best", "mean_gap_db", "median_gap_db", "max_gap_db"]
    for eps in margins:
        header += [f"share_{eps}db", f"lower_{eps}db"]
    lines = ["\t".join(header)]
    for result in report["partitions"]:
        for member in result["members"]:
            row = [result["partition"], result["n_prompts"], result["pool_size"],
                   member["rank"],
                   int(member["rank"] == report["selected_rank"]),
                   member["prompts_best"],
                   f"{member['mean_gap_db']:.6f}",
                   f"{member['median_gap_db']:.6f}",
                   f"{member['max_gap_db']:.6f}"]
            for m in range(len(margins)):
                row += [f"{member['share'][m]:.6f}", f"{member['lower'][m]:.6f}"]
            lines.append("\t".join(str(value) for value in row))
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_out() -> Path:
    data = os.environ.get("DATA")
    base = Path(data) if data else REPO / "resources"
    return base / "exhaustive_k41" / "deep_pool_check"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="held-out family wave root holding <set>_s<seed>/pairs")
    parser.add_argument("--out", type=Path, default=default_out(),
                        help="output path prefix; writes <out>.json and <out>.tsv")
    parser.add_argument("--selected_rank", type=int, default=DEFAULT_SELECTED_RANK,
                        help="the schedule the paper selected")
    parser.add_argument("--margins", type=float, nargs="+", default=list(MARGINS),
                        help="PSNR margins in dB, tightest first")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="one-sided binomial bound level")
    parser.add_argument("--expect_candidates", type=int, default=DEFAULT_CANDIDATES,
                        help="candidates every pair file must carry")
    parser.add_argument("--cp_backend", choices=("auto", "stdlib"), default="auto",
                        help="'stdlib' forces the bisection bound even with scipy")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="SET=IDX,IDX",
                        help="prompts to drop as in sample; 'none' keeps every "
                             f"prompt (default: {DEFAULT_EXCLUDE})")
    args = parser.parse_args(argv)

    margins = list(args.margins)
    if margins != sorted(margins):
        raise SystemExit("--margins must be given tightest first")

    exclude = parse_exclude(args.exclude)
    wave = load_wave(args.data_root.resolve(), args.expect_candidates, exclude)
    ranks = wave["ranks"]
    scores = wave["scores"]
    datasets = sorted(scores)

    partitions = []
    for dataset in datasets:
        partitions.append(analyze_partition(
            dataset, partition_units(scores, [dataset]), ranks, margins,
            args.alpha, args.selected_rank, args.cp_backend))
    partitions.append(analyze_partition(
        POOLED, partition_units(scores, datasets), ranks, margins,
        args.alpha, args.selected_rank, args.cp_backend))

    report = {
        "check": "gph_deep_pool_check.v1",
        "data_root": str(args.data_root.resolve()),
        "pool_size": len(ranks),
        "selected_rank": args.selected_rank,
        "margins_db": margins,
        "alpha": args.alpha,
        "cp_backend": cp_backend_name(args.cp_backend),
        "excluded_selection_prompts": {k: sorted(v) for k, v in exclude.items()},
        "env_counts": wave["env_counts"],
        "coverage": wave["coverage"],
        "problems": wave["problems"],
        "partitions": partitions,
    }

    print_report(report, margins)

    out = args.out
    write_json(out.with_suffix(out.suffix + ".json"), report)
    write_tsv(out.with_suffix(out.suffix + ".tsv"), report, margins)
    print(f"wrote {out}.json")
    print(f"wrote {out}.tsv")
    return 1 if wave["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
