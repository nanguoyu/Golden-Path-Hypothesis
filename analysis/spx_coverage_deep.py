#!/usr/bin/env python3
"""Deepen the reference pool of the oracle-relative coverage table.

Paper Table~\\ref{tab:oracle-coverage} (section 2.4, produced by
``analysis/gph_oracle_check.py``) asks, per (model, cache ratio) partition, how
often one fixed prompt-independent schedule stays within a margin of the best
quality any pool member reaches on that prompt.  There the pool is a curated
set of 9 to 17 paths built around the offline tables, the most frequent gate
paths and MeanCache neighbours.  A reader can object that the per-prompt best
is easy to stay near because the pool is shallow and clustered.

This script keeps the protocol of ``gph_oracle_check.py`` unchanged and only
enlarges the pool.  For every partition the deepened pool is EVERY distinct
fixed schedule that has staged per-prompt scores under the residual-reuse
payload at that exact cache ratio: the current pool members plus the random
controls, the free Hamming ladder, the rho_2 DP row and the geometry-screened
rows.  Rows whose realised cache count differs from the partition budget
(the ``*_off`` rows) are dropped; nothing else is dropped.

Everything else is identical to the paper table:

  * quality metric PSNR, same-seed reconstruction, read as recorded,
  * a prompt is the unit; scores are averaged over the seeds a cell carries,
  * image partitions drop the 544 discovery prompts, leaving 1,088 held out;
    video partitions pool 300 in-sample prompts over two datasets,
  * b(x) = max over the pool on prompt x,
  * margins 0.25 / 0.5 / 1.0 dB, share plus one-sided exact binomial
    (Clopper-Pearson) 95% lower bound, and the same bound at level 0.05/m
    with m the pool size,
  * the reported member is the one with the largest share at 0.25 dB.

Step 1 of ``main`` is a protocol-fidelity check: the loader here is compared
row by row with ``gph_oracle_check.load_image`` / ``load_video``, and the
twelve partitions are recomputed with the original pools and checked against
the values recorded in ``gph_oracle_check`` (which are the paper table).  If
any of that fails the script stops before touching the deepened pools.

Outputs
    resources/spx_coverage_deep/results.json
    resources/spx_coverage_deep/summary.md

Usage
    python analysis/spx_coverage_deep.py
    python analysis/spx_coverage_deep.py --no_write
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

import gph_oracle_check as base  # noqa: E402  the protocol this script reuses

OUT_DIR = REPO / "resources" / "spx_coverage_deep"

MARGINS = base.EPS                      # (0.25, 0.5, 1.0) dB
ALPHA = base.ALPHA                      # 0.05
IMAGE_KS = base.IMAGE_KS                # ("29", "37", "41")
METRIC = "psnr"

IMAGE_TABLE = REPO / "resources" / "spx" / "perprompt_spx_{model}.tsv.gz"
VIDEO_TABLE = (REPO / "resources" / "video_spx" / "{model}"
               / "pervideo_spx_{model}.tsv.gz")
SPLITS = base.SPLITS
VIDEO_MANIFEST = REPO / "resources" / "video_spx_schedules" / "manifest.tsv"
IMAGE_SCHEDULE_DIRS = ("sp_cross_schedules", "spx_supplement_schedules")

#: schedule family of every staged row name, for the "who won" column.
FAMILY_RULES = (
    (re.compile(r"^(budcache|dpcache|meancache|uniform|shared)$"), "offline-method"),
    (re.compile(r"^(rand_\d+)$"), "random"),
    (re.compile(r"_top1(_r2)?$"), "most-frequent"),
    (re.compile(r"^ham\d+f?(_d\d+)?$"), "neighbor"),
    (re.compile(r"^dp_rho2$"), "other-control"),
    (re.compile(r"^gpf_"), "other-control"),
)


def family_of(name: str) -> str:
    for pattern, label in FAMILY_RULES:
        if pattern.search(name):
            return label
    return "other-control"


# --------------------------------------------------------------- schedules


def image_bits() -> dict[tuple[str, str], dict[str, str]]:
    """(model, k) -> {schedule: 50-bit path}."""
    out: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    pattern = re.compile(r"^(flux|qwen)_k(\d+)_(.+)\.txt$")
    for directory in IMAGE_SCHEDULE_DIRS:
        for path in sorted((REPO / "resources" / directory).glob("*_k*_*.txt")):
            match = pattern.match(path.name)
            if not match:
                continue
            model, k, schedule = match.groups()
            bits = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            out[(model, k)][schedule] = bits
    return out


def video_manifest() -> dict[tuple[str, str], dict[str, dict]]:
    """(backbone, k) -> {row: manifest record}."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    with VIDEO_MANIFEST.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            k = record["budget"].lstrip("K")
            out[(record["backbone"], k)][record["row"]] = record
    return out


# ------------------------------------------------------------------ loading


def load_image_scores(model: str, keep: set[str] | None, metric: str = METRIC,
                      drop_discovery: bool = True, seeds: set[str] | None = None):
    """(k, schedule) -> {prompt: seed-averaged score}, mirroring base.load_image.

    ``keep`` of None keeps every staged schedule.  ``seeds`` of None keeps
    every staged seed, which is what the paper table does.
    """
    sign = base.METRICS[metric][0]
    discovery: set[int] = set()
    if drop_discovery:
        discovery = set(json.loads(SPLITS.read_text(encoding="utf-8"))
                        ["roles"]["discovery"])
    raw: dict = defaultdict(lambda: defaultdict(list))
    seen_seeds: dict = defaultdict(set)
    with gzip.open(str(IMAGE_TABLE).format(model=model), "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["payload"] != "reuse":
                continue
            if keep is not None and row["schedule"] not in keep:
                continue
            if seeds is not None and row["seed"] not in seeds:
                continue
            prompt = int(row["prompt_idx"])
            if prompt in discovery:
                continue
            raw[(row["k"], row["schedule"])][prompt].append(sign * float(row[metric]))
            seen_seeds[(row["k"], row["schedule"])].add(row["seed"])
    tables = {key: {p: base.mean(v) for p, v in table.items()}
              for key, table in raw.items()}
    return tables, {key: sorted(v) for key, v in seen_seeds.items()}


def load_video_scores(model: str, keep: set[str] | None, metric: str = METRIC):
    """(K, row) -> {(dataset, prompt): seed-averaged score}, as base.load_video."""
    sign = base.METRICS[metric][0]
    raw: dict = defaultdict(lambda: defaultdict(list))
    seen_seeds: dict = defaultdict(set)
    realized: dict = defaultdict(set)
    with gzip.open(str(VIDEO_TABLE).format(model=model), "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["payload"] != "reuse":
                continue
            if keep is not None and row["row"] not in keep:
                continue
            key = (row["dataset"], row["prompt_idx"])
            raw[(row["K"], row["row"])][key].append(sign * float(row[metric]))
            seen_seeds[(row["K"], row["row"])].add((row["dataset"], row["seed"]))
            realized[(row["K"], row["row"])].add(int(row["cache_count_realized"]))
    tables = {key: {p: base.mean(v) for p, v in table.items()}
              for key, table in raw.items()}
    return tables, {key: sorted(v) for key, v in seen_seeds.items()}, realized


# ------------------------------------------------------------- pool building


def ordered_pool(present: list[str], declared: tuple[str, ...]) -> list[str]:
    """Declared members first in their declared order, then the rest sorted.

    Restricted to the declared set this reproduces the pool order of
    ``gph_oracle_check``, so the tie-break of ``best_member`` is unchanged.
    """
    head = [s for s in declared if s in present]
    tail = sorted(s for s in present if s not in set(declared))
    return head + tail


def deep_image_pool(model: str, k: str, tables: dict, bits: dict):
    """Every distinct on-budget schedule with staged reuse scores at this K."""
    present = sorted(s for (kk, s) in tables if kk == k)
    table_bits = bits.get((model, k), {})
    dropped_off, dropped_dup, missing_bits = [], [], []
    seen: dict[str, str] = {}
    keep: list[str] = []
    for schedule in ordered_pool(present, base.IMAGE_CLEAN + base.IMAGE_RANDOM):
        path = table_bits.get(schedule)
        if path is None:
            missing_bits.append(schedule)
            continue
        if path.count("1") != int(k):
            dropped_off.append({"schedule": schedule, "realized_k": path.count("1")})
            continue
        if path in seen:
            dropped_dup.append({"schedule": schedule, "same_as": seen[path]})
            continue
        seen[path] = schedule
        keep.append(schedule)
    return keep, {"off_budget": dropped_off, "duplicate_path": dropped_dup,
                  "no_schedule_file": missing_bits}


def deep_video_pool(model: str, k: str, tables: dict, manifest: dict,
                    realized: dict):
    present = sorted(s for (kk, s) in tables if kk == k)
    records = manifest.get((model, k), {})
    dropped_off, dropped_dup, missing_bits = [], [], []
    seen: dict[str, str] = {}
    keep: list[str] = []
    for row in ordered_pool(present, base.VIDEO_CLEAN + base.VIDEO_RANDOM):
        record = records.get(row)
        counts = sorted(realized.get((k, row), set()))
        if record is None:
            missing_bits.append(row)
            continue
        off = record.get("off_budget") == "1" or counts != [int(k)]
        if off:
            dropped_off.append({"schedule": row, "realized_k": counts})
            continue
        path = record["bits"]
        if path in seen:
            dropped_dup.append({"schedule": row, "same_as": seen[path]})
            continue
        seen[path] = row
        keep.append(row)
    return keep, {"off_budget": dropped_off, "duplicate_path": dropped_dup,
                  "no_schedule_file": missing_bits}


# ---------------------------------------------------------------- reference


def reference(tables: dict, k: str, pool: list[str]):
    """Per-prompt oracle b(x) over the pool, on the prompts every member has."""
    prompts = sorted(set.intersection(*(set(tables[(k, s)]) for s in pool)))
    return prompts, {x: max(tables[(k, s)][x] for s in pool) for x in prompts}


def score_member(tables: dict, k: str, name: str, prompts: list,
                 best: dict, margins=MARGINS, alpha_corr: float | None = None):
    table = tables[(k, name)]
    rows = []
    for eps in margins:
        hits = sum(1 for x in prompts if x in table and table[x] >= best[x] - eps)
        n = sum(1 for x in prompts if x in table)
        rows.append({
            "eps": eps,
            "hits": hits,
            "n": n,
            "share": hits / n,
            "lcb95": base.cp_lower(hits, n),
            "lcb_bonferroni": (base.cp_lower(hits, n, alpha_corr)
                               if alpha_corr else None),
        })
    return rows


def pick_best(scored: dict, pool: list[str]):
    """Largest share at the tightest margin; ties keep the earlier pool member."""
    order = {name: i for i, name in enumerate(pool)}
    return min(scored, key=lambda s: (-scored[s][0]["share"], order[s]))


def member_block(name: str, rows: list, extra: dict | None = None) -> dict:
    block = {
        "schedule": name,
        "family": family_of(name),
        "margins": [{"eps": r["eps"], "share": round(r["share"], 6),
                     "lcb95": round(r["lcb95"], 6),
                     "lcb_bonferroni": (round(r["lcb_bonferroni"], 6)
                                        if r["lcb_bonferroni"] is not None else None)}
                    for r in rows],
    }
    if extra:
        block.update(extra)
    return block


# ------------------------------------------------------------------ fidelity


def check_fidelity(verbose: bool = True) -> dict:
    """Reproduce the recorded Table 2 numbers with this script's own pipeline."""
    report = {"loader_matches_base": {}, "partitions": [], "failures": []}

    image_tables, image_seeds = {}, {}
    for model in base.IMAGE_MODELS:
        keep = set(base.IMAGE_CLEAN) | set(base.IMAGE_RANDOM)
        mine, seeds = load_image_scores(model, keep)
        theirs = base.load_image(model)
        same = set(mine) == set(theirs) and all(
            set(mine[k]) == set(theirs[k])
            and all(abs(mine[k][p] - theirs[k][p]) < 1e-12 for p in mine[k])
            for k in mine)
        report["loader_matches_base"][model] = bool(same)
        if not same:
            report["failures"].append(f"{model}: loader differs from gph_oracle_check")
        image_tables[model], image_seeds[model] = mine, seeds

    video_tables = {}
    for model in base.VIDEO_MODELS:
        keep = set(base.VIDEO_CLEAN) | set(base.VIDEO_RANDOM)
        mine, _seeds, _real = load_video_scores(model, keep)
        theirs = base.load_video(model)
        same = set(mine) == set(theirs) and all(
            set(mine[k]) == set(theirs[k])
            and all(abs(mine[k][p] - theirs[k][p]) < 1e-12 for p in mine[k])
            for k in mine)
        report["loader_matches_base"][model] = bool(same)
        if not same:
            report["failures"].append(f"{model}: loader differs from gph_oracle_check")
        video_tables[model] = mine

    def one(model, k, tables, declared):
        pool = ordered_pool(sorted(s for (kk, s) in tables if kk == k), declared)
        pool = [s for s in pool if s in declared]
        prompts, best = reference(tables, k, pool)
        alpha_corr = ALPHA / len(pool)
        scored = {s: score_member(tables, k, s, prompts, best,
                                  alpha_corr=alpha_corr) for s in pool}
        name = pick_best(scored, pool)
        return pool, prompts, name, scored[name]

    for model in base.IMAGE_MODELS:
        for k in IMAGE_KS:
            pool, prompts, name, rows = one(model, k, image_tables[model],
                                            base.IMAGE_CLEAN)
            want_name, want_rows = base.EXPECT_IMAGE[(model, k)]
            entry = {"model": model, "k": k, "pool_size": len(pool),
                     "n_prompts": len(prompts), "best": name,
                     "recorded_best": want_name,
                     "got": [[round(r["share"], 3), round(r["lcb95"], 3)]
                             for r in rows],
                     "recorded": [list(w) for w in want_rows]}
            if name != want_name:
                report["failures"].append(
                    f"{model} K{k}: best {name}, recorded {want_name}")
            for r, (ws, wlb) in zip(rows, want_rows):
                if abs(r["share"] - ws) > base.TOL:
                    report["failures"].append(
                        f"{model} K{k} eps={r['eps']} share {r['share']:.4f} vs {ws}")
                if abs(r["lcb95"] - wlb) > base.TOL:
                    report["failures"].append(
                        f"{model} K{k} eps={r['eps']} lb {r['lcb95']:.4f} vs {wlb}")
            corr = rows[1]["lcb_bonferroni"]
            want_corr = base.EXPECT_CORRECTED_LB_E05[(model, k)]
            entry["corrected_lb_0.5"] = round(corr, 4)
            entry["recorded_corrected_lb_0.5"] = want_corr
            if abs(corr - want_corr) > base.TOL:
                report["failures"].append(
                    f"{model} K{k} corrected lb {corr:.4f} vs {want_corr}")
            report["partitions"].append(entry)

    for model in base.VIDEO_MODELS:
        for k in IMAGE_KS:
            pool, prompts, name, rows = one(model, k, video_tables[model],
                                            base.VIDEO_CLEAN)
            entry = {"model": model, "k": k, "pool_size": len(pool),
                     "n_prompts": len(prompts), "best": name,
                     "got": [[round(r["share"], 3), round(r["lcb95"], 3)]
                             for r in rows]}
            want = base.EXPECT_VIDEO_LB_E05[(model, k)]
            entry["recorded_lb_0.5"] = want
            if abs(rows[1]["lcb95"] - want) > base.TOL:
                report["failures"].append(
                    f"{model} K{k} lb@0.5 {rows[1]['lcb95']:.4f} vs {want}")
            if (model, k) in base.EXPECT_VIDEO_LB_E025:
                w25 = base.EXPECT_VIDEO_LB_E025[(model, k)]
                if abs(rows[0]["lcb95"] - w25) > base.TOL:
                    report["failures"].append(
                        f"{model} K{k} lb@0.25 {rows[0]['lcb95']:.4f} vs {w25}")
            corr = rows[1]["lcb_bonferroni"]
            want_corr = base.EXPECT_CORRECTED_LB_E05[(model, k)]
            entry["corrected_lb_0.5"] = round(corr, 4)
            entry["recorded_corrected_lb_0.5"] = want_corr
            if abs(corr - want_corr) > base.TOL:
                report["failures"].append(
                    f"{model} K{k} corrected lb {corr:.4f} vs {want_corr}")
            report["partitions"].append(entry)

    report["status"] = "reproduced" if not report["failures"] else "MISMATCH"
    if verbose:
        print("PROTOCOL FIDELITY: recompute Table 2 with the original pools")
        for entry in report["partitions"]:
            cells = "  ".join(f"{s:.3f}/{lb:.3f}" for s, lb in entry["got"])
            print(f"  {entry['model'] + ' K' + entry['k']:<22}"
                  f"{entry['best']:<16}{entry['pool_size']:>4}{entry['n_prompts']:>7}"
                  f"   {cells}")
        print(f"  status: {report['status']}")
        for line in report["failures"]:
            print("  FAIL " + line)
        print()
    return report


# ------------------------------------------------------------------ deepened


def analyse_partition(model: str, k: str, modality: str, tables: dict,
                      old_pool: list[str], deep_pool: list[str],
                      dropped: dict, seeds: dict | None) -> dict:
    prompts_old, best_old = reference(tables, k, old_pool)
    prompts_new, best_new = reference(tables, k, deep_pool)
    if prompts_old != prompts_new:
        raise SystemExit(f"{model} K{k}: prompt set changed with the deeper pool")
    prompts = prompts_old

    alpha_old = ALPHA / len(old_pool)
    alpha_new = ALPHA / len(deep_pool)

    scored_old = {s: score_member(tables, k, s, prompts, best_old,
                                  alpha_corr=alpha_old) for s in old_pool}
    scored_new = {s: score_member(tables, k, s, prompts, best_new,
                                  alpha_corr=alpha_new) for s in deep_pool}

    name_old = pick_best(scored_old, old_pool)
    name_new = pick_best(scored_new, deep_pool)
    old_members_in_deep = [s for s in deep_pool if s in set(old_pool)]
    name_clean = pick_best({s: scored_new[s] for s in old_members_in_deep},
                           old_members_in_deep)

    # how often each pool member is the per-prompt oracle
    argmax = defaultdict(int)
    for x in prompts:
        top = best_new[x]
        for s in deep_pool:
            if tables[(k, s)][x] == top:
                argmax[s] += 1
                break
    argmax_rows = sorted(({"schedule": s, "family": family_of(s), "prompts_best": c}
                          for s, c in argmax.items()),
                         key=lambda r: -r["prompts_best"])
    by_family: dict[str, int] = defaultdict(int)
    for row in argmax_rows:
        by_family[row["family"]] += row["prompts_best"]
    argmax_by_family = {f: round(c / len(prompts), 4)
                        for f, c in sorted(by_family.items(), key=lambda r: -r[1])}
    # share of prompts whose reference is set by a row whose construction saw
    # held-out quality (the geometry rows) or all prompt indices (the rho_2 DP row)
    exposed = sum(r["prompts_best"] for r in argmax_rows
                  if r["schedule"] == "dp_rho2" or r["schedule"].startswith("gpf_"))

    gains = [best_new[x] - best_old[x] for x in prompts]
    reference_gain = {
        "median_db": round(statistics.median(gains), 4),
        "mean_db": round(statistics.fmean(gains), 4),
        "p90_db": round(sorted(gains)[int(0.9 * (len(gains) - 1))], 4),
        "max_db": round(max(gains), 4),
        "share_improved": round(sum(1 for g in gains if g > 1e-9) / len(gains), 4),
    }

    added = [s for s in deep_pool if s not in set(old_pool)]
    seed_counts = None
    if seeds is not None:
        seed_counts = {s: len(seeds.get((k, s), [])) for s in deep_pool}

    return {
        "model": model,
        "k": int(k),
        "modality": modality,
        "n_prompts": len(prompts),
        "pool_old": {"size": len(old_pool), "members": old_pool},
        "pool_deep": {"size": len(deep_pool), "members": deep_pool, "added": added},
        "dropped": dropped,
        "seeds_per_member": seed_counts,
        "old": member_block(name_old, scored_old[name_old]),
        "deep": member_block(name_new, scored_new[name_new]),
        "deep_best_among_old_members": member_block(name_clean, scored_new[name_clean]),
        "old_best_under_deep_reference": member_block(name_old, scored_new[name_old]),
        "reference_gain": reference_gain,
        "oracle_argmax": {"distinct_winners": len(argmax_rows),
                          "top": argmax_rows[:8],
                          "share_by_family": argmax_by_family,
                          "share_set_by_exposed_rows": round(exposed / len(prompts), 4)},
        "members_deep": {s: member_block(s, scored_new[s]) for s in deep_pool},
        "verdict": {
            str(eps): {
                "old_lcb95": round(scored_old[name_old][i]["lcb95"], 6),
                "deep_lcb95": round(scored_new[name_new][i]["lcb95"], 6),
                "old_pass": scored_old[name_old][i]["lcb95"] > 0.5,
                "deep_pass": scored_new[name_new][i]["lcb95"] > 0.5,
                "flip": (scored_old[name_old][i]["lcb95"] > 0.5)
                        != (scored_new[name_new][i]["lcb95"] > 0.5),
                "old_lcb_bonferroni": round(scored_old[name_old][i]["lcb_bonferroni"], 6),
                "deep_lcb_bonferroni": round(scored_new[name_new][i]["lcb_bonferroni"], 6),
                "old_pass_bonferroni": scored_old[name_old][i]["lcb_bonferroni"] > 0.5,
                "deep_pass_bonferroni": scored_new[name_new][i]["lcb_bonferroni"] > 0.5,
            }
            for i, eps in enumerate(MARGINS)
        },
    }


def analyse_all(seeds_filter: set[str] | None = None) -> list[dict]:
    bits = image_bits()
    manifest = video_manifest()
    partitions = []

    for model in base.IMAGE_MODELS:
        tables, seeds = load_image_scores(model, None, seeds=seeds_filter)
        for k in IMAGE_KS:
            deep, dropped = deep_image_pool(model, k, tables, bits)
            old = [s for s in base.IMAGE_CLEAN if (k, s) in tables]
            partitions.append(analyse_partition(model, k, "image", tables,
                                                old, deep, dropped, seeds))

    for model in base.VIDEO_MODELS:
        tables, seeds, realized = load_video_scores(model, None)
        for k in IMAGE_KS:
            deep, dropped = deep_video_pool(model, k, tables, manifest, realized)
            old = [s for s in base.VIDEO_CLEAN if (k, s) in tables]
            partitions.append(analyse_partition(model, k, "video", tables,
                                                old, deep, dropped, seeds))
    return partitions


# -------------------------------------------------------------------- report


def print_partitions(partitions: list[dict]) -> None:
    header = (f"  {'partition':<20}{'pool':>10}  {'best member':<18}{'family':<16}"
              + "".join(f"{f'{e} dB':>16}" for e in MARGINS))
    print("DEEPENED POOL: share / one-sided 95% lower bound")
    print(header)
    for part in partitions:
        cells = "".join(f"{m['share']:>9.3f}/{m['lcb95']:.3f}"
                        for m in part["deep"]["margins"])
        pool = f"{part['pool_old']['size']}->{part['pool_deep']['size']}"
        print(f"  {part['model'] + ' K' + str(part['k']):<20}{pool:>10}  "
              f"{part['deep']['schedule']:<18}{part['deep']['family']:<16}{cells}")
    print()
    print("Reference gain b_deep(x) - b_old(x) over the reported prompts")
    print(f"  {'partition':<20}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}"
          f"{'improved':>10}")
    for part in partitions:
        g = part["reference_gain"]
        print(f"  {part['model'] + ' K' + str(part['k']):<20}"
              f"{g['median_db']:>9.3f}{g['mean_db']:>9.3f}{g['p90_db']:>9.3f}"
              f"{g['max_db']:>9.3f}{g['share_improved']:>10.3f}")
    print()
    for i, eps in enumerate(MARGINS):
        flips = [p for p in partitions if p["verdict"][str(eps)]["flip"]]
        old_pass = sum(1 for p in partitions if p["verdict"][str(eps)]["old_pass"])
        new_pass = sum(1 for p in partitions if p["verdict"][str(eps)]["deep_pass"])
        print(f"  {eps} dB: lower bound above one half {old_pass}/12 -> "
              f"{new_pass}/12; flips: "
              + (", ".join(f"{p['model']} K{p['k']}" for p in flips) or "none"))
    print()


def write_summary(path: Path, blob: dict) -> None:
    parts = blob["partitions"]
    lines = [
        "# Deepened reference pool for the oracle-relative coverage table",
        "",
        "Paper Table 2 (`tab:oracle-coverage`, section 2.4) reports how often one",
        "fixed prompt-independent path stays within a margin of the best path any",
        "pool member reaches on that prompt. Its pool holds 9 to 17 curated paths.",
        "This file keeps the protocol of `analysis/gph_oracle_check.py` unchanged",
        "and enlarges the pool to every distinct fixed schedule that has staged",
        "per-prompt PSNR under the residual-reuse payload at that exact cache",
        "ratio. Rows whose realised cache count differs from the budget are",
        "dropped; nothing else is.",
        "",
        f"Protocol-fidelity check: **{blob['fidelity']['status']}** "
        "(the twelve partitions recomputed here with the original pools match the",
        "values recorded in `analysis/gph_oracle_check.py`, which are the paper",
        "table).",
        "",
        "## Before and after",
        "",
        "Share of reported prompts within the margin of the per-prompt pool best,",
        "with the exact one-sided 95% binomial lower bound in brackets. `old` is",
        "the paper pool, `deep` the enlarged one.",
        "",
        "| Model | K | Pool | Best member (deep) | Family | 0.25 dB | 0.5 dB | 1.0 dB |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for part in parts:
        cells = " | ".join(f"{m['share']:.3f} [{m['lcb95']:.3f}]"
                           for m in part["deep"]["margins"])
        lines.append(
            f"| {part['model']} | {part['k']} | "
            f"{part['pool_old']['size']} -> {part['pool_deep']['size']} | "
            f"{part['deep']['schedule']} | {part['deep']['family']} | {cells} |")
    lines += ["", "Old pool, for comparison:", "",
              "| Model | K | Best member (old) | 0.25 dB | 0.5 dB | 1.0 dB |",
              "|---|---:|---|---|---|---|"]
    for part in parts:
        cells = " | ".join(f"{m['share']:.3f} [{m['lcb95']:.3f}]"
                           for m in part["old"]["margins"])
        lines.append(f"| {part['model']} | {part['k']} | "
                     f"{part['old']['schedule']} | {cells} |")

    lines += ["", "## How much harder the bar got", "",
              "Per-prompt gain of the deepened reference over the old one, in dB.",
              "",
              "| Model | K | median | mean | p90 | max | share improved |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for part in parts:
        g = part["reference_gain"]
        lines.append(f"| {part['model']} | {part['k']} | {g['median_db']:.3f} | "
                     f"{g['mean_db']:.3f} | {g['p90_db']:.3f} | {g['max_db']:.3f} | "
                     f"{g['share_improved']:.3f} |")

    families = ["offline-method", "most-frequent", "neighbor", "random",
                "other-control"]
    lines += ["", "## Who sets the deepened reference", "",
              "Share of reported prompts whose per-prompt best is attained by each",
              "family, plus the share set by a row whose construction saw held-out",
              "quality (`gpf_*`) or every prompt index (`dp_rho2`).",
              "",
              "| Model | K | distinct winners | "
              + " | ".join(families) + " | exposed |",
              "|---|---:|---:|" + "---:|" * (len(families) + 1)]
    for part in parts:
        share = part["oracle_argmax"]["share_by_family"]
        cells = " | ".join(f"{share.get(f, 0.0):.3f}" for f in families)
        lines.append(f"| {part['model']} | {part['k']} | "
                     f"{part['oracle_argmax']['distinct_winners']} | {cells} | "
                     f"{part['oracle_argmax']['share_set_by_exposed_rows']:.3f} |")

    lines += ["", "## Verdict at each margin", ""]
    for eps in MARGINS:
        key = str(eps)
        old_pass = sum(1 for p in parts if p["verdict"][key]["old_pass"])
        new_pass = sum(1 for p in parts if p["verdict"][key]["deep_pass"])
        flips = [f"{p['model']} K{p['k']}" for p in parts if p["verdict"][key]["flip"]]
        bonf_old = sum(1 for p in parts if p["verdict"][key]["old_pass_bonferroni"])
        bonf_new = sum(1 for p in parts if p["verdict"][key]["deep_pass_bonferroni"])
        lines.append(f"- **{eps} dB**: lower bound above one half in "
                     f"{old_pass}/12 partitions before, {new_pass}/12 after. "
                     f"With the Bonferroni level 0.05/m over the pool: "
                     f"{bonf_old}/12 before, {bonf_new}/12 after. "
                     f"Flips: {', '.join(flips) if flips else 'none'}.")

    same = sum(1 for p in parts
               if p["deep"]["schedule"] == p["old"]["schedule"])
    lines += ["", f"The deepened pool returns the same reported path in {same} of "
              f"{len(parts)} partitions, so the enlargement moves the numbers, not "
              "the identity of the path."]

    if blob.get("seed42_sensitivity"):
        lines += ["", "## Seed-matched sensitivity (image side)", "",
                  "The image pool mixes one-seed and three-seed cells, which was",
                  "already true of the paper table. Reading every image cell at the",
                  "one seed they all share removes that asymmetry from the maximum.",
                  "",
                  "| Model | K | Pool | Best member | 0.25 dB | 0.5 dB | 1.0 dB |",
                  "|---|---:|---:|---|---|---|---|"]
        for part in blob["seed42_sensitivity"]:
            cells = " | ".join(f"{m['share']:.3f} [{m['lcb95']:.3f}]"
                               for m in part["deep"]["margins"])
            lines.append(f"| {part['model']} | {part['k']} | "
                         f"{part['pool_deep']['size']} | "
                         f"{part['deep']['schedule']} | {cells} |")

    lines += ["", "## Notes", ""] + [f"- {note}" for note in blob["notes"]]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------- main


NOTES = [
    "The deepened pool includes the random controls, so a random path can now "
    "raise the per-prompt reference. The `oracle_argmax` block of results.json "
    "counts how often each family is the per-prompt best.",
    "It also includes `dp_rho2` (the rho_2 DP row, solved on a trajectory table "
    "spanning all prompt indices) and, on FLUX, the `gpf_*` geometry rows "
    "(screened on held-out quality). Both raise the bar, which is conservative "
    "for the hypothesis, but neither should be reported as a winning path. "
    "`deep_best_among_old_members` gives the best member of the original pool "
    "scored against the deepened reference, which is free of that exposure.",
    "Seed depth is uneven and was already uneven in the paper table: the "
    "supplement rows (`ham*_d*`, `rand_3..5`) carry one seed on the image side "
    "while the rest carry three. `seeds_per_member` records it per partition.",
    "The video pools deepen only from 9-11 to 15-17 members: the staged video "
    "SPX tables carry no further fixed schedules at those budgets beyond the "
    "free Hamming ladder, the rho_2 DP row and the two random rows.",
    "Off-budget rows dropped: image flux K29 `sencache_top1_off` (28 cached "
    "steps), video `sen_top1_off` / `tea_top1_off` (36) and `sea_top1_off` (40).",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no_write", action="store_true",
                        help="print everything without writing the outputs")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip_seed42", action="store_true",
                        help="skip the single-seed sensitivity pass")
    args = parser.parse_args(argv)

    fidelity = check_fidelity()
    if fidelity["status"] != "reproduced":
        print("Protocol fidelity failed; stopping before the deepened pools.")
        return 1

    partitions = analyse_all()
    print_partitions(partitions)

    seed42 = None
    if not args.skip_seed42:
        # sensitivity: every image cell read at its shared seed only, so that no
        # pool member enters the maximum with a noisier estimate than another.
        seed42 = [p for p in analyse_all(seeds_filter={"42"})
                  if p["modality"] == "image"]
        print("SEED-42-ONLY SENSITIVITY (image side; video is single-seed already)")
        for part in seed42:
            cells = "".join(f"{m['share']:>9.3f}/{m['lcb95']:.3f}"
                            for m in part["deep"]["margins"])
            print(f"  {part['model'] + ' K' + str(part['k']):<20}"
                  f"{part['deep']['schedule']:<18}{cells}")
        print()

    blob = {
        "check": "spx_coverage_deep.v1",
        "generated_by": "analysis/spx_coverage_deep.py",
        "protocol": {
            "source": "analysis/gph_oracle_check.py (paper Table tab:oracle-coverage)",
            "metric": METRIC,
            "margins_db": list(MARGINS),
            "alpha": ALPHA,
            "bound": "one-sided exact binomial (Clopper-Pearson)",
            "unit": "prompt, seed-averaged over the seeds the cell carries",
            "payload": "residual reuse",
            "image_population": "1088 PartiPrompts held out from the discovery split",
            "video_population": "300 in-sample prompts, penguin599 s54 + vbench944 s42",
            "pool_rule": ("every distinct schedule with staged per-prompt reuse "
                          "scores at the partition budget, minus rows whose "
                          "realised cache count differs from that budget"),
        },
        "fidelity": fidelity,
        "partitions": partitions,
        "seed42_sensitivity": seed42,
        "notes": NOTES,
    }

    if not args.no_write:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results = out_dir / "results.json"
        results.write_text(json.dumps(blob, indent=1, sort_keys=False) + "\n",
                           encoding="utf-8")
        write_summary(out_dir / "summary.md", blob)
        print(f"wrote {results} ({results.stat().st_size / 1e6:.1f} MB)")
        print(f"wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
