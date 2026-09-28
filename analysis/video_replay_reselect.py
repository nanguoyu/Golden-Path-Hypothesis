#!/usr/bin/env python3
"""Re-select the video fixed-replay schedules on prompts disjoint from the
replay's own evaluation set.

Review point this answers
-------------------------
Section 2.4's video half freezes, per (model, gate, budget), the path that gate
walked most often in the three-seed baseline matrix, and then replays that path
on 300 prompts.  The census that picked the path counted *every* prompt of the
matrix (599 + 944 per seed stream, three streams), and the 300 evaluation
prompts are a subset of it.  Selection and evaluation therefore overlap.

This program re-runs the identical selection rule on the complement -- every
matrix prompt that the replay does NOT evaluate -- and reports whether the
frozen 50-bit path survives.

Selection rule (read off `analysis/build_video_spx_schedules.py::_pick_top1`,
which produced `resources/video_spx_schedules/<model>/{sea,tea,sen,di}_top1_K*.json`)
  * unit of counting: one realized 50-bit path of one generation;
  * pool: both datasets and all three seed streams of that (gate, budget) cell,
    summed by raw count (datasets are NOT weighted equally);
  * filter: keep only paths whose cached-step count is exactly K;
  * pick: highest pooled count, ties broken by the lexicographically smallest
    bit string;
  * if no path realizes exactly K, the row falls back to the pooled top-1 over
    ALL realized paths and is marked off-budget.

Evaluation prompts (`docs/video_sp_cross_plan_zh.md` section 4.2, confirmed
against `resources/video_spx/video_spx_results.json` -> `prompt_indices`):
penguin599 idx 0-149 and vbench944 idx 0-149, one seed stream each
(penguin s54, vbench s42).  The disjoint recount drops those (dataset, idx)
pairs in every seed stream, leaving penguin 150-598 and vbench 150-943.

Input is `resources/video_full_results/perprompt_paths_<model>.tsv.gz`, which
this program first checks reproduces `dataset_path_counts.tsv` exactly.

    python analysis/video_replay_reselect.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]

MODELS = ("hunyuan_video", "wan21")
MODEL_LABEL = {"hunyuan_video": "HunyuanVideo", "wan21": "Wan2.1"}
GATES = ("seacache", "teacache", "sencache", "dicache")
GATE_LABEL = {"seacache": "SeaCache", "teacache": "TeaCache",
              "sencache": "SenCache", "dicache": "DiCache"}
GATE_ROW = {"seacache": "sea_top1", "teacache": "tea_top1",
            "sencache": "sen_top1", "dicache": "di_top1"}
KS = (29, 37, 41)
DATASETS = ("penguin599", "vbench944")

#: The replay/SPX evaluation prompts, per dataset.
EVAL_PROMPT_INDICES = {dataset: frozenset(range(150)) for dataset in DATASETS}

PERPROMPT = {m: REPO / f"resources/video_full_results/perprompt_paths_{m}.tsv.gz"
             for m in MODELS}
CENSUS = {m: REPO / f"resources/video_native_gate_paths/{m}/dataset_path_counts.tsv"
          for m in MODELS}
SCHEDULE_DIR = {m: REPO / f"resources/video_spx_schedules/{m}" for m in MODELS}

OUT_DIR = REPO / "resources/video_replay_reselect"


# -- data --------------------------------------------------------------------


def read_perprompt(model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(PERPROMPT[model], "rt", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            rows.append({
                "method": record["method"],
                "dataset": record["dataset"],
                "K": int(record["K"]),
                "seed": int(record["seed"]),
                "prompt_idx": int(record["prompt_idx"]),
                "path": record["path"],
                "n_cached": int(record["n_cached"]),
            })
    return rows


def check_census_reproduced(model: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The per-prompt table must aggregate to the frozen census, bit for bit."""
    mine: collections.Counter = collections.Counter()
    for r in rows:
        mine[(r["method"], r["dataset"], r["K"], r["path"])] += 1
    theirs: collections.Counter = collections.Counter()
    with CENSUS[model].open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            key = (record["method"], record["dataset"],
                   int(record["budget"][1:]), record["schedule"])
            theirs[key] += int(record["count"])
    return {"identical": mine == theirs,
            "n_path_rows_recomputed": len(mine),
            "n_path_rows_frozen": len(theirs)}


# -- the selection rule ------------------------------------------------------


def pick_top1(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """`_pick_top1` of `analysis/build_video_spx_schedules.py`, on raw rows.

    Counts one per generation instead of reading pre-summed counts, which is
    the same number: the census is exactly that aggregation (checked above).
    """
    total_all: collections.Counter = collections.Counter()
    for r in rows:
        total_all[r["path"]] += 1
    grand_total = sum(total_all.values())

    pool = [r for r in rows if r["n_cached"] == k]
    off_budget = not pool
    if off_budget:
        pool = rows
    total: collections.Counter = collections.Counter()
    per_dataset: dict[str, collections.Counter] = {d: collections.Counter() for d in DATASETS}
    for r in pool:
        total[r["path"]] += 1
        per_dataset[r["dataset"]][r["path"]] += 1
    best = max(total.values())
    tied = sorted(bits for bits, count in total.items() if count == best)
    bits = tied[0]
    return {
        "bits": bits,
        "count": int(best),
        "pool_size": sum(total.values()),
        "grand_total": grand_total,
        "off_budget": off_budget,
        "n_tied_at_top": len(tied),
        "tie_broken_lexicographically": len(tied) > 1,
        "share_of_exact_k_pool": best / sum(total.values()),
        "share_of_all_paths": best / grand_total if grand_total else None,
        "n_distinct_paths_in_pool": len(total),
        "realized_cache_count": bits.count("1"),
        "per_dataset_counts": {d: int(c.get(bits, 0)) for d, c in per_dataset.items()},
        "per_dataset_top1_agrees": {
            d: (bool(c) and max(c.items(), key=lambda kv: (kv[1], [-ord(ch) for ch in kv[0]]))[0] == bits)
            for d, c in per_dataset.items()},
        "runner_up": _runner_up(total, bits),
    }


def _runner_up(total: collections.Counter, bits: str) -> dict[str, Any] | None:
    rest = [(b, c) for b, c in total.items() if b != bits]
    if not rest:
        return None
    b, c = max(rest, key=lambda kv: (kv[1], [-ord(ch) for ch in kv[0]]))
    return {"bits": b, "count": int(c), "share": c / sum(total.values()),
            "hamming_to_top1": hamming(b, bits)}


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def frozen_bits(model: str, gate: str, k: int) -> dict[str, Any]:
    """The 50-bit path the replay actually ran, from the frozen schedule file."""
    base = GATE_ROW[gate]
    for name in (f"{base}_K{k}.json", f"{base}_off_K{k}.json"):
        path = SCHEDULE_DIR[model] / name
        if path.exists():
            payload = json.loads(path.read_text())
            return {"bits": payload["bits"], "file": str(path.relative_to(REPO)),
                    "row": payload["row"], "off_budget": bool(payload["off_budget"]),
                    "cache_count": int(payload["cache_count"]),
                    "provenance": payload.get("provenance", {})}
    raise SystemExit(f"no frozen schedule for {model} {gate} K{k}")


# -- main --------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "schema": "video_replay_reselect.v1",
        "question": ("does the fixed-replay video schedule survive re-selection on "
                     "prompts disjoint from the 300 the replay evaluates"),
        "selection_rule": {
            "source": "analysis/build_video_spx_schedules.py::_pick_top1",
            "unit": "one realized 50-bit path per generation",
            "pool": "both datasets and all three seed streams, summed by raw count",
            "filter": "cached-step count exactly K; else fall back to all paths (off-budget)",
            "tie_break": "lexicographically smallest bit string",
        },
        "eval_prompts": {
            "source": ("docs/video_sp_cross_plan_zh.md section 4.2; confirmed against "
                       "resources/video_spx/video_spx_results.json prompt_indices"),
            "per_dataset_indices": {d: [0, 149] for d in DATASETS},
            "n_eval_prompts": 300,
            "seed_streams_evaluated": {"penguin599": [54], "vbench944": [42]},
            "exclusion": "drop (dataset, prompt_idx) with idx < 150 in every seed stream",
        },
        "census_check": {},
        "pool_sizes": {},
        "settings": [],
    }

    for model in MODELS:
        rows = read_perprompt(model)
        results["census_check"][model] = check_census_reproduced(model, rows)
        if not results["census_check"][model]["identical"]:
            raise SystemExit(f"{model}: per-prompt table does not aggregate to the census")

        disjoint = [r for r in rows
                    if r["prompt_idx"] not in EVAL_PROMPT_INDICES[r["dataset"]]]
        results["pool_sizes"][model] = {
            "all_generations": len(rows),
            "disjoint_generations": len(disjoint),
            "prompts_per_dataset_all": {d: 599 if d == "penguin599" else 944 for d in DATASETS},
            "prompts_per_dataset_disjoint": {
                d: len({r["prompt_idx"] for r in disjoint if r["dataset"] == d}) for d in DATASETS},
        }

        by_cell_all: dict[tuple[str, int], list] = collections.defaultdict(list)
        by_cell_dis: dict[tuple[str, int], list] = collections.defaultdict(list)
        for r in rows:
            by_cell_all[(r["method"], r["K"])].append(r)
        for r in disjoint:
            by_cell_dis[(r["method"], r["K"])].append(r)

        for k in KS:
            for gate in GATES:
                frozen = frozen_bits(model, gate, k)
                pick_all = pick_top1(by_cell_all[(gate, k)], k)
                pick_dis = pick_top1(by_cell_dis[(gate, k)], k)

                # where does the frozen path sit in the disjoint pool?
                dis_pool = [r for r in by_cell_dis[(gate, k)]
                            if (r["n_cached"] == k) or pick_dis["off_budget"]]
                dis_counts: collections.Counter = collections.Counter(r["path"] for r in dis_pool)
                frozen_count = int(dis_counts.get(frozen["bits"], 0))
                ranked = sorted(dis_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                frozen_rank = next((i + 1 for i, (b, _) in enumerate(ranked)
                                    if b == frozen["bits"]), None)

                results["settings"].append({
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "gate": gate,
                    "gate_label": GATE_LABEL[gate],
                    "K": k,
                    "row": frozen["row"],
                    "off_budget_row": frozen["off_budget"],
                    "frozen": frozen,
                    "all_prompts": pick_all,
                    "disjoint": pick_dis,
                    "reproduces_frozen": pick_all["bits"] == frozen["bits"],
                    "disjoint_same_as_frozen": pick_dis["bits"] == frozen["bits"],
                    "hamming_disjoint_vs_frozen": hamming(pick_dis["bits"], frozen["bits"]),
                    "frozen_in_disjoint_pool": {
                        "count": frozen_count,
                        "share": frozen_count / sum(dis_counts.values()) if dis_counts else None,
                        "rank": frozen_rank,
                    },
                })

    n = len(results["settings"])
    margins = [(s["disjoint"]["share_of_exact_k_pool"] - s["disjoint"]["runner_up"]["share"],
                f"{s['model_label']} {s['gate_label']} K{s['K']}")
               for s in results["settings"] if s["disjoint"]["runner_up"]]
    worst = min(margins)
    results["summary"] = {
        "n_settings": n,
        "min_disjoint_margin": worst[0],
        "min_disjoint_margin_setting": worst[1],
        "n_reproduced": sum(s["reproduces_frozen"] for s in results["settings"]),
        "n_disjoint_same": sum(s["disjoint_same_as_frozen"] for s in results["settings"]),
        "disjoint_differs": [
            {"model": s["model_label"], "gate": s["gate_label"], "K": s["K"],
             "hamming": s["hamming_disjoint_vs_frozen"],
             "frozen_share_in_disjoint_pool": s["frozen_in_disjoint_pool"]["share"],
             "frozen_rank_in_disjoint_pool": s["frozen_in_disjoint_pool"]["rank"],
             "disjoint_top1_share": s["disjoint"]["share_of_exact_k_pool"]}
            for s in results["settings"] if not s["disjoint_same_as_frozen"]],
    }

    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (args.out / "summary.md").write_text(render_summary(results))
    print(f"{results['summary']['n_reproduced']}/{n} frozen schedules reproduced; "
          f"{results['summary']['n_disjoint_same']}/{n} unchanged under disjoint selection "
          f"-> {args.out}")
    return 0


def render_summary(results: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Video fixed-replay: re-selecting the schedule on disjoint prompts\n")
    a("Section 2.4's video half freezes, per (model, gate, budget), the 50-bit path")
    a("the gate walked most often in the three-seed baseline matrix, then replays it")
    a("on 300 prompts.  The census behind that choice counted every matrix prompt,")
    a("and the 300 evaluation prompts are inside it.  This re-runs the identical")
    a("selection rule on the complement.\n")

    a("## Selection rule\n")
    rule = results["selection_rule"]
    for key in ("source", "unit", "pool", "filter", "tie_break"):
        a(f"- **{key}**: {rule[key]}")
    a("")

    a("## Prompt split\n")
    ev = results["eval_prompts"]
    a(f"- evaluated: penguin599 idx 0-149 (seed 54) and vbench944 idx 0-149 (seed 42), {ev['n_eval_prompts']} prompts")
    a("- disjoint pool: penguin599 idx 150-598 and vbench944 idx 150-943, all three seed streams")
    for model, sizes in results["pool_sizes"].items():
        a(f"- {model}: {sizes['all_generations']:,} generations in the full census, "
          f"{sizes['disjoint_generations']:,} in the disjoint pool "
          f"({sizes['prompts_per_dataset_disjoint']['penguin599']} + "
          f"{sizes['prompts_per_dataset_disjoint']['vbench944']} prompts)")
    a("")

    a("## Protocol fidelity\n")
    for model, chk in results["census_check"].items():
        a(f"- {model}: per-prompt table aggregates to the frozen census "
          f"({chk['n_path_rows_recomputed']} path rows) -- {'identical' if chk['identical'] else 'MISMATCH'}")
    s = results["summary"]
    a(f"- all-prompt re-selection reproduces the frozen schedule in {s['n_reproduced']}/{s['n_settings']} settings")
    a("")

    a("## Per setting\n")
    a("| Model | Method | K | Row | Reproduced | Disjoint same | Hamming | Disjoint top-1 mass | All-prompt top-1 mass | Disjoint runner-up mass | Margin |")
    a("|---|---|---:|---|:--:|:--:|---:|---:|---:|---:|---:|")
    for st in results["settings"]:
        ham = "-" if st["disjoint_same_as_frozen"] else str(st["hamming_disjoint_vs_frozen"])
        ru = st["disjoint"]["runner_up"]
        ru_share = f"{ru['share']:.3f}" if ru else "-"
        margin = (f"{st['disjoint']['share_of_exact_k_pool'] - ru['share']:+.3f}"
                  if ru else "-")
        a(f"| {st['model_label']} | {st['gate_label']} | {st['K']} | "
          f"{st['row']}{' (off-budget)' if st['off_budget_row'] else ''} | "
          f"{'yes' if st['reproduces_frozen'] else 'NO'} | "
          f"{'yes' if st['disjoint_same_as_frozen'] else 'no'} | {ham} | "
          f"{st['disjoint']['share_of_exact_k_pool']:.3f} | "
          f"{st['all_prompts']['share_of_exact_k_pool']:.3f} | {ru_share} | {margin} |")
    a("")
    a("Mass is the top-1 path's share of the selection pool, i.e. of the generations")
    a("whose realized cache count is exactly K (for an off-budget row, of all")
    a("generations in the cell).  Margin is the top-1 mass minus the runner-up's,")
    a("which says how much would have to move for the disjoint selection to change")
    a("its mind.  The smallest margin over the 24 settings is "
      f"{results['summary']['min_disjoint_margin']:.3f} "
      f"({results['summary']['min_disjoint_margin_setting']}).\n")

    if s["disjoint_differs"]:
        a("## Settings where the disjoint selection picks another path\n")
        a("| Model | Method | K | Hamming | Disjoint top-1 mass | Frozen path's mass in the disjoint pool | Frozen path's rank |")
        a("|---|---|---:|---:|---:|---:|---:|")
        for d in s["disjoint_differs"]:
            share = d["frozen_share_in_disjoint_pool"]
            a(f"| {d['model']} | {d['gate']} | {d['K']} | {d['hamming']} | "
              f"{d['disjoint_top1_share']:.3f} | "
              f"{share:.3f} | {d['frozen_rank_in_disjoint_pool']} |")
        a("")
    else:
        a("## Settings where the disjoint selection picks another path\n")
        a("None.\n")

    thin = [s for s in results["settings"]
            if s["disjoint"]["runner_up"]
            and s["disjoint"]["share_of_exact_k_pool"] - s["disjoint"]["runner_up"]["share"] < 0.05]
    a("## Thin margins\n")
    if not thin:
        a("No setting decides its top-1 by less than 0.05 of the pool.\n")
    else:
        a("These settings keep the frozen path, but by a small share.  The runner-up's")
        a("Hamming distance says how different the alternative would have been.\n")
        a("| Model | Method | K | Top-1 count | Runner-up count | Pool | Hamming to runner-up | Pooled top-1 is also each dataset's top-1 |")
        a("|---|---|---:|---:|---:|---:|---:|---|")
        for s in thin:
            d, ru = s["disjoint"], s["disjoint"]["runner_up"]
            agree = ", ".join(f"{k}: {'yes' if v else 'no'}"
                              for k, v in sorted(d["per_dataset_top1_agrees"].items()))
            a(f"| {s['model_label']} | {s['gate_label']} | {s['K']} | {d['count']} | "
              f"{ru['count']} | {d['pool_size']} | {ru['hamming_to_top1']} | {agree} |")
        a("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
