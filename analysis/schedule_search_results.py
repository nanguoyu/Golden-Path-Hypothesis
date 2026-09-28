"""Numbers for the schedule-search results (P5 of docs/schedule_search_plan_zh.md).

Reads the staged P4 tables resources/schedule_search/perprompt_search_{flux,qwen}.tsv.gz
(schedule payload k dataset seed prompt_idx psnr ssim lpips image_reward clip) and the
SPX tables resources/spx/perprompt_spx_{flux,qwen}.tsv.gz (same columns without
dataset, Parti only), and writes resources/schedule_search/results.json:

  * per (model, K, schedule, dataset): mean of each metric over prompts x seeds,
    with the per-seed means;
  * the paired per-prompt PSNR difference of each searched schedule against the
    incumbents' schedules under the same reuse payload (meancache, budcache),
    with a 95% bootstrap interval over prompts, pairing on
    (dataset, seed, prompt_idx). The incumbents' reuse rows are taken from the
    staged search tables on every dataset where they exist, and from the SPX
    tables on Parti otherwise;
  * the incumbents' own Parti means per metric under that same payload, for the
    level the differences are taken against;
  * an `arbitration_check` block: for every (model, K, algorithm) whose
    50-caption arbitration delivered a schedule other than the algorithm's
    calibration-best candidate, the paired per-image difference on PartiPrompts
    between the delivered schedule and that rejected candidate (staged under the
    name `cal0_<algorithm>`), on all five metrics, together with both schedules'
    full-step lists and their arbitration means;
  * an `objectives` block (section 4b of the plan): per (K, objective) the
    schedules that objective delivered, their calibration means over the 8
    pairs, their arbitration means over the 50 captions, their structure
    numbers, their staged P4 cell means, and the paired per-image difference of
    each of them against the PSNR-objective schedule of the same algorithm
    family, on every dataset and all five metrics.

  * a `payloads` block: the arbitration-best schedule of each setting
    (`delivery_best.txt`) under the five payloads, with per (K, payload,
    dataset) cell means, an equal-weight four-dataset row, the paired
    per-image difference against the same schedule under reuse, and the paired
    difference on PartiPrompts against the baseline methods' schedules under
    the same payload.

The `cal0_*` rows exist on PartiPrompts only and are not delivered schedules, so
they stay out of the `cells` and `paired` blocks.

Everything compared in the blocks above the `payloads` block runs the residual
reuse payload, so only the schedule differs between the two sides of a
difference. Inside the `payloads` block a difference either holds the schedule
fixed and varies the payload, or holds the payload fixed and varies the
schedule.

    python analysis/schedule_search_results.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.schedule_search import OBJECTIVES, objective_units  # noqa: E402

SS_DIR = Path("resources/schedule_search")
SPX_DIR = Path("resources/spx")
#: Every baseline method's schedule that the SPX wave ran under reuse on Parti:
#: the fixed-schedule methods and the gate methods' modal realised paths.
INCUMBENTS = (
    "meancache", "budcache", "dpcache", "uniform",
    "dicache_top1", "seacache_top1", "teacache_top1", "sencache_top1",
)
#: The payload every row of this experiment runs, searched and incumbent alike.
PAYLOAD = "reuse"
METRICS = ("psnr", "ssim", "lpips", "image_reward", "clip")
#: Name prefix of the rejected calibration-best candidates. These rows were run
#: on PartiPrompts only, for the arbitration check.
CAL0_PREFIX = "cal0_"
#: The dataset the arbitration check runs on.
CAL0_DATASET = "parti_full"
#: The budgets of the experiment, in the order the report lists them.
BUDGETS = (29, 37, 41)
#: The payload axis of the SPX cross, reuse first. Sections 2 to 5 of the report
#: read `reuse` alone; the payload supplement ran the delivered schedule of each
#: setting under the other four as well.
SUPPLEMENT_PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel",
                       "di_two_anchor")
#: The four evaluation datasets, in the order the report lists them.
DATASETS = ("drawbench_full", "parti_full", "geneval_style", "diffusiondb_clean10k")
#: The schedule list the payload supplement was run on: the arbitration-best
#: delivery of each (model, K).
DELIVERY_BEST = "delivery_best.txt"
#: The baseline methods' schedules the supplement compares against on Parti.
SUPPLEMENT_INCUMBENTS = ("meancache", "budcache")
#: The step below which a full step counts as early, as section 3 of the report
#: counts them; `analysis/render_schedule_search_results.py` uses the same value.
EARLY_STEP = 13
#: A commented-out row of a delivery list, marking a schedule that another
#: objective already delivered and that is therefore evaluated under that name.
IDENTICAL_RE = re.compile(r"^#\s*identical to ([^:\s]+)\s*:\s*(.+)$")
#: The steps every schedule of the space computes at full cost, used by the
#: structure numbers. Read from the frozen config when it is present.
DEFAULT_FORCED_FULL_STEPS = (0, 1, 2, 49)


def forced_full_steps(model: str) -> tuple[int, ...]:
    path = SS_DIR / "config.v1.json"
    if not path.is_file():
        return DEFAULT_FORCED_FULL_STEPS
    spaces = json.loads(path.read_text(encoding="utf-8")).get("spaces", {})
    steps = spaces.get(model, {}).get("forced_full_steps")
    return tuple(int(s) for s in steps) if steps else DEFAULT_FORCED_FULL_STEPS


def read_tsv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def read_schedule_list(path: Path) -> list[dict]:
    """A `model \t K \t name \t bits` list such as delivery.txt."""

    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        model, k, name, bits = line.split("\t")
        rows.append({"model": model, "k": int(k), "name": name, "bits": bits})
    return rows


def read_delivery_list(path: Path) -> list[dict]:
    """A delivery list, including the rows commented out as already delivered.

    A row `# identical to <name>: model K name bits` says that this objective
    delivered a bitstring an earlier objective already delivered, so it was not
    evaluated again and its cells are the ones staged under `<name>`.
    """

    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        identical_to = None
        if line.startswith("#"):
            match = IDENTICAL_RE.match(line)
            if match is None:
                continue
            identical_to, line = match.group(1), match.group(2).strip()
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) != 4:
            continue
        model, k, name, bits = fields
        rows.append({"model": model, "k": int(k), "name": name, "bits": bits,
                     "identical_to": identical_to, "evaluated": identical_to is None})
    return rows


def full_steps(bits: str) -> list[int]:
    return [i for i, c in enumerate(bits) if c == "0"]


def first_cached(bits: str) -> int:
    return bits.index("1")


def longest_cached_run(bits: str) -> int:
    best = run = 0
    for c in bits:
        run = run + 1 if c == "1" else 0
        best = max(best, run)
    return best


def structure(bits: str, forced: tuple[int, ...]) -> dict:
    """The structure numbers section 3 of the report reads off a schedule."""

    free = [s for s in full_steps(bits) if s not in forced]
    return {
        "first_cached_step": first_cached(bits),
        "free_full_steps": len(free),
        "free_full_steps_below_early": sum(1 for s in free if s < EARLY_STEP),
        "early_step": EARLY_STEP,
        "last_free_full_step": max(free) if free else None,
        "longest_cached_run": longest_cached_run(bits),
    }


def bootstrap_ci(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(values.size, size=(reps, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_metrics(left: dict, right: dict) -> dict:
    """Left minus right, per image, on every metric both sides carry.

    `left` and `right` map (seed, prompt index) to that image's metrics, so the
    pairing is on the prompt and the seed and the two sides differ only in the
    schedule. Each metric gets its own bootstrap seed so a metric's interval
    does not depend on which other metrics are present.
    """

    common = sorted(set(left) & set(right))
    entry: dict = {"n_pairs": len(common)}
    if not common:
        return entry
    for i, m in enumerate(METRICS):
        pairs = [c for c in common if m in left[c] and m in right[c]]
        if not pairs:
            continue
        d = np.array([left[c][m] - right[c][m] for c in pairs])
        lo, hi = bootstrap_ci(d, seed=i)
        entry[f"delta_{m}_mean"] = float(d.mean())
        entry[f"ci95_{m}"] = [lo, hi]
        entry[f"n_pairs_{m}"] = int(d.size)
        entry[f"share_positive_{m}"] = float((d > 0).mean())
    return entry


def bootstrap_ci_grouped(groups: list[np.ndarray], reps: int = 2000,
                         seed: int = 0) -> tuple[float, float]:
    """Bootstrap interval of a mean that weights every group equally.

    Each group is resampled on its own and the group means are averaged, so a
    dataset with more images does not pull the average towards itself.
    """

    rng = np.random.default_rng(seed)
    means = np.zeros(reps)
    for values in groups:
        idx = rng.integers(values.size, size=(reps, values.size))
        means += values[idx].mean(axis=1)
    means /= len(groups)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_metrics_over_groups(groups: list[tuple[dict, dict]]) -> dict:
    """Left minus right per image, averaged over the groups with equal weight.

    Each group is one dataset: the per-image differences are taken inside it,
    the group means are averaged, and the interval comes from resampling every
    group at once.
    """

    entry: dict = {"n_groups": len(groups), "n_pairs": 0}
    commons = [sorted(set(left) & set(right)) for left, right in groups]
    entry["n_pairs"] = sum(len(c) for c in commons)
    if not groups or any(not c for c in commons):
        return entry
    for i, m in enumerate(METRICS):
        arrays = []
        for (left, right), common in zip(groups, commons):
            pairs = [c for c in common if m in left[c] and m in right[c]]
            if not pairs:
                arrays = []
                break
            arrays.append(np.array([left[c][m] - right[c][m] for c in pairs]))
        if not arrays:
            continue
        lo, hi = bootstrap_ci_grouped(arrays, seed=i)
        entry[f"delta_{m}_mean"] = float(np.mean([a.mean() for a in arrays]))
        entry[f"ci95_{m}"] = [lo, hi]
        entry[f"n_pairs_{m}"] = int(sum(a.size for a in arrays))
        entry[f"share_positive_{m}"] = float(np.mean([(a > 0).mean() for a in arrays]))
    return entry


def payload_supplement(model: str, rows: list[dict], spx_rows: list[dict]) -> dict:
    """The delivered schedule of each setting under the five payloads.

    `delivery_best.txt` names the arbitration-best schedule of each (model, K).
    The supplement ran those schedules under the four forecast payloads on the
    four datasets and the three seed streams; their reuse rows are the P4 cells
    section 4 reads. Every difference against reuse is therefore between two
    runs of one schedule and only the payload differs. The PartiPrompts block
    puts the same schedule against the baseline methods' schedules under the
    same payload, reading the baseline rows from the SPX tables.
    """

    delivered = {r["k"]: r for r in read_schedule_list(SS_DIR / DELIVERY_BEST)
                 if r["model"] == model}
    if not delivered:
        return {}

    #: (k, payload, dataset) -> (seed, prompt index) -> metrics.
    scores: dict[tuple, dict[tuple, dict[str, float]]] = defaultdict(dict)
    duplicates: Counter = Counter()
    seeds: dict[tuple, set] = defaultdict(set)
    for r in rows:
        k = int(r["k"])
        row = delivered.get(k)
        if row is None or r["schedule"] != row["name"] or r["payload"] not in SUPPLEMENT_PAYLOADS:
            continue
        key = (k, r["payload"], r["dataset"])
        image = (int(r["seed"]), int(r["prompt_idx"]))
        if image in scores[key]:
            duplicates[key + (int(r["seed"]),)] += 1
        scores[key][image] = {m: float(r[m]) for m in METRICS
                              if r.get(m) not in (None, "", "nan")}
        seeds[key].add(int(r["seed"]))

    def cell_means(values: dict[tuple, dict[str, float]]) -> dict:
        out = {}
        for m in METRICS:
            got = [v[m] for v in values.values() if m in v]
            if got:
                out[f"{m}_mean"] = float(np.mean(got))
        return out

    cells = []
    for key in sorted(scores):
        k, payload, dataset = key
        cells.append({"k": k, "payload": payload, "schedule": delivered[k]["name"],
                      "dataset": dataset, "n": len(scores[key]),
                      "seeds": sorted(seeds[key]), **cell_means(scores[key])})

    dataset_mean_cells = []
    for k in sorted(delivered):
        for payload in SUPPLEMENT_PAYLOADS:
            group = [c for c in cells if c["k"] == k and c["payload"] == payload]
            if not group:
                continue
            entry = {"k": k, "payload": payload, "schedule": delivered[k]["name"],
                     "n_datasets": len(group), "n": sum(c["n"] for c in group),
                     "datasets": sorted(c["dataset"] for c in group)}
            for m in METRICS:
                got = [c[f"{m}_mean"] for c in group if f"{m}_mean" in c]
                if len(got) == len(group):
                    entry[f"{m}_mean"] = float(np.mean(got))
            dataset_mean_cells.append(entry)

    vs_reuse, vs_reuse_dataset_mean = [], []
    for k in sorted(delivered):
        for payload in SUPPLEMENT_PAYLOADS:
            if payload == "reuse":
                continue
            groups = []
            for dataset in DATASETS:
                left = scores.get((k, payload, dataset))
                right = scores.get((k, "reuse", dataset))
                if not left or not right:
                    continue
                differences = paired_metrics(left, right)
                if not differences["n_pairs"]:
                    continue
                vs_reuse.append({"k": k, "payload": payload,
                                 "schedule": delivered[k]["name"],
                                 "dataset": dataset, "reference_payload": "reuse",
                                 **differences})
                groups.append((left, right))
            if len(groups) > 1:
                vs_reuse_dataset_mean.append(
                    {"k": k, "payload": payload, "schedule": delivered[k]["name"],
                     "dataset": None, "reference_payload": "reuse",
                     **paired_metrics_over_groups(groups)})

    #: (k, payload, incumbent) -> (seed, prompt index) -> metrics, on Parti.
    spx: dict[tuple, dict[tuple, dict[str, float]]] = defaultdict(dict)
    for r in spx_rows:
        if r["schedule"] not in SUPPLEMENT_INCUMBENTS or r["payload"] not in SUPPLEMENT_PAYLOADS:
            continue
        spx[(int(r["k"]), r["payload"], r["schedule"])][(int(r["seed"]), int(r["prompt_idx"]))] = {
            m: float(r[m]) for m in METRICS if r.get(m) not in (None, "", "nan")}

    vs_incumbent = []
    for k in sorted(delivered):
        for payload in SUPPLEMENT_PAYLOADS:
            left = scores.get((k, payload, "parti_full"))
            if not left:
                continue
            for inc in SUPPLEMENT_INCUMBENTS:
                right = spx.get((k, payload, inc))
                if not right:
                    continue
                differences = paired_metrics(left, right)
                if not differences["n_pairs"]:
                    continue
                vs_incumbent.append({"k": k, "payload": payload,
                                     "schedule": delivered[k]["name"],
                                     "incumbent": inc, "dataset": "parti_full",
                                     "reference_source": "spx", **differences})

    incumbent_payloads = {inc: sorted({p for (_, p, name) in spx if name == inc},
                                      key=SUPPLEMENT_PAYLOADS.index)
                          for inc in SUPPLEMENT_INCUMBENTS}
    stream_seeds = sorted({s for key, got in seeds.items() if key[1] == "reuse"
                           for s in got})
    missing = [{"k": k, "payload": payload, "dataset": dataset, "seed": seed}
               for k in sorted(delivered) for payload in SUPPLEMENT_PAYLOADS
               for dataset in DATASETS for seed in stream_seeds
               if seed not in seeds.get((k, payload, dataset), set())]
    return {
        "payloads": list(SUPPLEMENT_PAYLOADS),
        "datasets": list(DATASETS),
        "seed_streams": stream_seeds,
        "delivered": [{"k": k, "schedule": delivered[k]["name"],
                       "bits": delivered[k]["bits"],
                       "full_steps": full_steps(delivered[k]["bits"])}
                      for k in sorted(delivered)],
        "cells": cells,
        "cells_dataset_mean": dataset_mean_cells,
        "vs_reuse": vs_reuse,
        "vs_reuse_dataset_mean": vs_reuse_dataset_mean,
        "vs_incumbent_parti": vs_incumbent,
        "incumbent_payloads": incumbent_payloads,
        "missing_cells": missing,
        "duplicated_images": [{"k": k, "payload": payload, "dataset": dataset,
                               "seed": seed, "n_images": n}
                              for (k, payload, dataset, seed), n in sorted(duplicates.items())],
    }


def arbitration_check(model: str, staged: dict) -> list[dict]:
    """Delivered schedule minus rejected calibration-best, on PartiPrompts.

    `delivery_calbest.txt` names, per (model, K, algorithm) where arbitration
    delivered something other than the algorithm's calibration-best candidate,
    the rejected candidate as `cal0_<algorithm>`. Its counterpart is the
    schedule that same algorithm delivered, read from the (model, K)
    arbitration record. Both were generated on PartiPrompts x 3 seed streams
    under the residual reuse payload, so the pair differs only in the schedule.
    """

    entries = []
    calbest = [r for r in read_schedule_list(SS_DIR / "delivery_calbest.txt")
               if r["model"] == model]
    for row in sorted(calbest, key=lambda r: (r["k"], r["name"])):
        k, name, bits = row["k"], row["name"], row["bits"]
        algorithm = name[len(CAL0_PREFIX):]
        record = SS_DIR / "arbitration" / f"{model}_k{k}.json"
        if not record.is_file():
            continue
        arb = json.loads(record.read_text(encoding="utf-8"))["arbitration"]
        delivered = next((d for d in arb["delivery"] if algorithm in d["algorithms"]), None)
        candidate = next((c for c in arb["candidates"] if c["bits"] == bits), None)
        if delivered is None or candidate is None:
            continue
        left = staged.get((k, delivered["name"], CAL0_DATASET), {})
        right = staged.get((k, name, CAL0_DATASET), {})
        differences = paired_metrics(left, right)
        if not differences["n_pairs"]:
            continue
        entry = {
            "k": k, "algorithm": algorithm, "dataset": CAL0_DATASET, "payload": PAYLOAD,
            "delivered": delivered["name"], "delivered_bits": delivered["bits"],
            "delivered_full_steps": full_steps(delivered["bits"]),
            "delivered_arbitration_mean_psnr_db": delivered["arbitration_mean_psnr_db"],
            "calibration_best": name, "calibration_best_bits": bits,
            "calibration_best_full_steps": full_steps(bits),
            "calibration_best_arbitration_mean_psnr_db":
                candidate["arbitration_mean_psnr_db"],
            "arbitration_n_prompts": arb["n_prompts"],
            **differences,
        }
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------
# the objective variants of section 4b
# --------------------------------------------------------------------------


def objective_suffix(objective: str) -> str:
    return "" if objective == "psnr" else f"_{objective}"


def read_arbitration(model: str, k: int, objective: str) -> dict | None:
    path = SS_DIR / "arbitration" / f"{model}_k{k}{objective_suffix(objective)}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    record["_file"] = path.name
    return record


def read_search_runs(model: str, k: int, objective: str,
                     warnings: list[str] | None = None) -> dict[str, dict]:
    """The search summaries of one (model, K, objective), keyed by algorithm."""

    suffix = objective_suffix(objective)
    runs = {}
    for path in sorted((SS_DIR / "search").glob(f"{model}_k{k}_*.json")):
        stem = path.stem[len(f"{model}_k{k}_"):]
        if suffix:
            if not stem.endswith(suffix):
                continue
            algorithm = stem[: -len(suffix)]
        else:
            algorithm = stem
            if any(algorithm.endswith(f"_{other}") for other in OBJECTIVES if other != "psnr"):
                continue
        if not algorithm:
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        recorded = str(run.get("objective", "psnr"))
        if recorded != objective:
            if warnings is not None:
                warnings.append(f"{path.name} records objective {recorded}, the "
                                f"filename says {objective}")
            continue
        runs[algorithm] = run
    return runs


def calibration_row(runs: dict[str, dict], bits: str) -> dict | None:
    """The 8-pair record of one delivered schedule, from a search summary.

    A delivered schedule is one of the candidates its algorithm sent to
    arbitration, so its calibration means are in that algorithm's summary; the
    selected row is read as well for the case where the algorithm delivered
    exactly what its own selection rule picked.
    """

    for algorithm, run in sorted(runs.items()):
        rows = list(run.get("arbitration_candidates") or [])
        selected = run.get("selected")
        if selected:
            rows.append(selected)
        for row in rows:
            if str(row.get("bits")) != bits:
                continue
            out = {
                "algorithm": algorithm,
                "mean_psnr_db": row.get("mean_psnr_db"),
                "mean_objective": row.get("mean_objective", row.get("mean_psnr_db")),
            }
            if row.get("metrics"):
                out["metrics"] = {m: float(v) for m, v in row["metrics"].items()}
            return out
    return None


def objective_settings(model: str, cells: list[dict], staged: dict) -> dict:
    """Per (K, objective) the delivered schedules and their numbers.

    Every objective's delivery comes from its own arbitration record; the
    delivery lists on disk say which of those schedules were evaluated and
    which repeat a bitstring an earlier objective already delivered.
    """

    forced = forced_full_steps(model)
    cell_index: dict[tuple, dict] = {(c["k"], c["schedule"], c["dataset"]): c for c in cells}
    delivery: dict[str, list[dict]] = {
        objective: [r for r in read_delivery_list(
            SS_DIR / f"delivery{objective_suffix(objective)}.txt") if r["model"] == model]
        for objective in OBJECTIVES
    }
    evaluated = {r["name"] for r in read_delivery_list(SS_DIR / "delivery_objectives.txt")
                 if r["model"] == model and r["evaluated"]}
    settings: list[dict] = []
    warnings: list[str] = []
    #: (k, algorithm) -> the PSNR objective's delivery row, for the pairing.
    psnr_by_algorithm: dict[tuple, dict] = {}
    for k in BUDGETS:
        for objective in OBJECTIVES:
            record = read_arbitration(model, k, objective)
            if record is None:
                continue
            arb = record["arbitration"]
            recorded = str(arb.get("objective", "psnr"))
            if recorded != objective:
                warnings.append(f"{record['_file']} arbitrated objective {recorded}, "
                                f"the filename says {objective}")
                continue
            runs = read_search_runs(model, k, objective, warnings)
            listed = {r["name"]: r for r in delivery[objective] if r["k"] == k}
            rows = []
            for row in arb["delivery"]:
                bits = str(row["bits"])
                listed_row = listed.get(row["name"], {})
                identical_to = listed_row.get("identical_to")
                if identical_to is None and objective != "psnr":
                    same = next((d for d in psnr_by_algorithm.values()
                                 if d["k"] == k and d["bits"] == bits), None)
                    if same is not None:
                        identical_to = same["name"]
                cells_name = identical_to or row["name"]
                entry = {
                    "name": row["name"],
                    "bits": bits,
                    "full_steps": full_steps(bits),
                    "algorithms": list(row.get("algorithms") or []),
                    "listed_in_delivery": row["name"] in listed,
                    "identical_to": identical_to,
                    "evaluated_as": cells_name,
                    "in_delivery_objectives": (cells_name in evaluated) if (evaluated and objective != "psnr") else None,
                    "structure": structure(bits, forced),
                    "arbitration": {
                        "mean_objective": row.get("arbitration_mean_objective",
                                                  row.get("arbitration_mean_psnr_db")),
                        "min_objective": row.get("arbitration_min_objective",
                                                 row.get("arbitration_min_psnr_db")),
                        "mean_psnr_db": row.get("arbitration_mean_psnr_db"),
                        "metrics": dict(row.get("arbitration_metrics") or {}),
                        "n_prompts": arb.get("n_prompts"),
                    },
                    "calibration": calibration_row(runs, bits),
                    "cells": [cell_index[key] for key in sorted(cell_index)
                              if key[0] == k and key[1] == cells_name],
                }
                rows.append(entry)
                if objective == "psnr":
                    for algorithm in entry["algorithms"]:
                        psnr_by_algorithm[(k, algorithm)] = {
                            "k": k, "name": row["name"], "bits": bits,
                            "algorithms": entry["algorithms"]}
            units = next((str(r["objective_units"]) for r in runs.values()
                          if r.get("objective_units")), objective_units(objective))
            if units != objective_units(objective):
                warnings.append(f"{model} K{k} {objective} search summaries record "
                                f"units {units}, lib.schedule_search says "
                                f"{objective_units(objective)}")
            settings.append({
                "k": k,
                "objective": objective,
                "objective_units": units,
                "arbitration_file": record["_file"],
                "n_arbitration_prompts": arb.get("n_prompts"),
                "search_algorithms": sorted(runs),
                "schedules": rows,
            })
    return {"settings": settings, "psnr_by_algorithm": psnr_by_algorithm,
            "warnings": warnings}


def objective_paired(model: str, settings: dict, staged: dict) -> list[dict]:
    """Objective schedule minus the PSNR schedule of the same algorithm family.

    A delivered schedule can carry several algorithms, so the counterpart is
    looked up per algorithm and identical counterparts are reported once. Both
    sides run the residual reuse payload on the same prompts and seeds.
    """

    entries = []
    psnr_by_algorithm = settings["psnr_by_algorithm"]
    for setting in settings["settings"]:
        if setting["objective"] == "psnr":
            continue
        k = setting["k"]
        for row in setting["schedules"]:
            counterparts: dict[str, list[str]] = {}
            for algorithm in row["algorithms"]:
                other = psnr_by_algorithm.get((k, algorithm))
                if other is not None:
                    counterparts.setdefault(other["name"], []).append(algorithm)
            for name, algorithms in counterparts.items():
                reference = next(d for d in psnr_by_algorithm.values()
                                 if d["k"] == k and d["name"] == name)
                base = {"k": k, "objective": setting["objective"],
                        "schedule": row["name"], "bits": row["bits"],
                        "reference": name, "reference_bits": reference["bits"],
                        "shared_algorithms": algorithms, "payload": PAYLOAD,
                        "identical_bits": row["bits"] == reference["bits"]}
                if base["identical_bits"]:
                    entries.append({**base, "dataset": None, "n_pairs": 0})
                    continue
                datasets = sorted({key[2] for key in staged
                                   if key[0] == k and key[1] == row["evaluated_as"]})
                for dataset in datasets:
                    left = staged.get((k, row["evaluated_as"], dataset), {})
                    right = staged.get((k, name, dataset), {})
                    differences = paired_metrics(left, right)
                    if not differences["n_pairs"]:
                        continue
                    entries.append({**base, "dataset": dataset, **differences})
    return entries


def main() -> int:
    out: dict = {"models": {}}
    for model in ("flux", "qwen"):
        # One staged table per cluster that ran cells (Site C, Site B, ...).
        paths = sorted(SS_DIR.glob(f"perprompt_search_{model}*.tsv.gz"))
        if not paths:
            continue
        rows = []
        for path in paths:
            rows.extend(read_tsv(path))
        cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        per_seed: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        #: per-image metrics of every staged row, keyed (k, schedule, dataset)
        #: then (seed, prompt_idx) -> {metric: value}.
        staged: dict[tuple, dict[tuple, dict[str, float]]] = defaultdict(dict)
        for r in rows:
            # The cell key is (k, schedule, dataset); a cell run under another
            # payload would land on the same key and be pooled with the reuse
            # rows this comparison is defined over.
            if r["payload"] != PAYLOAD:
                continue
            key = (int(r["k"]), r["schedule"], r["dataset"])
            vals = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "", "nan")}
            staged[key][(int(r["seed"]), int(r["prompt_idx"]))] = vals
            if r["schedule"].startswith(CAL0_PREFIX):
                # A rejected calibration-best candidate, not a delivered
                # schedule: it feeds the arbitration check only.
                continue
            for m, v in vals.items():
                cells[key][m].append(v)
            per_seed[key][int(r["seed"])].append(float(r["psnr"]))
        model_out: dict = {"cells": [], "paired": [], "incumbent_parti": [],
                           "arbitration_check": []}
        for key in sorted(cells):
            k, schedule, dataset = key
            model_out["cells"].append(
                {"k": k, "schedule": schedule, "dataset": dataset,
                 "n": len(cells[key]["psnr"]),
                 **{f"{m}_mean": float(np.mean(cells[key][m])) for m in METRICS if cells[key][m]},
                 "psnr_by_seed": {str(s): float(np.mean(v)) for s, v in sorted(per_seed[key].items())}}
            )
        #: reference per-image PSNR under the reuse payload, keyed
        #: (k, incumbent, dataset), with the table it came from. The incumbents'
        #: rows come from the staged search tables wherever they exist, and from
        #: the SPX Parti tables otherwise.
        ref: dict[tuple, dict[tuple, dict[str, float]]] = {}
        ref_source: dict[tuple, str] = {}
        for (k, schedule, dataset), scores in staged.items():
            if schedule in INCUMBENTS:
                ref[(k, schedule, dataset)] = scores
                ref_source[(k, schedule, dataset)] = "schedule_search"
        spx_path = SPX_DIR / f"perprompt_spx_{model}.tsv.gz"
        spx_rows: list[dict] = []
        if spx_path.is_file():
            spx = spx_rows = read_tsv(spx_path)
            spx_scores: dict[tuple, dict[tuple, dict[str, float]]] = defaultdict(dict)
            ref_metrics: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in spx:
                if r["schedule"] not in INCUMBENTS or r["payload"] != PAYLOAD:
                    continue
                key = (int(r["k"]), r["schedule"], "parti_full")
                vals = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "", "nan")}
                spx_scores[key][(int(r["seed"]), int(r["prompt_idx"]))] = vals
                for m, v in vals.items():
                    ref_metrics[key][m].append(v)
            for key in sorted(spx_scores):
                if key not in ref:
                    ref[key] = spx_scores[key]
                    ref_source[key] = "spx"
            for key in sorted(ref_metrics):
                k, inc, _ = key
                model_out["incumbent_parti"].append(
                    {"k": k, "schedule": inc, "payload": PAYLOAD,
                     "n": len(ref_metrics[key]["psnr"]),
                     **{f"{m}_mean": float(np.mean(ref_metrics[key][m]))
                        for m in METRICS if ref_metrics[key][m]}}
                )
        for (k, schedule, dataset), scores in sorted(staged.items()):
            if schedule in INCUMBENTS or schedule.startswith(CAL0_PREFIX):
                continue
            for inc in INCUMBENTS:
                key = (k, inc, dataset)
                base = ref.get(key)
                if not base:
                    continue
                common = sorted(set(scores) & set(base))
                if not common:
                    continue
                diff = np.array([scores[c]["psnr"] - base[c]["psnr"] for c in common])
                lo, hi = bootstrap_ci(diff)
                entry = {"k": k, "schedule": schedule, "dataset": dataset,
                         "incumbent": inc, "payload": PAYLOAD,
                         "reference_source": ref_source[key],
                         "n_pairs": int(diff.size), "delta_psnr_mean": float(diff.mean()),
                         "ci95": [lo, hi], "share_positive": float((diff > 0).mean())}
                # The same paired difference on the other metrics (searched minus
                # reference; for LPIPS a negative value favours the searched schedule).
                for m in METRICS[1:]:
                    pairs_m = [c for c in common if m in scores[c] and m in base[c]]
                    if not pairs_m:
                        continue
                    d = np.array([scores[c][m] - base[c][m] for c in pairs_m])
                    lo_m, hi_m = bootstrap_ci(d, seed=1)
                    entry[f"delta_{m}_mean"] = float(d.mean())
                    entry[f"ci95_{m}"] = [lo_m, hi_m]
                model_out["paired"].append(entry)
        model_out["arbitration_check"] = arbitration_check(model, staged)
        settings = objective_settings(model, model_out["cells"], staged)
        model_out["objectives"] = {
            "early_step": EARLY_STEP,
            "settings": settings["settings"],
            "paired": objective_paired(model, settings, staged),
            "warnings": settings["warnings"],
        }
        model_out["payloads"] = payload_supplement(model, rows, spx_rows)
        out["models"][model] = model_out
    (SS_DIR / "results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for model, m in out["models"].items():
        for c in m["cells"]:
            print(f"{model} K{c['k']} {c['schedule']:18s} {c['dataset']:20s} n={c['n']:5d} psnr {c.get('psnr_mean', float('nan')):.3f} ssim {c.get('ssim_mean', float('nan')):.4f} lpips {c.get('lpips_mean', float('nan')):.4f} ir {c.get('image_reward_mean', float('nan')):.3f}")
        for p in m["paired"]:
            print(f"{model} K{p['k']} {p['schedule']:18s} {p['dataset']:20s} vs {p['incumbent']:10s} ({p['reference_source']}) dPSNR {p['delta_psnr_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  n={p['n_pairs']}  share>0 {p['share_positive']:.2f}")
        for a in m["arbitration_check"]:
            print(f"{model} K{a['k']} arbitration {a['algorithm']:8s} {a['delivered']:18s} vs {a['calibration_best']:14s} arb {a['delivered_arbitration_mean_psnr_db']:.3f} vs {a['calibration_best_arbitration_mean_psnr_db']:.3f}  dPSNR {a['delta_psnr_mean']:+.3f} [{a['ci95_psnr'][0]:+.3f}, {a['ci95_psnr'][1]:+.3f}]  n={a['n_pairs']}")
        for w in m["objectives"]["warnings"]:
            print(f"{model} objective artifact mismatch: {w}")
        for s in m["objectives"]["settings"]:
            names = ", ".join(r["name"] for r in s["schedules"])
            print(f"{model} K{s['k']} objective {s['objective']:14s} "
                  f"searches {','.join(s['search_algorithms']) or '--':22s} "
                  f"delivers {names}")
        for p in m["objectives"]["paired"]:
            if p["identical_bits"]:
                print(f"{model} K{p['k']} {p['schedule']:22s} is the same bitstring as {p['reference']}")
                continue
            print(f"{model} K{p['k']} {p['schedule']:22s} {p['dataset']:20s} vs {p['reference']:18s} "
                  f"dPSNR {p['delta_psnr_mean']:+.3f} [{p['ci95_psnr'][0]:+.3f}, {p['ci95_psnr'][1]:+.3f}]  "
                  f"dLPIPS {p.get('delta_lpips_mean', float('nan')):+.4f}  n={p['n_pairs']}")
        block = m.get("payloads") or {}
        for c in block.get("cells_dataset_mean", []):
            print(f"{model} K{c['k']} payload {c['payload']:14s} {c['schedule']:10s} "
                  f"{c['n_datasets']} datasets n={c['n']:6d} psnr {c.get('psnr_mean', float('nan')):.3f} "
                  f"ssim {c.get('ssim_mean', float('nan')):.4f} lpips {c.get('lpips_mean', float('nan')):.4f}")
        for p in block.get("vs_reuse_dataset_mean", []):
            print(f"{model} K{p['k']} payload {p['payload']:14s} vs reuse, four-dataset mean "
                  f"dPSNR {p['delta_psnr_mean']:+.3f} [{p['ci95_psnr'][0]:+.3f}, {p['ci95_psnr'][1]:+.3f}]  "
                  f"n={p['n_pairs']}")
        for p in block.get("vs_incumbent_parti", []):
            print(f"{model} K{p['k']} payload {p['payload']:14s} parti vs {p['incumbent']:10s} "
                  f"dPSNR {p['delta_psnr_mean']:+.3f} [{p['ci95_psnr'][0]:+.3f}, {p['ci95_psnr'][1]:+.3f}]  "
                  f"n={p['n_pairs']}")
        if block.get("missing_cells"):
            print(f"{model} payload cells missing: {len(block['missing_cells'])} "
                  f"{block['missing_cells'][:8]}")
        if block.get("duplicated_images"):
            print(f"{model} payload rows staged twice: {block['duplicated_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
