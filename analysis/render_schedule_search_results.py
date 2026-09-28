"""P5 of `docs/schedule_search_plan_zh.md`: render `docs/schedule_search_results.md`.

Every number in the document comes from the staged artifacts under
`resources/schedule_search/`:

    config.v1.json                 frozen calibration slots, arbitration set,
                                   caps, chain lengths, temperatures, warm starts
    probes/<model>_k<K>.json       P1 probe readings
    search/<model>_k<K>_<alg>.json P2 search summaries
    arbitration/<model>_k<K>.json  P3 50-caption arbitration and delivery
    delivery.txt                   the P4 delivery list
    k41_table_placement.json       FLUX K41 lookup in the exhaustive table
    results.json                   P4 cell means and the paired per-image
                                   differences against the baseline methods' schedules
                                   under the same residual reuse payload
                                   (written by analysis/schedule_search_results.py)

The objective variants of section 4b of the plan add, for each objective other
than PSNR:

    search/<model>_k<K>_<alg>_<objective>.json   its search summaries
    arbitration/<model>_k<K>_<objective>.json    its arbitration and delivery
    delivery_<objective>.txt                     its delivery list
    delivery_objectives.txt                      the union that was evaluated

Their numbers reach section 5 through the `objectives` block of `results.json`.

Section 6 reads the `payloads` block of the same file: the arbitration-best schedule
of each setting (`delivery_best.txt`) run under the four forecast payloads as well as
under residual reuse.

plus the baseline methods' schedules in `resources/sp_cross_schedules/`.

Settings whose artifacts have not arrived yet are omitted from the tables and
listed as absent in section 1.  Run from the repository root:

    python analysis/render_schedule_search_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SS_DIR = REPO / "resources" / "schedule_search"
SPX_SCHEDULE_DIR = REPO / "resources" / "sp_cross_schedules"
OUT = REPO / "docs" / "schedule_search_results.md"

MODELS = (("flux", "FLUX.1-dev"), ("qwen", "Qwen-Image"))
BUDGETS = (29, 37, 41)
ALGORITHMS = ("random", "hill", "anneal", "greedy")
INCUMBENTS = ("meancache", "budcache")
PAYLOAD_NAMES = {"reuse": "residual reuse",
                 "taylor_o1": "first-order Taylor extrapolation",
                 "hermite_o2": "second-order Hermite extrapolation",
                 "mean_avg_vel": "interval average velocity",
                 "di_two_anchor": "two-anchor extrapolation"}
#: The payload axis of section 6, reuse first.
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
SOURCE_NAMES = {"schedule_search": "this experiment's P4 tables",
                "spx": "`resources/spx/`"}
DATASETS = (
    ("drawbench_full", "DrawBench 200"),
    ("parti_full", "PartiPrompts 1632"),
    ("geneval_style", "GenEval-style 553"),
    ("diffusiondb_clean10k", "DiffusionDB clean10k"),
)
DATASET_NAME = dict(DATASETS)
CLUSTER_NAMES = {"site_c": "Site C", "site_b": "Site B", "site_a": "Site A"}
#: The search objectives of section 4b of the plan, PSNR first.
OBJECTIVES = ("psnr", "lpips", "psnr_lpips_z")
OBJECTIVE_NAMES = {"psnr": "PSNR", "lpips": "LPIPS",
                   "psnr_lpips_z": "standardized PSNR plus LPIPS"}
#: What each objective maximises, for the lead-in of section 5.
OBJECTIVE_RULES = {
    "psnr": "the mean PSNR of the 8 calibration pairs, the objective sections 2 "
            "to 4 report",
    "lpips": "the mean LPIPS of the 8 pairs with its sign flipped, so a lower "
             "LPIPS is a higher objective",
    "psnr_lpips_z": "the sum of the mean PSNR and the negated mean LPIPS after "
                    "each has been standardized by the mean and standard "
                    "deviation the setting's probe pool measured for it",
}
METRIC_COLUMNS = (("psnr", "PSNR (dB)", 3), ("ssim", "SSIM", 4),
                  ("lpips", "LPIPS", 4), ("image_reward", "ImageReward", 3),
                  ("clip", "CLIP", 3))
EARLY_STEP = 13  # the "early" window used by the structure columns of section 3


# ---------------------------------------------------------------- loading


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_bits(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def read_delivery(path: Path) -> list[dict]:
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


def load() -> dict:
    data = {
        "config": read_json(SS_DIR / "config.v1.json"),
        "placement": read_json(SS_DIR / "k41_table_placement.json"),
        "results": read_json(SS_DIR / "results.json") or {"models": {}},
        "delivery": read_delivery(SS_DIR / "delivery.txt"),
        "probes": {},
        "search": {},
        "arbitration": {},
        "incumbents": {},
        "objective_files": {},
        "staged_files": sorted(p.name for p in SS_DIR.glob("perprompt_search_*.tsv.gz")),
    }
    for model, _ in MODELS:
        for k in BUDGETS:
            key = (model, k)
            data["probes"][key] = read_json(SS_DIR / "probes" / f"{model}_k{k}.json")
            data["arbitration"][key] = read_json(SS_DIR / "arbitration" / f"{model}_k{k}.json")
            for alg in ALGORITHMS:
                run = read_json(SS_DIR / "search" / f"{model}_k{k}_{alg}.json")
                if run is not None:
                    data["search"][(model, k, alg)] = run
            for inc in INCUMBENTS:
                bits = read_bits(SPX_SCHEDULE_DIR / f"{model}_k{k}_{inc}.txt")
                if bits is not None:
                    data["incumbents"][(model, k, inc)] = bits
            for objective in OBJECTIVES:
                suffix = "" if objective == "psnr" else f"_{objective}"
                data["objective_files"][(model, k, objective)] = {
                    "arbitration": (SS_DIR / "arbitration"
                                    / f"{model}_k{k}{suffix}.json").is_file(),
                    "algorithms": [alg for alg in ALGORITHMS
                                   if (SS_DIR / "search"
                                       / f"{model}_k{k}_{alg}{suffix}.json").is_file()],
                }
    return data


# ---------------------------------------------------------------- helpers


def full_steps(bits: str) -> list[int]:
    return [i for i, c in enumerate(bits) if c == "0"]


def hamming(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def first_cached(bits: str) -> int:
    return bits.index("1")


def longest_cached_run(bits: str) -> int:
    best = run = 0
    for c in bits:
        run = run + 1 if c == "1" else 0
        best = max(best, run)
    return best


def free_steps(bits: str, forced: tuple[int, ...]) -> list[int]:
    return [s for s in full_steps(bits) if s not in forced]


def steps_str(bits: str) -> str:
    return ", ".join(str(s) for s in full_steps(bits))


def fmt(value, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def starred(entry: dict, metric: str, digits: int, ci_key: str | None = None) -> str:
    """A signed paired difference, with `*` when its interval excludes zero."""

    key = f"delta_{metric}_mean"
    if key not in entry:
        return "n/a"
    lo, hi = entry[ci_key or f"ci95_{metric}"]
    return f"{signed(entry[key], digits)}{'*' if (lo > 0 or hi < 0) else ''}"


def thousands(value: int) -> str:
    return f"{value:,}"


def table(header: list[str], rows: list[list[str]], align: str = "") -> list[str]:
    align = align or "l" * len(header)
    sep = {"l": "---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(sep[a] for a in align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def delivered_for(data: dict, model: str, k: int) -> list[dict]:
    """Delivered schedules of one setting, in the arbitration record's order."""
    record = data["arbitration"].get((model, k))
    if record is None:
        return []
    names = {row["name"] for row in data["delivery"]
             if row["model"] == model and row["k"] == k}
    rows = [dict(row) for row in record["arbitration"]["delivery"] if row["name"] in names]
    return rows


def settings_present(data: dict) -> list[tuple[str, int]]:
    return [(model, k) for model, _ in MODELS for k in BUDGETS
            if data["arbitration"].get((model, k)) is not None]


def setting_index(data: dict) -> dict[tuple[str, int], int]:
    """Subsection number of each rendered setting, 1-based and gap-free."""

    return {key: i + 1 for i, key in enumerate(settings_present(data))}


def rng(values: list) -> str:
    """`8` when the range is a point, `8--12` otherwise."""

    lo, hi = min(values), max(values)
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def join_and(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def model_title(model: str) -> str:
    return dict(MODELS)[model]


# ---------------------------------------------------------------- section 1


def section_setup(data: dict) -> list[str]:
    cfg = data["config"]
    out = ["## 1. Setup", ""]

    rendered = settings_present(data)
    rows = []
    for model, _ in MODELS:
        for k in BUDGETS:
            key = (model, k)
            probe = "yes" if data["probes"].get(key) else "no"
            n_search = sum(1 for alg in ALGORITHMS if (model, k, alg) in data["search"])
            arb = data["arbitration"].get(key)
            delivered = delivered_for(data, model, k)
            cells = [c for c in data["results"]["models"].get(model, {}).get("cells", [])
                     if c["k"] == k]
            sets_done = sorted({c["dataset"] for c in cells}, key=lambda d: [n for n, _ in DATASETS].index(d))
            rows.append([
                model_title(model), f"K{k}", probe, f"{n_search}/4",
                "yes" if arb else "no",
                str(len(delivered)) if delivered else "--",
                ", ".join(DATASET_NAME[d] for d in sets_done) if sets_done else "--",
            ])
    out += ["**Table 1.0: which artifacts are staged.** A setting is rendered in the "
            "sections below once its P3 arbitration record exists; sections 2 and 3 need "
            "only that, section 4 needs staged P4 cells as well.", ""]
    out += table(
        ["Model", "Budget", "P1 probe", "P2 searches", "P3 arbitration",
         "Delivered schedules", "P4 datasets staged"],
        rows, align="llcccrl")
    out += ["",
            f"Settings rendered below: "
            f"{join_and([f'{model_title(m)} K{k}' for m, k in rendered])}.", ""]

    out += ["### 1.1 The space that is searched", ""]
    rows = []
    for model, title in MODELS:
        space = cfg["spaces"][model]
        for k in BUDGETS:
            rows.append([
                title, f"K{k}", str(space["num_steps"]),
                ", ".join(str(s) for s in space["forced_full_steps"]),
                f"{space['variable_start']}..{space['variable_end']}",
                str(space["free_full_count"][str(k)]),
                thousands(space["space_size"][str(k)]),
            ])
    out += table(
        ["Model", "Budget", "Steps", "Forced full steps", "Free positions",
         "Free full steps", "Schedules in the space"],
        rows, align="llrllrr")
    out += ["",
            f"K is the number of cached steps out of {cfg['spaces']['flux']['num_steps']}. "
            f"The payload is {PAYLOAD_NAMES['reuse']} for every search and for every "
            "evaluation of sections 2 to 5, so only the position of the cached steps "
            "varies; section 6 keeps the schedule and varies the payload.", ""]

    out += ["### 1.2 Calibration set", ""]
    rows = [[str(i), pair["slot"], f"`{pair['prompt']}`", str(pair["seed"]),
             pair["source"]["dataset"]]
            for i, pair in enumerate(cfg["calibration"]["pairs"])]
    out += table(["#", "Slot", "Prompt", "Seed", "Source"], rows, align="rllrl")
    seeds = cfg["calibration"]["seeds"]
    out += ["",
            f"One evaluation of a schedule is the mean PSNR of these "
            f"{len(cfg['calibration']['pairs'])} pairs against their own full-compute "
            f"references at the same prompt and seed. Seeds {seeds[0]} and {seeds[1]} "
            f"({cfg['calibration']['seed_rule']}). The number of these prompts that also "
            "appear in one of the four evaluation sets is "
            f"{cfg['checks']['calibration_prompts_in_evaluation_sets']}.", ""]

    out += ["### 1.3 Arbitration set", ""]
    arb_cfg = cfg["arbitration"]
    out += [f"{len(arb_cfg['prompts'])} held-out COCO 2014 val captions at seed "
            f"{arb_cfg['seed']}, taken by the rule: {arb_cfg['rule']}. "
            f"Caption ids {arb_cfg['prompts'][0]['caption_id']} to "
            f"{arb_cfg['prompts'][-1]['caption_id']}; the full list is appendix A.", ""]

    out += ["### 1.4 Searchers", ""]
    labels = {}
    for (model, k, alg), run in data["search"].items():
        labels[alg] = run.get("label", alg)
    rows = [[f"`ss_{alg}`", labels.get(alg, alg)] for alg in ALGORITHMS]
    out += table(["Delivery name", "Algorithm"], rows)
    search_cfg = cfg["search"]
    out += ["",
            f"All four run on the same evaluation callback and the same budget rule. "
            f"Warm starts are the baseline schedules of the same setting "
            f"(section 3); the swap window is +/- {search_cfg['hill_window']} steps and "
            f"{search_cfg['local_probability']:.0%} of annealing proposals are local. "
            f"Search seed {search_cfg['search_seed']}, shared by both models, so the "
            "random control draws the same pool in each.", ""]

    out += ["### 1.5 Budget, stopping and temperature", ""]
    rows = []
    for k in BUDGETS:
        rows.append([f"K{k}", thousands(search_cfg["caps"][str(k)]),
                     thousands(search_cfg["chain_lengths"][str(k)]),
                     str(search_cfg["probe_evals"]),
                     f"{search_cfg['se_factor']:.0f} x SE over {search_cfg['stop_units']} units"])
    out += table(["Budget", "Cap (evaluations)", "Annealing chain length",
                  "Probe evaluations", "Stopping rule"], rows, align="lrrrl")
    out += ["",
            f"The cap is shared by all four algorithms of a setting and includes the "
            f"{search_cfg['probe_evals']} probe evaluations, so a search run is allowed "
            f"{search_cfg['caps']['41'] - search_cfg['probe_evals']} / "
            f"{search_cfg['caps']['37'] - search_cfg['probe_evals']} / "
            f"{search_cfg['caps']['29'] - search_cfg['probe_evals']} evaluations at "
            "K41 / K37 / K29. Repeat visits to an already scored schedule are replayed "
            "from the run's own trace and do not consume budget.", ""]

    rows = []
    for model, title in MODELS:
        for k in BUDGETS:
            probe = data["probes"].get((model, k))
            if probe is None:
                continue
            p = probe["probe"]
            rows.append([
                title, f"K{k}", str(p["n_evals"]),
                f"{p['mean_psnr_db']['min']:.2f}--{p['mean_psnr_db']['max']:.2f}",
                fmt(p["mean_psnr_db"]["std"], 3),
                fmt(p["swap_delta_db"]["median_abs"], 4),
                fmt(p["calibration_se_db"], 3),
                fmt(cfg["search"]["temperatures"][model][str(k)]["t_max"], 2),
                fmt(cfg["search"]["temperatures"][model][str(k)]["t_min"], 5),
            ])
    out += ["**Table 1.5: P1 probe readings and the temperatures backfilled from them.** "
            "The probe evaluates 50 random schedules of the setting, 25 of them as "
            "one-swap pairs; the swap delta is the absolute change in the eight-pair "
            "mean across a pair. All values in dB.", ""]
    out += table(["Model", "Budget", "Probe evaluations", "Mean PSNR range",
                  "Mean PSNR sd", "Median swap delta", "Calibration SE",
                  "t_max", "t_min"], rows, align="llrrrrrrr")
    out += ["", f"Temperature rule: {cfg['search']['temperatures_note']}", ""]

    out += ["### 1.6 Arbitration and delivery", ""]
    out += [f"Each algorithm hands the arbitration stage its top "
            f"{search_cfg['arbitration_candidates']} candidates that are pairwise at "
            f"least {search_cfg['arbitration_min_hamming']} bits apart. All candidates "
            f"of a setting are re-scored on the {len(arb_cfg['prompts'])} arbitration "
            "captions, and each algorithm delivers the candidate of highest arbitration "
            "mean. Algorithms that deliver the same bitstring are merged into one "
            "schedule under a joined name such as `ss_anneal+greedy`.", ""]

    out += ["### 1.7 Evaluation protocol", ""]
    streams = cfg["checks"]["evaluation_seed_streams"]
    rows = []
    for model, title in MODELS:
        rows.append([title,
                     ", ".join(str(a) for a, _ in streams[model]),
                     ", ".join(f"{a}..{b}" for a, b in streams[model])])
    out += ["**Table 1.7: evaluation seed streams.** A stream is labelled by its base "
            "seed; the seed of a prompt is that base plus the prompt index, so a stream "
            "covers the range in the last column. The calibration seeds "
            f"({', '.join(str(s) for s in cfg['checks']['calibration_seeds'])}) lie "
            "outside every stream.", ""]
    out += table(["Model", "Stream labels", "Seeds covered"], rows)
    out += ["",
            "Each delivered schedule is run through the image baseline-matrix "
            "protocol: four datasets "
            f"({', '.join(name for _, name in DATASETS)}) x three seed streams, "
            "residual reuse payload, compared image by image against the matrix's own "
            "full-compute references at the same prompt and the same seed. Metrics are "
            "PSNR (dB), SSIM, LPIPS, ImageReward and CLIP score.", ""]

    if data["staged_files"]:
        rows = []
        for name in data["staged_files"]:
            stem = name[len("perprompt_search_"):-len(".tsv.gz")]
            model, _, suffix = stem.rpartition("_")
            rows.append([model_title(model), CLUSTER_NAMES.get(suffix, suffix), f"`{name}`"])
        out += ["The per-image metrics are staged one table per cluster that ran cells:", ""]
        out += table(["Model", "Cluster", "Staged table"], rows)
        out += [""]
    return out


# ---------------------------------------------------------------- section 2


def section_agreement(data: dict) -> list[str]:
    out = ["## 2. Q1: do the four algorithms find the same schedule?", ""]
    rendered = settings_present(data)
    index = setting_index(data)
    out += ["Stop reasons in the per-run tables: `cap` means the run reached the hard "
            "evaluation cap of its setting, `loop_end` means the control arm finished its "
            "fixed sweep over that same budget. A restart unit is one completed chain, "
            "climb or slot sweep; the stopping rule needs two consecutive units improving "
            "the best mean by less than twice the standard error.", ""]

    summary_rows = []
    for model, k in rendered:
        delivered = delivered_for(data, model, k)
        bits = [row["bits"] for row in delivered]
        dists = [hamming(bits[i], bits[j]) for i in range(len(bits)) for j in range(i + 1, len(bits))]
        merged = [row["name"] for row in delivered if len(row["algorithms"]) > 1]
        best = max(delivered, key=lambda r: r["arbitration_mean_psnr_db"])
        rnd = next((r for r in delivered if r["algorithms"] == ["random"]), None)
        summary_rows.append([
            model_title(model), f"K{k}",
            f"{len(delivered)}/{len(ALGORITHMS)}",
            ", ".join(f"`{m}`" for m in merged) if merged else "none",
            rng(dists) if dists else "--",
            f"`{best['name']}`",
            fmt(best["arbitration_mean_psnr_db"], 3),
            signed(rnd["arbitration_mean_psnr_db"] - best["arbitration_mean_psnr_db"], 3) if rnd else "--",
        ])
    out += ["**Table 2.0: one row per setting.** \"Distinct delivered\" counts the "
            "bitstrings the four algorithms hand over after merging identical ones. "
            "The last column is the random control's arbitration mean minus the best "
            "delivered schedule's, in dB.", ""]
    out += table(["Model", "Budget", "Distinct delivered", "Merged", "Pairwise Hamming",
                  "Best delivered", "Its arbitration mean (dB)", "Random control (dB)"],
                 summary_rows, align="llcclrrr")
    out += [""]

    for model, k in rendered:
        record = data["arbitration"][(model, k)]["arbitration"]
        delivered = delivered_for(data, model, k)
        tag = f"2.{index[(model, k)]}"
        out += [f"### {tag} {model_title(model)} K{k}", ""]

        rows = []
        for alg in ALGORITHMS:
            run = data["search"].get((model, k, alg))
            if run is None:
                continue
            winner_bits = record["winners"].get(alg)
            winner = next((c for c in record["candidates"] if c["bits"] == winner_bits), None)
            delivered_name = next((r["name"] for r in delivered if alg in r["algorithms"]), "--")
            rows.append([
                f"`{alg}`",
                f"{thousands(run['evaluations'])} / {thousands(run['max_evals'])}",
                str(run["replayed_evaluations"]),
                str(run["restart_units"]),
                run["stop_reason"],
                fmt(run["selected"]["mean_psnr_db"], 3),
                fmt(run["arbitration_candidates"][0]["mean_psnr_db"], 3),
                fmt(winner["arbitration_mean_psnr_db"], 3) if winner else "--",
                fmt(winner["arbitration_min_psnr_db"], 3) if winner else "--",
                f"`{delivered_name}`" if delivered_name != "--" else "--",
            ])
        out += [f"**Table {tag}a: the four search runs.** \"Selected\" is the "
                "schedule the algorithm's own selection rule picked on the 8 calibration "
                "pairs; \"best candidate\" is the highest calibration mean among the three "
                "candidates it sent to arbitration; the arbitration columns are the "
                f"candidate that won the {record['n_prompts']}-caption re-score for that "
                "algorithm. All PSNR values in dB.", ""]
        out += table(["Algorithm", "Evaluations", "Replayed", "Restart units", "Stop reason",
                      "Selected calibration mean", "Best candidate calibration mean",
                      "Arbitration mean", "Arbitration worst caption", "Delivers"],
                     rows, align="lrrrlrrrrl")
        out += [""]

        rows = []
        for row in delivered:
            rows.append([
                f"`{row['name']}`",
                ", ".join(f"`{a}`" for a in row["algorithms"]),
                fmt(row["arbitration_mean_psnr_db"], 3),
                fmt(row["arbitration_min_psnr_db"], 3),
                steps_str(row["bits"]),
            ])
        out += [f"**Table {tag}b: the delivered schedules.** Full steps are the "
                "positions computed at full cost; every other step of the 50 is cached.", ""]
        out += table(["Name", "Algorithms", "Arbitration mean (dB)",
                      "Worst caption (dB)", "Full steps"], rows, align="llrrl")
        out += [""]

        if len(delivered) > 1:
            names = [row["name"] for row in delivered]
            rows = []
            for left in delivered:
                rows.append([f"`{left['name']}`"] +
                            [str(hamming(left["bits"], right["bits"])) for right in delivered])
            out += [f"**Table {tag}c: pairwise Hamming distance among the delivered "
                    "schedules**, in bits of the 50-step string. A distance of 2 is one "
                    "cached step moved to a different position.", ""]
            out += table([""] + [f"`{n}`" for n in names], rows,
                         align="l" + "r" * len(names))
            out += [""]

        non_random = [r for r in delivered if r["algorithms"] != ["random"]]
        rnd = next((r for r in delivered if r["algorithms"] == ["random"]), None)
        if non_random and rnd:
            nr_bits = [r["bits"] for r in non_random]
            nr_dists = [hamming(nr_bits[i], nr_bits[j])
                        for i in range(len(nr_bits)) for j in range(i + 1, len(nr_bits))]
            rnd_dists = [hamming(rnd["bits"], b) for b in nr_bits]
            means = [r["arbitration_mean_psnr_db"] for r in non_random]
            spread = (f"{max(means) - min(means):.3f} dB of each other"
                      if len(means) > 1 else "one schedule")
            out += [f"The {len(non_random)} schedules from the designed searchers are "
                    f"{rng(nr_dists) if nr_dists else '0'} bits apart and within "
                    f"{spread} on arbitration. The random control sits "
                    f"{rng(rnd_dists)} bits away from them and "
                    f"{max(means) - rnd['arbitration_mean_psnr_db']:.3f} dB "
                    f"below the best of them.", ""]

    same_random = {}
    for model, k in rendered:
        rnd = next((r for r in delivered_for(data, model, k) if r["algorithms"] == ["random"]), None)
        if rnd:
            same_random.setdefault(k, {})[model] = rnd["bits"]
    shared = [k for k, per_model in sorted(same_random.items())
              if len(per_model) == 2 and len(set(per_model.values())) == 1]
    if shared:
        out += [f"The random control evaluates the same seeded pool in both models, and "
                f"at {join_and([f'K{k}' for k in shared])} both models select the same "
                "schedule out of that pool.", ""]
    out += section_arbitration_check(data, len(index) + 1)
    return out


def section_arbitration_check(data: dict, number: int) -> list[str]:
    """2.x: what the 50-caption arbitration bought, measured on PartiPrompts."""

    rows = []
    for model, _ in MODELS:
        for entry in data["results"]["models"].get(model, {}).get("arbitration_check", []):
            rows.append([
                f"{model_title(model)} K{entry['k']}",
                f"`{entry['algorithm']}`",
                f"`{entry['delivered']}`",
                steps_str(entry["delivered_bits"]),
                steps_str(entry["calibration_best_bits"]),
                fmt(entry["delivered_arbitration_mean_psnr_db"], 3),
                fmt(entry["calibration_best_arbitration_mean_psnr_db"], 3),
                thousands(entry["n_pairs"]),
                starred(entry, "psnr", 3),
                starred(entry, "ssim", 4),
                starred(entry, "lpips", 4),
                starred(entry, "image_reward", 3),
                starred(entry, "clip", 3),
            ])
    if not rows:
        return []
    n_captions = next(
        entry["arbitration_n_prompts"]
        for model, _ in MODELS
        for entry in data["results"]["models"].get(model, {}).get("arbitration_check", []))
    out = [f"### 2.{number} What arbitration changed", ""]
    out += [f"Arbitration re-scores each algorithm's top three candidates on the "
            f"{n_captions} held-out captions and delivers the winner, which is not always "
            "the candidate with the highest calibration mean. The rows below are every "
            "(setting, algorithm) where the two differ. The rejected calibration-best "
            "candidate was generated on PartiPrompts x 3 seed streams under the same "
            f"{PAYLOAD_NAMES['reuse']} payload, staged as `cal0_<algorithm>`, so the pair "
            "differs only in which steps are computed at full cost. These candidates are "
            "not delivered schedules and carry no rows in sections 3 and 4.", ""]
    out += [f"**Table 2.{number}: the delivered schedule against the calibration-best "
            "candidate it displaced.** The two arbitration columns are the mean PSNR over "
            f"the {n_captions} captions that decided the choice. The metric columns are "
            "paired per-image differences on PartiPrompts, delivered minus "
            "calibration-best, paired on seed and prompt index, with a 95% bootstrap "
            "interval over the pairs (2,000 resamples, fixed seed). PSNR, SSIM, "
            "ImageReward and CLIP read the same way (positive favours the delivered "
            "schedule); LPIPS is a distance, so negative favours it. An asterisk marks an "
            "interval that excludes zero.", ""]
    out += table(["Setting", "Algorithm", "Delivered", "Delivered full steps",
                  "Calibration-best full steps", "Arbitration mean, delivered (dB)",
                  "Arbitration mean, calibration-best (dB)", "Pairs", "Delta PSNR (dB)",
                  "Delta SSIM", "Delta LPIPS", "Delta ImageReward", "Delta CLIP"],
                 rows, align="lllllrrrrrrrr")
    out += [""]

    deltas = []
    decided = positive = 0
    for model, _ in MODELS:
        for entry in data["results"]["models"].get(model, {}).get("arbitration_check", []):
            deltas.append((entry["delta_psnr_mean"],
                           f"{model_title(model)} K{entry['k']} `{entry['algorithm']}`"))
            lo, hi = entry["ci95_psnr"]
            if lo > 0 or hi < 0:
                decided += 1
                positive += lo > 0
    mean_delta = sum(d for d, _ in deltas) / len(deltas)
    best, best_where = max(deltas)
    worst, worst_where = min(deltas)
    out += [f"Across the {len(deltas)} rows the delivered schedule sits "
            f"{signed(mean_delta, 3)} dB from the calibration-best candidate on "
            f"PartiPrompts, with a largest gain of {signed(best, 3)} dB at {best_where} "
            f"and a most negative value of {signed(worst, 3)} dB at {worst_where}. The "
            f"PSNR interval excludes zero in {decided} of the {len(deltas)} rows, "
            f"{positive} of them in favour of the delivered schedule.", ""]
    return out


# ---------------------------------------------------------------- section 3


def section_structure(data: dict) -> list[str]:
    cfg = data["config"]
    out = ["## 3. Q2: what do the searched schedules look like?", ""]
    out += ["Each setting is shown against the two baseline schedules of the same "
            "(model, budget): `meancache` and `budcache`, read from "
            "`resources/sp_cross_schedules/<model>_k<K>_<name>.txt`. The structure "
            f"columns count the free full steps (the full steps outside the forced set) "
            f"below step {EARLY_STEP}, the first cached step, and the longest run of "
            "consecutive cached steps.", ""]

    index = setting_index(data)
    for model, k in settings_present(data):
        forced = tuple(cfg["spaces"][model]["forced_full_steps"])
        n_free = cfg["spaces"][model]["free_full_count"][str(k)]
        delivered = delivered_for(data, model, k)
        incs = [(inc, data["incumbents"].get((model, k, inc))) for inc in INCUMBENTS]
        incs = [(inc, bits) for inc, bits in incs if bits]

        tag = f"3.{index[(model, k)]}"
        out += [f"### {tag} {model_title(model)} K{k}", ""]
        rows = []
        entries = [(row["name"], row["bits"], "searched") for row in delivered]
        entries += [(inc, bits, "baseline") for inc, bits in incs]
        for name, bits, kind in entries:
            free = free_steps(bits, forced)
            rows.append([
                f"`{name}`", kind, steps_str(bits),
                str(first_cached(bits)),
                f"{sum(1 for s in free if s < EARLY_STEP)}/{len(free)}",
                str(max(free)),
                str(longest_cached_run(bits)),
            ])
        out += [f"**Table {tag}a: full-step lists side by side.** Each schedule has "
                f"{len(forced)} forced full steps ({', '.join(str(s) for s in forced)}) and "
                f"{n_free} free ones.", ""]
        out += table(["Schedule", "Kind", "Full steps", "First cached step",
                      f"Free full steps below {EARLY_STEP}", "Last free full step",
                      "Longest cached run"], rows, align="lllrrrr")
        out += [""]

        if incs:
            rows = []
            for row in delivered:
                rows.append([f"`{row['name']}`"] +
                            [str(hamming(row["bits"], bits)) for _, bits in incs])
            out += [f"**Table {tag}b: Hamming distance from each delivered schedule "
                    "to the baseline schedules**, in bits of the 50-step string.", ""]
            out += table(["Schedule"] + [f"`{inc}`" for inc, _ in incs], rows,
                         align="l" + "r" * len(incs))
            out += [""]

        searched = [(row["name"], row["bits"]) for row in delivered
                    if row["algorithms"] != ["random"]]
        if searched:
            firsts = [first_cached(b) for _, b in searched]
            earlies = [sum(1 for s in free_steps(b, forced) if s < EARLY_STEP) for _, b in searched]
            lasts = [max(free_steps(b, forced)) for _, b in searched]
            runs = [longest_cached_run(b) for _, b in searched]
            sentence = (f"The {len(searched)} schedules from the designed searchers start "
                        f"caching at step {rng(firsts)}, put {rng(earlies)} of their "
                        f"{n_free} free full steps below step {EARLY_STEP}, place their "
                        f"last free full step at {rng(lasts)}, and leave a longest cached "
                        f"run of {rng(runs)} steps.")
            rnd = next((r for r in delivered if r["algorithms"] == ["random"]), None)
            if rnd:
                rfree = free_steps(rnd["bits"], forced)
                sentence += (f" The random control starts caching at step "
                             f"{first_cached(rnd['bits'])}, has "
                             f"{sum(1 for s in rfree if s < EARLY_STEP)} free full steps below "
                             f"step {EARLY_STEP}, and a longest cached run of "
                             f"{longest_cached_run(rnd['bits'])} steps.")
            out += [sentence, ""]

    placement = data["placement"]
    if placement:
        tag = f"3.{len(index) + 1}"
        out += [f"### {tag} FLUX K41 against the exhaustive table", ""]
        out += [f"The K41 space is the one setting with a complete oracle: "
                f"`{placement['table']}` scores all "
                f"{thousands(placement['space_size'])} schedules on the four "
                f"exhaustive-run calibration pairs. Every schedule the four searchers "
                f"selected or sent to arbitration is looked up in it. The table best is "
                f"{placement['table_best_mean_psnr_db']:.3f} dB.", ""]
        rows = []
        for row in placement["schedules"]:
            uses = ", ".join(f"`{u['algorithm']}`:{u['role']}" for u in row["uses"])
            rows.append([
                row["schedule"].replace(",", ", "), uses,
                fmt(row["uses"][0]["calibration_mean_psnr_db"], 3),
                fmt(row["table_mean_psnr_db"], 3),
                fmt(row["gap_to_table_best_db"], 3),
                thousands(row["position"]),
                f"{100 * row['top_share']:.3f}",
            ])
        out += [f"**Table {tag}: FLUX K41 schedules in the exhaustive table**, sorted by rank "
                "position. \"Calibration mean\" is the 8-pair value the search worked "
                "with; \"table mean\" is the 4-pair value of the exhaustive run, computed on a "
                "different prompt set. \"Rank\" counts "
                "schedules in the table with a table mean at least as high.", ""]
        out += table(["Full steps", "Proposed by", "Calibration mean (dB)",
                      "Table mean (dB)", "Gap to table best (dB)", "Rank",
                      "Top share (%)"], rows, align="llrrrrr")
        out += [""]
        delivered = delivered_for(data, "flux", 41)
        by_steps = {row["schedule"].replace(",", ", "): row
                    for row in placement["schedules"]}
        lines = []
        for row in delivered:
            place = by_steps.get(steps_str(row["bits"]))
            if place:
                lines.append(f"`{row['name']}` is {place['gap_to_table_best_db']:.3f} dB "
                             f"below the table best at rank {thousands(place['position'])} "
                             f"of {thousands(placement['space_size'])} "
                             f"({100 * place['top_share']:.3f}% of the space)")
        if lines:
            out += ["Of the schedules actually delivered: " + "; ".join(lines) + ".", ""]
        gaps = [r["gap_to_table_best_db"] for r in placement["schedules"]]
        positions = [r["position"] for r in placement["schedules"]]
        out += [f"Across all {len(placement['schedules'])} looked-up schedules the gap to "
                f"the table best runs {min(gaps):.3f}--{max(gaps):.3f} dB and the rank "
                f"position {thousands(min(positions))}--{thousands(max(positions))}.", ""]
    return out


# ---------------------------------------------------------------- section 4


def section_evaluation(data: dict) -> list[str]:
    out = ["## 4. Q3: how do the delivered schedules score on the full evaluation?", ""]
    results = data["results"]["models"]
    if not results:
        out += ["No P4 cells are staged yet.", ""]
        return out

    index = setting_index(data)
    staged = [(model, k) for model, k in settings_present(data)
              if any(c["k"] == k for c in results.get(model, {}).get("cells", []))]
    out += ["Settings with staged P4 cells: "
            f"{join_and([f'{model_title(m)} K{k}' for m, k in staged]) if staged else 'none'}. "
            "Each cell mean is over prompts x seed streams; n is the number of images "
            "behind it.", ""]
    out += ["The reference is the baseline methods' own schedules, `meancache` and "
            f"`budcache`, run under the same {PAYLOAD_NAMES['reuse']} payload as the "
            "searched schedules, on the same prompts and the same seed streams. Both "
            "sides of every difference in this section therefore differ only in which "
            "steps are computed at full cost.", ""]

    for model, k in staged:
        cells = [c for c in results[model]["cells"] if c["k"] == k]
        order = [name for name, _ in DATASETS]
        cells.sort(key=lambda c: (order.index(c["dataset"]), c["schedule"]))
        tag = f"4.{index[(model, k)]}"
        out += [f"### {tag} {model_title(model)} K{k}", ""]
        rows = []
        for c in cells:
            by_seed = sorted(c["psnr_by_seed"].values())
            rows.append([
                DATASET_NAME[c["dataset"]], f"`{c['schedule']}`",
                "baseline" if c["schedule"] in INCUMBENTS else "searched",
                thousands(c["n"]),
                fmt(c.get("psnr_mean"), 3),
                f"{by_seed[0]:.3f}--{by_seed[-1]:.3f}" if len(by_seed) > 1 else fmt(by_seed[0], 3),
                str(len(by_seed)),
                fmt(c.get("ssim_mean"), 4),
                fmt(c.get("lpips_mean"), 4),
                fmt(c.get("image_reward_mean"), 3),
                fmt(c.get("clip_mean"), 3),
            ])
        out += [f"**Table {tag}a: cell means per dataset.** \"PSNR by seed\" is the "
                "range of the per-seed-stream means; \"seeds\" is how many streams are "
                "staged for that cell. Baseline rows are the baseline methods' own schedules "
                f"run under the same {PAYLOAD_NAMES['reuse']} payload.", ""]
        out += table(["Dataset", "Schedule", "Kind", "n", "PSNR (dB)", "PSNR by seed",
                      "Seeds", "SSIM", "LPIPS", "ImageReward", "CLIP"],
                     rows, align="lllrrrrrrrr")
        out += [""]

        incumbent_rows = [r for r in results[model].get("incumbent_parti", []) if r["k"] == k]
        if incumbent_rows:
            rows = []
            for r in sorted(incumbent_rows, key=lambda r: r["schedule"]):
                rows.append([
                    f"`{r['schedule']}`",
                    PAYLOAD_NAMES.get(r["payload"], r["payload"]),
                    thousands(r["n"]), fmt(r.get("psnr_mean"), 3), fmt(r.get("ssim_mean"), 4),
                    fmt(r.get("lpips_mean"), 4), fmt(r.get("image_reward_mean"), 3),
                    fmt(r.get("clip_mean"), 3),
                ])
            out += [f"**Table {tag}b: the baseline schedules on the same PartiPrompts rows**, "
                    "from `resources/spx/perprompt_spx_" + model + ".tsv.gz`, under the "
                    f"{PAYLOAD_NAMES['reuse']} payload.", ""]
            out += table(["Schedule", "Payload", "n", "PSNR (dB)", "SSIM", "LPIPS",
                          "ImageReward", "CLIP"], rows, align="llrrrrrr")
            out += [""]

        paired = [p for p in results[model].get("paired", []) if p["k"] == k]
        if paired:
            order = [name for name, _ in DATASETS]
            paired.sort(key=lambda p: (order.index(p["dataset"]), p["schedule"],
                                       p["incumbent"]))
            rows = []
            for p in paired:
                rows.append([
                    DATASET_NAME[p["dataset"]],
                    f"`{p['schedule']}`",
                    f"`{p['incumbent']}`",
                    SOURCE_NAMES.get(p["reference_source"], p["reference_source"]),
                    thousands(p["n_pairs"]),
                    signed(p["delta_psnr_mean"], 3),
                    f"[{signed(p['ci95'][0], 3)}, {signed(p['ci95'][1], 3)}]",
                    f"{p['share_positive']:.3f}",
                ])
            out += [f"**Table {tag}c: paired per-image PSNR difference**, searched "
                    "schedule minus baseline schedule, both under the "
                    f"{PAYLOAD_NAMES['reuse']} payload, paired on dataset, seed and "
                    "prompt index. The interval is a 95% bootstrap interval over the "
                    "paired differences (2,000 resamples, fixed seed). The source column "
                    "says which staged table the baseline rows came from.", ""]
            out += table(["Dataset", "Schedule", "Reference schedule", "Reference rows",
                          "Pairs", "Delta PSNR (dB)", "95% interval",
                          "Share of pairs positive"], rows, align="llllrrrr")
            out += [""]

            # The same paired difference on the other four metrics, MeanCache's
            # and BudCache's schedules only; a value whose interval excludes zero
            # is marked with an asterisk.
            rows_m = []
            for p in paired:
                if p["incumbent"] not in ("meancache", "budcache"):
                    continue
                rows_m.append([
                    DATASET_NAME[p["dataset"]], f"`{p['schedule']}`", f"`{p['incumbent']}`",
                    thousands(p["n_pairs"]),
                    starred(p, "ssim", 4), starred(p, "lpips", 4),
                    starred(p, "image_reward", 3), starred(p, "clip", 3),
                ])
            if rows_m:
                out += [f"**Table {tag}d: the same paired difference on the other metrics**, "
                        "searched schedule minus baseline schedule under the "
                        f"{PAYLOAD_NAMES['reuse']} payload, for `meancache` and `budcache`. "
                        "SSIM, ImageReward and CLIP read as PSNR does (positive favours the "
                        "searched schedule); LPIPS is a distance, so negative favours it. An "
                        "asterisk marks a 95% bootstrap interval that excludes zero.", ""]
                out += table(["Dataset", "Schedule", "Reference schedule", "Pairs",
                              "Delta SSIM", "Delta LPIPS", "Delta ImageReward", "Delta CLIP"],
                             rows_m, align="lllrrrrr")
                out += [""]

    covered: dict[str, set[str]] = {}
    missing: dict[str, list[str]] = {}
    for model, k in staged:
        label = f"{model_title(model)} K{k}"
        paired = [p for p in results[model].get("paired", []) if p["k"] == k]
        for p in paired:
            covered.setdefault(p["reference_source"], set()).add(p["dataset"])
        with_reference = {p["dataset"] for p in paired}
        for c in results[model]["cells"]:
            if c["k"] == k and c["dataset"] not in with_reference:
                missing.setdefault(c["dataset"], [])
                if label not in missing[c["dataset"]]:
                    missing[c["dataset"]].append(label)
    if covered:
        parts = []
        for source in sorted(covered):
            names = [DATASET_NAME[d] for d, _ in DATASETS if d in covered[source]]
            parts.append(f"{SOURCE_NAMES.get(source, source)} on {join_and(names)}")
        out += ["The baseline rows behind those differences come from "
                + join_and(parts) + ".", ""]
    if missing:
        names = [DATASET_NAME[d] for d, _ in DATASETS if d in missing]
        lead = (f"Staged searched cells whose baseline rows under the "
                f"{PAYLOAD_NAMES['reuse']} payload are not staged: ")
        if all(len(v) == len(staged) for v in missing.values()):
            body = f"{join_and(names)}, in every setting"
        else:
            parts = []
            for dataset, _ in DATASETS:
                if dataset not in missing:
                    continue
                where = ("every setting" if len(missing[dataset]) == len(staged)
                         else join_and(missing[dataset]))
                parts.append(f"{DATASET_NAME[dataset]} in {where}")
            body = "; ".join(parts)
        out += [lead + body + ". Those cells carry means in the tables above.", ""]
    return out


# ---------------------------------------------------------------- section 5


def objective_block(data: dict, model: str) -> dict:
    """The `objectives` block `analysis/schedule_search_results.py` wrote."""

    return data["results"]["models"].get(model, {}).get("objectives", {}) or {}


def objective_settings(data: dict, model: str, k: int) -> list[dict]:
    return [s for s in objective_block(data, model).get("settings", []) if s["k"] == k]


def objective_label(objective: str) -> str:
    return OBJECTIVE_NAMES.get(objective, objective)


def objective_settings_present(data: dict) -> list[tuple[str, int]]:
    """Settings that have a delivery under an objective other than PSNR."""

    out = []
    for model, _ in MODELS:
        for k in BUDGETS:
            rows = [s for s in objective_settings(data, model, k)
                    if s["objective"] != "psnr" and s["schedules"]]
            if rows:
                out.append((model, k))
    return out


def schedule_of(setting_rows: list[dict], name: str) -> dict | None:
    for row in setting_rows:
        if row["name"] == name:
            return row
    return None


def objective_metric_row(values: dict) -> list[str]:
    return [fmt(values.get(metric), digits) for metric, _, digits in METRIC_COLUMNS]


def sign_pattern(rows: list[dict]) -> str:
    """How the paired differences of one objective fall, metric by metric."""

    parts = []
    for metric, name, _ in METRIC_COLUMNS:
        values = [r[f"delta_{metric}_mean"] for r in rows if f"delta_{metric}_mean" in r]
        if not values:
            continue
        negative = sum(1 for v in values if v < 0)
        positive = len(values) - negative
        word = "negative" if negative >= positive else "positive"
        count = negative if negative >= positive else positive
        parts.append(f"{word} on {name.split(' (')[0]} in {count} of {len(values)}")
    return join_and(parts)


def section_objectives(data: dict) -> list[str]:
    out = ["## 5. The search objective", ""]
    out += ["Sections 2 to 4 read one search objective, the mean PSNR of the 8 "
            "calibration pairs. Section 4b of the plan adds two more objectives over "
            "the same pairs and the same five metrics every evaluation records; table "
            "5.0b lists which searchers ran for each. Each objective runs its own "
            "searches, its own "
            "arbitration on the same 50 held-out captions, and its own delivery; a "
            "delivered schedule whose bitstring another objective already delivered is "
            "evaluated once and carries the cells of that name.", ""]
    rows = [[f"`{objective}`", objective_label(objective), OBJECTIVE_RULES[objective]]
            for objective in OBJECTIVES]
    out += ["**Table 5.0a: the three objectives.** All three are maximised, so a higher "
            "value is a better schedule under that objective.", ""]
    out += table(["Objective", "Name", "What one evaluation is reduced to"], rows)
    out += [""]

    rows = []
    for model, _ in MODELS:
        for k in BUDGETS:
            for objective in OBJECTIVES:
                files = data["objective_files"].get((model, k, objective), {})
                setting = next((s for s in objective_settings(data, model, k)
                                if s["objective"] == objective), None)
                schedules = setting["schedules"] if setting else []
                datasets = sorted({c["dataset"] for row in schedules for c in row["cells"]},
                                  key=lambda d: [n for n, _ in DATASETS].index(d))
                if not files.get("arbitration") and not schedules:
                    continue
                rows.append([
                    model_title(model), f"K{k}", f"`{objective}`",
                    ", ".join(f"`{a}`" for a in files.get("algorithms", [])) or "--",
                    "yes" if files.get("arbitration") else "no",
                    str(len(schedules)) if schedules else "--",
                    ", ".join(DATASET_NAME[d] for d in datasets) if datasets else "--",
                ])
    out += ["**Table 5.0b: which objective variants are staged.** The searches column "
            "lists the search summaries on disk for that objective; the delivered count "
            "and the datasets come from the `objectives` block of `results.json`.", ""]
    out += table(["Model", "Budget", "Objective", "Searches", "Arbitration",
                  "Delivered schedules", "P4 datasets staged"], rows, align="llllcrl")
    out += [""]

    mismatched = [f"{model_title(model)}: {text}" for model, _ in MODELS
                  for text in objective_block(data, model).get("warnings", [])]
    if mismatched:
        out += ["Files whose recorded objective does not match their name, left out of "
                "the tables below: " + join_and(mismatched) + ".", ""]

    stale = [f"{model_title(m)} K{k} `{o}`"
             for (m, k, o), files in sorted(data["objective_files"].items())
             if files.get("arbitration") and o != "psnr"
             and not any(s["objective"] == o for s in objective_settings(data, m, k))]
    if stale:
        out += [f"Arbitration records present on disk with no entry in `results.json`: "
                f"{join_and(stale)}. Re-run `analysis/schedule_search_results.py` to "
                "bring them into the tables below.", ""]

    rendered = objective_settings_present(data)
    if not rendered:
        out += ["No setting has a delivery under an objective other than PSNR yet.", ""]
        return out
    out += [f"Settings with a delivery under another objective: "
            f"{join_and([f'{model_title(m)} K{k}' for m, k in rendered])}.", ""]

    paired_all = {model: objective_block(data, model).get("paired", [])
                  for model, _ in MODELS}
    order = [name for name, _ in DATASETS]
    for i, (model, k) in enumerate(rendered):
        tag = f"5.{i + 1}"
        settings = objective_settings(data, model, k)
        settings.sort(key=lambda s: OBJECTIVES.index(s["objective"]))
        psnr_rows = next((s["schedules"] for s in settings if s["objective"] == "psnr"), [])
        early = objective_block(data, model).get("early_step") or EARLY_STEP
        out += [f"### {tag} {model_title(model)} K{k}", ""]

        rows = []
        for setting in settings:
            for row in setting["schedules"]:
                structure = row["structure"]
                rows.append([
                    f"`{setting['objective']}`", f"`{row['name']}`",
                    ", ".join(f"`{a}`" for a in row["algorithms"]),
                    steps_str(row["bits"]),
                    str(structure["first_cached_step"]),
                    f"{structure['free_full_steps_below_early']}/"
                    f"{structure['free_full_steps']}",
                    str(structure["last_free_full_step"]),
                    str(structure["longest_cached_run"]),
                ])
        out += [f"**Table {tag}a: the schedules each objective delivered.** The "
                "structure columns are the ones section 3 uses: the first cached step, "
                f"the free full steps below step {early}, the last free full step, and "
                "the longest run of consecutive cached steps.", ""]
        out += table(["Objective", "Name", "Algorithms", "Full steps",
                      "First cached step", f"Free full steps below {early}",
                      "Last free full step", "Longest cached run"],
                     rows, align="lllrrrrr")
        out += [""]

        repeats = [(setting["objective"], row) for setting in settings
                   for row in setting["schedules"] if row.get("identical_to")]
        if repeats:
            out += ["Schedules that repeat a bitstring already delivered: "
                    + join_and([f"`{row['name']}` is `{row['identical_to']}`"
                                for _, row in repeats])
                    + ". They were evaluated once, under the earlier name.", ""]

        rows = []
        for setting in settings:
            for row in setting["schedules"]:
                for label, values, value_key in (
                        ("8 calibration pairs", (row.get("calibration") or {}).get("metrics", {}),
                         (row.get("calibration") or {}).get("mean_objective")),
                        (f"{setting['n_arbitration_prompts']} arbitration captions",
                         row["arbitration"].get("metrics", {}),
                         row["arbitration"].get("mean_objective"))):
                    rows.append([
                        f"`{setting['objective']}`", f"`{row['name']}`", label,
                        fmt(value_key, 4),
                    ] + objective_metric_row(values or {}))
        out += [f"**Table {tag}b: the delivered schedules on the sets the search and the "
                "arbitration read.** The objective column is in that objective's own "
                "units (dB for `psnr`, negative LPIPS for `lpips`, standardized units "
                "for `psnr_lpips_z`); the metric columns are the means of the five "
                "recorded metrics over that set. A dash is a value the run did not "
                "record.", ""]
        out += table(["Objective", "Schedule", "Set", "Objective value"]
                     + [name for _, name, _ in METRIC_COLUMNS],
                     rows, align="lllr" + "r" * len(METRIC_COLUMNS))
        out += [""]

        rows = []
        for setting in settings:
            for row in setting["schedules"]:
                for cell in sorted(row["cells"], key=lambda c: order.index(c["dataset"])):
                    rows.append([
                        DATASET_NAME[cell["dataset"]], f"`{setting['objective']}`",
                        f"`{row['name']}`", thousands(cell["n"]),
                    ] + [fmt(cell.get(f"{metric}_mean"), digits)
                         for metric, _, digits in METRIC_COLUMNS])
        if rows:
            rows.sort(key=lambda r: order.index(
                next(d for d, name in DATASETS if name == r[0])))
            out += [f"**Table {tag}c: P4 cell means of the delivered schedules.** Each "
                    "mean is over prompts times seed streams under the "
                    f"{PAYLOAD_NAMES['reuse']} payload; n is the number of images behind "
                    "it. A schedule that repeats an earlier bitstring appears under the "
                    "name its cells are staged with.", ""]
            out += table(["Dataset", "Objective", "Schedule", "n"]
                         + [name for _, name, _ in METRIC_COLUMNS],
                         rows, align="lllr" + "r" * len(METRIC_COLUMNS))
            out += [""]
        else:
            out += ["No P4 cells are staged for these schedules yet.", ""]

        paired = [p for p in paired_all.get(model, [])
                  if p["k"] == k and not p["identical_bits"] and p.get("dataset")]
        if paired:
            paired.sort(key=lambda p: (order.index(p["dataset"]),
                                       OBJECTIVES.index(p["objective"]), p["schedule"]))
            rows = []
            for p in paired:
                rows.append([
                    DATASET_NAME[p["dataset"]], f"`{p['objective']}`",
                    f"`{p['schedule']}`", f"`{p['reference']}`",
                    thousands(p["n_pairs"]),
                ] + [starred(p, metric, digits) for metric, _, digits in METRIC_COLUMNS])
            out += [f"**Table {tag}d: paired per-image difference against the "
                    "PSNR-objective schedule of the same algorithm family**, objective "
                    "schedule minus PSNR-objective schedule, both under the "
                    f"{PAYLOAD_NAMES['reuse']} payload, paired on seed and prompt index. "
                    "PSNR, SSIM, ImageReward and CLIP read the same way, a positive "
                    "value favouring the objective schedule; LPIPS is a distance, so "
                    "negative favours it. An asterisk marks a 95% bootstrap interval "
                    "that excludes zero (2,000 resamples, fixed seed).", ""]
            out += table(["Dataset", "Objective", "Schedule", "Reference schedule",
                          "Pairs"] + [f"Delta {name}" for _, name, _ in METRIC_COLUMNS],
                         rows, align="llllr" + "r" * len(METRIC_COLUMNS))
            out += [""]

        for setting in settings:
            if setting["objective"] == "psnr" or not setting["schedules"]:
                continue
            label = objective_label(setting["objective"])
            moves = []
            for row in setting["schedules"]:
                links = [p for p in paired_all.get(model, [])
                         if p["k"] == k and p["schedule"] == row["name"]]
                for reference in sorted({p["reference"] for p in links}):
                    other = schedule_of(psnr_rows, reference)
                    if other is None:
                        continue
                    family = join_and([f"`{a}`" for a in sorted(
                        {a for p in links if p["reference"] == reference
                         for a in p["shared_algorithms"]})])
                    here = row["structure"]["last_free_full_step"]
                    there = other["structure"]["last_free_full_step"]
                    if here == there:
                        moves.append((f"the {family} family keeps the last free full "
                                      f"step at {here}",
                                      f"the {family} family keeps it at {here}"))
                    else:
                        moves.append((f"the {family} family moves the last free full "
                                      f"step from {there} to {here}",
                                      f"the {family} family moves it from {there} to "
                                      f"{here}"))
            rows_o = [p for p in paired
                      if p["objective"] == setting["objective"]]
            sentence = f"Under the {label} objective "
            moves.sort()
            clauses = [long for long, _ in moves[:1]] + [short for _, short in moves[1:]]
            sentence += (join_and(clauses) if clauses else
                         "no delivered schedule has a PSNR-objective counterpart in the "
                         "same algorithm family")
            if rows_o:
                decided = sum(1 for p in rows_o for metric, _, _ in METRIC_COLUMNS
                              if f"ci95_{metric}" in p
                              and (p[f"ci95_{metric}"][0] > 0 or p[f"ci95_{metric}"][1] < 0))
                cells_n = sum(1 for p in rows_o for metric, _, _ in METRIC_COLUMNS
                              if f"ci95_{metric}" in p)
                sentence += (f". Across the {len(rows_o)} rows of table {tag}d the "
                             f"difference against the PSNR-objective schedule is "
                             f"{sign_pattern(rows_o)}, and {decided} of the {cells_n} "
                             "metric cells have an interval excluding zero; LPIPS is a "
                             "distance, so a positive difference there is a worse match")
            out += [sentence + ".", ""]
    return out


# ---------------------------------------------------------------- section 6


def payload_block(data: dict, model: str) -> dict:
    """The `payloads` block `analysis/schedule_search_results.py` wrote."""

    return data["results"]["models"].get(model, {}).get("payloads", {}) or {}


def payload_label(payload: str) -> str:
    return PAYLOAD_NAMES.get(payload, payload)


def signed_range(values: list[float], digits: int = 3) -> str:
    """`+0.012` when the range is a point, `-0.004 to +0.012` otherwise."""

    lo, hi = min(values), max(values)
    if f"{lo:.{digits}f}" == f"{hi:.{digits}f}":
        return signed(lo, digits)
    return f"{signed(lo, digits)} to {signed(hi, digits)}"


def section_payloads(data: dict) -> list[str]:
    blocks = [(model, payload_block(data, model)) for model, _ in MODELS]
    blocks = [(model, block) for model, block in blocks if block.get("cells")]
    out = ["## 6. The delivered schedule under other payloads", ""]
    if not blocks:
        out += ["No payload cells are staged yet.", ""]
        return out

    names = join_and([f"`{payload}` ({payload_label(payload)})" for payload in PAYLOADS
                      if payload != "reuse"])
    schedules = join_and([f"{model_title(model)} K{row['k']} `{row['schedule']}`"
                          for model, block in blocks for row in block["delivered"]])
    out += ["Sections 2 to 5 hold the payload at "
            f"{PAYLOAD_NAMES['reuse']} and vary the schedule. This section holds the "
            "schedule and varies the payload: the arbitration-best schedule of each "
            f"setting ({schedules}) was run again under {names}, on the same four "
            "datasets and the same three seed streams, and compared image by image "
            "against the matrix's own full-compute references. The reuse rows are the "
            "P4 cells of section 4.", ""]

    rows = []
    for model, block in blocks:
        for row in block["cells_dataset_mean"]:
            rows.append([
                model_title(model), f"K{row['k']}", f"`{row['payload']}`",
                payload_label(row["payload"]), f"`{row['schedule']}`",
                thousands(row["n"]),
            ] + [fmt(row.get(f"{metric}_mean"), digits)
                 for metric, _, digits in METRIC_COLUMNS])
    out += ["**Table 6.0: the delivered schedule's cell means, averaged over the four "
            "datasets with equal weight.** Each dataset contributes its own cell mean "
            "once, so DiffusionDB's 10,000 prompts do not outweigh DrawBench's 200. "
            "\"Images\" is the total behind the row.", ""]
    out += table(["Model", "Budget", "Payload", "What a cached step uses", "Schedule",
                  "Images"] + [name for _, name, _ in METRIC_COLUMNS],
                 rows, align="llllr" + "r" * (1 + len(METRIC_COLUMNS)))
    out += [""]

    order = [name for name, _ in DATASETS]
    rows = []
    for model, block in blocks:
        per_dataset = block["vs_reuse"]
        for row in block["vs_reuse_dataset_mean"]:
            here = sorted([p for p in per_dataset
                           if p["k"] == row["k"] and p["payload"] == row["payload"]],
                          key=lambda p: order.index(p["dataset"]))
            for p in here:
                rows.append([
                    model_title(model), f"K{p['k']}", f"`{p['payload']}`",
                    DATASET_NAME[p["dataset"]], thousands(p["n_pairs"]),
                    starred(p, "psnr", 3),
                    f"[{signed(p['ci95_psnr'][0], 3)}, {signed(p['ci95_psnr'][1], 3)}]",
                    f"{p['share_positive_psnr']:.3f}",
                ])
            rows.append([
                model_title(model), f"K{row['k']}", f"`{row['payload']}`",
                "four-dataset mean", thousands(row["n_pairs"]),
                starred(row, "psnr", 3),
                f"[{signed(row['ci95_psnr'][0], 3)}, {signed(row['ci95_psnr'][1], 3)}]",
                f"{row['share_positive_psnr']:.3f}",
            ])
    out += ["**Table 6.1a: paired per-image PSNR difference against the same schedule "
            f"under {PAYLOAD_NAMES['reuse']}.** Both sides are the same schedule on the "
            "same prompt and the same seed, so only the payload differs. The interval is "
            "a 95% bootstrap interval over the paired differences (2,000 resamples, "
            "fixed seed); the four-dataset row resamples each dataset on its own and "
            "averages the four means. An asterisk marks an interval that excludes zero.", ""]
    out += table(["Model", "Budget", "Payload", "Dataset", "Pairs", "Delta PSNR (dB)",
                  "95% interval", "Share of pairs positive"], rows, align="llllrrrr")
    out += [""]

    rows = []
    for model, block in blocks:
        for row in block["vs_reuse_dataset_mean"]:
            here = [p for p in block["vs_reuse"]
                    if p["k"] == row["k"] and p["payload"] == row["payload"]]
            cells = []
            for metric, _, digits in METRIC_COLUMNS[1:]:
                values = [p[f"delta_{metric}_mean"] for p in here
                          if f"delta_{metric}_mean" in p]
                cells.append(signed_range(values, digits) if values else "n/a")
            rows.append([model_title(model), f"K{row['k']}", f"`{row['payload']}`",
                         str(row["n_groups"])] + cells)
    out += ["**Table 6.1b: the same difference on the other four metrics**, as the range "
            "over the four datasets, payload minus reuse. SSIM, ImageReward and CLIP read "
            "as PSNR does, a positive value favouring the payload of the row; LPIPS is a "
            "distance, so negative favours it.", ""]
    out += table(["Model", "Budget", "Payload", "Datasets"]
                 + [f"Delta {name}" for _, name, _ in METRIC_COLUMNS[1:]],
                 rows, align="lllr" + "r" * (len(METRIC_COLUMNS) - 1))
    out += [""]

    rows = []
    for model, block in blocks:
        for row in block["vs_incumbent_parti"]:
            rows.append([
                model_title(model), f"K{row['k']}", f"`{row['payload']}`",
                f"`{row['schedule']}`", f"`{row['incumbent']}`",
                thousands(row["n_pairs"]), starred(row, "psnr", 3),
                f"[{signed(row['ci95_psnr'][0], 3)}, {signed(row['ci95_psnr'][1], 3)}]",
                starred(row, "ssim", 4), starred(row, "lpips", 4),
                starred(row, "image_reward", 3), starred(row, "clip", 3),
            ])
    if rows:
        present = {}
        for _, block in blocks:
            for inc, payloads in block.get("incumbent_payloads", {}).items():
                present.setdefault(inc, set()).update(payloads)
        listed = join_and([f"`{inc}` under "
                           + join_and([f"`{p}`" for p in PAYLOADS if p in present[inc]])
                           for inc in sorted(present)])
        out += ["**Table 6.2: the same payload, two schedules, on PartiPrompts.** The "
                "delivered schedule minus the baseline method's schedule, both run under "
                "the payload of the row, paired on seed and prompt index. The baseline "
                "rows come from `resources/spx/perprompt_spx_<model>.tsv.gz`, which "
                f"carries {listed}, so the rows below are every pair where both sides "
                "exist. An asterisk marks a 95% bootstrap interval that excludes zero.", ""]
        out += table(["Model", "Budget", "Payload", "Schedule", "Reference schedule",
                      "Pairs", "Delta PSNR (dB)", "95% interval", "Delta SSIM",
                      "Delta LPIPS", "Delta ImageReward", "Delta CLIP"],
                     rows, align="lllll" + "r" * 7)
        out += [""]

    for model, block in blocks:
        best = []
        for row in block["delivered"]:
            k = row["k"]
            cells = [c for c in block["cells_dataset_mean"] if c["k"] == k]
            if not cells:
                continue
            top = max(cells, key=lambda c: c["psnr_mean"])
            gain = next((p["delta_psnr_mean"] for p in block["vs_reuse_dataset_mean"]
                         if p["k"] == k and p["payload"] == top["payload"]), 0.0)
            best.append((k, top, gain))
        if not best:
            continue
        payloads = {top["payload"] for _, top, _ in best}
        sentences = []
        if len(payloads) == 1:
            payload = payloads.pop()
            sentences.append(
                f"On {model_title(model)} the delivered schedule scores highest under "
                f"`{payload}`, the {payload_label(payload)}, at all three budgets.")
            for k, top, gain in best:
                sentences.append(
                    f"At K{k} its four-dataset mean PSNR is {top['psnr_mean']:.3f} dB, "
                    f"which is above {PAYLOAD_NAMES['reuse']} by {abs(gain):.3f} dB."
                    if gain else
                    f"At K{k} its four-dataset mean PSNR is {top['psnr_mean']:.3f} dB.")
        else:
            sentences.append(
                f"On {model_title(model)} the payload with the highest four-dataset "
                "mean PSNR is not the same at every budget.")
            for k, top, gain in best:
                head = (f"At K{k} it is `{top['payload']}`, the "
                        f"{payload_label(top['payload'])}, at {top['psnr_mean']:.3f} dB")
                sentences.append(
                    f"{head}, which is above {PAYLOAD_NAMES['reuse']} by {abs(gain):.3f} dB."
                    if gain else f"{head}.")
        out += [" ".join(sentences), ""]

    problems = []
    for model, block in blocks:
        if block.get("missing_cells"):
            problems.append(f"{model_title(model)}: {len(block['missing_cells'])} of the "
                            f"(budget, payload, dataset, seed) cells are not staged")
        if block.get("duplicated_images"):
            problems.append(f"{model_title(model)}: "
                            f"{len(block['duplicated_images'])} cells appear in more than "
                            "one cluster's table")
    if problems:
        out += ["Gaps in the staged cells: " + join_and(problems) + ".", ""]
    return out


# ---------------------------------------------------------------- section 7


DEVIATIONS = [
    ("P0 fidelity anchor, 2026-09-01",
     "The first anchor run re-encoded the prompts in place and its Parti index 15 pair "
     "differed from the truth table by 0.1--0.7 dB while the other three matched bit for "
     "bit. Re-running it against the frozen conditioning artifact of the exhaustive run "
     "(`exhaustive_k41/conditioning/flux_dev_3de623fc_discovery4.pt`) made all four pairs "
     "bit-identical. The search's own references and candidates are generated in one "
     "process on one encoding path and do not use that artifact."),
    ("P1 temperature formula, 2026-09-01",
     "The plan's rule of putting t_min below the calibration standard error was not usable: "
     "the stratified 8-slot calibration set has a standard error of 0.54--0.69 dB, so "
     "t_min = SE/10 would sit within a factor of two of t_max and the 200-step cooling "
     "would be near-isothermal. The temperatures were instead set to t_max = 10 x the "
     "median one-swap change and t_min = that median / 100, the ratio the table benchmark "
     "used, backfilled by `analysis/backfill_schedule_search_temperatures.py`. The "
     "stopping rule was not changed; with that standard error the 2 x SE threshold "
     "rarely triggers and the cap takes over."),
    ("P2 intra-node fan-out, 2026-09-01",
     "The search is sequential, but the 8 calibration pairs of one evaluation are "
     "independent generations. The runner gained `--gpus N`: N worker processes each hold "
     "one model on one GPU and own the pairs with `index % N == rank`, and the parent "
     "collects the per-pair PSNR in the original order. Seeds, conditioning and decode "
     "path per pair are the same as on one GPU, and trace replay is unaffected."),
    ("P6 payload supplement, 2026-09-03",
     "The six arbitration-best schedules of `delivery_best.txt` were run again under the "
     "four forecast payloads on the four datasets and the three seed streams, 288 cells, "
     "through the same submitter with `--payload`. Section 6 reports these cells."),
]


def section_provenance(data: dict) -> list[str]:
    cfg = data["config"]
    out = ["## 7. Provenance", "", "### 7.1 Deviations recorded in the plan", ""]
    for title, text in DEVIATIONS:
        out += [f"- **{title}.** {text}"]
    out += [""]

    out += ["### 7.2 Artifacts", ""]
    rows = [
        ["`resources/schedule_search/config.v1.json`",
         "frozen calibration slots, arbitration captions, caps, chain lengths, "
         "temperatures, warm starts"],
        ["`resources/schedule_search/probes/<model>_k<K>.json`", "P1 probe readings"],
        ["`resources/schedule_search/search/<model>_k<K>_<algorithm>.json`",
         "P2 search summaries"],
        ["`resources/schedule_search/arbitration/<model>_k<K>.json`",
         "P3 arbitration and delivery"],
        ["`resources/schedule_search/delivery.txt`", "the P4 delivery list"],
        ["`resources/schedule_search/delivery_best.txt`",
         "the arbitration-best schedule of each setting, the schedule list section 6 "
         "runs under the other payloads"],
        ["`resources/schedule_search/search/<model>_k<K>_<algorithm>_<objective>.json`",
         "the objective variants' search summaries"],
        ["`resources/schedule_search/arbitration/<model>_k<K>_<objective>.json`",
         "the objective variants' arbitration and delivery"],
        ["`resources/schedule_search/delivery_<objective>.txt`",
         "one delivery list per objective, and `delivery_objectives.txt` for the union "
         "that was evaluated"],
        ["`resources/schedule_search/schedules/`", "one bitstring file per delivered schedule"],
        ["`resources/schedule_search/k41_table_placement.json`",
         "FLUX K41 lookup in the exhaustive table"],
        ["`resources/schedule_search/perprompt_search_<model>_<cluster>.tsv.gz`",
         "P4 per-image metrics, and the section 6 payload cells in the same tables"],
        ["`resources/schedule_search/results.json`",
         "cell means and paired differences, written by `analysis/schedule_search_results.py`"],
        ["`resources/sp_cross_schedules/<model>_k<K>_<name>.txt`", "the baseline methods' schedules"],
        ["`resources/spx/perprompt_spx_<model>.tsv.gz`",
         "the baseline methods' per-image PartiPrompts metrics, under all five payloads"],
    ]
    out += ["**Table 7.2a: staged artifacts.**", ""]
    out += table(["Path", "Contents"], rows)
    out += [""]

    runs = sorted(data["search"].items())
    revisions = sorted({run.get("model_revision") for _, run in runs if run.get("model_revision")})
    config_shas = sorted({run.get("config_sha256") for _, run in runs if run.get("config_sha256")})
    probe_shas = sorted({p.get("config_sha256") for p in data["probes"].values()
                         if p and p.get("config_sha256")})
    prompt_shas = sorted({run.get("prompt_text_sha256") for _, run in runs
                          if run.get("prompt_text_sha256")})
    rows = [
        ["Model weights revision recorded by the search runs", ", ".join(revisions) or "--"],
        ["Calibration prompt text sha256", ", ".join(s[:12] for s in prompt_shas) or "--"],
        ["Config sha256 recorded by the search and arbitration runs",
         ", ".join(s[:12] for s in config_shas) or "--"],
        ["Config sha256 recorded by the probe runs", ", ".join(s[:12] for s in probe_shas) or "--"],
        ["COCO captions sha256", cfg["sources"]["coco_captions"]["sha256"][:12]],
        ["COCO instances sha256", cfg["sources"]["coco_instances"]["sha256"][:12]],
        ["DiffusionDB calibration pool sha256", cfg["sources"]["diffusiondb_pool"]["sha256"][:12]],
    ]
    out += ["**Table 7.2b: identities the runs recorded.** Hashes are truncated to 12 "
            "hex characters; the full values are in the files listed above.", ""]
    out += table(["Item", "Value"], rows)
    out += ["",
            "The probes ran before the temperature backfill, so they record the config "
            "hash of the version whose `temperatures` field was still null; the search "
            "and arbitration runs record the hash after the backfill. The backfill "
            "touched only that field.", ""]

    total_wall = sum(run["wall_s"] for _, run in runs)
    rows = []
    for model, _ in MODELS:
        for k in BUDGETS:
            group = [run for (m, kk, _), run in runs if m == model and kk == k]
            if not group:
                continue
            rows.append([model_title(model), f"K{k}", str(len(group)),
                         f"{sum(r['wall_s'] for r in group) / 3600:.2f}",
                         f"{min(r['wall_s'] for r in group) / 3600:.2f}--"
                         f"{max(r['wall_s'] for r in group) / 3600:.2f}",
                         sorted({r["device"] for r in group})[0]])
    out += ["**Table 7.2c: search wall time.** One job per (model, budget, algorithm). "
            "Qwen K29, Qwen K37, and FLUX K29 resumed with four GPUs per job.", ""]
    out += table(["Model", "Budget", "Jobs", "Total wall (h)", "Per job (h)", "Device"],
                 rows, align="llrrrl")
    out += ["", f"Search wall time over the {len(runs)} staged jobs: "
                f"{total_wall / 3600:.1f} h.", ""]
    return out


# ---------------------------------------------------------------- appendix


def appendix(data: dict) -> list[str]:
    cfg = data["config"]
    out = ["## Appendix A: the arbitration captions", ""]
    rows = [[str(i + 1), str(p["caption_id"]), str(p["image_id"]), f"`{p['prompt']}`"]
            for i, p in enumerate(cfg["arbitration"]["prompts"])]
    out += table(["#", "Caption id", "Image id", "Prompt"], rows, align="rrrl")
    out += [""]

    out += ["## Appendix B: warm starts", ""]
    out += ["The hill climb and the annealer start from the baseline schedules of the "
            "same setting before falling back to random starts. Schedules whose full "
            "steps do not contain the forced set are not members of the space and were "
            "excluded.", ""]
    rows = []
    for model, title in MODELS:
        for k in BUDGETS:
            names = [w["name"] for w in cfg["warm_starts"][model][str(k)]]
            excluded = [w["name"] for w in cfg["warm_starts_excluded"]
                        if w["model"] == model and w["k"] == k]
            rows.append([title, f"K{k}", str(len(names)), ", ".join(f"`{n}`" for n in names),
                         ", ".join(f"`{n}`" for n in excluded) if excluded else "none"])
    out += table(["Model", "Budget", "Count", "Warm starts", "Excluded"], rows,
                 align="llrll")
    out += [""]

    anchor = cfg["anchor"]
    out += ["## Appendix C: the FLUX K41 fidelity anchor", ""]
    out += ["Three schedules re-scored on the exhaustive run's own four pairs before the "
            "searches started, to check that this harness reproduces the truth table "
            f"(`{anchor['file']}`).", ""]
    rows = [[f"`{s['name']}`", ", ".join(str(x) for x in s["full_steps"])]
            for s in anchor["schedules"]]
    out += table(["Name", "Full steps"], rows)
    out += [""]
    return out


# ---------------------------------------------------------------- document


def render(data: dict) -> str:
    rendered = settings_present(data)
    lines = ["# Searching for a fixed schedule on the real models -- results", ""]
    lines += ["Standalone experiment report, self-contained: sections 1 and 2 define "
              "everything the later tables use. Plan: `docs/schedule_search_plan_zh.md`. "
              "The searchers and their hyperparameters were selected on the exhaustive "
              "K41 truth table in `docs/golden_path_search_bench.md`; this experiment "
              "runs the same recipes against the real models, where one evaluation costs "
              "eight generations and carries sampling noise.", ""]
    lines += ["The report answers the plan's three descriptive questions: whether the "
              "four algorithms converge on the same schedule and at what cost (section "
              "2), what the schedules they deliver look like next to the baseline schedules and "
              "next to the exhaustive table (section 3), and how they score on the full "
              "four-dataset evaluation (section 4). Section 5 repeats the delivery and "
              "the evaluation for the two other search objectives of section 4b of the "
              "plan, LPIPS and the standardized combination of PSNR and LPIPS. Section 6 "
              "holds the schedule fixed instead and runs the best delivered schedule of "
              "each setting under four more payloads.", ""]
    missing = [f"{model_title(m)} K{k}" for m, _ in MODELS for k in BUDGETS
               if (m, k) not in rendered]
    lines += [f"Settings rendered: "
              f"{join_and([f'{model_title(m)} K{k}' for m, k in rendered]) or 'none'}. "
              f"Settings whose artifacts have not arrived: "
              f"{join_and(missing) or 'none'}.", ""]

    lines += ["### Glossary", ""]
    lines += table(["Term", "Meaning"], [
        ["schedule", "which of the 50 denoising steps are computed at full cost and which "
                     "are cached; written either as the list of full steps or as a 50-bit "
                     "string with `1` = cached"],
        ["payload", "what a cached step uses in place of the computed value; residual "
                    "reuse in sections 1 to 5, and five payloads in section 6"],
        ["`reuse`", "the residual of the last full step, reused unchanged"],
        ["`taylor_o1`", "first-order Taylor extrapolation of the residual"],
        ["`hermite_o2`", "second-order Hermite extrapolation of the residual"],
        ["`mean_avg_vel`", "the average velocity over the interval since the last full step"],
        ["`di_two_anchor`", "extrapolation from the two most recent full steps"],
        ["K", "the number of cached steps, so K29 / K37 / K41 are increasing compression"],
        ["evaluation", "one score of one schedule: 8 generations plus their full-compute "
                       "references, reduced to the mean PSNR over the 8 pairs"],
        ["calibration set", "the 8 frozen (prompt, seed) pairs the search optimises on"],
        ["arbitration", "re-scoring each algorithm's top candidates on 50 held-out COCO "
                        "captions and delivering the best of them"],
        ["search objective", "the single number an evaluation is reduced to, which the "
                             "search maximises; PSNR in sections 2 to 4, and two "
                             "further objectives in section 5"],
        ["baseline schedule", "a schedule published by an existing cache method, here `meancache` "
                      "and `budcache`, used as warm start and as comparison"],
        ["Hamming distance", "number of differing positions between two 50-bit schedule "
                             "strings; moving one cached step changes two bits"],
    ])
    lines += [""]

    lines += section_setup(data)
    lines += section_agreement(data)
    lines += section_structure(data)
    lines += section_evaluation(data)
    lines += section_objectives(data)
    lines += section_payloads(data)
    lines += section_provenance(data)
    lines += appendix(data)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    data = load()
    if data["config"] is None:
        raise SystemExit(f"missing {SS_DIR / 'config.v1.json'}")
    OUT.write_text(render(data), encoding="utf-8")
    rendered = settings_present(data)
    print(f"[render] {OUT.relative_to(REPO)}: {len(rendered)} settings "
          f"({', '.join(f'{m}_k{k}' for m, k in rendered)}), "
          f"{len(data['search'])} search runs, "
          f"{sum(len(v.get('cells', [])) for v in data['results']['models'].values())} P4 cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
