#!/usr/bin/env python3
"""Render the schedule-search LaTeX tables of the paper.

Reads resources/schedule_search/results.json (paired differences, arbitration
check) and the staged per-image tables (for the paired difference against the
random control, which results.json does not carry), and writes

  paper/tables/search_paired_main.tex     main text, PSNR, three references
  paper/tables/search_paired_full.tex     appendix, five metrics, two references
  paper/tables/search_arbitration.tex     appendix, arbitration check

Every number comes from those two sources.  Run from the repository root.
"""

import collections
import gzip
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources", "schedule_search")
OUT = os.path.join(ROOT, "paper", "tables")

# The selected schedule of each setting has the highest validation mean among
# the candidates retained from the four search algorithms.
BEST = {
    ("flux", 29): "ss_hill",
    ("flux", 37): "ss_hill",
    ("flux", 41): "ss_hill",
    ("qwen", 29): "ss_anneal",
    ("qwen", 37): "ss_anneal",
    ("qwen", 41): "ss_hill",
}
MODEL_NAME = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
RATIO = {29: "0.58", 37: "0.74", 41: "0.82"}
DATASETS = [
    ("drawbench_full", "DrawBench"),
    ("geneval_style", "GenEval-style"),
    ("parti_full", "PartiPrompts"),
    ("diffusiondb_clean10k", "DiffusionDB"),
]
PERPROMPT = {
    "flux": ["perprompt_search_flux_site_a.tsv.gz", "perprompt_search_flux_site_c.tsv.gz",
             "perprompt_search_flux_site_b.tsv.gz"],
    "qwen": ["perprompt_search_qwen_site_a.tsv.gz", "perprompt_search_qwen_site_c.tsv.gz"],
}


def load_results():
    with open(os.path.join(RES, "results.json")) as fh:
        return json.load(fh)


def paired(res, model, k, schedule, incumbent, dataset):
    for row in res["models"][model]["paired"]:
        if (row["k"] == k and row["schedule"] == schedule
                and row["incumbent"] == incumbent and row["dataset"] == dataset):
            return row
    raise KeyError((model, k, schedule, incumbent, dataset))


def random_control():
    """Paired PSNR difference, selected schedule minus the random-search schedule."""
    out = {}
    for model, files in PERPROMPT.items():
        table = collections.defaultdict(dict)
        for name in files:
            with gzip.open(os.path.join(RES, name), "rt") as fh:
                head = fh.readline().rstrip("\n").split("\t")
                col = {c: i for i, c in enumerate(head)}
                for line in fh:
                    p = line.rstrip("\n").split("\t")
                    key = (int(p[col["k"]]), p[col["dataset"]],
                           p[col["schedule"]], p[col["payload"]])
                    table[key][(p[col["seed"]], p[col["prompt_idx"]])] = float(p[col["psnr"]])
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            for dataset, _ in DATASETS:
                a = table[(k, dataset, sched, "reuse")]
                b = table[(k, dataset, "ss_random", "reuse")]
                keys = sorted(set(a) & set(b))
                diff = [a[key] - b[key] for key in keys]
                n = len(diff)
                mean = sum(diff) / n
                rng = random.Random(20260903)
                draws = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n
                               for _ in range(2000))
                out[(model, k, dataset)] = (n, mean, draws[49], draws[1949])
    return out


def fmt(value, digits=3):
    return f"{value:+.{digits}f}"


def ci(row, key="ci95"):
    lo, hi = row[key]
    return f"[{lo:.2f}, {hi:.2f}]"


def write_main_table(res, rc):
    """Write the compact 24-row transfer table used in the main text."""
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{\textbf{Searched schedules achieve higher or comparable mean PSNR",
        r"relative to all three reference schedules.}",
        r"For each prompt and seed, we subtract the reference schedule's output PSNR",
        r"from the searched schedule's output PSNR.  Entries are means of these",
        r"differences, so positive values favor the searched schedule.",
        r"All schedules use residual reuse.",
        r"Each of the four search procedures receives the same candidate-evaluation",
        r"budget within a model and cache ratio.  The searched schedule has the highest",
        r"mean PSNR on 50 validation prompts among candidates from all four procedures.",
        r"The random-search comparison uses the schedule selected from random search alone.",
        r"We average over 600 DrawBench, 1,659 GenEval-style, 4,896 PartiPrompts, or",
        r"30,000 DiffusionDB prompt--seed runs.  Confidence intervals and four additional",
        r"metrics appear in Appendix Table~\ref{tab:search-paired-full}.}",
        r"\label{tab:search-summary}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Model & Cache ratio & Dataset & \multicolumn{3}{c}{Mean $\Delta$PSNR: searched $-$ comparison schedule, dB} \\",
        r"\cmidrule(lr){4-6}",
        r" & & & MeanCache & BudCache & Random search \\",
        r"\midrule",
    ]
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            for i, (dataset, label) in enumerate(DATASETS):
                mean_row = paired(res, model, k, sched, "meancache", dataset)
                bud_row = paired(res, model, k, sched, "budcache", dataset)
                _, random_mean, _, _ = rc[(model, k, dataset)]
                model_head = MODEL_NAME[model] if i == 0 else ""
                ratio_head = RATIO[k] if i == 0 else ""
                lines.append(
                    f"{model_head} & {ratio_head} & {label} & "
                    f"{fmt(mean_row['delta_psnr_mean'], 2)} & "
                    f"{fmt(bud_row['delta_psnr_mean'], 2)} & "
                    f"{random_mean:+.2f} \\\\")
            if not (model == "qwen" and k == 41):
                lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(OUT, "search_paired_main.tex"), "w") as fh:
        fh.write("\n".join(lines))


def write_transposed_main_table(res, rc):
    """Write six settings with dataset columns grouped by reference schedule."""
    lines = [
        r"\begin{table}[!htbp]",
        "\\caption{\\textbf{Searched schedules achieve higher or comparable mean PSNR",
        "relative to all three reference schedules.}",
        "For each prompt and seed, we subtract the reference schedule's output PSNR",
        "from the searched schedule's output PSNR.  Entries are means of these",
        "differences, so positive values favor the searched schedule.",
        "All schedules use residual reuse.",
        "Each of the four search procedures receives the same candidate-evaluation",
        "budget within a model and cache ratio.  The searched schedule has the highest",
        "mean PSNR on 50 validation prompts among candidates from all four procedures.",
        "The random-search comparison uses the schedule selected from random search alone.",
        "FLUX denotes FLUX.1-dev and Qwen denotes Qwen-Image.",
        "DB, GE, PP, and DDB denote DrawBench, GenEval-style, PartiPrompts, and",
        "DiffusionDB-clean10k, respectively.",
        "We average over 600, 1,659, 4,896, or 30,000 prompt--seed runs per dataset,",
        "in that order.  Confidence",
        "intervals and four additional metrics appear in Appendix",
        "Table~\\ref{tab:search-paired-full}.}",
        r"\label{tab:search-summary}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabular}{ll*{4}{r}@{\hspace{3pt}}*{4}{r}@{\hspace{3pt}}*{4}{r}}",
        r"\toprule",
        r"Model & \shortstack{Cache\\ratio} & \multicolumn{12}{c}{Mean $\Delta$PSNR: searched $-$ comparison schedule, dB} \\",
        r"\cmidrule(lr){3-14}",
        r" & & \multicolumn{4}{c}{MeanCache} & \multicolumn{4}{c}{BudCache} &",
        r"\multicolumn{4}{c}{Random search} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r" & & DB & GE & PP & DDB & DB & GE & PP & DDB & DB & GE & PP & DDB \\",
        r"\midrule",
    ]
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            row = [{"flux": "FLUX", "qwen": "Qwen"}[model], RATIO[k]]
            for incumbent in ("meancache", "budcache"):
                for dataset, _ in DATASETS:
                    result = paired(res, model, k, sched, incumbent, dataset)
                    row.append(fmt(result["delta_psnr_mean"], 2))
            for dataset, _ in DATASETS:
                _, random_mean, _, _ = rc[(model, k, dataset)]
                row.append(f"{random_mean:+.2f}")
            lines.append(" & ".join(row) + r" \\")
        if model == "flux":
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(OUT, "search_paired_main.tex"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    res = load_results()
    rc = random_control()

    # ---- main text: one float, two panels ---------------------------------
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Searched schedules achieve higher or comparable mean PSNR"
        r" relative to all three reference schedules.}  For each metric, we subtract"
        r" the value for the reference schedule's output from the value for the searched"
        r" schedule's output, using the same prompt and seed.  Entries are means of these"
        r" differences. Positive PSNR, SSIM, and ImageReward differences and negative LPIPS"
        r" differences favor the searched schedule.  All schedules use residual reuse."
        r" Each of the four search procedures receives the same candidate-evaluation"
        r" budget within a model and cache ratio.  The searched schedule has the highest"
        r" mean PSNR on 50 validation prompts among candidates from all four procedures."
        r" The random-search comparison uses the schedule selected from random search alone."
        r" We average over 600 DrawBench, 1,659 GenEval-style, 4,896 PartiPrompts, or 30,000"
        r" DiffusionDB prompt--seed runs.  \emph{Top:} Mean PSNR differences in dB with 95\%"
        r" bootstrap intervals obtained by resampling prompt--seed runs individually."
        r" \emph{Bottom:} Ranges of mean metric differences relative to MeanCache over"
        r" the four datasets. Appendix Table~\ref{tab:search-paired-full} gives every"
        r" dataset and both references.}",
        r"\label{tab:search-paired}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Setting & Dataset & $-$ MeanCache & $-$ BudCache & $-$ random \\",
        r"\midrule",
    ]
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            for i, (dataset, label) in enumerate(DATASETS):
                mean_row = paired(res, model, k, sched, "meancache", dataset)
                bud_row = paired(res, model, k, sched, "budcache", dataset)
                n, rmean, rlo, rhi = rc[(model, k, dataset)]
                head = (f"{MODEL_NAME[model]}, ratio {RATIO[k]}" if i == 0 else "")
                lines.append(
                    f"{head} & {label} & "
                    f"{fmt(mean_row['delta_psnr_mean'], 2)} {ci(mean_row)} & "
                    f"{fmt(bud_row['delta_psnr_mean'], 2)} {ci(bud_row)} & "
                    f"{rmean:+.2f} [{rlo:+.2f}, {rhi:+.2f}] \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{4pt}",
        r"",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & $\Delta$PSNR (dB) & $\Delta$SSIM & $\Delta$LPIPS & $\Delta$ImageReward \\",
        r"\midrule",
    ]
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            rows = [paired(res, model, k, sched, "meancache", d) for d, _ in DATASETS]

            def span(key, digits):
                vals = sorted(r[key] for r in rows)
                return f"{vals[0]:+.{digits}f} to {vals[-1]:+.{digits}f}"

            lines.append(
                f"{MODEL_NAME[model]}, ratio {RATIO[k]} & {span('delta_psnr_mean', 2)} & "
                f"{span('delta_ssim_mean', 4)} & {span('delta_lpips_mean', 4)} & "
                f"{span('delta_image_reward_mean', 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(OUT, "search_paired_main.tex"), "w") as fh:
        fh.write("\n".join(lines))

    # ---- appendix, five metrics against both reference schedules ----------
    lines = [
        r"\begingroup",
        r"\normalsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\setlength{\LTpre}{0pt}",
        r"\renewcommand{\arraystretch}{0.7273}",
        r"\begin{longtable}{*{3}{>{\scriptsize}l}*{6}{>{\scriptsize}r}}",
        r"\caption{\textbf{Searched schedules improve PSNR over BudCache in every reported",
        r"model, cache ratio, and dataset combination.}  Each row compares the selected",
        r"schedule with one reference schedule under residual reuse.",
        r"For each metric, we subtract the value for the reference schedule's output",
        r"from the value for the searched schedule's output, using the same prompt and seed.",
        r"Entries are means of these differences over the listed prompt--seed runs.",
        r"Positive PSNR, SSIM, ImageReward, and CLIP differences and negative LPIPS",
        r"differences favor the searched schedule.  IR denotes ImageReward.",
        r"PSNR differences are in dB.  Brackets give 95\% bootstrap intervals obtained",
        r"by resampling prompt--seed runs individually.}",
        r"\label{tab:search-paired-full}\\",
        r"\toprule",
        r"Model, cache ratio & Dataset & Reference & Runs & $\Delta$PSNR (dB) & $\Delta$SSIM"
        r" & $\Delta$LPIPS & $\Delta$IR & $\Delta$CLIP \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{l}{\scriptsize Table~\thetable\ continued} \\",
        r"\toprule",
        r"Model, cache ratio & Dataset & Reference & Runs & $\Delta$PSNR (dB) & $\Delta$SSIM"
        r" & $\Delta$LPIPS & $\Delta$IR & $\Delta$CLIP \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{\scriptsize Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for model in ("flux", "qwen"):
        for k in (29, 37, 41):
            sched = BEST[(model, k)]
            first = True
            for dataset, label in DATASETS:
                for inc, inc_label in (("meancache", "MeanCache"), ("budcache", "BudCache")):
                    row = paired(res, model, k, sched, inc, dataset)
                    head = (f"{MODEL_NAME[model]}, ratio {RATIO[k]}" if first else "")
                    first = False
                    lines.append(
                        f"{head} & {label} & {inc_label} & {row['n_pairs']:,} & "
                        f"{row['delta_psnr_mean']:+.3f} {ci(row)} & "
                        f"{row['delta_ssim_mean']:+.4f} & {row['delta_lpips_mean']:+.4f} & "
                        f"{row['delta_image_reward_mean']:+.3f} & "
                        f"{row['delta_clip_mean']:+.3f} \\\\")
                    # Keep each model--ratio group together across pages.
                    if not (dataset == DATASETS[-1][0] and inc == "budcache"):
                        lines[-1] += "*"
            if not (model == "qwen" and k == 41):
                lines.append(r"\midrule")
    lines += [r"\end{longtable}", r"\endgroup", ""]
    with open(os.path.join(OUT, "search_paired_full.tex"), "w") as fh:
        fh.write("\n".join(lines))

    # ---- appendix, validation check ---------------------------------------
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{\textbf{Validation changes the selected candidate in eight search runs.}",
        r"The replaced candidate has the highest mean PSNR on the eight scoring runs.",
        r"The selected candidate has the highest mean PSNR on 50 validation prompts.",
        r"The first two numeric columns give the candidates' mean validation PSNR.",
        r"For each metric, we subtract the value for the replaced candidate's output",
        r"from the value for the selected candidate's output on the same prompt and seed.",
        r"The last three columns report the means of these differences over",
        r"4,896 PartiPrompts prompt--seed runs.  Positive PSNR and SSIM differences and",
        r"negative LPIPS differences favor the selected candidate.  All PSNR values are",
        r"in dB.  Brackets give a 95\% bootstrap interval for the PSNR difference.",
        r"Hill, anneal, and greedy denote hill climbing, annealing, and greedy coordinate ascent.}",
        r"\label{tab:search-arbitration}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model, cache ratio & Procedure & Selected PSNR & Replaced PSNR & $\Delta$PSNR (dB)"
        r" & $\Delta$SSIM & $\Delta$LPIPS \\",
        r"\midrule",
    ]
    deltas = []
    for model in ("flux", "qwen"):
        for row in sorted(res["models"][model]["arbitration_check"], key=lambda r: r["k"]):
            deltas.append(row["delta_psnr_mean"])
            lo, hi = row["ci95_psnr"]
            lines.append(
                f"{MODEL_NAME[model]}, ratio {RATIO[row['k']]} & {row['algorithm']} & "
                f"{row['delivered_arbitration_mean_psnr_db']:.3f} & "
                f"{row['calibration_best_arbitration_mean_psnr_db']:.3f} & "
                f"{row['delta_psnr_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] & "
                f"{row['delta_ssim_mean']:+.4f} & {row['delta_lpips_mean']:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(OUT, "search_arbitration.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print("arbitration rows:", len(deltas),
          "mean delta psnr: %+.3f" % (sum(deltas) / len(deltas)),
          "min %+.3f max %+.3f" % (min(deltas), max(deltas)))
    # This final write replaces the older two-panel table constructed above.
    write_transposed_main_table(res, rc)


if __name__ == "__main__":
    main()
