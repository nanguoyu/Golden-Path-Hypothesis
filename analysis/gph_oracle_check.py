"""Oracle-relative check of the Golden Path Hypothesis.

For a fixed model and cache ratio, ask how often a single prompt-independent
schedule stays within a margin of the best quality any evaluated schedule of
the same budget reaches on that prompt.

Per (model, cache ratio) partition:

  * take the reuse-payload rows of a clean pool of fixed schedules,
  * average the quality metric over the available seeds for every prompt,
  * b(x) = max over the pool on prompt x  (the per-prompt oracle of the pool),
  * for every pool member u and margin eps, the share of prompts with
    q(u;x) >= b(x) - eps, plus its one-sided exact binomial (Clopper-Pearson)
    95% lower bound, and the same bound at level 0.05/m with m the pool
    size, which is the Bonferroni correction for reporting the best member.

Image partitions drop the discovery prompts used to pick the modal schedules,
so the reported prompts are held out.  Video partitions are in sample.

The same test runs on three quality metrics.  PSNR and SSIM are read as
recorded; LPIPS is negated so that larger is always better, and its margins
are stated as positive LPIPS differences.  Each metric has its own margins,
anchored on the measured quality value of one cached step for that metric.

The exhaustive K41 anchor compares the schedule selected on the four
PartiPrompts selection pairs with the exact per-prompt optimum over all
1,370,754 feasible schedules.

Run from anywhere:

    python analysis/gph_oracle_check.py
"""
import argparse
import csv
import statistics
import gzip
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta as beta_dist

REPO = Path(__file__).resolve().parents[1]

ALPHA = 0.05
EPS = (0.25, 0.5, 1.0)

# metric -> (sign, margins).  The sign turns the recorded column into a
# larger-is-better score, so LPIPS enters as its negative and its margins stay
# positive LPIPS differences.  Margins are about one half, one and two times
# the median value of one cached step in that metric (see step_value_ladder).
METRICS = {
    "psnr": (1.0, EPS),
    "ssim": (1.0, (0.0075, 0.015, 0.03)),
    "lpips": (-1.0, (0.01, 0.02, 0.04)),
}
EXTRA_METRICS = ("ssim", "lpips")

# ---------------------------------------------------------------- image side
IMAGE_MODELS = ("flux", "qwen")
IMAGE_KS = ("29", "37", "41")
IMAGE_CLEAN = (
    "budcache", "dpcache", "meancache", "uniform",
    "seacache_top1", "teacache_top1", "sencache_top1", "dicache_top1",
    "ham2f", "ham4f", "ham8f",
    "ham2f_d2", "ham2f_d3", "ham4f_d2", "ham4f_d3", "ham8f_d2", "ham8f_d3",
)
IMAGE_RANDOM = ("rand_1", "rand_2", "rand_3", "rand_4", "rand_5")
IMAGE_TABLE = REPO / "resources" / "spx" / "perprompt_spx_{model}.tsv.gz"
SPLITS = REPO / "resources" / "sp_cross_schedules" / "parti_spx_splits.v1.json"

# ---------------------------------------------------------------- video side
VIDEO_MODELS = ("hunyuan_video", "wan21")
VIDEO_CLEAN = (
    "budcache", "meancache", "uniform", "shared",
    "sea_top1", "tea_top1", "sen_top1", "di_top1",
    "ham2f", "ham4f", "ham8f",
)
VIDEO_RANDOM = ("rand_1", "rand_2")
VIDEO_TABLE = REPO / "resources" / "video_spx" / "{model}" / "pervideo_spx_{model}.tsv.gz"

# ------------------------------------------------------------- K41 exhaustive
K41_SUMMARY = REPO / "resources" / "exhaustive_k41" / "formal_results" / "summary.json"
K41_MANIFEST = REPO / "resources" / "exhaustive_k41" / "formal_results" / "candidate_manifest.tsv"
K41_SELECTED_RANK = 164762
K41_PROMPTS = ("5", "8", "9", "15")

# Verified values reproduced by this script.  share / lower bound at the three
# margins, for the best pool member of each image partition.
EXPECT_IMAGE = {
    ("flux", "29"): ("meancache", ((0.694, 0.670), (0.776, 0.754), (0.869, 0.851))),
    ("flux", "37"): ("ham2f_d3", ((0.443, 0.418), (0.571, 0.546), (0.715, 0.692))),
    ("flux", "41"): ("budcache", ((0.653, 0.629), (0.752, 0.729), (0.882, 0.865))),
    ("qwen", "29"): ("meancache", ((0.482, 0.456), (0.583, 0.558), (0.767, 0.744))),
    ("qwen", "37"): ("meancache", ((0.530, 0.505), (0.617, 0.592), (0.763, 0.741))),
    ("qwen", "41"): ("meancache", ((0.756, 0.734), (0.849, 0.830), (0.949, 0.937))),
}
# lower bound of the best pool member at eps = 0.5 dB
EXPECT_VIDEO_LB_E05 = {
    ("hunyuan_video", "29"): 0.588, ("hunyuan_video", "37"): 0.657,
    ("hunyuan_video", "41"): 0.667,
    ("wan21", "29"): 0.716, ("wan21", "37"): 0.695, ("wan21", "41"): 0.787,
}
EXPECT_VIDEO_LB_E025 = {("hunyuan_video", "29"): 0.481}
# Bonferroni-corrected lower bound of the best pool member at eps = 0.5 dB,
# computed at level 0.05/m with m the pool size of that partition.
EXPECT_CORRECTED_LB_E05 = {
    ("flux", "29"): 0.739, ("flux", "37"): 0.529, ("flux", "41"): 0.717,
    ("qwen", "29"): 0.541, ("qwen", "37"): 0.575, ("qwen", "41"): 0.819,
    ("hunyuan_video", "29"): 0.561, ("hunyuan_video", "37"): 0.631,
    ("hunyuan_video", "41"): 0.641,
    ("wan21", "29"): 0.690, ("wan21", "37"): 0.670, ("wan21", "41"): 0.765,
}
EXPECT_CORRECTED_PASS = 12
EXPECT_CORRECTED_WORST = 0.529
EXPECT_RANDOM_MAX_E025 = {"image": 0.057, "video": 0.020}

# Same twelve partitions read with the two extra metrics: best pool member and
# its share / lower bound at the three margins of that metric.
EXPECT_EXTRA = {
    "ssim": {
        ("flux", "29"): ("meancache", ((0.825, 0.805), (0.917, 0.902), (0.968, 0.958))),
        ("flux", "37"): ("ham2f_d3", ((0.484, 0.459), (0.602, 0.577), (0.773, 0.751))),
        ("flux", "41"): ("seacache_top1", ((0.608, 0.583), (0.723, 0.700), (0.866, 0.848))),
        ("qwen", "29"): ("meancache", ((0.619, 0.595), (0.835, 0.816), (0.964, 0.953))),
        ("qwen", "37"): ("meancache", ((0.481, 0.455), (0.626, 0.601), (0.821, 0.801))),
        ("qwen", "41"): ("budcache", ((0.551, 0.526), (0.689, 0.665), (0.866, 0.848))),
        ("hunyuan_video", "29"): ("meancache", ((0.730, 0.685), (0.847, 0.808), (0.940, 0.912))),
        ("hunyuan_video", "37"): ("meancache", ((0.523, 0.474), (0.633, 0.585), (0.750, 0.705))),
        ("hunyuan_video", "41"): ("budcache", ((0.503, 0.454), (0.597, 0.548), (0.723, 0.678))),
        ("wan21", "29"): ("meancache", ((0.567, 0.518), (0.680, 0.633), (0.823, 0.783))),
        ("wan21", "37"): ("budcache", ((0.613, 0.565), (0.690, 0.643), (0.827, 0.787))),
        ("wan21", "41"): ("budcache", ((0.760, 0.716), (0.863, 0.826), (0.950, 0.924))),
    },
    "lpips": {
        ("flux", "29"): ("meancache", ((0.851, 0.832), (0.936, 0.922), (0.983, 0.976))),
        ("flux", "37"): ("ham2f_d3", ((0.480, 0.454), (0.623, 0.598), (0.791, 0.770))),
        ("flux", "41"): ("seacache_top1", ((0.660, 0.636), (0.761, 0.739), (0.870, 0.852))),
        ("qwen", "29"): ("budcache", ((0.649, 0.624), (0.846, 0.826), (0.955, 0.943))),
        ("qwen", "37"): ("meancache", ((0.463, 0.438), (0.592, 0.567), (0.798, 0.777))),
        ("qwen", "41"): ("budcache", ((0.545, 0.520), (0.675, 0.650), (0.853, 0.834))),
        ("hunyuan_video", "29"): ("meancache", ((0.727, 0.681), (0.880, 0.845), (0.967, 0.944))),
        ("hunyuan_video", "37"): ("sea_top1", ((0.493, 0.444), (0.553, 0.504), (0.647, 0.599))),
        ("hunyuan_video", "41"): ("sen_top1", ((0.487, 0.438), (0.557, 0.508), (0.690, 0.643))),
        ("wan21", "29"): ("meancache", ((0.690, 0.643), (0.840, 0.801), (0.960, 0.936))),
        ("wan21", "37"): ("meancache", ((0.617, 0.568), (0.790, 0.748), (0.947, 0.920))),
        ("wan21", "41"): ("meancache", ((0.690, 0.643), (0.857, 0.819), (0.970, 0.948))),
    },
}
EXPECT_EXTRA_MID_PASS = {"ssim": 12, "lpips": 12}
EXPECT_EXTRA_RANDOM = {"ssim": {"image": 0.118, "video": 0.117},
                       "lpips": {"image": 0.132, "video": 0.087}}
EXPECT_K41_GAPS = (0.030, 0.102, 0.269, 1.249)

TOL = 0.005


def cp_lower(k, n, alpha=ALPHA):
    """One-sided exact binomial (Clopper-Pearson) lower bound for k/n."""
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def mean(values):
    return sum(values) / len(values)


# ----------------------------------------------------------------- loading


def load_image(model, drop_discovery=True, metric="psnr"):
    """(k, schedule) -> {prompt: seed-averaged score} for the clean + random pool."""
    sign = METRICS[metric][0]
    keep = set(IMAGE_CLEAN) | set(IMAGE_RANDOM)
    discovery = set()
    if drop_discovery:
        with open(SPLITS) as fh:
            discovery = set(json.load(fh)["roles"]["discovery"])
    raw = defaultdict(lambda: defaultdict(list))
    with gzip.open(str(IMAGE_TABLE).format(model=model), "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["payload"] != "reuse" or row["schedule"] not in keep:
                continue
            prompt = int(row["prompt_idx"])
            if prompt in discovery:
                continue
            raw[(row["k"], row["schedule"])][prompt].append(sign * float(row[metric]))
    return {key: {p: mean(v) for p, v in table.items()} for key, table in raw.items()}


def load_video(model, metric="psnr"):
    """(K, schedule) -> {(dataset, prompt): seed-averaged score}."""
    sign = METRICS[metric][0]
    keep = set(VIDEO_CLEAN) | set(VIDEO_RANDOM)
    raw = defaultdict(lambda: defaultdict(list))
    with gzip.open(str(VIDEO_TABLE).format(model=model), "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["payload"] != "reuse" or row["row"] not in keep:
                continue
            key = (row["dataset"], row["prompt_idx"])
            raw[(row["K"], row["row"])][key].append(sign * float(row[metric]))
    return {key: {p: mean(v) for p, v in table.items()} for key, table in raw.items()}


# ----------------------------------------------------------------- analysis


def partition(tables, k, clean, random_rows, margins=EPS):
    """Shares and lower bounds for one (model, cache ratio) partition.

    Each margin gives three numbers: the share, its one-sided 95% lower
    bound, and the same bound recomputed at level 0.05/m with m the pool
    size.  The third number is the Bonferroni correction for reporting the
    best of the m pool members.
    """
    pool = [s for s in clean if (k, s) in tables]
    rands = [s for s in random_rows if (k, s) in tables]
    prompts = sorted(set.intersection(*(set(tables[(k, s)]) for s in pool)))
    best = {x: max(tables[(k, s)][x] for s in pool) for x in prompts}
    alpha_corr = ALPHA / len(pool)

    def score(name):
        table = tables[(k, name)]
        rows = []
        for eps in margins:
            hits = sum(1 for x in prompts if x in table and table[x] >= best[x] - eps)
            n = sum(1 for x in prompts if x in table)
            rows.append((hits / n, cp_lower(hits, n), cp_lower(hits, n, alpha_corr)))
        return rows

    return {
        "n_prompts": len(prompts),
        "pool_size": len(pool),
        "clean": {s: score(s) for s in pool},
        "random": {s: score(s) for s in rands},
    }


def best_member(result):
    """Pool member with the largest share at the tightest margin.

    Ties keep the earlier member of the declared pool order.  The same member
    is then read at every margin, so one path carries the whole row.
    """
    scores = result["clean"]
    name = max(scores, key=lambda s: scores[s][0][0])
    return name, scores[name]


def k41_gaps():
    """Per-prompt gap between the exact optimum and the selected schedule."""
    with open(K41_SUMMARY) as fh:
        summary = json.load(fh)
    optimum = {p: summary["top_per_prompt"][p][0]["psnr_db"] for p in K41_PROMPTS}
    selected = None
    with open(K41_MANIFEST) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(row["rank"]) == K41_SELECTED_RANK:
                selected = {p: float(row[f"psnr_p{p}"]) for p in K41_PROMPTS}
                break
    if selected is None:
        raise SystemExit(f"rank {K41_SELECTED_RANK} not in {K41_MANIFEST}")
    return {p: (optimum[p], selected[p], optimum[p] - selected[p]) for p in K41_PROMPTS}


# ------------------------------------------------------------------ report


TABLES = ("budcache", "dpcache", "meancache", "uniform")
EXPECT_STEP_VALUE = (0.171, 1.166)      # min and max of the 16 ladder slopes
# min, median and max of the 16 ladder slopes for the two extra metrics
EXPECT_STEP_VALUE_EXTRA = {
    "ssim": (0.0054, 0.0138, 0.0367),
    "lpips": (0.0080, 0.0202, 0.0495),
}


def step_value_ladder(metric="psnr"):
    """Quality value of one cached step: median held-out score of the four
    fixed tables (reuse payload), differenced across adjacent budgets."""
    sign = METRICS[metric][0]
    with open(REPO / "resources/sp_cross_schedules/parti_spx_splits.v1.json") as fh:
        disc = set(json.load(fh)["roles"]["discovery"])
    slopes = []
    for model in ("flux", "qwen"):
        med = {}
        acc = {}
        with gzip.open(REPO / f"resources/spx/perprompt_spx_{model}.tsv.gz", "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["schedule"] in TABLES and r["payload"] == "reuse":
                    pid = int(r["prompt_idx"])
                    if pid in disc:
                        continue
                    acc.setdefault((r["schedule"], r["k"]), {}).setdefault(pid, []).append(
                        sign * float(r[metric]))
        for key, d in acc.items():
            med[key] = statistics.median(sum(v) / len(v) for v in d.values())
        for s in TABLES:
            for k0, k1 in (("29", "37"), ("37", "41")):
                if (s, k0) in med and (s, k1) in med:
                    slopes.append((med[(s, k0)] - med[(s, k1)]) / (int(k1) - int(k0)))
    return slopes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no_assert", action="store_true",
                    help="print the tables without checking the recorded values")
    args = ap.parse_args()

    failures = []

    def check(label, got, want, tol=TOL):
        if abs(got - want) > tol:
            failures.append(f"{label}: got {got:.4f}, recorded {want:.4f}")

    def header_for(margins):
        cells = "".join(f"{'  eps=' + str(e):>14}" for e in margins)
        return f"{'partition':<22}{'best fixed path':<16}{'n':>6}{'pool':>6}{cells}"

    header = header_for(EPS)
    pools = {}

    def run_side(models, loader, clean, random_rows, margins=EPS, unit=" dB",
                 verbose=True):
        rows_out = []
        for model in models:
            tables = loader(model)
            for k in IMAGE_KS:
                res = partition(tables, k, clean, random_rows, margins)
                name, rows = best_member(res)
                pools[(model, k)] = res["pool_size"]
                cells = "".join(f"  {s:.3f}/{lb:.3f}" for s, lb, _ in rows)
                print(f"{model + ' K' + k:<22}{name:<16}{res['n_prompts']:>6}"
                      f"{res['pool_size']:>6}{cells}")
                counts = [sum(1 for s in res["clean"] if res["clean"][s][i][1] > 0.5)
                          for i in range(len(margins))]
                rand_best = max((v[0][0] for v in res["random"].values()), default=0.0)
                if verbose:
                    print(f"{'':<22}pool members with lower bound > 0.5: "
                          + ", ".join(f"{c} at {e}{unit}" for c, e in zip(counts, margins))
                          + f"; best random share at {margins[0]}{unit} {rand_best:.3f}")
                rows_out.append((model, k, name, rows, counts, rand_best))
        return rows_out

    print("Oracle-relative coverage: share of prompts within eps dB PSNR of the")
    print("per-prompt best of the evaluated pool (share / one-sided 95% lower bound).")
    print("The row reports the pool member with the largest share at 0.25 dB.")
    print()
    print("IMAGE (held-out PartiPrompts, seed-averaged)")
    print(header)
    image = run_side(IMAGE_MODELS, load_image, IMAGE_CLEAN, IMAGE_RANDOM)
    if not args.no_assert:
        for model, k, name, rows, _, _ in image:
            want_name, want_rows = EXPECT_IMAGE[(model, k)]
            if name != want_name:
                failures.append(
                    f"{model} K{k} best path: got {name}, recorded {want_name}")
            for (share, lb, _), (ws, wlb), eps in zip(rows, want_rows, EPS):
                check(f"{model} K{k} eps={eps} share", share, ws)
                check(f"{model} K{k} eps={eps} lower bound", lb, wlb)

    print()
    print("VIDEO (in-sample, two datasets pooled)")
    print(header)
    video = run_side(VIDEO_MODELS, load_video, VIDEO_CLEAN, VIDEO_RANDOM)
    if not args.no_assert:
        for model, k, name, rows, _, _ in video:
            check(f"{model} K{k} eps=0.5 lower bound",
                  rows[1][1], EXPECT_VIDEO_LB_E05[(model, k)])
            if (model, k) in EXPECT_VIDEO_LB_E025:
                check(f"{model} K{k} eps=0.25 lower bound",
                      rows[0][1], EXPECT_VIDEO_LB_E025[(model, k)])

    print()
    for label, side in (("image", image), ("video", video)):
        lbs = [rows[1][1] for _, _, _, rows, _, _ in side]
        print(f"{label}: lower bound at 0.5 dB spans {min(lbs):.3f} to {max(lbs):.3f} "
              f"over {len(side)} partitions")
    total = len(image) + len(video)
    for i, eps in enumerate(EPS):
        passed = sum(1 for _, _, _, _, counts, _ in image + video if counts[i] > 0)
        print(f"partitions with at least one path above one half at {eps} dB: "
              f"{passed}/{total}")
    print()
    print("BONFERRONI CORRECTION over the pool.  The reported path is the best of")
    print("m pool members, so its lower bound is recomputed at level 0.05/m.")
    print(f"{'partition':<22}{'best fixed path':<16}{'m':>4}"
          f"{'share':>9}{'95% LB':>9}{'corrected':>11}{'margin':>9}")
    corrected = []
    for model, k, name, rows, _, _ in image + video:
        m = pools[(model, k)]
        share, lb, lb_corr = rows[1]
        corrected.append((model, k, name, m, share, lb, lb_corr))
        print(f"{model + ' K' + k:<22}{name:<16}{m:>4}{share:>9.3f}{lb:>9.3f}"
              f"{lb_corr:>11.3f}{lb_corr - 0.5:>+9.3f}")
    n_corr_pass = sum(1 for row in corrected if row[6] > 0.5)
    worst = min(row[6] for row in corrected)
    print(f"partitions with a corrected lower bound above one half at 0.5 dB: "
          f"{n_corr_pass}/{len(corrected)}; worst corrected bound {worst:.3f} "
          f"(margin {worst - 0.5:+.3f})")
    if not args.no_assert:
        for model, k, name, m, share, lb, lb_corr in corrected:
            want = EXPECT_CORRECTED_LB_E05[(model, k)]
            check(f"{model} K{k} corrected lower bound", lb_corr, want)
        check("corrected pass count", n_corr_pass, EXPECT_CORRECTED_PASS, tol=0.5)
        check("worst corrected bound", worst, EXPECT_CORRECTED_WORST)

    print()
    image_random_max = max(r for _, _, _, _, _, r in image)
    video_random_max = max(r for _, _, _, _, _, r in video)
    print(f"best random-path share at 0.25 dB: image {image_random_max:.3f}, "
          f"video {video_random_max:.3f}")
    if not args.no_assert:
        check("image random max", image_random_max, EXPECT_RANDOM_MAX_E025["image"])
        check("video random max", video_random_max, EXPECT_RANDOM_MAX_E025["video"])

    print()
    print("EXHAUSTIVE K41 (all 1,370,754 schedules, four selection pairs)")
    print(f"{'prompt':<10}{'exact best dB':>15}{'selected dB':>14}{'gap dB':>10}")
    gaps = k41_gaps()
    for p in K41_PROMPTS:
        opt, sel, gap = gaps[p]
        print(f"{p:<10}{opt:>15.3f}{sel:>14.3f}{gap:>10.3f}")
    if not args.no_assert:
        got = sorted(g for _, _, g in gaps.values())
        for g, w in zip(got, sorted(EXPECT_K41_GAPS)):
            check("K41 gap", g, w)

    print()
    slopes = step_value_ladder()
    print(f"quality value of one cached step: {min(slopes):.3f} to {max(slopes):.3f} dB "
          f"(median {statistics.median(slopes):.3f}, 16 fixed-table ladder slopes)")
    if not args.no_assert:
        check("step value min", min(slopes), EXPECT_STEP_VALUE[0])
        check("step value max", max(slopes), EXPECT_STEP_VALUE[1])

    print()
    print("METRIC ROBUSTNESS: the same twelve partitions read with SSIM and LPIPS.")
    print("LPIPS is negated so that larger is better, so an LPIPS margin below is a")
    print("positive LPIPS difference.  Margins are about one half, one and two times")
    print("the median value of one cached step in that metric.")
    for metric in EXTRA_METRICS:
        margins = METRICS[metric][1]
        lad = step_value_ladder(metric)
        print()
        print(f"{metric.upper()}: one cached step is worth {min(lad):.4f} to "
              f"{max(lad):.4f} (median {statistics.median(lad):.4f}, 16 ladder "
              f"slopes); margins " + ", ".join(str(e) for e in margins))
        print(header_for(margins))
        side = []
        for models, loader, clean, rands, label in (
                (IMAGE_MODELS, load_image, IMAGE_CLEAN, IMAGE_RANDOM, "image"),
                (VIDEO_MODELS, load_video, VIDEO_CLEAN, VIDEO_RANDOM, "video")):
            got = run_side(models, lambda m, ld=loader, mt=metric: ld(m, metric=mt),
                           clean, rands, margins, verbose=False)
            side.append((label, got))
        rows_all = side[0][1] + side[1][1]
        mid_pass = sum(1 for _, _, _, r, _, _ in rows_all if r[1][1] > 0.5)
        print(f"partitions whose best path has a lower bound above one half at the "
              f"middle margin {margins[1]}: {mid_pass}/{len(rows_all)}")
        rand = {label: max(r for _, _, _, _, _, r in got) for label, got in side}
        print(f"best random-path share at {margins[0]}: image {rand['image']:.3f}, "
              f"video {rand['video']:.3f}")
        if not args.no_assert:
            lo, med_, hi = EXPECT_STEP_VALUE_EXTRA[metric]
            check(f"{metric} step min", min(lad), lo, tol=0.0005)
            check(f"{metric} step median", statistics.median(lad), med_, tol=0.0005)
            check(f"{metric} step max", max(lad), hi, tol=0.0005)
            for model, k, name, rows, _, _ in rows_all:
                want_name, want_rows = EXPECT_EXTRA[metric][(model, k)]
                if name != want_name:
                    failures.append(f"{metric} {model} K{k} best path: got {name}, "
                                    f"recorded {want_name}")
                for (share, lb, _), (ws, wlb), eps in zip(rows, want_rows, margins):
                    check(f"{metric} {model} K{k} eps={eps} share", share, ws)
                    check(f"{metric} {model} K{k} eps={eps} lower bound", lb, wlb)
            check(f"{metric} middle-margin pass count", mid_pass,
                  EXPECT_EXTRA_MID_PASS[metric], tol=0.5)
            for label in ("image", "video"):
                check(f"{metric} {label} random max", rand[label],
                      EXPECT_EXTRA_RANDOM[metric][label])

    print()
    if args.no_assert:
        print("checks skipped (--no_assert)")
    elif failures:
        print("MISMATCH against the recorded values:")
        for line in failures:
            print("  " + line)
        raise SystemExit(1)
    else:
        print("all recorded values reproduced")


if __name__ == "__main__":
    main()
