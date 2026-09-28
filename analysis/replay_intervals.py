"""Prompt-clustered intervals for the image fixed-replay differences.

The paper states one protocol for these intervals (Appendix C.2 and
Appendix F): average the three random-seed settings within a prompt, then
treat prompts as the sampling unit and resample them.  This script carries
that out for the held-out signed difference of the image fixed-replay table.

For each model, adaptive method and cache count K:

  * read the frozen modal path of that setting,
  * read the method's own run and the same path replayed with the method's
    payload, both per (seed, prompt),
  * keep the held-out prompts on which the method chose a different path,
  * form the paired difference, fixed minus adaptive,
  * average the seeds within each prompt,
  * resample prompts and report the 2.5 and 97.5 percentiles of the mean.

A prompt enters the resample with its seed average and with the number of
seeds behind it.  The mean of a resample weights each drawn prompt by that
count, so on the original sample the statistic is the table's own value and
the interval is centred on it.  Weighting matters here because a prompt
contributes a pair only on the seeds where the method left the frozen path,
so the three seeds are not always present.

The resampling uses a fixed seed, so the interval is reproducible.

    python analysis/replay_intervals.py
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SPLITS = REPO / "resources" / "sp_cross_schedules" / "parti_spx_splits.v1.json"
SPX_TABLE = REPO / "resources" / "spx" / "perprompt_spx_{model}.tsv.gz"
NATIVE_TABLE = REPO / "resources" / "spx" / "perprompt_native_{model}.tsv.gz"
SCHEDULE = REPO / "resources" / "sp_cross_schedules" / "{model}_k{k}_{gate}_top1.txt"

MODELS = ("flux", "qwen")
KS = ("29", "37", "41")
# Each adaptive method is replayed with its own payload.
GATES = (
    ("seacache", "reuse"),
    ("teacache", "reuse"),
    ("sencache", "reuse"),
    ("dicache", "di_two_anchor"),
)

N_BOOT = 20000
BOOT_SEED = 20260830


def heldout_prompts():
    with open(SPLITS) as fh:
        roles = json.load(fh)["roles"]
    return set(roles["validation"]) | set(roles["test"])


def load_native(model):
    """(method, K) -> {(seed, prompt): (psnr, path)}."""
    out = defaultdict(dict)
    with gzip.open(str(NATIVE_TABLE).format(model=model), "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (int(row["seed"]), int(row["prompt_idx"]))
            out[(row["method"], row["k"])][key] = (float(row["psnr"]), row["path"])
    return out


def load_fixed(model):
    """(schedule, payload, K) -> {(seed, prompt): psnr}."""
    out = defaultdict(dict)
    with gzip.open(str(SPX_TABLE).format(model=model), "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (int(row["seed"]), int(row["prompt_idx"]))
            out[(row["schedule"], row["payload"], row["k"])][key] = float(row["psnr"])
    return out


def modal_bits(model, k, gate):
    return Path(str(SCHEDULE).format(model=model, k=k, gate=gate)).read_text().strip()


def interval(per_prompt, rng):
    """Percentile interval of the mean, resampling prompts.

    `per_prompt` maps a prompt to its seed average and its seed count.  Each
    resample draws prompts with replacement and averages their values with
    those counts as weights.
    """
    rows = sorted(per_prompt.items())
    values = np.asarray([v for _, (v, _) in rows], dtype=float)
    counts = np.asarray([c for _, (_, c) in rows], dtype=float)
    n = values.size
    draws = rng.integers(0, n, size=(N_BOOT, n))
    weights = counts[draws]
    means = (values[draws] * weights).sum(axis=1) / weights.sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the values as JSON as well as printing them")
    args = ap.parse_args()

    held = heldout_prompts()
    rng = np.random.default_rng(BOOT_SEED)
    records = []

    print(f"{'model':<6}{'method':<10}{'K':>4}{'prompts':>9}{'pairs':>8}"
          f"{'pair mean':>11}{'95% interval':>22}")
    for model in MODELS:
        native = load_native(model)
        fixed = load_fixed(model)
        for k in KS:
            for gate, payload in GATES:
                bits = modal_bits(model, k, gate)
                own = native[(gate, k)]
                frozen = fixed[(f"{gate}_top1", payload, k)]
                diffs = defaultdict(list)
                pairs = 0
                for key, (psnr, path) in own.items():
                    if key[1] not in held or key not in frozen or path == bits:
                        continue
                    diffs[key[1]].append(frozen[key] - psnr)
                    pairs += 1
                per_prompt = {p: (sum(v) / len(v), len(v)) for p, v in diffs.items()}
                mean = sum(sum(v) for v in diffs.values()) / pairs
                lo, hi = interval(per_prompt, rng)
                records.append({
                    "model": model, "method": gate, "k": int(k),
                    "n_prompts": len(per_prompt), "n_pairs": pairs,
                    "mean": mean, "ci_low": lo, "ci_high": hi,
                })
                print(f"{model:<6}{gate:<10}{k:>4}{len(per_prompt):>9}{pairs:>8}"
                      f"{mean:>11.3f}"
                      f"{'[' + format(lo, '+.2f') + ', ' + format(hi, '+.2f') + ']':>22}")
    if args.out:
        args.out.write_text(json.dumps(
            {"n_bootstrap": N_BOOT, "seed": BOOT_SEED, "rows": records}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
