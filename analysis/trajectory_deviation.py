#!/usr/bin/env python3
"""Aggregate and visualize FLUX trajectory-deviation audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PLOT_CACHE = Path(tempfile.gettempdir()) / "trajectory-deviation-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_METRICS = [
    "latent_drift_pre",
    "latent_drift_post",
    "output_drift_rel",
    "action_defect_rel_to_full",
    "state_gap_rel",
]
PROMPT_METRICS = [
    "trajectory_auc_z",
    "trajectory_auc_z_rel_time",
    "total_action_exposure",
    "total_state_gap_exposure",
    "total_action_latent_exposure",
    "cache_rate",
    "max_D_z_pre",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate trajectory deviation audit outputs.")
    p.add_argument("--acc", type=Path, required=True,
                   help="Trajectory deviation output dir containing prompt_*/trajectory_rows.jsonl.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output dir for aggregate CSV/JSON/plots (default: --acc/plots + root CSVs).")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rankdata(xs: List[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return float((concordant - discordant) / denom)


def _mean_std(vals: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"mean": None, "std": None, "median": None}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def _load(acc: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for prompt_dir in sorted(acc.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue
        summary_path = prompt_dir / "trajectory_summary.json"
        rows_path = prompt_dir / "trajectory_rows.jsonl"
        if not summary_path.is_file() or not rows_path.is_file():
            print(f"[WARN] skip incomplete {prompt_dir.name}")
            continue
        try:
            summary = _read_json(summary_path)
            rows = _read_jsonl(rows_path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] skip unreadable {prompt_dir.name}: {e}")
            continue
        summaries.append(summary)
        step_rows.extend(rows)
    return step_rows, summaries


def _per_step_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step_index"]), []).append(row)
    out = []
    for step, step_rows in sorted(by_step.items()):
        rec: Dict[str, Any] = {
            "step_index": step,
            "n": len(step_rows),
            "cache_rate": float(sum(1 for r in step_rows if r.get("is_cached")) / len(step_rows)),
        }
        for metric in STEP_METRICS:
            stats = _mean_std(_coerce_float(r.get(metric)) for r in step_rows)
            for k, v in stats.items():
                rec[f"{metric}_{k}"] = v
        out.append(rec)
    return out


def _correlations(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target = "final_latent_drift"
    out = []
    for metric in PROMPT_METRICS:
        xs: List[float] = []
        ys: List[float] = []
        for row in summaries:
            x = _coerce_float(row.get(metric))
            y = _coerce_float(row.get(target))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        out.append({
            "metric": metric,
            "target": target,
            "n": len(xs),
            "spearman": _spearman(xs, ys),
            "kendall": _kendall(xs, ys),
        })
    return out


def _plot_curves(per_step: List[Dict[str, Any]], out_dir: Path) -> None:
    if not per_step:
        return
    steps = [int(r["step_index"]) for r in per_step]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, [r.get("latent_drift_pre_mean") for r in per_step],
             label="latent_drift_pre", linewidth=2)
    ax1.plot(steps, [r.get("action_defect_rel_to_full_mean") for r in per_step],
             label="action_defect_rel_to_full", linewidth=1.5)
    ax1.plot(steps, [r.get("state_gap_rel_mean") for r in per_step],
             label="state_gap_rel", linewidth=1.5)
    ax1.set_xlabel("step")
    ax1.set_ylabel("mean metric")
    ax2 = ax1.twinx()
    ax2.plot(steps, [r.get("cache_rate") for r in per_step], color="black",
             linestyle="--", alpha=0.5, label="cache_rate")
    ax2.set_ylabel("cache rate")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_curves.png", dpi=180)
    plt.close(fig)


def _plot_heatmap(rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]], out_dir: Path) -> None:
    if not rows or not summaries:
        return
    prompt_order = [int(s["prompt_id"]) for s in sorted(
        summaries, key=lambda s: _coerce_float(s.get("final_latent_drift")) or 0.0, reverse=True)]
    steps = sorted({int(r["step_index"]) for r in rows})
    step_to_col = {s: i for i, s in enumerate(steps)}
    pid_to_row = {pid: i for i, pid in enumerate(prompt_order)}
    mat = np.full((len(prompt_order), len(steps)), np.nan, dtype=float)
    cache = np.zeros_like(mat)
    for r in rows:
        pid = int(r["prompt_id"])
        if pid not in pid_to_row:
            continue
        mat[pid_to_row[pid], step_to_col[int(r["step_index"])]] = (
            _coerce_float(r.get("latent_drift_pre")) or np.nan
        )
        cache[pid_to_row[pid], step_to_col[int(r["step_index"])]] = 1.0 if r.get("is_cached") else 0.0
    fig, ax = plt.subplots(figsize=(12, max(3, len(prompt_order) * 0.18)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    if cache.shape[0] >= 2 and cache.shape[1] >= 2 and np.nanmin(cache) < 0.5 < np.nanmax(cache):
        ax.contour(cache, levels=[0.5], colors="white", linewidths=0.25)
    ax.set_xlabel("step")
    ax.set_ylabel("prompts sorted by final drift")
    ax.set_title("latent_drift_pre heatmap; white contours mark cached decisions")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_step_latent_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_correlations(corrs: List[Dict[str, Any]], out_dir: Path) -> None:
    if not corrs:
        return
    labels = [r["metric"] for r in corrs]
    vals = [r["spearman"] if r["spearman"] is not None else 0.0 for r in corrs]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), vals)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Spearman vs final_latent_drift")
    ax.set_title("Observed closed-loop correlations (not causal)")
    fig.tight_layout()
    fig.savefig(out_dir / "terminal_correlation.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.acc
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries = _load(args.acc)
    if not rows or not summaries:
        raise SystemExit(f"no complete trajectory audit outputs under {args.acc}")

    per_step = _per_step_summary(rows)
    corrs = _correlations(summaries)
    _write_csv(out_dir / "trajectory_deviation_steps.csv", rows)
    _write_csv(out_dir / "trajectory_deviation_prompt_summary.csv", summaries)
    _write_csv(out_dir / "trajectory_deviation_per_step.csv", per_step)
    _write_csv(out_dir / "trajectory_deviation_correlations.csv", corrs)

    summary = {
        "input_dir": str(args.acc),
        "n_step_rows": len(rows),
        "n_prompts": len(summaries),
        "n_steps": len(per_step),
        "per_step": per_step,
        "correlations": corrs,
    }
    (out_dir / "trajectory_deviation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_curves(per_step, plot_dir)
    _plot_heatmap(rows, summaries, plot_dir)
    _plot_correlations(corrs, plot_dir)
    print(f"[OK] aggregated {len(summaries)} prompts, {len(rows)} step rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
