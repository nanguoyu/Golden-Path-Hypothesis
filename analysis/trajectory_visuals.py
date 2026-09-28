#!/usr/bin/env python3
"""Additional CSV/JSON visualizations for trajectory-deviation audits.

These plots use only aggregate artifacts already produced by
`analysis/trajectory_deviation.py`, `analysis/trajectory_deviation_stats.py`,
and `evaluation/eval_trajectory_perceptual.py`. They deliberately avoid
latent/velocity tensor visualizations, which require `trace_tensors.pt` and live
in `analysis/trajectory_trace_visuals.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PLOT_CACHE = Path(tempfile.gettempdir()) / "trajectory-visuals-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEATMAP_METRICS = [
    ("latent_drift_pre_rel", "latent drift / full latent"),
    ("action_defect_rel_to_full", "cache velocity error / full velocity"),
    ("state_gap_rel", "state-induced velocity gap / full velocity"),
    ("output_drift_rel", "total velocity difference / full velocity"),
]
PROXY_METRICS = [
    "trajectory_auc_z",
    "trajectory_auc_z_rel_time",
    "total_action_exposure",
    "total_state_gap_exposure",
    "total_action_latent_exposure",
    "cache_rate",
    "max_D_z_pre",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extra trajectory-deviation visualizations.")
    p.add_argument("--acc", type=Path, required=True,
                   help="Trajectory audit dir containing aggregate CSV/JSON outputs.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Default: <acc>/plots.")
    p.add_argument("--max_prompt_profiles", type=int, default=5,
                   help="Number of prompt profile panels to draw.")
    return p.parse_args()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _coerce_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":
            return 1.0
        if low == "false":
            return 0.0
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(v)


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
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if prod > 0:
                conc += 1
            elif prod < 0:
                disc += 1
    denom = conc + disc
    return None if denom == 0 else float((conc - disc) / denom)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def _prompt_order(summaries: List[Dict[str, Any]]) -> List[int]:
    return [
        int(s["prompt_id"])
        for s in sorted(
            summaries,
            key=lambda s: _coerce_float(s.get("final_latent_drift")) or -math.inf,
            reverse=True,
        )
    ]


def _plot_multi_heatmaps(
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    prompt_order = _prompt_order(summaries)
    steps = sorted({int(r["step_index"]) for r in rows})
    pid_to_row = {pid: i for i, pid in enumerate(prompt_order)}
    step_to_col = {step: i for i, step in enumerate(steps)}
    cache = np.zeros((len(prompt_order), len(steps)), dtype=float)
    mats: Dict[str, np.ndarray] = {
        metric: np.full((len(prompt_order), len(steps)), np.nan, dtype=float)
        for metric, _ in HEATMAP_METRICS
    }
    for row in rows:
        pid = int(row["prompt_id"])
        if pid not in pid_to_row:
            continue
        i = pid_to_row[pid]
        j = step_to_col[int(row["step_index"])]
        cache[i, j] = 1.0 if _as_bool(row.get("is_cached")) else 0.0
        for metric, _ in HEATMAP_METRICS:
            val = _coerce_float(row.get(metric))
            if val is not None:
                mats[metric][i, j] = val

    fig, axes = plt.subplots(2, 2, figsize=(14, max(7, len(prompt_order) * 0.11)))
    for ax, (metric, title) in zip(axes.ravel(), HEATMAP_METRICS):
        mat = mats[metric]
        finite = mat[np.isfinite(mat)]
        vmax = float(np.percentile(finite, 98)) if finite.size else None
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
        if cache.shape[0] >= 2 and cache.shape[1] >= 2 and np.nanmin(cache) < 0.5 < np.nanmax(cache):
            ax.contour(cache, levels=[0.5], colors="white", linewidths=0.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("prompts by final drift")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Prompt x step metric heatmaps; white contours mark cache/full boundaries")
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_step_metric_heatmaps.png", dpi=180)
    plt.close(fig)


def _by_step(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(row["step_index"])].append(row)
    return dict(sorted(out.items()))


def _plot_defect_glyph(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    grouped = _by_step(rows)
    steps = sorted(grouped)
    action = [_mean(_coerce_float(r.get("action_defect_rel_to_full")) for r in grouped[s]) for s in steps]
    state = [_mean(_coerce_float(r.get("state_gap_rel")) for r in grouped[s]) for s in steps]
    total = [_mean(_coerce_float(r.get("output_drift_rel")) for r in grouped[s]) for s in steps]
    cos = [_mean(_coerce_float(r.get("cos_action_gap")) for r in grouped[s]) for s in steps]
    proj = [_mean(_coerce_float(r.get("projection_action_on_total")) for r in grouped[s]) for s in steps]
    cache = [sum(1 for r in grouped[s] if _as_bool(r.get("is_cached"))) / len(grouped[s]) for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(steps, action, label="||cache velocity error|| / ||full velocity||", linewidth=1.8)
    axes[0].plot(steps, state, label="||state-induced velocity gap|| / ||full velocity||", linewidth=1.8)
    axes[0].plot(steps, total, label="||total velocity difference|| / ||full velocity||", linewidth=2.2)
    axes[0].set_ylabel("relative norm")
    axes[0].set_title("Velocity decomposition: norms are not stacked")
    axes[0].legend(loc="best")

    axes[1].plot(steps, cos, color="tab:purple", label="mean cos(action, state gap)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("cosine")
    axes[1].legend(loc="best")

    axes[2].plot(steps, proj, color="tab:orange", label="projection action on total")
    axes[2].plot(steps, cache, color="black", linestyle="--", alpha=0.7, label="cache rate")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("projection / rate")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_decomposition_glyph.png", dpi=180)
    legacy = out_dir / ("defect_" + "decomposition_glyph.png")
    if legacy.is_file():
        legacy.unlink()
    plt.close(fig)


def _perceptual_rows(acc: Path) -> Dict[int, Dict[str, Any]]:
    data = _read_json(acc / "trajectory_perceptual_metrics.json")
    if not data:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in data.get("per_image", []):
        try:
            out[int(row["prompt_id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _plot_terminal_panel(
    acc: Path,
    summaries: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    percept = _perceptual_rows(acc)
    joined: List[Dict[str, Any]] = []
    for row in summaries:
        rec = dict(row)
        rec.update(percept.get(int(row["prompt_id"]), {}))
        joined.append(rec)

    target_defs = [
        ("final_latent_drift", "final latent drift", 1.0),
        ("lpips", "LPIPS", 1.0),
        ("psnr", "PSNR degradation (-PSNR)", -1.0),
        ("ssim", "SSIM degradation (1-SSIM)", -1.0),
    ]
    corr_rows: List[Dict[str, Any]] = []
    for proxy in PROXY_METRICS:
        for target, target_label, sign in target_defs:
            xs: List[float] = []
            ys: List[float] = []
            for row in joined:
                x = _coerce_float(row.get(proxy))
                y = _coerce_float(row.get(target))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(sign * y)
            corr_rows.append({
                "proxy": proxy,
                "target": target_label,
                "n": len(xs),
                "spearman": _spearman(xs, ys),
                "kendall": _kendall(xs, ys),
            })
    _write_csv(acc / "trajectory_visual_correlations.csv", corr_rows)

    targets_present = [label for _, label, _ in target_defs if any(r["target"] == label and r["spearman"] is not None for r in corr_rows)]
    if not targets_present:
        return
    labels = PROXY_METRICS
    x = np.arange(len(labels))
    width = 0.8 / len(targets_present)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, target_label in enumerate(targets_present):
        vals = []
        for proxy in labels:
            rec = next((r for r in corr_rows if r["proxy"] == proxy and r["target"] == target_label), None)
            vals.append(0.0 if rec is None or rec["spearman"] is None else float(rec["spearman"]))
        ax.bar(x - 0.4 + width / 2 + i * width, vals, width=width, label=target_label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Terminal correlations; observed closed-loop associations only")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "terminal_perceptual_correlation_panel.png", dpi=180)
    plt.close(fig)


def _select_prompt_ids(summaries: List[Dict[str, Any]], n: int) -> List[int]:
    ordered = _prompt_order(summaries)
    if not ordered:
        return []
    candidates = [ordered[0], ordered[len(ordered) // 4], ordered[len(ordered) // 2],
                  ordered[(3 * len(ordered)) // 4], ordered[-1]]
    out: List[int] = []
    for pid in candidates:
        if pid not in out:
            out.append(pid)
        if len(out) >= n:
            break
    return out


def _plot_prompt_profiles(
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    out_dir: Path,
    max_profiles: int,
) -> None:
    by_prompt: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    summary_by_prompt = {int(s["prompt_id"]): s for s in summaries}
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(row)
    selected = _select_prompt_ids(summaries, max_profiles)
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.8 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, pid in zip(axes, selected):
        prs = sorted(by_prompt[pid], key=lambda r: int(r["step_index"]))
        steps = [int(r["step_index"]) for r in prs]
        ax.plot(steps, [_coerce_float(r.get("latent_drift_pre_rel")) for r in prs],
                label="latent drift pre rel", linewidth=1.8)
        ax.plot(steps, [_coerce_float(r.get("latent_drift_post_rel")) for r in prs],
                label="latent drift post rel", linewidth=1.4, linestyle=":")
        ax.plot(steps, [_coerce_float(r.get("action_defect_rel_to_full")) for r in prs],
                label="cache velocity error / full velocity", linewidth=1.4)
        ax.plot(steps, [_coerce_float(r.get("state_gap_rel")) for r in prs],
                label="state gap rel", linewidth=1.4)
        cached_steps = [s for s, r in zip(steps, prs) if _as_bool(r.get("is_cached"))]
        if cached_steps:
            ymax = ax.get_ylim()[1]
            ax.scatter(cached_steps, [0.0] * len(cached_steps), marker="|", s=80,
                       color="black", label="cached step")
            ax.set_ylim(top=ymax)
        summ = summary_by_prompt.get(pid, {})
        final_drift = _coerce_float(summ.get("final_latent_drift"))
        ax.set_title(f"prompt {pid}; final latent drift={final_drift:.3f}" if final_drift is not None else f"prompt {pid}")
        ax.set_ylabel("relative metric")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Selected prompt trajectories by final-drift rank")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_prompt_profiles.png", dpi=180)
    plt.close(fig)


def _plot_fixed_effect_panel(acc: Path, out_dir: Path) -> None:
    stats = _read_json(acc / "trajectory_deviation_stats_checks.json")
    if not stats:
        return
    rows = stats.get("fixed_effects") or []
    if not rows:
        return
    labels = [r["metric"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for offset, key, label in [
        (-0.25, "step_fixed_r2", "step fixed effect"),
        (0.0, "prompt_fixed_r2", "prompt fixed effect"),
        (0.25, "prompt_plus_step_fixed_r2", "prompt + step"),
    ]:
        vals = [0.0 if r.get(key) is None else float(r[key]) for r in rows]
        ax.bar(x + offset, vals, width=0.24, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("R2")
    ax.set_title("Fixed-effect diagnostics for visual claims")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fixed_effect_r2_panel.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.acc / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.acc / "trajectory_deviation_steps.csv")
    summaries = _read_csv(args.acc / "trajectory_deviation_prompt_summary.csv")
    _plot_multi_heatmaps(rows, summaries, out_dir)
    _plot_defect_glyph(rows, out_dir)
    _plot_terminal_panel(args.acc, summaries, out_dir)
    _plot_prompt_profiles(rows, summaries, out_dir, max(1, args.max_prompt_profiles))
    _plot_fixed_effect_panel(args.acc, out_dir)
    print(f"[OK] wrote extra trajectory visualizations to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
