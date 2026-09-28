"""W&B logging for diffusion caching (gen + eval to the SAME run).

Design
------
Both the generation phase and the evaluation phase write to one wandb run,
keyed by `run_id = output_dir.name` (the user already encodes method + config
into the directory name, e.g. `teacache_t0.3_flux_s42_50`). We use
`wandb.init(id=run_id, resume="allow")` so:

  * first call (gen)  -> creates the run, logs 3 sample images + timing summary
  * second call (eval) -> resumes the same run, attaches eval metrics

This module is callable two ways:

  * As a library, from `evaluation/eval_metrics.py`:
        from lib.wandb_logger import log_eval_results
        log_eval_results(metrics_path, ...)

  * As a CLI, from shell launchers (after-merge in multi_gpu_flux.sh /
    after Slurm array completes for Slurm):
        python -m lib.wandb_logger gen  --run_dir DIR [--prompt_file F]
        python -m lib.wandb_logger eval --metrics PATH

Step semantics: eval writes only to `wandb.summary["eval/<metric>/mean"]`
(and friends), which is index-less, so it does NOT collide with whatever
step counter the gen phase used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROJECT = "gph-baselines"
DEFAULT_SAMPLE_INDICES: tuple[int, ...] = (0, 100, 199)

_TIMING_CONFIG_KEYS = (
    "cache_mode", "mode_raw", "num_steps", "interval", "max_order",
    "hicache_sigma", "seacache_thresh", "teacache_thresh",
    "foca_official_reproduction", "foca_target", "foca_heun_variant",
    "foca_history_policy", "foca_derivative", "foca_h", "foca_log_norms",
    "teacache_backbone", "teacache_variant", "first_enhance",
    "model_id", "model_name", "guidance", "width", "height",
    "dtype", "base_seed", "git_sha", "n_images", "n_shards", "batch_size",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def derive_run_id(run_dir: Path) -> str:
    """The output dir basename. User's naming convention is already unique
    per (method, op-point, seed, steps), so we re-use it as wandb run id."""
    return run_dir.name


def _require_wandb():
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[wandb_logger] `wandb` not installed. "
            "Run `pip install -e \".[eval]\"` (which now includes wandb) "
            "or `pip install wandb`."
        ) from exc
    import wandb
    return wandb


def _find_image(run_dir: Path, idx: int) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = run_dir / f"img_{idx}{ext}"
        if p.is_file():
            return p
    return None


def _read_prompts(prompt_file: Path | None) -> list[str] | None:
    if prompt_file is None or not prompt_file.is_file():
        return None
    return [ln.strip() for ln in prompt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _init_run(
    *,
    project: str,
    entity: str | None,
    run_id: str,
    job_type: str,
    config: dict[str, Any],
    tags: Iterable[str] | None = None,
):
    wandb = _require_wandb()
    return wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        name=run_id,
        resume="allow",
        job_type=job_type,
        config=config,
        tags=list(tags) if tags else None,
        # console="off" stops wandb from capturing stdout — keeps slurm logs clean.
        # Default system-stat collection (GPU util, mem, etc.) is kept ON.
        settings=wandb.Settings(console="off"),
    )


# ---------------------------------------------------------------------------
# generation upload
# ---------------------------------------------------------------------------
def log_generation(
    run_dir: Path,
    *,
    prompt_file: Path | None = None,
    sample_indices: Iterable[int] = DEFAULT_SAMPLE_INDICES,
    project: str = DEFAULT_PROJECT,
    entity: str | None = None,
    run_id: str | None = None,
) -> str:
    """Upload gen results from a finished (and merged) run dir.

    Expects `run_dir/timing.json` (written by multi_gpu_flux.sh's merge
    block or by RUN/merge_shard_timings.py).

    Logged:
      * 3 sample images at the given indices, with prompt as caption
      * timing summary (latency mean/std, throughput, wallclock) under
        `wandb.summary["gen/..."]`
      * config dict from timing.json (cache_mode, num_steps, etc.)

    Returns the wandb run id (`run_id` if given, else `run_dir.name`). Pass
    `run_id` when the directory basename is not unique across runs (SPX cells
    repeat `<schedule>x<payload>_s<seed>` under every model/K); the eval side
    takes the same override via `eval_metrics.py --wandb_run_id`.
    """
    timing_path = run_dir / "timing.json"
    if not timing_path.is_file():
        raise FileNotFoundError(
            f"[wandb_logger] no {timing_path}. "
            f"Run the merge step first (multi_gpu_flux.sh does this "
            f"automatically; on Slurm: `python RUN/merge_shard_timings.py {run_dir} <N>`)."
        )
    timing = json.loads(timing_path.read_text(encoding="utf-8"))

    run_id = run_id or derive_run_id(run_dir)
    config = {k: timing.get(k) for k in _TIMING_CONFIG_KEYS}
    config["output_dir"] = str(run_dir)

    tags = [str(timing.get("mode_raw") or "unknown"), str(timing.get("model_name") or "unknown")]

    wandb = _require_wandb()
    _init_run(
        project=project, entity=entity, run_id=run_id,
        job_type="generate", config=config, tags=tags,
    )

    # ---- sample images
    prompts = _read_prompts(prompt_file)
    images_to_log: dict[str, Any] = {}
    for idx in sample_indices:
        p = _find_image(run_dir, idx)
        if p is None:
            print(f"[wandb_logger] sample idx={idx} not found in {run_dir}; skipping",
                  file=sys.stderr)
            continue
        caption = prompts[idx] if (prompts is not None and idx < len(prompts)) else f"idx={idx}"
        images_to_log[f"samples/img_{idx}"] = wandb.Image(str(p), caption=caption)
    if images_to_log:
        wandb.log(images_to_log)

    # ---- timing summary (index-less)
    lat = timing.get("latency_per_image_s")
    if isinstance(lat, dict):
        for k in ("mean", "std", "min", "max"):
            v = lat.get(k)
            if v is not None:
                wandb.summary[f"gen/latency_s/{k}"] = float(v)
    elif isinstance(lat, (int, float)):
        wandb.summary["gen/latency_s/mean"] = float(lat)

    if "throughput_img_per_s" in timing and timing["throughput_img_per_s"]:
        wandb.summary["gen/throughput_img_s"] = float(timing["throughput_img_per_s"])

    wc = timing.get("wallclock_total_s")
    if isinstance(wc, dict):
        if wc.get("per_shard_max") is not None:
            wandb.summary["gen/wallclock_s_max"] = float(wc["per_shard_max"])
    elif isinstance(wc, (int, float)):
        wandb.summary["gen/wallclock_s_max"] = float(wc)

    if "n_images" in timing:
        wandb.summary["gen/n_images"] = int(timing["n_images"])
    if "n_shards" in timing:
        wandb.summary["gen/n_shards"] = int(timing["n_shards"])

    wandb.finish()
    print(f"[wandb_logger] gen logged to run_id={run_id} ({len(images_to_log)} samples)")
    return run_id


# ---------------------------------------------------------------------------
# eval upload
# ---------------------------------------------------------------------------
def log_eval_results(
    metrics_path: Path,
    *,
    run_id: str | None = None,
    project: str = DEFAULT_PROJECT,
    entity: str | None = None,
) -> str:
    """Upload eval metrics summary, resuming the gen run.

    ``metrics_path`` is the JSON produced by ``evaluation/eval_metrics.py``.
    ``run_id`` defaults to ``basename(payload['acc'])`` so gen and eval line up
    automatically (gen was keyed off ``output_dir.name``; eval's acc dir == gen's
    output dir).

    Writes only ``wandb.summary["eval/..."]`` — no ``wandb.log()`` steps — so
    this is safe to call after gen without step-counter collisions.
    """
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) or {}
    timing_block = payload.get("timing", {}) or {}

    if run_id is None:
        acc = payload.get("acc")
        if not acc:
            raise ValueError(
                f"[wandb_logger] no 'acc' field in {metrics_path}; pass --run_id explicitly"
            )
        run_id = Path(acc).name

    config = {
        "eval_metrics": sorted(summary.keys()),
        "n_pairs": payload.get("n_pairs"),
        "metrics_path": str(metrics_path),
    }
    gt = payload.get("gt")
    if gt:
        config["gt_dir"] = gt

    wandb = _require_wandb()
    _init_run(
        project=project, entity=entity, run_id=run_id,
        job_type="evaluate", config=config,
    )

    for k, v in summary.items():
        if not isinstance(v, dict):
            continue
        for stat in ("mean", "std", "min", "max"):
            if v.get(stat) is not None:
                wandb.summary[f"eval/{k}/{stat}"] = float(v[stat])
        if v.get("n") is not None:
            wandb.summary[f"eval/{k}/n"] = int(v["n"])

    wc = timing_block.get("wallclock_speedup_vs_gt")
    if wc is not None:
        wandb.summary["eval/speedup_vs_gt"] = float(wc)

    # convenient cross-table: gen latency from the same record (if gen was logged earlier).
    acc_timing = timing_block.get("acc") or {}
    acc_lat = acc_timing.get("latency_per_image_s")
    if isinstance(acc_lat, dict) and acc_lat.get("mean") is not None:
        wandb.summary["eval/acc_latency_s/mean"] = float(acc_lat["mean"])

    wandb.finish()
    print(f"[wandb_logger] eval logged to run_id={run_id} ({len(summary)} metrics)")
    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="W&B logger for diffusion caching. Generation and eval write "
                    "to the same run via resume='allow', keyed by output dir basename."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Upload generation timing + 3 sample images.")
    g.add_argument("--run_dir", type=Path, required=True,
                   help="Output directory of the generation run (contains timing.json + img_*.png).")
    g.add_argument("--prompt_file", type=Path, default=None,
                   help="Prompt file for image captions (optional but useful).")
    g.add_argument("--sample_indices", type=int, nargs="+", default=list(DEFAULT_SAMPLE_INDICES),
                   help=f"Which image indices to upload. Default: {list(DEFAULT_SAMPLE_INDICES)}.")
    g.add_argument("--run_id", default=None,
                   help="Override run_id (default: basename of --run_dir). Needed "
                        "when the basename is not unique across runs.")
    g.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    g.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))

    e = sub.add_parser("eval", help="Upload eval metrics summary (resumes gen run).")
    e.add_argument("--metrics", type=Path, required=True,
                   help="Path to metrics.json produced by evaluation/eval_metrics.py.")
    e.add_argument("--run_id", default=None,
                   help="Override run_id (default: basename of metrics.acc dir).")
    e.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    e.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))

    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "gen":
        log_generation(
            args.run_dir,
            prompt_file=args.prompt_file,
            sample_indices=tuple(args.sample_indices),
            project=args.project,
            entity=args.entity,
            run_id=args.run_id,
        )
    elif args.cmd == "eval":
        log_eval_results(
            args.metrics,
            run_id=args.run_id,
            project=args.project,
            entity=args.entity,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
