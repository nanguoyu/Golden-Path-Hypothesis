"""Single-pair evaluator: ImageReward, CLIP, PSNR, SSIM, LPIPS.

Compares an accelerated image directory against a ground-truth (baseline)
directory, both produced by ``RUN/multi_gpu_launcher.sh`` (image files named
``img_<idx>.jpg|png``). Pairs are matched by index; prompts are read line by
line from ``--prompts`` with line ``i`` corresponding to ``img_<i>.*``.

PSNR, SSIM, LPIPS need both --acc and --gt.
ImageReward and CLIP only need --acc + --prompts.

Outputs a JSON file (default: <acc>/metrics.json) with per-image scores and
mean/std summary, and prints the summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


METRIC_CHOICES = ["image_reward", "psnr", "ssim", "lpips", "clip"]
PAIRWISE_METRICS = {"psnr", "ssim", "lpips"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HiCache single-pair eval (5 metrics)")
    p.add_argument("--acc", required=True, type=Path, help="Accelerated image dir (img_*.jpg|png)")
    p.add_argument("--gt", type=Path, default=None,
                   help="Baseline image dir; required for psnr/ssim/lpips")
    p.add_argument("--prompts", required=True, type=Path,
                   help="Prompt file, one prompt per line (line i -> img_<i>)")
    p.add_argument("--output", type=Path, default=None,
                   help="JSON output path (default: <acc>/metrics.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of (idx, prompt) entries to evaluate")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="Evaluate only entries where idx %% shard_count == shard_idx.")
    p.add_argument("--shard_count", type=int, default=1,
                   help="Number of eval shards. Default 1 keeps legacy behavior.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--metrics", nargs="+", default=METRIC_CHOICES, choices=METRIC_CHOICES,
                   help="Which metrics to compute (default: all)")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    # ---- wandb (optional) ----
    p.add_argument("--wandb", action="store_true",
                   help="Upload eval summary to W&B, resuming the gen run "
                        "keyed by basename of --acc.")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project (default: env WANDB_PROJECT or gph-baselines).")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity / team (default: env WANDB_ENTITY).")
    p.add_argument("--wandb_run_id", default=None,
                   help="Override W&B run id (default: basename of --acc).")
    return p.parse_args()


def _find_image(d: Path, idx: int) -> Path | None:
    """Find image idx under d. Supports two layouts:
    - flat (paper-fidelity runners): `img_<idx>.{jpg,jpeg,png}`
    - nested per-prompt (SM-D state_gate_runner): `prompt_<05d-idx>/image.{png,jpg,jpeg}`
    Indexing is the same — line i of --prompts pairs with idx i in either layout.
    """
    for ext in (".jpg", ".jpeg", ".png"):
        p = d / f"img_{idx}{ext}"
        if p.is_file():
            return p
    for ext in (".png", ".jpg", ".jpeg"):
        p = d / f"prompt_{idx:05d}" / f"image{ext}"
        if p.is_file():
            return p
    return None


def collect_entries(
    acc_dir: Path,
    gt_dir: Path | None,
    prompts: list[str],
    limit: int | None,
    need_gt: bool,
    shard_idx: int = 0,
    shard_count: int = 1,
) -> list[tuple[int, Path, Path | None, str]]:
    n = len(prompts) if limit is None else min(limit, len(prompts))
    out: list[tuple[int, Path, Path | None, str]] = []
    missing_acc = missing_gt = 0
    for idx in range(n):
        if shard_count > 1 and (idx % shard_count) != shard_idx:
            continue
        acc_p = _find_image(acc_dir, idx)
        if acc_p is None:
            missing_acc += 1
            continue
        gt_p = _find_image(gt_dir, idx) if gt_dir is not None else None
        if need_gt and gt_p is None:
            missing_gt += 1
            continue
        out.append((idx, acc_p, gt_p, prompts[idx]))
    if missing_acc:
        print(f"[WARN] {missing_acc} acc images missing under {acc_dir}", file=sys.stderr)
    if missing_gt:
        print(f"[WARN] {missing_gt} gt images missing under {gt_dir}", file=sys.stderr)
    return out


def _to_chw_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def _to_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------------------
# model loading and single-pair scoring
#
# The batch functions below are loops over these, and so is the resident
# scoring of `lib/search_metrics.py`: a caller that keeps a model in memory
# (the schedule search keeps one next to the diffusion pipeline) loads it once
# here and scores one pair at a time, with the same arithmetic the matrix runs.
# ---------------------------------------------------------------------------


def score_psnr(gt: np.ndarray, acc: np.ndarray) -> float:
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(gt, acc, data_range=255))


def score_ssim(gt: np.ndarray, acc: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt, acc, data_range=255, channel_axis=-1))


def load_lpips_model(device: str, net: str = "alex"):
    import lpips
    return lpips.LPIPS(net=net).to(device).eval()


def score_lpips(model, acc: Image.Image, gt: Image.Image, device: str) -> float:
    with torch.no_grad():
        a = _to_chw_tensor(acc, device)
        g = _to_chw_tensor(gt, device)
        return float(model(a, g).squeeze().item())


def load_image_reward_model(device: str, model_name: str = "ImageReward-v1.0"):
    import ImageReward as RM
    return RM.load(model_name, device=device)


def score_image_reward(model, prompt: str, image: Image.Image) -> float:
    """One ImageReward score for an already-open image.

    The image is handed over as a `PIL.Image` so that the upstream's
    `os.path.isfile(image)` branch (which has been seen to misfire on some
    shared-filesystem paths) is bypassed entirely. The other branch —
    `isinstance(image, PIL.Image.Image)` — accepts subclasses like
    `PngImageFile` returned by `Image.open(...)`, so this is safe and
    equivalent.
    """
    return float(model.score(prompt, image.convert("RGB")))


def load_clip_model(device: str, model_name: str = "openai/clip-vit-large-patch14"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    return model, proc


def score_clip(model, proc, image: Image.Image, prompt: str, device: str) -> float:
    with torch.no_grad():
        inputs = proc(
            text=[prompt],
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        outputs = model(**inputs)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum(dim=-1).item()) * 100.0


def compute_psnr_ssim(entries, want_psnr: bool, want_ssim: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if want_psnr:
        out["psnr"] = []
    if want_ssim:
        out["ssim"] = []
    for _, acc_p, gt_p, _ in tqdm(entries, desc="PSNR/SSIM"):
        acc = _to_uint8(Image.open(acc_p))
        gt = _to_uint8(Image.open(gt_p))
        if acc.shape != gt.shape:
            raise ValueError(f"shape mismatch: acc {acc.shape} vs gt {gt.shape} at {acc_p}")
        if want_psnr:
            out["psnr"].append(score_psnr(gt, acc))
        if want_ssim:
            out["ssim"].append(score_ssim(gt, acc))
    return out


def compute_lpips(entries, device: str, net: str) -> list[float]:
    model = load_lpips_model(device, net)
    vals: list[float] = []
    for _, acc_p, gt_p, _ in tqdm(entries, desc=f"LPIPS({net})"):
        vals.append(
            score_lpips(model, Image.open(acc_p), Image.open(gt_p), device)
        )
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vals


def compute_image_reward(entries, device: str, model_name: str) -> list[float]:
    """Score acc images with ImageReward."""
    model = load_image_reward_model(device, model_name)
    vals: list[float] = []
    for _, acc_p, _, prompt in tqdm(entries, desc="ImageReward"):
        if not acc_p.is_file():
            print(f"[ImageReward] WARN missing file: {acc_p}; reporting NaN", file=sys.stderr)
            vals.append(float("nan"))
            continue
        with Image.open(acc_p) as im:
            vals.append(score_image_reward(model, prompt, im))
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vals


def compute_clip(entries, device: str, model_name: str) -> list[float]:
    model, proc = load_clip_model(device, model_name)
    vals: list[float] = []
    for _, acc_p, _, prompt in tqdm(entries, desc="CLIP"):
        vals.append(score_clip(model, proc, Image.open(acc_p), prompt, device))
    del model, proc
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vals


def load_timing(d: Path | None) -> dict | None:
    """Load timing.json (written by sample.py / merged by the launcher) if present."""
    if d is None:
        return None
    p = d / "timing.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latency_mean(timing: dict | None) -> float | None:
    if not timing:
        return None
    lat = timing.get("latency_per_image_s")
    if isinstance(lat, dict):
        v = lat.get("mean")
        return float(v) if v is not None else None
    if isinstance(lat, (int, float)):
        return float(lat)
    return None


def summarize(per_image: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k, vs in per_image.items():
        if not vs:
            summary[k] = None
            continue
        arr = np.asarray(vs, dtype=np.float64)
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(vs),
        }
    return summary


def main() -> int:
    args = parse_args()

    want = set(args.metrics)
    need_gt = bool(want & PAIRWISE_METRICS)
    if need_gt and args.gt is None:
        sys.exit("[ERROR] --gt is required for psnr/ssim/lpips. "
                 "Pass --gt or restrict --metrics to {image_reward, clip}.")

    if not args.prompts.is_file():
        sys.exit(f"[ERROR] prompts file not found: {args.prompts}")
    prompts = [ln.strip() for ln in args.prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not prompts:
        sys.exit(f"[ERROR] no prompts parsed from {args.prompts}")

    if not args.acc.is_dir():
        sys.exit(f"[ERROR] acc dir not found: {args.acc}")
    if args.gt is not None and not args.gt.is_dir():
        sys.exit(f"[ERROR] gt dir not found: {args.gt}")
    if args.shard_count < 1:
        sys.exit("[ERROR] --shard_count must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.shard_count:
        sys.exit("[ERROR] --shard_idx must satisfy 0 <= shard_idx < shard_count")

    entries = collect_entries(
        args.acc,
        args.gt,
        prompts,
        args.limit,
        need_gt,
        shard_idx=int(args.shard_idx),
        shard_count=int(args.shard_count),
    )
    if not entries:
        sys.exit("[ERROR] no eligible (acc[, gt], prompt) entries")
    print(
        f"[INFO] evaluating {len(entries)} entries; metrics = {sorted(want)}; "
        f"shard={args.shard_idx}/{args.shard_count}"
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"

    per_image: dict[str, list[float]] = {}
    if "psnr" in want or "ssim" in want:
        per_image.update(compute_psnr_ssim(entries, "psnr" in want, "ssim" in want))
    if "lpips" in want:
        per_image["lpips"] = compute_lpips(entries, device, args.lpips_net)
    if "image_reward" in want:
        per_image["image_reward"] = compute_image_reward(entries, device, args.image_reward_model)
    if "clip" in want:
        per_image["clip"] = compute_clip(entries, device, args.clip_model)

    summary = summarize(per_image)

    acc_timing = load_timing(args.acc)
    gt_timing = load_timing(args.gt)
    acc_lat = _latency_mean(acc_timing)
    gt_lat = _latency_mean(gt_timing)
    wallclock_speedup = (gt_lat / acc_lat) if (acc_lat and gt_lat and acc_lat > 0) else None
    timing_block = {
        "acc": acc_timing,
        "gt": gt_timing,
        "wallclock_speedup_vs_gt": wallclock_speedup,
    }

    print("\n=== Summary ===")
    print(f"  acc : {args.acc}")
    print(f"  gt  : {args.gt}")
    print(f"  n   : {len(entries)}")
    for k in METRIC_CHOICES:
        if k not in summary:
            continue
        v = summary[k]
        if v is None:
            print(f"  {k:13s}: N/A")
        else:
            print(f"  {k:13s}: mean={v['mean']:.4f}  std={v['std']:.4f}  "
                  f"[{v['min']:.4f}, {v['max']:.4f}]  (n={v['n']})")
    if acc_lat is not None:
        print(f"  {'latency(acc)':13s}: {acc_lat:.4f} s/img")
    if gt_lat is not None:
        print(f"  {'latency(gt)':13s}: {gt_lat:.4f} s/img")
    if wallclock_speedup is not None:
        print(f"  {'speedup':13s}: {wallclock_speedup:.3f}x  (gt / acc)")

    out_path = args.output if args.output is not None else (args.acc / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "acc": str(args.acc),
        "gt": str(args.gt) if args.gt is not None else None,
        "prompts": str(args.prompts),
        "n_pairs": len(entries),
        "indices": [int(entry[0]) for entry in entries],
        "summary": summary,
        "timing": timing_block,
        "per_image": per_image,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] wrote {out_path}")

    # ---- optional: upload to W&B, resuming the gen run by output-dir basename
    if args.wandb:
        # late import so this stays an optional dep
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from lib.wandb_logger import DEFAULT_PROJECT, log_eval_results
        project = args.wandb_project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        try:
            log_eval_results(
                out_path,
                run_id=args.wandb_run_id,
                project=project,
                entity=entity,
            )
        except Exception as exc:  # noqa: BLE001 — wandb errors shouldn't fail the eval
            print(f"[WARN] wandb upload failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
