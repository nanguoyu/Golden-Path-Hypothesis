#!/usr/bin/env python3
"""Perceptual metrics for trajectory-deviation audit outputs.

Compares each prompt's cached terminal image against the corresponding full
terminal image saved by `flux/trajectory_deviation_runner.py`:

  prompt_00000/cached.png  vs  prompt_00000/baseline.png

This script is intentionally separate from `eval_metrics.py` because the
trajectory audit has a nested layout and a different interpretation: the
ground truth is the full trajectory from the same audit run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


METRIC_CHOICES = ["psnr", "ssim", "lpips", "image_reward", "clip"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trajectory audit perceptual eval.")
    p.add_argument("--acc", type=Path, required=True,
                   help="Trajectory audit dir containing prompt_*/baseline.png and cached.png.")
    p.add_argument("--prompts", type=Path, required=True,
                   help="Prompt file; line i maps to prompt_<i:05d>.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on prompt ids to evaluate.")
    p.add_argument("--output_json", type=Path, default=None,
                   help="Default: <acc>/trajectory_perceptual_metrics.json.")
    p.add_argument("--output_csv", type=Path, default=None,
                   help="Default: <acc>/trajectory_perceptual_metrics.csv.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"],
                   choices=METRIC_CHOICES)
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    return p.parse_args()


def _load_prompts(path: Path) -> List[str]:
    if not path.is_file():
        raise SystemExit(f"[ERROR] prompts file not found: {path}")
    prompts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not prompts:
        raise SystemExit(f"[ERROR] no prompts parsed from {path}")
    return prompts


def _collect_entries(acc: Path, prompts: List[str], limit: int | None) -> List[Tuple[int, Path, Path, str]]:
    n = len(prompts) if limit is None else min(limit, len(prompts))
    out: List[Tuple[int, Path, Path, str]] = []
    missing = 0
    for idx in range(n):
        prompt_dir = acc / f"prompt_{idx:05d}"
        baseline = prompt_dir / "baseline.png"
        cached = prompt_dir / "cached.png"
        if baseline.is_file() and cached.is_file():
            out.append((idx, cached, baseline, prompts[idx]))
        else:
            missing += 1
    if missing:
        print(f"[WARN] {missing} trajectory image pairs missing under {acc}", file=sys.stderr)
    if not out:
        raise SystemExit(f"[ERROR] no eligible trajectory image pairs under {acc}")
    return out


def _to_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _to_chw_tensor(path: Path, device: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def _compute_psnr_ssim(entries: List[Tuple[int, Path, Path, str]], want_psnr: bool,
                       want_ssim: bool) -> Dict[int, Dict[str, float]]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    out: Dict[int, Dict[str, float]] = {}
    for idx, cached_p, baseline_p, _ in tqdm(entries, desc="PSNR/SSIM"):
        cached = _to_uint8(Image.open(cached_p))
        baseline = _to_uint8(Image.open(baseline_p))
        if cached.shape != baseline.shape:
            raise ValueError(f"shape mismatch for prompt {idx}: {cached.shape} vs {baseline.shape}")
        rec: Dict[str, float] = {}
        if want_psnr:
            rec["psnr"] = float(peak_signal_noise_ratio(baseline, cached, data_range=255))
        if want_ssim:
            rec["ssim"] = float(structural_similarity(baseline, cached, data_range=255, channel_axis=-1))
        out[idx] = rec
    return out


def _compute_lpips(entries: List[Tuple[int, Path, Path, str]], device: str, net: str) -> Dict[int, float]:
    import lpips
    model = lpips.LPIPS(net=net).to(device).eval()
    out: Dict[int, float] = {}
    with torch.no_grad():
        for idx, cached_p, baseline_p, _ in tqdm(entries, desc=f"LPIPS({net})"):
            cached = _to_chw_tensor(cached_p, device)
            baseline = _to_chw_tensor(baseline_p, device)
            out[idx] = float(model(cached, baseline).squeeze().item())
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _compute_image_reward(entries: List[Tuple[int, Path, Path, str]], device: str,
                          model_name: str) -> Dict[int, Dict[str, float]]:
    import ImageReward as RM
    model = RM.load(model_name, device=device)
    out: Dict[int, Dict[str, float]] = {}
    for idx, cached_p, baseline_p, prompt in tqdm(entries, desc="ImageReward"):
        with Image.open(cached_p) as cached_im, Image.open(baseline_p) as baseline_im:
            cached_score = float(model.score(prompt, cached_im.convert("RGB")))
            baseline_score = float(model.score(prompt, baseline_im.convert("RGB")))
        out[idx] = {
            "image_reward_cached": cached_score,
            "image_reward_baseline": baseline_score,
            "image_reward_delta": cached_score - baseline_score,
        }
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _compute_clip(entries: List[Tuple[int, Path, Path, str]], device: str,
                  model_name: str) -> Dict[int, Dict[str, float]]:
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    out: Dict[int, Dict[str, float]] = {}
    with torch.no_grad():
        for idx, cached_p, baseline_p, prompt in tqdm(entries, desc="CLIP"):
            images = [
                Image.open(cached_p).convert("RGB"),
                Image.open(baseline_p).convert("RGB"),
            ]
            inputs = proc(
                text=[prompt, prompt],
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            outputs = model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores = ((img_emb * txt_emb).sum(dim=-1) * 100.0).detach().cpu().tolist()
            out[idx] = {
                "clip_cached": float(scores[0]),
                "clip_baseline": float(scores[1]),
                "clip_delta": float(scores[0] - scores[1]),
            }
    del model, proc
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _summarize(per_image: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted({
        k for row in per_image for k, v in row.items()
        if k not in {"prompt_id", "cached_path", "baseline_path", "prompt"} and isinstance(v, (int, float))
    })
    summary: Dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in per_image
                if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        if not vals:
            summary[key] = None
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(vals)),
        }
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    want = set(args.metrics)
    prompts = _load_prompts(args.prompts)
    entries = _collect_entries(args.acc, prompts, args.limit)
    print(f"[INFO] evaluating {len(entries)} trajectory image pairs; metrics={sorted(want)}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"

    per_image: Dict[int, Dict[str, Any]] = {
        idx: {
            "prompt_id": idx,
            "cached_path": str(cached_p),
            "baseline_path": str(baseline_p),
            "prompt": prompt,
        }
        for idx, cached_p, baseline_p, prompt in entries
    }
    if "psnr" in want or "ssim" in want:
        for idx, rec in _compute_psnr_ssim(entries, "psnr" in want, "ssim" in want).items():
            per_image[idx].update(rec)
    if "lpips" in want:
        for idx, val in _compute_lpips(entries, device, args.lpips_net).items():
            per_image[idx]["lpips"] = val
    if "image_reward" in want:
        for idx, rec in _compute_image_reward(entries, device, args.image_reward_model).items():
            per_image[idx].update(rec)
    if "clip" in want:
        for idx, rec in _compute_clip(entries, device, args.clip_model).items():
            per_image[idx].update(rec)

    rows = [per_image[idx] for idx in sorted(per_image)]
    summary = _summarize(rows)
    out_json = args.output_json or (args.acc / "trajectory_perceptual_metrics.json")
    out_csv = args.output_csv or (args.acc / "trajectory_perceptual_metrics.csv")
    payload = {
        "acc": str(args.acc),
        "prompts": str(args.prompts),
        "n_pairs": len(rows),
        "metrics": sorted(want),
        "summary": summary,
        "per_image": rows,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_csv, rows)

    print("\n=== Trajectory Perceptual Summary ===")
    print(f"  acc : {args.acc}")
    print(f"  n   : {len(rows)}")
    for key, rec in summary.items():
        if rec is None:
            print(f"  {key:22s}: N/A")
        else:
            print(f"  {key:22s}: mean={rec['mean']:.4f} std={rec['std']:.4f} "
                  f"[{rec['min']:.4f}, {rec['max']:.4f}] (n={rec['n']})")
    print(f"\n[INFO] wrote {out_json}")
    print(f"[INFO] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
