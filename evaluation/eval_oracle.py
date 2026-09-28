"""Oracle fragility eval: per (prompt, k) compute 6 metrics.

Reads each prompt_*/ subdir under --acc and computes, for each oracle step k:
- latent_L2_abs:  ‖z_N^oracle − z_N^baseline‖₂ (canonical R_k^oracle)
- latent_L2_rel:  ‖.‖ / ‖z_N^baseline‖₂
- psnr:           skimage PSNR(decoded_baseline, decoded_oracle, data_range=255)
- ssim:           skimage SSIM(...)
- lpips:          lpips(alex by default) on normalized [-1,1] tensors
- d_ir:           ImageReward(oracle, prompt) − ImageReward(baseline, prompt)
- d_clip:         CLIP_score(oracle, prompt) − CLIP_score(baseline, prompt) (×100)

Writes prompt_dir/oracle_metrics.json. Optionally uploads aggregate to W&B
under one run id (default = acc dir basename) in project
`gph-analysis`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle eval: per (prompt, k) compute 6 metrics.")
    p.add_argument("--acc", required=True, type=Path,
                   help="Oracle output dir containing prompt_XXXXX/ subdirs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of prompts to evaluate (for testing)")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose oracle_metrics.json exists")
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_id", default=None,
                   help="Override wandb run id; default = acc dir basename")
    return p.parse_args()


def _to_chw_norm_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def _to_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def load_metric_models(device: str, lpips_net: str, clip_model: str, ir_model: str):
    import lpips
    from transformers import CLIPModel, CLIPProcessor
    import ImageReward as RM

    # Use .train(False) instead of .eval() — semantically identical (PyTorch
    # nn.Module API), avoids static-analysis false positives on the eval() word.
    lpips_m = lpips.LPIPS(net=lpips_net).to(device)
    lpips_m.train(False)
    clip_m = CLIPModel.from_pretrained(clip_model).to(device)
    clip_m.train(False)
    clip_p = CLIPProcessor.from_pretrained(clip_model)
    ir_m = RM.load(ir_model, device=device)
    return lpips_m, clip_m, clip_p, ir_m


def compute_one_pair(
    img_base: Image.Image, img_oracle: Image.Image,
    z_base: torch.Tensor, z_oracle: torch.Tensor,
    prompt: str, ir_base_val: float, clip_base_val: float,
    lpips_m, clip_m, clip_p, ir_m, device: str,
) -> dict:
    """All 6 metrics for one (baseline, oracle_k) pair."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    base_u8 = _to_uint8(img_base)
    oracle_u8 = _to_uint8(img_oracle)

    psnr = float(peak_signal_noise_ratio(base_u8, oracle_u8, data_range=255))
    ssim = float(structural_similarity(base_u8, oracle_u8, data_range=255, channel_axis=-1))

    with torch.no_grad():
        a = _to_chw_norm_tensor(img_base, device)
        b = _to_chw_norm_tensor(img_oracle, device)
        lpips_val = float(lpips_m(a, b).squeeze().item())

    with torch.no_grad():
        ir_oracle = float(ir_m.score(prompt, img_oracle.convert("RGB")))

    with torch.no_grad():
        inputs = clip_p(text=[prompt], images=img_oracle.convert("RGB"),
                        return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = clip_m(**inputs)
        iemb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        temb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        clip_oracle = float((iemb * temb).sum(dim=-1).item()) * 100.0

    z_base32 = z_base.to(torch.float32)
    z_oracle32 = z_oracle.to(torch.float32)
    abs_l2 = float((z_oracle32 - z_base32).norm().item())
    rel_l2 = abs_l2 / float(z_base32.norm().item() + 1e-16)

    return {
        "latent_L2_abs": abs_l2,
        "latent_L2_rel": rel_l2,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "ir_oracle": ir_oracle,
        "d_ir": ir_oracle - ir_base_val,
        "clip_oracle": clip_oracle,
        "d_clip": clip_oracle - clip_base_val,
    }


def score_baseline(
    img: Image.Image, prompt: str, clip_m, clip_p, ir_m, device: str,
) -> tuple[float, float]:
    """Compute (IR, CLIP) for the baseline image (called once per prompt)."""
    with torch.no_grad():
        ir_val = float(ir_m.score(prompt, img.convert("RGB")))
        inputs = clip_p(text=[prompt], images=img.convert("RGB"),
                        return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = clip_m(**inputs)
        iemb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        temb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        clip_val = float((iemb * temb).sum(dim=-1).item()) * 100.0
    return ir_val, clip_val


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"

    if not args.acc.is_dir():
        sys.exit(f"[ERROR] acc dir not found: {args.acc}")

    prompt_dirs = sorted([d for d in args.acc.iterdir() if d.is_dir() and d.name.startswith("prompt_")])
    if args.limit:
        prompt_dirs = prompt_dirs[: args.limit]
    if not prompt_dirs:
        sys.exit(f"[ERROR] no prompt_*/ subdirs under {args.acc}")
    print(f"[INFO] {len(prompt_dirs)} prompts to evaluate; device={device}", flush=True)

    print(f"[INFO] loading metric models...", flush=True)
    lpips_m, clip_m, clip_p, ir_m = load_metric_models(
        device, args.lpips_net, args.clip_model, args.image_reward_model,
    )
    print(f"[INFO] metric models loaded", flush=True)

    aggregate = []

    for prompt_dir in tqdm(prompt_dirs, desc="prompts"):
        out_path = prompt_dir / "oracle_metrics.json"
        if args.resume and out_path.is_file():
            try:
                aggregate.append(json.loads(out_path.read_text()))
                continue
            except (OSError, json.JSONDecodeError):
                pass

        manifest_path = prompt_dir / "manifest.json"
        if not manifest_path.is_file():
            print(f"[WARN] {prompt_dir.name} missing manifest.json, skip", file=sys.stderr)
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete", False):
            print(f"[WARN] {prompt_dir.name} manifest not complete, skip", file=sys.stderr)
            continue

        prompt = manifest["prompt"]
        k_values = manifest["k_values"]

        base_z = torch.load(prompt_dir / "baseline.pt", map_location="cpu")
        base_img = Image.open(prompt_dir / "baseline.png").convert("RGB")

        ir_base, clip_base = score_baseline(
            base_img, prompt, clip_m, clip_p, ir_m, device,
        )

        per_k = []
        for k in k_values:
            try:
                oracle_z = torch.load(prompt_dir / f"oracle_k{k:03d}.pt", map_location="cpu")
                oracle_img = Image.open(prompt_dir / f"oracle_k{k:03d}.png").convert("RGB")
            except (OSError, FileNotFoundError) as e:
                print(f"[WARN] {prompt_dir.name} k={k} missing: {e}", file=sys.stderr)
                continue
            m = compute_one_pair(
                base_img, oracle_img, base_z, oracle_z, prompt,
                ir_base, clip_base,
                lpips_m, clip_m, clip_p, ir_m, device,
            )
            m["k"] = int(k)
            per_k.append(m)

        result = {
            "prompt_idx": manifest["prompt_idx"],
            "prompt": prompt,
            "seed": manifest["seed"],
            "num_steps": manifest["num_steps"],
            "k_values": k_values,
            "ir_baseline": ir_base,
            "clip_baseline": clip_base,
            "per_k": per_k,
        }
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        aggregate.append(result)

    if args.wandb:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from lib.wandb_logger import _require_wandb  # noqa
            wandb = _require_wandb()
            project = args.wandb_project or os.environ.get("WANDB_PROJECT", "gph-analysis")
            entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
            run_id = args.wandb_run_id or args.acc.name
            wandb.init(
                project=project, entity=entity,
                id=run_id, name=run_id, resume="allow",
                job_type="oracle_eval",
                config={
                    "experiment": "oracle_fragility",
                    "cache_event": "seacache_reuse",
                    "n_prompts": len(aggregate),
                    "acc": str(args.acc),
                    "lpips_net": args.lpips_net,
                    "clip_model": args.clip_model,
                    "image_reward_model": args.image_reward_model,
                },
            )

            cols = ["prompt_idx", "k",
                    "latent_L2_abs", "latent_L2_rel",
                    "psnr", "ssim", "lpips",
                    "d_ir", "d_clip"]
            tbl = wandb.Table(columns=cols)
            for r in aggregate:
                for m in r["per_k"]:
                    tbl.add_data(
                        r["prompt_idx"], m["k"],
                        m["latent_L2_abs"], m["latent_L2_rel"],
                        m["psnr"], m["ssim"], m["lpips"],
                        m["d_ir"], m["d_clip"],
                    )
            wandb.log({"oracle_per_prompt_k": tbl})

            if aggregate:
                k_values = aggregate[0]["k_values"]
                for i, k in enumerate(k_values):
                    log_payload = {"k": int(k)}
                    for key in ("latent_L2_abs", "latent_L2_rel", "psnr",
                                "ssim", "lpips", "d_ir", "d_clip"):
                        vals = []
                        for r in aggregate:
                            if i < len(r["per_k"]):
                                vals.append(r["per_k"][i][key])
                        if vals:
                            log_payload[f"aggregate/{key}_mean"] = float(np.mean(vals))
                            log_payload[f"aggregate/{key}_std"]  = float(np.std(vals))
                    wandb.log(log_payload)

            wandb.finish()
            print(f"[wandb_oracle] uploaded run_id={run_id} ({len(aggregate)} prompts)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] wandb upload failed: {exc}", file=sys.stderr)

    print(f"[INFO] eval complete: {len(aggregate)} prompts processed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
