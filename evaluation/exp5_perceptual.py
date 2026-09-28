#!/usr/bin/env python3
"""Experiment 5: perceptual interval oracle.

docs/research_plan_after_extension.md §3 Experiment 5. The post-extension
H5-H7 analysis is latent-L2; the Phase 4 schedule failure was perceptual.
This connects the interval oracle back to image space.

The generation side reuses flux/interval_oracle.py (it already saves
prompt_XXXXX/baseline.png and interval_aAA_bBB.png). This script computes,
per interval [a,b] vs the no-cache baseline:
  psnr, ssim, lpips, d_ir = ImageReward(iv) - ImageReward(base),
  d_clip = CLIP(iv) - CLIP(base)  (x100).

Reuses evaluation/eval_oracle.py::load_metric_models.

Output: <interval_oracle_dir>/prompt_XXXXX/exp5_metrics.json
        <out_dir>/exp5_perceptual_summary.{json,csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.eval_oracle import (  # noqa: E402
    _to_chw_norm_tensor, _to_uint8, load_metric_models,
)


def _clip_score(img: Image.Image, prompt: str, clip_m, clip_p, device: str) -> float:
    with torch.no_grad():
        inp = clip_p(text=[prompt], images=img.convert("RGB"),
                     return_tensors="pt", padding=True, truncation=True).to(device)
        out = clip_m(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((ie * te).sum(dim=-1).item()) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp 5: perceptual interval oracle.")
    ap.add_argument("--interval_oracle", type=Path, required=True,
                    help="interval_oracle output dir (prompt_*/interval_*.png).")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lpips_net", default="alex")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--image_reward_model", default="ImageReward-v1.0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or args.interval_oracle
    out_dir.mkdir(parents=True, exist_ok=True)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    lpips_m, clip_m, clip_p, ir_m = load_metric_models(
        args.device, args.lpips_net, args.clip_model, args.image_reward_model)

    pdirs = sorted(args.interval_oracle.glob("prompt_*"))
    if args.limit:
        pdirs = pdirs[: args.limit]
    print(f"[INFO] prompt dirs: {len(pdirs)}")

    rows: List[Dict] = []
    for pdir in pdirs:
        man = pdir / "manifest.json"
        if not man.is_file():
            continue
        m = json.loads(man.read_text())
        if not m.get("complete"):
            continue
        prompt = m["prompt"]
        out_json = pdir / "exp5_metrics.json"
        if args.resume and out_json.is_file():
            try:
                rows.extend(json.loads(out_json.read_text())["per_interval"])
                continue
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        base_p = pdir / "baseline.png"
        if not base_p.is_file():
            continue
        img_base = Image.open(base_p).convert("RGB")
        base_u8 = _to_uint8(img_base)
        with torch.no_grad():
            ir_base = float(ir_m.score(prompt, img_base))
        clip_base = _clip_score(img_base, prompt, clip_m, clip_p, args.device)

        per_iv: List[Dict] = []
        for ivp in sorted(pdir.glob("interval_a*_b*.png")):
            parts = ivp.stem.split("_")          # interval_aAA_bBB
            a, b = int(parts[1][1:]), int(parts[2][1:])
            img = Image.open(ivp).convert("RGB")
            u8 = _to_uint8(img)
            psnr = float(peak_signal_noise_ratio(base_u8, u8, data_range=255))
            ssim = float(structural_similarity(base_u8, u8, data_range=255,
                                               channel_axis=-1))
            with torch.no_grad():
                lp = float(lpips_m(_to_chw_norm_tensor(img_base, args.device),
                                   _to_chw_norm_tensor(img, args.device)
                                   ).squeeze().item())
                ir = float(ir_m.score(prompt, img))
            clip = _clip_score(img, prompt, clip_m, clip_p, args.device)
            rec = {"prompt_idx": m["prompt_idx"], "a": a, "b": b, "gap": b - a,
                   "psnr": psnr, "ssim": ssim, "lpips": lp,
                   "d_ir": ir - ir_base, "d_clip": clip - clip_base}
            per_iv.append(rec)
            rows.append(rec)
        out_json.write_text(json.dumps(
            {"prompt_idx": m["prompt_idx"], "per_interval": per_iv,
             "complete": True}, ensure_ascii=False))
        print(f"[INFO] prompt {m['prompt_idx']} done ({len(per_iv)} intervals)",
              flush=True)

    if not rows:
        raise SystemExit("no intervals evaluated")

    def agg(sub: List[Dict]) -> Dict:
        return {"n": len(sub),
                **{f: float(np.mean([r[f] for r in sub]))
                   for f in ("psnr", "ssim", "lpips", "d_ir", "d_clip")}}

    by_iv = defaultdict(list)
    for r in rows:
        by_iv[(r["a"], r["b"])].append(r)
    summary = {"n_rows": len(rows), "overall": agg(rows),
               "per_interval": {f"{a}:{b}": agg(s)
                                for (a, b), s in sorted(by_iv.items())}}
    (out_dir / "exp5_perceptual_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "exp5_perceptual_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "gap", "n", "psnr", "ssim", "lpips", "d_ir", "d_clip"])
        for (a, b), s in sorted(by_iv.items()):
            d = agg(s)
            w.writerow([a, b, b - a, d["n"], "%.3f" % d["psnr"],
                        "%.4f" % d["ssim"], "%.4f" % d["lpips"],
                        "%.4f" % d["d_ir"], "%.4f" % d["d_clip"]])

    o = summary["overall"]
    print("\n=== Exp 5: perceptual interval oracle (n=%d) ===" % o["n"])
    print("  psnr=%.2f  ssim=%.4f  lpips=%.4f  d_ir=%.4f  d_clip=%.3f"
          % (o["psnr"], o["ssim"], o["lpips"], o["d_ir"], o["d_clip"]))
    print(f"[INFO] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
