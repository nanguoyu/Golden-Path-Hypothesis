"""Evaluate sampled-frame video runs against the no-cache reference.

The evaluator samples a fixed set of frames from each ``video_<idx>.mp4`` pair
and reports per-video means for PSNR, SSIM, LPIPS, CLIP, and ImageReward.
It intentionally mirrors ``evaluation/eval_metrics.py`` conventions where
PSNR/SSIM/CLIP/ImageReward are higher-is-better and LPIPS is lower-is-better.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


METRIC_CHOICES = ["psnr", "ssim", "lpips", "temporal_lpips_delta", "clip", "image_reward"]
PAIRWISE_METRICS = {"psnr", "ssim", "lpips", "temporal_lpips_delta"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sampled-frame video metrics")
    p.add_argument("--acc", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--prompts", type=Path, default=None,
                   help="prompt list, one per line; optional when --task_manifest is given")
    p.add_argument("--task_manifest", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backbone", default="video")
    p.add_argument("--eval_tag", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics", nargs="+", default=METRIC_CHOICES, choices=METRIC_CHOICES)
    p.add_argument("--frame_indices", default="0,16,32,48,64",
                   help="Comma-separated 0-based frame indices or 'all'.")
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_reward_model", default="ImageReward-v1.0")
    return p.parse_args()


def _parse_frame_indices(raw: str) -> list[int] | None:
    if raw.strip().lower() == "all":
        return None
    out = [int(x) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--frame_indices parsed to an empty list")
    if any(x < 0 for x in out):
        raise ValueError("--frame_indices must be non-negative")
    return out


def _find_video(d: Path, idx: int) -> Path | None:
    p = d / f"video_{idx:05d}.mp4"
    return p if p.is_file() else None


def collect_entries(
    acc_dir: Path,
    gt_dir: Path,
    prompts: list[str],
    limit: int | None,
    *,
    allow_missing: bool,
) -> list[tuple[int, Path, Path, str]]:
    n = len(prompts) if limit is None or limit <= 0 else min(limit, len(prompts))
    entries: list[tuple[int, Path, Path, str]] = []
    missing_acc = 0
    missing_gt = 0
    for idx in range(n):
        acc_p = _find_video(acc_dir, idx)
        gt_p = _find_video(gt_dir, idx)
        if acc_p is None:
            missing_acc += 1
        if gt_p is None:
            missing_gt += 1
        if acc_p is None or gt_p is None:
            continue
        entries.append((idx, acc_p, gt_p, prompts[idx]))
    if (missing_acc or missing_gt) and not allow_missing:
        raise FileNotFoundError(
            f"video pair coverage incomplete: missing_acc={missing_acc}, missing_gt={missing_gt}"
        )
    if missing_acc or missing_gt:
        print(
            f"[WARN] incomplete coverage allowed: missing_acc={missing_acc} missing_gt={missing_gt}",
            file=sys.stderr,
        )
    return entries


def _read_selected_frames(path: Path, frame_indices: list[int] | None) -> list[np.ndarray]:
    import imageio.v2 as imageio

    frames: list[np.ndarray] = []
    reader = imageio.get_reader(path, format="ffmpeg")
    try:
        iterator = enumerate(reader) if frame_indices is None else ((idx, reader.get_data(idx)) for idx in frame_indices)
        try:
            for frame_idx, frame in iterator:
                arr = np.asarray(frame)
                if arr.ndim != 3 or arr.shape[-1] < 3:
                    raise ValueError(f"unexpected frame shape {arr.shape} in {path} at {frame_idx}")
                frames.append(arr[..., :3].astype(np.uint8, copy=False))
        except IndexError as exc:
            raise ValueError(f"{path} does not contain every requested frame") from exc
    finally:
        reader.close()
    return frames


def _pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame, mode="RGB")


def _to_lpips_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    arr = frame.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def _summarize(vals: list[float]) -> dict[str, float | int] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "n": int(arr.size),
    }


class MetricState:
    def __init__(self, want: set[str], device: str, lpips_net: str, clip_model: str, ir_model: str) -> None:
        self.want = want
        self.device = device
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        self.ir_model = None

        # temporal_lpips_delta is LPIPS between consecutive frames, so it needs
        # the same network even when plain lpips was not asked for
        if want & {"lpips", "temporal_lpips_delta"}:
            import lpips
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(device).eval()
        if "clip" in want:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        if "image_reward" in want:
            import ImageReward as RM
            self.ir_model = RM.load(ir_model, device=device)


def evaluate_entry(
    idx: int,
    acc_path: Path,
    gt_path: Path,
    prompt: str,
    frame_indices: list[int] | None,
    state: MetricState,
) -> dict[str, Any]:
    acc_frames = _read_selected_frames(acc_path, frame_indices)
    gt_frames = _read_selected_frames(gt_path, frame_indices)
    if not acc_frames or not gt_frames:
        raise ValueError(f"empty decoded video for idx={idx}")
    if len(acc_frames) != len(gt_frames):
        raise ValueError(f"frame sample count mismatch for idx={idx}")
    for frame_idx, (acc_frame, gt_frame) in enumerate(zip(acc_frames, gt_frames)):
        if acc_frame.shape != gt_frame.shape:
            raise ValueError(
                f"frame shape mismatch for idx={idx}, frame={frame_idx}: "
                f"{acc_frame.shape} != {gt_frame.shape}"
            )

    result: dict[str, Any] = {
        "idx": idx,
        "video": acc_path.name,
        "acc_video_sha256": _sha256_file(acc_path),
        "gt_video_sha256": _sha256_file(gt_path),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "n_frames": len(acc_frames),
    }

    if "psnr" in state.want:
        from skimage.metrics import peak_signal_noise_ratio

        vals = [
            float(peak_signal_noise_ratio(g, a, data_range=255))
            for a, g in zip(acc_frames, gt_frames)
        ]
        result["psnr"] = float(np.mean(vals))
    if "ssim" in state.want:
        from skimage.metrics import structural_similarity

        vals = [
            float(structural_similarity(g, a, data_range=255, channel_axis=-1))
            for a, g in zip(acc_frames, gt_frames)
        ]
        result["ssim"] = float(np.mean(vals))
    if "lpips" in state.want:
        assert state.lpips_model is not None
        vals = []
        with torch.no_grad():
            for a, g in zip(acc_frames, gt_frames):
                vals.append(float(state.lpips_model(_to_lpips_tensor(a, state.device), _to_lpips_tensor(g, state.device)).item()))
        result["lpips"] = float(np.mean(vals))
    if "temporal_lpips_delta" in state.want:
        # Plan section 4.3: how much the acceleration changed the video's own
        # frame-to-frame motion, not its distance to the reference. Both sides
        # are LPIPS between CONSECUTIVE frames of one video; the metric is the
        # accelerated video's mean minus the reference's, so 0 means the cache
        # left temporal coherence alone and a positive value means it added
        # flicker. Needs adjacent frames, so it is only defined when every frame
        # was read.
        assert state.lpips_model is not None
        if frame_indices is not None:
            raise SystemExit(
                "temporal_lpips_delta compares consecutive frames and needs the "
                "whole video; pass --frame_indices all")
        if len(acc_frames) < 2:
            raise SystemExit("temporal_lpips_delta needs at least two frames")

        def _consecutive_mean(frames: list[np.ndarray]) -> float:
            with torch.no_grad():
                return float(np.mean([
                    float(state.lpips_model(
                        _to_lpips_tensor(first, state.device),
                        _to_lpips_tensor(second, state.device)).item())
                    for first, second in zip(frames, frames[1:])
                ]))

        acc_temporal = _consecutive_mean(acc_frames)
        gt_temporal = _consecutive_mean(gt_frames)
        result["temporal_lpips_acc"] = acc_temporal
        result["temporal_lpips_gt"] = gt_temporal
        result["temporal_lpips_delta"] = acc_temporal - gt_temporal
    if "clip" in state.want:
        assert state.clip_model is not None and state.clip_processor is not None
        images = [_pil(a) for a in acc_frames]
        inputs = state.clip_processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(state.device)
        with torch.no_grad():
            outputs = state.clip_model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            vals = (img_emb * txt_emb).sum(dim=-1).detach().cpu().numpy() * 100.0
        result["clip"] = float(np.mean(vals))
    if "image_reward" in state.want:
        assert state.ir_model is not None
        vals = [float(state.ir_model.score(prompt, _pil(a))) for a in acc_frames]
        result["image_reward"] = float(np.mean(vals))

    return result


def main() -> int:
    args = parse_args()
    want = set(args.metrics)
    if not args.acc.is_dir():
        sys.exit(f"[ERROR] acc dir not found: {args.acc}")
    if not args.gt.is_dir():
        sys.exit(f"[ERROR] gt dir not found: {args.gt}")
    if args.prompts is None and args.task_manifest is None:
        sys.exit("[ERROR] one of --prompts / --task_manifest is required")
    prompts: list[str] = []
    if args.prompts is not None:
        if not args.prompts.is_file():
            sys.exit(f"[ERROR] prompts file not found: {args.prompts}")
        prompts = [ln.strip() for ln in args.prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    task_manifest_sha256 = None
    if args.task_manifest is not None:
        rows = [
            json.loads(line)
            for line in args.task_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for idx, row in enumerate(rows):
            if int(row.get("task_idx", -1)) != idx:
                raise ValueError(f"task manifest is not contiguous at row {idx}")
        prompts = [str(row["prompt"]) for row in rows]
        task_manifest_sha256 = hashlib.sha256(args.task_manifest.read_bytes()).hexdigest()
    manifest_binding = None
    frame_indices = _parse_frame_indices(args.frame_indices)
    entries = collect_entries(
        args.acc, args.gt, prompts, args.limit, allow_missing=bool(args.allow_missing)
    )
    if not entries:
        sys.exit("[ERROR] no video pairs to evaluate")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU", file=sys.stderr)
        device = "cpu"

    print(f"[INFO] acc={args.acc}")
    print(f"[INFO] gt={args.gt}")
    print(f"[INFO] entries={len(entries)} frame_indices={frame_indices} metrics={sorted(want)}")
    state = MetricState(want, device, args.lpips_net, args.clip_model, args.image_reward_model)

    per_video: list[dict[str, Any]] = []
    for idx, acc_p, gt_p, prompt in tqdm(entries, desc="videos"):
        per_video.append(evaluate_entry(idx, acc_p, gt_p, prompt, frame_indices, state))

    summary = {
        metric: _summarize([float(row[metric]) for row in per_video if metric in row])
        for metric in METRIC_CHOICES
        if metric in want
    }

    out_path = args.output if args.output is not None else args.acc / "metrics.json"
    payload = {
        "schema": "video_pairwise_metrics.v2",
        "evaluator_source_sha256": _sha256_file(Path(__file__)),
        "backbone": args.backbone,
        "eval_tag": args.eval_tag,
        "acc": str(args.acc),
        "gt": str(args.gt),
        "prompts": str(args.prompts) if args.prompts else None,
        "n_pairs": len(per_video),
        "frame_indices": "all" if frame_indices is None else frame_indices,
        "frame_protocol": "all_aligned_frames" if frame_indices is None else "fixed_sampled_frames",
        "task_manifest": str(args.task_manifest) if args.task_manifest else None,
        "task_manifest_sha256": task_manifest_sha256,
        "manifest_binding": manifest_binding,
        "metrics": sorted(want),
        "summary": summary,
        "per_video": per_video,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    payload["pair_set_sha256"] = _canonical_sha256(
        [
            {
                "idx": row["idx"],
                "video": row["video"],
                "acc_video_sha256": row["acc_video_sha256"],
                "gt_video_sha256": row["gt_video_sha256"],
                "prompt_sha256": row["prompt_sha256"],
            }
            for row in per_video
        ]
    )
    payload["metrics_payload_sha256"] = _canonical_sha256(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(out_path)

    print("\n=== Summary ===")
    for metric in METRIC_CHOICES:
        block = summary.get(metric)
        if block is None:
            continue
        print(
            f"{metric:13s} mean={block['mean']:.6f} std={block['std']:.6f} "
            f"min={block['min']:.6f} max={block['max']:.6f} n={block['n']}"
        )
    print(f"[INFO] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
