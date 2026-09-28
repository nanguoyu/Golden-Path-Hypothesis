"""The matrix's five metrics, resident, for the schedule search.

`evaluation/eval_metrics.py` owns the implementations and the model loading;
this holds one LPIPS net, one ImageReward model and one CLIP model in memory
next to the diffusion pipeline, so a search evaluation scores its eight
calibration pairs without reloading anything.  PSNR is the uint8 RGB PSNR of
`flux/exhaustive_k41_runner.py::mse_and_psnr` -- the number the search has
recorded as `psnr_db` from the start, exact-match sentinel included.

One call scores one pair: the decoded candidate against that pair's
full-compute reference (psnr, ssim, lpips) and the candidate on its own with
that pair's prompt (image_reward, clip).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The matrix metric set, in the order `evaluation/eval_metrics.py` lists it.
METRIC_NAMES: tuple[str, ...] = ("psnr", "ssim", "lpips", "image_reward", "clip")


class MetricModels:
    """LPIPS + ImageReward + CLIP held on one device, scoring pairs in memory.

    On an 80 GB H100 the three sit beside a bf16 FLUX (~24 GB) or Qwen-Image
    (~41 GB at 1328^2) pipeline with room to spare: LPIPS(alex) is a few MB,
    ImageReward and CLIP ViT-L/14 about 1.7 GB each in fp32.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        lpips_net: str = "alex",
        clip_model: str = "openai/clip-vit-large-patch14",
        image_reward_model: str = "ImageReward-v1.0",
    ) -> None:
        from evaluation.eval_metrics import (
            load_clip_model,
            load_image_reward_model,
            load_lpips_model,
        )

        self.device = str(device)
        self.lpips_net = str(lpips_net)
        self.clip_model_name = str(clip_model)
        self.image_reward_model_name = str(image_reward_model)
        self._lpips = load_lpips_model(self.device, self.lpips_net)
        self._image_reward = load_image_reward_model(
            self.device, self.image_reward_model_name
        )
        self._clip, self._clip_proc = load_clip_model(self.device, self.clip_model_name)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "names": list(METRIC_NAMES),
            "lpips_net": self.lpips_net,
            "clip_model": self.clip_model_name,
            "image_reward_model": self.image_reward_model_name,
            "psnr": "uint8_rgb_psnr_data_range_255",
        }

    def score(
        self, reference: np.ndarray, candidate: np.ndarray, prompt: str
    ) -> dict[str, float]:
        """The five metrics of one (reference, candidate, prompt) triple."""

        from PIL import Image

        from evaluation.eval_metrics import score_clip, score_image_reward, score_lpips
        from evaluation.eval_metrics import score_ssim
        from flux.exhaustive_k41_runner import mse_and_psnr

        if reference.shape != candidate.shape:
            raise ValueError(
                f"image shapes differ: {reference.shape} vs {candidate.shape}"
            )
        reference_image = Image.fromarray(reference)
        candidate_image = Image.fromarray(candidate)
        return {
            "psnr": float(mse_and_psnr(reference, candidate)[1]),
            "ssim": score_ssim(reference, candidate),
            "lpips": score_lpips(
                self._lpips, candidate_image, reference_image, self.device
            ),
            "image_reward": score_image_reward(
                self._image_reward, prompt, candidate_image
            ),
            "clip": score_clip(
                self._clip, self._clip_proc, candidate_image, prompt, self.device
            ),
        }

    def score_pairs(
        self,
        references: Sequence[np.ndarray],
        candidates: Sequence[np.ndarray],
        prompts: Sequence[str],
    ) -> list[dict[str, float]]:
        """One metric dict per pair, in the order the pairs were given."""

        if not (len(references) == len(candidates) == len(prompts)):
            raise ValueError(
                f"{len(references)} references, {len(candidates)} candidates and "
                f"{len(prompts)} prompts do not line up"
            )
        return [
            self.score(reference, candidate, prompt)
            for reference, candidate, prompt in zip(references, candidates, prompts)
        ]
