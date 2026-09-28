from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from lib.l2p import append_history, load_l2p_weight_file, predict_l2p


class L2POutputMethod:
    """L2P predictor for the tensor returned by HunyuanVideo's final layer."""

    name = "l2p_output"

    def __init__(
        self,
        *,
        action: Any,
        weights_path: str | Path,
        num_steps: int,
        min_abs_weight: float = 0.0,
    ) -> None:
        metadata = load_l2p_weight_file(weights_path, num_steps=num_steps)
        target = str(metadata.get("target", "final_output"))
        if target not in {"final_output", "final_layer_output"}:
            raise ValueError(f"Hunyuan L2P requires final-output weights, got {target!r}")
        self.action = action
        self.weights = metadata["weights"]
        self.weights_metadata = {
            key: value for key, value in metadata.items() if key != "weights"
        }
        self.num_steps = int(num_steps)
        self.min_abs_weight = float(min_abs_weight)
        self.reset()

    def reset(self) -> None:
        self.action.reset()
        self.step = 0
        self.current_step = -1
        self.current_full = True
        self.history: dict[int, torch.Tensor] = {}
        self.last_prediction_fields: dict[str, Any] = {}

    def decide(self) -> MethodDecision:
        full, reason = self.action.decide_full()
        self.current_step = self.step
        self.step += 1
        if not full and not self.history:
            full = True
            reason = "history_unready"
        self.current_full = bool(full)
        self.last_prediction_fields = {}
        return MethodDecision(full=bool(full), reason=str(reason))

    def final_output(self, output: torch.Tensor | None = None) -> torch.Tensor:
        if self.current_step < 0:
            raise RuntimeError("L2P output requested before a step decision")
        if self.current_full:
            if output is None:
                raise RuntimeError("full L2P step requires the real final-layer output")
            value = output
            self.last_prediction_fields = {
                "l2p_target": "final_output",
                "l2p_weights_used": 0,
                "l2p_fallback_latest": False,
            }
        else:
            value, fields = predict_l2p(
                self.history,
                self.weights,
                current_step=self.current_step,
                min_abs_weight=self.min_abs_weight,
            )
            self.last_prediction_fields = {
                "l2p_target": "final_output",
                **fields,
            }
        self.history = append_history(self.history, self.current_step, value)
        return value
