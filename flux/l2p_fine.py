"""Project-specific fine-grained L2P predictor for FLUX.

The L2P paper predicts only final-layer features.  This file is a research
extension: it applies the same learned timestep weight matrix to each of the
existing 114 fine cache slots from ``lib.flux_fine_scaffold``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from lib.flux_fine_scaffold import CacheHistory, install_fine_cache, reset_per_image_state_fine
from lib.l2p import append_history, latest_history_step, load_l2p_weight_file, predict_l2p


class L2PFinePredictor:
    """Shared-weight L2P predictor for every fine cache slot.

    The paper's complete-history L2P semantics are memory-feasible for one
    final-hidden tensor, but not for 114 fine FLUX slots.  This research
    extension therefore stores full-anchor observations only, matching the
    existing fine Taylor/HiCache scaffold's memory contract.
    """

    max_order: int = 0

    def __init__(self, weights: torch.Tensor, *, min_abs_weight: float = 0.0) -> None:
        self.weights = weights
        self.min_abs_weight = float(min_abs_weight)

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        last_step = latest_history_step(prev_history or {})
        current_step = 0 if last_step is None else int(last_step) + int(step_gap)
        return append_history(prev_history, current_step, feature)

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        last_step = latest_history_step(history)
        if last_step is None:
            raise ValueError("empty L2P fine history")
        current_step = int(last_step) + int(step_offset)
        pred, _fields = predict_l2p(
            history,
            self.weights,
            current_step=current_step,
            min_abs_weight=self.min_abs_weight,
        )
        return pred


def install(
    pipe,
    *,
    weights_path: str | Path,
    interval: int = 7,
    first_enhance: int = 3,
    num_steps: int,
    min_abs_weight: float = 0.0,
) -> Callable[[], None]:
    """Install the fine 114-slot L2P research variant."""
    meta = load_l2p_weight_file(weights_path, num_steps=num_steps)
    target = meta.get("target")
    if target is not None and str(target) in ("final_hidden", "final_layer", "hidden_states"):
        raise ValueError(
            "L2P_fine expects fine-slot or target-free shared weights, "
            f"got target={target!r}"
        )
    predictor = L2PFinePredictor(meta["weights"], min_abs_weight=float(min_abs_weight))
    teardown_fine = install_fine_cache(
        pipe,
        predictor=predictor,
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
        method_tag="l2p_fine_shared_w",
    )
    tr = pipe.transformer
    tr._l2p_weights_meta = {k: v for k, v in meta.items() if k != "weights"}
    tr._l2p_min_abs_weight = float(min_abs_weight)
    tr._l2p_fine_variant = "shared_w_114_slots"

    def teardown() -> None:
        teardown_fine()
        for attr in ("_l2p_weights_meta", "_l2p_min_abs_weight", "_l2p_fine_variant"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass

    return teardown


def reset_per_image_state(
    pipe,
    *,
    action_steps: Optional[set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    reset_per_image_state_fine(pipe, action_steps=action_steps, prompt_idx=prompt_idx)
