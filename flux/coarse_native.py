"""Thin adapters around the locked native FLUX gate implementations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_native_module(mode: str) -> Any:
    return importlib.import_module(f"flux.{mode}")


@dataclass(frozen=True)
class FluxNativeGateConfig:
    mode: str
    num_steps: int = 50
    threshold: float = 0.3
    first_enhance: int = 1
    teacache_backbone: str = "flux"
    sencache_sensitivity_path: str | Path | None = None
    sencache_threshold_start: float = 0.005
    sencache_threshold_scale: str | float | int | None = "auto"
    sencache_switch_ratio: float = 0.2
    sencache_max_skip: int = 10
    sencache_ret_steps: int = 0
    sencache_cutoff_steps: int = -1


class FluxNativeGateAdapter:
    """Expose SeaCache, TeaCache, and SenCache through one screening interface."""

    _DECISION_ATTRS = {
        "seacache": "seacache_decisions",
        "teacache": "teacache_decisions",
        "sencache": "sencache_decisions",
    }

    def __init__(self, pipe: Any, config: FluxNativeGateConfig):
        if config.mode not in self._DECISION_ATTRS:
            raise ValueError(f"unsupported FLUX native gate: {config.mode}")
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.config = config
        self.prompt_idx: int | None = None
        self.seed: int | None = None
        self._module: Any = None
        self._teardown: Any = None

    def install(self) -> None:
        if self._teardown is not None:
            raise RuntimeError("FLUX native gate adapter already installed")
        module = _load_native_module(self.config.mode)
        if self.config.mode == "seacache":
            teardown = module.install(
                self.pipe,
                threshold=self.config.threshold,
                num_steps=self.config.num_steps,
                first_enhance=self.config.first_enhance,
            )
        elif self.config.mode == "teacache":
            teardown = module.install(
                self.pipe,
                threshold=self.config.threshold,
                num_steps=self.config.num_steps,
                first_enhance=self.config.first_enhance,
                backbone=self.config.teacache_backbone,
            )
        else:
            if self.config.sencache_sensitivity_path is None:
                raise ValueError("SenCache requires a sensitivity table")
            teardown = module.install(
                self.pipe,
                sensitivity_path=str(self.config.sencache_sensitivity_path),
                threshold_start=self.config.sencache_threshold_start,
                threshold_main=self.config.threshold,
                num_steps=self.config.num_steps,
                first_enhance=self.config.first_enhance,
                max_skip=self.config.sencache_max_skip,
                threshold_scale=self.config.sencache_threshold_scale,
                switch_ratio=self.config.sencache_switch_ratio,
                ret_steps=self.config.sencache_ret_steps,
                cutoff_steps=self.config.sencache_cutoff_steps,
            )
        self._module = module
        self._teardown = teardown

    def restore(self) -> None:
        if self._teardown is None:
            return
        self._teardown()
        self._teardown = None
        self._module = None

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        if self._module is None:
            raise RuntimeError("FLUX native gate adapter is not installed")
        self.prompt_idx = prompt_idx
        self.seed = seed
        self._module.reset_per_image_state(self.pipe)

    def decisions(self) -> dict[str, Any]:
        attr = self._DECISION_ATTRS[self.config.mode]
        raw_rows = list(getattr(self.transformer, attr, ()))
        if len(raw_rows) != int(self.config.num_steps):
            raise RuntimeError(
                f"{self.config.mode} recorded {len(raw_rows)} decisions; "
                f"expected {self.config.num_steps}"
            )
        rows = [
            {
                **row,
                "action": "cache" if int(row.get("u", 0)) == 1 else "full",
            }
            for row in raw_rows
        ]
        cached = sum(row["action"] == "cache" for row in rows)
        return {
            "schema": "flux_native_gate_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": f"{self.config.mode}_native",
            "num_steps": int(self.config.num_steps),
            "per_step": rows,
            "summary": {
                "n_total": len(rows),
                "n_full": len(rows) - cached,
                "n_cached": cached,
                "cache_ratio": float(cached / len(rows)) if rows else 0.0,
            },
        }
