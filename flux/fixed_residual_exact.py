"""Fixed-schedule whole-transformer residual reuse for FLUX."""

from __future__ import annotations

from typing import Any

from flux.oracle_runner import install_oracle, reset_oracle_state
from lib.fixed_schedule import validate_cache_steps


class FluxFixedResidualAdapter:
    """Thin decision-recording wrapper around the existing residual-reuse path."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        mode: str = "budcache_exact",
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.mode = str(mode)
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self._handles: list[Any] = []
        self._teardown = None
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        reset_oracle_state(self.pipe)
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        self._teardown = install_oracle(
            self.pipe,
            cache_steps=self.cache_steps,
            num_steps=self.num_steps,
            cache_mode="seacache",
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if self._teardown is not None:
            self._teardown()
            self._teardown = None

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        cache = self.step in self._cache_steps
        self.records.append(
            {
                "step": int(self.step),
                "action": "cache" if cache else "full",
                "u": int(cache),
                "payload": "latest_whole_transformer_residual",
            }
        )
        self.step += 1
        return output

    def decisions(self) -> dict[str, Any]:
        cached = sum(row["u"] for row in self.records)
        if len(self.records) != self.num_steps or cached != len(self.cache_steps):
            raise RuntimeError(
                f"{self.mode} recorded {len(self.records)} steps and K={cached}; "
                f"expected {self.num_steps} and K={len(self.cache_steps)}"
            )
        return {
            "schema": "flux_fixed_residual_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": self.mode,
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "per_step": list(self.records),
            "summary": {
                "n_total": len(self.records),
                "n_full": len(self.records) - cached,
                "n_cached": cached,
                "cache_ratio": cached / len(self.records),
            },
        }
