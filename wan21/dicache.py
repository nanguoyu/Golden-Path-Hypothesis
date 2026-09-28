"""DiCache adapter for the Wan2.1 t2v-1.3B backbone.

DiCache is the one matrix method whose gate and payload cannot be separated:
the decision statistic is produced by a shallow probe -- the first
`probe_depth` blocks run for real -- and the same probe output is what aligns
the reused residual.  So, like `hunyuan_video/dicache.py`, it is a fused
gate + adapter rather than a method object plus a generic adapter.

Structurally it is simpler than the Hunyuan twin: Wan is one uniform 30-block
stack over a single tensor, so there is no double/single split and no
`output[:, :img_len]` slice.

Gate semantics are the image-side locked set (plan section 2.2 item 2), which
this repository applies on every backbone so the matrix compares backbones and
not gate policies:

1. warmup forces full while `step <= int(ret_ratio * num_steps)` -- note the
   `<=`: 0.2 of 50 steps is 11 forced steps, not 10;
2. the terminal step `num_steps - 1` is forced full;
3. the threshold comparison is strict: cache only when
   `accumulated + delta_y < threshold`;
4. `error_choice = delta_y` is hardcoded -- `delta_x` is recorded but never
   enters the decision.

The official WAN2.1 variant
(`reference/dicache/code/WAN2.1/run_wan_dicache.py`) differs on all of these
and on branch independence; it is the recorded deviation, and
`reference/dicache/code/FLUX/run_flux_dicache.py` is what is followed.

CFG handling (plan sections 2.1 / 2.10): the decision is taken from the cond
branch's `delta_y`, but the probe runs on **both** branches on a cached step,
because aligning a branch's residual needs that branch's own probe residual.
Those probe blocks are real computation and are counted in
`original_block_calls` -- DiCache is the explicit exception to the "cached
steps make zero original block calls" criterion (plan section 2.0 item 2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from lib.dicache import aligned_residual, append_anchor, relative_l1
from wan21.adapter import WanCFGAdapter, _block_inputs
from wan21.methods_glue import CondDecidesArbiter, DICACHE_PROBE_DEPTH, DICACHE_RET_RATIO


@dataclass(frozen=True)
class WanDiCacheConfig:
    num_steps: int = 50
    threshold: float = 0.1
    ret_ratio: float = DICACHE_RET_RATIO
    probe_depth: int = DICACHE_PROBE_DEPTH
    #: When set, the action comes from this table instead of from the gate: the
    #: video SPX `di_two_anchor` payload column, which scores DiCache's payload
    #: on a schedule DiCache did not choose. The payload is untouched -- the
    #: probe still runs on both branches of every cached step, gamma is still
    #: estimated and clamped, full steps still refresh both anchors -- only the
    #: decision is read off the table, so `threshold` and `ret_ratio` stop
    #: being inputs. Twin of `hunyuan_video/dicache.py`.
    cache_steps: tuple[int, ...] | None = None


@dataclass(frozen=True)
class WanDiCacheDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    cond_block_calls: int
    uncond_block_calls: int
    accumulated: float
    threshold: float
    probe_depth: int
    branch_policy: str = CondDecidesArbiter.POLICY
    timestep: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    gamma: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WanDiCacheAdapter(WanCFGAdapter):
    """Run the official shallow probe and trajectory-alignment equations on Wan."""

    def __init__(self, model: Any, config: WanDiCacheConfig):
        self.config = config
        if not 1 <= int(config.probe_depth) <= len(model.blocks):
            raise ValueError(
                f"DiCache probe_depth {int(config.probe_depth)} is outside "
                f"Wan's 1..{len(model.blocks)} block stack"
            )
        if config.cache_steps is not None:
            steps = list(config.cache_steps)
            if steps != sorted(set(steps)):
                raise ValueError("DiCache cache_steps must be sorted and unique")
            if steps and (steps[0] < 2 or steps[-1] >= int(config.num_steps) - 1):
                raise ValueError(
                    "a fixed-schedule DiCache keeps steps 0, 1 and the terminal step full")
        super().__init__(model, num_steps=int(config.num_steps))

    def _reset_payload_state(self) -> None:
        # Gate state: one accumulator and one pair of previous observations,
        # both read on the cond branch only.
        self.accumulated = 0.0
        self.previous_input: torch.Tensor | None = None
        self.previous_probe: torch.Tensor | None = None
        # Payload state: two anchors per branch, since the residual a branch
        # reuses is that branch's own.
        self.residual_history: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.probe_history: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self._initial_x: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._probe_outputs: list[torch.Tensor] = []
        self._fields: dict[str, Any] = {}

    # -- probe -------------------------------------------------------------

    def _run_probe(
        self,
        x: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run the first `probe_depth` blocks on a clone of this step's input."""

        probe_x = x.clone()
        self._probe_outputs = []
        for index in range(int(self.config.probe_depth)):
            original = self._patches[index][1][1]
            probe_args = list(args)
            probe_kwargs = dict(kwargs)
            if probe_args:
                probe_args[0] = probe_x
            else:
                probe_kwargs["x"] = probe_x
            probe_x = original(*probe_args, **probe_kwargs)
            self._probe_outputs.append(probe_x)
            self._block_calls += 1
        return probe_x

    # -- decision ----------------------------------------------------------

    def _decide(
        self,
        x: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Cond branch: score the probe, settle the step, build the payload."""

        cfg = self.config
        warmup_last = int(float(cfg.ret_ratio) * int(cfg.num_steps))
        cache = False
        reason = "forced_boundary"
        proposed = self.accumulated
        delta_x: float | None = None
        delta_y: float | None = None
        if cfg.cache_steps is not None:
            # Fixed-schedule payload column: the table decides, and a cached
            # step it names before the anchors exist is an error rather than a
            # silent demotion, which would spend one of the K cached steps the
            # schedule promises.
            cache = self.step in cfg.cache_steps
            reason = "fixed_cache" if cache else "fixed_full"
            if cache:
                if (self.previous_input is None
                        or len(self.residual_history["cond"]) < 2
                        or len(self.residual_history["uncond"]) < 2):
                    raise RuntimeError(
                        f"fixed-schedule DiCache caches step {self.step} before two full "
                        f"steps have laid down anchors on both branches")
                probe = self._run_probe(x, args, kwargs)
                self._current_probe = probe
                delta_x = relative_l1(x, self.previous_input)
                delta_y = relative_l1(probe, self.previous_probe)
                # recorded, never compared: the schedule already decided
                proposed = self.accumulated + delta_y
        else:
            hard_full = (
                self.step <= warmup_last
                or self.step == int(cfg.num_steps) - 1
                or self.previous_input is None
                or self.previous_probe is None
                or not self.residual_history["cond"]
                or not self.residual_history["uncond"]
            )
            if not hard_full:
                probe = self._run_probe(x, args, kwargs)
                self._current_probe = probe
                delta_x = relative_l1(x, self.previous_input)
                delta_y = relative_l1(probe, self.previous_probe)
                proposed += delta_y
                cache = proposed < float(cfg.threshold)
                reason = "threshold_cache" if cache else "threshold_full"

        self._step_full = not cache
        self._step_reason = reason
        self.accumulated = float(proposed) if cache else 0.0
        self._fields = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "accumulated": float(self.accumulated),
            "threshold": float(cfg.threshold),
            "gamma": None,
            "probe_depth": int(cfg.probe_depth),
        }
        if not cache:
            return None
        payload, gamma = self._aligned_payload("cond")
        self._fields["gamma"] = gamma
        self.previous_input = x.detach()
        assert self._current_probe is not None
        self.previous_probe = self._current_probe.detach()
        return payload

    def _follow(
        self,
        x: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Uncond branch: execute the cond decision.

        On a cached step the probe still runs, because `aligned_residual` needs
        this branch's own probe residual; on a full step it does not, so a
        fully-computed step stays at exactly 60 original block calls.
        """

        if self._step_full:
            return None
        self._current_probe = self._run_probe(x, args, kwargs)
        payload, _gamma = self._aligned_payload("uncond")
        return payload

    def _aligned_payload(self, branch: str) -> tuple[torch.Tensor, float | None]:
        if self._current_probe is None:
            raise RuntimeError("DiCache cache action has no probe output")
        probe_residual = self._current_probe - self._initial_x
        return aligned_residual(
            probe_residual,
            self.residual_history[branch],
            self.probe_history[branch],
        )

    # -- block dispatch ----------------------------------------------------

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        branch = self.branch
        if index == 0:
            x, _e, _grid_sizes = _block_inputs(args, kwargs)
            self._initial_x = x.clone()
            self._current_probe = None
            self._probe_outputs = []
            if self.arbiter.is_cond:
                payload = self._decide(x, args, kwargs)
                # DiCache has no method object; it hands the arbiter a plain
                # MethodDecision so the cond-decides bookkeeping is one object
                # for all nine methods.
                self.arbiter.settle(
                    MethodDecision(full=bool(self._step_full), reason=str(self._step_reason))
                )
            else:
                decision = self.arbiter.follow()
                self._step_full = decision.full
                self._step_reason = decision.reason
                payload = self._follow(x, args, kwargs)
            if not self._step_full:
                if payload is None:
                    raise RuntimeError(f"DiCache cache at step {self.step} produced no payload")
                return x + payload

        if not self._step_full:
            return args[0]

        if index < len(self._probe_outputs):
            output = self._probe_outputs[index]
        else:
            output = original(*args, **kwargs)
            self._block_calls += 1

        if index == int(self.config.probe_depth) - 1:
            self._current_probe = output.detach()
        if index == len(self.model.blocks) - 1:
            if self._initial_x is None or self._current_probe is None:
                raise RuntimeError("DiCache full step is missing an anchor")
            append_anchor(self.residual_history[branch], output - self._initial_x)
            append_anchor(self.probe_history[branch], self._current_probe - self._initial_x)
            if branch == "cond":
                self.previous_input = self._initial_x.detach()
                self.previous_probe = self._current_probe.detach()
        return output

    def _decision_record(self) -> WanDiCacheDecisionRecord:
        if not self._fields:
            raise RuntimeError("DiCache adapter did not enter the first block")
        return WanDiCacheDecisionRecord(**self._fields, **self._common_record_fields())
