"""Backbone adapters that let the HunyuanVideo method objects drive Wan2.1.

These are the Wan siblings of `hunyuan_video/adapter.py`, written to the same
method-facing interface so `hunyuan_video.methods.*` runs unmodified (plan
section 2.3): the adapters call `decide()`, `update_slot()` / `predict_slot()` /
`finish_full_step()`, `final_output()` and `observe_input()` exactly as the
Hunyuan adapters do, and only the backbone plumbing differs.

Three things are Wan-specific:

* **A single block stack.**  Wan has 30 uniform `WanAttentionBlock`s over one
  tensor, so the whole-transformer residual is `last block output - block 0
  input` with no `(img, txt)` split and no `output[:, :img_len]` slice.
* **CFG dual forward.**  Each solver step runs two model forwards, cond then
  uncond (`wan21/runner.py:407-408`).  The action is decided once on cond and
  replayed on uncond; payload state is held per branch.  The ruling itself
  lives in `wan21.methods_glue.CondDecidesArbiter`.
* **`model.head`.**  Wan's final projection before `unpatchify` is the
  injection point L2P and MeanCache substitute, the way the Hunyuan adapters
  patch `final_layer`.

As on the Hunyuan side the upstream `WanModel.forward` is never replaced: the
model gets a forward pre/post hook pair and each block an instance-level
forward, so `wan21/backend.py::verify_untouched_transformer` can prove the
stack was handed back untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.cuda.amp as amp

from hunyuan_video.actions import MethodDecision
from hunyuan_video.adapter import _restore_instance_forward, _set_instance_forward
from wan21.methods_glue import CondDecidesArbiter, WAN_FINE_SLOT_COUNT, WAN_LATENT_NUMEL


def _block_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Pull (x, e, grid_sizes) out of a `WanAttentionBlock` call.

    Upstream dispatches `block(x, **kwargs)` (`.../wan/modules/model.py:569-570`),
    so everything but `x` arrives as a keyword; the positional fallbacks follow
    the declared signature `(x, e, seq_lens, grid_sizes, freqs, context,
    context_lens)`.
    """

    x = args[0] if args else kwargs["x"]
    e = kwargs["e"] if "e" in kwargs else args[1]
    grid_sizes = kwargs["grid_sizes"] if "grid_sizes" in kwargs else args[3]
    return x, e, grid_sizes


def _forward_timestep(args: tuple[Any, ...], kwargs: dict[str, Any]) -> float | None:
    """The scalar timestep `WanModel.forward` was called with.

    Recorded on every decision row: SenCache's `|t_k - t_a|` term and its
    sensitivity-table lookup are both in whatever units this value carries, and
    getting that unit wrong is a factor-1000 error (plan section 2.5).
    """

    t = kwargs.get("t")
    if t is None and len(args) > 1:
        t = args[1]
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return float(t.detach().to(torch.float32).reshape(-1)[0].item())
    return float(t)


def _model_latent(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    """The pre-`patch_embedding` latent, as a single tensor.

    `WanModel.forward` takes `x` as a list of `[C, F, H, W]` tensors; the Wan
    upstream SenCache scores `x[0]` (`knowledge/SenCache/code/Wan2.1/sencache.py:67`).
    """

    x = args[0] if args else kwargs["x"]
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"Wan2.1 matrix lane expects batch size 1, got {len(x)} latents")
        return x[0]
    return x


# ---------------------------------------------------------------------------
# Decision records -- one row per solver step, not per model forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WanDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    cond_block_calls: int
    uncond_block_calls: int
    branch_policy: str = CondDecidesArbiter.POLICY
    timestep: float | None = None
    gate_scalar: float | None = None
    accumulated: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WanL2PDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    cond_block_calls: int
    uncond_block_calls: int
    l2p_target: str
    l2p_weights_used: int
    branch_policy: str = CondDecidesArbiter.POLICY
    timestep: float | None = None
    l2p_current_step: int | None = None
    l2p_history_steps: list[int] | None = None
    l2p_weight_l1: float | None = None
    l2p_weight_l2: float | None = None
    l2p_weight_sum: float | None = None
    l2p_fallback_latest: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WanMeanCacheDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    cond_block_calls: int
    uncond_block_calls: int
    sigma_t: float
    sigma_s: float
    requested_jvp_span: int
    actual_jvp_span: int
    jvp_correction_used: bool
    branch_policy: str = CondDecidesArbiter.POLICY
    timestep: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


class WanCFGAdapter:
    """Install/restore plumbing plus the cond-decides CFG clock.

    Subclasses supply `_reset_payload_state`, `_block_forward` and
    `_decision_record`; everything else -- hook lifetime, per-branch block-call
    accounting, and the one-record-per-step contract -- is here.
    """

    def __init__(self, model: Any, *, num_steps: int):
        if not len(model.blocks):
            raise ValueError("Wan2.1 adapter requires a non-empty block stack")
        self.model = model
        self.num_steps = int(num_steps)
        self.arbiter = CondDecidesArbiter(self.num_steps)
        self.decisions: list[Any] = []
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-prompt state; hooks stay installed."""

        self.arbiter.reset()
        self.decisions.clear()
        self._step_full = True
        self._step_reason = "uninitialized"
        self._timestep: float | None = None
        self._block_calls = 0
        self._cond_block_calls = 0
        self._reset_payload_state()

    def _reset_payload_state(self) -> None:
        raise NotImplementedError

    @property
    def step(self) -> int:
        return self.arbiter.step

    @property
    def branch(self) -> str:
        return self.arbiter.branch

    def __enter__(self) -> "WanCFGAdapter":
        if self._installed:
            raise RuntimeError("adapter already installed")
        self._handles.append(
            self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True)
        )
        self._handles.append(
            self.model.register_forward_hook(self._model_post, with_kwargs=True)
        )
        self._install_patches()
        self._installed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_instance_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _install_patches(self) -> None:
        for index, block in enumerate(self.model.blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._block_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))

    # -- the CFG clock -----------------------------------------------------

    def _model_pre(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _step, branch = self.arbiter.begin_forward()
        if branch == "cond":
            # Block calls are counted per *step*, across both forwards, so a
            # fully-computed step reads 60 = 30 blocks x 2 branches.
            self._block_calls = 0
            self._cond_block_calls = 0
        self._timestep = _forward_timestep(args, kwargs)

    def _model_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self.arbiter.is_cond:
            self._cond_block_calls = self._block_calls
            self._on_cond_forward_end()
        if self.arbiter.end_forward():
            self.decisions.append(self._decision_record())
        return output

    def _on_cond_forward_end(self) -> None:
        """Hook for adapters whose record fields are produced on the cond pass."""

    # -- to be provided by subclasses -------------------------------------

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _decision_record(self) -> Any:
        raise NotImplementedError

    def _common_record_fields(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": "full" if self._step_full else "cache",
            "reason": self._step_reason,
            "original_block_calls": self._block_calls,
            "cond_block_calls": self._cond_block_calls,
            "uncond_block_calls": self._block_calls - self._cond_block_calls,
            "timestep": self._timestep,
        }


# ---------------------------------------------------------------------------
# Coarse whole-transformer residual (SeaCache / TeaCache / SenCache / BudCache)
# ---------------------------------------------------------------------------


class CoarseBackboneAdapter(WanCFGAdapter):
    """Whole-transformer residual reuse on Wan's single block stack.

    One method object gates the step, driven on the cond branch only; the
    residual payload is held per branch, as the upstream Wan SenCache does
    (`knowledge/SenCache/code/Wan2.1/sencache.py:152-160`, separate
    `cached_residual_cond` / `cached_residual_uncond`) and as the in-repo
    golden-path lane already does (`wan21/seacache.py:119`).
    """

    def __init__(self, model: Any, method: Any, *, num_steps: int):
        self.method = method
        super().__init__(model, num_steps=num_steps)

    def _reset_payload_state(self) -> None:
        self.method.reset()
        self.previous_residual: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self._initial_x: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self._decision: MethodDecision | None = None

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        branch = self.branch
        if index == 0:
            x, e, grid_sizes = _block_inputs(args, kwargs)
            if self.arbiter.is_cond:
                decision = self.method.decide(
                    x=x,
                    e=e,
                    first_block=module,
                    grid_sizes=grid_sizes,
                )
                if not decision.full and self.previous_residual["cond"] is None:
                    decision = MethodDecision(
                        full=True,
                        reason="missing_payload_forced_full",
                        gate_scalar=decision.gate_scalar,
                        accumulated=decision.accumulated,
                        threshold=decision.threshold,
                    )
                self._decision = self.arbiter.settle(decision)
            else:
                self._decision = self.arbiter.follow()
            self._step_full = self._decision.full
            self._step_reason = self._decision.reason
            if not self._step_full:
                residual = self.previous_residual[branch]
                if residual is None:
                    raise RuntimeError(
                        f"Wan coarse cache at step {self.step} has no {branch} residual"
                    )
                return x + residual
            self._initial_x[branch] = x.clone()
        if not self._step_full:
            return args[0]
        self._block_calls += 1
        output = original(*args, **kwargs)
        if index == len(self.model.blocks) - 1:
            anchor = self._initial_x[branch]
            if anchor is None:
                raise RuntimeError("missing coarse residual anchor")
            self.previous_residual[branch] = (output - anchor).detach()
        return output

    def _decision_record(self) -> WanDecisionRecord:
        decision = self._decision
        if decision is None:
            raise RuntimeError("coarse adapter did not enter the first block")
        return WanDecisionRecord(
            gate_scalar=decision.gate_scalar,
            accumulated=decision.accumulated,
            threshold=decision.threshold,
            **self._common_record_fields(),
        )


class SenCacheAdapter(CoarseBackboneAdapter):
    """Coarse residual reuse whose gate reads the pre-`patch_embedding` latent.

    SenCache scores `x` and `t` as `WanModel.forward` receives them, and Wan
    hands both in as explicit arguments -- one layer shallower than Hunyuan,
    which has to be intercepted before `img_in`.  The model pre-hook forwards
    them to the gate before block 0 dispatches.  Both CFG branches see the same
    `z_k`, so the observation is taken on cond only, matching the upstream Wan
    variant's `is_cond` guard.
    """

    #: Measured on the first observed latent; see the check in `_model_pre`.
    latent_numel: int | None = None

    def _model_pre(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super()._model_pre(module, args, kwargs)
        if not self.arbiter.is_cond:
            return
        latent = _model_latent(args, kwargs)
        t = kwargs.get("t")
        if t is None and len(args) > 1:
            t = args[1]
        if t is None:
            raise RuntimeError("Wan SenCache gate did not receive the transformer timestep")
        # sqrt(d) multiplies the raw threshold, so d is protocol-defining: at
        # 832x480x65 the Wan-VAE latent is 16 x 17 x 60 x 104 = 1,697,280 and
        # sqrt(d) ~= 1302.80.  Checking it here is the "verify the arithmetic at
        # runtime" clause of plan section 2.5; a different shape means a
        # different protocol and a threshold that no longer means what was frozen.
        self.latent_numel = int(latent.numel())
        if self.latent_numel != WAN_LATENT_NUMEL:
            raise RuntimeError(
                f"Wan SenCache latent has {self.latent_numel} elements, "
                f"protocol expects {WAN_LATENT_NUMEL}"
            )
        self.method.observe_input(latent, t)


# ---------------------------------------------------------------------------
# Fine component payload (TaylorSeer O1 / HiCache O2)
# ---------------------------------------------------------------------------


class FineAdapter(WanCFGAdapter):
    """90-slot component payload over Wan's 30 uniform blocks.

    Slots are the three sub-module outputs each `WanAttentionBlock` adds back
    into the residual stream -- `self_attn`, `cross_attn`, `ffn` -- captured
    *before* their modulation gates, which is where the upstream TaylorSeer-Wan
    port takes them too (`wan21/taylorseer_fine.py:248-265`).  On a cached step
    the predictions are re-gated with this step's live modulation chunks:
    `self_attn` by `e[2]`, `ffn` by `e[5]`, `cross_attn` added ungated.

    Each CFG branch owns a method object, because the slot histories differ
    between cond and uncond; both run the same frozen table and the arbiter
    asserts they agree.
    """

    SUBMODULES = ("self_attn", "cross_attn", "ffn")

    def __init__(self, model: Any, methods: Mapping[str, Any], *, num_steps: int):
        if methods["cond"] is methods["uncond"]:
            raise ValueError("Wan fine payload needs one method object per CFG branch")
        self.methods = {"cond": methods["cond"], "uncond": methods["uncond"]}
        super().__init__(model, num_steps=num_steps)
        if self.slot_count != WAN_FINE_SLOT_COUNT:
            raise ValueError(
                f"Wan fine payload expects {WAN_FINE_SLOT_COUNT} slots, got {self.slot_count}"
            )

    @property
    def slot_count(self) -> int:
        return len(self.model.blocks) * len(self.SUBMODULES)

    @staticmethod
    def slot_name(index: int, submodule: str) -> str:
        return f"block.{index}.{submodule}"

    def _reset_payload_state(self) -> None:
        for method in self.methods.values():
            method.reset()
        self._decision: MethodDecision | None = None

    def _install_patches(self) -> None:
        for index, block in enumerate(self.model.blocks):
            for submodule in self.SUBMODULES:
                slot = self.slot_name(index, submodule)
                self._handles.append(
                    getattr(block, submodule).register_forward_hook(self._slot_hook(slot))
                )
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._block_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))

    def _slot_hook(self, slot: str):
        def hook(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            if self._step_full:
                self.methods[self.branch].update_slot(slot, output)

        return hook

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        method = self.methods[self.branch]
        if index == 0:
            decision = method.decide()
            if self.arbiter.is_cond:
                self._decision = self.arbiter.settle(decision)
            else:
                self._decision = self.arbiter.follow(decision)
            self._step_full = self._decision.full
            self._step_reason = self._decision.reason
        if self._step_full:
            self._block_calls += 1
            output = original(*args, **kwargs)
            if index == len(self.model.blocks) - 1:
                method.finish_full_step()
            return output

        x, e, _grid_sizes = _block_inputs(args, kwargs)
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            chunks = (module.modulation + e).chunk(6, dim=1)
        assert chunks[0].dtype == torch.float32
        # The same three adds `WanAttentionBlock.forward` performs, with the
        # forecast standing in for each sub-module's output.  Wan's chunks are
        # (B, 1, C) and broadcast over (B, L, C) as-is -- no `unsqueeze(1)`, in
        # contrast to the Hunyuan gates.
        with amp.autocast(dtype=torch.float32):
            x = x + method.predict_slot(self.slot_name(index, "self_attn")) * chunks[2]
        x = x + method.predict_slot(self.slot_name(index, "cross_attn"))
        with amp.autocast(dtype=torch.float32):
            x = x + method.predict_slot(self.slot_name(index, "ffn")) * chunks[5]
        return x

    def _decision_record(self) -> WanDecisionRecord:
        if self._decision is None:
            raise RuntimeError("fine adapter did not enter the first block")
        return WanDecisionRecord(**self._common_record_fields())


# ---------------------------------------------------------------------------
# Head-output substitution (L2P)
# ---------------------------------------------------------------------------


class HeadOutputAdapter(WanCFGAdapter):
    """Skip every block and predict Wan's `head` output.

    `model.head` is the final projection before `unpatchify`
    (`.../wan/modules/model.py:571-575`), i.e. the Wan twin of the Hunyuan
    `final_layer` that `L2POutputAdapter` patches.  Prediction history is
    per branch; the weight matrix is a single shared asset (plan section 2.8).
    """

    def __init__(self, model: Any, methods: Mapping[str, Any], *, num_steps: int):
        if methods["cond"] is methods["uncond"]:
            raise ValueError("Wan L2P needs one method object per CFG branch")
        self.methods = {"cond": methods["cond"], "uncond": methods["uncond"]}
        super().__init__(model, num_steps=num_steps)

    def _reset_payload_state(self) -> None:
        for method in self.methods.values():
            method.reset()
        self._decision: MethodDecision | None = None
        self._cond_fields: dict[str, Any] = {}

    def _install_patches(self) -> None:
        super()._install_patches()
        head = self.model.head

        def head_forward(module: Any, *args: Any, **kwargs: Any):
            return self._head_forward(module, *args, **kwargs)

        self._patches.append((head, _set_instance_forward(head, head_forward)))

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            decision = self.methods[self.branch].decide()
            if self.arbiter.is_cond:
                self._decision = self.arbiter.settle(decision)
            else:
                self._decision = self.arbiter.follow(decision)
            self._step_full = self._decision.full
            self._step_reason = self._decision.reason
        if not self._step_full:
            return args[0]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _head_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        original = self._patches[-1][1][1]
        method = self.methods[self.branch]
        if self._step_full:
            return method.final_output(original(*args, **kwargs))
        return method.final_output()

    def _on_cond_forward_end(self) -> None:
        # Both branches predict, but the row describes the step, so it carries
        # the cond branch's prediction diagnostics.
        self._cond_fields = dict(self.methods["cond"].last_prediction_fields)

    def _decision_record(self) -> WanL2PDecisionRecord:
        if self._decision is None:
            raise RuntimeError("L2P adapter did not enter the first block")
        fields = dict(self._cond_fields)
        if not fields:
            raise RuntimeError("L2P adapter did not reach the head")
        return WanL2PDecisionRecord(
            l2p_target=str(fields.pop("l2p_target")),
            l2p_weights_used=int(fields.pop("l2p_weights_used")),
            **fields,
            **self._common_record_fields(),
        )


# ---------------------------------------------------------------------------
# Velocity substitution (MeanCache)
# ---------------------------------------------------------------------------


def patchify(
    latent: torch.Tensor,
    patch_size: Sequence[int],
    out_dim: int,
    *,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Inverse of `WanModel.unpatchify` (`.../wan/modules/model.py:579-602`).

    `unpatchify` views one head-output row of `prod(patch_size) * out_dim`
    values as `(f, h, w, pt, ph, pw, c)` and einsums it to `(c, F, H, W)`.
    MeanCache substitutes that head output, so the latent it differences
    against has to be written in the same token layout -- and padded to
    `seq_len` the way `WanModel.forward` pads the embedded tokens, so the two
    sequences line up element for element.
    """

    pt, ph, pw = (int(value) for value in patch_size)
    value = latent if latent.dim() == 5 else latent.unsqueeze(0)
    n, c, of, oh, ow = value.shape
    if c != int(out_dim):
        raise ValueError(f"latent has {c} channels, head emits {int(out_dim)}")
    if of % pt or oh % ph or ow % pw:
        raise ValueError(f"latent {tuple(value.shape)} does not tile with patch {tuple(patch_size)}")
    f, h, w = of // pt, oh // ph, ow // pw
    value = value.reshape(n, c, f, pt, h, ph, w, pw)
    # (n c f p h q w r) -> (n f h w p q r c), mirroring the unpatchify einsum.
    value = value.permute(0, 2, 4, 6, 3, 5, 7, 1)
    tokens = value.reshape(n, f * h * w, pt * ph * pw * c)
    if seq_len is not None and int(seq_len) != tokens.shape[1]:
        if int(seq_len) < tokens.shape[1]:
            raise ValueError(f"seq_len {int(seq_len)} is shorter than {tokens.shape[1]} tokens")
        pad = tokens.new_zeros(n, int(seq_len) - tokens.shape[1], tokens.shape[2])
        tokens = torch.cat([tokens, pad], dim=1)
    return tokens


class MeanCacheVelocityAdapter(WanCFGAdapter):
    """Skip every block and substitute MeanCache's velocity at Wan's head.

    Sibling of `HeadOutputAdapter`: same block skipping and same injection
    point, but the payload is an average-velocity integration that needs the
    current latent in token layout and the solver sigmas.  Wan is flow matching,
    so the model output *is* the velocity and the head is the natural injection
    point.  Latents and sigmas are identical across the CFG branches, the
    velocities are not, so each branch keeps its own history.
    """

    def __init__(self, model: Any, methods: Mapping[str, Any], *, num_steps: int):
        if methods["cond"] is methods["uncond"]:
            raise ValueError("Wan MeanCache needs one method object per CFG branch")
        self.methods = {"cond": methods["cond"], "uncond": methods["uncond"]}
        super().__init__(model, num_steps=num_steps)

    def _reset_payload_state(self) -> None:
        # Reset hygiene matters here: the JVP proxy is estimated on the *current*
        # trajectory, so a history surviving into the next prompt would silently
        # predict from the previous video.
        for method in self.methods.values():
            method.reset()
        self._decision: MethodDecision | None = None
        self._latent_tokens: torch.Tensor | None = None
        self._cond_fields: dict[str, Any] = {}

    def _install_patches(self) -> None:
        super()._install_patches()
        head = self.model.head

        def head_forward(module: Any, *args: Any, **kwargs: Any):
            return self._head_forward(module, *args, **kwargs)

        self._patches.append((head, _set_instance_forward(head, head_forward)))

    def _model_pre(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super()._model_pre(module, args, kwargs)
        self._latent_tokens = patchify(
            _model_latent(args, kwargs).detach(),
            self.model.patch_size,
            self.model.out_dim,
            seq_len=kwargs.get("seq_len"),
        )

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            if self._latent_tokens is None:
                raise RuntimeError("model pre-hook did not capture the latent")
            decision = self.methods[self.branch].decide(latent=self._latent_tokens)
            if self.arbiter.is_cond:
                self._decision = self.arbiter.settle(decision)
            else:
                self._decision = self.arbiter.follow(decision)
            self._step_full = self._decision.full
            self._step_reason = self._decision.reason
        if not self._step_full:
            return args[0]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _head_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        original = self._patches[-1][1][1]
        method = self.methods[self.branch]
        if self._step_full:
            return method.final_output(original(*args, **kwargs))
        return method.final_output()

    def _on_cond_forward_end(self) -> None:
        self._cond_fields = dict(self.methods["cond"].last_payload_fields)

    def _decision_record(self) -> WanMeanCacheDecisionRecord:
        if self._decision is None:
            raise RuntimeError("MeanCache adapter did not enter the first block")
        fields = dict(self._cond_fields)
        if not fields:
            raise RuntimeError("MeanCache adapter did not reach the head")
        return WanMeanCacheDecisionRecord(**fields, **self._common_record_fields())
