"""Fine-grained TaylorSeer/HiCache scaffold for Qwen-Image.

The implementation is deliberately local to Qwen-Image:

* one diffusion-step decision is shared by the true-CFG cond/uncond forwards;
* feature histories are isolated by branch (`cond`, `uncond`);
* each Qwen block has four gate-pre slots:
  `img_attn`, `txt_attn`, `img_mlp`, `txt_mlp`.

Only `TaylorSeer_fine` and `HiCache_fine` are implemented here. Coarse methods
live in their own Qwen adapters.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

from lib.fixed_schedule import validate_cache_steps
from lib.hermite import hermite_update, hicache_predict
from lib.taylor import taylor_predict


BRANCHES = ("cond", "uncond")
MODULES = ("img_attn", "txt_attn", "img_mlp", "txt_mlp")


@dataclass(frozen=True)
class QwenFineConfig:
    mode: str
    num_steps: int = 50
    interval: int = 7
    first_enhance: int = 3
    max_order: int = 2
    hicache_sigma: float = 0.5
    true_cfg: bool = True
    cache_steps: tuple[int, ...] = ()


def install_qwen_fine_forward(pipe: Any, config: QwenFineConfig) -> None:
    """Install instance-level transformer/block patches on a QwenImagePipeline."""
    if config.mode not in {"TaylorSeer_fine", "HiCache_fine"}:
        raise ValueError(f"unsupported Qwen fine mode: {config.mode!r}")
    if config.cache_steps:
        validate_cache_steps(
            config.cache_steps,
            num_steps=int(config.num_steps),
            forced_full_steps=range(int(config.first_enhance)),
        )
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not hasattr(transformer, "transformer_blocks"):
        raise TypeError(
            "pipe.transformer.transformer_blocks is required for Qwen fine cache"
        )
    restore_qwen_fine_forward(pipe)
    state = _new_state(config, num_layers=len(transformer.transformer_blocks))
    transformer._qwen_image_fine_state = state
    transformer._qwen_image_original_forward = transformer.forward
    transformer.forward = types.MethodType(_transformer_forward, transformer)
    for layer, block in enumerate(transformer.transformer_blocks):
        block._qwen_image_layer_index = int(layer)
        block._qwen_image_parent_transformer = transformer
        block._qwen_image_original_forward = block.forward
        block.forward = types.MethodType(_block_forward, block)


def restore_qwen_fine_forward(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return
    original = getattr(transformer, "_qwen_image_original_forward", None)
    if original is not None:
        transformer.forward = original
        delattr(transformer, "_qwen_image_original_forward")
    for block in getattr(transformer, "transformer_blocks", []):
        original_block = getattr(block, "_qwen_image_original_forward", None)
        if original_block is not None:
            block.forward = original_block
            delattr(block, "_qwen_image_original_forward")
        for attr in ("_qwen_image_layer_index", "_qwen_image_parent_transformer"):
            if hasattr(block, attr):
                delattr(block, attr)
    if hasattr(transformer, "_qwen_image_fine_state"):
        delattr(transformer, "_qwen_image_fine_state")


def reset_qwen_fine_state(pipe: Any, *, prompt_idx: int, seed: int) -> None:
    transformer = getattr(pipe, "transformer", None)
    state = getattr(transformer, "_qwen_image_fine_state", None)
    if state is None:
        return
    cfg: QwenFineConfig = state["config"]
    transformer._qwen_image_fine_state = _new_state(
        cfg,
        num_layers=int(state["num_layers"]),
        prompt_idx=int(prompt_idx),
        seed=int(seed),
    )


def qwen_fine_decisions(pipe: Any) -> Dict[str, Any]:
    transformer = getattr(pipe, "transformer", None)
    state = getattr(transformer, "_qwen_image_fine_state", None)
    if state is None:
        return {}
    cfg: QwenFineConfig = state["config"]
    rows = []
    for step in range(int(cfg.num_steps)):
        meta = state["step_meta"].get(step, {"action": "missing", "reason": "missing"})
        branches = state["decisions"].get(step, {})
        rows.append(
            {
                "step": step,
                "action": meta.get("action"),
                "reason": meta.get("reason"),
                "branches": {branch: branches.get(branch) for branch in BRANCHES},
            }
        )
    return {
        "schema": "qwen_image_decisions.v1",
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "mode": cfg.mode,
        "granularity": "fine_240",
        "shared_step_action": True,
        "num_steps": int(cfg.num_steps),
        "interval": int(cfg.interval),
        "first_enhance": int(cfg.first_enhance),
        "max_order": int(cfg.max_order),
        "hicache_sigma": float(cfg.hicache_sigma),
        "schedule_kind": "fixed" if cfg.cache_steps else "interval",
        "cache_steps": list(cfg.cache_steps),
        "activated_steps": list(state["activated_steps"]),
        "steps": rows,
        "summary": _summary(state),
    }


def _new_state(
    config: QwenFineConfig,
    *,
    num_layers: int,
    prompt_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "config": config,
        "num_layers": int(num_layers),
        "prompt_idx": prompt_idx,
        "seed": seed,
        "forward_call_count": 0,
        "cache_counter": 0,
        "activated_steps": [],
        "step_meta": {},
        "decisions": {},
        "history": {branch: {} for branch in BRANCHES},
        "current_step": 0,
        "current_branch": "cond",
        "current_action": "full",
        "current_reason": "init",
        "current_cache_ready": False,
        "stats": {branch: {"full_count": 0, "cache_count": 0} for branch in BRANCHES},
    }


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg: QwenFineConfig = state["config"]
    step_actions = [
        state["step_meta"].get(step, {}).get("action")
        for step in range(int(cfg.num_steps))
    ]
    n_full = sum(1 for action in step_actions if action == "full")
    n_cached = sum(1 for action in step_actions if action == "cache")
    out: Dict[str, Any] = {
        "n_total": int(cfg.num_steps),
        "n_full": int(n_full),
        "n_cached": int(n_cached),
        "cache_ratio": (
            float(n_cached / int(cfg.num_steps)) if int(cfg.num_steps) > 0 else 0.0
        ),
        "branches": {},
    }
    for branch, stats in state["stats"].items():
        full = int(stats["full_count"])
        cache = int(stats["cache_count"])
        total = full + cache
        out["branches"][branch] = {
            "full_count": full,
            "cache_count": cache,
            "cache_ratio": float(cache / total) if total else 0.0,
        }
    return out


def _transformer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    state = self._qwen_image_fine_state
    cfg: QwenFineConfig = state["config"]
    branches_per_step = 2 if bool(cfg.true_cfg) else 1
    call_idx = int(state["forward_call_count"])
    step = call_idx // branches_per_step
    branch = "cond" if (branches_per_step == 1 or call_idx % 2 == 0) else "uncond"
    if step >= int(cfg.num_steps):
        step = int(cfg.num_steps) - 1

    if step not in state["step_meta"]:
        state["step_meta"][step] = _decide_step(state, step)

    meta = dict(state["step_meta"][step])
    cache_ready = _branch_cache_complete(state, branch)
    action = str(meta["action"])
    reason = str(meta["reason"])
    if action == "cache" and not cache_ready:
        action = "full"
        reason = "cache_not_ready"
        state["step_meta"][step] = {"action": "full", "reason": reason}
        if step not in state["activated_steps"]:
            state["activated_steps"].append(step)

    state["current_step"] = int(step)
    state["current_branch"] = branch
    state["current_action"] = action
    state["current_reason"] = reason
    state["current_cache_ready"] = bool(cache_ready)
    try:
        return self._qwen_image_original_forward(*args, **kwargs)
    finally:
        _record_branch_decision(state, step, branch, action, reason, cache_ready)
        state["forward_call_count"] = call_idx + 1


def _decide_step(state: Dict[str, Any], step: int) -> Dict[str, str]:
    cfg: QwenFineConfig = state["config"]
    if cfg.cache_steps:
        if step in cfg.cache_steps:
            return {"action": "cache", "reason": "fixed_schedule_cache"}
        state["activated_steps"].append(int(step))
        return {"action": "full", "reason": "fixed_schedule_full"}
    if step < int(cfg.first_enhance):
        state["cache_counter"] = 0
        state["activated_steps"].append(int(step))
        return {"action": "full", "reason": "warmup"}
    if int(state["cache_counter"]) == int(cfg.interval) - 1:
        state["cache_counter"] = 0
        state["activated_steps"].append(int(step))
        return {"action": "full", "reason": "interval_refresh"}
    state["cache_counter"] = int(state["cache_counter"]) + 1
    return {"action": "cache", "reason": "interval_cache"}


def _record_branch_decision(
    state: Dict[str, Any],
    step: int,
    branch: str,
    action: str,
    reason: str,
    cache_ready: bool,
) -> None:
    state["stats"][branch][f"{action}_count"] += 1
    state["decisions"].setdefault(int(step), {})[branch] = {
        "action": action,
        "reason": reason,
        "cache_ready": bool(cache_ready),
    }


def _branch_cache_complete(state: Dict[str, Any], branch: str) -> bool:
    history = state["history"].get(branch, {})
    num_layers = int(state["num_layers"])
    if len(history) < num_layers:
        return False
    for layer in range(num_layers):
        layer_hist = history.get(layer, {})
        for module in MODULES:
            slot_hist = layer_hist.get(module)
            if not slot_hist or 0 not in slot_hist:
                return False
    return True


def _slot_history(
    state: Dict[str, Any], branch: str, layer: int, module: str
) -> Dict[int, torch.Tensor]:
    branch_hist = state["history"].setdefault(branch, {})
    layer_hist = branch_hist.setdefault(int(layer), {})
    return layer_hist.setdefault(module, {})


def _update_slot(
    state: Dict[str, Any], *, layer: int, module: str, feature: torch.Tensor
) -> None:
    cfg: QwenFineConfig = state["config"]
    branch = str(state["current_branch"])
    step = int(state["current_step"])
    prev = _slot_history(state, branch, layer, module)
    if len(state["activated_steps"]) >= 2:
        step_gap = int(state["activated_steps"][-1]) - int(state["activated_steps"][-2])
    else:
        step_gap = 0
    feature_detached = feature.detach()
    if prev and step > int(cfg.first_enhance) - 2 and step_gap > 0:
        updated = hermite_update(prev, feature_detached, step_gap, int(cfg.max_order))
    else:
        updated = {0: feature_detached}
    state["history"][branch].setdefault(int(layer), {})[module] = updated


def _predict_slot(state: Dict[str, Any], *, layer: int, module: str) -> torch.Tensor:
    cfg: QwenFineConfig = state["config"]
    branch = str(state["current_branch"])
    step = int(state["current_step"])
    activated = state["activated_steps"]
    if not activated:
        raise RuntimeError("Qwen fine cache has no activated full step")
    step_offset = step - int(activated[-1])
    hist = _slot_history(state, branch, layer, module)
    if not hist or 0 not in hist:
        raise RuntimeError(
            f"missing Qwen fine history: branch={branch} layer={layer} module={module}"
        )
    if cfg.mode == "TaylorSeer_fine":
        return taylor_predict(
            hist, step_offset=step_offset, max_order=int(cfg.max_order)
        )
    if cfg.mode == "HiCache_fine":
        return hicache_predict(
            hist,
            step_offset=step_offset,
            sigma=float(cfg.hicache_sigma),
            max_order=int(cfg.max_order),
        )
    raise ValueError(f"unsupported mode: {cfg.mode!r}")


def _chunk_gate(mod_params: torch.Tensor) -> torch.Tensor:
    _, _, gate = mod_params.chunk(3, dim=-1)
    return gate.unsqueeze(1)


def _block_forward(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_hidden_states_mask = kwargs.pop("encoder_hidden_states_mask", None)
    temb = kwargs.pop("temb", None)
    image_rotary_emb = kwargs.pop("image_rotary_emb", None)
    joint_attention_kwargs = kwargs.pop(
        "joint_attention_kwargs", kwargs.pop("attention_kwargs", None)
    )
    if args:
        if len(args) >= 3:
            encoder_hidden_states_mask = (
                args[0]
                if encoder_hidden_states_mask is None
                else encoder_hidden_states_mask
            )
            temb = args[1] if temb is None else temb
            image_rotary_emb = args[2] if image_rotary_emb is None else image_rotary_emb
            if len(args) >= 4 and joint_attention_kwargs is None:
                joint_attention_kwargs = args[3]
        elif len(args) == 2:
            temb = args[0] if temb is None else temb
            image_rotary_emb = args[1] if image_rotary_emb is None else image_rotary_emb
        elif len(args) == 1:
            temb = args[0] if temb is None else temb
    if temb is None:
        raise TypeError("Qwen fine block wrapper requires temb")
    parent = getattr(self, "_qwen_image_parent_transformer")
    state = parent._qwen_image_fine_state
    layer = int(getattr(self, "_qwen_image_layer_index"))
    action = str(state["current_action"])

    img_mod_params = self.img_mod(temb)
    txt_mod_params = self.txt_mod(temb)
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

    img_normed1 = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed1, img_mod1)
    txt_normed1 = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed1, txt_mod1)
    joint_attention_kwargs = joint_attention_kwargs or {}

    if action == "full":
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        img_attn_output, txt_attn_output = attn_output
        _update_slot(state, layer=layer, module="img_attn", feature=img_attn_output)
        _update_slot(state, layer=layer, module="txt_attn", feature=txt_attn_output)
    elif action == "cache":
        img_attn_output = _predict_slot(state, layer=layer, module="img_attn")
        txt_attn_output = _predict_slot(state, layer=layer, module="txt_attn")
    else:
        raise ValueError(f"bad Qwen fine action: {action!r}")

    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    img_gate2 = _chunk_gate(img_mod2)
    txt_gate2 = _chunk_gate(txt_mod2)
    if action == "full":
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, _ = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        _update_slot(state, layer=layer, module="img_mlp", feature=img_mlp_output)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, _ = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        _update_slot(state, layer=layer, module="txt_mlp", feature=txt_mlp_output)
    else:
        img_mlp_output = _predict_slot(state, layer=layer, module="img_mlp")
        txt_mlp_output = _predict_slot(state, layer=layer, module="txt_mlp")

    hidden_states = hidden_states + img_gate2 * img_mlp_output
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def expected_full_steps(num_steps: int, interval: int, first_enhance: int) -> list[int]:
    state = _new_state(
        QwenFineConfig(
            mode="TaylorSeer_fine",
            num_steps=int(num_steps),
            interval=int(interval),
            first_enhance=int(first_enhance),
        ),
        num_layers=1,
    )
    for step in range(int(num_steps)):
        state["step_meta"][step] = _decide_step(state, step)
    return [
        step
        for step in range(int(num_steps))
        if state["step_meta"][step]["action"] == "full"
    ]


def flatten_decision_rows(decisions: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for row in decisions.get("steps", []):
        step = int(row.get("step", -1))
        for branch, entry in (row.get("branches") or {}).items():
            if entry is None:
                yield {"step": step, "branch": branch, "missing": True}
            else:
                payload = dict(entry)
                payload["step"] = step
                payload["branch"] = branch
                yield payload
