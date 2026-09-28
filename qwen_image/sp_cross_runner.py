#!/usr/bin/env python3
"""SPX (schedule x payload cross) generator for Qwen-Image.

Mirror of `flux/sp_cross_runner.py` on the Qwen-Image protocol constants of
`qwen_image/runner.py` (1328x1328, 50 steps, true CFG 4.0, bf16). One cell =
one forced schedule x one payload x one seed; the schedule is a
`num_steps`-character bitstring file (`'1'` = cached, `'0'` = full).

Qwen runs true CFG, so every denoising step issues two transformer calls
(cond, then uncond). Both branches share the step action and keep separate
payload histories -- the convention of every locked Qwen adapter
(`qwen_image/coarse_cache.py`, `qwen_image/dicache.py`,
`qwen_image/meancache.py`).

Payloads
--------
`reuse`          zero-order residual reuse (BudCache payload of coarse_cache.py)
`taylor_o1`      `lib.taylor.taylor_predict` order 1
`hermite_o2`     `lib.hermite.hicache_predict` order 2, sigma 0.5
`mean_avg_vel`   `qwen_image/meancache.py::QwenMeanCacheAdapter`, unchanged
                 (note: unlike the residual payloads it also advances its
                 latent history on cached steps - the adapter's own semantics)
                 (its cells keep the adapter's own decisions schema/mode names;
                 SPX identity lives in the cell dir + timing `spx_payload`)
`di_two_anchor`  `lib.dicache.aligned_residual`, gamma clamped to [1, 1.5]

Warmup under an arbitrary schedule: histories advance only on full steps with
`step_gap` measured against that branch's most recent full step, and the
locked math functions fall back to the highest available order when the
history is short. Step 0 is always full, so every cache step has an anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from lib.dicache import aligned_residual, append_anchor  # noqa: E402
from lib.fixed_schedule import validate_cache_steps  # noqa: E402
from lib.hermite import hermite_update, hicache_predict  # noqa: E402
from lib.io_utils import (  # noqa: E402
    carry_forward_timing_records,
    read_prompts,
    split_shard,
    write_timing_json,
)
from lib.retained_trajectory import (  # noqa: E402
    install_z_t_capture,
    make_step_collector,
    parse_prompt_indices,
    save_trajectory,
    scheduler_sigmas,
    select_pairs,
    stack_trajectory,
    trajectory_filename,
)
from lib.taylor import taylor_predict  # noqa: E402
from qwen_image._helpers import (  # noqa: E402
    DEFAULT_PROMPT_FILE,
    atomic_write_json,
    decisions_filename,
    git_sha,
    image_filename,
    read_prompt_shard,
    seed_for,
    sha256_file,
)


BRANCHES = ("cond", "uncond")
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
RESIDUAL_PAYLOADS = ("reuse", "taylor_o1", "hermite_o2")


def load_schedule_file(path: Path, *, num_steps: int) -> tuple[int, ...]:
    """Read a `num_steps`-character '0'/'1' schedule file into cache steps."""

    bits = Path(path).read_text(encoding="utf-8").strip()
    if len(bits) != int(num_steps) or set(bits) - {"0", "1"}:
        raise SystemExit(
            f"schedule file {path} must hold {num_steps} characters of 0/1"
        )
    return tuple(step for step, bit in enumerate(bits) if bit == "1")


def _set_forward(module: Any, function: Callable[..., Any]) -> tuple[bool, Any]:
    had_instance = "forward" in module.__dict__
    original = module.forward
    module.forward = types.MethodType(function, module)
    return had_instance, original


def _restore_forward(module: Any, state: tuple[bool, Any]) -> None:
    had_instance, original = state
    if had_instance:
        module.forward = original
    else:
        delattr(module, "forward")


def _states(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    encoder = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
    if hidden is None or encoder is None:
        raise RuntimeError("Qwen SPX block wrapper could not find block inputs")
    return hidden, encoder


class _QwenSPCrossBase:
    """Shared schedule bookkeeping for the Qwen SPX block-level adapters."""

    schema = "qwen_image_sp_cross_decisions.v1"

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        payload: str,
        num_steps: int,
        true_cfg: bool,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.payload = str(payload)
        self.num_steps = int(num_steps)
        self.true_cfg = bool(true_cfg)
        self.num_layers = len(self.transformer.transformer_blocks)
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Qwen SPX adapter already installed")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.transformer_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._block_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_forward(block, wrapped)))
        self._installed = True

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _pre(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        branches_per_step = 2 if self.true_cfg else 1
        call = int(self.forward_call_count)
        self.step = min(call // branches_per_step, self.num_steps - 1)
        self.branch = "cond" if branches_per_step == 1 or call % 2 == 0 else "uncond"
        self.action = "cache" if self.step in self._cache_steps else "full"
        self._begin_step()

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        row = self.steps.setdefault(
            self.step,
            {
                "step": int(self.step),
                "action": self.action,
                "u": int(self.action == "cache"),
                "payload": self.payload,
                "branches": {},
            },
        )
        row["branches"][self.branch] = self._branch_fields()
        self.forward_call_count += 1
        return output

    def _begin_step(self) -> None:
        raise NotImplementedError

    def _branch_fields(self) -> dict[str, Any]:
        raise NotImplementedError

    def _block_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def decisions(self) -> dict[str, Any]:
        rows = [
            self.steps.get(
                step,
                {
                    "step": step,
                    "action": "missing",
                    "u": 0,
                    "payload": self.payload,
                    "branches": {},
                },
            )
            for step in range(self.num_steps)
        ]
        cached = sum(int(row["u"]) for row in rows)
        if cached != len(self.cache_steps):
            raise RuntimeError(
                f"Qwen SPX recorded K={cached}, expected K={len(self.cache_steps)}"
            )
        return {
            "schema": self.schema,
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "sp_cross",
            "payload": self.payload,
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "shared_step_action": True,
            **self.config_fields(),
            "steps": rows,
            "summary": {
                "n_total": self.num_steps,
                "n_full": self.num_steps - cached,
                "n_cached": cached,
                "cache_ratio": cached / self.num_steps,
            },
        }

    def config_fields(self) -> dict[str, Any]:
        return {}


class QwenSPCrossResidualAdapter(_QwenSPCrossBase):
    """Forced schedule with a whole-transformer residual payload per CFG branch."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        payload: str,
        num_steps: int = 50,
        hicache_sigma: float = 0.5,
        hicache_max_order: int = 2,
        taylorseer_max_order: int = 1,
        true_cfg: bool = True,
    ) -> None:
        if payload not in RESIDUAL_PAYLOADS:
            raise ValueError(f"unsupported residual payload: {payload!r}")
        self.hicache_sigma = float(hicache_sigma)
        self.hicache_max_order = int(hicache_max_order)
        self.taylorseer_max_order = int(taylorseer_max_order)
        self.history_max_order = max(self.hicache_max_order, self.taylorseer_max_order)
        super().__init__(
            pipe,
            cache_steps=cache_steps,
            payload=payload,
            num_steps=num_steps,
            true_cfg=true_cfg,
        )
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.step = 0
        self.branch = "cond"
        self.action = "full"
        self.steps: dict[int, dict[str, Any]] = {}
        self.branch_state: dict[str, dict[str, Any]] = {
            branch: {"history": {}, "last_full_step": None} for branch in BRANCHES
        }
        self._body_input: torch.Tensor | None = None
        self._step_offset: int | None = None
        self._order_used: int | None = None

    def _begin_step(self) -> None:
        self._body_input = None
        self._step_offset = None
        self._order_used = None

    def _branch_fields(self) -> dict[str, Any]:
        state = self.branch_state[self.branch]
        return {
            "action": self.action,
            "last_full_step": state["last_full_step"],
            "step_offset": self._step_offset,
            "order_used": self._order_used,
            "order_available": (len(state["history"]) - 1) if state["history"] else -1,
        }

    def _predict(self) -> torch.Tensor:
        state = self.branch_state[self.branch]
        history = state["history"]
        if not history:
            raise RuntimeError("Qwen SPX residual payload has no anchor")
        offset = int(self.step) - int(state["last_full_step"])
        self._step_offset = offset
        order_avail = len(history) - 1
        if self.payload == "reuse":
            self._order_used = 0
            return history[0]
        if self.payload == "taylor_o1":
            self._order_used = min(self.taylorseer_max_order, order_avail)
            return taylor_predict(
                history, step_offset=offset, max_order=self.taylorseer_max_order
            )
        self._order_used = min(self.hicache_max_order, order_avail)
        return hicache_predict(
            history,
            step_offset=offset,
            sigma=self.hicache_sigma,
            max_order=self.hicache_max_order,
        )

    def _update(self, residual: torch.Tensor) -> None:
        state = self.branch_state[self.branch]
        last_full = state["last_full_step"]
        gap = 1 if last_full is None else int(self.step) - int(last_full)
        state["history"] = hermite_update(
            state["history"],
            residual.detach(),
            step_gap=gap,
            max_order=self.history_max_order,
        )
        state["last_full_step"] = int(self.step)

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        hidden, encoder = _states(args, kwargs)
        if index == 0:
            self._body_input = hidden.detach()
            if self.action == "cache":
                return encoder, hidden + self._predict()
        if self.action == "cache":
            return encoder, hidden
        output = original(*args, **kwargs)
        if index == self.num_layers - 1:
            if self._body_input is None:
                raise RuntimeError("Qwen SPX full step is missing its block-0 input")
            self._update(output[1] - self._body_input)
        return output

    def config_fields(self) -> dict[str, Any]:
        return {
            "granularity": "coarse_residual",
            "hicache_sigma": self.hicache_sigma,
            "hicache_max_order": self.hicache_max_order,
            "taylorseer_max_order": self.taylorseer_max_order,
        }


class QwenSPCrossDiCacheAdapter(_QwenSPCrossBase):
    """Forced schedule with DiCache's two-anchor payload per CFG branch.

    Mirrors `qwen_image/dicache.py` with the accumulated-error gate removed:
    the schedule decides the action, the shallow probe still runs at cache
    steps to produce `gamma`, and the anchors are that branch's two most
    recent full steps.
    """

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        probe_depth: int = 1,
        true_cfg: bool = True,
    ) -> None:
        self.probe_depth = int(probe_depth)
        super().__init__(
            pipe,
            cache_steps=cache_steps,
            payload="di_two_anchor",
            num_steps=num_steps,
            true_cfg=true_cfg,
        )
        if not 1 <= self.probe_depth <= self.num_layers:
            raise ValueError("Qwen DiCache probe_depth is outside the block stack")
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.step = 0
        self.branch = "cond"
        self.action = "full"
        self.steps: dict[int, dict[str, Any]] = {}
        self.branch_state: dict[str, dict[str, Any]] = {
            branch: {
                "residual_history": [],
                "probe_history": [],
                "last_full_step": None,
            }
            for branch in BRANCHES
        }
        self._initial_hidden: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._gamma: float | None = None
        self._block_calls = 0

    def _begin_step(self) -> None:
        self._initial_hidden = None
        self._current_probe = None
        self._gamma = None
        self._block_calls = 0

    def _branch_fields(self) -> dict[str, Any]:
        state = self.branch_state[self.branch]
        return {
            "action": self.action,
            "last_full_step": state["last_full_step"],
            "gamma": self._gamma,
            "n_anchors": len(state["residual_history"]),
            "original_block_calls": int(self._block_calls),
        }

    def _run_probe(
        self,
        hidden: torch.Tensor,
        encoder: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        probe_hidden = hidden.clone()
        probe_encoder = encoder.clone()
        for index in range(self.probe_depth):
            original = self._patches[index][1][1]
            probe_kwargs = dict(kwargs)
            probe_args = list(args)
            if "hidden_states" in probe_kwargs:
                probe_kwargs["hidden_states"] = probe_hidden
                probe_kwargs["encoder_hidden_states"] = probe_encoder
            else:
                probe_args[0] = probe_hidden
                probe_args[1] = probe_encoder
            probe_encoder, probe_hidden = original(*probe_args, **probe_kwargs)
            self._block_calls += 1
        return probe_hidden

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        hidden, encoder = _states(args, kwargs)
        state = self.branch_state[self.branch]
        if index == 0:
            self._initial_hidden = hidden.detach()
            if self.action == "cache":
                if not state["residual_history"]:
                    raise RuntimeError("Qwen SPX DiCache cache step has no anchor")
                probe_hidden = self._run_probe(hidden, encoder, args, kwargs)
                payload, gamma = aligned_residual(
                    probe_hidden - hidden,
                    state["residual_history"],
                    state["probe_history"],
                )
                self._gamma = gamma
                return encoder, hidden + payload
        if self.action == "cache":
            return encoder, hidden
        output = original(*args, **kwargs)
        self._block_calls += 1
        if index == self.probe_depth - 1:
            self._current_probe = output[1].detach()
        if index == self.num_layers - 1:
            if self._initial_hidden is None or self._current_probe is None:
                raise RuntimeError("Qwen SPX DiCache full step is missing an anchor")
            append_anchor(state["residual_history"], output[1] - self._initial_hidden)
            append_anchor(
                state["probe_history"], self._current_probe - self._initial_hidden
            )
            state["last_full_step"] = int(self.step)
        return output

    def config_fields(self) -> dict[str, Any]:
        return {"granularity": "coarse_residual", "probe_depth": self.probe_depth}


def load_meancache_spans(args, cache_steps):
    """Per-edge spans from a frozen MeanCache solution file, or None.

    The matrix's frozen MeanCache schedules carry `jvp_spans` next to their
    `cache_steps` because the spans are half of the solved path
    (`flux/meancache_calibrate.py` sweeps them per edge). A scalar
    --meancache_jvp_span cannot express that, so the W1b homologous cell
    meancache x mean_avg_vel would not be the frozen method's payload without
    this. The file's cache_steps must equal the bits this cell runs: spans
    solved for one schedule applied to another is neither method nor payload,
    which is exactly the attribution ambiguity P4 forbids.
    """
    spans_file = getattr(args, "meancache_jvp_spans", None)
    if spans_file is None:
        return None
    payload = json.loads(Path(spans_file).read_text(encoding="utf-8"))
    spans = payload.get("jvp_spans")
    if not isinstance(spans, dict) or not spans:
        raise SystemExit(f"{spans_file} carries no jvp_spans")
    frozen_steps = payload.get("cache_steps")
    if frozen_steps is not None and \
            [int(step) for step in frozen_steps] != sorted(int(s) for s in cache_steps):
        raise SystemExit(
            f"{spans_file} solves a different schedule than the "
            f"--schedule_file bits; per-edge spans do not transfer across schedules")
    return {int(step): int(span) for step, span in spans.items()}


def build_adapter(
    pipe: Any,
    args: argparse.Namespace,
    cache_steps: tuple[int, ...],
) -> Any:
    if args.payload in RESIDUAL_PAYLOADS:
        adapter = QwenSPCrossResidualAdapter(
            pipe,
            cache_steps=cache_steps,
            payload=args.payload,
            num_steps=args.num_steps,
            hicache_sigma=args.hicache_sigma,
            hicache_max_order=args.hicache_max_order,
            taylorseer_max_order=args.taylorseer_max_order,
        )
    elif args.payload == "di_two_anchor":
        adapter = QwenSPCrossDiCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            probe_depth=args.dicache_probe_depth,
        )
    elif args.payload == "mean_avg_vel":
        from qwen_image.meancache import QwenMeanCacheAdapter

        adapter = QwenMeanCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            jvp_span=args.meancache_jvp_span,
            jvp_spans=load_meancache_spans(args, cache_steps),
            true_cfg=True,
            true_cfg_scale=args.true_cfg_scale,
        )
    else:
        raise AssertionError(f"unhandled payload: {args.payload}")
    return adapter


def _save_image(image: Any, path: Path) -> None:
    tmp = path.with_name(f"{path.stem}.tmp.{time.time_ns()}{path.suffix}")
    image.save(tmp)
    tmp.replace(path)


def _image_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule_file", type=Path, required=True)
    parser.add_argument("--payload", choices=PAYLOADS, required=True)
    parser.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hicache_sigma", type=float, default=0.5)
    parser.add_argument("--hicache_max_order", type=int, default=2)
    parser.add_argument("--taylorseer_max_order", type=int, default=1)
    parser.add_argument("--meancache_jvp_span", type=int, default=4)
    parser.add_argument(
        "--meancache_jvp_spans", type=Path, default=None,
        help="frozen MeanCache solution JSON whose per-edge jvp_spans override "
             "the scalar span; its cache_steps must equal the --schedule_file "
             "bits, or the cell is not the frozen method's payload")
    parser.add_argument("--dicache_probe_depth", type=int, default=1)
    parser.add_argument(
        "--retain_trajectory", action="store_true",
        help="also write latents_<idx>.pt, the [num_steps+1, d] path this "
             "generation walked (docs/image_cached_trajectory_plan_zh.md 7.1). "
             "Off by default; the SPX cells were generated without it.")
    parser.add_argument(
        "--retain_dtype", choices=("bf16", "fp32"), default="bf16",
        help="store dtype of the retained path. fp32 is the small dual-store "
             "subset that bounds what the store does to a direction.")
    parser.add_argument(
        "--prompt_indices", default=None,
        help="comma-separated GLOBAL prompt indices (ranges 'a-b' allowed) to "
             "run instead of the whole file. Indices are never renumbered: the "
             "seed is base_seed + global_idx, so renumbering would change the "
             "noise and break the pairing with the stored cell.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen-Image protocol requires --true_cfg_scale > 1")
    cache_steps = load_schedule_file(args.schedule_file, num_steps=args.num_steps)
    if args.prompt_indices is not None and args.limit:
        raise SystemExit("--prompt_indices and --limit select prompts two ways; pick one")
    indices = None
    if args.prompt_indices is None:
        shard_prompts, start_idx, _prompt_count = read_prompt_shard(
            args.prompt_file,
            limit=args.limit,
            shard_idx=args.shard_idx,
            shard_count=args.shard_count,
        )
        selected = [(start_idx + i, p) for i, p in enumerate(shard_prompts)]
    else:
        all_prompts = read_prompts(args.prompt_file)
        indices = parse_prompt_indices(args.prompt_indices, total=len(all_prompts))
        pairs = select_pairs(all_prompts, indices=indices)
        lo, hi = split_shard(len(pairs), args.shard_count, args.shard_idx)
        selected = pairs[lo:hi]
    if not selected:
        print(
            f"[qwen-spx] shard {args.shard_idx}/{args.shard_count} is empty, exiting.",
            flush=True,
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from diffusers import QwenImagePipeline

    load_start = time.perf_counter()
    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    load_s = time.perf_counter() - load_start

    adapter = build_adapter(pipe, args, cache_steps)
    adapter.install()
    z_t_holder = install_z_t_capture(pipe) if args.retain_trajectory else None
    per_image: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    try:
        for idx, prompt in selected:
            out_path = args.output_dir / image_filename(idx)
            dec_path = args.output_dir / decisions_filename(idx)
            traj_path = args.output_dir / trajectory_filename(idx)
            if (
                args.resume
                and _image_complete(out_path)
                and dec_path.is_file()
                and (not args.retain_trajectory or traj_path.is_file())
            ):
                print(f"[qwen-spx] resume skip idx={idx}", flush=True)
                continue
            seed = seed_for(args.seed, idx)
            adapter.reset(prompt_idx=idx, seed=seed)
            generator = torch.Generator(device=device).manual_seed(int(seed))
            trace = step_cb = None
            retain_kwargs: dict[str, Any] = {}
            if args.retain_trajectory:
                z_t_holder.pop("z_T", None)
                trace, step_cb = make_step_collector()
                retain_kwargs = {
                    "callback_on_step_end": step_cb,
                    "callback_on_step_end_tensor_inputs": ["latents"],
                }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            denoise_start = time.perf_counter()
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                true_cfg_scale=float(args.true_cfg_scale),
                height=int(args.height),
                width=int(args.width),
                num_inference_steps=int(args.num_steps),
                generator=generator,
                return_dict=True,
                output_type="pil",
                **retain_kwargs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            denoise_s = time.perf_counter() - denoise_start
            images = getattr(result, "images", None)
            if not images:
                raise RuntimeError(f"Qwen-Image returned no image for idx={idx}")
            traj_meta: dict[str, Any] | None = None
            if args.retain_trajectory:
                z_T = z_t_holder.pop("z_T", None)
                Z = stack_trajectory(z_T, trace, num_steps=args.num_steps)
                traj_meta = save_trajectory(
                    traj_path,
                    Z=Z,
                    latent_shape=list(z_T.shape),
                    sigmas=scheduler_sigmas(pipe, num_steps=args.num_steps),
                    prompt_idx=idx,
                    seed=seed,
                    num_steps=args.num_steps,
                    store_dtype=args.retain_dtype,
                    extra={
                        "backbone": "qwen_image",
                        "spx_payload": args.payload,
                        "schedule_name": args.schedule_file.stem,
                        "cache_steps": list(cache_steps),
                    },
                )
                del Z, z_T, trace
            decisions = adapter.decisions()
            decisions.update(
                {
                    "prompt": prompt,
                    "image_file": out_path.name,
                    "width": int(args.width),
                    "height": int(args.height),
                    "true_cfg_scale": float(args.true_cfg_scale),
                    "schedule_file": str(args.schedule_file),
                    "spx_payload": args.payload,
                }
            )
            _save_image(images[0], out_path)
            atomic_write_json(dec_path, decisions)
            per_image.append(
                {
                    "idx": int(idx),
                    "seed": int(seed),
                    "trajectory_file": (traj_path.name if traj_meta else None),
                    "z_T_sha256": (traj_meta or {}).get("z_T_sha256"),
                    "denoise_s": float(denoise_s),
                    "decode_s": 0.0,
                    "n_cached": int(decisions["summary"]["n_cached"]),
                    "cache_ratio": float(decisions["summary"]["cache_ratio"]),
                    "image_file": out_path.name,
                    "decisions_file": dec_path.name,
                }
            )
            print(
                f"[qwen-spx] payload={args.payload} idx={idx} "
                f"denoise={denoise_s:.2f}s",
                flush=True,
            )
    finally:
        adapter.restore()
        if z_t_holder is not None:
            z_t_holder["_restore"]()

    timing_path = (
        args.output_dir
        / f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json"
    )
    if args.resume and not per_image and timing_path.is_file():
        print(f"[qwen-spx] resume preserved existing timing shard: {timing_path}", flush=True)
        return 0

    carried = 0
    if args.resume:
        merged_records = carry_forward_timing_records(timing_path, per_image)
        carried = len(merged_records) - len(per_image)
        if carried:
            print(
                f"[qwen-spx] resume carried forward {carried} timing record(s) "
                f"from {timing_path.name}",
                flush=True,
            )
        per_image = merged_records

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    write_timing_json(
        timing_path,
        per_image=per_image,
        config={
            "cache_mode": f"sp_cross_{args.payload}",
            "mode": "sp_cross",
            "backbone": "qwen_image",
            "spx_payload": args.payload,
            "schedule_file": str(args.schedule_file),
            "schedule_file_sha256": sha256_file(args.schedule_file),
            "schedule_name": args.schedule_file.stem,
            "num_steps": int(args.num_steps),
            "cache_steps": list(cache_steps),
            "target_cache_count": len(cache_steps),
            "hicache_sigma": float(args.hicache_sigma),
            "hicache_max_order": int(args.hicache_max_order),
            "taylorseer_max_order": int(args.taylorseer_max_order),
            "meancache_jvp_span": int(args.meancache_jvp_span),
            "meancache_jvp_spans_file": (
                str(args.meancache_jvp_spans) if args.meancache_jvp_spans else None
            ),
            "dicache_probe_depth": int(args.dicache_probe_depth),
            "width": int(args.width),
            "height": int(args.height),
            "true_cfg_scale": float(args.true_cfg_scale),
            "dtype": args.dtype,
            "model_id": args.model_id,
            "base_seed": int(args.seed),
            "seed_rule": "base_plus_prompt_idx",
            "shard_idx": int(args.shard_idx),
            "shard_count": int(args.shard_count),
            "resume_carried_records": int(carried),
            "retain_trajectory": bool(args.retain_trajectory),
            "retain_dtype": (args.retain_dtype if args.retain_trajectory else None),
            "prompt_indices": (list(indices) if indices is not None else None),
            "timing_scope": "monolithic_pipeline_call_stored_as_denoise_s",
            "git_sha": git_sha(_ROOT),
        },
        model_load_s=load_s,
        wallclock_total_s=time.perf_counter() - wall_start + load_s,
        device=device_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
