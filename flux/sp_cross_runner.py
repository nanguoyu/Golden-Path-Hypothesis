#!/usr/bin/env python3
"""SPX (schedule x payload cross) generator for FLUX.

One cell = one forced schedule x one payload x one seed. The schedule comes
from a text file holding a single `num_steps`-character bitstring
(`'1'` = cached step, `'0'` = full step, see
`analysis/build_sp_cross_schedules.py`); the payload selects what replaces the
skipped computation. Locked baseline implementations are not modified: the
three residual payloads run through `flux/oracle_runner.py::install_oracle`
(the same engine `flux/fixed_residual_exact.py` and `flux/schedule_runner.py`
already use for forced schedules), `mean_avg_vel` reuses
`flux/meancache_exact.py::FluxMeanCacheAdapter` unchanged (note: those cells
therefore write decisions files under the adapter's own schema/mode names —
`flux_meancache_exact_decisions.v1` / `meancache_exact`; the SPX identity
lives in the cell directory name and timing.json `spx_payload`), and
`di_two_anchor` is a schedule-driven adapter over
`lib/dicache.py::aligned_residual` with DiCache's own gate removed.

Payloads
--------
`reuse`          zero-order residual reuse (SeaCache/TeaCache/BudCache payload)
`taylor_o1`      `lib.taylor.taylor_predict` order 1 (TaylorSeer O1)
`hermite_o2`     `lib.hermite.hicache_predict` order 2, sigma 0.5 (HiCache)
`mean_avg_vel`   MeanCache interval average velocity + trajectory-internal JVP
`di_two_anchor`  DiCache two-anchor gamma-clamped extrapolation, gamma in [1, 1.5]

Warmup under an arbitrary schedule
----------------------------------
Predictor state advances only on full steps for the residual payloads
(`install_oracle` with `reuse` / `taylor_o1` / `hermite_o2`) and for
`di_two_anchor` (`aligned_residual`'s two most recent full-step anchors),
with each method's own update rule and `step_gap` / `step_offset` measured
against the most recent full step. `mean_avg_vel` is the exception: the locked
MeanCache adapter is reused unchanged and it also advances its latent history
on cached steps (`flux/meancache_exact.py`) — its own semantics, kept rather
than altered. When the history is too short for the nominal order, the locked
math functions themselves fall back to the highest available order
(`min(max_order, len(history) - 1)` in `taylor_predict`/`hicache_predict`,
`residual_history[-1]` in `aligned_residual`, `velocities[-1]` in MeanCache).
No `first_enhance` warmup counter is imposed on top: a forced schedule has no
such parameter, and step 0 is always full so every cache step has an anchor.

Weights identity
----------------
`--revision` pins the model snapshot (`from_pretrained(revision=...)`). The
golden-path schedules are constructed from the trajectory campaign's geometry,
so a screening run must load the same weights that campaign used; the DGX
launcher passes the pinned revision for exactly that reason. Whatever the pin
resolved to is written into `timing.json` as `model_commit` (plus the requested
`model_revision` and how it was resolved), so a finished run can be checked
against the campaign snapshot after the fact instead of being taken on trust.

Output convention is the matrix gen step: `img_<idx>.png`,
`decisions_<idx>.json`, `timing_shard<i>of<n>.json`, so
`evaluation/eval_metrics.py` and `RUN/slurm_eval.sh` work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.dicache import aligned_residual, append_anchor  # noqa: E402
from lib.fixed_schedule import validate_cache_steps  # noqa: E402
from lib.io_utils import (  # noqa: E402
    carry_forward_timing_records,
    image_filename,
    read_prompts,
    seed_for,
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


PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
ORACLE_CACHE_MODE = {
    "reuse": "seacache",
    "taylor_o1": "taylorseer",
    "hermite_o2": "hicache",
}


def load_schedule_file(path: Path, *, num_steps: int) -> tuple[int, ...]:
    """Read a `num_steps`-character '0'/'1' schedule file into cache steps."""

    bits = Path(path).read_text(encoding="utf-8").strip()
    if len(bits) != int(num_steps) or set(bits) - {"0", "1"}:
        raise SystemExit(
            f"schedule file {path} must hold {num_steps} characters of 0/1"
        )
    return tuple(step for step, bit in enumerate(bits) if bit == "1")


def resolve_model_commit(model_id: str, revision: str | None) -> dict[str, Any]:
    """What the `--revision` pin actually resolves to, for `timing.json`.

    The pin says what was asked for; this says what was served, which is the
    part that can drift silently (a branch name moves, and the short sha the
    plan pins is not itself a cache directory name). Resolution is a file lookup
    for `model_index.json`: the HF cache answers offline for a full sha or a
    branch, and the hub expands a short sha with one metadata call. A failure is
    recorded as `unresolved` rather than guessed.
    """

    record: dict[str, Any] = {
        "model_revision": revision,
        "model_commit": None,
        "model_commit_source": "unresolved",
    }
    if Path(model_id).is_dir():
        record["model_commit_source"] = "local_dir"
        return record
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError:
        return record
    for source, lookup in (
        ("hf_cache", try_to_load_from_cache),
        ("hf_hub", hf_hub_download),
    ):
        try:
            resolved = lookup(model_id, "model_index.json", revision=revision)
        except Exception:  # noqa: BLE001 - offline, gated repo, unknown revision
            continue
        # `.../snapshots/<commit sha>/model_index.json`; anything else is not a
        # commit and must not be recorded as one.
        if isinstance(resolved, str) and Path(resolved).is_file():
            commit = Path(resolved).parent.name
            if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                record["model_commit"] = commit
                record["model_commit_source"] = source
                return record
    return record


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


class FluxSPCrossResidualAdapter:
    """Forced schedule with a whole-transformer residual payload."""

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
    ) -> None:
        if payload not in ORACLE_CACHE_MODE:
            raise ValueError(f"unsupported residual payload: {payload!r}")
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.payload = str(payload)
        self.cache_mode = ORACLE_CACHE_MODE[self.payload]
        self.num_steps = int(num_steps)
        self.hicache_sigma = float(hicache_sigma)
        self.hicache_max_order = int(hicache_max_order)
        self.taylorseer_max_order = int(taylorseer_max_order)
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
        # Imported lazily: flux/oracle_runner.py patches the diffusers FLUX
        # transformer class at import time, and the CPU test path never loads it.
        from flux.oracle_runner import reset_oracle_state

        reset_oracle_state(self.pipe)
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.last_full_step: int | None = None
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        from flux.oracle_runner import install_oracle

        self._teardown = install_oracle(
            self.pipe,
            cache_steps=self.cache_steps,
            num_steps=self.num_steps,
            cache_mode=self.cache_mode,
            hicache_sigma=self.hicache_sigma,
            hicache_max_order=self.hicache_max_order,
            taylorseer_max_order=self.taylorseer_max_order,
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
                "payload": self.payload,
                "last_full_step": self.last_full_step,
                "step_offset": (
                    None
                    if self.last_full_step is None
                    else int(self.step) - int(self.last_full_step)
                ),
            }
        )
        if not cache:
            self.last_full_step = int(self.step)
        self.step += 1
        return output

    def decisions(self) -> dict[str, Any]:
        return _decisions_payload(self, extra={"cache_mode": self.cache_mode})


class FluxSPCrossDiCacheAdapter:
    """Forced schedule with DiCache's two-anchor trajectory-aligned payload.

    Mirrors `flux/dicache_native.py` with the accumulated-error gate removed:
    the schedule decides the action, the shallow probe still runs at cache
    steps to produce `gamma`, and the two anchors are the two most recent full
    steps (`lib/dicache.py::aligned_residual`, gamma clamped to [1, 1.5]).
    """

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        probe_depth: int = 1,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.probe_depth = int(probe_depth)
        if not 1 <= self.probe_depth <= len(self.transformer.transformer_blocks):
            raise ValueError("DiCache probe_depth is outside the FLUX double-block stack")
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.last_full_step: int | None = None
        self.residual_history: list[torch.Tensor] = []
        self.probe_history: list[torch.Tensor] = []
        self.records: list[dict[str, Any]] = []
        self._current_action = "full"
        self._current_gamma: float | None = None
        self._initial_hidden: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._block_calls = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX SPX DiCache adapter already installed")
        blocks = [
            *self.transformer.transformer_blocks,
            *self.transformer.single_transformer_blocks,
        ]
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )
        for index, block in enumerate(blocks):
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
        cache = self.step in self._cache_steps
        if cache and not self.residual_history:
            raise RuntimeError("DiCache payload reached a cache step with no anchor")
        self._current_action = "cache" if cache else "full"
        self._current_gamma = None
        self._initial_hidden = None
        self._current_probe = None
        self._block_calls = 0

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        cache = self._current_action == "cache"
        self.records.append(
            {
                "step": int(self.step),
                "action": self._current_action,
                "u": int(cache),
                "payload": "di_two_anchor",
                "last_full_step": self.last_full_step,
                "step_offset": (
                    None
                    if self.last_full_step is None
                    else int(self.step) - int(self.last_full_step)
                ),
                "gamma": self._current_gamma,
                "n_anchors": len(self.residual_history),
                "original_block_calls": int(self._block_calls),
            }
        )
        if not cache:
            self.last_full_step = int(self.step)
        self.step += 1
        return output

    @staticmethod
    def _states(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        encoder = kwargs.get(
            "encoder_hidden_states", args[1] if len(args) > 1 else None
        )
        if hidden is None or encoder is None:
            raise RuntimeError("FLUX SPX DiCache could not find block inputs")
        return hidden, encoder

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
        hidden, encoder = self._states(args, kwargs)
        if index == 0:
            self._initial_hidden = hidden.detach()
            if self._current_action == "cache":
                probe_hidden = self._run_probe(hidden, encoder, args, kwargs)
                payload, gamma = aligned_residual(
                    probe_hidden - hidden,
                    self.residual_history,
                    self.probe_history,
                )
                self._current_gamma = gamma
                return encoder, hidden + payload

        if self._current_action == "cache":
            return encoder, hidden

        output = original(*args, **kwargs)
        self._block_calls += 1
        if index == self.probe_depth - 1:
            self._current_probe = output[1].detach()
        if index == len(self._patches) - 1:
            if self._initial_hidden is None or self._current_probe is None:
                raise RuntimeError("FLUX SPX DiCache full step is missing an anchor")
            append_anchor(self.residual_history, output[1] - self._initial_hidden)
            append_anchor(self.probe_history, self._current_probe - self._initial_hidden)
        return output

    def decisions(self) -> dict[str, Any]:
        return _decisions_payload(self, extra={"probe_depth": self.probe_depth})


def _decisions_payload(adapter: Any, *, extra: dict[str, Any]) -> dict[str, Any]:
    cached = sum(int(row["u"]) for row in adapter.records)
    if len(adapter.records) != adapter.num_steps or cached != len(adapter.cache_steps):
        raise RuntimeError(
            f"SPX recorded {len(adapter.records)} steps and K={cached}; expected "
            f"{adapter.num_steps} and K={len(adapter.cache_steps)}"
        )
    return {
        "schema": "flux_sp_cross_decisions.v1",
        "prompt_idx": adapter.prompt_idx,
        "seed": adapter.seed,
        "mode": "sp_cross",
        "payload": adapter.records[0]["payload"],
        "num_steps": adapter.num_steps,
        "cache_steps": list(adapter.cache_steps),
        **extra,
        "per_step": list(adapter.records),
        "summary": {
            "n_total": len(adapter.records),
            "n_full": len(adapter.records) - cached,
            "n_cached": cached,
            "cache_ratio": cached / len(adapter.records),
        },
    }


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
    if args.payload in ORACLE_CACHE_MODE:
        return FluxSPCrossResidualAdapter(
            pipe,
            cache_steps=cache_steps,
            payload=args.payload,
            num_steps=args.num_steps,
            hicache_sigma=args.hicache_sigma,
            hicache_max_order=args.hicache_max_order,
            taylorseer_max_order=args.taylorseer_max_order,
        )
    if args.payload == "mean_avg_vel":
        from flux.meancache_exact import FluxMeanCacheAdapter

        return FluxMeanCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            jvp_span=args.meancache_jvp_span,
            jvp_spans=load_meancache_spans(args, cache_steps),
        )
    if args.payload == "di_two_anchor":
        return FluxSPCrossDiCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            probe_depth=args.dicache_probe_depth,
        )
    raise AssertionError(f"unhandled payload: {args.payload}")


def _save_image(image: Any, path: Path) -> None:
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    image.save(tmp)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule_file", type=Path, required=True)
    parser.add_argument("--payload", choices=PAYLOADS, required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "model snapshot to pin (branch, tag or commit sha); default = the "
            "hub default branch. The resolved commit is recorded in timing.json."
        ),
    )
    parser.add_argument(
        "--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev"
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
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
    cache_steps = load_schedule_file(args.schedule_file, num_steps=args.num_steps)
    if args.prompt_indices is not None and args.limit:
        raise SystemExit("--prompt_indices and --limit select prompts two ways; pick one")
    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    indices = (
        None if args.prompt_indices is None
        else parse_prompt_indices(args.prompt_indices, total=len(prompts))
    )
    pairs = select_pairs(prompts, indices=indices)
    lo, hi = split_shard(len(pairs), args.shard_count, args.shard_idx)
    selected = pairs[lo:hi]
    if not selected:
        print(
            f"[flux-spx] shard {args.shard_idx}/{args.shard_count} is empty, exiting.",
            flush=True,
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    from diffusers import DiffusionPipeline

    from flux.oracle_runner import _decode_to_pil, _run_one_pipe_call

    weights = resolve_model_commit(args.model_id, args.revision)
    print(
        f"[flux-spx] model={args.model_id} revision={args.revision or '(hub default)'} "
        f"commit={weights['model_commit'] or 'UNRESOLVED'} "
        f"({weights['model_commit_source']})",
        flush=True,
    )

    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, revision=args.revision
    ).to("cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start

    adapter = build_adapter(pipe, args, cache_steps)
    adapter.install()
    z_t_holder = install_z_t_capture(pipe) if args.retain_trajectory else None
    per_image: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    try:
        for idx, prompt in selected:
            out_path = args.output_dir / image_filename(idx)
            dec_path = args.output_dir / f"decisions_{idx:05d}.json"
            traj_path = args.output_dir / trajectory_filename(idx)
            if (
                args.resume
                and out_path.is_file()
                and dec_path.is_file()
                and (not args.retain_trajectory or traj_path.is_file())
            ):
                print(f"[flux-spx] resume skip idx={idx}", flush=True)
                continue
            seed = seed_for(args.seed, idx)
            adapter.reset(prompt_idx=idx, seed=seed)
            trace = step_cb = None
            if args.retain_trajectory:
                z_t_holder.pop("z_T", None)
                trace, step_cb = make_step_collector()
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()
            latent = _run_one_pipe_call(pipe, prompt, seed, args, callback=step_cb)
            denoise_s = time.perf_counter() - denoise_start
            decode_start = time.perf_counter()
            image = _decode_to_pil(
                pipe,
                latent,
                (args.height // 16) * 16,
                (args.width // 16) * 16,
            )
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - decode_start
            decisions = adapter.decisions()
            decisions.update(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "image_file": out_path.name,
                    "schedule_file": str(args.schedule_file),
                    "spx_payload": args.payload,
                }
            )
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
                        "backbone": "flux",
                        "spx_payload": args.payload,
                        "schedule_name": args.schedule_file.stem,
                        "cache_steps": list(cache_steps),
                    },
                )
                del Z, z_T, trace
            _save_image(image, out_path)
            _atomic_write_json(dec_path, decisions)
            per_image.append(
                {
                    "idx": idx,
                    "seed": seed,
                    "trajectory_file": (traj_path.name if traj_meta else None),
                    "z_T_sha256": (traj_meta or {}).get("z_T_sha256"),
                    "denoise_s": denoise_s,
                    "decode_s": decode_s,
                    "n_cached": int(decisions["summary"]["n_cached"]),
                    "cache_ratio": float(decisions["summary"]["cache_ratio"]),
                    "image_file": out_path.name,
                    "decisions_file": dec_path.name,
                }
            )
            print(
                f"[flux-spx] payload={args.payload} idx={idx} "
                f"denoise={denoise_s:.2f}s decode={decode_s:.2f}s",
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
        print(f"[flux-spx] resume preserved existing timing shard: {timing_path}", flush=True)
        return 0

    carried = 0
    if args.resume:
        merged_records = carry_forward_timing_records(timing_path, per_image)
        carried = len(merged_records) - len(per_image)
        if carried:
            print(
                f"[flux-spx] resume carried forward {carried} timing record(s) "
                f"from {timing_path.name}",
                flush=True,
            )
        per_image = merged_records

    write_timing_json(
        timing_path,
        per_image=per_image,
        config={
            "cache_mode": f"sp_cross_{args.payload}",
            "mode": "sp_cross",
            "backbone": "flux",
            "spx_payload": args.payload,
            "schedule_file": str(args.schedule_file),
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
            "guidance": float(args.guidance),
            "dtype": args.dtype,
            "model_id": args.model_id,
            **weights,
            "base_seed": int(args.seed),
            "seed_rule": "base_plus_prompt_idx",
            "shard_idx": int(args.shard_idx),
            "shard_count": int(args.shard_count),
            "resume_carried_records": int(carried),
            "retain_trajectory": bool(args.retain_trajectory),
            "retain_dtype": (args.retain_dtype if args.retain_trajectory else None),
            "prompt_indices": (list(indices) if indices is not None else None),
        },
        model_load_s=load_s,
        wallclock_total_s=time.perf_counter() - wall_start + load_s,
        device=torch.cuda.get_device_name(0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
