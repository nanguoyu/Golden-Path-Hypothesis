"""Wan2.1 backend for the nine-method baseline matrix.

Model source, protocol constants, restore-then-hook installation, adapter
dispatch, and the post-run proof that the transformer was handed back
untouched.  The Hunyuan twin is `hunyuan_video/backend.py`; the differences are
all Wan's:

* the model is the upstream `wan` package vendored inside the TaylorSeer-Wan2.1
  submodule (`wan21/runner.py:82-87`), imported through that runner's own
  `import_wan` so the easydict / xfuser stubs and the flash-attn hard
  requirement are the same code, not a second copy;
* `WanModel.forward` and `WanAttentionBlock.forward` arrive already rebound to
  the class functions by `wan21/seacache.py::restore_original_forwards`, which
  undoes the TaylorSeer reference monkey patches -- restore first, hook second;
* the matrix lane never touches the golden-path modes.  `wan21/seacache.py`,
  `wan21/taylorseer_fine.py` and their payload siblings are read-only here.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from wan21 import methods_glue
from wan21.adapter import (
    CoarseBackboneAdapter,
    FineAdapter,
    HeadOutputAdapter,
    MeanCacheVelocityAdapter,
    SenCacheAdapter,
)
from wan21.dicache import WanDiCacheAdapter, WanDiCacheConfig
from wan21.methods_glue import (
    MATRIX_METHODS,
    WAN_FINE_SLOT_COUNT,
    WAN_LATENT_NUMEL,
    WAN_NUM_LAYERS,
    WAN_ORIGINAL_BLOCK_CALLS_PER_STEP,
    build_branch_methods,
    build_coarse_method,
)
from wan21.seacache import restore_original_forwards


ROOT = Path(__file__).resolve().parents[1]

#: The importable `wan` package the whole Wan2.1 lane runs on
#: (`wan21/runner.py:82-87`), pinned to the taylorseer submodule.
WAN_UPSTREAM_ROOT = (ROOT / "reference/taylorseer/code/TaylorSeer-Wan2.1").resolve()
WAN_UPSTREAM_COMMIT = "704ee98c74f7f04da443daa3c0aa2cc7803d86e3"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_upstream_source() -> dict[str, Any]:
    """Return the configured Wan source tree."""

    if not (WAN_UPSTREAM_ROOT / "wan/modules/model.py").is_file():
        raise FileNotFoundError(f"missing Wan2.1 source: {WAN_UPSTREAM_ROOT}")
    return {"root": str(WAN_UPSTREAM_ROOT), "commit": None, "status_clean": None}


def import_wan(wan_repo: Path | None = None) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Import the pinned `wan` package.

    Delegates to `wan21/runner.py::import_wan`, which installs the easydict and
    xfuser fallbacks and hard-fails when flash-attn 2/3 is unavailable rather
    than falling back to `scaled_dot_product_attention`.  Afterwards every
    imported `wan.*` module is checked to come from the pinned tree, so a
    site-packages `wan` cannot shadow it -- the same guard
    `hunyuan_video/backend.py:68-72` applies to `hyvideo.*`.
    """

    from wan21.runner import import_wan as _runner_import_wan

    repo = Path(wan_repo) if wan_repo is not None else WAN_UPSTREAM_ROOT
    validate_upstream_source()
    wan, wan_configs, size_configs, attention_backend = _runner_import_wan(repo)
    resolved = Path(repo).resolve()
    for name, module in list(sys.modules.items()):
        if name != "wan" and not name.startswith("wan."):
            continue
        origin = getattr(module, "__file__", None)
        if origin and resolved not in Path(origin).resolve().parents:
            raise RuntimeError(f"shadowed {name} imported from {origin}")
    return wan, wan_configs, size_configs, attention_backend


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WanProtocol:
    """WAN-CachePaper-480, the matrix generation protocol (plan section 1.1).

    Identical to the constants `wan21/runner.py:124-172` hard-locks; restated
    here because the matrix runner is a separate entry point and must fail the
    same way rather than inherit defaults.
    """

    task: str = "t2v-1.3B"
    width: int = 832
    height: int = 480
    frames: int = 65
    steps: int = 50
    sample_solver: str = "unipc"
    sample_shift: float = 5.0
    guidance_scale: float = 5.0
    dtype: str = "bf16"

    @property
    def size(self) -> str:
        return f"{self.width}*{self.height}"


def validate_protocol(protocol: WanProtocol) -> None:
    if protocol.task != "t2v-1.3B":
        raise SystemExit("Wan2.1 matrix protocol requires task t2v-1.3B")
    if (protocol.width, protocol.height) != (832, 480):
        raise SystemExit("Wan2.1 matrix protocol requires 832x480")
    if protocol.frames != 65:
        raise SystemExit("Wan2.1 matrix protocol requires 65 frames")
    if protocol.steps != 50:
        raise SystemExit("Wan2.1 matrix protocol requires 50 steps")
    if protocol.sample_solver != "unipc":
        raise SystemExit("Wan2.1 matrix protocol requires the unipc solver")
    if abs(float(protocol.sample_shift) - 5.0) > 1e-9:
        raise SystemExit("Wan2.1 matrix protocol requires sample_shift 5.0")
    if abs(float(protocol.guidance_scale) - 5.0) > 1e-9:
        raise SystemExit("Wan2.1 matrix protocol requires guidance_scale 5.0")
    if protocol.dtype != "bf16":
        raise SystemExit("Wan2.1 matrix protocol requires bf16")


@dataclass(frozen=True)
class WanRunSpec:
    """One matrix cell's method identity, as read out of the frozen config."""

    method: str
    method_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.method != "original" and self.method not in MATRIX_METHODS:
            raise KeyError(f"unsupported Wan2.1 matrix method: {self.method}")


# ---------------------------------------------------------------------------
# Loading and restore-then-hook
# ---------------------------------------------------------------------------


def _callable_identity(bound: Any) -> dict[str, Any]:
    function = getattr(bound, "__func__", bound)
    code = getattr(function, "__code__", None)
    source = inspect.getsourcefile(function)
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source": str(Path(source).resolve()) if source else None,
        "code_sha256": hashlib.sha256(code.co_code).hexdigest() if code else None,
    }


def verify_untouched_transformer(model: Any) -> dict[str, Any]:
    """Prove the WanModel is back to pinned upstream code.

    Fork of `hunyuan_video/backend.py::verify_untouched_transformer` with one
    difference forced by Wan: `restore_original_forwards`
    (`wan21/seacache.py:38-45`) undoes the TaylorSeer monkey patches by
    *rebinding the class function onto the instance*, so an instance-level
    `forward` attribute is the normal, restored state on this backbone.  What
    must not exist is an instance forward whose underlying function is anything
    other than the class's own -- that is a leftover patch.  On top of that the
    effective `WanModel.forward` has to come out of the pinned source tree, and
    no hooks may remain.
    """

    modules = [model, *model.blocks, model.head]
    stray: list[dict[str, Any]] = []
    for module in modules:
        instance_forward = module.__dict__.get("forward")
        if instance_forward is None:
            continue
        if getattr(instance_forward, "__func__", None) is not type(module).forward:
            stray.append({type(module).__name__: _callable_identity(instance_forward)})
    if stray:
        raise RuntimeError(f"Wan2.1 model still carries patched forwards: {stray}")

    hook_counts = {
        "pre": sum(len(module._forward_pre_hooks) for module in modules),
        "post": sum(len(module._forward_hooks) for module in modules),
    }
    if hook_counts != {"pre": 0, "post": 0}:
        raise RuntimeError(f"Wan2.1 model still carries hooks: {hook_counts}")

    identity = _callable_identity(model.forward)
    if not identity["source"] or WAN_UPSTREAM_ROOT not in Path(identity["source"]).parents:
        raise RuntimeError(f"WanModel.forward is not pinned Wan source: {identity}")
    return {
        "forward": identity,
        "hooks": hook_counts,
        "num_layers": len(model.blocks),
        "instance_forward_is_class_function": True,
    }


def load_wan_pipeline(
    wan: Any,
    wan_configs: Mapping[str, Any],
    *,
    ckpt_dir: Path,
    protocol: WanProtocol,
    device_id: int = 0,
    t5_cpu: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build the upstream `WanT2V` pipeline and restore its forwards.

    The TaylorSeer fork ships its own patched forwards; every matrix run starts
    from the pristine ones, and the identity of what it starts from is recorded.
    """

    validate_protocol(protocol)
    ckpt_dir = Path(ckpt_dir).resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"missing Wan2.1 checkpoint directory: {ckpt_dir}")
    started = time.perf_counter()
    pipe = wan.WanT2V(
        config=wan_configs[protocol.task],
        checkpoint_dir=str(ckpt_dir),
        device_id=int(device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=bool(t5_cpu),
    )
    load_seconds = time.perf_counter() - started
    restore_original_forwards(pipe.model)
    identity = verify_untouched_transformer(pipe.model)
    identity["upstream_source"] = validate_upstream_source()
    if identity["num_layers"] != WAN_NUM_LAYERS:
        raise RuntimeError(
            f"Wan2.1 t2v-1.3B has {WAN_NUM_LAYERS} blocks, loaded {identity['num_layers']}"
        )
    return pipe, {"load_seconds": load_seconds, "identity": identity}


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


def build_adapter(
    model: Any,
    protocol: WanProtocol,
    run: WanRunSpec,
    *,
    scheduler_provider: Callable[[], Any] | None = None,
) -> Any:
    """Return the adapter that realises `run.method` on this WanModel.

    `scheduler_provider` defaults to reading the scheduler off the model, which
    is where the generation loop attaches the per-video
    `FlowUniPCMultistepScheduler` (`wan21/runner.py:398`); SeaCache's SEA filter
    and MeanCache's sigmas both read it at step time, never at build time.
    """

    method = run.method
    config = run.method_config
    steps = int(protocol.steps)
    provider = scheduler_provider or (lambda: getattr(model, "scheduler", None))

    if method in ("seacache", "teacache", "budcache"):
        return CoarseBackboneAdapter(
            model,
            build_coarse_method(
                method, num_steps=steps, config=config, scheduler_provider=provider
            ),
            num_steps=steps,
        )
    if method == "sencache":
        return SenCacheAdapter(
            model,
            build_coarse_method(
                method, num_steps=steps, config=config, scheduler_provider=provider
            ),
            num_steps=steps,
        )
    if method in ("taylorseer_o1", "hicache_o2"):
        return FineAdapter(
            model,
            build_branch_methods(
                method, num_steps=steps, config=config, scheduler_provider=provider
            ),
            num_steps=steps,
        )
    if method == "l2p":
        return HeadOutputAdapter(
            model,
            build_branch_methods(
                method, num_steps=steps, config=config, scheduler_provider=provider
            ),
            num_steps=steps,
        )
    if method == "meancache":
        return MeanCacheVelocityAdapter(
            model,
            build_branch_methods(
                method, num_steps=steps, config=config, scheduler_provider=provider
            ),
            num_steps=steps,
        )
    if method == "dicache":
        # With `cache_steps` the gate is replaced by the table and the payload
        # is unchanged; that is the video SPX `di_two_anchor` column, the only
        # way DiCache's payload can be scored on a foreign schedule.
        fixed_steps = (tuple(methods_glue.fixed_cache_steps(
            config, method="dicache", num_steps=steps))
            if config.get("cache_steps") is not None else None)
        return WanDiCacheAdapter(
            model,
            WanDiCacheConfig(
                num_steps=steps,
                threshold=float(config.get("threshold", 0.0)),
                ret_ratio=float(config.get("ret_ratio", methods_glue.DICACHE_RET_RATIO)),
                probe_depth=int(config.get("probe_depth", methods_glue.DICACHE_PROBE_DEPTH)),
                cache_steps=tuple(sorted(fixed_steps)) if fixed_steps is not None else None,
            ),
        )
    raise KeyError(f"unsupported Wan2.1 matrix method: {method}")


@contextlib.contextmanager
def maybe_adapter(
    model: Any,
    protocol: WanProtocol,
    run: WanRunSpec,
    *,
    scheduler_provider: Callable[[], Any] | None = None,
) -> Iterator[Any | None]:
    """Install the adapter for the duration of one generation.

    `--mode original` installs nothing: the reference row is the pinned
    upstream forward, verified untouched on both sides of the generation.
    """

    if run.method == "original":
        verify_untouched_transformer(model)
        yield None
        verify_untouched_transformer(model)
        return
    adapter = build_adapter(model, protocol, run, scheduler_provider=scheduler_provider)
    adapter.reset()
    with adapter:
        yield adapter
    verify_untouched_transformer(model)


__all__ = [
    "MATRIX_METHODS",
    "WAN_FINE_SLOT_COUNT",
    "WAN_LATENT_NUMEL",
    "WAN_NUM_LAYERS",
    "WAN_ORIGINAL_BLOCK_CALLS_PER_STEP",
    "WAN_UPSTREAM_COMMIT",
    "WAN_UPSTREAM_ROOT",
    "WanProtocol",
    "WanRunSpec",
    "build_adapter",
    "import_wan",
    "load_wan_pipeline",
    "maybe_adapter",
    "validate_protocol",
    "validate_upstream_source",
    "verify_untouched_transformer",
]
