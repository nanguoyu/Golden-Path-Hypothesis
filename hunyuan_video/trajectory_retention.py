"""Trajectory geometry retention on HunyuanVideo (baseline-matrix plan section 4.2).

The whole latent path of one 480p generation is 133 MB (51 rows x 1,305,600
elements x bf16), so it is kept in three tiers instead of stored:

  T1  every generation      the per-generation scalar bundle of
                            `analysis/trajectory_math.py::trajectory_metrics`,
                            computed in-flight and written as one compact JSON
                            record (~30 KB)
  T2  reference runs only   the `[chord, PC1, PC2]` orthonormal frame of
                            `plane_frame`, float16 (3 x 1,305,600 x 2 = 7.8 MB).
                            Whether two trajectories bend in the same plane is a
                            pairwise question and no per-trajectory scalar
                            answers it, so the frame is the one direction-valued
                            object that has to survive the generation.
  T3  a few references      the full path, bf16 (133 MB), for figures and for
                            any later analysis that needs the real latents.

Nothing else is retained: `retain_trajectory` drops every captured row before it
returns, so the latents live only for the duration of one generation.

Capture point. `HunyuanVideoSampler.predict` builds the pipeline call itself and
forwards no callback (`reference/hunyuan_video/code/hyvideo/inference.py:648`),
so the `callback_on_step_end` route that `flux/full_trajectory_probe.py` and
`qwen_image/full_trajectory_probe.py` use is not reachable from our runners. The
denoise loop's own latent assignment is
`latents = self.scheduler.step(noise_pred, t, latents, ..., return_dict=False)[0]`
(`.../hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py:1021`), so wrapping
the scheduler's `step` sees `sample` (= z_T on the first call) and every
`prev_sample` after it, giving Z = [z_T, z_1, ..., z_N] with `num_steps + 1`
rows. It is also the only hook that leaves the transformer alone:
`backend.verify_untouched_transformer` rejects any instance-level forward or
module hook, and it runs on exactly the untouched `original` rows that T2/T3 are
collected from.

Three details of that wrapper are load-bearing:

  - it goes on the scheduler *class*, not on the live instance. `predict` builds
    a fresh `FlowMatchDiscreteScheduler` and assigns it to the pipeline on every
    call (`reference/hunyuan_video/code/hyvideo/inference.py:611-616`), and every
    generation in this repo goes through `predict`, so an instance-level patch
    installed before the call is thrown away before the first solver step runs
    and captures nothing. `analysis/hunyuan_video/stage_b_trajectory_validate.py`
    patches the class for the same reason.

  - `functools.wraps` is required, not cosmetic. The pipeline decides whether to
    pass `generator`/`eta` to the scheduler by inspecting the signature of
    `self.scheduler.step` (`prepare_extra_func_kwargs`, pipeline file line 477,
    called at line 941); a bare `*args, **kwargs` wrapper would change that
    decision. `inspect.signature` follows the `__wrapped__` that `wraps` sets.
  - row 0 is bf16 and rows 1..N are float32. `prepare_latents` builds z_T in the
    prompt-embedding dtype (bf16 under this protocol) while
    `FlowMatchDiscreteScheduler.step` upcasts before the Euler update
    (`.../schedulers/scheduling_flow_match_discrete.py:236`), so from step 0 on
    the pipeline carries float32 latents. The bf16 quantization floor that
    forces the coarse curvature windows on the image backbones therefore only
    touches the first row here.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from analysis.trajectory_math import plane_frame, segment_tag, trajectory_metrics
from lib.io_utils import write_plane_frame


RECORD_SCHEMA = "hunyuan_video.trajectory_geometry.v1"


@dataclass
class LatentPath:
    """Rows captured from one generation, plus the scheduler sigmas behind them.

    `rows[n]` is the flattened latent after solver step `n - 1` (row 0 = z_T),
    float32 on CPU. `sigmas` is read off the scheduler the pipeline is holding
    when the capture closes: the constructor builds the full 1,001-entry
    training grid and only `set_timesteps`, which runs on the new instance
    inside the pipeline call, narrows it to `num_steps + 1`.
    """

    rows: list[torch.Tensor] = field(default_factory=list)
    sigmas: tuple[float, ...] = ()
    latent_shape: tuple[int, ...] = ()
    z_t_dtype: str = ""
    path_dtype: str = ""

    def _observe(self, sample: torch.Tensor, prev_sample: torch.Tensor) -> None:
        if not self.rows:
            self.latent_shape = tuple(int(size) for size in sample.shape)
            self.z_t_dtype = str(sample.dtype).removeprefix("torch.")
            self.path_dtype = str(prev_sample.dtype).removeprefix("torch.")
            self.rows.append(_row(sample))
        self.rows.append(_row(prev_sample))

    def _seal(self, scheduler: Any) -> None:
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None:
            return
        values = sigmas.detach().cpu().tolist() if torch.is_tensor(sigmas) else list(sigmas)
        self.sigmas = tuple(float(value) for value in values)

    def clear(self) -> None:
        self.rows.clear()


@dataclass(frozen=True)
class RetentionPlan:
    """Which tiers above T1 this generation keeps. T1 is unconditional."""

    frame_segments: tuple[tuple[int, int], ...] = ()
    save_latents: bool = False


def retention_plan(
    *,
    is_reference: bool,
    prompt_idx: int,
    num_steps: int,
    base_seed: int | None = None,
    t3_seed: int | None = None,
    t3_prompt_count: int = 0,
    frame_segments: Sequence[tuple[int, int]] | None = None,
    allow_cached: bool = False,
) -> RetentionPlan:
    """Plan section 4.2 in one place: T2 for every reference, T3 for the first
    `t3_prompt_count` prompts of the single reference stream whose base seed is
    `t3_seed`, T1 for everything else.

    `allow_cached` (runner flag `--t3_cached`) extends the same T3 predicate to
    non-reference runs (video trajectory plan section 4.3 block B): a cached
    cell run at `--seed t3_seed` then keeps the whole path of its first
    `t3_prompt_count` prompts. T2 stays reference-only regardless.

    `base_seed` is the stream's `--seed`, not the per-video
    `seed_for(seed, prompt_idx)` = seed + prompt_idx: comparing the per-video
    seed against `t3_seed` would keep only prompt 0 of the intended stream and
    spuriously keep prompt `t3_seed - seed` of every other stream.

    A frame for all 129,612 generations would be 1 TB, which is why T2 is
    reference-only; the default segment is the whole path.

    T3 is budgeted at "30 references per dataset, 60 in total" (8 GB), and every
    prompt is generated at 3 reference seeds, so the seed has to be part of the
    predicate: a prompt-index-only rule would keep 3 x 30 x 2 = 180 paths, 24 GB,
    three times the tier and 16 GB over the 74 GB the plan sizes storage for.
    """
    if not is_reference and not allow_cached:
        return RetentionPlan()
    if int(t3_prompt_count) > 0 and t3_seed is None:
        raise ValueError("T3 needs a t3_seed, or it keeps one path per reference seed")
    save_latents = (
        int(t3_prompt_count) > 0
        and base_seed is not None
        and int(base_seed) == int(t3_seed)
        and int(prompt_idx) < int(t3_prompt_count)
    )
    if not is_reference:
        return RetentionPlan(save_latents=save_latents)
    segments = (
        tuple((int(a), int(b)) for a, b in frame_segments)
        if frame_segments
        else ((0, int(num_steps) + 1),)
    )
    return RetentionPlan(frame_segments=segments, save_latents=save_latents)


def _row(tensor: torch.Tensor) -> torch.Tensor:
    """One flattened float32 CPU copy. The clone matters: the pipeline keeps
    writing to the tensors it hands us."""
    return tensor.detach().to("cpu", torch.float32).reshape(-1).clone()


@contextlib.contextmanager
def capture_latent_path(sampler: Any) -> Iterator[LatentPath]:
    """Record the latent after every denoise step of the generations run inside.

    Wraps the scheduler class's `step` for the duration and restores it
    afterwards, leaving the transformer untouched. Class, not instance, and the
    sigmas read back from the pipeline rather than from the scheduler that was
    there at entry: `predict` swaps `pipeline.scheduler` for a fresh one on every
    call (`hyvideo/inference.py:611-616`).
    """
    scheduler_type = type(sampler.pipeline.scheduler)
    original = scheduler_type.step
    if getattr(original, "_latent_path_capture", False):
        raise RuntimeError("scheduler.step is already wrapped; nested capture would double-count")
    path = LatentPath()

    @functools.wraps(original)
    def step(
        scheduler: Any, model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = original(scheduler, model_output, timestep, sample, *args, **kwargs)
        prev_sample = result[0] if isinstance(result, (tuple, list)) else result.prev_sample
        path._observe(sample, prev_sample)
        return result

    step._latent_path_capture = True
    scheduler_type.step = step
    try:
        yield path
    finally:
        scheduler_type.step = original
        path._seal(sampler.pipeline.scheduler)


def record_path(output_dir: Path, stem: str) -> Path:
    """The T1 record. Skip-if-exists keys on this file alone, which is why it is
    written last."""
    return Path(output_dir) / f"traj_{stem}.json"


def frame_path(output_dir: Path, stem: str, segment: tuple[int, int]) -> Path:
    return Path(output_dir) / f"frame_{stem}_{segment_tag(*segment)}.npy"


def latents_path(output_dir: Path, stem: str) -> Path:
    return Path(output_dir) / f"latents_{stem}.pt"


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomic write: skip-if-exists must never see a half-written record.

    Plain `json.dumps`, not `config.canonical_json_bytes`: a degenerate
    trajectory legitimately reports NaN (a zero-length chord, a zero sigma gap)
    and the canonical writer refuses NaN. This matches the record the FLUX and
    Qwen probes write, so one reader handles all three backbones.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _save_latents(path: Path, Z: torch.Tensor) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(Z.to(torch.bfloat16), tmp)
    tmp.replace(target)


def retain_trajectory(
    path: LatentPath,
    *,
    output_dir: Path,
    stem: str,
    num_steps: int,
    plan: RetentionPlan = RetentionPlan(),
    record: Mapping[str, Any] | None = None,
) -> Path:
    """Write this generation's tiers and drop the latents. Returns the T1 record.

    Write order is T3, then T2, then T1, because resume skips on the record
    alone: a kill between two of them re-runs the generation rather than leaving
    a record that points at a missing frame.

    `record` carries the caller's identity fields. The cross-backbone readers
    (`analysis/merge_full_traj.py`, `analysis/trajectory_shape_scale.py`) key on
    `dataset`, `seed` and `prompt_idx`, so a matrix runner has to pass at least
    those, alongside the method/budget and the `matrix_config_sha256` that ties
    the row to its frozen configuration. Measured fields win over `record`.
    """
    output_dir = Path(output_dir)
    rows = len(path.rows)
    if rows != int(num_steps) + 1:
        raise RuntimeError(f"captured {rows} latents, expected {int(num_steps) + 1}")
    if len(path.sigmas) != rows:
        raise RuntimeError(f"scheduler has {len(path.sigmas)} sigmas, expected {rows}")

    try:
        Z = torch.stack(path.rows)
        path.clear()  # the stack is the only copy from here on
        # measured ~2 GB of host float64 transients on top of Z at this
        # protocol, and ~5 s: trajectory_math promotes so the small
        # perpendicular components survive a 1.3e6-dimensional sum.
        metrics = trajectory_metrics(Z.numpy(), path.sigmas)

        latent_file = None
        if plan.save_latents:
            target = latents_path(output_dir, stem)
            _save_latents(target, Z)
            latent_file = target.name

        frame_files: dict[str, str] = {}
        for segment in plan.frame_segments:
            a, b = segment
            frame = plane_frame(Z.numpy()[a:b])
            if frame is None:
                raise RuntimeError(f"{stem}: rows {a}:{b} have no chord-orthogonal plane")
            target = frame_path(output_dir, stem, segment)
            write_plane_frame(target, frame)
            frame_files[segment_tag(a, b)] = target.name

        payload: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "model": "hunyuan_video",
            **dict(record or {}),
            "num_steps": int(num_steps),
            "latent_shape": list(path.latent_shape),
            "d": int(Z.shape[1]),
            "z_T_dtype": path.z_t_dtype,
            "path_dtype": path.path_dtype,
            "z_T_sha256": hashlib.sha256(Z[0].numpy().tobytes()).hexdigest(),
            "sigmas": list(path.sigmas),
            "frame_files": frame_files,
            "frame_segments": [[a, b] for a, b in plan.frame_segments],
            "latent_file": latent_file,
        }
        payload.update(metrics)
        target = record_path(output_dir, stem)
        _write_record(target, payload)
        return target
    finally:
        path.clear()
