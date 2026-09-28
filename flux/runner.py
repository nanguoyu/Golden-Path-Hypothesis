#!/usr/bin/env python3
"""Single-GPU multi-prompt runner for FLUX via diffusers `FluxPipeline`.

Dispatches `--mode {original, HiCache, SeaCache}` to the corresponding
`install` function in `flux.hicache` / `flux.seacache`. Writes lossless PNGs
to `--output_dir` named `img_<global_idx>.png` and a per-shard
`timing_shard{i}of{N}.json`.

Per-image seed: `seed = base_seed + global_idx`. Sharding semantics match
`lib.io_utils.split_shard` so HiCache, SeaCache, and original outputs at the
same `(prompt_file, base_seed)` pair line up across runs.

Example (single GPU):

    PYTHONPATH=$PWD python flux/runner.py \\
        --prompt_file resources/prompts/prompt.txt \\
        --output_dir results/flux_hicache_s42 \\
        --mode HiCache --num_steps 50 --seed 42

Multi-GPU sharding: see `RUN/multi_gpu_flux.sh`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# NOTE: this file is meant to be run with PROJECT_ROOT on PYTHONPATH so that
# `import lib.xxx` and `import flux.xxx` both resolve. The launchers in `RUN/`
# set that up; running this file directly also works because Python prepends
# the script's directory (`flux/`) to sys.path, and we add the parent below
# to make `lib` importable in that case.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux import seacache_fine_payload as seacache_fine_payload_modes  # noqa: E402
from flux import seacache_payload as seacache_payload_modes  # noqa: E402
from flux import seacache_segment_payload as seacache_segment_payload_modes  # noqa: E402
from lib.foca import FOCA_DERIVATIVES, FOCA_HEUN_VARIANTS, FOCA_HISTORY_POLICIES  # noqa: E402
from lib.io_utils import image_filename, read_prompts, split_shard, write_timing_json  # noqa: E402
from lib.sencache import load_sensitivity_table  # noqa: E402

_PAYLOAD_MODE_CHOICES = tuple(
    sorted(
        set(seacache_payload_modes.PAYLOAD_MODES)
        | set(seacache_fine_payload_modes.PAYLOAD_MODES)
        | set(seacache_segment_payload_modes.PAYLOAD_MODES)
    )
)


def _git_rev_parse_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLUX via diffusers FluxPipeline with optional cache acceleration"
    )
    p.add_argument("--prompt_file", type=Path, required=True,
                   help="Prompt list, one per line.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output dir for img_<idx>.png and timing_shard*.json.")
    p.add_argument("--mode", choices=[
        "original", "HiCache", "TaylorSeer", "SeaCache", "TeaCache", "OriCache",
        "SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload",
        "SVDCache", "SenCache", "HiCache_fine", "TaylorSeer_fine",
        "L2P", "L2P_fine", "FoCa_fine",
    ], default="original",
        help="`original` = vanilla FLUX (no cache). "
             "`HiCache` = transformer-residual Hermite extrapolation (coarse). "
             "`TaylorSeer` = transformer-residual Taylor extrapolation (coarse). "
             "`SeaCache` = full forward replacement, Wiener-filtered rel_L1 gate. "
             "`OriCache` = research-only orientation/curvature threshold gate. "
             "`SeaCachePayload` = research-only SeaCache gate with forecast residual payload. "
             "`TeaCachePayload` = research-only TeaCache gate with forecast residual payload. "
             "`SeaCacheFinePayload` = research-only SeaCache gate with 114-slot fine feature forecast payload. "
             "`SeaCacheSegmentPayload` = research-only SeaCache gate with segment residual forecast payload. "
             "`SVDCache` = native coarse SVD-Cache interval schedule with principal EMA and residual reuse. "
             "`TeaCache` = full forward replacement, poly-rescaled rel_L1 gate. "
             "`SenCache` = full forward replacement, frozen sensitivity-aware latent gate. "
             "`HiCache_fine` = per-(block,sub_module) Hermite extrapolation (paper-faithful). "
             "`TaylorSeer_fine` = per-(block,sub_module) Taylor extrapolation (paper-faithful). "
             "`L2P` = learned linear final-hidden predictor (paper-faithful target). "
             "`L2P_fine` = research 114-slot fine learned linear predictor. "
             "`FoCa_fine` = research 114-slot fine BDF2/Heun predictor.")
    p.add_argument("--num_steps", type=int, default=50,
                   help="Sampling steps (paper Table 1 baseline: 50).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; per-image seed = seed + global_idx.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5,
                   help="CFG guidance for flux-dev (ignored for flux-schnell).")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace model id (or local path).")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Pipeline torch_dtype. Default bf16 matches the published convention.")

    # Cache args shared by HiCache + TaylorSeer (interval-style gating)
    p.add_argument("--interval", type=int, default=7,
                   help="HiCache/TaylorSeer: refresh every N steps after warmup "
                        "(HiCache paper: 7; TaylorSeer paper FLUX: 6).")
    p.add_argument("--max_order", type=int, default=2,
                   help="HiCache/TaylorSeer: extrapolation truncation order O (paper: 2).")
    p.add_argument("--hicache_sigma", type=float, default=0.5,
                   help="HiCache: dual-scaling sigma (paper: 0.5). Ignored by other modes.")
    p.add_argument("--first_enhance", type=int, default=3,
                   help="Number of initial sampling steps to force full-forward. "
                        "HiCache/TaylorSeer paper: 3; SeaCache/TeaCache upstream: 1.")

    # SeaCache args (Wiener-rescaled threshold gate)
    p.add_argument("--seacache_thresh", type=float, default=0.3,
                   help="SeaCache: accumulated rel_L1 threshold (0.3 → ~2×, 0.6 → ~3×).")
    p.add_argument("--payload_mode", choices=_PAYLOAD_MODE_CHOICES, default=None,
                   help=("SeaCachePayload: residual payload used on cached steps. "
                         "Research coarse RFC modes: rfc_rfe_taylor_o1 | rfc_rfe_taylor_o2. "
                         "SVD-Cache coarse residual modes: svdcache_taylor_o1_e85_r16 | "
                         "svdcache_taylor_o1_e90_r16 | svdcache_taylor_o1_e95_r16 | "
                         "svdcache_taylor_o1_e95_r32 | svdcache_ensemble_e85_r16 | "
                         "svdcache_ensemble_e95_r16 | svdcache_ema_e85_b0.9_r16 | "
                         "svdcache_ema_e95_b0.9_r16. "
                         "SeaCacheFinePayload: fine_reuse | fine_taylor_o1 | "
                         "fine_taylor_o2 | fine_hicache_o2 | "
                         "fine_rfc_rfe_taylor_o1 | fine_rfc_rfe_taylor_o2. "
                         "SeaCacheSegmentPayload: segment_reuse | segment_taylor_o1 | "
                         "segment_taylor_o2 | segment_hicache_o2 | segment_ensemble_mean."))
    p.add_argument("--payload_blend", type=float, default=1.0,
                   help=("SeaCachePayload: payload = reuse + w*(forecast-reuse); "
                         "w > 1 extrapolates for research scale sweeps."))
    p.add_argument("--payload_sigma", type=float, default=0.5,
                   help="SeaCachePayload: HiCache Hermite sigma for hicache_o2 forecasts.")
    p.add_argument("--segment_layout", choices=seacache_segment_payload_modes.LAYOUTS, default="seg8",
                   help="SeaCacheSegmentPayload: segment partition layout.")
    p.add_argument("--payload_schedule_dir", type=Path, default=None,
                   help=("SeaCachePayload/TeaCachePayload: baseline output dir with decisions_XXXXX.json "
                         "or prompt_XXXXX/decisions.json files. Required for non-reuse "
                         "payloads so cached/full actions are replayed and only the "
                         "payload changes. Also accepted by L2P/L2P_fine to replay "
                         "a fixed cache/full action schedule."))
    p.add_argument("--allow_online_payload_schedule_drift", action="store_true",
                   help=("Research-only SeaCachePayload mode: allow non-reuse forecast "
                         "payloads to run without --payload_schedule_dir under the native "
                         "SeaCache gate, so closed-loop payload-induced schedule drift can "
                         "be measured. Disabled by default to protect fixed-schedule "
                         "payload-only experiments."))
    p.add_argument("--payload_log_norms", action="store_true",
                   help=("SeaCacheFinePayload: log aggregate per-slot payload norms. "
                         "Disabled by default so generation timing is not contaminated "
                         "by extra GPU reductions and synchronizations."))
    p.add_argument("--payload_shadow_full_residual", action="store_true",
                   help=("SeaCachePayload/TeaCachePayload: on cached steps, run a shadow full transformer "
                         "to log counterfactual residual-prediction errors without changing "
                         "the committed trajectory. Observer runs are for mechanism analysis, "
                         "not timing/speedup claims."))
    p.add_argument("--payload_shadow_full_velocity", action="store_true",
                   help=("SeaCacheFinePayload: on cached steps, run a read-only full "
                         "transformer to log same-state velocity error without changing "
                         "the committed trajectory. Observer runs are for mechanism "
                         "analysis, not timing/speedup claims."))
    p.add_argument("--payload_bank_dir", type=Path, default=None,
                   help=("SeaCachePayload: tensor bank for bank-backed negative controls "
                         "such as _step_shuffle, _wrong_prompt, _step_only, and _prompt_only."))
    p.add_argument("--payload_wrong_seed_bank_dir", type=Path, default=None,
                   help="SeaCachePayload: alternate-seed tensor bank for _wrong_seed controls.")
    p.add_argument("--payload_bank_write_dir", type=Path, default=None,
                   help=("SeaCachePayload: write forecast tensors for cached steps into this bank. "
                         "Research-only; use with a schedule-locked run."))
    p.add_argument("--payload_gate_mode", choices=seacache_payload_modes.PAYLOAD_GATE_MODES,
                   default="seacache",
                   help=("SeaCachePayload: decision rule. `seacache` preserves the native "
                         "SeaCache/fixed-schedule behavior; `forecast_uncertainty` uses a "
                         "single-forecast calibrated uncertainty guard; "
                         "`seacache_forecast_intersection` caches only when both pass; "
                         "`rfc_input_error` uses RFC-style accumulated input prediction error. "
                         "`teacache` is used internally by TeaCachePayload."))
    p.add_argument("--rfc_gate_tau", type=float, default=None,
                   help=("SeaCachePayload payload_gate_mode=rfc_input_error: accumulated "
                         "relative L1 input-prediction-error threshold. Defaults to "
                         "--seacache_thresh when omitted."))
    p.add_argument("--forecast_guard_observe", action="store_true",
                   help=("SeaCachePayload: log single-forecast uncertainty fields without "
                         "letting them affect decisions."))
    p.add_argument("--forecast_guard_mode", default="auto",
                   choices=("auto",) + seacache_payload_modes.FORECAST_GUARD_MODES,
                   help="SeaCachePayload forecast uncertainty guard mode. `auto` follows payload_mode.")
    p.add_argument("--forecast_guard_tau", type=float, default=0.05,
                   help="SeaCachePayload forecast guard threshold; cache iff score <= tau.")
    p.add_argument("--forecast_guard_quantile", type=float, default=0.8,
                   help="SeaCachePayload forecast guard rolling nonconformity quantile.")
    p.add_argument("--forecast_guard_window", type=int, default=16,
                   help="SeaCachePayload forecast guard rolling calibration window.")
    p.add_argument("--forecast_guard_warmup_updates", type=int, default=1,
                   help="SeaCachePayload forecast guard full-refresh calibration warmup.")
    p.add_argument("--forecast_guard_age_gamma", type=float, default=0.0,
                   help="SeaCachePayload forecast guard multiplicative age exponent.")
    p.add_argument("--forecast_guard_scale_floor", type=float, default=1e-4,
                   help="SeaCachePayload forecast guard minimum self-scale for calibration.")

    # L2P args (learned linear predictor)
    p.add_argument("--l2p_weights", type=Path, default=None,
                   help="L2P/L2P_fine checkpoint containing `weights` or `W`.")
    p.add_argument("--l2p_min_abs_weight", type=float, default=0.0,
                   help="L2P: ignore coefficients with abs(weight) <= this value.")

    # FoCa args (training-free BDF2/Heun fine predictor)
    p.add_argument("--foca_heun_variant", choices=FOCA_HEUN_VARIANTS, default="paper_literal",
                   help="FoCa_fine: Heun calibration interpretation.")
    p.add_argument("--foca_history_policy", choices=FOCA_HISTORY_POLICIES, default="recursive",
                   help="FoCa_fine: whether cached predictions are committed to rolling history.")
    p.add_argument("--foca_derivative", choices=FOCA_DERIVATIVES, default="step_backward",
                   help="FoCa_fine: derivative estimator variant.")
    p.add_argument("--foca_h", type=float, default=1.0,
                   help="FoCa_fine: step-index ODE spacing h.")
    p.add_argument("--foca_log_norms", action="store_true",
                   help="FoCa_fine: log aggregate norm diagnostics; disabled for clean timing.")

    # SVD-Cache native args (interval schedule + principal EMA / residual reuse)
    p.add_argument("--svdcache_energy", type=float, default=0.85,
                   help="SVDCache: cumulative singular energy threshold tau.")
    p.add_argument("--svdcache_max_rank", type=int, default=32,
                   help="SVDCache: maximum right-basis rank.")
    p.add_argument("--svdcache_beta", type=float, default=0.9,
                   help="SVDCache: EMA beta for the principal subspace.")
    p.add_argument("--svdcache_basis_policy", choices=("first_full",), default="first_full",
                   help="SVDCache: coarse basis construction policy.")

    # TeaCache args (polynomial-rescaled threshold gate)
    p.add_argument("--teacache_thresh", type=float, default=0.3,
                   help="TeaCache: accumulated f(rel_L1) threshold. FLUX-specific scale: "
                        "0.25→1.5×, 0.4→1.8×, 0.6→2×, 0.8→2.25× per upstream README. "
                        "SeaCache paper Table 1 uses 0.3 (~50%% budget) and 0.6 (~30%%). "
                        "Note: paper TeaCache's video-task 0.1/0.2 settings DO NOT apply to FLUX "
                        "(the polynomial intercept is ~0.26 so single-step 0.1 always triggers).")
    p.add_argument("--teacache_backbone", default="flux",
                   help="TeaCache: polynomial coefficient backbone key "
                        "(see lib/teacache_coeffs.py).")
    p.add_argument("--teacache_variant", default=None,
                   help="TeaCache: optional variant key (e.g. `1.3b` / `14b` for wan21).")

    # OriCache args (orientation-guided accumulated threshold gate)
    p.add_argument("--oricache_thresh", type=float, default=1.5,
                   help="OriCache: accumulated normalized-curvature threshold tau.")
    p.add_argument("--oricache_signal", choices=("modulated", "raw"), default="modulated",
                   help=("OriCache gate signal. `modulated` matches the TeaCache/SeaCache "
                         "first-block norm1 proxy used in this repo; `raw` uses the embedded "
                         "first-block input and is a sensitivity check."))

    # SenCache args (frozen sensitivity-table gate)
    p.add_argument("--sencache_sensitivity_path", type=Path, default=None,
                   help="SenCache: frozen FLUX sensitivity npz with timesteps/J_x_norm/J_t_norm.")
    p.add_argument("--sencache_thresh", type=float, default=None,
                   help="SenCache convenience alias for --sencache_thresh_main.")
    p.add_argument("--sencache_thresh_start", type=float, default=0.005,
                   help="SenCache early-window threshold before --sencache_switch_ratio.")
    p.add_argument("--sencache_thresh_main", type=float, default=None,
                   help="SenCache main threshold. Default: --sencache_thresh if set, else 0.07.")
    p.add_argument("--sencache_K", type=int, default=10,
                   help="SenCache max consecutive cached steps.")
    p.add_argument("--sencache_threshold_scale", default="auto",
                   help="SenCache threshold scale. 'auto' uses sqrt(packed latent numel).")
    p.add_argument("--sencache_switch_ratio", type=float, default=0.2,
                   help="Fraction of trajectory using --sencache_thresh_start.")
    p.add_argument("--sencache_ret_steps", type=int, default=0,
                   help="SenCache minimum step index before cache is allowed.")
    p.add_argument("--sencache_cutoff_steps", type=int, default=-1,
                   help="SenCache cache-allowed cutoff step index; -1 means num_steps - 1.")

    # Sharding / IO
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose img_<idx>.png already exists.")
    return p.parse_args()


def _load_payload_action_steps(
    schedule_dir: Path,
    prompt_idx: int,
    *,
    expected_num_steps: Optional[int] = None,
    require_reuse_reference: bool = False,
    expected_reference_mode: Any = "SeaCachePayload",
    expected_reference_payload_mode: Any = "reuse",
    expected_payload_gate_mode: Optional[str] = None,
) -> set[int]:
    candidates = [
        schedule_dir / f"decisions_{prompt_idx:05d}.json",
        schedule_dir / f"prompt_{prompt_idx:05d}" / "decisions.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        joined = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"missing payload schedule for prompt {prompt_idx}: tried {joined}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        schedule_prompt_idx = payload.get("prompt_idx")
        if schedule_prompt_idx is not None and int(schedule_prompt_idx) != int(prompt_idx):
            raise ValueError(
                f"schedule prompt_idx mismatch in {path}: "
                f"expected {prompt_idx}, got {schedule_prompt_idx}"
            )
        if require_reuse_reference:
            schedule_mode = payload.get("mode")
            expected_modes = (
                (str(expected_reference_mode),)
                if isinstance(expected_reference_mode, str)
                else tuple(str(item) for item in expected_reference_mode)
            )
            if schedule_mode is not None and str(schedule_mode) not in expected_modes:
                raise ValueError(
                    f"schedule {path} must come from {expected_modes} reuse reference; "
                    f"got mode={schedule_mode!r}"
                )
            schedule_payload_mode = payload.get("payload_mode")
            expected_payload_modes = (
                (str(expected_reference_payload_mode),)
                if isinstance(expected_reference_payload_mode, str)
                else tuple(str(item) for item in expected_reference_payload_mode)
            )
            if schedule_payload_mode is not None and str(schedule_payload_mode) not in expected_payload_modes:
                raise ValueError(
                    f"schedule {path} must come from payload_mode in {expected_payload_modes}; "
                    f"got payload_mode={schedule_payload_mode!r}"
                )
            if expected_payload_gate_mode is not None:
                schedule_gate_mode = payload.get("payload_gate_mode")
                if schedule_gate_mode is not None and str(schedule_gate_mode) != str(expected_payload_gate_mode):
                    raise ValueError(
                        f"schedule {path} must use payload_gate_mode={expected_payload_gate_mode!r}; "
                        f"got {schedule_gate_mode!r}"
                    )
        rows = payload.get("per_step") or payload.get("decisions") or []
    else:
        raise ValueError(f"unsupported schedule JSON in {path}: {type(payload).__name__}")
    if not isinstance(rows, list):
        raise ValueError(f"schedule rows in {path} are not a list")
    if expected_num_steps is not None and int(expected_num_steps) > 0:
        steps = [
            int(row.get("step"))
            for row in rows
            if isinstance(row, dict) and row.get("step") is not None
        ]
        expected = list(range(int(expected_num_steps)))
        if steps != expected:
            raise ValueError(
                f"schedule step sequence mismatch in {path}: expected 0.."
                f"{int(expected_num_steps) - 1}, got {steps[:10]}... len={len(steps)}"
            )
    return {
        int(row["step"])
        for row in rows
        if isinstance(row, dict) and int(row.get("u", 0)) == 1
    }


def _normalize_payload_mode_default(args: argparse.Namespace) -> None:
    if args.mode == "TeaCachePayload" and str(args.payload_gate_mode) == "seacache":
        args.payload_gate_mode = "teacache"
    if args.payload_mode is not None:
        return
    if args.mode == "SeaCacheFinePayload":
        args.payload_mode = "fine_taylor_o1"
    elif args.mode == "SeaCacheSegmentPayload":
        args.payload_mode = "segment_taylor_o1"
    else:
        args.payload_mode = "reuse"


def _validate_seacache_payload_args(args: argparse.Namespace) -> None:
    if args.mode not in ("SeaCachePayload", "TeaCachePayload"):
        if getattr(args, "payload_shadow_full_residual", False):
            raise SystemExit("--payload_shadow_full_residual is only supported by --mode SeaCachePayload/TeaCachePayload")
        if getattr(args, "allow_online_payload_schedule_drift", False):
            raise SystemExit("--allow_online_payload_schedule_drift is only supported by --mode SeaCachePayload")
        return
    if args.mode == "SeaCachePayload" and str(args.payload_gate_mode) == "teacache":
        raise SystemExit("--payload_gate_mode teacache is exposed as --mode TeaCachePayload")
    if args.mode == "TeaCachePayload" and str(args.payload_gate_mode) != "teacache":
        raise SystemExit("--mode TeaCachePayload requires --payload_gate_mode teacache")
    base_mode, control = seacache_payload_modes._payload_spec(str(args.payload_mode))
    allowed_base_modes = (
        seacache_payload_modes.BASE_PAYLOAD_MODES
        + seacache_payload_modes.SELECTOR_PAYLOAD_MODES
        + ("update_target",)
    )
    if base_mode not in allowed_base_modes:
        raise SystemExit(f"unknown SeaCachePayload base mode: {base_mode}")
    if control == "update_scalar_calib" and float(args.payload_blend) != 1.0:
        raise SystemExit("--payload_mode update_alpha_* requires --payload_blend 1.0")
    posterior_oracle = str(args.payload_mode) in seacache_payload_modes.POSTERIOR_ORACLE_PAYLOAD_MODES
    active_guard = str(args.payload_gate_mode) in (
        "forecast_uncertainty",
        "seacache_forecast_intersection",
    )
    rfc_gate_active = str(args.payload_gate_mode) == "rfc_input_error"
    online_schedule_drift = bool(getattr(args, "allow_online_payload_schedule_drift", False))
    if online_schedule_drift:
        if args.mode != "SeaCachePayload":
            raise SystemExit("--allow_online_payload_schedule_drift is only supported by --mode SeaCachePayload")
        if args.payload_schedule_dir is not None:
            raise SystemExit("--allow_online_payload_schedule_drift must not be combined with --payload_schedule_dir")
        if str(args.payload_gate_mode) != "seacache":
            raise SystemExit("--allow_online_payload_schedule_drift requires --payload_gate_mode seacache")
        if active_guard:
            raise SystemExit("--allow_online_payload_schedule_drift is not an active forecast-guard run")
        if base_mode not in seacache_payload_modes.BASE_PAYLOAD_MODES or base_mode == "reuse":
            raise SystemExit(
                "--allow_online_payload_schedule_drift requires a forecast base payload "
                "mode: taylor_o1, taylor_o2, hicache_o2, or ensemble_mean"
            )
        if control != "none" and str(args.payload_mode) not in seacache_payload_modes.FORECAST_OPT_PAYLOAD_MODES:
            raise SystemExit(
                "--allow_online_payload_schedule_drift supports only base forecast payloads "
                "or forecast-opt research payloads such as rfc_rfe_taylor_o1/o2"
            )
        if float(args.payload_blend) != 1.0:
            raise SystemExit("--allow_online_payload_schedule_drift requires --payload_blend 1.0")
        if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
            raise SystemExit("--allow_online_payload_schedule_drift does not support payload banks")
        if args.payload_bank_write_dir is not None:
            raise SystemExit("--allow_online_payload_schedule_drift does not support bank writes")
    if posterior_oracle:
        if args.mode != "SeaCachePayload":
            raise SystemExit("--payload_mode posterior_oracle_* is only supported by --mode SeaCachePayload")
        if args.payload_schedule_dir is None:
            raise SystemExit("--payload_mode posterior_oracle_* requires --payload_schedule_dir")
        if active_guard:
            raise SystemExit("--payload_mode posterior_oracle_* requires a fixed schedule, not active online gating")
        if float(args.payload_blend) != 1.0:
            raise SystemExit("--payload_mode posterior_oracle_* requires --payload_blend 1.0")
        if args.payload_shadow_full_residual:
            raise SystemExit(
                "--payload_mode posterior_oracle_* runs its own shadow full pass; "
                "do not also pass --payload_shadow_full_residual"
            )
        if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
            raise SystemExit("--payload_mode posterior_oracle_* does not support payload banks")
        if args.payload_bank_write_dir is not None:
            raise SystemExit("--payload_mode posterior_oracle_* does not support bank writes")
    if rfc_gate_active:
        if args.mode != "SeaCachePayload":
            raise SystemExit("--payload_gate_mode rfc_input_error is only supported by --mode SeaCachePayload")
        if args.payload_schedule_dir is not None:
            raise SystemExit("--payload_gate_mode rfc_input_error is active online gating and must not use --payload_schedule_dir")
        if float(args.payload_blend) != 1.0:
            raise SystemExit("--payload_gate_mode rfc_input_error requires --payload_blend 1.0")
        if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
            raise SystemExit("--payload_gate_mode rfc_input_error does not support payload banks")
        if args.payload_bank_write_dir is not None:
            raise SystemExit("--payload_gate_mode rfc_input_error does not support bank writes")
        if args.rfc_gate_tau is not None and float(args.rfc_gate_tau) <= 0.0:
            raise SystemExit("--rfc_gate_tau must be positive")
    if active_guard:
        if args.payload_schedule_dir is not None:
            raise SystemExit(
                f"--payload_gate_mode {args.payload_gate_mode} is active online gating and "
                "must not be combined with --payload_schedule_dir."
            )
        if str(args.payload_mode) not in seacache_payload_modes.FORECAST_GUARD_MODES:
            raise SystemExit(
                f"--payload_gate_mode {args.payload_gate_mode} currently requires "
                f"--payload_mode in {seacache_payload_modes.FORECAST_GUARD_MODES}."
            )
        if str(args.forecast_guard_mode) not in ("auto", str(args.payload_mode)):
            raise SystemExit(
                "--forecast_guard_mode must be auto or match --payload_mode for active guard runs."
            )
        if float(args.payload_blend) != 1.0:
            raise SystemExit(f"--payload_gate_mode {args.payload_gate_mode} requires --payload_blend 1.0")
        if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
            raise SystemExit(f"--payload_gate_mode {args.payload_gate_mode} does not support payload banks")
        if args.payload_bank_write_dir is not None:
            raise SystemExit(f"--payload_gate_mode {args.payload_gate_mode} does not support bank writes")
    if (
        str(args.payload_mode) != "reuse"
        and args.payload_schedule_dir is None
        and not active_guard
        and not rfc_gate_active
        and not online_schedule_drift
    ):
        raise SystemExit(
            "--mode SeaCachePayload with --payload_mode != reuse requires "
            "--payload_schedule_dir. SeaCachePayload forecast experiments are "
            "defined as fixed-schedule payload interventions."
        )
    if args.payload_bank_write_dir is not None and args.payload_schedule_dir is None:
        raise SystemExit(
            "--payload_bank_write_dir requires --payload_schedule_dir so the tensor "
            "bank is collected under the same fixed SeaCachePayload(reuse) schedule."
        )
    if control in seacache_payload_modes.BANK_CONTROLS:
        if control == "wrong_seed":
            if args.payload_wrong_seed_bank_dir is None:
                raise SystemExit(
                    f"--payload_mode {args.payload_mode} requires "
                    "--payload_wrong_seed_bank_dir"
                )
        elif args.payload_bank_dir is None:
            raise SystemExit(
                f"--payload_mode {args.payload_mode} requires --payload_bank_dir"
            )
    if not (0.0 < float(args.forecast_guard_quantile) <= 1.0):
        raise SystemExit("--forecast_guard_quantile must be in (0, 1]")
    if int(args.forecast_guard_window) < 0:
        raise SystemExit("--forecast_guard_window must be >= 0")
    if int(args.forecast_guard_warmup_updates) < 0:
        raise SystemExit("--forecast_guard_warmup_updates must be >= 0")
    if float(args.forecast_guard_scale_floor) <= 0.0:
        raise SystemExit("--forecast_guard_scale_floor must be > 0")


def _validate_seacache_fine_payload_args(args: argparse.Namespace) -> None:
    if args.mode != "SeaCacheFinePayload":
        if getattr(args, "payload_shadow_full_velocity", False):
            raise SystemExit("--payload_shadow_full_velocity is only supported by --mode SeaCacheFinePayload")
        return
    if str(args.payload_mode) not in seacache_fine_payload_modes.PAYLOAD_MODES:
        raise SystemExit(
            "--mode SeaCacheFinePayload requires --payload_mode in "
            f"{seacache_fine_payload_modes.PAYLOAD_MODES}"
        )
    if float(args.payload_blend) != 1.0:
        raise SystemExit("--mode SeaCacheFinePayload requires --payload_blend 1.0")
    if args.payload_shadow_full_residual:
        raise SystemExit("--mode SeaCacheFinePayload does not support --payload_shadow_full_residual")
    if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
        raise SystemExit("--mode SeaCacheFinePayload does not support payload banks")
    if args.payload_bank_write_dir is not None:
        raise SystemExit("--mode SeaCacheFinePayload does not support --payload_bank_write_dir")
    if str(args.payload_gate_mode) not in seacache_fine_payload_modes.PAYLOAD_GATE_MODES:
        raise SystemExit(
            "--mode SeaCacheFinePayload supports --payload_gate_mode in "
            f"{seacache_fine_payload_modes.PAYLOAD_GATE_MODES}"
        )
    if str(args.payload_gate_mode) == "rfc_input_error":
        if args.payload_schedule_dir is not None:
            raise SystemExit(
                "--mode SeaCacheFinePayload --payload_gate_mode rfc_input_error "
                "is active online gating and must not use --payload_schedule_dir"
            )
        if str(args.payload_mode) not in (
            "fine_rfc_rfe_taylor_o1",
            "fine_rfc_rfe_taylor_o2",
        ):
            raise SystemExit(
                "--mode SeaCacheFinePayload --payload_gate_mode rfc_input_error "
                "requires --payload_mode fine_rfc_rfe_taylor_o1/o2"
            )
        if args.rfc_gate_tau is not None and float(args.rfc_gate_tau) <= 0.0:
            raise SystemExit("--rfc_gate_tau must be positive")
    if args.forecast_guard_observe or str(args.forecast_guard_mode) != "auto":
        raise SystemExit("--mode SeaCacheFinePayload does not support forecast guard options")


def _validate_seacache_segment_payload_args(args: argparse.Namespace) -> None:
    if args.mode != "SeaCacheSegmentPayload":
        return
    if str(args.payload_mode) not in seacache_segment_payload_modes.PAYLOAD_MODES:
        raise SystemExit(
            "--mode SeaCacheSegmentPayload requires --payload_mode in "
            f"{seacache_segment_payload_modes.PAYLOAD_MODES}"
        )
    if float(args.payload_blend) != 1.0:
        raise SystemExit("--mode SeaCacheSegmentPayload requires --payload_blend 1.0")
    if str(args.payload_mode) != "segment_reuse" and args.payload_schedule_dir is None:
        raise SystemExit(
            "--mode SeaCacheSegmentPayload with non-reuse payload requires "
            "--payload_schedule_dir for fixed-schedule intervention."
        )
    if args.payload_shadow_full_residual or args.payload_shadow_full_velocity:
        raise SystemExit("--mode SeaCacheSegmentPayload does not support payload shadow observers")
    if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
        raise SystemExit("--mode SeaCacheSegmentPayload does not support payload banks")
    if args.payload_bank_write_dir is not None:
        raise SystemExit("--mode SeaCacheSegmentPayload does not support --payload_bank_write_dir")
    if str(args.payload_gate_mode) != "seacache":
        raise SystemExit("--mode SeaCacheSegmentPayload currently supports only --payload_gate_mode seacache")
    if args.forecast_guard_observe or str(args.forecast_guard_mode) != "auto":
        raise SystemExit("--mode SeaCacheSegmentPayload does not support forecast guard options")


def _validate_l2p_args(args: argparse.Namespace) -> None:
    if args.mode not in ("L2P", "L2P_fine"):
        return
    if args.l2p_weights is None:
        raise SystemExit(f"--mode {args.mode} requires --l2p_weights")
    if not args.l2p_weights.is_file():
        raise SystemExit(f"--l2p_weights does not exist: {args.l2p_weights}")
    if float(args.l2p_min_abs_weight) < 0.0:
        raise SystemExit("--l2p_min_abs_weight must be >= 0")
    if args.payload_shadow_full_residual or args.payload_shadow_full_velocity:
        raise SystemExit(f"--mode {args.mode} does not support payload shadow observers")
    if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
        raise SystemExit(f"--mode {args.mode} does not support payload banks")
    if args.payload_bank_write_dir is not None:
        raise SystemExit(f"--mode {args.mode} does not support --payload_bank_write_dir")
    if str(args.payload_gate_mode) != "seacache":
        raise SystemExit(f"--mode {args.mode} does not support --payload_gate_mode")
    if args.forecast_guard_observe or str(args.forecast_guard_mode) != "auto":
        raise SystemExit(f"--mode {args.mode} does not support forecast guard options")


def _validate_foca_fine_args(args: argparse.Namespace) -> None:
    if args.mode != "FoCa_fine":
        if getattr(args, "foca_log_norms", False):
            raise SystemExit("--foca_log_norms is only supported by --mode FoCa_fine")
        return
    if float(args.foca_h) <= 0.0:
        raise SystemExit("--foca_h must be > 0")
    if args.payload_shadow_full_residual or args.payload_shadow_full_velocity:
        raise SystemExit("--mode FoCa_fine does not support payload shadow observers")
    if args.payload_bank_dir is not None or args.payload_wrong_seed_bank_dir is not None:
        raise SystemExit("--mode FoCa_fine does not support payload banks")
    if args.payload_bank_write_dir is not None:
        raise SystemExit("--mode FoCa_fine does not support --payload_bank_write_dir")
    if str(args.payload_gate_mode) != "seacache":
        raise SystemExit("--mode FoCa_fine uses fixed interval and does not support --payload_gate_mode")
    if args.forecast_guard_observe or str(args.forecast_guard_mode) != "auto":
        raise SystemExit("--mode FoCa_fine does not support forecast guard options")


def _payload_spec_for_args(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    if args.mode == "SeaCacheSegmentPayload":
        return seacache_segment_payload_modes._base_payload_mode(str(args.payload_mode)), "none"
    if args.mode == "SeaCacheFinePayload":
        return seacache_fine_payload_modes._payload_base_mode(str(args.payload_mode)), None
    return seacache_payload_modes._payload_spec(str(args.payload_mode))


def main() -> int:
    args = parse_args()
    _normalize_payload_mode_default(args)
    _validate_seacache_payload_args(args)
    _validate_seacache_fine_payload_args(args)
    _validate_seacache_segment_payload_args(args)
    _validate_l2p_args(args)
    _validate_foca_fine_args(args)
    if args.sencache_thresh_main is None:
        args.sencache_thresh_main = (
            float(args.sencache_thresh) if args.sencache_thresh is not None else 0.07
        )
    if args.mode == "SenCache" and args.sencache_sensitivity_path is None:
        raise SystemExit("--mode SenCache requires --sencache_sensitivity_path")
    sencache_table_identity = None
    if args.mode == "SenCache":
        table = load_sensitivity_table(args.sencache_sensitivity_path)
        sencache_table_identity = {
            "path": str(args.sencache_sensitivity_path),
            "sha256": table.sha256,
            "metadata": table.metadata,
        }

    # ---- Dtype ----
    import torch
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ---- Prompts + shard ----
    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load pipeline ----
    _tea_tag = (
        f"_{args.teacache_backbone}"
        + (f"-{args.teacache_variant}" if args.teacache_variant else "")
    )
    _l2p_weight_tag = args.l2p_weights.stem if args.l2p_weights is not None else "none"
    mode_label = {
        "original":        "original",
        "HiCache":         f"HiCache_i{args.interval}_o{args.max_order}_sig{args.hicache_sigma}_fe{args.first_enhance}",
        "TaylorSeer":      f"TaylorSeer_i{args.interval}_o{args.max_order}_fe{args.first_enhance}",
        "SeaCache":        f"SeaCache_t{args.seacache_thresh}_fe{args.first_enhance}",
        "SeaCachePayload": (
            f"SeaCachePayload_t{args.seacache_thresh}_pl{args.payload_mode}"
            f"_b{args.payload_blend}_sig{args.payload_sigma}_fe{args.first_enhance}"
            f"{'_gate' + args.payload_gate_mode if args.payload_gate_mode != 'seacache' else ''}"
            f"{'_fgobs' if args.forecast_guard_observe else ''}"
            f"{'_locked' if args.payload_schedule_dir is not None else ''}"
            f"{'_online_drift' if args.allow_online_payload_schedule_drift else ''}"
            f"{'_shadow' if args.payload_shadow_full_residual else ''}"
        ),
        "TeaCachePayload": (
            f"TeaCachePayload_t{args.teacache_thresh}_pl{args.payload_mode}"
            f"_b{args.payload_blend}_sig{args.payload_sigma}_fe{args.first_enhance}{_tea_tag}"
            f"{'_locked' if args.payload_schedule_dir is not None else ''}"
            f"{'_shadow' if args.payload_shadow_full_residual else ''}"
        ),
        "SeaCacheFinePayload": (
            f"SeaCacheFinePayload_t{args.seacache_thresh}_pl{args.payload_mode}"
            f"_sig{args.payload_sigma}_fe{args.first_enhance}"
            f"{'_gate' + args.payload_gate_mode if args.payload_gate_mode != 'seacache' else ''}"
            f"{'_locked' if args.payload_schedule_dir is not None else ''}"
            f"{'_vshadow' if args.payload_shadow_full_velocity else ''}"
        ),
        "SeaCacheSegmentPayload": (
            f"SeaCacheSegmentPayload_t{args.seacache_thresh}_layout{args.segment_layout}"
            f"_pl{args.payload_mode}_sig{args.payload_sigma}_fe{args.first_enhance}"
            f"{'_locked' if args.payload_schedule_dir is not None else ''}"
        ),
        "SVDCache": (
            f"SVDCache_i{args.interval}_e{args.svdcache_energy}_r{args.svdcache_max_rank}"
            f"_b{args.svdcache_beta}_fe{args.first_enhance}_{args.svdcache_basis_policy}"
        ),
        "TeaCache":        f"TeaCache_t{args.teacache_thresh}_fe{args.first_enhance}{_tea_tag}",
        "OriCache":        f"OriCache_t{args.oricache_thresh}_sig{args.oricache_signal}_fe{args.first_enhance}",
        "SenCache":        f"SenCache_ts{args.sencache_thresh_start}_tm{args.sencache_thresh_main}_K{args.sencache_K}_fe{args.first_enhance}",
        "HiCache_fine":    f"HiCache_fine_i{args.interval}_o{args.max_order}_sig{args.hicache_sigma}_fe{args.first_enhance}",
        "TaylorSeer_fine": f"TaylorSeer_fine_i{args.interval}_o{args.max_order}_fe{args.first_enhance}",
        "L2P": (
            f"L2P_final_hidden_i{args.interval}_fe{args.first_enhance}"
            f"_w{_l2p_weight_tag}{'_locked' if args.payload_schedule_dir is not None else ''}"
        ),
        "L2P_fine": (
            f"L2P_fine114_i{args.interval}_fe{args.first_enhance}"
            f"_w{_l2p_weight_tag}{'_locked' if args.payload_schedule_dir is not None else ''}"
        ),
        "FoCa_fine": (
            f"FoCa_fine_i{args.interval}_fe{args.first_enhance}"
            f"_hv{args.foca_heun_variant}_hp{args.foca_history_policy}"
            f"_d{args.foca_derivative}_h{args.foca_h:g}"
            f"{'_locked' if args.payload_schedule_dir is not None else ''}"
            f"{'_norms' if args.foca_log_norms else ''}"
        ),
    }[args.mode]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} dtype={args.dtype} mode={mode_label}",
          flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in {model_load_end - process_start:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
          f"(global idx {start}..{end - 1})", flush=True)

    # ---- Install cache method (after pipeline load) ----
    teardown = None
    reset_per_image = None
    l2p_weight_meta: dict[str, Any] = {}
    if args.mode == "HiCache":
        from flux import hicache
        teardown = hicache.install(
            pipe,
            interval=args.interval,
            max_order=args.max_order,
            sigma=args.hicache_sigma,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
        )
        reset_per_image = hicache.reset_per_image_state
    elif args.mode == "TaylorSeer":
        from flux import taylorseer
        teardown = taylorseer.install(
            pipe,
            interval=args.interval,
            max_order=args.max_order,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
        )
        reset_per_image = taylorseer.reset_per_image_state
    elif args.mode == "L2P":
        from flux import l2p
        teardown = l2p.install(
            pipe,
            weights_path=args.l2p_weights,
            interval=args.interval,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
            min_abs_weight=args.l2p_min_abs_weight,
        )
        reset_per_image = l2p.reset_per_image_state
    elif args.mode == "L2P_fine":
        from flux import l2p_fine
        teardown = l2p_fine.install(
            pipe,
            weights_path=args.l2p_weights,
            interval=args.interval,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
            min_abs_weight=args.l2p_min_abs_weight,
        )
        reset_per_image = l2p_fine.reset_per_image_state
    elif args.mode == "FoCa_fine":
        from flux import foca_fine
        teardown = foca_fine.install(
            pipe,
            interval=args.interval,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
            heun_variant=args.foca_heun_variant,
            history_policy=args.foca_history_policy,
            derivative=args.foca_derivative,
            h=args.foca_h,
            log_norms=args.foca_log_norms,
        )
        reset_per_image = foca_fine.reset_per_image_state
    elif args.mode == "SeaCache":
        from flux import seacache
        teardown = seacache.install(
            pipe,
            threshold=args.seacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
        )
        reset_per_image = seacache.reset_per_image_state
    elif args.mode == "SeaCachePayload":
        from flux import seacache_payload
        teardown = seacache_payload.install(
            pipe,
            threshold=args.seacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            payload_mode=args.payload_mode,
            payload_blend=args.payload_blend,
            payload_sigma=args.payload_sigma,
            shadow_full_residual=args.payload_shadow_full_residual,
            payload_bank_dir=args.payload_bank_dir,
            payload_wrong_seed_bank_dir=args.payload_wrong_seed_bank_dir,
            payload_bank_write_dir=args.payload_bank_write_dir,
            payload_gate_mode=args.payload_gate_mode,
            forecast_guard_observe=args.forecast_guard_observe,
            forecast_guard_mode=args.forecast_guard_mode,
            forecast_guard_tau=args.forecast_guard_tau,
            forecast_guard_quantile=args.forecast_guard_quantile,
            forecast_guard_window=args.forecast_guard_window,
            forecast_guard_warmup_updates=args.forecast_guard_warmup_updates,
            forecast_guard_age_gamma=args.forecast_guard_age_gamma,
            forecast_guard_scale_floor=args.forecast_guard_scale_floor,
            rfc_gate_tau=args.rfc_gate_tau,
        )
        reset_per_image = seacache_payload.reset_per_image_state
    elif args.mode == "TeaCachePayload":
        from flux import seacache_payload
        teardown = seacache_payload.install(
            pipe,
            threshold=args.teacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            payload_mode=args.payload_mode,
            payload_blend=args.payload_blend,
            payload_sigma=args.payload_sigma,
            shadow_full_residual=args.payload_shadow_full_residual,
            payload_bank_dir=args.payload_bank_dir,
            payload_wrong_seed_bank_dir=args.payload_wrong_seed_bank_dir,
            payload_bank_write_dir=args.payload_bank_write_dir,
            payload_gate_mode="teacache",
            forecast_guard_observe=args.forecast_guard_observe,
            forecast_guard_mode=args.forecast_guard_mode,
            forecast_guard_tau=args.forecast_guard_tau,
            forecast_guard_quantile=args.forecast_guard_quantile,
            forecast_guard_window=args.forecast_guard_window,
            forecast_guard_warmup_updates=args.forecast_guard_warmup_updates,
            forecast_guard_age_gamma=args.forecast_guard_age_gamma,
            forecast_guard_scale_floor=args.forecast_guard_scale_floor,
            teacache_backbone=args.teacache_backbone,
            teacache_variant=args.teacache_variant,
        )
        reset_per_image = seacache_payload.reset_per_image_state
    elif args.mode == "SeaCacheFinePayload":
        from flux import seacache_fine_payload
        teardown = seacache_fine_payload.install(
            pipe,
            threshold=args.seacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            payload_mode=args.payload_mode,
            payload_sigma=args.payload_sigma,
            log_payload_norms=args.payload_log_norms,
            shadow_full_velocity=args.payload_shadow_full_velocity,
            payload_gate_mode=args.payload_gate_mode,
            rfc_gate_tau=args.rfc_gate_tau,
        )
        reset_per_image = seacache_fine_payload.reset_per_image_state
    elif args.mode == "SeaCacheSegmentPayload":
        from flux import seacache_segment_payload
        teardown = seacache_segment_payload.install(
            pipe,
            threshold=args.seacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            payload_mode=args.payload_mode,
            payload_sigma=args.payload_sigma,
            segment_layout=args.segment_layout,
        )
        reset_per_image = seacache_segment_payload.reset_per_image_state
    elif args.mode == "SVDCache":
        from flux import svdcache
        teardown = svdcache.install(
            pipe,
            interval=args.interval,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
            energy=args.svdcache_energy,
            max_rank=args.svdcache_max_rank,
            beta=args.svdcache_beta,
            basis_policy=args.svdcache_basis_policy,
        )
        reset_per_image = svdcache.reset_per_image_state
    elif args.mode == "TeaCache":
        from flux import teacache
        teardown = teacache.install(
            pipe,
            threshold=args.teacache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            backbone=args.teacache_backbone,
            variant=args.teacache_variant,
        )
        reset_per_image = teacache.reset_per_image_state
    elif args.mode == "OriCache":
        from flux import oricache
        teardown = oricache.install(
            pipe,
            threshold=args.oricache_thresh,
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            signal=args.oricache_signal,
        )
        reset_per_image = oricache.reset_per_image_state
    elif args.mode == "SenCache":
        from flux import sencache
        teardown = sencache.install(
            pipe,
            sensitivity_path=str(args.sencache_sensitivity_path),
            threshold_start=float(args.sencache_thresh_start),
            threshold_main=float(args.sencache_thresh_main),
            num_steps=args.num_steps,
            first_enhance=args.first_enhance,
            max_skip=args.sencache_K,
            threshold_scale=args.sencache_threshold_scale,
            switch_ratio=args.sencache_switch_ratio,
            ret_steps=args.sencache_ret_steps,
            cutoff_steps=args.sencache_cutoff_steps,
        )
        reset_per_image = sencache.reset_per_image_state
    elif args.mode == "HiCache_fine":
        from flux import hicache_fine
        teardown = hicache_fine.install(
            pipe,
            interval=args.interval,
            max_order=args.max_order,
            sigma=args.hicache_sigma,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
        )
        reset_per_image = hicache_fine.reset_per_image_state
    elif args.mode == "TaylorSeer_fine":
        from flux import taylorseer_fine
        teardown = taylorseer_fine.install(
            pipe,
            interval=args.interval,
            max_order=args.max_order,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
        )
        reset_per_image = taylorseer_fine.reset_per_image_state
    # else: --mode original → no install, no teardown

    if args.mode in ("L2P", "L2P_fine"):
        l2p_weight_meta = dict(getattr(pipe.transformer, "_l2p_weights_meta", {}))

    print(f"[shard {args.shard_idx}] mode={mode_label} ready, beginning sampling", flush=True)

    # ---- Sampling loop ----
    per_image_records: list[dict] = []
    skipped = 0
    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            out_path = args.output_dir / image_filename(global_idx)
            if args.resume and out_path.is_file():
                skipped += 1
                continue

            per_image_seed = args.seed + global_idx
            generator = torch.Generator(device=device).manual_seed(int(per_image_seed))

            payload_action_steps = None
            if args.mode in (
                "SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload",
                "L2P", "L2P_fine", "FoCa_fine",
            ) and args.payload_schedule_dir is not None:
                expected_reference_mode: Any
                expected_reference_payload_mode: Any
                if args.mode == "TeaCachePayload":
                    expected_reference_mode = "TeaCachePayload"
                    expected_reference_payload_mode = "reuse"
                elif args.mode in ("L2P", "L2P_fine", "FoCa_fine"):
                    expected_reference_mode = (
                        "SeaCachePayload", "SeaCacheFinePayload",
                        "SeaCacheSegmentPayload", "L2P", "L2P_fine", "FoCa_fine",
                    )
                    expected_reference_payload_mode = (
                        "reuse", "fine_reuse", "segment_reuse", None,
                    )
                elif args.mode == "SeaCacheFinePayload":
                    expected_reference_mode = ("SeaCacheFinePayload", "SeaCachePayload")
                    expected_reference_payload_mode = ("fine_reuse", "reuse")
                elif args.mode == "SeaCacheSegmentPayload":
                    expected_reference_mode = ("SeaCacheSegmentPayload", "SeaCachePayload")
                    expected_reference_payload_mode = ("segment_reuse", "reuse")
                else:
                    expected_reference_mode = ("SeaCachePayload", "SenCache")
                    expected_reference_payload_mode = "reuse"
                payload_action_steps = _load_payload_action_steps(
                    args.payload_schedule_dir,
                    global_idx,
                    expected_num_steps=args.num_steps,
                    require_reuse_reference=True,
                    expected_reference_mode=expected_reference_mode,
                    expected_reference_payload_mode=expected_reference_payload_mode,
                    expected_payload_gate_mode=("teacache" if args.mode == "TeaCachePayload" else None),
                )
            if reset_per_image is not None:
                if args.mode in (
                    "SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload",
                    "SeaCacheSegmentPayload", "L2P", "L2P_fine", "FoCa_fine",
                ):
                    reset_per_image(pipe, action_steps=payload_action_steps, prompt_idx=global_idx)
                else:
                    reset_per_image(pipe)

            denoise_start = time.perf_counter()
            result = pipe(
                prompt=prompt,
                num_inference_steps=int(args.num_steps),
                guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
                height=(args.height // 16) * 16,
                width=(args.width // 16) * 16,
                max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
                num_images_per_prompt=1,
                generator=generator,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            denoise_s = time.perf_counter() - denoise_start

            result.images[0].save(out_path)  # PNG, lossless, no watermark

            cache_record = {}
            if args.mode in (
                "SeaCache", "SeaCachePayload", "SeaCacheFinePayload",
                "TeaCachePayload", "SeaCacheSegmentPayload", "TeaCache", "OriCache", "SenCache",
                "SVDCache", "L2P", "L2P_fine", "FoCa_fine",
            ):
                attr = {
                    "SeaCache": "seacache_decisions",
                    "SeaCachePayload": "seacache_payload_decisions",
                    "TeaCachePayload": "seacache_payload_decisions",
                    "SeaCacheFinePayload": "seacache_payload_decisions",
                    "SeaCacheSegmentPayload": "seacache_segment_payload_decisions",
                    "TeaCache": "teacache_decisions",
                    "OriCache": "oricache_decisions",
                    "SenCache": "sencache_decisions",
                    "SVDCache": "svdcache_decisions",
                    "L2P": "l2p_decisions",
                    "L2P_fine": "fine_cache_decisions",
                    "FoCa_fine": "fine_cache_decisions",
                }[args.mode]
                decisions = list(getattr(pipe.transformer, attr, []))
                if args.mode == "SenCache" and len(decisions) != int(args.num_steps):
                    raise RuntimeError(
                        f"SenCache decision logging failed for prompt {global_idx}: "
                        f"got {len(decisions)} rows, expected {int(args.num_steps)}"
                    )
                if decisions:
                    n_cached = sum(1 for d in decisions if int(d.get("u", 0)) == 1)
                    cache_record = {
                        "n_cached_steps": int(n_cached),
                        "n_full_steps": int(len(decisions) - n_cached),
                        "cached_ratio": float(n_cached / max(1, len(decisions))),
                    }
                    if args.mode == "TeaCache":
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "threshold": float(args.teacache_thresh),
                                "teacache_thresh": float(args.teacache_thresh),
                                "teacache_backbone": str(args.teacache_backbone),
                                "teacache_variant": args.teacache_variant,
                                "first_enhance": int(args.first_enhance),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode == "OriCache":
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "threshold": float(args.oricache_thresh),
                                "oricache_thresh": float(args.oricache_thresh),
                                "oricache_signal": str(args.oricache_signal),
                                "first_enhance": int(args.first_enhance),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode == "SenCache":
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "threshold": float(args.sencache_thresh_main),
                                "sencache_thresh_start": float(args.sencache_thresh_start),
                                "sencache_thresh_main": float(args.sencache_thresh_main),
                                "sencache_K": int(args.sencache_K),
                                "sencache_threshold_scale": str(args.sencache_threshold_scale),
                                "sencache_switch_ratio": float(args.sencache_switch_ratio),
                                "sencache_ret_steps": int(args.sencache_ret_steps),
                                "sencache_cutoff_steps": int(args.sencache_cutoff_steps),
                                "sencache_sensitivity_path": (
                                    None if args.sencache_sensitivity_path is None
                                    else str(args.sencache_sensitivity_path)
                                ),
                                "sencache_sensitivity_sha256": (
                                    sencache_table_identity["sha256"]
                                    if sencache_table_identity is not None else None
                                ),
                                "sencache_sensitivity_metadata": (
                                    sencache_table_identity["metadata"]
                                    if sencache_table_identity is not None else None
                                ),
                                "first_enhance": int(args.first_enhance),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode == "SVDCache":
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "interval": int(args.interval),
                                "first_enhance": int(args.first_enhance),
                                "energy_threshold": float(args.svdcache_energy),
                                "max_rank": int(args.svdcache_max_rank),
                                "beta": float(args.svdcache_beta),
                                "basis_policy": str(args.svdcache_basis_policy),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode in ("L2P", "L2P_fine"):
                        l2p_meta = dict(getattr(pipe.transformer, "_l2p_weights_meta", {}))
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "interval": int(args.interval),
                                "first_enhance": int(args.first_enhance),
                                "l2p_weights": None if args.l2p_weights is None else str(args.l2p_weights),
                                "l2p_weight_sha256": l2p_meta.get("sha256"),
                                "l2p_weight_format": l2p_meta.get("format"),
                                "l2p_target": (
                                    l2p_meta.get("target")
                                    if args.mode == "L2P" else l2p_meta.get("target", "fine_pregate_shared")
                                ),
                                "l2p_granularity": (
                                    "final_hidden" if args.mode == "L2P" else "fine_114_shared_w"
                                ),
                                "l2p_min_abs_weight": float(args.l2p_min_abs_weight),
                                "payload_schedule_dir": (
                                    None if args.payload_schedule_dir is None
                                    else str(args.payload_schedule_dir)
                                ),
                                "schedule_locked": bool(args.payload_schedule_dir is not None),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode == "FoCa_fine":
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "foca_official_reproduction": False,
                                "interval": int(args.interval),
                                "first_enhance": int(args.first_enhance),
                                "num_steps": int(args.num_steps),
                                "foca_target": "fine_pregate_114",
                                "foca_heun_variant": str(args.foca_heun_variant),
                                "foca_history_policy": str(args.foca_history_policy),
                                "foca_derivative": str(args.foca_derivative),
                                "foca_h": float(args.foca_h),
                                "foca_log_norms": bool(args.foca_log_norms),
                                "payload_schedule_dir": (
                                    None if args.payload_schedule_dir is None
                                    else str(args.payload_schedule_dir)
                                ),
                                "schedule_locked": bool(args.payload_schedule_dir is not None),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload"):
                        payload_base_mode, payload_control = _payload_spec_for_args(args)
                        segment_metadata = {}
                        if args.mode == "SeaCacheSegmentPayload":
                            from flux import seacache_segment_payload
                            segment_metadata = seacache_segment_payload.segment_metadata(pipe)
                        payload_threshold = (
                            float(args.teacache_thresh)
                            if args.mode == "TeaCachePayload" else float(args.seacache_thresh)
                        )
                        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
                            json.dumps({
                                "prompt_idx": int(global_idx),
                                "mode": args.mode,
                                "threshold": payload_threshold,
                                "seacache_thresh": (
                                    float(args.seacache_thresh)
                                    if args.mode != "TeaCachePayload" else None
                                ),
                                "teacache_thresh": (
                                    float(args.teacache_thresh)
                                    if args.mode == "TeaCachePayload" else None
                                ),
                                "teacache_backbone": (
                                    str(args.teacache_backbone)
                                    if args.mode == "TeaCachePayload" else None
                                ),
                                "teacache_variant": (
                                    args.teacache_variant
                                    if args.mode == "TeaCachePayload" else None
                                ),
                                "payload_mode": str(args.payload_mode),
                                "payload_base_mode": payload_base_mode,
                                "payload_control": payload_control,
                                "payload_blend": float(args.payload_blend),
                                "payload_sigma": float(args.payload_sigma),
                                "payload_log_norms": bool(args.payload_log_norms),
                                "segment_layout": (
                                    str(args.segment_layout)
                                    if args.mode == "SeaCacheSegmentPayload" else None
                                ),
                                **segment_metadata,
                                "payload_schedule_dir": (
                                    None if args.payload_schedule_dir is None
                                    else str(args.payload_schedule_dir)
                                ),
                                "allow_online_payload_schedule_drift": bool(args.allow_online_payload_schedule_drift),
                                "payload_bank_dir": (
                                    None if args.payload_bank_dir is None
                                    else str(args.payload_bank_dir)
                                ),
                                "payload_wrong_seed_bank_dir": (
                                    None if args.payload_wrong_seed_bank_dir is None
                                    else str(args.payload_wrong_seed_bank_dir)
                                ),
                                "payload_bank_write_dir": (
                                    None if args.payload_bank_write_dir is None
                                    else str(args.payload_bank_write_dir)
                                ),
                                "payload_gate_mode": str(args.payload_gate_mode),
                                "forecast_guard_observe": bool(args.forecast_guard_observe),
                                "forecast_guard_mode": str(args.forecast_guard_mode),
                                "forecast_guard_tau": float(args.forecast_guard_tau),
                                "forecast_guard_quantile": float(args.forecast_guard_quantile),
                                "forecast_guard_window": int(args.forecast_guard_window),
                                "forecast_guard_warmup_updates": int(args.forecast_guard_warmup_updates),
                                "forecast_guard_age_gamma": float(args.forecast_guard_age_gamma),
                                "forecast_guard_scale_floor": float(args.forecast_guard_scale_floor),
                                "rfc_gate_tau": (
                                    None if args.rfc_gate_tau is None else float(args.rfc_gate_tau)
                                ),
                                "payload_shadow_full_residual": bool(args.payload_shadow_full_residual),
                                "payload_shadow_full_velocity": bool(args.payload_shadow_full_velocity),
                                "n_cached": int(n_cached),
                                "n_total": int(len(decisions)),
                                "cached_ratio": float(n_cached / max(1, len(decisions))),
                                "per_step": decisions,
                            }, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

            per_image_records.append({
                "idx": global_idx,
                "denoise_s": float(denoise_s),
                # FluxPipeline runs VAE decode inside pipe(); we account the entire call
                # as "denoise" and keep decode at 0 for schema compatibility with the
                # BFL pipeline which used to break these out.
                "decode_s": 0.0,
                **cache_record,
            })
            if (local_idx + 1) % 10 == 0 or local_idx == len(shard_prompts) - 1:
                print(f"[shard {args.shard_idx}] {local_idx + 1}/{len(shard_prompts)}  "
                      f"idx={global_idx}  {denoise_s:.2f}s  -> {out_path.name}", flush=True)
    finally:
        if teardown is not None:
            teardown()

    if skipped:
        print(f"[shard {args.shard_idx}] resumed: skipped {skipped} existing images", flush=True)

    # ---- Per-shard timing ----
    if per_image_records:
        process_end = time.perf_counter()
        _interval_mode = args.mode in (
            "HiCache", "TaylorSeer", "SVDCache", "HiCache_fine", "TaylorSeer_fine",
            "L2P", "L2P_fine", "FoCa_fine",
        )
        _order_mode = args.mode in ("HiCache", "TaylorSeer", "HiCache_fine", "TaylorSeer_fine")
        _hermite_mode = args.mode in ("HiCache", "HiCache_fine")
        config = {
            "cache_mode": mode_label,
            "mode_raw": args.mode,
            "num_steps": int(args.num_steps),
            "threshold": (
                float(args.seacache_thresh)
                if args.mode in (
                    "SeaCache", "SeaCachePayload", "SeaCacheFinePayload",
                    "SeaCacheSegmentPayload",
                ) else float(args.teacache_thresh)
                if args.mode in ("TeaCache", "TeaCachePayload") else float(args.oricache_thresh)
                if args.mode == "OriCache" else float(args.sencache_thresh_main)
                if args.mode == "SenCache" else None
            ),
            "interval": int(args.interval) if _interval_mode else 0,
            "max_order": int(args.max_order) if _order_mode else 0,
            "hicache_sigma": float(args.hicache_sigma) if _hermite_mode else None,
            "seacache_thresh": (
                float(args.seacache_thresh)
                if args.mode in (
                    "SeaCache", "SeaCachePayload", "SeaCacheFinePayload",
                    "SeaCacheSegmentPayload",
                ) else None
            ),
            "payload_mode": (
                str(args.payload_mode)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
            ),
            "payload_base_mode": (
                _payload_spec_for_args(args)[0]
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
            ),
            "payload_control": (
                _payload_spec_for_args(args)[1]
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
            ),
            "payload_blend": (
                float(args.payload_blend)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
            ),
            "payload_sigma": (
                float(args.payload_sigma)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
            ),
            "segment_layout": str(args.segment_layout) if args.mode == "SeaCacheSegmentPayload" else None,
            "payload_log_norms": (
                bool(args.payload_log_norms)
                if args.mode in ("SeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "payload_shadow_full_residual": (
                bool(args.payload_shadow_full_residual)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "payload_shadow_full_velocity": (
                bool(args.payload_shadow_full_velocity)
                if args.mode in ("SeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "payload_schedule_dir": (
                str(args.payload_schedule_dir)
                if args.mode in (
                    "SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload",
                    "L2P", "L2P_fine", "FoCa_fine",
                ) and args.payload_schedule_dir is not None
                else None
            ),
            "allow_online_payload_schedule_drift": (
                bool(args.allow_online_payload_schedule_drift)
                if args.mode == "SeaCachePayload" else None
            ),
            "payload_bank_dir": (
                str(args.payload_bank_dir)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") and args.payload_bank_dir is not None
                else None
            ),
            "payload_wrong_seed_bank_dir": (
                str(args.payload_wrong_seed_bank_dir)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") and args.payload_wrong_seed_bank_dir is not None
                else None
            ),
            "payload_bank_write_dir": (
                str(args.payload_bank_write_dir)
                if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") and args.payload_bank_write_dir is not None
                else None
            ),
            "payload_gate_mode": str(args.payload_gate_mode) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None,
            "rfc_gate_tau": (
                None if args.rfc_gate_tau is None else float(args.rfc_gate_tau)
            ) if args.mode in ("SeaCachePayload", "SeaCacheFinePayload") else None,
            "forecast_guard_observe": (
                bool(args.forecast_guard_observe) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "forecast_guard_mode": str(args.forecast_guard_mode) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None,
            "forecast_guard_tau": float(args.forecast_guard_tau) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None,
            "forecast_guard_quantile": (
                float(args.forecast_guard_quantile) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "forecast_guard_window": int(args.forecast_guard_window) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None,
            "forecast_guard_warmup_updates": (
                int(args.forecast_guard_warmup_updates) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "forecast_guard_age_gamma": (
                float(args.forecast_guard_age_gamma) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "forecast_guard_scale_floor": (
                float(args.forecast_guard_scale_floor) if args.mode in ("SeaCachePayload", "TeaCachePayload", "SeaCacheFinePayload") else None
            ),
            "teacache_thresh": float(args.teacache_thresh) if args.mode in ("TeaCache", "TeaCachePayload") else None,
            "teacache_backbone": args.teacache_backbone if args.mode in ("TeaCache", "TeaCachePayload") else None,
            "teacache_variant": args.teacache_variant if args.mode in ("TeaCache", "TeaCachePayload") else None,
            "oricache_thresh": float(args.oricache_thresh) if args.mode == "OriCache" else None,
            "oricache_signal": str(args.oricache_signal) if args.mode == "OriCache" else None,
            "sencache_sensitivity_path": (str(args.sencache_sensitivity_path)
                                          if args.mode == "SenCache" else None),
            "sencache_sensitivity_sha256": (
                sencache_table_identity["sha256"]
                if args.mode == "SenCache" and sencache_table_identity is not None
                else None
            ),
            "sencache_sensitivity_metadata": (
                sencache_table_identity["metadata"]
                if args.mode == "SenCache" and sencache_table_identity is not None
                else None
            ),
            "sencache_thresh_start": (float(args.sencache_thresh_start)
                                      if args.mode == "SenCache" else None),
            "sencache_thresh_main": (float(args.sencache_thresh_main)
                                     if args.mode == "SenCache" else None),
            "sencache_K": int(args.sencache_K) if args.mode == "SenCache" else None,
            "sencache_threshold_scale": (str(args.sencache_threshold_scale)
                                         if args.mode == "SenCache" else None),
            "sencache_switch_ratio": (float(args.sencache_switch_ratio)
                                      if args.mode == "SenCache" else None),
            "sencache_ret_steps": int(args.sencache_ret_steps) if args.mode == "SenCache" else None,
            "sencache_cutoff_steps": int(args.sencache_cutoff_steps) if args.mode == "SenCache" else None,
            "svdcache_energy": float(args.svdcache_energy) if args.mode == "SVDCache" else None,
            "svdcache_max_rank": int(args.svdcache_max_rank) if args.mode == "SVDCache" else None,
            "svdcache_beta": float(args.svdcache_beta) if args.mode == "SVDCache" else None,
            "svdcache_basis_policy": str(args.svdcache_basis_policy) if args.mode == "SVDCache" else None,
            "l2p_weights": str(args.l2p_weights) if args.mode in ("L2P", "L2P_fine") else None,
            "l2p_weight_sha256": l2p_weight_meta.get("sha256") if args.mode in ("L2P", "L2P_fine") else None,
            "l2p_weight_format": l2p_weight_meta.get("format") if args.mode in ("L2P", "L2P_fine") else None,
            "l2p_target": (
                l2p_weight_meta.get("target", "final_hidden" if args.mode == "L2P" else "fine_pregate_shared")
                if args.mode in ("L2P", "L2P_fine") else None
            ),
            "l2p_granularity": (
                "final_hidden" if args.mode == "L2P"
                else ("fine_114_shared_w" if args.mode == "L2P_fine" else None)
            ),
            "l2p_min_abs_weight": (
                float(args.l2p_min_abs_weight) if args.mode in ("L2P", "L2P_fine") else None
            ),
            "foca_official_reproduction": False if args.mode == "FoCa_fine" else None,
            "foca_target": "fine_pregate_114" if args.mode == "FoCa_fine" else None,
            "foca_heun_variant": str(args.foca_heun_variant) if args.mode == "FoCa_fine" else None,
            "foca_history_policy": str(args.foca_history_policy) if args.mode == "FoCa_fine" else None,
            "foca_derivative": str(args.foca_derivative) if args.mode == "FoCa_fine" else None,
            "foca_h": float(args.foca_h) if args.mode == "FoCa_fine" else None,
            "foca_log_norms": bool(args.foca_log_norms) if args.mode == "FoCa_fine" else None,
            "first_enhance": int(args.first_enhance) if args.mode != "original" else None,
            "model_id": args.model_id,
            "model_name": args.model_name,
            "guidance": float(args.guidance),
            "width": int(args.width),
            "height": int(args.height),
            "dtype": args.dtype,
            "shard_idx": int(args.shard_idx),
            "shard_count": int(args.shard_count),
            "base_seed": int(args.seed),
            "batch_size": 1,
            "git_sha": _git_rev_parse_head(),
        }
        device_str = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
        write_timing_json(
            timing_path,
            per_image=per_image_records,
            config=config,
            model_load_s=float(model_load_end - process_start),
            wallclock_total_s=float(process_end - process_start),
            device=device_str,
        )
        print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
