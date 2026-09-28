"""Extract Q_k = |H_k| = |tau_{k+1} - tau_k| from the FLUX scheduler.

Per research plan §9.3:
    Q_k = |H_k| = |tau_{k+1} - tau_k|

For FLUX (rectified flow + dynamic shift), tau_k is taken to be sigmas[k] —
the scheduler's sigma values, which decrease monotonically from ~1.0 (full
noise at step 0) to 0.0 (clean latent at step N). The Euler update at step k
is x_{k+1} = x_k + (sigmas[k+1] - sigmas[k]) * v_k = x_k + H_k * v_k. The
solver step size H_k is therefore the canonical "tau increment" for FLUX.

Q_k is fully analytical given (num_steps, height, width, model_id); it is
identical across prompts and seeds. Output is a single JSON containing
sigmas, timesteps, H_k, Q_k arrays + metadata sufficient to reproduce.

CLI:
    python analysis/extract_q_k.py --output q_k_schedule.json \\
           --num_steps 50 --height 1024 --width 1024

Output schema:
    {
      "model_id": str, "num_steps": int, "height": int, "width": int,
      "image_seq_len": int, "mu": float,
      "sigmas_init": [...],         # the linspace fed into set_timesteps (length N)
      "tau_k": [sigmas[0..N]],      # length N+1
      "timesteps": [...],           # length N, scheduler timesteps (sigmas * 1000)
      "H_k": [tau[k+1]-tau[k]],     # length N (negative numbers)
      "Q_k": [|H_k|],               # length N (positive; the Q_k that pairs with step k)
      "doc": "..."
    }

Q_k indexing convention: Q_k[k] is the step size taken AT step k (from x_k
to x_{k+1}). For the oracle / probe sweep that measures effects at step k,
use Q_k[k] (k in 1..N-1; k=0 excluded because the oracle skips k=0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """Replicates diffusers FluxPipeline.calculate_shift (the FLUX-specific
    image-size-dependent time warp). Source:
    diffusers/src/diffusers/pipelines/flux/pipeline_flux.py

    All shift parameters are MANDATORY — pass the real values from
    scheduler.config (NOT hardcoded). FLUX.1-dev's scheduler config carries
    max_shift=1.15 (NOT 1.16 as some older diffusers used as fallback);
    using the wrong value compounds to ~19% z_N drift over 50 steps.
    Caught by Check 1 v2 diagnostic (job 50175).
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump per-step Q_k = |tau_{k+1} - tau_k| for FLUX runs."
    )
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path.")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy diffusers import (CPU-only; only the scheduler config is loaded).
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # image_seq_len for FLUX: latent = height // vae_scale_factor (=8),
    # then patch size = 2 in the transformer. So per-axis tokens = H // 16.
    H_p = args.height // 16
    W_p = args.width // 16
    image_seq_len = H_p * W_p

    # The default `sigmas` fed into set_timesteps is a uniform descending grid.
    # Matches diffusers FluxPipeline default when caller doesn't override sigmas.
    sigmas_init = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps).tolist()
    cfg = scheduler.config
    mu = calculate_shift(
        image_seq_len,
        base_seq_len=cfg.get("base_image_seq_len", 256),
        max_seq_len=cfg.get("max_image_seq_len", 4096),
        base_shift=cfg.get("base_shift", 0.5),
        max_shift=cfg.get("max_shift", 1.16),  # fallback only; FLUX.1-dev has 1.15
    )

    scheduler.set_timesteps(sigmas=sigmas_init, mu=mu, device="cpu")
    sigmas = scheduler.sigmas.detach().cpu().numpy().tolist()        # length N+1
    timesteps = scheduler.timesteps.detach().cpu().numpy().tolist()  # length N

    if len(sigmas) != args.num_steps + 1:
        raise RuntimeError(
            f"Unexpected sigmas length: got {len(sigmas)}, expected {args.num_steps + 1}"
        )

    tau_k = sigmas
    H_k = [float(sigmas[i + 1] - sigmas[i]) for i in range(args.num_steps)]
    Q_k = [abs(h) for h in H_k]

    payload = {
        "model_id": args.model_id,
        "num_steps": int(args.num_steps),
        "height": int(args.height),
        "width": int(args.width),
        "image_seq_len": int(image_seq_len),
        "mu": float(mu),
        "sigmas_init": sigmas_init,
        "tau_k": tau_k,
        "timesteps": timesteps,
        "H_k": H_k,
        "Q_k": Q_k,
        "doc": (
            "Q_k = |H_k| = |tau_{k+1} - tau_k| with tau_k = sigmas[k]. "
            "H_k and Q_k are length N; H_k[k] is the solver step taken AT step k "
            "(from x_k to x_{k+1}). Pair Q_k[k] with P_k(k), S_k(k), R_k^oracle(k). "
            "For rectified-flow Euler this is the canonical solver step size; "
            "for Heun/RK extend to Q_k^eff per plan §9.3."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Q_k schedule written to {args.output}")
    print(f"  model_id={args.model_id}  num_steps={args.num_steps}  "
          f"size={args.height}x{args.width}  image_seq_len={image_seq_len}  mu={mu:.4f}")
    print(f"  tau_k[0..5] = {[f'{v:.4f}' for v in tau_k[:6]]}")
    print(f"  tau_k[N-5..N] = {[f'{v:.4f}' for v in tau_k[-6:]]}")
    print(f"  Q_k[0..5] = {[f'{v:.4f}' for v in Q_k[:6]]}")
    print(f"  Q_k[N-5..N-1] = {[f'{v:.4f}' for v in Q_k[-5:]]}")
    print(f"  Q_k stats: min={min(Q_k):.6f}  max={max(Q_k):.6f}  "
          f"sum={sum(Q_k):.4f}  (should sum to ~1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
