"""Online marginal-risk features for the stateful-marginal plan.

docs/research_plan_stateful_marginal.md §4. `MarginalFeatureTracker`
maintains, per trajectory, the cheap online features that Experiment SM-A
logs, SM-B labels against, and an SM-D gate would consume — computed
identically across all three so the estimator trained in SM-C transfers.

The plan's §6 omits this shared module; it is added here because q_n /
C^stale / C^traj / C^mem must be byte-for-byte consistent across runners.

Features (plan §4.2-4.3), all observed *entering* step n:

  gap      g_n     = n - a_n,  a_n = last full-refresh step
  q        q_n     = P_hat_{a->n} * S_hat_n * |h_n| * A_hat_n   (clean-gap score)
  c_stale          staleness accumulator, resets to 0 on a full step
  c_traj           trajectory-risk accumulator, NO reset on a full step
  c_mem            memory-bias proxy (variant 0 / 1 / 2)

By default, P_hat is the accumulated gate-feature drift since the last refresh a:

  P_hat_{a->n} = sum_{j=a+1}^{n} psi_drift_j                    (plan §4.2 P_acc)

where psi_drift_j = rel_l1(psi_j, psi_{j-1}) is the per-step drift of the
gate feature (modulated first-block input). Some closed-loop gates may instead
provide an explicit P value, e.g. anchor-to-current displacement
rel_l1(psi_n, psi_a). The caller computes psi_drift and any explicit P inside
its own forward — where psi is already on hand — and passes the scalar in, so
this tracker stays torch-free. S_hat_n, A_hat_n and |h_n| are per-step
calibration arrays passed at construction: |h_n| from q_k_schedule.json (Q_k),
S/A from the Phase-1 s_k / a_k probe calibration tables (seacache method, mean
over prompts).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class MarginalFeatureTracker:
    """Per-trajectory online feature state machine (plan §4).

    Usage per step n (0..N-1), in order:
        feats = tracker.observe(n, psi_drift_n)  # feature dict entering step n
        ...decide / look up u_n...
        tracker.commit(n, u_n)                   # apply the cache/full decision

    `observe` must be called for every step (full and cache) so the
    accumulated gate-feature drift stays continuous. `psi_drift_n` is
    rel_l1(psi_n, psi_{n-1}); pass 0.0 for the first step.
    """

    def __init__(self, q_abs: Sequence[float], s_cal: Sequence[float],
                 a_cal: Sequence[float], *, lam_traj: float = 1.0,
                 lam_mem: float = 1.0, alpha: float = 1.0,
                 mem_variant: int = 1) -> None:
        self.q_abs = [float(x) for x in q_abs]   # |H_n|, per step
        self.s_cal = [float(x) for x in s_cal]   # S_hat_n, per step
        self.a_cal = [float(x) for x in a_cal]   # A_hat_n, per step
        n = len(self.q_abs)
        if not (len(self.s_cal) == n and len(self.a_cal) == n):
            raise ValueError("q_abs / s_cal / a_cal must have equal length")
        if mem_variant not in (0, 1, 2):
            raise ValueError("mem_variant must be 0, 1 or 2")
        self.lam_traj = float(lam_traj)
        self.lam_mem = float(lam_mem)
        self.alpha = float(alpha)
        self.mem_variant = int(mem_variant)
        self.reset()

    def reset(self) -> None:
        """Clear all per-trajectory state before a new run."""
        self._last_refresh = 0      # step index a of the most recent full step
        self._p_acc = 0.0           # P_hat_{a->n}
        self._last_q = 0.0          # q_n stashed by observe() for commit()
        self._c_stale = 0.0
        self._c_traj = 0.0
        self._c_mem = 0.0

    def observe(self, n: int, psi_drift: float,
                p_effective: Optional[float] = None) -> Dict[str, float]:
        """Process step n's gate-feature drift; return the online feature dict.

        `psi_drift` = rel_l1(psi_n, psi_{n-1}), 0.0 for the first step. It is
        accumulated into the path-length P_hat. If `p_effective` is provided,
        it is used as the P factor in q_n while `p_acc` is still reported as the
        path-length diagnostic. Does not apply the cache/full decision — call
        commit() for that.
        """
        d_n = float(psi_drift)
        self._p_acc += d_n
        p_eff = self._p_acc if p_effective is None else float(p_effective)

        g_n = n - self._last_refresh
        q_n = p_eff * self.s_cal[n] * self.q_abs[n] * self.a_cal[n]
        self._last_q = q_n
        return {
            "step": n,
            "gap": g_n,
            "psi_drift": d_n,
            "p_acc": self._p_acc,
            "p_eff": p_eff,
            "q": q_n,
            "c_stale": self._c_stale,
            "c_traj": self._c_traj,
            "c_mem": self._c_mem,
        }

    def commit(self, n: int, u_n: int, refreshed: Optional[bool] = None) -> None:
        """Apply the step-n decision u_n (0 full / 1 cache), advance state.

        `refreshed` marks whether a full step rebuilt the cache memory; for
        zero-order residual reuse a full step always does, so it defaults to
        `u_n == 0`. Accumulator recurrences are plan §4.3.
        """
        if u_n not in (0, 1):
            raise ValueError("u_n must be 0 (full) or 1 (cache)")
        refreshed = (u_n == 0) if refreshed is None else bool(refreshed)
        q = self._last_q
        old_traj = self._c_traj

        # C^traj: lambda * old + u_n * q   (no reset on full — plan §4.3)
        self._c_traj = self.lam_traj * old_traj + (q if u_n == 1 else 0.0)
        # C^stale: 0 on full, else accumulate
        self._c_stale = 0.0 if u_n == 0 else self._c_stale + q
        # C^mem: refresh on a drifted latent biases the new memory
        if self.mem_variant == 0:
            self._c_mem = 0.0
        elif u_n == 0 and refreshed:
            src = old_traj if self.mem_variant == 1 else q
            self._c_mem = self.alpha * src
        else:
            self._c_mem = self.lam_mem * self._c_mem

        if u_n == 0:                       # full step refreshes the gap origin
            self._p_acc = 0.0
            self._last_refresh = n


def features_header() -> List[str]:
    """Column order for the per-step feature rows logged by SM runners."""
    return ["step", "gap", "psi_drift", "p_acc", "p_eff", "q",
            "c_stale", "c_traj", "c_mem"]
