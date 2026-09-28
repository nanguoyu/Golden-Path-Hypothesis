"""SeaCache accumulator replay over a virtual stale-memory window.

`docs/research_plan_method_native_sqa.md` §4.1 defines the SeaCache state
machine and §6 E1 defines the synthetic stale-memory action used in the
forced-stale-gap ranking experiment. For an action `(a, n)` we ask: if the
SeaCache gate had decided to cache every step in `(a, n-1]` (forced virtual
cache from anchor `a`), what value would its accumulator have at the
decision point of step `n`?

This module implements that replay against a pre-logged full no-cache
trajectory. The trace is produced by `flux/sqa_trace.py` and is the
ground truth `psi_n` (raw first-block-norm modulated input) plus
`psi_filtered_n` (SEA / Wiener-filtered version, same as
`flux/seacache.py:_seacache_forward` lines 135-149).

Byte-faithful rule (matches `flux/seacache.py` exactly):

  * Step `a` is the virtual force-full anchor:
      - accumulator resets to 0 at the start of step `a`
      - `previous_modulated_input` after step `a` is the **raw**
        `psi_a` (force-full branch does not apply SEA filter)
  * Steps `j = a+1, a+2, ..., n` are virtually cached (gate would have
    decided cache, regardless of threshold):
      - increment = `rel_l1(psi_filtered[j], prev)` is added to acc
      - `prev` is then updated to `psi_filtered[j]`
  * `P_Sea(a, n)` is the accumulator value at the decision point of
    step `n`, i.e. after the step-`n` increment has been added.

The first increment after the anchor (j = a + 1) therefore compares
`psi_filtered[a+1]` against the raw `psi[a]` — this is the
"filtered current vs raw previous" cross-type comparison flagged by
plan §4.1.

A real SeaCache rollout caches OR resets at every step depending on
whether the accumulator crosses threshold. The replay deliberately
forces the cache decision at every step in (a, n] so the accumulator
never resets inside the window. The plan calls this a synthetic
stale-memory gap action; it is the cleanest single-action label and
is **not** the unique ground truth for deployment cache harm.

Public surface:
    replay_p_sea(psi_raw_a,        — accumulator value at step n's decision point
                 psi_filtered_seq)
    replay_p_sea_from_trace(trace, — convenience wrapper that pulls psi_raw[a] and
                            a, n)    psi_filtered[a+1:n+1] from a logged trace dict

The trace dict shape expected by the convenience wrapper is documented in
`flux/sqa_trace.py`; minimally it has `psi_raw[k]` and `psi_filtered[k]`
indexable by step `k`.

ASSUMPTION (important for callers): this module is designed for the E1
full-prefix / virtual-history setting where the trace is a real full
no-cache trajectory. The synthetic anchor `a` is treated as a virtual
force-full step regardless of whether the underlying rollout had any
caching. For a no-cache trace, every step's stored modulated_inp is the
raw `first_block.norm1(...)` output, so `psi_raw[a]` is the right
anchor reference. If a future caller feeds a trace from a real cached
rollout, the `psi_raw[a]` vs `psi_filtered[a]` distinction at the
anchor matters and the docstring promise breaks.

Callers should prefer `replay_p_sea_from_trace` over raw `replay_p_sea`
to avoid off-by-one slicing of `psi_filtered`. The window `(a, n]` is
inclusive of `n`, exclusive of `a`, so the right slice is
`psi_filtered[a+1:n+1]` (length `n - a`).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from lib.gates import rel_l1


def replay_p_sea(
    psi_raw_at_a: torch.Tensor,
    psi_filtered_seq: Sequence[torch.Tensor],
) -> float:
    """Replay the SeaCache accumulator over `(a, n]`.

    Args:
        psi_raw_at_a: raw `modulated_inp` at the virtual force-full anchor `a`.
            This is the output of `first_block.norm1(inp, emb=temb)[0]` BEFORE
            any SEA / Wiener filter is applied. Equivalent to the value
            `previous_modulated_input` would hold immediately after step `a`
            in `flux/seacache.py:_seacache_forward` line 157, when step `a`
            took the force-full branch.
        psi_filtered_seq: SEA-filtered `modulated_inp` at every step in
            `(a, n]`, in increasing step order. Length must equal `n - a`.
            Element `k` (0-indexed) corresponds to step `a + 1 + k`.

    Returns:
        Accumulator value at the decision point of step `n`, i.e. after the
        increment for step `n` has been added. This is `P_Sea(a, n)` per
        plan §4.1.

    Notes:
        * `len(psi_filtered_seq) == 0` returns 0.0 — defines the degenerate
          case `n == a`. The plan's action grid requires `n > a`, so callers
          should not hit this path in normal use, but we keep it well-defined
          rather than raising.
        * Tensor shapes must match between `psi_raw_at_a` and every element
          of `psi_filtered_seq`. We do not check this — `rel_l1` would fail
          loudly with a broadcasting error.
    """
    if len(psi_filtered_seq) == 0:
        return 0.0

    acc = 0.0
    prev = psi_raw_at_a
    for psi_filtered_j in psi_filtered_seq:
        acc += rel_l1(psi_filtered_j, prev)
        prev = psi_filtered_j
    return acc


def replay_p_sea_from_trace(
    trace: Any,
    a: int,
    n: int,
    *,
    psi_raw_key: str = "psi_raw",
    psi_filtered_key: str = "psi_filtered",
) -> float:
    """Convenience wrapper: pull the right tensors from a trace dict.

    Args:
        trace: a dict-like (or attribute-like) object with two keys
            `psi_raw` and `psi_filtered`, each a sequence indexable by step.
        a, n: anchor and decision-point step indices. Must satisfy
            `0 <= a < n < num_steps` and `a` should be a step at which
            SeaCache would force-full (so `psi_raw[a]` is what
            `previous_modulated_input` would hold after step `a` —
            see plan §6 E1 for the synthetic-anchor semantics).
        psi_raw_key, psi_filtered_key: override if the trace uses different
            names.

    Returns:
        `P_Sea(a, n)` per `replay_p_sea`.
    """
    def _get(k: str):
        if isinstance(trace, dict):
            return trace[k]
        return getattr(trace, k)

    psi_raw = _get(psi_raw_key)
    psi_filtered = _get(psi_filtered_key)

    if not (0 <= a < n):
        raise ValueError(f"replay_p_sea_from_trace requires 0 <= a < n; got a={a}, n={n}")
    if n >= len(psi_filtered):
        raise ValueError(
            f"trace length insufficient: need psi_filtered[{n}], have len={len(psi_filtered)}"
        )

    return replay_p_sea(
        psi_raw_at_a=psi_raw[a],
        psi_filtered_seq=[psi_filtered[j] for j in range(a + 1, n + 1)],
    )
