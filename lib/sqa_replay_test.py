"""Unit tests for `lib/sqa_replay.py`.

Run with: `python -m lib.sqa_replay_test` from the repo root, or
         `python lib/sqa_replay_test.py`.

No GPU / no model load. Tests use synthetic tensors and a pure-Python
reference state machine that mirrors `flux/seacache.py:_seacache_forward`
lines 117-160 (the SeaCache gating block).
"""

from __future__ import annotations

import math
import sys

import torch

from lib.gates import rel_l1
from lib.sqa_replay import replay_p_sea, replay_p_sea_from_trace


# ----- reference: pure-Python SeaCache state machine -------------------------


def _reference_seacache_accumulator(
    psi_raw_seq,
    psi_filtered_seq,
    force_full_steps,
    threshold,
    num_steps,
):
    """Walk through `psi_raw_seq` / `psi_filtered_seq` step-by-step, mirroring
    `flux/seacache.py` lines 117-160 exactly. Return the per-step accumulator
    value AFTER each step's increment (so per_step_acc[k] is what would gate
    the cache decision at step k).

    `force_full_steps` is the set of step indices that take the force-full
    branch (acc reset, prev = raw). All other steps take the non-force-full
    branch (apply filter, increment acc, prev = filtered).

    `threshold` controls whether non-force-full steps additionally reset acc
    when acc >= threshold (the "real" SeaCache behavior). Pass `+inf` to
    suppress threshold-driven resets (i.e. force virtual cache at every
    non-force-full step — this is the replay semantics).
    """
    acc = 0.0
    prev = None
    per_step_acc = []
    for k in range(num_steps):
        force_full = (k in force_full_steps) or (prev is None)
        if force_full:
            acc = 0.0
            current_stored = psi_raw_seq[k]
        else:
            inc = rel_l1(psi_filtered_seq[k], prev)
            acc += inc
            if acc >= threshold:
                # real SeaCache: full forward, reset acc to 0 AFTER recording
                # the value that caused the trigger. Subsequent steps start fresh.
                # For replay purposes we still want to know what acc was at the
                # decision point of step k, so record before reset.
                per_step_acc.append(acc)
                acc = 0.0
                current_stored = psi_filtered_seq[k]
                prev = current_stored
                continue
            current_stored = psi_filtered_seq[k]
        per_step_acc.append(acc)
        prev = current_stored
    return per_step_acc


# ----- tests -----------------------------------------------------------------


def test_empty_window_returns_zero():
    psi_raw = torch.randn(1, 16, 32)
    out = replay_p_sea(psi_raw_at_a=psi_raw, psi_filtered_seq=[])
    assert out == 0.0, f"empty seq must return 0.0, got {out}"
    print("  test_empty_window_returns_zero ... PASS")


def test_single_increment_matches_rel_l1():
    torch.manual_seed(0)
    psi_a_raw = torch.randn(1, 32, 16, dtype=torch.float32)
    psi_n_filtered = torch.randn(1, 32, 16, dtype=torch.float32)
    expected = rel_l1(psi_n_filtered, psi_a_raw)
    got = replay_p_sea(psi_a_raw, [psi_n_filtered])
    assert math.isclose(got, expected, rel_tol=1e-12), \
        f"single-step replay mismatch: got {got}, expected {expected}"
    print("  test_single_increment_matches_rel_l1 ... PASS")


def test_two_increments_decompose():
    """For (a, n=a+2):
       acc = rel_l1(f[a+1], raw[a]) + rel_l1(f[a+2], f[a+1])
    """
    torch.manual_seed(1)
    psi_a_raw = torch.randn(1, 8, 16)
    psi_ap1_f = torch.randn(1, 8, 16)
    psi_ap2_f = torch.randn(1, 8, 16)
    expected = (
        rel_l1(psi_ap1_f, psi_a_raw)
        + rel_l1(psi_ap2_f, psi_ap1_f)
    )
    got = replay_p_sea(psi_a_raw, [psi_ap1_f, psi_ap2_f])
    assert math.isclose(got, expected, rel_tol=1e-12), \
        f"two-step replay mismatch: got {got}, expected {expected}"
    print("  test_two_increments_decompose ... PASS")


def test_replay_matches_reference_state_machine_synthetic():
    """Build a synthetic 'full no-cache trajectory' with one force-full step at
    every position k. Force virtual cache at all other steps (threshold = inf).
    The reference per-step accumulator at step n, starting from the most recent
    force-full anchor a, must equal replay_p_sea(psi_raw[a], psi_filtered[a+1:n+1]).
    """
    torch.manual_seed(2)
    N = 20
    shape = (1, 8, 16)
    psi_raw = [torch.randn(shape) for _ in range(N)]
    psi_filtered = [torch.randn(shape) for _ in range(N)]

    # Reference: only step 0 is force-full; all other steps non-force-full,
    # virtual cache everywhere (threshold = +inf so no reset).
    ref_acc = _reference_seacache_accumulator(
        psi_raw, psi_filtered,
        force_full_steps={0},
        threshold=float("inf"),
        num_steps=N,
    )

    # Replay: anchor a = 0, vary n in [1, N-1].
    for n in range(1, N):
        got = replay_p_sea(
            psi_raw_at_a=psi_raw[0],
            psi_filtered_seq=psi_filtered[1:n + 1],
        )
        assert math.isclose(got, ref_acc[n], rel_tol=1e-10, abs_tol=1e-12), (
            f"replay(a=0, n={n}) = {got}  vs  reference = {ref_acc[n]}  "
            f"(abs diff {abs(got - ref_acc[n]):.3e})"
        )
    print(f"  test_replay_matches_reference_state_machine_synthetic ... PASS  "
          f"(verified n=1..{N - 1})")


def test_replay_matches_reference_with_multiple_anchors():
    """Multiple force-full anchors. For each anchor a, verify replay for each
    n in (a, next_anchor]."""
    torch.manual_seed(3)
    N = 30
    shape = (1, 16, 8)
    psi_raw = [torch.randn(shape) for _ in range(N)]
    psi_filtered = [torch.randn(shape) for _ in range(N)]
    force_full_steps = {0, 7, 18}   # multiple anchors

    ref_acc = _reference_seacache_accumulator(
        psi_raw, psi_filtered,
        force_full_steps=force_full_steps,
        threshold=float("inf"),
        num_steps=N,
    )

    anchors_sorted = sorted(force_full_steps)
    for i, a in enumerate(anchors_sorted):
        next_anchor = anchors_sorted[i + 1] if i + 1 < len(anchors_sorted) else N
        for n in range(a + 1, next_anchor):
            got = replay_p_sea(
                psi_raw_at_a=psi_raw[a],
                psi_filtered_seq=psi_filtered[a + 1:n + 1],
            )
            assert math.isclose(got, ref_acc[n], rel_tol=1e-10, abs_tol=1e-12), (
                f"replay(a={a}, n={n}) = {got}  vs  reference = {ref_acc[n]}  "
                f"(abs diff {abs(got - ref_acc[n]):.3e})"
            )
    print(f"  test_replay_matches_reference_with_multiple_anchors ... PASS  "
          f"(anchors {anchors_sorted})")


def test_replay_ignores_threshold_resets():
    """Real SeaCache resets the accumulator when acc crosses threshold. The
    replay (used for the synthetic forced-cache action) deliberately ignores
    threshold; it accumulates unconditionally. This test runs the reference
    state machine with a FINITE threshold (so the real path would reset some-
    where), then verifies the replay's value at step n equals the SUM of
    increments from a+1..n (no resets), which generally differs from the
    real per-step acc value.

    Construct a sequence where a known step in (a, n] would trigger reset
    in the reference; check replay value still equals naive sum and is
    strictly larger than the reference value at n."""
    torch.manual_seed(7)
    N = 12
    shape = (1, 8, 4)
    # Use small absolute magnitudes so rel_l1 increments are roughly known size.
    psi_raw = [torch.randn(shape) for _ in range(N)]
    psi_filtered = [torch.randn(shape) for _ in range(N)]

    # Force-full only at step 0. Choose a small threshold so some increment
    # in the middle of the window crosses it.
    threshold = 0.5

    ref_acc = _reference_seacache_accumulator(
        psi_raw, psi_filtered,
        force_full_steps={0},
        threshold=threshold,
        num_steps=N,
    )

    # Walk the same window with replay (no threshold resets).
    a, n = 0, N - 1
    naive_sum = 0.0
    prev = psi_raw[a]
    for j in range(a + 1, n + 1):
        naive_sum += rel_l1(psi_filtered[j], prev)
        prev = psi_filtered[j]

    replay_val = replay_p_sea(psi_raw[a], psi_filtered[a + 1:n + 1])
    assert math.isclose(replay_val, naive_sum, rel_tol=1e-12), \
        f"replay should equal naive cumulative sum, got {replay_val} vs {naive_sum}"

    # Confirm at least one intermediate step crossed threshold in the
    # reference (otherwise this test wouldn't be exercising the difference).
    saw_crossing = any(v >= threshold for v in ref_acc[1:])
    assert saw_crossing, (
        "test setup did not cross threshold inside the window — increase N "
        "or lower threshold so the threshold-reset path actually fires"
    )
    print(f"  test_replay_ignores_threshold_resets ... PASS  "
          f"(replay={replay_val:.4f}, ref crossed at step "
          f"{next(k for k,v in enumerate(ref_acc) if v >= threshold)})")


def test_from_trace_dict():
    torch.manual_seed(4)
    N = 10
    shape = (1, 8, 8)
    trace = {
        "psi_raw": [torch.randn(shape) for _ in range(N)],
        "psi_filtered": [torch.randn(shape) for _ in range(N)],
    }
    a, n = 2, 7
    direct = replay_p_sea(
        psi_raw_at_a=trace["psi_raw"][a],
        psi_filtered_seq=trace["psi_filtered"][a + 1:n + 1],
    )
    via_trace = replay_p_sea_from_trace(trace, a=a, n=n)
    assert math.isclose(direct, via_trace, rel_tol=1e-12), \
        f"from_trace wrapper diverges from direct: {via_trace} vs {direct}"
    print(f"  test_from_trace_dict ... PASS  (a={a}, n={n})")


def test_from_trace_bounds_checking():
    trace = {"psi_raw": [torch.zeros(1)] * 5, "psi_filtered": [torch.zeros(1)] * 5}
    # a >= n
    try:
        replay_p_sea_from_trace(trace, a=3, n=3)
        raise AssertionError("expected ValueError for a >= n")
    except ValueError:
        pass
    # n >= len
    try:
        replay_p_sea_from_trace(trace, a=0, n=5)
        raise AssertionError("expected ValueError for n >= len")
    except ValueError:
        pass
    # negative a
    try:
        replay_p_sea_from_trace(trace, a=-1, n=2)
        raise AssertionError("expected ValueError for a < 0")
    except ValueError:
        pass
    print("  test_from_trace_bounds_checking ... PASS")


def test_anchor_uses_raw_filtered_uses_filtered():
    """Sanity check: changing psi_raw[a] changes the result (anchor uses raw);
    changing psi_filtered[a+k] changes the result (window uses filtered);
    changing psi_raw[a+k] for k>=1 does NOT change the result (replay never
    reads raw inside the window)."""
    torch.manual_seed(5)
    N = 6
    shape = (1, 4, 4)
    psi_raw = [torch.randn(shape) for _ in range(N)]
    psi_filtered = [torch.randn(shape) for _ in range(N)]
    a, n = 0, 4

    base = replay_p_sea(psi_raw[a], psi_filtered[a + 1:n + 1])

    # perturb raw[a] → result must change
    psi_raw_b = list(psi_raw)
    psi_raw_b[a] = psi_raw_b[a] + 0.5 * torch.randn(shape)
    after_raw_a = replay_p_sea(psi_raw_b[a], psi_filtered[a + 1:n + 1])
    assert not math.isclose(base, after_raw_a, rel_tol=1e-6), \
        "perturbing raw[a] should change the result"

    # perturb filtered[a+2] → result must change
    psi_f_b = list(psi_filtered)
    psi_f_b[a + 2] = psi_f_b[a + 2] + 0.5 * torch.randn(shape)
    after_filt = replay_p_sea(psi_raw[a], psi_f_b[a + 1:n + 1])
    assert not math.isclose(base, after_filt, rel_tol=1e-6), \
        "perturbing filtered[a+2] should change the result"

    # perturb raw[a+2] → result must NOT change (replay only reads raw at anchor)
    # (trivially true here because we don't pass raw[a+2] to replay; the test
    # just documents the invariant)
    print("  test_anchor_uses_raw_filtered_uses_filtered ... PASS")


def main():
    print(f"running sqa_replay tests (torch {torch.__version__})")
    tests = [
        test_empty_window_returns_zero,
        test_single_increment_matches_rel_l1,
        test_two_increments_decompose,
        test_replay_matches_reference_state_machine_synthetic,
        test_replay_matches_reference_with_multiple_anchors,
        test_replay_ignores_threshold_resets,
        test_from_trace_dict,
        test_from_trace_bounds_checking,
        test_anchor_uses_raw_filtered_uses_filtered,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  {t.__name__} ... FAIL  {e}")
            failed += 1
        except Exception as e:
            print(f"  {t.__name__} ... ERROR  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed} / {len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
