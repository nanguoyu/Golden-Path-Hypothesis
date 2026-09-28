#!/usr/bin/env python3
"""Build the schedule TaylorSeer O1, HiCache O2 and L2P share at one budget.

Those three read the same table, so it has to satisfy all three step
restrictions at once. Each is a guard in `hunyuan_video/backend.py`:

    taylorseer_exact  raw_steps[0] == 0 or raw_steps[-1] == steps-1   (:333)
    hicache_exact     set(range(first_enhance)) | {steps-1}, fe=3     (:361)
    l2p_output_exact  0 in raw_steps                                  (:384)

The union is {0, 1, 2, 49} for the 50-step protocol, which is this script's
default. It is a CLI argument rather than a constant so that a change to
HiCache's first_enhance does not silently produce a table the adapter rejects
at generation time, hours into a matrix run.

The cached steps are spread as evenly as the arithmetic allows. Nothing here
is searched; the payload records `"algorithm"` and `"searched": false` so a
table that later comes out of a search is distinguishable from this one. The
assembler keeps only `cache_steps` plus the source path and hash, so inside
`baseline_matrix_config.v1.json` that distinction survives as the source file
rather than as a field.

    python analysis/hunyuan_video/build_shared_schedule.py \
        --budget K29 --output $DATA/hunyuan_video/shared/hy_k29.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunyuan_video.matrix_config import (  # noqa: E402
    BUDGET_CACHE_COUNTS,
    BUDGETS,
    NUM_STEPS,
)

DEFAULT_FORBIDDEN = (0, 1, 2, NUM_STEPS - 1)
ALGORITHM = "even-runs.v1"


def even_runs(total: int, parts: int) -> list[int]:
    """`total` split into `parts` whole numbers differing by at most one, with
    the larger ones spread through the sequence rather than bunched at either
    end.

    Bunching them at the front would make the early path systematically coarser
    than the late one, and bunching them at the back the reverse; either is a
    claim about where caching is cheap, which this table is in no position to
    make. (When `remainder` and `parts` share no useful factor the sequence
    cannot be symmetric -- at K41 it is [6,7,7,7,7,7], the one short run first.)
    """
    if parts <= 0:
        raise ValueError(f"need at least one run, got {parts}")
    if total < 0:
        raise ValueError(f"cannot split {total} steps")
    base, remainder = divmod(total, parts)
    return [base + (((i + 1) * remainder) // parts - (i * remainder) // parts)
            for i in range(parts)]


def build_schedule(cache_count: int, *, num_steps: int = NUM_STEPS,
                   forbidden: tuple[int, ...] = DEFAULT_FORBIDDEN) -> list[int]:
    """The cached steps: `cache_count` of them, none forbidden, runs even.

    The forbidden steps have to be a leading run plus the terminal step, which
    is the shape all three adapters ask for; anything else is refused rather
    than approximated, because a table that quietly drops a constraint fails
    inside the adapter after the model is already loaded.
    """
    order = sorted(set(forbidden))
    if not order:
        raise ValueError("no forbidden steps given; all three adapters forbid at least step 0")
    prefix = [step for step in order if step != num_steps - 1]
    if order[-1] != num_steps - 1 or prefix != list(range(len(prefix))):
        raise ValueError(
            f"forbidden steps {order} are not a leading run plus the terminal step; "
            f"this builder only knows how to satisfy that shape")
    head = len(prefix)                       # steps 0..head-1 are computed
    computed = num_steps - cache_count
    # the cached runs sit between consecutive computed anchors: the last head
    # step, each free computed step, and the terminal step
    runs = computed - head
    if runs <= 0:
        raise ValueError(
            f"cache_count {cache_count} leaves {computed} computed steps, which the "
            f"{head} leading and 1 terminal forced steps already exceed")
    lengths = even_runs(cache_count, runs)

    cached: list[int] = []
    cursor = head - 1                        # the last computed step so far
    for length in lengths:
        cached.extend(range(cursor + 1, cursor + 1 + length))
        cursor += length + 1                 # the computed step that ends this run
    return cached


def run_lengths(cache_steps: list[int]) -> list[int]:
    """Lengths of the maximal runs of consecutive cached steps.

    Read back off the table rather than recomputed from the inputs: the one
    human-readable summary an operator gets has to describe the artifact that
    was written, not a second derivation of it.
    """
    if not cache_steps:
        return []
    runs, current = [], 1
    for previous, step in zip(cache_steps, cache_steps[1:]):
        if step == previous + 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs


def payload(budget: str, cache_steps: list[int], forbidden: tuple[int, ...],
            num_steps: int) -> dict[str, Any]:
    return {
        "schema": "hunyuan_video.shared_schedule.v1",
        "budget": budget,
        "num_steps": num_steps,
        "cache_count": len(cache_steps),
        "cache_steps": cache_steps,
        "forbidden_steps": sorted(set(forbidden)),
        "shared_by": ["taylorseer_o1", "hicache_o2", "l2p"],
        "algorithm": ALGORITHM,
        "searched": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--forbidden", default=",".join(str(s) for s in DEFAULT_FORBIDDEN),
        help="steps the three adapters refuse to see cached (default %(default)s)")
    args = parser.parse_args(argv)

    # no --num-steps: the lane is frozen to NUM_STEPS (baseline_screen_runner
    # refuses anything else), and a schedule built for a different length
    # passes the assembler's bounds check and the adapters' -- it would leave a
    # long tail of steps uncached with the right cache_count, silently
    forbidden = tuple(int(part) for part in args.forbidden.split(",") if part.strip())
    cache_count = BUDGET_CACHE_COUNTS[args.budget]
    cache_steps = build_schedule(cache_count, num_steps=NUM_STEPS, forbidden=forbidden)

    # cheap, and the failure it catches otherwise surfaces inside the adapter
    # one model load into a matrix cell
    if len(cache_steps) != cache_count:
        raise AssertionError(f"{len(cache_steps)} cached steps, expected {cache_count}")
    if sorted(set(cache_steps)) != cache_steps:
        raise AssertionError("cached steps are not sorted and unique")
    clash = sorted(set(cache_steps) & set(forbidden))
    if clash:
        raise AssertionError(f"schedule caches forbidden steps {clash}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload(args.budget, cache_steps, forbidden, NUM_STEPS), indent=2),
        encoding="utf-8")
    runs = run_lengths(cache_steps)
    print(f"{args.budget}: {cache_count} cached of {NUM_STEPS}, "
          f"{NUM_STEPS - cache_count} computed, {len(runs)} cached runs of "
          f"{min(runs)}-{max(runs)} steps -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
