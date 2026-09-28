#!/usr/bin/env python3
"""Three things the P4 scripts (`step_profiles.py`, `shape_scale.py`,
`latent_paths.py`) all need, kept in one place so they cannot drift apart.

1. **Atomic output writing.** Every table, JSON, markdown page and figure is
   written to `<path>.tmp` and moved into place with `os.replace`, so a killed
   sbatch job leaves either the previous file or the new one — never a
   truncated file that the next run's existence test would accept as done.

2. **A reuse decision that looks at the run parameters, not just the file
   name.** The prescribed workflow is "run cheap with `--limit` / `--sample` /
   `--frames_per_stream`, then run for real"; a plain `is_file()` test lets the
   smoke run's numbers survive into the production outputs. Each script records
   the parameters that produced its JSON and `reuse_decision` compares them
   with the ones asked for now: identical -> skip, the stored run strictly
   smaller (a smoke run) -> recompute, anything else (the stored run is the
   larger one, or the parameters are not comparable) -> refuse and say so.

3. **One index convention for the single-step turn profile.** Plan section 2.4
   indexes `turn_angle_deg` by interior junction n = 0..48, while
   `trajectory_math.window_centers(51, 1)` returns the state centres 1..49 of
   the same 49 numbers. Both P4 scripts store this array and P9 reads them
   together, so the label comes from here.

Nothing in this module reads or writes experiment data; it is plain plumbing.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import window_centers  # noqa: E402

# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` through a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, payload: Any, *, indent: int = 1) -> Path:
    return atomic_write_text(Path(path), json.dumps(payload, indent=indent))


def atomic_savefig(fig, path: Path, **kwargs: Any) -> Path:
    """`fig.savefig` through a temporary file (same suffix, so the writer still
    picks the format from the extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp" + path.suffix)
    fig.savefig(tmp, **kwargs)
    os.replace(tmp, path)
    return path


_PNG_END = b"IEND\xaeB`\x82"


def output_complete(path: Path) -> bool:
    """Is this output file whole?

    Existence is not enough: a job killed mid-write used to leave a truncated
    file that the next run accepted as done. Writes here go through
    `os.replace` so they cannot produce one any more, but the check stays
    cheap and honest — non-empty, and for a PNG, actually ending in its IEND
    chunk.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() == ".png":
        with open(path, "rb") as fh:
            fh.seek(-len(_PNG_END), os.SEEK_END)
            return fh.read() == _PNG_END
    return True


def read_json_if_readable(path: Path) -> dict[str, Any] | None:
    """The JSON at `path`, or None when it is absent, empty or truncated.

    A corrupt file must fall through to a recompute rather than abort the run:
    the way it gets corrupted is a killed job, and the fix is to run again.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# reuse decision
# ---------------------------------------------------------------------------

_ABSENT = "<absent>"


def _cap(value: Any) -> float:
    """A 'keep at most N' knob as a comparable number.

    `None`, `0` and `-1` all mean "no cap" in the three CLIs (`--limit` unset,
    `--frames_per_stream 0` = every frame, `--reference_planes -1` = as many as
    the data), so they map to infinity; everything else is its own value.
    """
    if value is None:
        return math.inf
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return math.inf if value in (0, -1) else float(value)
    return math.nan


def reuse_decision(recorded: dict[str, Any] | None, current: dict[str, Any],
                   *, caps: Iterable[str] = ()) -> tuple[str, dict[str, tuple[Any, Any]]]:
    """`("skip" | "recompute" | "refuse", differences)`.

    `caps` names the parameters that mean "keep at most N" and are therefore
    ordered; every other parameter is compared for equality only. "recompute"
    is returned when every difference makes the STORED run the smaller one — a
    smoke run being replaced by the real one, which is the workflow. When the
    stored run is the larger one, or the parameters cannot be ordered
    (different streams, a different floor route), the caller is told to decide
    with `--force` instead of silently overwriting a bigger run's numbers.
    """
    if recorded is None:
        return "recompute", {}
    caps = set(caps)
    diffs: dict[str, tuple[Any, Any]] = {}
    for key, cur in current.items():
        rec = recorded.get(key, _ABSENT)
        if key in caps:
            same = rec is not _ABSENT and _cap(rec) == _cap(cur)
        else:
            same = rec == cur
        if not same:
            diffs[key] = (rec, cur)
    if not diffs:
        return "skip", diffs

    smaller = True
    for key, (rec, cur) in diffs.items():
        if rec is _ABSENT:
            smaller = False
        elif key in caps:
            smaller = smaller and _cap(rec) < _cap(cur)
        elif isinstance(rec, bool) and isinstance(cur, bool):
            # a boolean that skips work: True in the store, False now, means
            # the stored run did less
            smaller = smaller and (rec and not cur)
        else:
            smaller = False
    return ("recompute" if smaller else "refuse"), diffs


def describe_diffs(diffs: dict[str, tuple[Any, Any]]) -> str:
    return "; ".join(f"{k}: stored {r!r} vs requested {c!r}" for k, (r, c) in sorted(diffs.items()))


def resolve_reuse(json_path: Path, current: dict[str, Any], *,
                  caps: Iterable[str] = (), force: bool = False,
                  recorded_key: str = "run_params",
                  extra_outputs: Iterable[Path] = ()) -> dict[str, Any] | None:
    """Decide whether the outputs at `json_path` can stand for this run.

    Returns the stored report when it can be reused (the caller then only has
    to check whatever else it expects on disk), and None when the caller has to
    recompute. Refusal is a hard stop with the differing parameters printed —
    silently overwriting a larger run, or silently keeping a smaller one, are
    both worse than making the operator type `--force`.
    """
    if force:
        return None
    stored = read_json_if_readable(json_path)
    if stored is None:
        return None
    verdict, diffs = reuse_decision(stored.get(recorded_key), current, caps=caps)
    if verdict == "recompute":
        print(f"[recompute] {json_path.name} was produced by a smaller run "
              f"({describe_diffs(diffs)})")
        return None
    if verdict == "refuse":
        raise SystemExit(
            f"{json_path} was produced with different run parameters and this run is "
            f"not simply larger:\n  {describe_diffs(diffs)}\n"
            f"Pass --force to overwrite it, or point --out_tables / --out_figs "
            f"somewhere else.")
    for extra in extra_outputs:
        if not output_complete(extra):
            print(f"[recompute] {Path(extra).name} is missing, empty or truncated")
            return None
    return stored


# ---------------------------------------------------------------------------
# turn-profile index convention (plan section 2.4)
# ---------------------------------------------------------------------------

TURN_W1_INDEX_NOTE = (
    "interior junction n = 0..48 (between solver step n and solver step n+1). "
    "This is the plan section 2.4 convention for `turn_angle_deg`; "
    "`trajectory_math.window_centers(51, 1)` labels the same 49 numbers by "
    "their state centre n+1, which is one higher.")


def turn_index(n_states: int, window: int) -> tuple[list[int], str]:
    """`(labels, what the labels are)` for `turn_angles_window_deg(Z, window)`.

    w >= 2 keeps `window_centers` (state indices w .. n_states-1-w). w = 1 is
    `turn_angle_deg`, relabelled to the plan's junction index so the two P4
    scripts that store it agree; the state centre is the junction + 1.
    """
    centers = [int(c) for c in window_centers(n_states, window)]
    if int(window) == 1:
        return [c - 1 for c in centers], TURN_W1_INDEX_NOTE
    return centers, (f"window centre c = {window}..{n_states - 1 - window} "
                     f"(state index, not array index)")
