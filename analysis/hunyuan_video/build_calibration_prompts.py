#!/usr/bin/env python3
"""Export the two calibration prompt files the threshold sweeps read.

Plan section 3.1 freezes one threshold per (method, dataset, budget), so each
evaluation set calibrates its own: a dynamic gate's threshold decides how many
steps it caches, and that count moves with the prompt distribution. The image
side hit exactly this and re-calibrated for its new distribution rather than
carrying the old thresholds over.

    Penguin599 -> B-CAL48, already frozen in `dataset_splits.v2.json`
    VBench944  -> 48 artifacts, 3 round-robin passes over the 16 dimensions

Both are 48 prompts, so neither dataset's thresholds rest on more evidence than
the other's. Neither is disjoint from the set it calibrates -- this backbone has
no spare pool, every Penguin and VBench prompt is evaluated -- which is
acceptable only because the selection criterion is realized cache count and
nothing else. A quantity-only criterion cannot overfit quality, and the quantity
it does fit is the budget the calibration exists to hold.

The runner reads prompts as lines (`lib/io_utils.read_prompts`), which drops
blanks and anything starting with `#` and cannot represent an embedded newline.
No prompt in either source has any of those properties today, and this script
refuses rather than writes if that ever changes -- a silently dropped line is a
calibration set of the wrong size, and the sweep reports a mean over that set.

    python analysis/hunyuan_video/build_calibration_prompts.py \\
        --out-dir resources/hunyuan_video/calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunyuan_video.config import hash_json  # noqa: E402
from hunyuan_video.records import load_self_hashed_json  # noqa: E402

SPLITS = ROOT / "resources/hunyuan_video/dataset_splits.v2.json"
VBENCH = ROOT / "resources/hunyuan_video/vbench_metadata.v1.json"
PENGUIN_SPLIT = "B-CAL48"
DEFAULT_VBENCH_COUNT = 48
ALGORITHM = "dimension-lexicographic-unique-roundrobin.v1"


def check_line_safe(prompts: list[str], where: str) -> None:
    """Refuse anything the line reader would drop, alter or merge.

    Each of these silently changes the size of the calibration set, and the
    sweep's whole output is a mean over that set.
    """
    problems = []
    for index, prompt in enumerate(prompts):
        # blank first: "".splitlines() is [] rather than [""], so the line test
        # below would report an empty prompt as spanning several lines
        if not prompt.strip():
            problems.append(f"{where}[{index}] is blank")
        # `str.splitlines` is what the reader uses, and it breaks on eight
        # separators beyond \n and \r -- U+000B, U+000C, U+001C-U+001E, U+0085,
        # U+2028, U+2029. Asking it directly cannot fall behind that list.
        elif len(prompt.splitlines()) != 1:
            problems.append(f"{where}[{index}] spans more than one line as the reader "
                            f"splits them")
        elif "﻿" in prompt:
            problems.append(f"{where}[{index}] contains a byte-order mark, which is not "
                            f"whitespace and would reach the text encoder")
        elif prompt.lstrip().startswith("#"):
            problems.append(f"{where}[{index}] starts with '#', which the reader treats "
                            f"as a comment")
        elif prompt != prompt.strip():
            problems.append(f"{where}[{index}] has leading or trailing whitespace")
    duplicates = {p for p in prompts if prompts.count(p) > 1}
    if duplicates:
        problems.append(f"{where} repeats {len(duplicates)} prompt(s); the reader would "
                        f"generate them twice and the mean would be weighted")
    if problems:
        raise SystemExit("refusing to write a prompt file:\n  " + "\n  ".join(problems[:10]))


def penguin_calibration(split_name: str = PENGUIN_SPLIT) -> tuple[list[str], list[str]]:
    """`(prompt_ids, prompts)` of one frozen Penguin split, in the split's order."""
    payload = load_self_hashed_json(SPLITS, "manifest_sha256")
    splits = payload["splits"]
    if split_name not in splits:
        raise SystemExit(f"{split_name} is not a frozen split; {SPLITS} holds "
                         f"{sorted(splits)}")
    by_id = {row["canonical_sha256"]: row for row in payload["prompts"]}
    ids = list(splits[split_name]["prompt_ids"])
    missing = [pid for pid in ids if pid not in by_id]
    if missing:
        raise SystemExit(f"{split_name} names {len(missing)} prompt id(s) the registry "
                         f"does not hold, first {missing[0]}")
    # the split records the digest of its own id list; recomputing it and not
    # comparing would be a coincidence rather than a check
    recorded = splits[split_name].get("prompt_ids_sha256")
    if recorded is not None and hash_json(ids) != recorded:
        raise SystemExit(f"{split_name} prompt_ids do not match their recorded digest "
                         f"{recorded}; the split file has been edited")
    # the exact prompt, not the canonical one: the canonical form is lowercased
    # and NFKC-folded, it differs from the exact text for all 599 Penguin and 426
    # VBench prompts, and every other HunyuanVideo builder ships `prompt`. The
    # matrix generates from the exact text, so calibrating on anything else tunes
    # a threshold against conditioning the matrix never sees.
    return ids, [by_id[pid]["prompt"] for pid in ids]


def vbench_calibration(count: int = DEFAULT_VBENCH_COUNT
                       ) -> tuple[list[str], list[str], dict[str, list[str]],
                                  dict[str, int]]:
    """`(artifact_ids, prompts, dimension -> ids, dimension -> picks)`.

    Round-robin over the 16 dimensions rather than "first `count` of the whole
    set". A prefix is not degenerate -- the artifacts are ordered by artifact_id,
    which is a sha256, so the first 48 already touch all 16 dimensions -- but it
    is badly skewed: imaging_quality, aesthetic_quality and overall_consistency
    get 11 each while object_class and temporal_flickering get 1. Round-robin
    picks each dimension exactly `count / 16` times. VBench944 is the dataset
    scored per dimension, so a subset skewed that way would hold the budget on
    prompts unlike most of what the matrix then generates. Nothing depends on a
    seed.

    `count` need not divide 16; the loop just stops mid-round, and the manifest
    records the realized picks per dimension.
    """
    payload = load_self_hashed_json(VBENCH, "manifest_sha256")
    dimensions = list(payload["dimensions"])
    artifacts = {row["artifact_id"]: row for row in payload["artifacts"]}
    # The frozen metadata carries each row's dimension next to its artifact id,
    # so the join stays inside the hashed file. Reading it out of the upstream
    # submodule instead would make the answer depend on a checkout nothing here
    # verifies -- a VBench release with the same row count and a different order
    # silently relabels dimensions.
    artifact_dimensions: dict[str, set[str]] = defaultdict(set)
    for row in payload["metadata_rows"]:
        artifact_dimensions[row["artifact_id"]].update(row["metadata"]["dimension"])
    by_dimension: dict[str, list[str]] = defaultdict(list)
    for artifact_id, observed in artifact_dimensions.items():
        for dimension in observed:
            by_dimension[dimension].append(artifact_id)
    for dimension in dimensions:
        by_dimension[dimension].sort()
    missing = [d for d in dimensions if not by_dimension[d]]
    if missing:
        raise SystemExit(f"no artifact carries dimension(s) {missing}")

    selected: list[str] = []
    primary_picks: dict[str, int] = {}
    taken: set[str] = set()
    cursor = {dimension: 0 for dimension in dimensions}
    while len(selected) < count:
        progressed = False
        for dimension in dimensions:
            if len(selected) >= count:
                break
            candidates = by_dimension[dimension]
            index = cursor[dimension]
            while index < len(candidates) and candidates[index] in taken:
                index += 1
            cursor[dimension] = index
            if index >= len(candidates):
                continue
            artifact_id = candidates[index]
            selected.append(artifact_id)
            primary_picks[dimension] = primary_picks.get(dimension, 0) + 1
            taken.add(artifact_id)
            cursor[dimension] = index + 1
            progressed = True
        if not progressed:
            raise SystemExit(f"only {len(selected)} distinct artifacts available, "
                             f"asked for {count}")

    chosen: dict[str, list[str]] = defaultdict(list)
    for artifact_id in selected:
        for dimension in sorted(artifact_dimensions[artifact_id]):
            chosen[dimension].append(artifact_id)
    return (selected, [artifacts[a]["prompt"] for a in selected], dict(chosen),
            {d: primary_picks.get(d, 0) for d in dimensions})


def show(path: Path) -> str:
    """Repo-relative where that is meaningful, absolute otherwise: --out-dir is
    free to point outside the tree and a manifest must not die on that."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_prompt_file(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def digests(prompts: list[str]) -> list[str]:
    return [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "resources/hunyuan_video/calibration")
    parser.add_argument("--penguin-split", default=PENGUIN_SPLIT)
    parser.add_argument("--vbench-count", type=int, default=DEFAULT_VBENCH_COUNT)
    args = parser.parse_args(argv)

    if args.vbench_count <= 0:
        raise SystemExit(f"--vbench-count must be positive, got {args.vbench_count}")

    penguin_ids, penguin_prompts = penguin_calibration(args.penguin_split)
    check_line_safe(penguin_prompts, args.penguin_split)
    penguin_path = args.out_dir / f"penguin_{args.penguin_split.lower()}.txt"
    write_prompt_file(penguin_path, penguin_prompts)

    vbench_ids, vbench_prompts, per_dimension, primary_picks = vbench_calibration(
        args.vbench_count)
    check_line_safe(vbench_prompts, f"vbench{args.vbench_count}")
    vbench_path = args.out_dir / f"vbench_cal{args.vbench_count}.txt"
    write_prompt_file(vbench_path, vbench_prompts)

    manifest: dict[str, Any] = {
        "schema": "hunyuan_video.calibration_prompts.v1",
        "used_by": "threshold sweep and offline schedule search (plan section 3.1)",
        "scope": "one calibration set per evaluation dataset; each dataset's "
                 "thresholds are frozen on its own set",
        "line_text": "exact prompt",
        "sets": {
            "penguin599": {
                "calibrates": "penguin599",
                "split": args.penguin_split,
                "source": show(SPLITS),
                "file": show(penguin_path),
                "count": len(penguin_ids),
                "selection": f"frozen split {args.penguin_split}",
                "prompt_ids": penguin_ids,
                "prompt_ids_sha256": hash_json(penguin_ids),
                # the split's ids are digests of the CANONICAL text while the
                # lines are the exact text, so a consumer joining by hashing a
                # line needs these
                "prompt_sha256_of_lines": digests(penguin_prompts),
            },
            "vbench944": {
                "calibrates": "vbench944",
                "source": show(VBENCH),
                "file": show(vbench_path),
                "count": len(vbench_ids),
                "selection": ALGORITHM,
                "artifact_ids": vbench_ids,
                "artifact_ids_sha256": hash_json(vbench_ids),
                "prompt_sha256_of_lines": digests(vbench_prompts),
                "picks_per_dimension": primary_picks,
                "artifacts_per_dimension": {dimension: sorted(ids) for dimension, ids
                                            in sorted(per_dimension.items())},
            },
        },
    }
    manifest_path = args.out_dir / "calibration_prompts.v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"{show(penguin_path)}: {len(penguin_prompts)} prompts "
          f"({args.penguin_split}) -> calibrates penguin599")
    print(f"{show(vbench_path)}: {len(vbench_prompts)} prompts "
          f"({ALGORITHM}) -> calibrates vbench944")
    print(f"{show(manifest_path)}: ids and hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
