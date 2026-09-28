#!/usr/bin/env python3
"""Stage a Wan2.1 baseline-matrix cell's videos into the layout VBench reads.

The official VBench evaluator finds a video by its file name: `<prompt>-<k>.mp4`,
where `k` is that prompt's repeat index. `wan21/baseline_screen_runner.py` writes
a flat directory keyed by prompt index -- `video_{idx:05d}.mp4`
(`wan21/_helpers.py:21-22`) -- so the two have to be bridged, and the bridge is
symlinks plus a manifest of what points where (plan sections 2.3 and 4.3).

The HunyuanVideo twin (`analysis/hunyuan_video/baseline_vbench_staging.py`) is
the same bridge for that lane; this one differs only in whose cells it accepts:
the identity record it verifies is `wan21.baseline_screen_cell.v1`, so a HYV
directory cannot be staged through here and scored as a Wan row.

    python analysis/wan21/baseline_vbench_staging.py \\
        --cell $DATA/wan21/matrix/seacache_vbench944_K29_s42 --seed-index 0 \\
        --cell $DATA/wan21/matrix/seacache_vbench944_K29_s43 --seed-index 1 \\
        --cell $DATA/wan21/matrix/seacache_vbench944_K29_s44 --seed-index 2 \\
        --prompt-manifest resources/hunyuan_video/evaluation/vbench944.json \\
        --staging-dir $DATA/wan21/vbench/seacache_K29_staging \\
        --out $DATA/wan21/vbench/seacache_K29_staging.json

One `--cell` per seed, each with its own `--seed-index`, because the repeat index
is what distinguishes the three videos VBench averages over. VBench is scored on
VBench944 only; a Penguin599 manifest is refused rather than staged (plan
section 4.3: the two datasets' scores are never put side by side).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Unconditional, and before any wan21/hunyuan_video import: `analysis/wan21/` is
# itself importable as a `wan21` namespace package, so a guarded insert that
# loses to a PYTHONPATH entry makes `import wan21.matrix_config` resolve to this
# directory instead of the repository package (precedent: commit b35a0d3).
sys.path.insert(0, str(ROOT))

from hunyuan_video.config import hash_json  # noqa: E402
from hunyuan_video.records import load_self_hashed_json, sha256_file  # noqa: E402
from wan21.baseline_screen_runner import CELL_SCHEMA  # noqa: E402
from wan21.matrix_config import EVALUATION_MANIFEST_SCHEMA  # noqa: E402

SCHEMA = "wan21.baseline_vbench_staging.v1"
#: VBench scores one dataset on this lane (plan section 4.3).
VBENCH_DATASET = "vbench944"


def safe_name(prompt: str, seed_index: int) -> str:
    """VBench's own naming, refusing anything a file system would mangle.

    A prompt is arbitrary user text; it reaches a file name here, so the checks
    are not decoration. `os.fsencode` is what bounds the 255-byte limit -- a
    prompt well under 255 characters can exceed it in UTF-8.
    """
    name = f"{prompt}-{int(seed_index)}.mp4"
    if Path(name).name != name or "\n" in name or "\x00" in name:
        raise ValueError(f"unsafe VBench filename: {name!r}")
    if len(os.fsencode(name)) > 255:
        raise ValueError(f"overlong VBench filename ({len(os.fsencode(name))} bytes): "
                         f"{name[:60]!r}...")
    return name


def cell_identity(cell: Path) -> dict[str, Any]:
    """The cell's own identity record, which the runner writes beside the videos."""
    path = cell / "cell_identity.json"
    if not path.is_file():
        raise SystemExit(f"{cell} has no cell_identity.json; it was not written by "
                         f"wan21/baseline_screen_runner.py")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("schema") != CELL_SCHEMA:
        raise SystemExit(f"{path} is a {identity.get('schema')!r} cell, not {CELL_SCHEMA}; "
                         f"another backbone's videos would be scored as a Wan2.1 row")
    return identity


def stage(cells: list[tuple[Path, int]], *, prompts: list[str], staging_dir: Path,
          dataset: str) -> dict[str, Any]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    names: set[str] = set()
    identities = []
    for cell, seed_index in cells:
        identity = cell_identity(cell)
        if identity.get("dataset") != dataset:
            raise SystemExit(f"{cell} is a {identity.get('dataset')!r} cell, but the "
                             f"prompt manifest is {dataset!r}")
        identities.append({"cell": str(cell), "seed_index": seed_index, **identity})
        for index, prompt in enumerate(prompts):
            source = (cell / f"video_{index:05d}.mp4").resolve()
            if not source.is_file():
                raise SystemExit(f"missing video for prompt {index} of {cell}: {source}")
            name = safe_name(prompt, seed_index)
            if name in names:
                # two artifacts whose prompts are equal after nothing at all --
                # VBench would average them as one, silently halving the count
                raise SystemExit(
                    f"two videos claim the file name {name!r}; VBench keys on the "
                    f"prompt text, so the two artifacts are indistinguishable to it")
            names.add(name)
            destination = staging_dir / name
            if destination.is_symlink() or destination.exists():
                if not destination.is_symlink() or destination.resolve() != source:
                    raise SystemExit(f"VBench staging collision: {destination}")
            else:
                destination.symlink_to(source)
            items.append({
                "name": name,
                "prompt_idx": index,
                "seed_index": seed_index,
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            })

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "staging_dir": str(staging_dir.resolve()),
        "item_count": len(items),
        "prompt_count": len(prompts),
        "cells": identities,
        "items": items,
    }
    payload["manifest_sha256"] = hash_json(payload)
    return payload


def remove_staging(staging_manifest_path: Path) -> int:
    """Undo one staging, checking every link before unlinking it.

    A VBench score for a cell is a directory of 2,832 symlinks; leaving them
    behind means the next cell either collides or, worse, scores a mixture. This
    removes links this manifest created and pointing where it recorded, and
    refuses anything else -- a name that is not a bare filename, a real file
    where a link should be, a link that now points somewhere else. Deleting
    under a wrong assumption here would delete generations.
    """
    payload = load_self_hashed_json(staging_manifest_path, "manifest_sha256")
    if payload.get("schema") != SCHEMA:
        raise SystemExit(f"{staging_manifest_path} is not a Wan2.1 baseline VBench "
                         f"staging manifest")
    staging_dir = Path(payload["staging_dir"])
    removed = 0
    for item in payload["items"]:
        name = item["name"]
        if Path(name).name != name or Path(name).is_absolute():
            raise SystemExit(f"unsafe VBench staging removal name: {name!r}")
        # the name check above is what keeps this inside the directory: joining a
        # bare filename onto staging_dir gives a path whose parent IS staging_dir,
        # so a second containment test could never fire
        path = staging_dir / name
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_symlink():
            raise SystemExit(f"{path} is a real file, not the symlink this manifest "
                             f"created; refusing to delete it")
        if str(path.resolve()) != item["source_path"]:
            raise SystemExit(f"{path} now points at {path.resolve()}, not the "
                             f"{item['source_path']} this manifest recorded")
        path.unlink()
        removed += 1
    if staging_dir.exists() and not any(staging_dir.iterdir()):
        staging_dir.rmdir()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=Path, action="append", default=[],
                        help="one generation directory; repeat once per seed")
    parser.add_argument("--seed-index", type=int, action="append", default=[],
                        help="that cell's VBench repeat index, in the same order")
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--remove", type=Path,
                        help="undo the staging this manifest describes, then stop")
    args = parser.parse_args(argv)

    if args.remove is not None:
        if args.cell or args.seed_index or args.prompt_manifest or args.staging_dir:
            raise SystemExit("--remove undoes a staging and takes no build arguments")
        removed = remove_staging(args.remove)
        print(f"{args.remove}: removed {removed} symlink(s)")
        return 0

    missing = [name for name, value in (("--cell", args.cell),
                                        ("--seed-index", args.seed_index),
                                        ("--prompt-manifest", args.prompt_manifest),
                                        ("--staging-dir", args.staging_dir),
                                        ("--out", args.out)) if not value]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")
    if len(args.cell) != len(args.seed_index):
        raise SystemExit(f"{len(args.cell)} --cell but {len(args.seed_index)} "
                         f"--seed-index; they pair positionally")
    if len(set(args.seed_index)) != len(args.seed_index):
        raise SystemExit("two cells share a --seed-index; VBench would see one of "
                         "them overwrite the other")

    manifest = load_self_hashed_json(args.prompt_manifest, "manifest_sha256")
    if manifest.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise SystemExit(f"{args.prompt_manifest} is not an evaluation prompt manifest")
    dataset = str(manifest["dataset"])
    if dataset != VBENCH_DATASET:
        raise SystemExit(f"VBench is scored on {VBENCH_DATASET} only, not {dataset!r}")
    prompts = [str(item["prompt"]) for item in manifest["items"]]

    payload = stage(list(zip(args.cell, args.seed_index)), prompts=prompts,
                    staging_dir=args.staging_dir, dataset=dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"{args.staging_dir}: {payload['item_count']} videos "
          f"({payload['prompt_count']} prompts x {len(args.cell)} seeds)")
    print(f"{args.out}: staging manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
