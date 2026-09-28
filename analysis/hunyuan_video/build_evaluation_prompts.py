#!/usr/bin/env python3
"""Export the two evaluation sets as prompt manifests the matrix generates from.

The matrix generates 599 Penguin prompts and 944 VBench artifacts per (method,
seed, budget) cell. Neither can be transported as a line-per-prompt text file:

  - **Two of the 599 Penguin prompts contain a literal newline** (indices 289
    and 306 of the frozen order). Written one per line and read back with
    `lib/io_utils.read_prompts`, which splits on `str.splitlines`, the file
    yields 601 prompts, prompt 289 is truncated and every index from there on
    shifts by one. Since the per-prompt seed is `base + prompt_idx` and the
    evaluators pair `video_{idx:05d}.mp4` positionally, that misaligns prompt,
    seed and reference for 310 of 599 prompts with nothing raising. The frozen
    registry says so itself: `prompt_transport: "structured-json-only;
    embedded-newlines-preserved"`.
  - The plan's unit for VBench is the `artifact_id`, and a line index cannot
    carry one.

So the transport is JSON: an ordered list of `{prompt_id, prompt}` with the
exact text, self-hashed, read by `baseline_screen_runner.py --prompt_manifest`.
The order is the frozen registry's order and is what fixes every prompt's seed
and file name, so it must never be re-sorted.

    python analysis/hunyuan_video/build_evaluation_prompts.py \\
        --out-dir resources/hunyuan_video/evaluation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunyuan_video.config import hash_json  # noqa: E402
from hunyuan_video.matrix_config import DATASETS  # noqa: E402
from hunyuan_video.records import load_self_hashed_json  # noqa: E402

SPLITS = ROOT / "resources/hunyuan_video/dataset_splits.v2.json"
VBENCH = ROOT / "resources/hunyuan_video/vbench_metadata.v1.json"
SCHEMA = "hunyuan_video.evaluation_prompts.v1"
EXPECTED = {"penguin599": 599, "vbench944": 944}


def penguin_items() -> list[dict[str, str]]:
    """All 599 Penguin prompts, in the frozen registry's order.

    `prompt` is the exact text and `prompt_id` the registry's own
    `canonical_sha256`, which is the digest of the CANONICAL form -- lowercased
    and NFKC-folded -- so it is an identifier, not a digest of what is written
    here. The matrix conditions on the exact text.
    """
    payload = load_self_hashed_json(SPLITS, "manifest_sha256")
    return [{"prompt_id": row["canonical_sha256"], "prompt": row["prompt"]}
            for row in payload["prompts"]]


def vbench_items() -> list[dict[str, str]]:
    """All 944 VBench artifacts, in the frozen metadata's order (by artifact_id).

    The unit is the artifact, not the prompt: 946 upstream rows collapse to 944
    distinct texts, and one further pair collapses only under canonicalization,
    so two artifacts here carry near-identical prompts and are generated twice
    on purpose (plan section 1.2).
    """
    payload = load_self_hashed_json(VBENCH, "manifest_sha256")
    return [{"prompt_id": row["artifact_id"], "prompt": row["prompt"]}
            for row in payload["artifacts"]]


def check_items(items: list[dict[str, str]], dataset: str) -> None:
    """Refuse anything that would make the manifest a different set than it says.

    Line-safety is deliberately NOT checked: an embedded newline is exactly what
    this transport exists to carry. What is checked is what would still break --
    an empty prompt, a repeated id, a count that does not match the registry.
    """
    if len(items) != EXPECTED[dataset]:
        raise SystemExit(f"{dataset} has {len(items)} items, the frozen registry "
                         f"defines {EXPECTED[dataset]}")
    problems = []
    for index, item in enumerate(items):
        if not item["prompt"].strip():
            problems.append(f"{dataset}[{index}] is blank")
        if not item["prompt_id"]:
            problems.append(f"{dataset}[{index}] has no prompt_id")
    ids = [item["prompt_id"] for item in items]
    if len(set(ids)) != len(ids):
        problems.append(f"{dataset} repeats a prompt_id; the id is what joins a "
                        f"generation back to its registry row")
    if problems:
        raise SystemExit("refusing to write a prompt manifest:\n  "
                         + "\n  ".join(problems[:10]))


def payload_for(dataset: str, items: list[dict[str, str]], source: Path) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "source": str(source.relative_to(ROOT)),
        "count": len(items),
        "unit": "prompt" if dataset == "penguin599" else "artifact",
        "order": "the frozen registry's order; it fixes every prompt's seed "
                 "(base + prompt_idx) and file name, so it must not be re-sorted",
        "text": "exact",
        "items": items,
        # a consumer that wants to check the text it generated from, without
        # canonicalizing anything
        "prompt_sha256_of_items": [
            hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest() for item in items],
    }
    body["manifest_sha256"] = hash_json(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "resources/hunyuan_video/evaluation")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, build, source in (("penguin599", penguin_items, SPLITS),
                                   ("vbench944", vbench_items, VBENCH)):
        items = build()
        check_items(items, dataset)
        path = args.out_dir / f"{dataset}.json"
        path.write_text(json.dumps(payload_for(dataset, items, source), indent=2,
                                   ensure_ascii=False), encoding="utf-8")
        multiline = sum(1 for item in items if len(item["prompt"].splitlines()) != 1)
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"{shown}: {len(items)} {dataset} items, {multiline} of them containing a "
              f"newline no line-based file could carry")
    assert set(EXPECTED) == set(DATASETS), "the manifests must cover the matrix's datasets"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
