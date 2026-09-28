#!/usr/bin/env python3
"""Build GenEval-style prompt files for cache-fidelity experiments.

This is an adapted, repository-local version of the public GenEval prompt
generator. It writes the two artifacts we need:

- one-prompt-per-line text for the HiCache runners;
- JSONL metadata for future official GenEval evaluator integration.

The current cache-fidelity pipeline consumes only the text file. The JSONL is
kept so that later detector-based GenEval evaluation can use the same prompt
set, rather than silently changing the benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
POSITIONS = ["left of", "right of", "above", "below"]
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object_names", type=Path,
                   default=Path("resources/prompts/datasets/geneval_object_names.txt"))
    p.add_argument("--out_dir", type=Path, default=Path("resources/prompts"))
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--num_prompts_per_task", "-n", type=int, default=100,
                   help="GenEval's n for two-object/counting/colors/position/color-attr tasks.")
    p.add_argument("--prefix", default=None,
                   help="Output prefix. Default: geneval_seed<seed>_n<num_prompts_per_task>.")
    return p.parse_args()


def _read_object_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if "person" not in names:
        raise SystemExit(f"object names must include 'person': {path}")
    return names


def _with_article(name: str) -> str:
    return f"an {name}" if name[0] in "aeiou" else f"a {name}"


def _make_plural(name: str) -> str:
    return f"{name}es" if name.endswith("s") else f"{name}s"


def _single_object_samples(rng: np.random.Generator, classnames: list[str]) -> list[dict[str, Any]]:
    idxs = rng.choice(len(classnames), size=len(classnames), replace=False)
    return [
        {
            "tag": "single_object",
            "include": [{"class": classnames[int(idx)], "count": 1}],
            "prompt": f"a photo of {_with_article(classnames[int(idx)])}",
        }
        for idx in idxs
    ]


def _two_object_sample(rng: np.random.Generator, classnames: list[str]) -> dict[str, Any]:
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    a = classnames[int(idx_a)]
    b = classnames[int(idx_b)]
    return {
        "tag": "two_object",
        "include": [{"class": a, "count": 1}, {"class": b, "count": 1}],
        "prompt": f"a photo of {_with_article(a)} and {_with_article(b)}",
    }


def _counting_sample(rng: np.random.Generator, classnames: list[str], max_count: int = 4) -> dict[str, Any]:
    idx = int(rng.choice(len(classnames)))
    num = int(rng.integers(2, max_count, endpoint=True))
    name = classnames[idx]
    return {
        "tag": "counting",
        "include": [{"class": name, "count": num}],
        "exclude": [{"class": name, "count": num + 1}],
        "prompt": f"a photo of {NUMBERS[num]} {_make_plural(name)}",
    }


def _non_person_index(rng: np.random.Generator, classnames: list[str]) -> int:
    idx = int(rng.choice(len(classnames) - 1) + 1)
    return int((idx + classnames.index("person")) % len(classnames))


def _color_sample(rng: np.random.Generator, classnames: list[str]) -> dict[str, Any]:
    idx = _non_person_index(rng, classnames)
    color = COLORS[int(rng.choice(len(COLORS)))]
    name = classnames[idx]
    return {
        "tag": "colors",
        "include": [{"class": name, "count": 1, "color": color}],
        "prompt": f"a photo of {_with_article(color)} {name}",
    }


def _position_sample(rng: np.random.Generator, classnames: list[str]) -> dict[str, Any]:
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    a = classnames[int(idx_a)]
    b = classnames[int(idx_b)]
    position = POSITIONS[int(rng.choice(len(POSITIONS)))]
    return {
        "tag": "position",
        "include": [
            {"class": b, "count": 1},
            {"class": a, "count": 1, "position": [position, 0]},
        ],
        "prompt": f"a photo of {_with_article(a)} {position} {_with_article(b)}",
    }


def _color_attribution_sample(rng: np.random.Generator, classnames: list[str]) -> dict[str, Any]:
    raw = rng.choice(len(classnames) - 1, size=2, replace=False) + 1
    idx_a, idx_b = ((raw + classnames.index("person")) % len(classnames)).tolist()
    cidx_a, cidx_b = rng.choice(len(COLORS), size=2, replace=False)
    a = classnames[int(idx_a)]
    b = classnames[int(idx_b)]
    ca = COLORS[int(cidx_a)]
    cb = COLORS[int(cidx_b)]
    return {
        "tag": "color_attr",
        "include": [
            {"class": a, "count": 1, "color": ca},
            {"class": b, "count": 1, "color": cb},
        ],
        "prompt": f"a photo of {_with_article(ca)} {a} and {_with_article(cb)} {b}",
    }


def _dedupe(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        key = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def _generate(classnames: list[str], seed: int, n: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    samples.extend(_single_object_samples(rng, classnames))
    for _ in range(n):
        samples.append(_two_object_sample(rng, classnames))
    for _ in range(n):
        samples.append(_counting_sample(rng, classnames))
    for _ in range(n):
        samples.append(_color_sample(rng, classnames))
    for _ in range(n):
        samples.append(_position_sample(rng, classnames))
    for _ in range(n):
        samples.append(_color_attribution_sample(rng, classnames))
    return _dedupe(samples)


def main() -> int:
    args = parse_args()
    classnames = _read_object_names(args.object_names)
    prefix = args.prefix or f"geneval_seed{args.seed}_n{args.num_prompts_per_task}"
    samples = _generate(classnames, int(args.seed), int(args.num_prompts_per_task))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.out_dir / f"{prefix}.txt"
    metadata_path = args.out_dir / f"{prefix}_metadata.jsonl"
    manifest_path = args.out_dir / f"{prefix}_manifest.json"

    prompt_path.write_text("\n".join(str(s["prompt"]) for s in samples) + "\n", encoding="utf-8")
    metadata_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    tag_counts: dict[str, int] = {}
    for sample in samples:
        tag_counts[str(sample["tag"])] = tag_counts.get(str(sample["tag"]), 0) + 1
    manifest = {
        "schema": "hiche_geneval_prompt_file.v1",
        "source": {
            "repo": "https://github.com/djghosh13/geneval",
            "commit": "af4902f24d3ca90ebbb446dd9891a59e0f82725f",
            "generator": "prompts/create_prompts.py",
            "object_names": "prompts/object_names.txt",
            "note": "Adapted to write HiCache one-prompt-per-line text plus JSONL metadata.",
        },
        "seed": int(args.seed),
        "num_prompts_per_task": int(args.num_prompts_per_task),
        "prompt_count": len(samples),
        "tag_counts": tag_counts,
        "prompt_file": str(prompt_path),
        "metadata_jsonl": str(metadata_path),
        "object_names_file": str(args.object_names),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {prompt_path} ({len(samples)} prompts)")
    print(f"wrote {metadata_path}")
    print(f"wrote {manifest_path}")
    print(json.dumps(tag_counts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
