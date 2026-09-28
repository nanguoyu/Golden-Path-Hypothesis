#!/usr/bin/env python3
"""Convert a generation prompt manifest to the video evaluator's JSONL input."""
import argparse
import json
from pathlib import Path


def convert(source: Path, output: Path) -> int:
    data = json.loads(source.read_text(encoding="utf-8"))
    items = data["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("The generation manifest must contain a nonempty items list")
    rows = []
    seen = set()
    for index, item in enumerate(items):
        prompt = item["prompt"]
        prompt_id = item["prompt_id"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty or invalid prompt at index {index}")
        if prompt_id in seen:
            raise ValueError(f"Duplicate prompt_id: {prompt_id}")
        if "prompt_idx" in item and item["prompt_idx"] != index:
            raise ValueError(f"Noncontiguous prompt_idx at index {index}")
        seen.add(prompt_id)
        rows.append({"task_idx": index, "prompt_id": prompt_id, "prompt": prompt})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    print(f"Wrote {convert(args.input, args.output)} prompts to {args.output}")


if __name__ == "__main__":
    main()
