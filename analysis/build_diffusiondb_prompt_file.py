#!/usr/bin/env python3
"""Build a clean DiffusionDB prompt file for cache-fidelity experiments.

The script consumes DiffusionDB metadata only. It never downloads or reads the
image archives. Selection is deterministic: prompts are filtered, normalized,
bucketed, then sampled by a stable hash score so the output does not depend on
metadata row order.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from urllib.request import urlretrieve


DIFFUSIONDB_URLS = {
    "2m": "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet",
    "large": "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet",
}

KEEP_COLUMNS = (
    "image_name",
    "prompt",
    "part_id",
    "seed",
    "step",
    "cfg",
    "sampler",
    "width",
    "height",
    "user_name",
    "timestamp",
    "image_nsfw",
    "prompt_nsfw",
)

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is",
    "of", "on", "the", "to", "with",
}

STYLE_TERMS = {
    "artstation", "cinematic", "concept", "digital", "drawing", "engine",
    "fantasy", "highly", "illustration", "octane", "painting", "photo",
    "photorealistic", "render", "realistic", "studio", "trending", "unreal",
}

BLOCKLIST_TERMS = {
    "child porn", "explicit", "gore", "hentai", "naked", "nude", "nsfw",
    "porn", "pornographic", "sex", "sexual",
}

SUBJECT_BUCKETS = (
    ("person", ("person", "portrait", "woman", "man", "girl", "boy", "face", "character")),
    ("animal", ("animal", "cat", "dog", "bird", "horse", "dragon", "fish", "lion", "tiger")),
    ("landscape", ("landscape", "mountain", "forest", "river", "ocean", "city", "street", "sky")),
    ("architecture", ("architecture", "building", "interior", "room", "house", "castle", "temple")),
    ("vehicle", ("car", "truck", "train", "ship", "spaceship", "airplane", "motorcycle")),
    ("object", ("object", "product", "chair", "table", "phone", "watch", "weapon", "flower")),
    ("fantasy_scifi", ("alien", "cyberpunk", "fantasy", "future", "robot", "sci-fi", "space")),
    ("food", ("food", "cake", "coffee", "fruit", "meal", "pizza", "restaurant")),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata_parquet", type=Path, default=None,
                   help="Local DiffusionDB metadata parquet. If omitted, use --cache_dir.")
    p.add_argument("--subset", choices=sorted(DIFFUSIONDB_URLS), default="2m",
                   help="Official DiffusionDB metadata table to use when downloading.")
    p.add_argument("--cache_dir", type=Path, default=Path("resources/prompts/datasets/diffusiondb"),
                   help="Where to store downloaded metadata outside tracked prompt outputs.")
    p.add_argument("--download_if_missing", action="store_true",
                   help="Download the selected metadata parquet if it is missing.")
    p.add_argument("--out_dir", type=Path, default=Path("resources/prompts"))
    p.add_argument("--prefix", default=None,
                   help="Output prefix. Default: diffusiondb_<subset>_clean_<n>_seed<seed>.")
    p.add_argument(
        "--exclude_prompt_file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prompt file whose normalized prompts must be excluded. "
            "Repeat this option for multiple files."
        ),
    )
    p.add_argument("--target_n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=65_536)
    p.add_argument("--per_bucket_cap", type=int, default=5_000)
    p.add_argument("--min_words", type=int, default=5)
    p.add_argument("--max_words", type=int, default=90)
    p.add_argument("--min_chars", type=int, default=20)
    p.add_argument("--max_chars", type=int, default=650)
    p.add_argument("--max_prompt_nsfw", type=float, default=0.05)
    p.add_argument("--max_image_nsfw", type=float, default=0.20)
    p.add_argument("--min_ascii_ratio", type=float, default=0.92)
    p.add_argument("--min_source_step", type=int, default=10)
    p.add_argument("--max_source_step", type=int, default=150)
    p.add_argument("--min_source_cfg", type=float, default=1.0)
    p.add_argument("--max_source_cfg", type=float, default=30.0)
    p.add_argument("--min_source_side", type=int, default=256)
    p.add_argument("--max_source_side", type=int, default=1536)
    p.add_argument("--max_source_aspect", type=float, default=3.0)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "as_py"):
        return _jsonable(v.as_py())
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    try:
        import numpy as np  # type: ignore

        if isinstance(v, np.generic):
            return _jsonable(v.item())
    except Exception:
        pass
    return v


def _metadata_path(args: argparse.Namespace) -> Path:
    if args.metadata_parquet is not None:
        return args.metadata_parquet
    filename = "metadata-large.parquet" if args.subset == "large" else "metadata.parquet"
    return args.cache_dir / filename


def _ensure_metadata(args: argparse.Namespace) -> Path:
    path = _metadata_path(args)
    if path.is_file():
        return path
    if not args.download_if_missing:
        raise SystemExit(
            f"metadata parquet missing: {path}. Pass --download_if_missing or --metadata_parquet."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    url = DIFFUSIONDB_URLS[str(args.subset)]
    print(f"[DOWNLOAD] {url} -> {path}")
    urlretrieve(url, path)
    return path


def _iter_rows_pyarrow(path: Path, batch_size: int) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema.names)
    columns = [c for c in KEEP_COLUMNS if c in schema_names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        names = batch.schema.names
        cols = [batch.column(i).to_pylist() for i in range(len(names))]
        for values in zip(*cols):
            yield {name: _jsonable(value) for name, value in zip(names, values)}


def _iter_rows_pandas(path: Path) -> Iterator[Dict[str, Any]]:
    import pandas as pd  # type: ignore

    df = pd.read_parquet(path)
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]]
    for row in df.to_dict(orient="records"):
        yield {k: _jsonable(v) for k, v in row.items()}


def _iter_rows(path: Path, batch_size: int) -> Iterator[Dict[str, Any]]:
    try:
        yield from _iter_rows_pyarrow(path, batch_size)
    except ImportError:
        yield from _iter_rows_pandas(path)


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _as_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _canonical_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", prompt)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?]){3,}", r"\1\1", text)
    return text.strip(" ,")


def _dedupe_key(prompt: str) -> str:
    text = _canonical_prompt(prompt).lower()
    text = re.sub(r"\b(4k|8k|16k|uhd|hd)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip(" ,")


def _tokens(prompt: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'-]*", prompt)


def _ascii_ratio(prompt: str) -> float:
    if not prompt:
        return 0.0
    ascii_chars = sum(1 for ch in prompt if ord(ch) < 128)
    return ascii_chars / max(1, len(prompt))


def _reject_reason(prompt: str, args: argparse.Namespace) -> Optional[str]:
    if len(prompt) < int(args.min_chars):
        return "too_short_chars"
    if len(prompt) > int(args.max_chars):
        return "too_long_chars"
    toks = _tokens(prompt)
    if len(toks) < int(args.min_words):
        return "too_few_words"
    if len(toks) > int(args.max_words):
        return "too_many_words"
    if _ascii_ratio(prompt) < float(args.min_ascii_ratio):
        return "non_english_or_non_ascii"
    lower = prompt.lower()
    if any(term in lower for term in BLOCKLIST_TERMS):
        return "blocked_sensitive_term"
    if "negative prompt:" in lower or "parameters:" in lower:
        return "parameter_dump"
    if re.search(r"https?://|www\.|@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}", prompt):
        return "url_or_email"
    if re.search(r"(.)\1{7,}", prompt):
        return "repeated_character_noise"
    if len(re.findall(r"[{}<>|\\^~`]", prompt)) > 3:
        return "symbol_noise"
    letters = sum(ch.isalpha() for ch in prompt)
    if letters / max(1, len(prompt)) < 0.35:
        return "low_letter_ratio"
    return None


def _style_bucket(prompt: str) -> str:
    lower = prompt.lower()
    toks = [t.lower() for t in _tokens(prompt)]
    stop_count = sum(1 for t in toks if t in STOPWORDS)
    style_count = sum(1 for t in toks if t in STYLE_TERMS)
    comma_count = prompt.count(",")
    if comma_count >= 4 or style_count >= 3:
        return "keyword_style"
    if stop_count >= 3 and comma_count <= 2:
        return "natural_sentence"
    return "hybrid"


def _length_bucket(n_words: int) -> str:
    if n_words <= 12:
        return "short"
    if n_words <= 32:
        return "medium"
    return "long"


def _subject_bucket(prompt: str) -> str:
    lower = prompt.lower()
    for bucket, terms in SUBJECT_BUCKETS:
        if any(term in lower for term in terms):
            return bucket
    return "other"


def _stable_score(seed: int, text: str) -> float:
    h = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2 ** 64)


def _source_metadata_reason(row: Dict[str, Any], args: argparse.Namespace) -> Optional[str]:
    seed = _as_int(row.get("seed"))
    step = _as_int(row.get("step"))
    cfg = _as_float(row.get("cfg"))
    sampler = _as_int(row.get("sampler"))
    width = _as_int(row.get("width"))
    height = _as_int(row.get("height"))
    if seed is None or step is None or cfg is None or sampler is None or width is None or height is None:
        return "missing_source_metadata"
    if sampler < 1 or sampler > 9:
        return "sampler_out_of_range"
    if step < int(args.min_source_step) or step > int(args.max_source_step):
        return "step_out_of_range"
    if cfg < float(args.min_source_cfg) or cfg > float(args.max_source_cfg):
        return "cfg_out_of_range"
    if (
        width < int(args.min_source_side)
        or height < int(args.min_source_side)
        or width > int(args.max_source_side)
        or height > int(args.max_source_side)
    ):
        return "side_out_of_range"
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > float(args.max_source_aspect):
        return "aspect_out_of_range"
    return None


def _push_bucket(
    buckets: Dict[tuple[str, str, str], list[tuple[float, int, Dict[str, Any]]]],
    key: tuple[str, str, str],
    score: float,
    counter: int,
    record: Dict[str, Any],
    cap: int,
) -> None:
    heap = buckets[key]
    item = (-score, counter, record)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _select_records(
    buckets: Dict[tuple[str, str, str], list[tuple[float, int, Dict[str, Any]]]],
    target_n: int,
) -> list[Dict[str, Any]]:
    by_bucket: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for key, heap in buckets.items():
        rows = [item[2] for item in heap]
        rows.sort(key=lambda row: (float(row["_selection_score"]), str(row["prompt"])))
        by_bucket[key] = rows

    selected: list[Dict[str, Any]] = []
    selected_keys: set[str] = set()
    active = sorted(by_bucket)
    cursor = {key: 0 for key in active}
    while len(selected) < target_n and active:
        next_active = []
        for key in active:
            idx = cursor[key]
            rows = by_bucket[key]
            if idx < len(rows):
                row = rows[idx]
                cursor[key] += 1
                dkey = str(row["dedupe_key"])
                if dkey not in selected_keys:
                    selected.append(row)
                    selected_keys.add(dkey)
                    if len(selected) >= target_n:
                        break
            if cursor[key] < len(rows):
                next_active.append(key)
        active = next_active

    if len(selected) < target_n:
        leftovers = []
        for rows in by_bucket.values():
            leftovers.extend(rows)
        leftovers.sort(key=lambda row: (float(row["_selection_score"]), str(row["prompt"])))
        for row in leftovers:
            dkey = str(row["dedupe_key"])
            if dkey in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(dkey)
            if len(selected) >= target_n:
                break
    return selected


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_excluded_prompt_keys(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"excluded prompt file does not exist: {path}")
        for raw_prompt in path.read_text(encoding="utf-8").splitlines():
            prompt = _canonical_prompt(raw_prompt)
            if prompt:
                excluded.add(_dedupe_key(prompt))
    return excluded


def main() -> int:
    args = parse_args()
    metadata_path = _ensure_metadata(args)
    excluded_prompt_keys = _load_excluded_prompt_keys(args.exclude_prompt_file)
    prefix = args.prefix or f"diffusiondb_{args.subset}_clean_{args.target_n}_seed{args.seed}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    seen: set[str] = set()
    buckets: Dict[tuple[str, str, str], list[tuple[float, int, Dict[str, Any]]]] = defaultdict(list)

    kept_counter = 0
    for row_idx, row in enumerate(_iter_rows(metadata_path, int(args.batch_size))):
        counters["rows_seen"] += 1
        raw_prompt = row.get("prompt")
        if raw_prompt is None:
            counters["missing_prompt"] += 1
            continue
        prompt = _canonical_prompt(str(raw_prompt))
        reason = _reject_reason(prompt, args)
        if reason is not None:
            counters[f"reject_{reason}"] += 1
            continue
        reason = _source_metadata_reason(row, args)
        if reason is not None:
            counters[f"reject_{reason}"] += 1
            continue
        prompt_nsfw = _as_float(row.get("prompt_nsfw"))
        if prompt_nsfw is not None and prompt_nsfw > float(args.max_prompt_nsfw):
            counters["reject_prompt_nsfw"] += 1
            continue
        image_nsfw = _as_float(row.get("image_nsfw"))
        if image_nsfw is not None and image_nsfw > float(args.max_image_nsfw):
            counters["reject_image_nsfw"] += 1
            continue
        dkey = _dedupe_key(prompt)
        if not dkey:
            counters["reject_empty_dedupe_key"] += 1
            continue
        if dkey in excluded_prompt_keys:
            counters["reject_excluded_prompt"] += 1
            continue
        if dkey in seen:
            counters["reject_duplicate_normalized_prompt"] += 1
            continue
        seen.add(dkey)

        toks = _tokens(prompt)
        style = _style_bucket(prompt)
        length = _length_bucket(len(toks))
        subject = _subject_bucket(prompt)
        bucket_key = (style, length, subject)
        score = _stable_score(int(args.seed), dkey)
        record = {
            "source_row_idx": int(row_idx),
            "prompt": prompt,
            "dedupe_key": dkey,
            "selection_bucket": "|".join(bucket_key),
            "style_bucket": style,
            "length_bucket": length,
            "subject_bucket": subject,
            "word_count": len(toks),
            "char_count": len(prompt),
            "ascii_ratio": _ascii_ratio(prompt),
            "_selection_score": score,
            "source": {
                "image_name": row.get("image_name"),
                "part_id": _as_int(row.get("part_id")),
                "seed": _as_int(row.get("seed")),
                "step": _as_int(row.get("step")),
                "cfg": _as_float(row.get("cfg")),
                "sampler": _as_int(row.get("sampler")),
                "width": _as_int(row.get("width")),
                "height": _as_int(row.get("height")),
                "user_name_hash": row.get("user_name"),
                "timestamp": str(row.get("timestamp")) if row.get("timestamp") is not None else None,
                "image_nsfw": image_nsfw,
                "prompt_nsfw": prompt_nsfw,
            },
        }
        kept_counter += 1
        bucket_counts["|".join(bucket_key)] += 1
        style_counts[style] += 1
        length_counts[length] += 1
        subject_counts[subject] += 1
        _push_bucket(buckets, bucket_key, score, kept_counter, record, int(args.per_bucket_cap))

    selected = _select_records(buckets, int(args.target_n))
    if len(selected) != int(args.target_n):
        msg = f"selected {len(selected)} prompts, expected {args.target_n}"
        if args.strict:
            raise SystemExit(msg)
        print(f"[WARN] {msg}")

    for idx, row in enumerate(selected):
        row["selected_idx"] = idx
        row.pop("_selection_score", None)

    prompt_path = args.out_dir / f"{prefix}.txt"
    metadata_jsonl = args.out_dir / f"{prefix}_metadata.jsonl"
    manifest_path = args.out_dir / f"{prefix}_manifest.json"

    prompt_path.write_text("\n".join(str(row["prompt"]) for row in selected) + "\n", encoding="utf-8")
    metadata_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )

    selected_style = Counter(str(row["style_bucket"]) for row in selected)
    selected_length = Counter(str(row["length_bucket"]) for row in selected)
    selected_subject = Counter(str(row["subject_bucket"]) for row in selected)
    selected_bucket = Counter(str(row["selection_bucket"]) for row in selected)
    manifest = {
        "schema": "hiche_diffusiondb_prompt_file.v1",
        "source": {
            "dataset": "poloclub/diffusiondb",
            "subset": str(args.subset),
            "metadata_parquet": str(metadata_path),
            "metadata_url": DIFFUSIONDB_URLS[str(args.subset)],
            "metadata_sha256": _sha256_file(metadata_path),
            "note": "Metadata-only prompt selection; no DiffusionDB images are used.",
        },
        "selection": {
            "seed": int(args.seed),
            "target_n": int(args.target_n),
            "selected_n": len(selected),
            "per_bucket_cap": int(args.per_bucket_cap),
            "excluded_prompt_files": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for path in args.exclude_prompt_file
            ],
            "excluded_normalized_prompt_count": len(excluded_prompt_keys),
            "filters": {
                "min_words": int(args.min_words),
                "max_words": int(args.max_words),
                "min_chars": int(args.min_chars),
                "max_chars": int(args.max_chars),
                "max_prompt_nsfw": float(args.max_prompt_nsfw),
                "max_image_nsfw": float(args.max_image_nsfw),
                "min_ascii_ratio": float(args.min_ascii_ratio),
                "source_metadata": {
                    "min_source_step": int(args.min_source_step),
                    "max_source_step": int(args.max_source_step),
                    "min_source_cfg": float(args.min_source_cfg),
                    "max_source_cfg": float(args.max_source_cfg),
                    "min_source_side": int(args.min_source_side),
                    "max_source_side": int(args.max_source_side),
                    "max_source_aspect": float(args.max_source_aspect),
                    "sampler_range": [1, 9],
                },
                "blocklist_terms": sorted(BLOCKLIST_TERMS),
            },
            "algorithm": "filter -> normalized exact dedupe -> style/length/subject buckets -> stable-hash round-robin",
        },
        "stats": {
            "counters": dict(sorted(counters.items())),
            "accepted_unique_prompts": int(kept_counter),
            "accepted_bucket_counts": dict(sorted(bucket_counts.items())),
            "accepted_style_counts": dict(sorted(style_counts.items())),
            "accepted_length_counts": dict(sorted(length_counts.items())),
            "accepted_subject_counts": dict(sorted(subject_counts.items())),
            "selected_bucket_counts": dict(sorted(selected_bucket.items())),
            "selected_style_counts": dict(sorted(selected_style.items())),
            "selected_length_counts": dict(sorted(selected_length.items())),
            "selected_subject_counts": dict(sorted(selected_subject.items())),
        },
        "outputs": {
            "prompt_file": str(prompt_path),
            "metadata_jsonl": str(metadata_jsonl),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {prompt_path} ({len(selected)} prompts)")
    print(f"wrote {metadata_jsonl}")
    print(f"wrote {manifest_path}")
    print(json.dumps({
        "rows_seen": counters["rows_seen"],
        "accepted_unique_prompts": kept_counter,
        "selected": len(selected),
        "selected_style_counts": dict(sorted(selected_style.items())),
        "selected_length_counts": dict(sorted(selected_length.items())),
        "selected_subject_counts": dict(sorted(selected_subject.items())),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
