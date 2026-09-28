#!/usr/bin/env python3
"""Freeze the schedule-search (SS) calibration set, arbitration set and spaces.

P0 of `docs/schedule_search_plan_zh.md`.  The eight calibration slots are
declared in advance and filled by deterministic rules, so neither the prompts
nor the seeds are picked by looking at any score:

1. person          COCO 2014 val, first image id whose instance annotations
2. animal          carry the slot's supercategory; for the three non-person
3. indoor_object   slots the image must additionally carry no `person`
4. outdoor_scene   annotation, so the four slots are actually different.
                   The caption is that image's lowest caption annotation id.
5. stylized_a      first two prompts, in pool order, of the committed
6. stylized_b      DiffusionDB clean calibration pool that contain one of
                   `oil painting` / `watercolor` / `illustration` and are
                   5..40 words.  The pool already excludes every clean10k
                   evaluation prompt, so the slot cannot leak.
7. counting        first COCO val caption by annotation id containing a
                   number word of three or more, on an unused image.
8. text_render     the fixed template
                   "a storefront with the word 'gleam' written on the sign".

The arbitration set is the next 50 COCO val captions by annotation id after
the largest one the calibration slots consumed.  The two calibration base
seeds clear every evaluation seed stream on both models (the matrix uses
41/42/43 on FLUX and 42/100042/200042 on Qwen-Image with the
`base + prompt_index` convention of `lib/io_utils.py`).

COCO 2014 val annotations are not in the repository and nothing here
downloads them.  Pass the two official files explicitly:

    captions_val2014.json    from annotations_trainval2014.zip
    instances_val2014.json   from annotations_trainval2014.zip
    (http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

Usage:

    python analysis/build_schedule_search_config.py \
        --coco_captions  /path/to/captions_val2014.json \
        --coco_instances /path/to/instances_val2014.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.exhaustive_schedule import schedule_for_rank  # noqa: E402
from lib.schedule_search import SearchSpace  # noqa: E402

MODELS = ("flux", "qwen")
KS = (29, 37, 41)

#: The matrix evaluation sets, as tabulated in
#: `analysis/build_sencache_recal_cells.py:36-44`.
EVAL_PROMPT_FILES = {
    ("flux", "drawbench_full"): "resources/prompts/prompt.txt",
    ("flux", "parti_full"): "resources/prompts/partiprompts_full_eval1632_seed42.txt",
    ("flux", "geneval_style"): "resources/prompts/geneval_seed43_n100.txt",
    ("flux", "diffusiondb_clean10k"): (
        "resources/prompts/diffusiondb_2m_clean_10000_seed42.txt"
    ),
    ("qwen", "drawbench_full"): (
        "reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt"
    ),
}
#: The matrix seed streams, as tabulated in `analysis/spx_coverage.py:82`.
EVAL_BASE_SEEDS = {"flux": (41, 42, 43), "qwen": (42, 100042, 200042)}
#: Largest prompt index any evaluation set reaches (DiffusionDB clean10k).
EVAL_MAX_PROMPT_INDEX = 9999

#: Calibration base seeds: slots 0..3 take the first, slots 4..7 the second.
CALIBRATION_SEEDS = (50042, 60042)

#: COCO supercategory required (and, for the last three, forbidden) per slot.
COCO_SLOTS = (
    ("person", "person", ()),
    ("animal", "animal", ("person",)),
    ("indoor_object", "indoor", ("person",)),
    ("outdoor_scene", "outdoor", ("person",)),
)
COUNTING_WORDS = ("three", "four", "five", "six", "seven", "eight", "nine", "ten")
#: Two term groups so the two stylized slots come from different media
#: rather than both matching "illustration".
STYLE_GROUPS = (
    ("stylized_a", ("oil painting", "watercolor")),
    ("stylized_b", ("illustration",)),
)
STYLE_MIN_WORDS, STYLE_MAX_WORDS = 5, 40
TEXT_RENDER_PROMPT = "a storefront with the word 'gleam' written on the sign"
TEXT_RENDER_WORD = "gleam"

ARBITRATION_COUNT = 50

#: The eight SPX baseline schedules that warm-start the climbers, in the order
#: `analysis/build_sp_cross_schedules.py` emits them.
WARM_START_NAMES = (
    "budcache",
    "meancache",
    "dpcache",
    "uniform",
    "seacache_top1",
    "teacache_top1",
    "sencache_top1",
    "dicache_top1",
)

#: Search recipe, from `docs/golden_path_search_bench.md`; only the chain
#: length and the cap scale with the free-slot count.
SEARCH_RECIPE = {
    "search_seed": 20260901,
    "caps": {"29": 1400, "37": 700, "41": 400},
    "probe_evals": 50,
    "chain_lengths": {"29": 680, "37": 360, "41": 200},
    "hill_iters": 20,
    "hill_window": 3,
    "local_probability": 0.7,
    "se_factor": 2.0,
    "stop_units": 2,
    "arbitration_candidates": 3,
    "arbitration_min_hamming": 4,
    "temperatures_note": (
        "null until P1: the probe run reports a suggested t_max (the order of "
        "magnitude of the median one-swap difference) and t_min (a tenth of the "
        "calibration standard error), and P1 writes the chosen pair here. The "
        "P0-frozen prompts and seeds are not touched by that edit; until it "
        "happens the annealing runs take --t_max / --t_min."
    ),
    "temperatures": {
        model: {str(k): {"t_max": None, "t_min": None} for k in KS}
        for model in MODELS
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco_captions", type=Path, required=True)
    p.add_argument("--coco_instances", type=Path, required=True)
    p.add_argument(
        "--diffusiondb_pool",
        type=Path,
        default=_ROOT
        / "resources/prompts/diffusiondb_2m_clean_calib512_seed20260729.txt",
    )
    p.add_argument(
        "--diffusiondb_metadata",
        type=Path,
        default=_ROOT
        / "resources/prompts/diffusiondb_2m_clean_calib512_seed20260729_metadata.jsonl",
    )
    p.add_argument(
        "--warm_start_dir", type=Path, default=_ROOT / "resources/sp_cross_schedules"
    )
    p.add_argument(
        "--exhaustive_summary",
        type=Path,
        default=_ROOT / "resources/exhaustive_k41/formal_results/summary.json",
    )
    p.add_argument(
        "--out", type=Path, default=_ROOT / "resources/schedule_search/config.v1.json"
    )
    p.add_argument(
        "--anchor_out",
        type=Path,
        default=_ROOT / "resources/schedule_search/anchor_flux_k41.txt",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _rel(path: Path) -> str:
    """Repo-relative path when the file lives in the repo, else the full one."""

    try:
        return str(Path(path).resolve().relative_to(_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# COCO slots
# --------------------------------------------------------------------------


def coco_tables(
    captions_path: Path, instances_path: Path
) -> tuple[dict[int, list[tuple[int, str]]], dict[int, set[str]], dict[str, set[str]]]:
    """`image_id -> [(caption_id, caption)]`, `image_id -> supercategories`,
    and `supercategory -> its category names` (the caption vocabulary)."""

    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    by_image: dict[int, list[tuple[int, str]]] = {}
    for row in captions["annotations"]:
        by_image.setdefault(int(row["image_id"]), []).append(
            (int(row["id"]), normalize(row["caption"]))
        )
    for rows in by_image.values():
        rows.sort()

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    supercategory = {
        int(row["id"]): str(row["supercategory"]) for row in instances["categories"]
    }
    vocab: dict[str, set[str]] = {}
    for row in instances["categories"]:
        vocab.setdefault(str(row["supercategory"]), set()).add(
            str(row["name"]).lower()
        )
    groups: dict[int, set[str]] = {}
    for row in instances["annotations"]:
        groups.setdefault(int(row["image_id"]), set()).add(
            supercategory[int(row["category_id"])]
        )
    return by_image, groups, vocab


#: Caption words accepted as evidence of a person, beyond the category name.
PERSON_WORDS = (
    "person", "people", "man", "men", "woman", "women", "boy", "boys",
    "girl", "girls", "child", "children",
)


def _caption_hits(caption: str, vocabulary: set[str]) -> list[str]:
    """Vocabulary terms present in the caption as whole words (plural allowed)."""

    tokens = set(re.findall(r"[a-z]+", caption.lower()))
    text = " ".join(sorted(tokens))
    hits = []
    for term in sorted(vocabulary):
        pieces = term.split()
        last = pieces[-1]
        singular_ok = all(piece in tokens for piece in pieces)
        plural_ok = all(
            piece in tokens for piece in pieces[:-1]
        ) and (last + "s") in tokens
        if singular_ok or plural_ok:
            hits.append(term)
    del text
    return hits


def pick_coco_category_slots(
    by_image: dict[int, list[tuple[int, str]]],
    groups: dict[int, set[str]],
    vocab: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """One caption per category slot, by ascending image id.

    The image's instance annotations must carry the slot's supercategory AND
    the caption itself must name a category of that supercategory: the prompt
    is what conditions generation, so an image-level annotation alone (a
    background person, an off-caption animal) is not evidence.
    """

    used: set[int] = set()
    picked: list[dict[str, Any]] = []
    order = sorted(groups)
    for slot, required, forbidden in COCO_SLOTS:
        vocabulary = set(vocab.get(required, set()))
        if required == "person":
            vocabulary |= set(PERSON_WORDS)
        found = False
        for image_id in order:
            if found:
                break
            if image_id in used or image_id not in by_image:
                continue
            present = groups[image_id]
            if required not in present:
                continue
            if any(term in present for term in forbidden):
                continue
            for caption_id, caption in by_image[image_id]:
                hits = _caption_hits(caption, vocabulary)
                if not hits:
                    continue
                used.add(image_id)
                picked.append(
                    {
                        "slot": slot,
                        "prompt": caption,
                        "source": {
                            "dataset": "coco_2014_val",
                            "rule": (
                                f"first image id whose instances carry "
                                f"supercategory '{required}'"
                                + (
                                    f" and none of {list(forbidden)}"
                                    if forbidden
                                    else ""
                                )
                                + "; caption = that image's lowest caption id "
                                "naming a category of that supercategory"
                            ),
                            "image_id": image_id,
                            "caption_id": caption_id,
                            "supercategories": sorted(present),
                            "caption_evidence": hits,
                        },
                    }
                )
                found = True
                break
        if not found:
            raise SystemExit(f"no COCO val image satisfies slot {slot}")
    return picked


def pick_counting_slot(
    by_image: dict[int, list[tuple[int, str]]], used_images: set[int]
) -> dict[str, Any]:
    """First caption by annotation id carrying a number word of three or more."""

    rows = sorted(
        (caption_id, image_id, caption)
        for image_id, entries in by_image.items()
        for caption_id, caption in entries
    )
    for caption_id, image_id, caption in rows:
        if image_id in used_images:
            continue
        tokens = set(re.findall(r"[a-z]+", caption.lower()))
        if tokens & set(COUNTING_WORDS):
            return {
                "slot": "counting",
                "prompt": caption,
                "source": {
                    "dataset": "coco_2014_val",
                    "rule": (
                        "first caption by annotation id containing one of "
                        f"{list(COUNTING_WORDS)}, on an image no other slot used"
                    ),
                    "image_id": image_id,
                    "caption_id": caption_id,
                },
            }
    raise SystemExit("no COCO val caption carries a number word of three or more")


def pick_arbitration_captions(
    by_image: dict[int, list[tuple[int, str]]],
    *,
    after_caption_id: int,
    used_images: set[int],
    count: int,
) -> list[dict[str, Any]]:
    """The next `count` captions by annotation id after the calibration set."""

    rows = sorted(
        (caption_id, image_id, caption)
        for image_id, entries in by_image.items()
        for caption_id, caption in entries
    )
    out: list[dict[str, Any]] = []
    for caption_id, image_id, caption in rows:
        if caption_id <= after_caption_id or image_id in used_images:
            continue
        out.append(
            {"prompt": caption, "image_id": image_id, "caption_id": caption_id}
        )
        if len(out) >= count:
            break
    if len(out) < count:
        raise SystemExit("COCO val ran out of captions for the arbitration set")
    return out


# --------------------------------------------------------------------------
# DiffusionDB slots
# --------------------------------------------------------------------------


def pick_stylized_slots(pool: Path, metadata: Path) -> list[dict[str, Any]]:
    """First two stylized prompts in pool order, 5..40 words.

    The pool is the committed clean calibration file, which was built with
    every clean10k evaluation prompt excluded, so the two slots cannot
    overlap the DiffusionDB evaluation set.
    """

    lines = [
        normalize(line) for line in pool.read_text(encoding="utf-8").splitlines()
    ]
    records = [
        json.loads(line)
        for line in metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(lines):
        raise SystemExit(
            f"pool ({len(lines)}) and metadata ({len(records)}) disagree in length"
        )
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for slot, terms in STYLE_GROUPS:
        for index, (prompt, record) in enumerate(zip(lines, records)):
            if normalize(record["prompt"]) != prompt:
                raise SystemExit(f"pool line {index} does not match its metadata row")
            if index in used:
                continue
            words = int(record["word_count"])
            if not STYLE_MIN_WORDS <= words <= STYLE_MAX_WORDS:
                continue
            lower = prompt.lower()
            hits = [term for term in terms if term in lower]
            if not hits:
                continue
            used.add(index)
            picked.append(
                {
                    "slot": slot,
                    "prompt": prompt,
                    "source": {
                        "dataset": "diffusiondb_2m_clean_calib512_seed20260729",
                        "rule": (
                            f"first prompt in pool order matching {list(terms)} "
                            f"with {STYLE_MIN_WORDS}..{STYLE_MAX_WORDS} words; the "
                            "pool already excludes the clean10k evaluation subset"
                        ),
                        "pool_index": index,
                        "word_count": words,
                        "style_terms": hits,
                    },
                }
            )
            break
        else:
            raise SystemExit(f"no DiffusionDB pool prompt satisfies slot {slot}")
    return picked


# --------------------------------------------------------------------------
# spaces, warm starts and checks
# --------------------------------------------------------------------------


def space_payload() -> dict[str, Any]:
    spec: dict[str, Any] = {}
    for model in MODELS:
        base = SearchSpace()
        spec[model] = {
            "num_steps": base.num_steps,
            "forced_full_steps": list(base.forced_full_steps),
            "variable_start": base.variable_start,
            "variable_end": base.variable_end,
            "free_full_count": {
                str(k): SearchSpace(cache_count=k).free_full_count for k in KS
            },
            "space_size": {str(k): SearchSpace(cache_count=k).total for k in KS},
            "payload": "residual_reuse",
            "source": (
                "analysis/build_spx_supplement_schedules.py:107 "
                "(CONTROL_FORCED_FULL, model-agnostic) and "
                "lib/exhaustive_schedule.py:22-28"
            ),
        }
    return spec


def warm_start_payload(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The in-domain SPX baseline schedules that lie inside each space."""

    kept: dict[str, dict[str, list[dict[str, str]]]] = {m: {} for m in MODELS}
    dropped: list[dict[str, Any]] = []
    for model in MODELS:
        for k in KS:
            space = SearchSpace(cache_count=k)
            rows: list[dict[str, str]] = []
            for name in WARM_START_NAMES:
                path = directory / f"{model}_k{k}_{name}.txt"
                if not path.is_file():
                    dropped.append(
                        {"model": model, "k": k, "name": name, "reason": "missing"}
                    )
                    continue
                bits = path.read_text(encoding="utf-8").strip()
                try:
                    space.combo_of_bits(bits)
                except ValueError as exc:
                    dropped.append(
                        {
                            "model": model,
                            "k": k,
                            "name": name,
                            "reason": str(exc),
                            "bits": bits,
                        }
                    )
                    continue
                rows.append({"name": name, "bits": bits})
            kept[model][str(k)] = rows
    return kept, dropped


def eval_seed_streams() -> dict[str, list[list[int]]]:
    """Closed intervals of per-image seeds the matrix already occupies."""

    return {
        model: [
            [base, base + EVAL_MAX_PROMPT_INDEX] for base in EVAL_BASE_SEEDS[model]
        ]
        for model in MODELS
    }


def run_checks(calibration: list[dict[str, Any]]) -> dict[str, Any]:
    """Leakage and seed-disjointness checks; any failure stops the build."""

    word = re.compile(rf"\b{TEXT_RENDER_WORD}\b", re.IGNORECASE)
    derived = re.compile(rf"{TEXT_RENDER_WORD}[a-z]+", re.IGNORECASE)
    gleam: dict[str, Any] = {}
    prompt_leaks: list[dict[str, Any]] = []
    calibration_texts = {normalize(row["prompt"]).lower() for row in calibration}
    for (model, dataset), relative in EVAL_PROMPT_FILES.items():
        path = _ROOT / relative
        text = path.read_text(encoding="utf-8")
        gleam[f"{model}/{dataset}"] = {
            "file": relative,
            "exact_word_hits": len(word.findall(text)),
            "derived_form_hits": len(derived.findall(text)),
        }
        for line in text.splitlines():
            if normalize(line).lower() in calibration_texts:
                prompt_leaks.append({"dataset": f"{model}/{dataset}", "prompt": line})
    bad = [key for key, row in gleam.items() if row["exact_word_hits"]]
    if bad:
        raise SystemExit(f"the text-render word appears in evaluation sets: {bad}")
    if prompt_leaks:
        raise SystemExit(f"calibration prompt found in an evaluation set: {prompt_leaks}")

    seeds = sorted({int(row["seed"]) for row in calibration})
    streams = eval_seed_streams()
    clashes = [
        {"model": model, "seed": seed, "interval": interval}
        for model, intervals in streams.items()
        for interval in intervals
        for seed in seeds
        if interval[0] <= seed <= interval[1]
    ]
    if clashes:
        raise SystemExit(f"calibration seeds collide with evaluation streams: {clashes}")
    return {
        "text_render_word": gleam,
        "calibration_prompts_in_evaluation_sets": 0,
        "calibration_seeds": seeds,
        "evaluation_seed_streams": streams,
    }


def write_anchor_file(path: Path, summary_path: Path, warm_dir: Path) -> dict[str, Any]:
    """The three known K41 schedules the P0 fidelity anchor re-scores."""

    rank = int(json.loads(summary_path.read_text(encoding="utf-8"))["best_mean"]["rank"])
    full_steps, _cache, bits = schedule_for_rank(rank)
    rows = [{"name": f"table_best_rank_{rank}", "bits": bits}]
    for name in ("budcache", "uniform"):
        rows.append(
            {
                "name": name,
                "bits": (warm_dir / f"flux_k41_{name}.txt")
                .read_text(encoding="utf-8")
                .strip(),
            }
        )
    space = SearchSpace(cache_count=41)
    for row in rows:
        row["full_steps"] = list(space.full_steps(space.combo_of_bits(row["bits"])))
    header = "\n".join(f"# {i + 1}: {row['name']}" for i, row in enumerate(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        header + "\n" + "\n".join(row["bits"] for row in rows) + "\n", encoding="utf-8"
    )
    assert full_steps == tuple(rows[0]["full_steps"])
    return {"file": _rel(path), "schedules": rows}


def main() -> int:
    args = parse_args()

    by_image, groups, vocab = coco_tables(args.coco_captions, args.coco_instances)
    slots = pick_coco_category_slots(by_image, groups, vocab)
    used_images = {row["source"]["image_id"] for row in slots}
    slots.extend(pick_stylized_slots(args.diffusiondb_pool, args.diffusiondb_metadata))
    counting = pick_counting_slot(by_image, used_images)
    slots.append(counting)
    used_images.add(counting["source"]["image_id"])
    slots.append(
        {
            "slot": "text_render",
            "prompt": TEXT_RENDER_PROMPT,
            "source": {"dataset": "fixed_template", "rule": "frozen in the plan"},
        }
    )
    if len(slots) != 8:
        raise SystemExit(f"expected 8 calibration slots, built {len(slots)}")
    for index, row in enumerate(slots):
        row["seed"] = CALIBRATION_SEEDS[0 if index < 4 else 1]

    max_caption_id = max(
        int(row["source"]["caption_id"])
        for row in slots
        if "caption_id" in row["source"]
    )
    arbitration = pick_arbitration_captions(
        by_image,
        after_caption_id=max_caption_id,
        used_images=used_images,
        count=ARBITRATION_COUNT,
    )

    checks = run_checks(slots)
    warm_starts, warm_dropped = warm_start_payload(args.warm_start_dir)
    anchor = write_anchor_file(args.anchor_out, args.exhaustive_summary, args.warm_start_dir)

    config = {
        "schema": "schedule_search_config.v1",
        "plan": "docs/schedule_search_plan_zh.md",
        "calibration": {
            "seeds": list(CALIBRATION_SEEDS),
            "seed_rule": "slots 0..3 take seeds[0], slots 4..7 take seeds[1]",
            "pairs": slots,
        },
        "arbitration": {
            "seed": CALIBRATION_SEEDS[0],
            "rule": (
                "the next 50 COCO 2014 val captions by annotation id after the "
                "largest id the calibration slots used, skipping images the "
                "calibration already used"
            ),
            "prompts": arbitration,
        },
        "spaces": space_payload(),
        "search": SEARCH_RECIPE,
        "warm_starts": warm_starts,
        "warm_starts_excluded": warm_dropped,
        "anchor": anchor,
        "checks": checks,
        "sources": {
            "coco_captions": {
                "path": str(args.coco_captions),
                "sha256": file_sha256(args.coco_captions),
            },
            "coco_instances": {
                "path": str(args.coco_instances),
                "sha256": file_sha256(args.coco_instances),
            },
            "diffusiondb_pool": {
                "path": _rel(args.diffusiondb_pool),
                "sha256": file_sha256(args.diffusiondb_pool),
            },
            "diffusiondb_metadata": {
                "path": _rel(args.diffusiondb_metadata),
                "sha256": file_sha256(args.diffusiondb_metadata),
            },
            "warm_start_dir": _rel(args.warm_start_dir),
            "exhaustive_summary": _rel(args.exhaustive_summary),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {args.out}")
    for row in slots:
        print(f"  {row['slot']:<14} seed={row['seed']}  {row['prompt'][:76]}")
    print(f"  arbitration: {len(arbitration)} captions, "
          f"ids {arbitration[0]['caption_id']}..{arbitration[-1]['caption_id']}")
    for model in MODELS:
        counts = {k: len(warm_starts[model][str(k)]) for k in KS}
        print(f"  warm starts {model}: {counts} (excluded {len(warm_dropped)} in total)")
    print(f"[INFO] wrote {args.anchor_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
