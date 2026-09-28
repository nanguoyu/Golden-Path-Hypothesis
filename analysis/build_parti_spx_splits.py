#!/usr/bin/env python3
"""Declare the SPX prompt split of PartiPrompts 1632 (plan section 3, 冻结边界).

The SPX plan and the paper both commit to a three-way partition declared BEFORE
path extraction: gate-derived schedule candidates are read off the discovery
prompts only, analysis choices are frozen with discovery + validation, and the
final statistics are evaluated on test. Until this file exists that boundary is
an instruction with no implementation -- there is nothing to filter by, so the
pooled `parti_full` extractions stay the only extractions and W1 cannot be
compliantly submitted.

Equal thirds (544/544/544), one seeded shuffle, no stratification: symmetric
sizes leave no tuning temptation, discovery keeps 544 x 3 seeds = 1,632
generations per (model, K, gate) for modal-path extraction, and the roles are
disjoint by construction. The manifest binds the exact prompt file by sha256 --
an index list is meaningless against any other file -- and self-hashes, so an
edited split no longer loads.

    python analysis/build_parti_spx_splits.py \\
        --out resources/sp_cross_schedules/parti_spx_splits.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "sp_cross.parti_splits.v1"
PROMPT_FILE = ROOT / "resources/prompts/partiprompts_full_eval1632_seed42.txt"
ROLES = ("discovery", "validation", "test")
SEED = 42


def hash_payload(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_split(prompt_count: int, seed: int = SEED) -> dict[str, list[int]]:
    """Disjoint thirds of `range(prompt_count)`, one seeded shuffle.

    The remainder (1632 % 3 == 0 today, so none) would go to test, never to
    discovery: an extra held-out prompt is harmless, an extra extraction prompt
    moves the candidates.
    """
    indices = list(range(prompt_count))
    random.Random(seed).shuffle(indices)
    third = prompt_count // 3
    return {
        "discovery": sorted(indices[:third]),
        "validation": sorted(indices[third:2 * third]),
        "test": sorted(indices[2 * third:]),
    }


def load_split(path: Path, *, role: str, prompt_file_sha256: str | None = None
               ) -> set[int]:
    """One role's prompt indices, with the file's own integrity checks applied.

    `prompt_file_sha256`, when given, must match the digest the split was
    declared against -- an index list applied to any other prompt file selects
    different prompts under the same names.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise SystemExit(f"{path} is not a {SCHEMA} manifest")
    if payload.get("manifest_sha256") != hash_payload(payload):
        raise SystemExit(f"{path} failed its self-hash; the split has been edited")
    if role not in payload["roles"]:
        raise SystemExit(f"{path} has no role {role!r}; it declares "
                         f"{sorted(payload['roles'])}")
    if prompt_file_sha256 is not None and \
            payload["prompt_file_sha256"] != prompt_file_sha256:
        raise SystemExit(
            f"the split was declared against a prompt file with digest "
            f"{payload['prompt_file_sha256'][:12]}..., but the run used "
            f"{prompt_file_sha256[:12]}...; the indices do not name the same prompts")
    return set(int(index) for index in payload["roles"][role])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "resources/sp_cross_schedules/parti_spx_splits.v1.json")
    args = parser.parse_args(argv)

    raw = args.prompt_file.read_bytes()
    prompts = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    roles = build_split(len(prompts), args.seed)
    assert sorted(index for role in roles.values() for index in role) == \
        list(range(len(prompts))), "roles must partition the prompt list"

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "prompt_file": str(args.prompt_file.relative_to(ROOT)),
        "prompt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "prompt_count": len(prompts),
        "seed": args.seed,
        "rule": "one seeded shuffle of range(prompt_count), equal thirds in shuffle "
                "order, each role re-sorted ascending",
        "declared_before": "any discovery-split path extraction; the pooled "
                           "parti_full extractions predate this file and are "
                           "screening-only (plan section 3)",
        "roles": roles,
        "counts": {role: len(indices) for role, indices in roles.items()},
    }
    payload["manifest_sha256"] = hash_payload(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{args.out}: " + ", ".join(f"{role}={len(indices)}"
                                      for role, indices in roles.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
