#!/usr/bin/env python3
"""Merge the video-matrix T1 trajectory records of one backbone into one table,
and (``--inventory``) run the plan's P0 checks over the three tiers.

docs/video_full_trajectory_plan_zh.md sections 2, 5.2 and 6 (P0). Reads
``$DATA/<T>/matrix/{cells,references,cells_t3_rand50,references_t3}/*/traj_*.json``
plus the sibling ``decisions_*.json``; writes

    <out>/t1_merged.jsonl   one row per generation (references first, then cells)
    <out>/t1_index.json     sigma grid, directory counts, field list, mode map
    <out>/p0_inventory.md   (--inventory) the P0 checks, pass/fail with numbers
    <out>/p0_inventory.json

Row normalisation (plan section 2.1/2.3): ``mode`` is the canonical method
name (HYV ``*_exact`` runner names mapped through
``hunyuan_video.matrix_config.RUNNER_MODE_METHOD``; Wan is already canonical),
the runner name is kept in ``mode_raw``; ``dataset`` and ``base_seed`` come
from the directory name (reference records carry ``dataset=None`` and only the
per-video seed); ``actions`` is the 50-character 0/1 string of the decisions
file (``'1'`` = cache, the ``density_form_test.gaps_of`` convention) and
``n_cached`` its count; every cell row carries ``ref_source_dir`` and
``ref_z_T_match`` for its same-(dataset, base_seed, prompt_idx) reference.
The per-backbone constants (``sigmas``, ``schema``, ``model``, ``latent_shape``,
``frame_segments``) move to the index, everything else in the T1 record is
kept as-is.

Only numpy/json/multiprocessing; ``torch`` is imported lazily for the one
inventory check that has to open the 60 stored ``latents_*.pt`` files.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hunyuan_video.matrix_config as _mc  # noqa: E402
from hunyuan_video.matrix_config import RUNNER_MODE_METHOD  # noqa: E402

REPO_ROOT = Path(_mc.__file__).resolve().parents[1]  # wherever the imported repo lives

CANONICAL_METHODS = ("seacache", "teacache", "sencache", "dicache", "budcache",
                     "meancache", "taylorseer_o1", "hicache_o2", "l2p")
FIXED_TABLE_METHODS = ("budcache", "meancache", "taylorseer_o1", "hicache_o2", "l2p")
DYNAMIC_METHODS = ("seacache", "teacache", "sencache", "dicache")
BUDGET_K = {"K29": 29, "K37": 37, "K41": 41}
NUM_STEPS = 50

# T1 fields and their expected lengths (None = scalar/dict/string)
T1_FIELDS: dict[str, int | None] = {
    "schema": None, "model": None, "mode": None, "dataset": None, "budget": None,
    "seed": None, "prompt_idx": None, "prompt_id": None, "matrix_config_sha256": None,
    "num_steps": None, "latent_shape": 5, "d": None, "z_T_dtype": None,
    "path_dtype": None, "z_T_sha256": None, "sigmas": 51, "frame_files": None,
    "frame_segments": None, "latent_file": None,
    "chord_len": None, "path_len": None, "max_dev_ratio": None, "straightness": None,
    "d_perp": 51, "spacing": 50, "magnitude": 51, "turn_angle_deg": 49,
    "turn_angle_w5_deg": 41, "turn_angle_w7_deg": 37, "second_diff_norm": 49,
    "velocity_norm": 50, "pca_evr": 5, "perp_var_total": None,
    "recon_err_1d": None, "recon_err_rel_1d": None, "recon_err_2d": None,
    "recon_err_rel_2d": None, "recon_err_3d": None, "recon_err_rel_3d": None,
    "update_chord_share": None, "update_in_position_plane": None, "update_own_evr": 5,
}
# fields that are constant per backbone and move into t1_index.json
INDEX_ONLY = ("schema", "model", "sigmas", "latent_shape", "frame_segments")

# plan section 2.4 sigma table (4 dp) and dsigma extremes, checked in --inventory
PLAN_SIGMA = {
    "hunyuan_video": {0: 1.0000, 10: 0.9655, 20: 0.9130, 30: 0.8235, 40: 0.6364,
                      45: 0.4375, 48: 0.2258, 49: 0.1250, 50: 0.0},
    "wan21": {0: 0.9998, 10: 0.9522, 20: 0.8821, 30: 0.7689, 40: 0.5552,
              45: 0.3569, 48: 0.1723, 49: 0.0925, 50: 0.0},
}
PLAN_DSIGMA = {"hunyuan_video": (0.0029, 0.1250), "wan21": (0.0041, 0.0925)}
# docs/video_full_results_report_zh.md section 5.1: the structural K exceptions
EXCEPTIONS = {
    "hunyuan_video": [("sencache", "K37", 36), ("sencache", "K41", 36)],
    "wan21": [("seacache", "K41", 40), ("teacache", "K37", 36)],
}
# Tier design after blocks A/B/C and the T3 extension
# (docs/video_cached_trajectory_t3_extension_plan_zh.md section 3.1(c)):
#   T1 = 126,333 cell rows (124,983 matrix + 1,350 cells_t3_rand50)
#      +   5,349 reference rows (4,629 matrix wave + 6 streams x 120 re-runs)
#   T2 = one frame set per reference generation = 5,349
#   T3 = 60 original + 240 block A + 480 block C + 1,350 extension = 2,130
#   dirs: 162 matrix cells + 162 cells_t3_rand50 = 324; 6 references + 6 references_t3
# The block-B path layer (cells_t3/, 54 dirs x 10) is superseded by the 50-pair
# sample and is not scanned; the extension's directories are the path layer.
EXPECTED = {"t1_total": 131682, "t2_total": 5349, "t3_total": 2130,
            "cells": 324, "refs": 12, "penguin599": 599, "vbench944": 944}

#: The extension's sample table: which manifest indices each cells_t3_rand50
#: directory holds. Its per-directory count is 5-12, not one constant, so the
#: inventory reads it rather than a literal.
T3_SAMPLE_TABLE = REPO_ROOT / "resources/video_full_trajectory/t3_extension_samples.v1.json"


def t3_sample_counts(backbone: str) -> dict[str, int]:
    payload = json.loads(T3_SAMPLE_TABLE.read_text(encoding="utf-8"))
    return {name: len(indices)
            for name, indices in payload["by_directory"][backbone].items()}


# ---------------------------------------------------------------------------
# directory naming
# ---------------------------------------------------------------------------


def parse_dir(name: str, kind: str) -> dict[str, Any]:
    """``cells/<mode>_<dataset>_<K>_s<seed>`` or ``references/<dataset>_s<seed>``.
    Split from the right: HYV runner names contain underscores."""
    parts = name.split("_")
    m = re.fullmatch(r"s(\d+)", parts[-1])
    if m is None:
        raise ValueError(f"{name}: no trailing _s<seed>")
    base_seed = int(m.group(1))
    if kind == "reference":
        return {"mode_raw": "original", "mode": "original", "dataset": "_".join(parts[:-1]),
                "budget": None, "base_seed": base_seed}
    budget = parts[-2]
    if budget not in BUDGET_K:
        raise ValueError(f"{name}: budget {budget!r} not in {sorted(BUDGET_K)}")
    mode_raw = "_".join(parts[:-3])
    mode = RUNNER_MODE_METHOD.get(mode_raw, mode_raw)
    return {"mode_raw": mode_raw, "mode": mode, "dataset": parts[-3],
            "budget": budget, "base_seed": base_seed}


def list_dirs(root: Path) -> list[tuple[str, Path]]:
    out = []
    for sub, kind in (("references", "reference"), ("references_t3", "reference"),
                      ("cells", "cell"), ("cells_t3_rand50", "cell")):
        base = root / sub
        if base.is_dir():
            out.extend((kind, p) for p in sorted(base.iterdir()) if p.is_dir())
    return out


# ---------------------------------------------------------------------------
# per-file worker
# ---------------------------------------------------------------------------


def _actions_of(decisions_path: Path) -> tuple[str | None, int | None]:
    if not decisions_path.is_file():
        return None, None
    with open(decisions_path, encoding="utf-8") as fh:
        d = json.load(fh)
    bits = "".join("1" if r.get("action") == "cache" else "0" for r in d["records"])
    return bits, bits.count("1")


def _read_one(args: tuple[str, str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Returns (jsonl line, small summary). Runs in a worker process."""
    traj_path, kind, meta = args
    tp = Path(traj_path)
    with open(tp, encoding="utf-8") as fh:
        rec = json.load(fh)
    missing = [k for k in T1_FIELDS if k not in rec]
    bad_len = [k for k, n in T1_FIELDS.items()
               if n is not None and k in rec and len(rec[k]) != n]
    stem = tp.name[len("traj_"):-len(".json")]
    actions, n_cached = _actions_of(tp.parent / f"decisions_{stem}.json")
    row: dict[str, Any] = {k: v for k, v in rec.items() if k not in INDEX_ONLY}
    row["kind"] = kind
    row["mode_raw"] = rec.get("mode")
    row["mode"] = meta["mode"] if kind == "cell" else "original"
    row["dataset_raw"] = rec.get("dataset")
    row["dataset"] = meta["dataset"]
    row["base_seed"] = meta["base_seed"]
    row["budget"] = meta["budget"]
    row["source_dir"] = f"{tp.parent.parent.name}/{tp.parent.name}"
    row["actions"] = actions
    row["n_cached"] = n_cached
    frame_files = rec.get("frame_files") or {}
    summary = {
        "prompt_idx": rec.get("prompt_idx"), "seed": rec.get("seed"),
        "z_T_sha256": rec.get("z_T_sha256"), "z_T_dtype": rec.get("z_T_dtype"),
        "path_dtype": rec.get("path_dtype"), "sigmas": tuple(rec.get("sigmas") or ()),
        "mode_rec": rec.get("mode"), "mode_dir": meta["mode_raw"],
        "dataset_rec": rec.get("dataset"), "budget_rec": rec.get("budget"),
        "matrix_config_sha256": rec.get("matrix_config_sha256"),
        "d": rec.get("d"), "latent_shape": tuple(rec.get("latent_shape") or ()),
        "missing": missing, "bad_len": bad_len,
        "n_actions": None if actions is None else len(actions), "n_cached": n_cached,
        "actions": actions,
        "chord_len": rec.get("chord_len"), "path_len": rec.get("path_len"),
        "spacing": rec.get("spacing"),
        "latent_file": rec.get("latent_file"),
        "latent_exists": bool(rec.get("latent_file")) and (tp.parent / rec["latent_file"]).is_file(),
        "n_frames": len(frame_files),
        "frames_exist": all((tp.parent / f).is_file() for f in frame_files.values()),
    }
    return json.dumps(row, ensure_ascii=False), summary


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def merge(root: Path, out: Path, workers: int, limit: int | None
          ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Write t1_merged.jsonl; return per-directory summaries and the index."""
    out.mkdir(parents=True, exist_ok=True)
    dirs = list_dirs(root)
    if not dirs:
        raise SystemExit(f"no cells/ or references/ under {root}")
    summaries: dict[str, list[dict[str, Any]]] = {}
    ref_index: dict[tuple[str, int, int], tuple[str, str]] = {}
    index: dict[str, Any] = {"root": str(root), "dirs": {}, "fields": list(T1_FIELDS),
                             "mode_map": dict(RUNNER_MODE_METHOD)}
    n_rows = 0
    with open(out / "t1_merged.jsonl", "w", encoding="utf-8") as fh, Pool(workers) as pool:
        for kind, d in dirs:
            meta = parse_dir(d.name, kind)
            files = sorted(d.glob("traj_*.json"))
            if limit is not None:
                files = files[:limit]
            key = f"{d.parent.name}/{d.name}"
            n_t2 = len(list(d.glob("frame_*.npy")))
            n_t3 = len(list(d.glob("latents_*.pt")))
            n_dec = len(list(d.glob("decisions_*.json")))
            index["dirs"][key] = {**meta, "kind": kind, "n_t1": len(files), "n_t2": n_t2,
                                  "n_t3": n_t3, "n_decisions": n_dec}
            rows: list[dict[str, Any]] = []
            for line, summ in pool.imap(_read_one, [(str(f), kind, meta) for f in files],
                                        chunksize=32):
                if kind == "cell":
                    ref = ref_index.get((meta["dataset"], meta["base_seed"], summ["prompt_idx"]))
                    row = json.loads(line)
                    row["ref_source_dir"] = ref[0] if ref else None
                    row["ref_z_T_match"] = (ref[1] == summ["z_T_sha256"]) if ref else None
                    summ["ref_source_dir"] = row["ref_source_dir"]
                    summ["ref_z_T_match"] = row["ref_z_T_match"]
                    line = json.dumps(row, ensure_ascii=False)
                else:
                    ref_index[(meta["dataset"], meta["base_seed"], summ["prompt_idx"])] = (
                        key, summ["z_T_sha256"])
                fh.write(line + "\n")
                n_rows += 1
                summ["dir"] = key
                summ.update(meta)
                summ["kind"] = kind
                rows.append(summ)
            summaries[key] = rows
            print(f"  {key}: {len(rows)} T1, {n_t2} T2, {n_t3} T3", flush=True)
    index["n_rows"] = n_rows
    grids = {s["sigmas"] for rows in summaries.values() for s in rows}
    index["sigma_grids"] = [list(g) for g in grids]
    index["n_sigma_grids"] = len(grids)
    (out / "t1_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return summaries, index


# ---------------------------------------------------------------------------
# inventory (P0)
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 4) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _t3_vs_t1(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Recompute chord/path/spacing from the stored bf16 T3 and compare with T1."""
    import torch  # lazy: only this check opens .pt files
    chord_rel, path_rel, spacing_rel = [], [], []
    for s in rows:
        if not s.get("latent_file"):
            continue
        p = root / s["dir"] / s["latent_file"]
        Z = torch.load(p, map_location="cpu", weights_only=True)
        Z = Z.to(torch.float32).numpy().reshape(Z.shape[0], -1).astype(np.float64)
        chord = float(np.linalg.norm(Z[-1] - Z[0]))
        sp = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        chord_rel.append((chord - s["chord_len"]) / s["chord_len"])
        path_rel.append((float(sp.sum()) - s["path_len"]) / s["path_len"])
        spacing_rel.append((sp - np.asarray(s["spacing"])) / np.asarray(s["spacing"]))
    if not chord_rel:
        return {"n": 0}
    sp = np.array(spacing_rel)
    return {
        "n": len(chord_rel),
        "chord_rel_max_abs": float(np.max(np.abs(chord_rel))),
        "chord_rel_median": float(np.median(chord_rel)),
        "path_rel_median": float(np.median(path_rel)),
        "path_rel_min": float(np.min(path_rel)), "path_rel_max": float(np.max(path_rel)),
        "spacing_rel_median_by_step": np.median(sp, axis=0).tolist(),
        "spacing_rel_median_steps_0_8": float(np.median(sp[:, :9])),
        "spacing_rel_median_steps_9_49": float(np.median(sp[:, 9:])),
    }


def inventory(backbone: str, root: Path, out: Path, summaries: dict[str, list[dict[str, Any]]],
              index: dict[str, Any], config_path: Path | None) -> None:
    lines: list[str] = [f"# P0 inventory — {backbone}", "",
                        f"root `{root}`; {index['n_rows']} T1 rows merged into `t1_merged.jsonl`.", ""]
    res: dict[str, Any] = {"backbone": backbone, "root": str(root)}
    verdicts: list[tuple[str, str, str]] = []

    def verdict(name: str, ok: bool | None, detail: str) -> None:
        verdicts.append((name, "PASS" if ok else ("REPORT" if ok is None else "FAIL"), detail))

    all_rows = [s for rows in summaries.values() for s in rows]
    refs = [s for s in all_rows if s["kind"] == "reference"]
    cells = [s for s in all_rows if s["kind"] == "cell"]

    # 1. counts ---------------------------------------------------------------
    n_t1 = len(all_rows)
    n_t2 = sum(v["n_t2"] for v in index["dirs"].values())
    n_t3 = sum(v["n_t3"] for v in index["dirs"].values())
    n_cell_dirs = sum(1 for v in index["dirs"].values() if v["kind"] == "cell")
    n_ref_dirs = sum(1 for v in index["dirs"].values() if v["kind"] == "reference")
    sample_counts = t3_sample_counts(backbone)

    def _dir_expected(name: str, dataset: str) -> int:
        # matrix cells and matrix-wave references hold the full prompt set; the
        # T3 re-run streams hold idx 0-119; each extension directory holds the
        # 5-12 indices the frozen sample table drew for it
        if name.startswith("references_t3/"):
            return 120
        if name.startswith("cells_t3_rand50/"):
            return sample_counts[name.split("/", 1)[1]]
        return EXPECTED[dataset]

    per_dir_ok = all(
        v["n_t1"] == _dir_expected(name, v["dataset"]) == v["n_decisions"]
        for name, v in index["dirs"].items() if v["dataset"] in EXPECTED)
    lines += ["## 1. Tier counts", "",
              "| tier | found | expected |", "|---|---:|---:|",
              f"| T1 traj_*.json | {n_t1} | {EXPECTED['t1_total']} |",
              f"| T2 frame_*.npy | {n_t2} | {EXPECTED['t2_total']} |",
              f"| T3 latents_*.pt | {n_t3} | {EXPECTED['t3_total']} |",
              f"| cell dirs | {n_cell_dirs} | {EXPECTED['cells']} |",
              f"| reference dirs | {n_ref_dirs} | {EXPECTED['refs']} |", ""]
    t3_dirs = {k: v["n_t3"] for k, v in index["dirs"].items() if v["n_t3"]}
    t2_dirs = {k: v["n_t2"] for k, v in index["dirs"].items() if v["n_t2"]}
    lines += [f"T3 by dir: {t3_dirs}", "", f"T2 by dir: {t2_dirs}", "",
              "Per-directory T1 == decisions == dataset size (599/944): "
              + ("all dirs" if per_dir_ok else "MISMATCH (first 10) — " + str(
                  [(k, v['n_t1'], v['n_decisions']) for k, v in index['dirs'].items()
                   if v['dataset'] in EXPECTED
                   and not (v['n_t1'] == _dir_expected(k, v['dataset'])
                            == v['n_decisions'])][:10])), ""]
    verdict("T1/T2/T3 counts", n_t1 == EXPECTED["t1_total"] and n_t2 == EXPECTED["t2_total"]
            and n_t3 == EXPECTED["t3_total"] and per_dir_ok,
            f"T1={n_t1} T2={n_t2} T3={n_t3} cells={n_cell_dirs} refs={n_ref_dirs}")
    res["counts"] = {"t1": n_t1, "t2": n_t2, "t3": n_t3, "cell_dirs": n_cell_dirs,
                     "ref_dirs": n_ref_dirs, "t3_by_dir": t3_dirs}
    # T2 / T3 existence as referenced by the T1 records
    ref_frames_ok = sum(1 for s in refs if s["n_frames"] == 1 and s["frames_exist"])
    lat_ok = sum(1 for s in all_rows if s["latent_file"] and s["latent_exists"])
    lat_named = sum(1 for s in all_rows if s["latent_file"])
    cell_frames = sum(s["n_frames"] for s in cells)
    lines += [f"References whose one frame file exists: {ref_frames_ok}/{len(refs)}; "
              f"records naming a latent_file: {lat_named}, of which the file exists: {lat_ok}; "
              f"cell records naming frames: {cell_frames} (must be 0).", ""]
    verdict("T2/T3 files referenced by T1 exist", ref_frames_ok == len(refs) and lat_ok == lat_named
            and cell_frames == 0, f"frames {ref_frames_ok}/{len(refs)}, latents {lat_ok}/{lat_named}")

    # 2. field completeness ---------------------------------------------------
    miss = collections.Counter(k for s in all_rows for k in s["missing"])
    badlen = collections.Counter(k for s in all_rows for k in s["bad_len"])
    n_complete = sum(1 for s in all_rows if not s["missing"] and not s["bad_len"])
    lines += ["## 2. Field completeness", "",
              f"{n_complete}/{n_t1} records carry all {len(T1_FIELDS)} fields with the expected "
              f"lengths ({100.0 * n_complete / max(n_t1, 1):.2f} %).",
              f"Missing fields: {dict(miss) or 'none'}. Wrong-length fields: {dict(badlen) or 'none'}.", ""]
    dt = collections.Counter((s["kind"], s["z_T_dtype"], s["path_dtype"]) for s in all_rows)
    lines += [f"(kind, z_T_dtype, path_dtype) histogram: {dict(dt)}", ""]
    d_vals = collections.Counter((s["d"], s["latent_shape"]) for s in all_rows)
    lines += [f"(d, latent_shape) histogram: { {str(k): v for k, v in d_vals.items()} }", ""]
    verdict("field completeness 100 %", n_complete == n_t1, f"{n_complete}/{n_t1}")
    res["fields"] = {"complete": n_complete, "missing": dict(miss), "bad_len": dict(badlen),
                     "dtypes": {str(k): v for k, v in dt.items()}}

    # 3. mode normalisation ---------------------------------------------------
    cell_modes = collections.Counter(s["mode"] for s in cells)
    raw_modes = collections.Counter((s["mode_dir"], s["mode_rec"]) for s in cells)
    dir_rec_agree = all(a == b for a, b in raw_modes)
    ref_modes = collections.Counter(s["mode_rec"] for s in refs)
    lines += ["## 3. Method names", "",
              f"Canonical `mode` over cell rows: {dict(cell_modes)}",
              f"(dir name, record `mode`) pairs: {dict(raw_modes)}; dir == record for all: {dir_rec_agree}",
              f"Reference record `mode`: {dict(ref_modes)}", ""]
    ok3 = set(cell_modes) == set(CANONICAL_METHODS) and dir_rec_agree and set(ref_modes) == {"original"}
    verdict("9 canonical method names", ok3, f"{sorted(cell_modes)}")
    res["modes"] = {"canonical": dict(cell_modes), "raw": {f"{a}|{b}": n for (a, b), n in raw_modes.items()}}
    # budget/dataset agreement between dir name and record
    bd = sum(1 for s in cells if s["budget_rec"] == s["budget"] and s["dataset_rec"] == s["dataset"])
    lines += [f"Cell records whose `budget`/`dataset` equal the directory's: {bd}/{len(cells)}", ""]
    verdict("cell budget/dataset == dir", bd == len(cells), f"{bd}/{len(cells)}")

    # 4. sigma grid -------------------------------------------------------------
    grids = index["sigma_grids"]
    lines += ["## 4. Sigma grid", "", f"Distinct `sigmas` tuples over all {n_t1} records: {len(grids)}", ""]
    ok4 = len(grids) == 1
    if grids:
        g = np.asarray(grids[0], dtype=np.float64)
        ds = np.abs(np.diff(g))
        plan = PLAN_SIGMA[backbone]
        table = ["| n | " + " | ".join(str(n) for n in plan) + " |",
                 "|---|" + "---|" * len(plan),
                 "| measured | " + " | ".join(f"{g[n]:.4f}" for n in plan) + " |",
                 "| plan §2.4 | " + " | ".join(f"{plan[n]:.4f}" for n in plan) + " |"]
        sig_ok = all(abs(round(float(g[n]), 4) - plan[n]) <= 5e-5 for n in plan)
        mono = bool(np.all(np.diff(ds) > 0))
        pmin, pmax = PLAN_DSIGMA[backbone]
        ds_ok = (abs(round(float(ds.min()), 4) - pmin) <= 5e-5 and abs(round(float(ds.max()), 4) - pmax) <= 5e-5
                 and int(ds.argmin()) == 0 and int(ds.argmax()) == 49)
        lines += table + ["",
                          f"Δσ: min {ds.min():.4f} at step {int(ds.argmin())}, max {ds.max():.4f} at step "
                          f"{int(ds.argmax())}, max/min {ds.max() / ds.min():.1f}, last step {ds[-1]:.4f}, "
                          f"strictly increasing over steps 0..49: {mono}"
                          + ("" if mono else f" (first decrease at step {int(np.argmax(np.diff(ds) <= 0)) + 1})"),
                          f"Δσ steps 47/48/49: {ds[47]:.4f} / {ds[48]:.4f} / {ds[49]:.4f}",
                          f"σ length {len(g)}, σ[0]={g[0]:.6f}, σ[50]={g[-1]:.6f}", ""]
        ok4 = ok4 and sig_ok and ds_ok
        res["sigma"] = {"n_grids": len(grids), "sigmas": g.tolist(), "table_match": sig_ok,
                        "dsigma_min": float(ds.min()), "dsigma_max": float(ds.max()),
                        "dsigma_monotone": mono, "dsigma_extremes_match": ds_ok}
        verdict("sigma grid unique + §2.4 values + Δσ extremes/monotone", ok4,
                f"{len(grids)} grid(s); table match {sig_ok}; Δσ min/max {ds.min():.4f}/{ds.max():.4f} "
                f"monotone {mono}")
    else:
        verdict("sigma grid", False, "no records")

    # 5. reference dataset fill ---------------------------------------------------
    raw_none = sum(1 for s in refs if s["dataset_rec"] is None)
    filled = sum(1 for s in refs if s["dataset"] in ("penguin599", "vbench944"))
    per_stream = collections.Counter((s["dataset"], s["base_seed"]) for s in refs)
    seed_rule = sum(1 for s in refs if s["seed"] == s["base_seed"] + s["prompt_idx"])
    seed_rule_c = sum(1 for s in cells if s["seed"] == s["base_seed"] + s["prompt_idx"])
    lines += ["## 5. Reference `dataset` and seed rule", "",
              f"Reference records with `dataset=None` in the file: {raw_none}/{len(refs)}; "
              f"filled from the directory name: {filled}/{len(refs)}: {dict(per_stream)}",
              f"Per-video `seed == base_seed + prompt_idx`: refs {seed_rule}/{len(refs)}, cells {seed_rule_c}/{len(cells)}", ""]
    verdict("reference dataset filled from dir", filled == len(refs), f"{filled}/{len(refs)} (raw None: {raw_none})")
    verdict("seed = base + prompt_idx", seed_rule == len(refs) and seed_rule_c == len(cells),
            f"refs {seed_rule}/{len(refs)}, cells {seed_rule_c}/{len(cells)}")

    # 6. cell <-> ref z_T pairing -------------------------------------------------
    n_ref_found = sum(1 for s in cells if s.get("ref_source_dir"))
    n_match = sum(1 for s in cells if s.get("ref_z_T_match"))
    by_mode_mis = collections.Counter(s["mode"] for s in cells if s.get("ref_z_T_match") is False)
    lines += ["## 6. cell ↔ reference z_T_sha256", "",
              f"Cells with a same-(dataset, base_seed, prompt_idx) reference: {n_ref_found}/{len(cells)}; "
              f"z_T_sha256 equal: {n_match}/{len(cells)} ({100.0 * n_match / max(len(cells), 1):.3f} %)"
              + (f"; mismatches by method: {dict(by_mode_mis)}" if by_mode_mis else ""), ""]
    verdict("cell↔ref z_T_sha256 = 100 %", n_match == len(cells) and n_ref_found == len(cells),
            f"{n_match}/{len(cells)}")
    res["z_T_pairing"] = {"found": n_ref_found, "match": n_match, "cells": len(cells)}

    # 7. same-seed cross-stream ------------------------------------------------------
    by_seed: dict[int, set[str]] = collections.defaultdict(set)
    by_seed_n: collections.Counter = collections.Counter()
    for s in refs:
        by_seed[s["seed"]].add(s["z_T_sha256"])
        by_seed_n[s["seed"]] += 1
    multi = [seed for seed, n in by_seed_n.items() if n >= 2]
    consistent = sum(1 for seed in multi if len(by_seed[seed]) == 1)
    n_distinct = len({s["z_T_sha256"] for s in refs})
    size_hist = collections.Counter(by_seed_n.values())
    sha_groups = collections.Counter(s["z_T_sha256"] for s in refs)
    sha_size_hist = collections.Counter(sha_groups.values())
    lines += ["## 7. Same seed across streams", "",
              f"Reference per-video seeds: {len(by_seed_n)} distinct; seeds seen in ≥2 streams: {len(multi)}; "
              f"of those, all records share one z_T_sha256: {consistent}/{len(multi)}",
              f"Distinct z_T_sha256 among references: {n_distinct} (plan ≈946); "
              f"seed-group size histogram {dict(sorted(size_hist.items()))}; "
              f"z_T-group size histogram {dict(sorted(sha_size_hist.items()))}",
              f"seed range: {min(by_seed_n)}–{max(by_seed_n)}", ""]
    verdict("same seed ⇒ same z_T across streams", consistent == len(multi),
            f"{consistent}/{len(multi)} multi-stream seeds; {n_distinct} distinct z_T")
    res["same_seed"] = {"multi": len(multi), "consistent": consistent, "distinct_zT": n_distinct,
                        "sha_group_sizes": dict(sha_size_hist)}
    # sanity: distinct z_T == distinct seeds only if the same seed never gives two z_T
    lines += [f"Distinct seeds {len(by_seed_n)} vs distinct z_T {n_distinct} "
              f"({'equal' if len(by_seed_n) == n_distinct else 'DIFFER'})", ""]

    # 8. T3 vs T1 -----------------------------------------------------------------
    t3rows = [s for s in refs if s["latent_file"] and s["latent_exists"]]
    lines += ["## 8. Stored T3 (bf16) vs T1 (in-flight float32)", ""]
    if t3rows:
        cmp = _t3_vs_t1(t3rows, root)
        sp = cmp["spacing_rel_median_by_step"]
        lines += [f"{cmp['n']} paths. chord relative diff: max |Δ| {cmp['chord_rel_max_abs']:.2e}, "
                  f"median {cmp['chord_rel_median']:+.2e} (criterion ≤ 1e-3).",
                  f"path_len relative diff (reported, not thresholded): median {cmp['path_rel_median']:+.2e}, "
                  f"range [{cmp['path_rel_min']:+.2e}, {cmp['path_rel_max']:+.2e}].",
                  f"spacing relative diff, median over paths: steps 0–8 {cmp['spacing_rel_median_steps_0_8']:+.3f}, "
                  f"steps 9–49 {cmp['spacing_rel_median_steps_9_49']:+.2e}; per step "
                  + ", ".join(f"s{n}={sp[n]:+.3f}" for n in (0, 1, 2, 4, 8, 10, 20, 30, 40, 49)), ""]
        verdict("T3 vs T1 chord rel diff ≤ 1e-3", cmp["chord_rel_max_abs"] <= 1e-3,
                f"max {cmp['chord_rel_max_abs']:.2e} over {cmp['n']}")
        verdict("T3 vs T1 path_len/spacing inflation (report)", None,
                f"path {cmp['path_rel_median']:+.2e}; spacing s0..8 {cmp['spacing_rel_median_steps_0_8']:+.3f}")
        res["t3_vs_t1"] = cmp
    else:
        verdict("T3 vs T1", False, "no latents found")

    # 9./10. actions and n_cached -----------------------------------------------------
    n_act50 = sum(1 for s in cells if s["n_actions"] == NUM_STEPS)
    ref_act = collections.Counter(s["n_cached"] for s in refs)
    lines += ["## 9. `actions` (decisions) length", "",
              f"Cell rows with a 50-step action string: {n_act50}/{len(cells)}; "
              f"reference n_cached histogram (must be all 0): {dict(ref_act)}", ""]
    verdict("actions length 50 (all cells), refs all-full", n_act50 == len(cells) and set(ref_act) <= {0},
            f"{n_act50}/{len(cells)}")

    lines += ["## 10. n_cached vs K", "", "Fixed-table methods (per-video n_cached must equal K):", "",
              "| method | K | n videos | n_cached == K | n_cached ≠ K |", "|---|---|---:|---:|---:|"]
    fixed_ok = True
    fixed_res = {}
    for m in FIXED_TABLE_METHODS:
        for K, k in BUDGET_K.items():
            rows = [s for s in cells if s["mode"] == m and s["budget"] == K]
            eq = sum(1 for s in rows if s["n_cached"] == k)
            fixed_ok &= eq == len(rows) and len(rows) > 0
            fixed_res[f"{m}_{K}"] = (eq, len(rows))
            lines.append(f"| {m} | {K} | {len(rows)} | {eq} | {len(rows) - eq} |")
    # fixed tables: the action string must equal the frozen table
    tbl_note = ""
    if config_path is not None and config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        tables = {name: "".join("1" if i in set(t["cache_steps"]) else "0" for i in range(NUM_STEPS))
                  for name, t in cfg["schedule_tables"].items()}
        sched = cfg["schedules"]
        eqs = []
        for m in FIXED_TABLE_METHODS:
            for K in BUDGET_K:
                want = tables[sched[m][K]]
                rows = [s for s in cells if s["mode"] == m and s["budget"] == K]
                eqs.append((m, K, sum(1 for s in rows if s["actions"] == want), len(rows)))
        bad = [(m, K, e, n) for m, K, e, n in eqs if e != n]
        tbl_note = (f"Fixed-table action strings equal the frozen `schedule_tables` entry: "
                    f"{sum(e for *_, e, _ in eqs)}/{sum(n for *_, n in eqs)}"
                    + (f"; mismatches {bad}" if bad else ""))
        fixed_ok &= not bad
    lines += ["", tbl_note, "", "Dynamic gates (per-cell n_cached distribution):", "",
              "| method | dataset | K | seed | n | mean | min | max | frac == K | distinct paths | modal path count |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    verdict("fixed-table n_cached == K (5 methods × 3 K)", fixed_ok,
            "; ".join(f"{k}:{e}/{n}" for k, (e, n) in fixed_res.items() if e != n or n == 0) or "all equal")
    dyn_res = {}
    for m in DYNAMIC_METHODS:
        for K, k in BUDGET_K.items():
            for ds in ("penguin599", "vbench944"):
                for seed in sorted({s["base_seed"] for s in cells if s["dataset"] == ds}):
                    rows = [s for s in cells if s["mode"] == m and s["budget"] == K
                            and s["dataset"] == ds and s["base_seed"] == seed]
                    if not rows:
                        continue
                    nc = np.array([s["n_cached"] for s in rows], dtype=float)
                    paths = collections.Counter(s["actions"] for s in rows)
                    modal = paths.most_common(1)[0][1]
                    dyn_res[f"{m}_{ds}_{K}_s{seed}"] = {"n": len(rows), "mean": float(nc.mean()),
                                                        "min": int(nc.min()), "max": int(nc.max()),
                                                        "distinct": len(paths), "modal": modal}
                    lines.append(f"| {m} | {ds} | {K} | {seed} | {len(rows)} | {nc.mean():.2f} | {int(nc.min())} | "
                                 f"{int(nc.max())} | {float((nc == k).mean()):.3f} | {len(paths)} | {modal} |")
    verdict("dynamic-gate n_cached distributions", None, f"{len(dyn_res)} cells reported")
    res["n_cached"] = {"fixed": fixed_res, "dynamic": dyn_res}

    # the three structural exceptions
    lines += ["", "Structural exceptions recorded in docs/video_full_results_report_zh.md section 5.1:", ""]
    for m, K, kexp in EXCEPTIONS[backbone]:
        rows = [s for s in cells if s["mode"] == m and s["budget"] == K]
        nc = np.array([s["n_cached"] for s in rows], dtype=float)
        frac = float((nc == kexp).mean()) if len(rows) else float("nan")
        lines.append(f"- {m} {K}: expected realised K = {kexp}: n_cached == {kexp} in {frac:.4f} of "
                     f"{len(rows)} videos (mean {nc.mean():.2f}, min {int(nc.min())}, max {int(nc.max())})")
        res.setdefault("exceptions", {})[f"{m}_{K}"] = {"expected": kexp, "frac": frac, "n": len(rows),
                                                          "mean": float(nc.mean())}
    if backbone == "hunyuan_video":
        # SenCache K37 == K41 as the same configuration: same action string per (ds, seed, idx)
        a37 = {(s["dataset"], s["base_seed"], s["prompt_idx"]): s["actions"]
               for s in cells if s["mode"] == "sencache" and s["budget"] == "K37"}
        a41 = {(s["dataset"], s["base_seed"], s["prompt_idx"]): s["actions"]
               for s in cells if s["mode"] == "sencache" and s["budget"] == "K41"}
        common = set(a37) & set(a41)
        same = sum(1 for key in common if a37[key] == a41[key])
        lines.append(f"- sencache K37 vs K41 identical action string per video: {same}/{len(common)}")
        res["exceptions"]["sencache_K37_eq_K41"] = {"same": same, "n": len(common)}
    lines.append("")

    # matrix_config sha
    shas = collections.Counter(s["matrix_config_sha256"] for s in cells)
    lines += [f"Cell `matrix_config_sha256` histogram: {dict(shas)}", ""]

    # verdict table -----------------------------------------------------------------
    head = ["## 0. Verdicts", "", "| check | verdict | numbers |", "|---|---|---|"]
    head += [f"| {n} | {v} | {d} |" for n, v, d in verdicts]
    head.append("")
    body = lines[:4] + head + lines[4:]
    (out / "p0_inventory.md").write_text("\n".join(body), encoding="utf-8")
    res["verdicts"] = verdicts
    (out / "p0_inventory.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("\n".join(head))
    print(f"wrote {out / 'p0_inventory.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(PLAN_SIGMA), required=True)
    ap.add_argument("--root", type=Path, required=True, help="$DATA/<T>/matrix")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/trajectory")
    ap.add_argument("--matrix_config", type=Path, default=None,
                    help="frozen config for the fixed-table check "
                         "(default resources/<T>/baseline_matrix_config.v1.json)")
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4))))
    ap.add_argument("--limit", type=int, default=None, help="first N records per dir (smoke)")
    ap.add_argument("--inventory", action="store_true", help="also run the P0 checks")
    args = ap.parse_args()
    out = args.out or (args.root / "trajectory")
    cfg = args.matrix_config or (REPO_ROOT / "resources" / args.backbone
                                 / "baseline_matrix_config.v1.json")
    if not cfg.is_file():
        raise SystemExit(f"frozen matrix config not found: {cfg}")
    summaries, index = merge(args.root, out, args.workers, args.limit)
    if args.inventory:
        inventory(args.backbone, args.root, out, summaries, index, cfg)


if __name__ == "__main__":
    main()
