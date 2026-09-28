"""Fixed registries: orders, families, colours, metric directions, path constants.

Every table and figure in this package follows the orders declared here
(plan section 1.1).  Nothing in this module performs IO.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# repository / output locations
# --------------------------------------------------------------------------

# analysis/full_results_local/registry.py -> repo root is three parents up.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root(override: str | os.PathLike | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    env = os.environ.get("FULL_RESULTS_LOCAL_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return DEFAULT_REPO_ROOT


# relative sub-paths (joined against repo_root())
REL_STAGE_E_RESULTS = "resources/cross_model_multiseed_stage_e_results"
REL_STAGE_E_PATHS = "resources/cross_model_multiseed_stage_e_native_paths"
REL_DDB_RESULTS = "resources/diffusiondb_clean10k_formal_baseline_results"
REL_OUT_ROOT = "resources/full_results_local_analysis"
REL_OUT_TABLES = REL_OUT_ROOT + "/tables"
REL_OUT_FIGURES = REL_OUT_ROOT + "/figures"


def stage_e_results_dir(root: Path) -> Path:
    return root / REL_STAGE_E_RESULTS


def stage_e_paths_dir(root: Path) -> Path:
    return root / REL_STAGE_E_PATHS


def ddb_results_dir(root: Path) -> Path:
    return root / REL_DDB_RESULTS


def tables_dir(root: Path) -> Path:
    return root / REL_OUT_TABLES


def figures_dir(root: Path) -> Path:
    return root / REL_OUT_FIGURES


# --------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------

N_STEPS = 50

MODELS = ("flux", "qwen")

DATASETS = ("drawbench_full", "parti_full", "geneval_style", "diffusiondb_clean10k")
STAGE_E_DATASETS = ("drawbench_full", "parti_full", "geneval_style")
DDB_DATASET = "diffusiondb_clean10k"
EQUAL_DATASET_LABEL = "three_datasets_equal"  # label used by equal_dataset_path_counts.tsv

DATASET_N_PROMPTS = {
    "drawbench_full": 200,
    "parti_full": 1632,
    "geneval_style": 553,
    "diffusiondb_clean10k": 10000,
}

TARGET_KS = (29, 37, 41)
TARGET_RATIOS = {29: 0.58, 37: 0.74, 41: 0.82}

METHODS = (
    # dynamic_native
    "seacache",
    "teacache",
    "sencache",
    "dicache",
    # shared_predictor (identical bitstring within a (model, K))
    "taylorseer_o1",
    "hicache_o2",
    "l2p",
    # offline fixed searchers
    "dpcache",
    "budcache",
    "meancache",
)

SEED_STREAMS = ("S0", "S1", "S2")

# The plan text says "base seed 41/42/43 for both the formal sets and DDB".
# Measured 2026-07-31: that holds for flux only; qwen uses 42/100042/200042 in
# both stage-E and DDB.  The mapping below is the measured one (schema deviation
# SD-SEED-MAP recorded by the loader).
BASE_SEED_BY_MODEL = {
    "flux": {"S0": 41, "S1": 42, "S2": 43},
    "qwen": {"S0": 42, "S1": 100042, "S2": 200042},
}
SEED_STREAM_BY_MODEL = {
    model: {seed: stream for stream, seed in mapping.items()}
    for model, mapping in BASE_SEED_BY_MODEL.items()
}

# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------

FAMILIES = ("dynamic_native", "shared_predictor", "offline_fixed")

METHOD_FAMILY = {
    "seacache": "dynamic_native",
    "teacache": "dynamic_native",
    "sencache": "dynamic_native",
    "dicache": "dynamic_native",
    "taylorseer_o1": "shared_predictor",
    "hicache_o2": "shared_predictor",
    "l2p": "shared_predictor",
    "dpcache": "offline_fixed",
    "budcache": "offline_fixed",
    "meancache": "offline_fixed",
}

FAMILY_MEMBERS = {
    fam: tuple(m for m in METHODS if METHOD_FAMILY[m] == fam) for fam in FAMILIES
}

DYNAMIC_METHODS = FAMILY_MEMBERS["dynamic_native"]
FIXED_METHODS = FAMILY_MEMBERS["shared_predictor"] + FAMILY_MEMBERS["offline_fixed"]

# `method_kind` as written in the source tables (not the same axis as family).
METHOD_KIND = {
    m: ("dynamic_native" if m in DYNAMIC_METHODS else "fixed_schedule") for m in METHODS
}

REFERENCE_METHOD = "seacache"  # paired_intervals.tsv reference

# --------------------------------------------------------------------------
# colours: one hue family per method family, lightness separates members
# --------------------------------------------------------------------------

FAMILY_COLORMAP = {
    "dynamic_native": "Blues",
    "shared_predictor": "Oranges",
    "offline_fixed": "Greens",
}

METHOD_COLOR = {
    # dynamic_native: blue ramp (4 steps)
    "seacache": "#08306b",
    "teacache": "#2171b5",
    "sencache": "#6baed6",
    "dicache": "#bdd7e7",
    # shared_predictor: orange ramp (3 steps)
    "taylorseer_o1": "#8c2d04",
    "hicache_o2": "#e6550d",
    "l2p": "#fdae6b",
    # offline_fixed: green ramp (3 steps)
    "dpcache": "#00441b",
    "budcache": "#41ab5d",
    "meancache": "#a1d99b",
}

FAMILY_COLOR = {
    "dynamic_native": "#2171b5",
    "shared_predictor": "#e6550d",
    "offline_fixed": "#41ab5d",
}

# cliff classes (M3) and their tile colours
CLIFF_CLASSES = (
    "non-monotone",
    "degrading-cliff",
    "degrading-accelerating",
    "degrading-steady",
    "improving/flat",
)
CLIFF_CLASS_COLOR = {
    "non-monotone": "#762a83",
    "degrading-cliff": "#b2182b",
    "degrading-accelerating": "#ef8a62",
    "degrading-steady": "#fddbc7",
    "improving/flat": "#4393c3",
}

DISCIPLINE_CLASSES = ("near-degenerate", "tight", "dispersed")
AGREEMENT_CLASSES = ("consistent-signed", "mixed", "both-null")

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

METRICS = ("psnr", "ssim", "lpips", "clip", "image_reward")

# +1 -> larger is better, -1 -> smaller is better
METRIC_DIRECTION = {
    "psnr": +1,
    "ssim": +1,
    "lpips": -1,
    "clip": +1,
    "image_reward": +1,
}

METRIC_LABEL = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "clip": "CLIP score",
    "image_reward": "ImageReward",
}


def harm(metric: str, value):
    """Harm transform (plan section 2 preamble): larger = worse, used only for the
    ratio-response / cliff analysis.  The main tables keep the original scale.

        h_PSNR = 10 ** (-PSNR / 10)
        h_SSIM = 1 - SSIM
        h_LPIPS = LPIPS
        h_CLIP = -CLIP            (absolute score negated, not a delta vs Full)
        h_IR   = -ImageReward     (idem)
    """
    if metric == "psnr":
        return 10.0 ** (-value / 10.0)
    if metric == "ssim":
        return 1.0 - value
    if metric == "lpips":
        return value
    if metric == "clip":
        return -value
    if metric == "image_reward":
        return -value
    raise KeyError(f"unknown metric: {metric!r}")


# --------------------------------------------------------------------------
# data layers (plan section 1.3)
# --------------------------------------------------------------------------

DATA_LAYERS = (
    "cell-agg",
    "prompt-quantile",
    "paired-ci",
    "path-support",
    "timing-bench",
    "timing-run",
)

SOURCE_PACKAGES = ("stage_e", "ddb")

# --------------------------------------------------------------------------
# ordering helpers
# --------------------------------------------------------------------------

ENVIRONMENTS = tuple(
    (model, dataset, k) for model in MODELS for dataset in DATASETS for k in TARGET_KS
)  # 24, fixed column order: model outer -> dataset -> K

MODEL_K_BLOCKS = tuple((model, k) for model in MODELS for k in TARGET_KS)  # 6

METHOD_PAIRS = tuple(
    (METHODS[i], METHODS[j])
    for i in range(len(METHODS))
    for j in range(i + 1, len(METHODS))
)  # 45


def method_order(method: str) -> int:
    return METHODS.index(method)


def env_order(model: str, dataset: str, target_k: int) -> int:
    return ENVIRONMENTS.index((model, dataset, int(target_k)))


def env_label(model: str, dataset: str, target_k: int) -> str:
    return f"{model}/{dataset}/K{int(target_k)}"


def sort_cells(df, extra_last: tuple[str, ...] = ()):
    """Return `df` sorted by the canonical (model, dataset, target_k, method,
    seed_stream) order followed by `extra_last` columns.  Deterministic, used by
    every module before writing a TSV."""
    import pandas as pd  # local import: registry stays import-light

    order_cols = []
    work = df.copy()
    for col, order in (
        ("model", MODELS),
        ("dataset", DATASETS),
        ("target_k", TARGET_KS),
        ("method", METHODS),
        ("method_a", METHODS),
        ("method_b", METHODS),
        ("seed_stream", SEED_STREAMS),
        ("metric", METRICS),
    ):
        if col in work.columns:
            key = f"__ord_{col}"
            work[key] = pd.Categorical(work[col], categories=list(order), ordered=True)
            order_cols.append(key)
    order_cols.extend([c for c in extra_last if c in work.columns])
    if not order_cols:
        return df
    work = work.sort_values(order_cols, kind="mergesort").drop(columns=[c for c in work.columns if c.startswith("__ord_")])
    return work.reset_index(drop=True)


# --------------------------------------------------------------------------
# reproducibility banner (plan section 4.2)
# --------------------------------------------------------------------------

BANNER_PREFIX = "# generated_by=analysis/full_results_local"
