#!/usr/bin/env bash
# Convert completed Qwen calibration shards into directly runnable assets.

set -euo pipefail
shopt -s nullglob

TRAIN_ROOT=""
HOLDOUT_ROOT=""
OUTPUT_DIR=""
CACHE_COUNTS="29,37,41"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_root) TRAIN_ROOT="$2"; shift 2;;
    --holdout_root) HOLDOUT_ROOT="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --cache_counts) CACHE_COUNTS="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    *) echo "[ERROR] unknown option: $1" >&2; exit 2;;
  esac
done

[[ -n "$TRAIN_ROOT" && -n "$HOLDOUT_ROOT" && -n "$OUTPUT_DIR" ]] || {
  echo "Usage: bash RUN/freeze_qwen_baseline_assets.sh --train_root DIR --holdout_root DIR --output_dir DIR" >&2
  exit 2
}

tea_train=("$TRAIN_ROOT"/tea/pairs_shard*.json)
tea_holdout=("$HOLDOUT_ROOT"/tea/pairs_shard*.json)
l2p_train=("$TRAIN_ROOT"/l2p/gram_shard*.pt)
l2p_holdout=("$HOLDOUT_ROOT"/l2p/gram_shard*.pt)
dpcache_costs=("$TRAIN_ROOT"/dpcache/cost_shard*.npz)
meancache_costs=("$TRAIN_ROOT"/meancache/cost_shard*.npz)

for spec in \
  "Tea train:${#tea_train[@]}" \
  "Tea holdout:${#tea_holdout[@]}" \
  "L2P train:${#l2p_train[@]}" \
  "L2P holdout:${#l2p_holdout[@]}" \
  "DPCache:${#dpcache_costs[@]}" \
  "MeanCache:${#meancache_costs[@]}"; do
  name="${spec%%:*}"
  count="${spec##*:}"
  (( count > 0 )) || {
    echo "[ERROR] no $name shards found" >&2
    exit 3
  }
done

mkdir -p "$OUTPUT_DIR"
tea_args=()
for path in "${tea_train[@]}"; do tea_args+=(--train "$path"); done
for path in "${tea_holdout[@]}"; do tea_args+=(--holdout "$path"); done
"$PYTHON_BIN" analysis/fit_teacache_polynomial.py \
  "${tea_args[@]}" --out "$OUTPUT_DIR/teacache_coefficients.json"

l2p_args=()
for path in "${l2p_train[@]}"; do l2p_args+=(--gram "$path"); done
for path in "${l2p_holdout[@]}"; do l2p_args+=(--holdout_gram "$path"); done
"$PYTHON_BIN" analysis/solve_l2p_grams.py \
  "${l2p_args[@]}" --out "$OUTPUT_DIR/l2p_weights.pt"

"$PYTHON_BIN" analysis/merge_sencache_sensitivity.py \
  --acc "$TRAIN_ROOT/sen" \
  --out "$OUTPUT_DIR/sencache_q90.npz" \
  --aggregation q90 \
  --backbone qwen_image \
  --num_steps 50

IFS=',' read -ra counts <<< "$CACHE_COUNTS"
for cache_count in "${counts[@]}"; do
  schedule_dir="$OUTPUT_DIR/schedules/k${cache_count}"
  "$PYTHON_BIN" analysis/build_fixed_predictor_schedule.py \
    --model qwen_image \
    --cache_count "$cache_count" \
    --output "$schedule_dir/shared_predictor.json"
  "$PYTHON_BIN" analysis/build_dpcache_schedule.py \
    --model qwen_image \
    --cache_count "$cache_count" \
    --cost_shards "${dpcache_costs[@]}" \
    --output "$schedule_dir/dpcache.json"
  "$PYTHON_BIN" analysis/build_meancache_schedule.py \
    --model qwen_image \
    --cache_count "$cache_count" \
    --cost_shards "${meancache_costs[@]}" \
    --output "$schedule_dir/meancache.json"
done

echo "[complete] Qwen baseline assets: $OUTPUT_DIR"
echo "[note] BudCache schedules are produced separately by slurm_qwen_budcache_search.sh"
