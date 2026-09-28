#!/bin/bash
#
# Slurm wrapper for full-trajectory analysis:
# flux/full_trajectory_probe.py and
# qwen_image/full_trajectory_probe.py behind one --model switch.
#
# Topology: 1 sbatch array task = 1 node x 1 GPU = 1 shard, so pass
# --array=0-N-1 at submit time and the task id becomes `--shard i/N`.
# Submit from the repo root (the dataset->prompt-file map is repo-relative).
#
# Matrix seed streams (analysis/full_results_local/registry.py):
#   flux  S0/S1/S2 = 41 / 42 / 43
#   qwen  S0/S1/S2 = 42 / 100042 / 200042
#
# Examples:
#   # DrawBench example with five prompts, both models
#   sbatch --array=0-0 RUN/slurm_full_trajectory.sh \
#          --model flux --dataset drawbench_full --seed 41 --limit 5
#   sbatch --array=0-0 RUN/slurm_full_trajectory.sh \
#          --model qwen --dataset drawbench_full --seed 42 --limit 5
#
#   # DrawBench wave with the figure sample on the first seed
#   sbatch --array=0-9 RUN/slurm_full_trajectory.sh \
#          --model flux --dataset drawbench_full --seed 41 --save_latents_first 30
#
#SBATCH -J full_traj
#SBATCH --output=full_traj_%A_%a.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

set -euo pipefail

########################
# mamba / cache env
########################
if [[ -z "${MAMBA_ROOT:-}" ]]; then
  for _prefix in "$HOME/mambaforge" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/conda"; do
    if [[ -x "$_prefix/bin/mamba" ]]; then MAMBA_ROOT="$_prefix"; break; fi
  done
fi
[[ -x "${MAMBA_ROOT:-}/bin/mamba" ]] || { echo "[ERROR] mamba not found" >&2; exit 1; }
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export MAMBA_EXE="$MAMBA_ROOT/bin/mamba"
eval "$($MAMBA_ROOT/bin/mamba shell hook --shell bash)"
mamba activate "${HICACHE_ENV:-cache}"

########################
# Caches under $HOME (no /data on Slurm)
########################
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_ENABLE_HF_TRANSFER=1

JOB_ID="${SLURM_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export TRITON_CACHE_DIR="/tmp/triton_cache/${JOB_ID}/${TASK_ID}"
export CUDA_CACHE_PATH="/tmp/cuda_cache/${JOB_ID}/${TASK_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

########################
# CLI defaults
########################
MODEL="flux"
DATASET=""
SEED=""
PROMPT_FILE=""
LIMIT=0
SAVE_LATENTS_FIRST=0
SAVE_FRAME=false
FRAME_SEGMENTS=""
NUM_STEPS=50
DTYPE="bf16"
RUN_NAME=""
BASE_OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)               MODEL="$2"; shift 2;;
    --dataset)             DATASET="$2"; shift 2;;
    --seed)                SEED="$2"; shift 2;;
    -p|--prompts)          PROMPT_FILE="$2"; shift 2;;
    -l|--limit)            LIMIT="$2"; shift 2;;
    --save_latents_first)  SAVE_LATENTS_FIRST="$2"; shift 2;;
    --save_frame)          SAVE_FRAME=true; shift;;
    --frame_segments)      FRAME_SEGMENTS="$2"; shift 2;;
    -s|--num_steps)        NUM_STEPS="$2"; shift 2;;
    --dtype)               DTYPE="$2"; shift 2;;
    --run-name|--run_name) RUN_NAME="$2"; shift 2;;
    -d|--output_dir)       BASE_OUTPUT_DIR="$2"; shift 2;;
    *) echo "[ERROR] unknown option: $1" >&2; exit 1;;
  esac
done

[[ "$MODEL" == "flux" || "$MODEL" == "qwen" ]] || { echo "[ERROR] --model must be flux|qwen" >&2; exit 1; }
[[ -n "$DATASET" ]] || { echo "[ERROR] --dataset is required" >&2; exit 1; }
[[ -n "$SEED"    ]] || { echo "[ERROR] --seed is required" >&2; exit 1; }

# cwd is SLURM_SUBMIT_DIR = repo root, same convention the prompt paths below use
[[ -f RUN/full_traj_datasets.sh ]] || {
  echo "[ERROR] run from the repo root: RUN/full_traj_datasets.sh not found from $(pwd)" >&2; exit 1; }
source RUN/full_traj_datasets.sh

full_traj_check_seed "$MODEL" "$SEED"
full_traj_resolve_dataset "$DATASET" "$MODEL"
EXPECTED_PROMPTS="$FULL_TRAJ_EXPECTED"

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$FULL_TRAJ_PROMPT_FILE"
  full_traj_check_prompt_count "$PROMPT_FILE" "$EXPECTED_PROMPTS"
fi
[[ -f "$PROMPT_FILE" ]] || { echo "[ERROR] prompt file missing: $PROMPT_FILE" >&2; exit 1; }

if [[ -z "$BASE_OUTPUT_DIR" ]]; then
  BASE_OUTPUT_DIR="$HOME/full_traj_results/$MODEL"
fi
if [[ -z "$RUN_NAME" ]]; then
  if [[ "$LIMIT" -eq 0 ]]; then
    RUN_NAME="${DATASET}_nfull_s${SEED}_${NUM_STEPS}"
  else
    RUN_NAME="${DATASET}_n${LIMIT}_s${SEED}_${NUM_STEPS}"
  fi
fi
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "================================="
echo "[full-traj] model        = $MODEL"
echo "[full-traj] dataset      = $DATASET ($EXPECTED_PROMPTS prompts)"
echo "[full-traj] prompts      = $PROMPT_FILE"
echo "[full-traj] seed         = $SEED   num_steps = $NUM_STEPS   limit = $LIMIT"
echo "[full-traj] save_latents = first $SAVE_LATENTS_FIRST prompt indices"
echo "[full-traj] save_frame   = $SAVE_FRAME   segments = ${FRAME_SEGMENTS:-<whole path>}"
echo "[full-traj] output       = $OUTPUT_DIR"
echo "[full-traj] shard        = $SHARD_IDX / $SHARD_COUNT (Slurm array)"
echo "[full-traj] host         = $(hostname)"
echo "================================="

if [[ "$MODEL" == "flux" ]]; then
  PROBE="flux/full_trajectory_probe.py"
else
  PROBE="qwen_image/full_trajectory_probe.py"
fi

RUNNER_ARGS=(
  --prompts    "$PROMPT_FILE"
  --dataset    "$DATASET"
  --seed       "$SEED"
  --output_dir "$OUTPUT_DIR"
  --limit      "$LIMIT"
  --shard      "${SHARD_IDX}/${SHARD_COUNT}"
  --save_latents_first "$SAVE_LATENTS_FIRST"
  --num_steps  "$NUM_STEPS"
  --dtype      "$DTYPE"
)
if [[ "$SAVE_FRAME" == true ]]; then RUNNER_ARGS+=(--save_frame); fi  # `&&` would trip set -e
if [[ -n "$FRAME_SEGMENTS" ]]; then RUNNER_ARGS+=(--frame_segments "$FRAME_SEGMENTS"); fi

echo "[CMD] python $PROBE ${RUNNER_ARGS[*]}"
python "$PROBE" "${RUNNER_ARGS[@]}"

echo "[INFO] shard $SHARD_IDX/$SHARD_COUNT done on $(hostname)."
echo "[INFO] Merge ONCE per (model, dataset) after ALL THREE seed waves finish,"
echo "       passing every seed run dir in one call (a single-seed merge would"
echo "       overwrite the combined table):"
echo "       python analysis/merge_full_traj.py --output_dir $BASE_OUTPUT_DIR/tables \\"
echo "              --results_dir $BASE_OUTPUT_DIR/${DATASET}_nfull_s<S0>_${NUM_STEPS} \\"
echo "                            $BASE_OUTPUT_DIR/${DATASET}_nfull_s<S1>_${NUM_STEPS} \\"
echo "                            $BASE_OUTPUT_DIR/${DATASET}_nfull_s<S2>_${NUM_STEPS}"
