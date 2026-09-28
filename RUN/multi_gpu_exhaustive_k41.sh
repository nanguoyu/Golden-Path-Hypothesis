#!/bin/bash
# Run one Slurm-array task of the exhaustive FLUX K=41 search on local GPUs.

set -euo pipefail

OUTPUT_DIR=""
PROMPT_FILE="resources/prompts/partiprompts_full_eval1632_seed42.txt"
PROMPT_INDICES="5,8,9,15"
SEED=42
REVISION="3de623fc"
GPUS="0,1,2,3"
TASK_INDEX=0
TASK_COUNT=1
RANK_START=0
RANK_END=1370754
CHUNK_SIZE=256
CONDITIONING_FILE=""
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME=false

usage() {
  cat <<'USAGE'
Usage: bash RUN/multi_gpu_exhaustive_k41.sh --output_dir DIR [options]

  --task_index N       array task index                         [default: 0]
  --task_count N       total array tasks                        [default: 1]
  --gpus IDS           comma-separated local GPU ids            [default: 0,1,2,3]
  --rank_start N       requested global interval start          [default: 0]
  --rank_end N         requested global interval end            [default: 1370754]
  --chunk_size N       schedules per atomic part                [default: 256]
  --conditioning_file  frozen canonical prompt embeddings       [required]
  --prompt_file PATH   frozen PartiPrompts file
  --prompt_indices S   comma-separated global indices           [default: 5,8,9,15]
  --seed N             base seed                                [default: 42]
  --revision REV       FLUX snapshot                            [default: 3de623fc]
  --python PATH        Python interpreter
  --resume             validate and skip complete parts
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)      OUTPUT_DIR="$2"; shift 2;;
    --task_index)      TASK_INDEX="$2"; shift 2;;
    --task_count)      TASK_COUNT="$2"; shift 2;;
    --gpus)            GPUS="$2"; shift 2;;
    --rank_start)      RANK_START="$2"; shift 2;;
    --rank_end)        RANK_END="$2"; shift 2;;
    --chunk_size)      CHUNK_SIZE="$2"; shift 2;;
    --conditioning_file) CONDITIONING_FILE="$2"; shift 2;;
    --prompt_file)     PROMPT_FILE="$2"; shift 2;;
    --prompt_indices)  PROMPT_INDICES="$2"; shift 2;;
    --seed)            SEED="$2"; shift 2;;
    --revision)        REVISION="$2"; shift 2;;
    --python)          PYTHON_BIN="$2"; shift 2;;
    --resume)          RESUME=true; shift;;
    --help|-h)         usage; exit 0;;
    *) echo "[ERROR] unknown option: $1" >&2; usage; exit 1;;
  esac
done

[[ -n "$OUTPUT_DIR" ]] || { echo "[ERROR] --output_dir is required" >&2; exit 1; }
[[ -n "$CONDITIONING_FILE" ]] || { echo "[ERROR] --conditioning_file is required" >&2; exit 1; }
[[ -f "$CONDITIONING_FILE" ]] || { echo "[ERROR] conditioning file not found: $CONDITIONING_FILE" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "[ERROR] prompt file not found: $PROMPT_FILE" >&2; exit 1; }
[[ "$TASK_COUNT" -gt 0 ]] || { echo "[ERROR] --task_count must be positive" >&2; exit 1; }
[[ "$TASK_INDEX" -ge 0 && "$TASK_INDEX" -lt "$TASK_COUNT" ]] || {
  echo "[ERROR] --task_index must be in [0, task_count)" >&2; exit 1;
}

IFS=',' read -ra GPU_ARRAY <<<"$GPUS"
N_GPUS=${#GPU_ARRAY[@]}
[[ "$N_GPUS" -gt 0 ]] || { echo "[ERROR] no GPUs resolved" >&2; exit 1; }
GLOBAL_SHARD_COUNT=$((TASK_COUNT * N_GPUS))
mkdir -p "$OUTPUT_DIR/logs"

echo "[exhaustive-k41] task=$TASK_INDEX/$TASK_COUNT GPUs=$GPUS global_shards=$GLOBAL_SHARD_COUNT"
echo "[exhaustive-k41] ranks=[$RANK_START,$RANK_END) prompts=$PROMPT_INDICES output=$OUTPUT_DIR"

pids=()
for ((local_idx = 0; local_idx < N_GPUS; local_idx++)); do
  gpu="${GPU_ARRAY[$local_idx]}"
  global_shard_idx=$((TASK_INDEX * N_GPUS + local_idx))
  runner_args=(
    flux/exhaustive_k41_runner.py
    --output_dir "$OUTPUT_DIR"
    --prompt_file "$PROMPT_FILE"
    --prompt_indices "$PROMPT_INDICES"
    --seed "$SEED"
    --revision "$REVISION"
    --shard_idx "$global_shard_idx"
    --shard_count "$GLOBAL_SHARD_COUNT"
    --rank_start "$RANK_START"
    --rank_end "$RANK_END"
    --chunk_size "$CHUNK_SIZE"
    --conditioning_file "$CONDITIONING_FILE"
  )
  $RESUME && runner_args+=(--resume)
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${runner_args[@]}" \
    >"$OUTPUT_DIR/logs/worker_${global_shard_idx}_of_${GLOBAL_SHARD_COUNT}.log" 2>&1 &
  pids+=("$!")
done

status=0
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    status="$rc"
  fi
done
set -e
if [[ "$status" -ne 0 ]]; then
  echo "[ERROR] at least one local GPU worker failed; inspect $OUTPUT_DIR/logs" >&2
  exit "$status"
fi
echo "[exhaustive-k41] task $TASK_INDEX/$TASK_COUNT complete"
