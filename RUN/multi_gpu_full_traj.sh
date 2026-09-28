#!/bin/bash
#
# Single-node multi-GPU launcher for the full-trajectory regularity probes
# (docs/research_plan_full_trajectory.md), for hosts without Slurm.
#
# One work item = one (model, dataset, seed) cell, run whole on one GPU. Cells
# are pulled from a shared queue rather than pre-assigned, because a qwen cell
# costs ~3x a flux one and a round-robin split would leave GPUs idle for hours
# at the tail. Run the file longest-cell-first for the usual greedy bound.
#
# Cell file: one `model dataset seed` per line; `#` comments and blanks ignored.
#
#   bash RUN/multi_gpu_full_traj.sh \
#        --cells RUN/cells_full_traj_shape_scale.txt \
#        --gpus 0,1,2,3,4,6 --output_root ~/full_traj_shape --limit 120 --save_frame
#
# Resume is the probes' own skip-if-exists: re-running the same command picks
# up every cell where it stopped. Run from the repo root.

set -euo pipefail

CELLS=""
GPUS=""
OUTPUT_ROOT=""
LIMIT=0
NUM_STEPS=50
DTYPE="bf16"
SAVE_FRAME=false
FRAME_SEGMENTS=""
SAVE_LATENTS_FIRST=0
PYTHON_BIN="${PYTHON_BIN:-python}"

print_usage() {
  cat <<'USAGE'
Usage: bash RUN/multi_gpu_full_traj.sh --cells FILE --gpus IDS --output_root DIR [opts]

Required:
  --cells FILE          one `model dataset seed` per line
  --gpus IDS            comma-separated CUDA device ids, e.g. 0,1,2,3,4,6
  --output_root DIR     run dirs go to DIR/<model>/<dataset>_n<limit>_s<seed>_<steps>

Options:
  --limit N             prompts per cell, 0 = whole dataset (default 0)
  --num_steps N         solver steps (default 50)
  --dtype bf16|fp16|fp32
  --save_frame          store the per-generation [chord, PC1, PC2] frame
  --frame_segments SPEC store one frame per row range, e.g. "0:51,38:51"
  --save_latents_first M  store full latent paths for prompt idx < M
  --python PATH
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cells)       CELLS="$2"; shift 2;;
    --gpus)        GPUS="$2"; shift 2;;
    --output_root) OUTPUT_ROOT="$2"; shift 2;;
    --limit)       LIMIT="$2"; shift 2;;
    --num_steps)   NUM_STEPS="$2"; shift 2;;
    --dtype)       DTYPE="$2"; shift 2;;
    --save_frame)  SAVE_FRAME=true; shift;;
    --frame_segments) FRAME_SEGMENTS="$2"; shift 2;;
    --save_latents_first) SAVE_LATENTS_FIRST="$2"; shift 2;;
    --python)      PYTHON_BIN="$2"; shift 2;;
    --help|-h)     print_usage; exit 0;;
    *) echo "[ERROR] unknown option: $1" >&2; print_usage; exit 1;;
  esac
done

[[ -n "$CELLS" && -f "$CELLS" ]] || { echo "[ERROR] --cells FILE required and must exist" >&2; exit 1; }
[[ -n "$GPUS" ]] || { echo "[ERROR] --gpus required" >&2; exit 1; }
[[ -n "$OUTPUT_ROOT" ]] || { echo "[ERROR] --output_root required" >&2; exit 1; }
[[ -f RUN/full_traj_datasets.sh ]] || {
  echo "[ERROR] run from the repo root: RUN/full_traj_datasets.sh not found from $(pwd)" >&2; exit 1; }
source RUN/full_traj_datasets.sh

IFS=',' read -ra GPU_ARR <<<"$GPUS"
N_GPUS=${#GPU_ARR[@]}
[[ "$N_GPUS" -gt 0 ]] || { echo "[ERROR] no GPUs resolved from '$GPUS'" >&2; exit 1; }

########################
# Validate every cell BEFORE launching anything: a bad dataset name or an
# off-matrix seed must not surface three hours in, on one worker.
########################
QUEUE="$(mktemp)"
grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$CELLS" >"$QUEUE" || true
N_CELLS=$(wc -l <"$QUEUE" | tr -d ' ')
[[ "$N_CELLS" -gt 0 ]] || { echo "[ERROR] $CELLS holds no cells" >&2; exit 1; }

while read -r model dataset seed _rest; do
  [[ "$model" == "flux" || "$model" == "qwen" ]] || {
    echo "[ERROR] bad model '$model' in $CELLS" >&2; exit 1; }
  [[ -n "$seed" ]] || { echo "[ERROR] cell '$model $dataset' has no seed" >&2; exit 1; }
  full_traj_check_seed "$model" "$seed"
  full_traj_resolve_dataset "$dataset" "$model"
  full_traj_check_prompt_count "$FULL_TRAJ_PROMPT_FILE" "$FULL_TRAJ_EXPECTED"
  if [[ "$LIMIT" -gt 0 && "$LIMIT" -gt "$FULL_TRAJ_EXPECTED" ]]; then
    echo "[ERROR] --limit $LIMIT exceeds $dataset's $FULL_TRAJ_EXPECTED prompts" >&2; exit 1
  fi
done <"$QUEUE"

mkdir -p "$OUTPUT_ROOT/logs"
POS="$OUTPUT_ROOT/.queue_pos"
LOCK="$OUTPUT_ROOT/.queue_lock"
echo 0 >"$POS"
: >"$LOCK"
: >"$OUTPUT_ROOT/logs/failed_cells.txt"  # else a clean retry still reports the old run's failures

echo "================================="
echo "[full-traj] cells       = $N_CELLS ($CELLS)"
echo "[full-traj] gpus        = $GPUS ($N_GPUS workers)"
echo "[full-traj] limit       = $LIMIT prompts/cell   num_steps = $NUM_STEPS   dtype = $DTYPE"
echo "[full-traj] save_frame  = $SAVE_FRAME   segments = ${FRAME_SEGMENTS:-<whole path>}"
echo "[full-traj] save_latents_first = $SAVE_LATENTS_FIRST"
echo "[full-traj] output_root = $OUTPUT_ROOT"
echo "[full-traj] host        = $(hostname)"
echo "================================="

claim_next() {
  local idx
  exec 9>"$LOCK"
  flock 9
  idx=$(cat "$POS")
  echo $((idx + 1)) >"$POS"
  flock -u 9
  exec 9>&-
  [[ "$idx" -lt "$N_CELLS" ]] || return 1
  sed -n "$((idx + 1))p" "$QUEUE"
}

run_worker() {
  local gpu="$1" cell model dataset seed run_name output_dir log probe rc
  while cell=$(claim_next); do
    read -r model dataset seed _rest <<<"$cell"
    full_traj_resolve_dataset "$dataset" "$model"
    if [[ "$LIMIT" -eq 0 ]]; then
      run_name="${dataset}_nfull_s${seed}_${NUM_STEPS}"
    else
      run_name="${dataset}_n${LIMIT}_s${seed}_${NUM_STEPS}"
    fi
    output_dir="$OUTPUT_ROOT/$model/$run_name"
    log="$OUTPUT_ROOT/logs/${model}_${run_name}.log"
    mkdir -p "$output_dir"

    if [[ "$model" == "flux" ]]; then probe="flux/full_trajectory_probe.py"
    else probe="qwen_image/full_trajectory_probe.py"; fi

    local args=(
      --prompts    "$FULL_TRAJ_PROMPT_FILE"
      --dataset    "$dataset"
      --seed       "$seed"
      --output_dir "$output_dir"
      --limit      "$LIMIT"
      --shard      "0/1"
      --save_latents_first "$SAVE_LATENTS_FIRST"
      --num_steps  "$NUM_STEPS"
      --dtype      "$DTYPE"
    )
    if [[ "$SAVE_FRAME" == true ]]; then args+=(--save_frame); fi
    if [[ -n "$FRAME_SEGMENTS" ]]; then args+=(--frame_segments "$FRAME_SEGMENTS"); fi

    echo "[GPU $gpu] START $model $dataset s$seed -> $log"
    rc=0
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$probe" "${args[@]}" >>"$log" 2>&1 || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      # keep the wave going: one failed cell is re-runnable on its own, a
      # worker that dies takes every remaining cell in its share with it
      echo "[GPU $gpu] FAIL  $model $dataset s$seed rc=$rc (see $log)"
      echo "$model $dataset $seed rc=$rc" >>"$OUTPUT_ROOT/logs/failed_cells.txt"
    else
      echo "[GPU $gpu] DONE  $model $dataset s$seed ($(ls "$output_dir"/traj_*.json 2>/dev/null | wc -l | tr -d ' ') records)"
    fi
  done
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  run_worker "$gpu" &
  pids+=($!)
done
# `|| true`: under set -e a worker that dies on anything other than the probe
# (which is already caught) would abort the launcher here, leaving the other
# five workers running as orphans while the queue state is torn down
for pid in "${pids[@]}"; do wait "$pid" || echo "[WARN] a worker exited non-zero"; done

########################
# Completeness is decided by what is on disk, not by exit codes: a worker that
# dies outside the probe drops the cell it had already claimed without writing
# a failure line for it.
########################
short=0
while read -r model dataset seed _rest; do
  full_traj_resolve_dataset "$dataset" "$model"
  if [[ "$LIMIT" -eq 0 ]]; then
    expected="$FULL_TRAJ_EXPECTED"; run_name="${dataset}_nfull_s${seed}_${NUM_STEPS}"
  else
    expected="$LIMIT"; run_name="${dataset}_n${LIMIT}_s${seed}_${NUM_STEPS}"
  fi
  got=$(find "$OUTPUT_ROOT/$model/$run_name" -maxdepth 1 -name 'traj_*.json' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$got" -ne "$expected" ]]; then
    echo "[SHORT] $model $dataset s$seed: $got / $expected records"
    short=$((short + 1))
  fi
done <"$QUEUE"

rm -f "$QUEUE" "$POS" "$LOCK"

echo "================================="
if [[ -s "$OUTPUT_ROOT/logs/failed_cells.txt" || "$short" -gt 0 ]]; then
  if [[ -s "$OUTPUT_ROOT/logs/failed_cells.txt" ]]; then
    echo "FAILED CELLS:"; cat "$OUTPUT_ROOT/logs/failed_cells.txt"
  fi
  if [[ "$short" -gt 0 ]]; then
    echo "$short cell(s) short of the expected record count (see [SHORT] above)"
  fi
  echo "Re-run the same command to retry (completed records are skipped)."
  exit 1
fi
echo "ALL $N_CELLS CELLS COMPLETE under $OUTPUT_ROOT"
find "$OUTPUT_ROOT" -name 'traj_*.json' | wc -l | xargs echo "  records:"
find "$OUTPUT_ROOT" -name 'frame_*.npy' | wc -l | xargs echo "  frames: "
echo "================================="
