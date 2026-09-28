#!/bin/bash
#
# Dataset -> prompt-file map and seed whitelist for the full-trajectory probes,
# shared by RUN/slurm_full_trajectory.sh and RUN/multi_gpu_full_traj.sh.
# Source it; do not execute it.
#
# The registry it mirrors is analysis/full_results_local/registry.py: the whole
# point of the probe is that each trajectory pairs with a 720-cell baseline
# generation, which only holds if the prompt file and the seed stream are the
# matrix's own. Two copies of this map would eventually disagree and the
# pairing would break silently, so there is one.

# Sets FULL_TRAJ_PROMPT_FILE and FULL_TRAJ_EXPECTED for ($1 dataset, $2 model).
full_traj_resolve_dataset() {
  local dataset="$1" model="$2"
  case "$dataset" in
    drawbench_full)
      # the one model-specific file: flux ran the plain prompt list, qwen the
      # quoted DrawBench200 copy vendored under reference/
      if [[ "$model" == "qwen" ]]; then
        FULL_TRAJ_PROMPT_FILE="reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt"
      else
        FULL_TRAJ_PROMPT_FILE="resources/prompts/prompt.txt"
      fi
      FULL_TRAJ_EXPECTED=200;;
    parti_full)
      FULL_TRAJ_PROMPT_FILE="resources/prompts/partiprompts_full_eval1632_seed42.txt"
      FULL_TRAJ_EXPECTED=1632;;
    geneval_style)
      FULL_TRAJ_PROMPT_FILE="resources/prompts/geneval_seed43_n100.txt"
      FULL_TRAJ_EXPECTED=553;;
    diffusiondb_clean10k)
      FULL_TRAJ_PROMPT_FILE="resources/prompts/diffusiondb_2m_clean_10000_seed42.txt"
      FULL_TRAJ_EXPECTED=10000;;
    *) echo "[ERROR] unknown dataset: $dataset" >&2; return 1;;
  esac
  return 0
}

# Counted through the probe's own reader, not `grep -c`: prompt.txt has no
# trailing newline and BSD/GNU grep disagree on its last line.
full_traj_check_prompt_count() {
  local file="$1" expected="$2" actual
  [[ -f "$file" ]] || { echo "[ERROR] prompt file missing: $file" >&2; return 1; }
  actual=$(python -c \
    "import sys; sys.path.insert(0, '.'); from lib.io_utils import read_prompts; print(len(read_prompts(sys.argv[1])))" \
    "$file")
  [[ "$actual" -eq "$expected" ]] || {
    echo "[ERROR] $file has $actual prompts, registry expects $expected" >&2; return 1; }
  return 0
}

# A seed off the matrix stream pairs with no baseline generation.
full_traj_check_seed() {
  local model="$1" seed="$2"
  [[ "${ALLOW_OFF_MATRIX_SEED:-0}" == "1" ]] && return 0
  case "$model:$seed" in
    flux:41|flux:42|flux:43|qwen:42|qwen:100042|qwen:200042) return 0;;
    *) echo "[ERROR] seed $seed is not in the $model matrix seed stream" \
            "(flux: 41/42/43, qwen: 42/100042/200042); set ALLOW_OFF_MATRIX_SEED=1 to force" >&2
       return 1;;
  esac
}
