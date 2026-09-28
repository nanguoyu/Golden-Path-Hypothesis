#!/usr/bin/env bash
# Request site-specific resources through sbatch options, not this file.
set -euo pipefail
if (( $# == 0 )); then
  echo 'Usage: sbatch [resource options] scripts/slurm_job.sh python runner.py [options]' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
