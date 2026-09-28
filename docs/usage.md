# Running experiments

This guide covers preparation, cluster execution, output organization, and tests.
Use the [research and code index](../README.md#research-and-code) to choose an
experiment. Its linked guide gives the comparison, required inputs, commands,
and analysis outputs. An optional provided-schedule example appears at the end.

## Prepare an experiment

1. Follow [installation](installation.md). Image generation, HunyuanVideo
   generation, and VBench evaluation use separate environments where
   dependencies conflict.
2. For video experiments, fetch the pinned public source dependencies:

   ```bash
   python scripts/bootstrap_sources.py hunyuan_video taylorseer vbench
   ```

3. Prepare the required prompt sets:

   ```bash
   python scripts/prepare_data.py --dataset image
   python scripts/prepare_data.py --dataset video
   ```

   DiffusionDB is a separate download. See [data preparation](data.md).
4. Obtain pretrained models from their official sources and rebuild required
   method assets using [assets.md](assets.md).

Run commands from the repository root. Install the local package in each
environment with `python -m pip install -e . --no-deps`. Dataset files and
experiment configurations are read from the checkout, so keep this directory
available when running scripts.

## Running on a cluster

The video generation and several video calibration entry points require a
Slurm compute node. See [execution environments](installation.md#execution-environment).
`scripts/slurm_job.sh` forwards a command without selecting a cluster, account,
partition, or environment. Activate your environment before submission and pass
your site's resource options to `sbatch`:

```bash
sbatch --gpus=1 --cpus-per-task=8 --time=02:00:00 \
  scripts/slurm_job.sh python flux/schedule_search_runner.py \
  --k 41 --algorithm hill --objective psnr \
  --output_dir outputs/search/flux_k41_hill
```

Use distinct output directories for independent jobs. When sharding one run,
keep the prompt order, base seed, and shard count fixed. Specialized wrappers
under `RUN/` construct experiment-specific commands and divide work across
GPUs. The full-trajectory wrapper also expects a local Mamba environment.
Use the generic wrapper when those site conventions do not apply.

## Files and generated inputs

- `lib/` and the four model directories implement caching and generation.
- `evaluation/` computes image and video metrics.
- `analysis/` prepares inputs, assembles results, and produces plots and tables.
- `resources/` contains small experiment definitions, split indices, thresholds,
  and schedule bitstrings. A bit of `1` means a cached step, and `0` means a full step.
- `scripts/` prepares public data and source dependencies and converts evaluator
  outputs to the formats used by the analysis.
- `tests/` contains CPU unit tests for the included components.

Model weights, fitted weights, generated media, saved trajectories, and
per-example result tables are obtained or generated separately. The
exhaustive-search score table is produced by the exhaustive experiment.
Some analyses read generated files under `resources/`. Each experiment guide
identifies the required inputs and the scripts that produce them. Figure scripts
that write to `paper/figs/` create that directory locally. The README figures in
`docs/figures/` illustrate the paper's reported results.

Output paths default to `outputs/`. Matrix configurations can include hashes of
calibration assets. Rebuilt assets require new matrix configurations, as
described in [assets.md](assets.md).

## Tests

With the analysis dependencies and a CPU or CUDA build of PyTorch installed:

```bash
python -m pytest tests
```

These are CPU tests. Full experiment reproduction also requires the pretrained
models, model-specific dependencies, and GPU computation. Tests that rebuild
schedules from experiment measurements are skipped when those generated
measurements are absent.

## Optional: run a provided schedule

This example uses a FLUX.1-dev schedule selected by the paper's few-example
search with PSNR as its objective. It caches 41 of 50 steps and uses the same
schedule for both prompts. Run the commands from the repository root after
installing the [image environment](installation.md#flux-qwen-image-and-wan21)
and downloading the FLUX.1-dev checkpoint in Diffusers format.

Set the local checkpoint directory and prepare the DrawBench prompt file:

```bash
export GPH_FLUX_MODEL=/path/to/FLUX.1-dev
python scripts/prepare_data.py --dataset drawbench
```

Generate the full-compute outputs for the first two prompts:

```bash
python flux/runner.py \
  --mode original --model_id "$GPH_FLUX_MODEL" \
  --prompt_file resources/prompts/prompt.txt \
  --output_dir outputs/example/flux_full \
  --num_steps 50 --seed 42 --limit 2 \
  --height 1024 --width 1024 --guidance 3.5 --dtype bf16
```

Generate the cached outputs using the provided schedule and residual reuse.
Residual reuse adds the feature change from the most recent full step to the
current input features.

```bash
python flux/sp_cross_runner.py \
  --model_id "$GPH_FLUX_MODEL" \
  --schedule_file resources/schedule_search/schedules/flux_k41_ss_hill.txt \
  --payload reuse \
  --prompt_file resources/prompts/prompt.txt \
  --output_dir outputs/example/flux_k41 \
  --num_steps 50 --seed 42 --limit 2 \
  --height 1024 --width 1024 --guidance 3.5 --dtype bf16
```

Both commands use the same prompts and generation settings. The actual noise
seeds are 42 and 43 because each runner adds the prompt index to the base seed.
Each output directory contains `img_0.png` and `img_1.png`.
The cached directory also contains the per-prompt caching decisions.

Compare the cached images with their matching full-compute outputs:

```bash
python evaluation/eval_metrics.py \
  --acc outputs/example/flux_k41 \
  --gt outputs/example/flux_full \
  --prompts resources/prompts/prompt.txt \
  --limit 2 --metrics psnr ssim --device cpu
```

The evaluator writes `outputs/example/flux_k41/metrics.json`. Its
`summary.psnr.mean` and `summary.ssim.mean` fields contain the two-image means.
Higher values indicate closer agreement with the full-compute outputs.
These two metrics do not require additional learned metric models.
The cache ratio of 0.82 describes the fraction of cached steps. Generation
timings are recorded separately in each directory's `timing_shard0of1.json`.

Use the [baseline](baselines.md) and [schedule search](search-and-sharing.md)
guides for the full datasets and seed counts used in the paper.
