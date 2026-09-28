# Trajectories and error propagation

These experiments ask which changes during denoising are shared across prompts
and how caching changes the final output. Full-compute trajectories measure
denoising without caching. The error experiments compare cached and full-compute
generations, separating the current approximation error from earlier state
changes and measuring propagation from an isolated cached step.

Run commands from the repository root after installing the model-specific
dependencies. Set `GPH_DATA` to an absolute directory for your generated data.
Model checkpoints must be downloaded separately. The analysis scripts also use
NumPy, SciPy, pandas, Matplotlib, and PyTorch. JSONL output avoids the optional
PyArrow dependency.

This repository supplies code and small experimental inputs, not generated
trajectories, per-sample results, or figure summaries. Generation commands below
run models. Analysis commands recompute measurements from your own outputs.
Reducing the sample count is useful for testing, but does not reproduce the
paper's full evaluation.

## Full-compute image trajectories

The probes record 51 latent states from a 50-step generation without caching.
They write per-generation measurements as `traj_*.json`. `--save_frame` also
saves the PCA frames needed for plane comparisons. `--save_latents_first` retains
a subset of complete trajectories for plots and numerical-precision checks.

```bash
python flux/full_trajectory_probe.py \
  --prompts resources/prompts/prompt.txt --dataset drawbench_full --seed 41 \
  --output_dir "$GPH_DATA/full_traj/flux/drawbench_full_s41" \
  --num_steps 50 --save_frame --save_latents_first 30

python qwen_image/full_trajectory_probe.py \
  --prompts reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt \
  --dataset drawbench_full --seed 42 \
  --output_dir "$GPH_DATA/full_traj/qwen/drawbench_full_s42" \
  --num_steps 50 --save_frame --save_latents_first 30
```

Repeat for the datasets listed in `RUN/full_traj_datasets.sh`. The image study
uses FLUX seeds 41, 42, and 43, and Qwen seeds 42, 100042, and 200042. The
per-prompt seed is the base seed plus the global prompt index. Qwen uses its own
DrawBench file. The filename `geneval_seed43_n100.txt` contains 553 prompts.
Use `--shard i/n` to split generation while preserving global indices.

Merge the generated records and compute the population measurements:

```bash
python analysis/merge_full_traj.py \
  --results_dir "$GPH_DATA/full_traj" \
  --output_dir resources/full_trajectory/tables_jsonl --format jsonl

python analysis/full_trajectory_analysis.py \
  --figures outputs/full_trajectory \
  --step-profiles resources/full_trajectory_analysis/step_profiles.json

python analysis/trajectory_shape_scale.py \
  --root "$GPH_DATA/full_traj" \
  --out resources/full_trajectory_shape/shape_scale.json \
  --figures outputs/full_trajectory_shape
```

The shape analysis expects `ROOT/<model>/<run>/traj_*.json` and their frame
files. `full_trajectory_analysis.py` uses repository-relative table paths and
looks for stored examples under `resources/full_trajectory/latents_<model>/`.
Place the selected generated latent files there to reproduce its example-based
checks. Keep examples from one seed stream together to avoid filename collisions.

## Full-compute video trajectories

The video baseline runners collect trajectories with `--retain_trajectory`.
They compute step measurements during generation and save reference PCA frames.
The additional `--t3_seed` and `--t3_prompt_count` options retain full latent
trajectories for the specified seed and prompt subset.

```bash
python hunyuan_video/baseline_screen_runner.py \
  --mode original --model_base "$HY_MODEL" \
  --prompt_manifest resources/hunyuan_video/evaluation/penguin599.json \
  --seed 54 --retain_trajectory --t3_seed 54 --t3_prompt_count 120 \
  --output_dir "$GPH_DATA/hunyuan_video/matrix/references/penguin599_s54"

python wan21/baseline_screen_runner.py \
  --mode original --ckpt_dir "$WAN_MODEL" \
  --prompt_manifest resources/hunyuan_video/evaluation/penguin599.json \
  --seed 54 --retain_trajectory --t3_seed 54 --t3_prompt_count 120 \
  --output_dir "$GPH_DATA/wan21/matrix/references/penguin599_s54"
```

Set `HY_MODEL` and `WAN_MODEL` to your checkpoint directories. Repeat for
Penguin599 seeds 54, 55, and 56 and VBench944 seeds 42, 43, and 44, changing the
manifest, directory name, `--seed`, and `--t3_seed` together. Keep JSON prompt
manifests: some prompts contain line breaks. Do not supply a matrix configuration
to `--mode original`.

Use `matrix/references/<dataset>_s<seed>/` for these runs. The profile analysis
excludes `references_t3/`, which was used for supplementary repeats in the
original experiment.

The following commands illustrate HunyuanVideo analysis. Use `wan21` and its
`baseline_matrix_config.v1.json` for Wan. HunyuanVideo uses configuration v2.
The precision checks below generate required inputs, rather than reading
unprovided measurements.

```bash
python analysis/video_trajectory/merge_video_traj.py \
  --backbone hunyuan_video --root "$GPH_DATA/hunyuan_video/matrix" \
  --out "$GPH_DATA/hunyuan_video/matrix/trajectory" \
  --matrix_config resources/hunyuan_video/baseline_matrix_config.v2.json

for dtype in float32 bfloat16; do
  python analysis/trajectory_bf16_floor.py \
    --latents "$GPH_DATA/hunyuan_video/matrix/references/*/latents_*.pt" \
    --quant_dtype "$dtype" --windows 1 5 7 9 11 \
    --json "outputs/hunyuan_video/floors/p1_floor_${dtype}.json"
done

python analysis/video_trajectory/step_profiles.py \
  --backbone hunyuan_video --data_root "$GPH_DATA" \
  --p1_floor_dir outputs/hunyuan_video/floors \
  --out_tables resources/video_full_trajectory/hunyuan_video \
  --out_figs outputs/hunyuan_video/profiles

python analysis/video_trajectory/shape_scale.py \
  --backbone hunyuan_video --data_root "$GPH_DATA" \
  --p1_floor_dir outputs/hunyuan_video/floors \
  --out_tables resources/video_full_trajectory/hunyuan_video \
  --out_figs outputs/hunyuan_video/shape
```

Plane analysis reads the saved frame files, not only the merged scalar table.
The precision-check glob can be restricted to a documented subset for a smaller
run. It must refer to your retained latent files.

## Current approximation errors and earlier state changes

Section 5.1 uses `flux/trajectory_deviation_runner.py`. It evaluates the full
model at the cached trajectory's state to separate the current approximation
error from the effect of earlier state changes. This extra evaluation is for
measurement, not for choosing cached steps.

The four configurations are SeaCache thresholds 0.29 and 1.0, and TeaCache
thresholds 0.38 and 1.2. Each uses 100 held-out PartiPrompts, 50 steps, base
seed 42, and `first_enhance=1`. The supplied `prompt_ids_test100.txt` contains
global indices in the full prompt file. Do not replace it with `--limit 100`.

```bash
python flux/trajectory_deviation_runner.py \
  --prompt_file resources/prompts/partiprompts_full_eval1632_seed42.txt \
  --prompt_id_file resources/suffix_reversal/prompt_ids_test100.txt \
  --mode SeaCache --seacache_thresh 0.29 \
  --seed 42 --num_steps 50 --first_enhance 1 \
  --with_cf --no-save_images \
  --output_dir "$GPH_DATA/error_decomposition/seacache_t029"

python analysis/trajectory_deviation.py \
  --acc "$GPH_DATA/error_decomposition/seacache_t029"
```

Repeat with SeaCache threshold 1.0 and TeaCache thresholds 0.38 and 1.2.
For TeaCache, use `--mode TeaCache --teacache_thresh VALUE` instead of the
SeaCache options. Give each configuration its own output directory. The four
generation jobs are independent.

The aggregate prompt table contains `total_action_exposure`,
`total_state_gap_exposure`, and `final_latent_drift`, which are the two summed
error contributions and final latent-state error used in the correlation
analysis. `analysis/trajectory_deviation_stats.py --acc DIR` checks the recorded
decomposition. No generated images are needed for these latent-error statistics.

## Propagation from one cached step

Section 5.2 uses the first 100 DrawBench prompts, base seed 42, and 50 steps.
The sensitivity probe records both local feature error and its effect on the
model output. The amplification probe must use `--a_k_horizon 0` to propagate
to the final state. Its default horizon of five steps is a different experiment.

```bash
python flux/s_k_probe.py \
  --prompt_file resources/prompts/prompt.txt --limit 100 --seed 42 \
  --output_dir "$GPH_DATA/factors/s"

python flux/a_k_probe.py \
  --prompt_file resources/prompts/prompt.txt --limit 100 --seed 42 \
  --a_k_horizon 0 --output_dir "$GPH_DATA/factors/a"

python analysis/extract_q_k.py --output "$GPH_DATA/factors/q.json"

for method in seacache hicache taylorseer; do
  python flux/oracle_runner.py \
    --prompt_file resources/prompts/prompt.txt --limit 100 --seed 42 \
    --cache_mode "$method" \
    --output_dir "$GPH_DATA/factors/oracle_${method}"
  python evaluation/eval_oracle.py \
    --acc "$GPH_DATA/factors/oracle_${method}"
done
```

Each oracle run caches just one step at a time. The names correspond to
residual reuse, second-order Hermite, and first-order Taylor prediction.
Each method needs its own oracle outputs. The oracle evaluator also computes
image metrics and requires LPIPS, CLIP, and ImageReward dependencies and their
separately obtained weights. The generation jobs can be run independently;
evaluation depends on the corresponding generated outputs.

After generation and evaluation, compute the leave-one-prompt-out comparison:

```bash
python analysis/score_compare.py \
  --s_k "$GPH_DATA/factors/s" --q_k "$GPH_DATA/factors/q.json" \
  --a_k "$GPH_DATA/factors/a" \
  --oracle_seacache "$GPH_DATA/factors/oracle_seacache" \
  --oracle_hicache "$GPH_DATA/factors/oracle_hicache" \
  --oracle_taylorseer "$GPH_DATA/factors/oracle_taylorseer" \
  --calibration_mode loo --out_dir outputs/factors/scores

python analysis/four_factor_profile_ablation.py \
  --s_k "$GPH_DATA/factors/s" --q_k "$GPH_DATA/factors/q.json" \
  --a_k "$GPH_DATA/factors/a" \
  --oracle_seacache "$GPH_DATA/factors/oracle_seacache" \
  --oracle_hicache "$GPH_DATA/factors/oracle_hicache" \
  --oracle_taylorseer "$GPH_DATA/factors/oracle_taylorseer" \
  --out outputs/factors/profile_ablation.json
```

The last script compares local error, propagation factors, their product, and
the step-only control. For each prompt, the LOO factors use the other 99 prompts
at the same step and with the same approximation policy.

## Cached trajectory comparisons

### Images

Use `flux/sp_cross_runner.py` or `qwen_image/sp_cross_runner.py` with
`--retain_trajectory`. The sample indices are supplied in
`resources/image_trajectory/prompt_sample.v1.json`. The schedule/policy
combinations and their relative output directories are in `cells.v1.tsv`.
Generate the no-cache reference with `zero_schedule_50.txt` and `--payload reuse`
at `"$GPH_DATA/image_cached_traj/<model>/refs_parti_s42"`. Generate each cached
combination at `"$GPH_DATA/image_cached_traj/<cell_dir>"`, where `cell_dir` is
read from the manifest.

Both runners accept `--schedule_file`, `--payload`, `--prompt_file`,
`--prompt_indices`, `--seed 42`, `--output_dir`, and `--retain_trajectory`.
Pass the same comma-separated global prompt indices to reference and cached
runs. An all-zero schedule makes the reference use full computation at every
step. Preserve the original indices rather than creating a renumbered prompt file.

After generating both sides of each pair:

```bash
python analysis/image_trajectory/cached_paths.py --data_root "$GPH_DATA" \
  cell --out_dir outputs/image_cached/cells

python analysis/image_trajectory/cached_paths.py --data_root "$GPH_DATA" \
  merge --cells_dir outputs/image_cached/cells \
  --out_tables resources/image_trajectory
```

Use `cell --only CELL_ID` for one combination. The merge processes the generated
cell files; reproduce the manifest's combinations for the full study.
The supplied exclusion indices preserve the paper's
sample selection. `early_quality_link.py` additionally requires your evaluated
quality tables at `resources/spx/perprompt_spx_<model>.tsv.gz`; latent files
alone do not supply those quality measurements. Those tables are not bundled.

### Videos

Use the video baseline runners with the method's frozen matrix configuration,
`--retain_trajectory --t3_cached --t3_seed SEED --t3_prompt_count 120`, and the
`--prompt_indices` from
`resources/video_full_trajectory/t3_extension_samples.v1.json`. Preserve its
`cells_t3_rand50/<method>_<dataset>_<budget>_s<seed>` directory names. The cached
and reference runs must have the same prompt and initial seed.

`analysis/video_trajectory/cached_vs_reference.py` compares the merged step
measurements. `analysis/video_trajectory/latent_paths.py cached` compares retained
latent trajectories in three stages: `--stage cells`, `--stage refside`, and
`--stage merge`. Each accepts `--backbone`, `--data_root`, `--out_tables`, and
`--out_figs`. Pass all six streams with `--streams`, the supplied sample file
with `--sample_list`, and your generated precision-check directory with
`--p1_floor_dir`. The latent comparison reads references from `references_t3/`.
Before running these stages, generate independent reference repeats for all six
streams in that directory, retaining the first 120 trajectories of each stream.
Use the original-mode commands above with `--limit 120` and the output directory
changed from `references/` to `references_t3/`. Then rerun the video merge to
include these records. The `refside` stage compares these repeats with their
`references/` counterparts.
SenCache and L2P require their separately generated calibration assets; these
binary assets are not included in this repository.

These scripts retain the original directory conventions. Pass your own
`--data_root`; the release defaults to `outputs/`. Saved paper-figure
summaries are not included, so figure renderers that require those summaries
cannot be run until you have generated and analyzed the corresponding data.
