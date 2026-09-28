# Building model and method assets

This release contains code, not model weights or generated experiment outputs.
Obtain pretrained models from their official repositories. Build L2P weights,
SenCache sensitivity tables, Qwen TeaCache coefficients, and offline schedules
locally using the commands below. These jobs require GPU computation.

Run from the repository root. Use Bash for the examples with arrays. Set
`FLUX_MODEL`, `QWEN_MODEL`, `HUNYUAN_MODEL`, and `WAN_MODEL` to downloaded model
directories, and set `ASSETS_ROOT` to a writable directory. Keep separate
environments for HunyuanVideo, the newer image/Wan stack, and VBench where
their upstream dependency versions conflict.

## Model sources

| Model | Official source | Runtime argument |
|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` on Hugging Face | `--model_id` |
| Qwen-Image | `Qwen/Qwen-Image` on Hugging Face | `--model_id` |
| HunyuanVideo | Tencent HunyuanVideo official download instructions | `--model_base` |
| Wan2.1-T2V-1.3B | `Wan-AI/Wan2.1-T2V-1.3B` on Hugging Face | `--ckpt_dir` |

Accept any required model license before downloading. HunyuanVideo needs its
complete model directory, including the text encoders and VAE, in the layout
expected by the official implementation. The model checkpoints are distinct
from the small method-specific assets built below.

The evaluators also download LPIPS, CLIP, ImageReward, and VBench models.
Configure their cache locations before running on a machine without internet
access. No authentication credentials are included in this repository.

## Image assets

Use `resources/baseline_exact/image_calibration50.txt` for fitting and
`image_holdout10.txt` for the L2P and Qwen TeaCache holdout checks. First choose
one model. For FLUX:

```bash
image_model=flux
image_checkpoint="$FLUX_MODEL"
calibration_seed=42
l2p_collector=flux/l2p_output_collect.py
sen_collector=flux/sencache_sensitivity_runner.py
freeze_script=RUN/freeze_flux_baseline_assets.sh
bud_search=analysis/search_budcache_flux.py
bud_prompts=2
```

For Qwen, use these assignments instead. Its collectors default to
1328 × 1328, true CFG 4.0, and an ASCII space as the negative prompt.

```bash
image_model=qwen_image
image_checkpoint="$QWEN_MODEL"
calibration_seed=20260723
l2p_collector=qwen_image/l2p_collect.py
sen_collector=qwen_image/sencache_calibrate.py
freeze_script=RUN/freeze_qwen_baseline_assets.sh
bud_search=analysis/search_budcache_qwen.py
bud_prompts=3
```

Then collect the selected model's fitting data:

```bash
train="$ASSETS_ROOT/$image_model/calibration/train"
holdout="$ASSETS_ROOT/$image_model/calibration/holdout"
assets="$ASSETS_ROOT/$image_model"
train_prompts=resources/baseline_exact/image_calibration50.txt
holdout_prompts=resources/baseline_exact/image_holdout10.txt
mkdir -p "$train/l2p" "$holdout/l2p" "$train/dpcache" "$train/meancache"

python "$l2p_collector" --model_id "$image_checkpoint" \
  --prompt_file "$train_prompts" --limit 50 --seed "$calibration_seed" \
  --gram_out "$train/l2p/gram_shard0of1.pt"
python "$l2p_collector" --model_id "$image_checkpoint" \
  --prompt_file "$holdout_prompts" --limit 10 --seed "$calibration_seed" \
  --gram_out "$holdout/l2p/gram_shard0of1.pt"
python "$sen_collector" --model_id "$image_checkpoint" \
  --prompt_file "$train_prompts" --limit 50 --seed "$calibration_seed" \
  --output_dir "$train/sen"
python "$image_model/dpcache_calibrate.py" --model_id "$image_checkpoint" \
  --prompt_file "$train_prompts" --limit 50 --seed "$calibration_seed" \
  --out "$train/dpcache/cost_shard0of1.npz"
python "$image_model/meancache_calibrate.py" --model_id "$image_checkpoint" \
  --prompt_file "$train_prompts" --limit 50 --seed "$calibration_seed" \
  --out "$train/meancache/cost_shard0of1.npz"
```

For Qwen only, also collect TeaCache data before freezing:

```bash
mkdir -p "$train/tea" "$holdout/tea"
python qwen_image/teacache_fit_collect.py --model_id "$QWEN_MODEL" \
  --prompt_file "$train_prompts" --limit 50 --seed 20260723 \
  --out "$train/tea/pairs_shard0of1.json"
python qwen_image/teacache_fit_collect.py --model_id "$QWEN_MODEL" \
  --prompt_file "$holdout_prompts" --limit 10 --seed 20260723 \
  --out "$holdout/tea/pairs_shard0of1.json"
```

Freeze each model's completed collections, then search its BudCache schedules:

```bash
bash "$freeze_script" --train_root "$train" --holdout_root "$holdout" \
  --output_dir "$assets" --cache_counts 29,37,41
for cache_count in 29 37 41; do
  python "$bud_search" --model_id "$image_checkpoint" \
    --prompt_file "$train_prompts" --limit "$bud_prompts" \
    --seed "$calibration_seed" --search_seed "$calibration_seed" \
    --cache_count "$cache_count" \
    --output "$assets/schedules/k$cache_count/budcache.json"
done
```

The freeze scripts fit L2P with `analysis/solve_l2p_grams.py`, merge SenCache
with `analysis/merge_sencache_sensitivity.py --aggregation q90`, build the
shared prediction schedules, and solve DPCache and MeanCache schedules from
their cost tensors. Qwen's script also fits the TeaCache polynomial. These
scripts run locally and do not submit Slurm jobs.

For multiple GPUs, each collector accepts `--shard_idx` and `--shard_count`.
Use matching output filenames for each shard. The freeze scripts collect all
`gram_shard*.pt`, `pairs_shard*.json`, and `cost_shard*.npz` files.

## Adaptive thresholds

Threshold calibration measures cache counts, not evaluation-set quality.
For image models, `flux/native_gate_calibrate.py` and
`qwen_image/native_gate_calibrate.py` accept the same four mode names:
`SeaCache`, `TeaCache`, `SenCache`, and `DiCache`. Supply a comma-separated
`--thresholds` grid and keep each sweep in a separate directory.

For example, calibrate FLUX SenCache at the K37 skip settings:

```bash
python flux/native_gate_calibrate.py --mode SenCache \
  --model_id "$FLUX_MODEL" --seed 42 \
  --prompt_file resources/baseline_exact/image_calibration50.txt --limit 50 \
  --sencache_sensitivity "$ASSETS_ROOT/flux/sencache_q90.npz" \
  --sencache_threshold_start 0.005 --sencache_max_skip 39 \
  --sencache_switch_ratio 0.2 --thresholds 3.2,3.6,4.0 \
  --output_dir "$ASSETS_ROOT/flux/sweeps/sen_k37"
```

Repeat per target using the settings in [baselines.md](baselines.md). The grid
above is an example around the final operating point, not the full historical
search. Use the actual subdirectories printed by the collector when selecting:

```bash
python analysis/select_native_gate_threshold.py \
  --candidate lower="$LOWER_CANDIDATE_DIR" \
  --candidate middle="$MIDDLE_CANDIDATE_DIR" \
  --candidate upper="$UPPER_CANDIDATE_DIR" \
  --target 37 --mean-tolerance 0.3 --selection-policy mean-first \
  --out "$ASSETS_ROOT/flux/sen_k37_selection.json"
```

For Qwen TeaCache, add `--teacache_coefficients` to its calibration command.
For DiCache, preserve both `--dicache_ret_ratio` and `--dicache_probe_depth`
with the selected threshold. K41 requires a shorter initial full-compute
window than `ret_ratio=0.2` allows. Thresholds for the three smaller image
datasets and DiffusionDB were calibrated separately. The final SenCache
recalibration supersedes its older threshold entries.

Video thresholds can be calibrated by running the appropriate
`baseline_screen_runner.py` on `--prompt_file` calibration lists, without
`--matrix_config`, and varying `--threshold`. Use
`resources/hunyuan_video/calibration/penguin_b-cal48.txt` and `vbench_cal48.txt`
for the corresponding datasets. Use the same method parameters during the
sweep and evaluation, then select by the realized cache counts.

## Video assets

Use `resources/baseline_exact/hunyuan_calibration50.txt` and
`hunyuan_holdout10.txt` for both video models. Select one model:

```bash
video_model=hunyuan_video
model_args=(--model_base "$HUNYUAN_MODEL")
bud_search=analysis/search_budcache_hunyuan.py
# For Wan, use these three assignments instead:
# video_model=wan21
# model_args=(--ckpt_dir "$WAN_MODEL")
# bud_search=analysis/search_budcache_wan21.py

assets="$ASSETS_ROOT/$video_model"
train_prompts=resources/baseline_exact/hunyuan_calibration50.txt
holdout_prompts=resources/baseline_exact/hunyuan_holdout10.txt
mkdir -p "$assets/calibration"
python "$video_model/l2p_collect.py" "${model_args[@]}" \
  --prompt_file "$train_prompts" --limit 50 --seed 20260723 \
  --gram_out "$assets/calibration/l2p_train.pt"
python "$video_model/l2p_collect.py" "${model_args[@]}" \
  --prompt_file "$holdout_prompts" --limit 10 --seed 20260723 \
  --gram_out "$assets/calibration/l2p_holdout.pt"
python analysis/solve_l2p_grams.py \
  --gram "$assets/calibration/l2p_train.pt" \
  --holdout_gram "$assets/calibration/l2p_holdout.pt" \
  --ridge 1e-5 --out "$assets/l2p_weights.pt"
python "$video_model/sencache_sensitivity_runner.py" "${model_args[@]}" \
  --prompt_file "$train_prompts" --limit 50 --seed 20260723 \
  --output_dir "$assets/calibration/sen"
python analysis/merge_sencache_sensitivity.py \
  --acc "$assets/calibration/sen" --backbone "$video_model" \
  --aggregation q90 --out "$assets/sencache_q90.npz"
python "$video_model/meancache_calibrate.py" "${model_args[@]}" \
  --prompt_file "$train_prompts" --limit 50 --seed 20260723 \
  --out "$assets/calibration/meancache.npz"
for cache_count in 29 37 41; do
  schedule_dir="$assets/schedules/k$cache_count"
  mkdir -p "$schedule_dir"
  python "analysis/$video_model/build_shared_schedule.py" \
    --budget "K$cache_count" --output "$schedule_dir/shared_predictor.json"
  python analysis/build_meancache_schedule.py --model "$video_model" \
    --cost_shards "$assets/calibration/meancache.npz" \
    --cache_count "$cache_count" --output "$schedule_dir/meancache.json"
  python "$bud_search" "${model_args[@]}" \
    --prompt_file "$train_prompts" --limit 2 \
    --seed 20260723 --search_seed 20260723 --cache_count "$cache_count" \
    --output "$schedule_dir/budcache.json"
done
```

## Bind rebuilt video assets to a new configuration

The historical matrix configurations record hashes of the original fitted
files. A newly fitted file can have a different hash even when generated by the
same procedure. Build a new configuration that records your local assets and
schedules. Do not disable the runner's hash checks or replace historical files
in place.

The existing builders accept repeated `--threshold`, `--method-param`,
`--asset`, `--schedule`, and `--shared-schedule` arguments. The HunyuanVideo
builder also accepts `--calibration`. Both require complete settings for all
methods, datasets, and cache counts. The following constructs these arguments
from a parameter configuration and the assets built above:

```bash
export VIDEO_ASSET_MODEL="$video_model"
export VIDEO_ASSET_DIR="$assets"
export VIDEO_PARAMETER_CONFIG="resources/$video_model/baseline_matrix_config.v2.json"
python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

model = os.environ["VIDEO_ASSET_MODEL"]
assets = Path(os.environ["VIDEO_ASSET_DIR"])
parameters = json.loads(Path(os.environ["VIDEO_PARAMETER_CONFIG"]).read_text())
cmd = [sys.executable, f"analysis/{model}/build_baseline_matrix_config.py",
       "--output", str(assets / "matrix.json")]
for method, datasets in parameters["thresholds"].items():
    for dataset, budgets in datasets.items():
        for budget, value in budgets.items():
            cmd += ["--threshold", f"{method}:{dataset}:{budget}={value}"]
for method, datasets in parameters["method_params"].items():
    for dataset, budgets in datasets.items():
        for budget, knobs in budgets.items():
            for knob, value in knobs.items():
                cmd += ["--method-param", f"{method}:{dataset}:{budget}:{knob}={value}"]
cmd += ["--asset", f"l2p:l2p_weights={assets / 'l2p_weights.pt'}",
        "--asset", f"sencache:sencache_sensitivity_path={assets / 'sencache_q90.npz'}"]
for count in (29, 37, 41):
    folder = assets / "schedules" / f"k{count}"
    for method in ("budcache", "meancache"):
        cmd += ["--schedule", f"{method}:K{count}={folder / (method + '.json')}"]
    cmd += ["--shared-schedule", f"K{count}={folder / 'shared_predictor.json'}"]
if model == "hunyuan_video":
    for dataset, row in parameters["threshold_calibration"].items():
        cmd += ["--calibration", f"{dataset}={row['file']}"]
subprocess.run(cmd, check=True)
PY
```

This example retains the recorded thresholds and other numerical settings but
replaces the fitted assets and searched schedules. If you recalibrate
thresholds, point `VIDEO_PARAMETER_CONFIG` to your updated parameter file
instead. The builder validates the result and hashes the new asset files.
Use that new `matrix.json` for generation and pass the matching L2P or SenCache
asset path. Record the measured cache counts and quality from your own run.
