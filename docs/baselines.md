# Baseline generation and evaluation

This repository provides code to rebuild the experiments. Model weights,
fitted predictors, calibration outputs, generated images, and generated videos
are not distributed. Download the models from their official sources and build
the method assets as described in [assets.md](assets.md).

Run commands from the repository root. Examples below use local paths in
`FLUX_MODEL`, `QWEN_MODEL`, `HUNYUAN_MODEL`, and `WAN_MODEL`, with writable
`ASSETS_ROOT` and `OUTPUT_ROOT` directories. Generation requires CUDA. Use the
model-specific environments described in the repository installation notes.

## Protocol

Every model uses 50 denoising steps and target cache counts of 29, 37, and 41,
corresponding to cache ratios of 0.58, 0.74, and 0.82. Fixed schedules execute
exactly that many cached steps. Adaptive methods retain their native decisions.
Their thresholds are calibrated to bring the mean cache count near the target.
The same cache count does not imply the same latency or number of operations.

| Model | Resolution | Frames | Sampling parameters | Base seeds |
|---|---|---:|---|---|
| FLUX.1-dev | 1024 × 1024 | 1 | Guidance 3.5, BF16 | 41, 42, 43 |
| Qwen-Image | 1328 × 1328 | 1 | True CFG 4.0, negative prompt `" "`, BF16 | 42, 100042, 200042 |
| HunyuanVideo | 640 × 480 | 65 | `HY-CachePaper-480` protocol | See below |
| Wan2.1-T2V-1.3B | 832 × 480 | 65 | UniPC, shift 5.0, guidance 5.0, BF16 | See below |

For both video models, use base seeds 54, 55, and 56 on Penguin599 and 42, 43,
and 44 on VBench944. Each runner uses `base_seed + prompt_index`. A cached
sample is compared with an uncached sample from the same model, prompt, seed,
and generation settings. Keep prompt order unchanged when sharding.

The image comparison contains ten methods. The video comparison contains nine,
with DPCache omitted. Runner mode names are case-sensitive:

| Method | FLUX | Qwen-Image | HunyuanVideo | Wan2.1 |
|---|---|---|---|---|
| No cache | `original_control` | `original` | `original` | `original` |
| SeaCache | `seacache_native` | `SeaCache` | `seacache` | `seacache` |
| TeaCache | `teacache_native` | `TeaCache` | `teacache` | `teacache` |
| SenCache | `sencache_native` | `SenCache` | `sencache` | `sencache` |
| DiCache | `dicache_native` | `DiCache` | `dicache` | `dicache` |
| TaylorSeer O1 | `taylorseer_fine_exact` | `TaylorSeer_fine` | `taylorseer_exact` | `taylorseer_o1` |
| HiCache O2 | `hicache_fine_exact` | `HiCache_fine` | `hicache_exact` | `hicache_o2` |
| L2P | `l2p_output_exact` | `L2P_output` | `l2p_output_exact` | `l2p` |
| DPCache | `dpcache_exact` | `DPCache` | Not evaluated | Not evaluated |
| BudCache | `budcache_exact` | `BudCache` | `reuse_exact` | `budcache` |
| MeanCache | `meancache_exact` | `MeanCache` | `meancache_exact` | `meancache` |

## Image generation

The image prompt files are:

| Dataset | File under `resources/prompts/` | Prompts |
|---|---|---:|
| DrawBench | `prompt.txt` | 200 |
| PartiPrompts | `partiprompts_full_eval1632_seed42.txt` | 1,632 |
| GenEval-style | `geneval_seed43_n100.txt` | 553 |
| DiffusionDB | `diffusiondb_2m_clean_10000_seed42.txt` | 10,000 |

The GenEval-style filename is historical. The file contains 553 prompts.
Qwen uses `reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt`
for DrawBench. Its text differs slightly from the FLUX file. Use the same
model-specific file for generation and evaluation.
The three smaller datasets and DiffusionDB have separate native-gate
calibration inputs. SenCache uses the later recalibration described below.

Generate an uncached FLUX reference and a BudCache result:

```bash
python flux/baseline_screen_runner.py \
  --mode original_control --model_id "$FLUX_MODEL" \
  --prompt_file resources/prompts/prompt.txt \
  --output_dir "$OUTPUT_ROOT/flux/full_s41" --seed 41

python flux/baseline_screen_runner.py \
  --mode budcache_exact --model_id "$FLUX_MODEL" \
  --cache_count 29 \
  --schedule_json "$ASSETS_ROOT/flux/schedules/k29/budcache.json" \
  --prompt_file resources/prompts/prompt.txt \
  --output_dir "$OUTPUT_ROOT/flux/budcache_k29_s41" --seed 41
```

For Qwen, use `qwen_image/runner.py`. Fixed methods take `--schedule_file`
and `--exact_cache_count` instead of FLUX's `--schedule_json` and
`--cache_count`:

```bash
python qwen_image/runner.py \
  --mode MeanCache --model_id "$QWEN_MODEL" \
  --exact_cache_count 29 \
  --schedule_file "$ASSETS_ROOT/qwen_image/schedules/k29/meancache.json" \
  --prompt_file reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt \
  --output_dir "$OUTPUT_ROOT/qwen/meancache_k29_s42" --seed 42 \
  --width 1328 --height 1328 --true_cfg_scale 4.0 --negative_prompt ' '
```

TaylorSeer, HiCache, and L2P share `shared_predictor.json` at each cache count.
Use `--max_order 1` for TaylorSeer and `--max_order 2 --hicache_sigma 0.5`
for HiCache, with `--first_enhance 3`. L2P additionally needs
`--l2p_weights`. MeanCache schedule files include `jvp_spans`, which must be
preserved. Qwen TeaCache needs `--teacache_coeff_source qwen_fitted` and
`--teacache_coefficients`, pointing to the fitted polynomial JSON.

Adaptive FLUX modes take `--threshold`. Qwen uses `--seacache_thresh`,
`--teacache_thresh`, `--sencache_threshold_main`, or `--dicache_thresh`.
Pass the calibrated method parameters explicitly. Both runners support
`--shard_idx`, `--shard_count`, and `--resume`.

### SenCache recalibration

Do not use the old main-threshold defaults for the paper comparison. The final
image settings were:

| Model | K | Start threshold | Main threshold | Maximum consecutive skips | Switch ratio |
|---|---:|---:|---:|---:|---:|
| FLUX | 29 | 0.005 | 0.6 | 10 | 0.20 |
| FLUX | 37 | 0.005 | 3.6 | 39 | 0.20 |
| FLUX | 41 | 0.005 | 3.6 | 43 | 0.12 |
| Qwen | 29 | 0.005 | 0.36 | 10 | 0.20 |
| Qwen | 37 | 0.005 | 1.8 | 39 | 0.20 |
| Qwen | 41 | 0.005 | 1.8 | 43 | 0.12 |

The FLUX baseline runner takes `--sencache_sensitivity_path`,
`--sencache_threshold_start`, `--threshold`, `--sencache_max_skip`, and
`--sencache_switch_ratio`. Qwen uses `--sencache_sensitivity` and
`--sencache_threshold_main` for the first and third of these options. Both use
`--first_enhance 1` for the image comparison. Rebuild the sensitivity table
before running these settings, and report the realized cache counts.

Wan's corresponding main thresholds are 0.31, 1.6, and 2.9. It uses start
threshold 0.005, the same skip limits and switch ratios shown above, and
`sencache_first_enhance=3`. HunyuanVideo retains its separate settings:
start thresholds 0.2, 0.2, and 1.2, main thresholds 0.5, 3.0, and 3.0,
`first_enhance=3`, maximum skip count 10, and switch ratio 0.2.

## Video generation

Use the structured evaluation manifests in
`resources/hunyuan_video/evaluation/` for both models. Penguin599 contains two
prompts with embedded newlines. Converting it to a plain text prompt list
changes indices and breaks seed matching.

The following examples use a matrix configuration rebuilt with your local
calibration assets, as described in [assets.md](assets.md):

```bash
python hunyuan_video/baseline_screen_runner.py \
  --mode meancache_exact --model_base "$HUNYUAN_MODEL" \
  --matrix_config "$ASSETS_ROOT/hunyuan_video/matrix.json" \
  --dataset penguin599 --budget K29 \
  --prompt_manifest resources/hunyuan_video/evaluation/penguin599.json \
  --output_dir "$OUTPUT_ROOT/hunyuan/meancache_k29_s54" --seed 54

python wan21/baseline_screen_runner.py \
  --mode meancache --ckpt_dir "$WAN_MODEL" \
  --matrix_config "$ASSETS_ROOT/wan21/matrix.json" \
  --dataset penguin599 --budget K29 \
  --prompt_manifest resources/hunyuan_video/evaluation/penguin599.json \
  --output_dir "$OUTPUT_ROOT/wan/meancache_k29_s54" --seed 54
```

For uncached references, use `--mode original` without `--matrix_config`,
`--dataset`, or `--budget`. For SenCache or L2P, also pass
`--sencache_sensitivity_path` or `--l2p_weights`. The matrix configuration
checks the supplied asset's hash and supplies the method parameters.
Wan requires FlashAttention and the upstream Wan package under
`reference/taylorseer/code/TaylorSeer-Wan2.1`. HunyuanVideo uses the upstream
package under `reference/hunyuan_video/code`.

## Evaluation

Images are saved as `img_<index>.png`. Evaluate them against the matching
uncached directory:

```bash
python evaluation/eval_metrics.py \
  --acc "$OUTPUT_ROOT/flux/budcache_k29_s41" \
  --gt "$OUTPUT_ROOT/flux/full_s41" \
  --prompts resources/prompts/prompt.txt \
  --metrics psnr ssim lpips clip image_reward
```

Videos are saved as `video_<index>.mp4`. The video evaluator takes JSONL rows
with contiguous `task_idx` values and a `prompt` field. This is a different
format from the generator's JSON manifest. Prepare it without losing embedded
newlines:

```bash
python scripts/prepare_video_eval_manifest.py \
  --input resources/hunyuan_video/evaluation/penguin599.json \
  --output "$OUTPUT_ROOT/penguin599_eval.jsonl"

python evaluation/eval_video_metrics.py \
  --acc "$OUTPUT_ROOT/wan/meancache_k29_s54" \
  --gt "$OUTPUT_ROOT/wan/full_s54" \
  --task_manifest "$OUTPUT_ROOT/penguin599_eval.jsonl" \
  --frame_indices all \
  --metrics psnr ssim lpips temporal_lpips_delta
```

`--frame_indices all` is required for the baseline tables. The default samples
only five frames. Use AlexNet LPIPS, the default in both evaluators.

VBench applies only to VBench944. Use
`analysis/hunyuan_video/baseline_vbench_staging.py` or
`analysis/wan21/baseline_vbench_staging.py` to stage the three seed directories
with official VBench filenames. Pass each directory through `--cell`, its
repeat index through `--seed-index`, and supply `--prompt-manifest`,
`--staging-dir`, and `--out`. Then run
`analysis/hunyuan_video/run_baseline_vbench.py` with `--staging-dir`,
`--staging-manifest`, `--name`, and `--out-dir`. Install the official VBench
code and evaluation models separately.
