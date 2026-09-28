# Schedule and approximation-policy comparisons

The schedule chooses where to cache, while the approximation policy determines
how features are reused or predicted. These experiments test whether changing
the policy improves a schedule's output quality and whether the effect differs
across schedules.

SPX denotes the experiments that hold a schedule fixed while changing the
approximation policy, or hold the policy fixed while changing the schedule.
Commands run from the repository root. Download model weights and generate
full-compute references as described in the model instructions. This repository
does not include generated images, videos, metric tables, or fitted weights.

## Images

The two entry points are `flux/sp_cross_runner.py` and
`qwen_image/sp_cross_runner.py`. Both read a 50-bit schedule file. A `1` means a
cached step; a `0` means a full step. Frozen schedule definitions are in
`resources/sp_cross_schedules/` and `resources/spx_supplement_schedules/`.

The five `--payload` values are:

| Value | Approximation |
| --- | --- |
| `reuse` | Reuse the most recent full-step residual |
| `taylor_o1` | First-order Taylor prediction |
| `hermite_o2` | Second-order Hermite prediction |
| `mean_avg_vel` | Interval-average velocity prediction |
| `di_two_anchor` | DiCache's two-anchor prediction |

For example, run the BudCache schedule with Hermite prediction:

```bash
python flux/sp_cross_runner.py \
  --schedule_file resources/sp_cross_schedules/flux_k29_budcache.txt \
  --payload hermite_o2 \
  --prompt_file resources/prompts/partiprompts_full_eval1632_seed42.txt \
  --seed 42 \
  --output_dir outputs/spx/flux/k29/budcachexhermite_o2_s42
```

Keep the prompt file, seed, model, and schedule unchanged when comparing
policies. For Qwen, change the runner and schedule prefix to `qwen_image` and
`qwen`. The defaults are FLUX at 1024 by 1024 with guidance 3.5 and Qwen at
1328 by 1328 with true CFG scale 4.0. Both use 50 steps and BF16. Image evaluation
uses all 1,632 PartiPrompts with three base seeds: 41/42/43 for FLUX and
42/100042/200042 for Qwen. Each actual seed is the base seed plus prompt index.
`--shard_idx` and `--shard_count` distribute independent prompts across workers.

### MeanCache spans

MeanCache's own schedule must retain its per-interval JVP spans:

```bash
python flux/sp_cross_runner.py \
  --schedule_file resources/sp_cross_schedules/flux_k41_meancache.txt \
  --payload mean_avg_vel \
  --meancache_jvp_spans resources/spx_supplement_schedules/meancache_spans/flux_k41.json \
  --prompt_file resources/prompts/partiprompts_full_eval1632_seed42.txt \
  --seed 42 \
  --output_dir outputs/spx/flux/k41/meancachexmean_avg_vel_s42
```

The span file must match the schedule. The comparisons that apply mean-velocity
prediction to other schedules use global span 4, selected by
`--meancache_jvp_span 4`. A different global span is a different experiment.

### Evaluation and image statistics

Compare each generated directory with full-compute images from the same prompts
and actual seeds:

```bash
python evaluation/eval_metrics.py \
  --acc outputs/spx/flux/k29/budcachexhermite_o2_s42 \
  --gt outputs/references/flux/parti_full_s42 \
  --prompts resources/prompts/partiprompts_full_eval1632_seed42.txt \
  --output outputs/spx/flux/k29/budcachexhermite_o2_s42/metrics.json
```

Generate the complete schedule-policy grid before computing its interaction
statistics. The directory convention is
`<root>/<model>/k<K>/<schedule>x<payload>_s<seed>/metrics.json`.

`analysis/sp_cross.py` analyzes these cell metrics without requiring native
baseline results:

```bash
python analysis/sp_cross.py --root outputs/spx --output outputs/spx_summary.json
```

For its additional comparisons with native gates, pass `--native_reference FILE`
with columns `model`, `budget_k`, `method`, `metric`, and `value`, computed from
your native evaluations. Omitting this option skips those comparisons, not the
schedule-policy decomposition.

For the paired replay and coverage analyses, collect per-image rows:

```bash
python analysis/stage_spx_perprompt.py spx \
  --spx_root outputs/spx --out resources/spx
python analysis/stage_spx_perprompt.py native \
  --discovery outputs/discovery_path_counts.tsv --out resources/spx
```

The second command follows `run_dirs` in the discovery count table produced from
your native runs. Those runs need both decisions and evaluated image metrics.
It writes `perprompt_native_flux.tsv.gz` and `perprompt_native_qwen.tsv.gz`, joining
the output metrics to each run's own realized schedule. Thus image replay uses
new native results even when the optional mean-only reference table is omitted.
Use the frozen split indices and discovery-only selection procedure in
[search-and-sharing.md](search-and-sharing.md).

`analysis/spx_supplement.py` produces the broader supplement analysis after the
new tables are ready. It additionally consumes the full-trajectory measurements,
baseline means, and control-schedule definitions used by that experiment. These
measurements must also be generated; they are not supplied result files.

## Videos

Video SPX uses `hunyuan_video/baseline_screen_runner.py` and
`wan21/baseline_screen_runner.py`. `RUN/video_spx/spx_cells.py` maps each frozen
schedule and policy to the correct runner arguments. Set `MODEL_BASE` to the
HunyuanVideo checkpoint directory or `WAN21_CKPT_DIR` to the Wan2.1 directory.

List the experiment and inspect one exact generation command:

```bash
python RUN/video_spx/spx_cells.py plan --grid full --data outputs
python RUN/video_spx/spx_cells.py argv \
  --backbone wan21 --cell sharedxreuse_penguin599_K29_s54 --data outputs
```

The `argv` command prints arguments beginning with the runner filename. Execute
them with the Python interpreter for that model. The grid uses the first 150
prompts of each dataset, base seed 54 for Penguin599 and 42 for VBench944.
Schedule JSON files under `resources/video_spx_schedules/` define the cache steps,
policy modes, and any MeanCache spans. The command generator preserves the
required warmup flags and the Hermite order. Do not replace these with generic
fixed-step arguments.

Generate full-compute references through the same command generator, with
`--references`. For example:

```bash
python RUN/video_spx/spx_cells.py argv --references \
  --backbone wan21 --cell refs_penguin599_s54 --data outputs
```

### Video evaluation

Build a task manifest from the ordered prompt JSON. This preserves prompts that
contain a newline:

```bash
python RUN/video_spx/spx_cells.py manifest \
  --dataset penguin599 --out outputs/manifests/penguin599.jsonl
```

Then evaluate the generated videos. Replace the example directories with the
`--output_dir` values printed by the generation commands:

```bash
python evaluation/eval_video_metrics.py \
  --acc outputs/wan21/spx/sharedxreuse_penguin599_K29_s54 \
  --gt outputs/wan21/spx/refs_penguin599_s54 \
  --task_manifest outputs/manifests/penguin599.jsonl \
  --backbone wan21 --eval_tag spx --limit 150 --frame_indices all \
  --metrics psnr ssim lpips temporal_lpips_delta \
  --output resources/video_spx/wan21/sharedxreuse_penguin599_K29_s54.json
```

`--frame_indices all` is required for the consecutive-frame temporal metric.
Place the new metric JSONs under `resources/video_spx/<backbone>/` with their cell
names. `analysis/video_spx.py` reads them directly for the cross-policy analysis.
Its full report also uses native baseline metrics, native schedule counts, and
full-trajectory measurements generated in the corresponding experiments.

The video coverage scripts additionally read
`resources/video_spx/<backbone>/pervideo_spx_<backbone>.tsv.gz`. This is a derived
table, not an included artifact. Generate it from the new metrics and their
corresponding `decisions_*.json` files:

```bash
python scripts/stage_video_spx_results.py \
  --backbone wan21 --input-root resources/video_spx/wan21 \
  --output resources/video_spx/wan21/pervideo_spx_wan21.tsv.gz
```

Each row retains the dataset, prompt index, prompt ID, base seed, actual seed,
realized cache count, and metrics. `temporal_delta` is the evaluator's
`temporal_lpips_delta`, unchanged. If generation directories have moved, use
`--generation-root DIR`, where `DIR` contains the cell directories. The script
does not infer realized cache counts from the nominal budget or replace missing
metrics with zeros. It does not collect generation timing. Native-gate
comparisons require the baseline per-video table, generated by the same script:

```bash
python scripts/stage_video_spx_results.py --kind baseline \
  --backbone wan21 --input-root outputs/video_baseline_metrics/wan21 \
  --output resources/video_full_results/pervideo_wan21.tsv.gz
```

Save each baseline evaluation JSON as
`<method>_<dataset>_K<K>_s<base-seed>.json`, matching its generation directory.
The script checks these labels against the generation decisions. It preserves
positive-infinite PSNR for identical baseline/reference outputs; downstream
analyses apply their existing finite-PSNR selection. Other missing or invalid
metrics are errors. Timing is not fabricated.

To generate native schedule counts and per-prompt schedules from the full video
baseline matrix:

```bash
python analysis/video_native_gate_paths.py \
  --backbone wan21 --matrix-root outputs/wan21/matrix \
  --out resources/video_native_gate_paths/wan21
python analysis/video_perprompt_paths.py \
  --backbone wan21 --matrix-root outputs/wan21/matrix \
  --out resources/video_full_results/perprompt_paths_wan21.tsv.gz
```

Repeat for HunyuanVideo. These tools read the baseline matrix's `cells/` tree,
including all three seed streams, rather than only the 150-prompt SPX subset.
