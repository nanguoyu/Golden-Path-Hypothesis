# Schedule search and sharing

These experiments test how well schedules transfer across prompts. Schedule
reuse compares a shared schedule with a method's adaptive decisions. Coverage
compares a shared schedule with each prompt's best evaluated schedule. The
search experiments select schedules using a small set of examples, then measure
their quality on new prompts.

Start with [schedule reuse](#schedule-recurrence-reuse-and-coverage),
[search and validation](#search-and-validate-schedules), or
[exhaustive evaluation](#exhaustive-search-over-1370754-schedules).

Run commands from the repository root after installing the model and evaluation
dependencies. This repository contains code, experiment configurations, schedule
definitions, and split indices. Model weights, generated outputs, per-example
metrics, and the exhaustive score table must be obtained or generated separately.

## Search and validate schedules

`resources/schedule_search/config.v1.json` fixes eight scoring prompt-seed pairs,
50 validation captions, warm-start schedules, search seeds, budgets, and annealing
temperatures. The captions and warm-start bitstrings are stored in the file.
The search uses residual reuse and forces steps 0, 1, 2, and 49 to be full steps.
`K` is the number of cached steps out of 50. A schedule bitstring uses `1` for a
cached step and `0` for a full step.

For one model, cache count, and objective:

```bash
python flux/schedule_search_runner.py \
  --k 41 --algorithm hill --objective psnr \
  --output_dir outputs/search/flux_k41_hill
```

Use `random`, `hill`, `anneal`, or `greedy` for `--algorithm`, and repeat for
`--k 29`, `37`, and `41`. Use `qwen_image/schedule_search_runner.py` for Qwen-Image.
The objective experiments repeat hill climbing and annealing with `--objective
lpips`. Separate output directories keep these runs distinct. `--gpus N` splits
the examples of one candidate evaluation across N GPUs. It does not change the
search procedure. The frozen configuration already contains the temperatures.

After the searches for a model, cache count, and objective finish, rescore their
candidate schedules on the 50 validation captions:

```bash
python flux/schedule_search_runner.py \
  --k 41 --objective psnr --arbitrate \
  --candidate_summaries \
    outputs/search/flux_k41_random/summary.json \
    outputs/search/flux_k41_hill/summary.json \
    outputs/search/flux_k41_anneal/summary.json \
    outputs/search/flux_k41_greedy/summary.json \
  --output_dir outputs/search/flux_k41_validation
```

The resulting `arbitration.json` contains the selected schedules and their
bitstrings under `arbitration.delivery`. For the objective experiment, pass only
summaries with the same objective and add `--objective lpips`.

To use the existing delivery-list utility, place each newly generated validation
record at `outputs/arbitration/<model>_k<K>.json`, or
`<model>_k<K>_lpips.json` for LPIPS. Then run:

```bash
python analysis/schedule_search_delivery.py \
  --arbitration_dir outputs/arbitration --out outputs/delivery.txt
python RUN/schedule_search_eval_cells.py materialize \
  --schedules outputs/delivery.txt --schedule_dir outputs/schedules
```

The second command writes one bitstring file per selected schedule. The delivery
utility accepts `--objective lpips` for the LPIPS records.

## Evaluate selected schedules

Use the image SPX runners with `--payload reuse`, as described in [spx.md](spx.md).
Evaluate all selected schedules on the four image prompt sets and three seed
streams per model. FLUX uses base seeds 41, 42, and 43. Qwen uses 42, 100042, and
200042. In these runners, the image seed is the base seed plus the prompt index.
Each cached output is compared with its own full-compute reference.

Store generation and metric outputs as:

```text
outputs/search_eval/<dataset>/<model>/k<K>/<schedule>xreuse_s<seed>/
```

Each cell needs `metrics.json` from `evaluation/eval_metrics.py`. Then collect the
new metrics:

```bash
python analysis/stage_schedule_search_perprompt.py \
  --root outputs/search_eval --out resources/schedule_search
python analysis/schedule_search_results.py
```

The result analysis also reads the newly generated SPX reference tables in
`resources/spx/`, search summaries in `resources/schedule_search/search/`, and
validation records in `resources/schedule_search/arbitration/`. These result
directories are not distributed. Populate them from the corresponding runs
before requesting the complete comparisons.

## Exhaustive search over 1,370,754 schedules

This is a separate GPU experiment. It evaluates every schedule with 41 cached
steps, four forced full steps, and five further full steps chosen from 3 through
48. The four scoring examples are PartiPrompts indices 5, 8, 9, and 15, with
base seed 42. The actual seeds are 42 plus the prompt indices.

```bash
python analysis/build_exhaustive_k41_conditioning.py \
  --output outputs/exhaustive/conditioning.pt
python flux/exhaustive_k41_runner.py \
  --conditioning_file outputs/exhaustive/conditioning.pt \
  --output_dir outputs/exhaustive/discovery4 \
  --rank_start 0 --rank_end 1370754 \
  --shard_idx 0 --shard_count 1
python analysis/merge_exhaustive_k41.py \
  --parts_dir outputs/exhaustive/discovery4/parts \
  --output_dir outputs/exhaustive/merged
```

The single-shard command covers the full experiment, not a short example run.
Parallel workers must share the same conditioning file and use distinct
`--shard_idx` values with the same `--shard_count`. Merge their complete `parts`
after all workers finish. The merge creates `merged.tsv.gz`, `summary.json`, and
`candidates/`. The conditioning file and score table are generated artifacts,
not bundled inputs.

### Evaluate transfer on new prompts

The lightweight candidate manifest and bitstring files identify the schedules
used in the paper. They permit evaluation without rerunning discovery:

```bash
python flux/exhaustive_k41_family_runner.py \
  --prompt_file resources/prompts/prompt.txt \
  --base_seed 42 \
  --output_dir outputs/exhaustive/heldout/drawbench_s42
```

Repeat for DrawBench, GenEval-style, and PartiPrompts at seeds 41, 42, and 43.
Use dataset directory names `drawbench`, `geneval`, and `parti`. The coverage
analysis reads directories named `<dataset>_s<seed>/pairs/` and
excludes all four discovery prompt indices from the PartiPrompts population:

```bash
python analysis/gph_deep_pool_check.py \
  --data_root outputs/exhaustive/heldout \
  --out outputs/exhaustive/deep_pool_check
```

The family runner checks the recorded environment: Python 3.12.13, PyTorch
2.12.0, diffusers 0.38.0, transformers 5.8.1, tokenizers 0.22.2, NumPy 2.4.4,
and Pillow 10.4.0. Use that environment for this runner.

To reconstruct candidate selection after a new exhaustive run, the merge's
`--include_schedule NAME=PATH` arguments must also include the 11 comparison
schedules: `dicache_top1`, `meancache`, `ham2f`, `ham4f`, `rand_2`, `uniform`,
`budcache`, `rand_1`, `gpf_reuse_e05_1`, `dpcache`, and `dp_rho2`. The merge's
default high-score and random subsets alone are not the paper's candidate set.

### Search algorithms using the stored scores

This CPU experiment starts only after the exhaustive run has generated the score
table. It looks up scores instead of generating new images:

```bash
python analysis/golden_path_search_bench.py \
  --table outputs/exhaustive/merged/merged.tsv.gz \
  --summary outputs/exhaustive/merged/summary.json \
  --manifest outputs/exhaustive/merged/candidates/candidate_manifest.tsv \
  --out_json outputs/search_bench/results.json \
  --out_fig outputs/search_bench/curves.png
```

## Schedule recurrence, reuse, and coverage

The baseline runners save `decisions_*.json`. These record the schedule used for
each prompt. `analysis/analyze_native_schedule_paths.py` collects their counts
and aggregates them across runs. Its `collect` command reads the baseline run
inventory supplied through `--run-stats`; use `--help` for that interface.

For image schedule reuse, select schedules only from the discovery role in
`resources/sp_cross_schedules/parti_spx_splits.v1.json`. The remaining 1,088
prompts are the validation and test roles combined. For each model, method, and
cache count, collect the three new native runs:

```bash
python analysis/build_discovery_path_counts.py \
  --split resources/sp_cross_schedules/parti_spx_splits.v1.json \
  --model flux --method seacache --target_k 29 \
  --run outputs/native/flux_k29_seacache_s41 \
  --run outputs/native/flux_k29_seacache_s42 \
  --run outputs/native/flux_k29_seacache_s43 \
  --output outputs/discovery_path_counts.tsv --append
```

Repeat for every required combination. When rebuilding schedules, pass this
discovery-only table explicitly:

```bash
python analysis/build_sp_cross_schedules.py \
  --path_counts outputs/discovery_path_counts.tsv \
  --dataset parti_discovery \
  --fixed_paths resources/cross_model_multiseed_stage_e_native_paths/fixed_paths.tsv \
  --out_dir outputs/reselected_schedules
```

Do not use pooled evaluation counts to select a schedule and then describe its
quality on those same prompts as held-out quality. Run the selected schedules
with their methods' original approximation policies and evaluate the outputs.
For DiCache this is `di_two_anchor`; for SeaCache, TeaCache, and SenCache it is
`reuse`.

After collecting the newly generated metrics as explained in [spx.md](spx.md):

```bash
python analysis/replay_intervals.py --out outputs/replay_intervals.json
python analysis/gph_oracle_check.py
python analysis/spx_coverage_deep.py --out_dir outputs/coverage_deep
```

These analyses require the generated tables at their documented `resources/`
paths. The first compares fixed and adaptive schedules on held-out image
prompts where their decisions differ. The other two compute coverage against
each prompt's best evaluated schedule, using the original and enlarged candidate
sets. They also read video tables when computing the four-model results. The
video population uses 150 prompts per dataset and one seed stream per dataset.
`analysis/video_replay_reselect.py` separately checks schedule selection after
excluding those evaluation prompts. Its inputs are the new native schedule
counts and per-prompt decision tables. No historical metric tables are bundled.

`gph_oracle_check.py` also reports the original four-example exhaustive check.
That part requires the newly generated exhaustive `summary.json` and the scored
candidate manifest, placed under `resources/exhaustive_k41/formal_results/`.
The distributed schedule-only manifest has no PSNR columns and cannot supply
those measurements. The exhaustive merge produces them. These analysis scripts
include checks against the published values; `gph_oracle_check.py --no_assert`
prints newly computed values without requiring them to equal the recorded ones.

## Reading the overview figure

The [overview figure in the README](../README.md#research-in-brief) uses
FLUX.1-dev on PartiPrompts. Each cached output is compared with the full-compute
output from the same prompt and noise seed. PSNR measures pixel similarity and
LPIPS measures perceptual difference. Higher PSNR and lower LPIPS indicate closer
agreement with the full-compute output.

In **(a)**, each row shows one schedule at a 74% target cache ratio. Black or
dark-red cells mark full steps, and white cells mark cached steps. The five
most frequent schedules are shown for each adaptive method. Percentages give
their frequencies across 4,896 prompt–seed runs. The PSNR and LPIPS columns
average all runs for each method or for the searched schedule. Adaptive methods
keep their original feature approximations. The searched schedule uses
SeaCache's feature reuse.

In **(b)**, each method's most frequent schedule is selected on 544 prompts and
reused on 1,088 other prompts, keeping its feature approximation unchanged.
Both axes show output PSNR: adaptive choices on the horizontal axis and the
shared schedule on the vertical axis. The diagonal marks equal PSNR. Colors
identify target cache ratios, and large-marker shapes identify methods.
Small points show up to 300 runs per method and ratio where the two schedules
differ. Large markers average all such runs, including those not shown as
small points.
