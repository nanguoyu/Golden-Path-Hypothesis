# Experimental scale and compute

The [README](../README.md#experimental-scale) summarizes the models, methods,
datasets, sample counts, cache ratios, and generation costs. This page gives
the scope of those counts and the additional experimental subsets.

## Counting prompts and runs

A prompt is an input text. A prompt–seed run generates one output from one
prompt and one noise seed with a specified model and caching configuration.
Three seeds therefore give three runs per prompt. Different experiments reuse
some prompts, full-compute outputs, and comparison schedules, so their counts
are reported separately.

The baseline grid uses 50 denoising steps. Its three target cache ratios are
0.58, 0.74, and 0.82, corresponding to 29, 37, and 41 cached steps. The full-compute
reference uses no cached steps. Adaptive methods retain their prompt-specific
decisions and are calibrated to approach the target ratio on average.

Both image models use 12,385 prompts across four datasets and three seeds,
giving 37,155 runs per model, method, and ratio. Both video models use 1,543
prompts across two datasets and three seeds, giving 4,629 runs per model,
method, and ratio. Each video has 65 frames.

| Model | Output size | Base seeds |
|---|---|---|
| FLUX.1-dev | 1024 × 1024 | 41, 42, 43 |
| Qwen-Image | 1328 × 1328 | 42, 100042, 200042 |
| HunyuanVideo | 640 × 480, 65 frames | Penguin599: 54, 55, 56. VBench944: 42, 43, 44 |
| Wan2.1-T2V-1.3B | 832 × 480, 65 frames | Penguin599: 54, 55, 56. VBench944: 42, 43, 44 |

The actual generation seed is the base seed plus the prompt index.
The [baseline guide](baselines.md) lists the ten image methods, nine video
methods, sampling parameters, and calibrated thresholds. The
[data guide](data.md) specifies prompt preparation and ordering.

## Shared schedules and exhaustive evaluation

- **Schedule recurrence:** 891,720 image runs from SeaCache, TeaCache, SenCache,
  and DiCache. This is the four-adaptive-method subset of the baseline grid:
  two image models × four methods × three cache ratios × 37,155 runs.
- **Image schedule reuse:** select schedules using 544 PartiPrompts, then
  evaluate them on the remaining 1,088 prompts under three seeds. The comparison
  covers two image models, four adaptive methods, and three cache ratios.
  Paired quality statistics use runs where the shared and adaptive schedules
  differ.
- **Video schedule reuse:** 150 prompts from Penguin599 and 150 from VBench944
  per model, with one seed per prompt. The comparison retains 19 model–method–ratio
  combinations with different schedules and the required shared-schedule cache count.
- **Exhaustive selection:** all 1,370,754 eligible FLUX.1-dev schedules at cache
  ratio 0.82 are evaluated on four fixed prompt–seed pairs. This produces
  5,483,016 cached images.
- **Candidate evaluation:** all 337 selected and comparison schedules are
  evaluated on 7,151 runs across DrawBench, GenEval-style, and PartiPrompts,
  excluding the four selection runs. Coverage uses a stricter prompt-level
  separation: removing the four selection prompts under every seed leaves
  2,381 prompts and 7,143 runs. The two selected schedules and method comparisons
  are also evaluated on 30,000 DiffusionDB runs.

Commands are in [schedule search and sharing](search-and-sharing.md).

## Trajectories, errors, and approximation policies

Full-compute collection covers 74,310 image trajectories and 9,258 video
trajectories, totaling 83,568. Every trajectory contains 51 latent states from
50 denoising steps. The image turning and plane-direction analyses use a subset
of 120 prompts per dataset and seed, or 1,440 trajectories per image model.

The complete cached-generation error decomposition uses FLUX.1-dev on 100
PartiPrompts with one seed per prompt. SeaCache and TeaCache are each evaluated
at two ratios near 0.58 and 0.82, giving 400 cached runs. Additional full-model
predictions at the cached states measure the error contributions.

The single-step propagation experiment uses 100 DrawBench prompts with one seed
per prompt. It tests each of steps 1–49 under residual reuse, first-order Taylor
prediction, and second-order Hermite prediction. This gives 14,700 single-step
interventions. Each intervention caches only one of the 50 steps. The joint
factor comparison uses steps 1–48, giving 4,800 prompt–step measurements per policy.
See [trajectories and errors](trajectories-and-errors.md) for the measurements.

The main image schedule–policy experiment uses four schedules and five
approximation policies on both image models at each of the three cache ratios.
Each combination is evaluated on 1,632 PartiPrompts under three seeds. The
resulting grid contains 587,520 evaluated image outputs. The schedules are from
BudCache, DPCache, uniform spacing, and DiCache's most frequent choice.

The broader video schedule–policy grid uses 300 prompts per model, one seed
per prompt, and all three cache ratios. It contains 207 supported schedule–policy
configurations for HunyuanVideo and 210 for Wan2.1, giving 62,100 and 63,000
evaluated videos. Supplementary interval-average prediction comparisons add
further generations. The [schedule–policy guide](spx.md) describes the runners
and comparison grids.

## Few-example search

Search uses eight fixed scoring prompt–seed pairs. Candidate selection then
uses 50 separate COCO validation prompts. Each selected schedule is evaluated
on the four image datasets under three seeds, giving 37,155 evaluation runs.
The experiments cover FLUX.1-dev and Qwen-Image at all three cache ratios.

The candidate-evaluation budgets are 1,400, 700, and 400 per search procedure
at cache ratios 0.58, 0.74, and 0.82. These budgets include the 50 initial
schedules. Each candidate evaluation uses the eight scoring pairs.
PSNR experiments compare random sampling, hill climbing, simulated annealing,
and greedy coordinate ascent. LPIPS and the combined PSNR–LPIPS objective use
hill climbing and simulated annealing.

The scoring examples comprise five COCO captions, two prompts from a separate
DiffusionDB calibration pool, and one fixed text-rendering prompt. The validation
captions are also from COCO. The provided configuration specifies all eight
scoring pairs, the 50 validation captions, and the search parameters.
Some initial schedules were selected using the 544-prompt PartiPrompts split.
The final search evaluation includes the full PartiPrompts set.
See [search and validation](search-and-sharing.md#search-and-validate-schedules).

## Compute accounting

The following costs refer to the named experiments. Recorded generation times,
job runtimes, and latency-based estimates are identified separately.

| Experiment or stage | Hardware | GPU-hours | Accounting scope |
|---|---|---:|---|
| Exhaustive FLUX.1-dev evaluation | H100 | ≈3,600 | Estimated generation cost: 5,483,016 images × 2.37 seconds per image ÷ 3,600 |
| HunyuanVideo schedule–policy generation | H100 | 632.5 | Recorded generation times summed over 62,976 videos, including supplementary comparisons |
| Wan2.1 schedule–policy generation | H100 | 469.1 | Recorded generation times summed over 63,771 videos, including supplementary comparisons |
| HunyuanVideo adaptive-threshold calibration | H100 | 67.45 | Recorded runtime of the single-GPU calibration jobs |
| Wan2.1 adaptive-threshold calibration | H100 | 67.98 | Recorded runtime of the single-GPU calibration jobs |
| HunyuanVideo baseline reconstruction and temporal-metric evaluation | H100 | 198.1 | Recorded runtime of the metric-evaluation jobs |
| Wan2.1 baseline reconstruction and temporal-metric evaluation | H100 | 224.3 | Recorded runtime of the metric-evaluation jobs |
| HunyuanVideo baseline VBench evaluation | H100 | 20.2 | Recorded runtime of the VBench jobs |
| Wan2.1 baseline VBench evaluation | H100 | 29.5 | Recorded runtime of the VBench jobs |

The exhaustive estimate uses the measured H100 latency of a fixed FLUX.1-dev
schedule with residual reuse at cache ratio 0.82. It estimates generation alone.
The video generation sums use one H100 per video. They cover the generation
intervals stored with the outputs, excluding model loading and metric evaluation.

Offline method preparation also uses RTX 6000 Pro GPUs for HunyuanVideo:

| Preparation | FLUX.1-dev | Qwen-Image | HunyuanVideo | Wan2.1 |
|---|---|---|---|---|
| BudCache search, per cache ratio | 0.19–1.65 H100 GPU-h | 0.91–5.82 H100 GPU-h | 4.8–18.3 RTX 6000 Pro GPU-h | 1.4–8.2 H100 GPU-h |
| SenCache sensitivity preparation | Not recorded | Not recorded | 7.19 RTX 6000 Pro GPU-h | 2.87 H100 GPU-h |
| L2P predictor preparation | Not recorded | Not recorded | 3.07 RTX 6000 Pro GPU-h | 1.84 H100 GPU-h |
| MeanCache preparation | Not recorded | Not recorded | 2.53 RTX 6000 Pro GPU-h | 1.95 H100 GPU-h |

The SenCache, L2P, and MeanCache video assets are shared across the three cache
ratios. Their preparation costs are therefore listed once per model. Separate
preparation times were not recorded for those image implementations.
The image full-compute trajectory collection used RTX 6000 Pro GPUs, whereas
the paper's online latency benchmarks used H100 GPUs. Hours on different GPU
types are kept separate.

The PSNR search report contains 103.7 hours of cumulative job wall time across
24 search jobs. Some jobs used four GPUs and resumed earlier single-GPU runs.
Those wall-time summaries do not provide a complete GPU-hour total for search.
An all-experiment GPU-hour total has not been consolidated.
