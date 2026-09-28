# The Golden Path Hypothesis: Reusable Schedules in Diffusion Caching

**Dong Wang, Wenwu Tang, Francesco Corti, Yun Cheng, Lothar Thiele, Olga Saukh**

[Research](#research-in-brief) ·
[Code and experiments](#research-and-code) ·
[Experimental scale](#experimental-scale) ·
[Code structure](#code-structure) ·
[Setup](#setup-and-running-experiments) ·
[Citation](#citation-and-sources)

## Research in brief

Diffusion caching accelerates image and video generation by reusing or predicting
features at selected denoising steps. A **cache schedule** specifies which steps
run the full model and which use these approximations. The **cache ratio** is
the fraction of steps that use an approximation. We propose the **Golden
Path Hypothesis (GPH)**: with the model and inference settings held fixed,
schedules shared across prompts can achieve output quality comparable to the
best prompt-specific schedules. Such shared schedules are **golden paths**.

We test the GPH through schedule reuse, common denoising patterns, error
propagation, and search on a few examples. This repository provides the caching
implementations, experiment runners, and analysis code used in that study.

![Adaptive methods repeatedly choose a small number of schedules. Reusing each method's most frequent schedule retains output quality on new prompts.](docs/figures/figure-1.png)

**Shared schedules retain quality across prompts.** Both panels use FLUX.1-dev
on PartiPrompts. **Left:** black or red cells mark full steps, and white cells
mark cached steps. Adaptive methods repeatedly choose a few schedules. The
searched schedule at the bottom is used in every run. **Right:** reusing each
method's most frequent schedule on new prompts gives mean output PSNR close to
that obtained when the method selects a schedule separately for each run.
Quality is measured against matching outputs
generated without caching. Higher PSNR and lower LPIPS indicate closer agreement.
[How to read the figure](docs/search-and-sharing.md#reading-the-overview-figure).

## Research and code

Choose the research question you want to investigate. The code links below open
the main implementations. The guides connect their inputs, generation and
evaluation commands, and analysis outputs.

| Research finding | Code entry points | Experiment guide |
|---|---|---|
| **Schedules can be shared.** Quality comparisons and exhaustive evaluation show that shared schedules can approach each prompt's best evaluated result. | [Schedule counts](analysis/analyze_native_schedule_paths.py), [paired quality analysis](analysis/replay_intervals.py), [exhaustive evaluation](flux/exhaustive_k41_runner.py) | [Schedule sharing and transfer](docs/search-and-sharing.md#schedule-recurrence-reuse-and-coverage) |
| **Denoising has common patterns.** Without caching, we measure how far the latent state moves and how its direction of motion changes at each step. These measurements follow similar patterns across prompts within each model. | [Image trajectory collection](flux/full_trajectory_probe.py), [image analysis](analysis/full_trajectory_analysis.py), [video analysis](analysis/video_trajectory/step_profiles.py) | [Trajectories](docs/trajectories-and-errors.md#full-compute-image-trajectories) |
| **Error propagation guides schedule evaluation.** An exact per-step error decomposition motivates scoring complete schedules by final-output quality. How features are reused or predicted also affects schedule quality. | [Error decomposition](flux/trajectory_deviation_runner.py), [single-step caching](flux/oracle_runner.py), [schedule–policy analysis](analysis/sp_cross.py) | [Error propagation](docs/trajectories-and-errors.md#current-approximation-errors-and-earlier-state-changes), [approximation policies](docs/spx.md) |
| **Search can find golden paths.** Standard search on a few examples finds schedules that retain quality on new prompts. Different quality objectives select different schedules. | [Search procedures](lib/schedule_search.py), [FLUX runner](flux/schedule_search_runner.py), [Qwen runner](qwen_image/schedule_search_runner.py) | [Search and validation](docs/search-and-sharing.md#search-and-validate-schedules) |

The [baseline guide](docs/baselines.md) covers the caching methods used in these
comparisons. The [calibration guide](docs/assets.md) explains how to prepare their
predictors, sensitivity measurements, and schedules.

<details>
<summary>Shared-schedule quality on new prompts</summary>

**One shared schedule approaches prompt-specific quality.** For FLUX.1-dev at
a cache ratio of 0.82, the selected schedule is within 1 dB of each prompt's best
evaluated PSNR on 78.9% of 2,381 new prompts.

![The selected shared schedule covers 78.9% of prompts within a 1 dB PSNR gap, compared with 63.3% for MeanCache and 69.5% for BudCache.](docs/figures/schedule-transfer.png)

The horizontal axis is the PSNR gap from each prompt's best result among 337
evaluated schedules. The vertical axis is the percentage of prompts within that
gap. PSNR is averaged over three seeds per prompt. All three schedules use the
same feature-reuse rule. [Selection and evaluation details](docs/search-and-sharing.md#evaluate-transfer-on-new-prompts).

</details>

## Experimental scale

### Models, methods, and evaluation data

The image models are [FLUX.1-dev](flux/) and [Qwen-Image](qwen_image/).
The video models are [HunyuanVideo](hunyuan_video/) and
[Wan2.1-T2V-1.3B](wan21/). The baseline experiments use **50 denoising steps**
and target cache ratios **0.58, 0.74, and 0.82**, corresponding to 29, 37, and
41 cached steps. Each video contains 65 frames.

We compare **SeaCache, TeaCache, SenCache, DiCache, TaylorSeer, HiCache, L2P,
DPCache, BudCache, and MeanCache**, alongside full-compute references. All ten
methods are evaluated on images. Video evaluation uses nine methods, excluding
DPCache. Adaptive methods are calibrated toward the target mean cache ratio.

Each baseline model–method–ratio combination uses the datasets for its modality
below. A run is one generation from one prompt and one noise seed. Baseline
evaluation uses **three seeds per prompt**.

| Modality | Dataset | Prompts | Runs per model, method, and cache ratio |
|---|---|---:|---:|
| Image | DrawBench | 200 | 600 |
| Image | GenEval-style | 553 | 1,659 |
| Image | PartiPrompts | 1,632 | 4,896 |
| Image | DiffusionDB-clean10k | 10,000 | 30,000 |
| Video | Penguin599 | 599 | 1,797 |
| Video | VBench944 | 944 | 2,832 |

This gives **12,385 prompts and 37,155 runs** per image model, method, and
cache ratio, and **1,543 prompts and 4,629 runs** per video model, method, and
cache ratio. Resolutions, sampling parameters, and seed values are in the
[baseline protocol](docs/baselines.md#protocol).

### Scale of the main experiments

The studies below use different subsets and comparisons. Prompt sets and
full-compute references are reused across experiments.

| Experiment | Models | Sample scale |
|---|---|---|
| Adaptive schedule recurrence | Both image models | 891,720 runs from four adaptive methods, three cache ratios, and four datasets |
| Shared-schedule reuse | All four models | Images: 544 selection prompts and 1,088 held-out PartiPrompts, three seeds. Videos: 150 prompts from each of two datasets, one seed per prompt |
| Exhaustive schedule evaluation | FLUX.1-dev, cache ratio 0.82 | 1,370,754 schedules × four prompt–seed pairs = 5,483,016 cached images |
| Transfer after exhaustive selection | FLUX.1-dev, cache ratio 0.82 | 337 candidate schedules; coverage measured on 2,381 new prompts with three seeds each |
| Full-compute trajectory collection | All four models | 74,310 image trajectories and 4,629 trajectories from each video model |
| Current and earlier error contributions | FLUX.1-dev | 100 PartiPrompts, SeaCache and TeaCache, two cache ratios near 0.58 and 0.82, one seed per prompt |
| Single-step error propagation | FLUX.1-dev | 100 DrawBench prompts, three approximation policies, 49 tested step positions, one seed per prompt |
| Main image schedule–policy comparison | Both image models | Four schedules × five policies × 1,632 PartiPrompts × three seeds at each of the three cache ratios |
| Few-example schedule search | Both image models, all three cache ratios | Eight scoring prompt–seed pairs, 50 separate COCO validation prompts, and 37,155 evaluation runs per selected schedule across the four image datasets |

The PSNR search compares hill climbing, simulated annealing, greedy coordinate
ascent, and random search. The LPIPS objective experiments use hill climbing
and simulated annealing. See [experimental scope](docs/experimental-scale.md)
for the candidate budgets, analysis subsets, and additional video comparisons.

### GPU-hours

Compute is reported separately for each experiment and GPU type.

| Experiment | GPU | GPU-hours | Basis |
|---|---|---:|---|
| Exhaustive FLUX.1-dev schedule evaluation | NVIDIA H100 | ≈3,600 | Estimated generation cost for 5,483,016 images at the measured 2.37 seconds per image |
| HunyuanVideo schedule–policy experiments | NVIDIA H100 | 633 | Sum of recorded generation times for 62,976 videos |
| Wan2.1 schedule–policy experiments | NVIDIA H100 | 469 | Sum of recorded generation times for 63,771 videos |

The video timings include supplementary schedule–policy comparisons. These
rows describe generation costs for the named experiments, rather than a total
for the study. [Compute accounting](docs/experimental-scale.md#compute-accounting)
also lists video calibration and metric-evaluation costs, and distinguishes
H100 from RTX 6000 Pro measurements.

## Code structure

| Directory | Role |
|---|---|
| [`lib/`](lib/) | Cache approximations, schedule definitions, and search procedures shared by the model implementations |
| [`evaluation/`](evaluation/) | Image and video quality metrics |
| [`analysis/`](analysis/) | Measurements, result aggregation, statistics, and figures |
| [`resources/`](resources/) | Provided schedules, experiment configurations, thresholds, and prompt splits |
| [`scripts/`](scripts/), [`RUN/`](RUN/) | Data preparation, external source setup, and experiment launch scripts |
| [`tests/`](tests/) | CPU unit tests for the included components |

## Setup and running experiments

```bash
git clone https://github.com/nanguoyu/Golden-Path-Hypothesis.git
cd Golden-Path-Hypothesis
```

Choose an experiment above, then follow [installation](docs/installation.md)
for its model. Prepare the [benchmark prompts](docs/data.md) and obtain the
model checkpoints from their [official sources](docs/assets.md#model-sources).
Methods that need fitted predictors or calibration measurements use the
[asset-building instructions](docs/assets.md).

Generation requires CUDA. Image generation, HunyuanVideo generation, and VBench
evaluation use separate environments. The current video generation and several
video calibration entry points require a Slurm compute node.
The [CPU analysis environment](docs/installation.md#cpu-analysis) supports
analysis of generated experiment outputs.

The [execution guide](docs/usage.md) covers data preparation, cluster commands,
output organization, and tests. It also contains an optional example that runs
a provided schedule. Each experiment guide gives the model-specific commands
and the protocol used for the paper.

## Citation and sources

Citation metadata and the author list are in [CITATION.cff](CITATION.cff).
The implementations build on the methods and tools listed in
[THIRD_PARTY.md](THIRD_PARTY.md). See [LICENSE](LICENSE) for licensing information.
