# Preparing benchmark prompts

Benchmark prompt files are downloaded or rebuilt locally. The repository does
not distribute the original benchmark data, generated images, or generated
videos. Run from the repository root:

```bash
python scripts/prepare_data.py --dataset image
python scripts/prepare_data.py --dataset video
```

If you will run video generation, complete `scripts/bootstrap_sources.py`
first so the upstream source directories contain the full model code.

`image` prepares DrawBench, PartiPrompts, and GenEval-style prompts. `video`
prepares Penguin599 and VBench944, including their 48-prompt threshold
calibration lists. `light` prepares both groups. Individual image choices are
`drawbench`, `parti`, and `geneval`. These commands download only small prompt
or metadata files, not model weights or reference media.

The GenEval builder needs NumPy. DiffusionDB additionally needs PyArrow or a
working pandas Parquet backend. Video metadata preparation is CPU-only.
Install the Parquet backend with `python -m pip install pyarrow` before
preparing DiffusionDB.

## Sources and generated files

| Dataset | Public source | Prepared file |
|---|---|---|
| FLUX DrawBench | HiCache, `resources/prompts/prompt.txt` | `resources/prompts/prompt.txt` |
| Qwen DrawBench | HiCache, `models/qwen_image/prompts/DrawBench200.txt` | `reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt` |
| PartiPrompts | Google Research Parti, `PartiPrompts.tsv` | `resources/prompts/partiprompts_full_eval1632_seed42.txt` |
| GenEval-style | GenEval object names and the repository's adapted generator | `resources/prompts/geneval_seed43_n100.txt` |
| Penguin599 | Tencent HunyuanVideo, `assets/PenguinVideoBenchmark.csv` | `resources/hunyuan_video/evaluation/penguin599.json` |
| VBench944 | VBench, `vbench/VBench_full_info.json` | `resources/hunyuan_video/evaluation/vbench944.json` |

Source repositories:
[HiCache](https://github.com/fenglang918/HiCache),
[Parti](https://github.com/google-research/parti),
[GenEval](https://github.com/djghosh13/geneval),
[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), and
[VBench](https://github.com/Vchitect/VBench).

The script uses fixed commits for HiCache, GenEval, HunyuanVideo, and VBench.
Parti uses its official public URL with the SHA256 of the input used in our
experiments. Every source is checked against its recorded SHA256. A changed
download is rejected rather than silently substituted. You can provide an
already downloaded matching file with `--source`, as shown below.

### Preserve text, order, and indices

- The two DrawBench files each contain 200 prompts but are not byte-identical.
  The Qwen file includes quotation marks in some prompts. Use the model's
  corresponding file for both generation and evaluation.
- Parti's 1,632 prompts use the deterministic Category/Challenge ordering from
  `analysis/build_parti_prompt_splits.py` with seed 42. Extracting the first TSV
  column directly gives a different order and therefore different generation
  seeds.
- The GenEval-style builder uses seed 43 and 100 samples per task before
  deduplication. Its final output contains **553 prompts**, not 100. The
  historical `n100` filename is retained because experiment commands refer to
  it. This prompt set is used for cache-fidelity metrics, not the official
  detector-based GenEval score.
- Penguin's 600 CSV rows reduce to 599 unique prompts. The prepared manifest
  retains exact text and the established ordering. Two prompts contain
  embedded newlines, so do not convert this manifest to a plain text list.
- VBench's 946 metadata rows reduce to 944 exact-text entries. Their official
  dimension associations are retained for evaluation.

The preparation script checks the generated image prompt hashes and both
video manifest hashes. Generation uses `base_seed + prompt_index`. Do not
shuffle the prepared files or restart indices within a shard. See
[baselines.md](baselines.md) for the generation seeds and model settings.

## DiffusionDB is opt-in

DiffusionDB preparation reads the original two-million-image metadata table,
which is a multi-gigabyte download. It does not download the image archives.
Use a local metadata file when available:

```bash
python scripts/prepare_data.py --dataset diffusiondb \
  --source diffusiondb=/path/to/metadata.parquet --offline
```

To explicitly permit the metadata download:

```bash
python scripts/prepare_data.py --dataset diffusiondb --allow-large-download
```

The source is the official
[DiffusionDB metadata](https://huggingface.co/datasets/poloclub/diffusiondb).
The SHA256 must match the recorded input. The command runs
`analysis/build_diffusiondb_prompt_file.py` twice:

1. It filters and selects 10,000 evaluation prompts with seed 42.
2. It selects 512 calibration prompts with seed 20260729, excluding the
   normalized evaluation prompts.

Outputs are `resources/prompts/diffusiondb_2m_clean_10000_seed42.txt` and
`resources/prompts/diffusiondb_2m_clean_calib512_seed20260729.txt`, with metadata
and selection manifests beside them. Filtering and selection are deterministic.
Both final prompt hashes are checked.

`--dataset all` also includes DiffusionDB and therefore requires either a
matching local parquet or `--allow-large-download`. `image` and `light` never
download DiffusionDB.

## Offline sources

Repeat `--source NAME=FILE` to use local files. Names are `drawbench_flux`,
`drawbench_qwen`, `parti`, `geneval`, `penguin`, `vbench`, and `diffusiondb`.
For GenEval, supply `object_names.txt`, not a generated prompt list.

```bash
python scripts/prepare_data.py --dataset image --offline \
  --source drawbench_flux=/path/to/flux_drawbench.txt \
  --source drawbench_qwen=/path/to/qwen_drawbench.txt \
  --source parti=/path/to/PartiPrompts.tsv \
  --source geneval=/path/to/object_names.txt
```

Matching files already present at their expected paths are reused. Existing
source files with different hashes are not overwritten. For a custom dataset,
pass your own prompt file directly to a generation runner instead of replacing
these benchmark files.

Video preparation calls the existing builders in this order:
`build_stage_a_vbench.py`, `build_stage_a_manifests.py`,
`build_evaluation_prompts.py`, and `build_calibration_prompts.py`, all under
`analysis/hunyuan_video/`. If the generation protocol file is absent, it first
calls `build_stage_a_protocols.py`. The upstream source files can come from
`scripts/bootstrap_sources.py` or the checked raw downloads used here.

For video reconstruction metrics, use
`scripts/prepare_video_eval_manifest.py` to convert the prepared JSON manifest
to the evaluator's JSONL format. This conversion preserves prompt order and
embedded newlines. Keep the JSON format for video generation.
