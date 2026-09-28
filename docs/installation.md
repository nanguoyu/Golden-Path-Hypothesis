# Installation

Use separate environments for image generation, HunyuanVideo generation, and
VBench evaluation. Their Transformers and NumPy versions differ. Run the commands
below from the repository root.

The source files are provided separately from pretrained models and metric
weights. Fetch the required public source trees with:

```bash
python scripts/bootstrap_sources.py all
```

Download the model checkpoints under their respective licenses and pass their
locations to the runners. Calibration commands generate SenCache and L2P assets;
those assets are not included. See the experiment guides for the model and data
arguments.

## Execution environment

The HunyuanVideo and Wan2.1 baseline generation entry points require a CUDA GPU
inside a Slurm compute job. Their MeanCache and SenCache calibration and L2P
collection scripts also check for a Slurm job. Use an allocated compute session
or the wrapper described in the [cluster instructions](usage.md#running-on-a-cluster).
These entry points do not run directly on a workstation without Slurm.
The CPU analysis scripts and unit tests do not require a Slurm allocation.

## PyTorch and CUDA

Install a PyTorch build compatible with your NVIDIA driver and GPU, together
with its matching torchvision package, before the other requirements. The
commands below use `TORCH_WHEEL_INDEX` for the index URL of that build. Set it
to the appropriate PyTorch wheel index before running them. A system CUDA
compiler is also needed when building FlashAttention from source.

The recorded image/Wan installation used Torch 2.12.0 with CUDA 13.0. The
recorded HunyuanVideo installation used Torch 2.6.0 with CUDA 12.4. These are
separate environments, not alternative versions to install together.

## FLUX, Qwen-Image, and Wan2.1

Use Python **3.12.13**. The exhaustive-family runner checks the following
versions exactly:

| Package | Version |
| --- | --- |
| torch | 2.12.0 |
| diffusers | 0.38.0 |
| transformers | 5.8.1 |
| tokenizers | 0.22.2 |
| numpy | 2.4.4 |
| Pillow | 10.4.0 |

```bash
python3.12 -m venv .venv-image
source .venv-image/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.0 torchvision --index-url "$TORCH_WHEEL_INDEX"
python -m pip install -r requirements/image.txt
python -m pip install --no-deps -e .
```

The `python3.12` executable above must be Python 3.12.13 for the exhaustive-family
experiment. Other entry points do not all enforce this check. Keep the same
runtime for paired full-compute and cached generations: tokenizer changes can
change generated outputs even when the prompt bytes and seed are unchanged.

Wan2.1 uses the public source tree fetched by the bootstrap script. Its runner
requires FlashAttention rather than silently switching to another attention
implementation. The recorded image/Wan environment used FlashAttention 2.8.3.
On a CUDA development machine, install it after PyTorch:

```bash
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Use a compatible prebuilt wheel when available. Do not run this build command
in a CPU-only environment. Do not install the entire Wan upstream requirements
file into this environment: it contains different NumPy constraints. The runtime
dependencies used here, including `easydict`, `ftfy`, and video I/O, are listed
in `requirements/image.txt`.

## HunyuanVideo

Use a new environment with Python **3.10.9**. The recorded stack uses Torch
2.6.0+cu124, Transformers 4.46.3, NumPy 1.24.4, and FlashAttention 2.6.3. The
remaining generation pins in `requirements/hunyuan.txt` follow the public
HunyuanVideo source requirements.

```bash
python3.10 -m venv .venv-hunyuan
source .venv-hunyuan/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision --index-url "$TORCH_WHEEL_INDEX"
python -m pip install -r requirements/hunyuan.txt
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.6.3 --no-build-isolation
python -m pip install --no-deps -e .
```

Use the CUDA 12.4 PyTorch index for the recorded setup and a compatible CUDA
compiler when building FlashAttention. This environment does not use the image
environment's Transformers 5.x package. Prepare and load the Hunyuan text
encoder with the same Transformers version.

## Image and video quality metrics

`evaluation/eval_metrics.py` and `evaluation/eval_video_metrics.py` load only
the requested metrics. LPIPS uses `lpips`, CLIP uses Hugging Face
`transformers.CLIPModel`, and ImageReward uses the `image-reward` package,
imported as `ImageReward`. PSNR and SSIM use scikit-image. Video decoding uses
`imageio` with its FFmpeg backend. `evaluation/eval_oracle.py` loads all three
learned metric models, including when the final report uses latent-state error.
Metric weights are downloaded or cached separately from this repository.

The image requirements include these metrics. For video metrics or VBench, use
a separate Python 3.10 environment:

```bash
python3.10 -m venv .venv-video-eval
source .venv-video-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "$TORCH_WHEEL_INDEX"
python -m pip install -r requirements/video-eval.txt
python -m pip install --no-deps -e .
```

For VBench, additionally install its own requirements after fetching its source:

```bash
python -m pip install -r reference/vbench/code/requirements.txt
```

VBench pins Transformers 4.33.2 and requires NumPy below 2. It also uses
`openai-clip` and `decord`, unlike the paired-video metric script. Some dimensions
require further model downloads or optional upstream dependencies. Follow the
VBench setup for the dimensions you run. Do not install VBench's requirements
into either generation environment.

## CPU analysis

For JSON/CSV analysis and plotting without model inference:

```bash
python -m pip install -r requirements/analysis.txt
python -m pip install --no-deps -e .
```

Analyses that open `.pt` latent trajectories additionally need PyTorch, but not
FlashAttention. Generation and learned quality metrics require their model
dependencies. W&B logging is optional; install `wandb` only if you enable it.
Utilities that are not version-pinned in these files are not a complete record
of the original environment. Save the resolved package versions with your run.
