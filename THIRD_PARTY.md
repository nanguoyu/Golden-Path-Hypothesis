# Third-party sources

This repository uses modified implementations of the following methods for
the experiments in this paper.

## Caching methods

| Method | Paper | Official code | Main local files |
|---|---|---|---|
| SeaCache | [Paper](https://arxiv.org/abs/2602.18993) | [SeaCache](https://github.com/jiwoogit/SeaCache) | [lib/wiener.py](lib/wiener.py), [flux/seacache.py](flux/seacache.py) |
| TeaCache | [Paper](https://arxiv.org/abs/2411.19108) | [TeaCache](https://github.com/ali-vilab/TeaCache) | [lib/teacache_coeffs.py](lib/teacache_coeffs.py), [flux/teacache.py](flux/teacache.py) |
| SenCache | [Paper](https://arxiv.org/abs/2602.24208) | [SenCache](https://github.com/vita-epfl/SenCache) | [lib/sencache.py](lib/sencache.py), [flux/sencache.py](flux/sencache.py) |
| DiCache | [Paper](https://arxiv.org/abs/2508.17356) | [DiCache](https://github.com/Bujiazi/DiCache) | [lib/dicache.py](lib/dicache.py), [flux/dicache_native.py](flux/dicache_native.py) |
| TaylorSeer | [Paper](https://arxiv.org/abs/2503.06923) | [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) | [lib/taylor.py](lib/taylor.py), [flux/taylorseer_fine.py](flux/taylorseer_fine.py) |
| HiCache | [Paper](https://arxiv.org/abs/2508.16984) | [HiCache](https://github.com/fenglang918/HiCache) | [lib/hermite.py](lib/hermite.py), [flux/hicache_fine.py](flux/hicache_fine.py) |
| L2P-Cache | [Paper](https://arxiv.org/abs/2604.26365) | [L2P-Cache](https://github.com/Aredstone/L2P-Cache) | [lib/l2p.py](lib/l2p.py), [flux/l2p_output.py](flux/l2p_output.py) |
| DPCache | [Paper](https://arxiv.org/abs/2602.22654) | [DPCache](https://github.com/argsss/DPCache) | [lib/dpcache.py](lib/dpcache.py), [flux/dpcache_exact.py](flux/dpcache_exact.py) |
| BudCache | [Paper](https://arxiv.org/abs/2606.13496) | [BudCache](https://github.com/Westlake-AGI-Lab/BudCache) | [FLUX search](analysis/search_budcache_flux.py), [Qwen search](analysis/search_budcache_qwen.py) |
| MeanCache | [Paper](https://arxiv.org/abs/2601.19961) | [MeanCache](https://github.com/UnicomAI/MeanCache) | [flux/meancache_exact.py](flux/meancache_exact.py), [schedule construction](analysis/build_meancache_schedule.py) |

All ten methods are evaluated on image models; video evaluation excludes
DPCache. Model-specific entry points and settings are in
[docs/baselines.md](docs/baselines.md).

Additional experimental code draws on [ToCa](https://arxiv.org/abs/2410.05317)
([official code](https://github.com/Shenyi-Z/ToCa)),
[FoCa](https://arxiv.org/abs/2508.16211), and
[SVD-Cache](https://arxiv.org/abs/2601.07396).

## External source dependencies

[scripts/bootstrap_sources.py](scripts/bootstrap_sources.py) fetches these
three source trees at fixed revisions.

| Source | Revision | Purpose |
|---|---|---|
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | `e748c73ac064728bf6bd15b1cdb8161e55a4f331` | Native HunyuanVideo model and Penguin benchmark metadata |
| [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) | `704ee98c74f7f04da443daa3c0aa2cc7803d86e3` | Wan2.1 model package in `TaylorSeer-Wan2.1` |
| [VBench](https://github.com/Vchitect/VBench) | `45e79ec14e69a2187202c675d2dbce1a71843d53` | Video prompt metadata and evaluation |

## Models, data, and evaluation

Model sources for FLUX.1-dev, Qwen-Image, HunyuanVideo, and Wan2.1 are listed
in [docs/assets.md](docs/assets.md). Prompt sources for DrawBench, PartiPrompts,
GenEval, DiffusionDB, Penguin, and VBench are listed in
[docs/data.md](docs/data.md). Search also uses [COCO](https://cocodataset.org/#download)
captions and annotations.

The code uses [PyTorch](https://github.com/pytorch/pytorch),
[diffusers](https://github.com/huggingface/diffusers), and
[transformers](https://github.com/huggingface/transformers).
Evaluation uses [LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[CLIP](https://github.com/openai/CLIP),
[ImageReward](https://github.com/zai-org/ImageReward),
[scikit-image](https://github.com/scikit-image/scikit-image), and VBench.
The [GenEval-style prompt generator](analysis/build_geneval_prompt_file.py)
is a modified version of the [GenEval generator](https://github.com/djghosh13/geneval).
Package requirements are in [requirements/](requirements/).

Third-party components retain their original licenses and attribution.
