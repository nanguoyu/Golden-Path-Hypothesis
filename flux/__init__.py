"""FLUX backbone (diffusers `FluxPipeline`).

Clean, monkey-patch-style cache method implementations on top of the HuggingFace
diffusers FluxPipeline.

Modules:
  - _helpers          pipeline-attribute helpers + image-size hygiene
  - hicache           HiCache: whole-transformer-residual Hermite extrapolation (coarse)
  - taylorseer        TaylorSeer: whole-transformer-residual Taylor extrapolation (coarse)
  - seacache          SeaCache: full-forward replacement with Wiener-filtered gate
  - teacache          TeaCache: full-forward replacement with poly-rescaled gate
  - sencache          SenCache-style frozen sensitivity-aware latent gate
  - hicache_fine      HiCache: per-(block,sub_module) Hermite (paper-faithful)
  - taylorseer_fine   TaylorSeer: per-(block,sub_module) Taylor (paper-faithful)
  - runner            single-GPU multi-prompt entry point (--mode dispatch)

Each method module exposes an `install(pipe, **kwargs) -> teardown_fn`
contract: call once after `DiffusionPipeline.from_pretrained`, call the
returned teardown callable to undo the monkey-patches. The `_fine` variants
share infrastructure in `lib/flux_fine_scaffold.py`.
"""
