"""Shared math primitives and utilities for cache-acceleration research.

`lib/` is intentionally backbone-agnostic. Modules here:
  - hermite       Physicist's Hermite polynomial + HiCache predictor/updater
  - taylor        Taylor expansion predictor/updater (TaylorSeer-style)
  - analytic_sigma EMA / quantile estimator for HiCache-Analytic adaptive sigma
  - wiener        N-D separable Wiener filter (SeaCache)
  - gates         rel_L1 distance + step counter + first_enhance/interval gating
  - io_utils      timing.json schema + seed/path conventions

No `diffusers` / `transformers` imports in this package.
"""
