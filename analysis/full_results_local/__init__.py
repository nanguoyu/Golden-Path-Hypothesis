"""Local aggregate-layer + path-layer analysis package for the cross-model
four-dataset ten-method full result set.

Scope (see docs/cross_model_full_results_local_analysis_plan_zh.md):
  720 cells = model(2) x dataset(4) x target_k(3) x method(10) x seed_stream(3),
  read from git-tracked aggregate tables under resources/.

Layering
  registry.py   fixed orders, families, colours, metric directions, path constants
  rules.py      pure deterministic classification rules (no IO, no pandas state)
  distances.py  point-to-point + distribution-to-distribution distances (pure)
  paths.py      bitstring parsing, PathDist support-distribution object, M5 quantities
  loader.py     M0: schema validation, 720 coverage, container construction
  run_all.py    single entry point

`distances.py`, `paths.py` and `rules.py` are IO-free pure function layers.
"""

__version__ = "1.0.0"
