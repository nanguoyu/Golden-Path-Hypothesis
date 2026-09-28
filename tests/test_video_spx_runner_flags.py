"""Unit tests for the three video-SPX runner flags, on both video backbones.

The experiment transplants each schedule onto payloads that did not choose it
(`docs/video_sp_cross_plan_zh.md` section 8), which needs exactly three things
the matrix runners did not have:

  * `--meancache_jvp_span`, plus a span-cover rule that accepts a table with no
    per-edge spans at all -- the offline search solved spans for MeanCache's own
    tables only;
  * `--spx_relax_warmup`, which lowers a head-of-trajectory warmup that is a
    search convention rather than a structural requirement (MeanCache 5 -> 2 on
    both backbones, Wan's BudCache 3 -> 1);
  * a fixed-schedule entry for DiCache, so its two-anchor payload can be scored
    on a schedule DiCache did not pick.

Every default is asserted to be unchanged, because the frozen 162-cell matrices
run this same code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hunyuan_video import baseline_screen_runner as HY
from hunyuan_video.dicache import HunyuanDiCacheAdapter, HunyuanDiCacheConfig
from hunyuan_video.methods.meancache import MeanCacheMethod
from wan21 import baseline_screen_runner as WAN
from wan21 import methods_glue as GLUE

REPO = Path(__file__).resolve().parents[1]
PENGUIN = REPO / "resources/hunyuan_video/evaluation/penguin599.json"


# ---------------------------------------------------------------------------
# defaults are unchanged
# ---------------------------------------------------------------------------


def test_defaults_are_the_matrix_behaviour() -> None:
    hy = HY.build_parser().parse_args(
        ["--mode", "reuse_exact", "--output_dir", "/tmp/x", "--model_base", "/tmp/m"])
    assert hy.meancache_jvp_span == 4
    assert hy.spx_relax_warmup is False
    wan = WAN.build_parser().parse_args(
        ["--mode", "budcache", "--output_dir", "/tmp/x", "--ckpt_dir", "/tmp/c"])
    assert wan.meancache_jvp_span == GLUE.MEANCACHE_JVP_SPAN
    assert wan.spx_relax_warmup is False
    assert GLUE.forbidden_cache_steps("meancache", 50) == frozenset({0, 1, 2, 3, 4, 49})
    assert GLUE.forbidden_cache_steps("budcache", 50) == frozenset({0, 1, 2, 49})


# ---------------------------------------------------------------------------
# --spx_relax_warmup
# ---------------------------------------------------------------------------


def test_wan_relaxed_warmups() -> None:
    assert GLUE.forbidden_cache_steps("meancache", 50, relax_warmup=True) == frozenset({0, 1, 49})
    assert GLUE.forbidden_cache_steps("budcache", 50, relax_warmup=True) == frozenset({0, 49})
    # the payloads whose warmup is a real order requirement are untouched
    assert (GLUE.forbidden_cache_steps("hicache_o2", 50, relax_warmup=True)
            == GLUE.forbidden_cache_steps("hicache_o2", 50))
    assert (GLUE.forbidden_cache_steps("taylorseer_o1", 50, relax_warmup=True)
            == GLUE.forbidden_cache_steps("taylorseer_o1", 50))


def test_wan_fixed_cache_steps_reads_the_relaxation_off_the_config() -> None:
    config = {"cache_count": 2, "cache_steps": [2, 10]}
    with pytest.raises(ValueError):
        GLUE.fixed_cache_steps(config, method="budcache", num_steps=50)
    relaxed = GLUE.fixed_cache_steps({**config, "relax_warmup": True},
                                     method="budcache", num_steps=50)
    assert relaxed == frozenset({2, 10})


def test_hunyuan_meancache_first_full_steps_travels_in_the_method_config(tmp_path: Path) -> None:
    schedule = tmp_path / "sched.json"
    schedule.write_text(json.dumps({"cache_steps": [2, 7, 20]}), encoding="utf-8")
    base = ["--mode", "meancache_exact", "--output_dir", str(tmp_path),
            "--model_base", str(tmp_path), "--cache_count", "3",
            "--meancache_schedule", str(schedule)]
    plain = HY.build_parser().parse_args(base)
    # absent by default, so a matrix cell's recorded method_config is unchanged
    # and the backend keeps its own 5
    assert "first_full_steps" not in HY._method_config(plain)
    relaxed = HY.build_parser().parse_args([*base, "--spx_relax_warmup"])
    assert (HY._method_config(relaxed)["first_full_steps"]
            == HY.MEANCACHE_FIRST_FULL_STEPS_RELAXED == 2)


def test_relaxation_is_refused_together_with_a_frozen_config(tmp_path: Path) -> None:
    """A relaxed warmup is a deviation from the matrix, so a config-driven cell
    must not be able to carry one."""
    args = WAN.build_parser().parse_args(
        ["--mode", "meancache", "--output_dir", str(tmp_path), "--ckpt_dir", str(tmp_path),
         "--matrix_config", str(REPO / "resources/wan21/baseline_matrix_config.v1.json"),
         "--budget", "K29", "--dataset", "penguin599", "--spx_relax_warmup"])
    with pytest.raises(SystemExit, match="spx_relax_warmup"):
        WAN._apply_matrix_config(args)
    hy = HY.build_parser().parse_args(
        ["--mode", "meancache_exact", "--output_dir", str(tmp_path),
         "--model_base", str(tmp_path),
         "--matrix_config", str(REPO / "resources/hunyuan_video/baseline_matrix_config.v1.json"),
         "--budget", "K29", "--dataset", "penguin599", "--spx_relax_warmup"])
    with pytest.raises(SystemExit, match="spx_relax_warmup"):
        HY._apply_matrix_config(hy)


# ---------------------------------------------------------------------------
# --meancache_jvp_span and the relaxed span cover
# ---------------------------------------------------------------------------


def test_hunyuan_schedule_without_spans_runs_the_global_span(tmp_path: Path) -> None:
    schedule = tmp_path / "sched.json"
    schedule.write_text(json.dumps({"cache_steps": [5, 9, 30]}), encoding="utf-8")
    args = HY.build_parser().parse_args(
        ["--mode", "meancache_exact", "--output_dir", str(tmp_path),
         "--model_base", str(tmp_path), "--cache_count", "3",
         "--meancache_schedule", str(schedule), "--meancache_jvp_span", "6"])
    config = HY._method_config(args)
    assert config["jvp_spans"] == {}
    assert config["jvp_span"] == 6
    assert config["cache_steps"] == [5, 9, 30]


def test_hunyuan_refuses_a_partial_span_table(tmp_path: Path) -> None:
    """All edges or none: a half-solved table would run the search's span on
    some edges and a default on the rest, which is neither schedule."""
    schedule = tmp_path / "sched.json"
    schedule.write_text(json.dumps({"cache_steps": [5, 9, 30], "jvp_spans": {"5": 3}}),
                        encoding="utf-8")
    args = HY.build_parser().parse_args(
        ["--mode", "meancache_exact", "--output_dir", str(tmp_path),
         "--model_base", str(tmp_path), "--cache_count", "3",
         "--meancache_schedule", str(schedule)])
    with pytest.raises(SystemExit, match="cover only part of the cached steps"):
        HY._method_config(args)


def test_hunyuan_refuses_spans_for_uncached_steps(tmp_path: Path) -> None:
    schedule = tmp_path / "sched.json"
    schedule.write_text(
        json.dumps({"cache_steps": [5, 9], "jvp_spans": {"5": 3, "9": 3, "40": 3}}),
        encoding="utf-8")
    args = HY.build_parser().parse_args(
        ["--mode", "meancache_exact", "--output_dir", str(tmp_path),
         "--model_base", str(tmp_path), "--cache_count", "2",
         "--meancache_schedule", str(schedule)])
    with pytest.raises(SystemExit, match="uncached steps"):
        HY._method_config(args)


def test_wan_schedule_without_spans_runs_the_global_span(tmp_path: Path) -> None:
    schedule = tmp_path / "sched.json"
    schedule.write_text(json.dumps({"cache_steps": [5, 9, 30]}), encoding="utf-8")
    args = WAN.build_parser().parse_args(
        ["--mode", "meancache", "--output_dir", str(tmp_path), "--ckpt_dir", str(tmp_path),
         "--meancache_schedule", str(schedule), "--meancache_jvp_span", "6"])
    WAN._resolve_meancache_schedule(args)
    assert args.cache_steps == (5, 9, 30)
    config = WAN._method_config(args)
    assert config["jvp_spans"] == {}
    assert config["jvp_span"] == 6
    assert "relax_warmup" not in config


def test_the_global_span_is_what_span_for_falls_back_to() -> None:
    """The runner flag only matters because `MeanCacheMethod` reads it per
    step; this pins the two together."""
    method = MeanCacheMethod(action=SimpleNamespace(reset=lambda: None),
                             num_steps=50, scheduler_provider=lambda: None,
                             jvp_span=6, jvp_spans={7: 2})
    assert method.span_for(7) == 2
    assert method.span_for(20) == 6


# ---------------------------------------------------------------------------
# the fixed-schedule DiCache entry
# ---------------------------------------------------------------------------


class _Double(nn.Module):
    def forward(self, img, txt, **_kwargs):
        return img + 1.0, txt


class _Single(nn.Module):
    def forward(self, x, **_kwargs):
        return x + 1.0


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.double_blocks = nn.ModuleList([_Double(), _Double()])
        self.single_blocks = nn.ModuleList([_Single()])

    def forward(self, img, txt):
        for block in self.double_blocks:
            img, txt = block(img, txt)
        x = torch.cat([img, txt], dim=1)
        for block in self.single_blocks:
            x = block(x)
        return x


def _run(adapter: HunyuanDiCacheAdapter, transformer: _Transformer, steps: int) -> None:
    with adapter:
        for step in range(steps):
            transformer(torch.full((1, 3, 2), float(step + 1)), torch.zeros((1, 1, 2)))


def test_fixed_schedule_dicache_follows_the_table_not_the_threshold() -> None:
    transformer = _Transformer()
    adapter = HunyuanDiCacheAdapter(
        transformer,
        HunyuanDiCacheConfig(num_steps=8, threshold=0.0, ret_ratio=0.9,
                             probe_depth=1, cache_steps=(3, 5)))
    _run(adapter, transformer, 8)
    actions = [row.action for row in adapter.decisions]
    assert actions == ["full", "full", "full", "cache", "full", "cache", "full", "full"]
    # threshold 0.0 would cache nothing and ret_ratio 0.9 would force the first
    # seven steps full; neither input is read once the table is there
    assert {row.reason for row in adapter.decisions} == {"fixed_full", "fixed_cache"}


def test_fixed_schedule_dicache_keeps_the_two_anchor_payload() -> None:
    transformer = _Transformer()
    adapter = HunyuanDiCacheAdapter(
        transformer,
        HunyuanDiCacheConfig(num_steps=8, threshold=0.0, ret_ratio=0.0,
                             probe_depth=1, cache_steps=(4, 6)))
    _run(adapter, transformer, 8)
    cached = [row for row in adapter.decisions if row.action == "cache"]
    assert len(cached) == 2
    for row in cached:
        # a real gamma means `aligned_residual` had two anchors and really
        # extrapolated; None would mean it silently degraded to plain reuse
        assert row.gamma is not None and 1.0 <= row.gamma <= 1.5
        # the shallow probe is real computation and is still counted
        assert row.original_block_calls == 1
    assert all(row.original_block_calls == len(transformer.double_blocks)
               + len(transformer.single_blocks)
               for row in adapter.decisions if row.action == "full")


def test_fixed_schedule_dicache_refuses_a_table_without_two_anchors() -> None:
    transformer = _Transformer()
    with pytest.raises(ValueError, match="keeps steps 0, 1"):
        HunyuanDiCacheAdapter(
            transformer,
            HunyuanDiCacheConfig(num_steps=8, probe_depth=1, cache_steps=(1, 4)))
    with pytest.raises(ValueError, match="keeps steps 0, 1"):
        HunyuanDiCacheAdapter(
            transformer,
            HunyuanDiCacheConfig(num_steps=8, probe_depth=1, cache_steps=(4, 7)))


def test_native_dicache_is_unchanged_without_a_table() -> None:
    transformer = _Transformer()
    adapter = HunyuanDiCacheAdapter(
        transformer,
        HunyuanDiCacheConfig(num_steps=8, threshold=100.0, ret_ratio=0.2, probe_depth=1))
    _run(adapter, transformer, 8)
    actions = [row.action for row in adapter.decisions]
    # warmup is `step <= int(0.2 * 8)` = steps 0..1, terminal step forced full
    assert actions[0] == "full" and actions[1] == "full"
    assert actions[-1] == "full"
    assert {row.reason for row in adapter.decisions} <= {"forced_boundary", "threshold_cache",
                                                         "threshold_full"}


def test_runners_recognise_a_fixed_schedule_dicache(tmp_path: Path) -> None:
    hy = HY.build_parser().parse_args(
        ["--mode", "dicache", "--output_dir", str(tmp_path), "--model_base", str(tmp_path),
         "--cache_count", "3", "--cache_steps", "5,9,30"])
    assert HY.dicache_fixed_schedule(hy)
    HY._validate(hy, 50)
    config = HY._method_config(hy)
    assert config["cache_steps"] == [5, 9, 30] and config["cache_count"] == 3

    wan = WAN.build_parser().parse_args(
        ["--mode", "dicache", "--output_dir", str(tmp_path), "--ckpt_dir", str(tmp_path),
         "--cache_count", "3", "--cache_steps", "5,9,30"])
    assert WAN.dicache_fixed_schedule(wan)
    config = WAN._method_config(wan)
    assert config["cache_steps"] == [5, 9, 30] and config["cache_count"] == 3


def test_runners_refuse_a_dicache_table_inside_the_warmup(tmp_path: Path) -> None:
    hy = HY.build_parser().parse_args(
        ["--mode", "dicache", "--output_dir", str(tmp_path), "--model_base", str(tmp_path),
         "--cache_count", "2", "--cache_steps", "1,9"])
    with pytest.raises(SystemExit, match="two full steps"):
        HY._validate(hy, 50)


def test_native_dicache_still_refuses_a_schedule_free_cache_count(tmp_path: Path) -> None:
    hy = HY.build_parser().parse_args(
        ["--mode", "dicache", "--output_dir", str(tmp_path), "--model_base", str(tmp_path),
         "--cache_count", "3"])
    assert not HY.dicache_fixed_schedule(hy)
    with pytest.raises(SystemExit, match="do not accept cache_count"):
        HY._validate(hy, 50)
