"""`--prompt_indices` on the two video baseline-screen runners.

The T3 extension (docs/video_cached_trajectory_t3_extension_plan_zh.md) draws a
scattered sample of manifest positions per directory instead of the contiguous
prefix `--limit` gives. What has to hold for those generations to pair with the
references already on disk is that an index keeps its manifest position: the
seed is `base + idx` and the filenames are `*_{idx:05d}`, so a selection that
renumbered its picks 0..n-1 would silently generate different videos under the
reference's names. Both runners carry their own copy of the selection, so both
are checked here.
"""

import pytest

from hunyuan_video import baseline_screen_runner as HYV
from wan21 import baseline_screen_runner as WAN

RUNNERS = pytest.mark.parametrize("R", [HYV, WAN], ids=["hunyuan_video", "wan21"])

PROMPTS = [f"prompt {i}" for i in range(10)]


def args_for(R, argv: list[str]):
    parser = R.build_parser()
    return parser.parse_args(argv)


@RUNNERS
def test_the_picked_indices_keep_their_manifest_position(R):
    args = args_for(R, ["--mode", "seacache", "--prompt_indices", "1,4,9",
                        "--output_dir", "/tmp/x", *R_extra(R)])
    assert R.select_work(args, PROMPTS) == [(1, "prompt 1"), (4, "prompt 4"),
                                            (9, "prompt 9")]


@RUNNERS
def test_without_the_flag_the_contiguous_slice_is_unchanged(R):
    args = args_for(R, ["--mode", "seacache", "--output_dir", "/tmp/x", *R_extra(R)])
    assert R.select_work(args, PROMPTS) == list(enumerate(PROMPTS))


@RUNNERS
def test_shards_split_the_picked_list_and_still_carry_global_indices(R):
    seen = []
    for shard in range(2):
        args = args_for(R, ["--mode", "seacache", "--prompt_indices", "1,4,9",
                            "--shard_idx", str(shard), "--shard_count", "2",
                            "--output_dir", "/tmp/x", *R_extra(R)])
        seen += R.select_work(args, PROMPTS)
    assert seen == [(1, "prompt 1"), (4, "prompt 4"), (9, "prompt 9")]


@RUNNERS
def test_an_index_past_the_end_of_the_manifest_stops_the_run(R):
    """Silently dropping it would leave the directory one pair short of the
    sample table with nothing in the log to say which one."""
    args = args_for(R, ["--mode", "seacache", "--prompt_indices", "1,42",
                        "--output_dir", "/tmp/x", *R_extra(R)])
    with pytest.raises(SystemExit, match="past the end"):
        R.select_work(args, PROMPTS)


@RUNNERS
def test_it_is_not_combinable_with_a_limit_slice(R):
    args = args_for(R, ["--mode", "seacache", "--prompt_indices", "1,4",
                        "--limit", "3", "--output_dir", "/tmp/x", *R_extra(R)])
    with pytest.raises(SystemExit):
        R.select_work(args, PROMPTS)


@RUNNERS
def test_an_unsorted_or_repeated_list_is_rejected_by_the_parser(R):
    for bad in ("4,1", "1,1", "-1,4"):
        with pytest.raises(SystemExit):
            args_for(R, ["--mode", "seacache", "--prompt_indices", bad,
                         "--output_dir", "/tmp/x", *R_extra(R)])


def R_extra(R) -> list[str]:
    """The one required argument that differs between the two parsers."""
    return ["--model_base", "/tmp/model"] if R is HYV else []
