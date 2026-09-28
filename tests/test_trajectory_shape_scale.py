"""Unit tests for analysis/trajectory_shape_scale.py — the module that reads
the 2-model x 4-dataset x 3-seed grid back.

Everything here is synthetic and CPU-only. The store is built with the same
writers the probes use (`plane_frame` + `write_plane_frame`) so the float16
round trip is exercised, and with PLANTED structure so each statistic has a
known right answer: trajectories that share the initial noise are planted in
one plane, trajectories that share only the prompt are not.
"""

import hashlib
import json

import numpy as np
import pytest

from analysis import trajectory_shape_scale as SS
from analysis.trajectory_math import CURVATURE_WINDOW, plane_frame, trajectory_metrics
from lib.io_utils import write_plane_frame

D = 96
N = 30
SIGMAS = list(np.linspace(1.0, 1.0 / N, N)) + [0.0]


def basis_for(noise: int) -> np.ndarray:
    return np.linalg.qr(np.random.default_rng(1000 + noise).standard_normal((D, 3)))[0].T


def trajectory(noise: int, amp_key: int) -> np.ndarray:
    """Bends inside the plane belonging to `noise`; `amp_key` only scales it."""
    e0, e1, e2 = basis_for(noise)
    t = np.linspace(0.0, 1.0, N + 1)
    amp = 1.0 + 0.3 * ((amp_key % 5) / 5.0)
    return (np.outer(20.0 * t, e0)
            + np.outer(amp * np.sin(np.pi * t), e1)
            + np.outer(0.4 * amp * np.sin(2.0 * np.pi * t), e2))


def build_store(root, *, n_prompts=6, drop_frame_at=None, schema="full_trajectory.v3"):
    """A grid shaped like the launcher's output. `drop_frame_at` omits the
    frame of one (dataset, seed, idx) to mimic a cell run without --save_frame."""
    for model, seeds in (("flux", (41, 42, 43)),):
        for dataset in ("drawbench_full", "parti_full"):
            for seed in seeds:
                out = root / model / f"{dataset}_n{n_prompts}_s{seed}_50"
                out.mkdir(parents=True, exist_ok=True)
                for idx in range(n_prompts):
                    noise = seed + idx          # the probes' own seed_for
                    Z = trajectory(noise, idx)
                    record = {
                        "schema": schema, "model": model, "dataset": dataset,
                        "prompt_idx": idx, "seed": seed, "d": D,
                        "z_T_sha256": hashlib.sha256(str(noise).encode()).hexdigest(),
                        "frame_file": None,
                    }
                    record.update(trajectory_metrics(Z, SIGMAS))
                    if schema != "full_trajectory.v3":
                        record.pop(SS.PROFILE_KEY)
                    if drop_frame_at != (dataset, seed, idx):
                        name = f"frame_{idx:05d}_s{seed}.npy"
                        write_plane_frame(out / name, plane_frame(Z))
                        record["frame_file"] = name
                    (out / f"traj_{idx:05d}_s{seed}.json").write_text(json.dumps(record))
    return root


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("grid"))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def test_load_root_finds_every_cell_and_keys_it_from_the_records(store):
    cells = SS.load_root(store)
    assert len(cells) == 6
    for (model, dataset, seed), (records, frames) in cells.items():
        assert model == "flux"
        assert len(records) == 6 and frames.shape == (6, 3, D)
        assert {r["dataset"] for r in records} == {dataset}
        assert {r["seed"] for r in records} == {seed}


def test_load_cell_keeps_frameless_records_and_counts_frames_separately(tmp_path):
    root = build_store(tmp_path, drop_frame_at=("parti_full", 42, 3))
    records, frames = SS.load_cell(root / "flux" / "parti_full_n6_s42_50")
    assert len(records) == 6
    assert frames.shape[0] == 5
    assert sum(1 for r in records if r.get("_has_frame")) == 5


def test_a_pre_v3_record_stops_the_run_with_a_readable_error(tmp_path):
    root = build_store(tmp_path, schema="full_trajectory.v2")
    records, _ = SS.load_cell(root / "flux" / "parti_full_n6_s42_50")
    with pytest.raises(SystemExit, match="full_trajectory.v3"):
        SS.profiles_of(records)


# --------------------------------------------------------------------------
# 1. profile summary
# --------------------------------------------------------------------------

def test_profile_centers_and_length_follow_the_window(store):
    records, _ = SS.load_root(store)[("flux", "parti_full", 42)]
    summary = SS.profile_summary(records)
    assert len(summary["median_profile"]) == (N + 1) - 2 * CURVATURE_WINDOW
    assert summary["centers"][0] == CURVATURE_WINDOW
    assert len(summary["centers"]) == len(summary["median_profile"])


def test_profile_summary_reports_where_the_trough_sits():
    """A profile whose trough is planted at a known center must report it, and
    the level must not leak into the location."""
    centers = SS._centers_for(21)
    trough_at = 17
    base = 1.0 + 0.02 * (np.array(centers) - trough_at) ** 2
    records = [{"schema": "full_trajectory.v3", SS.PROFILE_KEY: list(scale * base)}
               for scale in (1.0, 3.0, 7.0)]
    summary = SS.profile_summary(records)
    assert summary["min_center"] == trough_at
    assert summary["trough_center_med"] == trough_at
    assert summary["trough_center_iqr"] == 0.0
    # scaling alone is not a shape difference
    assert summary["norm_dev_med"] < 1e-12


def test_profile_summary_separates_a_shifted_trough_from_a_rescaled_one():
    """The statistic that survived the audit has to react to a MOVED trough and
    not to a rescaled one — the failure mode of the correlation it replaced."""
    centers = np.array(SS._centers_for(21))
    def bowl(at):
        return list(1.0 + 0.02 * (centers - at) ** 2)
    same_place = [{"schema": "full_trajectory.v3", SS.PROFILE_KEY: list(s * np.array(bowl(17)))}
                  for s in (1.0, 2.0, 5.0)]
    moved = [{"schema": "full_trajectory.v3", SS.PROFILE_KEY: bowl(at)} for at in (11, 17, 23)]
    assert SS.profile_summary(same_place)["norm_dev_q95"] < 1e-12
    assert SS.profile_summary(moved)["norm_dev_q95"] > 0.05
    assert SS.profile_summary(moved)["trough_center_iqr"] == 6.0


def test_profile_agreement_reports_the_trough_shift_between_cells():
    centers = np.array(SS._centers_for(21))     # 5..25, mid 15
    # a and b are mirror images about the mid center, so they have the SAME
    # mean: the only thing separating them is where the trough sits
    medians = {
        ("a",): 1.0 + 0.02 * (centers - 12) ** 2,
        ("b",): 1.0 + 0.02 * (centers - 18) ** 2,
        ("c",): 3.0 * (1.0 + 0.02 * (centers - 12) ** 2),   # same shape, 3x level
    }
    out = SS.profile_agreement(medians)
    assert out["pairs"] == 3
    assert out["trough_shift_max"] == 6
    assert out["level_ratio_max"] == pytest.approx(3.0, rel=1e-12)
    # a/c differ only in level, so the unit-mean shapes coincide ...
    same = SS.profile_agreement({k: medians[k] for k in (("a",), ("c",))})
    assert same["norm_dev_max"] < 1e-12
    # ... while a/b differ only in WHERE the trough is, which the gap must see
    moved = SS.profile_agreement({k: medians[k] for k in (("a",), ("b",))})
    assert moved["norm_dev_max"] > 0.10
    assert moved["level_ratio_max"] < 1.01     # and it is not a level difference


# --------------------------------------------------------------------------
# 2. plane geometry
# --------------------------------------------------------------------------

def test_gram_block_angles_match_a_direct_computation(store):
    """The closed-form 2x2 route has to agree with an honest SVD of the two
    bases — it exists only because ~1e5 LAPACK calls on 2x2 inputs cost more
    in thread synchronization than the arithmetic."""
    from analysis.trajectory_math import principal_angles_deg

    cells = SS.load_root(store)
    frames = SS.orthonormalize(np.concatenate([cells[k][1] for k in sorted(cells)]))
    rows = frames[:, 1:, :].reshape(-1, D)
    gram = rows @ rows.T
    pairs = [(0, 1), (0, 7), (3, 20), (11, 30)]
    t1, t2 = SS.pair_angles_deg(gram, pairs)
    for k, (i, j) in enumerate(pairs):
        direct = principal_angles_deg(frames[i, 1:], frames[j, 1:])
        assert [t1[k], t2[k]] == pytest.approx(direct, abs=1e-3)


def test_gram_block_angles_stay_accurate_on_near_parallel_planes():
    """The unstable pairing — taking the SMALL singular value from the
    discriminant — cancels catastrophically exactly here, where s2 is tiny."""
    from analysis.trajectory_math import principal_angles_deg

    rng = np.random.default_rng(17)
    basis = np.linalg.qr(rng.standard_normal((64, 4)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    for tilt in (1e-4, 1e-2, 1.0, 45.0):
        rad = np.radians(tilt)
        a = np.stack([e0, e1])
        b = np.stack([e0, np.cos(rad) * e1 + np.sin(rad) * e2])
        rows = np.concatenate([a, b])
        t1, t2 = SS.pair_angles_deg(rows @ rows.T, [(0, 1)])
        assert [t1[0], t2[0]] == pytest.approx(principal_angles_deg(a, b), abs=1e-6)


def test_gram_block_angles_are_ordered_and_bounded(store):
    cells = SS.load_root(store)
    frames = SS.orthonormalize(np.concatenate([cells[k][1] for k in sorted(cells)]))
    rows = frames[:, 1:, :].reshape(-1, D)
    gram = rows @ rows.T
    n = len(frames)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t1, t2 = SS.pair_angles_deg(gram, pairs)
    assert np.all(t1 <= t2 + 1e-9)
    assert np.all((t1 >= 0.0) & (t2 <= 90.0 + 1e-9))
    # A plane against itself is the degenerate case the closed form must
    # survive. The tolerance is the float32 Gram's, not the formula's: arccos
    # is vertical at 0, so 1 - 1e-7 in the Gram is ~0.03 degrees here.
    same1, same2 = SS.pair_angles_deg(gram, [(3, 3)])
    assert same1[0] == pytest.approx(0.0, abs=0.05)
    assert same2[0] == pytest.approx(0.0, abs=0.05)
    # with a basis that is orthonormal to float64 rather than to float32, the
    # same self-pair is exact — the 0.03 above is the store's precision
    exact = np.linalg.qr(np.random.default_rng(0).standard_normal((D, 2)))[0].T
    e1, e2 = SS.pair_angles_deg(exact @ exact.T, [(0, 0)])
    assert max(e1[0], e2[0]) < 1e-5


def test_planted_shared_planes_come_back_as_zero_and_the_rest_as_the_null(store):
    cells = SS.load_root(store)
    keys = sorted(cells)
    records = [r for k in keys for r in cells[k][0] if r.get("_has_frame")]
    frames = SS.orthonormalize(np.concatenate([cells[k][1] for k in keys]))
    rows = frames[:, 1:, :].reshape(-1, D)
    gram = rows @ rows.T
    split = SS.split_pairs(records)

    shared = SS.angle_stats(gram, split["same_noise_diff_prompt"])
    assert shared["pairs"] > 0
    assert shared["theta2_med"] < 0.5          # planted identical plane

    prompt = SS.angle_stats(gram, split["same_prompt_diff_noise"])
    null = SS.random_plane_null(D, n_pairs=200)
    assert prompt["theta1_med"] > null["theta1_med"] - 15.0   # planted unrelated


def test_split_pairs_buckets_are_disjoint_and_read_from_the_record(store):
    cells = SS.load_root(store)
    records = [r for k in sorted(cells) for r in cells[k][0]]
    split = SS.split_pairs(records)
    noise = set(split["same_noise_diff_prompt"])
    prompt = set(split["same_prompt_diff_noise"])
    assert not (noise & prompt)
    assert split["same_both"] == []
    for i, j in noise:
        assert records[i]["z_T_sha256"] == records[j]["z_T_sha256"]
        assert (records[i]["dataset"], records[i]["prompt_idx"]) != \
               (records[j]["dataset"], records[j]["prompt_idx"])
    for i, j in prompt:
        assert (records[i]["dataset"], records[i]["prompt_idx"]) == \
               (records[j]["dataset"], records[j]["prompt_idx"])
        assert records[i]["z_T_sha256"] != records[j]["z_T_sha256"]


def test_chord_angles_are_acute_despite_the_qr_sign_convention(store):
    """`orthonormalize` runs each frame through QR, which flips the sign of any
    row with a positive leading component. Half the chord angles would come
    back as 180-theta if the cosine were not taken absolute."""
    cells = SS.load_root(store)
    keys = sorted(cells)
    records = [r for k in keys for r in cells[k][0] if r.get("_has_frame")]
    chords = SS.orthonormalize(np.concatenate([cells[k][1] for k in keys]))[:, 0, :]

    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    out = SS.chord_stats(chords, pairs)
    assert out["pairs"] == len(pairs)
    assert 0.0 <= out["chord_angle_med"] <= 90.0
    # planted: trajectories sharing a noise share e0, so their chords coincide
    same_noise = SS.chord_stats(chords, SS.split_pairs(records)["same_noise_diff_prompt"])
    assert same_noise["pairs"] > 0
    assert same_noise["chord_angle_med"] < 1.0

    # QR fixes the sign of each row independently, so flipping any subset of
    # the chords must not move a single reported angle
    flipped = chords.copy()
    flipped[::2] *= -1.0
    assert SS.chord_stats(flipped, pairs) == out
    assert SS.chord_stats(-chords, pairs) == out


def test_within_cell_pairs_stay_inside_their_cell_when_a_frame_is_missing(tmp_path):
    """The pair indices address the frame stack, so a frameless record must not
    shift the cell boundaries. With 6 cells of 6 and one frame dropped, the
    correct answer is 5 cells of C(6,2) plus one of C(5,2) = 85 pairs; counting
    records instead gives 90 and silently labels cross-cell pairs within-cell.
    """
    root = build_store(tmp_path, drop_frame_at=("drawbench_full", 42, 2))
    cells = SS.load_root(root)
    model_cells = [(k, cells[k]) for k in sorted(cells)]
    entry = SS.model_plane_report(model_cells, max_random_pairs=50)
    assert entry["n"] == 35                      # 36 records, 35 frames
    assert entry["within_cell"]["pairs"] == 5 * 15 + 10
    # a cross-cell pair would be planted-unrelated, dragging theta1 to the null
    assert entry["within_cell"]["theta1_min"] > 30.0


def test_model_plane_report_refuses_a_frame_record_mismatch(tmp_path):
    root = build_store(tmp_path)
    cells = SS.load_root(root)
    key = sorted(cells)[0]
    records, frames = cells[key]
    cells[key] = (records, frames[:-1])          # a frame vanished after loading
    with pytest.raises(ValueError, match="frames but"):
        SS.model_plane_report([(k, cells[k]) for k in sorted(cells)], max_random_pairs=10)


def test_population_reference_is_drawn_not_assumed():
    """Stacked random 2-planes do not have a flat spectrum, so the naive
    0.5 * rows reference overstates the no-alignment baseline. How far it
    overstates depends on rows/dim, which is why the reference is drawn at the
    run's own shape rather than assumed: the gap shrinks as the dimension grows
    and the naive number is only correct in the limit."""
    wide = SS.plane_population_reference(24, 256, seed=3)     # rows/dim = 0.19
    thin = SS.plane_population_reference(24, 8192, seed=3)    # rows/dim = 0.006
    assert wide["rows"] == thin["rows"] == 48
    assert wide["dims_for_50pct"] < thin["dims_for_50pct"] < 0.5 * 48
    assert wide["top2_share"] < 0.25


def test_population_detects_a_genuinely_shared_plane():
    rng = np.random.default_rng(9)
    shared = np.linalg.qr(rng.standard_normal((64, 2)))[0].T
    rows = np.repeat(shared[None, :, :], 20, axis=0).reshape(-1, 64)
    out = SS.plane_population(rows @ rows.T)
    assert out["top2_share"] > 0.999
    assert out["dims_for_90pct"] <= 2


def test_sample_pairs_is_capped_and_excludes():
    exclude = {(0, 1), (0, 2)}
    got = SS.sample_pairs(5, 1000, seed=0, exclude=exclude)
    assert len(got) == 5 * 4 // 2 - len(exclude)
    assert not (set(got) & exclude)
    assert all(i < j for i, j in got)
