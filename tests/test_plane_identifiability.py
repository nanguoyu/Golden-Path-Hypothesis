"""Unit tests for analysis/plane_identifiability.py — the module that measures
what a bend-plane angle of 61 degrees means before anyone reads it as large.

Everything here is synthetic and CPU-only. The store is written with the same
writers the probes use (`plane_frame` + `write_plane_frame`), so the float16
round trip is exercised, and the geometry is planted so that EACH STORED ROW
HAS A DIFFERENT SIGNATURE — reading the wrong one is the failure mode with no
symptom in the output, since a chord read as a bend direction would report a
spectacular confirmation for a reason that has nothing to do with bending:

    row 0  chord   one global direction   -> every pair agrees
    row 1  PC1     belongs to the noise   -> only same-noise pairs agree
    row 2  PC2     belongs to the prompt  -> only same-prompt pairs agree
"""

import hashlib
import json

import numpy as np
import pytest

from analysis import plane_identifiability as PI
from analysis.trajectory_math import plane_frame, segment_tag, trajectory_metrics
from lib.io_utils import write_plane_frame

D = 192
N = 50                                   # 51 rows, so the real segment tags apply
SIGMAS = list(np.linspace(1.0, 1.0 / N, N)) + [0.0]
EARLY, LATE = (0, 16), (38, 51)
EARLY_TAG, LATE_TAG = segment_tag(*EARLY), segment_tag(*LATE)
WHOLE_TAG = segment_tag(0, N + 1)
GLOBAL_CHORD = np.linalg.qr(np.random.default_rng(0).standard_normal((D, 1)))[0][:, 0]


def _perp_unit(seed: int) -> np.ndarray:
    """A direction orthogonal to the shared chord, so the planted plane really
    is the chord-orthogonal one the probe fits."""
    v = np.random.default_rng(seed).standard_normal(D)
    v -= (v @ GLOBAL_CHORD) * GLOBAL_CHORD
    return v / np.linalg.norm(v)


def noise_dir(noise: int) -> np.ndarray:
    return _perp_unit(1000 + noise)


def prompt_dir(dataset: str, idx: int) -> np.ndarray:
    return _perp_unit(9000 + 37 * idx + (dataset == "parti_full"))


def trajectory(noise: int, dataset: str, idx: int, *, turning: bool) -> np.ndarray:
    """Travels along the shared chord, bends mostly along the noise's direction
    and a little along the prompt's.

    Both bend terms vanish at either end, so the chord is exactly the global
    direction for every trajectory whatever the amplitudes are. The two time
    profiles are mutually orthogonal AND mean-zero, which makes the
    chord-orthogonal covariance exactly diagonal — without that the two
    directions mix into both principal components and the row signatures the
    tests read would not be the planted ones.

    `turning` swaps both bend directions over the late window, so that
    window's plane is orthogonal to the early one — a property of the whole
    store, never of one dataset, since a store where one dataset turned would
    stop same-noise pairs spanning the two datasets from sharing a plane.
    """
    t = np.linspace(0.0, 1.0, N + 1)
    lead, trail = np.sin(2 * np.pi * t), 0.4 * np.sin(4 * np.pi * t)
    e1, e2 = noise_dir(noise), prompt_dir(dataset, idx)
    alt1, alt2 = noise_dir(noise + 500), prompt_dir(dataset, idx + 500)
    first = np.outer(lead, e1) + np.outer(trail, e2)
    second = np.outer(lead, alt1) + np.outer(trail, alt2)
    late = (t > (LATE[0] - 1) / N)[:, None] & bool(turning)
    return np.outer(20.0 * t, GLOBAL_CHORD) + np.where(late, second, first)


def build_store(root, *, n_prompts=6, drop_late_frame_at=None, drop_whole_frame_at=None,
                turning=False, models=(("flux", (41, 42, 43)),)):
    """A grid shaped like the launcher's v5 output: a `frame_files` map holding
    the whole path plus two disjoint windows.

    `drop_*_frame_at` omits one (dataset, seed, idx) frame from one segment, so
    the frame stack and the record list stop lining up — the shape a cell run
    without `--save_frame` leaves behind, and the one that shifts every later
    pair index if the two are ever counted the same way.
    """
    for model, seeds in models:
        for dataset in ("drawbench_full", "parti_full"):
            for seed in seeds:
                out = root / model / f"{dataset}_n{n_prompts}_s{seed}_50"
                out.mkdir(parents=True, exist_ok=True)
                for idx in range(n_prompts):
                    noise = seed + idx          # the probes' own seed_for
                    Z = trajectory(noise, dataset, idx, turning=turning)
                    frame_files = {}
                    for tag, (lo, hi), drop in ((WHOLE_TAG, (0, N + 1), drop_whole_frame_at),
                                                (EARLY_TAG, EARLY, None),
                                                (LATE_TAG, LATE, drop_late_frame_at)):
                        if drop == (dataset, seed, idx):
                            continue
                        name = f"frame_{idx:05d}_s{seed}_{tag}.npy"
                        write_plane_frame(out / name, plane_frame(Z[lo:hi]))
                        frame_files[tag] = name
                    record = {
                        "schema": "full_trajectory.v5", "model": model, "dataset": dataset,
                        "prompt_idx": idx, "seed": seed, "d": D,
                        "device_name": "SYNTHETIC",
                        "z_T_sha256": hashlib.sha256(str(noise).encode()).hexdigest(),
                        "frame_files": frame_files,
                    }
                    record.update(trajectory_metrics(Z, SIGMAS))
                    (out / f"traj_{idx:05d}_s{seed}.json").write_text(json.dumps(record))
    return root


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("grid"))


@pytest.fixture(scope="module")
def turning_store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("turning"), turning=True)


@pytest.fixture(scope="module")
def report(store):
    return PI.model_report(sorted(store.glob("flux/*")), WHOLE_TAG, (EARLY_TAG, LATE_TAG))


# --------------------------------------------------------------------------
# store walking
# --------------------------------------------------------------------------

def test_parse_tag_reads_the_half_open_range():
    assert PI.parse_tag("038_051") == (38, 51)
    assert PI.parse_tag("000_016") == (0, 16)


def test_run_dirs_are_grouped_by_the_recorded_model(store):
    groups = PI.run_dirs_by_model(store)
    assert set(groups) == {"flux"}
    assert len(groups["flux"]) == 6                       # 2 datasets x 3 seeds


def test_run_dirs_refuses_a_dir_whose_records_disagree_about_the_model(tmp_path):
    """Checking only the first record would file the whole dir under one model
    and analyse two populations as one."""
    root = build_store(tmp_path, n_prompts=3)
    run_dir = sorted(root.glob("flux/*"))[0]
    path = sorted(run_dir.glob("traj_*.json"))[2]
    record = json.loads(path.read_text())
    record["model"] = "qwen"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="one model per run dir"):
        PI.run_dirs_by_model(root)


def test_run_dirs_names_the_file_when_a_record_has_no_model(tmp_path):
    root = build_store(tmp_path, n_prompts=2)
    path = sorted(root.glob("flux/*/traj_*.json"))[0]
    record = json.loads(path.read_text())
    record.pop("model")
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="traj_00000"):
        PI.run_dirs_by_model(root)


def test_frame_positions_counts_frames_not_records():
    records = [{"prompt_idx": 0, "_has_frame": True},
               {"prompt_idx": 1},                          # no frame for this segment
               {"prompt_idx": 2, "_has_frame": True}]
    assert PI.frame_positions(records) == {0: 0, 2: 1}


def test_frame_positions_refuses_a_repeated_prompt_index():
    records = [{"prompt_idx": 3, "_has_frame": True}, {"prompt_idx": 3, "_has_frame": True}]
    with pytest.raises(ValueError, match="twice"):
        PI.frame_positions(records)


def test_model_report_refuses_two_run_dirs_claiming_one_cell(tmp_path):
    root = build_store(tmp_path, n_prompts=3)
    original = sorted(root.glob("flux/drawbench_full*"))[0]
    copy = original.parent / (original.name + "_resubmit")
    copy.mkdir()
    for path in original.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    with pytest.raises(SystemExit, match="claim cell"):
        PI.model_report(sorted(root.glob("flux/*")), WHOLE_TAG, (EARLY_TAG, LATE_TAG))


# --------------------------------------------------------------------------
# argument validation, before any data is read
# --------------------------------------------------------------------------

def test_validate_segments_refuses_overlapping_turn_windows():
    with pytest.raises(SystemExit, match="overlap"):
        PI.validate_segments(WHOLE_TAG, ("000_051", "038_051"), [WHOLE_TAG, "038_051"])


def test_validate_segments_allows_windows_that_only_touch():
    """Half-open ranges: 000_016 ends before 016_051 begins, so they share no
    step and refusing them would rule out the natural split of a path."""
    PI.validate_segments("000_016", ("000_016", "016_051"), ["000_016", "016_051"])


def test_validate_segments_refuses_a_tag_the_store_does_not_hold():
    with pytest.raises(SystemExit, match="not in the store"):
        PI.validate_segments("not_a_tag", (EARLY_TAG, LATE_TAG), [WHOLE_TAG, EARLY_TAG, LATE_TAG])


def test_validate_segments_refuses_a_store_with_no_frames_at_all():
    """An empty segment list means nothing was stored, not that anything goes."""
    with pytest.raises(SystemExit, match="no frames"):
        PI.validate_segments(WHOLE_TAG, (EARLY_TAG, LATE_TAG), [])


def test_validate_segments_needs_exactly_two_turn_windows():
    with pytest.raises(SystemExit, match="exactly two"):
        PI.validate_segments(WHOLE_TAG, (EARLY_TAG,), [WHOLE_TAG, EARLY_TAG])


# --------------------------------------------------------------------------
# A. the measured resolution
# --------------------------------------------------------------------------

def test_plane_angles_matches_a_planted_rotation():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((40, 4)))[0].T
    a = basis[:2]
    angle = np.deg2rad(25.0)
    b = np.stack([np.cos(angle) * basis[0] + np.sin(angle) * basis[2], basis[1]])
    t1, t2 = PI.plane_angles(a, b)
    assert t1 == pytest.approx(0.0, abs=1e-6)             # basis[1] is shared exactly
    assert t2 == pytest.approx(25.0, abs=1e-6)


def test_floor_is_small_for_a_well_separated_plane_and_large_for_a_degenerate_one():
    """The floor has to track how far apart the second and third directions
    are in ENERGY, not how far apart they are as a ratio: a window whose second
    and third directions carry almost the same variance has an arbitrary plane
    and must report a large floor."""
    t = np.linspace(0.0, 1.0, N + 1)
    basis = np.linalg.qr(np.random.default_rng(5).standard_normal((D, 4)))[0].T
    # mutually orthogonal, mean-zero profiles, so the amplitudes ARE the
    # eigenvalues and "degenerate" means exactly what it says
    p1, p2, p3 = (np.sin(2 * np.pi * t), np.sin(4 * np.pi * t), np.sin(6 * np.pi * t))
    stem = np.outer(20.0 * t, basis[0]) + np.outer(p1, basis[1])
    separated = stem + np.outer(0.4 * p2, basis[2]) + np.outer(0.01 * p3, basis[3])
    degenerate = stem + np.outer(0.4 * p2, basis[2]) + np.outer(0.4 * p3, basis[3])
    sep = PI.floor_for_trajectory(separated, [WHOLE_TAG], np.random.default_rng(1), 5)[WHOLE_TAG]
    deg = PI.floor_for_trajectory(degenerate, [WHOLE_TAG], np.random.default_rng(1), 5)[WHOLE_TAG]
    assert np.median([r[1] for r in sep]) < 1.0
    assert np.median([r[1] for r in deg]) > 10.0 * np.median([r[1] for r in sep])


def test_floor_uses_a_relative_jitter_so_it_scales_with_the_trajectory():
    """An absolute perturbation would be enormous where the latent is small.
    Scaling the whole trajectory must leave the floor alone."""
    t = np.linspace(0.0, 1.0, N + 1)
    basis = np.linalg.qr(np.random.default_rng(6).standard_normal((D, 3)))[0].T
    Z = (np.outer(20.0 * t, basis[0]) + np.outer(np.sin(np.pi * t), basis[1])
         + np.outer(0.4 * np.sin(2 * np.pi * t), basis[2]))
    small = PI.floor_for_trajectory(Z, [WHOLE_TAG], np.random.default_rng(2), 4)[WHOLE_TAG]
    large = PI.floor_for_trajectory(1e4 * Z, [WHOLE_TAG], np.random.default_rng(2), 4)[WHOLE_TAG]
    assert np.median([r[0] for r in small]) == pytest.approx(
        np.median([r[0] for r in large]), rel=0.5)


def test_floor_report_is_empty_and_says_so_when_there_are_no_latents(tmp_path):
    assert PI.floor_report(tmp_path, [WHOLE_TAG]) == {}


# --------------------------------------------------------------------------
# B. one direction at a time
# --------------------------------------------------------------------------

def test_each_stored_row_carries_its_planted_signature(report):
    """The whole point of passing `row`: the chord agrees for every pair, the
    first bend direction only for same-noise pairs, the second only for
    same-prompt pairs. A report that read the wrong row would move a group
    from ~0 to ~90 degrees."""
    chord, first, second = (report["chord_control"], report["first_direction"],
                            report["second_direction"])
    assert chord["same_noise_diff_prompt"]["theta_med"] < 2.0
    assert chord["same_prompt_diff_noise"]["theta_med"] < 2.0
    assert chord["unrelated"]["theta_med"] < 2.0          # the chord is shared by construction

    assert first["same_noise_diff_prompt"]["theta_med"] < 5.0
    assert first["same_prompt_diff_noise"]["theta_med"] > 60.0
    assert first["unrelated"]["theta_med"] > 60.0

    assert second["same_prompt_diff_noise"]["theta_med"] < 15.0
    assert second["same_noise_diff_prompt"]["theta_med"] > 60.0


def test_direction_report_refuses_a_row_outside_the_frame(store):
    run_dirs = sorted(store.glob("flux/*"))
    recs, frames = PI.load_cell(run_dirs[0], WHOLE_TAG)
    framed = [r for r in recs if r.get("_has_frame")]
    splits = PI.split_pairs(framed)
    with pytest.raises(ValueError, match="outside the stored frame"):
        PI.direction_report(framed, PI.orthonormalize(frames), 3, splits=splits, unrelated=[])


def test_direction_stats_is_acute_because_the_stored_sign_is_arbitrary():
    """`plane_frame` takes its sign from a QR, so d and -d are one direction;
    a signed reading would give the same geometry two different answers."""
    cos = np.array([np.cos(np.deg2rad(70.0))])
    assert (PI.direction_stats(cos, 1)["theta_med"]
            == pytest.approx(PI.direction_stats(-cos, 1)["theta_med"], abs=1e-9))
    assert PI.direction_stats(cos, 1)["theta_med"] == pytest.approx(70.0, abs=1e-9)


def test_direction_stats_reports_no_pairs_rather_than_crashing():
    assert PI.direction_stats(np.array([]), 0) == {"pairs": 0}


def test_direction_stats_reports_no_minimum(report):
    """A minimum is an order statistic and these groups differ in size by more
    than an order of magnitude, so reporting one would invite comparing the
    smallest of 20,000 draws against the smallest of 500."""
    assert "theta_min" not in report["first_direction"]["unrelated"]
    assert "theta_min" not in report["random_direction_null"]


def test_random_direction_null_sits_just_under_ninety_and_is_drawn():
    null = PI.random_direction_null(4096, n_pairs=200, seed=3)
    assert 88.0 < null["theta_med"] < 90.0
    assert null["cos_med"] < 0.02
    assert null["theta_med"] != PI.random_direction_null(4096, n_pairs=200, seed=4)["theta_med"]
    assert "theta2" in null["also_the_null_for"]


def test_the_reports_unrelated_group_excludes_every_related_pair(store, report):
    """Leaving the same-noise pairs in the unrelated baseline would drag it
    toward the signal and shrink the very gap being reported. Counted through
    the report, not by re-running the sampler the way the module does — that
    would pass however the module actually calls it."""
    records = []
    for run_dir in sorted(store.glob("flux/*")):
        records.extend(PI.load_cell(run_dir, WHOLE_TAG)[0])
    framed = [r for r in records if r.get("_has_frame")]
    splits = PI.split_pairs(framed)
    related = (set(splits["same_noise_diff_prompt"]) | set(splits["same_prompt_diff_noise"])
               | set(splits["same_both"]))
    n = len(framed)
    assert report["first_direction"]["unrelated"]["pairs"] == n * (n - 1) // 2 - len(related)


def test_the_stratification_is_read_through_the_report_not_reimplemented(report):
    """The two halves have to be the report's own split, decided by the two
    records' datasets — checking a split the test computed itself would pass
    however the module partitions."""
    first = report["first_direction"]
    total = first["same_noise_diff_prompt"]["pairs"]
    same, cross = first["same_noise_same_dataset"], first["same_noise_cross_dataset"]
    assert same["pairs"] + cross["pairs"] == total
    assert same["pairs"] > 0 and cross["pairs"] > 0       # both non-empty on this grid
    # the seed arithmetic puts neighbouring indices in the same-dataset half and
    # never index 0 there; the cross-dataset half is the only place 0 can appear
    assert set(same["prompt_index_distances"]) <= {"1", "2"}
    assert "0" in cross["prompt_index_distances"]


def test_index_distances_counts_the_gap_between_the_two_prompt_indices():
    records = [{"prompt_idx": 0}, {"prompt_idx": 1}, {"prompt_idx": 5}]
    assert PI.index_distances(records, [(0, 1), (0, 2), (1, 2)]) == {"1": 1, "4": 1, "5": 1}


def test_pc1_against_the_other_chord_is_the_artefact_control(report):
    """If the stored PC1 were the partner's chord with its own taken out, this
    would return the chord-chord angle. Here the chord is shared by every
    trajectory (0 degrees) while PC1-vs-other-chord must not be."""
    assert report["chord_control"]["unrelated"]["theta_med"] < 2.0
    assert report["pc1_vs_other_chord"]["unrelated"]["theta_med"] > 60.0
    assert report["pc1_vs_other_chord"]["same_noise_diff_prompt"]["theta_med"] > 60.0


def test_the_report_removes_the_float16_chord_leak_and_says_how_much_is_left(store, report):
    """The store is float16, so the stored first bend direction comes back
    carrying a little of the chord. Every angle in the report is biased by
    whatever is left, so the report has to state it — and the stated figure has
    to come from frames that were actually re-orthonormalised (~1e-8), not from
    the raw ones (~1e-5)."""
    _, frames = PI.load_cell(sorted(store.glob("flux/*"))[0], WHOLE_TAG)
    raw_leak = float(np.abs(np.einsum("ij,ij->i", frames[:, 0, :].astype(np.float32),
                                      frames[:, 1, :].astype(np.float32))).max())
    assert raw_leak > 1e-6                               # the store really does leak
    assert report["max_chord_leak_into_pc1"] < 1e-6      # and the report really does fix it


# --------------------------------------------------------------------------
# alignment between records and the frame stack
# --------------------------------------------------------------------------

def test_a_missing_frame_does_not_shift_the_pair_indices(tmp_path):
    """A cell run without a frame for this segment leaves a record with no row
    in the stack. Pair indices address the STACK, so counting records instead
    would move every pair above the gap onto the wrong trajectory."""
    root = build_store(tmp_path, drop_whole_frame_at=("drawbench_full", 42, 2))
    entry = PI.model_report(sorted(root.glob("flux/*")), WHOLE_TAG, (EARLY_TAG, LATE_TAG))
    assert entry["records"] == 36 and entry["framed"] == 35
    # the planted signatures must survive the gap unchanged
    assert entry["first_direction"]["same_noise_diff_prompt"]["theta_med"] < 5.0
    assert entry["first_direction"]["unrelated"]["theta_med"] > 60.0
    assert entry["chord_control"]["unrelated"]["theta_med"] < 2.0


# --------------------------------------------------------------------------
# C. how far one trajectory's own plane turns
# --------------------------------------------------------------------------

def test_plane_turn_recovers_a_planted_turn(turning_store):
    turn = PI.plane_turn_report(sorted(turning_store.glob("flux/*")), EARLY_TAG, LATE_TAG)
    assert turn["trajectories"] == 36
    assert turn["theta1"]["med"] > 80.0
    assert turn["first_direction"]["med"] > 80.0


def test_the_steady_store_keeps_one_plane_across_both_windows(store):
    turn = PI.plane_turn_report(sorted(store.glob("flux/*")), EARLY_TAG, LATE_TAG)
    assert turn["trajectories"] == 36
    assert turn["theta1"]["med"] < 10.0


def test_a_shared_plane_does_not_force_a_shared_first_direction():
    """Both bend components live in one plane the whole way, but their
    envelopes cross: the first dominates early, the second late. The plane is
    identical in both windows and the leading direction inside it is not.

    This is why the report gives the plane and the first direction separately
    instead of treating either as a proxy for the other.
    """
    t = np.linspace(0.0, 1.0, N + 1)
    basis = np.linalg.qr(np.random.default_rng(11).standard_normal((D, 3)))[0].T
    wave = np.sin(6 * np.pi * t)
    Z = (np.outer(20.0 * t, basis[0])
         + np.outer((1.0 - t) * wave, basis[1]) + np.outer(t * wave, basis[2]))
    early = plane_frame(Z[EARLY[0]:EARLY[1]])
    late = plane_frame(Z[LATE[0]:LATE[1]])
    t1, _ = PI.plane_angles(early[1:], late[1:])
    pc1 = np.degrees(np.arccos(np.clip(abs(float(early[1] @ late[1])), 0.0, 1.0)))
    assert t1 < 5.0                                       # the same plane
    assert pc1 > 45.0                                     # a different direction inside it
    assert pc1 >= t1


def test_the_first_direction_angle_can_never_fall_below_the_plane_angle(store, turning_store):
    """theta1 is the smallest angle between ANY direction of one plane and any
    of the other, so a particular direction pair cannot beat it. Breaking this
    means the wrong frames were paired."""
    for root in (store, turning_store):
        turn = PI.plane_turn_report(sorted(root.glob("flux/*")), EARLY_TAG, LATE_TAG)
        assert turn["first_direction"]["med"] >= turn["theta1"]["med"] - 1e-6


def test_plane_turn_matches_by_prompt_index_not_by_stack_position(tmp_path):
    """Dropping one late frame must not shift every later trajectory onto the
    wrong partner. The median survives 3 bad pairs out of 17, so the maximum is
    what has to be asserted."""
    root = build_store(tmp_path, drop_late_frame_at=("drawbench_full", 42, 2))
    turn = PI.plane_turn_report(sorted(root.glob("flux/drawbench_full*")), EARLY_TAG, LATE_TAG)
    assert turn["trajectories"] == 17                     # 18 minus the dropped one
    assert turn["theta1"]["max"] < 10.0                   # every pair, not just most


def test_plane_turn_refuses_to_report_more_trajectories_than_angles(monkeypatch, store):
    """The count and the angles are accumulated separately, so they have to be
    checked against each other or the report can claim 36 alongside 30."""
    real = PI.pair_angles_deg
    monkeypatch.setattr(PI, "pair_angles_deg",
                        lambda gram, pairs: tuple(a[:-1] for a in real(gram, pairs)))
    with pytest.raises(ValueError, match="counted"):
        PI.plane_turn_report(sorted(store.glob("flux/*")), EARLY_TAG, LATE_TAG)


def test_plane_turn_reports_nothing_rather_than_guessing_when_a_window_is_absent(tmp_path):
    root = build_store(tmp_path, n_prompts=2)
    for path in root.glob(f"flux/*/frame_*_{LATE_TAG}.npy"):
        path.unlink()
    for path in root.glob("flux/*/traj_*.json"):
        record = json.loads(path.read_text())
        record["frame_files"].pop(LATE_TAG, None)
        path.write_text(json.dumps(record))
    assert PI.plane_turn_report(sorted(root.glob("flux/*")), EARLY_TAG, LATE_TAG) == {
        "trajectories": 0}


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_report_carries_what_it_was_measured_on(report):
    assert report["segment"] == WHOLE_TAG
    assert report["dim"] == D
    assert report["cells"] == 6 and report["records"] == 36 and report["framed"] == 36
    assert report["devices"] == ["SYNTHETIC"]
    assert report["plane_turn"]["segments"] == [EARLY_TAG, LATE_TAG]


def test_model_report_refuses_a_segment_with_no_frames(tmp_path):
    root = build_store(tmp_path, n_prompts=2)
    for path in root.glob(f"flux/*/frame_*_{LATE_TAG}.npy"):
        path.unlink()
    for path in root.glob("flux/*/traj_*.json"):
        record = json.loads(path.read_text())
        record["frame_files"].pop(LATE_TAG, None)
        path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="no frames stored"):
        PI.model_report(sorted(root.glob("flux/*")), LATE_TAG, (EARLY_TAG, LATE_TAG))
