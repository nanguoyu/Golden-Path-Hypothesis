"""Unit tests for analysis/trajectory_math.py on synthetic trajectories with
closed-form answers: a straight line (zero deviation / turn / curvature,
straightness 1), a planar circular arc (constant known turn angle), a planar
case with hand-set variances (PCA), and the scaling / rotation invariances the
ratio metrics are supposed to have.

CPU + numpy only: no GPU, no model, no diffusers.
"""

import math

import numpy as np
import pytest

from analysis import trajectory_math as TM


D = 37  # ambient dimension; anything > 3 exercises the Gram trick


def embed(planar: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Lift 2-D coordinates into R^D along the orthonormal pair (e1, e2)."""
    return np.outer(planar[:, 0], e1) + np.outer(planar[:, 1], e2)


def orthonormal_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(D)
    b = rng.standard_normal(D)
    e1 = a / np.linalg.norm(a)
    b = b - (b @ e1) * e1
    e2 = b / np.linalg.norm(b)
    return e1, e2


def straight_line(n_steps: int = 50) -> np.ndarray:
    """Line along a generic (non-axis-aligned) direction."""
    e1, _ = orthonormal_pair(1)
    return np.outer(np.linspace(0.0, 3.0, n_steps + 1) + 0.7, e1)


def axis_line(n_steps: int = 10) -> np.ndarray:
    """Line along coordinate axis 0: collinear to the last bit, so the
    degenerate branch of orthogonal_pca is reached exactly."""
    Z = np.zeros((n_steps + 1, D))
    Z[:, 0] = np.linspace(0.0, 3.0, n_steps + 1)
    return Z


def circular_arc(n_steps: int, delta_deg: float, radius: float = 2.0) -> np.ndarray:
    """n_steps+1 points equally spaced by `delta_deg` on a circle of `radius`."""
    e1, e2 = orthonormal_pair(2)
    ang = np.deg2rad(delta_deg) * np.arange(n_steps + 1)
    planar = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    return embed(planar, e1, e2)


def sigmas_for(n_steps: int) -> list[float]:
    """FLUX-shaped schedule: N descending values plus the trailing 0."""
    return list(np.linspace(1.0, 1.0 / n_steps, n_steps)) + [0.0]


# --------------------------------------------------------------------------
# straight line: the degenerate reference case
# --------------------------------------------------------------------------

def test_straight_line_has_no_deviation_curvature_or_turn():
    Z = straight_line()
    m = TM.trajectory_metrics(Z, sigmas_for(50))
    assert np.allclose(m["d_perp"], 0.0, atol=1e-13)
    assert m["max_dev_ratio"] == pytest.approx(0.0, abs=1e-13)
    assert np.allclose(m["turn_angle_deg"], 0.0, atol=1e-10)
    assert np.allclose(m["second_diff_norm"], 0.0, atol=1e-13)


def test_straight_line_straightness_is_one():
    Z = straight_line()
    path_len, straightness = TM.path_length(Z)
    _, chord_len = TM.chord(Z)
    assert straightness == pytest.approx(1.0, rel=1e-9)
    assert path_len == pytest.approx(chord_len, rel=1e-9)


def test_collinear_trajectory_hits_the_zero_variance_branch():
    pca = TM.orthogonal_pca(axis_line())
    assert pca["perp_var_total"] == 0.0
    assert pca["pca_evr"] == [0.0] * 5
    for k in (1, 2, 3):
        assert pca[f"recon_err_{k}d"] == 0.0
        assert pca[f"recon_err_rel_{k}d"] == 0.0


def test_straight_line_orthogonal_variance_is_rounding_only():
    """Same line along a generic direction: the residual is float64 noise, not
    structure, so it must stay ~30 orders below the chord energy."""
    Z = straight_line()
    _, chord_len = TM.chord(Z)
    pca = TM.orthogonal_pca(Z)
    assert pca["perp_var_total"] < 1e-24 * chord_len ** 2
    assert pca["recon_err_1d"] < 1e-12 * chord_len


def test_straight_line_spacing_and_velocity_are_exact():
    n = 10
    e1, _ = orthonormal_pair(1)
    Z = np.outer(np.linspace(0.0, 5.0, n + 1), e1)
    assert np.allclose(TM.step_spacing(Z), 0.5, atol=1e-12)
    # uniform sigma spacing of 0.1 -> velocity = spacing / 0.1
    sig = np.linspace(1.0, 0.0, n + 1)
    assert np.allclose(TM.velocity_norms(Z, sig), 5.0, atol=1e-10)


# --------------------------------------------------------------------------
# circular arc: known constant turn angle and curvature
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta_deg", [1.0, 5.0, 17.5])
def test_arc_turn_angle_equals_step_angle(delta_deg):
    Z = circular_arc(20, delta_deg)
    turns = TM.turn_angles_deg(Z)
    assert len(turns) == 19
    assert np.allclose(turns, delta_deg, atol=1e-3)


def test_arc_spacing_and_curvature_are_the_chord_formulas():
    delta_deg, radius, n = 6.0, 2.0, 20
    Z = circular_arc(n, delta_deg, radius=radius)
    d = math.radians(delta_deg)
    expected_spacing = 2.0 * radius * math.sin(d / 2.0)
    assert np.allclose(TM.step_spacing(Z), expected_spacing, rtol=1e-5)
    # consecutive equal-length chords separated by angle d
    expected_curv = 2.0 * expected_spacing * math.sin(d / 2.0)
    assert np.allclose(TM.second_difference_norms(Z), expected_curv, rtol=1e-4)


def test_arc_max_deviation_ratio_matches_circular_segment_geometry():
    # Half-circle: chord = 2R, max perpendicular deviation = R -> ratio 0.5.
    Z = circular_arc(180, 1.0, radius=2.0)
    _, _, ratio = TM.deviation_profile(Z)
    assert ratio == pytest.approx(0.5, rel=1e-4)


def test_arc_deviation_endpoints_are_zero():
    d_perp, chord_len, _ = TM.deviation_profile(circular_arc(30, 4.0))
    assert chord_len > 0
    assert d_perp[0] == pytest.approx(0.0, abs=1e-5)
    assert d_perp[-1] == pytest.approx(0.0, abs=1e-5)


def test_arc_is_less_straight_than_its_chord():
    _, straightness = TM.path_length(circular_arc(60, 3.0))
    assert straightness > 1.0


# --------------------------------------------------------------------------
# PCA: hand-set variances in the chord-orthogonal plane
# --------------------------------------------------------------------------

# chord runs along e0 (both residual coordinates vanish at the endpoints); the
# e1 / e2 coordinates are already centred and mutually orthogonal, so the two
# non-zero eigenvalues are exactly their sums of squares: 16 and 4.
PLANTED_ALONG = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
PLANTED_A = np.array([0.0, 2.0, -2.0, 2.0, -2.0, 0.0])   # sum a^2 = 16
PLANTED_B = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.0])   # sum b^2 = 4, sum a*b = 0
LAM_HI, LAM_LO = 16.0, 4.0


def build_known_variance_case() -> np.ndarray:
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.standard_normal((D, 3)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    return (np.outer(PLANTED_ALONG, e0)
            + np.outer(PLANTED_A, e1)
            + np.outer(PLANTED_B, e2))


def test_pca_recovers_the_two_planted_eigenvalues():
    pca = TM.orthogonal_pca(build_known_variance_case())
    total = LAM_HI + LAM_LO
    assert pca["perp_var_total"] == pytest.approx(total, rel=1e-12)
    assert pca["pca_evr"][0] == pytest.approx(LAM_HI / total, rel=1e-12)
    assert pca["pca_evr"][1] == pytest.approx(LAM_LO / total, rel=1e-12)
    assert pca["pca_evr"][2] == pytest.approx(0.0, abs=1e-12)


def test_pca_reconstruction_errors_match_the_eigenvalue_tail():
    # paper convention: k-D approximation keeps the chord + (k - 1) top PCs,
    # so 1-D error is the full perpendicular residual and 3-D is exact here.
    pca = TM.orthogonal_pca(build_known_variance_case())
    assert pca["recon_err_1d"] == pytest.approx(
        math.sqrt(LAM_HI + LAM_LO), rel=1e-10)
    assert pca["recon_err_2d"] == pytest.approx(math.sqrt(LAM_LO), rel=1e-10)
    assert pca["recon_err_3d"] == pytest.approx(0.0, abs=1e-6)
    assert pca["recon_err_rel_1d"] == pytest.approx(1.0, rel=1e-12)
    assert pca["recon_err_rel_2d"] == pytest.approx(
        math.sqrt(LAM_LO / (LAM_HI + LAM_LO)), rel=1e-10)


def test_pca_is_invariant_to_a_constant_offset():
    """Shifting every point by a constant chord-orthogonal vector changes
    neither the chord nor the centred residuals; an implementation that skips
    mean-centering picks up the offset as spurious variance and fails."""
    Z = build_known_variance_case()
    rng = np.random.default_rng(11)
    e1 = np.linalg.qr(rng.standard_normal((D, 1)))[0][:, 0]
    # any fixed offset direction works as long as it is not chord-parallel;
    # project out the chord so the offset fully survives into the residuals.
    u, _ = TM.chord(Z)
    off = e1 - (e1 @ u) * u
    shifted = Z + 3.0 * off[None, :]
    base = TM.orthogonal_pca(Z)
    moved = TM.orthogonal_pca(shifted)
    assert moved["perp_var_total"] == pytest.approx(
        base["perp_var_total"], rel=1e-9)
    assert moved["pca_evr"] == pytest.approx(base["pca_evr"], abs=1e-9)


def test_velocity_alignment_on_a_nonuniform_sigma_schedule():
    """Euler flow-matching integration with distinct per-step velocities and
    strictly non-uniform sigma gaps: velocity_norms must return each ||v_n||
    at its own index (any index misalignment divides by the wrong gap)."""
    sig = np.array([1.0, 0.7, 0.45, 0.25, 0.1, 0.0])
    speeds = np.array([3.0, 5.0, 2.0, 7.0, 4.0])
    e1, _ = orthonormal_pair(3)
    Z = np.zeros((6, D))
    for n in range(5):
        Z[n + 1] = Z[n] + (sig[n + 1] - sig[n]) * (speeds[n] * e1)
    assert TM.velocity_norms(Z, sig) == pytest.approx(speeds, rel=1e-12)


def test_pca_ignores_variance_along_the_chord():
    """Stretching the trajectory along the chord must not move any EVR."""
    Z = build_known_variance_case()
    u, _ = TM.chord(Z)
    stretched = Z + np.outer(np.linspace(0.0, 9.0, Z.shape[0]) ** 2, u)
    assert TM.orthogonal_pca(stretched)["pca_evr"] == pytest.approx(
        TM.orthogonal_pca(Z)["pca_evr"], rel=1e-7)


# --------------------------------------------------------------------------
# invariances
# --------------------------------------------------------------------------

def test_ratio_metrics_are_scale_invariant():
    Z = circular_arc(40, 4.0)
    base = TM.trajectory_metrics(Z, sigmas_for(40))
    scaled = TM.trajectory_metrics(3.75 * Z, sigmas_for(40))
    for key in ("max_dev_ratio", "straightness"):
        assert scaled[key] == pytest.approx(base[key], rel=1e-8)
    assert scaled["turn_angle_deg"] == pytest.approx(base["turn_angle_deg"], rel=1e-6)
    assert scaled["pca_evr"] == pytest.approx(base["pca_evr"], rel=1e-8)


def test_length_metrics_scale_linearly():
    Z = circular_arc(40, 4.0)
    c = 3.75
    base = TM.trajectory_metrics(Z, sigmas_for(40))
    scaled = TM.trajectory_metrics(c * Z, sigmas_for(40))
    for key in ("chord_len", "path_len"):
        assert scaled[key] == pytest.approx(c * base[key], rel=1e-8)
    for key in ("spacing", "magnitude", "second_diff_norm", "velocity_norm", "d_perp"):
        assert scaled[key] == pytest.approx(list(c * np.asarray(base[key])), rel=1e-7)


def test_reconstruction_error_scales_linearly():
    # measured on the planted case, where recon_err_1d is a real quantity
    # (sqrt(4) = 2) rather than the rounding floor a planar arc leaves behind
    c = 3.75
    Z = build_known_variance_case()
    assert TM.orthogonal_pca(c * Z)["recon_err_1d"] == pytest.approx(
        c * TM.orthogonal_pca(Z)["recon_err_1d"], rel=1e-10)


def test_metrics_are_rotation_invariant():
    rng = np.random.default_rng(11)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    Z = circular_arc(40, 4.0)
    base = TM.trajectory_metrics(Z, sigmas_for(40))
    rotated = TM.trajectory_metrics(Z @ Q.T, sigmas_for(40))
    for key in ("chord_len", "path_len", "max_dev_ratio", "straightness"):
        assert rotated[key] == pytest.approx(base[key], rel=1e-7)
    assert rotated["pca_evr"] == pytest.approx(base["pca_evr"], rel=1e-7)


def test_velocity_is_invariant_to_uniform_sigma_rescaling():
    Z = circular_arc(20, 4.0)
    sig = np.linspace(1.0, 0.0, 21)
    v1 = np.asarray(TM.velocity_norms(Z, sig))
    v2 = np.asarray(TM.velocity_norms(Z, 2.0 * sig))
    assert np.allclose(v1, 2.0 * v2, rtol=1e-10)


# --------------------------------------------------------------------------
# shapes, dtypes and degenerate inputs
# --------------------------------------------------------------------------

def test_metric_array_lengths_follow_n():
    n = 50
    Z = circular_arc(n, 2.0)
    m = TM.trajectory_metrics(Z, sigmas_for(n))
    assert len(m["d_perp"]) == n + 1
    assert len(m["magnitude"]) == n + 1
    assert len(m["spacing"]) == n
    assert len(m["velocity_norm"]) == n
    assert len(m["turn_angle_deg"]) == n - 1
    assert len(m["second_diff_norm"]) == n - 1
    assert len(m["pca_evr"]) == 5


def test_outputs_are_plain_floats():
    m = TM.trajectory_metrics(circular_arc(6, 5.0), sigmas_for(6))
    assert all(type(v) is float for v in m["spacing"])
    assert all(type(v) is float for v in m["pca_evr"])
    assert type(m["max_dev_ratio"]) is float


def test_accepts_torch_cpu_tensors():
    torch = pytest.importorskip("torch")
    n = 8
    Z = circular_arc(n, 5.0)
    ref = TM.trajectory_metrics(Z, sigmas_for(n))
    got = TM.trajectory_metrics(torch.from_numpy(Z), torch.tensor(sigmas_for(n)))
    assert got["max_dev_ratio"] == pytest.approx(ref["max_dev_ratio"], rel=1e-6)
    assert got["spacing"] == pytest.approx(ref["spacing"], rel=1e-6)


def test_fp32_input_agrees_with_fp64_to_capture_precision():
    """The probes hand in fp32 latents; the float64 promotion inside must keep
    the ratio metrics accurate to the input's own precision, not worse."""
    n = 40
    Z = circular_arc(n, 4.0)
    ref = TM.trajectory_metrics(Z, sigmas_for(n))
    got = TM.trajectory_metrics(Z.astype(np.float32), sigmas_for(n))
    assert got["max_dev_ratio"] == pytest.approx(ref["max_dev_ratio"], rel=1e-5)
    assert got["straightness"] == pytest.approx(ref["straightness"], rel=1e-5)
    assert got["pca_evr"] == pytest.approx(ref["pca_evr"], abs=1e-5)


def test_degenerate_chord_reports_nan_ratios():
    e1, e2 = orthonormal_pair(3)
    Z = np.stack([np.zeros(D), e1, e2, np.zeros(D)])
    d_perp, chord_len, ratio = TM.deviation_profile(Z)
    assert chord_len == 0.0
    assert math.isnan(ratio)
    assert all(math.isnan(v) for v in d_perp)
    assert math.isnan(TM.path_length(Z)[1])


def test_zero_sigma_gap_yields_nan_velocity():
    Z = circular_arc(3, 5.0)
    v = TM.velocity_norms(Z, [1.0, 0.5, 0.5, 0.0])
    assert math.isnan(v[1])
    assert not math.isnan(v[0])


def test_shape_mismatches_raise():
    Z = circular_arc(4, 5.0)
    with pytest.raises(ValueError):
        TM.velocity_norms(Z, [1.0, 0.0])
    with pytest.raises(ValueError):
        TM.trajectory_metrics(Z.ravel(), sigmas_for(4))


# --------------------------------------------------------------------------
# update_subspace: dimensionality of the step-to-step update
# --------------------------------------------------------------------------

def test_update_subspace_on_a_straight_line() -> None:
    """Every update is pure chord: chord_share 1, no off-chord content."""
    e1, _ = orthonormal_pair(1)
    Z = np.outer(np.linspace(0.0, 5.0, 11), e1)
    out = TM.update_subspace(Z)
    assert out["chord_share"] == pytest.approx(1.0, abs=1e-12)
    # off-chord content is rounding noise only: reported as nan, not fitted
    assert math.isnan(out["in_position_plane"])


def test_update_subspace_on_a_planar_arc_is_fully_in_plane() -> None:
    """A planar arc's updates lie in the same plane as its positions."""
    Z = circular_arc(20, 5.0)
    out = TM.update_subspace(Z)
    assert out["in_position_plane"] == pytest.approx(1.0, abs=1e-8)
    assert out["own_evr"][0] + out["own_evr"][1] == pytest.approx(1.0, abs=1e-8)


def test_update_subspace_detects_out_of_plane_update_content() -> None:
    """Add a zig-zag along a third direction: the positions stay near the
    plane (the zig-zag cancels) while the updates do not - exactly the case
    the metric exists to separate."""
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.standard_normal((D, 4)))[0]
    e0, e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]
    n = 20
    along = np.linspace(0.0, 10.0, n + 1)
    # a smooth 2-D bend dominates the position variance, so the top-2 plane is
    # (e1, e2); the small alternating term in e3 is left out of that plane but
    # is doubled by every difference
    bend1 = np.sin(np.linspace(0.0, np.pi, n + 1))
    bend2 = 0.6 * np.sin(np.linspace(0.0, 2 * np.pi, n + 1))
    zig = 0.05 * ((-1.0) ** np.arange(n + 1))
    Z = (np.outer(along, e0) + np.outer(bend1, e1)
         + np.outer(bend2, e2) + np.outer(zig, e3))
    out = TM.update_subspace(Z)
    pos = TM.orthogonal_pca(Z)
    # positions: the alternating component is small in absolute terms
    assert pos["pca_evr"][0] + pos["pca_evr"][1] > 0.99
    # updates: differencing doubles it every step, so it now shows up
    assert out["in_position_plane"] < 0.9
    assert out["own_evr"][0] + out["own_evr"][1] < 1.0


def test_update_subspace_is_scale_invariant() -> None:
    Z = circular_arc(15, 4.0)
    a = TM.update_subspace(Z)
    b = TM.update_subspace(7.3 * Z)
    assert b["chord_share"] == pytest.approx(a["chord_share"], rel=1e-10)
    assert b["in_position_plane"] == pytest.approx(a["in_position_plane"], rel=1e-10)


def test_trajectory_metrics_carries_the_update_fields() -> None:
    Z = circular_arc(12, 3.0)
    m = TM.trajectory_metrics(Z, sigmas_for(12))
    for key in ("update_chord_share", "update_in_position_plane", "update_own_evr"):
        assert key in m
    assert len(m["update_own_evr"]) == 5


def test_update_subspace_needs_the_centred_chord_orthogonal_residual() -> None:
    """The plane must come from the CENTRED, chord-removed positions.

    Fixture: the bending genuinely spans (e1, e2), a large constant offset sits
    in e3, and the travel is along e0. Skipping the centring drags e3 into the
    plane; skipping the chord removal drags e0 in. Either way the updates -
    which live purely in (e1, e2) - stop being fully covered, so the exact 1.0
    here is what pins both steps down.
    """
    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.standard_normal((D, 5)))[0]
    e0, e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2], basis[:, 3]
    n = 24
    t = np.linspace(0.0, 1.0, n + 1)
    Z = (np.outer(10.0 * t, e0)
         + np.outer(np.sin(np.pi * t), e1)
         + np.outer(0.6 * np.sin(2 * np.pi * t), e2)
         + 10.0 * e3)
    out = TM.update_subspace(Z)
    assert out["in_position_plane"] == pytest.approx(1.0, abs=1e-8)


def test_update_own_plane_is_never_worse_than_the_position_plane() -> None:
    """`own_evr` is the updates' own best 2-plane, `in_position_plane` scores
    them in the positions' plane, so by SVD optimality own2 >= in_plane always.
    A strict gap on a fixture built to separate the two catches the circular
    variant that fits the plane to the updates themselves and reports it as if
    it were the positions' plane.
    """
    rng = np.random.default_rng(13)
    basis = np.linalg.qr(rng.standard_normal((D, 6)))[0]
    e0, e1, e2, e3, e4 = (basis[:, i] for i in range(5))
    n = 40
    t = np.linspace(0.0, 1.0, n + 1)
    k = np.arange(n + 1)
    # slow bends carry the position variance (99% of it stays in e1, e2), while
    # two small alternating terms - period 2 and period 4 - are magnified by
    # every difference and take over the updates. The planes therefore differ
    # by construction, which is what makes the gap below meaningful.
    Z = (np.outer(10.0 * t, e0)
         + np.outer(np.sin(np.pi * t), e1)
         + np.outer(0.8 * np.sin(2.0 * np.pi * t), e2)
         + np.outer(0.05 * (-1.0) ** k, e3)
         + np.outer(0.05 * np.where((k // 2) % 2 == 0, 1.0, -1.0), e4))
    assert sum(TM.orthogonal_pca(Z)["pca_evr"][:2]) > 0.98   # positions: still (e1, e2)
    out = TM.update_subspace(Z)
    own2 = out["own_evr"][0] + out["own_evr"][1]
    assert own2 >= out["in_position_plane"] - 1e-12
    assert own2 - out["in_position_plane"] > 0.10


def test_update_own_plane_dominates_on_every_synthetic_fixture() -> None:
    """own2 >= in_position_plane is an SVD identity, not a fixture accident."""
    for Z in (circular_arc(20, 5.0), circular_arc(30, 2.0, radius=5.0),
              build_known_variance_case()):
        out = TM.update_subspace(Z)
        if math.isnan(out["in_position_plane"]):
            continue
        assert out["own_evr"][0] + out["own_evr"][1] >= out["in_position_plane"] - 1e-10


# --------------------------------------------------------------------------
# coarse-window turn angle
# --------------------------------------------------------------------------

def bent_3d_path(n: int = 50, seed: int = 7) -> np.ndarray:
    """A path whose chord-orthogonal residual has honest rank >= 2, i.e. the
    shape `plane_frame` is defined for. A planar arc is deliberately NOT that:
    its chord lies in its own plane, leaving a rank-1 residual."""
    basis = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, 4)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    t = np.linspace(0.0, 1.0, n + 1)
    return (np.outer(6.0 * t, e0)
            + np.outer(np.sin(np.pi * t), e1)
            + np.outer(0.4 * np.sin(2.0 * np.pi * t), e2))


@pytest.mark.parametrize("window", [1, 2, 5])
@pytest.mark.parametrize("delta_deg", [0.5, 3.0])
def test_window_turn_angle_on_an_arc_is_window_times_the_step_angle(window, delta_deg):
    """Successive equal chords of a circle turn by exactly the arc angle they
    subtend, so a w-step window must read w * delta. This is the identity that
    separates the correct pair of segments (n->n+w, n+w->n+2w) from the
    plausible wrong ones: pairing n->n+w with n->n+2w reads w*delta/2, and
    pairing n->n+w with n+w->n+2w+1 reads (2w+1)*delta/2."""
    n = 40
    Z = circular_arc(n, delta_deg)
    got = TM.turn_angles_window_deg(Z, window)
    assert len(got) == (n + 1) - 2 * window
    assert np.allclose(got, window * delta_deg, rtol=1e-9)


def test_window_one_reproduces_the_single_step_turn_angle():
    Z = bent_3d_path(30)
    assert TM.turn_angles_window_deg(Z, 1) == pytest.approx(TM.turn_angles_deg(Z), rel=1e-12)


def test_window_centers_line_up_with_the_profile():
    Z = bent_3d_path(30)
    for window in (1, 3, 5):
        profile = TM.turn_angles_window_deg(Z, window)
        centers = TM.window_centers(Z.shape[0], window)
        assert len(centers) == len(profile)
        assert centers[0] == window
        assert centers[-1] == Z.shape[0] - window - 1


def test_window_turn_angle_is_scale_and_rotation_invariant():
    Z = bent_3d_path(30)
    rng = np.random.default_rng(5)
    rot = np.linalg.qr(rng.standard_normal((D, D)))[0]
    ref = TM.turn_angles_window_deg(Z)
    assert TM.turn_angles_window_deg(3.7 * Z) == pytest.approx(ref, rel=1e-10)
    assert TM.turn_angles_window_deg(Z @ rot + 4.0) == pytest.approx(ref, abs=1e-9)


def test_window_turn_angle_reports_nan_on_a_stalled_window():
    Z = bent_3d_path(20)
    Z[5:11] = Z[5]  # window 5 centred on 10 has a zero-length incoming segment
    got = TM.turn_angles_window_deg(Z, 5)
    assert math.isnan(got[TM.window_centers(Z.shape[0], 5).index(10)])


def test_window_turn_angle_is_empty_when_the_path_is_too_short():
    assert TM.turn_angles_window_deg(bent_3d_path(8), 5) == []
    assert TM.window_centers(9, 5) == []


def test_window_turn_angle_rejects_a_nonpositive_window():
    with pytest.raises(ValueError):
        TM.turn_angles_window_deg(bent_3d_path(20), 0)


def test_metrics_carry_the_coarse_window_profile():
    n = 50
    m = TM.trajectory_metrics(bent_3d_path(n), sigmas_for(n))
    key = f"turn_angle_w{TM.CURVATURE_WINDOW}_deg"
    assert len(m[key]) == (n + 1) - 2 * TM.CURVATURE_WINDOW
    assert all(type(v) is float for v in m[key])


# --------------------------------------------------------------------------
# plane_frame / principal_angles_deg
# --------------------------------------------------------------------------

def test_plane_frame_is_orthonormal_and_leads_with_the_chord():
    Z = bent_3d_path()
    frame = TM.plane_frame(Z)
    assert frame.shape == (3, D)
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    u, _ = TM.chord(Z)
    assert abs(abs(float(frame[0] @ u)) - 1.0) < 1e-12


def test_plane_frame_spans_the_directions_the_trajectory_actually_occupies():
    """Built in span(e0, e1, e2); the frame must span exactly that, so the
    residual after projecting the centred path onto it is zero."""
    basis = np.linalg.qr(np.random.default_rng(11).standard_normal((D, 4)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    t = np.linspace(0.0, 1.0, 41)
    Z = (np.outer(5.0 * t, e0) + np.outer(np.sin(np.pi * t), e1)
         + np.outer(0.3 * np.sin(2 * np.pi * t), e2))
    frame = TM.plane_frame(Z)
    centred = Z - Z.mean(axis=0, keepdims=True)
    residual = centred - (centred @ frame.T) @ frame
    assert float((residual ** 2).sum()) < 1e-18 * float((centred ** 2).sum())


def test_plane_frame_rows_1_2_carry_the_top_two_chord_orthogonal_variances():
    """The stored rows must be the same object `orthogonal_pca` scores, else
    the frame and the EVR columns describe different planes."""
    Z = bent_3d_path()
    frame = TM.plane_frame(Z)
    u, _ = TM.chord(Z)
    centred = Z - Z.mean(axis=0, keepdims=True)
    P = centred - np.outer(centred @ u, u)
    share = float(((P @ frame[1:].T) ** 2).sum() / (P ** 2).sum())
    evr = TM.orthogonal_pca(Z)["pca_evr"]
    assert share == pytest.approx(evr[0] + evr[1], abs=1e-10)


def test_plane_frame_declines_a_rank_deficient_residual():
    """A planar arc's chord lies in its own plane, so only ONE chord-orthogonal
    direction exists; the second QR row would be an arbitrary direction that
    would then pollute every pairwise angle it entered."""
    assert TM.plane_frame(circular_arc(20, 5.0)) is None
    assert TM.plane_frame(straight_line()) is None
    assert TM.plane_frame(np.zeros((6, D))) is None


def test_plane_frame_declines_a_closed_path_with_a_real_spread():
    """A trajectory that returns to its start has no chord at all. The residual
    is nowhere near zero, so only the chord guard can catch this - the energy
    guard cannot."""
    e1, e2 = orthonormal_pair(3)
    Z = np.stack([np.zeros(D), e1, e2, np.zeros(D)])
    assert float((Z - Z.mean(axis=0)).__pow__(2).sum()) > 1.0  # real spread
    assert TM.plane_frame(Z) is None


def test_plane_frame_keeps_a_faint_but_real_second_direction():
    """The rank guard must reject only what is numerically unresolved. Real
    FLUX trajectories sit at lambda2/lambda1 ~ 0.12-0.19; a guard tightened
    anywhere near that would refuse to store most of the grid, so the fixture
    is six orders of magnitude below real data and must still come back."""
    basis = np.linalg.qr(np.random.default_rng(21).standard_normal((D, 4)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    t = np.linspace(0.0, 1.0, 41)
    Z = (np.outer(6.0 * t, e0) + np.outer(np.sin(np.pi * t), e1)
         + np.outer(1e-3 * np.sin(2 * np.pi * t), e2))
    evr = TM.orthogonal_pca(Z)["pca_evr"]
    assert 1e-7 < evr[1] / evr[0] < 1e-5
    frame = TM.plane_frame(Z)
    assert frame is not None and frame.shape == (3, D)


def test_curvature_window_constants_are_pinned():
    """These constants are record field names, so changing one silently renames
    a field across the whole store and moves every number. w=7 is not
    decoration: w=5 sits at 2.6x the bf16 floor in the middle of a Qwen
    trajectory and biased its measured trough four steps late, and a stored
    profile cannot be recomputed without the latents.
    """
    assert TM.CURVATURE_WINDOW == 5
    assert TM.CURVATURE_WINDOWS == (5, 7)
    assert TM.PLANE_PCS == 2
    n = 30
    m = TM.trajectory_metrics(bent_3d_path(n), sigmas_for(n))
    for w in TM.CURVATURE_WINDOWS:
        key = f"turn_angle_w{w}_deg"
        assert key in m, key
        assert len(m[key]) == (n + 1) - 2 * w
        assert m[key] == pytest.approx(TM.turn_angles_window_deg(bent_3d_path(n), w))


def test_principal_angles_identical_and_orthogonal_subspaces():
    basis = np.linalg.qr(np.random.default_rng(2).standard_normal((D, 4)))[0]
    a, b = basis[:, :2].T, basis[:, 2:4].T
    assert TM.principal_angles_deg(a, a) == pytest.approx([0.0, 0.0], abs=2e-6)
    assert TM.principal_angles_deg(a, b) == pytest.approx([90.0, 90.0], abs=1e-6)


def test_principal_angles_recover_a_planted_rotation():
    """Share one direction exactly, tilt the other by a known angle: the pair
    must come back as (0, tilt) in ascending order."""
    basis = np.linalg.qr(np.random.default_rng(4).standard_normal((D, 3)))[0]
    e0, e1, e2 = basis[:, 0], basis[:, 1], basis[:, 2]
    tilt = 23.0
    a = np.stack([e0, e1])
    b = np.stack([e0, math.cos(math.radians(tilt)) * e1 + math.sin(math.radians(tilt)) * e2])
    assert TM.principal_angles_deg(a, b) == pytest.approx([0.0, tilt], abs=1e-8)


def test_principal_angles_are_symmetric_and_basis_independent():
    rng = np.random.default_rng(6)
    basis = np.linalg.qr(rng.standard_normal((D, 4)))[0]
    a, b = basis[:, :2].T, np.linalg.qr(rng.standard_normal((D, 2)))[0].T
    ref = TM.principal_angles_deg(a, b)
    assert TM.principal_angles_deg(b, a) == pytest.approx(ref, abs=1e-9)
    # a different basis of the SAME subspace must not move the answer
    mix = np.linalg.qr(rng.standard_normal((2, 2)))[0]
    assert TM.principal_angles_deg(mix @ a, b) == pytest.approx(ref, abs=1e-9)


def test_principal_angles_survive_the_float16_store():
    """The frames go to disk as float16; that must not move an angle enough to
    matter, and the re-orthonormalization must absorb the lost orthogonality."""
    rng = np.random.default_rng(8)
    big = 4096  # closer to the real 2.6e5 than D, still cheap
    a = np.linalg.qr(rng.standard_normal((big, 2)))[0].T
    b = np.linalg.qr(rng.standard_normal((big, 2)))[0].T
    ref = TM.principal_angles_deg(a, b)
    got = TM.principal_angles_deg(a.astype(np.float16), b.astype(np.float16))
    assert got == pytest.approx(ref, abs=0.01)


def test_principal_angles_reject_mismatched_spaces():
    with pytest.raises(ValueError):
        TM.principal_angles_deg(np.eye(2, 7), np.eye(2, 9))
    with pytest.raises(ValueError):
        TM.principal_angles_deg(np.ones(7), np.eye(2, 7))


# --------------------------------------------------------------------------
# segment specs (what decides which frames a run stores)
# --------------------------------------------------------------------------

def test_parse_segments_round_trips_a_valid_spec():
    assert TM.parse_segments("0:51,0:16,25:51,32:51,38:51", 51) == [
        (0, 51), (0, 16), (25, 51), (32, 51), (38, 51)]
    assert TM.parse_segments(" 0:51 , 38:51 ", 51) == [(0, 51), (38, 51)]


@pytest.mark.parametrize("spec,why", [
    ("", "empty"),
    ("abc", "no colon"),
    ("0:60", "past the end"),
    ("5:3", "reversed"),
    ("-1:10", "negative start"),
    ("38:40", "too few rows for a chord plus two directions"),
    ("0:51,0:51", "duplicate"),
])
def test_parse_segments_rejects(spec, why):
    """A malformed spec has to fail at argument-parse time. The probe resolves
    it before the model loads, so the alternative is discovering it after a
    GPU hour."""
    with pytest.raises(ValueError):
        TM.parse_segments(spec, 51)


def test_segment_tag_is_stable_and_sorts_by_position():
    assert TM.segment_tag(0, 51) == "000_051"
    assert TM.segment_tag(38, 51) == "038_051"
    tags = [TM.segment_tag(a, b) for a, b in TM.parse_segments("25:51,0:16,0:51", 51)]
    assert sorted(tags) == ["000_016", "000_051", "025_051"]


def test_a_segment_frame_uses_only_that_segment():
    """Each stored frame must be exactly `plane_frame` of the sliced rows, with
    the segment's OWN chord — nothing about the surrounding path may leak in."""
    Z = bent_3d_path(50)
    for a, b in TM.parse_segments("0:51,25:51,38:51", 51):
        frame = TM.plane_frame(Z[a:b])
        assert frame is not None and frame.shape == (3, D)
        u, _ = TM.chord(Z[a:b])            # chord of the SEGMENT
        assert abs(abs(float(frame[0] @ u)) - 1.0) < 1e-12
        whole_u, _ = TM.chord(Z)
        if (a, b) != (0, 51):              # ...which is not the whole path's
            assert abs(float(frame[0] @ whole_u)) < 0.9999


def test_segments_of_a_three_direction_path_share_a_line_not_a_plane():
    """`bent_3d_path` spans chord + 2 directions, so every segment plane lies
    in that same 3-space and the two planes must intersect in a line
    (first angle 0). They are NOT identical, because a segment's chord is not
    the whole path's chord and the plane is defined orthogonally to it."""
    Z = bent_3d_path(50)
    whole = TM.plane_frame(Z)
    for a, b in ((25, 51), (38, 51), (0, 16)):
        theta1, theta2 = TM.principal_angles_deg(whole[1:], TM.plane_frame(Z[a:b])[1:])
        assert theta1 < 1e-4          # a shared direction: same 3-space
        assert theta2 > 5.0           # but a different plane: different chord


def test_a_late_only_bend_makes_the_tail_plane_differ():
    """Real trajectories are not confined to three directions — on the archived
    ones the unrelated-pair angle moves from 85.6 to 89.65 degrees once the
    path is truncated, i.e. the segment planes genuinely differ. A fourth
    direction that BENDS only late reproduces that; a late linear ramp would
    not, since it only moves the chord."""
    basis = np.linalg.qr(np.random.default_rng(31).standard_normal((D, 5)))[0]
    e0, e1, e2, e3 = (basis[:, i] for i in range(4))
    t = np.linspace(0.0, 1.0, 51)
    tail = np.where(t > 0.7, np.sin(np.pi * np.clip(t - 0.7, 0.0, None) / 0.3), 0.0)
    Z = (np.outer(6.0 * t, e0) + np.outer(np.sin(np.pi * t), e1)
         + np.outer(0.4 * np.sin(2 * np.pi * t), e2) + np.outer(2.0 * tail, e3))
    theta1, _ = TM.principal_angles_deg(TM.plane_frame(Z)[1:], TM.plane_frame(Z[38:])[1:])
    assert theta1 > 5.0
