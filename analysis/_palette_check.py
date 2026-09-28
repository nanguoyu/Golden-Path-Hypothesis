#!/usr/bin/env python3
"""Colour separation checks shared by the paper-figure renderers.

Every figure in the paper states, in a comment beside its palette, how far
apart its colours are.  This module measures those claims so a future edit
that breaks one fails the render instead of reaching the page.

Three readings are taken for each pair of colours:

  normal    CIELAB distance as printed
  protan    the same distance after simulating protanopia
  deutan    the same distance after simulating deuteranopia

and one reading for each colour on its own: CIELAB lightness, which is what a
greyscale print keeps.  The dichromat simulation is the Vienot, Brettel and
Mollon (1999) reduction in linear RGB, the standard one behind the usual
colour-blindness tools.

Run it on its own to print every palette group the renderers register:

    python analysis/_palette_check.py
"""

from __future__ import annotations

import math

__all__ = ["to_lab", "lightness", "chroma", "simulate", "distance",
           "check_group", "check_cross"]

# Linear sRGB to LMS and back (Vienot, Brettel and Mollon 1999).
_RGB_TO_LMS = ((17.8824, 43.5161, 4.11935),
               (3.45565, 27.1554, 3.86714),
               (0.0299566, 0.184309, 1.46709))
_LMS_TO_RGB = ((0.080944, -0.130504, 0.116721),
               (-0.010248, 0.054019, -0.113615),
               (-0.000365, -0.004120, 0.693513))


def _to_linear(hex_color: str) -> tuple[float, float, float]:
    if hex_color.startswith("#"):
        parts = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    else:                       # matplotlib grey string, e.g. "0.46"
        parts = [float(hex_color)] * 3
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                 for c in parts)


def _from_linear(rgb: tuple[float, float, float]) -> str:
    out = []
    for c in rgb:
        c = min(1.0, max(0.0, c))
        s = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out.append(round(255 * s))
    return "#{:02X}{:02X}{:02X}".format(*out)


def _matmul(matrix, vector):
    return tuple(sum(m * v for m, v in zip(row, vector)) for row in matrix)


def to_lab(color: str) -> tuple[float, float, float]:
    """sRGB to CIELAB under D65."""
    r, g, b = _to_linear(color)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lightness(color: str) -> float:
    """CIELAB L*, the reading a greyscale print keeps."""
    return to_lab(color)[0]


def chroma(color: str) -> float:
    _light, a, b = to_lab(color)
    return math.hypot(a, b)


def simulate(color: str, kind: str) -> str:
    """The colour as a protanope or a deuteranope sees it."""
    if kind == "normal":
        return color if color.startswith("#") else _from_linear(_to_linear(color))
    long, medium, short = _matmul(_RGB_TO_LMS, _to_linear(color))
    if kind == "protan":
        long = 2.02344 * medium - 2.52581 * short
    elif kind == "deutan":
        medium = 0.494207 * long + 1.24827 * short
    else:
        raise ValueError(f"unknown vision type {kind!r}")
    return _from_linear(_matmul(_LMS_TO_RGB, (long, medium, short)))


def distance(one: str, other: str, kind: str = "normal") -> float:
    """CIELAB distance between two colours, optionally under a dichromacy."""
    return math.dist(to_lab(simulate(one, kind)), to_lab(simulate(other, kind)))


VISION = ("normal", "protan", "deutan")


def _pairs(colors: dict[str, str]):
    names = list(colors)
    for i, one in enumerate(names):
        for other in names[i + 1:]:
            yield one, other


def check_group(figure: str, group: str, colors: dict[str, str], *,
                min_distance: float = 25.0,
                min_lightness_gap: float | None = None,
                min_chroma: float | None = None,
                vision: tuple[str, ...] = VISION,
                verbose: bool = True) -> None:
    """Assert that one figure's palette group stays separable.

    `min_distance` is enforced for every pair under normal, protanope and
    deuteranope vision.  `min_lightness_gap` additionally asks the pair to
    survive a greyscale print.  `min_chroma` keeps a colour from reading as
    the grey the paper reserves for random controls.
    """
    if verbose:
        print(f"== palette: {figure} / {group} ==")
    for name, color in colors.items():
        light = lightness(color)
        if verbose:
            print(f"  {name:22s} {color:8s} L* {light:5.1f}  "
                  f"chroma {chroma(color):5.1f}")
        if min_chroma is not None:
            assert chroma(color) >= min_chroma, (figure, group, name, color)
    for one, other in _pairs(colors):
        reading = {k: distance(colors[one], colors[other], k) for k in VISION}
        gap = abs(lightness(colors[one]) - lightness(colors[other]))
        if verbose:
            print(f"  {one:22s} vs {other:22s} "
                  + "  ".join(f"{k} {reading[k]:5.1f}" for k in VISION)
                  + f"  |dL*| {gap:5.1f}")
        for kind in vision:
            assert reading[kind] >= min_distance, (
                figure, group, one, other, kind,
                round(reading[kind], 1), min_distance)
        if min_lightness_gap is not None:
            assert gap >= min_lightness_gap, (figure, group, one, other,
                                              round(gap, 1), min_lightness_gap)


def check_cross(figure: str, group: str, colors: dict[str, str],
                other_group: str, others: dict[str, str], *,
                min_distance: float = 25.0,
                min_cvd_distance: float | None = None,
                verbose: bool = True) -> None:
    """Assert one group keeps its distance from another, e.g. the method hues.

    `min_distance` is the limit under normal vision.  `min_cvd_distance` is the
    limit under both dichromacies; leave it out to hold them to the same limit.
    """
    limit = {"normal": min_distance,
             "protan": min_distance if min_cvd_distance is None else min_cvd_distance,
             "deutan": min_distance if min_cvd_distance is None else min_cvd_distance}
    worst = {kind: None for kind in VISION}
    for name, color in colors.items():
        for other_name, other in others.items():
            for kind in VISION:
                value = distance(color, other, kind)
                if worst[kind] is None or value < worst[kind][0]:
                    worst[kind] = (value, name, other_name)
                assert value >= limit[kind], (figure, group, name,
                                              other_group, other_name, kind,
                                              round(value, 1), limit[kind])
    if verbose:
        for kind in VISION:
            value, name, other_name = worst[kind]
            print(f"  {group} vs {other_group} ({kind}): closest pair "
                  f"{name} / {other_name} at {value:.1f} "
                  f"(limit {limit[kind]:.0f})")


if __name__ == "__main__":
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for module in ("analysis.fig1_main",
                   "analysis.fig2_search_cost",
                   "analysis.fig3_schedule_strips",
                   "analysis.render_paper_cached_figures",
                   "analysis.render_paper_spx_figure",
                   "analysis.render_paper_plane_figure",
                   "analysis.render_paper_action_state_figure",
                   "analysis.render_paper_clock_figure",
                   "analysis.render_paper_exhaustive_existence",
                   "analysis.render_paper_k41_transfer"):
        importlib.import_module(module).check_palette()
