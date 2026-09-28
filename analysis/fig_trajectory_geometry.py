#!/usr/bin/env python3
"""Measured trajectory examples, PCA summaries, and high-dimensional angles.

The examples are the first three DrawBench prompts in the formal FLUX
full-compute trajectory measurement, with base seed 41 and actual seeds
41, 42, and 43. Each example uses its own centered off-chord PCA frame.
Each measured two-dimensional curve is embedded in its own orthonormal
display plane. All examples use the same uniform scale. Plane rotations
and translations provide a readable layout, not measured relative angles.
Every state has a marker. States 0-40 are solid, and the remaining return
to state 50 is dashed. The shared position of states 0 and 50 has an open ring.

The stored projection diagnostics also evaluate one joint top-three-PC
projection. That projection loses almost all of each second bending direction,
so it is not used to illustrate the shapes. Plane-angle statistics come from
the separate formal high-dimensional frame analysis, not the drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.palette import MODEL, OLIVE_RAMP, RANDOM_GREY

DATA = ROOT / "resources/full_trajectory_analysis/trajectory_geometry_examples.json"
IMAGE_SPECTRUM = ROOT / "resources/full_trajectory_analysis/evr_spectrum_image.json"
ANGLES = ROOT / "resources/full_trajectory_shape/shape_scale.json"
OUT = ROOT / "paper/figs/trajectory_geometry"
MODELS = ("flux", "qwen", "hunyuan_video", "wan21")
FONT = 7.2


def measure_examples() -> dict:
    import torch

    record_path = ROOT / "resources/full_trajectory/tables_jsonl/full_traj_flux_drawbench_full.jsonl"
    with record_path.open() as stream:
        records = [json.loads(line) for line in stream]
    centered, planes, examples = [], [], []
    for index in range(3):
        record = next(row for row in records
                      if row["prompt_idx"] == index and row["seed"] == 41)
        assert record["prompt_seed"] == 41 + index
        path = ROOT / f"resources/full_trajectory/latents_flux/latents_{index:05d}_s41.pt"
        states = torch.load(path, map_location="cpu", weights_only=True).double().numpy()
        assert states.shape == (51, record["d"])
        chord = states[-1] - states[0]
        length = np.linalg.norm(chord)
        unit = chord / length
        scaled = (states - states[0]) / length
        off = scaled - np.outer(scaled @ unit, unit)
        y = off - off.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(y @ y.T)
        order = np.argsort(eigenvalues)[::-1]
        values = np.maximum(eigenvalues[order], 0)
        directions = y.T @ eigenvectors[:, order[:2]] / np.sqrt(values[:2])
        coords = off @ directions
        # Resolve only PCA sign ambiguity, identically for every trajectory.
        # No rotation or translation between individual frames is introduced.
        for column in range(2):
            sign = 1 if coords[np.argmax(np.abs(coords[:, column])), column] >= 0 else -1
            directions[:, column] *= sign
            coords[:, column] *= sign
        fractions = values / values.sum()
        np.testing.assert_allclose(fractions[:5], record["pca_evr"], atol=2e-10)
        np.testing.assert_allclose(coords[[0, -1]], 0, atol=1e-12)
        centered.append(y)
        planes.append(directions)
        examples.append({
            "prompt_idx": index,
            "prompt_seed": record["prompt_seed"],
            "prompt_sha256": record["prompt_sha256"],
            "state_file": str(path.relative_to(ROOT)),
            "record_file": str(record_path.relative_to(ROOT)),
            "base_seed": 41,
            "states": list(range(51)),
            "own_plane_coordinates_per_chord_length": coords.tolist(),
            "pc1_fraction": float(fractions[0]),
            "pc2_fraction": float(fractions[1]),
            "pc1_plus_pc2_fraction": float(fractions[:2].sum()),
        })

    joint = np.concatenate(centered)
    vals, vecs = np.linalg.eigh(joint @ joint.T)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0)
    basis = joint.T @ vecs[:, order[:3]] / np.sqrt(vals[:3])
    for example, y, plane in zip(examples, centered, planes):
        example["joint_top3_variance_fraction"] = float(np.sum((y @ basis) ** 2) / np.sum(y ** 2))
        example["joint_top3_plane_direction_retention"] = np.linalg.svd(plane.T @ basis, compute_uv=False).tolist()
    return {
        "model": "flux", "dataset": "drawbench_full", "dimension": 262144,
        "selection": "First three prompt indices at base seed 41, chosen before projection.",
        "plotted_frame": "Each trajectory's own two centered off-chord principal components.",
        "plotted_states": [0, 50],
        "normalization": "Off-chord vectors divided by each trajectory's initial-to-final chord length.",
        "joint_top3_total_variance_fraction": float(vals[:3].sum() / vals.sum()),
        "examples": examples,
    }


def load_spectra() -> dict:
    data = json.loads(IMAGE_SPECTRUM.read_text())
    for model in MODELS[2:]:
        source = ROOT / f"resources/video_full_trajectory/{model}/step_profiles_{model}.json"
        source_data = json.loads(source.read_text())["evr_spectrum"]["by_dataset"]
        data[model] = {key: {**value, "top2_cum_median": value["top2_cumulative_median"]}
                       for key, value in source_data.items()}
    return data


def display_plane(u, v) -> np.ndarray:
    """An orthonormal basis for an illustrative plane, not a latent frame."""
    u = np.asarray(u, dtype=float)
    u /= np.linalg.norm(u)
    v = np.asarray(v, dtype=float)
    v -= np.dot(u, v) * u
    v /= np.linalg.norm(v)
    basis = np.column_stack((u, v))
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    return basis


def draw_examples(fig, data: dict) -> None:
    fig.text(.012, .97, "(a) Measured curves",
             ha="left", va="top", fontsize=FONT + .3)
    ax = fig.add_axes((.005, .025, .355, .910), projection="3d", computed_zorder=False)
    layouts = [
        ((-1.02, -.18, .20), (1.0, .10, .08), (-.08, .90, .38)),
        ((.02, .24, -.22), (.72, .58, .18), (-.20, .12, .97)),
        ((1.24, -.20, .28), (.73, .03, .68), (-.36, .84, .36)),
    ]
    # A single uniform enlargement preserves relative shape and scale in all
    # planes. The shift below only centers the shared local plotting window.
    scale = 8.0
    local_center = np.array([.060, .007])
    plane_corners = np.array([[-.010, -.031], [.131, -.031],
                              [.131, .046], [-.010, .046]])
    colors = [to_rgb(OLIVE_RAMP[0]), to_rgb(OLIVE_RAMP[1]),
              tuple(.62*np.array(to_rgb(OLIVE_RAMP[2]))
                    + .38*np.array(to_rgb(OLIVE_RAMP[0])))]
    for i, (example, layout, color) in enumerate(zip(data["examples"], layouts, colors)):
        coords = np.asarray(example["own_plane_coordinates_per_chord_length"])
        assert example["states"] == list(range(51))
        np.testing.assert_allclose(coords[[0, 50]], 0, atol=1e-12)
        center, u, v = layout
        center = np.asarray(center)
        basis = display_plane(u, v)
        curve = center + scale * (coords - local_center) @ basis.T
        corners = center + scale * (plane_corners - local_center) @ basis.T
        # Every pairwise distance must be preserved up to the one scale above.
        original_distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        plotted_distances = np.linalg.norm(curve[:, None] - curve[None, :], axis=2)
        np.testing.assert_allclose(plotted_distances, scale*original_distances,
                                   rtol=1e-12, atol=1e-14)
        ax.add_collection3d(Poly3DCollection([corners], facecolor=color,
            edgecolor=color, alpha=.16, linewidth=.65, zorder=1+i*.1))
        # State 40 is present in both segments, so the 40-to-41 edge remains.
        ax.plot(*curve[:41].T, color=color, lw=1.0, zorder=5+i*.1)
        ax.plot(*curve[40:].T, color=color, lw=.65, ls=(0, (3, 2)),
                alpha=.45, zorder=4+i*.1)
        ax.scatter(*curve[:41].T, s=3.4, color=[color], linewidths=0,
                   depthshade=False, zorder=7)
        ax.scatter(*curve[41:].T, s=3.4, color=[color], linewidths=0,
                   alpha=.45, depthshade=False, zorder=6)
        ax.scatter(*curve[0], s=14, facecolors="white", edgecolors=[color],
                   linewidths=.7, depthshade=False, zorder=7)
        label = center + .43*basis[:, 1]
        if i == 1:
            label = center - .70*basis[:, 1]
        if i == 2:
            label += np.array([.06, -.03, .31])
        ax.text(*label, str(i+1), color=".12", ha="center", va="bottom",
                fontsize=FONT, zorder=10)
    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-.75, .75)
    ax.set_zlim(-.82, .82)
    # Matching the display box to its coordinate limits avoids axis stretching.
    ax.set_box_aspect((3.44, 1.50, 1.64), zoom=1.08)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-65)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(".85")
        axis.line.set_color(".78")
        axis.line.set_linewidth(.45)


def draw_spectra(fig, data: dict) -> None:
    fig.text(.363, .97, "(b) Explained variance (%)", ha="left", va="top",
             fontsize=FONT + .3)
    ax = fig.add_axes((.363, .195, .255, .605))
    ax.axis("off")
    columns = [0.00, .59, 1.00]
    for x, text in zip(columns, ["Model", "PC1", "PC1+PC2"]):
        ax.text(x, 1.0, text, transform=ax.transAxes,
                ha="left" if x == 0 else "right", va="center", fontsize=FONT)
    ax.plot([0, 1], [.88, .88], color=".2", lw=.6, transform=ax.transAxes)
    for row, model in enumerate(MODELS):
        ds = list(data[model].values())
        a = [100*v["evr_median"][0] for v in ds]
        b = [100*v["top2_cum_median"] for v in ds]
        label = {"flux": "FLUX", "qwen": "Qwen", "hunyuan_video": "HYV", "wan21": "Wan"}[model]
        entries = [label, f"{min(a):.1f}-{max(a):.1f}", f"{min(b):.1f}-{max(b):.1f}"]
        for x, text in zip(columns, entries):
            ax.text(x, .71 - row*.22, text, transform=ax.transAxes,
                    ha="left" if x == 0 else "right", va="center", fontsize=FONT)
    ax.plot([0, 1], [-.08, -.08], color=".2", lw=.6, transform=ax.transAxes, clip_on=False)


def draw_angles(fig, data: dict) -> None:
    fig.text(.685, .97, "(c) Plane angles", ha="left", va="top",
             fontsize=FONT + .3)
    ax = fig.add_axes((.680, .225, .314, .485))
    categories = ["same_noise_diff_prompt", "same_prompt_diff_noise", "unrelated", "random_plane_null"]
    x = np.arange(4)
    for i, model in enumerate(MODELS[:2]):
        medians = [data[model][key]["theta1_med"] for key in categories]
        ax.bar(x[:3]+(i-.5)*.34, medians[:3], width=.34, color=MODEL[model], label={"flux": "FLUX", "qwen": "Qwen"}[model], zorder=3)
        ax.bar(x[3:]+(i-.5)*.34, medians[3:], width=.34, facecolor=RANDOM_GREY["fill"],
               edgecolor=MODEL[model], linewidth=.8, zorder=3)
    ax.axhline(90, color=".5", lw=.6, ls=":")
    ax.set_ylim(0, 100)
    ax.set_xlim(-.47, 3.53)
    ax.set_yticks([0, 45, 90])
    ax.set_ylabel("First principal angle (°)", labelpad=1.5)
    ax.set_xticks(x, ["Same\nnoise", "Same\nprompt", "Neither", "Random"])
    ax.tick_params(pad=1.5)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.34), ncols=2,
              frameon=False, fontsize=7.0, borderpad=0, handlelength=1.1, columnspacing=.8)
    ax.grid(axis="y", alpha=.2, lw=.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-examples", action="store_true")
    args = parser.parse_args()
    if args.recompute_examples or not DATA.exists():
        data = measure_examples()
        DATA.write_text(json.dumps(data, indent=2) + "\n")
    else:
        data = json.loads(DATA.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT,
        "axes.labelsize": FONT, "axes.titlesize": FONT,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(5.5, 1.55))
    draw_examples(fig, data)
    draw_spectra(fig, load_spectra())
    draw_angles(fig, json.loads(ANGLES.read_text())["planes"])
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"CreationDate": None, "Creator": "fig_trajectory_geometry.py"})
    fig.savefig(OUT.with_suffix(".png"), dpi=400)
    plt.close(fig)
    print(json.dumps({"examples": [{key: row[key] for key in
           ["prompt_idx", "prompt_seed", "pc1_plus_pc2_fraction", "joint_top3_variance_fraction"]}
          for row in data["examples"]], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
