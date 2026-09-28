#!/usr/bin/env python3
"""The single source of every colour the paper figures draw with.

A registry here is keyed by the ROLE the colour plays in the paper, never by
the figure that happens to draw it, so one meaning always wears one colour and
two figures cannot drift apart.  Reading a figure's palette therefore means
reading which roles it uses, not which literals it declares.

The roles, one line each:

METHOD        the four cache methods of Figure 1; a hue here means that method
              and nothing else, anywhere in the paper.
RANDOM_GREY   the random control; grey is reserved for it, so no coloured role
              may read as grey and no other role may be drawn in these tones.
DATASET       the prompt set a curve was measured on, four image and two video.
MODEL         the generative model a mark belongs to, four of them.
FAMILY        the family a fixed schedule was drawn from, in the drift figure.
COMPONENT     the three shares of one variance decomposition, read as an order.
OLIVE_RAMP    one hue at three lightnesses, for any ordered set of three.
SCHEMATIC     the two sides of the one drawn, unmeasured schematic.

`analysis/_palette_check.py` stays the measuring engine.  This module hands it
the groups to measure through the helpers at the bottom, and every renderer
still calls its own `check_palette()` before it draws.
"""

from __future__ import annotations

__all__ = ["METHOD", "METHOD_LABEL", "METHOD_NAMED", "RANDOM_GREY", "DATASET",
           "DATASET_LABEL", "DATASET_IMAGE", "DATASET_VIDEO", "MODEL",
           "MODEL_LABEL", "MODEL_NAMED", "FAMILY", "COMPONENT", "OLIVE_RAMP",
           "SCHEMATIC", "SCHEMATIC_LABEL", "SCHEMATIC_NAMED",
           "named", "image_datasets", "video_datasets", "all_datasets"]


# The four cache methods, from the Okabe-Ito colour-vision-safe set.  A figure
# that draws no cache method must still keep its distance from these four,
# because a reader carries them from Figure 1 to every later page.
METHOD = {
    "seacache": "#0072B2",
    "teacache": "#E69F00",
    "sencache": "#009E73",
    "dicache": "#D55E00",
}
METHOD_LABEL = {"seacache": "SeaCache", "teacache": "TeaCache",
                "sencache": "SenCache", "dicache": "DiCache"}

# The random control.  Every random draw in the paper is grey, which is why no
# coloured role may fall below a chroma of 20 and read as one of these.
RANDOM_GREY = {
    "mark": "0.46",     # the random control's own mark
    "dot": "0.45",      # a median dot on a random span, or a random bar's edge
    "span": "0.62",     # the 5-95 percent span a random median sits on
    "fill": "0.86",     # a filled random bar, drawn with `dot` as its edge
}

# The prompt set a measurement was taken on.  Four image sets on a lightness
# ladder and two video sets on a second one; no dataset is grey, and none is a
# method hue.
#
# GenEval's rose sits at L* 57.1, the midpoint of the gap between DrawBench at
# 69.8 and PartiPrompts at 44.8.  It cannot be darkened to L* 45 as the ladder
# stands, because PartiPrompts already holds that level and the two would then
# print as one grey.  The window below PartiPrompts is closed as well: a rose
# at L* 33 lands on Penguin599's maroon, and the only hues that clear it there
# are purples that collide with DiffusionDB's indigo.  At L* 57.1 every pair of
# image hues clears 10 L* in a greyscale print and 25 CIELAB under both
# dichromacies, which the previous #CC79A7 did not.
DATASET = {
    "drawbench_full": "#56B4E9",
    "geneval_style": "#BF728E",
    "parti_full": "#667000",
    "diffusiondb_clean10k": "#332288",
    "penguin599": "#882255",
    "vbench944": "#D6604D",
}
DATASET_LABEL = {
    "drawbench_full": "DrawBench",
    "geneval_style": "GenEval",
    "parti_full": "PartiPrompts",
    "diffusiondb_clean10k": "DiffusionDB",
    "penguin599": "Penguin599",
    "vbench944": "VBench944",
}
DATASET_IMAGE = ("drawbench_full", "geneval_style", "parti_full",
                 "diffusiondb_clean10k")
DATASET_VIDEO = ("penguin599", "vbench944")

# The generative model a mark belongs to.  The image pair is the indigo and
# teal the plane figure and the appendix separation figure both use; the video
# pair completes the set for the four-model figures.
MODEL = {
    "flux": "#332288",
    "qwen": "#44AA99",
    "hunyuan_video": "#882255",
    "wan21": "#999933",
}
MODEL_LABEL = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image",
               "hunyuan_video": "HunyuanVideo", "wan21": "Wan2.1"}

# The family a fixed schedule was drawn from.  These are not cache methods, so
# they stay clear of the method hues; the random family wears the reserved grey.
FAMILY = {
    "search": "#332288",
    "ladder": "#56B4E9",
    "other": "#999933",
    "random": RANDOM_GREY["mark"],
}

# The three shares of one variance decomposition.  They sit on one bar, so a
# greyscale print has to separate them by lightness alone.
COMPONENT = {
    "schedule": "#117733",
    "payload": "#DDCC77",
    "interaction": "#5C1141",
}

# One olive hue at three lightnesses, dark to light, for any ordered set of
# three: the stacked energy shares use it, and the step index of one drawn
# trajectory reads it as a continuous ramp.
OLIVE_RAMP = ("#3B4A1C", "#7E9445", "#C6D49B")

# The two sides of the action-state schematic.  Nothing on that figure is
# measured, so its colours must not read as any measured role: the full model
# takes a near-black ink at L* 11.8, which is 36 L* away from the darkest tone
# the random control uses, and the cached run takes a brown that clears every
# cache-method hue by 41 CIELAB under normal vision and by 22 under both
# dichromacies.
SCHEMATIC = {
    "full": "#1F1F1F",
    "cached": "#8A5A2B",
}
SCHEMATIC_LABEL = {"full": "full model", "cached": "cached run"}


# ------------------------------------------------------- groups to measure
def named(registry: dict[str, str], labels: dict[str, str]) -> dict[str, str]:
    """The same registry keyed by the name the reader sees on the page."""
    return {labels[key]: registry[key] for key in registry}


METHOD_NAMED = named(METHOD, METHOD_LABEL)
MODEL_NAMED = named(MODEL, MODEL_LABEL)
SCHEMATIC_NAMED = named(SCHEMATIC, SCHEMATIC_LABEL)


def image_datasets() -> dict[str, str]:
    """The four image prompt sets, the group that shares one panel."""
    return {DATASET_LABEL[k]: DATASET[k] for k in DATASET_IMAGE}


def video_datasets() -> dict[str, str]:
    """The two video prompt sets, the group that shares the other panel."""
    return {DATASET_LABEL[k]: DATASET[k] for k in DATASET_VIDEO}


def all_datasets() -> dict[str, str]:
    """All six, the group a reader compares across panels of one figure."""
    return {**image_datasets(), **video_datasets()}
