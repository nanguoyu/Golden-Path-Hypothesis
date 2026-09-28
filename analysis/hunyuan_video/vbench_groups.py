"""Frozen C4 partition of the 16 official VBench dimensions."""

DIMENSION_GROUPS = {
    "consistency": (
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "overall_consistency",
    ),
    "motion": (
        "motion_smoothness",
        "dynamic_degree",
        "human_action",
        "temporal_style",
    ),
    "quality": (
        "aesthetic_quality",
        "imaging_quality",
        "appearance_style",
        "scene",
    ),
    "semantics": (
        "object_class",
        "multiple_objects",
        "color",
        "spatial_relationship",
    ),
}
