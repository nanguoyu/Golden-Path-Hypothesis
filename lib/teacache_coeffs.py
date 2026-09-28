"""TeaCache polynomial-rescaling coefficients per backbone.

TeaCache's rescaling function $f(\\cdot)$ (paper Eq. 6) is a per-backbone
degree-4 polynomial fit on ~70 calibration prompts. Coefficients here are
extracted verbatim from the upstream code at
`reference/teacache/code/TeaCache4<Backbone>/`.

All entries follow numpy `poly1d` order (highest-degree coefficient first):

    f(x) = c[0] * x^4 + c[1] * x^3 + c[2] * x^2 + c[3] * x + c[4]

Usage:

    from lib.teacache_coeffs import get_coeffs
    import numpy as np
    coeffs = get_coeffs("flux")
    rescale = np.poly1d(coeffs)
    accumulated += rescale(rel_l1_value)

If a backbone has multiple sizes/variants (e.g. Wan2.1 has 1.3B and 14B), the
variant must be specified explicitly via `variant=...`. Add new entries by
copying the `coefficients = [...]` line from the corresponding upstream
`TeaCache4<Backbone>/*.py` file.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# (backbone_lowercase, variant_optional) -> coefficients (highest-degree first).
_COEFFS: dict[Tuple[str, Optional[str]], List[float]] = {
    # ---- FLUX family ----
    ("flux", None): [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    ("tangoflux", None): [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],

    # ---- Video DiTs ----
    ("wan21", "1.3b"): [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
    ("wan21", "14b"): [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
    ("hunyuan_video", None): [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    ("cogvideox", None): None,  # placeholder; upstream uses model.__class__.coefficients dynamic
    ("ltxvideo", None): [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
    ("mochi", None): [-3.51241319e+03, 8.11675948e+02, -6.09400215e+01, 2.42429681e+00, 3.05291719e-03],
    ("cosmos", None): [2.71156237e+02, -9.19775607e+01, 2.24437250e+00, 2.08355751e+00, 1.41776330e-01],
    ("consisid", None): [-1.53880483e+03, 8.43202495e+02, -1.34363087e+02, 7.97131516e+00, -5.23162339e-02],

    # ---- Image DiTs / next-gen ----
    ("hidream_i1", None): [-3.13605009e+04, -7.12425503e+02, 4.91363285e+01, 8.26515490e+00, 1.08053901e-01],
    ("lumina", None): [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
    ("lumina2", None): [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
}


def get_coeffs(backbone: str, variant: Optional[str] = None) -> List[float]:
    """Look up TeaCache polynomial coefficients by backbone (and optional variant).

    Args:
        backbone: lowercase backbone name (e.g. "flux" or "wan21").
        variant:  required iff the backbone has multiple sizes registered
                  (e.g. wan21 needs "1.3b" or "14b").

    Returns:
        Coefficients in numpy `poly1d` order (highest-degree first).

    Raises:
        KeyError if the (backbone, variant) combination is unknown OR if a
        variant is needed but not supplied.
    """
    key = (backbone.lower(), variant.lower() if variant else None)
    if key in _COEFFS and _COEFFS[key] is not None:
        return list(_COEFFS[key])  # defensive copy

    # If the user did not supply a variant but multiple are registered, error
    # with a helpful message.
    variants = sorted(
        {v for (b, v) in _COEFFS.keys() if b == backbone.lower() and v is not None}
    )
    if variants and variant is None:
        raise KeyError(
            f"TeaCache backbone '{backbone}' has multiple variants registered: "
            f"{variants}. Pass `variant=...`."
        )

    available = sorted({b for (b, _v) in _COEFFS.keys()})
    raise KeyError(
        f"No TeaCache coefficients registered for ({backbone!r}, {variant!r}). "
        f"Available backbones: {available}. "
        "Add entries to lib/teacache_coeffs.py by copying from "
        "reference/teacache/code/TeaCache4<Backbone>/."
    )


def list_backbones() -> List[Tuple[str, Optional[str]]]:
    """Return the (backbone, variant) keys for which coefficients are known."""
    return sorted(k for k, v in _COEFFS.items() if v is not None)
