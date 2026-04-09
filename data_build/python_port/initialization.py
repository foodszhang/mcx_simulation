from __future__ import annotations

import numpy as np


def initialization() -> tuple[np.ndarray, np.ndarray, float]:
    """Port of MATLAB ``Initialization.m``.

    Returns:
        uam: absorption per region label (1..7)
        dm: diffusion per region label (1..7)
        an: boundary Fresnel coefficient scalar
    """

    ua2 = np.array([0.08697, 0.05881, 0.01304, 0.35182, 0.06597, 0.19639], dtype=np.float64)
    us2 = np.array([4.29071, 6.42581, 17.9615, 6.78066, 16.09293, 36.52133], dtype=np.float64)
    g = np.array([0.9, 0.85, 0.92, 0.9, 0.85, 0.94], dtype=np.float64)
    n = 1.37
    dm2 = 1.0 / (3.0 * (ua2 + (1.0 - g) * us2))
    r = -1.4399 * (n**-2) + 0.7099 * (n**-1) + 0.6681 + 0.0636 * n
    an = (1.0 + r) / (1.0 - r)

    # MATLAB maps region 7 to same optical params as region 4
    uam = np.array([ua2[0], ua2[1], ua2[2], ua2[3], ua2[4], ua2[5], ua2[3]], dtype=np.float64)
    dm = np.array([dm2[0], dm2[1], dm2[2], dm2[3], dm2[4], dm2[5], dm2[3]], dtype=np.float64)
    return uam, dm, float(an)

