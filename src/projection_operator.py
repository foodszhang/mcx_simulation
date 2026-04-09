from __future__ import annotations

import numpy as np
from skimage.transform import resize

from src.gen_mul_projection import generate_projection_view_matrix


def resize_volume_for_projection(volume_data: np.ndarray, config: dict) -> np.ndarray:
    proj_params = config.get("projection", {})
    if "resize_shape" in proj_params:
        resize_shape = tuple(proj_params["resize_shape"])
        return resize(volume_data, resize_shape, mode="constant", preserve_range=True).astype(
            np.float32
        )
    return volume_data.astype(np.float32)


def project_volume_with_angles(
    volume_data: np.ndarray,
    angles: list[int] | tuple[int, ...] | None,
    config: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    proj_params = config.get("projection", {})
    if angles is None:
        angles = proj_params.get("angles", [-90, -60, -30, 0, 30, 60, 90])

    resized = resize_volume_for_projection(volume_data, config)
    detector_resolution = tuple(proj_params.get("detector_resolution", (256, 256)))
    camera_distance = float(proj_params.get("camera_distance", 256))
    detector_size = tuple(proj_params.get("detector_size", detector_resolution))

    flux_proj: dict[str, np.ndarray] = {}
    depth_proj: dict[str, np.ndarray] = {}
    for angle in angles:
        proj, depth = generate_projection_view_matrix(
            resized,
            int(angle),
            camera_distance,
            detector_size,
            detector_resolution,
        )
        flux_proj[str(angle)] = proj.astype(np.float32)
        depth_proj[str(angle)] = depth.astype(np.float32)
    return flux_proj, depth_proj

