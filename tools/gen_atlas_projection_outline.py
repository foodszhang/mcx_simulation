# -*- coding: utf-8 -*-
"""Generate minimalist outline projection images from ct_data atlas.

Usage (recommended):
  uv run python tools/gen_atlas_projection_outline.py --canvas 1024 --angles -90,0,90

Notes:
- Input atlas is read the same way as tools/main_volume_and_material_config.py.
- Before projection, all voxels > 0 are unified to a constant (default 200) to keep only boundary.
- Projection is generated via src/gen_mul_projection.py:generate_projection_view_matrix.
- Output is "paper schematic" style: black background + white outline only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

from skimage.measure import find_contours
from skimage.transform import resize

# Ensure repo root is on sys.path when running via `uv run python tools/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.gen_mul_projection import generate_projection_view_matrix


def _parse_angles(s: str) -> list[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _crop_bbox(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(mask.shape))
    return mask[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]


def _downsample(volume: np.ndarray, target_max_dim: int = 256) -> tuple[np.ndarray, int]:
    max_dim = int(max(volume.shape))
    factor = max(1, int(np.ceil(max_dim / float(target_max_dim))))
    if factor == 1:
        return volume, factor
    return volume[::factor, ::factor, ::factor], factor


def _normalize_mask_to_square(mask2d: np.ndarray, canvas: int, margin: float = 0.10) -> np.ndarray:
    coords = np.argwhere(mask2d)
    if coords.size == 0:
        return np.zeros((canvas, canvas), dtype=bool)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h = int(y1 - y0)
    w = int(x1 - x0)

    side = int(np.ceil(max(h, w) * (1.0 + 2.0 * margin)))
    side = max(side, 2)

    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0

    sy0 = int(round(cy - side / 2.0))
    sx0 = int(round(cx - side / 2.0))
    sy1 = sy0 + side
    sx1 = sx0 + side

    pad_top = max(0, -sy0)
    pad_left = max(0, -sx0)
    pad_bottom = max(0, sy1 - mask2d.shape[0])
    pad_right = max(0, sx1 - mask2d.shape[1])

    padded = np.pad(
        mask2d,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=False,
    )

    sy0 += pad_top
    sy1 += pad_top
    sx0 += pad_left
    sx1 += pad_left

    square = padded[sy0:sy1, sx0:sx1]

    square_resized = resize(
        square.astype(np.float32),
        (canvas, canvas),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return square_resized > 0.5


def _render_outline_png(mask2d: np.ndarray, out_path: Path, line_width: float) -> None:
    # Import matplotlib lazily to keep startup fast.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contours = find_contours(mask2d.astype(np.uint8), 0.5)

    canvas = int(mask2d.shape[0])
    dpi = 100
    fig = plt.figure(figsize=(canvas / dpi, canvas / dpi), dpi=dpi, facecolor="#000000")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#000000")
    ax.set_axis_off()
    ax.set_xlim(0, canvas)
    ax.set_ylim(canvas, 0)

    # Solid fill (same color as outline): white silhouette on pure black.
    ax.imshow(
        mask2d.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(0, canvas, canvas, 0),
    )

    # Optional outline on top (also white), helps the boundary read crisply.
    for c in contours:
        ax.plot(
            c[:, 1],
            c[:, 0],
            color="#FFFFFF",
            linewidth=line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )

    fig.savefig(out_path, dpi=dpi, facecolor="#000000", edgecolor="none", pad_inches=0)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--atlas",
        default="./ct_data/atlas_380x992x208.hdr",
        help="Path to atlas .hdr",
    )
    ap.add_argument("--out-dir", default="./outputs/atlas_outline_projections")
    ap.add_argument("--canvas", type=int, default=1024, choices=[512, 1024])
    ap.add_argument("--angles", default="-90,0,90")
    ap.add_argument("--unify-value", type=int, default=200)
    ap.add_argument("--target-max-dim", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--line-width", type=float, default=3.0)
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(atlas_path))
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]

    mask3d = data > 0
    mask3d = _crop_bbox(mask3d, pad=1)
    vol = np.where(mask3d, int(args.unify_value), 0).astype(np.uint16)

    vol, factor = _downsample(vol, target_max_dim=int(args.target_max_dim))

    # camera_distance must be > max extent in rotated Z to keep depth positive.
    max_dim = float(max(vol.shape))
    camera_distance = max_dim * 2.0

    detector_resolution = (int(args.canvas), int(args.canvas))
    detector_size = (float(args.canvas), float(args.canvas))

    angles = _parse_angles(args.angles)
    if not angles:
        raise SystemExit("--angles is empty")

    for angle in angles:
        proj, _depth = generate_projection_view_matrix(
            vol,
            float(angle),
            float(camera_distance),
            np.array(detector_size, dtype=np.float32),
            np.array(detector_resolution, dtype=np.int32),
        )

        mask2d = proj > 0
        mask2d = _normalize_mask_to_square(mask2d, canvas=int(args.canvas), margin=float(args.margin))

        out_path = out_dir / f"atlas_outline_angle_{angle:g}_canvas_{args.canvas}_ds{factor}.png"
        _render_outline_png(mask2d, out_path, line_width=float(args.line_width))
        print(f"[OK] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
