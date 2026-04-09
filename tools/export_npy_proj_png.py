#!/usr/bin/env python3
"""Batch export .npy projections to .png.

This is based on the visualization style in tools/show_proj.py (matplotlib imshow + optional
colorbar and stats overlay), but runs headless and saves pngs instead of showing windows.

Examples:
  uv run python tools/export_npy_proj_png.py --input ./some_dir --out-dir ./pngs --recursive
  uv run python tools/export_npy_proj_png.py --input ./a.npy --out-dir ./pngs --cmap viridis

Notes:
  - 2D array: saved as one png.
  - >=3D array: flattened into multiple 2D frames using the last 2 dims as (H, W).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _iter_array_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    exts = {".npy", ".npz"}

    if input_path.is_file():
        if input_path.suffix.lower() not in exts:
            raise ValueError(f"Input file must be one of {sorted(exts)}: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if recursive:
        files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    yield from sorted(files)

def _flatten_to_frames(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (frames, n_frames) where frames has shape (N, H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, :, :], 1
    if arr.ndim < 2:
        raise ValueError(f"Array must be at least 2D to render as image, got shape {arr.shape}")

    h, w = arr.shape[-2], arr.shape[-1]
    n = int(np.prod(arr.shape[:-2]))
    frames = arr.reshape((n, h, w))
    return frames, n


def _safe_percentile(a: np.ndarray, p: float) -> float:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, p))


def save_heatmap_png(
    array2d: np.ndarray,
    out_path: Path,
    *,
    cmap: str,
    dpi: int,
    colorbar: bool,
    stats: bool,
    title: bool,
    log1p: bool,
    clip_pct: Tuple[float, float] | None,
    rotate_right_90: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(array2d, dtype=np.float32)
    if log1p:
        img = np.log1p(np.maximum(img, 0))

    if rotate_right_90:
        img = np.rot90(img, k=-1)

    vmin = vmax = None
    if clip_pct is not None:
        lo, hi = clip_pct
        vmin = _safe_percentile(img, lo)
        vmax = _safe_percentile(img, hi)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

    # Default: only image (no axes/title/colorbar/stats)
    if not (colorbar or stats or title):
        plt.imsave(out_path, img, cmap=cmap, vmin=vmin, vmax=vmax)
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(out_path.stem)

    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if stats:
        finite = img[np.isfinite(img)]
        if finite.size:
            stats_text = f"Min: {finite.min():.2f}\nMax: {finite.max():.2f}\nMean: {finite.mean():.2f}"
        else:
            stats_text = "No finite values"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch export .npy/.npz arrays to heatmap .png")
    ap.add_argument("--input", required=True, help="Input .npy/.npz file or directory")
    ap.add_argument("--out-dir", required=True, help="Output directory to write pngs")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find .npy/.npz in input dir",
    )
    ap.add_argument("--cmap", default="hot", help="Matplotlib colormap name")
    ap.add_argument("--dpi", type=int, default=300, help="PNG dpi (used when drawing decorations)")
    ap.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw colorbar",
    )
    ap.add_argument(
        "--title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw title",
    )
    ap.add_argument("--stats", action="store_true", help="Overlay min/max/mean text")
    ap.add_argument("--log1p", action="store_true", help="Apply log1p(max(x,0)) before render")
    ap.add_argument(
        "--clip-pct",
        default=None,
        help="Clip visualization range by percentiles, e.g. '1,99'",
    )
    ap.add_argument(
        "--rotate-right-90",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate image 90° clockwise before saving",
    )
    ap.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays (unsafe; only if you trust the file)",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    clip_pct = None
    if args.clip_pct is not None:
        parts = [p.strip() for p in args.clip_pct.split(",")]
        if len(parts) != 2:
            raise ValueError("--clip-pct must be like '1,99'")
        clip_pct = (float(parts[0]), float(parts[1]))

    files = list(_iter_array_files(input_path, args.recursive))
    if not files:
        print(f"No .npy/.npz files found under: {input_path}")
        return 1

    for in_path in files:
        if input_path.is_dir():
            rel = in_path.relative_to(input_path)
            out_base = (out_dir / rel).with_suffix("")
        else:
            out_base = out_dir / in_path.stem

        if in_path.suffix.lower() == ".npz":
            data = np.load(in_path, allow_pickle=bool(args.allow_pickle))
            for key in data.files:
                arr = data[key]
                if input_path.is_dir():
                    # avoid key collisions across multiple npz files
                    key_base = out_base / str(key)
                else:
                    # single npz: output name is exactly the key
                    key_base = out_dir / str(key)
                frames, n_frames = _flatten_to_frames(arr)
                if n_frames == 1:
                    save_heatmap_png(
                        frames[0],
                        key_base.with_suffix(".png"),
                        cmap=args.cmap,
                        dpi=args.dpi,
                        colorbar=bool(args.colorbar),
                        stats=args.stats,
                        title=bool(args.title),
                        log1p=args.log1p,
                        clip_pct=clip_pct,
                        rotate_right_90=bool(args.rotate_right_90),
                    )
                else:
                    for i in range(n_frames):
                        save_heatmap_png(
                            frames[i],
                            Path(str(key_base) + f"_{i:04d}.png"),
                            cmap=args.cmap,
                            dpi=args.dpi,
                            colorbar=bool(args.colorbar),
                            stats=args.stats,
                            title=bool(args.title),
                            log1p=args.log1p,
                            clip_pct=clip_pct,
                            rotate_right_90=bool(args.rotate_right_90),
                        )
        else:
            arr = np.load(in_path, allow_pickle=bool(args.allow_pickle))
            frames, n_frames = _flatten_to_frames(arr)
            if n_frames == 1:
                save_heatmap_png(
                    frames[0],
                    out_base.with_suffix(".png"),
                    cmap=args.cmap,
                    dpi=args.dpi,
                    colorbar=bool(args.colorbar),
                    stats=args.stats,
                    title=bool(args.title),
                    log1p=args.log1p,
                    clip_pct=clip_pct,
                    rotate_right_90=bool(args.rotate_right_90),
                )
            else:
                for i in range(n_frames):
                    save_heatmap_png(
                        frames[i],
                        Path(str(out_base) + f"_{i:04d}.png"),
                        cmap=args.cmap,
                        dpi=args.dpi,
                        colorbar=bool(args.colorbar),
                        stats=args.stats,
                        title=bool(args.title),
                        log1p=args.log1p,
                        clip_pct=clip_pct,
                        rotate_right_90=bool(args.rotate_right_90),
                    )

    print(f"Wrote pngs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
