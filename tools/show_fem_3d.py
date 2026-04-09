#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.fem_mesh import load_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="导出FEM网格与光通量三维图")
    parser.add_argument(
        "--mesh",
        required=True,
        help="mesh文件路径（.npz）",
    )
    parser.add_argument(
        "--fluence",
        required=True,
        help="节点光通量文件路径（.npy）",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出图片目录",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="颜色映射",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=25.0,
        help="视角仰角",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=35.0,
        help="视角方位角",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="节点散点大小",
    )
    parser.add_argument(
        "--surface-alpha",
        type=float,
        default=0.88,
        help="表面透明度",
    )
    parser.add_argument(
        "--wire-alpha",
        type=float,
        default=0.20,
        help="网格线透明度",
    )
    parser.add_argument(
        "--clip-pct",
        default="1,99",
        help="光通量着色范围百分位，如 1,99",
    )
    return parser.parse_args()


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _add_surface(
    ax,
    nodes: np.ndarray,
    faces: np.ndarray,
    *,
    face_colors=None,
    edge_color=(0.2, 0.2, 0.2, 0.15),
    alpha: float = 1.0,
):
    polys = nodes[faces]
    collection = Poly3DCollection(
        polys,
        facecolors=face_colors,
        edgecolors=edge_color,
        linewidths=0.15,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    return collection


def render_mesh_only(
    nodes: np.ndarray,
    boundary_faces: np.ndarray,
    out_path: Path,
    elev: float,
    azim: float,
    surface_alpha: float,
    wire_alpha: float,
) -> None:
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    _add_surface(
        ax,
        nodes,
        boundary_faces,
        face_colors=(0.70, 0.78, 0.90, surface_alpha),
        edge_color=(0.15, 0.15, 0.20, wire_alpha),
        alpha=surface_alpha,
    )
    ax.set_title("FEM Mesh Surface")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_3d_axes(ax, nodes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_fluence_surface(
    nodes: np.ndarray,
    boundary_faces: np.ndarray,
    nodal_fluence: np.ndarray,
    out_path: Path,
    cmap_name: str,
    elev: float,
    azim: float,
    surface_alpha: float,
    wire_alpha: float,
) -> None:
    face_values = nodal_fluence[boundary_faces].mean(axis=1)
    finite = face_values[np.isfinite(face_values)]
    if finite.size == 0:
        raise ValueError("光通量中没有可视化的有限值。")

    cmap = colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
    colors = cmap(norm(face_values))
    colors[:, 3] = surface_alpha

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    _add_surface(
        ax,
        nodes,
        boundary_faces,
        face_colors=colors,
        edge_color=(0.05, 0.05, 0.05, wire_alpha),
        alpha=surface_alpha,
    )
    ax.set_title("FEM Fluence Surface")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_3d_axes(ax, nodes)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(finite)
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, shrink=0.72)
    cbar.set_label("Fluence")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_fluence_points(
    nodes: np.ndarray,
    nodal_fluence: np.ndarray,
    bndvtx: np.ndarray,
    out_path: Path,
    cmap_name: str,
    elev: float,
    azim: float,
    point_size: float,
    clip_pct: tuple[float, float],
) -> None:
    mask = bndvtx > 0
    plot_nodes = nodes[mask]
    plot_values = nodal_fluence[mask]
    finite = plot_values[np.isfinite(plot_values)]
    if finite.size == 0:
        raise ValueError("边界节点光通量中没有可视化的有限值。")

    vmin = float(np.percentile(finite, clip_pct[0]))
    vmax = float(np.percentile(finite, clip_pct[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(finite.min())
        vmax = float(finite.max())

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        plot_nodes[:, 0],
        plot_nodes[:, 1],
        plot_nodes[:, 2],
        c=plot_values,
        s=point_size,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        alpha=0.92,
        linewidths=0,
    )
    ax.set_title("FEM Boundary Node Fluence")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_3d_axes(ax, plot_nodes)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.05, shrink=0.72)
    cbar.set_label("Fluence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    nodal_fluence = np.load(args.fluence)

    if len(nodal_fluence) != len(mesh.nodes):
        raise ValueError(
            f"节点光通量长度 {len(nodal_fluence)} 与 mesh 节点数 {len(mesh.nodes)} 不一致。"
        )

    clip_parts = [float(x.strip()) for x in args.clip_pct.split(",")]
    if len(clip_parts) != 2:
        raise ValueError("--clip-pct 必须是 '1,99' 这种形式。")
    clip_pct = (clip_parts[0], clip_parts[1])

    render_mesh_only(
        mesh.nodes,
        mesh.boundary_faces,
        out_dir / "fem_mesh_surface.png",
        elev=args.elev,
        azim=args.azim,
        surface_alpha=args.surface_alpha,
        wire_alpha=args.wire_alpha,
    )
    render_fluence_surface(
        mesh.nodes,
        mesh.boundary_faces,
        nodal_fluence,
        out_dir / "fem_fluence_surface.png",
        cmap_name=args.cmap,
        elev=args.elev,
        azim=args.azim,
        surface_alpha=args.surface_alpha,
        wire_alpha=args.wire_alpha,
    )
    render_fluence_points(
        mesh.nodes,
        nodal_fluence,
        mesh.bndvtx,
        out_dir / "fem_fluence_points.png",
        cmap_name=args.cmap,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
        clip_pct=clip_pct,
    )

    print(f"已输出到: {out_dir}")
    print(f"- {out_dir / 'fem_mesh_surface.png'}")
    print(f"- {out_dir / 'fem_fluence_surface.png'}")
    print(f"- {out_dir / 'fem_fluence_points.png'}")


if __name__ == "__main__":
    main()
