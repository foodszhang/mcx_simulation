from __future__ import annotations

from pathlib import Path

import numpy as np


def _tet_faces(element_nodes: np.ndarray) -> np.ndarray:
    p = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    return element_nodes[p]


def pre_amira_mesh(file_am: str | Path, file_txt: str | Path | None = None) -> tuple[int, np.ndarray, int, np.ndarray, int, np.ndarray]:
    """Port of ``preAmiraMesh.m``.

    Reads Amira tetra mesh and extracts boundary triangular faces.
    Returns MATLAB-like outputs:
        sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface
    """

    file_am = Path(file_am)
    lines = file_am.read_text(encoding="utf-8", errors="ignore").splitlines()

    n_nodes = None
    n_tets = None
    nodes = None
    elem_nodes = None
    elem_regions = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("nNodes"):
            n_nodes = int(line.split()[-1])
            nodes = np.zeros((n_nodes, 3), dtype=np.float64)
        elif line.startswith("nTetrahedra"):
            n_tets = int(line.split()[-1])
            elem_nodes = np.zeros((n_tets, 4), dtype=np.int64)
            elem_regions = np.zeros((n_tets,), dtype=np.int64)
        elif line == "@1":
            if n_nodes is None:
                raise ValueError("Invalid amira mesh: nNodes not found before @1")
            for k in range(n_nodes):
                i += 1
                nodes[k] = np.fromstring(lines[i], sep=" ", dtype=np.float64)
        elif line == "@2":
            if n_tets is None:
                raise ValueError("Invalid amira mesh: nTetrahedra not found before @2")
            for k in range(n_tets):
                i += 1
                elem_nodes[k] = np.fromstring(lines[i], sep=" ", dtype=np.int64)
        elif line == "@3":
            if n_tets is None:
                raise ValueError("Invalid amira mesh: nTetrahedra not found before @3")
            for k in range(n_tets):
                i += 1
                # MATLAB +1 region label
                elem_regions[k] = int(lines[i].strip()) + 1
        i += 1

    if nodes is None or elem_nodes is None or elem_regions is None:
        raise ValueError("Failed to parse Amira mesh sections (@1/@2/@3).")

    # Amira tetra node IDs are 1-based in this legacy dataset.
    # Convert once here so downstream Python code can use native 0-based indexing.
    elem_nodes = elem_nodes - 1
    if elem_nodes.min() < 0 or elem_nodes.max() >= nodes.shape[0]:
        raise ValueError(
            f"Invalid tetra node indices after 0-based conversion: "
            f"min={elem_nodes.min()}, max={elem_nodes.max()}, n_nodes={nodes.shape[0]}"
        )

    # Find boundary faces: faces that appear once
    faces = np.vstack([_tet_faces(elem_nodes[t]) for t in range(elem_nodes.shape[0])])
    faces_sorted = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]

    # MATLAB packs as [label, n1, n2, n3], label default 1
    element_inside = np.column_stack([elem_regions, elem_nodes]).astype(np.int64)
    element_surface = np.column_stack([np.ones((boundary_faces.shape[0], 1), dtype=np.int64), boundary_faces]).astype(np.int64)

    if file_txt is not None:
        file_txt = Path(file_txt)
        with file_txt.open("w", encoding="utf-8") as f:
            f.write(f"{nodes.shape[0]}\n")
            for row in nodes:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")
            f.write(f"{element_inside.shape[0]}\n")
            for row in element_inside:
                f.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]}\n")
            f.write(f"{element_surface.shape[0]}\n")
            for row in element_surface[:, 1:]:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")

    return (
        int(nodes.shape[0]),
        nodes,
        int(element_inside.shape[0]),
        element_inside,
        int(element_surface.shape[0]),
        element_surface,
    )


def getsurface(min_boundary: float, max_boundary: float, element_surface: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Port of ``getsurface.m``.

    Keeps only triangle faces with mean z between [min_boundary, max_boundary].
    Input can be either shape (N,3) or (N,4). Output is (N,4) [flag,n1,n2,n3].
    """

    es = np.asarray(element_surface, dtype=np.int64)
    if es.ndim != 2:
        raise ValueError("element_surface must be 2D.")
    if es.shape[1] == 3:
        es = np.column_stack([np.ones((es.shape[0], 1), dtype=np.int64), es])
    elif es.shape[1] != 4:
        raise ValueError("element_surface must have 3 or 4 columns.")

    keep = np.ones((es.shape[0],), dtype=bool)
    for i in range(es.shape[0]):
        tri = es[i, 1:4]
        z_mean = float(np.mean(nodes[tri, 2]))
        if z_mean < min_boundary or z_mean > max_boundary:
            keep[i] = False
    return es[keep]
