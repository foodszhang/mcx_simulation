from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from .fem_matrices import fem_linear, form_matrix_f
from .initialization import initialization
from .laplacian import getlaplace, new_getlaplace_ghb
from .mesh_io import getsurface, pre_amira_mesh


def run_pipeline(file_am: str, file_txt: str, min_boundary: float, max_boundary: float, save_prefix: str) -> dict:
    """Python port of data_build/main.m (currently active branch)."""
    size_node, nodes, size_inside, element_inside, _size_surface, element_surface = pre_amira_mesh(file_am, file_txt)
    uam, dm, an = initialization()
    element_surface = getsurface(min_boundary, max_boundary, element_surface, nodes)

    fyx = np.ones((nodes.shape[0],), dtype=np.float64)
    sign = np.ones((element_inside.shape[0],), dtype=np.int64)
    f = form_matrix_f(fyx, sign, nodes, element_inside)
    mm = fem_linear(dm, uam, nodes, element_inside, element_surface, an)

    # Equivalent to MATLAB: A = full(Mm\F)
    # Column-wise solve for sparse RHS
    f_dense = f.toarray()
    a = np.column_stack([spsolve(mm, f_dense[:, j]) for j in range(f_dense.shape[1])])

    surface_idx = np.unique(element_surface[:, 1:4].astype(np.int64).reshape(-1))
    a_surface = a[surface_idx, :]

    lap = getlaplace(nodes, element_inside)
    n_lap0, n_lap1, n_lap2, n_lap3 = new_getlaplace_ghb(nodes)

    out = {
        "sizeNode": int(size_node),
        "sizeInsideElement": int(size_inside),
        "nodes": nodes,
        "elementInside": element_inside,
        "elementSurface": element_surface,
        "A_full": a,
        "A_surface": a_surface,
        "surface_index": surface_idx,
        "Lap": lap,
        "n_Lap0": n_lap0,
        "n_Lap1": n_lap1,
        "n_Lap2": n_lap2,
        "n_Lap3": n_lap3,
    }

    np.savez_compressed(
        f"{save_prefix}.npz",
        **out,
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python port for data_build/main.m")
    p.add_argument("--amira", required=True, help="Amira mesh file path (.am)")
    p.add_argument("--mesh_txt", default="mesh_from_amira.txt", help="Optional intermediate txt output path")
    p.add_argument("--min_boundary", type=float, default=0.3)
    p.add_argument("--max_boundary", type=float, default=33.3)
    p.add_argument("--save_prefix", default="recon_info_python_port")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        file_am=args.amira,
        file_txt=args.mesh_txt,
        min_boundary=args.min_boundary,
        max_boundary=args.max_boundary,
        save_prefix=args.save_prefix,
    )
    print(f"Saved python-port result: {Path(args.save_prefix).with_suffix('.npz')}")


if __name__ == "__main__":
    main()

