from __future__ import annotations

import numpy as np


def sorcesetup(nodes: np.ndarray, element_inside: np.ndarray, excite_nodes: np.ndarray) -> np.ndarray:
    """Port of ``sorcesetup.m``.

    For each excitation coordinate, find nearest tetra centroid and light its 4 nodes.
    """
    n_nodes = nodes.shape[0]
    n_src = excite_nodes.shape[0]
    lx = np.zeros((n_nodes, n_src), dtype=np.float64)
    centroids = np.mean(nodes[element_inside[:, 1:5].astype(np.int64)], axis=1)
    for i in range(n_src):
        d = np.linalg.norm(centroids - excite_nodes[i][None, :], axis=1)
        k = int(np.argmin(d))
        tet_nodes = element_inside[k, 1:5].astype(np.int64)
        exsign = np.zeros((n_nodes,), dtype=np.float64)
        exsign[tet_nodes] = 1.0
        lx[:, i] = exsign
    return lx


def get_surface_energy(sp3result: np.ndarray, element_surface: np.ndarray, n_source: int) -> np.ndarray:
    """Port of ``get_surface_energy.m``."""
    if element_surface.shape[1] == 4:
        surf_idx = np.unique(element_surface[:, 1:4].astype(np.int64).reshape(-1))
    else:
        surf_idx = np.unique(element_surface.astype(np.int64).reshape(-1))
    out = np.zeros((sp3result.shape[0], n_source), dtype=np.float64)
    out[surf_idx, :] = sp3result[surf_idx, :]
    return out

