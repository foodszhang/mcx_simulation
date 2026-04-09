from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def getlaplace(nodes: np.ndarray, element_inside: np.ndarray) -> np.ndarray:
    """Port of MATLAB ``getlaplace.m`` (topological normalized Laplacian)."""
    n = nodes.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    deg = np.zeros((n, n), dtype=np.float64)
    elem = element_inside[:, 1:5].astype(np.int64)
    for i in range(n):
        idx_u, _ = np.where(elem == i)
        hood = np.unique(elem[idx_u].reshape(-1))
        adj[i, hood] = 1.0
        adj[hood, i] = 1.0
        deg[i, i] = max(len(hood) - 1, 1)
    np.fill_diagonal(adj, 0.0)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(deg), 1e-12)))
    return np.eye(n, dtype=np.float64) - inv_sqrt @ adj @ inv_sqrt


def _adaptive_radius_neighbors(nodes: np.ndarray, min_count: int, scale: float) -> list[np.ndarray]:
    """Mimics MATLAB neighbor search by increasing radius until min_count reached."""
    n = nodes.shape[0]
    d_all = cdist(nodes, nodes)
    neighbors = []
    for i in range(n):
        r = 0.1
        idx = np.where(d_all[i] <= r)[0]
        while idx.size < min_count:
            r += 0.01
            idx = np.where(d_all[i] <= r)[0]
            if r > 10.0:
                break
        idx = np.where(d_all[i] <= scale * r)[0]
        neighbors.append(idx)
    return neighbors


def new_getlaplace(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of MATLAB ``new_getlaplace.m``."""
    n = nodes.shape[0]
    nbs1 = _adaptive_radius_neighbors(nodes, min_count=4, scale=1.0)
    nbs2 = _adaptive_radius_neighbors(nodes, min_count=4, scale=1.5)
    nbs3 = _adaptive_radius_neighbors(nodes, min_count=4, scale=2.0)

    def build_lap(nbs: list[np.ndarray]) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float64)
        deg = np.zeros((n, n), dtype=np.float64)
        for i, hood in enumerate(nbs):
            adj[i, hood] = 1.0
            adj[hood, i] = 1.0
            deg[i, i] = max(len(hood) - 1, 1)
        np.fill_diagonal(adj, 0.0)
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(deg), 1e-12)))
        return np.eye(n, dtype=np.float64) - inv_sqrt @ adj @ inv_sqrt

    return build_lap(nbs1), build_lap(nbs2), build_lap(nbs3)


def new_getlaplace_ghb(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of MATLAB ``new_getlaplace_GHB.m``."""
    n = nodes.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    d1 = np.zeros((n, n), dtype=np.float64)
    d2 = np.zeros((n, n), dtype=np.float64)
    d3 = np.zeros((n, n), dtype=np.float64)
    d_all = cdist(nodes, nodes)

    for i in range(n):
        r = 1.0
        idx = np.where(d_all[i] <= r)[0]
        while idx.size < 5:
            r += 0.01
            idx = np.where(d_all[i] <= r)[0]
            if r > 10.0:
                break
        i1 = np.where(d_all[i] <= r)[0]
        i2 = np.where(d_all[i] <= 2 * r)[0]
        i3 = np.where(d_all[i] <= 3 * r)[0]
        i4 = np.where(d_all[i] <= 4 * r)[0]
        for a in i1:
            for b in i1:
                if b > a:
                    d[a, b] = d[b, a] = np.sum((nodes[a] - nodes[b]) ** 2)
        for a in i2:
            for b in i2:
                if b > a:
                    d1[a, b] = d1[b, a] = np.sum((nodes[a] - nodes[b]) ** 2)
        for a in i3:
            for b in i3:
                if b > a:
                    d2[a, b] = d2[b, a] = np.sum((nodes[a] - nodes[b]) ** 2)
        for a in i4:
            for b in i4:
                if b > a:
                    d3[a, b] = d3[b, a] = np.sum((nodes[a] - nodes[b]) ** 2)

    def kernel_lap(dist_mat: np.ndarray) -> np.ndarray:
        denom = np.sum(dist_mat != 0, axis=0)
        denom = np.maximum(denom, 1)
        e = np.sum(np.sqrt(dist_mat), axis=0) / denom
        k = np.exp(np.outer(e, e) * (-dist_mat))
        k[k == 1.0] = 0.0
        dnorm = np.diag(1.0 / np.sqrt(np.sum(k, axis=1) + 1.0))
        return dnorm @ (k + np.eye(n, dtype=np.float64)) @ dnorm

    lap = kernel_lap(d)
    n_lap1 = kernel_lap(d1)
    n_lap2 = kernel_lap(d2)
    n_lap3 = kernel_lap(d3)
    return lap, n_lap1, n_lap2, n_lap3

