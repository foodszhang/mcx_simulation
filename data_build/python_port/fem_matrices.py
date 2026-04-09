from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def _tet_geom(nodes4: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute tet determinant and gradient terms as in MATLAB code."""
    a = np.vstack([np.ones((1, 4), dtype=np.float64), nodes4.T])  # 4x4
    d0 = abs(np.linalg.det(a))
    # Cofactor-like terms (match MATLAB index formulas)
    av1 = -np.linalg.det(a[np.ix_([0, 2, 3], [1, 2, 3])])
    bv1 = np.linalg.det(a[np.ix_([0, 1, 3], [1, 2, 3])])
    cv1 = -np.linalg.det(a[np.ix_([0, 1, 2], [1, 2, 3])])
    av2 = np.linalg.det(a[np.ix_([0, 2, 3], [0, 2, 3])])
    bv2 = -np.linalg.det(a[np.ix_([0, 1, 3], [0, 2, 3])])
    cv2 = np.linalg.det(a[np.ix_([0, 1, 2], [0, 2, 3])])
    av3 = -np.linalg.det(a[np.ix_([0, 2, 3], [0, 1, 3])])
    bv3 = np.linalg.det(a[np.ix_([0, 1, 3], [0, 1, 3])])
    cv3 = -np.linalg.det(a[np.ix_([0, 1, 2], [0, 1, 3])])
    av4 = np.linalg.det(a[np.ix_([0, 2, 3], [0, 1, 2])])
    bv4 = -np.linalg.det(a[np.ix_([0, 1, 3], [0, 1, 2])])
    cv4 = np.linalg.det(a[np.ix_([0, 1, 2], [0, 1, 2])])
    g = np.array(
        [
            [av1 * av1 + bv1 * bv1 + cv1 * cv1, av1 * av2 + bv1 * bv2 + cv1 * cv2, av1 * av3 + bv1 * bv3 + cv1 * cv3, av1 * av4 + bv1 * bv4 + cv1 * cv4],
            [av2 * av1 + bv2 * bv1 + cv2 * cv1, av2 * av2 + bv2 * bv2 + cv2 * cv2, av2 * av3 + bv2 * bv3 + cv2 * cv3, av2 * av4 + bv2 * bv4 + cv2 * cv4],
            [av3 * av1 + bv3 * bv1 + cv3 * cv1, av3 * av2 + bv3 * bv2 + cv3 * cv2, av3 * av3 + bv3 * bv3 + cv3 * cv3, av3 * av4 + bv3 * bv4 + cv3 * cv4],
            [av4 * av1 + bv4 * bv1 + cv4 * cv1, av4 * av2 + bv4 * bv2 + cv4 * cv2, av4 * av3 + bv4 * bv3 + cv4 * cv3, av4 * av4 + bv4 * bv4 + cv4 * cv4],
        ],
        dtype=np.float64,
    )
    return d0, g


def fem_linear(d: np.ndarray, u: np.ndarray, nodes: np.ndarray, element_inside: np.ndarray, element_surface: np.ndarray, an: float) -> csr_matrix:
    """Port of ``FEMLinear.m`` and ``FormMatrixM``."""
    n = nodes.shape[0]
    m = lil_matrix((n, n), dtype=np.float64)

    for l in range(element_inside.shape[0]):
        reg = int(element_inside[l, 0]) - 1
        tet = element_inside[l, 1:5].astype(np.int64)
        d0, g = _tet_geom(nodes[tet, :])
        if d0 <= 1e-15:
            continue
        kl = (d[reg] / (6.0 * d0)) * g
        cl = (d0 / 120.0) * np.ones((4, 4), dtype=np.float64)
        cl += np.diag(np.diag(cl))
        local = kl + u[reg] * cl
        for i in range(4):
            for j in range(4):
                m[tet[i], tet[j]] += local[i, j]

    # Robin boundary
    es = element_surface
    if es.shape[1] == 3:
        tri_nodes = es
    else:
        tri_nodes = es[:, 1:4]
    for l in range(tri_nodes.shape[0]):
        tri = tri_nodes[l].astype(np.int64)
        x = nodes[tri, :]
        ax = np.array([[1.0, 1.0, 1.0], [x[0, 0], x[1, 0], x[2, 0]], [x[0, 1], x[1, 1], x[2, 1]]], dtype=np.float64)
        ay = np.array([[1.0, 1.0, 1.0], [x[0, 0], x[1, 0], x[2, 0]], [x[0, 2], x[1, 2], x[2, 2]]], dtype=np.float64)
        az = np.array([[1.0, 1.0, 1.0], [x[0, 1], x[1, 1], x[2, 1]], [x[0, 2], x[1, 2], x[2, 2]]], dtype=np.float64)
        d0 = float(np.sqrt(np.linalg.det(ax) ** 2 + np.linalg.det(ay) ** 2 + np.linalg.det(az) ** 2))
        bl = (d0 / (48.0 * an)) * np.ones((3, 3), dtype=np.float64)
        bl += np.diag(np.diag(bl))
        for i in range(3):
            for j in range(3):
                m[tri[i], tri[j]] += bl[i, j]

    return m.tocsr()


def form_matrix_f(fy0: np.ndarray, sign: np.ndarray, nodes: np.ndarray, element_inside: np.ndarray) -> csr_matrix:
    """Port of ``FormMatrixF.m``."""
    n = nodes.shape[0]
    f = lil_matrix((n, n), dtype=np.float64)
    for l in range(element_inside.shape[0]):
        if int(sign[l]) != 1:
            continue
        tet = element_inside[l, 1:5].astype(np.int64)
        a = np.vstack([np.ones((1, 4), dtype=np.float64), nodes[tet, :].T])
        d0 = abs(np.linalg.det(a))
        fl = (d0 / 120.0) * np.ones((4, 4), dtype=np.float64)
        fl += np.diag(np.diag(fl))
        for i in range(4):
            for j in range(4):
                f[tet[i], tet[j]] += fl[i, j]
    f = f.tocsr()
    # MATLAB: each column j scales by fy0(j)
    fy = np.asarray(fy0, dtype=np.float64).reshape(-1)
    if fy.shape[0] != n:
        raise ValueError(f"fy0 length mismatch: expected {n}, got {fy.shape[0]}")
    return f @ csr_matrix(np.diag(fy))

