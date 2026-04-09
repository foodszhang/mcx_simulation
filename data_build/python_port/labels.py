from __future__ import annotations

import numpy as np


def get_singlelabel(nodes: np.ndarray, a: np.ndarray, test: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``get_signlelabel.m``."""
    n = nodes.shape[0]
    if test == 0:
        x = np.arange(10.5, 12.5 + 1e-9, 0.5)
        y = np.arange(10.0, 14.0 + 1e-9, 0.5)
        z = np.arange(16.0, 19.0 + 1e-9, 0.5)
        r = np.arange(1.0, 2.0 + 1e-9, 0.2)
    else:
        x = np.array([11.5])
        y = np.array([11.0])
        z = np.array([17.0])
        r = np.array([1.0])

    labels = []
    for zz in z:
        for xx in x:
            for yy in y:
                center = np.array([xx, yy, zz], dtype=np.float64)
                for rr in r:
                    mask = np.linalg.norm(nodes[:, :3] - center[None, :], axis=1) <= rr
                    if mask.any():
                        labels.append(mask.astype(np.float64))
    if not labels:
        return np.zeros((0, n), dtype=np.float64), np.zeros((a.shape[0], 0), dtype=np.float64)
    label_data = np.vstack(labels)
    test_b = np.maximum(a @ label_data.T, 0.0)
    c = np.sum(label_data, axis=1)
    c1 = np.sum(test_b, axis=0)
    keep = ~((c <= 0) & (c1 <= 0))
    return label_data[keep], test_b[:, keep]


def get_double(nodes: np.ndarray, a: np.ndarray, test: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``get_double.m``."""
    n = nodes.shape[0]
    if test == 0:
        x = np.arange(10.5, 12.5 + 1e-9, 0.5)
        y = np.arange(10.0, 14.0 + 1e-9, 0.5)
        z = np.arange(16.0, 19.0 + 1e-9, 0.5)
        r = np.arange(1.0, 2.0 + 1e-9, 0.2)
        dis = np.array([4.0, 6.0, 8.0])
    else:
        x = np.array([12.5])
        y = np.array([11.0, 12.0, 13.0])
        z = np.array([17.5])
        r = np.array([1.4])
        dis = np.array([4.0, 6.0, 8.0])

    labels = []
    for zz in z:
        for xx in x:
            for yy in y:
                c0 = np.array([xx, yy, zz], dtype=np.float64)
                for dd in dis:
                    for rr in r:
                        c1 = c0 + np.array([dd + 2.0 * rr, 0.0, 0.0], dtype=np.float64)
                        m0 = np.linalg.norm(nodes[:, :3] - c0[None, :], axis=1) <= rr
                        m1 = np.linalg.norm(nodes[:, :3] - c1[None, :], axis=1) <= rr
                        m = m0 | m1
                        if m.any():
                            labels.append(m.astype(np.float64))
    if not labels:
        return np.zeros((0, n), dtype=np.float64), np.zeros((a.shape[0], 0), dtype=np.float64)
    label_data = np.vstack(labels)
    test_b = np.maximum(a @ label_data.T, 0.0)
    c = np.sum(label_data, axis=1)
    c1 = np.sum(test_b, axis=0)
    keep = ~((c <= 0) & (c1 <= 0))
    return label_data[keep], test_b[:, keep]

