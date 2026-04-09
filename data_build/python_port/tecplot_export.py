from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_zone(fid, title: str, points: np.ndarray, elems: np.ndarray) -> None:
    fid.write(f'ZONE T={title}, N={points.shape[0]}, E={elems.shape[0]}, F=FEPOINT, ET=TETRAHEDRON\n')
    for row in points:
        fid.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")
    fid.write("\n")
    for row in elems:
        fid.write(f"{int(row[0])} {int(row[1])} {int(row[2])} {int(row[3])}\n")
    fid.write("\n")


def export_dat(path: str | Path, nodes: np.ndarray, element_inside: np.ndarray, density: np.ndarray, by_region: bool = True) -> None:
    """Generic Tecplot DAT exporter (port of step3/tecplot helpers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.zeros((nodes.shape[0], 4), dtype=np.float64)
    pts[:, :3] = nodes
    pts[:, 3] = density.reshape(-1)
    elems = element_inside[:, 1:5].astype(np.int64)

    with path.open("w", encoding="utf-8") as fid:
        fid.write('TITLE = "Result for invivo data"\n')
        fid.write('VARIABLES = "X", "Y", "Z", "Pred Density"\n')
        _write_zone(fid, '"all"', pts, elems)
        if by_region:
            regs = np.unique(element_inside[:, 0].astype(np.int64))
            for reg in regs:
                e = elems[element_inside[:, 0].astype(np.int64) == reg]
                if e.size == 0:
                    continue
                _write_zone(fid, f'"region_{reg}"', pts, e)

