from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NirfastStyleMesh:
    """参考 NIRFAST 的轻量 mesh 数据结构。"""

    nodes: np.ndarray
    elements: np.ndarray
    region: np.ndarray
    boundary_faces: np.ndarray
    bndvtx: np.ndarray
    dimension: int
    spacing: np.ndarray | None = None
    voxel_shape: np.ndarray | None = None
    cell_coords: np.ndarray | None = None
    cell_nodes: np.ndarray | None = None
    surface_faces: np.ndarray | None = None
    surface_face_regions: np.ndarray | None = None
    boundary_face_regions: np.ndarray | None = None
    mua: np.ndarray | None = None
    mus: np.ndarray | None = None
    g: np.ndarray | None = None
    ri: np.ndarray | None = None
    kappa: np.ndarray | None = None
    ksi: np.ndarray | None = None
    element_mua: np.ndarray | None = None
    element_mus: np.ndarray | None = None
    element_g: np.ndarray | None = None
    element_ri: np.ndarray | None = None
    element_kappa: np.ndarray | None = None


def _optional_array(value: np.ndarray | None) -> np.ndarray:
    return np.array([]) if value is None else value


def _downsample_label_volume(volume_data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume_data.astype(np.uint8, copy=True)

    nz, ny, nx = volume_data.shape
    tz = int(np.ceil(nz / factor))
    ty = int(np.ceil(ny / factor))
    tx = int(np.ceil(nx / factor))
    reduced = np.zeros((tz, ty, tx), dtype=np.uint8)

    for z in range(tz):
        z0, z1 = z * factor, min((z + 1) * factor, nz)
        for y in range(ty):
            y0, y1 = y * factor, min((y + 1) * factor, ny)
            for x in range(tx):
                x0, x1 = x * factor, min((x + 1) * factor, nx)
                block = volume_data[z0:z1, y0:y1, x0:x1]
                positive = block[block > 0]
                values = positive if positive.size > 0 else block.reshape(-1)
                bincount = np.bincount(values.astype(np.int32))
                reduced[z, y, x] = np.uint8(np.argmax(bincount))
    return reduced


def downsample_binary_max(binary_volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Efficiently downsample a binary volume using max pooling (sparse implementation)."""
    src_shape = np.array(binary_volume.shape, dtype=np.float64)
    target = np.array(target_shape, dtype=np.int32)
    scale = src_shape / target
    
    # Fast path for sparse binary volumes
    active_indices = np.argwhere(binary_volume > 0.5)
    if active_indices.size == 0:
        return np.zeros(tuple(target_shape), dtype=np.float32)
        
    # Map high-res indices to low-res indices
    reduced_indices = (active_indices / scale).astype(np.int32)
    
    # Clip to valid range (handling edge cases)
    np.clip(reduced_indices[:, 0], 0, target_shape[0] - 1, out=reduced_indices[:, 0])
    np.clip(reduced_indices[:, 1], 0, target_shape[1] - 1, out=reduced_indices[:, 1])
    np.clip(reduced_indices[:, 2], 0, target_shape[2] - 1, out=reduced_indices[:, 2])
    
    reduced = np.zeros(tuple(target_shape), dtype=np.float32)
    # Set active voxels to 1.0 (duplicate indices are handled naturally by assignment)
    reduced[reduced_indices[:, 0], reduced_indices[:, 1], reduced_indices[:, 2]] = 1.0
    
    return reduced


def _compute_surface_faces(
    elements: np.ndarray,
    regions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(elements) == 0:
        empty_faces = np.zeros((0, 3), dtype=np.int32)
        empty_regions = np.zeros((0, 2), dtype=np.int32)
        return empty_faces, empty_regions, empty_faces, empty_regions

    local_faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    faces = np.vstack([elements[:, f] for f in local_faces])
    owners = np.repeat(np.arange(len(elements), dtype=np.int32), len(local_faces))
    owner_regions = regions[owners]

    sorted_faces = np.sort(faces, axis=1)
    unique_faces, inverse, counts = np.unique(
        sorted_faces,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    order = np.argsort(inverse)
    inverse_sorted = inverse[order]
    owner_regions_sorted = owner_regions[order]

    surface_faces = []
    surface_face_regions = []
    boundary_faces = []
    boundary_face_regions = []

    start = 0
    for face_idx, count in enumerate(counts):
        end = start + count
        face = unique_faces[face_idx]
        regs = owner_regions_sorted[start:end]
        start = end

        if count == 1:
            reg_pair = np.array([int(regs[0]), 0], dtype=np.int32)
            boundary_faces.append(face)
            boundary_face_regions.append(reg_pair)
            surface_faces.append(face)
            surface_face_regions.append(reg_pair)
        elif count == 2 and regs[0] != regs[1]:
            reg_pair = np.array(sorted((int(regs[0]), int(regs[1]))), dtype=np.int32)
            surface_faces.append(face)
            surface_face_regions.append(reg_pair)

    return (
        np.asarray(boundary_faces, dtype=np.int32),
        np.asarray(boundary_face_regions, dtype=np.int32),
        np.asarray(surface_faces, dtype=np.int32),
        np.asarray(surface_face_regions, dtype=np.int32),
    )


def _compute_fresnel_ksi(ri: np.ndarray) -> np.ndarray:
    ri = np.asarray(ri, dtype=np.float64)
    ro = ((ri - 1.0) ** 2) / ((ri + 1.0) ** 2 + 1e-12)
    inv_ri = np.clip(1.0 / np.maximum(ri, 1.0), 0.0, 1.0)
    cos_theta_c = np.abs(np.cos(np.arcsin(inv_ri)))
    a = ((2.0 / (1.0 - ro + 1e-12)) - 1.0 + cos_theta_c**3) / (
        1.0 - cos_theta_c**2 + 1e-12
    )
    return 1.0 / (2.0 * a + 1e-12)


def build_tetrahedral_mesh_from_volume(
    volume_data: np.ndarray,
    dx: float,
    mesh_config: dict | None = None,
) -> NirfastStyleMesh:
    """从体素分割直接生成保结构四面体网格。"""
    if mesh_config is None:
        mesh_config = {}

    factor = int(mesh_config.get("downsample_factor", 8))
    reduced_volume = _downsample_label_volume(volume_data, factor=factor)

    nz, ny, nx = reduced_volume.shape
    node_grid_shape = (nz + 1, ny + 1, nx + 1)
    node_ids = np.arange(np.prod(node_grid_shape), dtype=np.int32).reshape(node_grid_shape)
    spacing = np.array([dx * factor, dx * factor, dx * factor], dtype=np.float64)

    tetra_pattern = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )

    elements: list[list[int]] = []
    regions: list[int] = []
    cell_coords: list[list[int]] = []
    cell_nodes: list[list[int]] = []

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                label = int(reduced_volume[z, y, x])
                if label <= 0:
                    continue

                corners = [
                    node_ids[z, y, x],
                    node_ids[z, y, x + 1],
                    node_ids[z, y + 1, x],
                    node_ids[z, y + 1, x + 1],
                    node_ids[z + 1, y, x],
                    node_ids[z + 1, y, x + 1],
                    node_ids[z + 1, y + 1, x],
                    node_ids[z + 1, y + 1, x + 1],
                ]

                cell_coords.append([z, y, x])
                cell_nodes.append(corners)
                for tet in tetra_pattern:
                    elements.append([corners[tet[0]], corners[tet[1]], corners[tet[2]], corners[tet[3]]])
                    regions.append(label)

    if not elements:
        raise ValueError("体素中没有有效组织区域，无法构建FEM网格。")

    elements_arr = np.asarray(elements, dtype=np.int32)
    regions_arr = np.asarray(regions, dtype=np.int32)
    cell_coords_arr = np.asarray(cell_coords, dtype=np.int32)
    cell_nodes_arr = np.asarray(cell_nodes, dtype=np.int32)

    used_nodes = np.unique(elements_arr)
    node_map = np.full(np.prod(node_grid_shape), -1, dtype=np.int32)
    node_map[used_nodes] = np.arange(len(used_nodes), dtype=np.int32)
    elements_arr = node_map[elements_arr]
    cell_nodes_arr = node_map[cell_nodes_arr]

    zyx = np.column_stack(np.unravel_index(used_nodes, node_grid_shape)).astype(np.float64)
    nodes = np.column_stack(
        [
            zyx[:, 2] * spacing[2],
            zyx[:, 1] * spacing[1],
            zyx[:, 0] * spacing[0],
        ]
    )

    (
        boundary_faces,
        boundary_face_regions,
        surface_faces,
        surface_face_regions,
    ) = _compute_surface_faces(elements_arr, regions_arr)

    bndvtx = np.zeros(len(nodes), dtype=np.int32)
    if len(boundary_faces) > 0:
        bndvtx[np.unique(boundary_faces)] = 1

    return NirfastStyleMesh(
        nodes=nodes.astype(np.float64),
        elements=elements_arr,
        region=regions_arr,
        boundary_faces=boundary_faces,
        bndvtx=bndvtx,
        dimension=3,
        spacing=spacing,
        voxel_shape=np.array(reduced_volume.shape, dtype=np.int32),
        cell_coords=cell_coords_arr,
        cell_nodes=cell_nodes_arr,
        surface_faces=surface_faces,
        surface_face_regions=surface_face_regions,
        boundary_face_regions=boundary_face_regions,
    )


def save_mesh(mesh: NirfastStyleMesh, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        nodes=mesh.nodes,
        elements=mesh.elements,
        region=mesh.region,
        boundary_faces=mesh.boundary_faces,
        bndvtx=mesh.bndvtx,
        dimension=np.array([mesh.dimension], dtype=np.int32),
        spacing=_optional_array(mesh.spacing),
        voxel_shape=_optional_array(mesh.voxel_shape),
        cell_coords=_optional_array(mesh.cell_coords),
        cell_nodes=_optional_array(mesh.cell_nodes),
        surface_faces=_optional_array(mesh.surface_faces),
        surface_face_regions=_optional_array(mesh.surface_face_regions),
        boundary_face_regions=_optional_array(mesh.boundary_face_regions),
        mua=_optional_array(mesh.mua),
        mus=_optional_array(mesh.mus),
        g=_optional_array(mesh.g),
        ri=_optional_array(mesh.ri),
        kappa=_optional_array(mesh.kappa),
        ksi=_optional_array(mesh.ksi),
        element_mua=_optional_array(mesh.element_mua),
        element_mus=_optional_array(mesh.element_mus),
        element_g=_optional_array(mesh.element_g),
        element_ri=_optional_array(mesh.element_ri),
        element_kappa=_optional_array(mesh.element_kappa),
    )


def load_mesh(mesh_path: str | Path) -> NirfastStyleMesh:
    with np.load(mesh_path) as data:
        def _optional(name: str):
            if name not in data:
                return None
            arr = data[name]
            return None if arr.size == 0 else arr

        return NirfastStyleMesh(
            nodes=data["nodes"],
            elements=data["elements"],
            region=data["region"],
            boundary_faces=data["boundary_faces"],
            bndvtx=data["bndvtx"],
            dimension=int(data["dimension"][0]),
            spacing=_optional("spacing"),
            voxel_shape=_optional("voxel_shape"),
            cell_coords=_optional("cell_coords"),
            cell_nodes=_optional("cell_nodes"),
            surface_faces=_optional("surface_faces"),
            surface_face_regions=_optional("surface_face_regions"),
            boundary_face_regions=_optional("boundary_face_regions"),
            mua=_optional("mua"),
            mus=_optional("mus"),
            g=_optional("g"),
            ri=_optional("ri"),
            kappa=_optional("kappa"),
            ksi=_optional("ksi"),
            element_mua=_optional("element_mua"),
            element_mus=_optional("element_mus"),
            element_g=_optional("element_g"),
            element_ri=_optional("element_ri"),
            element_kappa=_optional("element_kappa"),
        )


def attach_optical_properties(mesh: NirfastStyleMesh, media_list: list[dict]) -> NirfastStyleMesh:
    """按 region 把材料参数映射到单元，并计算 NIRFAST 风格 kappa/ksi。"""
    n_nodes = len(mesh.nodes)
    n_elements = len(mesh.elements)

    elem_mua = np.zeros(n_elements, dtype=np.float64)
    elem_mus = np.zeros(n_elements, dtype=np.float64)
    elem_g = np.zeros(n_elements, dtype=np.float64)
    elem_ri = np.zeros(n_elements, dtype=np.float64)

    for idx, reg in enumerate(mesh.region):
        if reg >= len(media_list):
            raise ValueError(f"区域标签 {reg} 超出材料参数范围。")
        media = media_list[reg]
        elem_mua[idx] = float(media.get("mua", 0.0))
        elem_mus[idx] = float(media.get("mus", 0.0))
        elem_g[idx] = float(media.get("g", 0.0))
        elem_ri[idx] = float(media.get("n", 1.0))

    elem_mus_prime = elem_mus * (1.0 - elem_g)
    elem_kappa = 1.0 / (3.0 * (elem_mua + elem_mus_prime) + 1e-12)

    node_mua = np.zeros(n_nodes, dtype=np.float64)
    node_mus = np.zeros(n_nodes, dtype=np.float64)
    node_g = np.zeros(n_nodes, dtype=np.float64)
    node_ri = np.zeros(n_nodes, dtype=np.float64)
    node_counts = np.zeros(n_nodes, dtype=np.float64)

    for elem_idx, element in enumerate(mesh.elements):
        node_mua[element] += elem_mua[elem_idx]
        node_mus[element] += elem_mus[elem_idx]
        node_g[element] += elem_g[elem_idx]
        node_ri[element] += elem_ri[elem_idx]
        node_counts[element] += 1.0

    node_counts = np.maximum(node_counts, 1.0)
    node_mua /= node_counts
    node_mus /= node_counts
    node_g /= node_counts
    node_ri /= node_counts

    node_mus_prime = node_mus * (1.0 - node_g)
    node_kappa = 1.0 / (3.0 * (node_mua + node_mus_prime) + 1e-12)
    node_ksi = _compute_fresnel_ksi(node_ri)

    mesh.mua = node_mua
    mesh.mus = node_mus
    mesh.g = node_g
    mesh.ri = node_ri
    mesh.kappa = node_kappa
    mesh.ksi = node_ksi
    mesh.element_mua = elem_mua
    mesh.element_mus = elem_mus
    mesh.element_g = elem_g
    mesh.element_ri = elem_ri
    mesh.element_kappa = elem_kappa
    return mesh
