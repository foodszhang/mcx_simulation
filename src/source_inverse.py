from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from src.fem_forward_solver import FemDiffusionSolver
from src.gen_mul_projection import (
    project_points_to_camera_batch_matrix,
    rotation_matrix_y,
)


def get_camera_pos(angle_deg: float, distance: float) -> tuple[float, float, float]:
    """
    计算给定角度和距离下的相机位置（世界坐标系）。
    假设默认相机在 (0, 0, distance)，物体绕 Y 轴旋转 angle_deg。
    相当于相机绕物体旋转 -angle_deg。
    
    R * P_world = P_camera_frame
    P_camera_frame_origin = [0, 0, distance]
    P_world_camera = R_inv * [0, 0, distance]
    
    Rotation matrix Y (angle):
    [ cos  0  sin]
    [  0   1   0 ]
    [-sin  0  cos]
    
    Inverse (transpose):
    [ cos  0 -sin]
    [  0   1   0 ]
    [ sin  0  cos]
    
    Product with [0, 0, d]:
    x = -d * sin(a)
    y = 0
    z = d * cos(a)
    """
    angle_rad = np.deg2rad(angle_deg)
    x = -distance * np.sin(angle_rad)
    y = 0.0
    z = distance * np.cos(angle_rad)
    return float(x), float(y), float(z)


@dataclass
class SourceBlockBasis:
    block_indices: list[np.ndarray]
    block_centers_zyx: np.ndarray
    block_grid_coords: np.ndarray
    block_shape: tuple[int, int, int]
    roi_shape: tuple[int, int, int]
    type: str = "voxel_block"


@dataclass
class SourceFemNodeBasis:
    node_indices: np.ndarray  # Indices of active nodes in the full mesh
    node_coords_zyx: np.ndarray # Coordinates of active nodes
    roi_bounds_zyx: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    type: str = "fem_node"


@dataclass
class VisibleSurfaceObservation:
    angles: list[int]
    row_slices_by_angle: dict[int, tuple[int, int]]
    patch_coords_by_angle: dict[int, np.ndarray]
    patch_centroids_by_angle: dict[int, np.ndarray]
    camera_depth_by_angle: dict[int, np.ndarray]
    patch_face_counts_by_angle: dict[int, np.ndarray]
    observation_matrix: csr_matrix
    observation_type: str = "surface_patch_partial_current"

    def rows_for_angle(self, angle: int) -> slice:
        start, end = self.row_slices_by_angle[angle]
        return slice(start, end)

    @property
    def n_measurements(self) -> int:
        return int(self.observation_matrix.shape[0])


def load_source_pattern_from_sample(config: dict, sample_dir: str, sample_id: int) -> np.ndarray:
    volume_shape = tuple(config["volume_shape"])
    src_range_z = config["src_range"]["z"]
    src_range_y = config["src_range"]["y"]
    src_range_x = config["src_range"]["x"]
    
    source_filename = f"source-{sample_id}.bin"
    source_path = Path(sample_dir) / source_filename
    if not source_path.exists():
        # Try checking if source file is in parent directory
        source_path = Path(sample_dir).parent / source_filename
        if not source_path.exists():
            raise FileNotFoundError(f"光源文件不存在: {source_path}")

    # Determine shape from file size if mismatch
    file_size = source_path.stat().st_size
    num_floats = file_size // 4
    
    expected_dz = src_range_z[1] - src_range_z[0]
    expected_dy = src_range_y[1] - src_range_y[0]
    expected_dx = src_range_x[1] - src_range_x[0]
    expected_size = expected_dz * expected_dy * expected_dx
    
    if num_floats == expected_size:
        src_shape = (expected_dz, expected_dy, expected_dx)
        z0, y0, x0 = src_range_z[0], src_range_y[0], src_range_x[0]
    else:
        # Fallback: try to read from JSON in the sample dir
        json_path = Path(sample_dir) / f"{sample_id}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    mcx_conf = json.load(f)
                pattern = mcx_conf.get("Optode", {}).get("Source", {}).get("Pattern", {})
                nx = pattern.get("Nx")
                ny = pattern.get("Ny")
                nz = pattern.get("Nz")
                if nx and ny and nz and (nx*ny*nz == num_floats):
                    src_shape = (nz, ny, nx)
                    # Use Pos as start index (1-based -> 0-based)
                    pos = mcx_conf.get("Optode", {}).get("Source", {}).get("Pos", [0,0,0])
                    x0 = int(pos[0]) - 1
                    y0 = int(pos[1]) - 1
                    z0 = int(pos[2]) - 1
                else:
                    raise ValueError(f"Source size mismatch: file {num_floats}, config {expected_size}")
            except Exception as e:
                raise ValueError(f"Failed to infer source shape from JSON: {e}")
        else:
             raise ValueError(f"Source size mismatch: file {num_floats}, config {expected_size}, and no JSON found")

    source_arr = np.fromfile(source_path, dtype=np.float32).reshape(src_shape)
    source_pattern = np.zeros(volume_shape, dtype=np.float32)
    
    # Safe assignment with clipping
    z1 = min(z0 + src_shape[0], volume_shape[0])
    y1 = min(y0 + src_shape[1], volume_shape[1])
    x1 = min(x0 + src_shape[2], volume_shape[2])
    
    dz = z1 - z0
    dy = y1 - y0
    dx = x1 - x0
    
    if dz > 0 and dy > 0 and dx > 0:
        source_pattern[z0:z1, y0:y1, x0:x1] = np.where(source_arr[:dz, :dy, :dx] > 0.5, 1.0, 0.0)

    return source_pattern


def _soft_threshold_positive(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(x - threshold, 0.0)


def _weighted_center(volume: np.ndarray) -> list[float]:
    coords = np.argwhere(volume > 1e-8)
    if coords.size == 0:
        return [0.0, 0.0, 0.0]
    weights = volume[tuple(coords.T)].astype(np.float64)
    center = (coords * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
    return center.tolist()


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


class VisiblePointSourceInverse:
    """基于“当前视角可见表面 patch partial current”的光源反演器。"""

    def __init__(self, config: dict, sample_dir: str, mesh_cache_path: str | None = None):
        self.config = config
        self.sample_dir = sample_dir
        self.inverse_config = config.get("inverse", {}).get("source_inverse", {})
        self.angles = [
            int(v)
            for v in self.inverse_config.get(
                "angles",
                config.get("projection", {}).get("angles", [0]),
            )
        ]
        # Prepare FEM solver arguments
        fem_config = config.get("fem", {})
        
        self.solver = FemDiffusionSolver(
            volume_path=config["generated_bin_path"],
            media_yaml_path=config["generated_material_yaml_path"],
            volume_shape=tuple(config["volume_shape"]),
            dx=config.get("lengthunit", 0.1),
            mesh_cache_path=config.get("fem", {}).get("mesh_cache_name", "brain_fem_mesh.npz"),
            fem_config=config.get("fem", {}),
        )
        self.basis_mode = self.inverse_config.get("basis_mode", "voxel_block")
        if self.basis_mode == "fem_node":
             self.basis = self._build_fem_node_basis()
        else:
             self.basis = self._build_source_basis()
        self.observation = self._build_observation()
        self._rhs_matrix: np.ndarray | None = None
        self._operator_cache: np.ndarray | None = None
        self._graph_edges: np.ndarray | None = None
        self._graph_laplacian: np.ndarray | None = None
        self._tv_difference: np.ndarray | None = None
        
        self._init_excitation()

    def _init_excitation(self):
        """初始化激发光场。支持固定光源或随相机移动的光源。"""
        ex_pos = self.inverse_config.get("excitation_pos")
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        
        self.angle_excitation_scalings: dict[int, np.ndarray] = {}
        self._excitation_volume = None

        if epi_mode:
            print(f"DEBUG: Initializing Epi-Illumination mode (Moving Source).")
            radius = float(self.inverse_config.get("excitation_radius", 0.5))
            for angle in self.angles:
                src_pos = self.compute_rotated_source_pos(angle)
                print(f"DEBUG: Angle {angle}, Excitation Source: {src_pos}")
                try:
                    phi_ex = self.solver.solve_excitation(src_pos, radius=radius)
                    if self.basis_mode == "fem_node":
                         if self.basis.node_indices.max() >= len(phi_ex): raise ValueError("Basis node indices exceed mesh node count")
                         basis_scaling = phi_ex[self.basis.node_indices]
                    else:
                        if self._excitation_volume is None: self._excitation_volume = self.solver.sample_to_volume(phi_ex)
                        flat_ex = self.solver.sample_to_volume(phi_ex).reshape(-1)
                        basis_scaling = np.zeros(len(self.basis.block_indices), dtype=np.float32)
                        for i, indices in enumerate(self.basis.block_indices): basis_scaling[i] = np.mean(flat_ex[indices])
                    self.angle_excitation_scalings[angle] = basis_scaling
                except Exception as e:
                    print(f"Warning: Failed to solve excitation for angle {angle}: {e}")
                    n_basis = len(self.basis.node_indices) if self.basis_mode == "fem_node" else len(self.basis.block_indices)
                    self.angle_excitation_scalings[angle] = np.ones(n_basis, dtype=np.float32)

        elif ex_pos:
            print(f"DEBUG: Initializing fixed excitation at {ex_pos} (mm)")
            # Mode 1: Fixed Source (Legacy)
            # Solve once
            phi_ex = self.solver.solve_excitation(tuple(ex_pos))
            print(f"DEBUG: Excitation field min/max: {phi_ex.min():.2e}/{phi_ex.max():.2e}, Mean: {phi_ex.mean():.2e}")
            
            self._excitation_volume = self.solver.sample_to_volume(phi_ex)
            
            # Map to basis scaling factors
            if self.basis_mode == "fem_node":
                # basis_scaling = phi_ex[self.basis.node_indices]
                # Check bounds
                if self.basis.node_indices.max() >= len(phi_ex):
                     raise ValueError("Basis node indices exceed mesh node count")
                basis_scaling = phi_ex[self.basis.node_indices]
                print(f"DEBUG: Basis scaling (FEM node) min/max: {basis_scaling.min():.2e}/{basis_scaling.max():.2e}")
            else:
                basis_scaling = np.zeros(len(self.basis.block_indices), dtype=np.float32)
                flat_ex = self._excitation_volume.reshape(-1)
                for i, indices in enumerate(self.basis.block_indices):
                    basis_scaling[i] = np.mean(flat_ex[indices])
                print(f"DEBUG: Basis scaling (Block) min/max: {basis_scaling.min():.2e}/{basis_scaling.max():.2e}")
            
            # Same scaling for all angles
            for angle in self.angles:
                self.angle_excitation_scalings[angle] = basis_scaling
                
        else:
            # Mode 2: Source follows Camera (FMT)
            # Or Mode 3: Passive (No excitation)
            
            proj_cfg = self.config.get("projection", {})
            camera_distance = float(proj_cfg.get("camera_distance", 256))
            
            # Support finite spot size and angle offset
            radius = float(self.inverse_config.get("excitation_radius", 0.0))
            azimuth_offset = float(self.inverse_config.get("source_azimuth_offset", 0.0))
            
            for angle in self.angles:
                # Calculate camera pos
                # Object rotates `angle`, Camera stays at 0 relative to World.
                # Relative to Object: Camera is at -angle.
                # Source is at -angle + offset.
                source_angle_deg = float(angle) + azimuth_offset
                
                # Note: get_camera_pos(ang, dist) returns position for "Object rotated by ang".
                # Effectively cam pos is rotated by -ang.
                # To get source at specific relative offset, we just use same function with modified angle.
                src_pos = get_camera_pos(source_angle_deg, camera_distance)
                
                # Solve excitation
                try:
                    phi_ex = self.solver.solve_excitation(src_pos, radius=radius)
                    
                    if self.basis_mode == "fem_node":
                         basis_scaling = phi_ex[self.basis.node_indices]
                    else:
                        # Store volume for first angle just in case needed for viz
                        # Only useful if single angle or similar pattern. 
                        if self._excitation_volume is None:
                             self._excitation_volume = self.solver.sample_to_volume(phi_ex)

                        flat_ex = self.solver.sample_to_volume(phi_ex).reshape(-1)
                        basis_scaling = np.zeros(len(self.basis.block_indices), dtype=np.float32)
                        for i, indices in enumerate(self.basis.block_indices):
                            basis_scaling[i] = np.mean(flat_ex[indices])
                    
                    self.angle_excitation_scalings[angle] = basis_scaling
                except Exception as e:
                    print(f"Warning: Failed to solve excitation for angle {angle}: {e}")
                    if self.basis_mode == "fem_node":
                        n_basis = len(self.basis.node_indices)
                    else:
                        n_basis = len(self.basis.block_indices)
                    self.angle_excitation_scalings[angle] = np.ones(n_basis, dtype=np.float32)

    def _build_fem_node_basis(self) -> SourceFemNodeBasis:
        src_range = self.config["src_range"]
        dx = self.solver.dx
        
        # Convert Voxel Indices to Physical Coordinates
        z0, z1 = src_range["z"][0] * dx, src_range["z"][1] * dx
        y0, y1 = src_range["y"][0] * dx, src_range["y"][1] * dx
        x0, x1 = src_range["x"][0] * dx, src_range["x"][1] * dx
        
        # Use existing solver mesh nodes (Physical Coordinates)
        nodes = self.solver.mesh.nodes # Shape (N, 3), typically XYZ
        
        # Filter nodes inside ROI
        in_x = (nodes[:, 0] >= x0) & (nodes[:, 0] < x1)
        in_y = (nodes[:, 1] >= y0) & (nodes[:, 1] < y1)
        in_z = (nodes[:, 2] >= z0) & (nodes[:, 2] < z1)
        mask = in_x & in_y & in_z
        
        indices = np.where(mask)[0]
        if indices.size == 0:
             print(f"DEBUG: ROI Physical Bounds: X[{x0:.1f}, {x1:.1f}], Y[{y0:.1f}, {y1:.1f}], Z[{z0:.1f}, {z1:.1f}]")
             print(f"DEBUG: Mesh Bounds: X[{nodes[:,0].min():.1f}, {nodes[:,0].max():.1f}], Y[{nodes[:,1].min():.1f}, {nodes[:,1].max():.1f}], Z[{nodes[:,2].min():.1f}, {nodes[:,2].max():.1f}]")
             raise ValueError("ROI contains no FEM nodes. Please check src_range and mesh density.")
             
        coords = nodes[indices]
        # coords is XYZ, convert to ZYX for consistency with block basis
        coords_zyx = coords[:, ::-1]
        
        return SourceFemNodeBasis(
            node_indices=indices,
            node_coords_zyx=coords_zyx,
            roi_bounds_zyx=((z0, z1), (y0, y1), (x0, x1)),
            type="fem_node"
        )

    def _build_source_basis(self) -> SourceBlockBasis:
        src_range = self.config["src_range"]
        z0, z1 = src_range["z"]
        y0, y1 = src_range["y"]
        x0, x1 = src_range["x"]
        roi = self.solver.volume_data[z0:z1, y0:y1, x0:x1]

        default_block = int(self.config.get("fem", {}).get("mesh", {}).get("downsample_factor", 8))
        block_shape = tuple(
            int(v)
            for v in self.inverse_config.get(
                "block_shape",
                [default_block, default_block, default_block],
            )
        )
        source_label = int(self.config.get("mask_background_value", 1))
        allowed = roi == source_label

        block_indices: list[np.ndarray] = []
        block_centers = []
        block_grid_coords = []
        roi_shape = roi.shape
        for z in range(0, roi_shape[0], block_shape[0]):
            zs = slice(z, min(z + block_shape[0], roi_shape[0]))
            for y in range(0, roi_shape[1], block_shape[1]):
                ys = slice(y, min(y + block_shape[1], roi_shape[1]))
                for x in range(0, roi_shape[2], block_shape[2]):
                    xs = slice(x, min(x + block_shape[2], roi_shape[2]))
                    block_mask = allowed[zs, ys, xs]
                    if not np.any(block_mask):
                        continue

                    local_coords = np.argwhere(block_mask)
                    global_coords = local_coords + np.array([zs.start, ys.start, xs.start], dtype=np.int32)
                    flat = np.ravel_multi_index(
                        (
                            global_coords[:, 0] + z0,
                            global_coords[:, 1] + y0,
                            global_coords[:, 2] + x0,
                        ),
                        dims=tuple(self.config["volume_shape"]),
                    )
                    block_indices.append(flat.astype(np.int64))
                    block_grid_coords.append(
                        np.array(
                            [
                                int(z // block_shape[0]),
                                int(y // block_shape[1]),
                                int(x // block_shape[2]),
                            ],
                            dtype=np.int32,
                        )
                    )
                    block_centers.append(
                        np.array(
                            [
                                z0 + (zs.start + zs.stop - 1) / 2.0,
                                y0 + (ys.start + ys.stop - 1) / 2.0,
                                x0 + (xs.start + xs.stop - 1) / 2.0,
                            ],
                            dtype=np.float64,
                        )
                    )

        if not block_indices:
            raise ValueError("source ROI 内没有可用的反演 basis 块。")

        return SourceBlockBasis(
            block_indices=block_indices,
            block_centers_zyx=np.vstack(block_centers),
            block_grid_coords=np.vstack(block_grid_coords),
            block_shape=block_shape,
            roi_shape=roi_shape,
        )

    def _mesh_nodes_to_projection_points(self, nodes_xyz: np.ndarray) -> np.ndarray:
        resize_shape = np.array(
            self.config.get("projection", {}).get("resize_shape", (256, 256, 256)),
            dtype=np.float64,
        )
        volume_shape = np.array(self.config["volume_shape"], dtype=np.float64)
        scale = resize_shape / volume_shape
        dx = max(float(self.config.get("lengthunit", 0.1)), 1e-12)

        x_idx = nodes_xyz[:, 0] / dx
        y_idx = nodes_xyz[:, 1] / dx
        z_idx = nodes_xyz[:, 2] / dx

        projector_points = np.column_stack(
            [
                z_idx * scale[0] - resize_shape[0] / 2.0 + 0.5,
                y_idx * scale[1] - resize_shape[1] / 2.0 + 0.5,
                x_idx * scale[2] - resize_shape[2] / 2.0 + 0.5,
            ]
        )
        return projector_points.astype(np.float32)

    def _build_observation(self) -> VisibleSurfaceObservation:
        observation_mode = str(
            self.inverse_config.get("observation_mode", "surface_patch_partial_current")
        ).lower()
        if observation_mode == "legacy_surface_nodes":
            return self._build_legacy_surface_node_observation()
        elif observation_mode == "visible_surface_nodes":
            return self._build_visible_node_observation()
        return self._build_visible_surface_observation()

    def _build_visible_node_observation(self) -> VisibleSurfaceObservation:
        """
        基于相机视角可见性的表面节点观测。
        对于每个角度，只观测相机可见的表面节点。
        """
        boundary_faces = np.asarray(self.solver.mesh.boundary_faces, dtype=np.int32)
        if boundary_faces.size == 0:
            raise ValueError("当前 mesh 没有外表面 boundary faces，无法建立表面观测。")

        face_nodes_xyz = self.solver.mesh.nodes[boundary_faces]
        
        # 预计算用于投影的面几何信息
        projector_face_nodes = self._mesh_nodes_to_projection_points(face_nodes_xyz.reshape(-1, 3)).reshape(-1, 3, 3)
        projector_face_centroids = projector_face_nodes.mean(axis=1)
        projector_face_normals = np.cross(
            projector_face_nodes[:, 1] - projector_face_nodes[:, 0],
            projector_face_nodes[:, 2] - projector_face_nodes[:, 0],
        )
        normal_norm = np.linalg.norm(projector_face_normals, axis=1, keepdims=True)
        valid_normal = normal_norm[:, 0] > 1e-12
        projector_face_normals[valid_normal] /= normal_norm[valid_normal]

        mesh_center = projector_face_centroids.mean(axis=0)
        outward_hint = projector_face_centroids - mesh_center
        flip_mask = np.sum(projector_face_normals * outward_hint, axis=1) < 0.0
        projector_face_normals[flip_mask] *= -1.0

        proj_cfg = self.config.get("projection", {})
        detector_resolution = tuple(proj_cfg.get("detector_resolution", (256, 256)))
        detector_size = tuple(proj_cfg.get("detector_size", detector_resolution))
        camera_distance = float(proj_cfg.get("camera_distance", 256))

        row_slices_by_angle: dict[int, tuple[int, int]] = {}
        patch_coords_by_angle: dict[int, np.ndarray] = {}
        patch_centroids_by_angle: dict[int, np.ndarray] = {}
        camera_depth_by_angle: dict[int, np.ndarray] = {}
        patch_face_counts_by_angle: dict[int, np.ndarray] = {}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        row_cursor = 0

        for angle in self.angles:
            # 投影并计算可见性
            projections, depths = project_points_to_camera_batch_matrix(
                projector_face_centroids,
                int(angle),
                camera_distance,
                np.asarray(detector_size, dtype=np.float32),
            )
            rotated_normals = projector_face_normals @ rotation_matrix_y(int(angle)).T
            
            visible_face_indices = []
            for face_idx in range(len(boundary_faces)):
                in_detector = float(depths[face_idx, 1]) > 0.5
                front_facing = float(rotated_normals[face_idx, 2]) > 1e-6
                if in_detector and front_facing:
                    visible_face_indices.append(face_idx)
            
            if not visible_face_indices:
                print(f"警告: 角度 {angle} 未找到可见表面 patch。")
                row_slices_by_angle[angle] = (row_cursor, row_cursor)
                patch_coords_by_angle[angle] = np.zeros((0, 2), dtype=np.int32)
                patch_centroids_by_angle[angle] = np.zeros((0, 3), dtype=np.float64)
                camera_depth_by_angle[angle] = np.zeros(0, dtype=np.float32)
                patch_face_counts_by_angle[angle] = np.zeros(0, dtype=np.int32)
                continue

            visible_face_indices = np.array(visible_face_indices, dtype=np.int32)
            
            # 获取这些可见面包含的所有唯一节点
            visible_nodes = np.unique(boundary_faces[visible_face_indices])
            
            # 添加到观测矩阵
            start_row = row_cursor
            for node_idx in visible_nodes:
                rows.append(row_cursor)
                cols.append(int(node_idx))
                data.append(1.0) # Partial current at node is usually just the nodal value * weight? Or just nodal value. Let's use 1.0 (identity map)
                row_cursor += 1
            
            n_nodes = len(visible_nodes)
            row_slices_by_angle[angle] = (start_row, row_cursor)
            
            # Dummy metadata for compatibility
            patch_coords_by_angle[angle] = np.zeros((n_nodes, 2), dtype=np.int32)
            patch_centroids_by_angle[angle] = self.solver.mesh.nodes[visible_nodes].astype(np.float64)
            camera_depth_by_angle[angle] = np.zeros(n_nodes, dtype=np.float32) # Not easily defined for node
            patch_face_counts_by_angle[angle] = np.ones(n_nodes, dtype=np.int32)

        n_mesh_nodes = len(self.solver.mesh.nodes)
        observation_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(row_cursor, n_mesh_nodes),
            dtype=np.float64,
        ).tocsr()

        return VisibleSurfaceObservation(
            angles=self.angles,
            row_slices_by_angle=row_slices_by_angle,
            patch_coords_by_angle=patch_coords_by_angle,
            patch_centroids_by_angle=patch_centroids_by_angle,
            camera_depth_by_angle=camera_depth_by_angle,
            patch_face_counts_by_angle=patch_face_counts_by_angle,
            observation_matrix=observation_matrix,
            observation_type="visible_surface_nodes",
        )

    def _build_legacy_surface_node_observation(self) -> VisibleSurfaceObservation:
        boundary_ids = np.where(self.solver.mesh.bndvtx > 0)[0].astype(np.int32)
        if boundary_ids.size == 0:
            raise ValueError("当前 mesh 没有外表面 boundary nodes，无法建立 legacy 观测。")

        boundary_nodes = self.solver.mesh.nodes[boundary_ids]
        z_min = self.inverse_config.get("surface_z_min", None)
        z_max = self.inverse_config.get("surface_z_max", None)
        mask = np.ones(boundary_ids.shape[0], dtype=bool)
        if z_min is not None:
            mask &= boundary_nodes[:, 2] >= float(z_min)
        if z_max is not None:
            mask &= boundary_nodes[:, 2] <= float(z_max)
        kept_ids = boundary_ids[mask]
        kept_nodes = boundary_nodes[mask]
        if kept_ids.size == 0:
            raise ValueError("legacy_surface_nodes 观测裁剪后为空，请检查 surface_z_min/max。")

        row_indices = np.arange(kept_ids.size, dtype=np.int32)
        observation_matrix = coo_matrix(
            (
                np.ones(kept_ids.size, dtype=np.float64),
                (row_indices, kept_ids),
            ),
            shape=(kept_ids.size, len(self.solver.mesh.nodes)),
            dtype=np.float64,
        ).tocsr()
        default_angle = 0
        return VisibleSurfaceObservation(
            angles=[default_angle],
            row_slices_by_angle={default_angle: (0, kept_ids.size)},
            patch_coords_by_angle={default_angle: row_indices[:, None]},
            patch_centroids_by_angle={default_angle: kept_nodes.astype(np.float64)},
            camera_depth_by_angle={default_angle: kept_nodes[:, 2].astype(np.float32)},
            patch_face_counts_by_angle={default_angle: np.ones(kept_ids.size, dtype=np.int32)},
            observation_matrix=observation_matrix,
            observation_type="legacy_surface_nodes",
        )

    def _build_visible_surface_observation(self) -> VisibleSurfaceObservation:
        boundary_faces = np.asarray(self.solver.mesh.boundary_faces, dtype=np.int32)
        if boundary_faces.size == 0:
            raise ValueError("当前 mesh 没有外表面 boundary faces，无法建立表面观测。")

        face_nodes_xyz = self.solver.mesh.nodes[boundary_faces]
        face_centroids_xyz = face_nodes_xyz.mean(axis=1)
        face_v1_xyz = face_nodes_xyz[:, 1] - face_nodes_xyz[:, 0]
        face_v2_xyz = face_nodes_xyz[:, 2] - face_nodes_xyz[:, 0]
        face_areas = 0.5 * np.linalg.norm(np.cross(face_v1_xyz, face_v2_xyz), axis=1)

        projector_face_nodes = self._mesh_nodes_to_projection_points(face_nodes_xyz.reshape(-1, 3)).reshape(-1, 3, 3)
        projector_face_centroids = projector_face_nodes.mean(axis=1)
        projector_face_normals = np.cross(
            projector_face_nodes[:, 1] - projector_face_nodes[:, 0],
            projector_face_nodes[:, 2] - projector_face_nodes[:, 0],
        )
        normal_norm = np.linalg.norm(projector_face_normals, axis=1, keepdims=True)
        valid_normal = normal_norm[:, 0] > 1e-12
        projector_face_normals[valid_normal] /= normal_norm[valid_normal]

        mesh_center = projector_face_centroids.mean(axis=0)
        outward_hint = projector_face_centroids - mesh_center
        flip_mask = np.sum(projector_face_normals * outward_hint, axis=1) < 0.0
        projector_face_normals[flip_mask] *= -1.0

        proj_cfg = self.config.get("projection", {})
        detector_resolution = tuple(proj_cfg.get("detector_resolution", (256, 256)))
        detector_size = tuple(proj_cfg.get("detector_size", detector_resolution))
        camera_distance = float(proj_cfg.get("camera_distance", 256))
        patch_size = max(int(self.inverse_config.get("patch_size_pixels", 8)), 1)
        normalize_patch_by_area = bool(self.inverse_config.get("normalize_patch_by_area", True))
        width_pixels, height_pixels = detector_resolution
        width_phys, height_phys = detector_size
        pixel_to_phys_x = width_phys / width_pixels
        pixel_to_phys_y = height_phys / height_pixels

        row_slices_by_angle: dict[int, tuple[int, int]] = {}
        patch_coords_by_angle: dict[int, np.ndarray] = {}
        patch_centroids_by_angle: dict[int, np.ndarray] = {}
        camera_depth_by_angle: dict[int, np.ndarray] = {}
        patch_face_counts_by_angle: dict[int, np.ndarray] = {}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        row_cursor = 0

        for angle in self.angles:
            projections, depths = project_points_to_camera_batch_matrix(
                projector_face_centroids,
                int(angle),
                camera_distance,
                np.asarray(detector_size, dtype=np.float32),
            )
            rotated_normals = projector_face_normals @ rotation_matrix_y(int(angle)).T
            patch_to_faces: dict[tuple[int, int], list[int]] = {}

            for face_idx in range(len(boundary_faces)):
                if face_areas[face_idx] <= 1e-12:
                    continue
                u = float(projections[face_idx, 0])
                v = float(projections[face_idx, 1])
                depth = float(depths[face_idx, 0])
                in_detector = float(depths[face_idx, 1]) > 0.5
                front_facing = float(rotated_normals[face_idx, 2]) > 1e-6
                if not in_detector or not front_facing:
                    continue

                pixel_u = int((u + width_phys / 2.0) / pixel_to_phys_x)
                pixel_v = int((v + height_phys / 2.0) / pixel_to_phys_y)
                if not (0 <= pixel_u < width_pixels and 0 <= pixel_v < height_pixels):
                    continue

                key = (pixel_v // patch_size, pixel_u // patch_size)
                patch_to_faces.setdefault(key, []).append(face_idx)

            if not patch_to_faces:
                raise ValueError(f"角度 {angle} 未找到可见表面 patch，无法建立观测。")

            start_row = row_cursor
            ordered_patches = sorted(patch_to_faces.keys())
            angle_patch_coords: list[list[int]] = []
            angle_patch_centroids: list[np.ndarray] = []
            angle_patch_depths: list[float] = []
            angle_patch_counts: list[int] = []

            for patch_key in ordered_patches:
                face_indices = np.asarray(patch_to_faces[patch_key], dtype=np.int32)
                total_area = float(np.sum(face_areas[face_indices]))
                if total_area <= 1e-12:
                    continue

                node_weights: dict[int, float] = {}
                for face_idx in face_indices:
                    nodes = boundary_faces[face_idx]
                    local_weights = face_areas[face_idx] * self.solver.mesh.ksi[nodes] / 3.0
                    for node_id, weight in zip(nodes, local_weights):
                        node_weights[int(node_id)] = node_weights.get(int(node_id), 0.0) + float(weight)

                normalizer = total_area if normalize_patch_by_area else 1.0
                for node_id, weight in node_weights.items():
                    rows.append(row_cursor)
                    cols.append(node_id)
                    data.append(weight / normalizer)

                area_weights = face_areas[face_indices].astype(np.float64)
                weighted_centroid = np.average(face_centroids_xyz[face_indices], axis=0, weights=area_weights)
                weighted_depth = float(np.average(depths[face_indices, 0], weights=area_weights))
                angle_patch_coords.append([int(patch_key[0]), int(patch_key[1])])
                angle_patch_centroids.append(weighted_centroid.astype(np.float64))
                angle_patch_depths.append(weighted_depth)
                angle_patch_counts.append(int(face_indices.size))
                row_cursor += 1

            if row_cursor == start_row:
                raise ValueError(f"角度 {angle} 的可见表面 patch 全部为空，无法建立观测。")

            row_slices_by_angle[angle] = (start_row, row_cursor)
            patch_coords_by_angle[angle] = np.asarray(angle_patch_coords, dtype=np.int32)
            patch_centroids_by_angle[angle] = np.vstack(angle_patch_centroids).astype(np.float64)
            camera_depth_by_angle[angle] = np.asarray(angle_patch_depths, dtype=np.float32)
            patch_face_counts_by_angle[angle] = np.asarray(angle_patch_counts, dtype=np.int32)

        observation_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(row_cursor, len(self.solver.mesh.nodes)),
            dtype=np.float64,
        ).tocsr()

        return VisibleSurfaceObservation(
            angles=self.angles,
            row_slices_by_angle=row_slices_by_angle,
            patch_coords_by_angle=patch_coords_by_angle,
            patch_centroids_by_angle=patch_centroids_by_angle,
            camera_depth_by_angle=camera_depth_by_angle,
            patch_face_counts_by_angle=patch_face_counts_by_angle,
            observation_matrix=observation_matrix,
        )

    def _build_rhs_matrix(self) -> np.ndarray:
        if self._rhs_matrix is not None:
            return self._rhs_matrix

        n_nodes = len(self.solver.mesh.nodes)
        
        if self.basis_mode == "fem_node":
            n_basis = len(self.basis.node_indices)
            rhs_matrix = np.zeros((n_nodes, n_basis), dtype=np.float64)
            # Each column i corresponds to node basis[i]
            # Set rhs[node_idx, i] = 1.0 (or normalized by volume?)
            # Partial current basis usually normalized by element volume if integral.
            # But point source is Dirac delta.
            # Usually in FEM: f_i = 1 for point source at node i.
            
            # Simple identity map for active nodes
            # Rows: All Nodes. Cols: Active Basis Nodes.
            for i, node_idx in enumerate(self.basis.node_indices):
                rhs_matrix[node_idx, i] = 1.0
                
        else:
            n_basis = len(self.basis.block_indices)
            rhs_matrix = np.zeros((n_nodes, n_basis), dtype=np.float64)
            source_volume = np.zeros(tuple(self.config["volume_shape"]), dtype=np.float32)
            flat = source_volume.reshape(-1)

            for idx, block in enumerate(self.basis.block_indices):
                flat.fill(0.0)
                flat[block] = 1.0
                rhs_matrix[:, idx] = self.solver.build_source_vector(source_volume, normalize=False)

        self._rhs_matrix = rhs_matrix
        return rhs_matrix

    def _observe_solution(self, nodal_solution: Union[np.ndarray, Dict[int, np.ndarray]]) -> np.ndarray:
        if isinstance(nodal_solution, dict):
            # Epi-mode: nodal_solution is {angle: array}
            # We must apply observation matrix per angle slice
            total_rows = self.observation.observation_matrix.shape[0]
            values = np.zeros(total_rows, dtype=np.float64)
            
            for angle, (start_row, end_row) in self.observation.row_slices_by_angle.items():
                if angle in nodal_solution:
                    obs_slice = self.observation.observation_matrix[start_row:end_row, :]
                    sol = np.asarray(nodal_solution[angle], dtype=np.float64).reshape(-1)
                    values[start_row:end_row] = obs_slice @ sol
            return values
        else:
            values = self.observation.observation_matrix @ np.asarray(nodal_solution, dtype=np.float64)
            return np.asarray(values, dtype=np.float64).reshape(-1)

    def _observe_solution_batch(self, nodal_solutions: np.ndarray) -> np.ndarray:
        values = self.observation.observation_matrix @ np.asarray(nodal_solutions, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def _get_gpu_options(self) -> dict:
        gpu_cfg = self.inverse_config.get("gpu", {})
        return {
            "enabled": bool(gpu_cfg.get("enabled", False)),
            "device_index": int(gpu_cfg.get("device_index", 0)),
        }

    def _build_graph_edges(self) -> np.ndarray:
        if self._graph_edges is not None:
            return self._graph_edges

        if self.basis_mode == "fem_node":
            # Extract edges from Mesh Elements (Tetrahedra)
            # Elements: (Ne, 4)
            elements = self.solver.mesh.elements
            
            # Map global node indices to local basis indices
            # global_idx -> local_idx
            global_to_local = {
                global_idx: local_idx 
                for local_idx, global_idx in enumerate(self.basis.node_indices)
            }
            
            edges_set = set()
            
            # Helper to add edge
            def add_edge(n1, n2):
                if n1 in global_to_local and n2 in global_to_local:
                    l1, l2 = global_to_local[n1], global_to_local[n2]
                    if l1 < l2:
                        edges_set.add((l1, l2))
                    elif l2 < l1:
                        edges_set.add((l2, l1))

            # Loop elements (expensive in python? ~20k elements is fine)
            # Or use numpy vectorization
            # (Ne, 4)
            # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            pairs = [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3),
                (2, 3)
            ]
            
            node_map = np.full(len(self.solver.mesh.nodes), -1, dtype=np.int32)
            node_map[self.basis.node_indices] = np.arange(len(self.basis.node_indices))
            
            edges_list = []
            for i, j in pairs:
                n1 = self.solver.mesh.elements[:, i]
                n2 = self.solver.mesh.elements[:, j]
                l1 = node_map[n1]
                l2 = node_map[n2]
                
                # Keep valid pairs (both nodes in basis)
                mask = (l1 >= 0) & (l2 >= 0)
                if np.any(mask):
                    valid_l1 = l1[mask]
                    valid_l2 = l2[mask]
                    
                    # Sort pairs to ensure (min, max)
                    stack = np.column_stack([np.minimum(valid_l1, valid_l2), np.maximum(valid_l1, valid_l2)])
                    edges_list.append(stack)
            
            if edges_list:
                all_edges = np.vstack(edges_list)
                # Unique rows
                self._graph_edges = np.unique(all_edges, axis=0)
            else:
                self._graph_edges = np.zeros((0, 2), dtype=np.int32)
                
            return self._graph_edges

        coord_to_idx = {
            tuple(coord.tolist()): idx for idx, coord in enumerate(self.basis.block_grid_coords)
        }
        neighbor_offsets = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        edges: list[tuple[int, int]] = []
        for idx, coord in enumerate(self.basis.block_grid_coords):
            for offset in neighbor_offsets:
                neigh = tuple((coord + offset).tolist())
                jdx = coord_to_idx.get(neigh)
                if jdx is not None:
                    edges.append((idx, jdx))

        if not edges:
            self._graph_edges = np.zeros((0, 2), dtype=np.int32)
        else:
            self._graph_edges = np.asarray(edges, dtype=np.int32)
        return self._graph_edges

    def _build_graph_laplacian(self) -> np.ndarray:
        if self._graph_laplacian is not None:
            return self._graph_laplacian

        n_basis = len(self.basis.node_indices) if self.basis_mode == "fem_node" else len(self.basis.block_indices)
        edges = self._build_graph_edges()
        lap = np.zeros((n_basis, n_basis), dtype=np.float64)
        for i, j in edges:
            lap[i, i] += 1.0
            lap[j, j] += 1.0
            lap[i, j] -= 1.0
            lap[j, i] -= 1.0
        self._graph_laplacian = lap
        return self._graph_laplacian

    def _build_tv_difference(self) -> np.ndarray:
        if self._tv_difference is not None:
            return self._tv_difference

        edges = self._build_graph_edges()
        n_basis = len(self.basis.node_indices) if self.basis_mode == "fem_node" else len(self.basis.block_indices)
        diff = np.zeros((len(edges), n_basis), dtype=np.float64)
        for row, (i, j) in enumerate(edges):
            diff[row, i] = 1.0
            diff[row, j] = -1.0
        self._tv_difference = diff
        return self._tv_difference

    def _build_support_mask(self, coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.size == 0:
            return np.zeros(0, dtype=bool)
        alpha = float(self.inverse_config.get("legacy_refine_alpha", 0.3))
        if np.max(coeff) <= 0:
            mask = np.zeros(coeff.size, dtype=bool)
            mask[int(np.argmax(coeff))] = True
            return mask
        mask = coeff >= alpha * float(np.max(coeff))
        if not np.any(mask):
            mask[int(np.argmax(coeff))] = True

        hops = max(int(self.inverse_config.get("legacy_refine_expand_hops", 1)), 0)
        if hops <= 0:
            return mask

        edges = self._build_graph_edges()
        if edges.size == 0:
            return mask
        active = set(np.where(mask)[0].tolist())
        frontier = set(active)
        for _ in range(hops):
            next_frontier: set[int] = set()
            for i, j in edges:
                if i in frontier and j not in active:
                    next_frontier.add(int(j))
                if j in frontier and i not in active:
                    next_frontier.add(int(i))
            active |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        expanded = np.zeros(coeff.size, dtype=bool)
        expanded[list(active)] = True
        return expanded

    def _dense_gram(self, operator: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gpu_opt = self._get_gpu_options()
        torch = _try_import_torch()
        if not gpu_opt["enabled"] or torch is None or not torch.cuda.is_available():
            return operator.T @ operator, operator.T @ measurement

        device = torch.device(f"cuda:{gpu_opt['device_index']}")
        operator_t = torch.as_tensor(operator, device=device, dtype=torch.float32)
        measurement_t = torch.as_tensor(measurement, device=device, dtype=torch.float32)
        hth = (operator_t.T @ operator_t).detach().cpu().numpy().astype(np.float64)
        hty = (operator_t.T @ measurement_t).detach().cpu().numpy().astype(np.float64)
        del operator_t, measurement_t
        torch.cuda.empty_cache()
        return hth, hty

    def build_operator(self) -> np.ndarray:
        if self._operator_cache is not None:
            return self._operator_cache

        rhs_matrix = self._build_rhs_matrix()
        n_basis = rhs_matrix.shape[1]
        n_obs = self.observation.n_measurements
        operator = np.zeros((n_obs, n_basis), dtype=np.float32)
        batch_size = int(self.inverse_config.get("h_batch_size", 8))

        # Check if we have angle-dependent excitation
        has_angle_excitation = hasattr(self, "angle_excitation_scalings") and self.angle_excitation_scalings

        for start in range(0, n_basis, batch_size):
            end = min(start + batch_size, n_basis)
            # Solve Green's functions (Fluence at all nodes for these basis sources)
            # solutions: [N_nodes, batch_size]
            solutions = self.solver.solve_multiple_rhs(rhs_matrix[:, start:end])
            
            # Map nodal fluence to observation vector
            # This applies the observation matrix P (selection of nodes/patches)
            # obs_batch: [N_obs, batch_size]
            obs_batch = self._observe_solution_batch(solutions).astype(np.float32)
            
            # Apply Excitation Scaling
            if has_angle_excitation:
                # Iterate angles and scale rows corresponding to that angle
                for angle, row_slice_indices in self.observation.row_slices_by_angle.items():
                    start_row, end_row = row_slice_indices
                    if start_row >= end_row:
                        continue
                        
                    # Get scaling vector for this angle (shape: [n_basis])
                    # Slice it for current batch
                    scaling = self.angle_excitation_scalings[angle][start:end]
                    
                    # Apply scaling: obs_batch[rows, cols] *= scaling[cols]
                    # Broadcast: (n_rows, n_cols) * (1, n_cols)
                    obs_batch[start_row:end_row, :] *= scaling[None, :]

            operator[:, start:end] = obs_batch

        self._operator_cache = operator
        return operator

    def compute_rotated_source_pos(self, angle: float) -> tuple[float, float, float]:
        """计算围绕体积中心旋转后的光源位置 (Epi-mode)。"""
        base_pos = np.array(self.config["excitation_source"]["position"], dtype=np.float64)
        volume_shape = np.array(self.config["volume_shape"], dtype=np.float64)
        dx = float(self.config.get("lengthunit", 0.1))
        center = (volume_shape[::-1] * dx) / 2.0
        
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        v = base_pos - center
        v_rot = R @ v
        new_pos = center + v_rot
        return tuple(new_pos)

    def solve_source(self, source_pattern: np.ndarray) -> np.ndarray | dict[int, np.ndarray]:
        """
        求解给定源分布的发射光场。
        支持 Epi-mode: 返回 {angle: nodal_fluence} 字典。
        """
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        
        if epi_mode:
            results = {}
            # Reconstruct tumor/source pattern vector ONCE
            # source_pattern is volume.
            tumor_rhs_raw = self.solver.build_source_vector(source_pattern, normalize=False)
            
            for angle in self.angles:
                # Compute Excitation
                ex_pos = self.compute_rotated_source_pos(angle)
                phi_ex = self.solver.solve_excitation(ex_pos, radius=0.5)
                
                # Effective Source
                effective_rhs = tumor_rhs_raw * phi_ex
                
                # Solve Emission
                nodal = self.solver.solve_from_rhs(effective_rhs)
                results[angle] = np.asarray(nodal, dtype=np.float64)
            return results

        if hasattr(self, "_excitation_volume") and self._excitation_volume is not None:
             effective_source = source_pattern * self._excitation_volume
        else:
             effective_source = source_pattern

        nodal = self.solver.solve(effective_source, normalize=False)
        return np.asarray(nodal, dtype=np.float64)


    def reconstruct_l2(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
    ) -> np.ndarray:
        hth, hty = self._dense_gram(operator, measurement)
        reg = np.eye(hth.shape[0], dtype=np.float64) * float(lambda_value)
        coeff = np.linalg.solve(hth + reg, hty)
        return np.maximum(coeff, 0.0)

    def reconstruct_laplacian(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
    ) -> np.ndarray:
        hth, hty = self._dense_gram(operator, measurement)
        lap = self._build_graph_laplacian()
        coeff = np.linalg.solve(hth + float(lambda_value) * lap + 1e-8 * np.eye(hth.shape[0]), hty)
        return np.maximum(coeff, 0.0)

    def reconstruct_l1_fista(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        max_iter: int = 200,
    ) -> np.ndarray:
        hth, hty = self._dense_gram(operator, measurement)
        lipschitz = float(np.max(np.linalg.eigvalsh(hth)) + 1e-8)
        x = np.zeros(operator.shape[1], dtype=np.float64)
        z = x.copy()
        t = 1.0

        for _ in range(max_iter):
            grad = hth @ z - hty
            x_new = _soft_threshold_positive(z - grad / lipschitz, lambda_value / lipschitz)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x = x_new
            t = t_new
        return x

    def _precondition_operator(self, operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col_norms = np.linalg.norm(operator, axis=0)
        scale = np.maximum(col_norms, 1e-12)
        return operator / scale[None, :], scale

    def reconstruct_l1_ls(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        max_iter: int = 200,
        err: float = 1e-10,
    ) -> np.ndarray:
        """
        L1-Regularized Least Squares (L1-LS) solver using FISTA with backtracking line search.
        Includes column normalization and measurement normalization for improved conditioning.
        """
        operator = np.asarray(operator, dtype=np.float64)
        measurement = np.asarray(measurement, dtype=np.float64).reshape(-1)
        
        # Measurement normalization
        m_norm = np.linalg.norm(measurement)
        if m_norm < 1e-12:
            return np.zeros(operator.shape[1])
        measurement_scaled = measurement / m_norm
        
        # Column normalization
        operator_normed, scale = self._precondition_operator(operator)
        
        # Scale operator by 1/m_norm effectively? 
        # No, we solve A_n * w = m_n.
        # A_n * (x * scale / m_norm) = m / m_norm.
        # So x = w * m_norm / scale.
        
        # However, we must ensure lambda is consistent.
        # Objective: 0.5 || A_n w - m_n ||^2 + lambda ||w||_1
        # If m_n is unit norm, w is roughly unit norm.
        # So lambda relative to 1 makes sense.

        x = np.zeros(operator.shape[1], dtype=np.float64)
        x_prev = x.copy()
        t_prev = 1.0
        t_cur = 1.0
        lipschitz = 1.0
        beta = 1.5
        gram = operator_normed.T @ operator_normed
        hty = operator_normed.T @ measurement_scaled
        
        # Initial objective
        obj_prev = 0.5 * np.linalg.norm(measurement_scaled - operator_normed @ x) ** 2 + lambda_value * np.linalg.norm(x, 1)

        for _ in range(max_iter):
            y = x + ((t_prev - 1.0) / max(t_cur, 1e-12)) * (x - x_prev)
            grad = gram @ y - hty
            
            # Backtracking line search
            while True:
                trial = _soft_threshold_positive(y - grad / lipschitz, lambda_value / lipschitz)
                lhs = 0.5 * np.linalg.norm(measurement_scaled - operator_normed @ trial) ** 2
                rhs = (
                    0.5 * np.linalg.norm(measurement_scaled - operator_normed @ y) ** 2
                    + (trial - y) @ grad
                    + 0.5 * lipschitz * np.linalg.norm(trial - y) ** 2
                )
                if lhs <= rhs + 1e-12:
                    break
                lipschitz *= beta

            obj = lhs + lambda_value * np.linalg.norm(trial, 1)
            rel = abs(obj - obj_prev) / max(abs(obj_prev), 1e-12)
            
            x_prev = x
            x = trial
            obj_prev = obj
            
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_cur * t_cur))
            t_prev, t_cur = t_cur, t_next
            
            if rel <= err:
                break

        # Rescale coefficients
        # x = w. real_x = w * m_norm / scale
        x = np.maximum(x * m_norm / scale, 0.0)
        return x

    def reconstruct_tv(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        n_iter: int = 15,
        eps: float = 1e-3,
        diff_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Total Variation (TV) solver using Iteratively Reweighted Least Squares (IRLS).
        Includes column normalization and measurement normalization for improved conditioning.
        """
        measurement = np.asarray(measurement, dtype=np.float64).reshape(-1)
        
        # Measurement normalization
        m_norm = np.linalg.norm(measurement)
        if m_norm < 1e-12:
             return np.zeros(operator.shape[1])
        measurement_scaled = measurement / m_norm

        # Column normalization
        operator_normed, scale = self._precondition_operator(operator)
        hth, hty = self._dense_gram(operator_normed, measurement_scaled)
        
        diff = self._build_tv_difference() if diff_matrix is None else diff_matrix
        
        if diff.shape[0] == 0:
            # Fallback to L2, but handle normalization manually since L2 helper doesn't do it?
            # Actually L2 is scale invariant for linear solve, but regularization lambda needs scaling.
            # Just call reconstruct_l2 with scaled measurement?
            # reconstruct_l2 doesn't have normalization logic.
            # Let's just implement quick L2 here.
            reg = np.eye(hth.shape[0], dtype=np.float64) * float(lambda_value)
            w = np.linalg.solve(hth + reg, hty)
            return np.maximum(w * m_norm / scale, 0.0)

        # Scale diff matrix for normalized variable w = scale * x / m_norm?
        # No, variable is w. x = w * m_norm / scale.
        # TV(x) = |D x|.
        # |D (w * m_norm / scale)| = |(D * m_norm / scale) w|.
        # So effective D for w is D * (m_norm / scale).
        diff_scaled = diff * (m_norm / scale)[None, :]
        
        # But wait, lambda is weighting |D x|.
        # Objective: 0.5 ||A x - y||^2 + lambda TV(x)
        # Scaled: 0.5 ||A_n w - y_n||^2 * m_norm^2 + lambda TV(w * m_norm / scale)
        # Divide by m_norm^2:
        # 0.5 ||A_n w - y_n||^2 + (lambda / m_norm^2) * |(D * m_norm / scale) w|
        # Let lambda_eff = lambda / m_norm. 
        # |(D/scale) w|.
        # This seems complicated.
        
        # Alternative: Just normalize measurement and use lambda relative to normalized problem.
        # Objective: 0.5 ||A_n w - y_n||^2 + lambda_value * TV_norm(w)
        # Where TV_norm(w) is some TV on w.
        # But w are coefficients of normalized columns. They don't have spatial smoothness directly?
        # Actually they do, if columns are spatially localized.
        # But scale factors vary with depth.
        # So D should act on x, not w.
        # x = w / scale * m_norm.
        # So we want to regularize D(w/scale).
        # And we solve for w.
        
        # Simplified approach:
        # 1. Solve unnormalized L2 first to get initial w.
        # 2. Iterate.
        
        # Let's stick to current implementation but fix the scale issue.
        # If we use scaled measurement y_n and A_n.
        # We solve for w.
        # We want to minimize: 0.5 ||A_n w - y_n||^2 + lambda * ||D (w/scale)||_1 / m_norm ?
        # No, let's just assume lambda is passed as "relative to energy".
        # So we minimize: 0.5 ||A_n w - y_n||^2 + lambda * ||D_weighted w||_1
        # where D_weighted accounts for scale.
        
        # Scale diff matrix for normalized variable w
        # x = w / scale * m_norm
        # We want to regularize TV(x) = TV(w/scale * m_norm) = m_norm * TV(w/scale)
        # Objective: 0.5 ||A_n w - y_n||^2 + lambda/m_norm * TV(x)
        #            0.5 ||A_n w - y_n||^2 + lambda * TV(w/scale)
        # So effective Diff is D @ diag(1/scale)
        
        diff_scaled = diff * (1.0 / scale)[None, :]

        # Initialize w using L2 (regularized)
        reg_l2 = np.eye(hth.shape[0], dtype=np.float64) * float(lambda_value)
        w = np.linalg.solve(hth + reg_l2, hty)
        w = np.maximum(w, 0.0)

        eye = np.eye(hth.shape[0], dtype=np.float64) * 1e-8
        
        for _ in range(n_iter):
            # grad here is TV argument: D * (w/scale)
            grad = diff_scaled @ w
            
            # Weight is 1 / |grad|
            # Note: We are minimizing L1 norm, which is sum(sqrt(grad^2 + eps^2))
            # Its derivative is grad / sqrt(...)
            # In IRLS form (linear system), we write this as weights * grad
            # weights = 1 / sqrt(grad^2 + eps^2)
            
            # However, we need to be careful with scaling.
            # Are we regularizing ||D (w/scale)||_1 ?
            # Yes, objective is ... + lambda * sum(|grad_i|)
            # Derivative w.r.t w is ... + lambda * sum(sign(grad_i) * d(grad_i)/dw)
            # grad = D_scaled @ w
            # d(grad)/dw = D_scaled
            # So term is D_scaled^T @ (sign(grad))
            # IRLS approximation: sign(grad) approx grad / |grad|
            # So term is D_scaled^T @ (grad / |grad|)
            #             = D_scaled^T @ (1/|grad| * D_scaled @ w)
            # So weights matrix W is diag(1/|grad|)
            
            weights = 1.0 / np.sqrt(grad * grad + eps * eps)
            
            # Reg matrix: D_scaled^T @ W @ D_scaled
            reg = diff_scaled.T @ (weights[:, None] * diff_scaled)
            
            # Solve (H^T H + lambda * Reg) w = H^T y
            # lambda_value is already scaled relative to data term in user config usually?
            # If user provides lambda for unnormalized problem:
            # 0.5 ||A x - y||^2 + lambda TV(x)
            # = 0.5 * m_norm^2 * ||A_n w - y_n||^2 + lambda * m_norm * TV(w/scale)
            # Divide by m_norm^2:
            # 0.5 ||A_n w - y_n||^2 + (lambda / m_norm) * TV(w/scale)
            # So effective lambda is lambda_value / m_norm
            
            lambda_eff = float(lambda_value) / m_norm
            
            lhs = hth + lambda_eff * reg + eye
            w = np.linalg.solve(lhs, hty)
            w = np.maximum(w, 0.0)
            
        x = w / scale * m_norm
        return x
        return x

    def reconstruct_l1_fista_legacy(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        max_iter: int = 200,
        err: float = 1e-10,
    ) -> np.ndarray:
        # Redirect to the clean implementation
        return self.reconstruct_l1_ls(measurement, operator, lambda_value, max_iter, err)

    def reconstruct_tv_irls(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        n_iter: int = 15,
        eps: float = 1e-3,
    ) -> np.ndarray:
        # Redirect to the clean implementation
        return self.reconstruct_tv(measurement, operator, lambda_value, n_iter, eps)

    def coefficients_to_volume(self, coeff: np.ndarray) -> np.ndarray:
        if self.basis_mode == "fem_node":
            # Map coeffs (on Active Nodes) to Full Mesh Nodes -> Volume Interpolation
            full_nodal = np.zeros(len(self.solver.mesh.nodes), dtype=np.float64)
            full_nodal[self.basis.node_indices] = coeff
            
            # Use solver's interpolation (linear barycentric)
            return self.solver.sample_to_volume(full_nodal)

        volume = np.zeros(tuple(self.config["volume_shape"]), dtype=np.float32)
        flat = volume.reshape(-1)
        for value, indices in zip(coeff, self.basis.block_indices):
            if value <= 0:
                continue
            flat[indices] += float(value)
        return volume

    def reconstruct_legacy_l1(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
    ) -> np.ndarray:
        lambda_value = float(self.inverse_config.get("legacy_l1_lambda", self.inverse_config.get("l1_lambda", 1e-4)))
        max_iter = int(self.inverse_config.get("legacy_l1_max_iter", self.inverse_config.get("l1_max_iter", 200)))
        err = float(self.inverse_config.get("legacy_l1_err", 1e-10))
        coeff = self.reconstruct_l1_fista_legacy(
            measurement,
            operator,
            lambda_value=lambda_value,
            max_iter=max_iter,
            err=err,
        )
        if not bool(self.inverse_config.get("legacy_refine_enabled", True)):
            return coeff

        support_mask = self._build_support_mask(coeff)
        if np.all(support_mask):
            return coeff
        refined = np.zeros_like(coeff)
        refined_coeff = self.reconstruct_l1_fista_legacy(
            measurement,
            operator[:, support_mask],
            lambda_value=lambda_value,
            max_iter=max_iter,
            err=err,
        )
        refined[support_mask] = refined_coeff
        return refined

    def reconstruct_legacy_tv(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
    ) -> np.ndarray:
        lambda_value = float(self.inverse_config.get("tv_lambda", 5e-2))
        n_iter = int(self.inverse_config.get("tv_irls_iter", 15))
        eps = float(self.inverse_config.get("tv_eps", 1e-3))

        # Initial TV (Full Domain)
        coeff = self.reconstruct_tv(
            measurement,
            operator,
            lambda_value=lambda_value,
            n_iter=n_iter,
            eps=eps,
        )

        if not bool(self.inverse_config.get("legacy_refine_enabled", True)):
            return coeff

        support_mask = self._build_support_mask(coeff)
        if np.all(support_mask):
            return coeff

        # Refined TV (Support Domain)
        # We need to project the difference matrix to the support subset.
        full_diff = self._build_tv_difference()
        # D_sub = D[:, mask] ensures D_sub @ x_sub == D @ x_full (where x_full is 0 outside mask)
        diff_sub = full_diff[:, support_mask]
        
        # Optimization: Prune edges that don't touch the support
        active_edges = np.abs(diff_sub).sum(axis=1) > 1e-12
        diff_sub = diff_sub[active_edges]

        refined = np.zeros_like(coeff)
        refined_coeff = self.reconstruct_tv(
            measurement,
            operator[:, support_mask],
            lambda_value=lambda_value,
            n_iter=n_iter,
            eps=eps,
            diff_matrix=diff_sub,
        )
        
        refined[support_mask] = refined_coeff
        return refined

    def reconstruct_legacy_itcg_vs(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
    ) -> np.ndarray:
        lambda_value = float(self.inverse_config.get("legacy_itcg_lambda", 1e-4))
        err = float(self.inverse_config.get("legacy_itcg_err", 1e-10))
        
        # ITCG-vs implementation
        # A: operator (M x N)
        # y: measurement (M)
        # tau: lambda_value
        
        M, N = operator.shape
        y = measurement.reshape(-1)
        
        # Normalize measurement (Standard practice to make lambda robust)
        y_norm = np.linalg.norm(y)
        if y_norm < 1e-12:
            return np.zeros(N)
        y = y / y_norm
        
        # Normalize columns (equivalent to MATLAB line 23)
        col_norms = np.linalg.norm(operator, axis=0)
        # Avoid division by zero
        scale = np.maximum(col_norms, 1e-12)
        # A_norm = A ./ scale
        A_norm = operator / scale
        
        # Calculate b = A' * y
        b = A_norm.T @ y
        
        # Check termination condition (line 25)
        if lambda_value >= np.max(np.abs(b)):
            return np.zeros(N)
            
        # A = [A, -A] (line 34)
        A_aug = np.hstack([A_norm, -A_norm])
        
        # c = tau + [-b; b] (line 35)
        c = lambda_value + np.concatenate([-b, b])
        
        Ns = int(M / 4)
        N_max = Ns + int(Ns / 8)
        
        gamma = 0.9
        beta = 0.01
        delta = 7.0
        alpha_max = 1e10
        ind = 0
        
        z = np.zeros(2 * N)
        g = c.copy()
        
        # v = min(0, g) (line 52)
        v = np.minimum(0, g)
        nv_itcg = np.linalg.norm(v)
        
        max_iter = 1000
        
        for iter_count in range(max_iter + 1):
            if nv_itcg < err:
                break
                
            # Gamma (line 61)
            gamma_mask = ((z > 0) & (g != 0)) | ((z == 0) & (g < 0))
            
            abs_g = np.abs(g)
            z_abs_g = z / (abs_g + np.finfo(float).eps)
            
            # Selection of I (lines 65-71)
            criteria = ((z > 0) & (z_abs_g > delta)) * abs_g
            I_hat = np.argsort(criteria)[::-1]
            Y_sorted = criteria[I_hat]
            
            n_Y = np.sum(Y_sorted != 0)
            if n_Y <= Ns:
                I = I_hat[:n_Y]
            else:
                I = I_hat[:Ns]
            
            if n_Y == 0:
                ind = 1
                d_I = np.array([]) 
            else:
                # Solve subproblem on I
                A_I = A_aug[:, I]
                b_I = g[I]
                z_I = z[I]
                
                MI, NI = A_I.shape
                s = np.zeros(NI)
                gI = b_I.copy()
                d_bar = -gI
                q = np.dot(gI, gI)
                
                err_sub = 1e-40
                
                # Subproblem loop (lines 87-117)
                for iter_sub in range(NI):
                    q1 = q
                    if q1 < err_sub:
                        break
                        
                    t = A_I @ d_bar
                    norm_t2 = np.dot(t, t)
                    
                    if q1 >= alpha_max * norm_t2:
                        alpha_I = alpha_max
                    else:
                        alpha_I = q1 / norm_t2
                    
                    alpha_I_d_bar = alpha_I * d_bar
                    z_bar = z_I + alpha_I_d_bar
                    
                    if np.any(z_bar < 0):
                        mask = d_bar < 0
                        ratios = np.zeros_like(d_bar)
                        ratios[mask] = -z_I[mask] / d_bar[mask]
                        
                        valid_ratios = ratios[ratios > 0]
                        if valid_ratios.size > 0:
                            alpha_star = np.min(valid_ratios)
                        else:
                            alpha_star = 0.0
                            
                        s = s + alpha_star * d_bar
                        break
                    else:
                        z_I = z_bar
                        s = s + alpha_I_d_bar
                        gI = gI + alpha_I * (A_I.T @ t)
                        q = np.dot(gI, gI)
                        beta_bar = q / q1
                        d_bar = -gI + beta_bar * d_bar
                
                d_I = s
            
            # Selection of J (lines 122-132)
            mask_J = np.ones(2 * N, dtype=bool)
            if n_Y > 0:
                mask_J[I] = False
            
            criteria_J = (np.abs(g) * gamma_mask) * mask_J
            J_hat = np.argsort(criteria_J)[::-1]
            Y_J = criteria_J[J_hat]
            
            n_Y_J = np.sum(Y_J != 0)
            limit_J = N_max - Ns
            if n_Y_J <= limit_J:
                J = J_hat[:n_Y_J]
            else:
                J = J_hat[:limit_J]
            
            d = np.zeros(2 * N)
            if ind == 1:
                d[J] = -v[J]
                ind = 0
            else:
                if n_Y > 0:
                    d[I] = d_I
                d[J] = -v[J]
            
            d_sparse = d # Dense vector
            g_d = np.dot(g, d_sparse)
            
            if g_d >= 0:
                d_sparse = -v
            
            # Line search (lines 148-173)
            A_d_Gamma = A_aug @ d_sparse
            d_B_d = np.dot(A_d_Gamma, A_d_Gamma)
            
            alpha_step = 1.0
            
            # Ensure descent condition check uses valid g_d
            if g_d >= 0:
                 g_d = np.dot(g, d_sparse)

            for l in range(101):
                if g_d < 0 and d_B_d <= 2 * (beta - 1) * g_d:
                    alpha_step = gamma ** l
                    break
                
                d_B_d = gamma * d_B_d
                
            # Update z (lines 175-177)
            z0 = z.copy()
            d_sparse = alpha_step * d_sparse
            z = z + d_sparse
            z = z * (z > 0) # Project to non-negative
            
            # "Normalization" of z (lines 179-181)
            z_min = np.minimum(z[:N], z[N:])
            z[:N] = z[:N] - z_min
            z[N:] = z[N:] - z_min
            
            # Update g (lines 182-187)
            d1 = z[:N] - z0[:N]
            d2 = z[N:] - z0[N:]
            A0_d1_d2 = A_norm @ (d1 - d2)
            A0T_res = A_norm.T @ A0_d1_d2
            
            g[:N] = g[:N] + A0T_res
            g[N:] = g[N:] - A0T_res
            
            # Update v
            v = np.minimum(z, g)
            nv_itcg = np.linalg.norm(v)
            
        # Final result (lines 207-214)
        x_itcg = z[:N] - z[N:]
        
        final_x = np.abs(x_itcg / scale) * y_norm
        return final_x


    def reconstruct(
        self,
        sample_id: int,
        method: str = "l1",
    ) -> dict:
        true_source = load_source_pattern_from_sample(self.config, self.sample_dir, sample_id)
        
        # Try loading pre-computed nodal solutions
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        true_nodal = None
        
        if epi_mode:
            true_nodal = {}
            loaded_all = True
            for angle in self.angles:
                p = Path(self.sample_dir) / f"fem_nodal_fluence_angle_{angle}.npy"
                if p.exists():
                    true_nodal[angle] = np.load(p)
                else:
                    loaded_all = False
                    break
            
            if not loaded_all:
                print("Warning: Pre-computed nodal fluence not found for all angles. Re-solving.")
                true_nodal = self.solve_source(true_source)
        else:
             p = Path(self.sample_dir) / "fem_nodal_fluence.npy"
             if p.exists():
                 true_nodal = np.load(p)
             else:
                 true_nodal = self.solve_source(true_source)

        measurement = self._observe_solution(true_nodal)
        operator = self.build_operator()

        row_mask = (np.abs(operator).sum(axis=1) > 1e-12) | (np.abs(measurement) > 1e-12)
        operator_used = operator[row_mask]
        measurement_used = measurement[row_mask]

        if method == "l2":
            lambda_value = float(self.inverse_config.get("l2_lambda", 1e-3))
            coeff = self.reconstruct_l2(measurement_used, operator_used, lambda_value=lambda_value)
        elif method == "legacy_l1":
            coeff = self.reconstruct_legacy_l1(measurement_used, operator_used)
        elif method == "legacy_tv":
            coeff = self.reconstruct_legacy_tv(measurement_used, operator_used)
        elif method == "legacy_itcg_vs":
            coeff = self.reconstruct_legacy_itcg_vs(measurement_used, operator_used)
        elif method == "laplacian":
            lambda_value = float(self.inverse_config.get("laplacian_lambda", 5e-2))
            coeff = self.reconstruct_laplacian(
                measurement_used,
                operator_used,
                lambda_value=lambda_value,
            )
        elif method == "tv":
            lambda_value = float(self.inverse_config.get("tv_lambda", 5e-2))
            coeff = self.reconstruct_tv(
                measurement_used,
                operator_used,
                lambda_value=lambda_value,
                n_iter=int(self.inverse_config.get("tv_irls_iter", 15)),
                eps=float(self.inverse_config.get("tv_eps", 1e-3)),
            )
        elif method == "l1_ls":
            lambda_value = float(self.inverse_config.get("l1_lambda", 1e-4))
            max_iter = int(self.inverse_config.get("l1_max_iter", 200))
            err = float(self.inverse_config.get("l1_err", 1e-10))
            coeff = self.reconstruct_l1_ls(
                measurement_used,
                operator_used,
                lambda_value=lambda_value,
                max_iter=max_iter,
                err=err,
            )
        else:
            lambda_value = float(self.inverse_config.get("l1_lambda", 1e-4))
            max_iter = int(self.inverse_config.get("l1_max_iter", 200))
            coeff = self.reconstruct_l1_fista(
                measurement_used,
                operator_used,
                lambda_value=lambda_value,
                max_iter=max_iter,
            )

        recon_source = self.coefficients_to_volume(coeff)
        recon_nodal = self.solve_source(recon_source)
        pred_measurement = self._observe_solution(recon_nodal)

        basis_centers = self.basis.node_coords_zyx if self.basis_mode == "fem_node" else self.basis.block_centers_zyx

        return {
            "sample_id": sample_id,
            "angles": list(self.observation.angles),
            "method": method,
            "operator": operator,
            "row_mask": row_mask,
            "measurement": measurement,
            "pred_measurement": pred_measurement,
            "coefficients": coeff,
            "true_source": true_source,
            "recon_source": recon_source,
            "true_nodal": true_nodal,
            "recon_nodal": recon_nodal,
            "basis_centers_zyx": basis_centers,
            "observation": self.observation,
        }


def save_reconstruction_report(result: dict, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "true_source.npy", result["true_source"])
    np.save(out_path / "recon_source.npy", result["recon_source"])
    np.save(out_path / "basis_centers_zyx.npy", result["basis_centers_zyx"])
    np.save(out_path / "source_coefficients.npy", result["coefficients"])

    observation: VisibleSurfaceObservation = result["observation"]
    if observation.observation_type != "legacy_surface_nodes":
        for angle in result["angles"]:
            row_slice = observation.rows_for_angle(angle)
            coords = observation.patch_centroids_by_angle[angle]
            patches = observation.patch_coords_by_angle[angle]
            depth = observation.camera_depth_by_angle[angle]
            face_counts = observation.patch_face_counts_by_angle[angle]
            np.save(out_path / f"visible_patch_centroids_{angle}.npy", coords)
            np.save(out_path / f"visible_patch_coords_{angle}.npy", patches)
            np.save(out_path / f"visible_patch_depth_{angle}.npy", depth)
            np.save(out_path / f"visible_patch_face_counts_{angle}.npy", face_counts)
            np.save(out_path / f"true_visible_signal_{angle}.npy", result["measurement"][row_slice])
            np.save(out_path / f"recon_visible_signal_{angle}.npy", result["pred_measurement"][row_slice])

    measurement = result["measurement"]
    prediction = result["pred_measurement"]
    rel_error = float(
        np.linalg.norm(prediction - measurement) / max(np.linalg.norm(measurement), 1e-12)
    )
    sse = float(np.sum((prediction - measurement) ** 2))

    summary = {
        "sample_id": int(result["sample_id"]),
        "angles": [int(v) for v in result["angles"]],
        "method": result["method"],
        "observation_type": observation.observation_type,
        "n_basis": int(len(result["coefficients"])),
        "n_measurements": int(measurement.size),
        "measurement_sse": sse,
        "measurement_relative_error": rel_error,
        "measurement_relative_error_percent": rel_error * 100.0,
        "true_source_center_zyx": _weighted_center(result["true_source"]),
        "recon_source_center_zyx": _weighted_center(result["recon_source"]),
        "true_source_mass": float(result["true_source"].sum()),
        "recon_source_mass": float(result["recon_source"].sum()),
    }
    
    # Compute extra metrics
    center_true = np.array(summary["true_source_center_zyx"])
    center_recon = np.array(summary["recon_source_center_zyx"])
    localization_error = np.linalg.norm(center_true - center_recon)
    
    # Quantitativeness: Max Recon / Max True
    quantitativeness = float(result["recon_source"].max() / max(result["true_source"].max(), 1e-12))
    
    # Volume Ratio: Vol Recon / Vol True (threshold > 0.5 max)
    true_norm = result["true_source"] / max(result["true_source"].max(), 1e-12)
    recon_norm = result["recon_source"] / max(result["recon_source"].max(), 1e-12)
    true_vol = (true_norm > 0.5).sum()
    recon_vol = (recon_norm > 0.5).sum()
    volume_ratio = float(recon_vol / max(true_vol, 1.0))

    # CNR: (Mean_ROI - Mean_BG) / Std_BG
    # Since background is 0, we can define ROI as > 50% max.
    roi_mask = recon_norm > 0.5
    bg_mask = ~roi_mask
    if roi_mask.sum() > 0:
        mean_roi = result["recon_source"][roi_mask].mean()
        mean_bg = result["recon_source"][bg_mask].mean() if bg_mask.sum() > 0 else 0.0
        std_bg = result["recon_source"][bg_mask].std() if bg_mask.sum() > 1 else 1e-12
        cnr = float((mean_roi - mean_bg) / max(std_bg, 1e-12))
    else:
        cnr = 0.0

    def _dice(v1, v2, threshold=0.5):
        # Normalize
        v1_norm = v1 / (v1.max() + 1e-12)
        v2_norm = v2 / (v2.max() + 1e-12)
        
        m1 = v1_norm > threshold
        m2 = v2_norm > threshold
        
        # Optional: Remove small components (like minr_fmt)
        # This requires connected component analysis, might be slow or need scipy
        # For now, let's just stick to raw threshold intersection.
        # But wait, previous memory says minr_fmt uses strict pre-processing.
        # If the result is sparse/noisy, direct threshold might pick up noise.
        
        if m1.sum() == 0 and m2.sum() == 0:
            return 0.0 
        
        intersection = np.logical_and(m1, m2).sum()
        return 2.0 * intersection / (m1.sum() + m2.sum() + 1e-12)

    dice_50 = _dice(result["true_source"], result["recon_source"], 0.5)
    # Also calculate a "relaxed" dice with lower threshold for sparse solvers like ITCG
    dice_20 = _dice(result["true_source"], result["recon_source"], 0.2)
    
    summary["metrics"] = {
        "LE": float(localization_error),
        "localization_error": float(localization_error),
        "Dice": float(dice_50),
        "Dice_20": float(dice_20),
        "Q": float(quantitativeness),
        "VR": float(volume_ratio),
        "CNR": float(cnr),
    }

    if observation.observation_type == "legacy_surface_nodes":
        summary["n_surface_nodes"] = int(measurement.size)
    else:
        summary["n_visible_patches"] = int(measurement.size)
    with open(out_path / "inverse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_visualization(result, out_path)


def _save_visualization(result: dict, out_path: Path) -> None:
    true_source = result["true_source"]
    recon_source = result["recon_source"]
    observation: VisibleSurfaceObservation = result["observation"]
    angles = result["angles"]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.imshow(true_source.max(axis=0), cmap="magma")
    ax1.set_title("True source MIP(Z)")
    ax2.imshow(recon_source.max(axis=0), cmap="magma")
    ax2.set_title("Recon source MIP(Z)")

    ref_angle = angles[len(angles) // 2]
    coords = observation.patch_centroids_by_angle[ref_angle]
    row_slice = observation.rows_for_angle(ref_angle)
    true_signal = result["measurement"][row_slice]
    recon_signal = result["pred_measurement"][row_slice]
    scatter = ax3.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=true_signal,
        s=8,
        cmap="inferno",
    )
    ax3.set_title(f"Visible surface patches {ref_angle}°")
    fig.colorbar(scatter, ax=ax3, shrink=0.65)

    ax4.plot(true_signal, label="true", linewidth=1.0)
    ax4.plot(recon_signal, label="recon", linewidth=1.0)
    ax4.set_title(f"Patch partial current compare {ref_angle}°")
    ax4.legend(loc="upper right")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path / "source_inverse_compare.png", dpi=180)
    plt.close(fig)
