from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.ndimage import map_coordinates
from scipy import ndimage

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

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
    # Use ZYX directly (matching batch_config_generator and standard volume order)
    volume_shape = tuple(config["volume_shape"])
    
    # Try to get src_range from config, otherwise None
    src_range = config.get("src_range")
    if src_range:
        src_range_z = src_range["z"]
        src_range_y = src_range["y"]
        src_range_x = src_range["x"]
        expected_dz = src_range_z[1] - src_range_z[0]
        expected_dy = src_range_y[1] - src_range_y[0]
        expected_dx = src_range_x[1] - src_range_x[0]
        expected_size = expected_dz * expected_dy * expected_dx
    else:
        src_range_z = src_range_y = src_range_x = None
        expected_size = -1

    source_filename = f"source-{sample_id}.bin"
    source_path = Path(sample_dir) / source_filename
    
    # Try loading from true_source.npy if bin doesn't exist (for re-running on results)
    npy_path = Path(sample_dir) / "true_source.npy"
    if not source_path.exists() and npy_path.exists():
        print(f"Loading source from {npy_path}")
        return np.load(npy_path)

    if not source_path.exists():
        # Try checking if source file is in parent directory
        source_path = Path(sample_dir).parent / source_filename
        if not source_path.exists():
            raise FileNotFoundError(f"光源文件不存在: {source_path}")

    # Determine shape from file size if mismatch
    file_size = source_path.stat().st_size
    num_floats = file_size // 4
    
    if num_floats == expected_size:
        # Use config ranges. Note: volume is (X, Y, Z). 
        # Source pattern should be placed at [x0:x1, y0:y1, z0:z1]
        x0, y0, z0 = src_range_x[0], src_range_y[0], src_range_z[0]
        # batch_config_generator saves as ZYX (transpose 2,1,0)
        # So we must reshape as (dz, dy, dx)
        src_shape = (src_range_z[1]-z0, src_range_y[1]-y0, src_range_x[1]-x0)
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
                    # MCX Pattern is Nx, Ny, Nz. 
                    # If we reshape linear array to (Nx, Ny, Nz), it assumes X-fastest?
                    # No, numpy reshape is C-order (last axis fastest).
                    # If MCX loads linear array, it fills X, then Y, then Z.
                    # So (Nz, Ny, Nx) in numpy matches MCX (Nx, Ny, Nz).
                    src_shape = (nz, ny, nx) 
                    # Use Pos as start index (1-based -> 0-based)
                    # Wait. MCX uses 1-based indexing for DETECTORS, but source Pos is usually coordinate (or grid unit).
                    # If Pos is [0,0,0], it means origin.
                    # We should NOT subtract 1 if it is 0-based coordinate.
                    # But if it is 1-based index...
                    # Usually MCX inputs are 1-based?
                    # But config generator uses 0-based logic for Pos.
                    # "Pos": [float(src_range_x[0]), ...]
                    # src_range_x[0] is 0.
                    # So Pos is 0.
                    # If we subtract 1, we get -1.
                    # So we should NOT subtract 1.
                    pos = mcx_conf.get("Optode", {}).get("Source", {}).get("Pos", [0,0,0])
                    # Pos is [x, y, z]
                    x0 = int(pos[0])
                    y0 = int(pos[1])
                    z0 = int(pos[2])
                else:
                    raise ValueError(f"Source size mismatch: file {num_floats}, config {expected_size}")
            except Exception as e:
                raise ValueError(f"Failed to infer source shape from JSON: {e}")
        else:
             raise ValueError(f"Source size mismatch: file {num_floats}, config {expected_size}, and no JSON found")

    # Reshape assuming (Nz, Ny, Nx) - ZYX order from batch_config_generator save
    source_arr_zyx = np.fromfile(source_path, dtype=np.float32).reshape(src_shape)
    
    # Transpose to XYZ to match volume shape
    # (Nz, Ny, Nx) -> (Nx, Ny, Nz)
    source_arr_xyz = source_arr_zyx.transpose(2, 1, 0)
    
    source_pattern = np.zeros(volume_shape, dtype=np.float32)
    
    # Safe assignment with clipping
    # volume_shape is (X, Y, Z)
    # source_arr_xyz is (Nx, Ny, Nz)
    
    # Calculate bounds
    # x0, y0, z0 are start indices for X, Y, Z
    nx_len = source_arr_xyz.shape[0]
    ny_len = source_arr_xyz.shape[1]
    nz_len = source_arr_xyz.shape[2]
    
    x1 = min(x0 + nx_len, volume_shape[0])
    y1 = min(y0 + ny_len, volume_shape[1])
    z1 = min(z0 + nz_len, volume_shape[2])
    
    # Effective lengths
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    
    if dx > 0 and dy > 0 and dz > 0:
        source_pattern[x0:x1, y0:y1, z0:z1] = source_arr_xyz[0:dx, 0:dy, 0:dz]
        
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
        
        # Helper to resolve mesh path
        def resolve_mesh_path(name: str) -> str:
            # 1. Check if absolute or in CWD
            if Path(name).exists():
                return name
            
            # 2. Check generated_mesh_path
            gen_path = config.get("generated_mesh_path")
            if gen_path and Path(gen_path).name == name and Path(gen_path).exists():
                return gen_path
                
            # 3. Check volume_bases/mesh/
            # Try to find base_input_dir from config or default
            base_input = config.get("base_input_dir", "./volume_bases")
            candidate = Path(base_input) / "mesh" / name
            if candidate.exists():
                return str(candidate)
                
            # 4. If nothing found, return original (will likely trigger rebuild in CWD)
            return name

        fem_cfg = config.get("fem", {})
        if mesh_cache_path:
             forward_mesh_name = mesh_cache_path
             print(f"DEBUG: Using mesh_cache_path from args: {forward_mesh_name}")
        else:
             raw_name = fem_cfg.get("mesh_cache_name", "brain_fem_mesh.npz")
             forward_mesh_name = resolve_mesh_path(raw_name)
             print(f"DEBUG: Using mesh_cache_name from config: {forward_mesh_name}")
             
        inverse_cfg = config.get("inverse", {})
        raw_recon_name = inverse_cfg.get("recon_mesh_cache_name", fem_cfg.get("mesh_cache_name", "brain_fem_mesh.npz"))
        recon_mesh_name = resolve_mesh_path(raw_recon_name)
        
        print(f"DEBUG: raw_recon_name={raw_recon_name}")
        print(f"DEBUG: resolved recon_mesh_name={recon_mesh_name}")
        
        # Override mesh config for reconstruction if specified
        recon_fem_config = fem_cfg.copy()
        if "recon_downsample_factor" in inverse_cfg:
            factor = inverse_cfg["recon_downsample_factor"]
            recon_fem_config.setdefault("mesh", {})["downsample_factor"] = factor
            print(f"DEBUG: Using Reconstruction Mesh Downsample Factor: {factor}")
        elif "recon_mesh" in inverse_cfg and "downsample_factor" in inverse_cfg["recon_mesh"]:
            factor = inverse_cfg["recon_mesh"]["downsample_factor"]
            recon_fem_config.setdefault("mesh", {})["downsample_factor"] = factor
            print(f"DEBUG: Using Reconstruction Mesh Downsample Factor (from mesh block): {factor}")

        self.solver = FemDiffusionSolver(
            volume_path=config["generated_bin_path"],
            media_yaml_path=config["generated_material_yaml_path"],
            volume_shape=tuple(config["volume_shape"]),
            dx=config.get("lengthunit", 0.1),
            mesh_cache_path=recon_mesh_name,
            fem_config=recon_fem_config,
        )
        
        if recon_mesh_name != forward_mesh_name:

            self.forward_solver = FemDiffusionSolver(
                volume_path=config["generated_bin_path"],
                media_yaml_path=config["generated_material_yaml_path"],
                volume_shape=tuple(config["volume_shape"]),
                dx=config.get("lengthunit", 0.1),
                mesh_cache_path=forward_mesh_name,
                fem_config=fem_cfg,
            )
        else:
            self.forward_solver = self.solver

        self.basis_mode = self.inverse_config.get("basis_mode", "voxel_block")
        print(f"DEBUG: Basis Mode: {self.basis_mode}")
        if self.basis_mode == "fem_node":
             self.basis = self._build_fem_node_basis()
             print(f"DEBUG: Built FEM Node Basis with {len(self.basis.node_indices)} nodes")
        else:
             self.basis = self._build_source_basis()
             print(f"DEBUG: Built Block Basis with {len(self.basis.block_indices)} blocks")
        self.observation = self._build_observation()
        self._rhs_matrix: np.ndarray | None = None
        self._operator_cache: np.ndarray | None = None
        self._graph_edges: np.ndarray | None = None
        self._graph_laplacian: np.ndarray | None = None
        self._tv_difference: np.ndarray | None = None
        
        self._init_excitation()

    def _init_excitation(self):
        """初始化激发光场。支持固定光源或随相机移动的光源。"""
        self.angle_excitation_scalings: dict[int, np.ndarray] = {}
        if self.inverse_config.get("disable_excitation_scaling", False):
            print("DEBUG: Excitation scaling disabled (Pure Emission Mode)")
            for angle in self.angles:
                if self.basis_mode == "fem_node":
                    n_basis = len(self.basis.node_indices)
                else:
                    n_basis = len(self.basis.block_indices)
                self.angle_excitation_scalings[angle] = np.ones(n_basis, dtype=np.float32)
            return

        ex_pos = self.inverse_config.get("excitation_pos")
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        
        self.angle_excitation_scalings: dict[int, np.ndarray] = {}
        self._excitation_volume = None

        if epi_mode:

            radius = float(self.inverse_config.get("excitation_radius", 0.5))
            for angle in self.angles:
                src_pos = self.compute_rotated_source_pos(angle)

                try:
                    phi_ex = self.solver.solve_excitation(src_pos, radius=radius)
                    if self.basis_mode == "fem_node":
                         if self.basis.node_indices.max() >= len(phi_ex): raise ValueError("Basis node indices exceed mesh node count")
                         basis_scaling = phi_ex[self.basis.node_indices]
                    else:
                        if self._excitation_volume is None: self._excitation_volume = self.solver.sample_to_volume(phi_ex).transpose(2, 1, 0)
                        flat_ex = self.solver.sample_to_volume(phi_ex).transpose(2, 1, 0).reshape(-1)
                        basis_scaling = np.zeros(len(self.basis.block_indices), dtype=np.float32)
                        for i, indices in enumerate(self.basis.block_indices): basis_scaling[i] = np.mean(flat_ex[indices])
                    self.angle_excitation_scalings[angle] = basis_scaling
                except Exception as e:
                    print(f"Warning: Failed to solve excitation for angle {angle}: {e}")
                    n_basis = len(self.basis.node_indices) if self.basis_mode == "fem_node" else len(self.basis.block_indices)
                    self.angle_excitation_scalings[angle] = np.ones(n_basis, dtype=np.float32)

        elif ex_pos:

            # Mode 1: Fixed Source (Legacy)
            # Solve once
            phi_ex = self.solver.solve_excitation(tuple(ex_pos))

            
            self._excitation_volume = self.solver.sample_to_volume(phi_ex).transpose(2, 1, 0)
            
            # Map to basis scaling factors
            if self.basis_mode == "fem_node":
                # basis_scaling = phi_ex[self.basis.node_indices]
                # Check bounds
                if self.basis.node_indices.max() >= len(phi_ex):
                     raise ValueError("Basis node indices exceed mesh node count")
                basis_scaling = phi_ex[self.basis.node_indices]

            else:
                basis_scaling = np.zeros(len(self.basis.block_indices), dtype=np.float32)
                flat_ex = self._excitation_volume.reshape(-1)
                for i, indices in enumerate(self.basis.block_indices):
                    basis_scaling[i] = np.mean(flat_ex[indices])

            
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
                             self._excitation_volume = self.solver.sample_to_volume(phi_ex).transpose(2, 1, 0)

                        flat_ex = self.solver.sample_to_volume(phi_ex).transpose(2, 1, 0).reshape(-1)
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
        if "src_range" in self.config:
            src_range = self.config["src_range"]
            # Map config ranges to Volume Axes (XYZ)
            r0_0, r0_1 = src_range["x"]
            r1_0, r1_1 = src_range["y"]
            r2_0, r2_1 = src_range["z"]
        else:
            shape = self.solver.volume_shape
            r0_0, r0_1 = 0, shape[0]
            r1_0, r1_1 = 0, shape[1]
            r2_0, r2_1 = 0, shape[2]
            
        roi = self.solver.volume_data[r0_0:r0_1, r1_0:r1_1, r2_0:r2_1]

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
        # Iterate over ROI axes (Axis 0=X, Axis 1=Y, Axis 2=Z)
        for i in range(0, roi_shape[0], block_shape[0]):
            is_slice = slice(i, min(i + block_shape[0], roi_shape[0]))
            for j in range(0, roi_shape[1], block_shape[1]):
                js_slice = slice(j, min(j + block_shape[1], roi_shape[1]))
                for k in range(0, roi_shape[2], block_shape[2]):
                    ks_slice = slice(k, min(k + block_shape[2], roi_shape[2]))
                    
                    block_mask = allowed[is_slice, js_slice, ks_slice]
                    if not np.any(block_mask):
                        continue

                    local_coords = np.argwhere(block_mask)
                    # Global coords in Volume (X, Y, Z)
                    # local_coords[:, 0] is i (Axis 0/X)
                    global_coords = local_coords + np.array([is_slice.start, js_slice.start, ks_slice.start], dtype=np.int32)
                    
                    flat = np.ravel_multi_index(
                        (
                            global_coords[:, 0] + r0_0, # X + X0
                            global_coords[:, 1] + r1_0, # Y + Y0
                            global_coords[:, 2] + r2_0, # Z + Z0
                        ),
                        dims=tuple(self.config["volume_shape"]),
                    )
                    block_indices.append(flat.astype(np.int64))
                    
                    # Store block metadata (Center in physical coords mm)
                    # Center of block (geometric center of the slice)
                    center_phys = np.array([
                        (r0_0 + (is_slice.start + is_slice.stop - 1) / 2.0) * self.solver.dx,
                        (r1_0 + (js_slice.start + js_slice.stop - 1) / 2.0) * self.solver.dx,
                        (r2_0 + (ks_slice.start + ks_slice.stop - 1) / 2.0) * self.solver.dx,
                    ])
                    block_centers.append(center_phys)
                    
                    block_grid_coords.append(
                        np.array(
                            [
                                i // block_shape[0],
                                j // block_shape[1],
                                k // block_shape[2],
                            ],
                            dtype=np.int32,
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
        # If resize_shape is not in config, we use volume_shape (no resizing)
        proj_config = self.config.get("projection", {})
        if "resize_shape" in proj_config:
            resize_shape = np.array(proj_config["resize_shape"], dtype=np.float64)
        else:
            resize_shape = np.array(self.config["volume_shape"], dtype=np.float64)

        # volume_shape is Z, Y, X
        volume_shape = np.array(self.config["volume_shape"], dtype=np.float64)
        scale = resize_shape / volume_shape
        dx = max(float(self.config.get("lengthunit", 0.1)), 1e-12)

        x_idx = nodes_xyz[:, 0] / dx
        y_idx = nodes_xyz[:, 1] / dx
        z_idx = nodes_xyz[:, 2] / dx

        # Correct mapping: Match X with X (index 2), Y with Y (index 1), Z with Z (index 0)
        # resize_shape and scale are Z, Y, X
        projector_points = np.column_stack(
            [
                x_idx * scale[2] - resize_shape[2] / 2.0 + 0.5,
                y_idx * scale[1] - resize_shape[1] / 2.0 + 0.5,
                z_idx * scale[0] - resize_shape[0] / 2.0 + 0.5,
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

        # Build observation for ALL angles
        rows = []
        cols = []
        data = []
        
        row_slices_by_angle = {}
        patch_coords_by_angle = {}
        patch_centroids_by_angle = {}
        camera_depth_by_angle = {}
        patch_face_counts_by_angle = {}
        
        row_cursor = 0
        n_per_angle = kept_ids.size
        
        for angle in self.angles:
            # For each angle, we observe the SAME surface nodes
            # Rows: [row_cursor, row_cursor + n_per_angle)
            # Cols: kept_ids
            # Data: 1.0
            
            current_rows = np.arange(row_cursor, row_cursor + n_per_angle, dtype=np.int32)
            rows.append(current_rows)
            cols.append(kept_ids) # Repeatedly observe same nodes
            data.append(np.ones(n_per_angle, dtype=np.float64))
            
            row_slices_by_angle[angle] = (row_cursor, row_cursor + n_per_angle)
            
            # Metadata duplicates
            patch_coords_by_angle[angle] = np.zeros((n_per_angle, 2), dtype=np.int32) # Dummy
            patch_centroids_by_angle[angle] = kept_nodes.astype(np.float64)
            camera_depth_by_angle[angle] = kept_nodes[:, 2].astype(np.float32)
            patch_face_counts_by_angle[angle] = np.ones(n_per_angle, dtype=np.int32)
            
            row_cursor += n_per_angle

        observation_matrix = coo_matrix(
            (
                np.concatenate(data),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
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
                rhs_matrix[:, idx] = self.solver.build_source_vector(source_volume.transpose(2, 1, 0), normalize=False)

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
        if HAS_CUPY and isinstance(coeff, cp.ndarray):
            coeff = coeff.get()
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
        # If inputs are CuPy arrays, perform computation on GPU and return CuPy arrays
        if HAS_CUPY and isinstance(operator, cp.ndarray):
            return operator.T @ operator, operator.T @ measurement

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

    def _power_iteration_norm_sq(
        self,
        operator: np.ndarray | "cp.ndarray",
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> float:
        xp = cp.get_array_module(operator) if (HAS_CUPY and isinstance(operator, cp.ndarray)) else np
        n_cols = int(operator.shape[1])
        if n_cols == 0:
            return 0.0
        v = xp.ones((n_cols,), dtype=operator.dtype)
        v /= max(float(xp.linalg.norm(v)), 1e-12)
        prev = 0.0
        for _ in range(max_iter):
            av = operator @ v
            atav = operator.T @ av
            nrm = float(xp.linalg.norm(atav))
            if nrm <= 1e-12:
                return 0.0
            v = atav / nrm
            eig = float(xp.dot(v, operator.T @ (operator @ v)))
            if abs(eig - prev) <= tol * max(abs(eig), 1e-12):
                return max(eig, 0.0)
            prev = eig
        return max(prev, 0.0)

    def reconstruct_baseline_v1_pgd(
        self,
        measurement: np.ndarray | "cp.ndarray",
        operator: np.ndarray | "cp.ndarray",
        reg_lambda: float,
    ) -> np.ndarray | "cp.ndarray":
        xp = cp.get_array_module(operator) if (HAS_CUPY and isinstance(operator, cp.ndarray)) else np
        y = xp.asarray(measurement, dtype=operator.dtype).reshape(-1)
        A = xp.asarray(operator, dtype=operator.dtype)
        n_cols = int(A.shape[1])
        if n_cols == 0:
            return xp.zeros((0,), dtype=A.dtype)
        x = xp.zeros((n_cols,), dtype=A.dtype)  # x0 = 0
        spectral_sq = self._power_iteration_norm_sq(
            A,
            max_iter=int(self.inverse_config.get("power_iter_max", 50)),
            tol=float(self.inverse_config.get("power_iter_tol", 1e-6)),
        )
        L = float(reg_lambda) + max(float(spectral_sq), 1e-12)
        step = 1.0 / L
        max_iter = int(self.inverse_config.get("baseline_v1_max_iter", 400))
        rel_tol = float(self.inverse_config.get("baseline_v1_rel_tol", 1e-6))
        for _ in range(max_iter):
            grad = A.T @ (A @ x - y) + float(reg_lambda) * x
            x_next = xp.maximum(x - step * grad, 0.0)
            rel = float(xp.linalg.norm(x_next - x) / max(float(xp.linalg.norm(x_next)), 1e-12))
            x = x_next
            if rel <= rel_tol:
                break
        return x

    def _map_source_volume_to_basis_coeff(self, source_xyz: np.ndarray) -> np.ndarray:
        if self.basis_mode != "fem_node":
            raise ValueError("baseline-v1 仅支持 fem_node basis_mode。")
        source_zyx = np.asarray(source_xyz, dtype=np.float64).transpose(2, 1, 0)
        rhs_full = self.solver.build_source_vector(source_zyx, normalize=False)
        coeff = rhs_full[self.basis.node_indices]
        return np.maximum(np.asarray(coeff, dtype=np.float64), 0.0)

    def _filter_small_components(
        self,
        mask: np.ndarray,
        min_size: int,
        connectivity_dim: int,
        dilation_iters: int,
    ) -> tuple[np.ndarray, int]:
        mask = np.asarray(mask, dtype=bool)
        conn = max(1, min(int(connectivity_dim), mask.ndim))
        structure = ndimage.generate_binary_structure(mask.ndim, conn)
        cc_mask = mask
        if dilation_iters > 0:
            cc_mask = ndimage.binary_dilation(
                mask,
                structure=np.ones((3,) * mask.ndim, dtype=bool),
                iterations=int(dilation_iters),
            )
        labeled, n_comp = ndimage.label(cc_mask, structure=structure)
        if min_size <= 1:
            return mask, int(n_comp)
        if n_comp == 0:
            return np.zeros_like(mask, dtype=bool), 0
        sizes = ndimage.sum(mask.astype(np.uint8), labeled, index=list(range(1, n_comp + 1)))
        keep = np.asarray(sizes) >= int(min_size)
        if not np.any(keep):
            return np.zeros_like(mask, dtype=bool), 0
        keep_ids = np.nonzero(keep)[0] + 1
        filtered = np.isin(labeled, keep_ids) & mask
        return filtered, int(len(keep_ids))

    def _assd_hd95(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> tuple[float, float]:
        pred = np.asarray(pred_mask, dtype=bool)
        gt = np.asarray(gt_mask, dtype=bool)
        if not pred.any() and not gt.any():
            return 0.0, 0.0
        if not pred.any() or not gt.any():
            return float("nan"), float("nan")
        struct = np.ones((3, 3, 3), dtype=bool)
        pred_s = pred ^ ndimage.binary_erosion(pred, structure=struct)
        gt_s = gt ^ ndimage.binary_erosion(gt, structure=struct)
        dt_gt = ndimage.distance_transform_edt(~gt_s, sampling=spacing)
        dt_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing)
        d = np.concatenate([dt_gt[pred_s], dt_pred[gt_s]], axis=0)
        return float(np.mean(d)), float(np.percentile(d, 95))

    def _compute_minrfmt_metrics(
        self,
        true_source: np.ndarray,
        recon_source: np.ndarray,
    ) -> dict:
        eval_cfg = self.inverse_config.get("evaluation", {})
        pred_threshold = float(eval_cfg.get("pred_threshold", 0.5))
        min_region_size = int(eval_cfg.get("min_region_size", 10))
        cc_connectivity = int(eval_cfg.get("cc_connectivity", 26))
        if cc_connectivity == 6:
            conn_dim = 1
        elif cc_connectivity == 18:
            conn_dim = 2
        else:
            conn_dim = 3
        cc_dilation_iters = int(eval_cfg.get("cc_dilation_iters", 1))
        spacing = tuple(float(v) for v in eval_cfg.get("voxel_spacing", [1.0, 1.0, 1.0]))
        gt_bin = np.asarray(true_source > 0.0, dtype=bool)
        vmax = float(np.max(recon_source))
        eps = float(eval_cfg.get("eps", 1e-12))
        reconstruction_failure = bool(vmax <= eps)
        if reconstruction_failure:
            pred_bin_raw = np.zeros_like(gt_bin, dtype=bool)
        else:
            pred_norm = recon_source / max(vmax, eps)
            pred_bin_raw = np.asarray(pred_norm >= pred_threshold, dtype=bool)
        pred_bin, pred_regions = self._filter_small_components(
            pred_bin_raw,
            min_size=min_region_size,
            connectivity_dim=conn_dim,
            dilation_iters=cc_dilation_iters,
        )
        gt_bin_f, gt_regions = self._filter_small_components(
            gt_bin,
            min_size=min_region_size,
            connectivity_dim=conn_dim,
            dilation_iters=cc_dilation_iters,
        )
        intersection = float(np.logical_and(pred_bin, gt_bin_f).sum())
        pred_sum = float(pred_bin.sum())
        gt_sum = float(gt_bin_f.sum())
        dice = 1.0 if (pred_sum + gt_sum) == 0.0 else (2.0 * intersection / (pred_sum + gt_sum + 1e-8))
        if (intersection + (pred_sum - intersection)) == 0.0:
            precision = 1.0 if gt_sum == 0.0 else 0.0
        else:
            precision = intersection / max(pred_sum, 1e-12)
        recall = 1.0 if gt_sum == 0.0 else (intersection / max(gt_sum, 1e-12))
        assd, hd95 = self._assd_hd95(pred_bin, gt_bin_f, spacing=spacing)
        return {
            "pred_threshold": pred_threshold,
            "min_region_size": min_region_size,
            "cc_connectivity": cc_connectivity,
            "cc_dilation_iters": cc_dilation_iters,
            "voxel_spacing": list(spacing),
            "reconstruction_failure": reconstruction_failure,
            "pred_regions": int(pred_regions),
            "gt_regions": int(gt_regions),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "assd": float(assd) if np.isfinite(assd) else None,
            "hd95": float(hd95) if np.isfinite(hd95) else None,
            "mr": None,
            "ms": None,
            "delta_cc": None,
        }

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
            if start % 200 == 0:
                print(f"Building Operator: {start}/{n_basis} ({start/n_basis:.1%})")
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
        center = (volume_shape * dx) / 2.0
        
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
        disable_excitation = self.inverse_config.get("disable_excitation_scaling", False)
        
        if epi_mode:
            results = {}
            # Reconstruct tumor/source pattern vector ONCE
            # source_pattern is volume (XYZ). Solver expects ZYX.
            tumor_rhs_raw = self.solver.build_source_vector(source_pattern.transpose(2, 1, 0), normalize=False)
            
            for angle in self.angles:
                # Compute Excitation
                if not disable_excitation:
                    ex_pos = self.compute_rotated_source_pos(angle)
                    radius = float(self.inverse_config.get("excitation_radius", 0.5))
                    phi_ex = self.solver.solve_excitation(ex_pos, radius=radius)
                    # Effective Source
                    effective_rhs = tumor_rhs_raw * phi_ex
                else:
                    effective_rhs = tumor_rhs_raw
                
                # Solve Emission
                nodal = self.solver.solve_from_rhs(effective_rhs)
                results[angle] = np.asarray(nodal, dtype=np.float64)
            return results

        if hasattr(self, "_excitation_volume") and self._excitation_volume is not None and not disable_excitation:
             effective_source = source_pattern * self._excitation_volume
        else:
             effective_source = source_pattern

        nodal = self.solver.solve(effective_source.transpose(2, 1, 0), normalize=False)
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

    def reconstruct_l1_fista_legacy(
        self,
        measurement: np.ndarray,
        operator: np.ndarray,
        lambda_value: float,
        max_iter: int = 200,
        err: float = 1e-10,
    ) -> np.ndarray:
        """
        Implementation of Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for L1 regularization.
        Matches legacy MATLAB implementation logic (FMT_DE_mouse/FISTA.m):
        1. Normalize operator columns: A = A ./ sqrt(sum(A.^2,1))
        2. Solve min 0.5||Ax - y||^2 + lambda||x||_1
        3. Denormalize solution: x = x ./ sqrt(sum(A.^2,1))
        """
        # Ensure correct shapes and types
        xp = cp.get_array_module(operator) if HAS_CUPY else np
        y = xp.asarray(measurement, dtype=np.float64).reshape(-1, 1)
        operator = xp.asarray(operator, dtype=np.float64)
        
        # 1. Column Normalization (Legacy: A = A1./repmat(sqrt(sum(A1.^2,1)),m,1))
        # Note: Legacy does NOT normalize y. It only normalizes A columns.
        col_norms = xp.sqrt(xp.sum(operator**2, axis=0))
        # Handle zero columns to avoid division by zero
        valid_cols = col_norms > 1e-12
        
        safe_norms = col_norms.copy()
        safe_norms[~valid_cols] = 1.0
        
        A_norm = operator / safe_norms[None, :]
        
        # Precompute gram matrix and A'y
        # G = A'*A (Legacy line 36)
        G = A_norm.T @ A_norm
        
        # c_F = A'*y (Legacy line 38)
        c_F = A_norm.T @ y
        
        n_features = operator.shape[1]
        
        # Initialization
        # Legacy: lambda0 = 0.5*L0*norm(c_F,inf), L0=1
        L_F = 1.0
        # Legacy uses constant lambda = tau (lambda_value)
        lambdaF = lambda_value
        
        xk = xp.zeros((n_features, 1), dtype=np.float64)
        xkm1 = xk.copy()
        t_k = 1.0
        t_km1 = 1.0
        
        beta_F = 1.5 # Backtracking multiplier
        
        for k in range(max_iter):
            # yk = xk + ((t_km1-1)/t_kF)*(xk-xkm1) (Legacy line 55)
            # Note: Legacy code has typo t_kF vs t_k, assuming t_kF is current t
            yk = xk + ((t_km1 - 1.0) / t_k) * (xk - xkm1)
            
            # Gradient of f at yk: G*yk - c_F
            grad_yk = G @ yk - c_F
            
            # Backtracking line search (Legacy lines 63-84)
            stop_backtrack = False
            
            while not stop_backtrack:
                # gk = yk - (1/L_F)*temp
                gk = yk - (1.0 / L_F) * grad_yk
                
                # xkp1 = soft(gk, lambdaF/L_F) (Legacy line 67)
                threshold = lambdaF / L_F
                xkp1 = xp.maximum(xp.abs(gk) - threshold, 0.0) * xp.sign(gk)
                
                # Check descent condition
                # temp1 = 0.5*norm(y-A*xkp1)^2
                # Efficient calculation without full A*x
                # ||Ax - y||^2 = x'G x - 2 x'c_F + y'y
                # We can ignore y'y term for comparison
                
                # temp1_part = 0.5 * (xkp1.T @ G @ xkp1 - 2 * xkp1.T @ c_F)
                # But legacy computes full norm: 0.5*norm(y-A*xkp1)^2
                # Let's do explicit for safety
                resid_xkp1 = y - A_norm @ xkp1
                temp1 = 0.5 * xp.sum(resid_xkp1**2)
                
                resid_yk = y - A_norm @ yk
                diff_x = xkp1 - yk
                temp2 = 0.5 * xp.sum(resid_yk**2) + xp.dot(diff_x.flatten(), grad_yk.flatten()) + (L_F / 2.0) * xp.sum(diff_x**2)
                
                if temp1 <= temp2 + 1e-12:
                    stop_backtrack = True
                else:
                    L_F *= beta_F
                    if L_F > 1e12: # Safety break
                        stop_backtrack = True

            # Update steps
            t_kp1 = 0.5 * (1.0 + xp.sqrt(1.0 + 4.0 * t_k * t_k))
            
            t_km1 = t_k
            t_k = t_kp1
            
            xkm1 = xk.copy()
            xk = xkp1.copy()
            
        # 3. Denormalization (Legacy line 112: xk=abs(xk./sqrt(sum(A1.^2,1))'))
        final_x = xp.abs(xk.flatten() / safe_norms)
        
        final_x[~valid_cols] = 0.0
        
        return final_x

    def _precondition_operator(self, operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xp = cp.get_array_module(operator) if HAS_CUPY else np
        col_norms = xp.linalg.norm(operator, axis=0)
        scale = xp.maximum(col_norms, 1e-12)
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
        # Ensure we are working with appropriate backend arrays
        if HAS_CUPY and isinstance(operator, cp.ndarray):
            # If operator is already CuPy, keep it as is.
            # But the logic below uses np.asarray which forces host transfer.
            # We should make this backend-agnostic or handle transfer explicitly.
            # For now, let's pull back to CPU for l1_ls to match original signature if intended,
            # OR better, update this function to support CuPy.
            
            # Since this method seems designed for CPU numpy (using np.*), let's ensure inputs are numpy.
            operator = operator.get()
            
        if HAS_CUPY and isinstance(measurement, cp.ndarray):
            measurement = measurement.get()

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
        Supports both NumPy and CuPy backends.
        """
        xp = cp.get_array_module(operator) if HAS_CUPY else np
        
        measurement = xp.asarray(measurement, dtype=np.float64).reshape(-1)
        
        # Measurement normalization
        m_norm = xp.linalg.norm(measurement)
        if m_norm < 1e-12:
             return xp.zeros(operator.shape[1])
        measurement_scaled = measurement / m_norm

        # Column normalization
        operator_normed, scale = self._precondition_operator(operator)
        hth, hty = self._dense_gram(operator_normed, measurement_scaled)
        
        diff = self._build_tv_difference() if diff_matrix is None else diff_matrix
        # Ensure diff is on the same device
        if xp == cp and not isinstance(diff, cp.ndarray):
            diff = cp.asarray(diff)
        
        if diff.shape[0] == 0:
            reg = xp.eye(hth.shape[0], dtype=np.float64) * float(lambda_value)
            w = xp.linalg.solve(hth + reg, hty)
            return xp.maximum(w * m_norm / scale, 0.0)

        diff_scaled = diff * (1.0 / scale)[None, :]

        # Initialize w using L2 (regularized)
        reg_l2 = xp.eye(hth.shape[0], dtype=np.float64) * float(lambda_value)
        w = xp.linalg.solve(hth + reg_l2, hty)
        w = xp.maximum(w, 0.0)

        eye = xp.eye(hth.shape[0], dtype=np.float64) * 1e-8
        
        for _ in range(n_iter):
            # grad here is TV argument: D * (w/scale)
            grad = diff_scaled @ w
            
            weights = 1.0 / xp.sqrt(grad * grad + eps * eps)
            
            # Reg matrix: D_scaled^T @ W @ D_scaled
            reg = diff_scaled.T @ (weights[:, None] * diff_scaled)
            
            lambda_eff = float(lambda_value)
            
            lhs = hth + lambda_eff * reg + eye
            w = xp.linalg.solve(lhs, hty)
            w = xp.maximum(w, 0.0)
            
        x = w / scale * m_norm
        return x

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
            lambda_value=1e-12,
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


    def _map_volume_to_recon(self, vol: np.ndarray) -> np.ndarray:
        """Map voxel fluence (Z,Y,X) to Recon mesh nodes."""
        from scipy.ndimage import map_coordinates
        nodes = self.solver.mesh.nodes 
        dx = float(self.solver.dx)
        # nodes are (X, Y, Z) in physical space
        # vol is (Z, Y, X)
        # So we need to map:
        # dim 0 of vol <- Z of nodes
        # dim 1 of vol <- Y of nodes
        # dim 2 of vol <- X of nodes
        z_coords = nodes[:, 2] / dx
        y_coords = nodes[:, 1] / dx
        x_coords = nodes[:, 0] / dx
        
        # Stack in order of volume dimensions: (Z, Y, X)
        coords = np.vstack([z_coords, y_coords, x_coords])
        return map_coordinates(vol, coords, order=1, mode='nearest')

    def _map_fluence_to_recon(self, fluence_data: np.ndarray | dict) -> np.ndarray | dict:
        """Map fluence from Forward mesh to Recon mesh if they are different."""
        if self.forward_solver == self.solver:
            return fluence_data


        
        def _map_single(arr: np.ndarray) -> np.ndarray:
            if len(arr) == len(self.solver.mesh.nodes):
                return arr
            elif len(arr) == len(self.forward_solver.mesh.nodes):
                vol = self.forward_solver.sample_to_volume(arr)
                return self._map_volume_to_recon(vol)
            else:
                return arr

        if isinstance(fluence_data, dict):
            return {k: _map_single(v) for k, v in fluence_data.items()}
        else:
            return _map_single(fluence_data)

    def _get_xp(self, arr: np.ndarray | None = None):
        """Returns the array module (numpy or cupy) and moves array to target if needed."""
        gpu_opts = self._get_gpu_options()
        if gpu_opts["enabled"] and HAS_CUPY:
             device = gpu_opts["device_index"]
             with cp.cuda.Device(device):
                 if arr is not None and not isinstance(arr, cp.ndarray):
                     return cp, cp.asarray(arr)
                 return cp, arr
        else:
             if arr is not None and HAS_CUPY and isinstance(arr, cp.ndarray):
                 return np, cp.asnumpy(arr)
             return np, arr

    def _build_roi_mask(self) -> np.ndarray:
        """
        Build a boolean mask of nodes that are within the ROI defined by 'src_range' in config.
        This restricts reconstruction to the permissible region (e.g., inside the brain/organ).
        """
        if "src_range" not in self.config:
            return np.ones(len(self.solver.mesh.nodes), dtype=bool)

        dx = float(self.solver.dx)
        nodes = self.solver.mesh.nodes # (N, 3) in physical coords (mm) X, Y, Z
        
        # src_range is in voxel indices (Z, Y, X order in config usually, but keys are explicit)
        src_z = self.config["src_range"]["z"]
        src_y = self.config["src_range"]["y"]
        src_x = self.config["src_range"]["x"]
        
        # Convert voxel indices to physical coordinates (mm)
        # Range is [start, end)
        z_min, z_max = src_z[0] * dx, src_z[1] * dx
        y_min, y_max = src_y[0] * dx, src_y[1] * dx
        x_min, x_max = src_x[0] * dx, src_x[1] * dx
        
        # Check nodes against bounding box
        # Nodes are (x, y, z)
        mask = (
            (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) &
            (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max) &
            (nodes[:, 2] >= z_min) & (nodes[:, 2] <= z_max)
        )
        
        n_active = np.sum(mask)
        print(f"ROI Restriction: {n_active}/{len(nodes)} nodes active ({n_active/len(nodes):.1%})")
        
        if n_active == 0:
            print("WARNING: ROI mask is empty! Falling back to full mesh.")
            return np.ones(len(nodes), dtype=bool)
            
        return mask

    def reconstruct(
        self,
        sample_id: int,
        method: str = "l1",
    ) -> dict:
        true_source = load_source_pattern_from_sample(self.config, self.sample_dir, sample_id)
        if method == "baseline_v1":
            if self.basis_mode != "fem_node":
                raise ValueError("baseline-v1 仅支持 node-based unknown，请设置 basis_mode=fem_node。")
            if not bool(self.inverse_config.get("disable_excitation_scaling", False)):
                raise ValueError("baseline-v1 为 emission-only，请设置 disable_excitation_scaling=true。")
            source_cc, source_cc_count = ndimage.label(
                np.asarray(true_source > 0.0, dtype=bool),
                structure=np.ones((3, 3, 3), dtype=bool),
            )
            _ = source_cc
            if int(source_cc_count) != 1:
                raise ValueError(f"baseline-v1 当前仅支持单灶，检测到连通域数量={int(source_cc_count)}")
            operator = np.asarray(self.build_operator(), dtype=np.float32)
            # matched/noiseless: y_true 由同一离散定义 x_true 生成，不读取外部投影文件
            x_true = self._map_source_volume_to_basis_coeff(true_source)
            true_source_eval = self.coefficients_to_volume(x_true)
            rhs_full = np.zeros(len(self.solver.mesh.nodes), dtype=np.float64)
            rhs_full[self.basis.node_indices] = x_true
            true_nodal = self.solver.solve_from_rhs(rhs_full)
            measurement = np.asarray(self._observe_solution(true_nodal), dtype=np.float32).reshape(-1)
            row_mask = (np.abs(operator).sum(axis=1) > 1e-12) | (np.abs(measurement) > 1e-12)
            operator_used_np = np.asarray(operator[row_mask], dtype=np.float32)
            measurement_used_np = np.asarray(measurement[row_mask], dtype=np.float32)
            reg_lambda = float(
                self.inverse_config.get(
                    "reg_lambda",
                    self.inverse_config.get("l2_lambda", 1e-3),
                )
            )
            gpu_opts = self._get_gpu_options()
            use_gpu = bool(gpu_opts["enabled"] and HAS_CUPY)
            if use_gpu:
                with cp.cuda.Device(gpu_opts["device_index"]):
                    operator_used = cp.asarray(operator_used_np, dtype=cp.float32)
                    measurement_used = cp.asarray(measurement_used_np, dtype=cp.float32)
                    coeff_dev = self.reconstruct_baseline_v1_pgd(
                        measurement=measurement_used,
                        operator=operator_used,
                        reg_lambda=reg_lambda,
                    )
                    coeff = np.asarray(cp.asnumpy(coeff_dev), dtype=np.float64)
            else:
                coeff = np.asarray(
                    self.reconstruct_baseline_v1_pgd(
                        measurement=measurement_used_np,
                        operator=operator_used_np,
                        reg_lambda=reg_lambda,
                    ),
                    dtype=np.float64,
                )
            y_true_from_x = np.asarray(operator_used_np, dtype=np.float64) @ np.asarray(x_true, dtype=np.float64)
            eps = float(self.inverse_config.get("consistency_eps", 1e-12))
            consistency_rel = float(
                np.linalg.norm(y_true_from_x - np.asarray(measurement_used_np, dtype=np.float64))
                / (np.linalg.norm(np.asarray(measurement_used_np, dtype=np.float64)) + eps)
            )
            consistency_tol = float(self.inverse_config.get("forward_consistency_tol", 1e-5))
            if use_gpu:
                consistency_tol = max(
                    consistency_tol,
                    float(self.inverse_config.get("forward_consistency_tol_fp32", 1e-4)),
                )
            if consistency_rel > consistency_tol:
                raise ValueError(
                    f"baseline-v1 forward consistency failed: rel={consistency_rel:.3e} > tol={consistency_tol:.3e}"
                )
            recon_source = self.coefficients_to_volume(coeff)
            pred_measurement = np.asarray(operator @ coeff, dtype=np.float64).reshape(-1)
            basis_centers = self.basis.node_coords_zyx
            metrics = self._compute_minrfmt_metrics(true_source=true_source_eval, recon_source=recon_source)
            return {
                "sample_id": sample_id,
                "angles": list(self.observation.angles),
                "method": method,
                "operator": operator,
                "row_mask": row_mask,
                "measurement": measurement.astype(np.float64),
                "pred_measurement": pred_measurement,
                "coefficients": coeff,
                "true_source": true_source_eval,
                "true_source_raw": true_source,
                "recon_source": recon_source,
                "true_nodal": true_nodal,
                "recon_nodal": None,
                "basis_centers_zyx": basis_centers,
                "observation": self.observation,
                "reg_lambda": reg_lambda,
                "forward_consistency_relative_error": consistency_rel,
                "forward_consistency_tolerance": consistency_tol,
                "reconstruction_failure": bool(metrics["reconstruction_failure"]),
                "metrics": metrics,
            }
        
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        true_nodal = None
        is_on_recon_mesh = False
        
        # Apply ROI restriction if enabled (implicit for most methods to improve sparsity/accuracy)
        # We can pass this mask to solvers or restrict operator beforehand.
        roi_mask = self._build_roi_mask()
        
        # Determine if we have a separate recon mesh
        # if self.recon_mesh_cache_path and (self.recon_mesh_cache_path != self.mesh_cache_path):
        
        def load_mcx_volume(path):
            try:
                import jdata as jd
                data = jd.loadjd(str(path))
                vol = data["NIFTIData"]
                if vol.ndim == 5: vol = vol[:,:,:,0,0]
                # MCX is (X,Y,Z). Convert to (Z,Y,X) for FEM compatibility
                return vol.transpose(2, 1, 0)
            except Exception as e:
                print(f"Warning: Failed to load MCX data from {path}: {e}")
                return None

        if epi_mode:
            true_nodal = {}
            loaded_all = True
            for angle in self.angles:
                # Try loading Nodal Fluence (Preferred if mesh matches)
                nodal_path = Path(self.sample_dir) / f"fem_nodal_fluence_angle_{angle}.npy"
                volume_path = Path(self.sample_dir) / f"fem_volume_fluence_angle_{angle}.npy"
                
                loaded_nodal = None
                if nodal_path.exists():
                    try:
                        temp_nodal = np.load(nodal_path)
                        # Check compatibility with current solver mesh
                        if len(temp_nodal) == len(self.solver.mesh.nodes):
                            loaded_nodal = temp_nodal
                        else:
                            print(f"Warning: Nodal fluence size {len(temp_nodal)} != Recon mesh size {len(self.solver.mesh.nodes)}. Trying volume mapping.")
                    except Exception as e:
                        print(f"Error loading nodal fluence: {e}")

                if loaded_nodal is None:
                    # Fallback to Volume Fluence
                    if volume_path.exists():
                         try:
                             vol = np.load(volume_path)
                             # Map Volume -> Recon Mesh Nodes
                             loaded_nodal = self._map_volume_to_recon(vol)
                         except Exception as e:
                             print(f"Error loading/mapping volume fluence: {e}")
                    
                    # Fallback to MCX Volume if FEM volume missing
                    else:
                         mcx_path = Path(self.sample_dir) / f"{sample_id}.jnii"
                         if angle == 0 and mcx_path.exists():
                             vol = load_mcx_volume(mcx_path)
                             if vol is not None:
                                 # DEBUG: Check Center of Mass
                                 # from scipy.ndimage import center_of_mass
                                 # cm = center_of_mass(vol)
                                 # print(f"DEBUG: MCX Volume Center (Z,Y,X): {cm}")
                                 loaded_nodal = self._map_volume_to_recon(vol)
                                 
                                 # DEBUG: Check Nodal Center
                                 # weighted_center = np.average(self.solver.mesh.nodes, axis=0, weights=loaded_nodal)
                                 # print(f"DEBUG: Mapped Nodal Center (X,Y,Z): {weighted_center}")

                if loaded_nodal is not None:
                    # DEBUG: Check mapped nodal center
                    if isinstance(loaded_nodal, dict):
                        chk_arr = list(loaded_nodal.values())[0]
                    else:
                        chk_arr = loaded_nodal
                    
                    recon_nodes = self.solver.mesh.nodes
                    w = np.abs(chk_arr)
                    w_sum = w.sum()
                    if w_sum > 1e-12:
                        com = (recon_nodes * w[:, None]).sum(axis=0) / w_sum
                        print(f"DEBUG: Mapped Nodal Fluence Center (Recon Coords XYZ): {com}")
                        
                        # Check original volume if available
                        vol_path = Path(self.sample_dir) / f"fem_volume_fluence_angle_{angle}.npy"
                        if vol_path.exists():
                            v_chk = np.load(vol_path)
                            c_v = _weighted_center(v_chk) # ZYX voxels
                            dx = self.solver.dx
                            # ZYX -> XYZ physical: x=X*dx, y=Y*dx, z=Z*dx.
                            # Index 0 is Z. Index 1 is Y. Index 2 is X.
                            p_v = np.array([c_v[2]*dx, c_v[1]*dx, c_v[0]*dx])
                            print(f"DEBUG: Volume Fluence Center (Physical XYZ): {p_v}")
                            print(f"DEBUG: Diff (XYZ): {p_v - com}")
                    else:
                        print("DEBUG: Mapped Nodal Fluence is empty/zero!")

                    true_nodal[angle] = loaded_nodal
                else:
                    loaded_all = False
                    print(f"Warning: No valid fluence found for angle {angle}")
            
            if not loaded_all:
                print("Warning: Pre-computed nodal fluence not found for all angles. Re-solving (using Recon solver).")
                true_nodal = self.solve_source(true_source)
                is_on_recon_mesh = True
        else:
             p = Path(self.sample_dir) / "fem_nodal_fluence.npy"
             mcx_path = Path(self.sample_dir) / f"{sample_id}.jnii"
             
             if p.exists():
                 true_nodal = np.load(p)
             elif mcx_path.exists():
                 # print(f"Loading MCX volume from {mcx_path}...")
                 vol = load_mcx_volume(mcx_path)
                 if vol is not None:
                     true_nodal = self._map_volume_to_recon(vol)
                     is_on_recon_mesh = True # Mapped to recon mesh
                 else:
                     true_nodal = self.solve_source(true_source)
                     is_on_recon_mesh = True
             else:
                 true_nodal = self.solve_source(true_source)
                 is_on_recon_mesh = True

        # Map to Recon Mesh if needed (and not already mapped/solved on recon)
        # Note: In Epi-mode, true_nodal is a dict {angle: array}. _map_fluence_to_recon handles dict.
        
        if not is_on_recon_mesh:
             true_nodal_recon = self._map_fluence_to_recon(true_nodal)
        else:
            true_nodal_recon = true_nodal
        
        measurement = self._observe_solution(true_nodal_recon)
        operator = self.build_operator()
        
        # --- ROI Restriction ---
        roi_mask = self._build_roi_mask()
        is_roi_active = np.sum(roi_mask) < len(roi_mask)
        
        # Check if operator matches full mesh (N=43912) or already reduced (N=1000)
        # If basis_mode is 'fem_node', operator cols == nodes.
        # If basis_mode is 'voxel_block', operator cols == blocks.
        
        if is_roi_active:
            if operator.shape[1] == len(roi_mask):
                print(f"Applying ROI restriction to operator columns: {operator.shape} -> ", end="")
                operator_used_cols = operator[:, roi_mask]
                print(f"{operator_used_cols.shape}")
            else:
                print(f"Warning: Operator columns ({operator.shape[1]}) != Mesh nodes ({len(roi_mask)}). Skipping ROI restriction (Assuming reduced basis).")
                operator_used_cols = operator
                is_roi_active = False # Disable ROI mapping back if we didn't slice
        else:
            operator_used_cols = operator

        # --- Row Selection ---
        row_mask = (np.abs(operator_used_cols).sum(axis=1) > 1e-12) | (np.abs(measurement) > 1e-12)
        operator_used = operator_used_cols[row_mask]
        measurement_used = measurement[row_mask]

        # Debug Scaling
        y_norm = np.linalg.norm(measurement)
        h_col_norms = np.linalg.norm(operator_used_cols, axis=0) # Use ROI-restricted cols for stats
        h_mean_norm = np.mean(h_col_norms)

        # GPU Handling
        gpu_opts = self._get_gpu_options()
        use_gpu = gpu_opts["enabled"] and HAS_CUPY
        
        if use_gpu:
            device = gpu_opts["device_index"]
            with cp.cuda.Device(device):
                measurement_used = cp.asarray(measurement_used)
                operator_used = cp.asarray(operator_used)
                print(f"DEBUG: Using GPU {device} (CuPy) for reconstruction.")

        if method == "l2":
            lambda_value = float(self.inverse_config.get("l2_lambda", 1e-3))
            coeff_roi = self.reconstruct_l2(measurement_used, operator_used, lambda_value=lambda_value)
        elif method == "legacy_l1":
            coeff_roi = self.reconstruct_legacy_l1(measurement_used, operator_used)
        elif method == "legacy_tv":
            coeff_roi = self.reconstruct_legacy_tv(measurement_used, operator_used)
        elif method == "l1_ls":
            coeff_roi = self.reconstruct_l1_ls(
                measurement_used,
                operator_used,
                lambda_value=float(self.inverse_config.get("l1_lambda", 1e-4)),
                max_iter=int(self.inverse_config.get("l1_max_iter", 200)),
            )
        elif method == "tv":
            coeff_roi = self.reconstruct_tv(
                measurement_used,
                operator_used,
                lambda_value=float(self.inverse_config.get("tv_lambda", 0.05)),
                n_iter=int(self.inverse_config.get("tv_irls_iter", 15)),
                eps=float(self.inverse_config.get("tv_eps", 1e-3)),
            )
        else:
             # Fallback for other methods not explicitly listed above but possibly valid
             if hasattr(self, f"reconstruct_{method}"):
                 func = getattr(self, f"reconstruct_{method}")
                 coeff_roi = func(measurement_used, operator_used)
             else:
                 raise ValueError(f"Unknown method: {method}")

        # Map back to full mesh if ROI was used
        # Handle coeff vs coeff_roi variable naming consistency
        
        # Helper to get numpy array
        if use_gpu and hasattr(coeff_roi, "get"):
             coeff_val = coeff_roi.get()
        else:
             coeff_val = coeff_roi
             
        if is_roi_active:
            coeff = np.zeros(len(self.solver.mesh.nodes), dtype=np.float64)
            coeff[roi_mask] = coeff_val
        else:
            coeff = coeff_val

        recon_vol = self.coefficients_to_volume(coeff)

        if use_gpu:
             coeff = cp.asnumpy(coeff)

        recon_source = self.coefficients_to_volume(coeff)
        recon_nodal = self.solve_source(recon_source)
        if isinstance(recon_nodal, dict):
             # Handle dict
             rn_max = max([v.max() for v in recon_nodal.values()]) if recon_nodal else 0
             # print(f"DEBUG: Recon Nodal Max: {rn_max}")
        else:
             # print(f"DEBUG: Recon Nodal Max: {recon_nodal.max()}")
             pass
             
        pred_measurement = self._observe_solution(recon_nodal)
        y = np.asarray(measurement).reshape(-1, 1)
        # print(f"DEBUG: y max: {y.max()}, y norm: {np.linalg.norm(y)}")
        # print(f"DEBUG: pred max: {pred_measurement.max()}, pred norm: {np.linalg.norm(pred_measurement)}")
        # print(f"DEBUG: recon_source max: {recon_source.max()}")

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
    if "true_source_raw" in result:
        np.save(out_path / "true_source_raw.npy", result["true_source_raw"])
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
    sample_id = int(result["sample_id"])
    method = str(result["method"])
    summary = {
        "sample_id": sample_id,
        "angles": [int(v) for v in result["angles"]],
        "method": method,
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
    if "forward_consistency_relative_error" in result:
        summary["forward_consistency_relative_error"] = float(result["forward_consistency_relative_error"])
    if "forward_consistency_tolerance" in result:
        summary["forward_consistency_tolerance"] = float(result["forward_consistency_tolerance"])
    if "reg_lambda" in result:
        summary["reg_lambda"] = float(result["reg_lambda"])
    if observation.observation_type == "legacy_surface_nodes":
        summary["n_surface_nodes"] = int(measurement.size)
    else:
        summary["n_visible_patches"] = int(measurement.size)
    if method == "baseline_v1" and "metrics" in result:
        metrics = dict(result["metrics"])
        summary["metrics"] = metrics
        summary["reconstruction_failure"] = bool(metrics.get("reconstruction_failure", False))
        with open(out_path / "inverse_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        keys = [
            "sample_id",
            "pred_regions",
            "gt_regions",
            "dice",
            "precision",
            "recall",
            "mr",
            "ms",
            "delta_cc",
            "assd",
            "hd95",
            "psnr",
            "ssim",
            "warning",
        ]
        row = {
            "sample_id": sample_id,
            "pred_regions": metrics.get("pred_regions"),
            "gt_regions": metrics.get("gt_regions", 1),
            "dice": metrics.get("dice"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mr": metrics.get("mr"),
            "ms": metrics.get("ms"),
            "delta_cc": metrics.get("delta_cc"),
            "assd": metrics.get("assd"),
            "hd95": metrics.get("hd95"),
            "psnr": None,
            "ssim": None,
            "warning": "reconstruction_failure" if metrics.get("reconstruction_failure", False) else None,
        }
        with open(out_path / "metrics_per_sample.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerow({k: row.get(k) for k in keys})
        summary_rows = []
        for group in ("1", "2", "3", "overall"):
            if group == "1":
                count = 1
                dsc = row["dice"]
                precision = row["precision"]
                recall = row["recall"]
                assd = row["assd"]
                hd95 = row["hd95"]
                mr = row["mr"]
                ms = row["ms"]
                delta_cc = row["delta_cc"]
            else:
                count = 0
                dsc = precision = recall = assd = hd95 = mr = ms = delta_cc = None
            summary_rows.append(
                {
                    "group": group,
                    "count": count,
                    "dsc": dsc,
                    "precision": precision,
                    "recall": recall,
                    "assd": assd,
                    "hd95": hd95,
                    "mr": mr,
                    "ms": ms,
                    "delta_cc": delta_cc,
                }
            )
        with open(out_path / "summary_by_num_lights.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group", "count", "dsc", "precision", "recall", "assd", "hd95", "mr", "ms", "delta_cc"],
            )
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        with open(out_path / "summary_by_num_lights.json", "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        _save_visualization(result, out_path)
        return

    # fallback: 保持旧路径兼容
    center_true = np.array(summary["true_source_center_zyx"])
    center_recon = np.array(summary["recon_source_center_zyx"])
    localization_error = np.linalg.norm(center_true - center_recon)
    quantitativeness = float(result["recon_source"].max() / max(result["true_source"].max(), 1e-12))
    true_norm = result["true_source"] / max(result["true_source"].max(), 1e-12)
    recon_norm = result["recon_source"] / max(result["recon_source"].max(), 1e-12)
    true_vol = (true_norm > 0.5).sum()
    recon_vol = (recon_norm > 0.5).sum()
    volume_ratio = float(recon_vol / max(true_vol, 1.0))
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
        v1_norm = v1 / (v1.max() + 1e-12)
        v2_norm = v2 / (v2.max() + 1e-12)
        m1 = v1_norm > threshold
        m2 = v2_norm > threshold
        if m1.sum() == 0 and m2.sum() == 0:
            return 0.0
        intersection = np.logical_and(m1, m2).sum()
        return 2.0 * intersection / (m1.sum() + m2.sum() + 1e-12)

    dice_50 = _dice(result["true_source"], result["recon_source"], 0.5)
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
