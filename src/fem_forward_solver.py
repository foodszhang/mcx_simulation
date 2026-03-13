from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import factorized, splu, spsolve
from scipy.sparse.linalg import MatrixRankWarning

from src.fem_mesh import (
    NirfastStyleMesh,
    attach_optical_properties,
    build_tetrahedral_mesh_from_volume,
    downsample_binary_max,
    load_mesh,
    save_mesh,
)
from src.forward_solver import VolumeReader


class FemDiffusionSolver:
    """基于保结构四面体网格的扩散近似 FEM 前向求解器。"""

    def __init__(
        self,
        volume_path: str,
        media_yaml_path: str,
        volume_shape: tuple[int, int, int],
        dx: float = 0.1,
        mesh_cache_path: str | None = None,
        fem_config: dict | None = None,
    ):
        self.volume_path = volume_path
        self.media_yaml_path = media_yaml_path
        self.volume_shape = tuple(volume_shape)
        self.dx = float(dx)
        self.fem_config = fem_config or {}
        self.mesh_cache_path = mesh_cache_path

        self.volume_data = VolumeReader(volume_path, self.volume_shape).read_volume()
        with open(media_yaml_path, "r", encoding="utf-8") as f:
            self.media_list = yaml.safe_load(f)

        self.mesh = self._load_or_build_mesh()
        
        # Build cell lookup if cell_coords is available (for voxel-based meshes)
        self.cell_lookup = {}
        if self.mesh.cell_coords is not None and self.mesh.cell_nodes is not None:
            self.cell_lookup = {
                tuple(coord.tolist()): nodes
                for coord, nodes in zip(self.mesh.cell_coords, self.mesh.cell_nodes)
            }
            
        self.system_matrix: csr_matrix | None = None
        self._linear_solver = None
        self._lu_solver = None

    def _load_or_build_mesh(self) -> NirfastStyleMesh:
        # Check if external mesh is configured
        # Note: self.fem_config structure might be nested. 
        # In config_legacy.yaml it is: fem: mesh: { method: external_mesh, path: ... }
        # The __init__ passes config.get("fem", {}) as fem_config.
        # So self.fem_config has "mesh" key.
        
        mesh_config = self.fem_config.get("mesh", {})
        mesh_method = mesh_config.get("method", "voxel_tet")
        
        if mesh_method == "external_mesh":
            mesh_path = mesh_config.get("path")
            if mesh_path and os.path.exists(mesh_path):
                # Use load_mesh to load external mesh (assuming it's .npz)
                mesh = load_mesh(mesh_path)
                return attach_optical_properties(mesh, self.media_list)
            else:
                raise FileNotFoundError(f"External mesh not found at: {mesh_path}")

        if self.mesh_cache_path and os.path.exists(self.mesh_cache_path):
            mesh = load_mesh(self.mesh_cache_path)
        else:
            mesh = build_tetrahedral_mesh_from_volume(
                self.volume_data,
                dx=self.dx,
                mesh_config=mesh_config,
            )
            if self.mesh_cache_path:
                save_mesh(mesh, self.mesh_cache_path)
        return attach_optical_properties(mesh, self.media_list)

    def build_system_matrix(self) -> csr_matrix:
        if self.system_matrix is not None:
            return self.system_matrix

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for elem_idx, element in enumerate(self.mesh.elements):
            self._current_element_idx = elem_idx
            points = self.mesh.nodes[element]
            local_matrix = self._assemble_element_matrix(points, element)
            for i_local, i_global in enumerate(element):
                for j_local, j_global in enumerate(element):
                    value = local_matrix[i_local, j_local]
                    if value != 0.0:
                        rows.append(int(i_global))
                        cols.append(int(j_global))
                        data.append(float(value))

        for face in self.mesh.boundary_faces:
            face_matrix = self._assemble_boundary_face_matrix(face)
            for i_local, i_global in enumerate(face):
                for j_local, j_global in enumerate(face):
                    value = face_matrix[i_local, j_local]
                    if value != 0.0:
                        rows.append(int(i_global))
                        cols.append(int(j_global))
                        data.append(float(value))

        n_nodes = len(self.mesh.nodes)
        self.system_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.float64,
        ).tocsr()
        diagonal_shift = float(self.fem_config.get("diagonal_regularization", 1e-8))
        self.system_matrix = self.system_matrix + sparse_eye(n_nodes, format="csr") * diagonal_shift
        return self.system_matrix

    def _assemble_element_matrix(
        self,
        points: np.ndarray,
        element: np.ndarray,
    ) -> np.ndarray:
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]
        x4, y4, z4 = points[3]
        transform = np.array(
            [
                [1.0, x1, y1, z1],
                [1.0, x2, y2, z2],
                [1.0, x3, y3, z3],
                [1.0, x4, y4, z4],
            ],
            dtype=np.float64,
        )
        det_transform = np.linalg.det(transform)
        volume = abs(det_transform) / 6.0
        if volume < 1e-12:
            return np.zeros((4, 4), dtype=np.float64)

        inv_transform = np.linalg.inv(transform)
        grads = inv_transform[1:, :]

        element_kappa = float(self.mesh.element_kappa[self._current_element_idx])
        element_mua = float(self.mesh.element_mua[self._current_element_idx])

        stiffness = element_kappa * volume * (grads.T @ grads)
        mass = element_mua * volume / 20.0 * np.array(
            [
                [2.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        return stiffness + mass

    def _assemble_boundary_face_matrix(self, face: np.ndarray) -> np.ndarray:
        face_points = self.mesh.nodes[face]
        v1 = face_points[1] - face_points[0]
        v2 = face_points[2] - face_points[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        if area < 1e-12:
            return np.zeros((3, 3), dtype=np.float64)
        face_ksi = float(np.mean(self.mesh.ksi[face]))
        return face_ksi * area / 12.0 * np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )

    def build_source_vector(self, source_pattern: np.ndarray, normalize: bool = True) -> np.ndarray:
        n_nodes = len(self.mesh.nodes)
        rhs = np.zeros(n_nodes, dtype=np.float64)
        
        # If we have cell_lookup (voxel-based mesh), use it for efficiency
        if self.cell_lookup:
            target_shape = tuple(int(v) for v in self.mesh.voxel_shape.tolist())
            source_reduced = downsample_binary_max(source_pattern, target_shape)
            active_cells = np.argwhere(source_reduced > 0.5)

            for coord in active_cells:
                nodes = self.cell_lookup.get(tuple(coord.tolist()))
                if nodes is None:
                    continue
                rhs[nodes] += 1.0 / len(nodes)
        else:
            # For unstructured/external mesh, interpolate source volume onto mesh nodes
            # nodes (N,3) in physical coords (mm)
            # source_pattern in voxel coords (indices)
            
            # Map physical coords to voxel coords
            # Assuming origin is (0,0,0) and spacing is dx
            dx = self.dx
            node_voxel_coords = self.mesh.nodes[:, ::-1] / dx  # XYZ -> ZYX?
            # Wait, mesh.nodes is usually XYZ (since it comes from tetgen/vtk)
            # Volume is ZYX (since it comes from numpy/nibabel)
            # My mesh_generator.py: nodes = np.column_stack([zyx[:,2]*s[2], zyx[:,1]*s[1], zyx[:,0]*s[0]]) -> X, Y, Z
            # So mesh.nodes is X, Y, Z.
            # Volume is Z, Y, X.
            # So to map nodes to volume indices:
            # z_idx = z_mm / dx
            # y_idx = y_mm / dx
            # x_idx = x_mm / dx
            # map_coordinates expects coordinates in (dim0, dim1, dim2) order i.e. (z, y, x)
            
            x_mm = self.mesh.nodes[:, 0]
            y_mm = self.mesh.nodes[:, 1]
            z_mm = self.mesh.nodes[:, 2]
            
            z_idx = z_mm / dx
            y_idx = y_mm / dx
            x_idx = x_mm / dx
            
            # Interpolate source value at each node
            # order=1 (linear) or 0 (nearest)? Since source is binary/sharp, 0 might be better or 1 for smoothing.
            # Let's use 1 to allow sub-voxel source distribution.
            rhs = map_coordinates(
                source_pattern, 
                np.vstack((z_idx, y_idx, x_idx)), 
                order=1, 
                mode='nearest'
            ).astype(np.float64)

        rhs_sum = rhs.sum()
        if rhs_sum <= 1e-12:
            # Fallback if source didn't hit any node
            if self.cell_lookup:
                 target_shape = tuple(int(v) for v in self.mesh.voxel_shape.tolist())
                 center_coord = np.array(target_shape, dtype=np.int32) // 2
                 nearest_key = min(
                     self.cell_lookup.keys(),
                     key=lambda key: np.linalg.norm(np.array(key, dtype=np.float64) - center_coord),
                 )
                 nodes = self.cell_lookup[nearest_key]
                 rhs[nodes] += 1.0 / len(nodes)
            else:
                 # Find nearest node to volume center
                 vol_center_idx = np.array(source_pattern.shape) / 2.0
                 vol_center_mm = vol_center_idx[::-1] * self.dx  # ZYX -> XYZ
                 dists = np.linalg.norm(self.mesh.nodes - vol_center_mm, axis=1)
                 nearest_node = np.argmin(dists)
                 rhs[nearest_node] = 1.0

        elif normalize:
            rhs /= rhs_sum
        return rhs

    def _get_linear_solver(self):
        if self._linear_solver is not None:
            return self._linear_solver

        matrix = self.build_system_matrix()
        try:
            self._lu_solver = splu(matrix.tocsc())
            self._linear_solver = self._lu_solver.solve
        except Exception:
            diagonal_shift = float(self.fem_config.get("fallback_regularization", 1e-6))
            stabilized = matrix + sparse_eye(matrix.shape[0], format="csr") * diagonal_shift
            self._lu_solver = splu(stabilized.tocsc())
            self._linear_solver = self._lu_solver.solve
        return self._linear_solver

    def _get_lu_solver(self):
        if self._lu_solver is not None:
            return self._lu_solver

        matrix = self.build_system_matrix()
        try:
            self._lu_solver = splu(matrix.tocsc())
            self._linear_solver = self._lu_solver.solve
        except Exception:
            diagonal_shift = float(self.fem_config.get("fallback_regularization", 1e-6))
            stabilized = matrix + sparse_eye(matrix.shape[0], format="csr") * diagonal_shift
            self._lu_solver = splu(stabilized.tocsc())
            self._linear_solver = self._lu_solver.solve
        return self._lu_solver

    def solve_from_rhs(self, rhs: np.ndarray) -> np.ndarray:
        solver = self._get_linear_solver()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solution = solver(np.asarray(rhs, dtype=np.float64))
            except MatrixRankWarning:
                diagonal_shift = float(self.fem_config.get("fallback_regularization", 1e-6))
                matrix = self.build_system_matrix()
                stabilized = matrix + sparse_eye(matrix.shape[0], format="csr") * diagonal_shift
                self._lu_solver = splu(stabilized.tocsc())
                self._linear_solver = self._lu_solver.solve
                solution = self._linear_solver(np.asarray(rhs, dtype=np.float64))
        return np.asarray(solution, dtype=np.float64)

    def solve(self, source_pattern: np.ndarray, normalize: bool = True) -> np.ndarray:
        rhs = self.build_source_vector(source_pattern, normalize=normalize)
        return self.solve_from_rhs(rhs)

    def solve_excitation(
        self,
        source_pos: tuple[float, float, float],
        radius: float = 0.0,
    ) -> np.ndarray:
        """
        求解激发光场 (Excitation Field)。
        在指定坐标处设置光源。
        radius: 光源半径 (mm)。如果 > 0，则将能量均匀分布在半径内的所有节点上。
        """
        n_nodes = len(self.mesh.nodes)
        rhs = np.zeros(n_nodes, dtype=np.float64)
        
        # Find nearest node or nodes within radius
        diff = self.mesh.nodes - np.array(source_pos, dtype=np.float64)
        dist_sq = np.sum(diff**2, axis=1)
        
        if radius > 1e-6:
            # Finite spot size
            mask = dist_sq <= (radius**2)
            if np.any(mask):
                rhs[mask] = 1.0 / np.sum(mask)
            else:
                # Fallback to nearest if no nodes within radius
                nearest_idx = np.argmin(dist_sq)
                rhs[nearest_idx] = 1.0
        else:
            # Point source
            nearest_idx = np.argmin(dist_sq)
            rhs[nearest_idx] = 1.0
            
        return self.solve_from_rhs(rhs)

    def solve_multiple_rhs(self, rhs_matrix: np.ndarray) -> np.ndarray:
        lu_solver = self._get_lu_solver()
        rhs_matrix = np.asarray(rhs_matrix, dtype=np.float64)
        if rhs_matrix.ndim != 2:
            raise ValueError("rhs_matrix 必须是二维数组，形状为 [n_nodes, n_rhs]。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solution = lu_solver.solve(rhs_matrix)
            except MatrixRankWarning:
                diagonal_shift = float(self.fem_config.get("fallback_regularization", 1e-6))
                matrix = self.build_system_matrix()
                stabilized = matrix + sparse_eye(matrix.shape[0], format="csr") * diagonal_shift
                self._lu_solver = splu(stabilized.tocsc())
                self._linear_solver = self._lu_solver.solve
                solution = self._lu_solver.solve(rhs_matrix)
        return np.asarray(solution, dtype=np.float64)

    def sample_to_volume(self, nodal_fluence: np.ndarray) -> np.ndarray:
        sampled = np.zeros(self.volume_shape, dtype=np.float32)
        
        # Case 1: Voxel-aligned mesh (Legacy)
        if self.mesh.voxel_shape is not None and self.mesh.cell_coords is not None:
            factor = int(self.fem_config.get("mesh", {}).get("downsample_factor", 8))
            coarse = np.zeros(tuple(int(v) for v in self.mesh.voxel_shape.tolist()), dtype=np.float32)
            for coord, nodes in zip(self.mesh.cell_coords, self.mesh.cell_nodes):
                coarse[coord[0], coord[1], coord[2]] = float(np.mean(nodal_fluence[nodes]))

            for z in range(coarse.shape[0]):
                z0, z1 = z * factor, min((z + 1) * factor, self.volume_shape[0])
                for y in range(coarse.shape[1]):
                    y0, y1 = y * factor, min((y + 1) * factor, self.volume_shape[1])
                    for x in range(coarse.shape[2]):
                        x0, x1 = x * factor, min((x + 1) * factor, self.volume_shape[2])
                        sampled[z0:z1, y0:y1, x0:x1] = coarse[z, y, x]
            return sampled

        # Case 2: Unstructured/External mesh
        # Fallback to Nearest Neighbor Interpolation (using KDTree if available or scipy)
        # For efficiency, we only interpolate onto a grid that covers the mesh bounding box
        from scipy.interpolate import griddata
        
        # Create grid coordinates (Z, Y, X)
        # This is very memory intensive for full volume. 
        # Optimize: Only interpolate inside mesh bbox.
        
        nodes = self.mesh.nodes # (N, 3) in mm [x, y, z] usually? 
        # Wait, build_source_vector used: idx = pos / dx. 
        # So mesh nodes are in mm.
        # volume indices are Z, Y, X.
        # Mesh nodes are X, Y, Z (mcx convention) or Z, Y, X?
        # In fem_mesh.py, we usually assume [x, y, z].
        
        min_b = np.min(nodes, axis=0)
        max_b = np.max(nodes, axis=0)
        
        # Convert bbox to indices
        dx = self.dx
        min_idx = np.floor(min_b / dx).astype(int)
        max_idx = np.ceil(max_b / dx).astype(int)
        
        # Clip to volume
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array(self.volume_shape)[[2, 1, 0]]) # vol shape is ZYX, idx is XYZ
        
        # Generate grid points in the bounding box
        # Mesh nodes are (x, y, z). Grid points should be (x, y, z).
        x_range = np.arange(min_idx[0], max_idx[0])
        y_range = np.arange(min_idx[1], max_idx[1])
        z_range = np.arange(min_idx[2], max_idx[2])
        
        if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
             return sampled

        # Create grid (mesh grid)
        # Note: griddata expects points as (N, D).
        # We want to interpolate nodal_fluence (defined at nodes) to grid points.
        
        # Optimization: griddata with method='nearest' uses KDTree.
        # We can construct the grid points directly.
        # grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        # This produces X, Y, Z ordering.
        # grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]) * dx
        
        # values = griddata(nodes, nodal_fluence, grid_points, method='nearest')
        
        # Fill sampled array
        # sampled[z, y, x] = value
        # meshgrid 'ij' -> (nx, ny, nz)
        # sampled expects (nz, ny, nx)
        
        # Let's use coordinate arrays
        grid_z, grid_y, grid_x = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        # Convert to physical coordinates (x, y, z)
        # grid_x is (nz, ny, nx)
        query_points = np.column_stack([
            grid_x.ravel() * dx,
            grid_y.ravel() * dx,
            grid_z.ravel() * dx
        ])
        
        values = griddata(nodes, nodal_fluence, query_points, method='nearest', fill_value=0.0)
        
        # Reshape to (nz, ny, nx)
        sub_volume = values.reshape(len(z_range), len(y_range), len(x_range))
        
        # Place into full volume
        sampled[min_idx[2]:max_idx[2], min_idx[1]:max_idx[1], min_idx[0]:max_idx[0]] = sub_volume
        
        return sampled


class BatchFemForwardSolver:
    """批量 FEM 前向求解：输出节点光通量与网格文件。"""

    def __init__(self, config: dict, output_base_dir: str):
        self.config = config
        self.output_base_dir = output_base_dir
        fem_cfg = config.get("fem", {})
        mesh_dir = Path(output_base_dir) / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_cache_name = fem_cfg.get("mesh_cache_name", "brain_fem_mesh.npz")
        self.mesh_cache_path = str(mesh_dir / mesh_cache_name)
        self.solver = FemDiffusionSolver(
            volume_path=config["generated_bin_path"],
            media_yaml_path=config["generated_material_yaml_path"],
            volume_shape=tuple(config["volume_shape"]),
            dx=config.get("lengthunit", 0.1),
            mesh_cache_path=self.mesh_cache_path,
            fem_config=fem_cfg,
        )
        self._save_shared_mesh_files(mesh_dir)

    def _save_shared_mesh_files(self, mesh_dir: Path) -> None:
        np.save(mesh_dir / "nodes.npy", self.solver.mesh.nodes)
        np.save(mesh_dir / "elements.npy", self.solver.mesh.elements)
        np.save(mesh_dir / "region.npy", self.solver.mesh.region)
        if self.solver.mesh.surface_faces is not None:
            np.save(mesh_dir / "surface_faces.npy", self.solver.mesh.surface_faces)
        if self.solver.mesh.surface_face_regions is not None:
            np.save(mesh_dir / "surface_face_regions.npy", self.solver.mesh.surface_face_regions)
        mesh_info = {
            "n_nodes": int(len(self.solver.mesh.nodes)),
            "n_elements": int(len(self.solver.mesh.elements)),
            "n_boundary_faces": int(len(self.solver.mesh.boundary_faces)),
            "n_surface_faces": int(len(self.solver.mesh.surface_faces) if self.solver.mesh.surface_faces is not None else 0),
            "dimension": int(self.solver.mesh.dimension),
        }
        with open(mesh_dir / "mesh_info.json", "w", encoding="utf-8") as f:
            json.dump(mesh_info, f, indent=2, ensure_ascii=False)

    def compute_rotated_source_pos(self, angle: float) -> tuple[float, float, float]:
        """计算围绕体积中心旋转后的光源位置 (Epi-mode)。"""
        base_pos = np.array(self.config["excitation_source"]["position"], dtype=np.float64)
        volume_shape = np.array(self.config["volume_shape"], dtype=np.float64)
        dx = float(self.config.get("lengthunit", 0.1))
        # Volume center in physical coords (XYZ)
        # Note: volume_shape is ZYX. center = (X_dim, Y_dim, Z_dim) / 2
        center = (volume_shape[::-1] * dx) / 2.0
        
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        
        # Rotation around Y axis
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        
        v = base_pos - center
        v_rot = R @ v
        new_pos = center + v_rot
        return tuple(new_pos)

    def solve_batch(self) -> None:
        epi_mode = self.config.get("fem", {}).get("epi_mode", False)
        # If epi_mode is true, we use projection angles. If not, just angle 0 (fixed source).
        angles = self.config.get("projection", {}).get("angles", [0]) if epi_mode else [0]
        if not angles:
            angles = [0]
            
        logger = logging.getLogger(__name__)
        
        for entry in sorted(os.listdir(self.output_base_dir)):
            config_subdir = os.path.join(self.output_base_dir, entry)
            if not (os.path.isdir(config_subdir) and entry.isdigit()):
                continue
            
            # Load Tumor Mask (as source pattern)
            source_pattern = self._reconstruct_source_pattern(int(entry), config_subdir)
            
            # Build RHS for the tumor (assuming unit concentration 1.0)
            # normalize=False ensures we keep the "mass" proportional to volume
            tumor_rhs_raw = self.solver.build_source_vector(source_pattern, normalize=False)
            
            for angle in angles:
                suffix = f"_angle_{angle}" if epi_mode else ""
                
                if epi_mode:
                    ex_pos = self.compute_rotated_source_pos(angle)
                    # Solve Excitation Field
                    # radius=0.5mm to simulate a small spot source
                    phi_ex = self.solver.solve_excitation(ex_pos, radius=0.5)
                    
                    # Save Excitation Field (Optional, useful for inverse/debug)
                    # np.save(os.path.join(config_subdir, f"fem_excitation_fluence{suffix}.npy"), phi_ex)
                    
                    # Compute Effective Source: Tumor Amount * Excitation Field
                    # Approximation: element-wise product at nodes
                    effective_rhs = tumor_rhs_raw * phi_ex
                    
                    if np.sum(effective_rhs) < 1e-12:
                        logger.warning(f"Sample {entry} Angle {angle}: Effective source is zero (Excitation didn't hit tumor).")
                else:
                    # Transmission / Legacy Mode
                    # Here we assume the source_pattern IS the effective source (Uniform Excitation)
                    # Or we could implement fixed excitation here too.
                    # For backward compatibility, keep as is (Unit Strength).
                    effective_rhs = tumor_rhs_raw
                
                # Solve Emission Field
                nodal_fluence = self.solver.solve_from_rhs(effective_rhs)
                
                self._save_sample_outputs(entry, config_subdir, nodal_fluence, suffix)

    def _reconstruct_source_pattern(self, config_idx: int, config_subdir: str) -> np.ndarray:
        volume_shape = tuple(self.config["volume_shape"])
        src_range_z = self.config["src_range"]["z"]
        src_range_y = self.config["src_range"]["y"]
        src_range_x = self.config["src_range"]["x"]
        source_filename = f"source-{config_idx}.bin"
        source_path = os.path.join(config_subdir, source_filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"光源文件不存在: {source_path}")

        # Compute valid ranges (truncation handling to match batch_config_generator)
        x0 = max(0, src_range_x[0])
        x1 = min(volume_shape[0], src_range_x[1])
        y0 = max(0, src_range_y[0])
        y1 = min(volume_shape[1], src_range_y[1])
        z0 = max(0, src_range_z[0])
        z1 = min(volume_shape[2], src_range_z[1])

        src_shape = (x1 - x0, y1 - y0, z1 - z0)
        
        source_arr = np.fromfile(source_path, dtype=np.float32).reshape(src_shape)
        source_pattern = np.zeros(volume_shape, dtype=np.float32)
        source_pattern[x0:x1, y0:y1, z0:z1] = np.where(source_arr > 0.5, 1.0, 0.0)
        return source_pattern

    def _save_sample_outputs(
        self,
        entry: str,
        config_subdir: str,
        nodal_fluence: np.ndarray,
        suffix: str = "",
    ) -> None:
        np.save(os.path.join(config_subdir, f"fem_nodal_fluence{suffix}.npy"), nodal_fluence)
        sampled_volume = self.solver.sample_to_volume(nodal_fluence)
        np.save(os.path.join(config_subdir, f"fem_volume_fluence{suffix}.npy"), sampled_volume)
        self._save_surface_outputs(config_subdir, nodal_fluence, suffix)

        summary = {
            "sample_id": entry,
            "angle_suffix": suffix,
            "mesh_cache_path": self.mesh_cache_path,
            "n_nodes": int(len(self.solver.mesh.nodes)),
            "n_elements": int(len(self.solver.mesh.elements)),
            "fluence_min": float(np.min(nodal_fluence)),
            "fluence_max": float(np.max(nodal_fluence)),
            "fluence_mean": float(np.mean(nodal_fluence)),
            "fluence_l2": float(np.linalg.norm(nodal_fluence)),
        }
        with open(
            os.path.join(config_subdir, f"fem_summary{suffix}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_surface_outputs(self, config_subdir: str, nodal_fluence: np.ndarray, suffix: str = "") -> None:
        boundary_mask = self.solver.mesh.bndvtx > 0
        boundary_nodes = self.solver.mesh.nodes[boundary_mask]
        boundary_values = nodal_fluence[boundary_mask]

        node_map = np.full(len(self.solver.mesh.nodes), -1, dtype=np.int32)
        node_map[np.where(boundary_mask)[0]] = np.arange(boundary_nodes.shape[0], dtype=np.int32)
        boundary_faces = node_map[self.solver.mesh.boundary_faces]
        face_values = nodal_fluence[self.solver.mesh.boundary_faces].mean(axis=1)

        np.save(os.path.join(config_subdir, f"fem_surface_nodes{suffix}.npy"), boundary_nodes)
        np.save(os.path.join(config_subdir, f"fem_surface_faces{suffix}.npy"), boundary_faces)
        np.save(os.path.join(config_subdir, f"fem_surface_nodal_fluence{suffix}.npy"), boundary_values)
        np.save(os.path.join(config_subdir, f"fem_surface_face_fluence{suffix}.npy"), face_values)

        surface_summary = {
            "n_surface_nodes": int(boundary_nodes.shape[0]),
            "n_surface_faces": int(boundary_faces.shape[0]),
            "fluence_min": float(np.min(boundary_values)),
            "fluence_max": float(np.max(boundary_values)),
            "fluence_mean": float(np.mean(boundary_values)),
        }
        with open(
            os.path.join(config_subdir, f"fem_surface_summary{suffix}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(surface_summary, f, indent=2, ensure_ascii=False)


def save_matrix(matrix: csr_matrix, path: str) -> None:
    save_npz(path, matrix)
