import numpy as np
import pyvista as pv
from pathlib import Path
import yaml
import json
import sys

sys.path.append("/home/foods/pro/mcx_simulation")
from src.source_inverse import load_source_pattern_from_sample

def main():
    config_path = "/home/foods/pro/mcx_simulation/config/config_legacy.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load Mesh
    mesh_path = config["fem"]["mesh"]["path"]
    data = np.load(mesh_path)
    nodes = data["nodes"]
    elements = data["elements"]
    
    print(f"Nodes shape: {nodes.shape}")
    print(f"Nodes Min: {nodes.min(axis=0)}")
    print(f"Nodes Max: {nodes.max(axis=0)}")
    print(f"Elements shape: {elements.shape}")
    print(f"Elements min index: {elements.min()}")
    
    if elements.min() == 1:
        print("Detected 1-based indexing, converting to 0-based...")
        elements -= 1
        
    # Create PyVista Mesh
    # elements are (N, 4). PyVista needs (N, 5) with [4, n1, n2, n3, n4]
    cells = np.column_stack([np.full(len(elements), 4), elements]).flatten()
    cell_types = np.full(len(elements), 10, dtype=np.uint8)
    
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white", show_edges=True)
    
    # Add Excitation Source
    if "inverse" in config and "source_inverse" in config["inverse"]:
        ex_pos = config["inverse"]["source_inverse"].get("excitation_pos")
        if ex_pos:
            plotter.add_mesh(pv.Sphere(radius=0.5, center=ex_pos), color="red", label="Excitation")
            print(f"Excitation Source (mm): {ex_pos}")
            
            # Add True Source (Tumor)
            sample_dir = "/home/foods/pro/mcx_simulation/output_legacy_shallow/0"
            try:
                source_vol = load_source_pattern_from_sample(config, sample_dir, 0)
                
                # Convert non-zero voxels to points
                dx = config.get("lengthunit", 0.1)
                
                # Find ROI > 0.5
                z_idx, y_idx, x_idx = np.where(source_vol > 0.5)
                
                if len(z_idx) > 0:
                    # Swap to x, y, z for plotting (x,y,z is standard, indices are z,y,x)
                    pts = np.column_stack([x_idx, y_idx, z_idx]) * dx
                    
                    poly = pv.PolyData(pts)
                    plotter.add_mesh(poly, color="green", point_size=5, render_points_as_spheres=True, label="True Source")
                    
                    center = np.mean(pts, axis=0)
                    print(f"True Source Center (mm): {center}")
                    
                    dist = np.linalg.norm(center - np.array(ex_pos))
                    print(f"Distance: {dist:.2f} mm")
                else:
                    print("Source volume is empty!")
            except Exception as e:
                print(f"Error loading source: {e}")
    
    plotter.add_axes()
    plotter.add_legend()
    plotter.show(screenshot="setup_vis.png")
    print("Saved setup_vis.png")

if __name__ == "__main__":
    main()
