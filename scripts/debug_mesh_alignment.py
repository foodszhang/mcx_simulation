import numpy as np
import nibabel as nib
import os

def load_mesh(path):
    data = np.load(path)
    return data['nodes'], data['elements']

def check_mesh(path, label=""):
    try:
        nodes, elements = load_mesh(path)
        print(f"--- {label} ---")
        print(f"Nodes: {nodes.shape}")
        print(f"Elements: {elements.shape}")
        print(f"Bounds X: {nodes[:, 0].min()} - {nodes[:, 0].max()}")
        print(f"Bounds Y: {nodes[:, 1].min()} - {nodes[:, 1].max()}")
        print(f"Bounds Z: {nodes[:, 2].min()} - {nodes[:, 2].max()}")
        print(f"Center: {nodes.mean(axis=0)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

fine_path = "rat_brain_mesh_fine.npz"
coarse_path = "rat_brain_mesh_coarse.npz"

check_mesh(fine_path, "Fine Mesh")
check_mesh(coarse_path, "Coarse Mesh")

# Check Volume Shape
bin_path = "volume_bases/bin/volume_brain.bin"
try:
    # Read first 3 bytes? No, binary.
    # Just check file size.
    size = os.path.getsize(bin_path)
    print(f"Volume Binary Size: {size}")
    # 182*164*210 = 6268080 bytes.
    print(f"Expected Size: {182*164*210}")
except Exception as e:
    print(f"Error checking volume: {e}")
