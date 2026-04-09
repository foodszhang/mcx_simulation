import trimesh
import numpy as np

# Create a sphere mesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
print(f"Original faces: {len(mesh.faces)}")

try:
    ratio = 0.1
    print(f"Target percent (float): {ratio}")
    simplified = mesh.simplify_quadric_decimation(percent=ratio)
    print(f"Simplified faces (percent arg): {len(simplified.faces)}")
except Exception as e:
    print(f"Simplification failed: {e}")
