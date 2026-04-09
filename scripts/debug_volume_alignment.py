import numpy as np
import nibabel as nib
import jdata as jd
import os
from scipy.ndimage import center_of_mass

def load_bin_volume(path, shape, dtype=np.uint8):
    return np.fromfile(path, dtype=dtype).reshape(shape)

def load_jnii(path):
    data = jd.load(path)
    return data['Data'] if 'Data' in data else data

def check_alignment(vol_seg, vol_flu, label=""):
    print(f"--- {label} ---")
    print(f"Seg Shape: {vol_seg.shape}")
    print(f"Flu Shape: {vol_flu.shape}")
    
    # Thresholds
    mask_seg = vol_seg > 0
    mask_flu = vol_flu > 0.01 * vol_flu.max()
    
    cm_seg = center_of_mass(mask_seg)
    cm_flu = center_of_mass(mask_flu)
    
    print(f"CM Seg: {cm_seg}")
    print(f"CM Flu: {cm_flu}")
    diff = np.array(cm_seg) - np.array(cm_flu)
    print(f"Diff: {diff}")
    
    # Overlap (Dice)
    if vol_seg.shape == vol_flu.shape:
        intersection = np.logical_and(mask_seg, mask_flu).sum()
        dice = 2 * intersection / (mask_seg.sum() + mask_flu.sum())
        print(f"Overlap Dice: {dice}")
    else:
        print("Shapes mismatch, cannot calculate Dice")

# Config
bin_path = "/home/foods/pro/mcx_simulation/volume_bases/bin/volume_brain.bin"
jnii_path = "/home/foods/pro/mcx_simulation/output_test_dual_v5/0/0.jnii"

try:
    # 1. Assume Seg=(X, Y, Z) (182, 164, 210)
    vol_xyz = load_bin_volume(bin_path, (182, 164, 210))
    # MCX from output usually matches input.
    vol_mcx = load_jnii(jnii_path) # MCX is (X, Y, Z) usually
    
    check_alignment(vol_xyz, vol_mcx, "Assume Seg=(X, Y, Z), MCX=(X, Y, Z)")
    
    # 2. Assume Seg=(Z, Y, X) (210, 164, 182)
    # Binary might be stored as ZYX.
    vol_zyx = load_bin_volume(bin_path, (210, 164, 182))
    
    # 3. Transpose MCX to (Z, Y, X) for comparison
    vol_mcx_T = vol_mcx.transpose(2, 1, 0)
    
    check_alignment(vol_zyx, vol_mcx_T, "Assume Seg=(Z, Y, X), MCX Transposed to (Z, Y, X)")
    
except Exception as e:
    print(f"Error: {e}")
