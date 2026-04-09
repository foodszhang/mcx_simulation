import numpy as np
import sys
from pathlib import Path

def compute_binary_dice(true_path, recon_path):
    print(f"Loading true source from {true_path}")
    true_source = np.load(true_path)
    print(f"Loading recon source from {recon_path}")
    recon_source = np.load(recon_path)

    # Normalize to max 1
    true_norm = true_source / (true_source.max() + 1e-12)
    recon_norm = recon_source / (recon_source.max() + 1e-12)

    # Thresholds to test
    thresholds = [0.5, 0.2, 0.1, 0.05, 0.01, 1e-3, 1e-6]
    
    print("\n--- Dice Calculation ---")
    print(f"{'Threshold':<10} | {'Dice':<10} | {'True Vol':<10} | {'Recon Vol':<10}")
    print("-" * 50)

    results = {}

    for th in thresholds:
        m1 = true_norm > th
        m2 = recon_norm > th
        
        vol1 = m1.sum()
        vol2 = m2.sum()
        
        if vol1 == 0 and vol2 == 0:
            dice = 0.0
        else:
            intersection = np.logical_and(m1, m2).sum()
            dice = 2.0 * intersection / (vol1 + vol2 + 1e-12)
        
        print(f"{th:<10} | {dice:<10.4f} | {vol1:<10} | {vol2:<10}")
        results[th] = dice

    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default paths if not provided
        base_dir = Path("output_tuning_v15_v10_inverse_crime_fixed_pure/sample_0_l1_ls_0")
        true_path = base_dir / "true_source.npy"
        recon_path = base_dir / "recon_source.npy"
    else:
        true_path = Path(sys.argv[1])
        recon_path = Path(sys.argv[2])
        
    if not true_path.exists() or not recon_path.exists():
        print(f"Error: Files not found.\n{true_path}\n{recon_path}")
        sys.exit(1)
        
    compute_binary_dice(true_path, recon_path)
