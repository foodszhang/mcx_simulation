# data_build Python Port

This folder stores a direct Python-oriented port of the MATLAB workflow under `data_build/`.

## Current scope

- Ported from MATLAB:
  - `Initialization.m`
  - `preAmiraMesh.m`
  - `getsurface.m`
  - `FEMLinear.m`
  - `FormMatrixF.m`
  - `getlaplace.m`
  - `new_getlaplace.m`
  - `new_getlaplace_GHB.m`
  - `get_signlelabel.m` (as `get_singlelabel`)
  - `get_double.m`
  - `sorcesetup.m`
  - `get_surface_energy.m`
  - Tecplot `.dat` export helpers
  - `main.m` active branch (`main_port.py`)

## Important note on excitation branch usage

In current MATLAB scripts, excitation-related components (`FEMLinear_x`, `uax`, `Dx`, `sorcesetup`) are present but **not invoked by `main.m` active path**.

`main.m` currently runs:

1. mesh preprocess
2. emission matrix assembly (`FEMLinear` + `FormMatrixF`)
3. `A = Mm\\F`
4. surface restriction + Laplacians + save

So the default run is emission-side matrix preparation, not a full excitation+emission two-stage solve.

## Run example

```bash
uv run python -m data_build.python_port.main_port \
  --amira /path/to/9029_with_cancer.am \
  --mesh_txt /tmp/9029_with_cancer.txt \
  --save_prefix /tmp/recon_info_python_port
```

Output is a single `.npz` containing:

- mesh arrays (`nodes`, `elementInside`, `elementSurface`)
- full and surface restricted operators (`A_full`, `A_surface`)
- Laplacian matrices (`Lap`, `n_Lap0..n_Lap3`)

