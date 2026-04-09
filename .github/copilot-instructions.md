# Copilot Instructions for MCX Simulation Project

## Build, Test, and Run Commands

- **Run Main Pipeline**: Use `uv` to run the main entry point.
  ```bash
  uv run python main.py --num_configs 2 --out_dir output_test
  ```
- **Run Tests**: Use `pytest` for the test suite.
  ```bash
  # Run all tests
  pytest tests/ -v
  
  # Run a specific test file
  pytest tests/test_forward_solver.py -v
  ```
- **Dependencies**: Managed via `pyproject.toml` and `uv`.
- **External Dependencies**: Requires `mcx` or `mcxcl` binary to be in the system PATH for simulations.

## High-Level Architecture

This project implements a pipeline for batch volume optical simulations and multi-view projection generation.

1.  **Configuration Generation** (`src/batch_config_generator.py`): Generates multiple simulation configurations based on a master YAML config.
2.  **Simulation Runner** (`src/batch_simulation_runner.py`): 
    - Auto-detects `mcx` (GPU) or `mcxcl` (CPU) binaries.
    - Iterates through generated configurations in numbered subdirectories.
    - Executes MCX simulations to produce `.jnii` output.
    - Supports normal and "no-scatter" (mus=0) modes.
3.  **Post-Processing** (`src/batch_postprocessor.py`):
    - Processes simulation outputs in parallel (ProcessPoolExecutor).
    - Generates projections (flux and depth) at specified angles using `src/gen_mul_projection.py`.
4.  **Forward Solver** (`src/forward_solver.py`): A separate Python-based solver implementation (likely FEM/FDM) with matrix construction tools, distinct from the MCX binary wrapper.

## Key Conventions

- **Language**: Comments and log messages are primarily in **Chinese**. Maintain this convention for consistency.
- **Output Structure**:
    - Default output directory format: `output_YYYYMMDD_HHMMSS/`.
    - Inside output dir: Numbered folders (`0/`, `1/`, ...) containing config and results for each sample.
- **Data Formats**:
    - **JSON/JNII**: Used for MCX configuration and volumetric output, handled via `jdata`.
    - **NPZ**: Used for storing projection results (`proj.npz`, `dep_proj.npz`).
- **Configuration**:
    - Centralized in `config/*.yaml` files.
    - File naming patterns (e.g., `{id}.json`, `{id}.jnii`) are configurable via the `file_naming` section in YAML.
- **Error Handling**: The simulation runner attempts to degrade gracefully (e.g., GPU -> CPU) and skips existing results to allow resuming interrupted batches.
