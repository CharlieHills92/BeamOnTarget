# Beam On Target simulation code

This is a high-performance Python application for simulating the interaction of particle beams with complex 3D geometries. It calculates the power deposited on the surfaces of mesh objects (from `.stl` files) and is optimized for parallel processing, memory efficiency, and batch-run capabilities.

The primary workflow involves defining geometry and particle sources, running the simulation, and analyzing the detailed, ParaView-compatible output files.

## Key Features

-   **High-Performance Engine:** Uses `trimesh` with the `pyembree` ray-tracing backend for extremely fast intersection calculations.
-   **Parallel Processing:** Leverages `joblib` to distribute the computational load across multiple CPU cores, dramatically reducing simulation time.
-   **Memory-Safe Design:** The engine is designed to handle billions of particles without storing them all in memory simultaneously, allowing for large-scale simulations on standard hardware.
-   **Folder-Centric Geometry:** Easily manage complex scenes by organizing `.stl` files into folders. Apply scaling and mesh refinement settings to entire groups of objects.
-   **Geometry Caching:** Automatically saves processed (refined) meshes to a cache, speeding up subsequent runs with the same geometry settings.
-   **Batch Simulation:** Automatically finds and runs a simulation for every particle source configuration (`.bl`) file in a specified directory.
-   **Advanced Particle Sources:** Includes various beam models, from simple planar/conical beams to realistic accelerator physics models like `GaussianTwissBeam`.
-   **Professional Output:** Saves results as ParaView-compatible `.vtp` files, which bundle the geometry with cell data (Deposited Power, Power Density) for easy and powerful post-processing.
-   **Automated Post-Processing:** Can automatically run a smoothing algorithm on the results to produce visually clearer heat maps.

## File Structure

-   `run_simulation.py`: The main entry point for the simulation. Handles command-line arguments and orchestrates the overall workflow.
-   `config.py`: The central user-facing configuration file. Control everything from CPU usage to geometry paths and output settings here.
-   `engine.py`: The core computational engine. Its only job is to efficiently calculate power deposition.
-   `geometry.py`: Handles loading, processing, caching, and grouping of `.stl` geometry files.
-   `particles.py`: Defines the various particle source classes and the logic for loading them from text files.
-   `output.py`: Manages all file outputs, including robust `.vtp` saving and summary `.csv` reports.
-   `batch_smoother.py`: A script to apply smoothing to result files in a directory.
-   `smooth_results.py`: A library file containing the core smoothing logic, used by `batch_smoother.py`.
-   `requirements.txt`: A list of all necessary Python packages.

## Installation

1.  **Prerequisites:** Python 3.8+ is recommended.

2.  **Virtual Environment:** It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start Guide

1.  **Prepare Geometry:**
    -   Create folders (e.g., `MY_GEOMETRY`, `TARGETS`).
    -   Place your `.stl` files inside these folders.
    -   In `config.py`, update the `GEOMETRY_FOLDERS` dictionary to point to your folders and set desired parameters like `scale` and `target_length` for mesh refinement.

2.  **Prepare Particle Sources:**
    -   Create a directory for your beam configurations (e.g., `BEAM_CONFIGS`).
    -   Inside this directory, create one or more beamlet definition files with a `.bl` extension. These are space-separated text files.
    -   **Example `my_beam_run.bl`:**
        ```
        # CenterX CenterY CenterZ DirX DirY DirZ Mass_kg Charge_e CurrentDensity_A_m2 SigmaY_m DeltaY_rad SigmaZ_m DeltaZ_rad HaloFraction DeltaHY_rad DeltaHZ_rad
        0 0 0 1 0 0 1.67e-27 1 1.0 0.005 0.005 0.005 0.005 0.1 0.01 0.01
        ```
    -   In `config.py`, set `PARTICLE_SOURCE_DIR = "BEAM_CONFIGS"`.

3.  **Configure Simulation:**
    -   Open `config.py`.
    -   Set `NUM_CPU_CORES` to the number of cores you want to use (`-1` for all).
    -   Adjust `NUM_PARTICLES_PER_BEAMLET` to control the simulation's resolution.
    -   Ensure `DETAILED_OUTPUT_DIR` is set to your desired output location (default is `"OUTPUT"`).

4.  **Run the Simulation:**
    ```bash
    python run_simulation.py
    ```
    The script will find each `.bl` file and run a full simulation for it, saving results in a uniquely named subfolder within your output directory.

5.  **View Results:**
    -   Navigate to the `OUTPUT` directory. You will find subfolders named after your `.bl` files (e.g., `OUTPUT/my_beam_run/`).
    -   Inside, you will find `results_*.vtp` files. Open these with [ParaView](https://www.paraview.org/) to visualize the power deposition.
    -   If `RUN_SMOOTHER_AFTER_SIM` was `True`, there will be a `SMOOTHED` subfolder containing smoothed versions of the `.vtp` files, which are often better for visualization.

## Advanced Usage

### Setup Preview

Before running a long simulation, you can preview the geometry and particle source locations.

-   **Preview geometry and particle sources:**
    ```bash
    python run_simulation.py --view-setup
    ```
-   **Preview geometry only:**
    ```bash
    python run_simulation.py --view-setup geo
    ```

### Manual Smoothing

If you want to re-run the smoothing process with different parameters or on existing results, you can run the `batch_smoother.py` script directly.

```bash
# Smooth all .vtp files in the "OUTPUT/my_beam_run" directory
python batch_smoother.py -i OUTPUT/my_beam_run
