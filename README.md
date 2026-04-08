# BeamOnTarget

A high-performance Python application for simulating the interaction of
particle beams with complex 3D geometries.  It calculates the power
deposited on the surfaces of mesh objects (from `.stl` files) and is
optimised for parallel processing, memory efficiency, and batch-run
capabilities.

The primary workflow involves defining geometry and particle sources,
running the simulation, and analysing the detailed, ParaView-compatible
output files — all from a graphical interface or the command line.

---

## Key Features

- **Graphical User Interface** — Tkinter-based GUI (`sim_gui.py`) for
  managing configurations, launching simulations, viewing results in an
  embedded 3D viewer (Open3D), and comparing CSV summaries with
  interactive bar-charts.
- **Built-in 3D Viewer** — GPU-accelerated Open3D renderer embedded
  inside the GUI with mouse-drag rotation, scroll-wheel zoom,
  right-click pan, orientation axis overlay, and jet colour-map with
  colour bar.
- **High-Performance Engine** — Uses `trimesh` with the `pyembree`
  ray-tracing backend for extremely fast intersection calculations.
- **Parallel Processing** — Leverages `joblib` to distribute work
  across multiple CPU cores.
- **Memory-Safe Design** — Handles billions of particles without
  storing them all in memory simultaneously.
- **Folder-Centric Geometry** — Organise `.stl` files into folders;
  apply scaling and mesh refinement to entire groups.
- **Geometry Caching** — Saves processed (refined) meshes to a cache,
  speeding up subsequent runs.
- **Batch Simulation** — Automatically finds and runs a simulation for
  every `.bl` file in the source directory.
- **Advanced Particle Sources** — Multiple beam models including
  `GaussianTwissBeam`.
- **Professional Output** — Saves results as ParaView-compatible `.vtp`
  files with cell data (Deposited Power, Power Density).
- **Automated Post-Processing** — Optional smoothing algorithm for
  clearer heat maps.
- **VTP Data Extraction** — Extract cell data from `.vtp` files to CSV
  directly from the GUI.
- **CSV Result Comparison** — Side-by-side bar-charts of peak power
  density and deposited power across multiple simulation runs, with
  per-component labels, multipliers, and log-scale toggles.
- **Cross-Platform** — Runs on Linux and Windows (Python ≥ 3.10).

---

## File Structure

| File | Description |
|---|---|
| `sim_gui.py` | Tkinter GUI — the main graphical entry point (`beamontarget` command). |
| `viewer.py` | Built-in 3D viewer (Open3D off-screen rendering in Tk canvas). |
| `run_simulation.py` | CLI entry point — handles arguments and orchestrates simulation runs. |
| `config.py` | Central configuration module; reads/writes `config.json`. |
| `config.json` | JSON file with all simulation parameters (edited via GUI or text). |
| `engine.py` | Core computational engine — power deposition calculation. |
| `geometry.py` | Loading, processing, caching, and grouping of `.stl` meshes. |
| `particles.py` | Particle source classes and `.bl` file loading. |
| `output.py` | File output — `.vtp` saving and summary `.csv` reports. |
| `smooth_results.py` | Core smoothing logic (library). |
| `batch_smoother.py` | Applies smoothing to result `.vtp` files in a directory. |
| `post_smooth.py` | Additional post-processing / smoothing utilities. |
| `generate_report.py` | Automated report generation from results. |
| `extract_mesh_data.py` | CLI tool for extracting mesh cell data from `.vtp` to CSV. |
| `pyproject.toml` | Modern Python packaging metadata (PEP 621). |
| `requirements.txt` | Pinned dependency list (legacy, kept for reference). |
| `install.sh` | One-step installer for **Linux**. |
| `install.bat` | One-step installer for **Windows**. |

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python ≥ 3.10** | On ITER HPC, load the EasyBuild module: `ml Python/3.11.5-GCCcore-13.2.0` |
| **Tkinter** | Usually bundled with Python. On ITER HPC: `ml Tkinter/3.11.5-GCCcore-13.2.0` |
| **C compiler** (Linux only) | Required to build `pyembree` / `embreex` from source. |

### Option A — Quick Install (recommended)

**Linux / macOS:**

```bash
git clone https://github.com/CharlieHills92/BeamOnTarget.git
cd BeamOnTarget
chmod +x install.sh
./install.sh            # creates ./venv and installs everything
source venv/bin/activate
beamontarget            # launch the GUI
```

**Windows:**

```batch
git clone https://github.com/CharlieHills92/BeamOnTarget.git
cd BeamOnTarget
install.bat             &:: creates .\venv and installs everything
venv\Scripts\activate
beamontarget            &:: launch the GUI
```

Both scripts accept an optional argument to specify a custom venv path:

```bash
./install.sh /opt/beamontarget/env    # Linux
install.bat  C:\bot\env               :: Windows
```

### Option B — Manual Install

```bash
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .                      # editable install from pyproject.toml
```

### Option C — pip install from requirements.txt (legacy)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note on Windows:** `pyembree` and `embreex` may fail to compile.
> If so, install them via conda instead:
> ```
> conda install -c conda-forge pyembree embreex
> ```
> All other dependencies (Open3D, pyvista, matplotlib, …) have
> pre-built wheels for Windows.

---

## Quick Start

### 1. Launch the GUI

```bash
beamontarget              # if installed via pip install -e .
# or
python sim_gui.py         # direct execution
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| **General** | ParaView path, number of CPU cores, particles per beamlet, energy. |
| **Geometry** | Add / edit / remove geometry folders and their STL files. |
| **Particles** | Select particle source directory and `.bl` files. |
| **Output** | Output directory, save options, CSV result comparison charts, VTP data extraction. |
| **Run** | Save configuration, launch simulation, view live log output. |

### 2. Prepare Geometry

- Create folders (e.g. `MY_GEOMETRY/`, `TARGETS/`) and place `.stl`
  files inside.
- In the **Geometry** tab, add each folder and configure its scale and
  mesh refinement settings.

### 3. Prepare Particle Sources

- Create a directory (e.g. `BEAM_CONFIGS/`) with one or more `.bl`
  files — space-separated text with columns:

  ```
  # CenterX CenterY CenterZ DirX DirY DirZ Mass_kg Charge_e CurrentDensity_A_m2 SigmaY_m DeltaY_rad SigmaZ_m DeltaZ_rad HaloFraction DeltaHY_rad DeltaHZ_rad
  0 0 0 1 0 0 1.67e-27 1 1.0 0.005 0.005 0.005 0.005 0.1 0.01 0.01
  ```

- Select the source directory in the **Particles** tab.

### 4. Run the Simulation

Click **▶ Run** in the **Run** tab, or from the command line:

```bash
python run_simulation.py
```

Results are saved as `.vtp` files in the output directory (default
`OUTPUT/`), organised by `.bl` filename.

### 5. View Results

- **Built-in viewer:** Use the 3D viewer buttons in the GUI
  (Geometry / Results / All / Sources).  The viewer supports rotation
  (left-drag), zoom (scroll), and pan (right-drag).
- **ParaView:** Open the `.vtp` files directly in
  [ParaView](https://www.paraview.org/), or use the GUI's ParaView
  integration buttons.
- **CSV comparison:** In the **Output** tab, load summary CSV files
  and compare peak power density / deposited power across runs with
  interactive bar charts.
- **Data extraction:** Use the **Extract results data** button to
  export VTP cell data (coordinates, area, power, power load) to CSV.

---

## Command-Line Usage

```bash
# Run all .bl files in the configured source directory
python run_simulation.py

# Preview geometry and source positions
python run_simulation.py --view-setup

# Preview geometry only
python run_simulation.py --view-setup geo
```

### Manual Smoothing

```bash
python batch_smoother.py -i OUTPUT/my_beam_run
```

### VTP Data Extraction (CLI)

```bash
python extract_mesh_data.py input.vtp -o output.csv
```

---

## Dependencies

All dependencies are declared in `pyproject.toml` and installed
automatically.  Key packages:

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `trimesh` + `pyembree` / `embreex` | Mesh loading and ray tracing |
| `pyvista` | VTP file I/O |
| `open3d ≥ 0.17` | GPU-accelerated 3D rendering (off-screen) |
| `matplotlib` | Bar-chart plotting in GUI |
| `Pillow` | Image processing for viewer overlay |
| `pandas` | CSV / DataFrame handling |
| `scipy` | Smoothing algorithms |
| `joblib` | Parallel processing |
| `tqdm` | Progress bars |

---

## License

This project is released under the [MIT License](LICENSE).
