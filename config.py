# config.py
"""
User-facing configuration file for the particle simulation.
"""
import numpy as np


# --- NEW: Parallel Processing Configuration ---
# Number of CPU cores to use for the simulation.
# -1 means use all available cores.
# 1 means run sequentially (useful for debugging).
NUM_CPU_CORES = 1 # Default to 8 cores

# --- 1. GEOMETRY CONFIGURATION (FOLDER-CENTRIC) ---
# --- NEW: Master switch for simulation mode ---
# True:  Use the slower, more general multi-hit engine for transparent surfaces.
# False: Use the extremely fast single-hit engine (all surfaces are solid).
ENABLE_DIAGNOSTIC_SURFACES = False

GEOMETRY_CACHE_DIR = "geometry_cache"
GEOMETRY_FOLDERS = {
    # Key: The path to the folder containing STL files.
    # Value: A dictionary of settings for that folder.
    #
    # New options:
    #   save_impact_data (bool): If True, store impact details (mass, charge, position, velocity, total kinetic energy) for impinging particles as a CSV.
    #   max_impact_records (int, optional): Maximum number of impact records to store per geometry. If more particles impinge, only the first N are stored, but the total number of impacts is always recorded.
    #
    "NEU": {
        "scale": 1,
        "target_length": 1.0,
        "save_details": False,  # <-- NEW: Yes, save detailed reports for this folder
        "show_in_plot": False,
        "is_diagnostic": False,  # <-- This marks them as solid surfaces
        "save_impact_data": False,  # <-- NEW: Store impact data for this geometry?
        # "max_impact_records": 10000,  # <-- Optional: Max number of impacts to store
    },
    
    "RID": {
        "scale": 1,
        "target_length": 1.0,
        "save_details": False,  # <-- Yes, save for this one too
        "show_in_plot": False,
        "is_diagnostic": False,  # <-- This marks them as solid surfaces
        "save_impact_data": False,  # <-- NEW: Store impact data for this geometry?
        # "max_impact_records": 10000,  # <-- Optional: Max number of impacts to store
    },

    "CAL": {
        "scale": 1,
        "target_length": 0.5,
        "save_details": True,  # <-- Yes, save for this one too
        "show_in_plot": False,
        "is_diagnostic": False,  # <-- This marks them as solid surfaces
        "save_impact_data": True,  # <-- NEW: Store impact data for this geometry?
        "max_impact_records": 10000,  # <-- Optional: Max number of impacts to store
    },

    # # --- NEW: A dedicated folder for diagnostic planes ---
    # "DIAGNOSTICS": {
    #     "scale": 1.0,           # Assume they are defined in meters
    #     "target_length": 0.02,  # Give them a fine mesh for good resolution
    #     "save_details": True,
    #     "show_in_plot": False,
    #     "is_diagnostic": True,  # <-- This marks them as transparent, pass-through surfaces
    #     "save_impact_data": True,  # <-- Store impact data for diagnostics
    #     "max_impact_records": 100000,  # <-- Example: store up to 100k impacts
    # }
}


# --- 2. PARTICLE SOURCE CONFIGURATION ---
# --- NEW: Specify a DIRECTORY containing all beam configuration files ---
# The simulation will run once for every .bl file found in this folder.
# Set to None to use the hardcoded fallback list below.
PARTICLE_SOURCE_DIR = "BEAM_CONFIGS"
# The old PARTICLE_SOURCE_FILE is now obsolete.
# PARTICLE_SOURCE_FILE = "beamlets_5mrad_0tilt.txt"
NUM_PARTICLES_PER_BEAMLET = 10_001
# NUM_PARTICLES_PER_BEAMLET = 10001
BEAMLET_AREA_FOR_CURRENT = np.pi * (0.007**2)
PARTICLE_SOURCES = []

# --- NEW: Controls the balance between vectorization and parallel overhead ---
# Each parallel worker will process this many sources at a time.
# A larger number increases vectorization speed but uses more memory per worker.
# A good starting point is (Total Sources / Num CPU Cores).
SOURCES_PER_WORKER = int(2_500_000/NUM_PARTICLES_PER_BEAMLET)

# # The old PARTICLE_BATCH_SIZE is now obsolete for this engine.
PARTICLE_BATCH_SIZE = 2_500_000

# --- 3. PHYSICS MODEL ---
def get_deposition_fraction(energy_eV):
    #scaled_energy = np.clip(energy_eV, 500, 8000)
    fraction = 1
    return fraction

# --- 4. OUTPUT CONFIGURATION ---

# --- Primary Output for Professional Workflow ---
# Saves results as ParaView-compatible .vtp files (geometry + data). This is the recommended workflow.
SAVE_PARAVIEW_FILES = True
# Directory for all detailed output reports (VTP, NPY, CSV)
DETAILED_OUTPUT_DIR = "OUTPUT" 

ENABLE_VISUALIZATION = True # Master switch for all plotting

# NEW: Simplified output control
# Saves results as fast, compact binary files (.npy). This is the recommended default.
SAVE_BINARY_POWERLOADS = False
# In ADDITION to binary files, also save human-readable .csv reports.
SAVE_CSV_REPORTS = False

# True: Saves results as ParaView-compatible .vtp files (geometry + data).
SAVE_PARAVIEW_FILES = True 

# If True, automatically runs the visualization after the simulation finishes.
# Set to False for long batch runs; you can then use post_process.py later.
RUN_VISUALIZATION_AFTER_SIM = False

# NEW: Control which rays are shown in the final visualization
# True:  Show a random sample of ALL generated rays (including misses). Good for debugging aiming.
# False: Show a random sample of ONLY the rays that hit a target. Good for seeing deposition patterns.
VISUALIZE_ALL_RAYS = False

# High-level summary file
SUMMARY_CSV_FILENAME = "power_summary_by_object.csv"

# Number of particle rays to show in the visualization
NUM_RAYS_TO_SHOW_IN_PLOT = 0

# --- NEW: Automatic Post-Processing Control ---
# If True, automatically runs the batch smoother script after the simulation is complete.
# This will create smoothed versions of all saved .vtp/.vtm files.
RUN_SMOOTHER_AFTER_SIM = False

# Smoothing radius in mesh units (metres).  All cells whose centroid is
# within this radius contribute to the smoothed power density.
SMOOTHING_RADIUS = 0.02

# Maximum cell area (m²) for smoothing.  Only cells with area below this
# threshold AND with deposited power > 0 will be smoothed.  Larger cells
# keep their original density.  Set to None to smooth all cells with power.
SMOOTHING_MAX_CELL_AREA = 4e-6
