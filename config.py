# config.py
"""
Configuration loader for the particle simulation.

Reads parameters from config.json and exposes them as module-level variables
so that all existing code (engine.py, run_simulation.py, output.py, etc.)
continues to work with ``import config; config.NUM_CPU_CORES``.

The JSON file can be edited by hand or via the GUI (sim_gui.py).
"""
import json
import math
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Path to the JSON configuration file (next to this script by default)
# ---------------------------------------------------------------------------
_CONFIG_DIR = (os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
               else os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")

# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------

def _load_json(path=None):
    """Read the JSON config and return a plain dict."""
    path = path or _CONFIG_FILE
    with open(path, "r") as f:
        return json.load(f)


def load_config(path=None):
    """Read the JSON config and return a plain dict (public alias for _load_json)."""
    return _load_json(path)


def apply_config(path=None, data=None):
    """Apply config values to module-level globals from a path or dict."""
    if data is None:
        data = _load_json(path)
    _apply(data)


def save_config(data=None, path=None):
    """Write the current (or given) config dict back to JSON."""
    path = path or _CONFIG_FILE
    if data is None:
        data = _build_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    # After saving, reload so module-level vars reflect the change
    _apply(data)


def _build_dict():
    """Snapshot the current module-level variables into a dict."""
    return {
        "NUM_CPU_CORES": NUM_CPU_CORES,
        "ENABLE_DIAGNOSTIC_SURFACES": ENABLE_DIAGNOSTIC_SURFACES,
        "GEOMETRY_CACHE_DIR": GEOMETRY_CACHE_DIR,
        "GEOMETRY_FOLDERS": GEOMETRY_FOLDERS,
        "PARTICLE_SOURCE_DIR": PARTICLE_SOURCE_DIR,
        "NUM_PARTICLES_PER_BEAMLET": NUM_PARTICLES_PER_BEAMLET,
        "BEAMLET_RADIUS_M": _BEAMLET_RADIUS_M,
        "SOURCES_PER_WORKER": SOURCES_PER_WORKER,
        "PARTICLE_BATCH_SIZE": PARTICLE_BATCH_SIZE,
        "DEPOSITION_FRACTION": _DEPOSITION_FRACTION,
        "SAVE_PARAVIEW_FILES": SAVE_PARAVIEW_FILES,
        "DETAILED_OUTPUT_DIR": DETAILED_OUTPUT_DIR,
        "ENABLE_VISUALIZATION": ENABLE_VISUALIZATION,
        "SAVE_BINARY_POWERLOADS": SAVE_BINARY_POWERLOADS,
        "SAVE_CSV_REPORTS": SAVE_CSV_REPORTS,
        "RUN_VISUALIZATION_AFTER_SIM": RUN_VISUALIZATION_AFTER_SIM,
        "VISUALIZE_ALL_RAYS": VISUALIZE_ALL_RAYS,
        "SUMMARY_CSV_FILENAME": SUMMARY_CSV_FILENAME,
        "NUM_RAYS_TO_SHOW_IN_PLOT": NUM_RAYS_TO_SHOW_IN_PLOT,
        "RUN_SMOOTHER_AFTER_SIM": RUN_SMOOTHER_AFTER_SIM,
        "SMOOTHING_RADIUS": SMOOTHING_RADIUS,
        "SMOOTHING_MAX_CELL_AREA": SMOOTHING_MAX_CELL_AREA,
        "PARAVIEW_PATH": PARAVIEW_PATH,
    }

# ---------------------------------------------------------------------------
# Apply a dict to module-level variables
# ---------------------------------------------------------------------------

def _apply(d):
    """Set module globals from a config dict."""
    g = globals()

    g["NUM_CPU_CORES"]              = d.get("NUM_CPU_CORES", 1)
    g["ENABLE_DIAGNOSTIC_SURFACES"] = d.get("ENABLE_DIAGNOSTIC_SURFACES", False)
    g["GEOMETRY_CACHE_DIR"]         = d.get("GEOMETRY_CACHE_DIR", "geometry_cache")
    g["GEOMETRY_FOLDERS"]           = d.get("GEOMETRY_FOLDERS", {})

    g["PARTICLE_SOURCE_DIR"]        = d.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS")
    g["NUM_PARTICLES_PER_BEAMLET"]  = d.get("NUM_PARTICLES_PER_BEAMLET", 10_001)

    radius = d.get("BEAMLET_RADIUS_M", 0.007)
    g["_BEAMLET_RADIUS_M"]         = radius
    g["BEAMLET_AREA_FOR_CURRENT"]  = math.pi * (radius ** 2)

    npb = g["NUM_PARTICLES_PER_BEAMLET"]
    batch = d.get("PARTICLE_BATCH_SIZE", 2_500_000)
    g["PARTICLE_BATCH_SIZE"]        = batch
    spw = d.get("SOURCES_PER_WORKER", None)
    g["SOURCES_PER_WORKER"]         = spw if spw is not None else int(batch / npb)

    frac = d.get("DEPOSITION_FRACTION", 1.0)
    g["_DEPOSITION_FRACTION"]       = frac

    g["SAVE_PARAVIEW_FILES"]        = d.get("SAVE_PARAVIEW_FILES", True)
    g["DETAILED_OUTPUT_DIR"]        = d.get("DETAILED_OUTPUT_DIR", "OUTPUT")
    g["ENABLE_VISUALIZATION"]       = d.get("ENABLE_VISUALIZATION", True)
    g["SAVE_BINARY_POWERLOADS"]     = d.get("SAVE_BINARY_POWERLOADS", False)
    g["SAVE_CSV_REPORTS"]           = d.get("SAVE_CSV_REPORTS", False)
    g["RUN_VISUALIZATION_AFTER_SIM"]= d.get("RUN_VISUALIZATION_AFTER_SIM", False)
    g["VISUALIZE_ALL_RAYS"]         = d.get("VISUALIZE_ALL_RAYS", False)
    g["SUMMARY_CSV_FILENAME"]       = d.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv")
    g["NUM_RAYS_TO_SHOW_IN_PLOT"]   = d.get("NUM_RAYS_TO_SHOW_IN_PLOT", 0)

    g["RUN_SMOOTHER_AFTER_SIM"]     = d.get("RUN_SMOOTHER_AFTER_SIM", False)
    g["SMOOTHING_RADIUS"]           = d.get("SMOOTHING_RADIUS", 0.02)
    g["SMOOTHING_MAX_CELL_AREA"]    = d.get("SMOOTHING_MAX_CELL_AREA", 4e-6)

    g["PARAVIEW_PATH"]              = d.get("PARAVIEW_PATH", "paraview")

    g["PARTICLE_SOURCES"]           = []  # always empty; sources come from .bl files


# ---------------------------------------------------------------------------
# Physics model — kept as a Python function (not serialisable to JSON).
# The JSON stores the constant fraction; this wraps it as a callable.
# ---------------------------------------------------------------------------

def get_deposition_fraction(energy_eV):
    """Return the fraction of kinetic energy deposited on impact."""
    return _DEPOSITION_FRACTION


# ---------------------------------------------------------------------------
# Initialise on first import
# ---------------------------------------------------------------------------
_apply(_load_json())
