# config.py
"""
Configuration loader for the particle simulation.

Reads parameters from config.json and exposes them as module-level variables
so that all existing code (engine.py, run_simulation.py, output.py, etc.)
continues to work with:

    import config
    config.NUM_CPU_CORES

The JSON file can be edited by hand or via the GUI.
"""

from pathlib import Path
import json
import math
import sys

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

if getattr(sys, "frozen", False):
    # PyInstaller / frozen executable
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    # config/config.py -> config/ -> project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config_GEOMETRY_DLM.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str):
    """
    Resolve a path relative to the project root.

    Absolute paths are returned unchanged.
    Relative paths become PROJECT_ROOT / relative_path.
    """
    if path_str is None:
        return None

    p = Path(path_str)

    if p.is_absolute():
        return str(p)

    return str((PROJECT_ROOT / p).resolve())


def _resolve_geometry_folders(folder_dict):
    """
    Resolve geometry folder paths while preserving settings.
    """

    if folder_dict is None:
        return {}

    resolved = {}

    for folder_path, settings in folder_dict.items():

        abs_path = _resolve_path(folder_path)

        resolved[abs_path] = settings

    return resolved


# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------

def _load_json(path=None):
    """Read the JSON config and return a plain dict."""
    path = Path(path) if path else CONFIG_FILE

    with open(path, "r") as f:
        return json.load(f)


def load_config(path=None):
    """Public config loader."""
    return _load_json(path)


def apply_config(path=None, data=None):
    """Apply config values to module-level globals."""
    if data is None:
        data = _load_json(path)

    _apply(data)


def save_config(data=None, path=None):
    """Write the current (or given) config dict back to JSON."""
    path = Path(path) if path else CONFIG_FILE

    if data is None:
        data = _build_dict()

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    # Reload globals after save
    _apply(data)


# ---------------------------------------------------------------------------
# Build config dict from current globals
# ---------------------------------------------------------------------------

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
        "PARTICLE_BATCH_SIZE": PARTICLE_BATCH_SIZE,
        "TRACKING_MODE": TRACKING_MODE,
        "EM_STEP_LENGTH_M": EM_STEP_LENGTH_M,
        "EM_MAX_STEPS": EM_MAX_STEPS,
        "EM_MAX_DISTANCE_M": EM_MAX_DISTANCE_M,
        "EM_MIN_ENERGY_EV": EM_MIN_ENERGY_EV,
        "EM_BOUNDING_BOX_MIN_CORNER_M": EM_BOUNDING_BOX_MIN_CORNER_M,
        "EM_BOUNDING_BOX_MAX_CORNER_M": EM_BOUNDING_BOX_MAX_CORNER_M,
        "EM_BVH_CHECKPOINT_DISTANCE_M": EM_BVH_CHECKPOINT_DISTANCE_M,
        "V_RID_V": V_RID_V,
        "DENSITY_DIRECTION": DENSITY_DIRECTION,
        "EXTERNAL_FIELD": EXTERNAL_FIELD,
        "REACTION_MODEL": REACTION_MODEL,
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

    # -----------------------------------------------------------------------
    # General
    # -----------------------------------------------------------------------

    g["NUM_CPU_CORES"] = d.get("NUM_CPU_CORES", 1)

    g["ENABLE_DIAGNOSTIC_SURFACES"] = d.get(
        "ENABLE_DIAGNOSTIC_SURFACES",
        False
    )

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    g["GEOMETRY_CACHE_DIR"] = _resolve_path(
        d.get("GEOMETRY_CACHE_DIR", "geometry_cache")
    )

    g["GEOMETRY_FOLDERS"] = _resolve_geometry_folders(
        d.get("GEOMETRY_FOLDERS", {})
    )

    # -----------------------------------------------------------------------
    # Particle sources
    # -----------------------------------------------------------------------

    g["PARTICLE_SOURCE_DIR"] = _resolve_path(
        d.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS")
    )

    g["NUM_PARTICLES_PER_BEAMLET"] = d.get(
        "NUM_PARTICLES_PER_BEAMLET",
        10_001
    )

    radius = d.get("BEAMLET_RADIUS_M", 0.007)

    g["_BEAMLET_RADIUS_M"] = radius

    g["BEAMLET_AREA_FOR_CURRENT"] = math.pi * (radius ** 2)

    batch = d.get("PARTICLE_BATCH_SIZE", 2_500_000)

    g["PARTICLE_BATCH_SIZE"] = batch

    # -----------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------

    g["TRACKING_MODE"] = d.get("TRACKING_MODE", "ray")

    g["EM_STEP_LENGTH_M"] = d.get("EM_STEP_LENGTH_M", 0.02)

    g["EM_MAX_STEPS"] = d.get("EM_MAX_STEPS", 500)

    g["EM_MAX_DISTANCE_M"] = d.get("EM_MAX_DISTANCE_M", None)

    g["EM_MIN_ENERGY_EV"] = d.get("EM_MIN_ENERGY_EV", None)

    g["EM_BOUNDING_BOX_MIN_CORNER_M"] = d.get(
        "EM_BOUNDING_BOX_MIN_CORNER_M",
        None
    )

    g["EM_BOUNDING_BOX_MAX_CORNER_M"] = d.get(
        "EM_BOUNDING_BOX_MAX_CORNER_M",
        None
    )

    g["EM_BVH_CHECKPOINT_DISTANCE_M"] = d.get(
        "EM_BVH_CHECKPOINT_DISTANCE_M",
        1.0
    )

    # -----------------------------------------------------------------------
    # Physics
    # -----------------------------------------------------------------------

    g["V_RID_V"] = d.get("V_RID_V", 20e3)

    g["DENSITY_DIRECTION"] = d.get(
        "DENSITY_DIRECTION",
        d.get(
            "MAIN_BEAM_AXIS_DIRECTION",
            d.get(
                "Main beam axis direction",
                [1.0, 0.0, 0.0]
            ),
        ),
    )

    g["EXTERNAL_FIELD"] = d.get(
        "EXTERNAL_FIELD",
        {
            "type": "zero",
            "electric_field_vpm": [0.0, 0.0, 0.0],
            "magnetic_field_t": [0.0, 0.0, 0.0],
        },
    )

    if str(g["EXTERNAL_FIELD"].get("type", "")).strip().lower() in (
        "rid_segment_y",
        "rid_piecewise",
    ):
        g["EXTERNAL_FIELD"].setdefault("v_rid_v", g["V_RID_V"])

    g["REACTION_MODEL"] = d.get(
        "REACTION_MODEL",
        {"type": "none"}
    )

    frac = d.get("DEPOSITION_FRACTION", 1.0)

    g["_DEPOSITION_FRACTION"] = frac

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    g["SAVE_PARAVIEW_FILES"] = d.get(
        "SAVE_PARAVIEW_FILES",
        True
    )

    g["DETAILED_OUTPUT_DIR"] = _resolve_path(
        d.get("DETAILED_OUTPUT_DIR", "OUTPUT")
    )

    g["ENABLE_VISUALIZATION"] = d.get(
        "ENABLE_VISUALIZATION",
        True
    )

    g["SAVE_BINARY_POWERLOADS"] = d.get(
        "SAVE_BINARY_POWERLOADS",
        False
    )

    g["SAVE_CSV_REPORTS"] = d.get(
        "SAVE_CSV_REPORTS",
        False
    )

    g["RUN_VISUALIZATION_AFTER_SIM"] = d.get(
        "RUN_VISUALIZATION_AFTER_SIM",
        False
    )

    g["VISUALIZE_ALL_RAYS"] = d.get(
        "VISUALIZE_ALL_RAYS",
        False
    )

    g["SUMMARY_CSV_FILENAME"] = d.get(
        "SUMMARY_CSV_FILENAME",
        "power_summary_by_object.csv"
    )

    g["NUM_RAYS_TO_SHOW_IN_PLOT"] = d.get(
        "NUM_RAYS_TO_SHOW_IN_PLOT",
        0
    )

    # -----------------------------------------------------------------------
    # Smoothing
    # -----------------------------------------------------------------------

    g["RUN_SMOOTHER_AFTER_SIM"] = d.get(
        "RUN_SMOOTHER_AFTER_SIM",
        False
    )

    g["SMOOTHING_RADIUS"] = d.get(
        "SMOOTHING_RADIUS",
        0.02
    )

    g["SMOOTHING_MAX_CELL_AREA"] = d.get(
        "SMOOTHING_MAX_CELL_AREA",
        4e-6
    )

    # -----------------------------------------------------------------------
    # External tools
    # -----------------------------------------------------------------------

    g["PARAVIEW_PATH"] = d.get(
        "PARAVIEW_PATH",
        "paraview"
    )

    # -----------------------------------------------------------------------
    # Runtime-only variables
    # -----------------------------------------------------------------------

    g["PARTICLE_SOURCES"] = []


# ---------------------------------------------------------------------------
# Physics model helper
# ---------------------------------------------------------------------------

def get_deposition_fraction(energy_eV):
    """Return the fraction of kinetic energy deposited on impact."""
    return _DEPOSITION_FRACTION


# ---------------------------------------------------------------------------
# Initialise on first import
# ---------------------------------------------------------------------------

_apply(_load_json())