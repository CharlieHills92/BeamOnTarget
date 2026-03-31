# config.py
"""
Configuration loader for FullBeamSimulation.

Reads parameters from config.json and exposes them as module-level variables
so every module can simply ``import config; config.NUM_CPU_CORES``.
"""
import json
import math
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")


# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------
def _load_json(path=None):
    path = path or _CONFIG_FILE
    with open(path, "r") as f:
        return json.load(f)


def save_config(data=None, path=None):
    path = path or _CONFIG_FILE
    if data is None:
        data = _build_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    _apply(data)


def _build_dict():
    return {
        "NUM_CPU_CORES":            NUM_CPU_CORES,
        "GEOMETRY_CACHE_DIR":       GEOMETRY_CACHE_DIR,
        "GEOMETRY_FOLDERS":         GEOMETRY_FOLDERS,
        "PARTICLE_SOURCE_DIR":      PARTICLE_SOURCE_DIR,
        "NUM_PARTICLES_PER_BEAMLET": NUM_PARTICLES_PER_BEAMLET,
        "BEAMLET_RADIUS_M":         _BEAMLET_RADIUS_M,
        "SOURCES_PER_WORKER":       SOURCES_PER_WORKER,
        "PARTICLE_BATCH_SIZE":      PARTICLE_BATCH_SIZE,
        # Tracking engine
        "TRACKING_ENGINE":          TRACKING_ENGINE,
        # Boris settings
        "BORIS_STEP_LENGTH_M":      BORIS_STEP_LENGTH_M,
        "BORIS_MAX_STEPS":          BORIS_MAX_STEPS,
        "BORIS_RELATIVISTIC":       BORIS_RELATIVISTIC,
        "BORIS_NULLCOLL_ENABLED":   BORIS_NULLCOLL_ENABLED,
        "BORIS_NULLCOLL_TRACK_ELECTRONS": BORIS_NULLCOLL_TRACK_ELECTRONS,
        "E_FIELD_FILE":             E_FIELD_FILE,
        "B_FIELD_FILE":             B_FIELD_FILE,
        "GAS_PROFILE_FILE":         GAS_PROFILE_FILE,
        "GAS_DENSITY_UNIFORM":      GAS_DENSITY_UNIFORM,
        # Deposition
        "DEPOSITION_FRACTION":      _DEPOSITION_FRACTION,
        # Output
        "SAVE_PARAVIEW_FILES":      SAVE_PARAVIEW_FILES,
        "DETAILED_OUTPUT_DIR":      DETAILED_OUTPUT_DIR,
        "SAVE_BINARY_POWERLOADS":   SAVE_BINARY_POWERLOADS,
        "SAVE_CSV_REPORTS":         SAVE_CSV_REPORTS,
        "SUMMARY_CSV_FILENAME":     SUMMARY_CSV_FILENAME,
        # Smoothing
        "RUN_SMOOTHER_AFTER_SIM":   RUN_SMOOTHER_AFTER_SIM,
        "SMOOTHING_RADIUS":         SMOOTHING_RADIUS,
        "SMOOTHING_MAX_CELL_AREA":  SMOOTHING_MAX_CELL_AREA,
    }


# ---------------------------------------------------------------------------
# Apply dict → module globals
# ---------------------------------------------------------------------------
def _apply(d):
    g = globals()

    g["NUM_CPU_CORES"]          = d.get("NUM_CPU_CORES", -1)
    g["GEOMETRY_CACHE_DIR"]     = d.get("GEOMETRY_CACHE_DIR", "geometry_cache")
    g["GEOMETRY_FOLDERS"]       = d.get("GEOMETRY_FOLDERS", {})

    g["PARTICLE_SOURCE_DIR"]    = d.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIGS")
    g["NUM_PARTICLES_PER_BEAMLET"] = d.get("NUM_PARTICLES_PER_BEAMLET", 10_001)

    radius = d.get("BEAMLET_RADIUS_M", 0.007)
    g["_BEAMLET_RADIUS_M"]     = radius
    g["BEAMLET_AREA_FOR_CURRENT"] = math.pi * (radius ** 2)

    npb = g["NUM_PARTICLES_PER_BEAMLET"]
    batch = d.get("PARTICLE_BATCH_SIZE", 2_500_000)
    g["PARTICLE_BATCH_SIZE"]    = batch
    spw = d.get("SOURCES_PER_WORKER", None)
    g["SOURCES_PER_WORKER"]     = spw if spw is not None else max(1, int(batch / npb))

    # Tracking engine selector
    g["TRACKING_ENGINE"]        = d.get("TRACKING_ENGINE", "raytrace")

    # Boris EM stepper settings
    g["BORIS_STEP_LENGTH_M"]    = d.get("BORIS_STEP_LENGTH_M", 0.001)
    g["BORIS_MAX_STEPS"]        = d.get("BORIS_MAX_STEPS", 5000)
    g["BORIS_RELATIVISTIC"]     = d.get("BORIS_RELATIVISTIC", False)
    g["E_FIELD_FILE"]           = d.get("E_FIELD_FILE", None)
    g["B_FIELD_FILE"]           = d.get("B_FIELD_FILE", None)

    # Null-collision MC settings (sub-option of Boris)
    g["BORIS_NULLCOLL_ENABLED"]         = d.get("BORIS_NULLCOLL_ENABLED",
                                                 d.get("NULLCOLL_ENABLED", False))
    g["BORIS_NULLCOLL_TRACK_ELECTRONS"] = d.get("BORIS_NULLCOLL_TRACK_ELECTRONS",
                                                 d.get("NULLCOLL_TRACK_ELECTRONS", False))
    g["GAS_PROFILE_FILE"]       = d.get("GAS_PROFILE_FILE", None)
    g["GAS_DENSITY_UNIFORM"]    = d.get("GAS_DENSITY_UNIFORM", None)

    # Deposition
    frac = d.get("DEPOSITION_FRACTION", 1.0)
    g["_DEPOSITION_FRACTION"]   = frac

    # Output
    g["SAVE_PARAVIEW_FILES"]    = d.get("SAVE_PARAVIEW_FILES", True)
    g["DETAILED_OUTPUT_DIR"]    = d.get("DETAILED_OUTPUT_DIR", "OUTPUT")
    g["SAVE_BINARY_POWERLOADS"] = d.get("SAVE_BINARY_POWERLOADS", False)
    g["SAVE_CSV_REPORTS"]       = d.get("SAVE_CSV_REPORTS", False)
    g["SUMMARY_CSV_FILENAME"]   = d.get("SUMMARY_CSV_FILENAME",
                                        "power_summary_by_object.csv")

    # Smoothing
    g["RUN_SMOOTHER_AFTER_SIM"] = d.get("RUN_SMOOTHER_AFTER_SIM", False)
    g["SMOOTHING_RADIUS"]       = d.get("SMOOTHING_RADIUS", 0.02)
    g["SMOOTHING_MAX_CELL_AREA"] = d.get("SMOOTHING_MAX_CELL_AREA", None)

    # Legacy placeholder
    g["PARTICLE_SOURCES"]       = []


def get_deposition_fraction(energy_eV):
    """Return the fraction of kinetic energy deposited on impact."""
    return _DEPOSITION_FRACTION


# ---------------------------------------------------------------------------
# Initialise on first import
# ---------------------------------------------------------------------------
_apply(_load_json())
