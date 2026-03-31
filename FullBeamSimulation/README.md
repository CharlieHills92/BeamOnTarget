# FullBeamSimulation

A comprehensive particle tracking and power/current deposition code for neutral
beam injection (NBI) systems.

## Features

- **Particle generation** from user-defined sources (Planar, Conical, Gaussian,
  Twiss, GaussianTwiss beams) or `.bl` beamlet configuration files.
- **STL geometry import** with automatic caching, scaling and refinement.
- **Three tracking engines** selectable per-run:
  - **Ray-trace** — infinite-length Embree ray cast (fastest; for line-of-sight
    power deposition).
  - **EM (Boris)** — stepped Boris integrator in static E + B fields with
    user-defined step length, optional relativistic corrections, and per-step
    short-ray Embree collision detection.
  - **Null-collision MC** — Monte-Carlo particle–background-gas interactions
    (stripping, ionisation, charge-exchange) with a pluggable cross-section
    module.  Optionally generates and tracks secondary electrons.
- **Per-species power and current deposition** on STL surfaces saved as VTP,
  CSV, and binary files.
- **Scales to ~1–2 × 10⁹ starting particles** via batched processing with
  constant memory footprint.

## Quick start

```bash
pip install -r requirements.txt
# Edit config.json, then:
python run_simulation.py
```

## Configuration

All parameters live in `config.json`.  See `config.py` for the mapping to
module-level variables.

## Project layout

```
config.py / config.json     — configuration loader & defaults
particles.py                — particle source classes + .bl file loader
geometry.py                 — STL loading, caching, refinement
fields.py                   — E(x), B(x) field interpolators
cross_sections.py           — reaction cross-section tables (σ)
background.py               — gas density / plasma profile interpolators
engine_raytrace.py          — infinite-ray Embree engine
engine_boris.py             — stepped Boris + short-ray collision engine
engine_nullcoll.py          — null-collision Monte-Carlo engine
deposition.py               — per-species power & current accumulator
output.py                   — VTP, CSV, summary output writers
run_simulation.py           — main entry point / batch runner
```
