#!/usr/bin/env python3
# results_csv_loader.py
"""
CSV parsing helpers for the BeamOnTarget Results tab.

Extracted from gui_results.py to keep that module under ~500 lines.

Public API
----------
find_csv_file(search_dir, extra_candidates=None)
    Locate the first existing CSV in *search_dir*, preferring explicit candidates.

read_csv(path)
    Read a CSV file to a list of dicts (via csv.DictReader).

detect_column_keys(rows)
    Heuristically detect name / total-power / peak-density column keys from
    the first row of a CSV.  Also returns per-species column dicts.

normalize_name(raw_name)
    Strip result-file prefixes ("results_", "smoothed_") from *raw_name* so
    object names compare cleanly across raw and smoothed reports.

load_simulation_results(outdir_abs, selected, use_smoothed)
    High-level loader that reads raw (and optionally smoothed) summary CSVs
    for each simulation in *selected* and returns a dict ready for plotting.
"""
import os
import glob
import csv as csv_mod


def find_csv_file(search_dir, extra_candidates=None):
    """Return the first existing CSV: explicit *extra_candidates* first,
    then any *.csv found alphabetically in *search_dir*."""
    candidates = list(extra_candidates or [])
    if os.path.isdir(search_dir):
        for f in sorted(glob.glob(os.path.join(search_dir, "*.csv"))):
            if f not in candidates:
                candidates.append(f)
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def read_csv(path):
    """Read *path* and return its rows as a list of dicts."""
    with open(path, "r", newline="") as fh:
        return list(csv_mod.DictReader(fh))


def detect_column_keys(rows):
    """Heuristically detect column keys from the first row of *rows*.

    Returns ``(name_key, power_key, density_key, species_power, species_density)``.
    *species_power* and *species_density* are ``{suffix: col_name}`` dicts for
    per-species columns (e.g. ``"H-"``, ``"H0"``, ``"H+"``).
    """
    name_key = power_key = density_key = None
    species_power = {}
    species_density = {}
    if not rows:
        return None, None, None, {}, {}
    keys = list(rows[0].keys())
    for k in keys:
        kl = k.lower()
        if "name" in kl or "file" in kl:
            name_key = k
        elif "total" in kl and "power" in kl:
            for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                if k.endswith(suffix):
                    species_power[suffix[1:]] = k
                    break
            else:
                if power_key is None:
                    power_key = k
        elif "peak" in kl or "density" in kl:
            for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                if k.endswith(suffix):
                    species_density[suffix[1:]] = k
                    break
            else:
                if density_key is None:
                    density_key = k
    # Fallback to positional keys when heuristics fail
    if name_key is None and len(keys) >= 1:
        name_key = keys[0]
    if power_key is None and len(keys) >= 2:
        power_key = keys[1]
    if density_key is None and len(keys) >= 3:
        density_key = keys[2]
    return name_key, power_key, density_key, species_power, species_density


def normalize_name(raw_name):
    """Strip common file-name prefixes so object names match across reports."""
    base = os.path.splitext(os.path.basename(str(raw_name)))[0]
    for prefix in ("results_", "smoothed_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return base


def load_simulation_results(outdir_abs, selected, use_smoothed):
    """Load raw (and optionally smoothed) summary CSVs for each result set.

    Parameters
    ----------
    outdir_abs : str
        Absolute path to the output directory.
    selected : list[str]
        Names of the result-set sub-folders to load.
    use_smoothed : bool
        When True, replace raw values with smoothed ones where available.

    Returns
    -------
    dict with keys:
      ``"plot_data"``      : {set_name: [entry_dicts]} ready for bar-chart rendering
      ``"first_merged"``   : {obj_name: entry_dict} for the first loaded sim (treeview)
      ``"first_order"``    : [obj_name, ...] display order for the first sim
      ``"first_raw_path"`` : str path to the first raw CSV found (for status display)
      ``"n_loaded"``       : int number of successfully loaded result sets
      ``"n_errors"``       : int number of result sets that failed to load
    """
    plot_data = {}
    first_merged = None
    first_display_order = None
    first_raw_path = None
    n_loaded = 0
    n_errors = 0

    for result_set in selected:
        base_dir = os.path.join(outdir_abs, result_set)
        raw_candidates = [
            os.path.join(base_dir, f"summary_{result_set}.csv"),
            os.path.join(base_dir, "summary_report.csv"),
            os.path.join(base_dir, "power_summary_by_object.csv"),
        ]
        raw_path = find_csv_file(base_dir, raw_candidates)
        if raw_path is None:
            n_errors += 1
            continue

        try:
            raw_rows = read_csv(raw_path)
        except Exception:
            n_errors += 1
            continue

        raw_nk, raw_pk, raw_dk, raw_sp, raw_sd = detect_column_keys(raw_rows)

        merged = {}
        display_order = []
        for row in raw_rows:
            obj_raw = row.get(raw_nk, "?") if raw_nk else "?"
            obj = normalize_name(obj_raw)
            tp = row.get(raw_pk, "N/A") if raw_pk else "N/A"
            pd_ = row.get(raw_dk, "N/A") if raw_dk else "N/A"
            entry = {"name": obj, "tp": tp, "pd": pd_, "source": "raw"}
            for sp_key, col in raw_sp.items():
                entry[f"tp_{sp_key}"] = row.get(col, "N/A")
            for sp_key, col in raw_sd.items():
                entry[f"pd_{sp_key}"] = row.get(col, "N/A")
            merged[obj] = entry
            if obj not in display_order:
                display_order.append(obj)

        if use_smoothed:
            sm_dir = os.path.join(base_dir, "SMOOTHED")
            sm_candidates = [
                os.path.join(sm_dir, "smoothed_summary.csv"),
                os.path.join(sm_dir, "summary_report.csv"),
            ]
            smoothed_path = find_csv_file(sm_dir, sm_candidates)
            if smoothed_path:
                try:
                    sm_rows = read_csv(smoothed_path)
                    sm_nk, sm_pk, sm_dk, sm_sp, sm_sd = detect_column_keys(sm_rows)
                    for row in sm_rows:
                        obj_raw = row.get(sm_nk, "?") if sm_nk else "?"
                        obj = normalize_name(obj_raw)
                        tp = row.get(sm_pk, "N/A") if sm_pk else "N/A"
                        pd_ = row.get(sm_dk, "N/A") if sm_dk else "N/A"
                        prev_entry = merged.get(obj, {"name": obj})
                        entry = dict(prev_entry)
                        entry.update({"tp": tp, "pd": pd_, "source": "smoothed"})
                        for sp_key, col in sm_sp.items():
                            entry[f"tp_{sp_key}"] = row.get(col, "N/A")
                        for sp_key, col in sm_sd.items():
                            entry[f"pd_{sp_key}"] = row.get(col, "N/A")
                        merged[obj] = entry
                        if obj not in display_order:
                            display_order.append(obj)
                except Exception:
                    pass

        entries = [merged[o] for o in display_order]
        plot_data[result_set] = entries
        n_loaded += 1

        if first_merged is None:
            first_merged = merged
            first_display_order = display_order
            first_raw_path = raw_path

    return {
        "plot_data": plot_data,
        "first_merged": first_merged,
        "first_order": first_display_order,
        "first_raw_path": first_raw_path,
        "n_loaded": n_loaded,
        "n_errors": n_errors,
    }
