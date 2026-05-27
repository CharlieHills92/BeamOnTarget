"""Fields tab — per-component EM field configuration.

Provides :class:`FieldsTab`, a ``ttk.Frame`` that can be embedded in the
main notebook.  It exposes ``collect(d)`` and ``refresh(c)`` so the host
window can serialise / deserialise the field config without reaching into
widget internals.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

import numpy as np
# Using standard scipy constants. If your custom 'particles' package
# is mandatory, change this back to: from particles.constants import ...
from scipy.constants import e as ELEMENTARY_CHARGE_C, m_p as HYDROGEN_MASS_KG

# Deuterium mass isn't standard in scipy.constants; approximating as ~2 * proton mass
DEUTERIUM_MASS_KG = HYDROGEN_MASS_KG * 1.999

from fields.field_provider import create_field_provider
from gui_widgets.gui_widgets import make_card, _SCRIPT_DIR


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
_MODE_LABELS = {"zero": "None", "fixed": "Fixed value",
                "file": "Field file (.fld)", "rid_ey": "ERID (simplified)"}
_MODE_MAP = {v: k for k, v in _MODE_LABELS.items()}   # reverse


def _legacy_to_components(ef):
    """Translate an old-style flat EXTERNAL_FIELD dict into a components dict."""
    ef_type = ef.get("type", "zero")
    comps = {}
    if ef_type == "zero":
        return comps
    bvec = ef.get("magnetic_field_t", [0.0, 0.0, 0.0])
    evec = ef.get("electric_field_vpm", [0.0, 0.0, 0.0])
    for i, key in enumerate(("Bx", "By", "Bz")):
        if bvec[i] != 0:
            comps[key] = {"mode": "fixed", "value": bvec[i]}
    if ef_type in ("rid_segment_y", "rid_piecewise"):
        comps["Ey"] = {"mode": "rid_ey",
                       "v_rid_v": ef.get("v_rid_v", 20e3),
                       "x_min_m": ef.get("x_min_m", 5.4),
                       "x_max_m": ef.get("x_max_m", 7.2)}
    else:
        for i, key in enumerate(("Ex", "Ey", "Ez")):
            if evec[i] != 0:
                comps[key] = {"mode": "fixed", "value": evec[i]}
    return comps


def _parse_components(ef):
    """Return the per-component dict from an EXTERNAL_FIELD config,
    handling both new (``components``) and legacy formats."""
    comps = dict(ef.get("components", {}))
    if not comps:
        comps = _legacy_to_components(ef)
    return comps


# ---------------------------------------------------------------------------
#  FieldsTab
# ---------------------------------------------------------------------------
class FieldsTab(ttk.Frame):
    """Per-component EM field configuration panel."""

    def __init__(self, parent, cfg, *, get_collect_fn=None, **kw):
        super().__init__(parent, **kw)
        self._comp_vars = {}     # key -> {mode, val, file, rid, scale}
        self._comp_widgets = {}  # key -> {val_frame, file_frame, rid_frame, scale_frame}
        self._get_collect = get_collect_fn

        ef = cfg.get("EXTERNAL_FIELD", {})
        comps = _parse_components(ef)

        # --- Magnetic Field card ---
        b_card = make_card(self, "Magnetic Field", pady=(12, 10))
        b_modes = ["None", "Fixed value", "Field file (.fld)"]
        for i, (label, key) in enumerate([("Bx:", "Bx"), ("By:", "By"), ("Bz:", "Bz")]):
            self._make_comp_row(b_card, i, label, key, b_modes, comps.get(key, {}))
        b_card.columnconfigure(2, weight=1)

        # --- Electric Field card ---
        e_card = make_card(self, "Electric Field")
        e_modes = ["None", "Fixed value", "Field file (.fld)"]
        ey_modes = ["None", "Fixed value", "Field file (.fld)", "ERID (simplified)"]
        for i, (label, key, modes) in enumerate([
            ("Ex:", "Ex", e_modes), ("Ey:", "Ey", ey_modes), ("Ez:", "Ez", e_modes)
        ]):
            self._make_comp_row(e_card, i, label, key, modes, comps.get(key, {}))
        e_card.columnconfigure(2, weight=1)

        # --- Larmor Radius estimate card ---
        lr_card = make_card(self, "Larmor Radius Estimate")

        param_row = ttk.Frame(lr_card, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 4))
        ttk.Label(param_row, text="Species:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_lr_species = tk.StringVar(value="H")
        ttk.Combobox(param_row, textvariable=self.var_lr_species,
                      values=["H", "D"], state="readonly", width=4).pack(
            side="left", padx=(0, 12))
        ttk.Label(param_row, text="Energy [eV]:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_lr_energy = tk.DoubleVar(value=870e3)
        ttk.Entry(param_row, textvariable=self.var_lr_energy, width=12).pack(
            side="left")

        lr_row = ttk.Frame(lr_card, style="Card.TFrame")
        lr_row.pack(fill="x")
        ttk.Button(lr_row, text="Calculate", style="Secondary.TButton",
                    command=self._calc_larmor_radius).pack(side="left")
        self._var_larmor = tk.StringVar(value="—")
        ttk.Label(lr_row, textvariable=self._var_larmor,
                  style="Card.TLabel", font=("Segoe UI", 10)).pack(
            side="left", padx=(12, 0))

    # ------------------------------------------------------------------
    #  Larmor radius estimate
    # ------------------------------------------------------------------
    def _calc_larmor_radius(self):
        """Estimate Larmor radius from current field config + user-specified species/energy."""
        try:  # <--- FIXED: Added missing try block
            species = self.var_lr_species.get()
            energy_ev = self.var_lr_energy.get()
            if energy_ev <= 0:
                self._var_larmor.set("Energy must be > 0")
                return

            mass = HYDROGEN_MASS_KG if species == "H" else DEUTERIUM_MASS_KG
            q_c = ELEMENTARY_CHARGE_C          # |charge| = 1 e
            speed = np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE_C / mass)

            # Sample B field on a coarse grid through the bbox
            cfg = self._get_collect()
            ef_cfg = cfg.get("EXTERNAL_FIELD", {})
            fp = create_field_provider(ef_cfg)
            pts = np.zeros((1, 3))             # sample at origin
            _, b_field = fp.sample(pts, np.zeros(len(pts)))
            b_norm = np.linalg.norm(np.asarray(b_field, dtype=np.float64), axis=1)
            max_b = float(np.max(b_norm)) if b_norm.size > 0 else 0.0

            if max_b <= 0:
                self._var_larmor.set("r_L = ∞  (|B| = 0)")
            else:
                r_l = mass * speed / (q_c * max_b)
                self._var_larmor.set(
                    f"r_L ≈ {r_l:.4e} m   |   v = {speed:.3e} m/s   |   |B| = {max_b:.3e} T"
                )
        except Exception as exc:
            self._var_larmor.set(f"Error: {exc}")

    # ------------------------------------------------------------------
    #  Build one component row
    # ------------------------------------------------------------------
    def _make_comp_row(self, card, row, label, key, modes, init_cfg):
        mode = init_cfg.get("mode", "zero")
        init_mode = _MODE_LABELS.get(mode, "None")

        ttk.Label(card, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=3)
        mode_var = tk.StringVar(value=init_mode)
        ttk.Combobox(card, textvariable=mode_var, values=modes,
                      state="readonly", width=18).grid(
            row=row, column=1, sticky="w", padx=(8, 0), pady=3)

        # Fixed value entry
        val_var = tk.StringVar(value=str(init_cfg.get("value", 0.0)))
        val_frm = ttk.Frame(card, style="Card.TFrame")
        val_frm.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(val_frm, textvariable=val_var, width=12).pack(side="left")

        # File path entry + browse
        file_var = tk.StringVar(value=init_cfg.get("file", ""))
        file_frm = ttk.Frame(card, style="Card.TFrame")
        file_frm.grid(row=row, column=2, sticky="we", padx=(8, 0), pady=3)
        ttk.Entry(file_frm, textvariable=file_var, width=28).pack(
            side="left", fill="x", expand=True)
        ttk.Button(file_frm, text="Browse…", style="Secondary.TButton",
                    command=lambda fv=file_var: self._browse_fld_file(fv)).pack(
                        side="left", padx=(4, 0))

        # RID params frame (only for Ey)
        rid_frm = None
        rid_vars = {}
        if "ERID (simplified)" in modes:
            rid_frm = ttk.Frame(card, style="Card.TFrame")
            rid_frm.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=3)
            for rlbl, rkey, rdef in [("V:", "v_rid_v", 20000.0),
                                      ("x_min:", "x_min_m", 5.4),
                                      ("x_max:", "x_max_m", 7.2)]:
                ttk.Label(rid_frm, text=rlbl, style="Card.TLabel").pack(side="left")
                rv = tk.StringVar(value=str(init_cfg.get(rkey, rdef)))
                ttk.Entry(rid_frm, textvariable=rv, width=8).pack(
                    side="left", padx=(2, 8))
                rid_vars[rkey] = rv

        # Scale factor entry (shown for all non-None modes)
        scale_var = tk.StringVar(value=str(init_cfg.get("scale", 1.0)))
        scale_frm = ttk.Frame(card, style="Card.TFrame")
        scale_frm.grid(row=row, column=3, sticky="e", padx=(8, 0), pady=3)
        ttk.Label(scale_frm, text="×", style="Card.TLabel").pack(side="left")
        ttk.Entry(scale_frm, textvariable=scale_var, width=6).pack(
            side="left", padx=(2, 0))

        self._comp_vars[key] = {"mode": mode_var, "val": val_var,
                                 "file": file_var, "rid": rid_vars,
                                 "scale": scale_var}
        self._comp_widgets[key] = {"val_frame": val_frm, "file_frame": file_frm,
                                    "rid_frame": rid_frm, "scale_frame": scale_frm}

        def _on_mode_change(*_, k=key):
            m = self._comp_vars[k]["mode"].get()
            w = self._comp_widgets[k]
            w["val_frame"].grid() if m == "Fixed value" else w["val_frame"].grid_remove()
            w["file_frame"].grid() if m == "Field file (.fld)" else w["file_frame"].grid_remove()
            if w["rid_frame"]:
                w["rid_frame"].grid() if m == "ERID (simplified)" else w["rid_frame"].grid_remove()
            w["scale_frame"].grid() if m != "None" else w["scale_frame"].grid_remove()

        mode_var.trace_add("write", _on_mode_change)
        _on_mode_change()

    # ------------------------------------------------------------------
    #  Browse for a .fld file
    # ------------------------------------------------------------------
    def _browse_fld_file(self, var):
        p = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Select field file",
            filetypes=[("Field files", "*.fld"), ("All files", "*")])
        if p:
            try:
                rel = os.path.relpath(p, _SCRIPT_DIR)
                var.set(rel)
            except ValueError:
                var.set(p)

    # ------------------------------------------------------------------
    #  Collect field config into dict *d*
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write the EXTERNAL_FIELD key into *d*."""
        components = {}
        for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            cv = self._comp_vars[key]
            mode = _MODE_MAP.get(cv["mode"].get(), "zero")
            if mode == "zero":
                continue
            comp = {"mode": mode}
            scale = float(cv["scale"].get())
            if scale != 1.0:
                comp["scale"] = scale
            if mode == "fixed":
                comp["value"] = float(cv["val"].get())
            elif mode == "file":
                comp["file"] = cv["file"].get()
            elif mode == "rid_ey":
                for rk in ("v_rid_v", "x_min_m", "x_max_m"):
                    comp[rk] = float(cv["rid"][rk].get())
            components[key] = comp

        if components:
            d["EXTERNAL_FIELD"] = {"type": "composite", "components": components}
        else:
            d["EXTERNAL_FIELD"] = {"type": "zero"}

    # ------------------------------------------------------------------
    #  Refresh widgets from config dict *c*
    # ------------------------------------------------------------------
    def refresh(self, c):
        """Push values from config dict *c* into our widgets."""
        ef = c.get("EXTERNAL_FIELD", {})
        comps = _parse_components(ef)
        for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            cv = self._comp_vars[key]
            cfg_comp = comps.get(key, {})
            mode = cfg_comp.get("mode", "zero")
            cv["mode"].set(_MODE_LABELS.get(mode, "None"))
            cv["val"].set(str(cfg_comp.get("value", 0.0)))
            cv["file"].set(cfg_comp.get("file", ""))
            cv["scale"].set(str(cfg_comp.get("scale", 1.0)))
            for rk in ("v_rid_v", "x_min_m", "x_max_m"):
                if rk in cv["rid"]:
                    cv["rid"][rk].set(str(cfg_comp.get(rk, {"v_rid_v": 20000.0,
                                                              "x_min_m": 5.4,
                                                              "x_max_m": 7.2}[rk])))