"""Reactions tab — background gas density, cross-section settings, diagnostic plots.

Provides :class:`ReactionsTab`, a ``ttk.Frame`` that the main window embeds
in the notebook.  Exposes ``collect(d, is_em)`` and ``refresh(c)`` so the
host can serialise / deserialise the reaction config without reaching into
widget internals.

The diagnostic plots (cross sections, gas density, species evolution) need
bounding-box and EM-step values that live in the General / Fields tabs.
Those are supplied via the *get_bbox_min*, *get_bbox_max* and *get_em_step*
callables passed at construction time.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from gui.gui_widgets import make_card, parse_vec3, _SCRIPT_DIR

# ---------------------------------------------------------------------------
from gui.reaction_diagnostics import ReactionDiagnosticsMixin


# ---------------------------------------------------------------------------
#  ReactionsTab
# ---------------------------------------------------------------------------
class ReactionsTab(ReactionDiagnosticsMixin, ttk.Frame):
    """Background gas & cross-section configuration plus diagnostic plots."""
    def __init__(self, parent, cfg, colours, *,
                 get_bbox_min, get_bbox_max, get_em_step,
                 get_collect_fn=None, **kw):
        super().__init__(parent, **kw)
        self._get_bbox_min = get_bbox_min
        self._get_bbox_max = get_bbox_max
        self._get_em_step = get_em_step
        self._get_collect = get_collect_fn

        top_pw = ttk.PanedWindow(self, orient="horizontal")
        top_pw.pack(fill="both", expand=True)

        # ===========================================================
        # LEFT side — scrollable cards
        # ===========================================================
        left_wrapper = ttk.Frame(top_pw)
        top_pw.add(left_wrapper, weight=1)

        rxn_canvas = tk.Canvas(left_wrapper, borderwidth=0,
                                highlightthickness=0, bg=colours["bg"])
        rxn_vscroll = ttk.Scrollbar(left_wrapper, orient="vertical",
                                     command=rxn_canvas.yview)
        outer = ttk.Frame(rxn_canvas)
        outer.bind("<Configure>",
                   lambda e: rxn_canvas.configure(
                       scrollregion=rxn_canvas.bbox("all")))
        rxn_canvas.create_window((0, 0), window=outer, anchor="nw")
        rxn_canvas.configure(yscrollcommand=rxn_vscroll.set)
        rxn_canvas.pack(side="left", fill="both", expand=True)
        rxn_vscroll.pack(side="right", fill="y")

        rm = cfg.get("REACTION_MODEL", {})

        # --- Background Gas Density card ---
        dens_card = make_card(outer, "Background Gas Density", pady=(12, 10))

        has_profile = bool(rm.get("density_profile_file", ""))
        dens_init = "Density profile file (.dens)" if has_profile else "Uniform density"

        row = 0
        ttk.Label(dens_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_density_mode = tk.StringVar(value=dens_init)
        ttk.Combobox(dens_card, textvariable=self.var_density_mode,
                      values=["Uniform density", "Density profile file (.dens)"],
                      state="readonly", width=28).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        # -- Uniform density widgets --
        row += 1
        self._dens_uniform_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_uniform_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self._dens_uniform_frame, text="Density (m⁻³):",
                  style="Card.TLabel").pack(side="left")
        self.var_bg_density = tk.DoubleVar(value=rm.get("background_density_m3", 0.0))
        ttk.Entry(self._dens_uniform_frame, textvariable=self.var_bg_density,
                  width=14).pack(side="left", padx=(8, 0))

        # -- Profile file widgets --
        row += 1
        self._dens_profile_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_profile_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self._dens_profile_frame, text="File:",
                  style="Card.TLabel").pack(side="left")
        self.var_density_file = tk.StringVar(value=rm.get("density_profile_file", ""))
        ttk.Entry(self._dens_profile_frame, textvariable=self.var_density_file,
                  width=30).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(self._dens_profile_frame, text="Browse…",
                    style="Secondary.TButton",
                    command=self._browse_density_file).pack(side="left")

        row += 1
        dens_scale_frame = ttk.Frame(dens_card, style="Card.TFrame")
        dens_scale_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(dens_scale_frame, text="Profile multiplier:",
                  style="Card.TLabel").pack(side="left")
        self.var_density_profile_scale = tk.DoubleVar(
            value=rm.get("density_profile_scale", 1.0)
        )
        ttk.Entry(dens_scale_frame, textvariable=self.var_density_profile_scale,
                  width=12).pack(side="left", padx=(8, 0))

        row += 1
        dens_dir_frame = ttk.Frame(dens_card, style="Card.TFrame")
        dens_dir_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(dens_dir_frame, text="Profile direction:",
                  style="Card.TLabel").pack(side="left")
        dd = rm.get("density_profile_direction",
                     cfg.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir = tk.StringVar(value=f"{dd[0]}, {dd[1]}, {dd[2]}")
        ttk.Entry(dens_dir_frame, textvariable=self.var_density_dir,
                  width=24).pack(side="left", padx=(8, 0))

        def _on_density_mode(*_):
            if self.var_density_mode.get() == "Uniform density":
                self._dens_uniform_frame.grid()
                self._dens_profile_frame.grid_remove()
            else:
                self._dens_uniform_frame.grid_remove()
                self._dens_profile_frame.grid()

        self.var_density_mode.trace_add("write", _on_density_mode)
        _on_density_mode()
        dens_card.columnconfigure(1, weight=1)

        # --- Reaction Definitions (Cross Sections) card ---
        cs_card = make_card(outer, "Reaction Definitions (Cross Sections)")

        self._cs_channels = [
            ("H⁻/D⁻ → H⁰/D⁰", "Single stripping",  "single_strip_neg_to_neutral"),
            ("H⁻/D⁻ → H⁺/D⁺", "Double stripping",   "double_strip_neg_to_positive"),
            ("H⁰/D⁰ → H⁺/D⁺", "Neutral stripping",  "strip_neutral_to_positive"),
            ("H⁺/D⁺ → H⁰/D⁰", "Charge exchange",    "charge_exchange_pos_to_neutral"),
        ]
        manual_cs = rm.get("manual_cross_sections") or {}
        has_manual = bool(manual_cs)
        cs_mode_init = "Manual (m²)" if has_manual else "Built-in polynomial fit"

        row = 0
        ttk.Label(cs_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_cs_mode = tk.StringVar(value=cs_mode_init)
        ttk.Combobox(cs_card, textvariable=self.var_cs_mode,
                      values=["Built-in polynomial fit", "Manual (m²)"],
                      state="readonly", width=24).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=(8, 0))

        row += 1
        self.var_fixed_cs = tk.BooleanVar(value=rm.get("fixed_cs", False))
        self._cs_freeze_check = ttk.Checkbutton(
            cs_card, text="Freeze at initial energy",
            variable=self.var_fixed_cs, style="Card.TCheckbutton")
        self._cs_freeze_check.grid(row=row, column=0, columnspan=3,
                                    sticky="w", pady=(2, 6))

        self._cs_manual_vars = {}
        self._cs_source_labels = {}
        self._cs_entry_widgets = {}
        self._cs_unit_labels = {}

        for i, (label, desc, key) in enumerate(self._cs_channels):
            r = row + 1 + i
            ttk.Label(cs_card, text=label, style="Card.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            ttk.Label(cs_card, text=desc, style="Card.TLabel").grid(
                row=r, column=1, sticky="w", padx=(8, 0), pady=2)

            lbl = ttk.Label(cs_card, text="(built-in polynomial fit)",
                            style="Card.TLabel")
            lbl.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_source_labels[key] = lbl

            var = tk.StringVar(value=f"{manual_cs.get(key, 0.0):.3e}")
            self._cs_manual_vars[key] = var
            ent = ttk.Entry(cs_card, textvariable=var, width=12)
            ent.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_entry_widgets[key] = ent
            unit = ttk.Label(cs_card, text="m²", style="Card.TLabel")
            unit.grid(row=r, column=3, sticky="w", padx=(4, 0), pady=2)
            self._cs_unit_labels[key] = unit

        def _on_cs_mode(*_):
            is_manual = self.var_cs_mode.get() == "Manual (m²)"
            for key in self._cs_manual_vars:
                if is_manual:
                    self._cs_source_labels[key].grid_remove()
                    self._cs_entry_widgets[key].grid()
                    self._cs_unit_labels[key].grid()
                else:
                    self._cs_entry_widgets[key].grid_remove()
                    self._cs_unit_labels[key].grid_remove()
                    self._cs_source_labels[key].grid()
            if is_manual:
                self._cs_freeze_check.grid_remove()
            else:
                self._cs_freeze_check.grid()

        self.var_cs_mode.trace_add("write", _on_cs_mode)
        _on_cs_mode()
        cs_card.columnconfigure(1, weight=0)

        # ===========================================================
        # RIGHT side — embedded diagnostic plots
        # ===========================================================
        right_frm = ttk.Frame(top_pw, style="Card.TFrame", padding=8)
        top_pw.add(right_frm, weight=1)

        ttk.Label(right_frm, text="Diagnostic Plots",
                  style="CardHeader.TLabel").pack(anchor="w", pady=(0, 4))

        param_row = ttk.Frame(right_frm, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 4))
        ttk.Label(param_row, text="Species:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_species = tk.StringVar(value="H")
        ttk.Combobox(param_row, textvariable=self.var_plot_species,
                      values=["H", "D"], state="readonly", width=4).pack(
            side="left", padx=(0, 12))
        ttk.Label(param_row, text="Energy [eV]:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_energy = tk.DoubleVar(value=870e3)
        ttk.Entry(param_row, textvariable=self.var_plot_energy, width=12).pack(
            side="left")

        btn_row = ttk.Frame(right_frm, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Cross Sections", style="Secondary.TButton",
                    command=self._plot_cross_sections).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Gas Density", style="Secondary.TButton",
                    command=self._plot_gas_density).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Species Evolution", style="Secondary.TButton",
                    command=self._plot_species_evolution).pack(side="left")

        # --- Mean Free Path estimate ---
        mfp_row = ttk.Frame(right_frm, style="Card.TFrame")
        mfp_row.pack(fill="x", pady=(4, 6))
        ttk.Button(mfp_row, text="Calc Mean Free Path", style="Secondary.TButton",
                    command=self._calc_mean_free_path).pack(side="left")
        self._var_mfp = tk.StringVar(value="—")
        ttk.Label(mfp_row, textvariable=self._var_mfp,
                  style="Card.TLabel", font=("Segoe UI", 9)).pack(
            side="left", padx=(10, 0))

        self._rxn_fig = Figure(figsize=(5.5, 4.5), dpi=100)
        self._rxn_canvas = FigureCanvasTkAgg(self._rxn_fig, master=right_frm)
        self._rxn_toolbar = NavigationToolbar2Tk(self._rxn_canvas, right_frm)
        self._rxn_toolbar.update()
        self._rxn_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._rxn_cursor_artists = []
        self._rxn_canvas.mpl_connect("motion_notify_event",
                                      self._on_rxn_mouse_move)

    # ------------------------------------------------------------------
    #  Collect reactions config into dict *d*
    # ------------------------------------------------------------------
    def collect(self, d, is_em):
        """Write REACTION_MODEL and DENSITY_DIRECTION keys into *d*."""
        if is_em:
            dd_vec = parse_vec3(self.var_density_dir.get())
            rm_dict = {
                "type": "beam_gas_cross_sections",
                "background_density_m3": self.var_bg_density.get(),
                "density_profile_file": self.var_density_file.get(),
                "density_profile_scale": self.var_density_profile_scale.get(),
                "density_profile_direction": dd_vec,
            }
            if self.var_cs_mode.get() == "Manual (m²)":
                rm_dict["manual_cross_sections"] = {
                    k: float(v.get()) for k, v in self._cs_manual_vars.items()
                }
            else:
                rm_dict["fixed_cs"] = self.var_fixed_cs.get()
            d["REACTION_MODEL"] = rm_dict
            d["DENSITY_DIRECTION"] = dd_vec
        else:
            d["REACTION_MODEL"] = {"type": "none"}

    # ------------------------------------------------------------------
    #  Refresh widgets from config dict *c*
    # ------------------------------------------------------------------
    def refresh(self, c):
        """Push values from config dict *c* into our widgets."""
        rm = c.get("REACTION_MODEL", {})
        self.var_bg_density.set(rm.get("background_density_m3", 0.0))
        self.var_density_file.set(rm.get("density_profile_file", ""))
        self.var_density_profile_scale.set(rm.get("density_profile_scale", 1.0))
        dd = rm.get("density_profile_direction",
                     c.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir.set(f"{dd[0]}, {dd[1]}, {dd[2]}")
        manual_cs = rm.get("manual_cross_sections") or {}
        if manual_cs:
            self.var_cs_mode.set("Manual (m²)")
            for key, var in self._cs_manual_vars.items():
                var.set(f"{manual_cs.get(key, 0.0):.3e}")
        else:
            self.var_cs_mode.set("Built-in polynomial fit")
            self.var_fixed_cs.set(rm.get("fixed_cs", False))

