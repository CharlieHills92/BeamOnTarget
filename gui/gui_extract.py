#!/usr/bin/env python3
# gui_extract.py
"""
Utility dialogs for BeamOnTarget -- VTP data extraction and result-set picker.

Classes:
  - _ExtractDialog -- dialog for exporting VTP cell data to CSV
  - _PickDialog    -- simple list-picker for selecting a result set
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from gui.gui_widgets import _SCRIPT_DIR

# ===================================================================
#  VTP data extraction dialog
# ===================================================================
class _ExtractDialog(tk.Toplevel):
    """Dialog for extracting mesh cell data from a VTP file to CSV."""

    def __init__(self, parent, initial_dir):
        super().__init__(parent)
        self.title("Extract Results Data")
        self.geometry("560x440")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self._parent = parent

        pad = dict(padx=12, pady=4)

        # --- Input VTP file ---
        ttk.Label(self, text="Input VTP file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=0, column=0, sticky="w", **pad)

        self.var_input = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_input, width=48).grid(
            row=0, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=lambda: self._browse_vtp(initial_dir)).grid(
            row=0, column=2, padx=(0, 12), pady=4)

        # --- Output CSV file ---
        ttk.Label(self, text="Output CSV file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=1, column=0, sticky="w", **pad)

        self.var_output = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_output, width=48).grid(
            row=1, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=self._browse_output).grid(
            row=1, column=2, padx=(0, 12), pady=4)

        # --- Properties to export ---
        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        prop_lbl = ttk.Label(self, text="Properties to export:",
                              font=("Segoe UI", 10, "bold"),
                              background="#f0f2f5")
        prop_lbl.grid(row=3, column=0, sticky="w", **pad)

        prop_frm = ttk.Frame(self)
        prop_frm.grid(row=4, column=0, columnspan=3, sticky="w", padx=16)

        self.var_geometry = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export geometry (X, Y, Z)",
                         variable=self.var_geometry).pack(anchor="w", pady=1)

        self.var_area = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export cell area",
                         variable=self.var_area).pack(anchor="w", pady=1)

        self.var_power = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power (Deposited_Power_W)",
                         variable=self.var_power).pack(anchor="w", pady=1)

        self.var_powerload = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power load (Power_Density_W_m2)",
                         variable=self.var_powerload).pack(anchor="w", pady=1)

        # --- Multiplier ---
        ttk.Separator(self, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        mult_frm = ttk.Frame(self)
        mult_frm.grid(row=6, column=0, columnspan=3, sticky="w", padx=12)

        ttk.Label(mult_frm, text="Multiplication factor:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").pack(side="left")
        self.var_mult = tk.StringVar(value="1.0")
        ttk.Entry(mult_frm, textvariable=self.var_mult, width=10).pack(
            side="left", padx=(8, 0))
        ttk.Label(mult_frm, text="(applied to power & power load)",
                  background="#f0f2f5",
                  foreground="#64748b").pack(side="left", padx=(8, 0))

        # --- Ignore zeros ---
        self.var_ignore_zeros = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Ignore zero-valued rows",
                         variable=self.var_ignore_zeros).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=16, pady=(8, 4))

        # --- Status label ---
        self.var_status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.var_status,
                  background="#f0f2f5", foreground="#64748b",
                  font=("Segoe UI", 9)).grid(
            row=8, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 0))

        # --- Buttons ---
        btn_frm = ttk.Frame(self)
        btn_frm.grid(row=9, column=0, columnspan=3, pady=12)

        ttk.Button(btn_frm, text="💾  Save CSV",
                    command=self._do_extract).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="Cancel",
                    command=self.destroy).pack(side="left", padx=4)

        self.columnconfigure(1, weight=1)

    def _browse_vtp(self, initial_dir):
        p = filedialog.askopenfilename(
            parent=self,
            initialdir=initial_dir,
            title="Select VTP file",
            filetypes=[("VTP files", "*.vtp"), ("All files", "*")])
        if p:
            self.var_input.set(p)
            # Auto-fill output name
            if not self.var_output.get():
                base = os.path.splitext(p)[0]
                self.var_output.set(base + "_extracted.csv")

    def _browse_output(self):
        init_dir = os.path.dirname(self.var_input.get()) or _SCRIPT_DIR
        p = filedialog.asksaveasfilename(
            parent=self,
            initialdir=init_dir,
            title="Save CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if p:
            self.var_output.set(p)

    def _do_extract(self):
        """Run the extraction in a background thread."""
        input_vtp = self.var_input.get().strip()
        output_csv = self.var_output.get().strip()
        if not input_vtp:
            messagebox.showwarning("Missing", "Please select an input VTP file.",
                                    parent=self)
            return
        if not output_csv:
            messagebox.showwarning("Missing", "Please specify an output file.",
                                    parent=self)
            return
        if not os.path.isfile(input_vtp):
            messagebox.showerror("Not found",
                                  f"Input file not found:\n{input_vtp}",
                                  parent=self)
            return

        try:
            mult = float(self.var_mult.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Multiplication factor must be a number.",
                                    parent=self)
            return

        export_geom = self.var_geometry.get()
        export_area = self.var_area.get()
        export_power = self.var_power.get()
        export_pload = self.var_powerload.get()
        ignore_zeros = self.var_ignore_zeros.get()

        if not any([export_geom, export_area, export_power, export_pload]):
            messagebox.showwarning("Nothing selected",
                                    "Select at least one property to export.",
                                    parent=self)
            return

        self.var_status.set("Extracting…")
        self.update_idletasks()

        def _worker():
            try:
                import pyvista as pv
                import pandas as pd
                import numpy as np

                dataset = pv.read(input_vtp)
                if isinstance(dataset, pv.MultiBlock):
                    if dataset.n_blocks == 0:
                        self.after(0, lambda: self.var_status.set(
                            "ERROR: MultiBlock dataset is empty."))
                        return
                    mesh = dataset.combine()
                elif isinstance(dataset, pv.PolyData):
                    mesh = dataset
                else:
                    self.after(0, lambda: self.var_status.set(
                        f"ERROR: Unsupported type {type(dataset)}"))
                    return

                mesh.clean(inplace=True)
                if mesh.n_cells == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: Mesh has 0 cells."))
                    return

                face_centers = mesh.cell_centers().points
                valid = np.isfinite(face_centers).all(axis=1)
                n_valid = int(np.sum(valid))
                if n_valid == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: No cells with finite coordinates."))
                    return

                # Build DataFrame
                if export_geom:
                    df = pd.DataFrame(face_centers[valid],
                                       columns=["X", "Y", "Z"])
                else:
                    df = pd.DataFrame(index=range(n_valid))

                if export_area:
                    try:
                        mesh = mesh.compute_cell_sizes()
                        df["Area"] = mesh.cell_data["Area"][valid]
                    except Exception:
                        pass

                power_cols = []
                if export_power:
                    key = "Deposited_Power_W"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                if export_pload:
                    key = "Power_Density_W_m2"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                # Filter zeros
                if ignore_zeros and power_cols:
                    mask = pd.Series([True] * len(df))
                    for col in power_cols:
                        if col in df.columns:
                            mask &= (np.abs(df[col]) > 1e-12)
                    df = df[mask].reset_index(drop=True)

                n_rows = len(df)
                df.to_csv(output_csv, index=False, float_format="%.6e")

                self.after(0, lambda: self.var_status.set(
                    f"✔ Saved {n_rows} rows to {os.path.basename(output_csv)}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Done",
                    f"Extracted {n_rows} rows to:\n{output_csv}",
                    parent=self))

            except Exception as e:
                self.after(0, lambda: self.var_status.set(f"ERROR: {e}"))

        threading.Thread(target=_worker, daemon=True).start()


# ===================================================================
#  Small pick-list dialog
# ===================================================================
class _PickDialog(tk.Toplevel):
    def __init__(self, parent, title, items):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x340")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self.result = None

        ttk.Label(self, text="Select a result set:",
                   font=("Segoe UI", 11, "bold"),
                   background="#f0f2f5", foreground="#2563eb").pack(
            anchor="w", padx=12, pady=(12, 6))

        lb = tk.Listbox(self, height=12, font=("Segoe UI", 10),
                          bg="white", fg="#1e293b",
                          selectbackground="#2563eb",
                          selectforeground="white",
                          highlightthickness=0, bd=1, relief="solid")
        lb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        for item in items:
            lb.insert("end", item)
        lb.selection_set(0)

        def _ok():
            sel = lb.curselection()
            if sel:
                self.result = lb.get(sel[0])
            self.destroy()

        ttk.Button(self, text="Open in ParaView", command=_ok).pack(pady=(0, 12))
        lb.bind("<Double-1>", lambda e: _ok())


# ===================================================================
#  Entry point
# ===================================================================
