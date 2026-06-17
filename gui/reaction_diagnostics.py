#!/usr/bin/env python3
# reaction_diagnostics.py
"""
Diagnostic plot mixin for ReactionsTab.

Extracted from gui_reactions.py to keep that module under ~400 lines.
Provides cross-section plots, gas density profile, species evolution,
mean-free-path calculator, and interactive cursor for the plot canvas.
"""
import os
import numpy as np
from tkinter import messagebox, filedialog

from gui.gui_widgets import _SCRIPT_DIR, parse_vec3


class ReactionDiagnosticsMixin:
    """Mixin providing diagnostic plot methods for ReactionsTab."""

    # ------------------------------------------------------------------
    #  Interactive cursor
    # ------------------------------------------------------------------
    def _on_rxn_mouse_move(self, event):
        for a in self._rxn_cursor_artists:
            try:
                a.remove()
            except (NotImplementedError, ValueError):
                pass
        self._rxn_cursor_artists.clear()

        if not self._rxn_fig.axes:
            return
        ax = self._rxn_fig.axes[0]
        if event.inaxes != ax or event.xdata is None:
            self._rxn_canvas.draw_idle()
            return

        x = event.xdata
        is_logx = ax.get_xscale() == "log"
        is_logy = ax.get_yscale() == "log"

        vl = ax.axvline(x, color="#888888", ls=":", lw=0.8, alpha=0.5)
        self._rxn_cursor_artists.append(vl)

        items = []
        for line in ax.get_lines():
            lbl = line.get_label()
            if not lbl or lbl.startswith("_"):
                continue
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if len(xd) <= 2:
                continue
            if x < xd.min() or x > xd.max():
                continue
            if is_logx and is_logy:
                yi = 10 ** np.interp(np.log10(x),
                                     np.log10(xd),
                                     np.log10(np.maximum(yd, 1e-300)))
            elif is_logx:
                yi = np.interp(np.log10(x), np.log10(xd), yd)
            else:
                yi = np.interp(x, xd, yd)
            items.append((yi, lbl, line.get_color()))

        for idx, (yi, lbl, col) in enumerate(items):
            dot, = ax.plot(x, yi, "o", color=col, ms=5, zorder=10)
            self._rxn_cursor_artists.append(dot)
            y_off = 6 * (1 if idx % 2 == 0 else -1)
            txt = ax.annotate(
                f"{lbl}: {yi:.3e}", xy=(x, yi),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=7.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col,
                          alpha=0.85, lw=0.6),
                zorder=11,
            )
            self._rxn_cursor_artists.append(txt)

        self._rxn_canvas.draw_idle()

    # ------------------------------------------------------------------
    #  Diagnostic plots
    # ------------------------------------------------------------------
    def _plot_cross_sections(self):
        import cross_sections as cs
        iso = self.var_plot_species.get()
        energy = np.logspace(1, 7, 500)
        sigma_ss = np.maximum(cs.cs_hm_single_strip(energy, isotope=iso), 1e-25)
        sigma_ds = np.maximum(cs.cs_hm_double_strip(energy, isotope=iso), 1e-25)
        sigma_ns = np.maximum(cs.cs_proj_ionization_h0(energy, isotope=iso), 1e-25)
        sigma_cx = np.maximum(cs.cs_cx_hp(energy, isotope=iso), 1e-25)

        neg = "H⁻" if iso == "H" else "D⁻"
        neu = "H⁰" if iso == "H" else "D⁰"
        pos = "H⁺" if iso == "H" else "D⁺"

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.loglog(energy, sigma_ss, label=f"{neg}→{neu} (single strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ds, label=f"{neg}→{pos} (double strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ns, label=f"{neu}→{pos} (neutral strip)", linewidth=1.8)
        ax.loglog(energy, sigma_cx, label=f"{pos}→{neu} (charge exchange)", linewidth=1.8)
        ax.set_xlabel("Particle energy [eV]")
        ax.set_ylabel("Cross section [m²]")
        ax.set_ylim(bottom=1e-25)
        ax.set_title(f"Beam–Gas Cross Sections ({iso})")
        ax.axvline(self.var_plot_energy.get(), color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.7,
                    label=f"E = {self.var_plot_energy.get():.0f} eV")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_gas_density(self):
        from reactions import BeamCrossSectionReaction

        density_dir = parse_vec3(self.var_density_dir.get())
        bbox_min = self._get_bbox_min()
        bbox_max = self._get_bbox_max()

        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Gas Density", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Gas Density",
                                 "Density direction does not traverse the bounding box.")
            return

        is_uniform = self.var_density_mode.get() == "Uniform density"
        density_file = None if is_uniform else self.var_density_file.get()
        bg_density = self.var_bg_density.get()

        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_scale=self.var_density_profile_scale.get(),
            density_profile_direction=density_dir,
            verbose=False,
        )

        steps = max(2, int(np.ceil(line_len / 0.01)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]
        dens = np.maximum(np.asarray(model._density_at_positions(positions),
                                      dtype=np.float64), 0.0)

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        if is_uniform:
            lbl = f"Uniform: {bg_density:.3e} m⁻³"
        else:
            lbl = os.path.basename(density_file) if density_file else "profile"
        ax.plot(distance, dens, color="tab:green", linewidth=2.0, label=lbl)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Gas density [m⁻³]")
        ax.set_title("Background Gas Density Profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_species_evolution(self):
        import cross_sections as cs
        from reactions import BeamCrossSectionReaction

        bbox_min = self._get_bbox_min()
        bbox_max = self._get_bbox_max()
        step_len = self._get_em_step()
        density_dir = parse_vec3(self.var_density_dir.get())

        density_file = self.var_density_file.get() \
            if self.var_density_mode.get() != "Uniform density" else None
        bg_density = self.var_bg_density.get()
        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_scale=self.var_density_profile_scale.get(),
            density_profile_direction=density_dir,
            verbose=False,
        )

        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Species Evolution", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Species Evolution",
                                 "Beam direction does not traverse the bounding box.")
            return

        steps = max(2, int(np.ceil(line_len / step_len)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]

        density_m3 = np.maximum(
            np.asarray(model._density_at_positions(positions), dtype=np.float64), 0.0)

        iso = self.var_plot_species.get()
        avg_energy_ev = self.var_plot_energy.get()
        from constants import HYDROGEN_MASS_KG, DEUTERIUM_MASS_KG, ELEMENTARY_CHARGE_C
        avg_mass_kg = HYDROGEN_MASS_KG if iso == "H" else DEUTERIUM_MASS_KG
        avg_speed = np.sqrt(2.0 * avg_energy_ev * ELEMENTARY_CHARGE_C / avg_mass_kg)

        if self.var_cs_mode.get() == "Manual (m²)":
            s_ss = float(self._cs_manual_vars["single_strip_neg_to_neutral"].get())
            s_ds = float(self._cs_manual_vars["double_strip_neg_to_positive"].get())
            s_ns = float(self._cs_manual_vars["strip_neutral_to_positive"].get())
            s_cx = float(self._cs_manual_vars["charge_exchange_pos_to_neutral"].get())
        else:
            sigma = cs.channel_cross_sections(avg_energy_ev, isotope=iso)
            s_ss = float(np.asarray(sigma[cs.CH_SINGLE_STRIP]))
            s_ds = float(np.asarray(sigma[cs.CH_DOUBLE_STRIP]))
            s_ns = float(np.asarray(sigma[cs.CH_NEUTRAL_STRIP]))
            s_cx = float(np.asarray(sigma[cs.CH_CHARGE_EXCHANGE]))

        fractions = np.zeros((steps, 3), dtype=np.float64)
        fractions[0] = [1.0, 0.0, 0.0]

        for i in range(steps - 1):
            ds = distance[i + 1] - distance[i]
            dt = ds / avg_speed
            f_neg, f_neu, f_pos = fractions[i]
            n = density_m3[i]
            dn = -(s_ss + s_ds) * n * avg_speed * f_neg
            d0 = (s_ss * n * avg_speed * f_neg
                   - s_ns * n * avg_speed * f_neu
                   + s_cx * n * avg_speed * f_pos)
            dp = (s_ds * n * avg_speed * f_neg
                   + s_ns * n * avg_speed * f_neu
                   - s_cx * n * avg_speed * f_pos)
            nxt = fractions[i] + dt * np.array([dn, d0, dp])
            nxt = np.maximum(nxt, 0.0)
            s = nxt.sum()
            if s > 0:
                nxt /= s
            fractions[i + 1] = nxt

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.plot(distance, fractions[:, 0], label="H⁻/D⁻", linewidth=2.0)
        ax.plot(distance, fractions[:, 1], label="H⁰/D⁰", linewidth=2.0)
        ax.plot(distance, fractions[:, 2], label="H⁺/D⁺", linewidth=2.0)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Species fraction")
        ax.set_title(f"Analytical Species Evolution (E={avg_energy_ev/1e3:.0f} keV)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    # ------------------------------------------------------------------
    def _browse_density_file(self):
        p = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Select density profile file",
            filetypes=[("Density files", "*.dens"), ("All files", "*")])
        if p:
            try:
                rel = os.path.relpath(p, _SCRIPT_DIR)
                self.var_density_file.set(rel)
            except ValueError:
                self.var_density_file.set(p)

    # ------------------------------------------------------------------
    #  Mean free path estimate
    # ------------------------------------------------------------------
    def _calc_mean_free_path(self):
        """Estimate minimum mean free path from reaction config + user-specified species/energy."""
        if not self._get_collect:
            self._var_mfp.set("(no config callback)")
            return
        try:
            import cross_sections as cs
            from reactions import BeamCrossSectionReaction
            from prerun_analysis import _ray_exit_distance_from_box

            bbox_min = self._get_bbox_min()
            bbox_max = self._get_bbox_max()

            # Species and energy from user fields
            isotope = self.var_plot_species.get()
            avg_energy_ev = self.var_plot_energy.get()
            if avg_energy_ev <= 0:
                self._var_mfp.set("Energy must be > 0")
                return

            # Cross sections at specified energy
            sigma = cs.channel_cross_sections(avg_energy_ev, isotope=isotope)
            sigma_neg = float(np.asarray(sigma[cs.CH_SINGLE_STRIP])) + \
                        float(np.asarray(sigma[cs.CH_DOUBLE_STRIP]))
            sigma_neu = float(np.asarray(sigma[cs.CH_NEUTRAL_STRIP]))
            sigma_pos = float(np.asarray(sigma[cs.CH_CHARGE_EXCHANGE]))

            # Density along beam direction
            density_dir = parse_vec3(self.var_density_dir.get())
            dir_arr = np.array(density_dir, dtype=np.float64)
            dir_norm = np.linalg.norm(dir_arr)
            if dir_norm <= 0:
                self._var_mfp.set("Invalid density direction.")
                return
            dir_arr /= dir_norm

            start = np.array(bbox_min, dtype=np.float64)
            line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
            if line_len is None or line_len <= 0:
                self._var_mfp.set("Direction doesn't traverse bbox.")
                return

            is_uniform = self.var_density_mode.get() == "Uniform density"
            density_file = None if is_uniform else self.var_density_file.get()
            bg_density = self.var_bg_density.get()

            model = BeamCrossSectionReaction(
                background_density_m3=bg_density,
                density_profile_file=density_file,
                density_profile_scale=self.var_density_profile_scale.get(),
                density_profile_direction=density_dir,
                verbose=False,
            )

            steps = max(2, int(np.ceil(line_len / 0.01)) + 1)
            distance = np.linspace(0.0, line_len, steps)
            positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]
            density_m3 = np.maximum(
                np.asarray(model._density_at_positions(positions), dtype=np.float64), 0.0)

            # mfp = 1 / (n * sigma) for each species transition
            mfp_arrays = []
            for s_total in (sigma_neg, sigma_neu, sigma_pos):
                if s_total > 0:
                    denom = density_m3 * s_total
                    mfp = np.where(denom > 0, 1.0 / np.maximum(denom, 1e-300), np.inf)
                    mfp_arrays.append(mfp)

            if not mfp_arrays:
                self._var_mfp.set("λ = ∞ (all σ = 0)")
                return

            all_mfp = np.concatenate(mfp_arrays)
            finite = all_mfp[np.isfinite(all_mfp) & (all_mfp > 0)]
            if finite.size == 0:
                self._var_mfp.set("λ = ∞ (zero density)")
                return

            mfp_min = float(np.min(finite))
            mfp_max = float(np.max(finite))
            em_step = self._get_em_step()
            ratio = em_step / mfp_min if mfp_min > 0 else float("inf")
            self._var_mfp.set(
                f"λ_min ≈ {mfp_min:.3e} m  |  λ_max ≈ {mfp_max:.3e} m  |  step/λ_min = {ratio:.3e}"
            )
        except Exception as exc:
            self._var_mfp.set(f"Error: {exc}")
