#!/usr/bin/env python3
# paraview_utils.py
"""
ParaView script generation and launcher helpers for BeamOnTarget.

Functions:
  - _pv_geometry_script(cfg)  -- build a Python script to load STL geometry
  - _pv_results_script(dir)   -- build a Python script to load VTP results
  - launch_paraview(...)       -- write temp script and spawn ParaView process
"""
import os
import sys
import glob
import subprocess
import threading
from tkinter import messagebox

from gui.gui_widgets import resolve_path as _resolve_path, _SCRIPT_DIR

# ===================================================================
#  ParaView launcher helpers
# ===================================================================

def _pv_geometry_script(cfg):
    """Return a Python script string that ParaView will execute to show STLs."""
    folders = cfg.get("GEOMETRY_FOLDERS", {})
    colors = [
        (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
        (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
        (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
    ]
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
        "rv.ResetCamera()",
    ]
    ci = 0
    for folder, settings in folders.items():
        scale = settings.get("scale", 1)
        folder_abs = _resolve_path(folder)
        if not os.path.isdir(folder_abs):
            continue
        stl_files = sorted(glob.glob(os.path.join(folder_abs, "*.stl")) +
                           glob.glob(os.path.join(folder_abs, "*.STL")))
        r, g, b = colors[ci % len(colors)]
        ci += 1
        for stl in stl_files:
            name = os.path.splitext(os.path.basename(stl))[0]
            lines.append(f"reader = STLReader(FileNames=[r'{stl}'])")
            if scale != 1:
                lines.append(f"t = Transform(Input=reader)")
                lines.append(f"t.Transform.Scale = [{scale}, {scale}, {scale}]")
                lines.append(f"dp = Show(t, rv)")
            else:
                lines.append(f"dp = Show(reader, rv)")
            lines.append(f"dp.DiffuseColor = [{r}, {g}, {b}]")
            lines.append(f"dp.Opacity = 0.85")
            lines.append(f"RenameSource('{folder}/{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    return "\n".join(lines)


def _pv_results_script(results_dir):
    """Return a Python script that opens all .vtp files in a results dir."""
    vtp_files = sorted(glob.glob(os.path.join(results_dir, "**", "*.vtp"), recursive=True))
    if not vtp_files:
        return None
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
    ]
    for vtp in vtp_files:
        name = os.path.splitext(os.path.basename(vtp))[0]
        lines.append(f"reader = XMLPolyDataReader(FileName=[r'{vtp}'])")
        lines.append(f"dp = Show(reader, rv)")
        lines.append(f"dp.SetRepresentationType('Surface')")
        lines.append(f"ColorBy(dp, ('CELLS', 'Power_Density_W_m2'))")
        lines.append(f"dp.RescaleTransferFunctionToDataRange(True, False)")
        lines.append(f"RenameSource('{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    lines.append("# Show color bar")
    lines.append("dp.SetScalarBarVisibility(rv, True)")
    return "\n".join(lines)


def launch_paraview(script_content, pv_path, pv_module="ParaView"):
    """Write a temp script and launch ParaView.

    On Linux the EasyBuild module *pv_module* is loaded first (via
    ``bash -lc 'ml …'``) so that LD_LIBRARY_PATH etc. are set.

    On Windows ParaView is invoked directly — no module system needed.
    """
    tmp_script = os.path.join(_SCRIPT_DIR, ".pv_temp_script.py")
    with open(tmp_script, "w") as f:
        f.write(script_content)

    try:
        if sys.platform == "win32":
            # Windows: call ParaView executable directly
            proc = subprocess.Popen(
                [pv_path, "--script=" + tmp_script],
                cwd=_SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
        else:
            # Linux / macOS: load EasyBuild module, then launch
            shell_cmd = (f'ml {pv_module} 2>/dev/null; '
                         f'"{pv_path}" --script="{tmp_script}"')
            proc = subprocess.Popen(
                ["bash", "-l", "-c", shell_cmd],
                cwd=_SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)

        # Start a thread to watch for early failure (give it a few seconds)
        def _watch():
            try:
                proc.wait(timeout=5)
                if proc.returncode and proc.returncode != 0:
                    err = proc.stderr.read().strip() if proc.stderr else ""
                    msg = f"ParaView exited with code {proc.returncode}."
                    if err:
                        msg += f"\n\n{err[:600]}"
                    try:
                        messagebox.showerror("ParaView Error", msg)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                pass  # still running — good
        threading.Thread(target=_watch, daemon=True).start()
    except FileNotFoundError:
        messagebox.showerror("ParaView not found",
                             f"Cannot find ParaView at:\n{pv_path}\n\n"
                             "Update the path in the General tab.")


# ===================================================================
#  Main GUI Application
# ===================================================================
