"""Shared GUI helpers, constants, and the ``make_card`` factory.

All tab modules import from here so that styling is consistent.
"""
import tkinter as tk
from tkinter import ttk
import os, sys

# ---------------------------------------------------------------------------
#  Path helpers
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else os.path.dirname(os.path.abspath(__file__)))


def resolve_path(relative_path):
    """Resolve a simulation-file path relative to the main application folder."""
    return os.path.join(_SCRIPT_DIR, relative_path)


def parse_vec3(text):
    """Parse a comma-separated string into a list of 3 floats."""
    parts = [s.strip() for s in text.split(",")]
    return [float(parts[i]) if i < len(parts) else 0.0 for i in range(3)]


# ---------------------------------------------------------------------------
#  Card factory
# ---------------------------------------------------------------------------
def make_card(parent, title=None, padx=12, pady=(0, 10)):
    """Return a content Frame inside a white card with rounded-look padding."""
    wrapper = ttk.Frame(parent, style="Card.TFrame", padding=14)
    wrapper.pack(fill="x", padx=padx, pady=pady)
    if title:
        ttk.Label(wrapper, text=title, style="CardHeader.TLabel").pack(
            anchor="w", pady=(0, 8))
    content = ttk.Frame(wrapper, style="Card.TFrame")
    content.pack(fill="both", expand=True)
    return content


# ---------------------------------------------------------------------------
#  Common browsing helpers
# ---------------------------------------------------------------------------
def browse_directory(var, title="Select directory"):
    """Ask user for a directory and store the (relative) path in *var*."""
    from tkinter import filedialog
    d = filedialog.askdirectory(initialdir=_SCRIPT_DIR, title=title)
    if d:
        try:
            rel = os.path.relpath(d, _SCRIPT_DIR)
            var.set(rel)
        except ValueError:
            var.set(d)
