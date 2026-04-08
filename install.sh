#!/usr/bin/env bash
# install.sh — Deploy BeamOnTarget into a Python virtual environment.
#
# Usage:
#   ./install.sh              # installs into ./venv (default)
#   ./install.sh /opt/bot     # installs into /opt/bot
#
# After installation, activate the venv and run:
#   beamontarget              # launches the GUI
#   python run_simulation.py  # runs a headless simulation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-${SCRIPT_DIR}/venv}"

echo "=== BeamOnTarget Installer ==="
echo "  Source:  ${SCRIPT_DIR}"
echo "  Venv:   ${VENV_DIR}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install the package in editable mode (so config.json / STLs stay in place)
echo "Installing BeamOnTarget..."
pip install -e "${SCRIPT_DIR}"

echo ""
echo "=== Installation complete ==="
echo ""
echo "To use:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  beamontarget                  # launch the GUI"
echo "  python run_simulation.py      # run a simulation"
echo ""
