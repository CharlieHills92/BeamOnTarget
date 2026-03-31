@echo off
REM install.bat — Deploy BeamOnTarget into a Python virtual environment on Windows.
REM
REM Usage:
REM   install.bat              — installs into .\venv (default)
REM   install.bat C:\bot\env   — installs into C:\bot\env
REM
REM After installation:
REM   venv\Scripts\activate
REM   beamontarget              — launches the GUI
REM   python run_simulation.py  — runs a headless simulation

setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%1
if "%VENV_DIR%"=="" set VENV_DIR=%SCRIPT_DIR%venv

echo === BeamOnTarget Installer ===
echo   Source:  %SCRIPT_DIR%
echo   Venv:   %VENV_DIR%
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

echo Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
pip install --upgrade pip setuptools wheel

echo Installing BeamOnTarget...
pip install -e "%SCRIPT_DIR%"

echo.
echo === Installation complete ===
echo.
echo To use:
echo   %VENV_DIR%\Scripts\activate
echo   beamontarget                  — launch the GUI
echo   python run_simulation.py      — run a simulation
echo.

endlocal
