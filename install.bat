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
REM   python -m beamontarget.workflows.run_simulation -i config.json  — runs a headless simulation

setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "VENV_DIR=%~1"
if "%VENV_DIR%"=="" set "VENV_DIR=%SCRIPT_DIR%\venv"

echo === BeamOnTarget Installer ===
echo   Source:  %SCRIPT_DIR%
echo   Venv:   %VENV_DIR%
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

echo Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo Installing BeamOnTarget...
"%VENV_DIR%\Scripts\python.exe" -m pip install -e "%SCRIPT_DIR%"
if errorlevel 1 goto :error

echo.
echo === Installation complete ===
echo.
echo To use:
echo   BeamOnTarget.cmd              ^<^< double-click to launch the GUI
echo   %VENV_DIR%\Scripts\activate
echo   beamontarget                  — launch the GUI
echo   python -m beamontarget.workflows.run_simulation -i config.json
echo.

endlocal
exit /b 0

:error
echo.
echo ERROR: Installation failed. Review the message above.
endlocal
exit /b 1
