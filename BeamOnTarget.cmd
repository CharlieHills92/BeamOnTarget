@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\venv"
set "PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "GUI_SCRIPT=%SCRIPT_DIR%\main.py"
if not exist "%GUI_SCRIPT%" (
    echo ERROR: Cannot find main.py in:
    echo   %SCRIPT_DIR%
    pause
    exit /b 1
)
if exist "%PYTHONW%" (
    start "" "%PYTHONW%" "%GUI_SCRIPT%"
    exit /b 0
)
if exist "%PYTHON%" (
    start "" "%PYTHON%" "%GUI_SCRIPT%"
    exit /b 0
)
echo ERROR: Virtual environment not found in:
echo   %VENV_DIR%
echo.
echo Run install.bat first.
pause
exit /b 1