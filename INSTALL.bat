@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title StrimerCameraControl - Installation

echo.
echo ================================================
echo       StrimerCameraControl - First setup
echo ================================================
echo.

rem Use Python 3.12, then 3.11. Keep this BAT ASCII-only and without BOM.
set "PYTHON_CMD="
py -3.12 -c "import sys; print(sys.version)" >nul 2>&1 && set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD py -3.11 -c "import sys; print(sys.version)" >nul 2>&1 && set "PYTHON_CMD=py -3.11"

if not defined PYTHON_CMD (
    where winget >nul 2>&1
    if errorlevel 1 goto :python_missing
    echo Python 3.12 was not found. Installing it with Windows Package Manager...
    winget install --id Python.Python.3.12 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto :python_missing
    set "PYTHON_CMD=py -3.12"
    py -3.12 -c "import sys" >nul 2>&1
    if errorlevel 1 goto :python_missing
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the local Python environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :python_error
)
if not exist ".venv\Scripts\python.exe" goto :python_error

echo.
echo Installing Python packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check "PySide6>=6.8" "opencv-python>=4.10" "mediapipe>=1.2.0" "pyautogui>=0.9.54" "pyvirtualcam>=0.14.0"
if errorlevel 1 goto :pip_error

if not exist "assets\hand_landmarker.task" goto :file_error
if not exist "assets\face_detector.tflite" goto :file_error
if not exist "assets\virtual_camera\InstallDriver.bat" goto :file_error

echo.
echo Installing the StrimerCameraControl virtual camera...
call "assets\virtual_camera\InstallDriver.bat"
if errorlevel 1 goto :driver_error

echo.
echo ================================================
echo Installation completed successfully!
echo.
echo To start the app, double-click START.vbs.
echo The app itself runs without administrator rights.
echo ================================================
echo.
echo Press any key to close this window...
pause >nul
exit /b 0

:python_missing
echo.
echo ERROR: Python 3.11 or 3.12 x64 could not be installed or found.
echo Install Python 3.12 x64 and run INSTALL.bat again.
echo.
goto :wait_error

:python_error
echo.
echo ERROR: Could not create the local Python environment.
echo Check that Python 3.11/3.12 x64 is installed.
echo.
goto :wait_error

:pip_error
echo.
echo ERROR: Installing Python packages failed.
echo Check your internet connection and run INSTALL.bat again.
echo.
goto :wait_error

:driver_error
echo.
echo ERROR: Virtual camera driver installation failed or was cancelled.
echo Run INSTALL.bat again and approve the administrator prompt.
echo.
goto :wait_error

:file_error
echo.
echo ERROR: One or more required project files are missing.
echo Extract the complete project folder and run INSTALL.bat again.
echo.
goto :wait_error

:wait_error
echo Press any key to close this window...
pause >nul
exit /b 1
