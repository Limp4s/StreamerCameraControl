@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if /i "%~1"=="--register-virtual-camera" goto :register_virtual_camera

title Streamer Camera Control - Installation

echo.
echo ================================================
echo       Streamer Camera Control - First setup
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
if not exist "assets\virtual_camera\UnityCaptureFilter32.dll" goto :file_error
if not exist "assets\virtual_camera\UnityCaptureFilter64.dll" goto :file_error

echo.
echo Registering the Streamer Camera Control virtual camera.
echo A SECOND CMD window will request administrator permission.
set "INSTALL_SCRIPT=%~f0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$script=$env:INSTALL_SCRIPT; $q=[char]34; $cmdArgs='/c '+$q+$q+$script+$q+' --register-virtual-camera'+$q; try { $p=Start-Process -FilePath $env:ComSpec -ArgumentList $cmdArgs -Verb RunAs -Wait -PassThru -WindowStyle Normal; if ($null -eq $p) { exit 1 }; exit $p.ExitCode } catch { exit 1223 }"
set "DRIVER_RESULT=%ERRORLEVEL%"
if not "%DRIVER_RESULT%"=="0" goto :driver_error

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
echo ERROR: Virtual camera driver registration failed or UAC was cancelled.
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

:register_virtual_camera
title Streamer Camera Control - Virtual Camera Registration
set "DLL32=%~dp0assets\virtual_camera\UnityCaptureFilter32.dll"
set "DLL64=%~dp0assets\virtual_camera\UnityCaptureFilter64.dll"
set "REG32=%SystemRoot%\SysWOW64\regsvr32.exe"
set "REG64=%SystemRoot%\System32\regsvr32.exe"
set "DRIVER_RC=0"

echo.
echo ================================================
echo   Streamer Camera Control - Virtual Camera
echo ================================================
echo.

if not exist "%DLL32%" (
    echo ERROR: Missing 32-bit Unity Capture DLL.
    set "DRIVER_RC=2"
    goto :driver_registration_done
)
if not exist "%DLL64%" (
    echo ERROR: Missing 64-bit Unity Capture DLL.
    set "DRIVER_RC=2"
    goto :driver_registration_done
)
if not exist "%REG32%" (
    echo ERROR: 32-bit regsvr32.exe was not found. This installer requires 64-bit Windows.
    set "DRIVER_RC=5"
    goto :driver_registration_done
)
if not exist "%REG64%" (
    echo ERROR: 64-bit regsvr32.exe was not found.
    set "DRIVER_RC=6"
    goto :driver_registration_done
)

echo Registering 32-bit Unity Capture filter...
"%REG32%" /s "/i:UnityCaptureName=Streamer Camera Control" "%DLL32%"
if errorlevel 1 (
    echo ERROR: 32-bit Unity Capture registration failed.
    set "DRIVER_RC=3"
    goto :driver_registration_done
)

echo Registering 64-bit Unity Capture filter...
"%REG64%" /s "/i:UnityCaptureName=Streamer Camera Control" "%DLL64%"
if errorlevel 1 (
    echo ERROR: 64-bit Unity Capture registration failed.
    set "DRIVER_RC=4"
    goto :driver_registration_done
)

echo.
echo SUCCESS: Both Unity Capture filters were registered as Streamer Camera Control.
echo Restart OBS, Discord and other camera applications to refresh their device lists.

goto :driver_registration_pause

:driver_registration_done
echo.
echo Virtual camera registration FAILED. Exit code: !DRIVER_RC!

:driver_registration_pause
echo.
echo Press any key to close this administrator CMD window...
pause >nul
exit /b !DRIVER_RC!
