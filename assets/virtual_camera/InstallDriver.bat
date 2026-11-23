@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title StrimerCameraControl - Virtual Camera Registration

set "DLL32=%~dp0UnityCaptureFilter32.dll"
set "DLL64=%~dp0UnityCaptureFilter64.dll"
set "REG32=%SystemRoot%\SysWOW64\regsvr32.exe"
set "REG64=%SystemRoot%\System32\regsvr32.exe"

if not exist "%DLL32%" (
    echo ERROR: Missing 32-bit DLL: "%DLL32%"
    exit /b 2
)
if not exist "%DLL64%" (
    echo ERROR: Missing 64-bit DLL: "%DLL64%"
    exit /b 2
)
if not exist "%REG32%" (
    echo ERROR: 32-bit regsvr32.exe was not found.
    exit /b 5
)
if not exist "%REG64%" (
    echo ERROR: 64-bit regsvr32.exe was not found.
    exit /b 6
)

:: This script is called by INSTALL.bat after that main installer has elevated.
fltmc >nul 2>&1
if errorlevel 1 (
    echo ERROR: Run the main INSTALL.bat. It is responsible for requesting administrator rights.
    exit /b 7
)

echo.
echo Registering 32-bit Unity Capture filter...
"%REG32%" /s /i:UnityCaptureName=StrimerCameraControl "%DLL32%"
if errorlevel 1 (
    echo ERROR: 32-bit Unity Capture registration failed.
    exit /b 3
)

echo Registering 64-bit Unity Capture filter...
"%REG64%" /s /i:UnityCaptureName=StrimerCameraControl "%DLL64%"
if errorlevel 1 (
    echo ERROR: 64-bit Unity Capture registration failed.
    exit /b 4
)

echo.
echo SUCCESS: Both Unity Capture filters were registered as StrimerCameraControl.
echo Restart OBS, Discord and other camera applications to refresh their device lists.
exit /b 0
