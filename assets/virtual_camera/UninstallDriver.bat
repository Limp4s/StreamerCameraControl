@echo off
setlocal
fltmc >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator permission to remove the virtual camera...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process -FilePath '%~f0' -Verb RunAs -Wait -PassThru; exit $p.ExitCode"
    exit /b 0
)
"%SystemRoot%\SysWOW64\regsvr32.exe" /s /u "%~dp0UnityCaptureFilter32.dll"
"%SystemRoot%\System32\regsvr32.exe" /s /u "%~dp0UnityCaptureFilter64.dll"
echo StrimerCameraControl virtual camera has been unregistered.
pause
