@echo off
cd /d "%~dp0"
echo === Gesture Control: first-time setup ===
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Python 3.12 or 3.13 x64 was not found. Install Python and try again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check the error above.
  pause
  exit /b 1
)
echo.
echo Setup complete. Double-click START.vbs to launch without a command window.
pause
