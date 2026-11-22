@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run INSTALL.bat to install dependencies.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0
