@echo off
setlocal
set SCRIPT_DIR=%~dp0

REM Prefer workspace venv python if it exists
set VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe
if exist "%VENV_PY%" (
  set PYTHON_EXE=%VENV_PY%
) else (
  if not defined PYTHON_EXE set PYTHON_EXE=C:\Users\willi\AppData\Local\Programs\Python\Python313\python.exe
)

"%PYTHON_EXE%" "%SCRIPT_DIR%run_all.py"
endlocal
pause