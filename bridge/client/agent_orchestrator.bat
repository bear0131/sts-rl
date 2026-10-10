@echo off
setlocal
cd /d "%~dp0..\.."
if defined STS_RL_PYTHON if exist "%STS_RL_PYTHON%" (
  "%STS_RL_PYTHON%" -u -m bridge.client.agent_orchestrator %*
  exit /b %ERRORLEVEL%
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    ".venv\Scripts\python.exe" -u -m bridge.client.agent_orchestrator %*
    exit /b %ERRORLEVEL%
  )
)
if exist "%~dp0..\..\..\..\..\.venv\Scripts\python.exe" (
  "%~dp0..\..\..\..\..\.venv\Scripts\python.exe" -c "import numpy" >nul 2>nul
  if not errorlevel 1 (
    "%~dp0..\..\..\..\..\.venv\Scripts\python.exe" -u -m bridge.client.agent_orchestrator %*
    exit /b %ERRORLEVEL%
  )
)
uv run --no-sync python -m bridge.client.agent_orchestrator %*
