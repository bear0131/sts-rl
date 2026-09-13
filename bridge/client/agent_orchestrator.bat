@echo off
setlocal
cd /d "%~dp0..\.."
uv run --no-sync python -m bridge.client.agent_orchestrator %*
