@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python -m src.data.preprocess
) else (
  python -m src.data.preprocess
)
pause
