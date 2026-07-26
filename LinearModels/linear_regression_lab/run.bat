@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if exist "D:\anaconda3\python.exe" (
  "D:\anaconda3\python.exe" main.py %*
) else (
  python main.py %*
)
pause
