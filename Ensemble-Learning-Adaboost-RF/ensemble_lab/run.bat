@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
"D:\Anaconda3\envs\Pytorch\python.exe" -u main.py %*
