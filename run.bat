@echo off

title LoRA Trainer - 67372a Fork - Refresh Branch
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -X utf8 main.py
if errorlevel 1 pause
