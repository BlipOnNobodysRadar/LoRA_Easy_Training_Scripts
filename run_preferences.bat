@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -X utf8 -m main_ui_files.PreferenceWindow %*
if errorlevel 1 pause
