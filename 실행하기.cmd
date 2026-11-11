@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import tkinter, cv2, numpy, matplotlib; import behavtrack" >nul 2>&1
if errorlevel 1 goto setup
goto launch
:setup
echo Preparing BehavTrack. First setup may take several minutes.
if exist "offline_files\python-3.11.9-amd64.exe" goto offline_setup
call "%~dp0설치하기.cmd"
if errorlevel 1 goto failed
goto launch
:offline_setup
call "%~dp0offline_setup.cmd"
if errorlevel 1 goto failed
:launch
echo Recording is available without AI models. Use ai_setup.cmd only when AI setup is needed.
:start_program
echo Starting BehavTrack...
".venv\Scripts\python.exe" windows_launcher.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup or launch failed. Please photograph the messages above.
pause
exit /b 1
