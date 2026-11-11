@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "offline_ai\wheelhouse\*.whl" goto missing_files
if not exist "models\topviewmouse\pose.pt" goto missing_files
if not exist "models\topviewmouse\detector.pt" goto missing_files
if not exist "models\torch\hub\checkpoints\mobilenet_v3_large-8738ca79.pth" goto missing_files
if not exist ".ai_venv\Scripts\python.exe" goto check_old_env
".ai_venv\Scripts\python.exe" -c "import sys,importlib.metadata as m; assert sys.version_info[:2] in ((3,11),(3,12)); assert m.version('deeplabcut') == '3.0.2'; import torch,cv2" >nul 2>&1
if not errorlevel 1 exit /b 0
:check_old_env
if exist ".ai_venv" goto backup_env
goto create_env
:backup_env
set "BACKUP_NAME=.ai_venv_backup_%RANDOM%"
if exist "%BACKUP_NAME%" goto backup_env
echo Saving the old AI environment as %BACKUP_NAME%.
ren ".ai_venv" "%BACKUP_NAME%"
if errorlevel 1 goto failed
:create_env
echo Creating the CPU AI environment...
set "AI_BASE=.venv\Scripts\python.exe"
"%AI_BASE%" -c "import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto base_ready
set "AI_BASE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%AI_BASE%" goto base_ready
if not exist "offline_files\python-3.11.9-amd64.exe" goto wrong_python
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; if ((Get-FileHash -Algorithm SHA256 -LiteralPath 'offline_files\python-3.11.9-amd64.exe').Hash -ne '5EE42C4EEE1E6B4464BB23722F90B45303F79442DF63083F05322F1785F5FDDE') { exit 1 }"
if errorlevel 1 goto failed
echo Installing Python 3.11 for the current user...
start "" /wait "%~dp0offline_files\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 Include_test=0
if errorlevel 1 goto failed
:base_ready
"%AI_BASE%" -c "import sys,tkinter; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto wrong_python
"%AI_BASE%" -m venv .ai_venv
if errorlevel 1 goto failed
echo Installing offline AI packages. This can take several minutes...
".ai_venv\Scripts\python.exe" -m pip install --no-index --find-links "%~dp0offline_ai\wheelhouse" --only-binary=:all: "deeplabcut==3.0.2"
if errorlevel 1 goto failed
".ai_venv\Scripts\python.exe" -c "import importlib.metadata as m; assert m.version('deeplabcut') == '3.0.2'; import torch,cv2"
if errorlevel 1 goto failed
echo CPU AI setup completed.
exit /b 0
:missing_files
echo Bundled AI files are missing. Copy or extract the entire package and retry.
exit /b 1
:wrong_python
echo The AI offline package requires Python 3.11 x64 in the main environment.
exit /b 1
:failed
echo AI setup failed. Check the messages above.
exit /b 1
