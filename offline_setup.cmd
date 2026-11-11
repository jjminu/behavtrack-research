@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if /I "%PROCESSOR_ARCHITECTURE%"=="AMD64" goto architecture_ok
if /I "%PROCESSOR_ARCHITEW6432%"=="AMD64" goto architecture_ok
echo This offline package requires Windows x64.
exit /b 1
:architecture_ok
if not exist "offline_files\python-3.11.9-amd64.exe" goto missing_files
if not exist "offline_files\wheelhouse\*.whl" goto missing_files
if not exist ".venv\Scripts\python.exe" goto check_old_env
".venv\Scripts\python.exe" -c "import sys, tkinter; assert sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto install_packages
:check_old_env
if exist ".venv" goto backup_env
goto find_python
:backup_env
set "BACKUP_NAME=.venv_backup_%RANDOM%"
if exist "%BACKUP_NAME%" goto backup_env
echo Saving the old virtual environment as %BACKUP_NAME%.
ren ".venv" "%BACKUP_NAME%"
if errorlevel 1 goto failed
:find_python
set "LOCAL_PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%LOCAL_PYTHON%" goto create_env
echo Checking the bundled Python installer...
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; if ((Get-FileHash -Algorithm SHA256 -LiteralPath 'offline_files\python-3.11.9-amd64.exe').Hash -ne '5EE42C4EEE1E6B4464BB23722F90B45303F79442DF63083F05322F1785F5FDDE') { exit 1 }"
if errorlevel 1 goto installer_invalid
echo Installing Python for the current user. Please wait...
start "" /wait "%~dp0offline_files\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 Include_test=0
if errorlevel 1 goto python_failed
:create_env
"%LOCAL_PYTHON%" -c "import sys, tkinter; assert sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto python_failed
echo Creating a fresh environment...
"%LOCAL_PYTHON%" -m venv .venv
if errorlevel 1 goto failed
:install_packages
echo Installing bundled packages without internet. Please wait...
".venv\Scripts\python.exe" -m pip install --no-index --find-links "%~dp0offline_files\wheelhouse" --only-binary=:all: "opencv-python>=4.8" "numpy>=1.24" "matplotlib>=3.7" "h5py>=3.8" "scikit-learn>=1.3"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import tkinter, cv2, numpy, matplotlib, h5py, sklearn; import behavtrack"
if errorlevel 1 goto failed
echo Offline setup completed.
exit /b 0
:missing_files
echo Bundled setup files are missing. Extract the complete ZIP and retry.
exit /b 1
:installer_invalid
echo The bundled Python installer failed its integrity check. Extract the ZIP again.
exit /b 1
:python_failed
echo Python installation or tkinter validation failed. Check permissions and Windows version.
exit /b 1
:failed
echo Offline setup failed. Check the messages above.
exit /b 1
