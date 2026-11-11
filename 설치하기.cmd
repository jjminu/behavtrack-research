@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys, tkinter" >nul 2>&1
  if errorlevel 1 goto broken_env
  goto install
)
if exist ".venv" goto broken_env
where py >nul 2>&1
if errorlevel 1 goto check_local_python
py -3.11 -c "import sys, tkinter; assert sys.version_info >= (3, 10)" >nul 2>&1
if errorlevel 1 goto check_local_python
py -3.11 -m venv .venv
if errorlevel 1 goto failed
goto install
:check_local_python
set "LOCAL_PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%LOCAL_PYTHON%" goto use_local_python
echo Python 3.11 was not found. Downloading and installing it for this user.
echo Internet access is required. This can take several minutes.
set "PYTHON_INSTALLER=%TEMP%\behavtrack-python-3.11.9-amd64.exe"
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $env:PYTHON_INSTALLER; if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:PYTHON_INSTALLER).Hash -ne '5EE42C4EEE1E6B4464BB23722F90B45303F79442DF63083F05322F1785F5FDDE') { throw 'Python installer checksum mismatch' }"
if errorlevel 1 goto python_download_failed
echo Installing Python 3.11.9. Please wait...
start "" /wait "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_tcltk=1 Include_test=0
if errorlevel 1 goto python_install_failed
:use_local_python
"%LOCAL_PYTHON%" -c "import sys, tkinter; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if errorlevel 1 goto python_install_failed
"%LOCAL_PYTHON%" -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install ".[pose,classify]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import cv2, numpy, matplotlib, h5py, sklearn, tkinter; import behavtrack"
if errorlevel 1 goto failed
echo.
echo Installation completed. Double-click the start file.
pause
exit /b 0
:python_download_failed
echo Python download or verification failed. Check internet access and try again.
echo You can also install Python 3.11 from https://www.python.org/downloads/release/python-3119/
pause
exit /b 1
:python_install_failed
echo Python installation failed or tkinter is not available.
echo Check permissions and install Python 3.11 from https://www.python.org/downloads/release/python-3119/
pause
exit /b 1
:broken_env
echo The existing .venv cannot run on this computer.
echo Close this window, delete only the .venv folder in this program folder,
echo and double-click this installer again. Do not delete your results folder.
pause
exit /b 1
:failed
echo Installation failed. Check the messages above and the Windows instructions.
pause
exit /b 1
