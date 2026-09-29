@echo off
rem Crop Yield Prediction - one-click start (the equivalent of "npm run dev").
rem Double-click to set up anything missing and launch the app.
rem Arguments pass through to run.py, e.g.:  start.bat check   /   start.bat build

rem Work from this script's folder, wherever it was launched from.
cd /d "%~dp0"

rem Find a real Python 3.10+ (the "python" Microsoft Store stub does not count).
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
if not defined PY (
    echo Python 3.10 or newer was not found on this computer.
    echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
    echo then double-click start.bat again.
    pause
    exit /b 1
)

%PY% run.py %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open only on an error (0 = success, 130 = stopped with Ctrl+C).
if not "%CODE%"=="0" if not "%CODE%"=="130" pause
exit /b %CODE%
