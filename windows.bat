@echo off
REM ——————————————————————————————
REM  Qwirkle Project Launcher for Windows
REM ——————————————————————————————

REM 1) Create venv if it doesn’t exist
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Creating virtual environment…
    python -m venv .venv
) ELSE (
    echo Virtual environment already exists.
)

REM 2) Activate the venv
echo Activating .venv…
CALL ".venv\Scripts\activate.bat"

REM 3) Install / upgrade dependencies
echo.
echo Installing/updating requirements…
pip install --upgrade pip
pip install -r requirements.txt

REM 4) Prompt to run the main script
echo.
set /p run="Do you want to run project.py now? [Y/N] "
if /I "%run%"=="Y" (
    echo.
    echo Running project.py…
    python project.py
) ELSE (
    echo.
    echo Skipping script execution.
)

REM 5) Inform about artifacts & prompt cleanup
echo.
echo Note: The virtualenv (.venv) and any log files (e.g. app.log) remain.
set /p clean="Do you wish to remove the virtualenv and logs? [Y/N] "
if /I "%clean%"=="Y" (
    echo.
    echo Deactivating and removing .venv and logs…
    REM deactivate isn’t needed in batch; just remove
    CALL ".venv\Scripts\deactivate.bat" 2>nul
    rmdir /s /q .venv
    if exist .venv del /q .venv 
    echo Cleanup complete.
) ELSE (
    echo.
    echo Leaving .venv and logs in place.
)

echo.
echo All done.
pause
