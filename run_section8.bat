@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Section 8 reproducibility run -- current paper version
echo High-d CLT - v3 (20260930-215525)
echo ============================================================
echo.

rem ------------------------------------------------------------
rem Find Python 3. Prefer the standard Windows launcher "py -3".
rem ------------------------------------------------------------
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
    goto check_packages
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
    goto check_packages
)

echo ERROR: Python 3 was not found.
echo Please install Python 3 and run this file again.
echo.
pause
exit /b 1

:check_packages
echo Using: %PYTHON_CMD%
echo Checking required Python packages:
echo   numpy, scipy, matplotlib
echo.

%PYTHON_CMD% -c "import numpy, scipy, matplotlib" >nul 2>nul
if %errorlevel%==0 goto run_script

echo One or more required packages are missing.
echo Installing them now for your user account...
echo.
%PYTHON_CMD% -m pip install --user numpy scipy matplotlib
if not %errorlevel%==0 goto install_failed

echo.
echo Verifying the installation...
%PYTHON_CMD% -c "import numpy, scipy, matplotlib"
if not %errorlevel%==0 goto install_failed

:run_script
echo.
echo All required packages are available.
echo Running section8_reproduce_all.py ...
echo.
%PYTHON_CMD% section8_reproduce_all.py
if not %errorlevel%==0 goto run_failed

echo.
echo ============================================================
echo SUCCESS
echo The current Section 8 figures and numerical output have been
echo regenerated in the Section8_reproduction folder.
echo ============================================================
echo.
pause
exit /b 0

:install_failed
echo.
echo ============================================================
echo ERROR: Automatic package installation failed.
echo ============================================================
echo Please copy the full message above and send it to me.
echo.
pause
exit /b 1

:run_failed
echo.
echo ============================================================
echo ERROR: The reproduction script returned an error.
echo ============================================================
echo Please copy the full Python error message above and send it to me.
echo.
pause
exit /b 1
