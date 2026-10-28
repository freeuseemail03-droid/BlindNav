@echo off
setlocal
title BlindNav AI - Web Server

rem ---- Project location -------------------------------------------------
rem This launcher runs the app from D:\BlindNavigator 2.0.
rem If that folder is missing (or has no server.py), fall back to the
rem folder this script lives in, so a copy elsewhere still works.
set "PROJECT_DIR=D:\BlindNavigator 2.0"
if not exist "%PROJECT_DIR%\server.py" set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

rem ---- Python interpreter selection ------------------------------------
rem Libraries live on D: so the project folder stays small.
rem Order of preference:
rem   1) Shared libs  D:\program\pylibs   (PYTHONPATH, Python on PATH)
rem   2) Project .venv                    (fallback, if you copied the folder)
set "PYTHON="
set "PYLIBS=D:\program\pylibs"
if exist "%PYLIBS%\numpy\__init__.py" (
    set "PYTHON=python"
)
rem pywin32 relies on a .pth file, which is ignored when the libraries are
rem installed with "pip install --target". Without these extra directories
rem `import pywintypes` fails and the app loses its spoken voice.
if exist "%PYLIBS%\win32\lib" (
    set "PYTHONPATH=%PYLIBS%;%PYLIBS%\win32;%PYLIBS%\win32\lib"
) else (
    set "PYTHONPATH=%PYLIBS%"
)
if not defined PYTHON if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

if not defined PYTHON (
    echo [ERROR] No Python found.
    echo Install shared libraries to %PYLIBS% :
    echo    python -m pip install --target "%PYLIBS%" -r requirements.txt
    echo ...or create a project .venv:
    echo    python -m venv .venv
    echo    .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "server.py" (
    echo [ERROR] server.py not found in:
    echo    %PROJECT_DIR%
    echo Copy the project there, or edit PROJECT_DIR in this script.
    echo.
    pause
    exit /b 1
)

rem ---- Make sure the web port is usable ---------------------------------
rem A server that was closed with the X button can stay alive in the
rem background and keep port 8000 busy, which makes every new launch fail
rem with "winerror 10048 only one usage of each socket address". Close such
rem a leftover instance, or move to the next free port.
rem Set PORT before running this script to force a specific port.
if not defined PORT set "PORT=8000"
set "PORTMAX=8020"

:checkport
set "BUSYPID="
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr /c:LISTENING ^| findstr /c:":%PORT% "') do set "BUSYPID=%%P"
if not defined BUSYPID goto :portfree

set "BUSYNAME="
for /f "tokens=1 delims=," %%N in ('tasklist /fi "pid eq %BUSYPID%" /nh /fo csv 2^>nul') do set "BUSYNAME=%%~N"

rem A leftover BlindNav server is a python process on our own port: safe to close.
if /i "%BUSYNAME%"=="python.exe" goto :reclaim
if /i "%BUSYNAME%"=="python3.13.exe" goto :reclaim
if /i "%BUSYNAME%"=="pythonw.exe" goto :reclaim

echo [WARN] Port %PORT% is already used by "%BUSYNAME%" (PID %BUSYPID%).
set /a PORT=%PORT%+1
if %PORT% GTR %PORTMAX% goto :noport
echo [WARN] Trying port %PORT% instead...
goto :checkport

:reclaim
echo [INFO] Closing leftover BlindNav server on port %PORT% (PID %BUSYPID%)...
taskkill /f /pid %BUSYPID% >nul 2>&1
rem Give Windows a moment to release the socket before re-checking.
ping -n 3 127.0.0.1 >nul
goto :checkport

:noport
echo [ERROR] No free port between 8000 and %PORTMAX%.
echo Close whatever is using them, then run this script again.
echo.
pause
exit /b 1

:portfree

echo ==================================================
echo   BlindNav AI - AI-Powered Blind Navigation
echo --------------------------------------------------
echo   Project   : %PROJECT_DIR%
echo   URL       : http://127.0.0.1:%PORT%
echo   Python    : %PYTHON%
echo   Camera    : make sure no other app uses it
echo   Stop      : close this window or press Ctrl+C
echo ==================================================
echo.

rem Open the browser automatically once the server has loaded (~5 s).
rem "ping" is used instead of "timeout" because timeout refuses to run when
rem the window's input is redirected (it would fail with "Input redirection
rem is not supported" and the browser would never open).
start "" cmd /c "ping -n 6 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"

"%PYTHON%" server.py --port %PORT%

echo.
echo Server stopped.
pause
goto :eof
