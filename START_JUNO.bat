@echo off
title JUNO - Starting...
color 0B

echo ====================================================
echo           JUNO AI Voice Assistant
echo ====================================================
echo.
echo [1/3] Checking server status...

:: Check if server is already running
curl -s http://localhost:8000/status >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Server already running
    goto :open_ui
)

echo [2/3] Starting JUNO server...
echo.
echo Opening server in new window...
start "JUNO Server" cmd /k "cd /d %~dp0 && python server.py"

echo [3/3] Waiting for server initialization...
timeout /t 5 /nobreak >nul

:open_ui
echo.
echo ====================================================
echo [OK] Opening JUNO in your browser...
echo ====================================================
echo.
echo Your browser will open with face authentication.
echo.
echo What happens next:
echo   1. Camera starts automatically
echo   2. Face verification (every 2 seconds)
echo   3. Once recognized, you're logged in!
echo.
echo ====================================================
echo.

:: Open authentication page
start http://localhost:8000/ui/auth.html

echo.
echo [OK] JUNO is ready!
echo.
echo Server window: Keep it running in background
echo Browser: Complete face authentication
echo.
echo Press any key to close this window...
pause >nul
