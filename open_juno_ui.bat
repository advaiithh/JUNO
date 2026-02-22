@echo off
echo ================================================================
echo  JUNO AI Voice Assistant - Opening Web UI
echo ================================================================
echo.
echo  Opening browser to: http://localhost:8000/ui/index.html
echo.
echo  You will need to:
echo    1. Verify your face using the webcam
echo    2. Once authenticated, you can use voice assistant
echo.
echo ================================================================
echo.

start http://localhost:8000/ui/index.html

echo  Browser opened! Check your browser window.
echo.
pause
