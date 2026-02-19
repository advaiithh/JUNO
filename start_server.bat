@echo off
echo.
echo ====================================================================
echo   JUNO - AI Voice Assistant Web Server
echo ====================================================================
echo.
echo Starting server...
echo.
echo Web UI will be available at:
echo   http://localhost:8000/ui/index.html
echo.
echo Press CTRL+C to stop the server
echo.
echo ====================================================================
echo.

REM Start the server
C:\Users\advai\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn server:app --reload --port 8000

pause
