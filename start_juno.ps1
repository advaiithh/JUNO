Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   JUNO AI Voice Assistant Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already registered
if (Test-Path "registered_faces_advanced.pkl") {
    Write-Host "OK: Face already registered" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting JUNO server..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Web UI will be available at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:8000/ui/index.html" -ForegroundColor White
    Write-Host ""
    Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    # Activate venv and start server
    & .\venv\Scripts\Activate.ps1
    python server.py
}
else {
    Write-Host "WARNING: No registered face found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please register your face first:" -ForegroundColor White
    Write-Host "  python recognition_advanced.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "After registration, run this script again." -ForegroundColor White
}
