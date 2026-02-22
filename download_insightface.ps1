# PowerShell script to download InsightFace models
# More reliable than Python's urllib

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "InsightFace Model Download (PowerShell)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

$url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
$outputFile = "buffalo_l.zip"
$extractPath = "."

# Check if already exists
if (Test-Path "buffalo_l") {
    Write-Host "buffalo_l folder already exists!" -ForegroundColor Green
    Write-Host "Checking contents..." -ForegroundColor Yellow
    
    $detModel = Join-Path "buffalo_l" "det_10g.onnx"
    $recModel = Join-Path "buffalo_l" "w600k_r50.onnx"
    
    if ((Test-Path $detModel) -and (Test-Path $recModel)) {
        Write-Host "All models present!" -ForegroundColor Green
        Write-Host "  - det_10g.onnx: $([math]::Round((Get-Item $detModel).Length/1MB, 1)) MB" -ForegroundColor White
        Write-Host "  - w600k_r50.onnx: $([math]::Round((Get-Item $recModel).Length/1MB, 1)) MB" -ForegroundColor White
        Write-Host ""
        Write-Host "Ready to use! Run: python recognition_advanced.py" -ForegroundColor Cyan
        exit 0
    }
    Write-Host "Some models missing, re-downloading..." -ForegroundColor Yellow
    Remove-Item -Path "buffalo_l" -Recurse -Force
}

Write-Host "Downloading buffalo_l models from GitHub..." -ForegroundColor Yellow
Write-Host "URL: $url" -ForegroundColor Gray
Write-Host "Size: ~90 MB" -ForegroundColor Gray
Write-Host ""

try {
    # Download with progress
    $ProgressPreference = 'Continue'
    Invoke-WebRequest -Uri $url -OutFile $outputFile -UseBasicParsing
    Write-Host "Download complete!" -ForegroundColor Green
    
    # Extract
    Write-Host ""
    Write-Host "Extracting..." -ForegroundColor Yellow
    Expand-Archive -Path $outputFile -DestinationPath $extractPath -Force
    Write-Host "Extraction complete!" -ForegroundColor Green
    
    # Clean up zip
    Remove-Item $outputFile
    Write-Host "Cleaned up zip file" -ForegroundColor Green
    
    # Verify
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Verifying installation..." -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    
    if (Test-Path "buffalo_l") {
        $detModel = Join-Path "buffalo_l" "det_10g.onnx"
        $recModel = Join-Path "buffalo_l" "w600k_r50.onnx"
        
        if ((Test-Path $detModel) -and (Test-Path $recModel)) {
            Write-Host ""
            Write-Host "SUCCESS! InsightFace models installed!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Models location: $(Resolve-Path 'buffalo_l')" -ForegroundColor White
            Write-Host "  - det_10g.onnx: $([math]::Round((Get-Item $detModel).Length/1MB, 1)) MB" -ForegroundColor White
            Write-Host "  - w600k_r50.onnx: $([math]::Round((Get-Item $recModel).Length/1MB, 1)) MB" -ForegroundColor White
            Write-Host ""
            Write-Host "=" * 70 -ForegroundColor Cyan
            Write-Host "Next Steps:" -ForegroundColor Cyan
            Write-Host "=" * 70 -ForegroundColor Cyan
            Write-Host "1. Delete old registration:" -ForegroundColor Yellow
            Write-Host "   del registered_faces_advanced.pkl" -ForegroundColor White
            Write-Host ""
            Write-Host "2. Run face recognition:" -ForegroundColor Yellow
            Write-Host "   python recognition_advanced.py" -ForegroundColor White
            Write-Host ""
            Write-Host "3. Register your face (Option 1)" -ForegroundColor Yellow
            Write-Host "   - Capture 12 samples" -ForegroundColor Gray
            Write-Host ""
            Write-Host "4. Test recognition (Option 2)" -ForegroundColor Yellow
            Write-Host "   - Uses 512-D InsightFace embeddings!" -ForegroundColor Gray
            Write-Host ""
        }
        else {
            Write-Host "ERROR: Some models are missing!" -ForegroundColor Red
            Write-Host "Expected:" -ForegroundColor Yellow
            Write-Host "  - buffalo_l/det_10g.onnx" -ForegroundColor Gray
            Write-Host "  - buffalo_l/w600k_r50.onnx" -ForegroundColor Gray
            exit 1
        }
    }
    else {
        Write-Host "ERROR: buffalo_l folder not found after extraction!" -ForegroundColor Red
        Write-Host "The zip file structure may be different than expected." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Please manually download and extract:" -ForegroundColor Yellow
        Write-Host "  $url" -ForegroundColor White
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "ERROR: Download failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Gray
    Write-Host ""
    Write-Host "Manual download instructions:" -ForegroundColor Yellow
    Write-Host "1. Open: $url" -ForegroundColor White
    Write-Host "2. Save to: $PWD" -ForegroundColor White
    Write-Host "3. Extract buffalo_l.zip here" -ForegroundColor White
    exit 1
}
