# PowerShell script to run the FastAPI application
Write-Host "Starting Sign Language Detector Web Application..." -ForegroundColor Green

# Check if .venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found! Running setup..." -ForegroundColor Red
    & ".\setup_venv.ps1"
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# Start FastAPI server
Write-Host "`nStarting server at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "----------------------------------------------`n" -ForegroundColor Cyan

uvicorn app:app --reload --host 0.0.0.0 --port 8000
