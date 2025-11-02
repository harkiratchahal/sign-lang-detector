# PowerShell script to set up virtual environment
Write-Host "Setting up Sign Language Detector virtual environment..." -ForegroundColor Green

# Create virtual environment
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Skipping creation..." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`nSetup complete! Virtual environment is ready." -ForegroundColor Green
Write-Host "To activate the environment manually, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "To start the application, run: .\run_app.ps1" -ForegroundColor Yellow
