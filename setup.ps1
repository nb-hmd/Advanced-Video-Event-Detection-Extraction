# Video Event Detection Setup Script for Windows
# Run this script in PowerShell as Administrator

Write-Host "🎥 Video Event Detection Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check if Python version is 3.8+
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $majorVersion = [int]$matches[1]
    $minorVersion = [int]$matches[2]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
        Write-Host "❌ Python 3.8+ required. Current version: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}

# Check FFmpeg installation
Write-Host "Checking FFmpeg installation..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
    Write-Host "✅ Found: $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  FFmpeg not found. Installing via chocolatey..." -ForegroundColor Yellow
    
    # Check if chocolatey is installed
    try {
        choco --version | Out-Null
        choco install ffmpeg -y
        Write-Host "✅ FFmpeg installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Chocolatey not found. Please install FFmpeg manually from https://ffmpeg.org" -ForegroundColor Red
        Write-Host "   Or install Chocolatey first: https://chocolatey.org/install" -ForegroundColor Yellow
    }
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Create data directories
Write-Host "Creating data directories..." -ForegroundColor Yellow
$directories = @("data\videos", "data\frames", "data\clips", "data\embeddings", "models\openclip", "models\blip", "models\univtg")

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    }
}

# Create .env file if it doesn't exist
Write-Host "Setting up configuration..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    $envContent = @"
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Processing Settings
BATCH_SIZE=16
CONFIDENCE_THRESHOLD=0.3
TOP_K_RESULTS=10

# Model Settings
OPENCLIP_MODEL=ViT-B-32
BLIP_MODEL=Salesforce/blip2-opt-2.7b
"@
    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ Created .env configuration file" -ForegroundColor Green
}

# Test installation
Write-Host "Testing installation..." -ForegroundColor Yellow
try {
    python -c "from src.utils.config import settings; print('✅ Configuration loaded successfully')"
    python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
    python -c "import open_clip; print('✅ OpenCLIP imported successfully')"
    python -c "import transformers; print(f'✅ Transformers version: {transformers.__version__}')"
    Write-Host "✅ All core components working" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Some components may not be working correctly" -ForegroundColor Yellow
}

# Setup complete
Write-Host "" 
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "================" -ForegroundColor Green
Write-Host ""
Write-Host "To start using the system:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start the web interface:" -ForegroundColor White
Write-Host "   streamlit run src\web\streamlit_app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or start the API server:" -ForegroundColor White
Write-Host "   python -m src.api.main" -ForegroundColor Gray
Write-Host ""
Write-Host "📖 For more information, see README.md" -ForegroundColor Cyan
Write-Host ""