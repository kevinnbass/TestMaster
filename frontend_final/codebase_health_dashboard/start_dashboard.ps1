# Codebase Health Dashboard Startup Script
# Ensure we're in the correct directory
Set-Location $PSScriptRoot

Write-Host "🚀 Starting Codebase Health Dashboard..." -ForegroundColor Green

# Check if Node.js is installed
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js is not installed. Please install Node.js first." -ForegroundColor Red
    exit 1
}

# Check if npm is installed
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "❌ npm is not installed. Please install npm first." -ForegroundColor Red
    exit 1
}

# Install dependencies if node_modules doesn't exist
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Check if backend service is running
Write-Host "🔍 Checking if backend service is running..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8088/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend service is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Backend service returned status code: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Backend service is not running on port 8088" -ForegroundColor Red
    Write-Host "   Please start the backend service first:" -ForegroundColor Yellow
    Write-Host "   cd ../../scripts && ./scan_server.ps1" -ForegroundColor Yellow
    Write-Host "   Or run: python -m backend.codebase_monitor.service" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔄 Starting dashboard anyway (will show connection errors)..." -ForegroundColor Cyan
}

Write-Host "🌐 Starting development server on http://localhost:3000" -ForegroundColor Green
Write-Host "📊 Dashboard will be available once loaded" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

# Start the development server
npm run dev