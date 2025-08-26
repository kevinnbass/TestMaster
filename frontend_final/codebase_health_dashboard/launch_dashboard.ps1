# Launch script for Codebase Health Dashboard
# This script starts both the backend API service and frontend dashboard

Write-Host "🚀 Starting Codebase Health Dashboard..." -ForegroundColor Cyan

# Check if running from correct directory
if (-not (Test-Path "package.json")) {
    Write-Error "Please run this script from the codebase_health_dashboard directory"
    exit 1
}

# Set environment variables
$env:NODE_ENV = "development"
$env:BROWSER = "none"  # Prevent automatic browser opening

# Start the dashboard
Write-Host "📊 Starting React Dashboard on http://127.0.0.1:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

# Launch the dev server
npm run dev