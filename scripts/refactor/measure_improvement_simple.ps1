param(
    [string]$BaselineFile = "tools\codebase_monitor\outputs\refactor_baseline.json",
    [switch]$SetBaseline
)

Write-Host "Refactor Campaign Improvement Measurement" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Run analyzer and update latest folder
Write-Host "Running analyzer..." -ForegroundColor Yellow
python "tools\codebase_monitor\analyzer.py" --root . --output-dir "tools\codebase_monitor\reports" | Out-Null

$latestScan = Get-ChildItem "tools\codebase_monitor\reports\scan_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $latestScan.FullName "tools\codebase_monitor\reports\latest\scan.json" -Force

# Load current scan
$currentScan = Get-Content "tools\codebase_monitor\reports\latest\scan.json" | ConvertFrom-Json

# Current metrics
$currentMetrics = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    total_files = $currentScan.total_files
    total_lines = $currentScan.total_code_lines
    duplicate_groups = $currentScan.duplicates.Count
    duplicate_files = ($currentScan.duplicates | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
}

if ($SetBaseline) {
    Write-Host "Setting baseline..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Split-Path $BaselineFile) | Out-Null
    $currentMetrics | ConvertTo-Json | Out-File $BaselineFile -Encoding utf8 -Force
    
    Write-Host "Baseline set:" -ForegroundColor Green
    Write-Host "  Files: $($currentMetrics.total_files)" -ForegroundColor Gray
    Write-Host "  Lines: $($currentMetrics.total_lines)" -ForegroundColor Gray
    Write-Host "  Duplicate groups: $($currentMetrics.duplicate_groups)" -ForegroundColor Gray
    Write-Host "  Files in duplicates: $($currentMetrics.duplicate_files)" -ForegroundColor Gray
    exit 0
}

# Load baseline
if (-not (Test-Path $BaselineFile)) {
    Write-Host "No baseline found. Please run with -SetBaseline first." -ForegroundColor Red
    exit 1
}

$baseline = Get-Content $BaselineFile | ConvertFrom-Json

# Calculate changes
Write-Host ""
Write-Host "Comparison vs Baseline ($($baseline.timestamp))" -ForegroundColor Cyan
Write-Host ""

$fileDelta = $currentMetrics.total_files - $baseline.total_files
$lineDelta = $currentMetrics.total_lines - $baseline.total_lines  
$dupeDelta = $currentMetrics.duplicate_groups - $baseline.duplicate_groups
$dupeFilesDelta = $currentMetrics.duplicate_files - $baseline.duplicate_files

Write-Host "Files: $($baseline.total_files) -> $($currentMetrics.total_files) ($fileDelta)" -ForegroundColor $(if ($fileDelta -le 0) { "Green" } else { "Red" })
Write-Host "Lines: $($baseline.total_lines) -> $($currentMetrics.total_lines) ($lineDelta)" -ForegroundColor $(if ($lineDelta -le 0) { "Green" } else { "Red" })
Write-Host "Duplicate Groups: $($baseline.duplicate_groups) -> $($currentMetrics.duplicate_groups) ($dupeDelta)" -ForegroundColor $(if ($dupeDelta -le 0) { "Green" } else { "Red" })
Write-Host "Duplicate Files: $($baseline.duplicate_files) -> $($currentMetrics.duplicate_files) ($dupeFilesDelta)" -ForegroundColor $(if ($dupeFilesDelta -le 0) { "Green" } else { "Red" })

Write-Host ""
if ($dupeDelta -lt 0 -and $dupeFilesDelta -lt 0) {
    Write-Host "Great progress! Duplicates reduced." -ForegroundColor Green
} elseif ($dupeDelta -eq 0 -and $dupeFilesDelta -eq 0) {
    Write-Host "No change in duplicates. Continue refactoring." -ForegroundColor Yellow  
} else {
    Write-Host "Duplicates increased. Review recent changes." -ForegroundColor Red
}

Write-Host ""
Write-Host "Next: Continue refactoring high-priority files" -ForegroundColor Cyan