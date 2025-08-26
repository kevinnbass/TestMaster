param(
    [string]$BaselineFile = "tools\codebase_monitor\outputs\refactor_baseline.json",
    [switch]$SetBaseline,
    [switch]$Verbose,
    [string]$ReportFile = "tools\codebase_monitor\outputs\improvement_report.json"
)

Write-Host "Refactor Campaign Improvement Measurement" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the project root
if (-not (Test-Path "CLAUDE.md")) {
    Write-Error "Please run this script from the project root directory"
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Run fresh analyzer scan
Write-Host "Running codebase analyzer..." -ForegroundColor Yellow
try {
    $analyzerOutput = & python "tools\codebase_monitor\analyzer.py" --root . --output-dir "tools\codebase_monitor\reports" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Analyzer failed with exit code $LASTEXITCODE"
    }
    Write-Host "Analyzer completed successfully" -ForegroundColor Green
} catch {
    Write-Error "Failed to run analyzer: $_"
    exit 1
}

# Update latest folder
Write-Host "Updating latest reports..." -ForegroundColor Yellow
$latestScan = Get-ChildItem "tools\codebase_monitor\reports\scan_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$latestSummary = $latestScan.FullName -replace 'scan_', 'summary_' -replace '.json', '.md'

Copy-Item $latestScan.FullName "tools\codebase_monitor\reports\latest\scan.json" -Force
if (Test-Path $latestSummary) {
    Copy-Item $latestSummary "tools\codebase_monitor\reports\latest\summary.md" -Force
}

# Load current scan results
$currentScan = Get-Content "tools\codebase_monitor\reports\latest\scan.json" | ConvertFrom-Json

# Extract current metrics
$currentMetrics = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    scan_file = $latestScan.Name
    total_files = $currentScan.total_files
    total_size_bytes = $currentScan.total_size_bytes
    total_code_lines = $currentScan.total_code_lines
    duplicate_groups = $currentScan.duplicates.Count
    duplicate_files = ($currentScan.duplicates | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
    hotspots = @{}
    top_issues = @()
}

# Extract hotspot counts
foreach ($hotspotType in $currentScan.hotspots.PSObject.Properties.Name) {
    $currentMetrics.hotspots[$hotspotType] = $currentScan.hotspots.$hotspotType.Count
}

# Find top issues (files appearing in multiple hotspot categories)
$fileHotspotCount = @{}
foreach ($hotspotType in $currentScan.hotspots.PSObject.Properties.Name) {
    foreach ($file in $currentScan.hotspots.$hotspotType) {
        if (-not $fileHotspotCount.ContainsKey($file)) {
            $fileHotspotCount[$file] = 0
        }
        $fileHotspotCount[$file]++
    }
}

$currentMetrics.top_issues = $fileHotspotCount.GetEnumerator() | 
    Sort-Object Value -Descending | 
    Select-Object -First 20 | 
    ForEach-Object { @{ file = $_.Key; hotspot_count = $_.Value } }

if ($SetBaseline) {
    Write-Host ""
    Write-Host "Setting new baseline..." -ForegroundColor Yellow
    
    # Ensure output directory exists
    New-Item -ItemType Directory -Force -Path (Split-Path $BaselineFile) | Out-Null
    
    # Save current metrics as baseline
    $currentMetrics | ConvertTo-Json -Depth 10 | Out-File $BaselineFile -Encoding utf8 -Force
    
    Write-Host "Baseline set successfully!" -ForegroundColor Green
    Write-Host "Baseline file: $BaselineFile" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Current metrics:" -ForegroundColor White
    Write-Host "  Total files: $($currentMetrics.total_files)" -ForegroundColor Gray
    Write-Host "  Total lines: $($currentMetrics.total_code_lines)" -ForegroundColor Gray
    Write-Host "  Duplicate groups: $($currentMetrics.duplicate_groups)" -ForegroundColor Gray
    Write-Host "  Files in duplicates: $($currentMetrics.duplicate_files)" -ForegroundColor Gray
    Write-Host "  Hotspot categories: $($currentMetrics.hotspots.Count)" -ForegroundColor Gray
    
    exit 0
}

# Load baseline if it exists
if (-not (Test-Path $BaselineFile)) {
    Write-Warning "No baseline found at: $BaselineFile"
    Write-Host "Please set a baseline first with: -SetBaseline" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Current metrics (no comparison available):" -ForegroundColor White
    Write-Host "  Total files: $($currentMetrics.total_files)" -ForegroundColor Gray
    Write-Host "  Total lines: $($currentMetrics.total_code_lines)" -ForegroundColor Gray
    Write-Host "  Duplicate groups: $($currentMetrics.duplicate_groups)" -ForegroundColor Gray
    Write-Host "  Files in duplicates: $($currentMetrics.duplicate_files)" -ForegroundColor Gray
    exit 1
}

$baseline = Get-Content $BaselineFile | ConvertFrom-Json

# Calculate improvements
Write-Host ""
Write-Host "Improvement Analysis" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

$improvements = @{
    timestamp = $currentMetrics.timestamp
    baseline_timestamp = $baseline.timestamp
    changes = @{}
    summary = @{}
}

# File count changes
$fileDelta = $currentMetrics.total_files - $baseline.total_files
$improvements.changes.total_files = @{
    baseline = $baseline.total_files
    current = $currentMetrics.total_files
    delta = $fileDelta
    percent_change = if ($baseline.total_files -gt 0) { [math]::Round(($fileDelta / $baseline.total_files) * 100, 2) } else { 0 }
}

# Line count changes
$lineDelta = $currentMetrics.total_code_lines - $baseline.total_code_lines
$improvements.changes.total_lines = @{
    baseline = $baseline.total_code_lines
    current = $currentMetrics.total_code_lines
    delta = $lineDelta
    percent_change = if ($baseline.total_code_lines -gt 0) { [math]::Round(($lineDelta / $baseline.total_code_lines) * 100, 2) } else { 0 }
}

# Duplicate improvements
$dupeDelta = $currentMetrics.duplicate_groups - $baseline.duplicate_groups
$improvements.changes.duplicate_groups = @{
    baseline = $baseline.duplicate_groups
    current = $currentMetrics.duplicate_groups
    delta = $dupeDelta
    percent_change = if ($baseline.duplicate_groups -gt 0) { [math]::Round(($dupeDelta / $baseline.duplicate_groups) * 100, 2) } else { 0 }
}

$dupeFilesDelta = $currentMetrics.duplicate_files - $baseline.duplicate_files
$improvements.changes.duplicate_files = @{
    baseline = $baseline.duplicate_files
    current = $currentMetrics.duplicate_files
    delta = $dupeFilesDelta
    percent_change = if ($baseline.duplicate_files -gt 0) { [math]::Round(($dupeFilesDelta / $baseline.duplicate_files) * 100, 2) } else { 0 }
}

# Hotspot changes
$improvements.changes.hotspots = @{}
foreach ($hotspotType in @($baseline.hotspots.PSObject.Properties.Name + $currentMetrics.hotspots.PSObject.Properties.Name) | Sort-Object -Unique) {
    $baselineCount = if ($baseline.hotspots.$hotspotType) { $baseline.hotspots.$hotspotType } else { 0 }
    $currentCount = if ($currentMetrics.hotspots.$hotspotType) { $currentMetrics.hotspots.$hotspotType } else { 0 }
    $delta = $currentCount - $baselineCount
    
    $improvements.changes.hotspots[$hotspotType] = @{
        baseline = $baselineCount
        current = $currentCount
        delta = $delta
        percent_change = if ($baselineCount -gt 0) { [math]::Round(($delta / $baselineCount) * 100, 2) } else { if ($currentCount -gt 0) { 100 } else { 0 } }
    }
}

# Display results
Write-Host ""
Write-Host "📊 Overall Metrics" -ForegroundColor White
Write-Host "Total Files: $($baseline.total_files) → $($currentMetrics.total_files) ($($improvements.changes.total_files.delta >= 0 ? '+' : '')$($improvements.changes.total_files.delta), $($improvements.changes.total_files.percent_change)%)" -ForegroundColor $(if ($improvements.changes.total_files.delta -le 0) { "Green" } else { "Red" })
Write-Host "Total Lines: $($baseline.total_code_lines) → $($currentMetrics.total_code_lines) ($($improvements.changes.total_lines.delta >= 0 ? '+' : '')$($improvements.changes.total_lines.delta), $($improvements.changes.total_lines.percent_change)%)" -ForegroundColor $(if ($improvements.changes.total_lines.delta -le 0) { "Green" } else { "Red" })

Write-Host ""
Write-Host "🔄 Duplication Improvements" -ForegroundColor White
Write-Host "Duplicate Groups: $($baseline.duplicate_groups) → $($currentMetrics.duplicate_groups) ($($improvements.changes.duplicate_groups.delta >= 0 ? '+' : '')$($improvements.changes.duplicate_groups.delta), $($improvements.changes.duplicate_groups.percent_change)%)" -ForegroundColor $(if ($improvements.changes.duplicate_groups.delta -le 0) { "Green" } else { "Red" })
Write-Host "Files in Duplicates: $($baseline.duplicate_files) → $($currentMetrics.duplicate_files) ($($improvements.changes.duplicate_files.delta >= 0 ? '+' : '')$($improvements.changes.duplicate_files.delta), $($improvements.changes.duplicate_files.percent_change)%)" -ForegroundColor $(if ($improvements.changes.duplicate_files.delta -le 0) { "Green" } else { "Red" })

Write-Host ""
Write-Host "🔥 Hotspot Analysis" -ForegroundColor White
foreach ($hotspotType in ($improvements.changes.hotspots.Keys | Sort-Object)) {
    $change = $improvements.changes.hotspots[$hotspotType]
    $color = if ($change.delta -le 0) { "Green" } else { "Red" }
    Write-Host "$hotspotType : $($change.baseline) → $($change.current) ($($change.delta >= 0 ? '+' : '')$($change.delta), $($change.percent_change)%)" -ForegroundColor $color
}

# Generate summary
$totalHotspotsBaseline = ($baseline.hotspots.PSObject.Properties.Value | Measure-Object -Sum).Sum
$totalHotspotsCurrent = ($currentMetrics.hotspots.PSObject.Properties.Value | Measure-Object -Sum).Sum
$hotspotsDelta = $totalHotspotsCurrent - $totalHotspotsBaseline

$improvements.summary = @{
    overall_score = @{
        baseline = $totalHotspotsBaseline + $baseline.duplicate_groups * 2
        current = $totalHotspotsCurrent + $currentMetrics.duplicate_groups * 2
        improvement_percent = if ($totalHotspotsBaseline -gt 0) { [math]::Round((($totalHotspotsBaseline - $totalHotspotsCurrent) / $totalHotspotsBaseline) * 100, 2) } else { 0 }
    }
    recommendation = if ($hotspotsDelta -lt 0 -and $dupeDelta -lt 0) { "Great progress! Continue refactoring." } 
                     elseif ($hotspotsDelta -eq 0 -and $dupeDelta -eq 0) { "No change detected. Focus on high-impact files." }
                     else { "Issues increased. Review recent changes." }
}

Write-Host ""
Write-Host "📈 Overall Assessment" -ForegroundColor Cyan
Write-Host "Problem Score: $($improvements.summary.overall_score.baseline) → $($improvements.summary.overall_score.current) ($(if ($improvements.summary.overall_score.improvement_percent -ge 0) { $improvements.summary.overall_score.improvement_percent }%)% improvement)" -ForegroundColor $(if ($improvements.summary.overall_score.improvement_percent -ge 0) { "Green" } else { "Red" })
Write-Host "Recommendation: $($improvements.summary.recommendation)" -ForegroundColor Yellow

# Save detailed report
$improvements.baseline_metrics = $baseline
$improvements.current_metrics = $currentMetrics

New-Item -ItemType Directory -Force -Path (Split-Path $ReportFile) | Out-Null
$improvements | ConvertTo-Json -Depth 10 | Out-File $ReportFile -Encoding utf8 -Force

Write-Host ""
Write-Host "📄 Reports Generated" -ForegroundColor Cyan
Write-Host "Detailed report: $ReportFile" -ForegroundColor Gray
Write-Host "Latest scan: tools\codebase_monitor\reports\latest\scan.json" -ForegroundColor Gray

Write-Host ""
Write-Host "🎯 Next Actions" -ForegroundColor Yellow
if ($improvements.summary.overall_score.improvement_percent -ge 10) {
    Write-Host "  ✅ Excellent progress! Consider setting a new baseline."
} elseif ($improvements.summary.overall_score.improvement_percent -ge 0) {
    Write-Host "  📈 Good progress. Continue with current refactoring approach."
} else {
    Write-Host "  ⚠️  Regression detected. Review recent changes and revert if needed."
}

Write-Host "  📋 Focus on files with highest hotspot counts"
Write-Host "  🔄 Run tests after each refactor: .\scripts\refactor\run_tests.ps1"
Write-Host "  📊 Re-run this script after significant changes"