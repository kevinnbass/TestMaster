param(
    [string]$TestDir = "tests\refactor_targets",
    [string]$OutputFile = "tools\codebase_monitor\outputs\test_results.json",
    [switch]$Verbose,
    [switch]$StopOnFail,
    [string]$Pattern = "*",
    [int]$Workers = 4
)

Write-Host "Refactor Campaign Test Runner" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
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

# Install pytest if needed
Write-Host "Ensuring pytest is installed..." -ForegroundColor Yellow
python -m pip install pytest --quiet

# Check if test directory exists
if (-not (Test-Path $TestDir)) {
    Write-Error "Test directory not found: $TestDir"
    Write-Host "Please run the test scaffold generator first:"
    Write-Host "  .\scripts\refactor\generate_test_scaffolds.ps1"
    exit 1
}

# Count test files
$testFiles = Get-ChildItem -Path $TestDir -Filter "test_*.py" -Name
$totalTests = $testFiles.Count

Write-Host "Found $totalTests test files in $TestDir" -ForegroundColor Green
Write-Host ""

# Build pytest command
$pytestArgs = @()
$pytestArgs += $TestDir

# Add verbosity
if ($Verbose) {
    $pytestArgs += "-v"
    $pytestArgs += "-s"
} else {
    $pytestArgs += "-q"
}

# Add pattern matching
if ($Pattern -ne "*") {
    $pytestArgs += "-k"
    $pytestArgs += $Pattern
}

# Add stop on first failure
if ($StopOnFail) {
    $pytestArgs += "-x"
}

# Skip JSON report and parallel execution for basic pytest
# These features require additional plugins that may not be available

# Add additional options
$pytestArgs += "--tb=short"

Write-Host "Running pytest with args: $($pytestArgs -join ' ')" -ForegroundColor Yellow
Write-Host ""

# Run tests
$startTime = Get-Date
try {
    $result = & python -m pytest @pytestArgs 2>&1
    $exitCode = $LASTEXITCODE
} catch {
    Write-Error "Failed to run pytest: $_"
    exit 1
}

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

# Display results
Write-Host ""
Write-Host "Test Results" -ForegroundColor Cyan
Write-Host "============" -ForegroundColor Cyan
Write-Host "Duration: $([math]::Round($duration, 2)) seconds" -ForegroundColor Yellow

# Simple result parsing from pytest output
$passedCount = ($result | Select-String "passed" | Measure-Object).Count
$failedCount = ($result | Select-String "failed" | Measure-Object).Count  
$skippedCount = ($result | Select-String "skipped" | Measure-Object).Count

if ($result -match "(\d+) passed") {
    $passedCount = [int]$matches[1]
}
if ($result -match "(\d+) failed") {
    $failedCount = [int]$matches[1]
}
if ($result -match "(\d+) skipped") {
    $skippedCount = [int]$matches[1]
}

$totalCount = $passedCount + $failedCount + $skippedCount

Write-Host "Tests Run: $totalCount" -ForegroundColor White
Write-Host "Passed: $passedCount" -ForegroundColor Green
Write-Host "Failed: $failedCount" -ForegroundColor $(if ($failedCount -eq 0) { "Green" } else { "Red" })
Write-Host "Skipped: $skippedCount" -ForegroundColor Yellow

if ($totalCount -gt 0) {
    $successRate = ($passedCount / $totalCount) * 100
    Write-Host "Success Rate: $([math]::Round($successRate, 1))%" -ForegroundColor $(if ($successRate -ge 90) { "Green" } elseif ($successRate -ge 70) { "Yellow" } else { "Red" })
}

Write-Host ""
Write-Host "Full pytest output:" -ForegroundColor Gray
Write-Host ($result -join "`n") -ForegroundColor DarkGray

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "All tests completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Some tests failed (exit code: $exitCode)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test report saved to: $OutputFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review failing tests and fix import issues"
Write-Host "  2. Begin refactoring high-priority files"
Write-Host "  3. Run tests after each refactor to ensure no regressions"
Write-Host "  4. Use: .\scripts\refactor\measure_improvement.ps1 to track progress"

exit $exitCode