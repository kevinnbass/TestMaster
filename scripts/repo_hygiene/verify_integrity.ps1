$root = "C:\Users\kbass\OneDrive\Documents\testmaster"
$tw = Get-Content "$root\tools\codebase_monitor\tripwire.json" | ConvertFrom-Json

Write-Host "Verifying critical file integrity..."

foreach ($rel in $tw.critical_files) {
    $p = Join-Path $root $rel
    if (-not (Test-Path $p)) {
        Write-Warning "Missing critical file: $rel"
        
        $backup = Join-Path $root ("tools\codebase_monitor\backup\" + (Split-Path $rel -Leaf))
        if (Test-Path $backup) {
            Copy-Item $backup $p -Force
            Write-Host "Restored $rel from backup" -ForegroundColor Green
        } else {
            Write-Error "No backup available for $rel"
            exit 1
        }
    } else {
        Write-Host "File $rel exists" -ForegroundColor Green
    }
}

Write-Host "Critical file integrity verification complete" -ForegroundColor Cyan