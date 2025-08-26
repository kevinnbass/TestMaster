param(
    [string]$RepoRoot = "C:\Users\kbass\OneDrive\Documents\testmaster",
    [switch]$WhatIf,
    [int]$MaxGroups = 200
)

Write-Host "Deduplication cleanup script"
Write-Host "Repository: $RepoRoot"
Write-Host "WhatIf Mode: $($WhatIf.IsPresent)"

$duplicateGroupsFile = Join-Path $RepoRoot "tools\codebase_monitor\outputs\duplicate_groups_topN.json"

if (-not (Test-Path $duplicateGroupsFile)) {
    Write-Error "Duplicate groups file not found: $duplicateGroupsFile"
    Write-Host "Generate it first using the scan report"
    exit 1
}

$groups = Get-Content $duplicateGroupsFile -Encoding UTF8 | ConvertFrom-Json
Write-Host "Processing $($groups.Count) duplicate groups..."

$totalFilesProcessed = 0
$totalFilesRemoved = 0
$safelyRemoved = 0

foreach ($group in $groups) {
    if (-not $group -or $group.Count -lt 2) { 
        continue 
    }
    
    $totalFilesProcessed += $group.Count
    Write-Host ""
    Write-Host "--- Group with $($group.Count) duplicates ---"
    
    # Sort by path to make deterministic (keep first)
    $sortedGroup = $group | Sort-Object
    $keepFile = Join-Path $RepoRoot $sortedGroup[0]
    
    Write-Host "KEEP: $($sortedGroup[0])"
    
    for ($i = 1; $i -lt $sortedGroup.Count; $i++) {
        $candidate = Join-Path $RepoRoot $sortedGroup[$i]
        $relPath = $sortedGroup[$i]
        
        # Safety checks - only remove from known artifact/temporary directories
        $isSafeToRemove = $false
        $safePatterns = @(
            "*\telemetry\*",
            "*\logs\*", 
            "*\PRODUCTION_PACKAGES\*",
            "*\organized_codebase\monitoring\*",
            "*\backup*",
            "*\archive\*",
            "*\.venv\*",
            "*\__pycache__\*",
            "*\node_modules\*",
            "*\dist\*",
            "*\build\*",
            "*\.pytest_cache\*",
            "*\coverage\*",
            "*.tmp",
            "*.bak",
            "*.log"
        )
        
        foreach ($pattern in $safePatterns) {
            if ($relPath -like $pattern) {
                $isSafeToRemove = $true
                break
            }
        }
        
        if ($isSafeToRemove) {
            if (Test-Path $candidate) {
                if ($WhatIf) {
                    Write-Host "  [WHATIF] REMOVE: $relPath (safe pattern match)" -ForegroundColor Yellow
                    $safelyRemoved++
                } else {
                    try {
                        Remove-Item $candidate -Force -ErrorAction Stop
                        Write-Host "  ✓ REMOVED: $relPath" -ForegroundColor Green
                        $safelyRemoved++
                        $totalFilesRemoved++
                    } catch {
                        Write-Host "  ✗ FAILED to remove: $relPath - $_" -ForegroundColor Red
                    }
                }
            } else {
                Write-Host "  ! NOT FOUND: $relPath" -ForegroundColor Gray
            }
        } else {
            # Files in source code areas - require manual review
            Write-Host "  MANUAL: $relPath (requires manual review)" -ForegroundColor Cyan
        }
    }
}

Write-Host ""
Write-Host "=== DEDUPLICATION SUMMARY ==="
Write-Host "Total files in groups: $totalFilesProcessed"
Write-Host "Files safely removed: $safelyRemoved"
if (-not $WhatIf) {
    Write-Host "Actual files deleted: $totalFilesRemoved"
}
Write-Host "Files requiring manual review: $($totalFilesProcessed - $safelyRemoved - $groups.Count)"

if ($WhatIf) {
    Write-Host ""
    Write-Host "This was a dry run. Use without -WhatIf to execute removals."
}