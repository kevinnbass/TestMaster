param(
  [string]$RepoRoot = "C:\Users\kbass\OneDrive\Documents\testmaster",
  [string]$ArtifactRoot = "C:\Users\kbass\OneDrive\Documents\testmaster_artifacts",
  [switch]$WhatIf
)

Write-Host "Relocating artifacts from $RepoRoot to $ArtifactRoot"

# Create artifact root directory
if (-not $WhatIf) {
    New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
}

$manifestPath = Join-Path $RepoRoot "tools\codebase_monitor\outputs\artifact_manifest.json"
if (-not (Test-Path $manifestPath)) { 
    Write-Error "Manifest not found at $manifestPath"
    Write-Host "Run generate_ignore_from_report.py first to create the manifest"
    exit 1 
}

$manifest = Get-Content $manifestPath | ConvertFrom-Json
$patterns = $manifest.ignore_patterns

Write-Host "Processing $($patterns.Count) artifact patterns..."

foreach ($pat in $patterns) {
    # Handle directory patterns
    if ($pat -like "*/") {
        $dirPattern = $pat.TrimEnd('/')
        $srcPath = Join-Path $RepoRoot $dirPattern
        
        if (Test-Path $srcPath -PathType Container) {
            $destPath = Join-Path $ArtifactRoot $dirPattern
            
            if ($WhatIf) {
                Write-Host "[WHATIF] Would move directory: $srcPath -> $destPath"
            } else {
                Write-Host "Moving directory: $srcPath -> $destPath"
                $destParent = Split-Path $destPath -Parent
                if ($destParent) {
                    New-Item -ItemType Directory -Force -Path $destParent | Out-Null
                }
                try {
                    Move-Item -Path $srcPath -Destination $destPath -Force -ErrorAction Stop
                    Write-Host "  ✓ Moved successfully"
                } catch {
                    Write-Warning "  ✗ Failed to move: $_"
                }
            }
        }
    }
    # Handle file patterns (exact paths, not globs)
    elseif (-not $pat.Contains('*') -and -not $pat.Contains('?')) {
        $srcPath = Join-Path $RepoRoot $pat
        
        if (Test-Path $srcPath -PathType Leaf) {
            $destPath = Join-Path $ArtifactRoot $pat
            
            if ($WhatIf) {
                Write-Host "[WHATIF] Would move file: $srcPath -> $destPath"
            } else {
                Write-Host "Moving file: $srcPath -> $destPath"
                $destParent = Split-Path $destPath -Parent
                if ($destParent) {
                    New-Item -ItemType Directory -Force -Path $destParent | Out-Null
                }
                try {
                    Move-Item -Path $srcPath -Destination $destPath -Force -ErrorAction Stop
                    Write-Host "  ✓ Moved successfully"
                } catch {
                    Write-Warning "  ✗ Failed to move: $_"
                }
            }
        }
    } else {
        Write-Host "Skipping glob pattern: $pat (requires manual review)"
    }
}

if ($WhatIf) {
    Write-Host ""
    Write-Host "WhatIf mode complete. Run without -WhatIf to execute moves."
} else {
    Write-Host ""
    Write-Host "Artifact relocation complete."
    Write-Host "Artifacts moved to: $ArtifactRoot"
}