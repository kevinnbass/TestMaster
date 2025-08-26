param(
    [int]$MaxBytes = 5242880,  # 5MB default
    [switch]$CheckAll,
    [string]$Path = "."
)

Write-Host "File size enforcement script"
Write-Host "Maximum file size: $($MaxBytes / 1MB) MB"

$oversizedFiles = @()
$totalChecked = 0

if ($CheckAll) {
    # Check all files in repository
    Write-Host "Checking all files in repository..."
    $files = Get-ChildItem -Path $Path -Recurse -File | Where-Object {
        # Skip common directories that should contain large files
        $_.FullName -notlike "*\.git\*" -and
        $_.FullName -notlike "*\node_modules\*" -and
        $_.FullName -notlike "*\.venv\*" -and
        $_.FullName -notlike "*\__pycache__\*" -and
        $_.FullName -notlike "*\dist\*" -and
        $_.FullName -notlike "*\build\*"
    }
    
    foreach ($file in $files) {
        $totalChecked++
        if ($file.Length -gt $MaxBytes) {
            $oversizedFiles += [PSCustomObject]@{
                Path = $file.FullName
                Size = $file.Length
                SizeMB = [math]::Round($file.Length / 1MB, 2)
            }
        }
    }
} else {
    # Check only staged files (git pre-commit hook mode)
    Write-Host "Checking staged files for commit..."
    
    try {
        $stagedFiles = git diff --cached --name-only --diff-filter=ACMR 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Not in a git repository or no staged files"
            exit 0
        }
        
        foreach ($relPath in $stagedFiles) {
            if ([string]::IsNullOrWhiteSpace($relPath)) { continue }
            
            $fullPath = Join-Path $Path $relPath
            if (Test-Path $fullPath -PathType Leaf) {
                $totalChecked++
                $fileSize = (Get-Item $fullPath).Length
                
                if ($fileSize -gt $MaxBytes) {
                    $oversizedFiles += [PSCustomObject]@{
                        Path = $relPath
                        Size = $fileSize
                        SizeMB = [math]::Round($fileSize / 1MB, 2)
                    }
                }
            }
        }
    } catch {
        Write-Warning "Error checking staged files: $_"
        exit 1
    }
}

Write-Host "Files checked: $totalChecked"

if ($oversizedFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  OVERSIZED FILES DETECTED:" -ForegroundColor Red
    Write-Host ""
    
    foreach ($file in $oversizedFiles) {
        Write-Host "  $($file.Path)" -ForegroundColor Red
        Write-Host "    Size: $($file.SizeMB) MB ($($file.Size) bytes)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Files exceeding $($MaxBytes / 1MB) MB limit: $($oversizedFiles.Count)" -ForegroundColor Red
    
    if (-not $CheckAll) {
        Write-Host ""
        Write-Host "COMMIT BLOCKED - Please:" -ForegroundColor Red
        Write-Host "1. Remove or compress large files"
        Write-Host "2. Add large files to .gitignore if they're artifacts"
        Write-Host "3. Use Git LFS for legitimate large files"
        Write-Host "4. Split large source files into smaller modules"
        exit 2
    }
} else {
    Write-Host "✓ All files are within size limits" -ForegroundColor Green
    exit 0
}