# 🎯 PHASE A: HTML BACKUP DASHBOARDS MODULE
# Mission: Collect HTML backup dashboard files with bulletproof error handling
# Created: 2025-08-23 | Modular Version

Write-Host "Phase A: Collecting HTML backup dashboards (BULLETPROOF)..." -ForegroundColor Cyan

try {
    # Create backup dashboards directory with error handling
    $backupDir = ".\frontend_final\backup_dashboards"
    New-Item -ItemType Directory -Path $backupDir -Force -ErrorAction Stop | Out-Null
    Write-Host "✓ Created backup dashboards directory" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create backup directory: $($_.Exception.Message)" -ForegroundColor Red
    throw "Directory creation failed"
}

# Collect all .backup HTML files from TestMaster with bulletproof handling
if (Test-Path ".\TestMaster") {
    Write-Host "Scanning TestMaster for backup files..." -ForegroundColor Yellow
    
    try {
        # Get all backup files with comprehensive error handling
        $backupFiles = Get-ChildItem -Path ".\TestMaster" -Recurse -Filter "*.html.backup" -ErrorAction SilentlyContinue
        
        if ($backupFiles.Count -eq 0) {
            Write-Host "⚠ No .html.backup files found in TestMaster" -ForegroundColor Yellow
        } else {
            Write-Host "Found $($backupFiles.Count) backup files to process" -ForegroundColor Cyan
            
            $currentFile = 0
            foreach ($file in $backupFiles) {
                $currentFile++
                $percent = [math]::Round(($currentFile / $backupFiles.Count) * 100, 1)
                Write-Progress -Activity "Processing backup files" -Status "$currentFile of $($backupFiles.Count) ($percent%)" -PercentComplete $percent
                
                try {
                    # BULLETPROOF PATH HANDLING: Proper escaping and Unicode support
                    $currentLocation = Get-Location
                    $testMasterPath = Join-Path $currentLocation.Path "TestMaster"
                    
                    # Calculate relative path with proper escaping
                    $relativePath = $file.FullName.Substring($testMasterPath.Length + 1)
                    $destParentDir = Split-Path $relativePath -Parent
                    
                    # Create destination directory preserving full structure
                    if ([string]::IsNullOrEmpty($destParentDir)) {
                        $finalDestDir = Join-Path $backupDir "TestMaster"
                    } else {
                        $finalDestDir = Join-Path $backupDir "TestMaster\$destParentDir"
                    }
                    
                    # Handle long paths (Windows 260 char limit)
                    if ($finalDestDir.Length -gt 240) {
                        Write-Host "  ⚠ Long path detected: $($finalDestDir.Length) chars" -ForegroundColor Yellow
                    }
                    
                    # Create directory with error handling
                    New-Item -ItemType Directory -Path $finalDestDir -Force -ErrorAction Stop | Out-Null
                    
                    # Copy file with comprehensive error handling
                    $destFile = Join-Path $finalDestDir $file.Name
                    Copy-Item -Path $file.FullName -Destination $destFile -ErrorAction Stop
                    
                    Write-Host "  ✓ Copied TestMaster\$relativePath" -ForegroundColor Green
                    $global:phaseASuccess++
                    
                } catch [System.UnauthorizedAccessException] {
                    Write-Host "  ⚠ Permission denied: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseAErrors++
                } catch [System.IO.IOException] {
                    Write-Host "  ⚠ File in use or I/O error: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseAErrors++
                } catch [System.IO.DirectoryNotFoundException] {
                    Write-Host "  ⚠ Directory not found for: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseAErrors++
                } catch [System.IO.PathTooLongException] {
                    Write-Host "  ⚠ Path too long: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseAErrors++
                } catch {
                    Write-Host "  ❌ Unexpected error with $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
                    $global:phaseAErrors++
                }
            }
            
            Write-Progress -Activity "Processing backup files" -Completed
        }
        
    } catch {
        Write-Host "❌ Failed to scan TestMaster directory: $($_.Exception.Message)" -ForegroundColor Red
        $global:phaseAErrors++
    }
    
} else {
    Write-Host "⚠ TestMaster directory not found" -ForegroundColor Yellow
}

# Final verification and reporting
try {
    $backupCount = (Get-ChildItem -Path $backupDir -Recurse -Filter "*.backup" -ErrorAction SilentlyContinue).Count
    Write-Host "✅ Phase A Complete: $global:phaseASuccess files copied successfully, $global:phaseAErrors errors" -ForegroundColor Green
    Write-Host "   Total backup files in collection: $backupCount" -ForegroundColor Cyan
} catch {
    Write-Host "⚠ Could not verify backup file count" -ForegroundColor Yellow
}