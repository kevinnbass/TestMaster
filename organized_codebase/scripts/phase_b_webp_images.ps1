# 🖼️ PHASE B: WEBP IMAGE FILES MODULE
# Mission: Collect WebP image files with bulletproof path preservation and deduplication
# Created: 2025-08-23 | Modular Version

Write-Host "Phase B: Collecting WebP image files (BULLETPROOF)..." -ForegroundColor Cyan

$processedHashes = @{} # Deduplication tracking

try {
    # Create assets directory with error handling
    $assetsDir = ".\frontend_final\additional_assets"
    New-Item -ItemType Directory -Path $assetsDir -Force -ErrorAction Stop | Out-Null
    Write-Host "✓ Created additional assets directory" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create assets directory: $($_.Exception.Message)" -ForegroundColor Red
    throw "Directory creation failed"
}

# Define source roots to search
$sourceRoots = @("agentops", "autogen", "agent-squad", "AgentVerse", "TestMaster")

foreach ($sourceRoot in $sourceRoots) {
    if (Test-Path ".\$sourceRoot") {
        Write-Host "Scanning $sourceRoot for WebP files..." -ForegroundColor Yellow
        
        try {
            $currentPath = Get-Location
            $rootPath = Join-Path $currentPath.Path $sourceRoot
            
            # Get all WebP files excluding already collected ones
            $webpFiles = Get-ChildItem -Path ".\$sourceRoot" -Recurse -Filter "*.webp" -ErrorAction SilentlyContinue | Where-Object {
                $_.FullName -notlike "*frontend_final*" -and 
                $_.FullName -notlike "*frontend_ui_only*" -and
                $_.FullName -notlike "*node_modules*"
            }
            
            Write-Host "Found $($webpFiles.Count) WebP files in $sourceRoot" -ForegroundColor Cyan
            $currentFile = 0
            
            foreach ($file in $webpFiles) {
                $currentFile++
                $percent = [math]::Round(($currentFile / $webpFiles.Count) * 100, 1)
                Write-Progress -Activity "Processing $sourceRoot WebP files" -Status "$currentFile of $($webpFiles.Count) ($percent%)" -PercentComplete $percent
                
                try {
                    # DEDUPLICATION: Check file hash to prevent duplicates
                    $fileHash = (Get-FileHash $file.FullName -Algorithm MD5 -ErrorAction Stop).Hash
                    if ($processedHashes.ContainsKey($fileHash)) {
                        # Skip duplicate file
                        continue
                    }
                    $processedHashes[$fileHash] = "$sourceRoot\$($file.Name)"
                    
                    # BULLETPROOF PATH CALCULATION: Handle relative paths properly
                    if ($file.FullName.StartsWith($rootPath)) {
                        $relativePath = $file.FullName.Substring($rootPath.Length + 1)
                        $destParentDir = Split-Path $relativePath -Parent
                        
                        if ([string]::IsNullOrEmpty($destParentDir)) {
                            $finalDestDir = Join-Path $assetsDir $sourceRoot
                        } else {
                            $finalDestDir = Join-Path $assetsDir "$sourceRoot\$destParentDir"
                        }
                        
                        # Handle very deep hierarchies
                        if ($finalDestDir.Length -gt 240) {
                            Write-Host "    ⚠ Very long path: $($finalDestDir.Length) chars" -ForegroundColor Yellow
                        }
                        
                        # Create destination directory structure
                        New-Item -ItemType Directory -Path $finalDestDir -Force -ErrorAction Stop | Out-Null
                        
                        # Copy WebP file
                        $destFile = Join-Path $finalDestDir $file.Name
                        Copy-Item -Path $file.FullName -Destination $destFile -ErrorAction Stop
                        
                        Write-Host "    ✓ Copied $sourceRoot\$relativePath" -ForegroundColor Green
                        $global:phaseBSuccess++
                    }
                    
                } catch [System.UnauthorizedAccessException] {
                    Write-Host "    ⚠ Permission denied: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseBErrors++
                } catch [System.IO.IOException] {
                    Write-Host "    ⚠ File in use: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseBErrors++
                } catch [System.IO.PathTooLongException] {
                    Write-Host "    ⚠ Path too long: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseBErrors++
                } catch {
                    Write-Host "    ❌ Error with $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
                    $global:phaseBErrors++
                }
            }
            
            Write-Progress -Activity "Processing $sourceRoot WebP files" -Completed
            
        } catch {
            Write-Host "❌ Failed to process $sourceRoot: $($_.Exception.Message)" -ForegroundColor Red
            $global:phaseBErrors++
        }
        
    } else {
        Write-Host "$sourceRoot directory not found, skipping" -ForegroundColor Yellow
    }
}

# Final verification and reporting
try {
    $webpCount = (Get-ChildItem -Path $assetsDir -Recurse -Filter "*.webp" -ErrorAction SilentlyContinue).Count
    Write-Host "✅ Phase B Complete: $global:phaseBSuccess files copied successfully, $global:phaseBErrors errors" -ForegroundColor Green
    Write-Host "   Total WebP files in collection: $webpCount" -ForegroundColor Cyan
    Write-Host "   Duplicates prevented: $($processedHashes.Count - $global:phaseBSuccess)" -ForegroundColor Cyan
} catch {
    Write-Host "⚠ Could not verify WebP file count" -ForegroundColor Yellow
}