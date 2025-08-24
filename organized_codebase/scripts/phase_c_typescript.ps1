# 📋 PHASE C: TYPESCRIPT DEFINITIONS MODULE
# Mission: Collect TypeScript definition files with bulletproof deduplication and path preservation
# Created: 2025-08-23 | Modular Version

Write-Host "Phase C: Collecting TypeScript definition files (BULLETPROOF)..." -ForegroundColor Cyan

$processedDtsHashes = @{} # Deduplication for .d.ts files

try {
    # Create TypeScript definitions directory
    $tsDir = ".\frontend_final\typescript_definitions"
    New-Item -ItemType Directory -Path $tsDir -Force -ErrorAction Stop | Out-Null
    Write-Host "✓ Created TypeScript definitions directory" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create TypeScript directory: $($_.Exception.Message)" -ForegroundColor Red
    throw "Directory creation failed"
}

# Define source roots prioritizing AgentVerse (main source of .d.ts files)
$sourceRoots = @("AgentVerse", "agent-squad", "autogen", "agentops")

foreach ($sourceRoot in $sourceRoots) {
    if (Test-Path ".\$sourceRoot") {
        Write-Host "Scanning $sourceRoot for TypeScript definition files..." -ForegroundColor Yellow
        
        try {
            $currentPath = Get-Location
            $rootPath = Join-Path $currentPath.Path $sourceRoot
            
            # Get all .d.ts files excluding node_modules and already collected files
            $dtsFiles = Get-ChildItem -Path ".\$sourceRoot" -Recurse -Filter "*.d.ts" -ErrorAction SilentlyContinue | Where-Object {
                $_.FullName -notlike "*node_modules*" -and
                $_.FullName -notlike "*frontend_final*" -and
                $_.FullName -notlike "*frontend_ui_only*" -and
                $_.FullName -notlike "*.cache*" -and
                $_.FullName -notlike "*build*" -and
                $_.FullName -notlike "*dist*"
            }
            
            Write-Host "Found $($dtsFiles.Count) TypeScript definition files in $sourceRoot" -ForegroundColor Cyan
            $currentFile = 0
            
            foreach ($file in $dtsFiles) {
                $currentFile++
                $percent = [math]::Round(($currentFile / $dtsFiles.Count) * 100, 1)
                Write-Progress -Activity "Processing $sourceRoot .d.ts files" -Status "$currentFile of $($dtsFiles.Count) ($percent%)" -PercentComplete $percent
                
                try {
                    # DEDUPLICATION: Check file hash to prevent duplicates
                    $fileHash = (Get-FileHash $file.FullName -Algorithm MD5 -ErrorAction Stop).Hash
                    if ($processedDtsHashes.ContainsKey($fileHash)) {
                        # Skip duplicate file
                        continue
                    }
                    $processedDtsHashes[$fileHash] = "$sourceRoot\$($file.Name)"
                    
                    # BULLETPROOF PATH CALCULATION: Handle relative paths properly
                    if ($file.FullName.StartsWith($rootPath)) {
                        $relativePath = $file.FullName.Substring($rootPath.Length + 1)
                        $destParentDir = Split-Path $relativePath -Parent
                        
                        if ([string]::IsNullOrEmpty($destParentDir)) {
                            $finalDestDir = Join-Path $tsDir $sourceRoot
                        } else {
                            $finalDestDir = Join-Path $tsDir "$sourceRoot\$destParentDir"
                        }
                        
                        # Handle very deep hierarchies (like Phaser3 plugins)
                        if ($finalDestDir.Length -gt 240) {
                            Write-Host "    ⚠ Very long path: $($finalDestDir.Length) chars" -ForegroundColor Yellow
                        }
                        
                        # Create destination directory structure
                        New-Item -ItemType Directory -Path $finalDestDir -Force -ErrorAction Stop | Out-Null
                        
                        # Copy .d.ts file
                        $destFile = Join-Path $finalDestDir $file.Name
                        Copy-Item -Path $file.FullName -Destination $destFile -ErrorAction Stop
                        
                        if ($currentFile % 50 -eq 0 -or $currentFile -le 10) {
                            Write-Host "    ✓ Copied $sourceRoot\$relativePath" -ForegroundColor Green
                        }
                        $global:phaseCSuccess++
                    }
                    
                } catch [System.UnauthorizedAccessException] {
                    Write-Host "    ⚠ Permission denied: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseCErrors++
                } catch [System.IO.IOException] {
                    Write-Host "    ⚠ File in use: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseCErrors++
                } catch [System.IO.PathTooLongException] {
                    Write-Host "    ⚠ Path too long: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseCErrors++
                } catch {
                    Write-Host "    ❌ Error with $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
                    $global:phaseCErrors++
                }
            }
            
            Write-Progress -Activity "Processing $sourceRoot .d.ts files" -Completed
            
            $sourceCount = (Get-ChildItem -Path (Join-Path $tsDir $sourceRoot) -Recurse -Filter "*.d.ts" -ErrorAction SilentlyContinue).Count
            Write-Host "✓ Collected $sourceCount TypeScript definition files from $sourceRoot" -ForegroundColor Green
            
        } catch {
            Write-Host "❌ Failed to process $sourceRoot: $($_.Exception.Message)" -ForegroundColor Red
            $global:phaseCErrors++
        }
        
    } else {
        Write-Host "$sourceRoot directory not found, skipping" -ForegroundColor Yellow
    }
}

# Final verification and reporting
try {
    $totalDtsCount = (Get-ChildItem -Path $tsDir -Recurse -Filter "*.d.ts" -ErrorAction SilentlyContinue).Count
    Write-Host "✅ Phase C Complete: $global:phaseCSuccess files copied successfully, $global:phaseCErrors errors" -ForegroundColor Green
    Write-Host "   Total TypeScript definition files: $totalDtsCount" -ForegroundColor Cyan
    Write-Host "   Duplicates prevented: $($processedDtsHashes.Count - $global:phaseCSuccess)" -ForegroundColor Cyan
} catch {
    Write-Host "⚠ Could not verify TypeScript file count" -ForegroundColor Yellow
}