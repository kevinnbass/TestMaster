# 🗂️ PHASE D: ADDITIONAL TEMPLATES MODULE
# Mission: Collect template files with refined patterns and comprehensive exclusions
# Created: 2025-08-23 | Modular Version

Write-Host "Phase D: Collecting additional template files (ENHANCED)..." -ForegroundColor Cyan

$processedTemplateHashes = @{} # Deduplication

try {
    $templatesDir = ".\frontend_final\additional_templates"
    New-Item -ItemType Directory -Path $templatesDir -Force -ErrorAction Stop | Out-Null
    Write-Host "✓ Created additional templates directory" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create templates directory: $($_.Exception.Message)" -ForegroundColor Red
    throw "Directory creation failed"
}

# Define template sources and patterns
$templateSources = @("TestMaster", "AWorld", "lagent", "crewAI", "MetaGPT", "competitors")
$templatePatterns = @("*template*", "*Template*", "*TEMPLATE*")

foreach ($source in $templateSources) {
    if (Test-Path ".\$source") {
        Write-Host "Searching for templates in $source..." -ForegroundColor Yellow
        
        try {
            $currentPath = Get-Location
            $rootPath = Join-Path $currentPath.Path $source
            
            foreach ($pattern in $templatePatterns) {
                $templateFiles = Get-ChildItem -Path ".\$source" -Recurse -Filter $pattern -ErrorAction SilentlyContinue | Where-Object {
                    $_.Extension -in @('.py', '.html', '.js', '.ts', '.tsx', '.jsx', '.yaml', '.yml', '.json', '.md') -and
                    $_.FullName -notlike "*frontend_final*" -and
                    $_.FullName -notlike "*frontend_ui_only*" -and
                    $_.FullName -notlike "*node_modules*" -and
                    $_.FullName -notlike "*.cache*" -and
                    $_.FullName -notlike "*build*" -and
                    $_.FullName -notlike "*dist*" -and
                    $_.FullName -notlike "*.git*" -and
                    $_.Length -lt 10MB  # Skip very large files
                }
                
                foreach ($file in $templateFiles) {
                    try {
                        # DEDUPLICATION: Check file hash to prevent duplicates
                        $fileHash = (Get-FileHash $file.FullName -Algorithm MD5 -ErrorAction Stop).Hash
                        if ($processedTemplateHashes.ContainsKey($fileHash)) {
                            # Skip duplicate file
                            continue
                        }
                        $processedTemplateHashes[$fileHash] = "$source\$($file.Name)"
                        
                        # BULLETPROOF PATH CALCULATION: Preserve source structure
                        if ($file.FullName.StartsWith($rootPath)) {
                            $relativePath = $file.FullName.Substring($rootPath.Length + 1)
                            $destParentDir = Split-Path $relativePath -Parent
                            
                            if ([string]::IsNullOrEmpty($destParentDir)) {
                                $finalDestDir = Join-Path $templatesDir $source
                            } else {
                                $finalDestDir = Join-Path $templatesDir "$source\$destParentDir"
                            }
                            
                            # Create destination directory structure
                            New-Item -ItemType Directory -Path $finalDestDir -Force -ErrorAction Stop | Out-Null
                            
                            # Copy template file
                            $destFile = Join-Path $finalDestDir $file.Name
                            Copy-Item -Path $file.FullName -Destination $destFile -ErrorAction Stop
                            
                            Write-Host "  ✓ Copied $source\$relativePath" -ForegroundColor Green
                            $global:phaseDSuccess++
                        }
                        
                    } catch [System.UnauthorizedAccessException] {
                        Write-Host "  ⚠ Permission denied: $($file.Name)" -ForegroundColor Yellow
                        $global:phaseDErrors++
                    } catch [System.IO.IOException] {
                        Write-Host "  ⚠ File in use: $($file.Name)" -ForegroundColor Yellow
                        $global:phaseDErrors++
                    } catch [System.IO.PathTooLongException] {
                        Write-Host "  ⚠ Path too long: $($file.Name)" -ForegroundColor Yellow
                        $global:phaseDErrors++
                    } catch {
                        Write-Host "  ❌ Error with $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
                        $global:phaseDErrors++
                    }
                }
            }
            
            $sourceCount = (Get-ChildItem -Path (Join-Path $templatesDir $source) -Recurse -ErrorAction SilentlyContinue).Count
            if ($sourceCount -gt 0) {
                Write-Host "✓ Collected $sourceCount template files from $source" -ForegroundColor Green
            }
            
        } catch {
            Write-Host "❌ Failed to process $source: $($_.Exception.Message)" -ForegroundColor Red
            $global:phaseDErrors++
        }
        
    } else {
        Write-Host "$source directory not found, skipping" -ForegroundColor Yellow
    }
}

# Final verification and reporting
try {
    $totalTemplateCount = (Get-ChildItem -Path $templatesDir -Recurse -ErrorAction SilentlyContinue).Count
    Write-Host "✅ Phase D Complete: $global:phaseDSuccess files copied successfully, $global:phaseDErrors errors" -ForegroundColor Green
    Write-Host "   Total additional template files: $totalTemplateCount" -ForegroundColor Cyan
    Write-Host "   Duplicates prevented: $($processedTemplateHashes.Count - $global:phaseDSuccess)" -ForegroundColor Cyan
} catch {
    Write-Host "⚠ Could not verify template file count" -ForegroundColor Yellow
}