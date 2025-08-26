# ⚙️ PHASE E: SPECIALIZED CONFIGURATIONS MODULE
# Mission: Collect specialized configuration files with bulletproof deduplication
# Created: 2025-08-23 | Modular Version

Write-Host "Phase E: Collecting specialized configuration files (BULLETPROOF)..." -ForegroundColor Cyan

$processedConfigHashes = @{} # Deduplication

try {
    $configsDir = ".\frontend_final\specialized_configs"
    New-Item -ItemType Directory -Path $configsDir -Force -ErrorAction Stop | Out-Null
    Write-Host "✓ Created specialized configs directory" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create configs directory: $($_.Exception.Message)" -ForegroundColor Red
    throw "Directory creation failed"
}

# Define specialized config file patterns by category
$configPatterns = @{
    "webmanifest" = @("*.webmanifest")
    "service_workers" = @("*service-worker*", "*sw.js")
    "pwa_configs" = @("*pwa*", "*PWA*")
    "babel_configs" = @(".babelrc*", "*babel*")
    "postcss_configs" = @("*postcss*", "*PostCSS*")
    "tailwind_configs" = @("*tailwind*", "*Tailwind*")
    "vite_configs" = @("*vite*", "*Vite*")
    "webpack_configs" = @("*webpack*", "*Webpack*")
    "rollup_configs" = @("*rollup*", "*Rollup*")
    "parcel_configs" = @("*parcel*", "*Parcel*")
}

foreach ($configType in $configPatterns.GetEnumerator()) {
    Write-Host "Searching for $($configType.Key) files..." -ForegroundColor Yellow
    $destDir = Join-Path $configsDir $configType.Key
    
    try {
        New-Item -ItemType Directory -Path $destDir -Force -ErrorAction Stop | Out-Null
        
        foreach ($pattern in $configType.Value) {
            $configFiles = Get-ChildItem -Path "." -Recurse -Filter $pattern -ErrorAction SilentlyContinue | Where-Object {
                $_.FullName -notlike "*node_modules*" -and
                $_.FullName -notlike "*frontend_final*" -and
                $_.FullName -notlike "*frontend_ui_only*" -and
                $_.FullName -notlike "*.cache*" -and
                $_.FullName -notlike "*build*" -and
                $_.FullName -notlike "*dist*" -and
                $_.FullName -notlike "*.git*" -and
                $_.Extension -in @('.js', '.json', '.yml', '.yaml', '.ts', '.mjs', '.webmanifest', '.config.js', '') -and
                $_.Length -lt 5MB  # Skip very large config files
            }
            
            foreach ($file in $configFiles) {
                try {
                    # DEDUPLICATION: Check file hash to prevent duplicates
                    $fileHash = (Get-FileHash $file.FullName -Algorithm MD5 -ErrorAction Stop).Hash
                    if ($processedConfigHashes.ContainsKey($fileHash)) {
                        # Skip duplicate file
                        continue
                    }
                    $processedConfigHashes[$fileHash] = "$($configType.Key)\$($file.Name)"
                    
                    # Calculate source framework for better organization
                    $sourceFramework = "unknown"
                    if ($file.DirectoryName -match "agentops") { $sourceFramework = "agentops" }
                    elseif ($file.DirectoryName -match "autogen") { $sourceFramework = "autogen" }
                    elseif ($file.DirectoryName -match "agent-squad") { $sourceFramework = "agent-squad" }
                    elseif ($file.DirectoryName -match "AgentVerse") { $sourceFramework = "AgentVerse" }
                    elseif ($file.DirectoryName -match "TestMaster") { $sourceFramework = "TestMaster" }
                    
                    # Create framework-specific subdirectory for better organization
                    $frameworkDestDir = Join-Path $destDir $sourceFramework
                    New-Item -ItemType Directory -Path $frameworkDestDir -Force -ErrorAction Stop | Out-Null
                    
                    # Copy config file preserving original name
                    $destFile = Join-Path $frameworkDestDir $file.Name
                    Copy-Item -Path $file.FullName -Destination $destFile -ErrorAction Stop
                    
                    Write-Host "  ✓ Collected $($file.Name) from $sourceFramework" -ForegroundColor Green
                    $global:phaseESuccess++
                    
                } catch [System.UnauthorizedAccessException] {
                    Write-Host "  ⚠ Permission denied: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseEErrors++
                } catch [System.IO.IOException] {
                    Write-Host "  ⚠ File in use: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseEErrors++
                } catch [System.IO.PathTooLongException] {
                    Write-Host "  ⚠ Path too long: $($file.Name)" -ForegroundColor Yellow
                    $global:phaseEErrors++
                } catch {
                    Write-Host "  ❌ Error with $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
                    $global:phaseEErrors++
                }
            }
        }
        
        $typeCount = (Get-ChildItem -Path $destDir -Recurse -ErrorAction SilentlyContinue).Count
        if ($typeCount -gt 0) {
            Write-Host "✓ Collected $typeCount $($configType.Key) files" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "❌ Failed to process $($configType.Key): $($_.Exception.Message)" -ForegroundColor Red
        $global:phaseEErrors++
    }
}

# Final verification and reporting
try {
    $totalConfigCount = (Get-ChildItem -Path $configsDir -Recurse -ErrorAction SilentlyContinue).Count
    Write-Host "✅ Phase E Complete: $global:phaseESuccess files copied successfully, $global:phaseEErrors errors" -ForegroundColor Green
    Write-Host "   Total specialized configuration files: $totalConfigCount" -ForegroundColor Cyan
    Write-Host "   Duplicates prevented: $($processedConfigHashes.Count - $global:phaseESuccess)" -ForegroundColor Cyan
    
    # Show breakdown by config type
    Write-Host "   Config types collected:" -ForegroundColor Cyan
    foreach ($configType in $configPatterns.Keys) {
        $typeDir = Join-Path $configsDir $configType
        if (Test-Path $typeDir) {
            $count = (Get-ChildItem -Path $typeDir -Recurse -ErrorAction SilentlyContinue).Count
            if ($count -gt 0) {
                Write-Host "     - $configType`: $count files" -ForegroundColor DarkCyan
            }
        }
    }
} catch {
    Write-Host "⚠ Could not verify config file count" -ForegroundColor Yellow
}