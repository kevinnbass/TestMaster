# 🛡️ BULLETPROOF COLLECTION MASTER ORCHESTRATOR
# Mission: Execute all collection phases with individual error recovery
# Created: 2025-08-23 | Modular Architecture

param(
    [switch]$SkipPhaseA,
    [switch]$SkipPhaseB,
    [switch]$SkipPhaseC,
    [switch]$SkipPhaseD,
    [switch]$SkipPhaseE,
    [switch]$VerboseOutput
)

# GLOBAL PHASE TRACKING VARIABLES
$global:phaseAErrors = 0; $global:phaseASuccess = 0
$global:phaseBErrors = 0; $global:phaseBSuccess = 0  
$global:phaseCErrors = 0; $global:phaseCSuccess = 0
$global:phaseDErrors = 0; $global:phaseDSuccess = 0
$global:phaseEErrors = 0; $global:phaseESuccess = 0

$global:masterScriptErrors = 0
$global:completedPhases = @()
$global:failedPhases = @()

Write-Host "🛡️ BULLETPROOF COLLECTION ORCHESTRATOR INITIALIZED" -ForegroundColor Green
Write-Host "Global phase tracking variables declared" -ForegroundColor Cyan

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# PHASE EXECUTION FUNCTIONS
function Invoke-Phase {
    param(
        [string]$PhaseName,
        [string]$ScriptPath,
        [string]$Description
    )
    
    Write-Host "`n🎯 EXECUTING $PhaseName" -ForegroundColor Cyan
    
    if (-not (Test-Path $ScriptPath)) {
        Write-Host "❌ Phase script not found: $ScriptPath" -ForegroundColor Red
        $global:failedPhases += "$PhaseName - Script not found"
        $global:masterScriptErrors++
        return
    }
    
    try {
        # Execute phase script in current scope to maintain global variables
        . $ScriptPath
        $global:completedPhases += "$PhaseName - $Description"
        Write-Host "✅ $PhaseName completed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $PhaseName failed: $($_.Exception.Message)" -ForegroundColor Red
        $global:failedPhases += "$PhaseName - $($_.Exception.Message)"
        $global:masterScriptErrors++
    }
}

# MAIN EXECUTION SEQUENCE
Write-Host "🚀 STARTING BULLETPROOF ULTRA-CREATIVE COLLECTION" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Execute phases with individual error recovery
if (-not $SkipPhaseA) {
    Invoke-Phase "Phase A" (Join-Path $scriptDir "phase_a_html_backups.ps1") "HTML Backup Dashboards"
}

if (-not $SkipPhaseB) {
    Invoke-Phase "Phase B" (Join-Path $scriptDir "phase_b_webp_images.ps1") "WebP Image Files"
}

if (-not $SkipPhaseC) {
    Invoke-Phase "Phase C" (Join-Path $scriptDir "phase_c_typescript.ps1") "TypeScript Definitions"
}

if (-not $SkipPhaseD) {
    Invoke-Phase "Phase D" (Join-Path $scriptDir "phase_d_templates.ps1") "Additional Templates"
}

if (-not $SkipPhaseE) {
    Invoke-Phase "Phase E" (Join-Path $scriptDir "phase_e_configs.ps1") "Specialized Configurations"
}

# FINAL VERIFICATION
Write-Host "`n📊 EXECUTING FINAL VERIFICATION" -ForegroundColor Cyan
try {
    . (Join-Path $scriptDir "phase_f_verification.ps1")
    $global:completedPhases += "Final Verification"
}
catch {
    Write-Host "❌ Final verification failed: $($_.Exception.Message)" -ForegroundColor Red
    $global:failedPhases += "Final Verification - $($_.Exception.Message)"
    $global:masterScriptErrors++
}

# COMPLETION SUMMARY
Write-Host "`n🏁 BULLETPROOF COLLECTION COMPLETE" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host "Completed Phases: $($global:completedPhases.Count)" -ForegroundColor Green
Write-Host "Failed Phases: $($global:failedPhases.Count)" -ForegroundColor $(if ($global:failedPhases.Count -eq 0) { "Green" } else { "Red" })
Write-Host "Master Script Errors: $global:masterScriptErrors" -ForegroundColor $(if ($global:masterScriptErrors -eq 0) { "Green" } else { "Red" })

if ($global:completedPhases.Count -gt 0) {
    Write-Host "`n✅ SUCCESSFULLY COMPLETED:" -ForegroundColor Green
    foreach ($phase in $global:completedPhases) {
        Write-Host "  - $phase" -ForegroundColor Cyan
    }
}

if ($global:failedPhases.Count -gt 0) {
    Write-Host "`n❌ FAILED PHASES:" -ForegroundColor Red
    foreach ($failure in $global:failedPhases) {
        Write-Host "  - $failure" -ForegroundColor Yellow
    }
}

if ($global:masterScriptErrors -eq 0) {
    Write-Host "`n🎉 PERFECT EXECUTION: All phases completed without cascade failures!" -ForegroundColor Green
} elseif ($global:completedPhases.Count -ge 4) {
    Write-Host "`n✅ SUBSTANTIAL SUCCESS: Most phases completed successfully" -ForegroundColor Green
} else {
    Write-Host "`n⚠ REVIEW REQUIRED: Multiple phase failures detected" -ForegroundColor Yellow
}