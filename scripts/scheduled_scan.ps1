# Scheduled Codebase Scan Script
# Triggers automated scans via the codebase monitor API

param(
    [string]$ApiUrl = "http://127.0.0.1:8088",
    [string]$Root = ".",
    [switch]$Force = $false,
    [int]$TimeoutMinutes = 10,
    [string]$LogFile = "",
    [switch]$Quiet = $false
)

# Set working directory to project root
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Set up logging
if ([string]::IsNullOrEmpty($LogFile)) {
    $LogFile = "tools\codebase_monitor\scheduled_scan.log"
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    if (-not $Quiet) {
        Write-Host $logEntry
    }
    
    # Append to log file
    New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
    Add-Content -Path $LogFile -Value $logEntry
}

function Test-ServiceHealth {
    param([string]$Url)
    
    try {
        $healthUrl = "$Url/health"
        $response = Invoke-RestMethod -Uri $healthUrl -Method GET -TimeoutSec 10
        return $response.status -eq "ok"
    } catch {
        Write-Log "Health check failed: $_" "ERROR"
        return $false
    }
}

function Start-Scan {
    param([string]$Url, [string]$ScanRoot, [bool]$ForceRun)
    
    try {
        $scanUrl = "$Url/scan/run"
        $body = @{
            root = $ScanRoot
            force = $ForceRun
        } | ConvertTo-Json
        
        Write-Log "Starting scan for root: $ScanRoot"
        $response = Invoke-RestMethod -Uri $scanUrl -Method POST -Body $body -ContentType "application/json" -TimeoutSec ($TimeoutMinutes * 60)
        
        return $response
    } catch {
        Write-Log "Scan failed: $_" "ERROR"
        return $null
    }
}

function Get-LatestScan {
    param([string]$Url)
    
    try {
        $latestUrl = "$Url/scan/latest"
        $response = Invoke-RestMethod -Uri $latestUrl -Method GET -TimeoutSec 10
        return $response
    } catch {
        Write-Log "Failed to get latest scan: $_" "ERROR"
        return $null
    }
}

# Main execution
Write-Log "=== Scheduled Scan Started ===" "INFO"
Write-Log "API URL: $ApiUrl"
Write-Log "Scan Root: $Root"
Write-Log "Force: $Force"
Write-Log "Timeout: $TimeoutMinutes minutes"

# Check if service is running
Write-Log "Checking service health..."
if (-not (Test-ServiceHealth $ApiUrl)) {
    Write-Log "Service is not healthy. Attempting to start..." "WARN"
    
    # Try to start the service
    try {
        $serverScript = Join-Path $PSScriptRoot "scan_server.ps1"
        if (Test-Path $serverScript) {
            Write-Log "Starting service..." "INFO"
            Start-Process -FilePath "powershell.exe" -ArgumentList "-File", $serverScript, "-Host", "127.0.0.1", "-Port", "8088" -WindowStyle Hidden
            
            # Wait for service to start
            Start-Sleep -Seconds 10
            
            if (-not (Test-ServiceHealth $ApiUrl)) {
                Write-Log "Failed to start service automatically" "ERROR"
                exit 1
            }
        } else {
            Write-Log "Service startup script not found: $serverScript" "ERROR"
            exit 1
        }
    } catch {
        Write-Log "Failed to start service: $_" "ERROR"
        exit 1
    }
}

# Get current scan status
$latestScan = Get-LatestScan $ApiUrl
if ($latestScan -and $latestScan.scan_id -gt 0) {
    $ageHours = [math]::Round($latestScan.age_seconds / 3600, 2)
    Write-Log "Latest scan ID: $($latestScan.scan_id), Age: $ageHours hours"
    
    # Skip scan if recent (less than 1 hour) and not forced
    if (-not $Force -and $latestScan.age_seconds -lt 3600) {
        Write-Log "Recent scan found (age: $ageHours hours). Skipping." "INFO"
        Write-Log "=== Scheduled Scan Completed (Skipped) ===" "INFO"
        exit 0
    }
}

# Run the scan
Write-Log "Triggering new scan..."
$scanResult = Start-Scan $ApiUrl $Root $Force

if ($scanResult) {
    Write-Log "Scan completed successfully!" "SUCCESS"
    Write-Log "Scan ID: $($scanResult.scan_id)"
    Write-Log "Duration: $($scanResult.duration_seconds) seconds"
    
    if ($scanResult.summary) {
        Write-Log "Total Files: $($scanResult.summary.total_files)"
        Write-Log "Total Size: $([math]::Round($scanResult.summary.total_size_bytes / 1MB, 2)) MB"
        Write-Log "Total Code Lines: $($scanResult.summary.total_code_lines)"
    }
    
    # Update latest symlink/copy
    try {
        $latestDir = "tools\codebase_monitor\reports\latest"
        New-Item -ItemType Directory -Force -Path $latestDir | Out-Null
        
        if ($scanResult.json_report -and (Test-Path $scanResult.json_report)) {
            Copy-Item $scanResult.json_report "$latestDir\scan.json" -Force
            Write-Log "Updated latest scan report"
        }
        
        if ($scanResult.markdown_report -and (Test-Path $scanResult.markdown_report)) {
            Copy-Item $scanResult.markdown_report "$latestDir\summary.md" -Force
            Write-Log "Updated latest summary report"
        }
    } catch {
        Write-Log "Failed to update latest reports: $_" "WARN"
    }
} else {
    Write-Log "Scan failed!" "ERROR"
    Write-Log "=== Scheduled Scan Failed ===" "ERROR"
    exit 1
}

Write-Log "=== Scheduled Scan Completed Successfully ===" "SUCCESS"

# Optional: Clean up old scans (keep last 10)
try {
    $allScans = Invoke-RestMethod -Uri "$ApiUrl/scans/history?limit=50" -Method GET -TimeoutSec 30
    if ($allScans.scans.Count -gt 10) {
        $oldScans = $allScans.scans | Sort-Object generated_at_epoch | Select-Object -First ($allScans.scans.Count - 10)
        foreach ($oldScan in $oldScans) {
            try {
                Invoke-RestMethod -Uri "$ApiUrl/scan/$($oldScan.id)" -Method DELETE -TimeoutSec 30 | Out-Null
                Write-Log "Cleaned up old scan ID: $($oldScan.id)" "INFO"
            } catch {
                Write-Log "Failed to clean up scan ID $($oldScan.id): $_" "WARN"
            }
        }
    }
} catch {
    Write-Log "Failed to clean up old scans: $_" "WARN"
}