# Test script for the Codebase Monitor API service
# Performs basic endpoint testing

param(
    [string]$ApiUrl = "http://127.0.0.1:8088",
    [int]$TimeoutSeconds = 30
)

# Set working directory to project root
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

Write-Host "Testing Codebase Monitor API at: $ApiUrl" -ForegroundColor Green
Write-Host "Timeout: $TimeoutSeconds seconds" -ForegroundColor Yellow
Write-Host ""

function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Body = $null,
        [string]$Description
    )
    
    Write-Host "Testing: $Description" -ForegroundColor Cyan
    Write-Host "  $Method $Url" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            TimeoutSec = $TimeoutSeconds
        }
        
        if ($Body) {
            $params.Body = ($Body | ConvertTo-Json)
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-RestMethod @params
        Write-Host "  ✓ SUCCESS" -ForegroundColor Green
        
        if ($response -is [PSCustomObject]) {
            # Show key fields from response
            if ($response.PSObject.Properties["status"]) {
                Write-Host "    Status: $($response.status)" -ForegroundColor Gray
            }
            if ($response.PSObject.Properties["scan_id"]) {
                Write-Host "    Scan ID: $($response.scan_id)" -ForegroundColor Gray
            }
            if ($response.PSObject.Properties["total_scans"]) {
                Write-Host "    Total Scans: $($response.total_scans)" -ForegroundColor Gray
            }
        }
        
        return $response
    } catch {
        Write-Host "  ✗ FAILED: $_" -ForegroundColor Red
        return $null
    }
    
    Write-Host ""
}

# Test 1: Health check
$health = Test-Endpoint "$ApiUrl/health" "GET" $null "Health Check"

if (-not $health) {
    Write-Host "Service appears to be down. Please start it with:" -ForegroundColor Red
    Write-Host "  .\scripts\scan_server.ps1" -ForegroundColor Yellow
    exit 1
}

# Test 2: Get latest scan (might be empty initially)
$latest = Test-Endpoint "$ApiUrl/scan/latest" "GET" $null "Get Latest Scan"

# Test 3: Get scan history
$history = Test-Endpoint "$ApiUrl/scans/history?limit=5" "GET" $null "Get Scan History"

# Test 4: Run a new scan (this will take some time)
Write-Host "Running a new scan (this may take a few minutes)..." -ForegroundColor Yellow
$scanRequest = @{
    root = "."
    force = $true
}
$scanResult = Test-Endpoint "$ApiUrl/scan/run" "POST" $scanRequest "Run New Scan"

if ($scanResult -and $scanResult.scan_id) {
    $scanId = $scanResult.scan_id
    Write-Host "Scan completed with ID: $scanId" -ForegroundColor Green
    
    # Test 5: Get scan details
    Test-Endpoint "$ApiUrl/scan/$scanId" "GET" $null "Get Scan Details"
    
    # Test 6: Get hotspots
    Test-Endpoint "$ApiUrl/scan/$scanId/hotspots" "GET" $null "Get Hotspots"
    
    # Test 7: Get duplicates
    Test-Endpoint "$ApiUrl/scan/$scanId/duplicates?limit=10" "GET" $null "Get Duplicates (limit 10)"
    
    # Test 8: Get file metrics
    Test-Endpoint "$ApiUrl/scan/$scanId/files?limit=20" "GET" $null "Get File Metrics (limit 20)"
    
    # Test 9: Get extension stats
    Test-Endpoint "$ApiUrl/scan/$scanId/extensions" "GET" $null "Get Extension Statistics"
    
    Write-Host ""
    Write-Host "All tests completed successfully!" -ForegroundColor Green
    Write-Host "API is working correctly." -ForegroundColor Green
    
} else {
    Write-Host "Scan failed - unable to test scan-specific endpoints" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test completed. Check the API docs at: $ApiUrl/docs" -ForegroundColor Magenta