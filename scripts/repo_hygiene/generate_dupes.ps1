# Generate duplicate groups JSON for cleanup
$scanFile = "tools\codebase_monitor\reports\latest\scan.json"
$outputFile = "tools\codebase_monitor\outputs\duplicate_groups_topN.json"

if (-not (Test-Path $scanFile)) {
    Write-Error "Scan file not found: $scanFile"
    exit 1
}

try {
    $scanContent = Get-Content $scanFile -Encoding UTF8 -Raw
    $scan = $scanContent | ConvertFrom-Json
    
    if ($scan.duplicates -and $scan.duplicates.Count -gt 0) {
        $groups = $scan.duplicates | Select-Object -First 200
        $outputData = @{
            "duplicate_groups" = $groups
            "generated_at" = (Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC")
            "total_groups" = $groups.Count
            "source_scan" = $scanFile
        }
        
        $outputData | ConvertTo-Json -Depth 8 | Out-File $outputFile -Encoding utf8
        
        Write-Host "Generated duplicate groups: $($groups.Count) groups"
        Write-Host "Saved to: $outputFile"
    } else {
        # Create empty file if no duplicates
        @{
            "duplicate_groups" = @()
            "generated_at" = (Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC")
            "total_groups" = 0
            "source_scan" = $scanFile
            "note" = "No duplicate groups found in scan"
        } | ConvertTo-Json -Depth 8 | Out-File $outputFile -Encoding utf8
        
        Write-Host "No duplicate groups found in scan"
        Write-Host "Empty file saved to: $outputFile"
    }
} catch {
    Write-Error "Failed to process scan file: $_"
    exit 1
}