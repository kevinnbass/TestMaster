# Setup latest reports directory
New-Item -ItemType Directory -Force -Path tools\codebase_monitor\reports\latest | Out-Null

$scans = Get-ChildItem tools\codebase_monitor\reports\scan_*.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending

if ($scans) {
    $latest = $scans[0]
    Copy-Item $latest.FullName tools\codebase_monitor\reports\latest\scan.json -Force
    $summaryPath = $latest.FullName -replace 'scan_', 'summary_' -replace '.json', '.md'
    if (Test-Path $summaryPath) {
        Copy-Item $summaryPath tools\codebase_monitor\reports\latest\summary.md -Force
    }
    Write-Host "Copied latest scan to reports/latest"
} else {
    Write-Host "No scan files found - will create placeholder"
    # Create a placeholder if no scans exist yet
    @{
        root = "."
        generated_at_epoch = [int][double]::Parse((Get-Date -UFormat %s))
        total_files = 0
        total_size_bytes = 0
        total_code_lines = 0
        duplicates = @()
        hotspots = @{}
        directory_summaries = @()
        extensions = @{}
    } | ConvertTo-Json -Depth 10 | Out-File tools\codebase_monitor\reports\latest\scan.json -Encoding utf8
}