param(
    [string]$Base = "C:\Users\kbass\OneDrive\Documents\testmaster"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RelativePath {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Root
    )
    $full = [System.IO.Path]::GetFullPath($Path)
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    if ($full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($rootFull.Length).TrimStart('\\')
    }
    return $null
}

function Ensure-ParentDirectory {
    param([Parameter(Mandatory=$true)][string]$TargetPath)
    $parent = Split-Path -Path $TargetPath -Parent
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Safe-GetFileHash {
    param([Parameter(Mandatory=$true)][string]$Path)
    try {
        return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path -ErrorAction Stop).Hash
    } catch {
        return $null
    }
}

$srcs = @(
    [PSCustomObject]@{ Name = 'frontend_ui_only'; Path = (Join-Path $Base 'frontend_ui_only') },
    [PSCustomObject]@{ Name = 'frontend_unified';  Path = (Join-Path $Base 'frontend_unified') }
)

$destRoot     = Join-Path $Base 'frontend_final'
$timestamp    = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportRoot   = Join-Path $Base ("_migration_reports\full_migration_" + $timestamp)
$conflictRoot = Join-Path $destRoot ("_conflicts\full_migration_" + $timestamp)

New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
New-Item -ItemType Directory -Path $conflictRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $destRoot)) { throw "Destination root does not exist: $destRoot" }

$logCopied       = New-Object System.Collections.Generic.List[object]
$logSkippedSame  = New-Object System.Collections.Generic.List[object]
$logConflicts    = New-Object System.Collections.Generic.List[object]
$logErrors       = New-Object System.Collections.Generic.List[object]
$allEnumerated   = New-Object System.Collections.Generic.List[object]

foreach ($src in $srcs) {
    if (-not (Test-Path -LiteralPath $src.Path)) { continue }
    $files = Get-ChildItem -Path $src.Path -Recurse -File -Force -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $rel = Get-RelativePath -Path $f.FullName -Root $src.Path
        if ([string]::IsNullOrWhiteSpace($rel)) { continue }
        $allEnumerated.Add([PSCustomObject]@{ Source=$src.Name; SourcePath=$f.FullName; RelativePath=$rel }) | Out-Null
    }
}

foreach ($src in $srcs) {
    if (-not (Test-Path -LiteralPath $src.Path)) { continue }
    $files = Get-ChildItem -Path $src.Path -Recurse -File -Force -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        try {
            $rel = Get-RelativePath -Path $f.FullName -Root $src.Path
            if ([string]::IsNullOrWhiteSpace($rel)) { continue }
            $destPath = Join-Path $destRoot $rel

            if (-not (Test-Path -LiteralPath $destPath)) {
                Ensure-ParentDirectory -TargetPath $destPath
                Copy-Item -LiteralPath $f.FullName -Destination $destPath -Force
                $dstHash = Safe-GetFileHash -Path $destPath
                $srcHash = Safe-GetFileHash -Path $f.FullName
                $logCopied.Add([PSCustomObject]@{
                    Time=(Get-Date); Action='copied_new'; Source=$src.Name; SourcePath=$f.FullName; RelativePath=$rel; DestPath=$destPath; SourceHash=$srcHash; DestHash=$dstHash; Note='new file'
                }) | Out-Null
                continue
            }

            $srcHash2 = Safe-GetFileHash -Path $f.FullName
            $dstHash2 = Safe-GetFileHash -Path $destPath
            if ($srcHash2 -and $dstHash2 -and ($srcHash2 -eq $dstHash2)) {
                $logSkippedSame.Add([PSCustomObject]@{
                    Time=(Get-Date); Action='skipped_same'; Source=$src.Name; SourcePath=$f.FullName; RelativePath=$rel; DestPath=$destPath; SourceHash=$srcHash2; DestHash=$dstHash2; Note='identical content'
                }) | Out-Null
            } else {
                # Conflict: archive both versions to ensure unique content is preserved inside frontend_final
                $archiveSrc  = Join-Path $conflictRoot (Join-Path $src.Name $rel)
                Ensure-ParentDirectory -TargetPath $archiveSrc
                Copy-Item -LiteralPath $f.FullName -Destination $archiveSrc -Force

                $archiveDest = Join-Path $conflictRoot (Join-Path 'dest_existing' $rel)
                if (-not (Test-Path -LiteralPath $archiveDest)) {
                    Ensure-ParentDirectory -TargetPath $archiveDest
                    Copy-Item -LiteralPath $destPath -Destination $archiveDest -Force
                }

                $logConflicts.Add([PSCustomObject]@{
                    Time=(Get-Date); Action='conflict_archived'; Source=$src.Name; SourcePath=$f.FullName; RelativePath=$rel; DestPath=$destPath; SourceHash=$srcHash2; DestHash=$dstHash2; ArchiveSrc=$archiveSrc; ArchiveDest=$archiveDest; Note='content differs; archived source and existing dest'
                }) | Out-Null
            }
        } catch {
            $logErrors.Add([PSCustomObject]@{
                Time=(Get-Date); Action='error'; Source=$src.Name; SourcePath=$f.FullName; RelativePath=$rel; Error=$_.Exception.Message
            }) | Out-Null
        }
    }
}

# Verification: ensure every enumerated source file is present either at dest relative path or archived under conflicts
$missing = New-Object System.Collections.Generic.List[object]
foreach ($item in $allEnumerated) {
    $destPath = Join-Path $destRoot $item.RelativePath
    $archSrc  = Join-Path $conflictRoot (Join-Path $item.Source $item.RelativePath)
    if (-not (Test-Path -LiteralPath $destPath) -and -not (Test-Path -LiteralPath $archSrc)) {
        $missing.Add([PSCustomObject]@{ Source=$item.Source; SourcePath=$item.SourcePath; RelativePath=$item.RelativePath; DestPath=$destPath }) | Out-Null
    }
}

# Write reports
$copiedPath      = Join-Path $reportRoot 'copied.csv'
$skippedPath     = Join-Path $reportRoot 'skipped_identical.csv'
$conflictsPath   = Join-Path $reportRoot 'conflicts.csv'
$errorsPath      = Join-Path $reportRoot 'errors.csv'
$missingPath     = Join-Path $reportRoot 'missing.csv'
$summaryPath     = Join-Path $reportRoot 'summary.txt'

($logCopied | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path $copiedPath -Encoding UTF8
($logSkippedSame | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path $skippedPath -Encoding UTF8
($logConflicts | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path $conflictsPath -Encoding UTF8
($logErrors | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path $errorsPath -Encoding UTF8
($missing | ConvertTo-Csv -NoTypeInformation) | Set-Content -Path $missingPath -Encoding UTF8

$summary = @()
$summary += "Full migration run: $timestamp"
$summary += "Base: $Base"
$summary += "Destination: $destRoot"
$summary += "Reports: $reportRoot"
$summary += "Conflicts archived to: $conflictRoot"
$summary += "Copied: $($logCopied.Count)"
$summary += "Skipped identical: $($logSkippedSame.Count)"
$summary += "Conflicts archived: $($logConflicts.Count)"
$summary += "Errors: $($logErrors.Count)"
$summary += "Missing coverage: $($missing.Count)"
Set-Content -Path $summaryPath -Value $summary -Encoding UTF8

Write-Host "Migration complete. Summary: $summaryPath"
Write-Host "Copied: $($logCopied.Count); Identical: $($logSkippedSame.Count); Conflicts: $($logConflicts.Count); Errors: $($logErrors.Count); Missing: $($missing.Count)"

if ($missing.Count -gt 0) {
    Write-Warning "Some source files were not accounted for (see missing.csv)."
}

