## Frontend UI/Dashboard/Visualization Migration Roadmap

This roadmap consolidates only UI/frontend/dashboard/visualization components into `frontend_final` from:
- `frontend_ui_only`
- `frontend_unified`

It excludes non-UI items (docker, dependency locks, Python bridges, test artifacts, docs autogen, etc.). Use the dry‑run script below to generate a manifest and conflicts report before any copy operations.

### Canonical target
- **Canonical frontend root**: `frontend_final`

### Scope: includes vs. excludes
- **Includes (only UI/frontend/visualization):**
  - From `frontend_ui_only`
    - `agent_specific\...` (TSX UI components)
    - `discovered\charts\...` (chart/dash UI)
    - `templates\js\...` (UI scripts referenced by dashboards/templates)
    - `yaml_components\...` (UI component definitions)
    - `assets\images\...` (images referenced by included UI)
    - `minified\...` (only files actually referenced by included UI/templates)
  - From `frontend_unified`
    - `external_ui\agentops\...` (TSX UI components)
    - `external_ui\fix_analytics_charts.js`

- **Excludes (not UI/frontend/visualization):**
  - `frontend_ui_only\discovered\dependencies\...`
  - `frontend_ui_only\discovered\docker\...`
  - `frontend_ui_only\discovered\tests\...`
  - `frontend_ui_only\python_ui\...` and `frontend_ui_only\discovered\python_ui\...`
  - `frontend_ui_only\json_configs\...` (tooling/config/sample data)
  - `frontend_unified\external_ui\autogen\...` and `frontend_unified\external_ui\coverage_report.html`

### Target layout in `frontend_final`
- `frontend_final\agent_specific\...`
- `frontend_final\discovered\charts\...`
- `frontend_final\templates\js\...`
- `frontend_final\yaml_components\...`
- `frontend_final\assets\images\...`
- `frontend_final\minified\...` (subset that is referenced)
- `frontend_final\external_ui\agentops\...`
- `frontend_final\external_ui\fix_analytics_charts.js`

## Implementation phases
1) Inventory and preflight
- Verify source paths exist and list counts for each include path.

2) Generate dry‑run manifest (no file copies)
- Run the PowerShell script below to:
  - Enumerate only in-scope files
  - Detect referenced `minified` assets (searches filename references across relevant code files)
  - Produce a migration manifest and a conflicts report (target already exists)

3) Review conflicts
- For each `source -> target` in conflicts, prefer `frontend_final` versions unless the incoming file is a missing visualization.
- Record resolutions in a small table for audit.

4) (Optional) Stage and integrate
- After review, copy files from the manifest (excluding conflicts) into the target layout above.
- Keep any replaced files in `frontend_final\_archive_conflicts\...` for audit if overrides are approved.

5) Wire‑up and verify
- Validate imports for `external_ui\agentops\...` and `templates\js\...`.
- Ensure only the referenced `minified` assets are present.
- Build/test UI; verify dashboards/charts render without broken imports.

6) Cleanup
- Remove temporary staging (if used) and any unused `minified` assets.

7) Documentation and VCS
- Update `README.md` to describe the canonical structure and migration outcome.
- Commit and push the migration once validated.

## Acceptance criteria
- Only UI/frontend/visualization content is added; no Python/test/docker/lock/doc-autogen content introduced.
- Dashboards and charts render correctly with resolved imports.
- No unintended overwrites; all conflicts reviewed and resolved.

## Dry‑run manifest script (PowerShell)
The script below is safe: it does not copy or modify source/target files. It writes reports to `_migration_reports` at the workspace root.

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

param(
    [string]$Base = "C:\Users\kbass\OneDrive\Documents\testmaster"
)

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

function Get-AllFilesUnder {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (Test-Path $Path) {
        return Get-ChildItem -Path $Path -Recurse -File -Force -ErrorAction SilentlyContinue
    }
    @()
}

$srcUIOnly   = Join-Path $Base 'frontend_ui_only'
$srcUnified  = Join-Path $Base 'frontend_unified'
$dest        = Join-Path $Base 'frontend_final'
$reportDir   = Join-Path $Base '_migration_reports'

New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

$uiOnlyDirs = @(
    'agent_specific',
    'discovered\charts',
    'templates\js',
    'yaml_components',
    'assets\images'
)

$uiOnlyFiles = @()
foreach ($d in $uiOnlyDirs) {
    $p = Join-Path $srcUIOnly $d
    $uiOnlyFiles += Get-AllFilesUnder -Path $p
}

$uiOnlyMinifiedRoot = Join-Path $srcUIOnly 'minified'
$codeExts = @('.tsx','.ts','.js','.jsx','.html','.htm','.yml','.yaml','.css')

# Build the code search set: included UI files + existing frontend_final code files
$destCodeFiles = @()
if (Test-Path $dest) {
    $destCodeFiles = Get-ChildItem -Path $dest -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $codeExts -contains $_.Extension.ToLower() }
}
$codeFiles = ($uiOnlyFiles + $destCodeFiles) | Select-Object -ExpandProperty FullName -Unique

# Determine referenced vs unused minified assets by filename occurrence across code files
$minifiedReferenced = @()
$minifiedUnused = @()
if (Test-Path $uiOnlyMinifiedRoot) {
    $minifiedAll = Get-AllFilesUnder -Path $uiOnlyMinifiedRoot
    foreach ($mf in $minifiedAll) {
        $pattern = [Regex]::Escape($mf.Name)
        $isRef = $false
        if ($codeFiles.Count -gt 0) {
            try {
                $isRef = Select-String -Path $codeFiles -Pattern $pattern -SimpleMatch -Quiet -ErrorAction SilentlyContinue
            } catch {
                $isRef = $false
            }
        }
        if ($isRef) { $minifiedReferenced += $mf } else { $minifiedUnused += $mf }
    }
}

# Unified sources
$unifiedAgentOpsRoot = Join-Path $srcUnified 'external_ui\agentops'
$unifiedAgentOpsFiles = Get-AllFilesUnder -Path $unifiedAgentOpsRoot

$unifiedSingles = @(
    Join-Path $srcUnified 'external_ui\fix_analytics_charts.js'
)
$unifiedSingleFiles = $unifiedSingles | Where-Object { Test-Path $_ } | ForEach-Object { Get-Item $_ }

# Build candidate list
$candidates = @()
$candidates += $uiOnlyFiles
$candidates += $minifiedReferenced
$candidates += $unifiedAgentOpsFiles
$candidates += $unifiedSingleFiles

if ($candidates.Count -eq 0) {
    Write-Host 'No candidate files found in scope.'
}

$manifestPath   = Join-Path $reportDir 'ui_migration_manifest.txt'
$conflictsPath  = Join-Path $reportDir 'ui_migration_conflicts.txt'
$unusedMinPath  = Join-Path $reportDir 'unused_minified.txt'
$summaryPath    = Join-Path $reportDir 'summary.txt'

Remove-Item -Path $manifestPath,$conflictsPath,$unusedMinPath,$summaryPath -ErrorAction SilentlyContinue

$manifestLines = New-Object System.Collections.Generic.List[string]
$conflictLines = New-Object System.Collections.Generic.List[string]

foreach ($f in $candidates) {
    $root = if ($f.FullName.StartsWith(([IO.Path]::GetFullPath($srcUIOnly)), [System.StringComparison]::OrdinalIgnoreCase)) { $srcUIOnly } else { $srcUnified }
    $rel = Get-RelativePath -Path $f.FullName -Root $root
    if ([string]::IsNullOrWhiteSpace($rel)) { continue }
    $destPath = Join-Path $dest $rel
    $line = "COPY `"$($f.FullName)`" -> `"$destPath`""
    $manifestLines.Add($line) | Out-Null
    if (Test-Path $destPath) {
        $conflictLines.Add($line) | Out-Null
    }
}

Set-Content -Path $manifestPath -Value $manifestLines -Encoding UTF8
Set-Content -Path $conflictsPath -Value $conflictLines -Encoding UTF8
Set-Content -Path $unusedMinPath -Value ($minifiedUnused | Select-Object -ExpandProperty FullName) -Encoding UTF8

$summary = @()
$summary += "Base: $Base"
$summary += "Destination (canonical): $dest"
$summary += "Candidates: $($candidates.Count)"
$summary += "Manifest: $manifestPath"
$summary += "Conflicts: $conflictsPath (count: $($conflictLines.Count))"
$summary += "Unused minified: $unusedMinPath (count: $($minifiedUnused.Count))"
Set-Content -Path $summaryPath -Value $summary -Encoding UTF8

Write-Host 'Dry-run complete.'
Write-Host "Manifest: $manifestPath"
Write-Host "Conflicts: $conflictsPath"
Write-Host "Unused minified: $unusedMinPath"
Write-Host "Summary: $summaryPath"
```

### How to use the script
1. Save the script to a file, for example: `ui_migration_dryrun.ps1`.
2. Open PowerShell and run:
   - `powershell -ExecutionPolicy Bypass -File .\ui_migration_dryrun.ps1` (or pass `-Base` if your workspace path differs)
3. Review files generated in `_migration_reports`:
   - `ui_migration_manifest.txt` — every source file and its intended target path
   - `ui_migration_conflicts.txt` — subset that would overwrite existing files; require manual review
   - `unused_minified.txt` — minified assets not referenced by code; safe to omit
   - `summary.txt` — counts and quick links

### Post‑dry‑run next steps
- Approve or edit the manifest; resolve conflicts explicitly.
- Execute controlled copies based on the approved manifest.
- Rebuild/verify dashboards and charts; prune unreferenced assets.
- Update `README.md` with the canonical structure and migration notes.
- Commit and push changes after validation.


