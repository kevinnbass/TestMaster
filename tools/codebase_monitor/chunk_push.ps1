Param(
    [string]$SourceRoot = (Resolve-Path "$PSScriptRoot\..\.."),
    [string]$Branch = "chunk-master",
    [string]$Remote = "origin",
    [string]$StartRef = "origin/master",
    [string]$WorktreesDir = "_worktrees",
    [string]$LogDir = "$PSScriptRoot\reports",
    [switch]$ForceData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Function Write-Log([string]$message) {
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $message"
    $existing = Get-Variable -Name 'LogFile' -Scope Script -ErrorAction SilentlyContinue
    if (-not $existing) {
        $Script:LogFile = Join-Path $LogDir ("chunk_push_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
    }
    Add-Content -Path $Script:LogFile -Value $line
    Write-Host $line
}

Function Invoke-Git($argsArray) {
    # Use Start-Process with redirected streams to avoid NativeCommandError noise
    $joined = ($argsArray -join ' ')
    $tmpDir = Join-Path $env:TEMP ("chunk_push_" + [guid]::NewGuid().ToString())
    New-Item -ItemType Directory -Path $tmpDir | Out-Null
    $outFile = Join-Path $tmpDir "stdout.txt"
    $errFile = Join-Path $tmpDir "stderr.txt"
    $proc = Start-Process -FilePath "git" -ArgumentList $argsArray -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile -Wait
    $code = $proc.ExitCode
    $out = if (Test-Path $outFile) { Get-Content $outFile -Raw } else { "" }
    $err = if (Test-Path $errFile) { Get-Content $errFile -Raw } else { "" }
    if ($code -ne 0) {
        Write-Log ("git $joined FAILED [$code]")
        if ($out) { Write-Log ("STDOUT:`n" + $out) }
        if ($err) { Write-Log ("STDERR:`n" + $err) }
        throw "git command failed"
    }
    if ($out) { Write-Log $out.TrimEnd() }
    if ($err) { Write-Log $err.TrimEnd() }
    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
    return $out
}

Function Ensure-Worktree() {
    Push-Location $SourceRoot
    try {
        git config core.longpaths true | Out-Null
        if (-not (Test-Path $WorktreesDir)) { New-Item -ItemType Directory -Path $WorktreesDir | Out-Null }
        $worktreePath = Join-Path $WorktreesDir $Branch
        if (-not (Test-Path $worktreePath)) {
            Write-Log "Creating worktree $worktreePath from $StartRef"
            Invoke-Git @("worktree","add","-B",$Branch,$worktreePath,$StartRef) | Out-Null
        } else {
            Write-Log "Worktree already exists: $worktreePath"
        }
        return (Resolve-Path $worktreePath)
    } finally {
        Pop-Location
    }
}

Function Test-HasLargeFiles([string]$path, [int]$mbThreshold = 95) {
    if (-not (Test-Path $path)) { return $false }
    $big = Get-ChildItem -Path $path -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt ($mbThreshold * 1MB) }
    if ($big) {
        foreach ($b in $big) { Write-Log ("SKIP (big file >" + $mbThreshold + "MB): " + $b.FullName + " [" + [math]::Round($b.Length/1MB,2) + " MB]") }
        return $true
    }
    return $false
}

Function Copy-And-CommitChunk([string]$srcRelative, [string]$worktreePath, [switch]$ForceAdd) {
    $srcPath = Join-Path $SourceRoot $srcRelative
    if (-not (Test-Path $srcPath)) { Write-Log "Missing: $srcRelative (skip)"; return $false }
    if (Test-HasLargeFiles $srcPath) { return $false }
    $dstPath = Join-Path $worktreePath $srcRelative
    $dstDir = Split-Path $dstPath -Parent
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    Write-Log "Copy: $srcRelative"
    Copy-Item -Path $srcPath -Destination $dstPath -Recurse -Force -ErrorAction SilentlyContinue
    Push-Location $worktreePath
    try {
        # If this path (or an ancestor) is ignored in the worktree, skip early
        if (Test-PathOrAncestorIgnored -worktreePath $worktreePath -relPath $srcRelative) {
            Write-Log ("Skip ignored: " + $srcRelative)
            return $false
        }
        $addArgs = @("add")
        if ($ForceAdd) { $addArgs += "-f" }
        $addArgs += $srcRelative
        Invoke-Git $addArgs | Out-Null
        $files = (git diff --cached --name-only | Measure-Object).Count
        if ($files -eq 0) { Write-Log "No staged files for $srcRelative (skip commit)"; return $false }
        $msg = "Chunk: add " + $srcRelative
        Invoke-Git @("commit","-m","`"$msg`"") | Out-Null
        Invoke-Git @("push",$Remote,$Branch) | Out-Null
        Write-Log ("PUSH OK: " + $srcRelative)
        return $true
    } finally {
        Pop-Location
    }
}

Function Get-ImmediateSubdirs([string]$relative) {
    $full = Join-Path $SourceRoot $relative
    if (-not (Test-Path $full)) { return @() }
    return (Get-ChildItem -Path $full -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $relative $_.Name })
}

Function Test-GitIgnored([string]$worktreePath, [string]$relPath) {
    # quick basename filter for very common ignored folders
    $base = Split-Path $relPath -Leaf
    if ($base -in @("__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", "dist", "build", "out", "coverage")) {
        return $true
    }
    Push-Location $worktreePath
    try {
        $relPathPosix = $relPath -replace "\\", "/"
        $null = & git check-ignore -q -- $relPathPosix 2>$null
        $code = $LASTEXITCODE
        if ($code -eq 0) { return $true }
        # Try with trailing slash and trailing "." to catch directory-only patterns
        $null = & git check-ignore -q -- ($relPathPosix.TrimEnd('/') + '/') 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
        $null = & git check-ignore -q -- ($relPathPosix.TrimEnd('/') + '/.') 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
        return $false
    } finally {
        Pop-Location
    }
}

Function Test-PathOrAncestorIgnored([string]$worktreePath, [string]$relPath) {
    # Check if the path or any of its ancestors are ignored
    $posix = ($relPath -replace "\\", "/").Trim('/')
    $segments = $posix.Split('/')
    for ($i = 0; $i -lt $segments.Count; $i++) {
        $prefix = ($segments[0..$i] -join '/')
        if (Test-GitIgnored -worktreePath $worktreePath -relPath $prefix) { return $true }
    }
    return $false
}

Function Push-Category([string]$relative, [string]$worktreePath, [switch]$ForceAdd) {
    # Skip whole category if ignored
    if (Test-PathOrAncestorIgnored -worktreePath $worktreePath -relPath $relative) {
        Write-Log ("Skip ignored: " + $relative)
        return
    }
    # Try small subdirs first; if none, push the directory itself
    $subs = @(Get-ImmediateSubdirs $relative)
    # Filter out .gitignored subdirs
    $validSubs = @()
    foreach ($s in $subs) {
        if (Test-PathOrAncestorIgnored -worktreePath $worktreePath -relPath $s) {
            Write-Log ("Skip ignored: " + $s)
        } else {
            $validSubs += $s
        }
    }
    if ($validSubs.Count -eq 0) {
        if (Test-GitIgnored -worktreePath $worktreePath -relPath $relative) {
            Write-Log ("Skip ignored: " + $relative)
            return
        }
        [void](Copy-And-CommitChunk -srcRelative $relative -worktreePath $worktreePath -ForceAdd:$ForceAdd)
        return
    }
    foreach ($s in $validSubs) {
        [void](Copy-And-CommitChunk -srcRelative $s -worktreePath $worktreePath -ForceAdd:$ForceAdd)
    }
}

# Main
Write-Log "=== Chunk Push Automation Start ==="
$worktreePath = Ensure-Worktree

$small = @(
    "organized_codebase/assets",
    "organized_codebase/scripts",
    "organized_codebase/documentation",
    "organized_codebase/sandbox",
    "organized_codebase/legacy",
    "organized_codebase/configuration",
    "organized_codebase/utilities"
)

foreach ($rel in $small) {
    Push-Category -relative $rel -worktreePath $worktreePath
}

# Data (ignored by .gitignore) – force add on request
if ($ForceData) {
    Push-Category -relative "organized_codebase/data" -worktreePath $worktreePath -ForceAdd
}

# Larger categories – push immediate subfolders as tiny chunks
$large = @(
    "organized_codebase/backups",
    "organized_codebase/testing",
    "organized_codebase/frontend",
    "organized_codebase/core"
)

foreach ($rel in $large) {
    Push-Category -relative $rel -worktreePath $worktreePath
}

Write-Log "=== Chunk Push Automation Complete ==="


