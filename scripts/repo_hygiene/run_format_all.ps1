# Code formatting script for repository hygiene
param(
    [string]$Path = ".",
    [switch]$PythonOnly,
    [switch]$JSOnly,
    [switch]$DryRun
)

Write-Host "Code Formatting Script"
Write-Host "Path: $Path"
Write-Host "Dry Run: $($DryRun.IsPresent)"

$pythonFiles = @()
$jsFiles = @()
$errors = @()

# Check for Python files
if (-not $JSOnly) {
    Write-Host ""
    Write-Host "=== PYTHON FORMATTING ==="
    
    $pythonFiles = Get-ChildItem -Path $Path -Recurse -Filter "*.py" | Where-Object {
        $_.FullName -notlike "*\.venv\*" -and
        $_.FullName -notlike "*\__pycache__\*" -and
        $_.FullName -notlike "*\build\*" -and
        $_.FullName -notlike "*\dist\*"
    }
    
    Write-Host "Found $($pythonFiles.Count) Python files"
    
    if ($pythonFiles.Count -gt 0) {
        # Install/check Python formatting tools
        Write-Host "Checking Python formatting tools..."
        
        $tools = @("black", "isort", "ruff")
        foreach ($tool in $tools) {
            try {
                if ($DryRun) {
                    Write-Host "[DRYRUN] Would check: python -m pip show $tool" -ForegroundColor Cyan
                } else {
                    python -m pip show $tool >$null 2>&1
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "Installing $tool..."
                        python -m pip install $tool
                    }
                }
            } catch {
                $errors += "Failed to install/check $tool`: $_"
            }
        }
        
        # Run Black formatter
        try {
            if ($DryRun) {
                Write-Host "[DRYRUN] Would run: black --check $Path" -ForegroundColor Cyan
            } else {
                Write-Host "Running Black formatter..."
                python -m black $Path
                Write-Host "✓ Black formatting complete"
            }
        } catch {
            $errors += "Black formatting failed: $_"
        }
        
        # Run isort for imports
        try {
            if ($DryRun) {
                Write-Host "[DRYRUN] Would run: isort --check-only $Path" -ForegroundColor Cyan
            } else {
                Write-Host "Running isort for import sorting..."
                python -m isort $Path
                Write-Host "✓ Import sorting complete"
            }
        } catch {
            $errors += "isort failed: $_"
        }
        
        # Run ruff for linting/fixing
        try {
            if ($DryRun) {
                Write-Host "[DRYRUN] Would run: ruff check $Path" -ForegroundColor Cyan
            } else {
                Write-Host "Running ruff linter with fixes..."
                python -m ruff check $Path --fix --silent
                Write-Host "✓ Ruff linting complete"
            }
        } catch {
            $errors += "Ruff linting failed: $_"
        }
    }
}

# Check for JavaScript/TypeScript files
if (-not $PythonOnly) {
    Write-Host ""
    Write-Host "=== JAVASCRIPT/TYPESCRIPT FORMATTING ==="
    
    $jsFiles = Get-ChildItem -Path $Path -Recurse | Where-Object {
        ($_.Extension -eq ".js" -or $_.Extension -eq ".jsx" -or 
         $_.Extension -eq ".ts" -or $_.Extension -eq ".tsx" -or
         $_.Extension -eq ".vue") -and
        $_.FullName -notlike "*\node_modules\*" -and
        $_.FullName -notlike "*\dist\*" -and
        $_.FullName -notlike "*\build\*"
    }
    
    Write-Host "Found $($jsFiles.Count) JS/TS files"
    
    if ($jsFiles.Count -gt 0) {
        # Check if we're in a Node.js project
        $hasPackageJson = Test-Path (Join-Path $Path "package.json")
        
        if ($hasPackageJson) {
            # Try to use project's Prettier/ESLint
            if ($DryRun) {
                Write-Host "[DRYRUN] Would run: npx prettier --check ." -ForegroundColor Cyan
                Write-Host "[DRYRUN] Would run: npx eslint . --fix" -ForegroundColor Cyan
            } else {
                try {
                    Write-Host "Running Prettier..."
                    npx prettier --write . 2>$null
                    Write-Host "✓ Prettier formatting complete"
                } catch {
                    Write-Warning "Prettier not available or failed: $_"
                }
                
                try {
                    Write-Host "Running ESLint..."
                    npx eslint . --fix --silent 2>$null
                    Write-Host "✓ ESLint fixing complete"
                } catch {
                    Write-Warning "ESLint not available or failed: $_"
                }
            }
        } else {
            Write-Host "No package.json found - skipping JS/TS formatting"
        }
    }
}

Write-Host ""
Write-Host "=== SUMMARY ==="
Write-Host "Python files processed: $($pythonFiles.Count)"
Write-Host "JS/TS files found: $($jsFiles.Count)"

if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  ERRORS ENCOUNTERED:" -ForegroundColor Yellow
    foreach ($error in $errors) {
        Write-Host "  • $error" -ForegroundColor Red
    }
} else {
    Write-Host "✓ Formatting completed successfully" -ForegroundColor Green
}

if ($DryRun) {
    Write-Host ""
    Write-Host "This was a dry run. Remove -DryRun to execute formatting."
}