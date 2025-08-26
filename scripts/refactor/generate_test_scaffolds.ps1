param(
    [string]$RefactorListPath = "tools\codebase_monitor\outputs\refactor_top100.json",
    [string]$TestOutputDir = "tests\refactor_targets",
    [int]$MaxTests = 100
)

# Load the refactor targets
if (-not (Test-Path $RefactorListPath)) {
    Write-Error "Refactor list not found at: $RefactorListPath"
    Write-Host "Please run: python .\tools\codebase_monitor\top_refactor_picker.py"
    exit 1
}

$targets = (Get-Content $RefactorListPath | ConvertFrom-Json).top100

# Ensure test output directory exists
New-Item -ItemType Directory -Force -Path $TestOutputDir | Out-Null

$generatedCount = 0
foreach ($relPath in $targets) {
    if ($generatedCount -ge $MaxTests) { break }
    
    # Skip if not a Python file
    if (-not ($relPath -match "\.py$")) { continue }
    
    # Create a safe filename for the test
    $safeName = $relPath -replace '[^A-Za-z0-9_]', '_'
    $safeName = $safeName -replace '__+', '_'  # Replace multiple underscores with single
    $safeName = $safeName -replace '^_|_$', '' # Remove leading/trailing underscores
    
    $testFileName = "test_$safeName.py"
    $testFilePath = Join-Path $TestOutputDir $testFileName
    
    # Extract module name from path for import
    $modulePath = $relPath -replace '\\', '/' -replace '\.py$', '' -replace '/', '.'
    $fileName = Split-Path $relPath -Leaf
    $fileNameNoExt = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
    
    # Generate test scaffold content
    $testContent = @"
"""
Test scaffold for refactoring target: $relPath

This test ensures that basic functionality is preserved during refactoring.
Add specific tests as you refactor the module.

Original file: $relPath
Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
"""
import os
import sys
import importlib
import pytest
from pathlib import Path

# Add the project root to Python path to enable imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_FILE = r"$relPath"
TARGET_MODULE = "$modulePath"

class TestRefactoringTarget:
    """Test class for refactoring target: $fileNameNoExt"""
    
    def test_file_exists(self):
        """Ensure the target file exists."""
        file_path = PROJECT_ROOT / TARGET_FILE
        assert file_path.exists(), f"Target file not found: {TARGET_FILE}"
    
    def test_file_not_empty(self):
        """Ensure the target file is not empty."""
        file_path = PROJECT_ROOT / TARGET_FILE
        assert file_path.stat().st_size > 0, f"Target file is empty: {TARGET_FILE}"
    
    def test_basic_import(self):
        """Test that the module can be imported without errors."""
        try:
            # Handle relative imports and different module structures
            if '.' in TARGET_MODULE:
                # Try importing as package
                module = importlib.import_module(TARGET_MODULE)
            else:
                # Try direct import
                spec = importlib.util.spec_from_file_location("target_module", PROJECT_ROOT / TARGET_FILE)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            
            assert module is not None, f"Failed to import module: {TARGET_MODULE}"
        except ImportError as e:
            pytest.skip(f"Module import failed (may have dependencies): {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing module: {e}")
    
    def test_python_syntax(self):
        """Ensure the file has valid Python syntax."""
        file_path = PROJECT_ROOT / TARGET_FILE
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(file_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {TARGET_FILE}: {e}")
    
    @pytest.mark.skipif(not Path(PROJECT_ROOT / TARGET_FILE).exists(), reason="Target file not found")
    def test_has_functions_or_classes(self):
        """Ensure the module defines functions or classes."""
        file_path = PROJECT_ROOT / TARGET_FILE
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_functions = 'def ' in content
        has_classes = 'class ' in content
        
        assert has_functions or has_classes, f"No functions or classes found in {TARGET_FILE}"

# Add specific tests as you refactor this module
# TODO: Add functional tests for key behavior
# TODO: Add performance tests if applicable
# TODO: Add integration tests for external dependencies
"@

    # Write the test file
    $testContent | Out-File -FilePath $testFilePath -Encoding utf8 -Force
    $generatedCount++
    
    Write-Host "Generated test scaffold: $testFileName" -ForegroundColor Green
}

Write-Host ""
Write-Host "Test scaffold generation complete!" -ForegroundColor Cyan
Write-Host "Generated $generatedCount test files in: $TestOutputDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the tests:" -ForegroundColor Yellow
Write-Host "  cd $TestOutputDir"
Write-Host "  pytest -v"
Write-Host ""
Write-Host "To run a specific test:" -ForegroundColor Yellow
Write-Host "  pytest -v test_specific_file.py"