# Refactor Campaign System (Agent Delta)

Complete refactoring workflow system for systematic code improvement.

## Overview

This system provides a data-driven approach to code refactoring by:
1. Identifying the top 100 files needing refactoring based on analyzer metrics
2. Generating test scaffolds to prevent regressions
3. Providing tools to measure improvement over time

## Files Created

### Core Scripts
- `tools\codebase_monitor\top_refactor_picker.py` - Generates top 100 refactoring targets
- `scripts\refactor\generate_test_scaffolds.ps1` - Creates pytest scaffolds for target files
- `scripts\refactor\run_tests.ps1` - Runs all refactor tests with reporting
- `scripts\refactor\measure_improvement.ps1` - Tracks refactoring progress over time

### Generated Outputs
- `tools\codebase_monitor\outputs\refactor_top100.json` - Top 100 files to refactor
- `tests\refactor_targets\test_*.py` - 100 test scaffold files
- `tools\codebase_monitor\outputs\test_results.json` - Latest test results
- `tools\codebase_monitor\outputs\improvement_report.json` - Progress tracking

## Usage Workflow

### 1. Generate Refactoring Targets
```powershell
python .\tools\codebase_monitor\top_refactor_picker.py
```

### 2. Create Test Scaffolds
```powershell
.\scripts\refactor\generate_test_scaffolds.ps1
```

### 3. Run Initial Tests
```powershell
.\scripts\refactor\run_tests.ps1 -Verbose
```

### 4. Set Baseline for Measurement
```powershell
.\scripts\refactor\measure_improvement.ps1 -SetBaseline
```

### 5. Refactor and Measure Progress
After making changes to target files:
```powershell
# Run tests to ensure no regressions
.\scripts\refactor\run_tests.ps1

# Measure improvement
.\scripts\refactor\measure_improvement.ps1
```

## Test Scaffold Features

Each generated test includes:
- **File existence check** - Ensures target file exists
- **Import validation** - Tests that module can be imported
- **Syntax validation** - Ensures valid Python syntax
- **Content validation** - Checks for functions/classes
- **Extensibility** - Placeholders for functional tests

## Improvement Tracking

The measurement system tracks:
- Total file/line count changes
- Duplicate group reduction
- Hotspot category improvements
- Overall problem score
- Recommendations for next steps

## Scoring Algorithm

Files are prioritized based on:
- Branch complexity (py_branch_nodes × 3)
- Average function length (py_avg_function_len × 2)
- Large files (1000+ lines: +50 points, 500+ lines: +25 points)
- Mixed indentation (+10 points)
- TODO count (num_todos × 2)
- Hotspot frequency (appears in multiple categories)

## Integration with Other Agents

- Uses output from the codebase analyzer
- Coordinates with Alpha agent's cleanup scripts
- Provides data for Beta's service endpoints
- Supports Gamma's dashboard visualization

## Example Output

```
Top file: TestMaster/archive/20250818/web_monitor.py (score: 40)
Total candidates considered: 2078
Generated 100 test files in: tests\refactor_targets

Test Results:
Tests Run: 100
Passed: 85
Failed: 10
Skipped: 5
Success Rate: 85.0%

Improvement Analysis:
Duplicate Groups: 150 → 120 (-30, -20%)
Files in Duplicates: 450 → 360 (-90, -20%)
Overall Score Improvement: 15.5%
```

## Best Practices

1. **Set baseline before starting** - Use `-SetBaseline` on your first run
2. **Run tests frequently** - After each significant refactor
3. **Focus on high-score files** - Start with files scoring 30+ points
4. **Track progress regularly** - Weekly measurement runs
5. **Update test scaffolds** - Add functional tests as you refactor

## Troubleshooting

### Import Errors in Tests
- Check Python path configuration
- Verify target files have valid syntax
- Add dependencies to project requirements

### No Improvement Detected
- Ensure you're working on files from the top 100 list
- Verify analyzer is detecting changes
- Check that duplicates are actually being removed

### Performance Issues
- Use `-Workers` parameter for parallel test execution
- Consider filtering tests with `-Pattern` parameter
- Skip time-consuming tests during development

## Future Enhancements

- Integration with git hooks for automatic measurement
- Web dashboard for progress visualization
- Machine learning for refactoring priority prediction
- Automated refactoring suggestions
- Integration with code quality metrics