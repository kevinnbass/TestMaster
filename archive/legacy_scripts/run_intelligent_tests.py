#!/usr/bin/env python3
"""
Run all intelligent tests and generate a comprehensive report.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import time

def run_single_test(test_file: Path) -> dict:
    """Run a single test file and capture results."""
    
    print(f"Testing: {test_file.name}...", end=" ")
    
    # Run pytest on the file
    cmd = [
        sys.executable, "-m", "pytest", 
        str(test_file),
        "-v",
        "--tb=short",
        "--json-report",
        "--json-report-file=test_result.json",
        "-q"
    ]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="tests/unit"
    )
    duration = time.time() - start_time
    
    # Parse results
    test_result = {
        "file": test_file.name,
        "exit_code": result.returncode,
        "duration": duration,
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "total": 0,
        "success": False,
        "error_summary": ""
    }
    
    # Try to parse json report if it exists
    json_report = Path("tests/unit/test_result.json")
    if json_report.exists():
        try:
            with open(json_report, 'r') as f:
                report_data = json.load(f)
                summary = report_data.get("summary", {})
                test_result["passed"] = summary.get("passed", 0)
                test_result["failed"] = summary.get("failed", 0)
                test_result["error"] = summary.get("error", 0)
                test_result["skipped"] = summary.get("skipped", 0)
                test_result["total"] = summary.get("total", 0)
        except:
            pass
        json_report.unlink()  # Clean up
    
    # Parse stdout for basic metrics if json report failed
    if test_result["total"] == 0:
        stdout = result.stdout
        if "passed" in stdout:
            # Try to extract metrics from output like "5 passed in 0.50s"
            import re
            passed_match = re.search(r'(\d+) passed', stdout)
            failed_match = re.search(r'(\d+) failed', stdout)
            error_match = re.search(r'(\d+) error', stdout)
            skipped_match = re.search(r'(\d+) skipped', stdout)
            
            if passed_match:
                test_result["passed"] = int(passed_match.group(1))
            if failed_match:
                test_result["failed"] = int(failed_match.group(1))
            if error_match:
                test_result["error"] = int(error_match.group(1))
            if skipped_match:
                test_result["skipped"] = int(skipped_match.group(1))
            
            test_result["total"] = (
                test_result["passed"] + 
                test_result["failed"] + 
                test_result["error"] + 
                test_result["skipped"]
            )
    
    # Determine success
    test_result["success"] = (
        result.returncode == 0 or 
        (test_result["passed"] > 0 and test_result["error"] == 0)
    )
    
    # Capture error summary if failed
    if not test_result["success"] and result.stderr:
        lines = result.stderr.split('\n')
        for line in lines:
            if 'ModuleNotFoundError' in line or 'ImportError' in line:
                test_result["error_summary"] = line.strip()
                break
        if not test_result["error_summary"] and len(lines) > 0:
            test_result["error_summary"] = lines[0][:100]
    
    # Print status
    if test_result["success"]:
        status = f"[PASS] {test_result['passed']}/{test_result['total']} tests"
    elif test_result["error"] > 0:
        status = f"[ERROR] Import/syntax error"
    elif test_result["failed"] > 0:
        status = f"[FAIL] {test_result['failed']} failures"
    else:
        status = "[SKIP] No tests collected"
    
    print(status)
    
    return test_result

def run_all_intelligent_tests():
    """Run all intelligent test files and generate report."""
    
    print("="*70)
    print("INTELLIGENT TEST SUITE RUNNER")
    print("="*70)
    
    # Find all intelligent test files
    test_dir = Path("tests/unit")
    intelligent_tests = sorted(test_dir.glob("*_intelligent.py"))
    
    print(f"Found {len(intelligent_tests)} intelligent test files")
    print("-"*70)
    
    # Run each test
    results = []
    for test_file in intelligent_tests:
        result = run_single_test(test_file)
        results.append(result)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    total_files = len(results)
    successful_files = sum(1 for r in results if r["success"])
    error_files = sum(1 for r in results if r["error"] > 0)
    failed_files = sum(1 for r in results if r["failed"] > 0 and r["error"] == 0)
    
    total_tests = sum(r["total"] for r in results)
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_errors = sum(r["error"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    
    print(f"\nFile Statistics:")
    print(f"  Total Files: {total_files}")
    print(f"  Successful: {successful_files} ({successful_files/total_files*100:.1f}%)")
    print(f"  With Errors: {error_files}")
    print(f"  With Failures: {failed_files}")
    
    print(f"\nTest Statistics:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_passed} ({total_passed/total_tests*100:.1f}% if total_tests > 0 else 0)")
    print(f"  Failed: {total_failed}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    
    # List problematic files
    if error_files > 0:
        print(f"\nFiles with Import/Syntax Errors:")
        for r in results:
            if r["error"] > 0:
                print(f"  - {r['file']}: {r['error_summary'][:80] if r['error_summary'] else 'Unknown error'}")
    
    if failed_files > 0:
        print(f"\nFiles with Test Failures:")
        for r in results:
            if r["failed"] > 0 and r["error"] == 0:
                print(f"  - {r['file']}: {r['failed']} failures")
    
    # Generate detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total_files,
            "successful_files": successful_files,
            "error_files": error_files,
            "failed_files": failed_files,
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_failed,
            "error_tests": total_errors,
            "skipped_tests": total_skipped,
            "success_rate": successful_files/total_files*100 if total_files > 0 else 0,
            "test_pass_rate": total_passed/total_tests*100 if total_tests > 0 else 0
        },
        "details": results
    }
    
    # Save report
    report_path = Path("intelligent_test_results.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Generate markdown report
    generate_markdown_report(report)
    
    return report

def generate_markdown_report(report: dict):
    """Generate a markdown report from test results."""
    
    md_content = f"""# Intelligent Test Suite Results

**Generated**: {report['timestamp']}

## Summary

### File Statistics
- **Total Files**: {report['summary']['total_files']}
- **Successful**: {report['summary']['successful_files']} ({report['summary']['success_rate']:.1f}%)
- **With Errors**: {report['summary']['error_files']}
- **With Failures**: {report['summary']['failed_files']}

### Test Statistics
- **Total Tests**: {report['summary']['total_tests']}
- **Passed**: {report['summary']['passed_tests']} ({report['summary']['test_pass_rate']:.1f}%)
- **Failed**: {report['summary']['failed_tests']}
- **Errors**: {report['summary']['error_tests']}
- **Skipped**: {report['summary']['skipped_tests']}

## Detailed Results

| Test File | Status | Tests | Passed | Failed | Errors | Duration |
|-----------|--------|-------|--------|--------|--------|----------|
"""
    
    for result in report['details']:
        status = "✅" if result['success'] else "❌"
        md_content += f"| {result['file']} | {status} | {result['total']} | {result['passed']} | {result['failed']} | {result['error']} | {result['duration']:.2f}s |\n"
    
    # Add error details if any
    error_files = [r for r in report['details'] if r['error'] > 0]
    if error_files:
        md_content += "\n## Import/Syntax Errors\n\n"
        for r in error_files:
            md_content += f"- **{r['file']}**: {r['error_summary']}\n"
    
    md_path = Path("INTELLIGENT_TEST_RESULTS.md")
    md_path.write_text(md_content)
    print(f"Markdown report saved to: {md_path}")

if __name__ == "__main__":
    report = run_all_intelligent_tests()
    
    # Exit with appropriate code
    if report['summary']['error_files'] > 0:
        sys.exit(1)
    elif report['summary']['failed_files'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)