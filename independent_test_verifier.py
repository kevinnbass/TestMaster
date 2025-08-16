#!/usr/bin/env python3
"""
Independent Test Verifier
Analyzes existing tests for completeness without modifying them.
Generates quality reports and improvement suggestions.
"""

import os
import ast
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, List, Tuple
import re

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

import google.generativeai as genai

# Configure API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def analyze_test_coverage(test_file: Path, module_file: Path) -> Dict:
    """Analyze test coverage and quality for a single module."""
    
    # Read test code
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_code = f.read()
    except:
        return {"error": "Cannot read test file"}
    
    # Read module code
    try:
        with open(module_file, 'r', encoding='utf-8') as f:
            module_code = f.read()
    except:
        return {"error": "Cannot read module file"}
    
    # Parse AST to extract functions/classes
    try:
        module_ast = ast.parse(module_code)
        module_functions = [node.name for node in ast.walk(module_ast) 
                           if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
        module_classes = [node.name for node in ast.walk(module_ast) 
                         if isinstance(node, ast.ClassDef)]
    except:
        module_functions = []
        module_classes = []
    
    # Analyze test coverage
    try:
        test_ast = ast.parse(test_code)
        test_functions = [node.name for node in ast.walk(test_ast) 
                         if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
    except:
        test_functions = []
    
    # Basic metrics
    metrics = {
        "module_functions": len(module_functions),
        "module_classes": len(module_classes),
        "test_functions": len(test_functions),
        "uses_mocks": "Mock" in test_code or "patch" in test_code,
        "has_fixtures": "@pytest.fixture" in test_code,
        "has_edge_cases": any(keyword in test_code.lower() for keyword in 
                              ['edge', 'boundary', 'empty', 'null', 'invalid', 'error']),
        "has_error_handling": "raises" in test_code or "assertRaises" in test_code
    }
    
    # Calculate basic score
    coverage_ratio = metrics["test_functions"] / max(metrics["module_functions"] + metrics["module_classes"], 1)
    mock_penalty = -20 if metrics["uses_mocks"] else 0
    edge_bonus = 10 if metrics["has_edge_cases"] else 0
    error_bonus = 10 if metrics["has_error_handling"] else 0
    
    metrics["basic_score"] = min(100, max(0, coverage_ratio * 70 + mock_penalty + edge_bonus + error_bonus))
    
    return metrics

def verify_with_llm(test_file: Path, module_file: Path, basic_metrics: Dict) -> Dict:
    """Use LLM to provide detailed verification."""
    
    # Read files
    with open(test_file, 'r', encoding='utf-8') as f:
        test_code = f.read()[:4000]  # Truncate for API
    with open(module_file, 'r', encoding='utf-8') as f:
        module_code = f.read()[:4000]
    
    prompt = f"""Analyze this test file for quality and completeness.

MODULE CODE:
```python
{module_code}
```

TEST CODE:
```python
{test_code}
```

BASIC METRICS:
- Functions in module: {basic_metrics['module_functions']}
- Classes in module: {basic_metrics['module_classes']}
- Test functions: {basic_metrics['test_functions']}
- Uses mocks: {basic_metrics['uses_mocks']}
- Has edge cases: {basic_metrics['has_edge_cases']}

Provide analysis in this format:
QUALITY_SCORE: [0-100]
COVERAGE_GAPS: [list missing test scenarios]
IMPROVEMENT_PRIORITY: [HIGH/MEDIUM/LOW]
TOP_3_IMPROVEMENTS:
1. [specific improvement]
2. [specific improvement]
3. [specific improvement]
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return {"llm_analysis": response.text, "status": "success"}
    except Exception as e:
        return {"llm_analysis": None, "status": f"LLM error: {e}"}

def verify_test_suite(test_dir: Path = Path("tests/unit"), 
                      module_dir: Path = Path("multi_coder_analysis"),
                      limit: int = 10) -> Dict:
    """Verify entire test suite quality."""
    
    results = []
    
    # Get all test files
    test_files = list(test_dir.glob("test_*_intelligent.py"))[:limit]
    
    print(f"Analyzing {len(test_files)} test files...")
    print("="*60)
    
    for test_file in test_files:
        # Find corresponding module
        module_name = test_file.stem.replace("test_", "").replace("_intelligent", "")
        
        # Search for module file
        module_file = None
        for py_file in module_dir.rglob(f"{module_name}.py"):
            if not py_file.name.startswith("test"):
                module_file = py_file
                break
        
        if not module_file:
            print(f"[SKIP] {module_name}: Module not found")
            continue
        
        # Analyze
        print(f"[ANALYZE] {module_name}...")
        basic_metrics = analyze_test_coverage(test_file, module_file)
        
        # Get LLM verification (optional, comment out to save API calls)
        # llm_result = verify_with_llm(test_file, module_file, basic_metrics)
        # basic_metrics.update(llm_result)
        
        results.append({
            "module": module_name,
            "test_file": str(test_file),
            "module_file": str(module_file),
            "metrics": basic_metrics
        })
        
        # Print summary
        score = basic_metrics.get("basic_score", 0)
        status = "GOOD" if score >= 70 else "NEEDS WORK" if score >= 40 else "POOR"
        print(f"  Score: {score:.0f}/100 - {status}")
    
    # Summary statistics
    scores = [r["metrics"].get("basic_score", 0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Files analyzed: {len(results)}")
    print(f"Average score: {avg_score:.1f}/100")
    print(f"Using mocks: {sum(1 for r in results if r['metrics'].get('uses_mocks'))} files")
    print(f"With edge cases: {sum(1 for r in results if r['metrics'].get('has_edge_cases'))} files")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "files_analyzed": len(results),
        "average_score": avg_score,
        "results": results
    }
    
    with open("test_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: test_verification_report.json")
    
    return report

if __name__ == "__main__":
    verify_test_suite()