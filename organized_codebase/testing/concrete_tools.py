from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Concrete Tool Implementations
=======================================

Concrete implementations of type-safe tools for various TestMaster operations.
Demonstrates Agency-Swarm patterns with full type safety.

Author: TestMaster Team
"""

import os
import subprocess
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .type_safe_tools import (
    TypeSafeTool, ToolMetadata, ToolCategory, ToolInput, ToolOutput,
    TestExecutionInput, TestExecutionOutput, CoverageAnalysisInput, CoverageAnalysisOutput,
    ValidationLevel, register_tool
)

class PytestExecutionTool(TypeSafeTool[TestExecutionInput, TestExecutionOutput]):
    """Type-safe tool for executing pytest with comprehensive validation"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
    
    async def execute(self, input_data: TestExecutionInput) -> TestExecutionOutput:
        """Execute pytest with the provided configuration"""
        start_time = time.time()
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        if input_data.coverage_enabled:
            cmd.extend(["--cov=.", "--cov-report=json"])
        
        if input_data.test_pattern:
            cmd.extend(["-k", input_data.test_pattern])
        
        if input_data.parallel_workers > 1:
            cmd.extend(["-n", str(input_data.parallel_workers)])
        
        cmd.extend(["-v", "--json-report", "--json-report-file=test_results.json"])
        cmd.append(input_data.test_path)
        
        # Execute pytest
        try:
            process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse results
            test_results = []
            coverage_percentage = None
            
            try:
                # Load test results from JSON report
                if os.path.exists("test_results.json"):
                    with open("test_results.json", "r") as f:
                        json_report = json.load(f)
                        test_results = json_report.get("tests", [])
                
                # Load coverage data
                if os.path.exists(".coverage") and input_data.coverage_enabled:
                    coverage_cmd = ["python", "-m", "coverage", "json"]
                    coverage_process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                        *coverage_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await coverage_process.communicate()
                    
                    if os.path.exists("coverage.json"):
                        with open("coverage.json", "r") as f:
                            coverage_data = json.load(f)
                            coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                            
            except Exception as e:
                self.logger.warning(f"Failed to parse test results: {e}")
            
            # Count test results
            tests_passed = sum(1 for test in test_results if test.get("outcome") == "passed")
            tests_failed = sum(1 for test in test_results if test.get("outcome") == "failed")
            tests_run = len(test_results)
            
            return TestExecutionOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="completed",
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                coverage_percentage=coverage_percentage,
                test_results=test_results,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestExecutionOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="failed",
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def validate_input(self, input_data: Any) -> TestExecutionInput:
        """Validate input data for pytest execution"""
        if isinstance(input_data, TestExecutionInput):
            return input_data
        
        if isinstance(input_data, dict):
            return TestExecutionInput(**input_data)
        
        raise ValueError(f"Invalid input type: {type(input_data)}")
    
    def validate_output(self, output_data: Any) -> TestExecutionOutput:
        """Validate output data from pytest execution"""
        if isinstance(output_data, TestExecutionOutput):
            return output_data
        
        if isinstance(output_data, dict):
            return TestExecutionOutput(**output_data)
        
        raise ValueError(f"Invalid output type: {type(output_data)}")

class CoverageAnalysisTool(TypeSafeTool[CoverageAnalysisInput, CoverageAnalysisOutput]):
    """Type-safe tool for analyzing code coverage"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
    
    async def execute(self, input_data: CoverageAnalysisInput) -> CoverageAnalysisOutput:
        """Analyze code coverage for the provided source"""
        start_time = time.time()
        
        try:
            # Generate coverage report
            cmd = ["python", "-m", "coverage", "json", "--pretty-print"]
            
            if input_data.include_patterns:
                for pattern in input_data.include_patterns:
                    cmd.extend(["--include", pattern])
            
            if input_data.exclude_patterns:
                for pattern in input_data.exclude_patterns:
                    cmd.extend(["--omit", pattern])
            
            process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=input_data.source_path
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse coverage data
            coverage_file = "coverage.json"
            if input_data.coverage_data_path:
                coverage_file = input_data.coverage_data_path
            
            file_coverage = {}
            missing_lines = {}
            overall_coverage = 0.0
            line_coverage = 0.0
            
            if os.path.exists(coverage_file):
                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)
                    
                    # Extract overall metrics
                    totals = coverage_data.get("totals", {})
                    overall_coverage = totals.get("percent_covered", 0.0)
                    line_coverage = totals.get("percent_covered_display", 0.0)
                    
                    # Extract file-level coverage
                    files = coverage_data.get("files", {})
                    for file_path, file_data in files.items():
                        file_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0.0)
                        missing_lines[file_path] = file_data.get("missing_lines", [])
            
            return CoverageAnalysisOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="completed",
                overall_coverage=overall_coverage,
                line_coverage=line_coverage,
                file_coverage=file_coverage,
                missing_lines=missing_lines,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return CoverageAnalysisOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="failed",
                error=str(e),
                overall_coverage=0.0,
                line_coverage=0.0,
                execution_time=time.time() - start_time
            )
    
    def validate_input(self, input_data: Any) -> CoverageAnalysisInput:
        """Validate input data for coverage analysis"""
        if isinstance(input_data, CoverageAnalysisInput):
            return input_data
        
        if isinstance(input_data, dict):
            return CoverageAnalysisInput(**input_data)
        
        raise ValueError(f"Invalid input type: {type(input_data)}")
    
    def validate_output(self, output_data: Any) -> CoverageAnalysisOutput:
        """Validate output data from coverage analysis"""
        if isinstance(output_data, CoverageAnalysisOutput):
            return output_data
        
        if isinstance(output_data, dict):
            return CoverageAnalysisOutput(**output_data)
        
        raise ValueError(f"Invalid output type: {type(output_data)}")

class CodeQualityInput(ToolInput):
    """Input model for code quality analysis"""
    source_path: str
    quality_checks: List[str] = ["pylint", "flake8", "mypy"]
    fail_threshold: float = 8.0

class CodeQualityOutput(ToolOutput):
    """Output model for code quality analysis"""
    quality_score: float = 0.0
    quality_issues: List[Dict[str, Any]] = []
    check_results: Dict[str, Any] = {}

class CodeQualityAnalysisTool(TypeSafeTool[CodeQualityInput, CodeQualityOutput]):
    """Type-safe tool for code quality analysis"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
    
    async def execute(self, input_data: CodeQualityInput) -> CodeQualityOutput:
        """Analyze code quality using various tools"""
        start_time = time.time()
        
        try:
            check_results = {}
            quality_issues = []
            quality_score = 10.0  # Start with perfect score
            
            for check_tool in input_data.quality_checks:
                if check_tool == "pylint":
                    result = await self._run_pylint(input_data.source_path)
                elif check_tool == "flake8":
                    result = await self._run_flake8(input_data.source_path)
                elif check_tool == "mypy":
                    result = await self._run_mypy(input_data.source_path)
                else:
                    continue
                
                check_results[check_tool] = result
                quality_issues.extend(result.get("issues", []))
                
                # Adjust quality score based on issues
                if result.get("score") is not None:
                    quality_score = min(quality_score, result["score"])
            
            return CodeQualityOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="completed",
                quality_score=quality_score,
                quality_issues=quality_issues,
                check_results=check_results,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return CodeQualityOutput(
                tool_id=self.metadata.name,
                execution_id=input_data.execution_id,
                status="failed",
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _run_pylint(self, source_path: str) -> Dict[str, Any]:
        """Run pylint analysis"""
        try:
            cmd = ["python", "-m", "pylint", "--output-format=json", source_path]
            process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            issues = []
            if stdout:
                try:
                    pylint_results = json.loads(stdout.decode())
                    issues = [
                        {
                            "tool": "pylint",
                            "type": issue.get("type"),
                            "message": issue.get("message"),
                            "line": issue.get("line"),
                            "file": issue.get("path")
                        }
                        for issue in pylint_results
                    ]
                except json.JSONDecodeError:
                    pass
            
            # Calculate score (pylint default is 0-10)
            score = max(0, 10 - len(issues) * 0.1)
            
            return {"score": score, "issues": issues}
            
        except Exception as e:
            return {"score": 0.0, "issues": [{"tool": "pylint", "error": str(e)}]}
    
    async def _run_flake8(self, source_path: str) -> Dict[str, Any]:
        """Run flake8 analysis"""
        try:
            cmd = ["python", "-m", "flake8", "--format=json", source_path]
            process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            issues = []
            if stderr:  # flake8 outputs to stderr
                for line in stderr.decode().split('\n'):
                    if line.strip():
                        parts = line.split(':')
                        if len(parts) >= 4:
                            issues.append({
                                "tool": "flake8",
                                "file": parts[0],
                                "line": parts[1],
                                "column": parts[2],
                                "message": ':'.join(parts[3:]).strip()
                            })
            
            # Calculate score based on issues
            score = max(0, 10 - len(issues) * 0.2)
            
            return {"score": score, "issues": issues}
            
        except Exception as e:
            return {"score": 0.0, "issues": [{"tool": "flake8", "error": str(e)}]}
    
    async def _run_mypy(self, source_path: str) -> Dict[str, Any]:
        """Run mypy analysis"""
        try:
            cmd = ["python", "-m", "mypy", "--json-report", "/tmp/mypy_report", source_path]
            process = await asyncio.create_subprocess_SafeCodeExecutor.safe_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            issues = []
            if stderr:
                for line in stderr.decode().split('\n'):
                    if line.strip() and ':' in line:
                        issues.append({
                            "tool": "mypy",
                            "message": line.strip()
                        })
            
            # Calculate score based on type errors
            score = max(0, 10 - len(issues) * 0.3)
            
            return {"score": score, "issues": issues}
            
        except Exception as e:
            return {"score": 0.0, "issues": [{"tool": "mypy", "error": str(e)}]}
    
    def validate_input(self, input_data: Any) -> CodeQualityInput:
        """Validate input data for code quality analysis"""
        if isinstance(input_data, CodeQualityInput):
            return input_data
        
        if isinstance(input_data, dict):
            return CodeQualityInput(**input_data)
        
        raise ValueError(f"Invalid input type: {type(input_data)}")
    
    def validate_output(self, output_data: Any) -> CodeQualityOutput:
        """Validate output data from code quality analysis"""
        if isinstance(output_data, CodeQualityOutput):
            return output_data
        
        if isinstance(output_data, dict):
            return CodeQualityOutput(**output_data)
        
        raise ValueError(f"Invalid output type: {type(output_data)}")

# Register all concrete tools
def register_all_tools():
    """Register all concrete tools with the global registry"""
    from .type_safe_tools import global_tool_registry
    
    # Register Pytest Execution Tool
    pytest_tool = PytestExecutionTool(
        metadata=ToolMetadata(
            name="pytest_executor",
            description="Execute pytest with comprehensive coverage and result analysis",
            category=ToolCategory.TEST_EXECUTION,
            version="1.0.0",
            timeout_seconds=600.0,
            tags=["pytest", "testing", "coverage"]
        )
    )
    global_tool_registry.register_tool(pytest_tool)
    
    # Register Coverage Analysis Tool
    coverage_tool = CoverageAnalysisTool(
        metadata=ToolMetadata(
            name="coverage_analyzer",
            description="Analyze code coverage metrics and generate detailed reports",
            category=ToolCategory.COVERAGE_ANALYSIS,
            version="1.0.0",
            timeout_seconds=300.0,
            tags=["coverage", "analysis", "metrics"]
        )
    )
    global_tool_registry.register_tool(coverage_tool)
    
    # Register Code Quality Tool
    quality_tool = CodeQualityAnalysisTool(
        metadata=ToolMetadata(
            name="code_quality_analyzer",
            description="Analyze code quality using pylint, flake8, and mypy",
            category=ToolCategory.QUALITY_ASSESSMENT,
            version="1.0.0",
            timeout_seconds=600.0,
            tags=["quality", "pylint", "flake8", "mypy"]
        )
    )
    global_tool_registry.register_tool(quality_tool)

# Auto-register tools when module is imported
register_all_tools()

# Export components
__all__ = [
    'PytestExecutionTool',
    'CoverageAnalysisTool', 
    'CodeQualityAnalysisTool',
    'CodeQualityInput',
    'CodeQualityOutput',
    'register_all_tools'
]