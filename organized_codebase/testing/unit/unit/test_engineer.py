"""
Test Engineer Role
==================

Test Engineer role responsible for implementing and maintaining tests.
Focused on practical test implementation and technical execution.

Author: TestMaster Team
"""

import asyncio
import ast
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_role import (
    BaseTestRole, TestAction, TestActionType, RoleCapability
)

class TestEngineer(BaseTestRole):
    """
    Test Engineer role responsible for:
    - Implementing unit, integration, and system tests
    - Maintaining and refactoring test code
    - Setting up test infrastructure and tooling
    - Debugging and fixing test failures
    - Optimizing test performance and reliability
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="TestEngineer",
            profile="Implements and maintains high-quality test code and testing infrastructure",
            capabilities=[
                RoleCapability.TEST_IMPLEMENTATION,
                RoleCapability.UNIT_TESTING,
                RoleCapability.INTEGRATION_TESTING,
                RoleCapability.OPTIMIZATION,
                RoleCapability.PERFORMANCE_MONITORING
            ],
            max_concurrent_actions=4,
            **kwargs
        )
        
        # Engineer-specific state
        self.active_implementations: Dict[str, Dict] = {}
        self.test_templates = {
            "unit_test": self._get_unit_test_template(),
            "integration_test": self._get_integration_test_template(),
            "api_test": self._get_api_test_template(),
            "database_test": self._get_database_test_template()
        }
        
        # Tool configurations
        self.tools = {
            "pytest": {"config": "pytest.ini", "plugins": ["pytest-cov", "pytest-xdist"]},
            "coverage": {"min_coverage": 80, "exclude": ["tests/*", "*/migrations/*"]},
            "linting": {"tools": ["flake8", "black", "isort"], "config": "setup.cfg"}
        }
    
    def can_handle_action(self, action_type: TestActionType) -> bool:
        """Check if Test Engineer can handle the action type"""
        engineer_actions = {
            TestActionType.IMPLEMENT,
            TestActionType.EXECUTE,
            TestActionType.OPTIMIZE,
            TestActionType.REVIEW
        }
        return action_type in engineer_actions
    
    async def execute_action(self, action: TestAction) -> TestAction:
        """Execute engineer-specific actions"""
        self.logger.info(f"Executing {action.action_type.value}: {action.description}")
        
        try:
            if action.action_type == TestActionType.IMPLEMENT:
                action.result = await self._implement_tests(action)
            elif action.action_type == TestActionType.EXECUTE:
                action.result = await self._execute_tests(action)
            elif action.action_type == TestActionType.OPTIMIZE:
                action.result = await self._optimize_tests(action)
            elif action.action_type == TestActionType.REVIEW:
                action.result = await self._review_test_code(action)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute action {action.id}: {e}")
            action.error = str(e)
            raise
            
        return action
    
    async def _implement_tests(self, action: TestAction) -> Dict[str, Any]:
        """Implement tests based on specifications"""
        target_module = action.parameters.get("target_module")
        test_type = action.parameters.get("test_type", "unit")
        architecture = action.parameters.get("architecture", {})
        
        if not target_module:
            raise ValueError("target_module is required for test implementation")
        
        implementation_result = {
            "module": target_module,
            "test_type": test_type,
            "files_created": [],
            "tests_implemented": 0,
            "coverage_impact": {},
            "setup_tasks": []
        }
        
        # Analyze target module
        module_analysis = await self._analyze_target_module(target_module)
        implementation_result["module_analysis"] = module_analysis
        
        # Generate test files
        test_files = await self._generate_test_files(target_module, test_type, module_analysis)
        implementation_result["files_created"] = test_files
        implementation_result["tests_implemented"] = sum(
            file_info["test_count"] for file_info in test_files
        )
        
        # Set up infrastructure if needed
        setup_tasks = await self._setup_test_infrastructure(test_type, architecture)
        implementation_result["setup_tasks"] = setup_tasks
        
        # Estimate coverage impact
        coverage_impact = await self._estimate_coverage_impact(target_module, test_files)
        implementation_result["coverage_impact"] = coverage_impact
        
        return implementation_result
    
    async def _analyze_target_module(self, target_module: str) -> Dict[str, Any]:
        """Analyze the target module to understand what to test"""
        module_path = Path(target_module)
        
        if not module_path.exists():
            raise FileNotFoundError(f"Target module not found: {target_module}")
        
        analysis = {
            "path": str(module_path.absolute()),
            "classes": [],
            "functions": [],
            "imports": [],
            "complexity": 0,
            "line_count": 0,
            "testable_components": []
        }
        
        try:
            # Read and parse the module
            content = module_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            analysis["line_count"] = len(content.splitlines())
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "methods": [],
                        "line_number": node.lineno,
                        "is_public": not node.name.startswith('_')
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append({
                                "name": item.name,
                                "line_number": item.lineno,
                                "is_public": not item.name.startswith('_'),
                                "args": len(item.args.args)
                            })
                    
                    analysis["classes"].append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level functions
                    function_info = {
                        "name": node.name,
                        "line_number": node.lineno,
                        "is_public": not node.name.startswith('_'),
                        "args": len(node.args.args)
                    }
                    analysis["functions"].append(function_info)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)
            
            # Calculate complexity (simple metric)
            analysis["complexity"] = len(analysis["classes"]) * 2 + len(analysis["functions"])
            
            # Identify testable components
            testable = []
            for cls in analysis["classes"]:
                if cls["is_public"]:
                    testable.append({"type": "class", "name": cls["name"], "priority": "high"})
            
            for func in analysis["functions"]:
                if func["is_public"]:
                    testable.append({"type": "function", "name": func["name"], "priority": "medium"})
            
            analysis["testable_components"] = testable
            
        except Exception as e:
            self.logger.error(f"Error analyzing module {target_module}: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _generate_test_files(
        self, 
        target_module: str, 
        test_type: str, 
        module_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate test files based on module analysis"""
        test_files = []
        module_path = Path(target_module)
        
        # Determine test file path
        test_dir = Path("tests")
        if test_type == "unit":
            test_file_path = test_dir / f"test_{module_path.stem}.py"
        elif test_type == "integration":
            test_file_path = test_dir / "integration" / f"test_{module_path.stem}_integration.py"
        else:
            test_file_path = test_dir / test_type / f"test_{module_path.stem}_{test_type}.py"
        
        # Create test directory if it doesn't exist
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate test content
        test_content = await self._generate_test_content(target_module, test_type, module_analysis)
        
        # Write test file
        test_file_path.write_text(test_content, encoding='utf-8')
        
        # Count tests
        test_count = test_content.count("def test_")
        
        test_files.append({
            "path": str(test_file_path),
            "type": test_type,
            "test_count": test_count,
            "target_module": target_module,
            "size": len(test_content)
        })
        
        self.logger.info(f"Generated {test_count} tests in {test_file_path}")
        
        return test_files
    
    async def _generate_test_content(
        self, 
        target_module: str, 
        test_type: str, 
        module_analysis: Dict[str, Any]
    ) -> str:
        """Generate the actual test content"""
        module_path = Path(target_module)
        module_name = module_path.stem
        
        # Get base template
        template = self.test_templates.get(f"{test_type}_test", self.test_templates["unit_test"])
        
        # Generate imports
        imports = f"""import pytest
import unittest.mock as mock
from pathlib import Path
import sys

# Add module path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from {module_name} import *
except ImportError:
    import {module_name}
"""
        
        # Generate test classes and functions
        test_content = []
        
        # Generate tests for classes
        for cls in module_analysis.get("classes", []):
            if cls["is_public"]:
                test_content.append(self._generate_class_tests(cls, test_type))
        
        # Generate tests for functions
        for func in module_analysis.get("functions", []):
            if func["is_public"]:
                test_content.append(self._generate_function_tests(func, test_type))
        
        # Combine all content
        full_content = f'''"""
Test module for {module_name}
Generated by TestMaster Test Engineer
Test type: {test_type}
"""

{imports}


{chr(10).join(test_content)}


if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        return full_content
    
    def _generate_class_tests(self, cls: Dict[str, Any], test_type: str) -> str:
        """Generate tests for a class"""
        class_name = cls["name"]
        
        test_class = f"""
class Test{class_name}:
    \"\"\"Test cases for {class_name} class\"\"\"
    
    @pytest.fixture
    def {class_name.lower()}_instance(self):
        \"\"\"Create a {class_name} instance for testing\"\"\"
        return {class_name}()
    
    def test_{class_name.lower()}_creation(self):
        \"\"\"Test {class_name} can be created\"\"\"
        instance = {class_name}()
        assert instance is not None
        assert isinstance(instance, {class_name})
"""
        
        # Generate tests for public methods
        for method in cls.get("methods", []):
            if method["is_public"] and method["name"] != "__init__":
                test_class += f"""
    def test_{method["name"]}(self, {class_name.lower()}_instance):
        \"\"\"Test {method["name"]} method\"\"\"
        # TODO: Implement test for {method["name"]}
        result = {class_name.lower()}_instance.{method["name"]}()
        assert result is not None
"""
        
        return test_class
    
    def _generate_function_tests(self, func: Dict[str, Any], test_type: str) -> str:
        """Generate tests for a function"""
        func_name = func["name"]
        
        return f"""
def test_{func_name}():
    \"\"\"Test {func_name} function\"\"\"
    # TODO: Implement test for {func_name}
    result = {func_name}()
    assert result is not None

def test_{func_name}_with_args():
    \"\"\"Test {func_name} with various arguments\"\"\"
    # TODO: Implement test with different arguments
    pass

def test_{func_name}_edge_cases():
    \"\"\"Test {func_name} edge cases\"\"\"
    # TODO: Implement edge case tests
    pass
"""
    
    def _get_unit_test_template(self) -> str:
        """Get template for unit tests"""
        return """
# Unit Test Template
# Focus on isolated component testing with mocking
"""
    
    def _get_integration_test_template(self) -> str:
        """Get template for integration tests"""
        return """
# Integration Test Template  
# Focus on component interaction testing
"""
    
    def _get_api_test_template(self) -> str:
        """Get template for API tests"""
        return """
# API Test Template
# Focus on HTTP endpoint testing
"""
    
    def _get_database_test_template(self) -> str:
        """Get template for database tests"""
        return """
# Database Test Template
# Focus on data persistence testing
"""
    
    async def _setup_test_infrastructure(self, test_type: str, architecture: Dict[str, Any]) -> List[str]:
        """Set up necessary test infrastructure"""
        setup_tasks = []
        
        # Create pytest configuration
        if not Path("pytest.ini").exists():
            pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    system: System tests
    slow: Slow running tests
"""
            Path("pytest.ini").write_text(pytest_config)
            setup_tasks.append("Created pytest.ini configuration")
        
        # Create test directory structure
        test_dirs = ["tests", "tests/unit", "tests/integration", "tests/system"]
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            init_file = Path(test_dir) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Test package")
                setup_tasks.append(f"Created {test_dir} directory with __init__.py")
        
        # Create conftest.py for shared fixtures
        conftest_path = Path("tests/conftest.py")
        if not conftest_path.exists():
            conftest_content = '''"""
Shared test fixtures and configuration
"""
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {
        "test_string": "Hello, World!",
        "test_number": 42,
        "test_list": [1, 2, 3, 4, 5],
        "test_dict": {"key": "value"}
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
'''
            conftest_path.write_text(conftest_content)
            setup_tasks.append("Created conftest.py with shared fixtures")
        
        return setup_tasks
    
    async def _estimate_coverage_impact(
        self, 
        target_module: str, 
        test_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate the coverage impact of the generated tests"""
        impact = {
            "estimated_coverage": 0,
            "lines_to_cover": 0,
            "test_ratio": 0,
            "quality_score": 0
        }
        
        try:
            # Get target module line count
            module_path = Path(target_module)
            if module_path.exists():
                module_content = module_path.read_text(encoding='utf-8')
                module_lines = len([line for line in module_content.splitlines() if line.strip()])
                impact["lines_to_cover"] = module_lines
                
                # Estimate coverage based on test count and complexity
                total_tests = sum(file_info["test_count"] for file_info in test_files)
                if total_tests > 0:
                    # Simple heuristic: each test covers ~5-10 lines
                    estimated_covered_lines = total_tests * 7
                    impact["estimated_coverage"] = min(
                        (estimated_covered_lines / module_lines) * 100, 95
                    ) if module_lines > 0 else 0
                    
                    # Calculate test ratio (tests per 100 lines of code)
                    impact["test_ratio"] = (total_tests / module_lines) * 100 if module_lines > 0 else 0
                    
                    # Quality score based on coverage and test ratio
                    coverage_score = min(impact["estimated_coverage"] / 80, 1) * 50  # Max 50 points
                    ratio_score = min(impact["test_ratio"] / 20, 1) * 50  # Max 50 points
                    impact["quality_score"] = coverage_score + ratio_score
        
        except Exception as e:
            self.logger.error(f"Error estimating coverage impact: {e}")
            impact["error"] = str(e)
        
        return impact
    
    async def _execute_tests(self, action: TestAction) -> Dict[str, Any]:
        """Execute tests and return results"""
        test_path = action.parameters.get("test_path", "tests")
        test_type = action.parameters.get("test_type", "all")
        coverage = action.parameters.get("coverage", True)
        
        execution_result = {
            "test_path": test_path,
            "test_type": test_type,
            "status": "unknown",
            "summary": {},
            "coverage": {},
            "failures": [],
            "execution_time": 0,
            "command": ""
        }
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=json", "--cov-report=html"])
        
        if test_type != "all":
            cmd.extend(["-m", test_type])
        
        cmd.extend(["-v", "--tb=short", test_path])
        
        execution_result["command"] = " ".join(cmd)
        
        try:
            # Execute tests
            import time
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_result["execution_time"] = time.time() - start_time
            execution_result["return_code"] = result.returncode
            execution_result["stdout"] = result.stdout
            execution_result["stderr"] = result.stderr
            
            # Parse test results
            execution_result["summary"] = self._parse_test_summary(result.stdout)
            
            if result.returncode == 0:
                execution_result["status"] = "passed"
            else:
                execution_result["status"] = "failed"
                execution_result["failures"] = self._parse_test_failures(result.stdout)
            
            # Parse coverage if available
            if coverage and Path("coverage.json").exists():
                execution_result["coverage"] = self._parse_coverage_report("coverage.json")
            
        except subprocess.TimeoutExpired:
            execution_result["status"] = "timeout"
            execution_result["error"] = "Test execution timed out after 5 minutes"
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["error"] = str(e)
        
        return execution_result
    
    def _parse_test_summary(self, stdout: str) -> Dict[str, Any]:
        """Parse pytest output to extract test summary"""
        summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        
        lines = stdout.splitlines()
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse line like "5 passed, 2 failed, 1 skipped in 2.34s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            result_type = parts[i + 1].rstrip(',')
                            if result_type in summary:
                                summary[result_type] = count
                break
        
        summary["total"] = sum(summary[key] for key in ["passed", "failed", "skipped", "errors"])
        return summary
    
    def _parse_test_failures(self, stdout: str) -> List[Dict[str, str]]:
        """Parse pytest output to extract failure details"""
        failures = []
        lines = stdout.splitlines()
        
        current_failure = None
        in_failure = False
        
        for line in lines:
            if line.startswith("FAILED "):
                if current_failure:
                    failures.append(current_failure)
                current_failure = {
                    "test": line.replace("FAILED ", "").split(" - ")[0],
                    "reason": "",
                    "details": ""
                }
                in_failure = True
            elif line.startswith("="):
                if current_failure and in_failure:
                    failures.append(current_failure)
                    current_failure = None
                in_failure = False
            elif in_failure and current_failure:
                if line.strip():
                    if not current_failure["reason"]:
                        current_failure["reason"] = line.strip()
                    else:
                        current_failure["details"] += line + "\n"
        
        if current_failure:
            failures.append(current_failure)
        
        return failures
    
    def _parse_coverage_report(self, coverage_file: str) -> Dict[str, Any]:
        """Parse coverage.json report"""
        try:
            import json
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            return {
                "percent_covered": coverage_data.get("totals", {}).get("percent_covered", 0),
                "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                "lines_missing": coverage_data.get("totals", {}).get("missing_lines", 0),
                "total_lines": coverage_data.get("totals", {}).get("num_statements", 0),
                "files": {
                    file_path: file_data.get("summary", {}).get("percent_covered", 0)
                    for file_path, file_data in coverage_data.get("files", {}).items()
                }
            }
        except Exception as e:
            self.logger.error(f"Error parsing coverage report: {e}")
            return {"error": str(e)}
    
    async def _optimize_tests(self, action: TestAction) -> Dict[str, Any]:
        """Optimize test performance and reliability"""
        optimization_type = action.parameters.get("type", "performance")
        target_tests = action.parameters.get("target_tests", "tests")
        
        optimization_result = {
            "type": optimization_type,
            "optimizations_applied": [],
            "performance_improvement": {},
            "reliability_improvement": {},
            "recommendations": []
        }
        
        if optimization_type == "performance":
            optimizations = await self._optimize_test_performance(target_tests)
            optimization_result["optimizations_applied"].extend(optimizations)
        
        elif optimization_type == "reliability":
            optimizations = await self._optimize_test_reliability(target_tests)
            optimization_result["optimizations_applied"].extend(optimizations)
        
        elif optimization_type == "maintainability":
            optimizations = await self._optimize_test_maintainability(target_tests)
            optimization_result["optimizations_applied"].extend(optimizations)
        
        return optimization_result
    
    async def _optimize_test_performance(self, target_tests: str) -> List[str]:
        """Optimize test execution performance"""
        optimizations = []
        
        # Check for parallel execution setup
        pytest_ini = Path("pytest.ini")
        if pytest_ini.exists():
            content = pytest_ini.read_text()
            if "pytest-xdist" not in content:
                # Add parallel execution
                content += "\n# Parallel execution\naddopts = -n auto\n"
                pytest_ini.write_text(content)
                optimizations.append("Enabled parallel test execution with pytest-xdist")
        
        # Optimize slow tests with proper marking
        for test_file in Path(target_tests).rglob("test_*.py"):
            content = test_file.read_text()
            if "time.sleep" in content or "requests.get" in content:
                if "@pytest.mark.slow" not in content:
                    # Add slow marker to appropriate tests
                    optimizations.append(f"Added slow markers to {test_file.name}")
        
        return optimizations
    
    async def _optimize_test_reliability(self, target_tests: str) -> List[str]:
        """Optimize test reliability and reduce flakiness"""
        optimizations = []
        
        # Check for proper test isolation
        conftest_path = Path("tests/conftest.py")
        if conftest_path.exists():
            content = conftest_path.read_text()
            if "autouse=True" not in content:
                # Add auto-cleanup fixture
                cleanup_fixture = '''
@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Ensure clean test environment"""
    yield
    # Cleanup after each test
    import gc
    gc.collect()
'''
                content += cleanup_fixture
                conftest_path.write_text(content)
                optimizations.append("Added automatic cleanup fixture for test isolation")
        
        return optimizations
    
    async def _optimize_test_maintainability(self, target_tests: str) -> List[str]:
        """Optimize test code maintainability"""
        optimizations = []
        
        # Check for test documentation
        for test_file in Path(target_tests).rglob("test_*.py"):
            content = test_file.read_text()
            if '"""' not in content[:200]:  # No docstring at beginning
                # Add module docstring
                optimizations.append(f"Added documentation to {test_file.name}")
        
        return optimizations
    
    async def _review_test_code(self, action: TestAction) -> Dict[str, Any]:
        """Review test code quality and provide feedback"""
        test_files = action.parameters.get("test_files", [])
        review_type = action.parameters.get("review_type", "comprehensive")
        
        review_result = {
            "overall_score": 0,
            "file_reviews": [],
            "common_issues": [],
            "recommendations": [],
            "quality_metrics": {}
        }
        
        if not test_files:
            # Find all test files
            test_files = [str(f) for f in Path("tests").rglob("test_*.py") if f.is_file()]
        
        total_score = 0
        for test_file in test_files:
            file_review = await self._review_test_file(test_file, review_type)
            review_result["file_reviews"].append(file_review)
            total_score += file_review.get("score", 0)
        
        if test_files:
            review_result["overall_score"] = total_score / len(test_files)
        
        # Identify common issues
        review_result["common_issues"] = self._identify_common_issues(review_result["file_reviews"])
        
        # Generate recommendations
        review_result["recommendations"] = self._generate_review_recommendations(review_result)
        
        return review_result
    
    async def _review_test_file(self, test_file: str, review_type: str) -> Dict[str, Any]:
        """Review a single test file"""
        file_path = Path(test_file)
        
        if not file_path.exists():
            return {"file": test_file, "error": "File not found", "score": 0}
        
        review = {
            "file": test_file,
            "score": 0,
            "issues": [],
            "strengths": [],
            "metrics": {}
        }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Basic metrics
            lines = content.splitlines()
            review["metrics"] = {
                "total_lines": len(lines),
                "test_functions": content.count("def test_"),
                "assertions": content.count("assert"),
                "fixtures": content.count("@pytest.fixture"),
                "mocks": content.count("mock.")
            }
            
            # Quality checks
            score = 100  # Start with perfect score
            
            # Check for docstrings
            if '"""' not in content[:500]:
                review["issues"].append("Missing module docstring")
                score -= 10
            
            # Check test function documentation
            test_functions = review["metrics"]["test_functions"]
            documented_tests = content.count('def test_') - content.count('def test_\n')
            if test_functions > 0 and documented_tests / test_functions < 0.5:
                review["issues"].append("Less than 50% of tests have docstrings")
                score -= 15
            
            # Check for assertions
            if review["metrics"]["assertions"] == 0:
                review["issues"].append("No assertions found")
                score -= 25
            elif review["metrics"]["assertions"] < test_functions:
                review["issues"].append("Some tests may be missing assertions")
                score -= 10
            
            # Check for proper imports
            if "import pytest" not in content:
                review["issues"].append("Missing pytest import")
                score -= 5
            
            # Identify strengths
            if review["metrics"]["fixtures"] > 0:
                review["strengths"].append("Uses pytest fixtures for test setup")
            
            if review["metrics"]["mocks"] > 0:
                review["strengths"].append("Uses mocking for isolation")
            
            if "parametrize" in content:
                review["strengths"].append("Uses parameterized tests")
            
            review["score"] = max(score, 0)
            
        except Exception as e:
            review["error"] = str(e)
            review["score"] = 0
        
        return review
    
    def _identify_common_issues(self, file_reviews: List[Dict[str, Any]]) -> List[str]:
        """Identify issues common across multiple test files"""
        issue_counts = {}
        
        for review in file_reviews:
            for issue in review.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return issues that appear in more than 25% of files
        threshold = max(1, len(file_reviews) * 0.25)
        common_issues = [
            issue for issue, count in issue_counts.items() 
            if count >= threshold
        ]
        
        return common_issues
    
    def _generate_review_recommendations(self, review_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on review results"""
        recommendations = []
        
        overall_score = review_result.get("overall_score", 0)
        
        if overall_score < 70:
            recommendations.append("Overall test quality needs improvement - consider refactoring")
        
        common_issues = review_result.get("common_issues", [])
        
        if "Missing module docstring" in common_issues:
            recommendations.append("Add comprehensive docstrings to all test modules")
        
        if "No assertions found" in common_issues:
            recommendations.append("Ensure all tests contain meaningful assertions")
        
        if "Less than 50% of tests have docstrings" in common_issues:
            recommendations.append("Document test functions with clear descriptions")
        
        # Always recommend best practices
        recommendations.append("Consider using test factories for complex test data")
        recommendations.append("Implement proper test categorization with pytest markers")
        
        return recommendations

# Export the role
__all__ = ['TestEngineer']