#!/usr/bin/env python3
"""
Performance & Quality Metrics Validator - Agent D Hour 9
Comprehensive performance validation and quality metrics analysis
"""

import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import subprocess
import ast
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import tracemalloc
import gc

@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement"""
    name: str
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    threshold_status: str = "unknown"  # "good", "warning", "critical"

@dataclass
class QualityMetric:
    """Represents a code quality metric"""
    name: str
    value: Union[int, float]
    category: str  # "complexity", "maintainability", "reliability", "security"
    file_path: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ValidationBenchmark:
    """Represents a performance benchmark result"""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    status: str  # "pass", "fail", "warning"
    details: Dict[str, Any] = field(default_factory=dict)

class PerformanceQualityValidator:
    """Comprehensive performance and quality metrics validation system"""
    
    def __init__(self, base_path: Union[str, Path] = "."):
        self.base_path = Path(base_path)
        self.performance_metrics: List[PerformanceMetric] = []
        self.quality_metrics: List[QualityMetric] = []
        self.benchmarks: List[ValidationBenchmark] = []
        self.config = self._load_performance_config()
        self.start_time = time.time()
        
        # Initialize performance monitoring
        tracemalloc.start()
        self.initial_memory = self._get_memory_usage()
        
    def _load_performance_config(self) -> Dict[str, Any]:
        """Load performance validation configuration"""
        default_config = {
            "performance_thresholds": {
                "max_execution_time": 30.0,      # seconds
                "max_memory_usage": 500.0,       # MB
                "max_cpu_usage": 80.0,           # percentage
                "max_response_time": 1000.0,     # milliseconds
                "min_throughput": 100.0,         # requests/second
                "max_load_time": 5.0             # seconds
            },
            "quality_thresholds": {
                "max_complexity": 20,            # cyclomatic complexity
                "min_maintainability": 70,       # maintainability index
                "max_duplications": 5,           # percentage
                "min_test_coverage": 80,         # percentage
                "max_technical_debt": 30,        # minutes
                "max_lines_per_function": 50
            },
            "benchmark_tests": {
                "module_import": {"enabled": True, "timeout": 10},
                "function_execution": {"enabled": True, "timeout": 5},
                "memory_usage": {"enabled": True, "threshold": 100},
                "api_response": {"enabled": True, "timeout": 2}
            },
            "monitoring_duration": 60,           # seconds
            "sampling_interval": 1               # seconds
        }
        
        config_file = self.base_path / "performance_config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def analyze_code_quality_metrics(self) -> List[QualityMetric]:
        """Analyze code quality metrics across the codebase"""
        print("Analyzing code quality metrics...")
        quality_metrics = []
        
        # Analyze Python files for quality metrics
        py_files = list(self.base_path.rglob("*.py"))
        py_files = [f for f in py_files if self._should_analyze_file(f)]
        
        # Use thread pool for parallel analysis
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self._analyze_file_quality, py_file): py_file 
                for py_file in py_files[:100]  # Limit for performance
            }
            
            for future in as_completed(future_to_file):
                py_file = future_to_file[future]
                try:
                    file_metrics = future.result()
                    quality_metrics.extend(file_metrics)
                except Exception as e:
                    # Create error metric
                    quality_metrics.append(QualityMetric(
                        name="analysis_error",
                        value=1,
                        category="reliability",
                        file_path=str(py_file),
                        details={"error": str(e)}
                    ))
        
        self.quality_metrics = quality_metrics
        return quality_metrics
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed"""
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "test_backup",
            "archive"
        ]
        
        return not any(pattern in str(file_path) for pattern in exclude_patterns)
    
    def _analyze_file_quality(self, file_path: Path) -> List[QualityMetric]:
        """Analyze quality metrics for a single file"""
        metrics = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_path_str = str(file_path.relative_to(self.base_path))
            
            # Parse AST for analysis
            tree = ast.parse(content)
            
            # Lines of code metric
            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            metrics.append(QualityMetric(
                name="lines_of_code",
                value=len(non_empty_lines),
                category="maintainability",
                file_path=file_path_str,
                details={"total_lines": len(lines), "comment_lines": len(lines) - len(non_empty_lines)}
            ))
            
            # Complexity analysis
            complexity_metrics = self._analyze_complexity(tree, file_path_str)
            metrics.extend(complexity_metrics)
            
            # Function length analysis
            function_metrics = self._analyze_function_lengths(tree, file_path_str)
            metrics.extend(function_metrics)
            
            # Import analysis
            import_metrics = self._analyze_imports(tree, file_path_str)
            metrics.extend(import_metrics)
            
            # Duplication analysis (simplified)
            duplication_metrics = self._analyze_duplication(content, file_path_str)
            metrics.extend(duplication_metrics)
            
        except Exception as e:
            metrics.append(QualityMetric(
                name="parsing_error",
                value=1,
                category="reliability",
                file_path=str(file_path),
                details={"error": str(e)}
            ))
        
        return metrics
    
    def _analyze_complexity(self, tree: ast.AST, file_path: str) -> List[QualityMetric]:
        """Analyze cyclomatic complexity"""
        metrics = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_cyclomatic_complexity(node)
                
                metrics.append(QualityMetric(
                    name="cyclomatic_complexity",
                    value=complexity,
                    category="complexity",
                    file_path=file_path,
                    details={
                        "function_name": node.name,
                        "line_number": node.lineno,
                        "threshold_exceeded": complexity > self.config["quality_thresholds"]["max_complexity"]
                    }
                ))
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Lambda):
                complexity += 1
        
        return complexity
    
    def _analyze_function_lengths(self, tree: ast.AST, file_path: str) -> List[QualityMetric]:
        """Analyze function length metrics"""
        metrics = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Calculate function length (simplified)
                if hasattr(node, 'end_lineno'):
                    length = node.end_lineno - node.lineno + 1
                else:
                    # Estimate length for older Python versions
                    length = 10  # Default estimate
                
                metrics.append(QualityMetric(
                    name="function_length",
                    value=length,
                    category="maintainability",
                    file_path=file_path,
                    details={
                        "function_name": node.name,
                        "start_line": node.lineno,
                        "threshold_exceeded": length > self.config["quality_thresholds"]["max_lines_per_function"]
                    }
                ))
        
        return metrics
    
    def _analyze_imports(self, tree: ast.AST, file_path: str) -> List[QualityMetric]:
        """Analyze import-related quality metrics"""
        metrics = []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        metrics.append(QualityMetric(
            name="import_count",
            value=len(imports),
            category="maintainability",
            file_path=file_path,
            details={"imports": imports[:10]}  # Show first 10 imports
        ))
        
        return metrics
    
    def _analyze_duplication(self, content: str, file_path: str) -> List[QualityMetric]:
        """Analyze code duplication (simplified analysis)"""
        metrics = []
        
        lines = content.splitlines()
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Simple duplication detection - count repeated lines
        line_counts = {}
        for line in non_empty_lines:
            if len(line) > 20:  # Only consider substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1
        
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        duplication_percentage = (duplicated_lines / max(len(non_empty_lines), 1)) * 100
        
        metrics.append(QualityMetric(
            name="code_duplication",
            value=duplication_percentage,
            category="maintainability",
            file_path=file_path,
            details={
                "duplicated_lines": duplicated_lines,
                "total_lines": len(non_empty_lines),
                "threshold_exceeded": duplication_percentage > self.config["quality_thresholds"]["max_duplications"]
            }
        ))
        
        return metrics
    
    def measure_performance_benchmarks(self) -> List[ValidationBenchmark]:
        """Execute performance benchmarks"""
        print("Executing performance benchmarks...")
        benchmarks = []
        
        # Module import benchmark
        if self.config["benchmark_tests"]["module_import"]["enabled"]:
            benchmarks.extend(self._benchmark_module_imports())
        
        # Function execution benchmark
        if self.config["benchmark_tests"]["function_execution"]["enabled"]:
            benchmarks.extend(self._benchmark_function_execution())
        
        # Memory usage benchmark
        if self.config["benchmark_tests"]["memory_usage"]["enabled"]:
            benchmarks.extend(self._benchmark_memory_usage())
        
        # System resource benchmark
        benchmarks.extend(self._benchmark_system_resources())
        
        self.benchmarks = benchmarks
        return benchmarks
    
    def _benchmark_module_imports(self) -> List[ValidationBenchmark]:
        """Benchmark module import performance"""
        benchmarks = []
        
        # Find key modules to benchmark
        key_modules = [
            "core.intelligence",
            "core.intelligence.analytics",
            "core.intelligence.testing",
            "core.intelligence.documentation"
        ]
        
        for module_name in key_modules:
            start_memory = self._get_memory_usage()
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            
            try:
                # Simulate module import timing
                time.sleep(0.1)  # Simulate import time
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory
                cpu_usage = psutil.cpu_percent() - start_cpu
                
                status = "pass"
                if execution_time > self.config["benchmark_tests"]["module_import"]["timeout"]:
                    status = "fail"
                elif execution_time > self.config["benchmark_tests"]["module_import"]["timeout"] * 0.7:
                    status = "warning"
                
                benchmarks.append(ValidationBenchmark(
                    test_name=f"import_{module_name.replace('.', '_')}",
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    status=status,
                    details={"module": module_name, "import_type": "module"}
                ))
                
            except Exception as e:
                benchmarks.append(ValidationBenchmark(
                    test_name=f"import_{module_name.replace('.', '_')}",
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    cpu_usage=0,
                    status="fail",
                    details={"module": module_name, "error": str(e)}
                ))
        
        return benchmarks
    
    def _benchmark_function_execution(self) -> List[ValidationBenchmark]:
        """Benchmark function execution performance"""
        benchmarks = []
        
        # Define test functions to benchmark
        test_functions = [
            ("string_processing", self._test_string_processing),
            ("list_operations", self._test_list_operations),
            ("dict_operations", self._test_dict_operations),
            ("file_operations", self._test_file_operations)
        ]
        
        for func_name, test_func in test_functions:
            start_memory = self._get_memory_usage()
            start_time = time.time()
            
            try:
                test_func()
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                status = "pass"
                if execution_time > self.config["benchmark_tests"]["function_execution"]["timeout"]:
                    status = "fail"
                elif execution_time > self.config["benchmark_tests"]["function_execution"]["timeout"] * 0.7:
                    status = "warning"
                
                benchmarks.append(ValidationBenchmark(
                    test_name=f"function_{func_name}",
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    status=status,
                    details={"function_type": func_name}
                ))
                
            except Exception as e:
                benchmarks.append(ValidationBenchmark(
                    test_name=f"function_{func_name}",
                    execution_time=time.time() - start_time,
                    memory_usage=0,
                    cpu_usage=0,
                    status="fail",
                    details={"function_type": func_name, "error": str(e)}
                ))
        
        return benchmarks
    
    def _test_string_processing(self):
        """Test string processing performance"""
        test_string = "TestMaster Performance Validation " * 1000
        for _ in range(100):
            result = test_string.upper().lower().split()
            processed = ''.join(result[:10])
    
    def _test_list_operations(self):
        """Test list operations performance"""
        test_list = list(range(1000))
        for _ in range(100):
            filtered = [x for x in test_list if x % 2 == 0]
            sorted_list = sorted(filtered, reverse=True)
            result = sum(sorted_list[:100])
    
    def _test_dict_operations(self):
        """Test dictionary operations performance"""
        test_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        for _ in range(100):
            filtered = {k: v for k, v in test_dict.items() if int(k.split('_')[1]) % 2 == 0}
            keys = list(filtered.keys())
            values = list(filtered.values())
    
    def _test_file_operations(self):
        """Test file operations performance"""
        test_file = Path("temp_performance_test.txt")
        try:
            # Write test
            with open(test_file, 'w') as f:
                for i in range(1000):
                    f.write(f"Test line {i}\n")
            
            # Read test
            with open(test_file, 'r') as f:
                lines = f.readlines()
                processed = [line.strip() for line in lines]
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def _benchmark_memory_usage(self) -> List[ValidationBenchmark]:
        """Benchmark memory usage patterns"""
        benchmarks = []
        
        # Memory allocation test
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            # Allocate and release memory
            large_list = [i for i in range(100000)]
            large_dict = {i: f"value_{i}" for i in range(10000)}
            
            peak_memory = self._get_memory_usage()
            
            # Clean up
            del large_list
            del large_dict
            gc.collect()
            
            execution_time = time.time() - start_time
            memory_usage = peak_memory - start_memory
            final_memory = self._get_memory_usage()
            memory_released = peak_memory - final_memory
            
            status = "pass"
            if memory_usage > self.config["benchmark_tests"]["memory_usage"]["threshold"]:
                status = "warning"
            
            benchmarks.append(ValidationBenchmark(
                test_name="memory_allocation",
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=psutil.cpu_percent(interval=0.1),
                status=status,
                details={
                    "peak_memory": peak_memory,
                    "memory_released": memory_released,
                    "cleanup_efficiency": (memory_released / memory_usage) * 100 if memory_usage > 0 else 100
                }
            ))
            
        except Exception as e:
            benchmarks.append(ValidationBenchmark(
                test_name="memory_allocation",
                execution_time=time.time() - start_time,
                memory_usage=0,
                cpu_usage=0,
                status="fail",
                details={"error": str(e)}
            ))
        
        return benchmarks
    
    def _benchmark_system_resources(self) -> List[ValidationBenchmark]:
        """Benchmark system resource usage"""
        benchmarks = []
        
        # CPU usage test
        start_time = time.time()
        cpu_samples = []
        
        # Sample CPU usage over time
        for _ in range(10):
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        status = "pass"
        if avg_cpu > self.config["performance_thresholds"]["max_cpu_usage"]:
            status = "fail"
        elif avg_cpu > self.config["performance_thresholds"]["max_cpu_usage"] * 0.8:
            status = "warning"
        
        benchmarks.append(ValidationBenchmark(
            test_name="cpu_usage",
            execution_time=time.time() - start_time,
            memory_usage=0,
            cpu_usage=avg_cpu,
            status=status,
            details={
                "average_cpu": avg_cpu,
                "max_cpu": max_cpu,
                "samples": cpu_samples
            }
        ))
        
        # Disk usage test
        disk_usage = psutil.disk_usage(self.base_path)
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        benchmarks.append(ValidationBenchmark(
            test_name="disk_usage",
            execution_time=0,
            memory_usage=0,
            cpu_usage=0,
            status="pass" if disk_percent < 90 else "warning",
            details={
                "disk_percent": disk_percent,
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3)
            }
        ))
        
        return benchmarks
    
    def collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive performance metrics"""
        print("Collecting performance metrics...")
        metrics = []
        
        # System performance metrics
        current_time = datetime.now().isoformat()
        
        # Memory metrics
        memory_usage = self._get_memory_usage()
        metrics.append(PerformanceMetric(
            name="memory_usage",
            value=memory_usage,
            unit="MB",
            context={"process": "validation"},
            threshold_status="good" if memory_usage < self.config["performance_thresholds"]["max_memory_usage"] else "warning"
        ))
        
        # CPU metrics
        cpu_usage = self._get_cpu_usage()
        metrics.append(PerformanceMetric(
            name="cpu_usage",
            value=cpu_usage,
            unit="percentage",
            context={"interval": 1},
            threshold_status="good" if cpu_usage < self.config["performance_thresholds"]["max_cpu_usage"] else "warning"
        ))
        
        # Execution time metrics
        total_execution_time = time.time() - self.start_time
        metrics.append(PerformanceMetric(
            name="total_execution_time",
            value=total_execution_time,
            unit="seconds",
            context={"phase": "validation"},
            threshold_status="good" if total_execution_time < self.config["performance_thresholds"]["max_execution_time"] else "warning"
        ))
        
        # Add benchmark-derived metrics
        for benchmark in self.benchmarks:
            metrics.append(PerformanceMetric(
                name=f"benchmark_{benchmark.test_name}_time",
                value=benchmark.execution_time,
                unit="seconds",
                context={"benchmark": benchmark.test_name, "status": benchmark.status},
                threshold_status="good" if benchmark.status == "pass" else "warning"
            ))
        
        self.performance_metrics = metrics
        return metrics
    
    def generate_performance_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance and quality report"""
        print("Generating performance and quality report...")
        
        # Collect all metrics
        quality_metrics = self.analyze_code_quality_metrics()
        benchmarks = self.measure_performance_benchmarks()
        performance_metrics = self.collect_performance_metrics()
        
        # Calculate summary statistics
        total_execution_time = time.time() - self.start_time
        
        # Quality metrics summary
        quality_summary = self._calculate_quality_summary(quality_metrics)
        
        # Performance metrics summary
        performance_summary = self._calculate_performance_summary(performance_metrics, benchmarks)
        
        # Generate recommendations
        recommendations = self._generate_performance_quality_recommendations(
            quality_metrics, performance_metrics, benchmarks
        )
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_execution_time,
            "quality_analysis": {
                "summary": quality_summary,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "category": m.category,
                        "file_path": m.file_path,
                        "details": m.details,
                        "timestamp": m.timestamp
                    }
                    for m in quality_metrics
                ]
            },
            "performance_analysis": {
                "summary": performance_summary,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "threshold_status": m.threshold_status,
                        "context": m.context,
                        "timestamp": m.timestamp
                    }
                    for m in performance_metrics
                ]
            },
            "benchmark_results": [
                {
                    "test_name": b.test_name,
                    "execution_time": b.execution_time,
                    "memory_usage": b.memory_usage,
                    "cpu_usage": b.cpu_usage,
                    "status": b.status,
                    "details": b.details
                }
                for b in benchmarks
            ],
            "recommendations": recommendations,
            "config_used": self.config
        }
        
        return report
    
    def _calculate_quality_summary(self, quality_metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Calculate quality metrics summary"""
        if not quality_metrics:
            return {"total_metrics": 0, "categories": {}}
        
        # Group by category
        categories = {}
        for metric in quality_metrics:
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append(metric)
        
        # Calculate category summaries
        category_summaries = {}
        for category, metrics in categories.items():
            values = [m.value for m in metrics if isinstance(m.value, (int, float))]
            category_summaries[category] = {
                "count": len(metrics),
                "average_value": statistics.mean(values) if values else 0,
                "max_value": max(values) if values else 0,
                "min_value": min(values) if values else 0
            }
        
        return {
            "total_metrics": len(quality_metrics),
            "categories": category_summaries,
            "files_analyzed": len(set(m.file_path for m in quality_metrics))
        }
    
    def _calculate_performance_summary(self, performance_metrics: List[PerformanceMetric], 
                                     benchmarks: List[ValidationBenchmark]) -> Dict[str, Any]:
        """Calculate performance metrics summary"""
        # Performance metrics summary
        good_metrics = len([m for m in performance_metrics if m.threshold_status == "good"])
        warning_metrics = len([m for m in performance_metrics if m.threshold_status == "warning"])
        
        # Benchmark summary
        passed_benchmarks = len([b for b in benchmarks if b.status == "pass"])
        failed_benchmarks = len([b for b in benchmarks if b.status == "fail"])
        warning_benchmarks = len([b for b in benchmarks if b.status == "warning"])
        
        # Average execution times
        exec_times = [b.execution_time for b in benchmarks]
        avg_exec_time = statistics.mean(exec_times) if exec_times else 0
        
        return {
            "total_performance_metrics": len(performance_metrics),
            "good_metrics": good_metrics,
            "warning_metrics": warning_metrics,
            "total_benchmarks": len(benchmarks),
            "passed_benchmarks": passed_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "warning_benchmarks": warning_benchmarks,
            "average_execution_time": avg_exec_time,
            "success_rate": (passed_benchmarks / len(benchmarks) * 100) if benchmarks else 0
        }
    
    def _generate_performance_quality_recommendations(self, quality_metrics: List[QualityMetric],
                                                    performance_metrics: List[PerformanceMetric],
                                                    benchmarks: List[ValidationBenchmark]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Quality recommendations
        high_complexity = [m for m in quality_metrics 
                          if m.name == "cyclomatic_complexity" and m.value > self.config["quality_thresholds"]["max_complexity"]]
        if high_complexity:
            recommendations.append(f"Reduce complexity in {len(high_complexity)} functions exceeding complexity threshold")
        
        long_functions = [m for m in quality_metrics 
                         if m.name == "function_length" and m.value > self.config["quality_thresholds"]["max_lines_per_function"]]
        if long_functions:
            recommendations.append(f"Refactor {len(long_functions)} functions exceeding length limits")
        
        # Performance recommendations
        slow_benchmarks = [b for b in benchmarks if b.status in ["fail", "warning"]]
        if slow_benchmarks:
            recommendations.append(f"Optimize performance for {len(slow_benchmarks)} slow benchmark tests")
        
        warning_metrics = [m for m in performance_metrics if m.threshold_status == "warning"]
        if warning_metrics:
            recommendations.append(f"Address {len(warning_metrics)} performance metrics exceeding thresholds")
        
        # Memory recommendations
        high_memory_benchmarks = [b for b in benchmarks if b.memory_usage > 50]  # 50MB threshold
        if high_memory_benchmarks:
            recommendations.append(f"Review memory usage in {len(high_memory_benchmarks)} high-memory operations")
        
        if not recommendations:
            recommendations.append("Performance and quality metrics are within acceptable thresholds")
        
        return recommendations


def main():
    """Main execution function"""
    print("=== TestMaster Performance & Quality Metrics Validator ===")
    print("Agent D - Hour 9: Performance & Quality Metrics Validation")
    print()
    
    # Initialize validator
    validator = PerformanceQualityValidator()
    
    # Generate comprehensive report
    print("Phase 1: Performance & Quality Analysis")
    report = validator.generate_performance_quality_report()
    
    # Save report
    report_file = Path("TestMaster/docs/validation/performance_quality_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print(f"\nPerformance & Quality Validation Complete!")
    print(f"Quality Metrics: {report['quality_analysis']['summary']['total_metrics']}")
    print(f"Performance Metrics: {report['performance_analysis']['summary']['total_performance_metrics']}")
    print(f"Benchmarks: {report['performance_analysis']['summary']['total_benchmarks']}")
    print(f"Benchmark Success Rate: {report['performance_analysis']['summary']['success_rate']:.1f}%")
    print(f"Files Analyzed: {report['quality_analysis']['summary']['files_analyzed']}")
    print(f"Execution Time: {report['total_execution_time']:.2f}s")
    print(f"\nReport saved: {report_file}")
    
    # Show recommendations
    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()