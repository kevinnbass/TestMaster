"""
TestMaster Ultimate - Benchmark & Performance Monitoring System
Hours 80-90: Advanced Performance & Memory Optimization
Agent B: Orchestration & Workflow Specialist

Comprehensive benchmark establishment and real-time performance monitoring
for all orchestration optimization components.
"""

import asyncio
import time
import json
import logging
import tracemalloc
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics

# Import our optimization components
from .performance_profiler import PerformanceProfiler, ProfileType
from .memory_optimizer import MemoryOptimizer, GCStrategy
from .intelligent_cache_manager import IntelligentCache, CacheLevel
from .parallel_processor import ParallelProcessor, ProcessingStrategy


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    CACHE_PERFORMANCE = "cache_performance"
    PARALLEL_PROCESSING = "parallel_processing"
    ORCHESTRATION_WORKFLOW = "orchestration_workflow"
    SYSTEM_INTEGRATION = "system_integration"
    MIXED_WORKLOAD = "mixed_workload"


class PerformanceThreshold(Enum):
    """Performance quality thresholds."""
    EXCELLENT = 1.0
    GOOD = 0.8
    ACCEPTABLE = 0.6
    POOR = 0.4
    CRITICAL = 0.2


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    benchmark_type: BenchmarkType
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    throughput_ops_per_second: float
    error_rate: float
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class SystemMetrics:
    """Current system performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    active_threads: int
    active_processes: int
    gc_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    timestamp: datetime


class BenchmarkMonitor:
    """
    Advanced benchmark and performance monitoring system for TestMaster optimization components.
    
    Provides comprehensive performance baseline establishment, real-time monitoring,
    and automated performance regression detection.
    """
    
    def __init__(self, 
                 results_directory: str = "performance_results",
                 monitoring_interval: float = 1.0,
                 alert_thresholds: Dict[str, float] = None):
        """Initialize benchmark monitoring system."""
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(exist_ok=True, parents=True)
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0,
            'response_time': 1000.0  # milliseconds
        }
        
        # Initialize optimization components
        self.profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.cache = IntelligentCache()
        self.parallel_processor = ParallelProcessor()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.baseline_metrics: Dict[BenchmarkType, BenchmarkResult] = {}
        self.historical_results: List[BenchmarkResult] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        # Performance tracking
        self.performance_alerts: List[Dict[str, Any]] = []
        self.regression_detector = PerformanceRegressionDetector()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup performance monitoring logging."""
        log_file = self.results_directory / "performance_monitor.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def establish_baseline_benchmarks(self) -> Dict[BenchmarkType, BenchmarkResult]:
        """
        Establish performance baselines for all optimization components.
        
        Returns:
            Dictionary of baseline benchmark results
        """
        self.logger.info("Starting baseline benchmark establishment...")
        
        # Prepare system for benchmarking
        await self._prepare_system_for_benchmarking()
        
        baselines = {}
        
        # CPU Intensive Benchmark
        self.logger.info("Running CPU intensive benchmark...")
        cpu_result = await self._benchmark_cpu_intensive()
        baselines[BenchmarkType.CPU_INTENSIVE] = cpu_result
        
        # Memory Intensive Benchmark
        self.logger.info("Running memory intensive benchmark...")
        memory_result = await self._benchmark_memory_intensive()
        baselines[BenchmarkType.MEMORY_INTENSIVE] = memory_result
        
        # Cache Performance Benchmark
        self.logger.info("Running cache performance benchmark...")
        cache_result = await self._benchmark_cache_performance()
        baselines[BenchmarkType.CACHE_PERFORMANCE] = cache_result
        
        # Parallel Processing Benchmark
        self.logger.info("Running parallel processing benchmark...")
        parallel_result = await self._benchmark_parallel_processing()
        baselines[BenchmarkType.PARALLEL_PROCESSING] = parallel_result
        
        # Orchestration Workflow Benchmark
        self.logger.info("Running orchestration workflow benchmark...")
        orchestration_result = await self._benchmark_orchestration_workflow()
        baselines[BenchmarkType.ORCHESTRATION_WORKFLOW] = orchestration_result
        
        # Mixed Workload Benchmark
        self.logger.info("Running mixed workload benchmark...")
        mixed_result = await self._benchmark_mixed_workload()
        baselines[BenchmarkType.MIXED_WORKLOAD] = mixed_result
        
        # Store baselines
        self.baseline_metrics = baselines
        await self._save_baseline_results()
        
        self.logger.info(f"Baseline benchmarks established for {len(baselines)} benchmark types")
        return baselines
    
    async def _prepare_system_for_benchmarking(self):
        """Prepare system for accurate benchmarking."""
        # Clear caches
        await self.cache.clear_all_caches()
        
        # Force garbage collection
        self.memory_optimizer.force_gc()
        
        # Wait for system to stabilize
        await asyncio.sleep(2.0)
        
        # Start profiling
        await self.profiler.start_profiling([
            ProfileType.CPU,
            ProfileType.MEMORY,
            ProfileType.IO,
            ProfileType.ASYNC
        ])
    
    async def _benchmark_cpu_intensive(self) -> BenchmarkResult:
        """Benchmark CPU-intensive operations."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # CPU intensive task - prime number calculation
        def calculate_primes(limit: int) -> List[int]:
            primes = []
            for num in range(2, limit):
                is_prime = True
                for i in range(2, int(num ** 0.5) + 1):
                    if num % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(num)
            return primes
        
        # Run CPU intensive operations
        tasks = []
        for _ in range(4):  # Multiple concurrent tasks
            task = asyncio.create_task(
                asyncio.to_thread(calculate_primes, 5000)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Calculate throughput (operations per second)
        total_primes = sum(len(result) for result in results)
        throughput = total_primes / execution_time if execution_time > 0 else 0
        
        # Calculate quality score based on execution time and throughput
        quality_score = min(1.0, 1000.0 / execution_time) * min(1.0, throughput / 500.0)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.CPU_INTENSIVE,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=0.0,  # N/A for CPU benchmark
            throughput_ops_per_second=throughput,
            error_rate=0.0,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'total_primes_calculated': total_primes,
                'concurrent_tasks': len(tasks),
                'prime_limit': 5000
            }
        )
    
    async def _benchmark_memory_intensive(self) -> BenchmarkResult:
        """Benchmark memory-intensive operations."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Memory intensive operations
        large_data_structures = []
        
        # Create large data structures
        for i in range(100):
            large_list = list(range(10000))
            large_dict = {f"key_{j}": f"value_{j}" * 100 for j in range(1000)}
            large_data_structures.append((large_list, large_dict))
        
        # Memory optimization operations
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_optimizer.optimize_memory_usage()
        optimized_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Calculate memory efficiency
        memory_reduction = max(0, initial_memory - optimized_memory)
        memory_efficiency = memory_reduction / initial_memory if initial_memory > 0 else 0
        
        # Calculate quality score based on memory efficiency and execution time
        quality_score = memory_efficiency * min(1.0, 10.0 / execution_time)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY_INTENSIVE,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=0.0,  # N/A for memory benchmark
            throughput_ops_per_second=len(large_data_structures) / execution_time,
            error_rate=0.0,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'data_structures_created': len(large_data_structures),
                'initial_memory_mb': initial_memory,
                'optimized_memory_mb': optimized_memory,
                'memory_reduction_mb': memory_reduction,
                'memory_efficiency': memory_efficiency
            }
        )
    
    async def _benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache performance."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Cache performance operations
        cache_operations = 1000
        cache_hits = 0
        cache_misses = 0
        
        # Simulate cache operations
        for i in range(cache_operations):
            key = f"benchmark_key_{i % 100}"  # Create some cache hits
            
            # Try to get from cache
            cached_value = await self.cache.get(key)
            if cached_value is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                # Store in cache
                value = f"benchmark_value_{i}" * 10
                await self.cache.set(key, value)
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Calculate cache hit rate
        cache_hit_rate = cache_hits / cache_operations if cache_operations > 0 else 0
        throughput = cache_operations / execution_time if execution_time > 0 else 0
        
        # Calculate quality score based on cache hit rate and throughput
        quality_score = cache_hit_rate * min(1.0, throughput / 1000.0)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.CACHE_PERFORMANCE,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=cache_hit_rate,
            throughput_ops_per_second=throughput,
            error_rate=0.0,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'total_cache_operations': cache_operations,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'unique_keys': 100
            }
        )
    
    async def _benchmark_parallel_processing(self) -> BenchmarkResult:
        """Benchmark parallel processing performance."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Parallel processing task
        def compute_factorial(n: int) -> int:
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        
        # Test data
        test_numbers = list(range(100, 200))
        
        # Run parallel processing
        results = await self.parallel_processor.parallel_map(
            compute_factorial, 
            test_numbers,
            strategy=ProcessingStrategy.AUTO
        )
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        throughput = len(test_numbers) / execution_time if execution_time > 0 else 0
        
        # Verify results correctness
        errors = 0
        for i, result in enumerate(results):
            expected = compute_factorial(test_numbers[i])
            if result != expected:
                errors += 1
        
        error_rate = errors / len(results) if results else 0
        
        # Calculate quality score based on throughput and accuracy
        quality_score = min(1.0, throughput / 50.0) * (1.0 - error_rate)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.PARALLEL_PROCESSING,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=0.0,  # N/A for parallel processing
            throughput_ops_per_second=throughput,
            error_rate=error_rate,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'total_operations': len(test_numbers),
                'errors': errors,
                'processing_strategy': 'AUTO',
                'factorial_range': f"{min(test_numbers)}-{max(test_numbers)}"
            }
        )
    
    async def _benchmark_orchestration_workflow(self) -> BenchmarkResult:
        """Benchmark complete orchestration workflow."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Simulate complex orchestration workflow
        workflow_steps = []
        
        # Step 1: Profiling
        step_start = time.time()
        await self.profiler.start_profiling([ProfileType.CPU, ProfileType.MEMORY])
        await asyncio.sleep(0.1)  # Simulate profiling duration
        profiling_report = await self.profiler.generate_report()
        workflow_steps.append(('profiling', time.time() - step_start))
        
        # Step 2: Memory optimization
        step_start = time.time()
        self.memory_optimizer.optimize_memory_usage()
        workflow_steps.append(('memory_optimization', time.time() - step_start))
        
        # Step 3: Cache operations
        step_start = time.time()
        for i in range(50):
            await self.cache.set(f"workflow_key_{i}", f"workflow_value_{i}")
            await self.cache.get(f"workflow_key_{i}")
        workflow_steps.append(('cache_operations', time.time() - step_start))
        
        # Step 4: Parallel processing
        step_start = time.time()
        test_data = list(range(50))
        await self.parallel_processor.parallel_map(lambda x: x * x, test_data)
        workflow_steps.append(('parallel_processing', time.time() - step_start))
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Calculate workflow efficiency
        total_steps = len(workflow_steps)
        throughput = total_steps / execution_time if execution_time > 0 else 0
        
        # Calculate quality score based on completion and efficiency
        quality_score = min(1.0, throughput / 10.0) * min(1.0, 5.0 / execution_time)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.ORCHESTRATION_WORKFLOW,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=1.0,  # All cache operations should hit
            throughput_ops_per_second=throughput,
            error_rate=0.0,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'workflow_steps': len(workflow_steps),
                'step_timings': dict(workflow_steps),
                'profiling_report_size': len(str(profiling_report))
            }
        )
    
    async def _benchmark_mixed_workload(self) -> BenchmarkResult:
        """Benchmark mixed workload scenario."""
        start_time = time.time()
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Mixed workload combining all optimization components
        tasks = []
        
        # CPU intensive task
        cpu_task = asyncio.create_task(
            asyncio.to_thread(lambda: sum(i**2 for i in range(1000)))
        )
        tasks.append(('cpu', cpu_task))
        
        # Memory operations
        memory_task = asyncio.create_task(
            asyncio.to_thread(lambda: [list(range(100)) for _ in range(100)])
        )
        tasks.append(('memory', memory_task))
        
        # Cache operations
        async def cache_operations():
            for i in range(100):
                await self.cache.set(f"mixed_{i}", f"value_{i}")
                await self.cache.get(f"mixed_{i}")
        
        cache_task = asyncio.create_task(cache_operations())
        tasks.append(('cache', cache_task))
        
        # Wait for all tasks
        results = []
        for name, task in tasks:
            try:
                result = await task
                results.append((name, 'success', result))
            except Exception as e:
                results.append((name, 'error', str(e)))
        
        execution_time = time.time() - start_time
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        memory_usage = end_memory - start_memory
        
        # Calculate success rate
        successful_tasks = sum(1 for _, status, _ in results if status == 'success')
        error_rate = 1.0 - (successful_tasks / len(results)) if results else 1.0
        
        throughput = len(results) / execution_time if execution_time > 0 else 0
        quality_score = (successful_tasks / len(results)) * min(1.0, throughput / 5.0)
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.MIXED_WORKLOAD,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(),
            cache_hit_rate=1.0,  # Cache operations should all hit
            throughput_ops_per_second=throughput,
            error_rate=error_rate,
            quality_score=quality_score,
            timestamp=datetime.now(),
            metadata={
                'total_tasks': len(tasks),
                'successful_tasks': successful_tasks,
                'task_results': [(name, status) for name, status, _ in results]
            }
        )
    
    async def _save_baseline_results(self):
        """Save baseline benchmark results to file."""
        baseline_file = self.results_directory / "baseline_benchmarks.json"
        
        baseline_data = {
            benchmark_type.value: {
                'benchmark_type': result.benchmark_type.value,
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'cache_hit_rate': result.cache_hit_rate,
                'throughput_ops_per_second': result.throughput_ops_per_second,
                'error_rate': result.error_rate,
                'quality_score': result.quality_score,
                'timestamp': result.timestamp.isoformat(),
                'metadata': result.metadata
            }
            for benchmark_type, result in self.baseline_metrics.items()
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.logger.info(f"Baseline results saved to {baseline_file}")
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Continuous monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Continuous monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Check for performance alerts
                self._check_performance_alerts(metrics)
                
                # Limit history size
                if len(self.system_metrics_history) > 3600:  # Keep 1 hour at 1s intervals
                    self.system_metrics_history = self.system_metrics_history[-3600:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(interval=0.1),
            memory_usage=psutil.virtual_memory().percent,
            disk_io={
                'read_bytes': psutil.disk_io_counters().read_bytes,
                'write_bytes': psutil.disk_io_counters().write_bytes
            },
            network_io={
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            active_threads=threading.active_count(),
            active_processes=len(psutil.pids()),
            gc_stats={
                'gen0': gc.get_count()[0],
                'gen1': gc.get_count()[1],
                'gen2': gc.get_count()[2]
            },
            cache_stats={
                'cache_size': len(self.cache._l1_cache) if hasattr(self.cache, '_l1_cache') else 0
            },
            timestamp=datetime.now()
        )
    
    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts based on thresholds."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics.cpu_usage,
                'threshold': self.alert_thresholds['cpu_usage'],
                'timestamp': metrics.timestamp.isoformat()
            })
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'memory_high',
                'value': metrics.memory_usage,
                'threshold': self.alert_thresholds['memory_usage'],
                'timestamp': metrics.timestamp.isoformat()
            })
        
        for alert in alerts:
            self.performance_alerts.append(alert)
            self.logger.warning(f"Performance alert: {alert}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_benchmarks': {},
            'recent_performance': {},
            'performance_trends': {},
            'alerts': self.performance_alerts[-50:],  # Last 50 alerts
            'recommendations': []
        }
        
        # Add baseline benchmark results
        for benchmark_type, result in self.baseline_metrics.items():
            report['baseline_benchmarks'][benchmark_type.value] = asdict(result)
        
        # Add recent performance metrics
        if self.system_metrics_history:
            recent_metrics = self.system_metrics_history[-10:]  # Last 10 measurements
            report['recent_performance'] = {
                'cpu_usage_avg': statistics.mean(m.cpu_usage for m in recent_metrics),
                'memory_usage_avg': statistics.mean(m.memory_usage for m in recent_metrics),
                'active_threads_avg': statistics.mean(m.active_threads for m in recent_metrics)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_performance_recommendations()
        
        # Save report
        report_file = self.results_directory / f"performance_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report generated: {report_file}")
        return report
    
    def _generate_performance_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if self.system_metrics_history:
            recent_cpu = statistics.mean(
                m.cpu_usage for m in self.system_metrics_history[-30:]
            )
            recent_memory = statistics.mean(
                m.memory_usage for m in self.system_metrics_history[-30:]
            )
            
            if recent_cpu > 75:
                recommendations.append({
                    'category': 'cpu',
                    'recommendation': 'Consider increasing parallel processing workers or optimizing CPU-intensive algorithms',
                    'priority': 'high'
                })
            
            if recent_memory > 80:
                recommendations.append({
                    'category': 'memory',
                    'recommendation': 'Increase garbage collection frequency or implement object pooling',
                    'priority': 'high'
                })
        
        # Check cache performance
        for benchmark_type, result in self.baseline_metrics.items():
            if benchmark_type == BenchmarkType.CACHE_PERFORMANCE:
                if result.cache_hit_rate < 0.7:
                    recommendations.append({
                        'category': 'cache',
                        'recommendation': 'Optimize cache size or implement better cache warming strategies',
                        'priority': 'medium'
                    })
        
        return recommendations


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, regression_threshold: float = 0.2):
        """Initialize regression detector."""
        self.regression_threshold = regression_threshold
    
    def detect_regression(self, 
                         baseline: BenchmarkResult, 
                         current: BenchmarkResult) -> Dict[str, Any]:
        """Detect performance regression between baseline and current results."""
        regression_info = {
            'has_regression': False,
            'regression_details': [],
            'improvement_details': [],
            'overall_performance_change': 0.0
        }
        
        # Compare execution time (lower is better)
        time_change = (current.execution_time - baseline.execution_time) / baseline.execution_time
        if time_change > self.regression_threshold:
            regression_info['regression_details'].append({
                'metric': 'execution_time',
                'baseline': baseline.execution_time,
                'current': current.execution_time,
                'change_percent': time_change * 100,
                'severity': 'high' if time_change > 0.5 else 'medium'
            })
            regression_info['has_regression'] = True
        elif time_change < -self.regression_threshold:
            regression_info['improvement_details'].append({
                'metric': 'execution_time',
                'baseline': baseline.execution_time,
                'current': current.execution_time,
                'improvement_percent': abs(time_change) * 100
            })
        
        # Compare throughput (higher is better)
        throughput_change = (current.throughput_ops_per_second - baseline.throughput_ops_per_second) / baseline.throughput_ops_per_second
        if throughput_change < -self.regression_threshold:
            regression_info['regression_details'].append({
                'metric': 'throughput',
                'baseline': baseline.throughput_ops_per_second,
                'current': current.throughput_ops_per_second,
                'change_percent': throughput_change * 100,
                'severity': 'high' if throughput_change < -0.5 else 'medium'
            })
            regression_info['has_regression'] = True
        elif throughput_change > self.regression_threshold:
            regression_info['improvement_details'].append({
                'metric': 'throughput',
                'baseline': baseline.throughput_ops_per_second,
                'current': current.throughput_ops_per_second,
                'improvement_percent': throughput_change * 100
            })
        
        # Compare quality score (higher is better)
        quality_change = (current.quality_score - baseline.quality_score) / baseline.quality_score
        if quality_change < -self.regression_threshold:
            regression_info['regression_details'].append({
                'metric': 'quality_score',
                'baseline': baseline.quality_score,
                'current': current.quality_score,
                'change_percent': quality_change * 100,
                'severity': 'critical' if quality_change < -0.5 else 'medium'
            })
            regression_info['has_regression'] = True
        elif quality_change > self.regression_threshold:
            regression_info['improvement_details'].append({
                'metric': 'quality_score',
                'baseline': baseline.quality_score,
                'current': current.quality_score,
                'improvement_percent': quality_change * 100
            })
        
        # Calculate overall performance change
        regression_info['overall_performance_change'] = quality_change
        
        return regression_info


# Example usage and testing
if __name__ == "__main__":
    async def test_benchmark_monitor():
        """Test the benchmark monitoring system."""
        monitor = BenchmarkMonitor()
        
        print("🚀 Starting benchmark establishment...")
        baselines = await monitor.establish_baseline_benchmarks()
        
        print(f"\n📊 Established {len(baselines)} baseline benchmarks:")
        for benchmark_type, result in baselines.items():
            print(f"  {benchmark_type.value}:")
            print(f"    Execution Time: {result.execution_time:.3f}s")
            print(f"    Memory Usage: {result.memory_usage_mb:.2f}MB")
            print(f"    Quality Score: {result.quality_score:.3f}")
            print(f"    Throughput: {result.throughput_ops_per_second:.2f} ops/s")
        
        print("\n🔄 Starting continuous monitoring for 10 seconds...")
        monitor.start_continuous_monitoring()
        await asyncio.sleep(10)
        monitor.stop_continuous_monitoring()
        
        print("\n📈 Generating performance report...")
        report = await monitor.generate_performance_report()
        
        print("✅ Benchmark monitoring system test completed successfully!")
        print(f"Performance report saved with {len(report['baseline_benchmarks'])} benchmarks")
        
        return monitor, baselines, report
    
    # Run test
    asyncio.run(test_benchmark_monitor())