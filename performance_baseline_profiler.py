#!/usr/bin/env python3
"""
AGENT BETA - PERFORMANCE BASELINE PROFILER
Phase 1, Hour 0-5: Core Performance Profiling System
=====================================

Complete system performance baseline measurement and profiling framework.
Integrates cProfile, py-spy, memory_profiler for comprehensive bottleneck identification.

Created: 2025-08-23 16:30:00 UTC
Agent: Beta (Performance Optimization Specialist)
Phase: 1 (Hours 0-5)
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import cProfile
import pstats
import io
from contextlib import contextmanager
import logging

# Performance monitoring imports
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Install with: pip install memory-profiler")

try:
    import py_spy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False
    print("Warning: py-spy not available. Install with: pip install py-spy")

class PerformanceProfiler:
    """
    Comprehensive performance profiling and baseline measurement system.
    
    Features:
    - CPU profiling with cProfile and py-spy
    - Memory usage tracking with memory_profiler and psutil
    - Network latency measurement
    - Database query performance analysis
    - Response time measurement for all operations
    - Real-time system resource monitoring
    """
    
    def __init__(self, output_dir: str = "performance_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance metrics storage
        self.metrics: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'cpu_profiles': [],
            'memory_profiles': [],
            'response_times': [],
            'system_metrics': [],
            'database_queries': [],
            'network_metrics': [],
            'bottlenecks': []
        }
        
        # System baseline
        self.system_baseline = self._collect_system_baseline()
        
        # Profiling state
        self.profiling_active = False
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'profiler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PerformanceProfiler')
        
        self.logger.info(f"Performance Profiler initialized. Output: {self.output_dir}")
        self.logger.info(f"System Baseline - CPU: {self.system_baseline['cpu_count']} cores, "
                        f"Memory: {self.system_baseline['total_memory_gb']:.2f}GB")

    def _collect_system_baseline(self) -> Dict[str, Any]:
        """Collect comprehensive system baseline metrics"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': {
                path: {
                    'total_gb': usage.total / (1024**3),
                    'used_gb': usage.used / (1024**3),
                    'free_gb': usage.free / (1024**3),
                    'percent': usage.percent
                } for path, usage in [
                    ('/', psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:'))
                ]
            },
            'network_interfaces': list(psutil.net_if_addrs().keys()),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'python_version': sys.version,
            'platform': sys.platform
        }

    @contextmanager
    def profile_cpu(self, operation_name: str):
        """Context manager for CPU profiling with cProfile"""
        profiler = cProfile.Profile()
        start_time = time.perf_counter()
        
        self.logger.info(f"Starting CPU profiling for: {operation_name}")
        profiler.enable()
        
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Save profile data
            profile_filename = self.output_dir / f"cpu_profile_{operation_name}_{int(time.time())}.prof"
            profiler.dump_stats(str(profile_filename))
            
            # Generate human-readable stats
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'operation_name': operation_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': duration,
                'profile_file': str(profile_filename),
                'top_functions': stats_io.getvalue(),
                'total_calls': stats.total_calls,
                'primitive_calls': stats.prim_calls
            }
            
            self.metrics['cpu_profiles'].append(profile_data)
            self.logger.info(f"CPU profiling completed for {operation_name}: {duration:.4f}s, "
                           f"{stats.total_calls} total calls")

    def profile_memory(self, operation_name: str, func, *args, **kwargs):
        """Profile memory usage for a specific operation"""
        if not MEMORY_PROFILER_AVAILABLE:
            self.logger.warning("Memory profiler not available, using psutil fallback")
            return self._profile_memory_fallback(operation_name, func, *args, **kwargs)
        
        self.logger.info(f"Starting memory profiling for: {operation_name}")
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Use memory_profiler
        mem_usage = memory_profiler.memory_usage((func, args, kwargs), interval=0.1)
        
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        profile_data = {
            'operation_name': operation_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_delta_mb': end_memory - start_memory,
            'peak_memory_mb': max(mem_usage) if mem_usage else end_memory,
            'memory_timeline': mem_usage,
            'memory_efficient': end_memory <= start_memory * 1.1  # Within 10% growth
        }
        
        self.metrics['memory_profiles'].append(profile_data)
        self.logger.info(f"Memory profiling completed for {operation_name}: "
                        f"Delta {end_memory - start_memory:.2f}MB, Peak {max(mem_usage) if mem_usage else 0:.2f}MB")
        
        return profile_data

    def _profile_memory_fallback(self, operation_name: str, func, *args, **kwargs):
        """Fallback memory profiling using psutil"""
        start_memory = psutil.Process().memory_info().rss / (1024**2)
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024**2)
        
        profile_data = {
            'operation_name': operation_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': end_time - start_time,
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_delta_mb': end_memory - start_memory,
            'method': 'psutil_fallback'
        }
        
        self.metrics['memory_profiles'].append(profile_data)
        return result

    @contextmanager
    def measure_response_time(self, operation_name: str, operation_type: str = "general"):
        """Context manager for measuring response times"""
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            
            response_data = {
                'operation_name': operation_name,
                'operation_type': operation_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'wall_time_seconds': end_time - start_time,
                'cpu_time_seconds': end_cpu_time - start_cpu_time,
                'efficiency_ratio': (end_cpu_time - start_cpu_time) / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }
            
            self.metrics['response_times'].append(response_data)
            
            # Log slow operations (>100ms)
            if response_data['wall_time_seconds'] > 0.1:
                self.logger.warning(f"Slow operation detected: {operation_name} took {response_data['wall_time_seconds']:.4f}s")

    def start_system_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring in background thread"""
        if self.profiling_active:
            self.logger.warning("System monitoring already active")
            return
        
        self.profiling_active = True
        self.stop_monitoring.clear()
        
        def monitor_loop():
            self.logger.info(f"System monitoring started (interval: {interval}s)")
            
            while not self.stop_monitoring.wait(interval):
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
                    network = psutil.net_io_counters()
                    
                    # Collect process-specific metrics
                    process = psutil.Process()
                    process_cpu = process.cpu_percent()
                    process_memory = process.memory_info()
                    
                    metric_data = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'system_cpu_percent': cpu_percent,
                        'system_memory_percent': memory.percent,
                        'system_memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3),
                        'network_bytes_sent': network.bytes_sent,
                        'network_bytes_recv': network.bytes_recv,
                        'process_cpu_percent': process_cpu,
                        'process_memory_mb': process_memory.rss / (1024**2),
                        'process_memory_vms_mb': process_memory.vms / (1024**2),
                        'thread_count': process.num_threads()
                    }
                    
                    self.metrics['system_metrics'].append(metric_data)
                    
                    # Log resource alerts
                    if cpu_percent > 80:
                        self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    if memory.percent > 80:
                        self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                    if process_memory.rss > 1024**3:  # > 1GB
                        self.logger.warning(f"High process memory: {process_memory.rss / (1024**2):.1f}MB")
                
                except Exception as e:
                    self.logger.error(f"Error in system monitoring: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_system_monitoring(self):
        """Stop system monitoring"""
        if not self.profiling_active:
            return
        
        self.logger.info("Stopping system monitoring...")
        self.stop_monitoring.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.profiling_active = False
        self.logger.info("System monitoring stopped")

    def measure_database_query(self, query: str, execution_time: float, result_count: int = 0):
        """Record database query performance metrics"""
        query_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query': query[:500],  # Truncate long queries
            'execution_time_seconds': execution_time,
            'result_count': result_count,
            'queries_per_second': 1.0 / execution_time if execution_time > 0 else float('inf'),
            'performance_tier': (
                'excellent' if execution_time < 0.01 else
                'good' if execution_time < 0.1 else
                'acceptable' if execution_time < 1.0 else
                'poor'
            )
        }
        
        self.metrics['database_queries'].append(query_data)
        
        if execution_time > 1.0:
            self.logger.warning(f"Slow database query: {execution_time:.4f}s - {query[:100]}...")

    def measure_network_latency(self, endpoint: str, latency: float, status_code: int = 200):
        """Record network operation performance"""
        network_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'endpoint': endpoint,
            'latency_seconds': latency,
            'status_code': status_code,
            'success': status_code < 400,
            'performance_tier': (
                'excellent' if latency < 0.05 else
                'good' if latency < 0.1 else
                'acceptable' if latency < 0.5 else
                'poor'
            )
        }
        
        self.metrics['network_metrics'].append(network_data)
        
        if latency > 0.5:
            self.logger.warning(f"High network latency: {latency:.4f}s for {endpoint}")

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze collected metrics to identify performance bottlenecks"""
        bottlenecks = []
        
        # CPU bottlenecks
        if self.metrics['cpu_profiles']:
            avg_cpu_time = sum(p['duration_seconds'] for p in self.metrics['cpu_profiles']) / len(self.metrics['cpu_profiles'])
            if avg_cpu_time > 1.0:
                bottlenecks.append({
                    'type': 'cpu',
                    'severity': 'high',
                    'description': f'Average CPU operation time: {avg_cpu_time:.2f}s',
                    'recommendation': 'Consider algorithm optimization, caching, or async processing'
                })
        
        # Memory bottlenecks
        if self.metrics['memory_profiles']:
            high_memory_ops = [p for p in self.metrics['memory_profiles'] if p['memory_delta_mb'] > 100]
            if high_memory_ops:
                bottlenecks.append({
                    'type': 'memory',
                    'severity': 'medium',
                    'description': f'{len(high_memory_ops)} operations with >100MB memory growth',
                    'recommendation': 'Implement memory pooling, optimize data structures, add garbage collection'
                })
        
        # Response time bottlenecks
        if self.metrics['response_times']:
            slow_responses = [r for r in self.metrics['response_times'] if r['wall_time_seconds'] > 0.1]
            if len(slow_responses) > len(self.metrics['response_times']) * 0.1:  # >10% slow
                bottlenecks.append({
                    'type': 'response_time',
                    'severity': 'high',
                    'description': f'{len(slow_responses)} slow responses (>100ms) out of {len(self.metrics["response_times"])}',
                    'recommendation': 'Add caching, optimize database queries, implement async processing'
                })
        
        # Database bottlenecks
        if self.metrics['database_queries']:
            slow_queries = [q for q in self.metrics['database_queries'] if q['execution_time_seconds'] > 0.1]
            if slow_queries:
                bottlenecks.append({
                    'type': 'database',
                    'severity': 'high',
                    'description': f'{len(slow_queries)} slow database queries (>100ms)',
                    'recommendation': 'Add database indexes, optimize queries, implement query caching'
                })
        
        # System resource bottlenecks
        if self.metrics['system_metrics']:
            high_cpu_readings = [m for m in self.metrics['system_metrics'] if m['system_cpu_percent'] > 80]
            high_memory_readings = [m for m in self.metrics['system_metrics'] if m['system_memory_percent'] > 80]
            
            if len(high_cpu_readings) > len(self.metrics['system_metrics']) * 0.1:
                bottlenecks.append({
                    'type': 'system_cpu',
                    'severity': 'high',
                    'description': f'High CPU usage detected in {len(high_cpu_readings)} measurements',
                    'recommendation': 'Scale horizontally, optimize CPU-intensive operations, add load balancing'
                })
            
            if len(high_memory_readings) > len(self.metrics['system_metrics']) * 0.1:
                bottlenecks.append({
                    'type': 'system_memory',
                    'severity': 'medium',
                    'description': f'High memory usage detected in {len(high_memory_readings)} measurements',
                    'recommendation': 'Implement memory management, add swapping, scale vertically'
                })
        
        self.metrics['bottlenecks'] = bottlenecks
        return bottlenecks

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        bottlenecks = self.identify_bottlenecks()
        
        # Calculate summary statistics
        report = {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'profiling_duration_hours': (
                    datetime.now(timezone.utc) - datetime.fromisoformat(self.metrics['start_time'].replace('Z', '+00:00'))
                ).total_seconds() / 3600,
                'system_baseline': self.system_baseline
            },
            'summary': {
                'total_cpu_profiles': len(self.metrics['cpu_profiles']),
                'total_memory_profiles': len(self.metrics['memory_profiles']),
                'total_response_measurements': len(self.metrics['response_times']),
                'total_database_queries': len(self.metrics['database_queries']),
                'total_network_measurements': len(self.metrics['network_metrics']),
                'system_monitoring_points': len(self.metrics['system_metrics']),
                'bottlenecks_identified': len(bottlenecks)
            },
            'performance_metrics': {
                'average_response_time': (
                    sum(r['wall_time_seconds'] for r in self.metrics['response_times']) / len(self.metrics['response_times'])
                    if self.metrics['response_times'] else 0
                ),
                'average_database_query_time': (
                    sum(q['execution_time_seconds'] for q in self.metrics['database_queries']) / len(self.metrics['database_queries'])
                    if self.metrics['database_queries'] else 0
                ),
                'average_network_latency': (
                    sum(n['latency_seconds'] for n in self.metrics['network_metrics']) / len(self.metrics['network_metrics'])
                    if self.metrics['network_metrics'] else 0
                ),
                'total_memory_allocated_mb': sum(
                    max(0, p['memory_delta_mb']) for p in self.metrics['memory_profiles']
                ),
                'peak_cpu_usage': (
                    max(m['system_cpu_percent'] for m in self.metrics['system_metrics'])
                    if self.metrics['system_metrics'] else 0
                ),
                'peak_memory_usage': (
                    max(m['system_memory_percent'] for m in self.metrics['system_metrics'])
                    if self.metrics['system_metrics'] else 0
                )
            },
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(bottlenecks),
            'raw_data': self.metrics
        }
        
        return report

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations based on bottlenecks"""
        recommendations = []
        
        # Priority recommendations based on bottleneck severity
        high_severity = [b for b in bottlenecks if b['severity'] == 'high']
        if high_severity:
            recommendations.append("[HIGH PRIORITY] Address high-severity bottlenecks immediately")
            for bottleneck in high_severity:
                recommendations.append(f"  - {bottleneck['type'].upper()}: {bottleneck['recommendation']}")
        
        # General performance recommendations
        recommendations.extend([
            "GENERAL RECOMMENDATIONS:",
            "  - Implement multi-layer caching (Redis + application-level)",
            "  - Add performance monitoring and alerting system",
            "  - Optimize database with proper indexing and query analysis",
            "  - Implement async processing for I/O operations",
            "  - Add load balancing and horizontal scaling capabilities",
            "  - Use CDN for static assets and content delivery",
            "  - Implement connection pooling for databases and external services",
            "  - Add compression for API responses and static content",
            "  - Optimize data structures and algorithms for critical paths",
            "  - Implement graceful degradation and circuit breaker patterns"
        ])
        
        return recommendations

    def save_report(self, filename: Optional[str] = None) -> Path:
        """Save performance report to JSON file"""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.output_dir / filename
        report = self.generate_performance_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved: {report_path}")
        return report_path

    def __enter__(self):
        """Context manager entry"""
        self.start_system_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_system_monitoring()


# Example usage and testing functions
def test_cpu_intensive_operation():
    """Test CPU-intensive operation for profiling"""
    # Simulate CPU-intensive work
    total = 0
    for i in range(1000000):
        total += i * i
    return total

def test_memory_intensive_operation():
    """Test memory-intensive operation for profiling"""
    # Create large data structures
    data = []
    for i in range(100000):
        data.append([j for j in range(100)])
    return len(data)

def test_io_operation():
    """Test I/O operation for profiling"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
        for i in range(10000):
            f.write(f"Test line {i}\n")
        f.flush()
    return "IO operation completed"


def main():
    """Main function to demonstrate profiling capabilities"""
    print("AGENT BETA - Performance Baseline Profiler")
    print("=" * 50)
    
    profiler = PerformanceProfiler()
    
    try:
        # Start system monitoring
        profiler.start_system_monitoring(interval=0.5)
        
        # Test CPU profiling
        print("\n[CPU] Testing CPU profiling...")
        with profiler.profile_cpu("cpu_intensive_test"):
            with profiler.measure_response_time("cpu_intensive_test", "computation"):
                result = test_cpu_intensive_operation()
                print(f"   CPU test result: {result}")
        
        # Test memory profiling
        print("\n[MEMORY] Testing memory profiling...")
        with profiler.measure_response_time("memory_intensive_test", "memory_allocation"):
            profiler.profile_memory("memory_intensive_test", test_memory_intensive_operation)
        
        # Test I/O profiling
        print("\n[I/O] Testing I/O profiling...")
        with profiler.measure_response_time("io_test", "file_io"):
            result = test_io_operation()
            print(f"   I/O test result: {result}")
        
        # Wait for some system monitoring data
        print("\n[SYSTEM] Collecting system metrics...")
        time.sleep(5)
        
        # Simulate database queries
        print("\n[DATABASE] Simulating database queries...")
        profiler.measure_database_query("SELECT * FROM test_table", 0.05, 100)
        profiler.measure_database_query("SELECT COUNT(*) FROM large_table", 0.8, 1)
        
        # Simulate network operations
        print("\n[NETWORK] Simulating network operations...")
        profiler.measure_network_latency("/api/health", 0.02, 200)
        profiler.measure_network_latency("/api/data", 0.15, 200)
        profiler.measure_network_latency("/api/slow", 0.6, 200)
        
    finally:
        # Stop monitoring and generate report
        profiler.stop_system_monitoring()
        
        print("\n[REPORT] Generating performance report...")
        report_path = profiler.save_report()
        
        # Display summary
        report = profiler.generate_performance_report()
        print(f"\nPERFORMANCE BASELINE SUMMARY")
        print(f"   Total measurements: {report['summary']['total_response_measurements']}")
        print(f"   Average response time: {report['performance_metrics']['average_response_time']:.4f}s")
        print(f"   Bottlenecks identified: {report['summary']['bottlenecks_identified']}")
        print(f"   Peak CPU usage: {report['performance_metrics']['peak_cpu_usage']:.1f}%")
        print(f"   Peak memory usage: {report['performance_metrics']['peak_memory_usage']:.1f}%")
        
        if report['bottlenecks']:
            print(f"\n[WARNING] BOTTLENECKS DETECTED:")
            for bottleneck in report['bottlenecks']:
                print(f"   - {bottleneck['type'].upper()}: {bottleneck['description']}")
        
        print(f"\nFull report saved: {report_path}")
        print(f"\nPerformance baseline profiling completed!")


if __name__ == "__main__":
    main()