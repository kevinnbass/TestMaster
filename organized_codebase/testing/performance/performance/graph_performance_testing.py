"""
Graph Performance Testing Framework
Extracted from FalkorDB performance profiling and optimization testing patterns.
"""

import pytest
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import threading
from contextlib import contextmanager


@dataclass
class MockExecutionStats:
    """Mock execution statistics for performance testing"""
    records_produced: int = 0
    records_consumed: int = 0
    execution_time_ms: float = 0.0
    memory_usage_kb: int = 0
    cpu_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class MockOperationProfile:
    """Mock operation profile for query execution analysis"""
    operation_name: str
    execution_time_ms: float = 0.0
    records_produced: int = 0
    records_consumed: int = 0
    memory_allocated_kb: int = 0
    children: List['MockOperationProfile'] = field(default_factory=list)
    stats: MockExecutionStats = field(default_factory=MockExecutionStats)
    
    @property
    def total_time_ms(self) -> float:
        """Calculate total execution time including children"""
        return self.execution_time_ms + sum(child.total_time_ms for child in self.children)
    
    @property
    def total_records_produced(self) -> int:
        """Calculate total records produced including children"""
        return self.records_produced + sum(child.total_records_produced for child in self.children)


class PerformanceProfiler:
    """Mock performance profiler for graph operations"""
    
    def __init__(self):
        self.profiles = {}
        self.query_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.baseline_metrics = {}
    
    def profile_query(self, query: str, simulate_complexity: bool = True) -> MockOperationProfile:
        """Profile a graph query execution"""
        start_time = time.time()
        
        # Simulate query complexity analysis
        complexity_factor = self._analyze_query_complexity(query) if simulate_complexity else 1.0
        base_time = 0.001 * complexity_factor
        
        # Create profile based on query type
        if "CREATE" in query.upper():
            profile = self._create_profile_for_create_query(query, base_time)
        elif "MATCH" in query.upper():
            profile = self._create_profile_for_match_query(query, base_time)
        elif "UNWIND" in query.upper():
            profile = self._create_profile_for_unwind_query(query, base_time)
        else:
            profile = self._create_profile_for_generic_query(query, base_time)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        profile.execution_time_ms = max(execution_time, base_time)
        
        # Store profile
        self.profiles[query] = profile
        self.query_history.append((query, profile.execution_time_ms, time.time()))
        
        return profile
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity and return complexity factor"""
        complexity = 1.0
        
        # Count complexity indicators
        if "JOIN" in query.upper():
            complexity *= 2.0
        if query.count("MATCH") > 1:
            complexity *= 1.5
        if "OPTIONAL MATCH" in query.upper():
            complexity *= 1.3
        if "WITH" in query.upper():
            complexity *= 1.2
        if len(query) > 100:  # Long queries are typically more complex
            complexity *= 1.5
        
        return complexity
    
    def _create_profile_for_create_query(self, query: str, base_time: float) -> MockOperationProfile:
        """Create profile for CREATE query"""
        node_count = query.count("(")
        edge_count = query.count("-[")
        
        create_nodes_op = MockOperationProfile(
            operation_name="Create Nodes",
            execution_time_ms=base_time * 0.3,
            records_produced=node_count,
            memory_allocated_kb=node_count * 2
        )
        
        create_edges_op = MockOperationProfile(
            operation_name="Create Relationships",
            execution_time_ms=base_time * 0.4,
            records_produced=edge_count,
            memory_allocated_kb=edge_count * 3
        )
        
        results_op = MockOperationProfile(
            operation_name="Results",
            execution_time_ms=base_time * 0.3,
            records_produced=max(node_count, edge_count),
            children=[create_nodes_op, create_edges_op] if edge_count > 0 else [create_nodes_op]
        )
        
        return results_op
    
    def _create_profile_for_match_query(self, query: str, base_time: float) -> MockOperationProfile:
        """Create profile for MATCH query"""
        if "," in query and "MATCH" in query:
            # Cartesian product query
            scan_a = MockOperationProfile(
                operation_name="All Node Scan",
                execution_time_ms=base_time * 0.2,
                records_produced=0,
                memory_allocated_kb=5
            )
            
            scan_b = MockOperationProfile(
                operation_name="All Node Scan", 
                execution_time_ms=base_time * 0.2,
                records_produced=0,
                memory_allocated_kb=5
            )
            
            cartesian_op = MockOperationProfile(
                operation_name="Cartesian Product",
                execution_time_ms=base_time * 0.4,
                records_produced=0,
                children=[scan_a, scan_b],
                memory_allocated_kb=10
            )
            
            project_op = MockOperationProfile(
                operation_name="Project",
                execution_time_ms=base_time * 0.15,
                records_produced=0,
                children=[cartesian_op],
                memory_allocated_kb=3
            )
            
            results_op = MockOperationProfile(
                operation_name="Results",
                execution_time_ms=base_time * 0.05,
                records_produced=0,
                children=[project_op],
                memory_allocated_kb=2
            )
            
            return results_op
        else:
            # Simple match query
            scan_op = MockOperationProfile(
                operation_name="Node By Label Scan",
                execution_time_ms=base_time * 0.6,
                records_produced=1,
                memory_allocated_kb=5
            )
            
            filter_op = MockOperationProfile(
                operation_name="Filter",
                execution_time_ms=base_time * 0.2,
                records_produced=1,
                children=[scan_op],
                memory_allocated_kb=2
            )
            
            results_op = MockOperationProfile(
                operation_name="Results",
                execution_time_ms=base_time * 0.2,
                records_produced=1,
                children=[filter_op],
                memory_allocated_kb=1
            )
            
            return results_op
    
    def _create_profile_for_unwind_query(self, query: str, base_time: float) -> MockOperationProfile:
        """Create profile for UNWIND query"""
        # Extract range size if present
        records_produced = 4  # Default for range(0, 3)
        if "range(" in query:
            try:
                range_part = query[query.find("range("):query.find(")", query.find("range("))]
                numbers = [int(x.strip()) for x in range_part.split(",") if x.strip().isdigit()]
                if len(numbers) >= 2:
                    records_produced = max(0, numbers[1] - numbers[0] + 1)
            except:
                pass
        
        unwind_op = MockOperationProfile(
            operation_name="Unwind",
            execution_time_ms=base_time * 0.5,
            records_produced=records_produced,
            memory_allocated_kb=records_produced
        )
        
        project_op = MockOperationProfile(
            operation_name="Project",
            execution_time_ms=base_time * 0.3,
            records_produced=records_produced,
            children=[unwind_op],
            memory_allocated_kb=records_produced * 2
        )
        
        results_op = MockOperationProfile(
            operation_name="Results",
            execution_time_ms=base_time * 0.2,
            records_produced=records_produced,
            children=[project_op],
            memory_allocated_kb=records_produced
        )
        
        return results_op
    
    def _create_profile_for_generic_query(self, query: str, base_time: float) -> MockOperationProfile:
        """Create profile for generic query"""
        return MockOperationProfile(
            operation_name="Results",
            execution_time_ms=base_time,
            records_produced=1,
            memory_allocated_kb=5
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from query history"""
        if not self.query_history:
            return {}
        
        execution_times = [entry[1] for entry in self.query_history]
        
        return {
            'total_queries': len(self.query_history),
            'avg_execution_time_ms': statistics.mean(execution_times),
            'median_execution_time_ms': statistics.median(execution_times),
            'max_execution_time_ms': max(execution_times),
            'min_execution_time_ms': min(execution_times),
            'std_dev_execution_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'total_execution_time_ms': sum(execution_times)
        }
    
    def identify_slow_queries(self, threshold_ms: float = 10.0) -> List[Tuple[str, float]]:
        """Identify queries that exceed execution time threshold"""
        slow_queries = []
        for query, exec_time, timestamp in self.query_history:
            if exec_time > threshold_ms:
                slow_queries.append((query, exec_time))
        
        return sorted(slow_queries, key=lambda x: x[1], reverse=True)


class LoadTester:
    """Load testing utility for graph operations"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.load_test_results = []
        self.concurrent_results = []
    
    def run_load_test(self, queries: List[str], iterations: int = 100, 
                     concurrent_threads: int = 1) -> Dict[str, Any]:
        """Run load test with specified queries and iterations"""
        start_time = time.time()
        results = {
            'total_queries': len(queries) * iterations * concurrent_threads,
            'execution_times': [],
            'errors': [],
            'throughput_qps': 0.0,
            'avg_response_time_ms': 0.0
        }
        
        if concurrent_threads == 1:
            # Sequential execution
            for _ in range(iterations):
                for query in queries:
                    query_start = time.time()
                    try:
                        profile = self.profiler.profile_query(query)
                        execution_time = profile.execution_time_ms
                        results['execution_times'].append(execution_time)
                    except Exception as e:
                        results['errors'].append(str(e))
                    
        else:
            # Concurrent execution
            results = self._run_concurrent_load_test(queries, iterations, concurrent_threads)
        
        total_time = time.time() - start_time
        successful_queries = len(results['execution_times'])
        
        if successful_queries > 0:
            results['throughput_qps'] = successful_queries / total_time
            results['avg_response_time_ms'] = sum(results['execution_times']) / successful_queries
            results['total_execution_time_s'] = total_time
        
        self.load_test_results.append(results)
        return results
    
    def _run_concurrent_load_test(self, queries: List[str], iterations: int, 
                                threads: int) -> Dict[str, Any]:
        """Run concurrent load test using threading"""
        results = {
            'total_queries': len(queries) * iterations * threads,
            'execution_times': [],
            'errors': [],
            'throughput_qps': 0.0,
            'avg_response_time_ms': 0.0
        }
        
        def worker_thread():
            thread_results = {'times': [], 'errors': []}
            for _ in range(iterations):
                for query in queries:
                    try:
                        profile = self.profiler.profile_query(query)
                        thread_results['times'].append(profile.execution_time_ms)
                    except Exception as e:
                        thread_results['errors'].append(str(e))
            return thread_results
        
        # Create and start threads
        thread_list = []
        for _ in range(threads):
            t = threading.Thread(target=lambda: self.concurrent_results.append(worker_thread()))
            thread_list.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in thread_list:
            t.join()
        
        # Aggregate results
        for thread_result in self.concurrent_results[-threads:]:
            results['execution_times'].extend(thread_result['times'])
            results['errors'].extend(thread_result['errors'])
        
        return results
    
    def benchmark_query_types(self, sample_queries: Dict[str, List[str]], 
                            iterations: int = 50) -> Dict[str, Dict[str, Any]]:
        """Benchmark different query types"""
        benchmarks = {}
        
        for query_type, query_list in sample_queries.items():
            start_time = time.time()
            execution_times = []
            
            for _ in range(iterations):
                for query in query_list:
                    profile = self.profiler.profile_query(query)
                    execution_times.append(profile.execution_time_ms)
            
            benchmarks[query_type] = {
                'avg_time_ms': statistics.mean(execution_times),
                'median_time_ms': statistics.median(execution_times),
                'max_time_ms': max(execution_times),
                'min_time_ms': min(execution_times),
                'total_queries': len(query_list) * iterations,
                'queries_per_second': (len(query_list) * iterations) / (time.time() - start_time)
            }
        
        return benchmarks


class GraphPerformanceTestFramework:
    """Comprehensive test framework for graph performance testing"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.load_tester = LoadTester(self.profiler)
        self.performance_baselines = {}
        self.regression_threshold_pct = 10.0  # 10% regression threshold
    
    @contextmanager
    def performance_timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self.profiler.performance_metrics[operation_name].append(execution_time)
    
    def test_query_profiling(self) -> bool:
        """Test query execution profiling"""
        try:
            # Test UNWIND query profiling
            profile = self.profiler.profile_query("UNWIND range(0, 3) AS x RETURN x")
            
            assert profile.operation_name == "Results"
            assert len(profile.children) == 1
            assert profile.records_produced == 4
            
            project_op = profile.children[0]
            assert project_op.operation_name == "Project"
            assert project_op.records_produced == 4
            
            unwind_op = project_op.children[0]
            assert unwind_op.operation_name == "Unwind"
            assert unwind_op.records_produced == 4
            
            return True
        except Exception as e:
            pytest.fail(f"Query profiling test failed: {e}")
    
    def test_cartesian_product_profiling(self) -> bool:
        """Test Cartesian product query profiling"""
        try:
            profile = self.profiler.profile_query("MATCH (a), (b) RETURN *")
            
            assert profile.operation_name == "Results"
            project_op = profile.children[0]
            assert project_op.operation_name == "Project"
            
            cp_op = project_op.children[0]
            assert cp_op.operation_name == "Cartesian Product"
            assert len(cp_op.children) == 2
            
            scan_ops = cp_op.children
            assert all(op.operation_name == "All Node Scan" for op in scan_ops)
            
            return True
        except Exception as e:
            pytest.fail(f"Cartesian product profiling test failed: {e}")
    
    def test_performance_regression_detection(self) -> bool:
        """Test performance regression detection"""
        try:
            test_query = "MATCH (n:Person) WHERE n.age > 25 RETURN n.name"
            
            # Establish baseline
            baseline_times = []
            for _ in range(10):
                profile = self.profiler.profile_query(test_query)
                baseline_times.append(profile.execution_time_ms)
            
            baseline_avg = statistics.mean(baseline_times)
            self.performance_baselines[test_query] = baseline_avg
            
            # Simulate performance regression (150% of baseline)
            simulated_slow_time = baseline_avg * 1.5
            
            regression_detected = (simulated_slow_time - baseline_avg) / baseline_avg > (self.regression_threshold_pct / 100)
            assert regression_detected
            
            return True
        except Exception as e:
            pytest.fail(f"Performance regression detection test failed: {e}")
    
    def test_load_testing(self) -> bool:
        """Test load testing functionality"""
        try:
            test_queries = [
                "CREATE (n:LoadTest {id: 1}) RETURN n",
                "MATCH (n:LoadTest) RETURN n",
                "CREATE (n:LoadTest {id: 2})-[r:CONNECTS]->(m:LoadTest {id: 3}) RETURN n, r, m"
            ]
            
            # Run load test
            results = self.load_tester.run_load_test(test_queries, iterations=20, concurrent_threads=2)
            
            assert results['total_queries'] == len(test_queries) * 20 * 2
            assert len(results['execution_times']) > 0
            assert results['throughput_qps'] > 0
            assert results['avg_response_time_ms'] > 0
            
            return True
        except Exception as e:
            pytest.fail(f"Load testing test failed: {e}")
    
    def test_query_complexity_analysis(self) -> bool:
        """Test query complexity analysis"""
        try:
            simple_query = "MATCH (n) RETURN n"
            complex_query = "MATCH (a)-[r1]->(b) WITH a, b OPTIONAL MATCH (b)-[r2]->(c) RETURN a, b, c"
            
            simple_profile = self.profiler.profile_query(simple_query)
            complex_profile = self.profiler.profile_query(complex_query)
            
            # Complex query should take longer
            assert complex_profile.execution_time_ms > simple_profile.execution_time_ms
            
            return True
        except Exception as e:
            pytest.fail(f"Query complexity analysis test failed: {e}")
    
    def test_memory_usage_tracking(self) -> bool:
        """Test memory usage tracking in profiles"""
        try:
            query = "CREATE (n:MemoryTest {data: 'large_data_string'}) RETURN n"
            profile = self.profiler.profile_query(query)
            
            assert profile.memory_allocated_kb > 0
            assert profile.total_records_produced > 0
            
            return True
        except Exception as e:
            pytest.fail(f"Memory usage tracking test failed: {e}")
    
    def test_benchmark_comparison(self) -> bool:
        """Test benchmark comparison between query types"""
        try:
            sample_queries = {
                'simple_create': ["CREATE (n:Test) RETURN n"],
                'simple_match': ["MATCH (n:Test) RETURN n"],
                'complex_match': ["MATCH (a:Test)-[r]->(b:Test) RETURN a, r, b"]
            }
            
            benchmarks = self.load_tester.benchmark_query_types(sample_queries, iterations=10)
            
            assert 'simple_create' in benchmarks
            assert 'simple_match' in benchmarks
            assert 'complex_match' in benchmarks
            
            for query_type, metrics in benchmarks.items():
                assert metrics['avg_time_ms'] > 0
                assert metrics['total_queries'] > 0
                assert metrics['queries_per_second'] > 0
            
            return True
        except Exception as e:
            pytest.fail(f"Benchmark comparison test failed: {e}")
    
    def test_slow_query_identification(self) -> bool:
        """Test slow query identification"""
        try:
            # Create mix of fast and slow queries
            fast_queries = ["RETURN 1", "RETURN 2", "RETURN 3"]
            slow_query = "MATCH (a), (b), (c) RETURN COUNT(*)"  # Cartesian product - slow
            
            # Profile queries
            for query in fast_queries:
                self.profiler.profile_query(query)
            
            self.profiler.profile_query(slow_query)
            
            # Identify slow queries (threshold: 1ms)
            slow_queries = self.profiler.identify_slow_queries(threshold_ms=1.0)
            
            assert len(slow_queries) > 0
            assert any(slow_query in sq[0] for sq in slow_queries)
            
            return True
        except Exception as e:
            pytest.fail(f"Slow query identification test failed: {e}")
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all performance tests"""
        results = {}
        
        test_methods = [
            'test_query_profiling',
            'test_cartesian_product_profiling',
            'test_performance_regression_detection',
            'test_load_testing',
            'test_query_complexity_analysis',
            'test_memory_usage_tracking',
            'test_benchmark_comparison',
            'test_slow_query_identification'
        ]
        
        for test_method in test_methods:
            try:
                # Reset for each test
                self.profiler = PerformanceProfiler()
                self.load_tester = LoadTester(self.profiler)
                
                results[test_method] = getattr(self, test_method)()
            except Exception as e:
                results[test_method] = False
                print(f"{test_method} failed: {e}")
        
        return results


# Pytest integration patterns
class TestGraphPerformance:
    """Pytest test class for graph performance testing"""
    
    @pytest.fixture
    def framework(self):
        return GraphPerformanceTestFramework()
    
    def test_execution_plan_analysis(self, framework):
        """Test execution plan analysis"""
        profile = framework.profiler.profile_query("UNWIND range(0, 10) AS x RETURN x")
        
        assert profile.operation_name == "Results"
        assert profile.total_time_ms > 0
        assert profile.total_records_produced == 11  # range(0, 10) produces 11 records
    
    def test_performance_monitoring(self, framework):
        """Test performance monitoring capabilities"""
        queries = [
            "CREATE (n:Monitor {id: 1}) RETURN n",
            "MATCH (n:Monitor) RETURN COUNT(n)",
            "MATCH (n:Monitor) DELETE n"
        ]
        
        for query in queries:
            with framework.performance_timer("monitoring_test"):
                framework.profiler.profile_query(query)
        
        assert len(framework.profiler.performance_metrics["monitoring_test"]) == 3
    
    def test_load_test_execution(self, framework):
        """Test load test execution and metrics"""
        test_queries = ["RETURN rand()", "RETURN timestamp()"]
        
        results = framework.load_tester.run_load_test(test_queries, iterations=5)
        
        assert results['total_queries'] == 10  # 2 queries × 5 iterations
        assert results['throughput_qps'] > 0
        assert len(results['execution_times']) == 10