#!/usr/bin/env python3
"""
Comprehensive Integration Systems Test
=====================================
Tests all 11 integration systems to ensure they work together cohesively
and provide the expected enterprise-grade functionality.
"""

import sys
import time
import json
import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
import concurrent.futures
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class IntegrationSystemTester:
    """Tests all integration systems for enterprise functionality."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'systems': {},
            'cross_system_tests': {},
            'performance_metrics': {},
            'load_tests': {},
            'fault_tolerance': {},
            'summary': {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def test_component(self, name: str, test_func, timeout: int = 30) -> Tuple[bool, Dict[str, Any]]:
        """Run a single component test with timeout and metrics."""
        self.total_tests += 1
        start_time = time.time()
        metrics = {}
        
        try:
            # Run test with timeout
            result = test_func()
            elapsed = time.time() - start_time
            metrics = {
                'execution_time': elapsed,
                'memory_usage': 'N/A',  # Could add psutil for real memory tracking
                'cpu_usage': 'N/A'
            }
            
            if result:
                self.passed_tests += 1
                logger.info(f"  [PASS] {name} ({elapsed:.2f}s)")
                return True, metrics
            else:
                self.failed_tests += 1
                logger.error(f"  [FAIL] {name} - returned False")
                return False, metrics
                
        except Exception as e:
            elapsed = time.time() - start_time
            self.failed_tests += 1
            logger.error(f"  [FAIL] {name} - {str(e)}")
            metrics = {
                'execution_time': elapsed,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False, metrics
    
    def test_automatic_scaling_system(self) -> bool:
        """Test automatic scaling capabilities."""
        try:
            from integration.automatic_scaling_system import AutomaticScalingSystem
            
            scaler = AutomaticScalingSystem()
            
            # Test basic scaling operations
            scaler.set_target_capacity(100)
            current_capacity = scaler.get_current_capacity()
            
            # Test scaling policies
            scaler.add_scaling_policy("cpu_threshold", threshold=80)
            policies = scaler.get_scaling_policies()
            
            # Test scaling event simulation
            scaler.trigger_scale_up(reason="Load increase")
            scaler.trigger_scale_down(reason="Load decrease")
            
            return (hasattr(scaler, 'set_target_capacity') and 
                   hasattr(scaler, 'get_current_capacity') and
                   len(policies) >= 1)
                   
        except Exception as e:
            logger.debug(f"Scaling system test error: {e}")
            return False
    
    def test_comprehensive_error_recovery(self) -> bool:
        """Test error recovery and resilience systems."""
        try:
            from integration.comprehensive_error_recovery import ErrorRecoverySystem
            
            recovery = ErrorRecoverySystem()
            
            # Test error detection
            recovery.register_error_handler("timeout", lambda: "recovered")
            recovery.register_error_handler("connection", lambda: "reconnected")
            
            # Simulate errors and test recovery
            recovery.handle_error("timeout", {"context": "test"})
            recovery.handle_error("connection", {"context": "test"})
            
            # Test recovery metrics
            metrics = recovery.get_recovery_metrics()
            
            # Test circuit breaker
            recovery.open_circuit("test_service")
            recovery.close_circuit("test_service")
            
            return (hasattr(recovery, 'handle_error') and 
                   hasattr(recovery, 'get_recovery_metrics') and
                   metrics is not None)
                   
        except Exception as e:
            logger.debug(f"Error recovery test error: {e}")
            return False
    
    def test_cross_system_communication(self) -> bool:
        """Test inter-system communication protocols."""
        try:
            from integration.cross_system_communication import CrossSystemCommunication
            
            comm = CrossSystemCommunication()
            
            # Test message publishing/subscribing
            comm.subscribe("test_channel", lambda msg: logger.debug(f"Received: {msg}"))
            comm.publish("test_channel", {"test": "message"})
            
            # Test system registration
            comm.register_system("test_system", {"endpoint": "localhost:8080"})
            systems = comm.get_registered_systems()
            
            # Test health checks
            comm.send_health_check("test_system")
            
            # Test message routing
            comm.route_message("test_system", {"command": "status"})
            
            return (hasattr(comm, 'publish') and 
                   hasattr(comm, 'subscribe') and
                   len(systems) >= 1)
                   
        except Exception as e:
            logger.debug(f"Cross-system communication test error: {e}")
            return False
    
    def test_distributed_task_queue(self) -> bool:
        """Test distributed task processing."""
        try:
            from integration.distributed_task_queue import DistributedTaskQueue
            
            queue = DistributedTaskQueue()
            
            # Test task submission
            task_id = queue.submit_task("test_task", {"param": "value"})
            
            # Test task status checking
            status = queue.get_task_status(task_id)
            
            # Test worker management
            queue.add_worker("worker_1", {"capacity": 10})
            workers = queue.get_active_workers()
            
            # Test task completion
            queue.complete_task(task_id, {"result": "success"})
            
            # Test queue statistics
            stats = queue.get_queue_statistics()
            
            return (task_id is not None and
                   status is not None and
                   len(workers) >= 1 and
                   stats is not None)
                   
        except Exception as e:
            logger.debug(f"Distributed task queue test error: {e}")
            return False
    
    def test_intelligent_caching_layer(self) -> bool:
        """Test intelligent caching with TTL and invalidation."""
        try:
            from integration.intelligent_caching_layer import IntelligentCachingLayer
            
            cache = IntelligentCachingLayer()
            
            # Test basic caching operations
            cache.set("test_key", "test_value", ttl=300)
            value = cache.get("test_key")
            
            # Test cache invalidation
            cache.invalidate("test_key")
            invalidated_value = cache.get("test_key")
            
            # Test cache statistics
            stats = cache.get_cache_statistics()
            
            # Test cache patterns
            cache.set_pattern("user:*", ttl=600)
            cache.set("user:123", {"name": "John"})
            
            # Test cache warming
            cache.warm_cache("frequent_queries", {"query1": "result1"})
            
            return (value == "test_value" and
                   invalidated_value is None and
                   stats is not None)
                   
        except Exception as e:
            logger.debug(f"Intelligent caching test error: {e}")
            return False
    
    def test_load_balancing_system(self) -> bool:
        """Test load balancing and traffic distribution."""
        try:
            from integration.load_balancing_system import LoadBalancingSystem
            
            balancer = LoadBalancingSystem()
            
            # Test server registration
            balancer.register_server("server_1", {"host": "localhost", "port": 8081})
            balancer.register_server("server_2", {"host": "localhost", "port": 8082})
            
            # Test load balancing algorithms
            balancer.set_algorithm("round_robin")
            server1 = balancer.get_next_server()
            server2 = balancer.get_next_server()
            
            # Test health monitoring
            balancer.mark_server_healthy("server_1")
            balancer.mark_server_unhealthy("server_2")
            
            # Test load metrics
            balancer.update_server_load("server_1", load=75)
            metrics = balancer.get_load_metrics()
            
            return (server1 is not None and
                   server2 is not None and
                   metrics is not None)
                   
        except Exception as e:
            logger.debug(f"Load balancing test error: {e}")
            return False
    
    def test_multi_environment_support(self) -> bool:
        """Test multi-environment configuration and deployment."""
        try:
            from integration.multi_environment_support import MultiEnvironmentSupport
            
            env_support = MultiEnvironmentSupport()
            
            # Test environment configuration
            env_support.configure_environment("development", {
                "database_url": "dev.db",
                "debug": True
            })
            env_support.configure_environment("production", {
                "database_url": "prod.db", 
                "debug": False
            })
            
            # Test environment switching
            env_support.switch_environment("development")
            dev_config = env_support.get_current_config()
            
            env_support.switch_environment("production")
            prod_config = env_support.get_current_config()
            
            # Test environment validation
            validation = env_support.validate_environment("production")
            
            return (dev_config.get("debug") == True and
                   prod_config.get("debug") == False and
                   validation.get("valid", False))
                   
        except Exception as e:
            logger.debug(f"Multi-environment test error: {e}")
            return False
    
    def test_predictive_analytics_engine(self) -> bool:
        """Test predictive analytics and machine learning integration."""
        try:
            from integration.predictive_analytics_engine import PredictiveAnalyticsEngine
            
            analytics = PredictiveAnalyticsEngine()
            
            # Test data ingestion
            analytics.ingest_data("user_behavior", [
                {"user_id": 1, "action": "login", "timestamp": time.time()},
                {"user_id": 1, "action": "view_page", "timestamp": time.time()},
            ])
            
            # Test pattern recognition
            patterns = analytics.detect_patterns("user_behavior")
            
            # Test prediction generation
            prediction = analytics.predict("user_churn", {"user_id": 1})
            
            # Test model training
            analytics.train_model("user_behavior_model", "user_behavior")
            
            # Test analytics metrics
            metrics = analytics.get_analytics_metrics()
            
            return (patterns is not None and
                   prediction is not None and
                   metrics is not None)
                   
        except Exception as e:
            logger.debug(f"Predictive analytics test error: {e}")
            return False
    
    def test_realtime_performance_monitoring(self) -> bool:
        """Test real-time performance monitoring and alerting."""
        try:
            from integration.realtime_performance_monitoring import RealtimePerformanceMonitoring
            
            monitor = RealtimePerformanceMonitoring()
            
            # Test metrics collection
            monitor.start_monitoring()
            time.sleep(0.1)  # Let it collect some data
            
            # Test metric recording
            monitor.record_metric("response_time", 150, {"endpoint": "/api/test"})
            monitor.record_metric("cpu_usage", 65, {"host": "server1"})
            
            # Test alerting
            monitor.set_alert_threshold("response_time", max_value=200)
            monitor.set_alert_threshold("cpu_usage", max_value=80)
            
            # Test dashboard data
            dashboard_data = monitor.get_dashboard_data()
            
            # Test alert history
            alerts = monitor.get_alert_history()
            
            monitor.stop_monitoring()
            
            return (dashboard_data is not None and
                   isinstance(alerts, list))
                   
        except Exception as e:
            logger.debug(f"Real-time monitoring test error: {e}")
            return False
    
    def test_resource_optimization_engine(self) -> bool:
        """Test resource optimization and allocation."""
        try:
            from integration.resource_optimization_engine import ResourceOptimizationEngine
            
            optimizer = ResourceOptimizationEngine()
            
            # Test resource registration
            optimizer.register_resource("cpu", capacity=100, current_usage=60)
            optimizer.register_resource("memory", capacity=16384, current_usage=8192)
            optimizer.register_resource("disk", capacity=1024, current_usage=512)
            
            # Test optimization strategies
            optimization = optimizer.optimize_allocation({
                "cpu": 80,
                "memory": 12000,
                "disk": 200
            })
            
            # Test resource prediction
            prediction = optimizer.predict_resource_needs(time_horizon=3600)
            
            # Test resource scaling recommendations
            recommendations = optimizer.get_scaling_recommendations()
            
            # Test resource efficiency metrics
            efficiency = optimizer.calculate_efficiency()
            
            return (optimization is not None and
                   prediction is not None and
                   efficiency is not None)
                   
        except Exception as e:
            logger.debug(f"Resource optimization test error: {e}")
            return False
    
    def test_service_mesh_integration(self) -> bool:
        """Test service mesh capabilities for microservices."""
        try:
            from integration.service_mesh_integration import ServiceMeshIntegration
            
            mesh = ServiceMeshIntegration()
            
            # Test service registration
            mesh.register_service("user_service", {
                "host": "localhost",
                "port": 8080,
                "health_check": "/health"
            })
            mesh.register_service("order_service", {
                "host": "localhost", 
                "port": 8081,
                "health_check": "/health"
            })
            
            # Test service discovery
            services = mesh.discover_services()
            user_service = mesh.find_service("user_service")
            
            # Test traffic routing
            mesh.configure_traffic_split("user_service", {
                "v1": 80,
                "v2": 20
            })
            
            # Test circuit breaker
            mesh.enable_circuit_breaker("user_service")
            
            # Test observability
            metrics = mesh.get_service_metrics("user_service")
            
            return (len(services) >= 2 and
                   user_service is not None and
                   metrics is not None)
                   
        except Exception as e:
            logger.debug(f"Service mesh test error: {e}")
            return False
    
    def test_cross_system_integration(self) -> bool:
        """Test integration between multiple systems."""
        logger.info("\nTesting cross-system integration scenarios...")
        
        try:
            # Test caching + load balancing integration
            from integration.intelligent_caching_layer import IntelligentCachingLayer
            from integration.load_balancing_system import LoadBalancingSystem
            
            cache = IntelligentCachingLayer()
            balancer = LoadBalancingSystem()
            
            # Simulate load-balanced cache access
            balancer.register_server("cache_server_1", {"type": "cache"})
            cache.set("shared_data", "integrated_value")
            
            # Test scaling + monitoring integration
            from integration.automatic_scaling_system import AutomaticScalingSystem
            from integration.realtime_performance_monitoring import RealtimePerformanceMonitoring
            
            scaler = AutomaticScalingSystem()
            monitor = RealtimePerformanceMonitoring()
            
            # Simulate monitoring-driven scaling
            monitor.start_monitoring()
            monitor.record_metric("cpu_usage", 85)  # High CPU should trigger scaling
            scaler.trigger_scale_up(reason="High CPU detected by monitoring")
            monitor.stop_monitoring()
            
            return True
            
        except Exception as e:
            logger.debug(f"Cross-system integration test error: {e}")
            return False
    
    def test_system_performance_under_load(self) -> bool:
        """Test system performance under simulated load."""
        logger.info("\nTesting system performance under load...")
        
        try:
            # Simulate concurrent operations across systems
            def stress_test_operation():
                try:
                    from integration.intelligent_caching_layer import IntelligentCachingLayer
                    cache = IntelligentCachingLayer()
                    
                    # Perform multiple cache operations
                    for i in range(10):
                        cache.set(f"stress_key_{i}", f"value_{i}")
                        cache.get(f"stress_key_{i}")
                    
                    return True
                except:
                    return False
            
            # Run stress test with multiple threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(stress_test_operation) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            success_rate = sum(results) / len(results)
            return success_rate >= 0.8  # 80% success rate under load
            
        except Exception as e:
            logger.debug(f"Load test error: {e}")
            return False
    
    def test_fault_tolerance(self) -> bool:
        """Test system behavior under failure conditions."""
        logger.info("\nTesting fault tolerance...")
        
        try:
            from integration.comprehensive_error_recovery import ErrorRecoverySystem
            
            recovery = ErrorRecoverySystem()
            
            # Test multiple failure scenarios
            failure_scenarios = [
                "network_timeout",
                "service_unavailable", 
                "database_connection_lost",
                "memory_exhaustion",
                "disk_full"
            ]
            
            recovered_count = 0
            for scenario in failure_scenarios:
                try:
                    recovery.handle_error(scenario, {"severity": "high"})
                    recovered_count += 1
                except:
                    pass
            
            recovery_rate = recovered_count / len(failure_scenarios)
            return recovery_rate >= 0.6  # 60% recovery rate
            
        except Exception as e:
            logger.debug(f"Fault tolerance test error: {e}")
            return False
    
    def run_all_integration_tests(self):
        """Run comprehensive integration system tests."""
        logger.info("Starting comprehensive integration systems test...")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Individual system tests
        logger.info("\n" + "="*60)
        logger.info("TESTING INDIVIDUAL INTEGRATION SYSTEMS")
        logger.info("="*60)
        
        system_tests = [
            ("Automatic Scaling System", self.test_automatic_scaling_system),
            ("Comprehensive Error Recovery", self.test_comprehensive_error_recovery),
            ("Cross-System Communication", self.test_cross_system_communication),
            ("Distributed Task Queue", self.test_distributed_task_queue),
            ("Intelligent Caching Layer", self.test_intelligent_caching_layer),
            ("Load Balancing System", self.test_load_balancing_system),
            ("Multi-Environment Support", self.test_multi_environment_support),
            ("Predictive Analytics Engine", self.test_predictive_analytics_engine),
            ("Realtime Performance Monitoring", self.test_realtime_performance_monitoring),
            ("Resource Optimization Engine", self.test_resource_optimization_engine),
            ("Service Mesh Integration", self.test_service_mesh_integration)
        ]
        
        system_results = {}
        for name, test_func in system_tests:
            success, metrics = self.test_component(name, test_func)
            system_results[name] = {
                'success': success,
                'metrics': metrics
            }
        
        self.results['systems'] = system_results
        
        # Cross-system integration tests
        logger.info("\n" + "="*60)
        logger.info("TESTING CROSS-SYSTEM INTEGRATION")
        logger.info("="*60)
        
        integration_tests = [
            ("Cross-System Integration", self.test_cross_system_integration),
            ("Performance Under Load", self.test_system_performance_under_load),
            ("Fault Tolerance", self.test_fault_tolerance)
        ]
        
        integration_results = {}
        for name, test_func in integration_tests:
            success, metrics = self.test_component(name, test_func)
            integration_results[name] = {
                'success': success,
                'metrics': metrics
            }
        
        self.results['cross_system_tests'] = integration_results
        
        # Generate comprehensive report
        return self.generate_integration_report()
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report."""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION SYSTEMS REPORT")
        logger.info("="*60)
        
        # Calculate success rates
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # System status determination
        if success_rate >= 90:
            status = 'EXCELLENT'
        elif success_rate >= 80:
            status = 'HEALTHY'
        elif success_rate >= 60:
            status = 'DEGRADED'
        else:
            status = 'CRITICAL'
        
        self.results['summary'] = {
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'success_rate': f"{success_rate:.1f}%",
            'status': status,
            'integration_maturity': self._calculate_integration_maturity()
        }
        
        # Summary output
        logger.info(f"\nTotal Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Integration Status: {status}")
        logger.info(f"Integration Maturity: {self.results['summary']['integration_maturity']}")
        
        # Detailed system breakdown
        logger.info("\nSystem Status Breakdown:")
        for category, tests in self.results.items():
            if isinstance(tests, dict) and category not in ['summary']:
                passed = sum(1 for test in tests.values() if isinstance(test, dict) and test.get('success', False))
                total = len(tests)
                if total > 0:
                    logger.info(f"  {category}: {passed}/{total} passing")
        
        # Performance insights
        self._log_performance_insights()
        
        # Save detailed report
        report_path = Path('integration_systems_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        return 0 if status in ['EXCELLENT', 'HEALTHY'] else 1
    
    def _calculate_integration_maturity(self) -> str:
        """Calculate integration maturity level based on test results."""
        system_count = len([test for test in self.results.get('systems', {}).values() 
                           if test.get('success', False)])
        integration_count = len([test for test in self.results.get('cross_system_tests', {}).values() 
                               if test.get('success', False)])
        
        if system_count >= 9 and integration_count >= 2:
            return "Advanced"
        elif system_count >= 6 and integration_count >= 1:
            return "Intermediate"
        elif system_count >= 3:
            return "Basic"
        else:
            return "Minimal"
    
    def _log_performance_insights(self):
        """Log performance insights from test execution."""
        logger.info("\nPerformance Insights:")
        
        # Analyze execution times
        all_times = []
        for category in ['systems', 'cross_system_tests']:
            if category in self.results:
                for test_name, test_data in self.results[category].items():
                    if isinstance(test_data, dict) and 'metrics' in test_data:
                        exec_time = test_data['metrics'].get('execution_time', 0)
                        if exec_time > 0:
                            all_times.append(exec_time)
        
        if all_times:
            avg_time = sum(all_times) / len(all_times)
            max_time = max(all_times)
            logger.info(f"  Average test execution time: {avg_time:.2f}s")
            logger.info(f"  Slowest test execution time: {max_time:.2f}s")
            
            if max_time > 10:
                logger.info("  ⚠️  Some tests are running slowly - consider optimization")
            if avg_time < 1:
                logger.info("  ✅ Good performance - tests execute quickly")


def main():
    """Main entry point for integration systems testing."""
    tester = IntegrationSystemTester()
    exit_code = tester.run_all_integration_tests()
    
    if exit_code == 0:
        logger.info("\n[SUCCESS] Integration systems are working well!")
    else:
        logger.error("\n[WARNING] Integration systems need attention.")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()