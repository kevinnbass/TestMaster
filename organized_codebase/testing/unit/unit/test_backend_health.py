#!/usr/bin/env python3
"""
Comprehensive Backend Health Test
==================================
Tests all backend components to ensure they're functional and properly exposed.
"""

import sys
import time
import json
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BackendHealthTester:
    """Comprehensive backend health testing."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'api_endpoints': {},
            'integration_systems': {},
            'state_managers': {},
            'monitoring': {},
            'summary': {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def test_component(self, name: str, test_func) -> bool:
        """Run a single component test."""
        self.total_tests += 1
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result:
                self.passed_tests += 1
                logger.info(f"  [PASS] {name} ({elapsed:.2f}s)")
                return True
            else:
                self.failed_tests += 1
                logger.error(f"  [FAIL] {name} - returned False")
                return False
        except Exception as e:
            self.failed_tests += 1
            logger.error(f"  [FAIL] {name} - {str(e)}")
            return False
    
    def test_core_components(self):
        """Test core TestMaster components."""
        logger.info("\n" + "="*60)
        logger.info("TESTING CORE COMPONENTS")
        logger.info("="*60)
        
        results = {}
        
        # Test orchestration
        def test_orchestration():
            from core.orchestration import TestOrchestrationEngine
            engine = TestOrchestrationEngine()
            return hasattr(engine, 'execute_task')
        
        results['orchestration'] = self.test_component(
            "TestOrchestrationEngine", test_orchestration
        )
        
        # Test observability
        def test_observability():
            from observability import UnifiedObservabilitySystem
            obs = UnifiedObservabilitySystem()
            return hasattr(obs, 'track_session')
        
        results['observability'] = self.test_component(
            "UnifiedObservabilitySystem", test_observability
        )
        
        # Test tools registry
        def test_tools():
            from core.tools import ToolRegistry
            registry = ToolRegistry()
            return hasattr(registry, 'register_tool')
        
        results['tools'] = self.test_component(
            "ToolRegistry", test_tools
        )
        
        self.results['components'] = results
        return all(results.values())
    
    def test_state_managers(self):
        """Test all state management systems."""
        logger.info("\n" + "="*60)
        logger.info("TESTING STATE MANAGERS")
        logger.info("="*60)
        
        results = {}
        
        # Test SharedState
        def test_shared_state():
            from core.shared_state import get_shared_state
            state = get_shared_state()
            state.set("test_key", "test_value")
            return state.get("test_key") == "test_value"
        
        results['shared_state'] = self.test_component(
            "SharedState", test_shared_state
        )
        
        # Test AsyncStateManager
        def test_async_state():
            from core.async_state_manager import AsyncStateManager
            manager = AsyncStateManager()
            return hasattr(manager, 'update_state')
        
        results['async_state'] = self.test_component(
            "AsyncStateManager", test_async_state
        )
        
        # Test UnifiedStateManager
        def test_unified_state():
            from state.unified_state_manager import UnifiedStateManager
            manager = UnifiedStateManager()
            return hasattr(manager, 'set_state')
        
        results['unified_state'] = self.test_component(
            "UnifiedStateManager", test_unified_state
        )
        
        # Test FeatureFlags
        def test_feature_flags():
            from core.feature_flags import FeatureFlags
            flags = FeatureFlags()
            flags.enable_feature("test_feature")
            return flags.is_enabled("test_feature")
        
        results['feature_flags'] = self.test_component(
            "FeatureFlags", test_feature_flags
        )
        
        self.results['state_managers'] = results
        return all(results.values())
    
    def test_integration_systems(self):
        """Test all 11 integration systems."""
        logger.info("\n" + "="*60)
        logger.info("TESTING INTEGRATION SYSTEMS")
        logger.info("="*60)
        
        integration_modules = [
            'automatic_scaling_system',
            'comprehensive_error_recovery',
            'cross_system_communication',
            'distributed_task_queue', 
            'intelligent_caching_layer',
            'load_balancing_system',
            'multi_environment_support',
            'predictive_analytics_engine',
            'realtime_performance_monitoring',
            'resource_optimization_engine',
            'service_mesh_integration'
        ]
        
        results = {}
        
        for module_name in integration_modules:
            def test_integration(mod=module_name):
                module = importlib.import_module(f'integration.{mod}')
                # Check for key classes/functions
                return hasattr(module, '__file__')  # Basic import test
            
            results[module_name] = self.test_component(
                module_name, lambda m=module_name: test_integration(m)
            )
        
        self.results['integration_systems'] = results
        return all(results.values())
    
    def test_monitoring_systems(self):
        """Test monitoring and analytics systems."""
        logger.info("\n" + "="*60)
        logger.info("TESTING MONITORING SYSTEMS")
        logger.info("="*60)
        
        results = {}
        
        # Test RealTimeMonitor
        def test_realtime_monitor():
            from dashboard.dashboard_core.monitor import RealTimeMonitor
            monitor = RealTimeMonitor()
            monitor.start_monitoring()
            time.sleep(0.2)  # Let it collect some data
            metrics = monitor.get_current_metrics()
            monitor.stop_monitoring()
            return metrics is not None
        
        results['realtime_monitor'] = self.test_component(
            "RealTimeMonitor", test_realtime_monitor
        )
        
        # Test MetricsCache
        def test_metrics_cache():
            from dashboard.dashboard_core.cache import MetricsCache
            cache = MetricsCache()
            cache.set("test", {"value": 123})
            return cache.get("test") == {"value": 123}
        
        results['metrics_cache'] = self.test_component(
            "MetricsCache", test_metrics_cache
        )
        
        # Test AnalyticsAggregator
        def test_analytics():
            from dashboard.dashboard_core.analytics_aggregator import AnalyticsAggregator
            aggregator = AnalyticsAggregator()
            return hasattr(aggregator, 'aggregate_metrics')
        
        results['analytics_aggregator'] = self.test_component(
            "AnalyticsAggregator", test_analytics
        )
        
        self.results['monitoring'] = results
        return all(results.values())
    
    def test_api_endpoints(self):
        """Test API blueprint availability and routes."""
        logger.info("\n" + "="*60)
        logger.info("TESTING API ENDPOINTS")
        logger.info("="*60)
        
        api_blueprints = [
            ('dashboard.api.performance', 'performance_bp'),
            ('dashboard.api.analytics', 'analytics_bp'),
            ('dashboard.api.workflow', 'workflow_bp'),
            ('dashboard.api.tests', 'tests_bp'),
            ('dashboard.api.refactor', 'refactor_bp'),
            ('dashboard.api.llm', 'llm_bp'),
            ('dashboard.api.health', 'HealthCheckAPI'),
            ('dashboard.api.monitoring', 'MonitoringAPI'),
            ('dashboard.api.intelligence', 'IntelligenceAPI'),
            ('dashboard.api.test_generation', 'TestGenerationAPI'),
            ('dashboard.api.security', 'SecurityAPI'),
            ('dashboard.api.coverage', 'CoverageAPI'),
            ('dashboard.api.flow_optimization', 'FlowOptimizationAPI'),
            ('dashboard.api.quality_assurance', 'QualityAssuranceAPI'),
            ('dashboard.api.telemetry', 'TelemetryAPI'),
            ('dashboard.api.async_processing', 'AsyncProcessingAPI'),
            ('dashboard.api.real_codebase_scanner', 'RealCodebaseScanner'),
            ('dashboard.api.crew_orchestration', 'crew_orchestration_bp'),
            ('dashboard.api.swarm_orchestration', 'swarm_orchestration_bp'),
            ('dashboard.api.observability', 'observability_bp'),
            ('dashboard.api.production_deployment', 'production_bp'),
            ('dashboard.api.enhanced_telemetry', 'enhanced_telemetry_bp'),
            ('dashboard.api.backend_health_monitor', 'health_monitor_bp'),
            ('dashboard.api.frontend_data_contracts', 'data_contract_bp'),
            ('dashboard.api.enhanced_analytics', 'enhanced_analytics_bp'),
            ('dashboard.api.orchestration_flask', 'orchestration_bp'),
            ('dashboard.api.phase2_api', 'phase2_bp')
        ]
        
        results = {}
        route_counts = {}
        
        for module_path, blueprint_name in api_blueprints:
            def test_blueprint(mod=module_path, bp=blueprint_name):
                try:
                    module = importlib.import_module(mod)
                    blueprint = getattr(module, bp, None)
                    if blueprint:
                        # Count routes if it's a Flask blueprint
                        if hasattr(blueprint, 'deferred_functions'):
                            route_count = len(blueprint.deferred_functions)
                            route_counts[bp] = route_count
                            return route_count > 0
                        return True
                    return False
                except Exception:
                    return False
            
            results[blueprint_name] = self.test_component(
                f"{blueprint_name} ({module_path})", 
                lambda m=module_path, b=blueprint_name: test_blueprint(m, b)
            )
        
        # Log route counts
        logger.info("\nRoute counts:")
        for bp, count in route_counts.items():
            if count > 0:
                logger.info(f"  {bp}: {count} routes")
        
        self.results['api_endpoints'] = results
        return all(results.values())
    
    def test_intelligence_agents(self):
        """Test intelligence agent systems."""
        logger.info("\n" + "="*60)
        logger.info("TESTING INTELLIGENCE AGENTS")
        logger.info("="*60)
        
        results = {}
        
        # Test consensus engine
        def test_consensus():
            from testmaster.intelligence.consensus import ConsensusEngine
            engine = ConsensusEngine()
            return hasattr(engine, 'reach_consensus')
        
        results['consensus_engine'] = self.test_component(
            "ConsensusEngine", test_consensus
        )
        
        # Test security intelligence
        def test_security():
            from testmaster.intelligence.security import SecurityIntelligenceAgent
            agent = SecurityIntelligenceAgent()
            return hasattr(agent, 'scan_vulnerabilities')
        
        results['security_intelligence'] = self.test_component(
            "SecurityIntelligenceAgent", test_security
        )
        
        # Test optimization agent
        def test_optimization():
            from testmaster.intelligence.optimization import MultiObjectiveOptimizationAgent
            agent = MultiObjectiveOptimizationAgent()
            return hasattr(agent, 'optimize')
        
        results['optimization_agent'] = self.test_component(
            "MultiObjectiveOptimizationAgent", test_optimization
        )
        
        self.results['intelligence_agents'] = results
        return len([v for v in results.values() if v]) >= 2  # At least 2 should work
    
    def generate_report(self):
        """Generate comprehensive health report."""
        logger.info("\n" + "="*60)
        logger.info("BACKEND HEALTH REPORT")
        logger.info("="*60)
        
        # Calculate success rates
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        self.results['summary'] = {
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'success_rate': f"{success_rate:.1f}%",
            'status': 'HEALTHY' if success_rate >= 80 else 'DEGRADED' if success_rate >= 50 else 'CRITICAL'
        }
        
        # Summary by category
        logger.info(f"\nTotal Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"System Status: {self.results['summary']['status']}")
        
        # Component breakdown
        logger.info("\nComponent Status:")
        for category, tests in self.results.items():
            if isinstance(tests, dict) and category != 'summary':
                passed = sum(1 for v in tests.values() if v)
                total = len(tests)
                logger.info(f"  {category}: {passed}/{total} passing")
        
        # Save report
        report_path = Path('backend_health_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        return self.results['summary']['status']
    
    def run_all_tests(self):
        """Run all backend health tests."""
        logger.info("Starting comprehensive backend health check...")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Run test suites
        self.test_core_components()
        self.test_state_managers()
        self.test_integration_systems()
        self.test_monitoring_systems()
        self.test_api_endpoints()
        self.test_intelligence_agents()
        
        # Generate report
        status = self.generate_report()
        
        # Return exit code
        return 0 if status == 'HEALTHY' else 1


def main():
    """Main entry point."""
    tester = BackendHealthTester()
    exit_code = tester.run_all_tests()
    
    if exit_code == 0:
        logger.info("\n[SUCCESS] Backend is healthy!")
    else:
        logger.error("\n[WARNING] Backend has issues that need attention.")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()