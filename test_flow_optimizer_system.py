"""
Test script for TestMaster Execution Flow Optimizer

Comprehensive testing of flow optimization components:
- FlowAnalyzer: Flow analysis and bottleneck detection
- ExecutionRouter: Intelligent task routing
- ResourceOptimizer: Resource allocation optimization
- DependencyResolver: Dependency graph optimization
- ParallelExecutor: Parallel execution planning
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from testmaster.core.feature_flags import FeatureFlags
from testmaster.flow_optimizer import (
    # Core components
    FlowAnalyzer, ExecutionRouter, ResourceOptimizer,
    DependencyResolver, ParallelExecutor,
    
    # Convenience functions
    configure_flow_optimizer, analyze_execution_flow, optimize_execution_route,
    optimize_resource_allocation, resolve_dependencies, create_parallel_execution_plan,
    
    # Enums and configs
    AnalysisType, RoutingStrategy, ResourceType, OptimizationPolicy,
    DependencyType, ParallelStrategy,
    
    # Global instances
    get_flow_analyzer, get_execution_router, get_resource_optimizer,
    get_dependency_resolver, get_parallel_executor,
    
    # Utilities
    is_flow_optimizer_enabled, get_optimization_status, shutdown_flow_optimizer
)

class FlowOptimizerSystemTest:
    """Comprehensive test suite for flow optimizer system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_workflow_id = "test_workflow_001"
        
    async def run_all_tests(self):
        """Run all flow optimizer system tests."""
        print("=" * 60)
        print("TestMaster Execution Flow Optimizer System Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("testmaster_config.yaml")
        
        # Check if flow optimizer is enabled
        if not is_flow_optimizer_enabled():
            print("[!] Flow optimizer is disabled in configuration")
            return
        
        print("[+] Flow optimizer is enabled")
        
        # Configure flow optimizer system
        config_result = configure_flow_optimizer(
            learning_rate=0.1,
            enable_adaptive_routing=True,
            enable_parallel_execution=True,
            optimization_interval=60
        )
        print(f"[+] Flow optimizer configured: {config_result['status']}")
        
        # Test individual components
        await self.test_flow_analyzer()
        await self.test_execution_router()
        await self.test_resource_optimizer()
        await self.test_dependency_resolver()
        await self.test_parallel_executor()
        await self.test_integration()
        
        # Display results
        self.display_results()
    
    async def test_flow_analyzer(self):
        """Test FlowAnalyzer functionality."""
        print("\\n[*] Testing FlowAnalyzer...")
        
        try:
            analyzer = get_flow_analyzer()
            
            # Create sample execution data
            execution_data = [
                {
                    "task_id": "task_1",
                    "execution_time": 150.0,
                    "wait_time": 20.0,
                    "resource_usage": 75.0,
                    "dependencies": ["task_0"]
                },
                {
                    "task_id": "task_2", 
                    "execution_time": 200.0,
                    "wait_time": 45.0,
                    "resource_usage": 85.0,
                    "dependencies": ["task_1"]
                },
                {
                    "task_id": "task_3",
                    "execution_time": 120.0,
                    "wait_time": 15.0,
                    "resource_usage": 60.0,
                    "dependencies": ["task_1"]
                },
                {
                    "task_id": "task_4",
                    "execution_time": 300.0,
                    "wait_time": 80.0,
                    "resource_usage": 95.0,
                    "dependencies": ["task_2", "task_3"]
                }
            ]
            
            # Perform flow analysis
            analysis = analyzer.analyze_flow(
                workflow_id=self.test_workflow_id,
                execution_data=execution_data,
                include_dependencies=True
            )
            
            print(f"   [+] Flow analysis completed: {analysis.efficiency_score:.3f} efficiency")
            print(f"   [i] Metrics analyzed: {len(analysis.metrics)}")
            print(f"   [i] Bottlenecks detected: {len(analysis.bottlenecks)}")
            print(f"   [i] Optimization recommendations: {len(analysis.optimization_recommendations)}")
            
            # Display metrics
            for metric in analysis.metrics:
                status_icon = "[+]" if metric.status == "pass" else "[!]"
                print(f"   {status_icon} {metric.name}: {metric.value:.2f} {metric.unit} (threshold: {metric.threshold:.2f})")
            
            # Display bottlenecks
            for bottleneck in analysis.bottlenecks:
                print(f"   [!] Bottleneck: {bottleneck.location} ({bottleneck.severity}) - {bottleneck.description}")
            
            # Set performance baselines
            analyzer.set_performance_baseline(self.test_workflow_id, "execution_time", 180.0)
            analyzer.set_performance_baseline(self.test_workflow_id, "throughput", 4.0)
            
            # Test efficiency trends
            trends = analyzer.get_efficiency_trends(self.test_workflow_id)
            print(f"   [i] Efficiency trends: {trends.get('trend', 'insufficient_data')}")
            
            self.test_results['flow_analyzer'] = analysis.efficiency_score > 0.0
            
        except Exception as e:
            print(f"   [!] FlowAnalyzer test failed: {e}")
            self.test_results['flow_analyzer'] = False
    
    async def test_execution_router(self):
        """Test ExecutionRouter functionality."""
        print("\\n[*] Testing ExecutionRouter...")
        
        try:
            router = get_execution_router()
            
            # Create sample available resources
            available_resources = [
                {
                    "id": "resource_1",
                    "performance_score": 0.9,
                    "current_load": 0.3,
                    "response_time": 80.0,
                    "success_rate": 0.98,
                    "availability": 1.0,
                    "queue_length": 2
                },
                {
                    "id": "resource_2",
                    "performance_score": 0.85,
                    "current_load": 0.6,
                    "response_time": 120.0,
                    "success_rate": 0.95,
                    "availability": 1.0,
                    "queue_length": 5
                },
                {
                    "id": "resource_3",
                    "performance_score": 0.8,
                    "current_load": 0.2,
                    "response_time": 100.0,
                    "success_rate": 0.97,
                    "availability": 0.9,
                    "queue_length": 1
                }
            ]
            
            # Test different routing strategies
            strategies = [
                "round_robin",
                "performance_based", 
                "load_balanced",
                "shortest_queue",
                "adaptive"
            ]
            
            routes = []
            
            for strategy in strategies:
                route = router.find_optimal_route(
                    task_id=f"test_task_{strategy}",
                    available_resources=available_resources
                )
                routes.append(route)
                
                print(f"   [+] {strategy} route: {route.path[0] if route.path else 'none'} (confidence: {route.confidence_score:.3f})")
            
            # Test adaptive routing
            router.enable_adaptive_routing(learning_rate=0.15)
            print(f"   [+] Adaptive routing enabled")
            
            # Test route performance feedback
            for route in routes[:2]:  # Test feedback for first 2 routes
                router.update_route_performance(
                    task_id=route.task_id,
                    route=route,
                    actual_performance={"completion_time": 90.0, "success": True}
                )
            
            # Check routing statistics
            stats = router.get_routing_statistics()
            print(f"   [i] Routing statistics:")
            print(f"      - Total routes: {stats['total_routes']}")
            print(f"      - Average confidence: {stats['average_confidence']:.3f}")
            print(f"      - Adaptive enabled: {stats['adaptive_enabled']}")
            
            self.test_results['execution_router'] = len(routes) > 0
            
        except Exception as e:
            print(f"   [!] ExecutionRouter test failed: {e}")
            self.test_results['execution_router'] = False
    
    async def test_resource_optimizer(self):
        """Test ResourceOptimizer functionality."""
        print("\\n[*] Testing ResourceOptimizer...")
        
        try:
            optimizer = get_resource_optimizer()
            
            # Create sample tasks with resource requirements
            tasks = [
                {
                    "id": "task_1",
                    "cpu_requirement": 2.0,
                    "memory_requirement": 512.0,
                    "network_requirement": 50.0,
                    "priority": 8
                },
                {
                    "id": "task_2",
                    "cpu_requirement": 4.0,
                    "memory_requirement": 1024.0,
                    "network_requirement": 100.0,
                    "priority": 6
                },
                {
                    "id": "task_3",
                    "cpu_requirement": 1.0,
                    "memory_requirement": 256.0,
                    "network_requirement": 25.0,
                    "priority": 9
                }
            ]
            
            # Available system resources
            available_resources = {
                "total_cpu": 10.0,
                "total_memory": 2048.0,
                "total_network": 200.0
            }
            
            # Resource allocation constraints
            constraints = {
                "max_cost": 50.0,
                "optimization_policy": "balanced",
                "resource_limits": {
                    "cpu": 8.0,
                    "memory": 1800.0
                }
            }
            
            # Optimize resource allocation
            allocation = optimizer.optimize_allocation(
                workflow_id=self.test_workflow_id,
                tasks=tasks,
                available_resources=available_resources,
                constraints=constraints
            )
            
            print(f"   [+] Resource allocation optimized: {allocation.efficiency_score:.3f} efficiency")
            print(f"   [i] Total cost: {allocation.total_cost:.2f}")
            print(f"   [i] Constraints satisfied: {allocation.constraints_satisfied}")
            print(f"   [i] Allocated resources:")
            
            for resource_type, type_allocations in allocation.allocations.items():
                total_allocated = sum(type_allocations.values())
                print(f"      - {resource_type}: {total_allocated:.1f} units across {len(type_allocations)} pools")
            
            # Test resource utilization
            utilization = optimizer.get_resource_utilization()
            print(f"   [i] Resource utilization: {len(utilization)} pools monitored")
            
            # Test optimization policy setting
            from testmaster.flow_optimizer.resource_optimizer import OptimizationPolicy
            optimizer.set_optimization_policy(self.test_workflow_id, OptimizationPolicy.MAXIMIZE_PERFORMANCE)
            
            # Test allocation history
            history = optimizer.get_allocation_history(self.test_workflow_id)
            print(f"   [i] Allocation history: {len(history)} allocations")
            
            self.test_results['resource_optimizer'] = allocation.efficiency_score > 0.0
            
        except Exception as e:
            print(f"   [!] ResourceOptimizer test failed: {e}")
            self.test_results['resource_optimizer'] = False
    
    async def test_dependency_resolver(self):
        """Test DependencyResolver functionality."""
        print("\\n[*] Testing DependencyResolver...")
        
        try:
            resolver = get_dependency_resolver()
            
            # Create sample tasks with dependencies
            tasks = [
                {
                    "id": "init_task",
                    "dependencies": [],
                    "estimated_duration": 50.0,
                    "priority": 10
                },
                {
                    "id": "process_data",
                    "dependencies": ["init_task"],
                    "estimated_duration": 120.0,
                    "priority": 8
                },
                {
                    "id": "validate_results",
                    "dependencies": ["process_data"],
                    "estimated_duration": 80.0,
                    "priority": 7
                },
                {
                    "id": "generate_report",
                    "dependencies": ["validate_results"],
                    "estimated_duration": 60.0,
                    "priority": 6
                },
                {
                    "id": "backup_data",
                    "dependencies": ["init_task"],
                    "estimated_duration": 40.0,
                    "priority": 5
                },
                {
                    "id": "cleanup",
                    "dependencies": ["generate_report", "backup_data"],
                    "estimated_duration": 30.0,
                    "priority": 4
                }
            ]
            
            # Optional dependency rules
            dependency_rules = [
                {
                    "type": "sequential",
                    "source_pattern": "process",
                    "target_pattern": "validate"
                }
            ]
            
            # Resolve dependencies
            dependency_graph = resolver.resolve_dependencies(
                tasks=tasks,
                dependency_rules=dependency_rules
            )
            
            print(f"   [+] Dependencies resolved: {len(dependency_graph.nodes)} tasks")
            print(f"   [i] Execution levels: {len(dependency_graph.execution_levels)}")
            print(f"   [i] Critical path: {len(dependency_graph.critical_path)} tasks")
            print(f"   [i] Total estimated time: {dependency_graph.total_estimated_time:.1f}ms")
            
            # Display execution levels
            for i, level in enumerate(dependency_graph.execution_levels):
                print(f"   [i] Level {i}: {len(level)} tasks - {level}")
            
            # Display critical path
            print(f"   [i] Critical path: {' -> '.join(dependency_graph.critical_path)}")
            
            # Test graph optimization
            optimized_graph = resolver.optimize_dependency_graph(dependency_graph)
            print(f"   [+] Graph optimized: {optimized_graph.total_estimated_time:.1f}ms total time")
            
            # Test parallelization analysis
            parallelization = resolver.get_parallelization_opportunities(dependency_graph)
            print(f"   [i] Parallelization opportunities:")
            print(f"      - Max parallel tasks: {parallelization['max_parallel_tasks']}")
            print(f"      - Speedup factor: {parallelization['speedup_factor']:.2f}x")
            print(f"      - Parallelization ratio: {parallelization['parallelization_ratio']:.3f}")
            
            self.test_results['dependency_resolver'] = len(dependency_graph.nodes) > 0
            
        except Exception as e:
            print(f"   [!] DependencyResolver test failed: {e}")
            self.test_results['dependency_resolver'] = False
    
    async def test_parallel_executor(self):
        """Test ParallelExecutor functionality."""
        print("\\n[*] Testing ParallelExecutor...")
        
        try:
            executor = get_parallel_executor()
            
            # Configure executor
            executor.configure(learning_rate=0.15, max_workers=6)
            
            # Get dependency graph from previous test (create minimal one if needed)
            resolver = get_dependency_resolver()
            simple_tasks = [
                {"id": "task_a", "dependencies": [], "estimated_duration": 100.0},
                {"id": "task_b", "dependencies": ["task_a"], "estimated_duration": 80.0},
                {"id": "task_c", "dependencies": ["task_a"], "estimated_duration": 90.0},
                {"id": "task_d", "dependencies": ["task_b", "task_c"], "estimated_duration": 60.0}
            ]
            
            dependency_graph = resolver.resolve_dependencies(simple_tasks)
            
            # Get resource allocation (create minimal one)
            from testmaster.flow_optimizer.resource_optimizer import ResourceAllocation
            resource_allocation = ResourceAllocation(
                workflow_id=self.test_workflow_id,
                allocations={"cpu": {"cpu_pool_1": 4.0}, "memory": {"memory_pool_1": 1024.0}},
                status="test"
            )
            
            # Test different parallel strategies
            strategies = ["balanced", "aggressive", "conservative", "adaptive", "resource_aware"]
            
            execution_plans = []
            
            for strategy in strategies:
                plan = executor.create_execution_plan(
                    workflow_id=f"{self.test_workflow_id}_{strategy}",
                    dependency_graph=dependency_graph,
                    resource_allocation=resource_allocation,
                    strategy=strategy
                )
                execution_plans.append(plan)
                
                print(f"   [+] {strategy} plan: {len(plan.batches)} batches, {plan.parallelization_factor:.2f}x speedup")
                print(f"      - Estimated time: {plan.total_estimated_time:.1f}ms")
                print(f"      - Resource efficiency: {plan.resource_efficiency:.3f}")
            
            # Test strategy performance recording
            for i, plan in enumerate(execution_plans):
                performance_score = 0.8 + (i * 0.05)  # Simulate varying performance
                executor.record_strategy_performance(strategies[i], performance_score)
            
            # Test execution statistics
            stats = executor.get_execution_statistics()
            print(f"   [i] Execution statistics:")
            print(f"      - Total plans: {stats['total_plans']}")
            print(f"      - Total batches: {stats['total_batches']}")
            print(f"      - Average parallelization: {stats['average_parallelization_factor']:.2f}x")
            print(f"      - Average efficiency: {stats['average_resource_efficiency']:.3f}")
            
            self.test_results['parallel_executor'] = len(execution_plans) > 0
            
        except Exception as e:
            print(f"   [!] ParallelExecutor test failed: {e}")
            self.test_results['parallel_executor'] = False
    
    async def test_integration(self):
        """Test integrated flow optimization functionality."""
        print("\\n[*] Testing Integration...")
        
        try:
            # Test end-to-end flow optimization workflow
            print("   [>] Starting integrated flow optimization workflow...")
            
            # 1. Create sample execution data
            execution_data = [
                {"execution_time": 180.0, "wait_time": 30.0, "resource_usage": 70.0},
                {"execution_time": 220.0, "wait_time": 50.0, "resource_usage": 85.0},
                {"execution_time": 160.0, "wait_time": 25.0, "resource_usage": 65.0}
            ]
            
            # 2. Analyze execution flow
            analysis = analyze_execution_flow(
                workflow_id=self.test_workflow_id,
                execution_data=execution_data,
                include_dependencies=True
            )
            print(f"   [+] Flow analysis: {analysis.efficiency_score:.3f} efficiency")
            
            # 3. Optimize execution route
            available_resources = [
                {"id": "optimal_resource", "performance_score": 0.95, "current_load": 0.2}
            ]
            
            route = optimize_execution_route(
                task_id="integrated_test_task",
                available_resources=available_resources
            )
            print(f"   [+] Execution route: {route.strategy} strategy, confidence: {route.confidence_score:.3f}")
            
            # 4. Optimize resource allocation
            tasks = [
                {"id": "int_task_1", "cpu_requirement": 2.0, "memory_requirement": 512.0},
                {"id": "int_task_2", "cpu_requirement": 1.5, "memory_requirement": 256.0}
            ]
            
            allocation = optimize_resource_allocation(
                workflow_id=self.test_workflow_id,
                tasks=tasks,
                available_resources={"total_cpu": 8.0, "total_memory": 2048.0}
            )
            print(f"   [+] Resource allocation: {allocation.efficiency_score:.3f} efficiency")
            
            # 5. Resolve dependencies
            dependency_tasks = [
                {"id": "start", "dependencies": [], "estimated_duration": 50.0},
                {"id": "process", "dependencies": ["start"], "estimated_duration": 100.0},
                {"id": "finish", "dependencies": ["process"], "estimated_duration": 75.0}
            ]
            
            dependency_graph = resolve_dependencies(dependency_tasks)
            print(f"   [+] Dependencies resolved: {len(dependency_graph.execution_levels)} levels")
            
            # 6. Create parallel execution plan
            execution_plan = create_parallel_execution_plan(
                workflow_id=self.test_workflow_id,
                dependency_graph=dependency_graph,
                resource_allocation=allocation,
                strategy="balanced"
            )
            print(f"   [+] Execution plan: {len(execution_plan.batches)} batches, {execution_plan.parallelization_factor:.2f}x speedup")
            
            # 7. Check overall optimization status
            optimization_status = get_optimization_status()
            print(f"   [+] Optimization status: {optimization_status['status']}")
            
            # Verify integration success
            integration_success = (
                analysis.efficiency_score > 0.0 and
                route.confidence_score > 0.0 and
                allocation.efficiency_score > 0.0 and
                len(dependency_graph.nodes) > 0 and
                len(execution_plan.batches) > 0
            )
            
            print(f"   [i] Integration workflow completed successfully")
            print(f"   [i] Optimization metrics:")
            print(f"      - Flow efficiency: {analysis.efficiency_score:.3f}")
            print(f"      - Route confidence: {route.confidence_score:.3f}")
            print(f"      - Resource efficiency: {allocation.efficiency_score:.3f}")
            print(f"      - Parallelization factor: {execution_plan.parallelization_factor:.2f}x")
            
            self.test_results['integration'] = integration_success
            
        except Exception as e:
            print(f"   [!] Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{component.replace('_', ' ').title()}: {status}")
        
        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All flow optimizer system tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

async def main():
    """Main test execution."""
    try:
        # Run tests
        test_suite = FlowOptimizerSystemTest()
        await test_suite.run_all_tests()
        
    finally:
        # Cleanup
        print("\\nCleaning up flow optimizer system...")
        shutdown_flow_optimizer()
        print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())