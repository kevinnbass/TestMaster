"""
Test script for TestMaster Async Processing System

Comprehensive testing of all async processing components:
- AsyncExecutor: Priority-based async task execution
- ThreadPoolManager: Advanced thread pool management
- AsyncMonitor: Real-time async operation monitoring
- ConcurrentScheduler: Task scheduling with multiple strategies
- AsyncStateManager: Hierarchical state management
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from testmaster.async_processing import (
    # Core components
    AsyncExecutor, ThreadPoolManager, AsyncMonitor, 
    ConcurrentScheduler, AsyncStateManager,
    
    # Convenience functions
    async_execute, submit_task, track_async_execution,
    schedule_task, async_context,
    
    # Enums and configs
    TaskPriority, ScheduleType, StateScope, ScheduleConfig,
    
    # Global instances
    get_async_executor, get_thread_pool_manager, get_async_monitor,
    get_concurrent_scheduler, get_async_state_manager,
    
    # Utilities
    is_async_enabled, configure_async_processing, shutdown_async_processing
)

class AsyncProcessingTest:
    """Comprehensive test suite for async processing."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    async def run_all_tests(self):
        """Run all async processing tests."""
        print("=" * 60)
        print("TestMaster Async Processing System Test")
        print("=" * 60)
        
        # Check if async processing is enabled
        if not is_async_enabled():
            print("X Async processing is disabled")
            return
        
        print("✓ Async processing is enabled")
        
        # Test individual components
        await self.test_async_executor()
        await self.test_thread_pool_manager()
        await self.test_async_monitor()
        await self.test_concurrent_scheduler()
        await self.test_async_state_manager()
        await self.test_integration()
        
        # Display results
        self.display_results()
    
    async def test_async_executor(self):
        """Test AsyncExecutor functionality."""
        print("\n[*] Testing AsyncExecutor...")
        
        try:
            executor = get_async_executor()
            
            # Test async task submission
            async def test_async_task():
                await asyncio.sleep(0.1)
                return "async_result"
            
            task_id = executor.submit_async_task(
                test_async_task(),
                priority=TaskPriority.HIGH,
                timeout_seconds=5.0,
                metadata={"test": "async_executor"}
            )
            
            print(f"   [+] Submitted async task: {task_id}")
            
            # Test sync task submission
            def test_sync_task():
                time.sleep(0.1)
                return "sync_result"
            
            sync_task_id = executor.submit_sync_task(
                test_sync_task,
                priority=TaskPriority.NORMAL,
                metadata={"test": "sync_task"}
            )
            
            print(f"   [+] Submitted sync task: {sync_task_id}")
            
            # Wait a bit for tasks to complete
            await asyncio.sleep(1.0)
            
            # Check statistics
            stats = executor.get_executor_statistics()
            print(f"   [i] Executor stats: {stats['tasks_executed']} executed, {stats['success_rate']}% success")
            
            self.test_results['async_executor'] = True
            
        except Exception as e:
            print(f"   [!] AsyncExecutor test failed: {e}")
            self.test_results['async_executor'] = False
    
    async def test_thread_pool_manager(self):
        """Test ThreadPoolManager functionality."""
        print("\n[*] Testing ThreadPoolManager...")
        
        try:
            manager = get_thread_pool_manager()
            
            # Test task submission to different pools
            def cpu_task(n):
                return sum(i * i for i in range(n))
            
            def io_task():
                time.sleep(0.1)
                return "io_completed"
            
            # Submit CPU-intensive task
            cpu_task_id = manager.submit_task(
                "cpu_intensive", cpu_task, 1000,
                metadata={"type": "cpu_test"}
            )
            print(f"   [+] Submitted CPU task: {cpu_task_id}")
            
            # Submit IO-intensive task
            io_task_id = manager.submit_task(
                "io_intensive", io_task,
                metadata={"type": "io_test"}
            )
            print(f"   [+] Submitted IO task: {io_task_id}")
            
            # Wait for completion
            time.sleep(0.5)
            
            # Check pool status
            status = manager.get_pool_status()
            print(f"   [i] Pool status: {status['total_pools']} pools, {status['success_rate']:.1f}% success")
            
            self.test_results['thread_pool_manager'] = True
            
        except Exception as e:
            print(f"   [!] ThreadPoolManager test failed: {e}")
            self.test_results['thread_pool_manager'] = False
    
    async def test_async_monitor(self):
        """Test AsyncMonitor functionality."""
        print("\n[*] Testing AsyncMonitor...")
        
        try:
            monitor = get_async_monitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Test task tracking
            task_id = "test_monitor_task"
            task_info = monitor.track_task_start(
                task_id, "Monitor Test Task", "test_component",
                metadata={"test": "monitoring"}
            )
            
            print(f"   [+] Started tracking task: {task_id}")
            
            # Simulate task execution
            await asyncio.sleep(0.1)
            monitor.track_task_running(task_id)
            
            await asyncio.sleep(0.1)
            monitor.track_task_completion(task_id, success=True)
            
            print(f"   [+] Completed task tracking")
            
            # Check performance summary
            summary = monitor.get_performance_summary()
            print(f"   [i] Monitor summary: {summary['total_tasks_tracked']} tracked, {summary['success_rate']}% success")
            
            self.test_results['async_monitor'] = True
            
        except Exception as e:
            print(f"   [!] AsyncMonitor test failed: {e}")
            self.test_results['async_monitor'] = False
    
    async def test_concurrent_scheduler(self):
        """Test ConcurrentScheduler functionality."""
        print("\n[*] Testing ConcurrentScheduler...")
        
        try:
            scheduler = get_concurrent_scheduler()
            scheduler.start_scheduler()
            
            # Test once-off task
            def once_task():
                print("   [>] Once-off task executed")
                return "once_complete"
            
            once_config = ScheduleConfig(
                schedule_type=ScheduleType.ONCE,
                delay_seconds=0.1
            )
            
            once_task_id = scheduler.schedule_task(
                "Once Task", once_task, once_config,
                metadata={"test": "once_task"}
            )
            
            print(f"   [+] Scheduled once task: {once_task_id}")
            
            # Test interval task
            interval_count = 0
            def interval_task():
                nonlocal interval_count
                interval_count += 1
                print(f"   [>] Interval task executed (count: {interval_count})")
                return f"interval_{interval_count}"
            
            interval_config = ScheduleConfig(
                schedule_type=ScheduleType.INTERVAL,
                interval_seconds=0.2,
                max_executions=3
            )
            
            interval_task_id = scheduler.schedule_task(
                "Interval Task", interval_task, interval_config,
                metadata={"test": "interval_task"}
            )
            
            print(f"   [+] Scheduled interval task: {interval_task_id}")
            
            # Wait for tasks to execute
            await asyncio.sleep(1.0)
            
            # Check scheduler statistics
            stats = scheduler.get_scheduler_statistics()
            print(f"   [i] Scheduler stats: {stats['scheduled_tasks']} scheduled, {stats['success_rate']}% success")
            
            # Cancel interval task
            scheduler.cancel_task(interval_task_id)
            print(f"   [+] Cancelled interval task")
            
            self.test_results['concurrent_scheduler'] = True
            
        except Exception as e:
            print(f"   [!] ConcurrentScheduler test failed: {e}")
            self.test_results['concurrent_scheduler'] = False
    
    async def test_async_state_manager(self):
        """Test AsyncStateManager functionality."""
        print("\n[*] Testing AsyncStateManager...")
        
        try:
            manager = get_async_state_manager()
            
            # Test hierarchical state management
            async with manager.async_context(scope=StateScope.SESSION, 
                                           metadata={"test": "session"}) as session_ctx:
                
                # Set states at different scopes
                manager.set_state("global_key", "global_value", StateScope.GLOBAL)
                manager.set_state("session_key", "session_value", StateScope.SESSION)
                manager.set_state("context_key", "context_value", StateScope.CONTEXT)
                
                print(f"   [+] Set states in session context: {session_ctx}")
                
                # Test hierarchical lookup
                global_val = manager.get_state("global_key")
                session_val = manager.get_state("session_key")
                context_val = manager.get_state("context_key")
                
                print(f"   [i] Retrieved values: global={global_val}, session={session_val}, context={context_val}")
                
                # Test nested context
                async with manager.async_context(parent_id=session_ctx, 
                                               scope=StateScope.TASK,
                                               metadata={"test": "nested"}) as task_ctx:
                    
                    manager.set_state("task_key", "task_value", StateScope.TASK)
                    
                    # Should access parent context values
                    inherited_val = manager.get_state("session_key")
                    task_val = manager.get_state("task_key")
                    
                    print(f"   [i] Nested context: inherited={inherited_val}, task={task_val}")
            
            # Check state summary
            summary = manager.get_state_summary()
            print(f"   [i] State summary: {summary['total_states']} states, {summary['active_contexts']} active contexts")
            
            self.test_results['async_state_manager'] = True
            
        except Exception as e:
            print(f"   [!] AsyncStateManager test failed: {e}")
            self.test_results['async_state_manager'] = False
    
    async def test_integration(self):
        """Test integrated functionality across components."""
        print("\n[*] Testing Integration...")
        
        try:
            # Test convenience functions
            async def integrated_task():
                state_manager = get_async_state_manager()
                
                async with state_manager.async_context(metadata={"test": "integration"}) as ctx:
                    # Use async monitoring
                    with track_async_execution("Integration Task", "integration_test"):
                        # Set some state
                        state_manager.set_state("integration_key", "integration_value")
                        
                        # Simulate work
                        await asyncio.sleep(0.1)
                        
                        # Get state
                        value = state_manager.get_state("integration_key")
                        return f"integration_complete_{value}"
            
            # Execute using convenience function
            task_id = async_execute(
                integrated_task(),
                priority=TaskPriority.HIGH,
                metadata={"test": "integration"}
            )
            
            print(f"   [+] Executed integrated task: {task_id}")
            
            # Wait for completion
            await asyncio.sleep(0.5)
            
            # Check all component statistics
            executor_stats = get_async_executor().get_executor_statistics()
            monitor_stats = get_async_monitor().get_performance_summary()
            pool_stats = get_thread_pool_manager().get_pool_status()
            scheduler_stats = get_concurrent_scheduler().get_scheduler_statistics()
            state_stats = get_async_state_manager().get_state_summary()
            
            print(f"   [i] Integration stats:")
            print(f"      - Executor: {executor_stats['tasks_executed']} tasks")
            print(f"      - Monitor: {monitor_stats['total_tasks_tracked']} tracked")
            print(f"      - Pools: {pool_stats['completed_tasks']} completed")
            print(f"      - Scheduler: {scheduler_stats['total_executions']} executions")
            print(f"      - State: {state_stats['state_operations']} operations")
            
            self.test_results['integration'] = True
            
        except Exception as e:
            print(f"   [!] Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{component.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All async processing tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

async def main():
    """Main test execution."""
    try:
        # Initialize feature flags first
        from testmaster.core.feature_flags import FeatureFlags
        FeatureFlags.initialize("testmaster_config.yaml")
        
        # Configure async processing
        configure_async_processing(
            max_workers=10,
            enable_monitoring=True,
            enable_scheduling=True
        )
        
        # Run tests
        test_suite = AsyncProcessingTest()
        await test_suite.run_all_tests()
        
    finally:
        # Cleanup
        print("\nCleaning up async processing...")
        shutdown_async_processing()
        print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())