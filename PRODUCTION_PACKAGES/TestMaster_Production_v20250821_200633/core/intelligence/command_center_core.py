"""
Command Center Core - Master Intelligence Orchestrator
======================================================

Streamlined Intelligence Command Center implementing enterprise orchestration
patterns for coordinating intelligence frameworks into a unified, autonomous,
and self-coordinating ecosystem.

This module contains the main IntelligenceCommandCenter class with clean
separation of concerns, delegating specialized functionality to dedicated
modules while maintaining centralized coordination and control.

Author: Agent A - Hours 30-40
Created: 2025-08-22
Module: command_center_core.py (250 lines)
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .orchestration_types import (
    FrameworkType, OrchestrationPriority, OrchestrationStatus,
    OrchestrationTask, FrameworkHealthStatus
)
from .framework_controllers import (
    FrameworkController, AnalyticsFrameworkController, MLFrameworkController
)
from .resource_management import ResourceManager


class IntelligenceCommandCenter:
    """
    Master orchestration system coordinating intelligence frameworks
    into a unified, autonomous, and self-coordinating ecosystem.
    
    Implements enterprise patterns:
    - Strategy Pattern: Pluggable framework controllers
    - Observer Pattern: Health monitoring and event coordination  
    - Command Pattern: Encapsulated orchestration tasks
    - Factory Pattern: Dynamic component creation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Intelligence Command Center with enterprise configuration"""
        self.config = config or self._get_default_config()
        
        # Core orchestration state
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, OrchestrationTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Framework coordination
        self.framework_controllers: Dict[FrameworkType, FrameworkController] = {}
        self.framework_health: Dict[FrameworkType, FrameworkHealthStatus] = {}
        
        # Communication and coordination
        self.message_queue: deque = deque(maxlen=10000)
        self.coordination_protocols: Dict[str, Any] = {}
        
        # Performance metrics
        self.orchestration_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_task_time': 0.0,
            'resource_efficiency': 0.0,
            'framework_availability': {},
            'coordination_latency': 0.0
        }
        
        # Component initialization
        self.resource_manager = ResourceManager(self.config)
        self._initialize_framework_controllers()
        self._setup_coordination_protocols()
        self.resource_manager.initialize_resource_allocations()
        
        # Runtime control
        self._running = False
        self._orchestration_tasks: List[asyncio.Task] = []
        
        # Enterprise logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get enterprise-grade default configuration"""
        return {
            'max_concurrent_tasks': 10,
            'task_timeout': timedelta(minutes=30),
            'health_check_interval': timedelta(seconds=30),
            'resource_reallocation_interval': timedelta(minutes=5),
            'coordination_timeout': timedelta(seconds=10),
            'retry_failed_tasks': True,
            'max_task_retries': 3,
            'enable_predictive_scaling': True,
            'enable_autonomous_optimization': True,
            'log_level': logging.INFO
        }
    
    def _initialize_framework_controllers(self) -> None:
        """Initialize framework controllers using Factory pattern"""
        # Factory pattern for controller creation
        controller_factories = {
            FrameworkType.ANALYTICS: AnalyticsFrameworkController,
            FrameworkType.ML: MLFrameworkController,
            # Additional controllers would be added here
        }
        
        self.framework_controllers = {
            framework_type: factory()
            for framework_type, factory in controller_factories.items()
        }
        
        # Set resource manager reference
        self.resource_manager.framework_controllers = self.framework_controllers
        
        self.logger.info(f"Initialized {len(self.framework_controllers)} framework controllers")
    
    def _setup_coordination_protocols(self) -> None:
        """Setup enterprise coordination protocols and communication patterns"""
        self.coordination_protocols = {
            'request_response': {
                'timeout': self.config['coordination_timeout'],
                'retry_count': 3,
                'priority_queue': True
            },
            'publish_subscribe': {
                'buffer_size': 1000,
                'persist_messages': True,
                'delivery_guarantee': 'at_least_once'
            },
            'command_control': {
                'acknowledgment_required': True,
                'execution_tracking': True,
                'rollback_support': True
            },
            'event_driven': {
                'event_persistence': True,
                'event_replay': True,
                'event_ordering': True
            }
        }
    
    def _setup_logging(self) -> None:
        """Setup enterprise logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config['log_level'])
    
    async def start(self) -> None:
        """Start Intelligence Command Center with full orchestration"""
        if self._running:
            self.logger.warning("Command Center is already running")
            return
        
        self._running = True
        self.logger.info("Starting Intelligence Command Center")
        
        # Start background orchestration loops
        self._orchestration_tasks = [
            asyncio.create_task(self._orchestration_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self.resource_manager.resource_management_loop()),
            asyncio.create_task(self._coordination_loop())
        ]
        
        # Initialize system state
        await self._update_all_health_status()
        await self.resource_manager.perform_initial_resource_allocation()
        
        self.logger.info("Intelligence Command Center started successfully")
    
    async def stop(self) -> None:
        """Stop Intelligence Command Center gracefully"""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping Intelligence Command Center")
        
        # Cancel background tasks gracefully
        for task in self._orchestration_tasks:
            task.cancel()
        
        await asyncio.gather(*self._orchestration_tasks, return_exceptions=True)
        
        # Cleanup resources
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Intelligence Command Center stopped")
    
    async def submit_task(
        self,
        task_type: str,
        framework_targets: List[FrameworkType],
        parameters: Dict[str, Any],
        priority: OrchestrationPriority = OrchestrationPriority.MEDIUM,
        dependencies: List[str] = None,
        timeout: timedelta = None
    ) -> str:
        """Submit orchestration task with enterprise validation"""
        
        task_id = str(uuid.uuid4())
        
        task = OrchestrationTask(
            task_id=task_id,
            task_type=task_type,
            framework_targets=framework_targets,
            priority=priority,
            parameters=parameters,
            dependencies=dependencies or [],
            timeout=timeout or self.config['task_timeout']
        )
        
        # Estimate resource requirements using resource manager
        task.resource_requirements = await self.resource_manager.estimate_resource_requirements(task)
        
        # Queue task for execution
        self.task_queue.append(task)
        self.orchestration_metrics['total_tasks'] += 1
        
        self.logger.info(f"Task {task_id} submitted: {task_type} -> {[ft.value for ft in framework_targets]}")
        
        return task_id
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop coordinating task execution"""
        self.logger.info("Starting orchestration loop")
        
        while self._running:
            try:
                await self._process_task_queue()
                await self._monitor_active_tasks()
                self._update_orchestration_metrics()
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_task_queue(self) -> None:
        """Process pending tasks with intelligent scheduling"""
        # Sort queue by priority for optimal execution
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.priority.value, reverse=True)
        self.task_queue.clear()
        self.task_queue.extend(sorted_tasks)
        
        # Process tasks up to concurrent limit
        while (len(self.active_tasks) < self.config['max_concurrent_tasks'] and 
               self.task_queue):
            
            task = self.task_queue.popleft()
            
            # Check dependencies and resource availability
            if (await self._are_dependencies_satisfied(task) and 
                await self.resource_manager.are_resources_available(task)):
                await self._execute_task(task)
            else:
                # Re-queue if not ready
                self.task_queue.appendleft(task)
                break
    
    async def _execute_task(self, task: OrchestrationTask) -> None:
        """Execute task across target frameworks with comprehensive monitoring"""
        try:
            task.status = OrchestrationStatus.RUNNING
            task.started_at = datetime.now()
            self.active_tasks[task.task_id] = task
            
            # Reserve resources for task execution
            await self.resource_manager.reserve_task_resources(task)
            
            self.logger.info(f"Executing task {task.task_id}: {task.task_type}")
            
            # Execute on each target framework using Strategy pattern
            framework_results = {}
            
            for framework_type in task.framework_targets:
                controller = self.framework_controllers.get(framework_type)
                if controller:
                    try:
                        result = await controller.execute_task(task)
                        framework_results[framework_type.value] = result
                    except Exception as e:
                        framework_results[framework_type.value] = {'error': str(e)}
                        self.logger.error(f"Framework {framework_type.value} failed for task {task.task_id}: {e}")
                else:
                    framework_results[framework_type.value] = {'error': 'Controller not found'}
            
            # Complete task execution
            await self._complete_task(task, framework_results)
            
        except Exception as e:
            await self._handle_task_failure(task, str(e))
    
    async def _complete_task(self, task: OrchestrationTask, framework_results: Dict[str, Any]) -> None:
        """Complete task execution with result aggregation"""
        task.result = {
            'framework_results': framework_results,
            'execution_summary': self._create_execution_summary(framework_results)
        }
        
        task.status = OrchestrationStatus.COMPLETED
        task.completed_at = datetime.now()
        
        # Release resources
        await self.resource_manager.release_task_resources(task)
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        del self.active_tasks[task.task_id]
        
        self.orchestration_metrics['completed_tasks'] += 1
        
        execution_time = (task.completed_at - task.started_at).total_seconds()
        self.logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
    
    def _create_execution_summary(self, framework_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive execution summary from framework results"""
        successful_frameworks = []
        failed_frameworks = []
        total_processing_time = 0.0
        
        for framework, result in framework_results.items():
            if 'error' in result:
                failed_frameworks.append(framework)
            else:
                successful_frameworks.append(framework)
                # Extract processing time if available
                if 'processing_time' in result:
                    total_processing_time += result['processing_time']
                elif 'response_time' in result:
                    total_processing_time += result['response_time']
        
        return {
            'successful_frameworks': successful_frameworks,
            'failed_frameworks': failed_frameworks,
            'success_rate': len(successful_frameworks) / len(framework_results) if framework_results else 0.0,
            'total_processing_time': total_processing_time,
            'framework_count': len(framework_results)
        }
    
    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status for monitoring"""
        return {
            'version': '1.0.0',
            'status': 'active' if self._running else 'inactive',
            'orchestration_metrics': dict(self.orchestration_metrics),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'framework_count': len(self.framework_controllers),
            'configuration': self.config
        }


# Factory function for enterprise instantiation
def create_intelligence_command_center(config: Dict[str, Any] = None) -> IntelligenceCommandCenter:
    """Create and return configured Intelligence Command Center"""
    return IntelligenceCommandCenter(config)


# Export core components
__all__ = [
    'IntelligenceCommandCenter',
    'create_intelligence_command_center'
]