"""
Performance Execution Manager (Part 2/2) - TestMaster Advanced ML
Coordinates with performance_ml_engine.py for complete enterprise performance optimization
Extracted from analytics_performance_booster.py (555 lines) → 2 coordinated ML modules
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, PriorityQueue
from threading import Event, Lock, RLock, Semaphore
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
import traceback

from .performance_ml_engine import AdvancedPerformanceMLEngine, PerformanceMetrics, OptimizationPlan


@dataclass
class ExecutionTask:
    """Advanced task execution with ML-driven optimization"""
    
    task_id: str
    priority: int
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    timeout: Optional[float] = None
    retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class ExecutionResult:
    """Comprehensive execution result with performance analytics"""
    
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    memory_peak: float = 0.0
    cpu_usage: float = 0.0
    performance_metrics: Optional[PerformanceMetrics] = None
    retry_count: int = 0
    finished_at: datetime = field(default_factory=datetime.now)


class AdvancedPerformanceExecutionManager:
    """
    Enterprise execution manager with ML-driven performance optimization
    Coordinates with AdvancedPerformanceMLEngine for intelligent resource management
    """
    
    def __init__(self, 
                 max_workers: int = 8,
                 max_process_workers: int = 4,
                 enable_ml_optimization: bool = True,
                 performance_threshold: float = 0.85):
        """Initialize execution manager with ML performance optimization"""
        
        self.max_workers = max_workers
        self.max_process_workers = max_process_workers
        self.enable_ml_optimization = enable_ml_optimization
        self.performance_threshold = performance_threshold
        
        # ML Performance Engine Integration
        self.ml_engine = AdvancedPerformanceMLEngine() if enable_ml_optimization else None
        
        # Execution Infrastructure
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_process_workers)
        
        # Task Management
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        
        # Performance Monitoring
        self.execution_metrics: List[PerformanceMetrics] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Synchronization
        self.execution_lock = RLock()
        self.metrics_lock = Lock()
        self.shutdown_event = Event()
        
        # Resource Management
        self.memory_semaphore = Semaphore(max_workers * 2)
        self.cpu_semaphore = Semaphore(max_workers)
        
        # Adaptive Configuration
        self.adaptive_config = {
            'batch_size': 10,
            'processing_interval': 0.1,
            'optimization_interval': 60.0,
            'circuit_breaker_threshold': 0.7
        }
        
        # Circuit Breaker State
        self.circuit_breaker = {
            'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
            'failure_count': 0,
            'last_failure_time': None,
            'reset_timeout': 30.0
        }
        
        self.logger = logging.getLogger(__name__)
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        
        asyncio.create_task(self._performance_monitor_loop())
        asyncio.create_task(self._task_processor_loop())
        
        if self.ml_engine:
            asyncio.create_task(self._ml_optimization_loop())
    
    async def _performance_monitor_loop(self):
        """Continuous performance monitoring with ML analysis"""
        
        while not self.shutdown_event.is_set():
            try:
                current_metrics = await self._collect_execution_metrics()
                
                with self.metrics_lock:
                    self.execution_metrics.append(current_metrics)
                    
                    # Keep only recent metrics
                    if len(self.execution_metrics) > 1000:
                        self.execution_metrics = self.execution_metrics[-500:]
                
                # ML-driven performance analysis
                if self.ml_engine and len(self.execution_metrics) >= 10:
                    await self._analyze_performance_with_ml(current_metrics)
                
                await asyncio.sleep(self.adaptive_config['processing_interval'])
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _task_processor_loop(self):
        """Intelligent task processing with ML optimization"""
        
        while not self.shutdown_event.is_set():
            try:
                if self._check_circuit_breaker():
                    await self._process_task_batch()
                else:
                    await asyncio.sleep(self.circuit_breaker['reset_timeout'])
                    self._reset_circuit_breaker()
                
                await asyncio.sleep(self.adaptive_config['processing_interval'])
                
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _ml_optimization_loop(self):
        """ML-driven optimization and adaptation"""
        
        while not self.shutdown_event.is_set():
            try:
                if len(self.execution_metrics) >= 20:
                    optimization_plan = await self._generate_optimization_plan()
                    
                    if optimization_plan:
                        await self._apply_optimization_plan(optimization_plan)
                
                await asyncio.sleep(self.adaptive_config['optimization_interval'])
                
            except Exception as e:
                self.logger.error(f"ML optimization error: {e}")
                await asyncio.sleep(10.0)
    
    async def submit_task(self, 
                         task_id: str,
                         function: Callable,
                         *args,
                         priority: int = 5,
                         timeout: Optional[float] = None,
                         **kwargs) -> str:
        """Submit task for ML-optimized execution"""
        
        task = ExecutionTask(
            task_id=task_id,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            timeout=timeout
        )
        
        # ML-driven priority adjustment
        if self.ml_engine:
            adjusted_priority = await self._ml_adjust_priority(task)
            task.priority = adjusted_priority
        
        with self.execution_lock:
            self.task_queue.put(task)
            self.active_tasks[task_id] = task
        
        self.logger.info(f"Task submitted: {task_id} (priority: {task.priority})")
        return task_id
    
    async def _process_task_batch(self):
        """Process tasks in optimized batches"""
        
        batch = []
        batch_size = self.adaptive_config['batch_size']
        
        # Collect batch of tasks
        for _ in range(batch_size):
            if not self.task_queue.empty():
                with self.execution_lock:
                    try:
                        task = self.task_queue.get_nowait()
                        batch.append(task)
                    except:
                        break
        
        if batch:
            await self._execute_task_batch(batch)
    
    async def _execute_task_batch(self, batch: List[ExecutionTask]):
        """Execute batch of tasks with performance monitoring"""
        
        execution_futures = []
        
        for task in batch:
            future = asyncio.create_task(self._execute_single_task(task))
            execution_futures.append(future)
        
        # Wait for batch completion
        results = await asyncio.gather(*execution_futures, return_exceptions=True)
        
        # Process results
        for task, result in zip(batch, results):
            if isinstance(result, Exception):
                await self._handle_task_failure(task, result)
            else:
                await self._handle_task_success(task, result)
    
    async def _execute_single_task(self, task: ExecutionTask) -> ExecutionResult:
        """Execute single task with comprehensive monitoring"""
        
        start_time = time.time()
        start_metrics = await self._collect_execution_metrics()
        
        try:
            # Resource acquisition
            async with self._acquire_resources():
                
                # Determine execution strategy
                if self._should_use_process_executor(task):
                    result = await self._execute_in_process(task)
                else:
                    result = await self._execute_in_thread(task)
                
                # Calculate performance metrics
                end_time = time.time()
                end_metrics = await self._collect_execution_metrics()
                
                execution_result = ExecutionResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    duration=end_time - start_time,
                    performance_metrics=self._calculate_task_metrics(start_metrics, end_metrics)
                )
                
                return execution_result
                
        except Exception as e:
            end_time = time.time()
            
            execution_result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                error=e,
                duration=end_time - start_time
            )
            
            return execution_result
    
    @asynccontextmanager
    async def _acquire_resources(self):
        """Smart resource acquisition with ML-driven optimization"""
        
        acquired_memory = False
        acquired_cpu = False
        
        try:
            # Adaptive resource acquisition
            if await self._predict_high_memory_task():
                acquired_memory = self.memory_semaphore.acquire(blocking=False)
            
            if await self._predict_high_cpu_task():
                acquired_cpu = self.cpu_semaphore.acquire(blocking=False)
            
            yield
            
        finally:
            if acquired_memory:
                self.memory_semaphore.release()
            if acquired_cpu:
                self.cpu_semaphore.release()
    
    async def _analyze_performance_with_ml(self, current_metrics: PerformanceMetrics):
        """ML-driven performance analysis and optimization"""
        
        if not self.ml_engine:
            return
        
        try:
            # Analyze current performance
            analysis = await self.ml_engine.analyze_performance(current_metrics)
            
            # Detect anomalies
            if analysis.get('anomaly_detected'):
                await self._handle_performance_anomaly(analysis)
            
            # Update adaptive configuration
            if analysis.get('optimization_suggestions'):
                await self._update_adaptive_config(analysis['optimization_suggestions'])
            
        except Exception as e:
            self.logger.error(f"ML performance analysis error: {e}")
    
    async def _generate_optimization_plan(self) -> Optional[OptimizationPlan]:
        """Generate ML-driven optimization plan"""
        
        if not self.ml_engine or len(self.execution_metrics) < 10:
            return None
        
        try:
            recent_metrics = self.execution_metrics[-50:]
            optimization_plan = await self.ml_engine.generate_optimization_plan(recent_metrics)
            
            return optimization_plan
            
        except Exception as e:
            self.logger.error(f"Optimization plan generation error: {e}")
            return None
    
    async def _apply_optimization_plan(self, plan: OptimizationPlan):
        """Apply ML-generated optimization plan"""
        
        try:
            # Apply resource optimizations
            if plan.resource_adjustments:
                await self._apply_resource_adjustments(plan.resource_adjustments)
            
            # Apply configuration optimizations
            if plan.config_updates:
                await self._apply_config_updates(plan.config_updates)
            
            # Apply algorithm optimizations
            if plan.algorithm_changes:
                await self._apply_algorithm_changes(plan.algorithm_changes)
            
            self.logger.info("Applied ML optimization plan successfully")
            
        except Exception as e:
            self.logger.error(f"Optimization plan application error: {e}")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        with self.metrics_lock:
            if not self.execution_metrics:
                return {}
            
            recent_metrics = self.execution_metrics[-100:]
            
            stats = {
                'total_tasks': len(self.completed_tasks),
                'active_tasks': len(self.active_tasks),
                'avg_duration': sum(m.duration for m in recent_metrics) / len(recent_metrics),
                'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                'circuit_breaker_state': self.circuit_breaker['state'],
                'optimization_config': self.adaptive_config.copy()
            }
            
            if self.ml_engine:
                stats['ml_predictions'] = self.ml_engine.get_prediction_stats()
            
            return stats
    
    async def shutdown(self):
        """Graceful shutdown with cleanup"""
        
        self.logger.info("Initiating graceful shutdown...")
        
        self.shutdown_event.set()
        
        # Wait for active tasks
        await asyncio.sleep(1.0)
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        self.logger.info("Shutdown completed")