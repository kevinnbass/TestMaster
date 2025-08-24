#!/usr/bin/env python3
"""
Advanced Parallel Processor
Agent B Hours 80-90: Parallel Processing Optimization

Advanced parallel processing system with dynamic load balancing, intelligent task distribution,
adaptive thread/process management, and performance optimization for compute-intensive operations.
"""

import asyncio
import multiprocessing
import threading
import concurrent.futures
import logging
import time
import psutil
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import functools

T = TypeVar('T')
R = TypeVar('R')

class ProcessingMode(Enum):
    """Parallel processing modes"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_POOL = "async_pool"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    GPU_ACCELERATED = "gpu_accelerated"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    CAPABILITY_BASED = "capability_based"
    PREDICTIVE = "predictive"
    WORK_STEALING = "work_stealing"

@dataclass
class ProcessingTask:
    """Task for parallel processing"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    estimated_duration: float
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class WorkerMetrics:
    """Worker performance metrics"""
    worker_id: str
    tasks_completed: int
    total_processing_time: float
    average_task_time: float
    cpu_utilization: float
    memory_usage: float
    queue_size: int
    error_rate: float
    efficiency_score: float

@dataclass
class ProcessingResult:
    """Result of parallel processing"""
    task_id: str
    result: Any
    success: bool
    execution_time: float
    worker_id: str
    error: Optional[str] = None
    memory_used: int = 0
    cpu_time: float = 0.0

class AdaptiveWorkerPool:
    """Adaptive worker pool with dynamic scaling"""
    
    def __init__(self, pool_type: str, min_workers: int = 2, max_workers: int = None):
        self.pool_type = pool_type
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        
        # Worker management
        self.current_workers = min_workers
        self.worker_pool: Optional[concurrent.futures.Executor] = None
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        
        # Task queues
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        
        # Performance tracking
        self.queue_length_history: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=100)
        self.response_time_history: deque = deque(maxlen=100)
        
        # Scaling parameters
        self.scale_up_threshold = 0.8    # Scale up when utilization > 80%
        self.scale_down_threshold = 0.3  # Scale down when utilization < 30%
        self.last_scale_action = datetime.now()
        self.scale_cooldown = 60  # seconds
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the worker pool"""
        if self.pool_type == "thread":
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix=f"ParallelProcessor-{self.pool_type}"
            )
        elif self.pool_type == "process":
            self.worker_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.current_workers
            )
        
        # Initialize worker metrics
        for i in range(self.current_workers):
            worker_id = f"{self.pool_type}-worker-{i}"
            self.worker_metrics[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                tasks_completed=0,
                total_processing_time=0.0,
                average_task_time=0.0,
                cpu_utilization=0.0,
                memory_usage=0.0,
                queue_size=0,
                error_rate=0.0,
                efficiency_score=1.0
            )
    
    def should_scale_up(self) -> bool:
        """Determine if pool should scale up"""
        if self.current_workers >= self.max_workers:
            return False
        
        # Check if cooldown period has passed
        if (datetime.now() - self.last_scale_action).seconds < self.scale_cooldown:
            return False
        
        # Check utilization
        avg_utilization = np.mean([m.cpu_utilization for m in self.worker_metrics.values()])
        queue_pressure = self.task_queue.qsize() / max(1, self.current_workers)
        
        return avg_utilization > self.scale_up_threshold or queue_pressure > 2.0
    
    def should_scale_down(self) -> bool:
        """Determine if pool should scale down"""
        if self.current_workers <= self.min_workers:
            return False
        
        # Check if cooldown period has passed
        if (datetime.now() - self.last_scale_action).seconds < self.scale_cooldown:
            return False
        
        # Check utilization
        avg_utilization = np.mean([m.cpu_utilization for m in self.worker_metrics.values()])
        queue_pressure = self.task_queue.qsize() / max(1, self.current_workers)
        
        return avg_utilization < self.scale_down_threshold and queue_pressure < 0.5
    
    def scale_up(self):
        """Scale up the worker pool"""
        new_workers = min(self.current_workers * 2, self.max_workers)
        if new_workers > self.current_workers:
            self._resize_pool(new_workers)
            self.last_scale_action = datetime.now()
    
    def scale_down(self):
        """Scale down the worker pool"""
        new_workers = max(self.current_workers // 2, self.min_workers)
        if new_workers < self.current_workers:
            self._resize_pool(new_workers)
            self.last_scale_action = datetime.now()
    
    def _resize_pool(self, new_size: int):
        """Resize the worker pool"""
        old_pool = self.worker_pool
        self.current_workers = new_size
        
        # Create new pool
        if self.pool_type == "thread":
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.current_workers
            )
        elif self.pool_type == "process":
            self.worker_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.current_workers
            )
        
        # Shutdown old pool gracefully
        if old_pool:
            old_pool.shutdown(wait=False)

class AdvancedParallelProcessor:
    """
    Advanced Parallel Processor
    
    Comprehensive parallel processing system with dynamic load balancing,
    intelligent task distribution, adaptive scaling, and performance optimization
    for compute-intensive operations in orchestration components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AdvancedParallelProcessor")
        
        # Processor configuration
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.processing_mode = ProcessingMode.ADAPTIVE
        self.load_balancing_strategy = LoadBalancingStrategy.PREDICTIVE
        
        # Worker pools
        self.thread_pool = AdaptiveWorkerPool("thread", min_workers=2, max_workers=self.cpu_count * 4)
        self.process_pool = AdaptiveWorkerPool("process", min_workers=2, max_workers=self.cpu_count)
        self.async_pool: Optional[asyncio.Semaphore] = asyncio.Semaphore(self.cpu_count * 2)
        
        # Task management
        self.pending_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.task_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.task_counter = 0
        
        # Performance monitoring
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "throughput": 0.0,
            "cpu_efficiency": 0.0,
            "memory_efficiency": 0.0,
            "queue_time": 0.0
        }
        
        # Load balancer
        self.worker_loads: Dict[str, float] = defaultdict(float)
        self.task_predictor: Optional[Any] = None
        
        # Monitoring and optimization
        self.monitoring_enabled = True
        self.optimization_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Advanced parallel processor initialized")
    
    async def start_processing(self):
        """Start parallel processing engine"""
        try:
            # Start monitoring task
            if self.monitoring_enabled:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start optimization task
            if self.optimization_enabled:
                asyncio.create_task(self._optimization_loop())
            
            self.logger.info("Parallel processing engine started")
            
        except Exception as e:
            self.logger.error(f"Failed to start parallel processing: {e}")
    
    async def stop_processing(self) -> Dict[str, Any]:
        """Stop parallel processing and generate report"""
        try:
            # Cancel monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown worker pools
            self.thread_pool.worker_pool.shutdown(wait=True)
            self.process_pool.worker_pool.shutdown(wait=True)
            
            # Generate final report
            report = self._generate_performance_report()
            
            self.logger.info("Parallel processing engine stopped")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to stop parallel processing: {e}")
            return {"error": str(e)}
    
    async def submit_task(self, 
                         function: Callable,
                         args: tuple = (),
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         processing_mode: Optional[ProcessingMode] = None,
                         timeout: Optional[float] = None,
                         callback: Optional[Callable] = None) -> str:
        """Submit task for parallel processing"""
        
        kwargs = kwargs or {}
        self.task_counter += 1
        task_id = f"task-{self.task_counter}-{int(time.time() * 1000)}"
        
        # Estimate resource requirements
        resource_requirements = self._estimate_resource_requirements(function, args, kwargs)
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            estimated_duration=resource_requirements.get("duration", 1.0),
            resource_requirements=resource_requirements,
            callback=callback,
            timeout=timeout
        )
        
        # Store task
        self.pending_tasks[task_id] = task
        self.performance_metrics["total_tasks"] += 1
        
        # Select processing mode
        if processing_mode is None:
            processing_mode = self._select_optimal_processing_mode(task)
        
        # Schedule task
        await self._schedule_task(task, processing_mode)
        
        return task_id
    
    async def submit_batch(self, 
                          tasks: List[Dict[str, Any]],
                          batch_processing_mode: Optional[ProcessingMode] = None) -> List[str]:
        """Submit batch of tasks for parallel processing"""
        task_ids = []
        
        for task_config in tasks:
            task_id = await self.submit_task(
                function=task_config["function"],
                args=task_config.get("args", ()),
                kwargs=task_config.get("kwargs", {}),
                priority=task_config.get("priority", TaskPriority.NORMAL),
                processing_mode=batch_processing_mode,
                timeout=task_config.get("timeout"),
                callback=task_config.get("callback")
            )
            task_ids.append(task_id)
        
        return task_ids
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """Get result of completed task"""
        start_time = time.time()
        
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            await asyncio.sleep(0.1)
        
        return self.completed_tasks[task_id]
    
    async def get_batch_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """Get results for batch of tasks"""
        results = {}
        
        # Wait for all tasks to complete
        tasks = [self.get_result(task_id, timeout) for task_id in task_ids]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_id, result in zip(task_ids, completed_results):
            if isinstance(result, Exception):
                results[task_id] = ProcessingResult(
                    task_id=task_id,
                    result=None,
                    success=False,
                    execution_time=0.0,
                    worker_id="error",
                    error=str(result)
                )
            else:
                results[task_id] = result
        
        return results
    
    def _estimate_resource_requirements(self, function: Callable, args: tuple, kwargs: dict) -> Dict[str, float]:
        """Estimate resource requirements for task"""
        # Basic estimation (can be enhanced with profiling)
        base_duration = 1.0
        base_memory = 100.0  # MB
        base_cpu = 1.0
        
        # Adjust based on function characteristics
        if hasattr(function, '__name__'):
            func_name = function.__name__
            if any(keyword in func_name.lower() for keyword in ['ml', 'neural', 'train', 'model']):
                base_duration *= 10
                base_memory *= 5
                base_cpu *= 2
            elif any(keyword in func_name.lower() for keyword in ['io', 'file', 'network', 'db']):
                base_duration *= 3
                base_memory *= 0.5
            elif any(keyword in func_name.lower() for keyword in ['compute', 'calculate', 'process']):
                base_duration *= 2
                base_cpu *= 1.5
        
        # Adjust based on data size
        total_size = sum(len(str(arg)) for arg in args) + sum(len(str(v)) for v in kwargs.values())
        size_factor = max(1.0, total_size / 1000)  # Scale with data size
        
        return {
            "duration": base_duration * size_factor,
            "memory": base_memory * size_factor,
            "cpu": base_cpu,
            "io_bound": any(keyword in function.__name__.lower() if hasattr(function, '__name__') else False 
                           for keyword in ['io', 'file', 'network', 'db']),
            "cpu_bound": any(keyword in function.__name__.lower() if hasattr(function, '__name__') else False 
                            for keyword in ['compute', 'calculate', 'process', 'ml', 'neural'])
        }
    
    def _select_optimal_processing_mode(self, task: ProcessingTask) -> ProcessingMode:
        """Select optimal processing mode for task"""
        if self.processing_mode != ProcessingMode.ADAPTIVE:
            return self.processing_mode
        
        # Adaptive selection based on task characteristics
        if task.resource_requirements.get("io_bound", False):
            return ProcessingMode.ASYNC_POOL
        elif task.resource_requirements.get("cpu_bound", False):
            if task.estimated_duration > 5.0:  # Long-running CPU task
                return ProcessingMode.PROCESS_POOL
            else:
                return ProcessingMode.THREAD_POOL
        elif task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            return ProcessingMode.THREAD_POOL  # Faster startup
        else:
            return ProcessingMode.HYBRID
    
    async def _schedule_task(self, task: ProcessingTask, processing_mode: ProcessingMode):
        """Schedule task for execution"""
        try:
            if processing_mode == ProcessingMode.THREAD_POOL:
                await self._execute_in_thread_pool(task)
            elif processing_mode == ProcessingMode.PROCESS_POOL:
                await self._execute_in_process_pool(task)
            elif processing_mode == ProcessingMode.ASYNC_POOL:
                await self._execute_in_async_pool(task)
            elif processing_mode == ProcessingMode.HYBRID:
                # Choose based on current load
                if self.thread_pool.task_queue.qsize() < self.process_pool.task_queue.qsize():
                    await self._execute_in_thread_pool(task)
                else:
                    await self._execute_in_process_pool(task)
            else:
                await self._execute_in_thread_pool(task)  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Task scheduling failed for {task.task_id}: {e}")
            await self._handle_task_error(task, str(e))
    
    async def _execute_in_thread_pool(self, task: ProcessingTask):
        """Execute task in thread pool"""
        try:
            loop = asyncio.get_event_loop()
            future = self.thread_pool.worker_pool.submit(
                self._execute_task_wrapper,
                task
            )
            
            # Wait for completion with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(None, future.result),
                timeout=task.timeout
            )
            
            await self._handle_task_completion(task, result)
            
        except Exception as e:
            await self._handle_task_error(task, str(e))
    
    async def _execute_in_process_pool(self, task: ProcessingTask):
        """Execute task in process pool"""
        try:
            loop = asyncio.get_event_loop()
            future = self.process_pool.worker_pool.submit(
                self._execute_task_wrapper,
                task
            )
            
            # Wait for completion with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(None, future.result),
                timeout=task.timeout
            )
            
            await self._handle_task_completion(task, result)
            
        except Exception as e:
            await self._handle_task_error(task, str(e))
    
    async def _execute_in_async_pool(self, task: ProcessingTask):
        """Execute task in async pool"""
        try:
            async with self.async_pool:
                start_time = time.time()
                
                # Execute task
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(*task.args, **task.kwargs)
                else:
                    result = task.function(*task.args, **task.kwargs)
                
                execution_time = time.time() - start_time
                
                # Create result
                processing_result = ProcessingResult(
                    task_id=task.task_id,
                    result=result,
                    success=True,
                    execution_time=execution_time,
                    worker_id="async-pool"
                )
                
                await self._handle_task_completion(task, processing_result)
                
        except Exception as e:
            await self._handle_task_error(task, str(e))
    
    def _execute_task_wrapper(self, task: ProcessingTask) -> ProcessingResult:
        """Wrapper for task execution with monitoring"""
        start_time = time.time()
        worker_id = f"{threading.current_thread().name}-{multiprocessing.current_process().pid}"
        
        try:
            # Execute the task
            result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                result=None,
                success=False,
                execution_time=execution_time,
                worker_id=worker_id,
                error=str(e)
            )
    
    async def _handle_task_completion(self, task: ProcessingTask, result: ProcessingResult):
        """Handle task completion"""
        try:
            # Store result
            self.completed_tasks[task.task_id] = result
            self.pending_tasks.pop(task.task_id, None)
            
            # Update metrics
            if result.success:
                self.performance_metrics["completed_tasks"] += 1
            else:
                self.performance_metrics["failed_tasks"] += 1
            
            # Update average execution time
            total_completed = self.performance_metrics["completed_tasks"]
            if total_completed > 0:
                current_avg = self.performance_metrics["average_execution_time"]
                self.performance_metrics["average_execution_time"] = (
                    (current_avg * (total_completed - 1) + result.execution_time) / total_completed
                )
            
            # Execute callback if provided
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result)
                    else:
                        task.callback(result)
                except Exception as e:
                    self.logger.error(f"Task callback failed: {e}")
            
            # Process dependent tasks
            await self._process_dependent_tasks(task.task_id)
            
        except Exception as e:
            self.logger.error(f"Task completion handling failed: {e}")
    
    async def _handle_task_error(self, task: ProcessingTask, error: str):
        """Handle task execution error"""
        # Retry if allowed
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            self.logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count})")
            processing_mode = self._select_optimal_processing_mode(task)
            await self._schedule_task(task, processing_mode)
            return
        
        # Create error result
        result = ProcessingResult(
            task_id=task.task_id,
            result=None,
            success=False,
            execution_time=0.0,
            worker_id="error",
            error=error
        )
        
        await self._handle_task_completion(task, result)
    
    async def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on completed task"""
        # This would be implemented to handle task dependencies
        pass
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for scaling opportunities
                await self._check_scaling()
                
                # Log statistics
                self.logger.info(f"Processing stats: {self.get_statistics()}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Optimize worker pools
                await self._optimize_worker_pools()
                
                # Optimize load balancing
                await self._optimize_load_balancing()
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        # Calculate throughput
        total_completed = self.performance_metrics["completed_tasks"]
        total_time = time.time()  # Would track actual start time
        
        if total_time > 0:
            self.performance_metrics["throughput"] = total_completed / total_time
        
        # Calculate CPU efficiency
        cpu_usage = psutil.cpu_percent()
        self.performance_metrics["cpu_efficiency"] = min(1.0, cpu_usage / 100.0)
        
        # Calculate memory efficiency
        memory_info = psutil.virtual_memory()
        self.performance_metrics["memory_efficiency"] = memory_info.percent / 100.0
        
        # Calculate queue time
        pending_count = len(self.pending_tasks)
        if pending_count > 0:
            avg_queue_time = sum(
                (datetime.now() - datetime.fromtimestamp(time.time())).seconds
                for _ in range(min(10, pending_count))  # Sample estimation
            ) / min(10, pending_count)
            self.performance_metrics["queue_time"] = avg_queue_time
    
    async def _check_scaling(self):
        """Check if worker pools need scaling"""
        # Check thread pool
        if self.thread_pool.should_scale_up():
            self.thread_pool.scale_up()
            self.logger.info("Scaled up thread pool")
        elif self.thread_pool.should_scale_down():
            self.thread_pool.scale_down()
            self.logger.info("Scaled down thread pool")
        
        # Check process pool
        if self.process_pool.should_scale_up():
            self.process_pool.scale_up()
            self.logger.info("Scaled up process pool")
        elif self.process_pool.should_scale_down():
            self.process_pool.scale_down()
            self.logger.info("Scaled down process pool")
    
    async def _optimize_worker_pools(self):
        """Optimize worker pool configurations"""
        # Analyze task patterns and optimize pool sizes
        pass
    
    async def _optimize_load_balancing(self):
        """Optimize load balancing strategy"""
        # Analyze worker performance and adjust load balancing
        pass
    
    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks"""
        # Keep only recent results to manage memory
        if len(self.completed_tasks) > 10000:
            # Remove oldest tasks
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].task_id
            )
            tasks_to_remove = sorted_tasks[:5000]
            for task_id, _ in tasks_to_remove:
                del self.completed_tasks[task_id]
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "processing_summary": {
                "total_tasks": self.performance_metrics["total_tasks"],
                "completed_tasks": self.performance_metrics["completed_tasks"],
                "failed_tasks": self.performance_metrics["failed_tasks"],
                "success_rate": (
                    self.performance_metrics["completed_tasks"] / 
                    max(1, self.performance_metrics["total_tasks"]) * 100
                )
            },
            "performance_metrics": self.performance_metrics,
            "worker_pools": {
                "thread_pool": {
                    "current_workers": self.thread_pool.current_workers,
                    "queue_size": self.thread_pool.task_queue.qsize(),
                    "metrics": dict(self.thread_pool.worker_metrics)
                },
                "process_pool": {
                    "current_workers": self.process_pool.current_workers,
                    "queue_size": self.process_pool.task_queue.qsize(),
                    "metrics": dict(self.process_pool.worker_metrics)
                }
            },
            "system_resources": {
                "cpu_count": self.cpu_count,
                "memory_total_gb": self.memory_total / (1024**3),
                "current_cpu_usage": psutil.cpu_percent(),
                "current_memory_usage": psutil.virtual_memory().percent
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": self.performance_metrics["completed_tasks"],
            "failed_tasks": self.performance_metrics["failed_tasks"],
            "average_execution_time": self.performance_metrics["average_execution_time"],
            "throughput": self.performance_metrics["throughput"],
            "cpu_efficiency": self.performance_metrics["cpu_efficiency"],
            "thread_pool_workers": self.thread_pool.current_workers,
            "process_pool_workers": self.process_pool.current_workers
        }