"""
Intelligent Resource Scheduler - TestMaster Advanced ML
ML-driven resource allocation and task scheduling with predictive optimization
Enterprise ML Module #2/8 for comprehensive system intelligence
"""

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    CUSTOM = "custom"


class TaskState(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass
class ResourceRequirement:
    """ML-enhanced resource requirement specification"""
    
    resource_type: ResourceType
    min_amount: float
    max_amount: float
    preferred_amount: float
    
    # ML Enhancement
    predicted_usage: float = 0.0
    usage_variance: float = 0.0
    priority_weight: float = 1.0


@dataclass
class ScheduledTask:
    """ML-enhanced task with intelligent scheduling metadata"""
    
    task_id: str
    name: str
    priority: TaskPriority
    estimated_duration: int  # seconds
    resource_requirements: List[ResourceRequirement]
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    
    # Execution tracking
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Resource allocation
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    actual_resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # ML Enhancement Fields
    predicted_duration: float = 0.0
    execution_score: float = 0.0
    optimization_factor: float = 1.0
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling metadata
    scheduling_attempts: int = 0
    last_failure_reason: Optional[str] = None
    preemption_count: int = 0


@dataclass
class ResourceNode:
    """ML-enhanced compute resource node"""
    
    node_id: str
    node_type: str
    capacity: Dict[ResourceType, float]
    available: Dict[ResourceType, float]
    utilization: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Performance metrics
    task_completion_rate: float = 1.0
    average_task_duration: float = 0.0
    failure_rate: float = 0.0
    
    # ML Enhancement
    performance_score: float = 1.0
    efficiency_rating: float = 1.0
    predicted_load: Dict[ResourceType, float] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # State tracking
    active_tasks: List[str] = field(default_factory=list)
    maintenance_mode: bool = False
    last_health_check: datetime = field(default_factory=datetime.now)


class IntelligentResourceScheduler:
    """
    ML-enhanced resource scheduler with predictive optimization
    """
    
    def __init__(self,
                 scheduling_interval: int = 10,
                 enable_ml_optimization: bool = True,
                 preemption_enabled: bool = True):
        """Initialize intelligent resource scheduler"""
        
        self.scheduling_interval = scheduling_interval
        self.enable_ml_optimization = enable_ml_optimization
        self.preemption_enabled = preemption_enabled
        
        # ML Models for Scheduling Intelligence
        self.duration_predictor: Optional[RandomForestRegressor] = None
        self.resource_predictor: Optional[RandomForestRegressor] = None
        self.placement_optimizer: Optional[GradientBoostingClassifier] = None
        self.task_clusterer: Optional[KMeans] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.resource_scaler = RobustScaler()
        self.scheduling_feature_history: deque = deque(maxlen=2000)
        
        # Scheduling State
        self.pending_tasks: List[ScheduledTask] = []  # Priority queue
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=500)
        
        # Resource Management
        self.resource_nodes: Dict[str, ResourceNode] = {}
        self.resource_reservations: Dict[str, Dict[str, float]] = {}  # node_id -> resource allocations
        self.node_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Scheduling Algorithms
        self.scheduling_strategies = {
            'fifo': self._schedule_fifo,
            'priority': self._schedule_priority,
            'shortest_job_first': self._schedule_sjf,
            'ml_optimized': self._schedule_ml_optimized,
            'resource_aware': self._schedule_resource_aware,
            'deadline_aware': self._schedule_deadline_aware
        }
        
        self.default_strategy = 'ml_optimized' if enable_ml_optimization else 'priority'
        
        # ML Insights
        self.ml_predictions: Dict[str, Any] = {}
        self.optimization_insights: List[Dict[str, Any]] = []
        self.resource_utilization_predictions: Dict[str, Dict[str, float]] = {}
        
        # Configuration
        self.max_scheduling_attempts = 5
        self.resource_fragmentation_threshold = 0.8
        self.preemption_score_threshold = 0.3
        
        # Statistics
        self.scheduler_stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'preemptions_performed': 0,
            'ml_optimizations': 0,
            'average_task_duration': 0.0,
            'resource_efficiency': 0.0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.scheduling_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start scheduling loop
        if enable_ml_optimization:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_optimization_loop())
        
        asyncio.create_task(self._scheduling_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for intelligent scheduling"""
        
        try:
            # Task duration prediction
            self.duration_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                min_samples_split=5
            )
            
            # Resource usage prediction
            self.resource_predictor = RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                random_state=42
            )
            
            # Optimal placement classification
            self.placement_optimizer = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Task similarity clustering
            self.task_clusterer = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            self.logger.info("Scheduler ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Scheduler ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def add_resource_node(self,
                         node_id: str,
                         node_type: str,
                         capacity: Dict[ResourceType, float]) -> bool:
        """Add compute resource node to scheduler"""
        
        try:
            with self.scheduling_lock:
                node = ResourceNode(
                    node_id=node_id,
                    node_type=node_type,
                    capacity=capacity.copy(),
                    available=capacity.copy()
                )
                
                # Initialize utilization tracking
                for resource_type in capacity:
                    node.utilization[resource_type] = 0.0
                    node.predicted_load[resource_type] = 0.0
                
                self.resource_nodes[node_id] = node
                self.resource_reservations[node_id] = {}
            
            self.logger.info(f"Resource node added: {node_id} ({node_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add resource node {node_id}: {e}")
            return False
    
    def submit_task(self,
                   task_name: str,
                   priority: TaskPriority,
                   estimated_duration: int,
                   resource_requirements: List[ResourceRequirement],
                   dependencies: List[str] = None,
                   deadline: Optional[datetime] = None) -> str:
        """Submit task for scheduling"""
        
        try:
            task_id = str(uuid.uuid4())
            
            task = ScheduledTask(
                task_id=task_id,
                name=task_name,
                priority=priority,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements,
                dependencies=dependencies or [],
                deadline=deadline
            )
            
            # ML enhancement: predict actual duration and resource usage
            if self.enable_ml_optimization:
                await self._enhance_task_with_ml(task)
            
            with self.scheduling_lock:
                # Insert task into priority queue
                heapq.heappush(self.pending_tasks, (priority.value, time.time(), task))
            
            self.logger.info(f"Task submitted: {task_name} (ID: {task_id})")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Task submission failed: {e}")
            return ""
    
    async def _enhance_task_with_ml(self, task: ScheduledTask):
        """Enhance task with ML predictions"""
        
        try:
            with self.ml_lock:
                # Extract task features for ML analysis
                features = await self._extract_task_features(task)
                
                # Predict actual task duration
                if self.duration_predictor and len(self.scheduling_feature_history) >= 30:
                    predicted_duration = await self._predict_task_duration(features)
                    task.predicted_duration = predicted_duration
                    task.ml_insights['duration_prediction'] = predicted_duration
                
                # Predict resource usage
                if self.resource_predictor:
                    for req in task.resource_requirements:
                        predicted_usage = await self._predict_resource_usage(features, req.resource_type)
                        req.predicted_usage = predicted_usage
                
                # Task clustering for similarity analysis
                if self.task_clusterer and len(self.scheduling_feature_history) >= 20:
                    cluster_id = await self._assign_task_cluster(features)
                    task.ml_insights['task_cluster'] = cluster_id
                
                # Calculate optimization factor
                task.optimization_factor = await self._calculate_optimization_factor(task)
                
        except Exception as e:
            self.logger.error(f"ML task enhancement failed: {e}")
    
    async def _extract_task_features(self, task: ScheduledTask) -> np.ndarray:
        """Extract ML features from task specification"""
        
        # Basic task features
        priority_value = task.priority.value
        duration_hours = task.estimated_duration / 3600.0
        num_dependencies = len(task.dependencies)
        
        # Resource requirement features
        total_cpu = sum(req.preferred_amount for req in task.resource_requirements 
                       if req.resource_type == ResourceType.CPU)
        total_memory = sum(req.preferred_amount for req in task.resource_requirements 
                          if req.resource_type == ResourceType.MEMORY)
        total_disk_io = sum(req.preferred_amount for req in task.resource_requirements 
                           if req.resource_type == ResourceType.DISK_IO)
        
        # Temporal features
        hour_of_day = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Deadline pressure
        deadline_pressure = 0.0
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            deadline_pressure = max(0.0, 1.0 - time_to_deadline / (24 * 3600))  # Normalize to 0-1
        
        # Create feature vector
        features = np.array([
            priority_value,
            duration_hours,
            num_dependencies,
            total_cpu,
            total_memory,
            total_disk_io,
            hour_of_day / 24.0,
            day_of_week / 7.0,
            deadline_pressure,
            task.scheduling_attempts,
            len(task.resource_requirements)
        ])
        
        return features.astype(np.float64)
    
    async def _scheduling_loop(self):
        """Main scheduling loop with ML optimization"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.scheduling_interval)
                
                # Update resource availability
                await self._update_resource_availability()
                
                # Check for completed tasks
                await self._check_task_completions()
                
                # Schedule pending tasks
                await self._schedule_pending_tasks()
                
                # Consider preemption if enabled
                if self.preemption_enabled:
                    await self._evaluate_preemption_opportunities()
                
            except Exception as e:
                self.logger.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(5)
    
    async def _schedule_pending_tasks(self):
        """Schedule pending tasks using selected strategy"""
        
        try:
            with self.scheduling_lock:
                if not self.pending_tasks:
                    return
                
                # Get scheduling strategy
                strategy = self.scheduling_strategies.get(
                    self.default_strategy, 
                    self._schedule_priority
                )
                
                # Execute scheduling
                scheduled_count = await strategy()
                
                if scheduled_count > 0:
                    self.scheduler_stats['tasks_scheduled'] += scheduled_count
                    self.logger.info(f"Scheduled {scheduled_count} tasks using {self.default_strategy}")
                
        except Exception as e:
            self.logger.error(f"Task scheduling failed: {e}")
    
    async def _schedule_ml_optimized(self) -> int:
        """ML-optimized task scheduling"""
        
        scheduled_count = 0
        
        try:
            # Get available tasks (considering dependencies)
            available_tasks = []
            for _, _, task in self.pending_tasks:
                if await self._are_dependencies_satisfied(task):
                    available_tasks.append(task)
            
            if not available_tasks:
                return 0
            
            # ML-based task prioritization
            task_scores = {}
            for task in available_tasks:
                score = await self._calculate_ml_task_score(task)
                task_scores[task.task_id] = score
            
            # Sort tasks by ML score (descending)
            sorted_tasks = sorted(available_tasks, 
                                key=lambda t: task_scores[t.task_id], 
                                reverse=True)
            
            # Schedule tasks in ML-optimized order
            for task in sorted_tasks:
                best_node = await self._find_optimal_node_ml(task)
                
                if best_node and await self._can_schedule_task(task, best_node):
                    await self._assign_task_to_node(task, best_node)
                    scheduled_count += 1
                    
                    # Remove from pending queue
                    self.pending_tasks = [(p, t, tsk) for p, t, tsk in self.pending_tasks 
                                        if tsk.task_id != task.task_id]
                    heapq.heapify(self.pending_tasks)
            
        except Exception as e:
            self.logger.error(f"ML-optimized scheduling failed: {e}")
        
        return scheduled_count
    
    async def _calculate_ml_task_score(self, task: ScheduledTask) -> float:
        """Calculate ML-driven task priority score"""
        
        try:
            score = 0.0
            
            # Base priority score
            priority_score = (6 - task.priority.value) / 5.0  # Higher priority = higher score
            score += 0.3 * priority_score
            
            # Deadline urgency
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds()
                urgency_score = max(0.0, 1.0 - time_to_deadline / (7 * 24 * 3600))  # Week normalization
                score += 0.25 * urgency_score
            
            # Resource efficiency prediction
            efficiency_score = await self._predict_resource_efficiency(task)
            score += 0.2 * efficiency_score
            
            # ML optimization factor
            score += 0.15 * task.optimization_factor
            
            # Waiting time penalty
            wait_time_hours = (datetime.now() - task.created_at).total_seconds() / 3600.0
            wait_penalty = min(1.0, wait_time_hours / 24.0)  # Max 1.0 after 24 hours
            score += 0.1 * wait_penalty
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"ML task score calculation failed: {e}")
            return 0.5  # Default neutral score
    
    async def _find_optimal_node_ml(self, task: ScheduledTask) -> Optional[ResourceNode]:
        """Find optimal node using ML analysis"""
        
        try:
            best_node = None
            best_score = 0.0
            
            for node in self.resource_nodes.values():
                if node.maintenance_mode or not await self._can_schedule_task(task, node):
                    continue
                
                # Calculate node score for this task
                node_score = await self._calculate_node_task_score(task, node)
                
                if node_score > best_score:
                    best_score = node_score
                    best_node = node
            
            return best_node
            
        except Exception as e:
            self.logger.error(f"Optimal node finding failed: {e}")
            return None
    
    async def _calculate_node_task_score(self, task: ScheduledTask, node: ResourceNode) -> float:
        """Calculate how well a node matches a task"""
        
        try:
            score = 0.0
            
            # Resource availability score
            resource_match_score = 0.0
            for req in task.resource_requirements:
                if req.resource_type in node.available:
                    available = node.available[req.resource_type]
                    needed = req.preferred_amount
                    
                    if available >= needed:
                        # Score based on resource fit (prefer not over-allocating)
                        fit_ratio = needed / available if available > 0 else 0
                        resource_match_score += min(1.0, fit_ratio)
            
            score += 0.4 * (resource_match_score / len(task.resource_requirements))
            
            # Node performance score
            score += 0.3 * node.performance_score
            
            # Node efficiency rating
            score += 0.2 * node.efficiency_rating
            
            # Current utilization (prefer less loaded nodes)
            avg_utilization = np.mean(list(node.utilization.values()))
            utilization_score = 1.0 - avg_utilization
            score += 0.1 * utilization_score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Node-task score calculation failed: {e}")
            return 0.0
    
    async def _assign_task_to_node(self, task: ScheduledTask, node: ResourceNode):
        """Assign task to resource node"""
        
        try:
            # Allocate resources
            for req in task.resource_requirements:
                if req.resource_type in node.available:
                    allocated = min(req.preferred_amount, node.available[req.resource_type])
                    node.available[req.resource_type] -= allocated
                    node.utilization[req.resource_type] += allocated / node.capacity[req.resource_type]
                    
                    task.allocated_resources[req.resource_type.value] = allocated
            
            # Update task state
            task.state = TaskState.SCHEDULED
            task.scheduled_at = datetime.now()
            
            # Add to node's active tasks
            node.active_tasks.append(task.task_id)
            
            # Move to running tasks
            self.running_tasks[task.task_id] = task
            
            # Start task execution (simulated)
            asyncio.create_task(self._execute_task(task, node))
            
            self.logger.info(f"Task {task.name} assigned to node {node.node_id}")
            
        except Exception as e:
            self.logger.error(f"Task assignment failed: {e}")
    
    async def _execute_task(self, task: ScheduledTask, node: ResourceNode):
        """Simulate task execution with resource monitoring"""
        
        try:
            task.state = TaskState.RUNNING
            task.started_at = datetime.now()
            
            # Use ML-predicted duration if available, otherwise use estimated
            execution_duration = task.predicted_duration or task.estimated_duration
            
            # Simulate task execution
            await asyncio.sleep(min(execution_duration, 60))  # Cap simulation time
            
            # Task completion
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()
            
            # Calculate actual resource usage (simulated)
            for req in task.resource_requirements:
                # Simulate actual usage with some variance
                variance_factor = np.random.normal(1.0, 0.1)
                actual_usage = req.predicted_usage * variance_factor
                task.actual_resource_usage[req.resource_type.value] = actual_usage
            
            # Release resources
            await self._release_task_resources(task, node)
            
            # Update statistics
            self.scheduler_stats['tasks_completed'] += 1
            
            # Store for ML learning
            await self._store_task_completion_data(task, node)
            
            self.logger.info(f"Task {task.name} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            task.state = TaskState.FAILED
            task.last_failure_reason = str(e)
            
            await self._release_task_resources(task, node)
            self.scheduler_stats['tasks_failed'] += 1
    
    async def _ml_optimization_loop(self):
        """ML optimization and model training loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if len(self.scheduling_feature_history) >= 100:
                    # Retrain ML models
                    await self._retrain_ml_models()
                    
                    # Update resource predictions
                    await self._update_resource_predictions()
                    
                    # Generate optimization insights
                    await self._generate_optimization_insights()
                
            except Exception as e:
                self.logger.error(f"ML optimization loop error: {e}")
                await asyncio.sleep(10)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        
        # Resource node summary
        node_summary = {}
        for node_id, node in self.resource_nodes.items():
            node_summary[node_id] = {
                'node_type': node.node_type,
                'capacity': {rt.value: cap for rt, cap in node.capacity.items()},
                'available': {rt.value: avail for rt, avail in node.available.items()},
                'utilization': {rt.value: util for rt, util in node.utilization.items()},
                'active_tasks': len(node.active_tasks),
                'performance_score': node.performance_score,
                'efficiency_rating': node.efficiency_rating
            }
        
        # Task queue status
        queue_status = {
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }
        
        # ML status
        ml_status = {
            'ml_optimization_enabled': self.enable_ml_optimization,
            'feature_history_size': len(self.scheduling_feature_history),
            'ml_optimizations_performed': self.scheduler_stats['ml_optimizations'],
            'recent_insights': self.optimization_insights[-5:] if self.optimization_insights else []
        }
        
        return {
            'scheduler_overview': {
                'total_nodes': len(self.resource_nodes),
                'active_nodes': len([n for n in self.resource_nodes.values() if not n.maintenance_mode]),
                'resource_efficiency': self.scheduler_stats['resource_efficiency'],
                'average_task_duration': self.scheduler_stats['average_task_duration']
            },
            'resource_nodes': node_summary,
            'task_queues': queue_status,
            'statistics': self.scheduler_stats.copy(),
            'ml_status': ml_status
        }
    
    async def shutdown(self):
        """Graceful shutdown of resource scheduler"""
        
        self.logger.info("Shutting down resource scheduler...")
        
        # Cancel pending tasks
        with self.scheduling_lock:
            for _, _, task in self.pending_tasks:
                task.state = TaskState.CANCELLED
        
        # Wait for running tasks to complete (with timeout)
        timeout = 60
        while self.running_tasks and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Resource scheduler shutdown complete")