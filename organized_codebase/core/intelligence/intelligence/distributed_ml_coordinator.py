"""
Distributed ML Coordinator - TestMaster Advanced ML
Coordinate distributed ML operations across multiple nodes with intelligent workload distribution
Enterprise ML Module #7/8 for comprehensive system intelligence
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import uuid
import json
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


class NodeType(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    COMPUTE = "compute"
    HYBRID = "hybrid"


class TaskType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DATA_PROCESSING = "data_processing"
    MODEL_SERVING = "model_serving"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_ENGINEERING = "feature_engineering"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class MLNode:
    """ML-enhanced distributed node with intelligent capabilities"""
    
    node_id: str
    node_type: NodeType
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    
    # Resource specifications
    cpu_cores: int = 4
    memory_gb: float = 16.0
    gpu_count: int = 0
    storage_gb: float = 100.0
    
    # Current state
    status: str = "active"  # active, inactive, maintenance, failed
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    network_latency_ms: float = 0.0
    
    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    success_rate: float = 1.0
    
    # ML Enhancement Fields
    performance_score: float = 1.0
    reliability_score: float = 1.0
    load_prediction: float = 0.0
    optimal_task_types: List[TaskType] = field(default_factory=list)
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Coordination state
    last_heartbeat: datetime = field(default_factory=datetime.now)
    assigned_tasks: List[str] = field(default_factory=list)
    task_queue_size: int = 0


@dataclass
class DistributedMLTask:
    """ML task for distributed execution"""
    
    task_id: str
    task_type: TaskType
    priority: int = 5
    data_size_mb: float = 0.0
    estimated_duration: float = 0.0  # seconds
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Task configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    data_location: str = ""
    output_location: str = ""
    
    # Execution tracking
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # ML Enhancement
    predicted_duration: float = 0.0
    predicted_resource_usage: Dict[str, float] = field(default_factory=dict)
    optimization_score: float = 1.0
    execution_efficiency: float = 1.0
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    # Dependencies and coordination
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None


@dataclass
class WorkloadDistribution:
    """ML-optimized workload distribution strategy"""
    
    strategy_name: str
    description: str
    node_assignments: Dict[str, List[str]] = field(default_factory=dict)  # node_id -> task_ids
    
    # Performance predictions
    predicted_completion_time: float = 0.0
    predicted_resource_efficiency: float = 0.0
    load_balance_score: float = 0.0
    fault_tolerance_score: float = 0.0
    
    # ML optimization
    ml_optimized: bool = True
    optimization_confidence: float = 0.0
    alternative_strategies: List[str] = field(default_factory=list)


class DistributedMLCoordinator:
    """
    ML-enhanced coordinator for distributed machine learning operations
    """
    
    def __init__(self,
                 coordinator_id: str,
                 enable_ml_optimization: bool = True,
                 heartbeat_interval: int = 30,
                 auto_scaling: bool = True,
                 fault_tolerance: bool = True):
        """Initialize distributed ML coordinator"""
        
        self.coordinator_id = coordinator_id
        self.enable_ml_optimization = enable_ml_optimization
        self.heartbeat_interval = heartbeat_interval
        self.auto_scaling = auto_scaling
        self.fault_tolerance = fault_tolerance
        
        # ML Models for Distributed Intelligence
        self.task_duration_predictor: Optional[RandomForestRegressor] = None
        self.node_performance_classifier: Optional[GradientBoostingClassifier] = None
        self.workload_optimizer: Optional[Ridge] = None
        self.failure_predictor: Optional[RandomForestRegressor] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.performance_scaler = RobustScaler()
        self.coordination_feature_history: deque = deque(maxlen=10000)
        
        # Distributed System State
        self.ml_nodes: Dict[str, MLNode] = {}
        self.distributed_tasks: Dict[str, DistributedMLTask] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: deque = deque(maxlen=1000)
        self.failed_tasks: deque = deque(maxlen=500)
        
        # Workload Management
        self.workload_strategies: Dict[str, WorkloadDistribution] = {}
        self.active_distribution: Optional[WorkloadDistribution] = None
        self.resource_utilization: Dict[str, Dict[str, float]] = {}
        
        # Performance Monitoring
        self.node_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.task_execution_history: deque = deque(maxlen=2000)
        self.coordination_metrics: deque = deque(maxlen=500)
        
        # ML Insights and Optimization
        self.ml_recommendations: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.failure_predictions: Dict[str, float] = {}
        
        # Configuration
        self.max_tasks_per_node = 10
        self.task_timeout = 3600  # 1 hour
        self.node_failure_threshold = 0.3
        self.load_balance_threshold = 0.2
        
        # Statistics
        self.coordination_stats = {
            'nodes_managed': 0,
            'tasks_distributed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'workload_optimizations': 0,
            'node_failures_detected': 0,
            'auto_scaling_events': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.coordination_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and coordination loops
        if enable_ml_optimization:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_coordination_loop())
        
        asyncio.create_task(self._coordination_monitoring_loop())
        asyncio.create_task(self._task_distribution_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for distributed coordination intelligence"""
        
        try:
            # Task duration prediction
            self.task_duration_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                min_samples_split=5
            )
            
            # Node performance classification
            self.node_performance_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Workload optimization
            self.workload_optimizer = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # Node failure prediction
            self.failure_predictor = RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                random_state=42
            )
            
            self.logger.info("Distributed ML coordination models initialized")
            
        except Exception as e:
            self.logger.error(f"Distributed ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def register_node(self,
                     node_id: str,
                     node_type: NodeType,
                     host: str,
                     port: int,
                     capabilities: List[str] = None,
                     cpu_cores: int = 4,
                     memory_gb: float = 16.0,
                     gpu_count: int = 0,
                     storage_gb: float = 100.0) -> bool:
        """Register distributed ML node for coordination"""
        
        try:
            with self.coordination_lock:
                node = MLNode(
                    node_id=node_id,
                    node_type=node_type,
                    host=host,
                    port=port,
                    capabilities=capabilities or [],
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb,
                    gpu_count=gpu_count,
                    storage_gb=storage_gb
                )
                
                self.ml_nodes[node_id] = node
                self.node_performance_history[node_id] = deque(maxlen=100)
                self.resource_utilization[node_id] = {
                    'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0
                }
                
                self.coordination_stats['nodes_managed'] += 1
            
            # ML enhancement for node
            if self.enable_ml_optimization:
                asyncio.create_task(self._analyze_node_capabilities(node))
            
            self.logger.info(f"ML node registered: {node_id} ({node_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Node registration failed: {e}")
            return False
    
    def submit_distributed_task(self,
                               task_type: TaskType,
                               model_config: Dict[str, Any],
                               data_location: str,
                               priority: int = 5,
                               estimated_duration: float = 0.0,
                               resource_requirements: Dict[str, float] = None) -> str:
        """Submit task for distributed ML execution"""
        
        try:
            task_id = str(uuid.uuid4())
            
            task = DistributedMLTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements or {},
                model_config=model_config,
                data_location=data_location
            )
            
            # ML enhancement for task
            if self.enable_ml_optimization:
                await self._enhance_task_with_ml(task)
            
            with self.coordination_lock:
                self.distributed_tasks[task_id] = task
                self.task_queue.append(task)
                self.coordination_stats['tasks_distributed'] += 1
            
            self.logger.info(f"Distributed ML task submitted: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Task submission failed: {e}")
            return ""
    
    async def _enhance_task_with_ml(self, task: DistributedMLTask):
        """Enhance task with ML predictions and optimization"""
        
        try:
            with self.ml_lock:
                # Extract task features
                features = await self._extract_task_features(task)
                
                # Predict task duration
                if self.task_duration_predictor and len(self.coordination_feature_history) >= 50:
                    predicted_duration = await self._predict_task_duration(features, task)
                    task.predicted_duration = predicted_duration
                
                # Predict resource usage
                predicted_resources = await self._predict_resource_usage(task, features)
                task.predicted_resource_usage = predicted_resources
                
                # Calculate optimization score
                task.optimization_score = await self._calculate_task_optimization_score(task, features)
                
                # Find optimal node types
                optimal_nodes = await self._identify_optimal_node_types(task, features)
                task.predicted_resource_usage['optimal_node_types'] = optimal_nodes
                
                # Store features for model training
                self.coordination_feature_history.append(features)
                
        except Exception as e:
            self.logger.error(f"ML task enhancement failed: {e}")
    
    def _extract_task_features(self, task: DistributedMLTask) -> np.ndarray:
        """Extract ML features from distributed task"""
        
        try:
            # Task characteristics
            task_type_encoded = list(TaskType).index(task.task_type)
            priority_normalized = task.priority / 10.0
            data_size_gb = task.data_size_mb / 1024.0
            
            # Resource requirements
            cpu_requirement = task.resource_requirements.get('cpu', 1.0)
            memory_requirement = task.resource_requirements.get('memory', 1.0)
            gpu_requirement = task.resource_requirements.get('gpu', 0.0)
            
            # Temporal features
            hour = datetime.now().hour
            day_of_week = datetime.now().weekday()
            
            # System state features
            available_nodes = len([n for n in self.ml_nodes.values() if n.status == 'active'])
            system_load = self._calculate_system_load()
            
            # Historical performance
            similar_tasks = [
                t for t in list(self.completed_tasks)[-50:]
                if t.task_type == task.task_type
            ]
            avg_duration = np.mean([
                (t.completed_at - t.started_at).total_seconds()
                for t in similar_tasks if t.completed_at and t.started_at
            ]) if similar_tasks else task.estimated_duration or 300.0
            
            # Create feature vector
            features = np.array([
                task_type_encoded,
                priority_normalized,
                data_size_gb,
                cpu_requirement,
                memory_requirement,
                gpu_requirement,
                hour / 24.0,
                day_of_week / 7.0,
                available_nodes / 10.0,  # Normalize
                system_load,
                avg_duration / 3600.0,  # Convert to hours
                len(task.dependencies),
                task.retry_count,
                len(self.task_queue) / 100.0,  # Normalize queue size
                len(self.ml_nodes) / 10.0  # Normalize node count
            ])
            
            return features.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Task feature extraction failed: {e}")
            return np.zeros(15)  # Default feature vector
    
    async def _task_distribution_loop(self):
        """Main task distribution and assignment loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # Check for tasks every 5 seconds
                
                if not self.task_queue:
                    continue
                
                # Get next task
                with self.coordination_lock:
                    if self.task_queue:
                        task = self.task_queue.popleft()
                    else:
                        continue
                
                # Find optimal node for task
                optimal_node = await self._find_optimal_node_for_task(task)
                
                if optimal_node:
                    await self._assign_task_to_node(task, optimal_node)
                else:
                    # No suitable node available, put back in queue
                    with self.coordination_lock:
                        self.task_queue.appendleft(task)
                    
                    await asyncio.sleep(10)  # Wait before trying again
                
            except Exception as e:
                self.logger.error(f"Task distribution loop error: {e}")
                await asyncio.sleep(10)
    
    async def _find_optimal_node_for_task(self, task: DistributedMLTask) -> Optional[MLNode]:
        """Find optimal node for task execution using ML analysis"""
        
        try:
            # Get available nodes
            available_nodes = [
                node for node in self.ml_nodes.values()
                if (node.status == 'active' and
                    len(node.assigned_tasks) < self.max_tasks_per_node)
            ]
            
            if not available_nodes:
                return None
            
            # ML-based node selection if enabled
            if self.enable_ml_optimization:
                optimal_node = await self._ml_node_selection(task, available_nodes)
                if optimal_node:
                    return optimal_node
            
            # Fallback to rule-based selection
            return await self._rule_based_node_selection(task, available_nodes)
            
        except Exception as e:
            self.logger.error(f"Optimal node finding failed: {e}")
            return None
    
    async def _ml_node_selection(self, task: DistributedMLTask, nodes: List[MLNode]) -> Optional[MLNode]:
        """ML-driven node selection for task assignment"""
        
        try:
            best_node = None
            best_score = 0.0
            
            for node in nodes:
                # Extract node-task compatibility features
                compatibility_features = await self._extract_node_task_features(node, task)
                
                # Predict node performance for this task
                if self.node_performance_classifier and len(self.coordination_feature_history) >= 100:
                    performance_score = await self._predict_node_task_performance(
                        compatibility_features, node, task
                    )
                else:
                    performance_score = node.performance_score
                
                # Calculate resource fit score
                resource_fit = await self._calculate_resource_fit_score(node, task)
                
                # Calculate load balance impact
                load_balance_score = await self._calculate_load_balance_impact(node)
                
                # Combined selection score
                selection_score = (
                    0.4 * performance_score +
                    0.3 * resource_fit +
                    0.2 * load_balance_score +
                    0.1 * node.reliability_score
                )
                
                if selection_score > best_score:
                    best_score = selection_score
                    best_node = node
            
            return best_node
            
        except Exception as e:
            self.logger.error(f"ML node selection failed: {e}")
            return None
    
    async def _assign_task_to_node(self, task: DistributedMLTask, node: MLNode):
        """Assign task to node and initiate execution"""
        
        try:
            with self.coordination_lock:
                # Update task state
                task.status = TaskStatus.ASSIGNED
                task.assigned_node = node.node_id
                
                # Update node state
                node.assigned_tasks.append(task.task_id)
                node.task_queue_size += 1
            
            # Initiate task execution (simulated)
            asyncio.create_task(self._execute_distributed_task(task, node))
            
            self.logger.info(f"Task {task.task_id} assigned to node {node.node_id}")
            
        except Exception as e:
            self.logger.error(f"Task assignment failed: {e}")
    
    async def _execute_distributed_task(self, task: DistributedMLTask, node: MLNode):
        """Execute distributed ML task on assigned node"""
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Use ML-predicted duration or estimated duration
            execution_duration = task.predicted_duration or task.estimated_duration or 300.0
            
            # Simulate task execution
            await asyncio.sleep(min(execution_duration, 60))  # Cap simulation time
            
            # Task completion
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update node statistics
            node.tasks_completed += 1
            node.assigned_tasks.remove(task.task_id)
            node.task_queue_size -= 1
            
            # Calculate execution efficiency
            actual_duration = (task.completed_at - task.started_at).total_seconds()
            if task.predicted_duration > 0:
                task.execution_efficiency = min(1.0, task.predicted_duration / actual_duration)
            
            # Move to completed tasks
            with self.coordination_lock:
                self.completed_tasks.append(task)
                del self.distributed_tasks[task.task_id]
            
            self.coordination_stats['tasks_completed'] += 1
            
            # Update ML models with execution data
            if self.enable_ml_optimization:
                await self._update_ml_models_with_execution_data(task, node)
            
            self.logger.info(f"Distributed task completed: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Distributed task execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            node.tasks_failed += 1
            node.assigned_tasks.remove(task.task_id)
            node.task_queue_size -= 1
            
            with self.coordination_lock:
                self.failed_tasks.append(task)
                del self.distributed_tasks[task.task_id]
            
            self.coordination_stats['tasks_failed'] += 1
    
    async def _coordination_monitoring_loop(self):
        """Monitor coordination system health and performance"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Monitor node health
                await self._monitor_node_health()
                
                # Check for failed nodes
                await self._check_node_failures()
                
                # Update system metrics
                await self._update_coordination_metrics()
                
                # Auto-scaling if enabled
                if self.auto_scaling:
                    await self._evaluate_auto_scaling()
                
            except Exception as e:
                self.logger.error(f"Coordination monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _ml_coordination_loop(self):
        """ML optimization and insights generation for coordination"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                if len(self.coordination_feature_history) >= 200:
                    # Retrain coordination models
                    await self._retrain_coordination_models()
                    
                    # Generate workload optimization strategies
                    await self._generate_workload_strategies()
                    
                    # Predict node failures
                    await self._predict_node_failures()
                    
                    # Generate coordination insights
                    await self._generate_coordination_insights()
                
            except Exception as e:
                self.logger.error(f"ML coordination loop error: {e}")
                await asyncio.sleep(30)
    
    def get_coordination_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive distributed coordination dashboard"""
        
        # Node status summary
        node_summary = {}
        for node_id, node in self.ml_nodes.items():
            node_summary[node_id] = {
                'node_type': node.node_type.value,
                'status': node.status,
                'cpu_usage': node.cpu_usage,
                'memory_usage': node.memory_usage,
                'gpu_usage': node.gpu_usage,
                'assigned_tasks': len(node.assigned_tasks),
                'performance_score': node.performance_score,
                'reliability_score': node.reliability_score,
                'tasks_completed': node.tasks_completed,
                'success_rate': node.success_rate
            }
        
        # Task distribution summary
        task_status_counts = defaultdict(int)
        for task in self.distributed_tasks.values():
            task_status_counts[task.status.value] += 1
        
        # System performance metrics
        system_metrics = {
            'total_nodes': len(self.ml_nodes),
            'active_nodes': len([n for n in self.ml_nodes.values() if n.status == 'active']),
            'system_load': self._calculate_system_load(),
            'average_node_utilization': self._calculate_average_node_utilization(),
            'task_completion_rate': self._calculate_task_completion_rate()
        }
        
        # ML insights
        ml_status = {
            'ml_optimization_enabled': self.enable_ml_optimization,
            'feature_history_size': len(self.coordination_feature_history),
            'workload_strategies': len(self.workload_strategies),
            'failure_predictions': len(self.failure_predictions),
            'ml_recommendations': len(self.ml_recommendations)
        }
        
        return {
            'coordination_overview': system_metrics,
            'nodes': node_summary,
            'task_distribution': dict(task_status_counts),
            'active_tasks': len(self.distributed_tasks),
            'task_queue_size': len(self.task_queue),
            'statistics': self.coordination_stats.copy(),
            'ml_status': ml_status,
            'recent_insights': self.ml_recommendations[-5:] if self.ml_recommendations else []
        }
    
    def _calculate_system_load(self) -> float:
        """Calculate overall system load"""
        
        if not self.ml_nodes:
            return 0.0
        
        total_capacity = sum(node.cpu_cores for node in self.ml_nodes.values() if node.status == 'active')
        total_usage = sum(node.cpu_cores * node.cpu_usage / 100.0 for node in self.ml_nodes.values() if node.status == 'active')
        
        return total_usage / total_capacity if total_capacity > 0 else 0.0
    
    def _calculate_average_node_utilization(self) -> float:
        """Calculate average node utilization"""
        
        active_nodes = [n for n in self.ml_nodes.values() if n.status == 'active']
        if not active_nodes:
            return 0.0
        
        return np.mean([
            (n.cpu_usage + n.memory_usage + n.gpu_usage) / 3.0
            for n in active_nodes
        ])
    
    def _calculate_task_completion_rate(self) -> float:
        """Calculate task completion rate"""
        
        total_tasks = self.coordination_stats['tasks_distributed']
        if total_tasks == 0:
            return 1.0
        
        completed_tasks = self.coordination_stats['tasks_completed']
        return completed_tasks / total_tasks
    
    async def shutdown(self):
        """Graceful shutdown of distributed coordinator"""
        
        self.logger.info("Shutting down distributed ML coordinator...")
        
        # Wait for active tasks to complete (with timeout)
        timeout = 120
        while self.distributed_tasks and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Distributed ML coordinator shutdown complete")