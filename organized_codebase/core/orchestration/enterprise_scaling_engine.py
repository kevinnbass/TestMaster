#!/usr/bin/env python3
"""
Enterprise Scaling Engine
Agent B Hours 80-90: Advanced Performance & Memory Optimization

Enterprise-grade horizontal scaling and auto-scaling system with predictive scaling,
resource optimization, cloud integration, and comprehensive monitoring.
"""

import asyncio
import logging
import time
import json
import psutil
import docker
import kubernetes
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import threading
import subprocess

class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"
    ML_BASED = "ml_based"

class ResourceType(Enum):
    """Types of resources to scale"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    CUSTOM = "custom"

class ScalingDirection(Enum):
    """Scaling direction"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"

class DeploymentTarget(Enum):
    """Deployment targets"""
    LOCAL_DOCKER = "local_docker"
    KUBERNETES = "kubernetes"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINERS = "azure_containers"
    GCP_RUN = "gcp_run"
    HYBRID_CLOUD = "hybrid_cloud"

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_length: int
    active_connections: int
    resource_utilization: Dict[str, float]
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ScalingRule:
    """Rule for scaling decisions"""
    rule_id: str
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    scaling_direction: ScalingDirection
    cooldown_seconds: int
    min_instances: int
    max_instances: int
    scaling_factor: float
    enabled: bool = True

@dataclass
class ScalingAction:
    """Scaling action to be executed"""
    action_id: str
    rule_id: str
    direction: ScalingDirection
    current_instances: int
    target_instances: int
    reason: str
    timestamp: datetime
    estimated_duration: float
    priority: int = 1

@dataclass
class DeploymentConfig:
    """Deployment configuration for scaling"""
    name: str
    target: DeploymentTarget
    image: str
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str
    replicas: int
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    ports: List[int] = field(default_factory=list)

class PredictiveScaler:
    """Predictive scaling using time series analysis"""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.prediction_window = 300  # 5 minutes
        self.confidence_threshold = 0.7
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics data point"""
        self.metrics_history.append(metrics)
    
    def predict_load(self, minutes_ahead: int = 5) -> Dict[str, float]:
        """Predict resource load using simple trend analysis"""
        if len(self.metrics_history) < 10:
            return {}
        
        # Simple linear trend prediction
        recent_metrics = list(self.metrics_history)[-60:]  # Last 60 data points
        
        predictions = {}
        
        # Predict CPU usage
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        predicted_cpu = cpu_values[-1] + (cpu_trend * minutes_ahead)
        predictions['cpu_usage'] = max(0, min(100, predicted_cpu))
        
        # Predict memory usage
        memory_values = [m.memory_usage for m in recent_metrics]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        predicted_memory = memory_values[-1] + (memory_trend * minutes_ahead)
        predictions['memory_usage'] = max(0, min(100, predicted_memory))
        
        # Predict request rate
        request_values = [m.request_rate for m in recent_metrics]
        request_trend = np.polyfit(range(len(request_values)), request_values, 1)[0]
        predicted_requests = request_values[-1] + (request_trend * minutes_ahead)
        predictions['request_rate'] = max(0, predicted_requests)
        
        return predictions
    
    def should_preemptive_scale(self, threshold: float = 80.0) -> bool:
        """Determine if preemptive scaling is needed"""
        predictions = self.predict_load()
        
        if not predictions:
            return False
        
        # Check if any metric is predicted to exceed threshold
        return (predictions.get('cpu_usage', 0) > threshold or 
                predictions.get('memory_usage', 0) > threshold)

class EnterpriseScalingEngine:
    """
    Enterprise Scaling Engine
    
    Advanced horizontal scaling and auto-scaling system with predictive scaling,
    resource optimization, multi-cloud deployment, and comprehensive monitoring
    for orchestration components at enterprise scale.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EnterpriseScalingEngine")
        
        # Scaling configuration
        self.scaling_strategy = ScalingStrategy.HYBRID
        self.scaling_enabled = True
        self.min_instances = 1
        self.max_instances = 100
        self.default_cooldown = 300  # 5 minutes
        
        # Metrics and monitoring
        self.metrics_history: deque = deque(maxlen=10000)
        self.current_metrics: Optional[ScalingMetrics] = None
        self.predictive_scaler = PredictiveScaler()
        
        # Scaling rules and actions
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.pending_actions: List[ScalingAction] = []
        self.action_history: List[ScalingAction] = []
        self.last_scaling_actions: Dict[str, datetime] = {}
        
        # Deployment configurations
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.current_instances: Dict[str, int] = defaultdict(int)
        
        # Container orchestration clients
        self.docker_client: Optional[docker.DockerClient] = None
        self.k8s_client: Optional[kubernetes.client.ApiClient] = None
        
        # Cloud integration
        self.cloud_providers: Dict[str, Any] = {}
        self.multi_cloud_enabled = False
        
        # Monitoring and optimization
        self.monitoring_enabled = True
        self.monitoring_interval = 30  # seconds
        self.optimization_enabled = True
        
        self._initialize_default_rules()
        self._initialize_container_clients()
        
        self.logger.info("Enterprise scaling engine initialized")
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules"""
        # CPU-based scaling
        cpu_rule = ScalingRule(
            rule_id="cpu_scaling",
            resource_type=ResourceType.CPU,
            metric_name="cpu_usage",
            threshold_up=75.0,
            threshold_down=30.0,
            scaling_direction=ScalingDirection.SCALE_OUT,
            cooldown_seconds=300,
            min_instances=1,
            max_instances=20,
            scaling_factor=1.5
        )
        self.scaling_rules[cpu_rule.rule_id] = cpu_rule
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            rule_id="memory_scaling",
            resource_type=ResourceType.MEMORY,
            metric_name="memory_usage",
            threshold_up=80.0,
            threshold_down=40.0,
            scaling_direction=ScalingDirection.SCALE_OUT,
            cooldown_seconds=300,
            min_instances=1,
            max_instances=15,
            scaling_factor=1.3
        )
        self.scaling_rules[memory_rule.rule_id] = memory_rule
        
        # Request rate-based scaling
        request_rule = ScalingRule(
            rule_id="request_rate_scaling",
            resource_type=ResourceType.CUSTOM,
            metric_name="request_rate",
            threshold_up=1000.0,
            threshold_down=200.0,
            scaling_direction=ScalingDirection.SCALE_OUT,
            cooldown_seconds=180,
            min_instances=2,
            max_instances=50,
            scaling_factor=2.0
        )
        self.scaling_rules[request_rule.rule_id] = request_rule
    
    def _initialize_container_clients(self):
        """Initialize container orchestration clients"""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
        
        try:
            # Initialize Kubernetes client
            kubernetes.config.load_incluster_config()  # Try in-cluster config first
            self.k8s_client = kubernetes.client.ApiClient()
            self.logger.info("Kubernetes client initialized (in-cluster)")
        except:
            try:
                kubernetes.config.load_kube_config()  # Try local config
                self.k8s_client = kubernetes.client.ApiClient()
                self.logger.info("Kubernetes client initialized (local config)")
            except Exception as e:
                self.logger.warning(f"Kubernetes client initialization failed: {e}")
    
    async def start_scaling_engine(self):
        """Start the scaling engine"""
        try:
            self.scaling_enabled = True
            
            # Start monitoring tasks
            if self.monitoring_enabled:
                asyncio.create_task(self._metrics_collection_loop())
                asyncio.create_task(self._scaling_decision_loop())
                asyncio.create_task(self._action_execution_loop())
            
            # Start optimization tasks
            if self.optimization_enabled:
                asyncio.create_task(self._optimization_loop())
            
            self.logger.info("Enterprise scaling engine started")
            
        except Exception as e:
            self.logger.error(f"Failed to start scaling engine: {e}")
    
    async def stop_scaling_engine(self) -> Dict[str, Any]:
        """Stop scaling engine and generate report"""
        try:
            self.scaling_enabled = False
            self.monitoring_enabled = False
            
            # Generate final report
            report = await self._generate_scaling_report()
            
            self.logger.info("Enterprise scaling engine stopped")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to stop scaling engine: {e}")
            return {"error": str(e)}
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.monitoring_enabled:
            try:
                metrics = await self._collect_scaling_metrics()
                if metrics:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    self.predictive_scaler.add_metrics(metrics)
                
                # Limit history size
                if len(self.metrics_history) > 10000:
                    # Keep recent metrics
                    self.metrics_history = deque(list(self.metrics_history)[-5000:], maxlen=10000)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _scaling_decision_loop(self):
        """Background scaling decision loop"""
        while self.scaling_enabled:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.current_metrics:
                    continue
                
                # Check reactive scaling
                await self._evaluate_reactive_scaling()
                
                # Check predictive scaling
                if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
                    await self._evaluate_predictive_scaling()
                
                # Check scheduled scaling
                if self.scaling_strategy in [ScalingStrategy.SCHEDULED, ScalingStrategy.HYBRID]:
                    await self._evaluate_scheduled_scaling()
                
            except Exception as e:
                self.logger.error(f"Scaling decision error: {e}")
    
    async def _action_execution_loop(self):
        """Background action execution loop"""
        while self.scaling_enabled:
            try:
                if self.pending_actions:
                    # Sort actions by priority
                    self.pending_actions.sort(key=lambda x: x.priority, reverse=True)
                    
                    # Execute highest priority action
                    action = self.pending_actions.pop(0)
                    await self._execute_scaling_action(action)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Action execution error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_scaling_metrics(self) -> Optional[ScalingMetrics]:
        """Collect current scaling metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            request_rate = getattr(net_io, 'packets_recv', 0)  # Approximation
            
            # Process metrics
            active_connections = len(psutil.net_connections())
            
            # Custom metrics (would be integrated with application metrics)
            response_time = self._get_average_response_time()
            error_rate = self._get_error_rate()
            queue_length = self._get_queue_length()
            
            # Resource utilization
            resource_utilization = {
                "cpu_cores": psutil.cpu_count(),
                "memory_gb": memory_info.total / (1024**3),
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=request_rate,
                response_time=response_time,
                error_rate=error_rate,
                queue_length=queue_length,
                active_connections=active_connections,
                resource_utilization=resource_utilization
            )
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return None
    
    def _get_average_response_time(self) -> float:
        """Get average response time (placeholder - integrate with actual metrics)"""
        return 100.0  # milliseconds
    
    def _get_error_rate(self) -> float:
        """Get current error rate (placeholder - integrate with actual metrics)"""
        return 0.1  # 0.1% error rate
    
    def _get_queue_length(self) -> int:
        """Get current queue length (placeholder - integrate with actual metrics)"""
        return 5
    
    async def _evaluate_reactive_scaling(self):
        """Evaluate reactive scaling based on current metrics"""
        if not self.current_metrics:
            return
        
        for rule_id, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            last_action = self.last_scaling_actions.get(rule_id)
            if last_action and (datetime.now() - last_action).seconds < rule.cooldown_seconds:
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(self.current_metrics, rule.metric_name)
            if metric_value is None:
                continue
            
            # Check if scaling is needed
            current_instances = self.current_instances.get(rule_id, rule.min_instances)
            
            if metric_value > rule.threshold_up and current_instances < rule.max_instances:
                # Scale up
                target_instances = min(
                    rule.max_instances,
                    int(current_instances * rule.scaling_factor)
                )
                
                action = ScalingAction(
                    action_id=f"reactive-{rule_id}-{int(time.time())}",
                    rule_id=rule_id,
                    direction=ScalingDirection.SCALE_OUT,
                    current_instances=current_instances,
                    target_instances=target_instances,
                    reason=f"Reactive scaling: {rule.metric_name} ({metric_value:.1f}) > threshold ({rule.threshold_up})",
                    timestamp=datetime.now(),
                    estimated_duration=60.0,
                    priority=2
                )
                
                self.pending_actions.append(action)
                self.logger.info(f"Scheduled scale-out action: {action.reason}")
            
            elif metric_value < rule.threshold_down and current_instances > rule.min_instances:
                # Scale down
                target_instances = max(
                    rule.min_instances,
                    int(current_instances / rule.scaling_factor)
                )
                
                action = ScalingAction(
                    action_id=f"reactive-{rule_id}-{int(time.time())}",
                    rule_id=rule_id,
                    direction=ScalingDirection.SCALE_IN,
                    current_instances=current_instances,
                    target_instances=target_instances,
                    reason=f"Reactive scaling: {rule.metric_name} ({metric_value:.1f}) < threshold ({rule.threshold_down})",
                    timestamp=datetime.now(),
                    estimated_duration=30.0,
                    priority=1
                )
                
                self.pending_actions.append(action)
                self.logger.info(f"Scheduled scale-in action: {action.reason}")
    
    async def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling based on trend analysis"""
        if len(self.metrics_history) < 60:  # Need sufficient history
            return
        
        # Check if preemptive scaling is needed
        if self.predictive_scaler.should_preemptive_scale():
            predictions = self.predictive_scaler.predict_load(minutes_ahead=5)
            
            for rule_id, rule in self.scaling_rules.items():
                if not rule.enabled:
                    continue
                
                predicted_value = predictions.get(rule.metric_name)
                if predicted_value is None:
                    continue
                
                current_instances = self.current_instances.get(rule_id, rule.min_instances)
                
                if predicted_value > rule.threshold_up and current_instances < rule.max_instances:
                    target_instances = min(
                        rule.max_instances,
                        int(current_instances * 1.2)  # Conservative predictive scaling
                    )
                    
                    action = ScalingAction(
                        action_id=f"predictive-{rule_id}-{int(time.time())}",
                        rule_id=rule_id,
                        direction=ScalingDirection.SCALE_OUT,
                        current_instances=current_instances,
                        target_instances=target_instances,
                        reason=f"Predictive scaling: {rule.metric_name} predicted to reach {predicted_value:.1f}",
                        timestamp=datetime.now(),
                        estimated_duration=45.0,
                        priority=3
                    )
                    
                    self.pending_actions.append(action)
                    self.logger.info(f"Scheduled predictive scale-out: {action.reason}")
    
    async def _evaluate_scheduled_scaling(self):
        """Evaluate scheduled scaling based on time patterns"""
        # This would implement time-based scaling patterns
        # For example, scale up during business hours, scale down at night
        current_hour = datetime.now().hour
        
        # Business hours scaling (9 AM to 6 PM)
        if 9 <= current_hour <= 18:
            # Ensure minimum instances for business hours
            for rule_id, rule in self.scaling_rules.items():
                current_instances = self.current_instances.get(rule_id, rule.min_instances)
                business_hours_min = max(rule.min_instances, 2)
                
                if current_instances < business_hours_min:
                    action = ScalingAction(
                        action_id=f"scheduled-{rule_id}-{int(time.time())}",
                        rule_id=rule_id,
                        direction=ScalingDirection.SCALE_OUT,
                        current_instances=current_instances,
                        target_instances=business_hours_min,
                        reason=f"Scheduled scaling: Business hours minimum instances",
                        timestamp=datetime.now(),
                        estimated_duration=60.0,
                        priority=1
                    )
                    
                    self.pending_actions.append(action)
    
    def _get_metric_value(self, metrics: ScalingMetrics, metric_name: str) -> Optional[float]:
        """Get metric value by name"""
        metric_map = {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "request_rate": metrics.request_rate,
            "response_time": metrics.response_time,
            "error_rate": metrics.error_rate,
            "queue_length": float(metrics.queue_length),
            "active_connections": float(metrics.active_connections)
        }
        
        return metric_map.get(metric_name) or metrics.custom_metrics.get(metric_name)
    
    async def _execute_scaling_action(self, action: ScalingAction):
        """Execute scaling action"""
        try:
            self.logger.info(f"Executing scaling action: {action.action_id}")
            
            # Get deployment configuration
            deployment = self.deployments.get(action.rule_id)
            if not deployment:
                # Use default deployment configuration
                deployment = self._create_default_deployment(action.rule_id)
                self.deployments[action.rule_id] = deployment
            
            # Execute scaling based on deployment target
            if deployment.target == DeploymentTarget.LOCAL_DOCKER:
                await self._scale_docker_containers(deployment, action)
            elif deployment.target == DeploymentTarget.KUBERNETES:
                await self._scale_kubernetes_deployment(deployment, action)
            else:
                self.logger.warning(f"Unsupported deployment target: {deployment.target}")
            
            # Update instance count
            self.current_instances[action.rule_id] = action.target_instances
            
            # Record action
            self.action_history.append(action)
            self.last_scaling_actions[action.rule_id] = datetime.now()
            
            self.logger.info(f"Scaling action completed: {action.rule_id} -> {action.target_instances} instances")
            
        except Exception as e:
            self.logger.error(f"Scaling action failed: {action.action_id}: {e}")
    
    def _create_default_deployment(self, name: str) -> DeploymentConfig:
        """Create default deployment configuration"""
        return DeploymentConfig(
            name=name,
            target=DeploymentTarget.LOCAL_DOCKER,
            image="testmaster-orchestration:latest",
            cpu_request="100m",
            memory_request="128Mi",
            cpu_limit="500m",
            memory_limit="512Mi",
            replicas=1,
            environment={"SERVICE_NAME": name},
            ports=[8080]
        )
    
    async def _scale_docker_containers(self, deployment: DeploymentConfig, action: ScalingAction):
        """Scale Docker containers"""
        if not self.docker_client:
            raise Exception("Docker client not available")
        
        try:
            # Get current containers
            containers = self.docker_client.containers.list(
                filters={"label": f"service={deployment.name}"}
            )
            
            current_count = len(containers)
            target_count = action.target_instances
            
            if target_count > current_count:
                # Scale out - create new containers
                for i in range(target_count - current_count):
                    container_name = f"{deployment.name}-{current_count + i + 1}"
                    
                    self.docker_client.containers.run(
                        deployment.image,
                        name=container_name,
                        environment=deployment.environment,
                        ports={f"{deployment.ports[0]}/tcp": None} if deployment.ports else None,
                        labels={"service": deployment.name},
                        detach=True
                    )
                    
                    self.logger.info(f"Created container: {container_name}")
            
            elif target_count < current_count:
                # Scale in - remove containers
                containers_to_remove = containers[target_count:]
                for container in containers_to_remove:
                    container.stop()
                    container.remove()
                    self.logger.info(f"Removed container: {container.name}")
            
        except Exception as e:
            self.logger.error(f"Docker scaling failed: {e}")
            raise
    
    async def _scale_kubernetes_deployment(self, deployment: DeploymentConfig, action: ScalingAction):
        """Scale Kubernetes deployment"""
        if not self.k8s_client:
            raise Exception("Kubernetes client not available")
        
        try:
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)
            
            # Update deployment replica count
            body = {'spec': {'replicas': action.target_instances}}
            
            apps_v1.patch_namespaced_deployment_scale(
                name=deployment.name,
                namespace="default",  # Could be configurable
                body=body
            )
            
            self.logger.info(f"Scaled Kubernetes deployment {deployment.name} to {action.target_instances} replicas")
            
        except Exception as e:
            self.logger.error(f"Kubernetes scaling failed: {e}")
            raise
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while self.optimization_enabled:
            try:
                await asyncio.sleep(600)  # Optimize every 10 minutes
                
                # Optimize scaling rules based on performance
                await self._optimize_scaling_rules()
                
                # Optimize resource allocation
                await self._optimize_resource_allocation()
                
                # Clean up old actions
                await self._cleanup_action_history()
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    async def _optimize_scaling_rules(self):
        """Optimize scaling rules based on historical performance"""
        if len(self.action_history) < 10:
            return
        
        # Analyze scaling action effectiveness
        for rule_id, rule in self.scaling_rules.items():
            rule_actions = [a for a in self.action_history if a.rule_id == rule_id]
            if len(rule_actions) < 5:
                continue
            
            # Calculate average time between actions
            action_intervals = []
            for i in range(1, len(rule_actions)):
                interval = (rule_actions[i].timestamp - rule_actions[i-1].timestamp).seconds
                action_intervals.append(interval)
            
            if action_intervals:
                avg_interval = np.mean(action_intervals)
                
                # Adjust cooldown based on action frequency
                if avg_interval < rule.cooldown_seconds * 0.5:
                    # Actions are too frequent - increase cooldown
                    rule.cooldown_seconds = min(rule.cooldown_seconds * 1.2, 900)
                    self.logger.info(f"Increased cooldown for {rule_id} to {rule.cooldown_seconds}s")
                elif avg_interval > rule.cooldown_seconds * 2:
                    # Actions are infrequent - decrease cooldown
                    rule.cooldown_seconds = max(rule.cooldown_seconds * 0.9, 60)
                    self.logger.info(f"Decreased cooldown for {rule_id} to {rule.cooldown_seconds}s")
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation across deployments"""
        # This would implement resource optimization logic
        pass
    
    async def _cleanup_action_history(self):
        """Clean up old scaling actions"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        initial_count = len(self.action_history)
        
        self.action_history = [a for a in self.action_history if a.timestamp > cutoff_time]
        
        cleaned_count = initial_count - len(self.action_history)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old scaling actions")
    
    async def _generate_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling report"""
        return {
            "scaling_summary": {
                "scaling_enabled": self.scaling_enabled,
                "scaling_strategy": self.scaling_strategy.value,
                "total_rules": len(self.scaling_rules),
                "active_deployments": len(self.deployments),
                "current_instances": dict(self.current_instances)
            },
            "scaling_rules": [
                {
                    "rule_id": rule.rule_id,
                    "resource_type": rule.resource_type.value,
                    "metric_name": rule.metric_name,
                    "threshold_up": rule.threshold_up,
                    "threshold_down": rule.threshold_down,
                    "enabled": rule.enabled
                }
                for rule in self.scaling_rules.values()
            ],
            "recent_actions": [
                {
                    "action_id": action.action_id,
                    "rule_id": action.rule_id,
                    "direction": action.direction.value,
                    "instances_change": f"{action.current_instances} -> {action.target_instances}",
                    "reason": action.reason,
                    "timestamp": action.timestamp.isoformat()
                }
                for action in self.action_history[-10:]
            ],
            "current_metrics": {
                "cpu_usage": self.current_metrics.cpu_usage if self.current_metrics else 0,
                "memory_usage": self.current_metrics.memory_usage if self.current_metrics else 0,
                "request_rate": self.current_metrics.request_rate if self.current_metrics else 0,
                "active_connections": self.current_metrics.active_connections if self.current_metrics else 0
            } if self.current_metrics else {},
            "deployments": [
                {
                    "name": dep.name,
                    "target": dep.target.value,
                    "replicas": dep.replicas,
                    "current_instances": self.current_instances.get(dep.name, 0)
                }
                for dep in self.deployments.values()
            ]
        }
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get current scaling statistics"""
        return {
            "total_instances": sum(self.current_instances.values()),
            "total_deployments": len(self.deployments),
            "pending_actions": len(self.pending_actions),
            "completed_actions": len(self.action_history),
            "scaling_enabled": self.scaling_enabled,
            "current_metrics": {
                "cpu": self.current_metrics.cpu_usage if self.current_metrics else 0,
                "memory": self.current_metrics.memory_usage if self.current_metrics else 0
            }
        }
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.scaling_rules[rule.rule_id] = rule
        self.logger.info(f"Added scaling rule: {rule.rule_id}")
    
    def add_deployment(self, deployment: DeploymentConfig):
        """Add deployment configuration"""
        self.deployments[deployment.name] = deployment
        self.current_instances[deployment.name] = deployment.replicas
        self.logger.info(f"Added deployment: {deployment.name}")
    
    async def manual_scale(self, deployment_name: str, target_instances: int, reason: str = "Manual scaling"):
        """Manually trigger scaling action"""
        current_instances = self.current_instances.get(deployment_name, 1)
        
        action = ScalingAction(
            action_id=f"manual-{deployment_name}-{int(time.time())}",
            rule_id=deployment_name,
            direction=ScalingDirection.SCALE_OUT if target_instances > current_instances else ScalingDirection.SCALE_IN,
            current_instances=current_instances,
            target_instances=target_instances,
            reason=reason,
            timestamp=datetime.now(),
            estimated_duration=60.0,
            priority=5
        )
        
        self.pending_actions.append(action)
        self.logger.info(f"Manual scaling scheduled: {deployment_name} -> {target_instances} instances")