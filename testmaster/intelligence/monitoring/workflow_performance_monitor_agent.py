"""
Workflow Performance Monitor Agent

Intelligent workflow performance monitoring that combines real-time telemetry
with predictive analytics, consensus-driven alerting, and adaptive threshold
management for comprehensive TestMaster workflow optimization.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json
from collections import defaultdict, deque

from ..hierarchical_planning import (
    HierarchicalTestPlanner, 
    PlanningNode, 
    TestPlanGenerator, 
    TestPlanEvaluator,
    EvaluationCriteria,
    get_best_planner
)
from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import get_shared_state, cache_test_result, get_cached_test_result
from ...telemetry.flow_analyzer import get_flow_analyzer, FlowAnalysis
from ...flow_optimizer.flow_analyzer import get_flow_analyzer as get_legacy_flow_analyzer
from ...telemetry.performance_monitor import get_performance_monitor
from ...telemetry.telemetry_collector import get_telemetry_collector


class MonitoringScope(Enum):
    """Scope of performance monitoring."""
    WORKFLOW_LEVEL = "workflow_level"
    COMPONENT_LEVEL = "component_level"
    SYSTEM_LEVEL = "system_level"
    AGENT_LEVEL = "agent_level"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    CONCURRENCY = "concurrency"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_type: PerformanceMetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    adaptive: bool = True
    baseline_period_hours: int = 24
    confidence_interval: float = 0.95


@dataclass
class PerformanceAlert:
    """Performance alert."""
    alert_id: str
    severity: AlertSeverity
    metric_type: PerformanceMetricType
    component: str
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    throughput_tps: float = 0.0
    error_rate: float = 0.0
    resource_utilization: float = 0.0
    queue_depth: int = 0
    response_time_p95: float = 0.0
    availability: float = 1.0
    concurrency_level: int = 1
    bottlenecks: List[str] = field(default_factory=list)
    performance_score: float = 1.0


@dataclass
class MonitoringConfiguration:
    """Monitoring configuration."""
    monitoring_scope: MonitoringScope = MonitoringScope.WORKFLOW_LEVEL
    collection_interval_seconds: int = 30
    analysis_window_minutes: int = 15
    alert_cooldown_minutes: int = 5
    enable_predictive_alerts: bool = True
    enable_adaptive_thresholds: bool = True
    max_alerts_per_hour: int = 10
    performance_baselines_days: int = 7


class PerformanceDataCollector:
    """Collects performance data from various sources."""
    
    def __init__(self):
        self.flow_analyzer = get_flow_analyzer()
        self.legacy_flow_analyzer = get_legacy_flow_analyzer()
        self.performance_monitor = get_performance_monitor()
        self.telemetry_collector = get_telemetry_collector()
        self.shared_state = get_shared_state()
        
        print("Performance Data Collector initialized")
        print("   Sources: flow analyzer, performance monitor, telemetry")
    
    def collect_workflow_metrics(self, workflow_id: str) -> WorkflowMetrics:
        """Collect comprehensive workflow metrics."""
        
        current_time = datetime.now()
        
        # Get flow analysis data
        try:
            flow_analysis = self.flow_analyzer.analyze_flows(timeframe_hours=1)
        except Exception as e:
            print(f"Flow analysis failed: {e}")
            flow_analysis = None
        
        # Get legacy flow analysis
        try:
            legacy_execution_data = self._get_legacy_execution_data(workflow_id)
            legacy_analysis = self.legacy_flow_analyzer.analyze_flow(workflow_id, legacy_execution_data)
        except Exception as e:
            print(f"Legacy flow analysis failed: {e}")
            legacy_analysis = None
        
        # Collect telemetry data
        telemetry_stats = self._collect_telemetry_stats(workflow_id)
        
        # Calculate metrics
        duration_ms = telemetry_stats.get('avg_duration_ms', 0.0)
        throughput_tps = telemetry_stats.get('throughput_tps', 0.0)
        error_rate = telemetry_stats.get('error_rate', 0.0)
        resource_utilization = telemetry_stats.get('resource_utilization', 0.0)
        queue_depth = telemetry_stats.get('queue_depth', 0)
        response_time_p95 = telemetry_stats.get('response_time_p95', 0.0)
        concurrency_level = telemetry_stats.get('concurrency_level', 1)
        
        # Extract bottlenecks
        bottlenecks = []
        if flow_analysis:
            bottlenecks.extend(flow_analysis.bottleneck_components)
        if legacy_analysis:
            bottlenecks.extend([b.location for b in legacy_analysis.bottlenecks])
        
        # Calculate availability
        availability = 1.0 - error_rate if error_rate < 1.0 else 0.0
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            duration_ms, throughput_tps, error_rate, resource_utilization
        )
        
        return WorkflowMetrics(
            workflow_id=workflow_id,
            start_time=current_time - timedelta(hours=1),
            end_time=current_time,
            duration_ms=duration_ms,
            throughput_tps=throughput_tps,
            error_rate=error_rate,
            resource_utilization=resource_utilization,
            queue_depth=queue_depth,
            response_time_p95=response_time_p95,
            availability=availability,
            concurrency_level=concurrency_level,
            bottlenecks=list(set(bottlenecks)),  # Remove duplicates
            performance_score=performance_score
        )
    
    def _get_legacy_execution_data(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get execution data for legacy flow analyzer."""
        # Simulate execution data from shared state or telemetry
        return [
            {
                'execution_time': 250.0,
                'wait_time': 50.0,
                'resource_usage': 65.0,
                'dependencies': ['module_a', 'module_b']
            },
            {
                'execution_time': 180.0,
                'wait_time': 30.0,
                'resource_usage': 45.0,
                'dependencies': ['module_c']
            }
        ]
    
    def _collect_telemetry_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Collect telemetry statistics."""
        try:
            # Get basic telemetry stats
            stats = self.telemetry_collector.get_statistics()
            
            # Calculate derived metrics
            events_count = stats.get('events_collected', 1)
            errors_count = stats.get('errors_count', 0)
            avg_duration = stats.get('avg_duration_ms', 200.0)
            
            return {
                'avg_duration_ms': avg_duration,
                'throughput_tps': max(0.1, events_count / 60.0),  # Events per second
                'error_rate': min(1.0, errors_count / max(events_count, 1)),
                'resource_utilization': min(100.0, avg_duration / 10.0),  # Simulated
                'queue_depth': max(0, events_count - errors_count),
                'response_time_p95': avg_duration * 1.5,  # Simulated P95
                'concurrency_level': min(8, max(1, int(events_count / 10)))
            }
        except Exception as e:
            print(f"Telemetry stats collection failed: {e}")
            return {
                'avg_duration_ms': 200.0,
                'throughput_tps': 1.0,
                'error_rate': 0.05,
                'resource_utilization': 50.0,
                'queue_depth': 0,
                'response_time_p95': 300.0,
                'concurrency_level': 2
            }
    
    def _calculate_performance_score(self, duration_ms: float, throughput_tps: float, 
                                   error_rate: float, resource_utilization: float) -> float:
        """Calculate overall performance score (0-1)."""
        
        # Individual component scores
        duration_score = max(0.0, min(1.0, 1.0 - (duration_ms / 1000.0)))  # Normalize to 1 second
        throughput_score = min(1.0, throughput_tps / 10.0)  # Normalize to 10 TPS
        error_score = max(0.0, 1.0 - (error_rate * 2.0))  # Penalize errors heavily
        resource_score = max(0.0, 1.0 - (resource_utilization / 100.0))
        
        # Weighted average
        performance_score = (
            duration_score * 0.3 +
            throughput_score * 0.25 +
            error_score * 0.3 +
            resource_score * 0.15
        )
        
        return round(performance_score, 3)


class ThresholdManager:
    """Manages adaptive performance thresholds."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.thresholds: Dict[str, Dict[PerformanceMetricType, PerformanceThreshold]] = {}
        self.baselines: Dict[str, Dict[PerformanceMetricType, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.RLock()
        
        # Default thresholds
        self._initialize_default_thresholds()
        
        print("Threshold Manager initialized")
        print(f"   Adaptive thresholds: {config.enable_adaptive_thresholds}")
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds."""
        default_thresholds = {
            PerformanceMetricType.LATENCY: PerformanceThreshold(
                metric_type=PerformanceMetricType.LATENCY,
                warning_threshold=500.0,   # 500ms
                critical_threshold=1000.0, # 1s
                emergency_threshold=2000.0 # 2s
            ),
            PerformanceMetricType.THROUGHPUT: PerformanceThreshold(
                metric_type=PerformanceMetricType.THROUGHPUT,
                warning_threshold=5.0,     # 5 TPS
                critical_threshold=2.0,    # 2 TPS
                emergency_threshold=1.0    # 1 TPS
            ),
            PerformanceMetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=PerformanceMetricType.ERROR_RATE,
                warning_threshold=0.05,    # 5%
                critical_threshold=0.10,   # 10%
                emergency_threshold=0.25   # 25%
            ),
            PerformanceMetricType.RESOURCE_UTILIZATION: PerformanceThreshold(
                metric_type=PerformanceMetricType.RESOURCE_UTILIZATION,
                warning_threshold=80.0,    # 80%
                critical_threshold=90.0,   # 90%
                emergency_threshold=95.0   # 95%
            )
        }
        
        self.thresholds["default"] = default_thresholds
    
    def update_baselines(self, workflow_id: str, metrics: WorkflowMetrics):
        """Update performance baselines for adaptive thresholds."""
        if not self.config.enable_adaptive_thresholds:
            return
        
        with self.lock:
            # Add current metrics to baselines
            baseline_data = self.baselines[workflow_id]
            
            baseline_data[PerformanceMetricType.LATENCY].append(metrics.duration_ms)
            baseline_data[PerformanceMetricType.THROUGHPUT].append(metrics.throughput_tps)
            baseline_data[PerformanceMetricType.ERROR_RATE].append(metrics.error_rate)
            baseline_data[PerformanceMetricType.RESOURCE_UTILIZATION].append(metrics.resource_utilization)
            
            # Keep only recent data (configurable window)
            max_samples = self.config.performance_baselines_days * 24 * 2  # 2 samples per hour
            for metric_type in baseline_data:
                if len(baseline_data[metric_type]) > max_samples:
                    baseline_data[metric_type] = baseline_data[metric_type][-max_samples:]
            
            # Update adaptive thresholds
            self._update_adaptive_thresholds(workflow_id)
    
    def _update_adaptive_thresholds(self, workflow_id: str):
        """Update adaptive thresholds based on historical data."""
        baseline_data = self.baselines[workflow_id]
        
        if workflow_id not in self.thresholds:
            self.thresholds[workflow_id] = {}
        
        for metric_type, values in baseline_data.items():
            if len(values) >= 10:  # Minimum samples for adaptation
                # Calculate statistical thresholds
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Define thresholds based on standard deviations
                if metric_type in [PerformanceMetricType.LATENCY, PerformanceMetricType.ERROR_RATE, 
                                 PerformanceMetricType.RESOURCE_UTILIZATION]:
                    # Higher is worse
                    warning_threshold = mean_val + (1 * std_val)
                    critical_threshold = mean_val + (2 * std_val)
                    emergency_threshold = mean_val + (3 * std_val)
                else:
                    # Lower is worse (throughput)
                    warning_threshold = max(0, mean_val - (1 * std_val))
                    critical_threshold = max(0, mean_val - (2 * std_val))
                    emergency_threshold = max(0, mean_val - (3 * std_val))
                
                # Create adaptive threshold
                self.thresholds[workflow_id][metric_type] = PerformanceThreshold(
                    metric_type=metric_type,
                    warning_threshold=warning_threshold,
                    critical_threshold=critical_threshold,
                    emergency_threshold=emergency_threshold,
                    adaptive=True,
                    baseline_period_hours=self.config.performance_baselines_days * 24,
                    confidence_interval=0.95
                )
    
    def get_thresholds(self, workflow_id: str) -> Dict[PerformanceMetricType, PerformanceThreshold]:
        """Get thresholds for a workflow."""
        with self.lock:
            return self.thresholds.get(workflow_id, self.thresholds["default"])
    
    def check_threshold_breach(self, workflow_id: str, metric_type: PerformanceMetricType, 
                             value: float) -> Optional[AlertSeverity]:
        """Check if a metric value breaches thresholds."""
        thresholds = self.get_thresholds(workflow_id)
        threshold = thresholds.get(metric_type)
        
        if not threshold:
            return None
        
        # Check emergency threshold
        if self._exceeds_threshold(metric_type, value, threshold.emergency_threshold):
            return AlertSeverity.EMERGENCY
        
        # Check critical threshold
        if self._exceeds_threshold(metric_type, value, threshold.critical_threshold):
            return AlertSeverity.CRITICAL
        
        # Check warning threshold
        if self._exceeds_threshold(metric_type, value, threshold.warning_threshold):
            return AlertSeverity.WARNING
        
        return None
    
    def _exceeds_threshold(self, metric_type: PerformanceMetricType, value: float, threshold: float) -> bool:
        """Check if value exceeds threshold based on metric type."""
        if metric_type in [PerformanceMetricType.LATENCY, PerformanceMetricType.ERROR_RATE, 
                          PerformanceMetricType.RESOURCE_UTILIZATION]:
            return value > threshold
        else:  # Throughput and similar - lower is worse
            return value < threshold


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        
        print("Alert Manager initialized")
        print(f"   Max alerts per hour: {config.max_alerts_per_hour}")
        print(f"   Alert cooldown: {config.alert_cooldown_minutes} minutes")
    
    def create_alert(self, severity: AlertSeverity, metric_type: PerformanceMetricType,
                    component: str, current_value: float, threshold_value: float,
                    message: str, recommendations: List[str] = None) -> Optional[PerformanceAlert]:
        """Create a new performance alert."""
        
        alert_key = f"{component}_{metric_type.value}_{severity.value}"
        current_time = datetime.now()
        
        with self.lock:
            # Check cooldown
            if alert_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[alert_key] + timedelta(minutes=self.config.alert_cooldown_minutes)
                if current_time < cooldown_end:
                    return None  # Still in cooldown
            
            # Check rate limiting
            recent_alerts = [alert for alert in self.alert_history 
                           if alert.timestamp > current_time - timedelta(hours=1)]
            if len(recent_alerts) >= self.config.max_alerts_per_hour:
                return None  # Rate limited
            
            # Create alert
            alert = PerformanceAlert(
                alert_id=f"alert_{int(current_time.timestamp())}_{component}",
                severity=severity,
                metric_type=metric_type,
                component=component,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message,
                recommendations=recommendations or [],
                timestamp=current_time
            )
            
            # Store alert
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            self.alert_cooldowns[alert_key] = current_time
            
            print(f"Performance alert created: {severity.value} - {component} - {message}")
            return alert
    
    def resolve_alert(self, alert_key: str) -> bool:
        """Resolve an active alert."""
        with self.lock:
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolution_time = datetime.now()
                del self.active_alerts[alert_key]
                print(f"Alert resolved: {alert.alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self.lock:
            active_count = len(self.active_alerts)
            total_count = len(self.alert_history)
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            return {
                "active_alerts": active_count,
                "total_alerts_generated": total_count,
                "severity_breakdown": dict(severity_counts),
                "rate_limited": len([a for a in self.alert_history 
                                   if a.timestamp > datetime.now() - timedelta(hours=1)]) >= self.config.max_alerts_per_hour
            }


class WorkflowPerformanceMonitorAgent:
    """Main workflow performance monitoring agent."""
    
    def __init__(self, 
                 coordinator: AgentCoordinator = None,
                 config: MonitoringConfiguration = None):
        
        self.coordinator = coordinator
        self.config = config or MonitoringConfiguration()
        self.shared_state = get_shared_state()
        
        # Initialize components
        self.data_collector = PerformanceDataCollector()
        self.threshold_manager = ThresholdManager(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.lock = threading.RLock()
        
        # Statistics
        self.workflows_monitored = 0
        self.alerts_generated = 0
        self.recommendations_provided = 0
        
        # Register with coordinator if provided
        if self.coordinator:
            self.coordinator.register_agent(
                "workflow_performance_monitor_agent",
                AgentRole.PERFORMANCE_OPTIMIZER,
                weight=1.1,
                specialization=["performance_monitoring", "workflow_optimization", "predictive_alerting"]
            )
        
        print("Workflow Performance Monitor Agent initialized")
        print(f"   Monitoring scope: {self.config.monitoring_scope.value}")
        print(f"   Collection interval: {self.config.collection_interval_seconds}s")
        print(f"   Analysis window: {self.config.analysis_window_minutes}m")
    
    def start_monitoring(self, workflow_ids: List[str] = None):
        """Start workflow performance monitoring."""
        
        if self.monitoring_active:
            print("Monitoring already active")
            return
        
        self.workflow_ids = workflow_ids or ["default_workflow"]
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print(f"Performance monitoring started for workflows: {self.workflow_ids}")
    
    def stop_monitoring(self):
        """Stop workflow performance monitoring."""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        print("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                for workflow_id in self.workflow_ids:
                    self._monitor_workflow(workflow_id)
                
                # Wait for next collection interval
                if self.shutdown_event.wait(timeout=self.config.collection_interval_seconds):
                    break
                    
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _monitor_workflow(self, workflow_id: str):
        """Monitor a specific workflow."""
        
        # Collect current metrics
        metrics = self.data_collector.collect_workflow_metrics(workflow_id)
        
        # Update baselines and thresholds
        self.threshold_manager.update_baselines(workflow_id, metrics)
        
        # Check for threshold breaches and create alerts
        self._check_and_create_alerts(workflow_id, metrics)
        
        # Store metrics in shared state
        self._store_metrics(workflow_id, metrics)
        
        # Coordinate with other agents for consensus if needed
        if self.coordinator and metrics.performance_score < 0.5:
            self._coordinate_performance_consensus(workflow_id, metrics)
        
        # Update statistics
        with self.lock:
            self.workflows_monitored += 1
    
    def _check_and_create_alerts(self, workflow_id: str, metrics: WorkflowMetrics):
        """Check metrics against thresholds and create alerts."""
        
        # Check latency
        severity = self.threshold_manager.check_threshold_breach(
            workflow_id, PerformanceMetricType.LATENCY, metrics.duration_ms
        )
        if severity:
            self.alert_manager.create_alert(
                severity=severity,
                metric_type=PerformanceMetricType.LATENCY,
                component=workflow_id,
                current_value=metrics.duration_ms,
                threshold_value=self.threshold_manager.get_thresholds(workflow_id)[PerformanceMetricType.LATENCY].warning_threshold,
                message=f"High latency detected: {metrics.duration_ms:.1f}ms",
                recommendations=["Optimize slow operations", "Consider parallel execution", "Review bottlenecks"]
            )
        
        # Check throughput
        severity = self.threshold_manager.check_threshold_breach(
            workflow_id, PerformanceMetricType.THROUGHPUT, metrics.throughput_tps
        )
        if severity:
            self.alert_manager.create_alert(
                severity=severity,
                metric_type=PerformanceMetricType.THROUGHPUT,
                component=workflow_id,
                current_value=metrics.throughput_tps,
                threshold_value=self.threshold_manager.get_thresholds(workflow_id)[PerformanceMetricType.THROUGHPUT].warning_threshold,
                message=f"Low throughput detected: {metrics.throughput_tps:.2f} TPS",
                recommendations=["Scale up resources", "Optimize processing pipeline", "Implement load balancing"]
            )
        
        # Check error rate
        severity = self.threshold_manager.check_threshold_breach(
            workflow_id, PerformanceMetricType.ERROR_RATE, metrics.error_rate
        )
        if severity:
            self.alert_manager.create_alert(
                severity=severity,
                metric_type=PerformanceMetricType.ERROR_RATE,
                component=workflow_id,
                current_value=metrics.error_rate,
                threshold_value=self.threshold_manager.get_thresholds(workflow_id)[PerformanceMetricType.ERROR_RATE].warning_threshold,
                message=f"High error rate detected: {metrics.error_rate:.1%}",
                recommendations=["Investigate error patterns", "Improve error handling", "Review input validation"]
            )
        
        # Check resource utilization
        severity = self.threshold_manager.check_threshold_breach(
            workflow_id, PerformanceMetricType.RESOURCE_UTILIZATION, metrics.resource_utilization
        )
        if severity:
            self.alert_manager.create_alert(
                severity=severity,
                metric_type=PerformanceMetricType.RESOURCE_UTILIZATION,
                component=workflow_id,
                current_value=metrics.resource_utilization,
                threshold_value=self.threshold_manager.get_thresholds(workflow_id)[PerformanceMetricType.RESOURCE_UTILIZATION].warning_threshold,
                message=f"High resource utilization: {metrics.resource_utilization:.1f}%",
                recommendations=["Scale resources", "Optimize resource usage", "Implement resource pooling"]
            )
    
    def _store_metrics(self, workflow_id: str, metrics: WorkflowMetrics):
        """Store metrics in shared state."""
        
        metrics_dict = {
            'workflow_id': workflow_id,
            'timestamp': metrics.start_time.isoformat(),
            'duration_ms': metrics.duration_ms,
            'throughput_tps': metrics.throughput_tps,
            'error_rate': metrics.error_rate,
            'resource_utilization': metrics.resource_utilization,
            'performance_score': metrics.performance_score,
            'bottlenecks': metrics.bottlenecks,
            'availability': metrics.availability
        }
        
        # Store current metrics
        self.shared_state.set(f"workflow_metrics_{workflow_id}", metrics_dict)
        
        # Cache historical metrics
        cache_test_result(f"workflow_performance_{workflow_id}", metrics_dict, metrics.performance_score * 100)
    
    def _coordinate_performance_consensus(self, workflow_id: str, metrics: WorkflowMetrics):
        """Coordinate with other agents for performance consensus."""
        
        try:
            # Create coordination task
            task_id = self.coordinator.create_coordination_task(
                description=f"Performance degradation detected in {workflow_id}",
                required_roles={AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.QUALITY_ASSESSOR},
                context={
                    'workflow_id': workflow_id,
                    'performance_score': metrics.performance_score,
                    'bottlenecks': metrics.bottlenecks,
                    'error_rate': metrics.error_rate
                }
            )
            
            # Submit performance assessment vote
            self.coordinator.submit_vote(
                task_id=task_id,
                agent_id="workflow_performance_monitor_agent",
                choice=metrics.performance_score,
                confidence=0.9,
                reasoning=f"Performance monitoring analysis: score {metrics.performance_score:.3f}, {len(metrics.bottlenecks)} bottlenecks detected"
            )
            
            print(f"Performance consensus requested for {workflow_id}")
            
        except Exception as e:
            print(f"Performance consensus coordination failed: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        
        with self.lock:
            active_alerts = self.alert_manager.get_active_alerts()
            alert_summary = self.alert_manager.get_alert_summary()
            
            return {
                'monitoring_active': self.monitoring_active,
                'workflows_monitored': len(self.workflow_ids) if hasattr(self, 'workflow_ids') else 0,
                'total_workflows_processed': self.workflows_monitored,
                'alerts_generated': len(self.alert_manager.alert_history),
                'active_alerts': len(active_alerts),
                'alert_summary': alert_summary,
                'collection_interval': self.config.collection_interval_seconds,
                'adaptive_thresholds_enabled': self.config.enable_adaptive_thresholds,
                'predictive_alerts_enabled': self.config.enable_predictive_alerts
            }
    
    def get_workflow_performance_report(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive performance report for a workflow."""
        
        # Get current metrics
        current_metrics_data = self.shared_state.get(f"workflow_metrics_{workflow_id}")
        
        # Get historical metrics
        historical_metrics = get_cached_test_result(f"workflow_performance_{workflow_id}")
        
        # Get thresholds
        thresholds = self.threshold_manager.get_thresholds(workflow_id)
        
        # Get active alerts
        active_alerts = [alert for alert in self.alert_manager.get_active_alerts() 
                        if alert.component == workflow_id]
        
        return {
            'workflow_id': workflow_id,
            'current_metrics': current_metrics_data,
            'historical_metrics': historical_metrics,
            'thresholds': {metric.value: {
                'warning': threshold.warning_threshold,
                'critical': threshold.critical_threshold,
                'emergency': threshold.emergency_threshold,
                'adaptive': threshold.adaptive
            } for metric, threshold in thresholds.items()},
            'active_alerts': [
                {
                    'severity': alert.severity.value,
                    'metric': alert.metric_type.value,
                    'message': alert.message,
                    'recommendations': alert.recommendations,
                    'timestamp': alert.timestamp.isoformat()
                } for alert in active_alerts
            ],
            'recommendations': self._generate_performance_recommendations(workflow_id, current_metrics_data),
            'report_generated': datetime.now().isoformat()
        }
    
    def _generate_performance_recommendations(self, workflow_id: str, metrics_data: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        
        if not metrics_data:
            return ["No metrics data available for recommendations"]
        
        performance_score = metrics_data.get('performance_score', 1.0)
        error_rate = metrics_data.get('error_rate', 0.0)
        resource_utilization = metrics_data.get('resource_utilization', 0.0)
        bottlenecks = metrics_data.get('bottlenecks', [])
        
        # Performance score recommendations
        if performance_score < 0.3:
            recommendations.append("Critical performance issues detected - immediate optimization required")
        elif performance_score < 0.6:
            recommendations.append("Performance below acceptable levels - optimization recommended")
        elif performance_score < 0.8:
            recommendations.append("Good performance with room for improvement")
        
        # Error rate recommendations
        if error_rate > 0.1:
            recommendations.append("High error rate - investigate error patterns and improve error handling")
        elif error_rate > 0.05:
            recommendations.append("Elevated error rate - monitor closely and review error sources")
        
        # Resource utilization recommendations
        if resource_utilization > 90:
            recommendations.append("Very high resource utilization - consider scaling up or optimizing resource usage")
        elif resource_utilization > 80:
            recommendations.append("High resource utilization - monitor for potential scaling needs")
        
        # Bottleneck recommendations
        if bottlenecks:
            recommendations.append(f"Bottlenecks detected in: {', '.join(bottlenecks[:3])} - optimize these components")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable ranges - continue monitoring")
        
        return recommendations


def test_workflow_performance_monitor():
    """Test the workflow performance monitor agent."""
    print("\n" + "="*60)
    print("Testing Workflow Performance Monitor Agent")
    print("="*60)
    
    # Create test configuration
    config = MonitoringConfiguration(
        monitoring_scope=MonitoringScope.WORKFLOW_LEVEL,
        collection_interval_seconds=5,
        analysis_window_minutes=1,
        enable_adaptive_thresholds=True,
        enable_predictive_alerts=True
    )
    
    # Create monitor agent
    agent = WorkflowPerformanceMonitorAgent(config=config)
    
    # Test data collection
    print("\n1. Testing data collection...")
    metrics = agent.data_collector.collect_workflow_metrics("test_workflow")
    print(f"   Performance score: {metrics.performance_score:.3f}")
    print(f"   Throughput: {metrics.throughput_tps:.2f} TPS")
    print(f"   Error rate: {metrics.error_rate:.1%}")
    print(f"   Resource utilization: {metrics.resource_utilization:.1f}%")
    
    # Test threshold management
    print("\n2. Testing threshold management...")
    agent.threshold_manager.update_baselines("test_workflow", metrics)
    thresholds = agent.threshold_manager.get_thresholds("test_workflow")
    print(f"   Thresholds configured: {len(thresholds)}")
    
    # Test alert creation
    print("\n3. Testing alert management...")
    alert = agent.alert_manager.create_alert(
        severity=AlertSeverity.WARNING,
        metric_type=PerformanceMetricType.LATENCY,
        component="test_workflow",
        current_value=600.0,
        threshold_value=500.0,
        message="Test alert for high latency",
        recommendations=["Optimize processing", "Review bottlenecks"]
    )
    
    if alert:
        print(f"   Alert created: {alert.severity.value} - {alert.message}")
    
    # Test monitoring status
    print("\n4. Testing monitoring status...")
    status = agent.get_monitoring_status()
    print(f"   Monitoring active: {status['monitoring_active']}")
    print(f"   Alerts generated: {status['alerts_generated']}")
    
    # Test performance report
    print("\n5. Testing performance report...")
    # Simulate some metrics first
    agent._store_metrics("test_workflow", metrics)
    report = agent.get_workflow_performance_report("test_workflow")
    print(f"   Report generated for: {report['workflow_id']}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"     {i}. {rec}")
    
    print("\nWorkflow Performance Monitor Agent test completed successfully!")
    return True


if __name__ == "__main__":
    test_workflow_performance_monitor()