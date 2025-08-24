"""
Performance Optimization Engine
===============================
"""Core Module - Split from performance_optimization_engine.py"""


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import psutil
import collections


# ============================================================================
# PERFORMANCE MONITORING TYPES
# ============================================================================


class MetricType(Enum):
    """Performance metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTILE = "percentile"


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    REACTIVE = "reactive"         # React to performance issues
    PROACTIVE = "proactive"       # Prevent performance issues
    PREDICTIVE = "predictive"     # Predict and prevent issues
    ADAPTIVE = "adaptive"         # Adapt to changing conditions
    AGGRESSIVE = "aggressive"     # Maximize performance
    CONSERVATIVE = "conservative" # Stable performance


class AlertSeverity(Enum):
    """Performance alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_id: str = field(default_factory=lambda: f"metric_{uuid.uuid4().hex[:8]}")
    name: str = ""
    metric_type: MetricType = MetricType.GAUGE
    value: float = 0.0
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_system: str = ""
    component: str = ""
    
    # Statistical data
    min_value: float = float('inf')
    max_value: float = float('-inf')
    avg_value: float = 0.0
    sample_count: int = 0
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class PerformanceProfile:
    """System performance profile"""
    profile_id: str = field(default_factory=lambda: f"profile_{uuid.uuid4().hex[:8]}")
    system_name: str = ""
    component: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    
    # Performance metrics
    response_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate_percent: float = 0.0
    success_rate_percent: float = 100.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Analysis results
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str = field(default_factory=lambda: f"rec_{uuid.uuid4().hex[:8]}")
    title: str = ""
    description: str = ""
    category: str = ""
    priority: int = 5  # 1-10 scale
    impact_estimate: str = ""
    effort_estimate: str = ""
    implementation_steps: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, rejected


# ============================================================================
# REAL-TIME METRICS COLLECTOR
# ============================================================================

class RealTimeMetricsCollector:
    """Advanced real-time metrics collection system"""
    
    def __init__(self):
        self.logger = logging.getLogger("realtime_metrics_collector")
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metric_history: Dict[str, collections.deque] = {}
        self.collection_lock = threading.Lock()
        
        # Collection configuration
        self.collection_interval = 1.0  # seconds
        self.max_history_size = 1000
        self.collection_enabled = True
        
        # System monitoring
        self.system_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_usage_percent": 0.0,
            "network_bytes_sent": 0,
            "network_bytes_recv": 0
        }
        
        # Collection workers
        self.collection_tasks: Dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("Real-time metrics collector initialized")
    
    async def start_collection(self):
        """Start metrics collection"""
        self.collection_enabled = True
        self.shutdown_event.clear()
        
        # Start system metrics collection
        self.collection_tasks["system"] = asyncio.create_task(self._collect_system_metrics())
        
        # Start application metrics collection
        self.collection_tasks["application"] = asyncio.create_task(self._collect_application_metrics())
        
        self.logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.collection_enabled = False
        self.shutdown_event.set()
        
        # Wait for collection tasks to complete
        for task_name, task in self.collection_tasks.items():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                self.logger.warning(f"Collection task {task_name} was cancelled due to timeout")
        
        self.collection_tasks.clear()
        self.logger.info("Metrics collection stopped")
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        while not self.shutdown_event.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.record_metric("system.cpu.usage_percent", cpu_percent, MetricType.GAUGE, "percent")
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system.memory.usage_percent", memory.percent, MetricType.GAUGE, "percent")
                self.record_metric("system.memory.usage_mb", memory.used / 1024 / 1024, MetricType.GAUGE, "MB")
                self.record_metric("system.memory.available_mb", memory.available / 1024 / 1024, MetricType.GAUGE, "MB")
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.record_metric("system.disk.usage_percent", disk_percent, MetricType.GAUGE, "percent")
                
                # Network I/O
                network = psutil.net_io_counters()
                self.record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER, "bytes")
                self.record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER, "bytes")
                
                # Process count
                process_count = len(psutil.pids())
                self.record_metric("system.processes.count", process_count, MetricType.GAUGE, "count")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_application_metrics(self):
        """Collect application-level performance metrics"""
        while not self.shutdown_event.is_set():
            try:
                # Simulate application metrics
                current_time = time.time()
                
                # API Gateway metrics
                self.record_metric("api_gateway.requests_per_second", 
                                 50 + (current_time % 10) * 5, MetricType.RATE, "req/s")
                self.record_metric("api_gateway.response_time_ms", 
                                 45 + (current_time % 5) * 10, MetricType.TIMER, "ms")
                self.record_metric("api_gateway.error_rate_percent", 
                                 0.5 + (current_time % 20) * 0.1, MetricType.RATE, "percent")
                
                # Orchestration metrics
                self.record_metric("orchestration.workflows_active", 
                                 10 + int(current_time % 15), MetricType.GAUGE, "count")
                self.record_metric("orchestration.tasks_per_second", 
                                 25 + (current_time % 8) * 3, MetricType.RATE, "tasks/s")
                self.record_metric("orchestration.queue_depth", 
                                 5 + int(current_time % 12), MetricType.GAUGE, "count")
                
                # Integration metrics
                self.record_metric("integration.messages_per_second", 
                                 100 + (current_time % 6) * 20, MetricType.RATE, "msg/s")
                self.record_metric("integration.transformation_time_ms", 
                                 15 + (current_time % 4) * 5, MetricType.TIMER, "ms")
                
                # Coordination metrics
                self.record_metric("coordination.services_registered", 
                                 20 + int(current_time % 5), MetricType.GAUGE, "count")
                self.record_metric("coordination.health_check_success_rate", 
                                 98 + (current_time % 3), MetricType.RATE, "percent")
                
                # Analytics metrics
                self.record_metric("analytics.events_processed_per_second", 
                                 200 + (current_time % 10) * 30, MetricType.RATE, "events/s")
                self.record_metric("analytics.processing_latency_ms", 
                                 80 + (current_time % 7) * 15, MetricType.TIMER, "ms")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        try:
            with self.collection_lock:
                # Update or create metric
                if name in self.metrics:
                    metric = self.metrics[name]
                    metric.value = value
                    metric.timestamp = datetime.now()
                    
                    # Update statistics
                    metric.sample_count += 1
                    metric.min_value = min(metric.min_value, value)
                    metric.max_value = max(metric.max_value, value)
                    metric.avg_value = ((metric.avg_value * (metric.sample_count - 1)) + value) / metric.sample_count
                else:
                    # Create new metric
                    metric = PerformanceMetric(
                        name=name,
                        metric_type=metric_type,
                        value=value,
                        unit=unit,
                        tags=tags or {},
                        min_value=value,
                        max_value=value,
                        avg_value=value,
                        sample_count=1
                    )
                    self.metrics[name] = metric
                
                # Store in history
                if name not in self.metric_history:
                    self.metric_history[name] = collections.deque(maxlen=self.max_history_size)
                
                self.metric_history[name].append({
                    "value": value,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get specific metric"""
        with self.collection_lock:
            return self.metrics.get(name)
    
    def get_metric_history(self, name: str, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metric history for specified duration"""
        with self.collection_lock:
            if name not in self.metric_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            history = list(self.metric_history[name])
            
            # Filter by time
            filtered_history = []
            for entry in history:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time >= cutoff_time:
                    filtered_history.append(entry)
            
            return filtered_history
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all current metrics"""
        with self.collection_lock:
            return self.metrics.copy()


# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """Advanced performance analysis and bottleneck detection"""
    
    def __init__(self, metrics_collector: RealTimeMetricsCollector):
        self.logger = logging.getLogger("performance_analyzer")
        self.metrics_collector = metrics_collector
        
        # Analysis configuration
        self.analysis_interval = 60  # seconds
        self.baseline_window = 24    # hours
        self.anomaly_threshold = 2.0  # standard deviations
        
        # Analysis results
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.bottlenecks: List[Dict[str, Any]] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Analysis patterns
        self.analysis_patterns = {
            "high_cpu": self._analyze_high_cpu,
            "high_memory": self._analyze_high_memory,
            "slow_response": self._analyze_slow_response,
            "high_error_rate": self._analyze_high_error_rate,
            "resource_contention": self._analyze_resource_contention,
            "capacity_limits": self._analyze_capacity_limits
        }
        
        self.logger.info("Performance analyzer initialized")
    
    async def analyze_system_performance(self) -> Dict[str, Any]:
        """Perform comprehensive system performance analysis"""
        start_time = time.time()
        
        try:
            analysis_results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_duration_ms": 0.0,
                "overall_health_score": 0.0,
                "bottlenecks_detected": [],
                "performance_summary": {},
                "recommendations": [],
                "trend_analysis": {}