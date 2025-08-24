"""
Adaptive Load Balancer - TestMaster Advanced ML
ML-driven load balancing with predictive scaling and intelligent routing
Enterprise ML Module #1/8 for comprehensive system intelligence
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
import random
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, accuracy_score


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    ML_PREDICTIVE = "ml_predictive"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    GEOGRAPHIC = "geographic"
    RESOURCE_AWARE = "resource_aware"


class BackendState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"


@dataclass
class BackendNode:
    """ML-enhanced backend node with intelligent metrics"""
    
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 1000
    current_connections: int = 0
    state: BackendState = BackendState.HEALTHY
    
    # Performance metrics
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    total_requests: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # ML Enhancement Fields
    predicted_load: float = 0.0
    capacity_score: float = 1.0
    reliability_score: float = 1.0
    performance_trend: str = "stable"
    optimal_weight: float = 1.0
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Health tracking
    last_health_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    downtime_duration: float = 0.0


@dataclass
class LoadBalancingDecision:
    """ML-driven load balancing decision with reasoning"""
    
    decision_id: str
    timestamp: datetime
    selected_backend: str
    strategy_used: RoutingStrategy
    request_hash: Optional[str] = None
    
    # Decision factors
    backend_scores: Dict[str, float] = field(default_factory=dict)
    ml_confidence: float = 0.0
    decision_reasoning: List[str] = field(default_factory=list)
    predicted_response_time: float = 0.0
    
    # Outcome tracking
    actual_response_time: Optional[float] = None
    request_success: Optional[bool] = None
    learning_feedback: Dict[str, Any] = field(default_factory=dict)


class AdvancedLoadBalancer:
    """
    ML-enhanced load balancer with predictive scaling and adaptive routing
    """
    
    def __init__(self,
                 enable_ml_routing: bool = True,
                 health_check_interval: int = 30,
                 learning_rate: float = 0.1):
        """Initialize advanced load balancer"""
        
        self.enable_ml_routing = enable_ml_routing
        self.health_check_interval = health_check_interval
        self.learning_rate = learning_rate
        
        # ML Models for Load Balancing Intelligence
        self.load_predictor: Optional[RandomForestRegressor] = None
        self.performance_classifier: Optional[GradientBoostingClassifier] = None
        self.capacity_optimizer: Optional[Ridge] = None
        self.backend_clusterer: Optional[KMeans] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.performance_scaler = MinMaxScaler()
        self.ml_feature_history: deque = deque(maxlen=1000)
        
        # Load Balancing State
        self.backend_nodes: Dict[str, BackendNode] = {}
        self.routing_history: deque = deque(maxlen=5000)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Request Distribution
        self.request_counter: int = 0
        self.session_affinity: Dict[str, str] = {}  # session_id -> backend_id
        self.sticky_sessions_enabled: bool = False
        
        # ML Predictions and Insights
        self.load_predictions: Dict[str, Dict[str, float]] = {}
        self.capacity_recommendations: Dict[str, float] = {}
        self.routing_insights: List[Dict[str, Any]] = []
        
        # Configuration
        self.default_strategy = RoutingStrategy.ML_PREDICTIVE
        self.failover_strategy = RoutingStrategy.LEAST_CONNECTIONS
        self.max_backend_failures = 3
        self.health_check_timeout = 5
        
        # Statistics
        self.lb_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'ml_decisions': 0,
            'ml_accuracy': 0.0,
            'average_response_time': 0.0,
            'backend_switches': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.routing_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start background tasks
        if enable_ml_routing:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_optimization_loop())
        
        asyncio.create_task(self._health_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for intelligent load balancing"""
        
        try:
            # Load prediction model
            self.load_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                min_samples_split=5
            )
            
            # Performance classification
            self.performance_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Capacity optimization
            self.capacity_optimizer = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # Backend clustering for similarity grouping
            self.backend_clusterer = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )
            
            self.logger.info("Load balancer ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Load balancer ML model initialization failed: {e}")
            self.enable_ml_routing = False
    
    def add_backend(self,
                   node_id: str,
                   host: str,
                   port: int,
                   weight: float = 1.0,
                   max_connections: int = 1000) -> bool:
        """Add backend node to load balancer"""
        
        try:
            with self.routing_lock:
                backend = BackendNode(
                    node_id=node_id,
                    host=host,
                    port=port,
                    weight=weight,
                    max_connections=max_connections
                )
                
                self.backend_nodes[node_id] = backend
                self.performance_metrics[node_id] = deque(maxlen=100)
            
            self.logger.info(f"Backend added: {node_id} ({host}:{port})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add backend {node_id}: {e}")
            return False
    
    def remove_backend(self, node_id: str) -> bool:
        """Remove backend node from load balancer"""
        
        try:
            with self.routing_lock:
                if node_id in self.backend_nodes:
                    # Drain existing connections first
                    self.backend_nodes[node_id].state = BackendState.DRAINING
                    
                    # Remove after brief delay to handle ongoing requests
                    asyncio.create_task(self._delayed_backend_removal(node_id))
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove backend {node_id}: {e}")
            return False
    
    async def route_request(self,
                           request_hash: Optional[str] = None,
                           session_id: Optional[str] = None,
                           client_ip: Optional[str] = None,
                           request_type: str = "http") -> Optional[BackendNode]:
        """Route request using ML-enhanced decision making"""
        
        try:
            with self.routing_lock:
                self.request_counter += 1
                self.lb_stats['total_requests'] += 1
                
                # Check session affinity first
                if session_id and self.sticky_sessions_enabled:
                    if session_id in self.session_affinity:
                        backend_id = self.session_affinity[session_id]
                        if (backend_id in self.backend_nodes and 
                            self.backend_nodes[backend_id].state == BackendState.HEALTHY):
                            return self.backend_nodes[backend_id]
                
                # Get available backends
                healthy_backends = [
                    backend for backend in self.backend_nodes.values()
                    if backend.state in [BackendState.HEALTHY, BackendState.DEGRADED]
                    and backend.current_connections < backend.max_connections
                ]
                
                if not healthy_backends:
                    self.logger.error("No healthy backends available")
                    return None
                
                # Select backend using ML or fallback strategy
                selected_backend = await self._select_backend(
                    healthy_backends, request_hash, client_ip, request_type
                )
                
                if selected_backend:
                    # Update connection count
                    selected_backend.current_connections += 1
                    
                    # Store session affinity if enabled
                    if session_id and self.sticky_sessions_enabled:
                        self.session_affinity[session_id] = selected_backend.node_id
                    
                    # Record decision
                    await self._record_routing_decision(selected_backend, request_hash)
                
                return selected_backend
                
        except Exception as e:
            self.logger.error(f"Request routing failed: {e}")
            return None
    
    async def _select_backend(self,
                             backends: List[BackendNode],
                             request_hash: Optional[str],
                             client_ip: Optional[str],
                             request_type: str) -> Optional[BackendNode]:
        """Select optimal backend using ML analysis"""
        
        try:
            # Use ML routing if enabled and sufficient data available
            if (self.enable_ml_routing and 
                len(self.ml_feature_history) >= 20):
                
                backend = await self._ml_backend_selection(backends, request_hash, request_type)
                if backend:
                    self.lb_stats['ml_decisions'] += 1
                    return backend
            
            # Fallback to traditional routing strategies
            return await self._traditional_backend_selection(backends, request_hash)
            
        except Exception as e:
            self.logger.error(f"Backend selection failed: {e}")
            return backends[0] if backends else None
    
    async def _ml_backend_selection(self,
                                   backends: List[BackendNode],
                                   request_hash: Optional[str],
                                   request_type: str) -> Optional[BackendNode]:
        """ML-driven backend selection"""
        
        try:
            with self.ml_lock:
                backend_scores = {}
                
                for backend in backends:
                    # Extract features for ML prediction
                    features = await self._extract_backend_features(backend, request_type)
                    
                    # Predict performance score
                    if self.load_predictor:
                        load_score = await self._predict_backend_load(features)
                        backend.predicted_load = load_score
                    else:
                        load_score = backend.current_connections / backend.max_connections
                    
                    # Calculate capacity score
                    capacity_score = await self._calculate_capacity_score(backend)
                    
                    # Calculate reliability score
                    reliability_score = await self._calculate_reliability_score(backend)
                    
                    # Combined ML score
                    ml_score = (
                        0.4 * (1.0 - load_score) +  # Lower load is better
                        0.3 * capacity_score +       # Higher capacity is better
                        0.3 * reliability_score      # Higher reliability is better
                    )
                    
                    backend_scores[backend.node_id] = ml_score
                    backend.ml_insights['selection_score'] = ml_score
                
                # Select backend with highest score
                best_backend_id = max(backend_scores, key=backend_scores.get)
                best_backend = next(b for b in backends if b.node_id == best_backend_id)
                
                # Add ML confidence
                score_variance = np.var(list(backend_scores.values()))
                confidence = 1.0 - min(score_variance, 0.5)  # Higher variance = lower confidence
                best_backend.ml_insights['selection_confidence'] = confidence
                
                return best_backend
                
        except Exception as e:
            self.logger.error(f"ML backend selection failed: {e}")
            return None
    
    async def _traditional_backend_selection(self,
                                           backends: List[BackendNode],
                                           request_hash: Optional[str]) -> BackendNode:
        """Traditional routing strategies as fallback"""
        
        strategy = self.failover_strategy
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            index = self.request_counter % len(backends)
            return backends[index]
        
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(backends, key=lambda b: b.current_connections)
        
        elif strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
            return min(backends, key=lambda b: b.response_time_ms)
        
        elif strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection based on backend weights
            total_weight = sum(b.weight for b in backends)
            random_weight = random.uniform(0, total_weight)
            
            current_weight = 0
            for backend in backends:
                current_weight += backend.weight
                if random_weight <= current_weight:
                    return backend
            
            return backends[0]
        
        else:
            # Default to round robin
            index = self.request_counter % len(backends)
            return backends[index]
    
    async def _extract_backend_features(self, backend: BackendNode, request_type: str) -> np.ndarray:
        """Extract ML features from backend state"""
        
        # Calculate recent performance metrics
        recent_metrics = list(self.performance_metrics[backend.node_id])[-10:]
        
        avg_response_time = np.mean([m.get('response_time', 0) for m in recent_metrics]) if recent_metrics else 0
        avg_cpu = np.mean([m.get('cpu_usage', 0) for m in recent_metrics]) if recent_metrics else 0
        avg_memory = np.mean([m.get('memory_usage', 0) for m in recent_metrics]) if recent_metrics else 0
        
        # Create feature vector
        features = np.array([
            backend.current_connections / backend.max_connections,
            backend.response_time_ms / 1000.0,  # Convert to seconds
            backend.success_rate,
            backend.cpu_usage / 100.0,
            backend.memory_usage / 100.0,
            backend.weight,
            avg_response_time / 1000.0,
            avg_cpu / 100.0,
            avg_memory / 100.0,
            backend.consecutive_failures,
            backend.total_requests / 1000.0,  # Scale down
            datetime.now().hour / 24.0,  # Time of day
            len(recent_metrics) / 10.0  # Data availability score
        ])
        
        return features.astype(np.float64)
    
    async def complete_request(self,
                              backend: BackendNode,
                              response_time_ms: float,
                              success: bool) -> None:
        """Complete request and update backend metrics"""
        
        try:
            with self.routing_lock:
                # Update backend metrics
                backend.current_connections = max(0, backend.current_connections - 1)
                backend.total_requests += 1
                
                # Update response time (exponential moving average)
                alpha = 0.1
                backend.response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * backend.response_time_ms
                )
                
                # Update success rate
                if success:
                    backend.success_rate = (
                        0.9 * backend.success_rate + 0.1 * 1.0
                    )
                    backend.consecutive_failures = 0
                    self.lb_stats['successful_requests'] += 1
                else:
                    backend.success_rate = (
                        0.9 * backend.success_rate + 0.1 * 0.0
                    )
                    backend.error_count += 1
                    backend.consecutive_failures += 1
                    self.lb_stats['failed_requests'] += 1
                
                # Store performance metrics
                metric_entry = {
                    'timestamp': datetime.now(),
                    'response_time': response_time_ms,
                    'success': success,
                    'cpu_usage': backend.cpu_usage,
                    'memory_usage': backend.memory_usage,
                    'connections': backend.current_connections
                }
                
                self.performance_metrics[backend.node_id].append(metric_entry)
                
                # Update ML features
                if self.enable_ml_routing:
                    features = await self._extract_backend_features(backend, "http")
                    self.ml_feature_history.append(features)
                
                # Check if backend should be marked as failed
                if backend.consecutive_failures >= self.max_backend_failures:
                    backend.state = BackendState.FAILED
                    self.logger.warning(f"Backend {backend.node_id} marked as failed")
            
        except Exception as e:
            self.logger.error(f"Request completion handling failed: {e}")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring for all backends"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for backend_id, backend in list(self.backend_nodes.items()):
                    await self._check_backend_health(backend)
                
                # Update ML models if sufficient data
                if self.enable_ml_routing and len(self.ml_feature_history) >= 50:
                    await self._update_ml_models()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _ml_optimization_loop(self):
        """ML optimization and learning loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if len(self.ml_feature_history) >= 100:
                    # Optimize backend weights
                    await self._optimize_backend_weights()
                    
                    # Update capacity recommendations
                    await self._update_capacity_recommendations()
                    
                    # Generate routing insights
                    await self._generate_routing_insights()
                
            except Exception as e:
                self.logger.error(f"ML optimization loop error: {e}")
                await asyncio.sleep(10)
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        
        # Backend summary
        backend_summary = {}
        for node_id, backend in self.backend_nodes.items():
            backend_summary[node_id] = {
                'state': backend.state.value,
                'current_connections': backend.current_connections,
                'response_time_ms': backend.response_time_ms,
                'success_rate': backend.success_rate,
                'cpu_usage': backend.cpu_usage,
                'memory_usage': backend.memory_usage,
                'predicted_load': backend.predicted_load,
                'capacity_score': backend.capacity_score,
                'reliability_score': backend.reliability_score
            }
        
        # ML insights
        ml_status = {
            'ml_routing_enabled': self.enable_ml_routing,
            'feature_history_size': len(self.ml_feature_history),
            'ml_decisions_made': self.lb_stats['ml_decisions'],
            'ml_accuracy': self.lb_stats['ml_accuracy'],
            'recent_insights': self.routing_insights[-5:] if self.routing_insights else []
        }
        
        return {
            'load_balancer_overview': {
                'total_backends': len(self.backend_nodes),
                'healthy_backends': len([b for b in self.backend_nodes.values() 
                                       if b.state == BackendState.HEALTHY]),
                'total_connections': sum(b.current_connections for b in self.backend_nodes.values()),
                'average_response_time': self.lb_stats['average_response_time']
            },
            'backends': backend_summary,
            'statistics': self.lb_stats.copy(),
            'ml_status': ml_status
        }
    
    async def shutdown(self):
        """Graceful shutdown of load balancer"""
        
        self.logger.info("Shutting down load balancer...")
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Load balancer shutdown complete")