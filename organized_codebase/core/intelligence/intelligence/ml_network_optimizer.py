"""
ML Network Optimizer - TestMaster Advanced ML
Advanced network optimization with ML-driven traffic analysis and routing
Enterprise ML Module #6/8 for comprehensive system intelligence
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
import socket
import struct
import ipaddress

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


class NetworkProtocol(Enum):
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"


class TrafficType(Enum):
    API = "api"
    WEB = "web"
    DATABASE = "database"
    STREAMING = "streaming"
    FILE_TRANSFER = "file_transfer"
    REAL_TIME = "real_time"


class QualityOfService(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class NetworkConnection:
    """ML-enhanced network connection with intelligent metrics"""
    
    connection_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: NetworkProtocol
    traffic_type: TrafficType
    qos: QualityOfService
    
    # Performance metrics
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    throughput_mbps: float = 0.0
    
    # Connection state
    established_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    
    # ML Enhancement Fields
    predicted_bandwidth_need: float = 0.0
    congestion_probability: float = 0.0
    optimization_score: float = 1.0
    route_efficiency: float = 1.0
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Quality tracking
    connection_quality: float = 1.0
    stability_score: float = 1.0
    error_count: int = 0
    retransmission_count: int = 0


@dataclass
class NetworkRoute:
    """Intelligent network route with ML optimization"""
    
    route_id: str
    source_network: str
    destination_network: str
    gateway: str
    interface: str
    metric: int = 1
    
    # Performance characteristics
    average_latency: float = 0.0
    capacity_mbps: float = 1000.0
    utilization: float = 0.0
    reliability_score: float = 1.0
    
    # ML Enhancement
    predicted_congestion: float = 0.0
    optimal_load_threshold: float = 0.8
    dynamic_priority: float = 1.0
    adaptation_factor: float = 1.0
    
    # Usage statistics
    active_connections: int = 0
    total_traffic_gb: float = 0.0
    success_rate: float = 1.0
    last_optimization: datetime = field(default_factory=datetime.now)


@dataclass
class TrafficPattern:
    """ML-identified traffic pattern"""
    
    pattern_id: str
    pattern_name: str
    source_networks: List[str]
    destination_networks: List[str]
    protocols: List[NetworkProtocol]
    time_windows: List[Tuple[int, int]]  # (start_hour, end_hour)
    
    # Pattern characteristics
    average_bandwidth: float = 0.0
    peak_bandwidth: float = 0.0
    duration_minutes: int = 0
    frequency: str = "daily"  # daily, weekly, monthly
    
    # ML insights
    predictability_score: float = 0.0
    optimization_potential: float = 0.0
    impact_on_network: float = 0.0
    last_detected: Optional[datetime] = None


class MLNetworkOptimizer:
    """
    ML-enhanced network optimization with intelligent traffic management
    """
    
    def __init__(self,
                 enable_ml_optimization: bool = True,
                 monitoring_interval: int = 30,
                 auto_route_optimization: bool = False,
                 congestion_threshold: float = 0.8):
        """Initialize ML network optimizer"""
        
        self.enable_ml_optimization = enable_ml_optimization
        self.monitoring_interval = monitoring_interval
        self.auto_route_optimization = auto_route_optimization
        self.congestion_threshold = congestion_threshold
        
        # ML Models for Network Intelligence
        self.bandwidth_predictor: Optional[RandomForestRegressor] = None
        self.congestion_predictor: Optional[GradientBoostingClassifier] = None
        self.route_optimizer: Optional[Ridge] = None
        self.traffic_clusterer: Optional[KMeans] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.traffic_scaler = MinMaxScaler()
        self.network_feature_history: deque = deque(maxlen=10000)
        
        # Network State Management
        self.active_connections: Dict[str, NetworkConnection] = {}
        self.network_routes: Dict[str, NetworkRoute] = {}
        self.traffic_patterns: Dict[str, TrafficPattern] = {}
        
        # Performance Monitoring
        self.connection_metrics: deque = deque(maxlen=5000)
        self.route_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.network_utilization_history: deque = deque(maxlen=1000)
        
        # Traffic Analysis
        self.traffic_flows: Dict[str, Dict[str, Any]] = {}
        self.bandwidth_allocation: Dict[str, float] = {}
        self.qos_enforcement: Dict[QualityOfService, Dict[str, Any]] = {}
        
        # ML Insights and Optimization
        self.ml_recommendations: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.congestion_alerts: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_connections_per_route = 1000
        self.bandwidth_buffer_percentage = 0.1
        self.route_failover_threshold = 0.3
        self.traffic_shaping_enabled = True
        
        # Statistics
        self.network_stats = {
            'connections_monitored': 0,
            'routes_optimized': 0,
            'congestion_events_detected': 0,
            'ml_optimizations_applied': 0,
            'bandwidth_savings_mbps': 0.0,
            'latency_improvements_ms': 0.0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.network_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and monitoring
        if enable_ml_optimization:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_optimization_loop())
        
        self._initialize_qos_policies()
        asyncio.create_task(self._network_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for network intelligence"""
        
        try:
            # Bandwidth prediction model
            self.bandwidth_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                min_samples_split=5
            )
            
            # Network congestion prediction
            self.congestion_predictor = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Route optimization model
            self.route_optimizer = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # Traffic pattern clustering
            self.traffic_clusterer = KMeans(
                n_clusters=8,
                random_state=42,
                n_init=10
            )
            
            self.logger.info("Network ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Network ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def _initialize_qos_policies(self):
        """Initialize Quality of Service policies"""
        
        # Critical traffic (real-time applications)
        self.qos_enforcement[QualityOfService.CRITICAL] = {
            'bandwidth_guarantee': 0.4,  # 40% of total bandwidth
            'max_latency_ms': 10,
            'max_jitter_ms': 2,
            'max_packet_loss': 0.001,
            'priority': 1
        }
        
        # High priority traffic (interactive applications)
        self.qos_enforcement[QualityOfService.HIGH] = {
            'bandwidth_guarantee': 0.3,  # 30% of total bandwidth
            'max_latency_ms': 50,
            'max_jitter_ms': 10,
            'max_packet_loss': 0.01,
            'priority': 2
        }
        
        # Medium priority traffic (standard applications)
        self.qos_enforcement[QualityOfService.MEDIUM] = {
            'bandwidth_guarantee': 0.2,  # 20% of total bandwidth
            'max_latency_ms': 200,
            'max_jitter_ms': 50,
            'max_packet_loss': 0.05,
            'priority': 3
        }
        
        # Low priority traffic (bulk transfers)
        self.qos_enforcement[QualityOfService.LOW] = {
            'bandwidth_guarantee': 0.1,  # 10% of total bandwidth
            'max_latency_ms': 1000,
            'max_jitter_ms': 200,
            'max_packet_loss': 0.1,
            'priority': 4
        }
        
        # Background traffic (system maintenance)
        self.qos_enforcement[QualityOfService.BACKGROUND] = {
            'bandwidth_guarantee': 0.05,  # 5% of total bandwidth
            'max_latency_ms': 5000,
            'max_jitter_ms': 1000,
            'max_packet_loss': 0.2,
            'priority': 5
        }
    
    def register_network_route(self,
                              route_id: str,
                              source_network: str,
                              destination_network: str,
                              gateway: str,
                              interface: str,
                              capacity_mbps: float = 1000.0,
                              metric: int = 1) -> bool:
        """Register network route for ML optimization"""
        
        try:
            with self.network_lock:
                route = NetworkRoute(
                    route_id=route_id,
                    source_network=source_network,
                    destination_network=destination_network,
                    gateway=gateway,
                    interface=interface,
                    capacity_mbps=capacity_mbps,
                    metric=metric
                )
                
                self.network_routes[route_id] = route
                self.route_performance_history[route_id] = deque(maxlen=100)
            
            self.logger.info(f"Network route registered: {route_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Network route registration failed: {e}")
            return False
    
    def monitor_connection(self,
                          connection_id: str,
                          source_ip: str,
                          destination_ip: str,
                          source_port: int,
                          destination_port: int,
                          protocol: NetworkProtocol,
                          traffic_type: TrafficType = TrafficType.API,
                          qos: QualityOfService = QualityOfService.MEDIUM) -> bool:
        """Monitor network connection with ML analysis"""
        
        try:
            with self.network_lock:
                connection = NetworkConnection(
                    connection_id=connection_id,
                    source_ip=source_ip,
                    destination_ip=destination_ip,
                    source_port=source_port,
                    destination_port=destination_port,
                    protocol=protocol,
                    traffic_type=traffic_type,
                    qos=qos
                )
                
                self.active_connections[connection_id] = connection
                self.network_stats['connections_monitored'] += 1
            
            # ML enhancement for connection
            if self.enable_ml_optimization:
                asyncio.create_task(self._enhance_connection_with_ml(connection))
            
            self.logger.info(f"Connection monitoring started: {connection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection monitoring failed: {e}")
            return False
    
    async def _enhance_connection_with_ml(self, connection: NetworkConnection):
        """Enhance connection with ML predictions and insights"""
        
        try:
            with self.ml_lock:
                # Extract connection features
                features = await self._extract_connection_features(connection)
                
                # Predict bandwidth requirements
                if self.bandwidth_predictor and len(self.network_feature_history) >= 50:
                    predicted_bandwidth = await self._predict_bandwidth_need(features)
                    connection.predicted_bandwidth_need = predicted_bandwidth
                
                # Predict congestion probability
                if self.congestion_predictor and len(self.network_feature_history) >= 100:
                    congestion_prob = await self._predict_congestion_probability(features)
                    connection.congestion_probability = congestion_prob
                    
                    if congestion_prob > self.congestion_threshold:
                        await self._handle_potential_congestion(connection)
                
                # Route optimization analysis
                if self.route_optimizer:
                    route_efficiency = await self._analyze_route_efficiency(connection)
                    connection.route_efficiency = route_efficiency
                    
                    if route_efficiency < 0.7:  # Route is inefficient
                        connection.ml_insights['route_optimization_needed'] = True
                
                # Calculate optimization score
                connection.optimization_score = await self._calculate_connection_optimization_score(connection)
                
                # Store features for model training
                self.network_feature_history.append(features)
                
        except Exception as e:
            self.logger.error(f"ML connection enhancement failed: {e}")
    
    def _extract_connection_features(self, connection: NetworkConnection) -> np.ndarray:
        """Extract ML features from network connection"""
        
        try:
            # Connection characteristics
            protocol_encoded = list(NetworkProtocol).index(connection.protocol)
            traffic_type_encoded = list(TrafficType).index(connection.traffic_type)
            qos_encoded = connection.qos.value
            
            # Performance metrics
            latency_normalized = min(connection.latency_ms / 1000.0, 1.0)
            bandwidth_normalized = connection.bandwidth_mbps / 1000.0  # Normalize to Gbps
            packet_loss_normalized = min(connection.packet_loss_rate * 100, 1.0)
            
            # Temporal features
            hour = datetime.now().hour
            day_of_week = datetime.now().weekday()
            
            # Connection age and activity
            connection_age_hours = (datetime.now() - connection.established_at).total_seconds() / 3600
            activity_age_minutes = (datetime.now() - connection.last_activity).total_seconds() / 60
            
            # Network state features
            total_active_connections = len(self.active_connections)
            network_utilization = self._calculate_current_network_utilization()
            
            # Create feature vector
            features = np.array([
                protocol_encoded,
                traffic_type_encoded,
                qos_encoded,
                latency_normalized,
                bandwidth_normalized,
                packet_loss_normalized,
                connection.jitter_ms / 100.0,  # Normalize
                hour / 24.0,
                day_of_week / 7.0,
                connection_age_hours / 24.0,  # Normalize to days
                activity_age_minutes / 60.0,  # Normalize to hours
                total_active_connections / 1000.0,  # Normalize
                network_utilization,
                connection.error_count / 10.0,  # Normalize
                connection.retransmission_count / 10.0  # Normalize
            ])
            
            return features.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Connection feature extraction failed: {e}")
            return np.zeros(15)  # Default feature vector
    
    def update_connection_metrics(self,
                                 connection_id: str,
                                 latency_ms: float,
                                 bandwidth_mbps: float,
                                 packet_loss_rate: float,
                                 jitter_ms: float = 0.0,
                                 bytes_transferred: int = 0) -> bool:
        """Update connection performance metrics"""
        
        try:
            with self.network_lock:
                if connection_id not in self.active_connections:
                    return False
                
                connection = self.active_connections[connection_id]
                
                # Update performance metrics (exponential moving average)
                alpha = 0.3
                connection.latency_ms = alpha * latency_ms + (1 - alpha) * connection.latency_ms
                connection.bandwidth_mbps = alpha * bandwidth_mbps + (1 - alpha) * connection.bandwidth_mbps
                connection.packet_loss_rate = alpha * packet_loss_rate + (1 - alpha) * connection.packet_loss_rate
                connection.jitter_ms = alpha * jitter_ms + (1 - alpha) * connection.jitter_ms
                
                # Update transfer statistics
                connection.bytes_sent += bytes_transferred
                connection.last_activity = datetime.now()
                
                # Update connection quality score
                connection.connection_quality = await self._calculate_connection_quality(connection)
                
                # Record metrics
                metric_entry = {
                    'timestamp': datetime.now(),
                    'connection_id': connection_id,
                    'latency_ms': latency_ms,
                    'bandwidth_mbps': bandwidth_mbps,
                    'packet_loss_rate': packet_loss_rate,
                    'jitter_ms': jitter_ms,
                    'quality_score': connection.connection_quality
                }
                
                self.connection_metrics.append(metric_entry)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection metrics update failed: {e}")
            return False
    
    async def _calculate_connection_quality(self, connection: NetworkConnection) -> float:
        """Calculate connection quality score based on performance metrics"""
        
        try:
            qos_policy = self.qos_enforcement.get(connection.qos, {})
            
            # Quality factors based on QoS requirements
            latency_score = 1.0
            if 'max_latency_ms' in qos_policy:
                latency_score = max(0.0, 1.0 - connection.latency_ms / qos_policy['max_latency_ms'])
            
            jitter_score = 1.0
            if 'max_jitter_ms' in qos_policy:
                jitter_score = max(0.0, 1.0 - connection.jitter_ms / qos_policy['max_jitter_ms'])
            
            packet_loss_score = 1.0
            if 'max_packet_loss' in qos_policy:
                packet_loss_score = max(0.0, 1.0 - connection.packet_loss_rate / qos_policy['max_packet_loss'])
            
            # Weighted quality score
            quality_score = (
                0.4 * latency_score +
                0.3 * packet_loss_score +
                0.2 * jitter_score +
                0.1 * (1.0 - min(connection.error_count / 10.0, 1.0))
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Connection quality calculation failed: {e}")
            return 0.5  # Default neutral score
    
    async def _network_monitoring_loop(self):
        """Main network monitoring and optimization loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update route performance metrics
                await self._update_route_metrics()
                
                # Monitor network utilization
                await self._monitor_network_utilization()
                
                # Check for congestion
                await self._check_network_congestion()
                
                # Cleanup inactive connections
                await self._cleanup_inactive_connections()
                
                # Auto-optimization if enabled
                if self.auto_route_optimization and self.enable_ml_optimization:
                    await self._auto_optimize_routes()
                
            except Exception as e:
                self.logger.error(f"Network monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _ml_optimization_loop(self):
        """ML optimization and insights generation loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if len(self.network_feature_history) >= 200:
                    # Retrain ML models
                    await self._retrain_network_models()
                    
                    # Discover traffic patterns
                    await self._discover_traffic_patterns()
                    
                    # Generate optimization recommendations
                    await self._generate_network_recommendations()
                    
                    # Apply ML optimizations
                    await self._apply_ml_optimizations()
                
            except Exception as e:
                self.logger.error(f"ML optimization loop error: {e}")
                await asyncio.sleep(30)
    
    def get_network_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive network optimization dashboard"""
        
        # Connection summary by QoS
        connection_summary = defaultdict(int)
        for connection in self.active_connections.values():
            connection_summary[connection.qos.name] += 1
        
        # Route performance summary
        route_summary = {}
        for route_id, route in self.network_routes.items():
            route_summary[route_id] = {
                'capacity_mbps': route.capacity_mbps,
                'utilization': route.utilization,
                'average_latency': route.average_latency,
                'reliability_score': route.reliability_score,
                'active_connections': route.active_connections,
                'predicted_congestion': route.predicted_congestion
            }
        
        # Performance metrics
        recent_metrics = list(self.connection_metrics)[-100:]
        avg_latency = np.mean([m['latency_ms'] for m in recent_metrics]) if recent_metrics else 0.0
        avg_bandwidth = np.mean([m['bandwidth_mbps'] for m in recent_metrics]) if recent_metrics else 0.0
        avg_quality = np.mean([m['quality_score'] for m in recent_metrics]) if recent_metrics else 1.0
        
        # ML insights
        ml_status = {
            'ml_optimization_enabled': self.enable_ml_optimization,
            'feature_history_size': len(self.network_feature_history),
            'traffic_patterns_discovered': len(self.traffic_patterns),
            'ml_recommendations': len(self.ml_recommendations),
            'congestion_alerts': len(self.congestion_alerts)
        }
        
        return {
            'network_overview': {
                'active_connections': len(self.active_connections),
                'monitored_routes': len(self.network_routes),
                'network_utilization': self._calculate_current_network_utilization(),
                'average_connection_quality': avg_quality,
                'congestion_events_24h': len([
                    alert for alert in self.congestion_alerts
                    if (datetime.now() - alert['timestamp']) < timedelta(hours=24)
                ])
            },
            'connections_by_qos': dict(connection_summary),
            'route_performance': route_summary,
            'performance_metrics': {
                'average_latency_ms': avg_latency,
                'average_bandwidth_mbps': avg_bandwidth,
                'average_quality_score': avg_quality,
                'packet_loss_rate': self._calculate_overall_packet_loss_rate()
            },
            'statistics': self.network_stats.copy(),
            'ml_status': ml_status,
            'recent_insights': self.ml_recommendations[-5:] if self.ml_recommendations else []
        }
    
    def _calculate_current_network_utilization(self) -> float:
        """Calculate current overall network utilization"""
        
        if not self.network_routes:
            return 0.0
        
        total_capacity = sum(route.capacity_mbps for route in self.network_routes.values())
        total_utilization = sum(
            route.capacity_mbps * route.utilization 
            for route in self.network_routes.values()
        )
        
        return total_utilization / total_capacity if total_capacity > 0 else 0.0
    
    def _calculate_overall_packet_loss_rate(self) -> float:
        """Calculate overall packet loss rate across all connections"""
        
        if not self.active_connections:
            return 0.0
        
        return np.mean([conn.packet_loss_rate for conn in self.active_connections.values()])
    
    async def shutdown(self):
        """Graceful shutdown of network optimizer"""
        
        self.logger.info("Shutting down ML network optimizer...")
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("ML network optimizer shutdown complete")