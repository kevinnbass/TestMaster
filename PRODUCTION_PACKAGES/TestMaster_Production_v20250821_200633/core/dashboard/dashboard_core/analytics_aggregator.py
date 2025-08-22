"""
Enhanced Analytics Aggregator
==============================

Aggregates analytics from all TestMaster intelligence systems for the dashboard.
Provides comprehensive metrics collection and real-time insights.

Author: TestMaster Team
"""

import logging
import time
import psutil
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import json

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class AnalyticsAggregator:
    """
    Central analytics aggregator that collects metrics from all TestMaster components.
    """
    
    def __init__(self, cache_ttl: int = 60):
        """
        Initialize the analytics aggregator.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamps = {}
        
        # Metrics storage
        self.test_metrics = defaultdict(dict)
        self.code_quality_metrics = defaultdict(float)
        self.performance_trends = deque(maxlen=1000)
        self.security_scan_results = {}
        self.workflow_metrics = defaultdict(list)
        self.agent_activity = defaultdict(int)
        self.bridge_metrics = defaultdict(dict)
        
        # Initialize component connections
        self._init_components()
        
        # Initialize all enhancement components
        try:
            from .test_collector import TestResultsCollector
            from .data_store import MetricsDataStore
            from .analytics_validator import AnalyticsValidator
            from .analytics_correlator import AnalyticsCorrelator
            from .analytics_performance_monitor import AnalyticsPerformanceMonitor
            from .analytics_performance_booster import AnalyticsPerformanceBooster
            from .analytics_streaming import AnalyticsStreamManager, AnalyticsStreamIntegrator
            from .analytics_event_queue import AnalyticsEventQueue, EventPriority
            from .analytics_anomaly_detector import AnalyticsAnomalyDetector
            from .analytics_export_manager import AnalyticsExportManager, ExportFormat
            from .analytics_persistence import AnalyticsPersistenceEngine
            from .analytics_pipeline import AnalyticsPipeline
            from .analytics_health_monitor import AnalyticsHealthMonitor, ComponentType
            from .analytics_smart_cache import SmartAnalyticsCache
            from .analytics_normalizer import AnalyticsDataNormalizer
            from .analytics_quality_assurance import AnalyticsQualityAssurance
            from .analytics_circuit_breaker import AnalyticsCircuitBreakerManager
            from .analytics_metrics_collector import AnalyticsMetricsCollector
            from .analytics_redundancy import AnalyticsRedundancyManager
            from .analytics_watchdog import AnalyticsWatchdog
            from .analytics_telemetry import AnalyticsTelemetryCollector, TelemetryLevel, traced_operation
            from .analytics_performance_optimizer import AnalyticsPerformanceOptimizer, OptimizationLevel
            # Additional robustness enhancement components - import separately for better error handling
            try:
                from .analytics_data_sanitizer import AnalyticsDataSanitizer, ValidationLevel
                data_sanitizer_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsDataSanitizer: {e}")
                data_sanitizer_available = False
            
            try:
                from .analytics_deduplication_engine import AnalyticsDeduplicationEngine, ConflictResolutionStrategy
                deduplication_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsDeduplicationEngine: {e}")
                deduplication_available = False
            
            try:
                from .analytics_rate_limiter import AnalyticsRateLimiter
                rate_limiter_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsRateLimiter: {e}")
                rate_limiter_available = False
            
            try:
                from .analytics_integrity_verifier import AnalyticsIntegrityVerifier, IntegrityLevel
                integrity_verifier_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsIntegrityVerifier: {e}")
                integrity_verifier_available = False
            
            try:
                from .analytics_error_recovery import AnalyticsErrorRecovery, ErrorSeverity
                error_recovery_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsErrorRecovery: {e}")
                error_recovery_available = False
            
            try:
                from .analytics_connectivity_monitor import AnalyticsConnectivityMonitor, DashboardEndpoint, MonitoringLevel
                connectivity_monitor_available = True
            except ImportError as e:
                logger.warning(f"Could not import AnalyticsConnectivityMonitor: {e}")
                connectivity_monitor_available = False
            
            # Initialize core components
            self.test_collector = TestResultsCollector()
            self.data_store = MetricsDataStore()
            self.validator = AnalyticsValidator()
            self.correlator = AnalyticsCorrelator()
            self.performance_monitor = AnalyticsPerformanceMonitor()
            
            # Initialize performance booster
            self.performance_booster = AnalyticsPerformanceBooster()
            self.performance_booster.start_optimization()
            
            # Initialize new robustness components
            self.event_queue = AnalyticsEventQueue()
            self.event_queue.start_processing()
            
            self.anomaly_detector = AnalyticsAnomalyDetector()
            
            self.export_manager = AnalyticsExportManager()
            
            # Initialize new robustness components
            from .analytics_retry_manager import AnalyticsRetryManager
            from .analytics_flow_monitor import AnalyticsFlowMonitor
            from .analytics_heartbeat_monitor import DashboardHeartbeatMonitor
            from .analytics_fallback_system import AnalyticsFallbackSystem
            from .analytics_compressor import AnalyticsCompressor
            from .analytics_dead_letter_queue import AnalyticsDeadLetterQueue
            from .analytics_batch_processor import AnalyticsBatchProcessor, BatchPriority
            
            self.retry_manager = AnalyticsRetryManager()
            self.flow_monitor = AnalyticsFlowMonitor()
            self.heartbeat_monitor = DashboardHeartbeatMonitor()
            self.fallback_system = AnalyticsFallbackSystem()
            self.compressor = AnalyticsCompressor()
            self.dead_letter_queue = AnalyticsDeadLetterQueue()
            self.batch_processor = AnalyticsBatchProcessor(
                batch_size=50,
                flush_interval=3.0,
                processor_func=self._process_analytics_batch
            )
            
            # Register dashboard endpoint for heartbeat monitoring
            self.heartbeat_monitor.register_endpoint(
                'main_dashboard',
                'http://localhost:5000/api/health/live',
                'http',
                critical=True
            )
            
            # Initialize new enhancement components
            self.stream_manager = AnalyticsStreamManager(self, stream_interval=2.0)
            self.persistence_engine = AnalyticsPersistenceEngine()
            self.pipeline = AnalyticsPipeline(max_workers=4)
            self.health_monitor = AnalyticsHealthMonitor(check_interval=60)
            self.smart_cache = SmartAnalyticsCache(max_memory_size=200*1024*1024)  # 200MB
            self.normalizer = AnalyticsDataNormalizer()
            self.quality_assurance = AnalyticsQualityAssurance(check_interval=30)
            self.circuit_breaker_manager = AnalyticsCircuitBreakerManager()
            self.metrics_collector = AnalyticsMetricsCollector(collection_interval=10.0)
            self.redundancy_manager = AnalyticsRedundancyManager()
            self.watchdog = AnalyticsWatchdog(check_interval=30, max_restart_attempts=3)
            self.telemetry_collector = AnalyticsTelemetryCollector(
                service_name="analytics_aggregator", 
                export_interval=60,
                max_events=5000
            )
            self.performance_optimizer = AnalyticsPerformanceOptimizer(
                optimization_level=OptimizationLevel.MODERATE,
                monitoring_interval=90
            )
            
            # Initialize additional robustness enhancement components based on availability
            if data_sanitizer_available:
                try:
                    self.data_sanitizer = AnalyticsDataSanitizer(
                        validation_level=ValidationLevel.STANDARD,
                        max_issues_per_batch=1000
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsDataSanitizer: {e}")
                    self.data_sanitizer = None
            else:
                self.data_sanitizer = None
            
            if deduplication_available:
                try:
                    self.deduplication_engine = AnalyticsDeduplicationEngine(
                        max_history_size=50000,
                        cleanup_interval=3600,
                        default_strategy=ConflictResolutionStrategy.LATEST_WINS
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsDeduplicationEngine: {e}")
                    self.deduplication_engine = None
            else:
                self.deduplication_engine = None
            
            if rate_limiter_available:
                try:
                    self.rate_limiter = AnalyticsRateLimiter(
                        max_queue_size=10000,
                        monitoring_interval=1.0,
                        adaptive_adjustment_factor=0.1
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsRateLimiter: {e}")
                    self.rate_limiter = None
            else:
                self.rate_limiter = None
            
            if integrity_verifier_available:
                try:
                    self.integrity_verifier = AnalyticsIntegrityVerifier(
                        integrity_level=IntegrityLevel.STANDARD,
                        secret_key = os.getenv('KEY')
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsIntegrityVerifier: {e}")
                    self.integrity_verifier = None
            else:
                self.integrity_verifier = None
            
            if error_recovery_available:
                try:
                    self.error_recovery = AnalyticsErrorRecovery(
                        max_error_history=10000,
                        recovery_timeout=30.0,
                        health_check_interval=60.0
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsErrorRecovery: {e}")
                    self.error_recovery = None
            else:
                self.error_recovery = None
            
            if connectivity_monitor_available:
                try:
                    self.connectivity_monitor = AnalyticsConnectivityMonitor(
                        monitoring_level=MonitoringLevel.STANDARD,
                        check_interval=10.0,
                        data_flow_timeout=60.0
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize AnalyticsConnectivityMonitor: {e}")
                    self.connectivity_monitor = None
            else:
                self.connectivity_monitor = None
            
            # Setup integrations
            self.stream_integrator = AnalyticsStreamIntegrator(self, self.stream_manager)
            
            # Register components with health monitor
            self.health_monitor.register_component(ComponentType.AGGREGATOR, self)
            self.health_monitor.register_component(ComponentType.VALIDATOR, self.validator)
            self.health_monitor.register_component(ComponentType.CORRELATOR, self.correlator)
            self.health_monitor.register_component(ComponentType.PERFORMANCE_MONITOR, self.performance_monitor)
            self.health_monitor.register_component(ComponentType.STREAMING, self.stream_manager)
            self.health_monitor.register_component(ComponentType.PERSISTENCE, self.persistence_engine)
            self.health_monitor.register_component(ComponentType.PIPELINE, self.pipeline)
            
            # Register components with watchdog for monitoring
            self.watchdog.register_component('aggregator', self, is_critical=True)
            self.watchdog.register_component('data_store', self.data_store, is_critical=True)
            self.watchdog.register_component('validator', self.validator)
            self.watchdog.register_component('correlator', self.correlator)
            self.watchdog.register_component('performance_monitor', self.performance_monitor)
            self.watchdog.register_component('stream_manager', self.stream_manager, is_critical=True)
            self.watchdog.register_component('persistence_engine', self.persistence_engine, is_critical=True)
            self.watchdog.register_component('smart_cache', self.smart_cache)
            # Register additional robustness components (only if available)
            if self.data_sanitizer:
                self.watchdog.register_component('data_sanitizer', self.data_sanitizer)
            if self.deduplication_engine:
                self.watchdog.register_component('deduplication_engine', self.deduplication_engine)
            if self.rate_limiter:
                self.watchdog.register_component('rate_limiter', self.rate_limiter)
            if self.integrity_verifier:
                self.watchdog.register_component('integrity_verifier', self.integrity_verifier, is_critical=True)
            if self.error_recovery:
                self.watchdog.register_component('error_recovery', self.error_recovery, is_critical=True)
            if self.connectivity_monitor:
                self.watchdog.register_component('connectivity_monitor', self.connectivity_monitor)
            
            # Register redundancy nodes
            # TODO: Add redundancy node registration when implemented
            
        except Exception as e:
            logger.error(f"Failed to initialize all analytics components: {e}")
            # Continue with basic functionality even if some components fail
    
    def aggregate_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple metrics into summary."""
        if not metrics:
            return {}
        
        aggregated = {
            'count': len(metrics),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Calculate averages for numeric fields
        numeric_fields = {}
        for metric in metrics:
            for key, value in metric.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        for field, values in numeric_fields.items():
            aggregated[f'avg_{field}'] = sum(values) / len(values)
            aggregated[f'max_{field}'] = max(values)
            aggregated[f'min_{field}'] = min(values)
        
        return aggregated
    
    def _misplaced_init_code(self):
        """This code was misplaced in aggregate_metrics - moved here temporarily."""
        # TODO: Move this code to proper location in __init__ or remove if duplicate
        if hasattr(self, 'performance_optimizer'):
            self.performance_optimizer.register_analytics_component('cache_smart', self.smart_cache)
            self.performance_optimizer.register_analytics_component('data_store', self.data_store)
            self.performance_optimizer.register_analytics_component('pipeline', self.pipeline)
            
            # Start services
            self.performance_monitor.start_monitoring()
            self.stream_manager.start_streaming()
            self.health_monitor.start_monitoring()
            self.quality_assurance.start_monitoring()
            self.circuit_breaker_manager.start_monitoring()
            self.metrics_collector.start_collection()
            self.redundancy_manager.start_redundancy()
            self.watchdog.start_monitoring()
            self.telemetry_collector.start_collection()
            self.performance_optimizer.start_optimization()
            
            # Start additional robustness enhancement services (only if available)
            if self.data_sanitizer:
                try:
                    self.data_sanitizer.start_background_processing()
                except Exception as e:
                    logger.error(f"Failed to start data sanitizer: {e}")
            
            if self.deduplication_engine:
                try:
                    self.deduplication_engine.start_engine()
                except Exception as e:
                    logger.error(f"Failed to start deduplication engine: {e}")
            
            if self.rate_limiter:
                try:
                    self.rate_limiter.start_rate_limiting()
                except Exception as e:
                    logger.error(f"Failed to start rate limiter: {e}")
            
            if self.integrity_verifier:
                try:
                    self.integrity_verifier.start_verification()
                except Exception as e:
                    logger.error(f"Failed to start integrity verifier: {e}")
            
            if self.error_recovery:
                try:
                    self.error_recovery.start_error_recovery()
                except Exception as e:
                    logger.error(f"Failed to start error recovery: {e}")
            
            if self.connectivity_monitor:
                try:
                    self.connectivity_monitor.start_monitoring()
                    
                    # Register dashboard endpoints for connectivity monitoring
                    dashboard_endpoint = DashboardEndpoint(
                        endpoint_id="main_dashboard",
                        url="http://localhost:5000",
                        endpoint_type="http",
                        expected_update_interval=5.0,
                        timeout_seconds=10.0,
                        retry_attempts=3,
                        health_check_path="/api/health/live"
                    )
                    self.connectivity_monitor.register_dashboard_endpoint(dashboard_endpoint)
                except Exception as e:
                    logger.error(f"Failed to start connectivity monitor: {e}")
            
            logger.info("All analytics enhancement components initialized successfully")
    

    def _fallback_init(self):

        """Fallback initialization for missing components."""
        pass
    
    def _init_components(self):
        """Initialize connections to TestMaster components."""
        self.components_available = {
            'hierarchical_analyzer': False,
            'security_intelligence': False,
            'workflow_monitor': False,
            'quality_monitor': False,
            'metrics_analyzer': False
        }
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'operational'
        }
    
    def _process_analytics_batch(self, batch):
        """Process a batch of analytics data."""
        return batch
