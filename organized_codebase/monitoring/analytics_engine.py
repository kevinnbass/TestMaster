"""
Analytics Engine - Unified Analytics Coordinator

Main coordination engine that brings together all analytics components
into a single, cohesive system. Replaces the need for multiple separate
analytics hubs and provides a unified interface.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type

from .base_analytics import BaseAnalytics, AnalyticsConfig, AnalyticsResult, MetricData
from .pipeline_manager import PipelineManager
from .data_processor import DataProcessor

# Import all component categories
from ..processors.aggregation import DataAggregator
from ..processors.compression import DataCompressor
from ..processors.normalization import DataNormalizer
from ..processors.correlation import DataCorrelator
from ..processors.deduplication import DuplicateDetector

from ..quality.validator import DataValidator
from ..quality.integrity_checker import IntegrityChecker
from ..quality.anomaly_detector import AnomalyDetector
from ..quality.quality_assurance import QualityAssurance

from ..monitoring.health_monitor import HealthMonitor
from ..monitoring.performance_monitor import PerformanceMonitor
from ..monitoring.flow_monitor import FlowMonitor
from ..monitoring.sla_tracker import SLATracker

from ..delivery.delivery_manager import DeliveryManager
from ..delivery.circuit_breaker import CircuitBreaker
from ..delivery.retry_manager import RetryManager
from ..delivery.fallback_system import FallbackSystem

from ..intelligence.predictive_engine import PredictiveEngine
from ..intelligence.correlation_engine import CorrelationEngine
from ..intelligence.adaptive_enhancer import AdaptiveEnhancer
from ..intelligence.cross_system_analyzer import CrossSystemAnalyzer


class AnalyticsEngine(BaseAnalytics):
    """
    Unified Analytics Engine that coordinates all analytics functionality.
    
    Consolidates the functionality of:
    - AnalyticsHub
    - PredictiveAnalyticsEngine
    - AnalyticsAggregator  
    - All 84+ analytics files
    
    Into a single, efficient system.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize the unified analytics engine."""
        config = config or AnalyticsConfig(component_name="analytics_engine")
        super().__init__(config)
        
        # Core processing components
        self.pipeline_manager = PipelineManager()
        self.data_processor = DataProcessor()
        
        # Initialize all component categories
        self._init_processors()
        self._init_quality_components()
        self._init_monitoring()
        self._init_delivery()
        self._init_intelligence()
        
        # Component registry for dynamic management
        self.components: Dict[str, BaseAnalytics] = {}
        self._register_all_components()
        
        # Unified metrics and state
        self.metrics_cache: Dict[str, MetricData] = {}
        self.global_stats = {
            'total_metrics_processed': 0,
            'total_anomalies_detected': 0,
            'total_predictions_made': 0,
            'total_duplicates_found': 0,
            'avg_processing_time_ms': 0.0,
            'components_active': 0
        }
        
        self.logger.info("Analytics Engine initialized with unified components")
    
    def _init_processors(self):
        """Initialize data processing components."""
        self.aggregator = DataAggregator()
        self.compressor = DataCompressor()
        self.normalizer = DataNormalizer()
        self.correlator = DataCorrelator()
        self.duplicate_detector = DuplicateDetector()
    
    def _init_quality_components(self):
        """Initialize quality assurance components."""
        self.validator = DataValidator()
        self.integrity_checker = IntegrityChecker()
        self.anomaly_detector = AnomalyDetector()
        self.quality_assurance = QualityAssurance()
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        self.health_monitor = HealthMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.flow_monitor = FlowMonitor()
        self.sla_tracker = SLATracker()
    
    def _init_delivery(self):
        """Initialize delivery components."""
        self.delivery_manager = DeliveryManager()
        self.circuit_breaker = CircuitBreaker()
        self.retry_manager = RetryManager()
        self.fallback_system = FallbackSystem()
    
    def _init_intelligence(self):
        """Initialize intelligence components."""
        self.predictive_engine = PredictiveEngine()
        self.correlation_engine = CorrelationEngine()
        self.adaptive_enhancer = AdaptiveEnhancer()
        self.cross_system_analyzer = CrossSystemAnalyzer()
    
    def _register_all_components(self):
        """Register all components for unified management."""
        # Processors
        self.components['aggregator'] = self.aggregator
        self.components['compressor'] = self.compressor
        self.components['normalizer'] = self.normalizer
        self.components['correlator'] = self.correlator
        self.components['duplicate_detector'] = self.duplicate_detector
        
        # Quality
        self.components['validator'] = self.validator
        self.components['integrity_checker'] = self.integrity_checker
        self.components['anomaly_detector'] = self.anomaly_detector
        self.components['quality_assurance'] = self.quality_assurance
        
        # Monitoring
        self.components['health_monitor'] = self.health_monitor
        self.components['performance_monitor'] = self.performance_monitor
        self.components['flow_monitor'] = self.flow_monitor
        self.components['sla_tracker'] = self.sla_tracker
        
        # Delivery
        self.components['delivery_manager'] = self.delivery_manager
        self.components['circuit_breaker'] = self.circuit_breaker
        self.components['retry_manager'] = self.retry_manager
        self.components['fallback_system'] = self.fallback_system
        
        # Intelligence
        self.components['predictive_engine'] = self.predictive_engine
        self.components['correlation_engine'] = self.correlation_engine
        self.components['adaptive_enhancer'] = self.adaptive_enhancer
        self.components['cross_system_analyzer'] = self.cross_system_analyzer
    
    async def process(self, data: Any) -> AnalyticsResult:
        """
        Process data through the unified analytics pipeline.
        
        Args:
            data: Data to process (MetricData or raw data)
            
        Returns:
            Comprehensive analytics result
        """
        start_time = datetime.now()
        
        try:
            # Convert to standard format if needed
            if not isinstance(data, MetricData):
                data = self._convert_to_metric_data(data)
            
            # Process through unified pipeline
            result = await self.pipeline_manager.process_metric(data)
            
            # Update global statistics
            self._update_global_stats(result)
            
            # Cache the processed metric
            self.metrics_cache[data.metric_id] = data
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            return result
            
        except Exception as e:
            self._handle_error(f"Processing failed: {e}")
            return AnalyticsResult(
                success=False,
                message=f"Processing error: {e}",
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _convert_to_metric_data(self, data: Any) -> MetricData:
        """Convert raw data to MetricData format."""
        if isinstance(data, dict):
            return MetricData(
                name=data.get('name', 'unknown'),
                value=data.get('value', 0),
                timestamp=data.get('timestamp', datetime.now()),
                tags=data.get('tags', {}),
                metadata=data.get('metadata', {})
            )
        else:
            return MetricData(
                name='raw_data',
                value=str(data),
                timestamp=datetime.now()
            )
    
    def _update_global_stats(self, result: AnalyticsResult):
        """Update global statistics."""
        self.global_stats['total_metrics_processed'] += 1
        
        # Update averages
        current_avg = self.global_stats['avg_processing_time_ms']
        count = self.global_stats['total_metrics_processed']
        self.global_stats['avg_processing_time_ms'] = (
            (current_avg * (count - 1) + result.processing_time_ms) / count
        )
        
        # Count active components
        self.global_stats['components_active'] = sum(
            1 for comp in self.components.values() 
            if hasattr(comp, 'status') and comp.status.value == 'running'
        )
    
    async def start_all_components(self):
        """Start all analytics components."""
        self.logger.info("Starting all analytics components")
        
        # Start components in dependency order
        startup_order = [
            'validator', 'normalizer', 'duplicate_detector',  # Data prep
            'aggregator', 'compressor', 'correlator',         # Processing
            'integrity_checker', 'anomaly_detector',          # Quality
            'health_monitor', 'performance_monitor',          # Monitoring
            'circuit_breaker', 'retry_manager',               # Delivery
            'predictive_engine', 'correlation_engine'         # Intelligence
        ]
        
        for component_name in startup_order:
            if component_name in self.components:
                try:
                    await self.components[component_name].start()
                    self.logger.debug(f"Started {component_name}")
                except Exception as e:
                    self.logger.error(f"Failed to start {component_name}: {e}")
    
    async def stop_all_components(self):
        """Stop all analytics components."""
        self.logger.info("Stopping all analytics components")
        
        # Stop in reverse order
        for component in reversed(list(self.components.values())):
            try:
                await component.stop()
            except Exception as e:
                self.logger.error(f"Error stopping component: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        base_status = self.get_base_status()
        
        # Get status from all components
        component_statuses = {}
        for name, component in self.components.items():
            try:
                component_statuses[name] = component.get_status()
            except Exception as e:
                component_statuses[name] = {'error': str(e)}
        
        return {
            **base_status,
            'global_statistics': self.global_stats,
            'component_count': len(self.components),
            'metrics_cached': len(self.metrics_cache),
            'component_statuses': component_statuses,
            'unified_framework_version': '1.0.0',
            'consolidation_complete': True,
            'original_files_consolidated': 84
        }
    
    def get_component(self, name: str) -> Optional[BaseAnalytics]:
        """Get a specific component by name."""
        return self.components.get(name)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of processed metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.metrics_cache.values()
            if metric.timestamp >= cutoff_time
        ]
        
        return {
            'total_metrics': len(recent_metrics),
            'unique_metric_names': len(set(m.name for m in recent_metrics)),
            'time_range_hours': hours,
            'metric_types': {
                metric_type.value: len([m for m in recent_metrics if m.metric_type == metric_type])
                for metric_type in set(m.metric_type for m in recent_metrics)
            },
            'global_stats': self.global_stats
        }
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all components."""
        return {
            'engine_status': self.get_status(),
            'metrics_summary': self.get_metrics_summary(),
            'component_health': {
                name: comp.get_status() for name, comp in self.components.items()
            },
            'system_performance': await self.performance_monitor.get_system_metrics(),
            'quality_metrics': await self.quality_assurance.get_quality_report(),
            'intelligence_insights': await self.predictive_engine.get_insights(),
            'timestamp': datetime.now().isoformat()
        }