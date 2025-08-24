"""
Telemetry Export Manager (Part 3/3) - TestMaster Advanced ML
Advanced export system with ML-driven data processing and multiple format support
Extracted from analytics_telemetry.py (680 lines) → 3 coordinated ML modules
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union
import csv
import gzip

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .telemetry_ml_collector import MLTelemetryEvent, MLSpan, MLMetricPoint, TelemetryLevel
from .telemetry_observability_engine import ObservabilityAlert, PerformanceInsight


@dataclass
class ExportConfiguration:
    """Export configuration with ML optimization settings"""
    
    format_type: str  # json, csv, prometheus, parquet, ml_features
    destination: str  # file, http, kafka, s3, database
    compression: str = "gzip"  # none, gzip, lz4, snappy
    batch_size: int = 1000
    export_interval: int = 300  # seconds
    retention_days: int = 7
    enable_ml_processing: bool = True
    feature_extraction: bool = False
    anonymization: bool = False
    
    # ML-specific settings
    dimensionality_reduction: bool = False
    clustering_enabled: bool = False
    anomaly_filtering: bool = False


class AdvancedTelemetryExportManager:
    """
    Advanced telemetry export system with ML-driven processing
    Part 3/3 of the complete telemetry system
    """
    
    def __init__(self,
                 telemetry_collector,
                 observability_engine,
                 export_dir: str = None,
                 default_config: ExportConfiguration = None):
        """Initialize export manager"""
        
        self.telemetry_collector = telemetry_collector
        self.observability_engine = observability_engine
        
        # Export configuration
        self.export_dir = export_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'telemetry_exports'
        )
        os.makedirs(self.export_dir, exist_ok=True)
        
        self.default_config = default_config or ExportConfiguration(
            format_type="json",
            destination="file",
            enable_ml_processing=True
        )
        
        # Export configurations by name
        self.export_configs: Dict[str, ExportConfiguration] = {}
        self.export_handlers: Dict[str, Callable] = {}
        
        # ML Processing Components
        self.feature_scaler = StandardScaler()
        self.dimensionality_reducer: Optional[PCA] = None
        self.export_clusterer: Optional[KMeans] = None
        
        # Export State
        self.export_queue: deque = deque(maxlen=10000)
        self.export_buffers: Dict[str, List] = defaultdict(list)
        self.last_export_times: Dict[str, datetime] = {}
        
        # Processing Statistics
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'bytes_exported': 0,
            'ml_processing_time': 0.0,
            'compression_ratio': 0.0,
            'last_export_time': None
        }
        
        # Synchronization
        self.export_lock = RLock()
        self.processing_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Setup default configurations and start export loop
        self._setup_default_configurations()
        self._initialize_ml_components()
        asyncio.create_task(self._export_loop())
    
    def _setup_default_configurations(self):
        """Setup default export configurations"""
        
        # JSON export for general telemetry
        self.add_export_config("telemetry_json", ExportConfiguration(
            format_type="json",
            destination="file",
            batch_size=500,
            export_interval=300,
            enable_ml_processing=True
        ))
        
        # CSV export for data analysis
        self.add_export_config("analytics_csv", ExportConfiguration(
            format_type="csv",
            destination="file",
            batch_size=1000,
            export_interval=600,
            feature_extraction=True
        ))
        
        # Prometheus metrics export
        self.add_export_config("metrics_prometheus", ExportConfiguration(
            format_type="prometheus",
            destination="file",
            batch_size=100,
            export_interval=60,
            enable_ml_processing=False
        ))
        
        # ML features export
        self.add_export_config("ml_features", ExportConfiguration(
            format_type="ml_features",
            destination="file",
            batch_size=200,
            export_interval=300,
            enable_ml_processing=True,
            dimensionality_reduction=True,
            clustering_enabled=True
        ))
    
    def _initialize_ml_components(self):
        """Initialize ML components for export processing"""
        
        try:
            # PCA for dimensionality reduction
            self.dimensionality_reducer = PCA(n_components=10, random_state=42)
            
            # Clustering for export data organization
            self.export_clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
            
            self.logger.info("Export ML components initialized")
            
        except Exception as e:
            self.logger.error(f"Export ML initialization failed: {e}")
    
    def add_export_config(self, name: str, config: ExportConfiguration):
        """Add export configuration"""
        
        self.export_configs[name] = config
        self.export_buffers[name] = []
        self.last_export_times[name] = datetime.now()
        
        # Setup handler based on format
        handler = self._get_format_handler(config.format_type)
        if handler:
            self.export_handlers[name] = handler
        
        self.logger.info(f"Export configuration added: {name}")
    
    def _get_format_handler(self, format_type: str) -> Optional[Callable]:
        """Get handler function for format type"""
        
        handlers = {
            "json": self._export_json,
            "csv": self._export_csv,
            "prometheus": self._export_prometheus,
            "ml_features": self._export_ml_features,
            "parquet": self._export_parquet
        }
        
        return handlers.get(format_type)
    
    async def _export_loop(self):
        """Main export processing loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.now()
                
                # Process each export configuration
                for config_name, config in self.export_configs.items():
                    last_export = self.last_export_times.get(config_name, datetime.min)
                    time_since_export = (current_time - last_export).total_seconds()
                    
                    # Check if export is due
                    buffer_size = len(self.export_buffers[config_name])
                    if (time_since_export >= config.export_interval or 
                        buffer_size >= config.batch_size):
                        
                        await self._process_export(config_name, config)
                
                # Cleanup old exports
                await self._cleanup_old_exports()
                
            except Exception as e:
                self.logger.error(f"Export loop error: {e}")
                await asyncio.sleep(5)
    
    async def queue_for_export(self, data_type: str, data: Any):
        """Queue data for export processing"""
        
        export_item = {
            'type': data_type,
            'data': data,
            'timestamp': datetime.now(),
            'processed': False
        }
        
        with self.export_lock:
            self.export_queue.append(export_item)
            
            # Add to relevant buffers
            for config_name, config in self.export_configs.items():
                if self._should_export_to_config(data_type, config):
                    self.export_buffers[config_name].append(export_item)
    
    def _should_export_to_config(self, data_type: str, config: ExportConfiguration) -> bool:
        """Determine if data should be exported to specific configuration"""
        
        # Simple mapping - can be made more sophisticated
        type_mappings = {
            "json": ["events", "spans", "metrics", "alerts", "insights"],
            "csv": ["events", "spans", "metrics"],
            "prometheus": ["metrics"],
            "ml_features": ["events", "spans", "metrics"],
            "parquet": ["events", "spans", "metrics", "alerts"]
        }
        
        allowed_types = type_mappings.get(config.format_type, [])
        return data_type in allowed_types
    
    async def _process_export(self, config_name: str, config: ExportConfiguration):
        """Process export for specific configuration"""
        
        try:
            with self.export_lock:
                buffer_data = list(self.export_buffers[config_name])
                self.export_buffers[config_name].clear()
                self.last_export_times[config_name] = datetime.now()
            
            if not buffer_data:
                return
            
            # ML processing if enabled
            if config.enable_ml_processing:
                buffer_data = await self._apply_ml_processing(buffer_data, config)
            
            # Get export handler
            handler = self.export_handlers.get(config_name)
            if not handler:
                self.logger.error(f"No handler for export config: {config_name}")
                return
            
            # Export data
            start_time = time.time()
            success = await handler(config_name, config, buffer_data)
            processing_time = time.time() - start_time
            
            # Update statistics
            with self.processing_lock:
                self.export_stats['total_exports'] += 1
                if success:
                    self.export_stats['successful_exports'] += 1
                else:
                    self.export_stats['failed_exports'] += 1
                
                self.export_stats['ml_processing_time'] += processing_time
                self.export_stats['last_export_time'] = datetime.now()
            
            self.logger.info(f"Export processed: {config_name} ({len(buffer_data)} items)")
            
        except Exception as e:
            self.logger.error(f"Export processing error for {config_name}: {e}")
            self.export_stats['failed_exports'] += 1
    
    async def _apply_ml_processing(self, data: List[Dict], config: ExportConfiguration) -> List[Dict]:
        """Apply ML processing to export data"""
        
        try:
            processed_data = list(data)
            
            # Anonymization
            if config.anonymization:
                processed_data = await self._anonymize_data(processed_data)
            
            # Anomaly filtering
            if config.anomaly_filtering:
                processed_data = await self._filter_anomalies(processed_data)
            
            # Feature extraction
            if config.feature_extraction:
                processed_data = await self._extract_features(processed_data)
            
            # Dimensionality reduction
            if config.dimensionality_reduction and len(processed_data) > 50:
                processed_data = await self._reduce_dimensions(processed_data)
            
            # Clustering
            if config.clustering_enabled and len(processed_data) > 20:
                processed_data = await self._apply_clustering(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"ML processing error: {e}")
            return data
    
    async def _extract_features(self, data: List[Dict]) -> List[Dict]:
        """Extract ML features from telemetry data"""
        
        enhanced_data = []
        
        for item in data:
            enhanced_item = dict(item)
            
            if item['type'] == 'events':
                event_data = item['data']
                if isinstance(event_data, MLTelemetryEvent):
                    # Extract event features
                    features = {
                        'message_length': len(event_data.message),
                        'attribute_count': len(event_data.attributes),
                        'metric_count': len(event_data.metrics) if event_data.metrics else 0,
                        'has_error': 1 if event_data.error else 0,
                        'duration_log': np.log1p(event_data.duration_ms or 0),
                        'level_numeric': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(event_data.level.value),
                        'hour_of_day': event_data.timestamp.hour,
                        'day_of_week': event_data.timestamp.weekday()
                    }
                    enhanced_item['ml_features'] = features
            
            elif item['type'] == 'spans':
                span_data = item['data']
                if isinstance(span_data, MLSpan):
                    # Extract span features
                    features = {
                        'duration_ms': span_data.duration_ms or 0,
                        'event_count': len(span_data.events),
                        'attribute_count': len(span_data.attributes),
                        'has_parent': 1 if span_data.parent_span_id else 0,
                        'status_numeric': 1 if span_data.status == 'ok' else 0,
                        'hour_of_day': span_data.start_time.hour,
                        'operation_hash': hash(span_data.operation_name) % 1000
                    }
                    enhanced_item['ml_features'] = features
            
            enhanced_data.append(enhanced_item)
        
        return enhanced_data
    
    async def _export_json(self, config_name: str, config: ExportConfiguration, 
                          data: List[Dict]) -> bool:
        """Export data in JSON format"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config_name}_{timestamp}.json"
            filepath = os.path.join(self.export_dir, filename)
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'config_name': config_name,
                    'item_count': len(data),
                    'service_name': self.telemetry_collector.service_name
                },
                'items': []
            }
            
            # Process each item
            for item in data:
                processed_item = await self._serialize_item(item)
                export_data['items'].append(processed_item)
            
            # Write to file (with compression if enabled)
            if config.compression == "gzip":
                with gzip.open(f"{filepath}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                filepath = f"{filepath}.gz"
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            # Update statistics
            file_size = os.path.getsize(filepath)
            self.export_stats['bytes_exported'] += file_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"JSON export error: {e}")
            return False
    
    async def _export_csv(self, config_name: str, config: ExportConfiguration,
                         data: List[Dict]) -> bool:
        """Export data in CSV format"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config_name}_{timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Flatten data for CSV
            flattened_data = []
            for item in data:
                flat_item = await self._flatten_item(item)
                flattened_data.append(flat_item)
            
            if not flattened_data:
                return True
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
            
            # Compress if enabled
            if config.compression == "gzip":
                with open(filepath, 'rb') as f_in:
                    with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(filepath)
                filepath = f"{filepath}.gz"
            
            # Update statistics
            file_size = os.path.getsize(filepath)
            self.export_stats['bytes_exported'] += file_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return False
    
    async def _export_prometheus(self, config_name: str, config: ExportConfiguration,
                               data: List[Dict]) -> bool:
        """Export metrics in Prometheus format"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config_name}_{timestamp}.prom"
            filepath = os.path.join(self.export_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Group metrics by name
                metrics_by_name = defaultdict(list)
                
                for item in data:
                    if item['type'] == 'metrics' and isinstance(item['data'], MLMetricPoint):
                        metric = item['data']
                        metrics_by_name[metric.name].append(metric)
                
                # Write Prometheus format
                for metric_name, metrics in metrics_by_name.items():
                    # Get latest metric for each label combination
                    latest_metrics = {}
                    for metric in metrics:
                        label_key = str(sorted(metric.labels.items()) if metric.labels else [])
                        if (label_key not in latest_metrics or 
                            metric.timestamp > latest_metrics[label_key].timestamp):
                            latest_metrics[label_key] = metric
                    
                    # Write metrics
                    for metric in latest_metrics.values():
                        labels_str = ""
                        if metric.labels:
                            label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                            labels_str = "{" + ",".join(label_pairs) + "}"
                        
                        metric_line = f"{metric_name.replace('.', '_')}{labels_str} {metric.value}"
                        f.write(metric_line + "\n")
            
            # Update statistics
            file_size = os.path.getsize(filepath)
            self.export_stats['bytes_exported'] += file_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Prometheus export error: {e}")
            return False
    
    async def _export_ml_features(self, config_name: str, config: ExportConfiguration,
                                 data: List[Dict]) -> bool:
        """Export ML-processed features"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config_name}_{timestamp}.json"
            filepath = os.path.join(self.export_dir, filename)
            
            # Extract and process features
            feature_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'feature_extraction_config': {
                        'dimensionality_reduction': config.dimensionality_reduction,
                        'clustering_enabled': config.clustering_enabled,
                        'anomaly_filtering': config.anomaly_filtering
                    }
                },
                'features': [],
                'clusters': {},
                'anomalies': []
            }
            
            # Process items with ML features
            for item in data:
                if 'ml_features' in item:
                    feature_item = {
                        'timestamp': item['timestamp'].isoformat(),
                        'type': item['type'],
                        'features': item['ml_features']
                    }
                    
                    # Add clustering info if available
                    if 'cluster_id' in item:
                        feature_item['cluster_id'] = item['cluster_id']
                    
                    # Add anomaly info if available
                    if 'anomaly_score' in item:
                        feature_item['anomaly_score'] = item['anomaly_score']
                    
                    feature_data['features'].append(feature_item)
            
            # Write ML features export
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(feature_data, f, indent=2, default=str)
            
            # Update statistics
            file_size = os.path.getsize(filepath)
            self.export_stats['bytes_exported'] += file_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"ML features export error: {e}")
            return False
    
    def get_export_status(self) -> Dict[str, Any]:
        """Get comprehensive export status"""
        
        config_status = {}
        for config_name, config in self.export_configs.items():
            buffer_size = len(self.export_buffers[config_name])
            last_export = self.last_export_times.get(config_name)
            
            config_status[config_name] = {
                'format': config.format_type,
                'buffer_size': buffer_size,
                'last_export': last_export.isoformat() if last_export else None,
                'export_interval': config.export_interval,
                'ml_processing': config.enable_ml_processing
            }
        
        return {
            'export_statistics': self.export_stats.copy(),
            'configurations': config_status,
            'export_directory': self.export_dir,
            'queue_size': len(self.export_queue)
        }
    
    async def shutdown(self):
        """Graceful shutdown with final export"""
        
        self.logger.info("Shutting down export manager...")
        
        # Process remaining exports
        for config_name, config in self.export_configs.items():
            if self.export_buffers[config_name]:
                await self._process_export(config_name, config)
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Export manager shutdown complete")