"""
Intelligent Data Pipeline - TestMaster Advanced ML
ML-driven data processing pipeline with adaptive optimization and quality assurance
Enterprise ML Module #5/8 for comprehensive system intelligence
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
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class PipelineStage(Enum):
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    AGGREGATION = "aggregation"
    OUTPUT = "output"


class DataQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    CRITICAL = 1


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class DataBatch:
    """ML-enhanced data batch with quality metrics"""
    
    batch_id: str
    timestamp: datetime
    source: str
    size_bytes: int
    record_count: int
    data_schema: Dict[str, str] = field(default_factory=dict)
    
    # Quality metrics
    data_quality: DataQuality = DataQuality.GOOD
    completeness_score: float = 1.0
    accuracy_score: float = 1.0
    consistency_score: float = 1.0
    timeliness_score: float = 1.0
    
    # ML Enhancement Fields
    anomaly_score: float = 0.0
    predicted_processing_time: float = 0.0
    optimization_score: float = 1.0
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Processing tracking
    current_stage: PipelineStage = PipelineStage.INGESTION
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    stage_durations: Dict[PipelineStage, float] = field(default_factory=dict)
    
    # Error handling
    error_count: int = 0
    retry_count: int = 0
    last_error: Optional[str] = None


@dataclass
class PipelineStageConfig:
    """Configuration for pipeline stage with ML optimization"""
    
    stage: PipelineStage
    name: str
    enabled: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    timeout: int = 300
    retry_attempts: int = 3
    
    # ML Enhancement
    ml_optimized: bool = True
    adaptive_scaling: bool = True
    quality_threshold: float = 0.7
    performance_target: float = 1.0  # processing time multiplier
    
    # Processing function
    processor_function: Optional[Callable] = None
    
    # Statistics
    batches_processed: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 1.0
    throughput_score: float = 1.0


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline performance metrics"""
    
    timestamp: datetime
    total_batches: int
    successful_batches: int
    failed_batches: int
    average_batch_size: float
    
    # Performance metrics
    overall_throughput: float  # batches per second
    average_latency: float     # seconds per batch
    resource_utilization: float
    quality_score: float
    
    # Stage-specific metrics
    stage_performance: Dict[PipelineStage, Dict[str, float]] = field(default_factory=dict)
    
    # ML insights
    ml_optimizations_applied: int = 0
    anomalies_detected: int = 0
    quality_improvements: float = 0.0
    performance_improvements: float = 0.0


class IntelligentDataPipeline:
    """
    ML-enhanced data processing pipeline with adaptive optimization
    """
    
    def __init__(self,
                 enable_ml_optimization: bool = True,
                 quality_monitoring: bool = True,
                 auto_scaling: bool = True,
                 metrics_interval: int = 60):
        """Initialize intelligent data pipeline"""
        
        self.enable_ml_optimization = enable_ml_optimization
        self.quality_monitoring = quality_monitoring
        self.auto_scaling = auto_scaling
        self.metrics_interval = metrics_interval
        
        # ML Models for Pipeline Intelligence
        self.processing_time_predictor: Optional[RandomForestClassifier] = None
        self.quality_classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.optimization_recommender: Optional[Ridge] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.quality_scaler = RobustScaler()
        self.pipeline_feature_history: deque = deque(maxlen=5000)
        
        # Pipeline Configuration
        self.stage_configs: Dict[PipelineStage, PipelineStageConfig] = {}
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.data_outputs: Dict[str, Dict[str, Any]] = {}
        
        # Data Processing
        self.processing_queue: deque = deque()
        self.active_batches: Dict[str, DataBatch] = {}
        self.completed_batches: deque = deque(maxlen=1000)
        self.failed_batches: deque = deque(maxlen=500)
        
        # Performance Monitoring
        self.pipeline_metrics: deque = deque(maxlen=1000)
        self.stage_performance_history: Dict[PipelineStage, deque] = defaultdict(lambda: deque(maxlen=100))
        self.quality_trends: deque = deque(maxlen=500)
        
        # ML Insights and Optimization
        self.ml_recommendations: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.quality_alerts: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_concurrent_batches = 10
        self.quality_alert_threshold = 0.6
        self.performance_degradation_threshold = 0.2
        self.auto_optimization_enabled = True
        
        # Statistics
        self.pipeline_stats = {
            'batches_processed': 0,
            'total_data_volume': 0,
            'quality_checks_performed': 0,
            'ml_optimizations_applied': 0,
            'anomalies_detected': 0,
            'pipeline_restarts': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.pipeline_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and pipeline stages
        if enable_ml_optimization:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_optimization_loop())
        
        self._initialize_default_stages()
        asyncio.create_task(self._pipeline_processing_loop())
        asyncio.create_task(self._metrics_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for pipeline intelligence"""
        
        try:
            # Processing time prediction
            self.processing_time_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                min_samples_split=5
            )
            
            # Data quality classification
            self.quality_classifier = RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Anomaly detection for unusual data patterns
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Pipeline optimization recommendations
            self.optimization_recommender = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            self.logger.info("Pipeline ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Pipeline ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def _initialize_default_stages(self):
        """Initialize default pipeline stages"""
        
        # Ingestion stage
        ingestion_config = PipelineStageConfig(
            stage=PipelineStage.INGESTION,
            name="Data Ingestion",
            parallel_processing=True,
            max_workers=4,
            timeout=120
        )
        
        # Validation stage
        validation_config = PipelineStageConfig(
            stage=PipelineStage.VALIDATION,
            name="Data Validation",
            parallel_processing=True,
            max_workers=2,
            timeout=60
        )
        
        # Transformation stage
        transformation_config = PipelineStageConfig(
            stage=PipelineStage.TRANSFORMATION,
            name="Data Transformation",
            parallel_processing=True,
            max_workers=6,
            timeout=180
        )
        
        # Enrichment stage
        enrichment_config = PipelineStageConfig(
            stage=PipelineStage.ENRICHMENT,
            name="Data Enrichment",
            parallel_processing=False,
            max_workers=2,
            timeout=300
        )
        
        # Aggregation stage
        aggregation_config = PipelineStageConfig(
            stage=PipelineStage.AGGREGATION,
            name="Data Aggregation",
            parallel_processing=True,
            max_workers=4,
            timeout=120
        )
        
        # Output stage
        output_config = PipelineStageConfig(
            stage=PipelineStage.OUTPUT,
            name="Data Output",
            parallel_processing=True,
            max_workers=3,
            timeout=90
        )
        
        self.stage_configs = {
            config.stage: config
            for config in [ingestion_config, validation_config, transformation_config,
                          enrichment_config, aggregation_config, output_config]
        }
    
    def register_data_source(self,
                           source_id: str,
                           source_type: str,
                           connection_config: Dict[str, Any],
                           schema: Dict[str, str] = None) -> bool:
        """Register data source for pipeline processing"""
        
        try:
            with self.pipeline_lock:
                self.data_sources[source_id] = {
                    'source_type': source_type,
                    'connection_config': connection_config,
                    'schema': schema or {},
                    'last_processed': None,
                    'total_batches': 0,
                    'total_volume': 0,
                    'quality_score': 1.0
                }
            
            self.logger.info(f"Data source registered: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Data source registration failed: {e}")
            return False
    
    def submit_data_batch(self,
                         source: str,
                         data: Any,
                         batch_metadata: Dict[str, Any] = None) -> str:
        """Submit data batch for pipeline processing"""
        
        try:
            batch_id = str(uuid.uuid4())
            
            # Calculate batch size
            if isinstance(data, (list, tuple)):
                record_count = len(data)
                size_bytes = len(str(data).encode('utf-8'))
            elif isinstance(data, pd.DataFrame):
                record_count = len(data)
                size_bytes = data.memory_usage(deep=True).sum()
            elif isinstance(data, dict):
                record_count = 1
                size_bytes = len(json.dumps(data).encode('utf-8'))
            else:
                record_count = 1
                size_bytes = len(str(data).encode('utf-8'))
            
            # Create data batch
            batch = DataBatch(
                batch_id=batch_id,
                timestamp=datetime.now(),
                source=source,
                size_bytes=size_bytes,
                record_count=record_count
            )
            
            # ML enhancement for batch prediction
            if self.enable_ml_optimization:
                await self._enhance_batch_with_ml(batch, data)
            
            # Add to processing queue
            with self.pipeline_lock:
                self.processing_queue.append(batch)
                self.pipeline_stats['total_data_volume'] += size_bytes
            
            self.logger.info(f"Data batch submitted: {batch_id} ({record_count} records)")
            return batch_id
            
        except Exception as e:
            self.logger.error(f"Data batch submission failed: {e}")
            return ""
    
    async def _enhance_batch_with_ml(self, batch: DataBatch, data: Any):
        """Enhance batch with ML predictions and insights"""
        
        try:
            with self.ml_lock:
                # Extract batch features
                features = await self._extract_batch_features(batch, data)
                
                # Predict processing time
                if self.processing_time_predictor and len(self.pipeline_feature_history) >= 50:
                    predicted_time = await self._predict_processing_time(features)
                    batch.predicted_processing_time = predicted_time
                
                # Predict data quality
                if self.quality_classifier and self.quality_monitoring:
                    quality_prediction = await self._predict_data_quality(features, data)
                    batch.data_quality = quality_prediction
                    batch.ml_insights['quality_prediction'] = quality_prediction.value
                
                # Anomaly detection
                if self.anomaly_detector and len(self.pipeline_feature_history) >= 30:
                    anomaly_score = self.anomaly_detector.decision_function([features])[0]
                    batch.anomaly_score = float(anomaly_score)
                    
                    if anomaly_score < -0.5:
                        batch.ml_insights['anomaly_detected'] = True
                        self.pipeline_stats['anomalies_detected'] += 1
                
                # Optimization score calculation
                batch.optimization_score = await self._calculate_batch_optimization_score(batch, features)
                
                # Store features for model training
                self.pipeline_feature_history.append(features)
                
        except Exception as e:
            self.logger.error(f"ML batch enhancement failed: {e}")
    
    def _extract_batch_features(self, batch: DataBatch, data: Any) -> np.ndarray:
        """Extract ML features from data batch"""
        
        try:
            # Basic batch features
            size_mb = batch.size_bytes / (1024 * 1024)
            record_count_normalized = min(batch.record_count / 10000, 1.0)
            
            # Temporal features
            hour = batch.timestamp.hour
            day_of_week = batch.timestamp.weekday()
            
            # Data characteristics
            data_complexity = 0.5  # Default
            if isinstance(data, pd.DataFrame):
                data_complexity = len(data.columns) / 50.0  # Normalize by typical column count
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                data_complexity = len(data[0].keys()) / 20.0
            
            # Source characteristics
            source_hash = hash(batch.source) % 1000 / 1000.0
            
            # Historical performance
            recent_batches = list(self.completed_batches)[-10:]
            avg_processing_time = np.mean([
                sum(b.stage_durations.values()) for b in recent_batches
                if b.source == batch.source
            ]) if recent_batches else 30.0
            
            # Create feature vector
            features = np.array([
                size_mb,
                record_count_normalized,
                data_complexity,
                source_hash,
                hour / 24.0,
                day_of_week / 7.0,
                avg_processing_time / 100.0,  # Normalize
                len(self.active_batches) / self.max_concurrent_batches,
                batch.completeness_score,
                batch.accuracy_score
            ])
            
            return features.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.zeros(10)  # Default feature vector
    
    async def _pipeline_processing_loop(self):
        """Main pipeline processing loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1)  # Process batches frequently
                
                # Check if we can process more batches
                if (len(self.active_batches) >= self.max_concurrent_batches or 
                    not self.processing_queue):
                    continue
                
                # Get next batch from queue
                with self.pipeline_lock:
                    if self.processing_queue:
                        batch = self.processing_queue.popleft()
                        self.active_batches[batch.batch_id] = batch
                    else:
                        continue
                
                # Start batch processing
                asyncio.create_task(self._process_data_batch(batch))
                
            except Exception as e:
                self.logger.error(f"Pipeline processing loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_data_batch(self, batch: DataBatch):
        """Process data batch through all pipeline stages"""
        
        try:
            batch.processing_status = ProcessingStatus.PROCESSING
            batch.processing_start = datetime.now()
            
            # Process through each enabled stage
            for stage in PipelineStage:
                if stage not in self.stage_configs or not self.stage_configs[stage].enabled:
                    continue
                
                batch.current_stage = stage
                stage_start = time.time()
                
                # Execute stage processing
                success = await self._execute_pipeline_stage(batch, stage)
                
                stage_duration = time.time() - stage_start
                batch.stage_durations[stage] = stage_duration
                
                if not success:
                    batch.processing_status = ProcessingStatus.FAILED
                    batch.error_count += 1
                    
                    # Retry logic
                    if batch.retry_count < self.stage_configs[stage].retry_attempts:
                        batch.retry_count += 1
                        batch.processing_status = ProcessingStatus.RETRYING
                        await asyncio.sleep(2 ** batch.retry_count)  # Exponential backoff
                        continue
                    else:
                        break
            
            # Finalize batch processing
            if batch.processing_status != ProcessingStatus.FAILED:
                batch.processing_status = ProcessingStatus.COMPLETED
                batch.processing_end = datetime.now()
                
                # Update statistics
                self.pipeline_stats['batches_processed'] += 1
                
                # Move to completed batches
                with self.pipeline_lock:
                    self.completed_batches.append(batch)
                    del self.active_batches[batch.batch_id]
                
                self.logger.info(f"Batch processed successfully: {batch.batch_id}")
            else:
                # Move to failed batches
                with self.pipeline_lock:
                    self.failed_batches.append(batch)
                    del self.active_batches[batch.batch_id]
                
                self.logger.error(f"Batch processing failed: {batch.batch_id}")
            
            # Update ML models with processing data
            if self.enable_ml_optimization:
                await self._update_ml_models_with_batch_data(batch)
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            batch.processing_status = ProcessingStatus.FAILED
            batch.last_error = str(e)
    
    async def _execute_pipeline_stage(self, batch: DataBatch, stage: PipelineStage) -> bool:
        """Execute specific pipeline stage"""
        
        try:
            stage_config = self.stage_configs[stage]
            
            # Update stage statistics
            stage_config.batches_processed += 1
            
            # Execute stage-specific processing
            if stage == PipelineStage.INGESTION:
                success = await self._execute_ingestion_stage(batch)
            elif stage == PipelineStage.VALIDATION:
                success = await self._execute_validation_stage(batch)
            elif stage == PipelineStage.TRANSFORMATION:
                success = await self._execute_transformation_stage(batch)
            elif stage == PipelineStage.ENRICHMENT:
                success = await self._execute_enrichment_stage(batch)
            elif stage == PipelineStage.AGGREGATION:
                success = await self._execute_aggregation_stage(batch)
            elif stage == PipelineStage.OUTPUT:
                success = await self._execute_output_stage(batch)
            else:
                success = True  # Default success for unknown stages
            
            # Update stage performance metrics
            if success:
                stage_config.success_rate = (
                    0.9 * stage_config.success_rate + 0.1 * 1.0
                )
            else:
                stage_config.success_rate = (
                    0.9 * stage_config.success_rate + 0.1 * 0.0
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Stage execution failed for {stage.value}: {e}")
            return False
    
    async def _execute_validation_stage(self, batch: DataBatch) -> bool:
        """Execute data validation stage"""
        
        try:
            if not self.quality_monitoring:
                return True
            
            # Perform quality checks
            quality_scores = await self._perform_quality_checks(batch)
            
            # Update batch quality metrics
            batch.completeness_score = quality_scores.get('completeness', 1.0)
            batch.accuracy_score = quality_scores.get('accuracy', 1.0)
            batch.consistency_score = quality_scores.get('consistency', 1.0)
            batch.timeliness_score = quality_scores.get('timeliness', 1.0)
            
            # Calculate overall quality score
            overall_quality = np.mean(list(quality_scores.values()))
            
            if overall_quality >= 0.8:
                batch.data_quality = DataQuality.EXCELLENT
            elif overall_quality >= 0.6:
                batch.data_quality = DataQuality.GOOD
            elif overall_quality >= 0.4:
                batch.data_quality = DataQuality.FAIR
            elif overall_quality >= 0.2:
                batch.data_quality = DataQuality.POOR
            else:
                batch.data_quality = DataQuality.CRITICAL
            
            # Quality alert if below threshold
            if overall_quality < self.quality_alert_threshold:
                await self._generate_quality_alert(batch, quality_scores)
            
            self.pipeline_stats['quality_checks_performed'] += 1
            
            return overall_quality >= 0.2  # Minimum acceptable quality
            
        except Exception as e:
            self.logger.error(f"Validation stage failed: {e}")
            return False
    
    async def _metrics_monitoring_loop(self):
        """Pipeline metrics monitoring and analysis"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.metrics_interval)
                
                # Calculate current metrics
                current_metrics = await self._calculate_pipeline_metrics()
                self.pipeline_metrics.append(current_metrics)
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
                # Check for performance degradation
                await self._check_performance_degradation()
                
                # Auto-optimization if enabled
                if self.auto_optimization_enabled and self.enable_ml_optimization:
                    await self._auto_optimize_pipeline()
                
            except Exception as e:
                self.logger.error(f"Metrics monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _ml_optimization_loop(self):
        """ML optimization and insights generation loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if len(self.pipeline_feature_history) >= 100:
                    # Retrain ML models
                    await self._retrain_ml_models()
                    
                    # Generate optimization recommendations
                    await self._generate_ml_recommendations()
                    
                    # Update pipeline configurations
                    await self._apply_ml_optimizations()
                
            except Exception as e:
                self.logger.error(f"ML optimization loop error: {e}")
                await asyncio.sleep(30)
    
    def get_pipeline_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive pipeline dashboard"""
        
        # Stage performance summary
        stage_summary = {}
        for stage, config in self.stage_configs.items():
            stage_summary[stage.value] = {
                'enabled': config.enabled,
                'batches_processed': config.batches_processed,
                'average_processing_time': config.average_processing_time,
                'success_rate': config.success_rate,
                'throughput_score': config.throughput_score,
                'parallel_processing': config.parallel_processing,
                'max_workers': config.max_workers
            }
        
        # Current pipeline status
        pipeline_status = {
            'active_batches': len(self.active_batches),
            'queued_batches': len(self.processing_queue),
            'completed_batches_24h': len([
                b for b in self.completed_batches
                if (datetime.now() - b.timestamp) < timedelta(hours=24)
            ]),
            'failed_batches_24h': len([
                b for b in self.failed_batches
                if (datetime.now() - b.timestamp) < timedelta(hours=24)
            ])
        }
        
        # ML insights
        ml_status = {
            'ml_optimization_enabled': self.enable_ml_optimization,
            'feature_history_size': len(self.pipeline_feature_history),
            'ml_recommendations': len(self.ml_recommendations),
            'anomalies_detected': self.pipeline_stats['anomalies_detected'],
            'optimizations_applied': self.pipeline_stats['ml_optimizations_applied']
        }
        
        # Recent metrics
        recent_metrics = self.pipeline_metrics[-1] if self.pipeline_metrics else None
        
        return {
            'pipeline_overview': {
                'total_batches_processed': self.pipeline_stats['batches_processed'],
                'total_data_volume_gb': self.pipeline_stats['total_data_volume'] / (1024**3),
                'overall_success_rate': self._calculate_overall_success_rate(),
                'average_throughput': recent_metrics.overall_throughput if recent_metrics else 0.0,
                'average_latency': recent_metrics.average_latency if recent_metrics else 0.0
            },
            'pipeline_status': pipeline_status,
            'stage_performance': stage_summary,
            'statistics': self.pipeline_stats.copy(),
            'ml_status': ml_status,
            'recent_insights': self.ml_recommendations[-5:] if self.ml_recommendations else []
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall pipeline success rate"""
        
        total_batches = self.pipeline_stats['batches_processed']
        if total_batches == 0:
            return 1.0
        
        failed_batches = len(self.failed_batches)
        return max(0.0, (total_batches - failed_batches) / total_batches)
    
    async def shutdown(self):
        """Graceful shutdown of data pipeline"""
        
        self.logger.info("Shutting down intelligent data pipeline...")
        
        # Process remaining batches (with timeout)
        timeout = 60
        while self.active_batches and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Intelligent data pipeline shutdown complete")