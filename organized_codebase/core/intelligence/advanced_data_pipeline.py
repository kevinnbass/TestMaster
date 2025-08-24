"""
Advanced Data Pipeline Manager
=============================

Enterprise-grade data pipeline with sanitization, validation, transformation,
and real-time processing. Extracted from 892-line archive data sanitizer
and other pipeline components.

Provides comprehensive ETL/ELT capabilities with data quality assurance.
"""

import asyncio
import hashlib
import html
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Data validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class DataType(Enum):
    """Supported data types for validation"""
    NUMERIC = "numeric"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    SQL_QUERY = "sql_query"
    FILE_PATH = "file_path"
    IDENTIFIER = "identifier"


class PipelineStage(Enum):
    """Data pipeline processing stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    SANITIZATION = "sanitization"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    QUALITY_CHECK = "quality_check"
    OUTPUT = "output"


class ProcessingStatus(Enum):
    """Data processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


@dataclass
class ValidationRule:
    """Data validation rule definition"""
    rule_id: str = field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    field_path: str = ""
    data_type: DataType = DataType.STRING
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value against this rule"""
        try:
            # Check required
            if self.required and (value is None or value == ""):
                return False, "Required field is missing"
            
            if value is None:
                return True, None
            
            # Type-specific validation
            if self.data_type == DataType.NUMERIC:
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, "Invalid numeric value"
            
            elif self.data_type == DataType.EMAIL:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    return False, "Invalid email format"
            
            elif self.data_type == DataType.URL:
                url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
                if not re.match(url_pattern, str(value)):
                    return False, "Invalid URL format"
            
            elif self.data_type == DataType.DATE:
                try:
                    datetime.fromisoformat(str(value))
                except ValueError:
                    return False, "Invalid date format"
            
            # Length validation
            if isinstance(value, str):
                if self.min_length and len(value) < self.min_length:
                    return False, f"Minimum length {self.min_length} required"
                if self.max_length and len(value) > self.max_length:
                    return False, f"Maximum length {self.max_length} exceeded"
            
            # Pattern validation
            if self.pattern and isinstance(value, str):
                if not re.match(self.pattern, value):
                    return False, "Pattern validation failed"
            
            # Allowed values
            if self.allowed_values and value not in self.allowed_values:
                return False, f"Value not in allowed list: {self.allowed_values}"
            
            # Custom validation
            if self.custom_validator:
                if not self.custom_validator(value):
                    return False, "Custom validation failed"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class DataRecord:
    """Data record for pipeline processing"""
    record_id: str = field(default_factory=lambda: f"record_{uuid.uuid4().hex[:8]}")
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.now)
    current_stage: PipelineStage = PipelineStage.INGESTION
    status: ProcessingStatus = ProcessingStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    
    def add_error(self, error: str, stage: PipelineStage):
        """Add validation or processing error"""
        self.validation_errors.append(f"[{stage.value}] {error}")
    
    def add_transformation(self, transformation: str):
        """Record applied transformation"""
        self.transformations_applied.append(transformation)


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    records_rejected: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    quality_score_avg: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedDataPipeline:
    """Enterprise-grade data pipeline with comprehensive processing capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Pipeline configuration
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'moderate'))
        self.max_queue_size = self.config.get('max_queue_size', 10000)
        self.batch_size = self.config.get('batch_size', 100)
        self.processing_timeout = self.config.get('processing_timeout', 30)
        
        # Validation rules
        self.validation_rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        
        # Processing queues for each stage
        self.stage_queues: Dict[PipelineStage, deque] = {
            stage: deque() for stage in PipelineStage
        }
        
        # Active processing
        self.active_records: Dict[str, DataRecord] = {}
        self.completed_records: deque = deque(maxlen=1000)
        self.quarantined_records: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.metrics = PipelineMetrics()
        self.stage_metrics: Dict[PipelineStage, PipelineMetrics] = {
            stage: PipelineMetrics() for stage in PipelineStage
        }
        
        # Processing threads
        self.is_running = True
        self.processor_threads = {}
        self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start processing
        self._start_processors()
        self.metrics_thread.start()
        
        self.logger.info("Advanced Data Pipeline initialized")
    
    def _start_processors(self):
        """Start processing threads for each pipeline stage"""
        for stage in PipelineStage:
            thread = threading.Thread(
                target=self._stage_processor,
                args=(stage,),
                name=f"processor_{stage.value}",
                daemon=True
            )
            self.processor_threads[stage] = thread
            thread.start()
    
    def add_validation_rule(self, data_source: str, rule: ValidationRule):
        """Add validation rule for specific data source"""
        with self.lock:
            self.validation_rules[data_source].append(rule)
            self.logger.debug(f"Added validation rule {rule.rule_id} for {data_source}")
    
    def ingest_data(self, data: Dict[str, Any], source: str = "default",
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ingest data into pipeline
        
        Args:
            data: Data to process
            source: Data source identifier
            metadata: Additional metadata
            
        Returns:
            Record ID for tracking
        """
        with self.lock:
            # Check queue capacity
            if len(self.stage_queues[PipelineStage.INGESTION]) >= self.max_queue_size:
                raise Exception("Pipeline ingestion queue is full")
            
            # Create data record
            record = DataRecord(
                data=data.copy(),
                metadata=metadata or {},
                current_stage=PipelineStage.INGESTION
            )
            
            # Add source information
            record.metadata['source'] = source
            record.metadata['validation_rules'] = self.validation_rules.get(source, [])
            
            # Add to ingestion queue
            self.stage_queues[PipelineStage.INGESTION].append(record)
            self.active_records[record.record_id] = record
            
            self.logger.debug(f"Ingested record {record.record_id} from {source}")
            return record.record_id
    
    def _stage_processor(self, stage: PipelineStage):
        """Process records for specific pipeline stage"""
        while self.is_running:
            try:
                queue = self.stage_queues[stage]
                
                if queue:
                    with self.lock:
                        record = queue.popleft()
                    
                    # Process record
                    start_time = time.time()
                    success = self._process_record_stage(record, stage)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Update metrics
                    self._update_stage_metrics(stage, success, processing_time)
                    
                    # Move to next stage or complete
                    if success:
                        self._advance_record(record, stage)
                    else:
                        self._handle_processing_failure(record, stage)
                
                else:
                    time.sleep(0.01)  # Small delay when queue is empty
                    
            except Exception as e:
                self.logger.error(f"Stage processor error for {stage.value}: {e}")
                time.sleep(1)
    
    def _process_record_stage(self, record: DataRecord, stage: PipelineStage) -> bool:
        """Process record for specific stage"""
        try:
            record.current_stage = stage
            record.status = ProcessingStatus.PROCESSING
            
            if stage == PipelineStage.VALIDATION:
                return self._validate_record(record)
            elif stage == PipelineStage.SANITIZATION:
                return self._sanitize_record(record)
            elif stage == PipelineStage.TRANSFORMATION:
                return self._transform_record(record)
            elif stage == PipelineStage.ENRICHMENT:
                return self._enrich_record(record)
            elif stage == PipelineStage.QUALITY_CHECK:
                return self._quality_check_record(record)
            elif stage == PipelineStage.OUTPUT:
                return self._output_record(record)
            else:
                return True  # INGESTION stage - just pass through
                
        except Exception as e:
            record.add_error(f"Processing error: {str(e)}", stage)
            return False
    
    def _validate_record(self, record: DataRecord) -> bool:
        """Validate record against rules"""
        validation_rules = record.metadata.get('validation_rules', [])
        if not validation_rules:
            return True
        
        errors = []
        
        for rule in validation_rules:
            # Extract field value
            field_value = self._get_nested_field(record.data, rule.field_path)
            
            # Validate
            is_valid, error_msg = rule.validate(field_value)
            if not is_valid:
                errors.append(f"Field '{rule.field_path}': {error_msg}")
        
        if errors:
            for error in errors:
                record.add_error(error, PipelineStage.VALIDATION)
            
            # Check validation level
            if self.validation_level == ValidationLevel.STRICT:
                return False
            elif self.validation_level == ValidationLevel.MODERATE and len(errors) > 3:
                return False
            # LENIENT - continue processing with warnings
        
        return True
    
    def _sanitize_record(self, record: DataRecord) -> bool:
        """Sanitize record data"""
        try:
            sanitized_data = self._sanitize_dict(record.data)
            record.data = sanitized_data
            record.add_transformation("data_sanitization")
            return True
            
        except Exception as e:
            record.add_error(f"Sanitization failed: {str(e)}", PipelineStage.SANITIZATION)
            return False
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key))
            
            # Sanitize value
            if isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    self._sanitize_dict(item) if isinstance(item, dict)
                    else self._sanitize_value(item)
                    for item in value
                ]
            else:
                sanitized[clean_key] = self._sanitize_value(value)
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string value"""
        if not isinstance(text, str):
            return text
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize individual value"""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, (int, float, bool)):
            return value
        elif value is None:
            return None
        else:
            return self._sanitize_string(str(value))
    
    def _transform_record(self, record: DataRecord) -> bool:
        """Apply transformations to record"""
        try:
            # Add standard metadata
            record.data['_pipeline_processed_at'] = datetime.now().isoformat()
            record.data['_pipeline_record_id'] = record.record_id
            
            # Calculate data hash for integrity
            data_str = json.dumps(record.data, sort_keys=True)
            record.data['_data_hash'] = hashlib.sha256(data_str.encode()).hexdigest()
            
            record.add_transformation("metadata_enrichment")
            return True
            
        except Exception as e:
            record.add_error(f"Transformation failed: {str(e)}", PipelineStage.TRANSFORMATION)
            return False
    
    def _enrich_record(self, record: DataRecord) -> bool:
        """Enrich record with additional data"""
        try:
            # Add processing timestamps
            record.data['_processing_timestamps'] = {
                'ingested_at': record.ingested_at.isoformat(),
                'enriched_at': datetime.now().isoformat()
            }
            
            # Add data quality indicators
            record.data['_quality_indicators'] = {
                'validation_errors': len(record.validation_errors),
                'transformations_applied': len(record.transformations_applied),
                'data_completeness': self._calculate_completeness(record.data)
            }
            
            record.add_transformation("data_enrichment")
            return True
            
        except Exception as e:
            record.add_error(f"Enrichment failed: {str(e)}", PipelineStage.ENRICHMENT)
            return False
    
    def _quality_check_record(self, record: DataRecord) -> bool:
        """Perform final quality check"""
        try:
            # Calculate quality score
            completeness = self._calculate_completeness(record.data)
            error_penalty = min(len(record.validation_errors) * 0.1, 0.5)
            transformation_bonus = min(len(record.transformations_applied) * 0.05, 0.2)
            
            quality_score = max(0, completeness - error_penalty + transformation_bonus)
            record.quality_score = min(1.0, quality_score)
            
            # Check quality threshold
            quality_threshold = self.config.get('quality_threshold', 0.5)
            if record.quality_score < quality_threshold:
                record.add_error(f"Quality score {record.quality_score:.2f} below threshold {quality_threshold}", 
                               PipelineStage.QUALITY_CHECK)
                return False
            
            return True
            
        except Exception as e:
            record.add_error(f"Quality check failed: {str(e)}", PipelineStage.QUALITY_CHECK)
            return False
    
    def _output_record(self, record: DataRecord) -> bool:
        """Output processed record"""
        try:
            record.status = ProcessingStatus.COMPLETED
            
            # Move to completed records
            with self.lock:
                self.completed_records.append(record)
                self.active_records.pop(record.record_id, None)
            
            self.logger.debug(f"Record {record.record_id} completed pipeline processing")
            return True
            
        except Exception as e:
            record.add_error(f"Output failed: {str(e)}", PipelineStage.OUTPUT)
            return False
    
    def _advance_record(self, record: DataRecord, current_stage: PipelineStage):
        """Advance record to next pipeline stage"""
        stages = list(PipelineStage)
        current_index = stages.index(current_stage)
        
        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            with self.lock:
                self.stage_queues[next_stage].append(record)
        else:
            # Record completed pipeline
            record.status = ProcessingStatus.COMPLETED
    
    def _handle_processing_failure(self, record: DataRecord, stage: PipelineStage):
        """Handle processing failure"""
        record.status = ProcessingStatus.FAILED
        
        # Move to quarantine or reject based on severity
        if len(record.validation_errors) > 5 or stage == PipelineStage.QUALITY_CHECK:
            record.status = ProcessingStatus.REJECTED
            with self.lock:
                self.active_records.pop(record.record_id, None)
        else:
            record.status = ProcessingStatus.QUARANTINED
            with self.lock:
                self.quarantined_records.append(record)
                self.active_records.pop(record.record_id, None)
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        total_fields = 0
        complete_fields = 0
        
        def count_fields(obj):
            nonlocal total_fields, complete_fields
            
            if isinstance(obj, dict):
                for value in obj.values():
                    total_fields += 1
                    if value is not None and value != "":
                        complete_fields += 1
                    
                    if isinstance(value, (dict, list)):
                        count_fields(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_fields(item)
        
        count_fields(data)
        return complete_fields / total_fields if total_fields > 0 else 1.0
    
    def _update_stage_metrics(self, stage: PipelineStage, success: bool, processing_time_ms: float):
        """Update processing metrics for stage"""
        with self.lock:
            metrics = self.stage_metrics[stage]
            metrics.records_processed += 1
            
            if success:
                metrics.records_successful += 1
            else:
                metrics.records_failed += 1
            
            # Update processing time
            if not hasattr(metrics, '_processing_times'):
                metrics._processing_times = []
            
            metrics._processing_times.append(processing_time_ms)
            if len(metrics._processing_times) > 100:
                metrics._processing_times = metrics._processing_times[-50:]
            
            metrics.avg_processing_time_ms = sum(metrics._processing_times) / len(metrics._processing_times)
            metrics.last_updated = datetime.now()
    
    def _metrics_loop(self):
        """Background metrics calculation loop"""
        while self.is_running:
            try:
                with self.lock:
                    # Calculate overall metrics
                    total_processed = sum(m.records_processed for m in self.stage_metrics.values())
                    total_successful = sum(m.records_successful for m in self.stage_metrics.values())
                    total_failed = sum(m.records_failed for m in self.stage_metrics.values())
                    
                    self.metrics.records_processed = total_processed
                    self.metrics.records_successful = total_successful
                    self.metrics.records_failed = total_failed
                    self.metrics.records_rejected = len(self.quarantined_records)
                    
                    # Calculate quality score average
                    if self.completed_records:
                        quality_scores = [r.quality_score for r in self.completed_records if r.quality_score > 0]
                        if quality_scores:
                            self.metrics.quality_score_avg = sum(quality_scores) / len(quality_scores)
                    
                    self.metrics.last_updated = datetime.now()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
                time.sleep(30)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        with self.lock:
            return {
                'overall_metrics': {
                    'records_processed': self.metrics.records_processed,
                    'records_successful': self.metrics.records_successful,
                    'records_failed': self.metrics.records_failed,
                    'records_rejected': self.metrics.records_rejected,
                    'quality_score_avg': self.metrics.quality_score_avg,
                    'last_updated': self.metrics.last_updated.isoformat()
                },
                'stage_metrics': {
                    stage.value: {
                        'records_processed': metrics.records_processed,
                        'records_successful': metrics.records_successful,
                        'records_failed': metrics.records_failed,
                        'avg_processing_time_ms': metrics.avg_processing_time_ms,
                        'last_updated': metrics.last_updated.isoformat()
                    }
                    for stage, metrics in self.stage_metrics.items()
                },
                'queue_sizes': {
                    stage.value: len(queue) for stage, queue in self.stage_queues.items()
                },
                'active_records': len(self.active_records),
                'completed_records': len(self.completed_records),
                'quarantined_records': len(self.quarantined_records),
                'validation_rules_count': sum(len(rules) for rules in self.validation_rules.values())
            }
    
    def get_record_status(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific record"""
        # Check active records
        if record_id in self.active_records:
            record = self.active_records[record_id]
            return {
                'record_id': record_id,
                'status': record.status.value,
                'current_stage': record.current_stage.value,
                'quality_score': record.quality_score,
                'validation_errors': record.validation_errors,
                'transformations_applied': record.transformations_applied,
                'ingested_at': record.ingested_at.isoformat()
            }
        
        # Check completed records
        for record in self.completed_records:
            if record.record_id == record_id:
                return {
                    'record_id': record_id,
                    'status': record.status.value,
                    'quality_score': record.quality_score,
                    'validation_errors': record.validation_errors,
                    'transformations_applied': record.transformations_applied,
                    'completed': True
                }
        
        return None
    
    def shutdown(self):
        """Gracefully shutdown pipeline"""
        self.logger.info("Shutting down Advanced Data Pipeline")
        self.is_running = False
        
        # Wait for threads to complete
        for thread in self.processor_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        if self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)
        
        self.logger.info("Advanced Data Pipeline shutdown complete")


# Global pipeline instance
advanced_data_pipeline = AdvancedDataPipeline()

# Export
__all__ = [
    'ValidationLevel', 'DataType', 'PipelineStage', 'ProcessingStatus',
    'ValidationRule', 'DataRecord', 'PipelineMetrics',
    'AdvancedDataPipeline', 'advanced_data_pipeline'
]