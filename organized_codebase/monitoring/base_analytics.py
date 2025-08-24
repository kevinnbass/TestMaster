"""
Base Analytics Framework

Provides the foundational classes and interfaces for all analytics components.
All analytics components inherit from these base classes to ensure consistency.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union


class AnalyticsStatus(Enum):
    """Status of analytics components."""
    INITIALIZING = "initializing"
    RUNNING = "running" 
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class MetricType(Enum):
    """Types of metrics that can be processed."""
    GAUGE = "gauge"           # Point-in-time value
    COUNTER = "counter"       # Cumulative value
    HISTOGRAM = "histogram"   # Distribution of values
    TIMER = "timer"          # Duration measurements
    EVENT = "event"          # Discrete events
    WORKFLOW = "workflow"     # Workflow-related metrics
    SEMANTIC = "semantic"     # Semantic analysis metrics
    COORDINATION = "coordination"  # Cross-system coordination metrics


class ProcessingStrategy(Enum):
    """Processing strategies for analytics components."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAMING = "streaming"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"


@dataclass
class AnalyticsConfig:
    """Configuration for analytics components."""
    component_name: str = "analytics"
    enabled: bool = True
    log_level: str = "INFO"
    cache_ttl_seconds: int = 300
    max_queue_size: int = 10000
    processing_interval_ms: int = 1000
    batch_size: int = 100
    retention_days: int = 30
    async_processing: bool = True
    error_threshold: int = 10
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing consolidation enhancements
    processing_strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    enable_workflow_analytics: bool = False
    enable_semantic_analytics: bool = False
    enable_coordination_analytics: bool = False
    consolidation_enabled: bool = True
    cross_system_integration: bool = False
    intelligent_routing: bool = False
    adaptive_batching: bool = True
    pattern_recognition: bool = False


@dataclass 
class MetricData:
    """Standard metric data structure."""
    name: str
    value: Union[float, int, str]
    metric_type: MetricType = MetricType.GAUGE
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def metric_id(self) -> str:
        """Generate unique metric ID."""
        return f"{self.name}_{int(self.timestamp.timestamp())}"


@dataclass
class AnalyticsResult:
    """Standard result structure for analytics operations."""
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: List[MetricData] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BaseAnalytics(ABC):
    """
    Abstract base class for all analytics components.
    
    Provides common functionality including:
    - Standardized configuration
    - Logging and monitoring
    - Error handling
    - Status management
    - Async processing support
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """
        Initialize base analytics component.
        
        Args:
            config: Component configuration
        """
        self.config = config or AnalyticsConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.config.component_name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Component state
        self.status = AnalyticsStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        
        # Processing metrics
        self.processed_count = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        
        # Queue for async processing
        if self.config.async_processing:
            self.processing_queue: asyncio.Queue = asyncio.Queue(
                maxsize=self.config.max_queue_size
            )
            self.processing_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"Initialized {self.config.component_name}")
    
    @abstractmethod
    async def process(self, data: Any) -> AnalyticsResult:
        """
        Process analytics data. Must be implemented by subclasses.
        
        Args:
            data: Data to process
            
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status. Must be implemented by subclasses.
        
        Returns:
            Status information
        """
        pass
    
    async def start(self):
        """Start the analytics component."""
        if self.status == AnalyticsStatus.RUNNING:
            self.logger.warning("Component already running")
            return
        
        self.logger.info(f"Starting {self.config.component_name}")
        self.start_time = datetime.now()
        self.status = AnalyticsStatus.RUNNING
        
        # Start async processing if enabled
        if self.config.async_processing:
            self.processing_task = asyncio.create_task(self._processing_loop())
        
        await self._on_start()
        self.logger.info(f"Started {self.config.component_name}")
    
    async def stop(self):
        """Stop the analytics component."""
        if self.status == AnalyticsStatus.STOPPED:
            self.logger.warning("Component already stopped")
            return
        
        self.logger.info(f"Stopping {self.config.component_name}")
        self.status = AnalyticsStatus.STOPPED
        
        # Stop async processing
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        await self._on_stop()
        self.logger.info(f"Stopped {self.config.component_name}")
    
    async def pause(self):
        """Pause the analytics component."""
        if self.status != AnalyticsStatus.RUNNING:
            self.logger.warning("Component not running, cannot pause")
            return
        
        self.logger.info(f"Pausing {self.config.component_name}")
        self.status = AnalyticsStatus.PAUSED
        await self._on_pause()
    
    async def resume(self):
        """Resume the analytics component."""
        if self.status != AnalyticsStatus.PAUSED:
            self.logger.warning("Component not paused, cannot resume")
            return
        
        self.logger.info(f"Resuming {self.config.component_name}")
        self.status = AnalyticsStatus.RUNNING
        await self._on_resume()
    
    async def _processing_loop(self):
        """Main async processing loop."""
        while self.status in [AnalyticsStatus.RUNNING, AnalyticsStatus.PAUSED]:
            try:
                if self.status == AnalyticsStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                # Get data from queue with timeout
                data = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=self.config.processing_interval_ms / 1000
                )
                
                # Process the data
                start_time = time.time()
                result = await self.process(data)
                processing_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self._update_processing_metrics(processing_time, result.success)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No data in queue, continue
                continue
            except Exception as e:
                self._handle_error(f"Processing loop error: {e}")
                await asyncio.sleep(1)
    
    def _update_processing_metrics(self, processing_time: float, success: bool):
        """Update internal processing metrics."""
        if success:
            self.processed_count += 1
            self.total_processing_time += processing_time
            self.avg_processing_time = self.total_processing_time / self.processed_count
    
    def _handle_error(self, error: str):
        """Handle errors consistently."""
        self.error_count += 1
        self.last_error = error
        self.logger.error(error)
        
        if self.error_count >= self.config.error_threshold:
            self.logger.critical(f"Error threshold exceeded: {self.error_count}")
            self.status = AnalyticsStatus.ERROR
    
    def get_base_status(self) -> Dict[str, Any]:
        """Get base status information."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'component': self.config.component_name,
            'status': self.status.value,
            'uptime_seconds': uptime,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'avg_processing_time_ms': self.avg_processing_time,
            'queue_size': self.processing_queue.qsize() if self.config.async_processing else 0,
            'configuration': {
                'enabled': self.config.enabled,
                'async_processing': self.config.async_processing,
                'batch_size': self.config.batch_size,
                'cache_ttl': self.config.cache_ttl_seconds,
                'processing_strategy': self.config.processing_strategy.value,
                'consolidation_enabled': self.config.consolidation_enabled,
                'workflow_analytics': self.config.enable_workflow_analytics,
                'semantic_analytics': self.config.enable_semantic_analytics,
                'coordination_analytics': self.config.enable_coordination_analytics
            }
        }
    
    # ========================================================================
    # PROCESSING CONSOLIDATION METHODS (Enhanced from orchestration analysis)
    # ========================================================================
    
    async def process_with_strategy(self, data: Any, strategy: ProcessingStrategy = None) -> AnalyticsResult:
        """Process data using specified strategy or configured default."""
        strategy = strategy or self.config.processing_strategy
        
        if strategy == ProcessingStrategy.SEQUENTIAL:
            return await self._process_sequential(data)
        elif strategy == ProcessingStrategy.PARALLEL:
            return await self._process_parallel(data)
        elif strategy == ProcessingStrategy.BATCH:
            return await self._process_batch(data)
        elif strategy == ProcessingStrategy.STREAMING:
            return await self._process_streaming(data)
        elif strategy == ProcessingStrategy.ADAPTIVE:
            return await self._process_adaptive(data)
        elif strategy == ProcessingStrategy.INTELLIGENT:
            return await self._process_intelligent(data)
        else:
            return await self.process(data)
    
    async def _process_sequential(self, data: Any) -> AnalyticsResult:
        """Sequential processing strategy."""
        return await self.process(data)
    
    async def _process_parallel(self, data: Any) -> AnalyticsResult:
        """Parallel processing strategy."""
        if isinstance(data, list):
            tasks = [self.process(item) for item in data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return self._consolidate_results(results)
        return await self.process(data)
    
    async def _process_batch(self, data: Any) -> AnalyticsResult:
        """Batch processing strategy."""
        if isinstance(data, list):
            batch_size = self.config.batch_size
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_result = await self._process_parallel(batch)
                results.append(batch_result)
            return self._consolidate_results(results)
        return await self.process(data)
    
    async def _process_streaming(self, data: Any) -> AnalyticsResult:
        """Streaming processing strategy."""
        # Implement streaming logic - process data as it arrives
        return await self.process(data)
    
    async def _process_adaptive(self, data: Any) -> AnalyticsResult:
        """Adaptive processing strategy - chooses best strategy based on data."""
        if isinstance(data, list) and len(data) > self.config.batch_size:
            return await self._process_batch(data)
        elif isinstance(data, list) and len(data) > 1:
            return await self._process_parallel(data)
        else:
            return await self._process_sequential(data)
    
    async def _process_intelligent(self, data: Any) -> AnalyticsResult:
        """Intelligent processing strategy - learns from patterns."""
        # Enhanced processing with pattern recognition and optimization
        if self.config.pattern_recognition:
            patterns = self._analyze_data_patterns(data)
            # Use patterns to optimize processing
        
        return await self._process_adaptive(data)
    
    def _consolidate_results(self, results: List[AnalyticsResult]) -> AnalyticsResult:
        """Consolidate multiple analytics results."""
        if not results:
            return AnalyticsResult(success=False, message="No results to consolidate")
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, AnalyticsResult)]
        if not valid_results:
            return AnalyticsResult(success=False, message="All processing failed")
        
        # Consolidate metrics and data
        consolidated_data = {}
        consolidated_metrics = []
        all_errors = []
        all_warnings = []
        
        success_count = sum(1 for r in valid_results if r.success)
        total_processing_time = sum(r.processing_time_ms for r in valid_results)
        
        for result in valid_results:
            consolidated_data.update(result.data)
            consolidated_metrics.extend(result.metrics)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return AnalyticsResult(
            success=success_count > 0,
            message=f"Consolidated {len(valid_results)} results, {success_count} successful",
            data=consolidated_data,
            metrics=consolidated_metrics,
            processing_time_ms=total_processing_time,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def _analyze_data_patterns(self, data: Any) -> Dict[str, Any]:
        """Analyze patterns in data for intelligent processing."""
        patterns = {
            'data_type': type(data).__name__,
            'size': len(data) if hasattr(data, '__len__') else 1,
            'complexity': 'simple'  # Could be enhanced with actual analysis
        }
        
        if isinstance(data, list):
            patterns['is_list'] = True
            patterns['list_length'] = len(data)
            if data:
                patterns['element_type'] = type(data[0]).__name__
        
        return patterns
    
    # Hook methods for subclasses
    async def _on_start(self):
        """Hook called when component starts."""
        pass
    
    async def _on_stop(self):
        """Hook called when component stops."""
        pass
    
    async def _on_pause(self):
        """Hook called when component pauses."""
        pass
    
    async def _on_resume(self):
        """Hook called when component resumes.""" 
        pass


class ProcessorMixin:
    """Mixin for data processing components."""
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        if data is None:
            return False
        
        if isinstance(data, MetricData):
            return bool(data.name and data.value is not None)
        
        return True
    
    def create_result(self, success: bool = True, message: str = "", 
                     data: Dict[str, Any] = None, 
                     processing_time: float = 0.0) -> AnalyticsResult:
        """Create standardized result."""
        return AnalyticsResult(
            success=success,
            message=message,
            data=data or {},
            processing_time_ms=processing_time
        )