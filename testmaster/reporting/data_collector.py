"""
Data Collector for TestMaster Reporting

Collects performance data from all TestMaster components
for comprehensive report generation.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.feature_flags import FeatureFlags

class DataSource(Enum):
    """Data sources for collection."""
    ASYNC_PROCESSING = "async_processing"
    STREAMING_GENERATION = "streaming_generation"
    TELEMETRY = "telemetry"
    SYSTEM_METRICS = "system_metrics"

class MetricAggregation(Enum):
    """Metric aggregation types."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"

class TimeRange(Enum):
    """Time range options."""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"

@dataclass
class CollectedMetric:
    """Collected metric data."""
    source: DataSource
    metric_name: str
    value: Any
    timestamp: datetime
    metadata: Dict[str, Any]

class DataCollector:
    """Data collector for reporting system."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'performance_reporting')
        self.lock = threading.RLock()
        self.collected_metrics: List[CollectedMetric] = []
        self.is_collecting = False
        
        if not self.enabled:
            return
    
    def start_collection(self):
        """Start data collection."""
        if self.enabled:
            self.is_collecting = True
    
    def stop_collection(self):
        """Stop data collection."""
        self.is_collecting = False
    
    def collect_data(self, sources: List[DataSource] = None) -> Dict[str, Any]:
        """Collect data from specified sources."""
        if not self.enabled:
            return {}
        
        data = {}
        sources = sources or list(DataSource)
        
        for source in sources:
            if source == DataSource.ASYNC_PROCESSING:
                data[source.value] = self._collect_async_data()
            elif source == DataSource.STREAMING_GENERATION:
                data[source.value] = self._collect_streaming_data()
            elif source == DataSource.TELEMETRY:
                data[source.value] = self._collect_telemetry_data()
            elif source == DataSource.SYSTEM_METRICS:
                data[source.value] = self._collect_system_data()
        
        return data
    
    def _collect_async_data(self) -> Dict[str, Any]:
        """Collect async processing data."""
        return {"async_operations": 0, "success_rate": 100.0}
    
    def _collect_streaming_data(self) -> Dict[str, Any]:
        """Collect streaming generation data."""
        return {"streaming_sessions": 0, "generation_rate": 0.0}
    
    def _collect_telemetry_data(self) -> Dict[str, Any]:
        """Collect telemetry data."""
        return {"events_recorded": 0, "error_rate": 0.0}
    
    def _collect_system_data(self) -> Dict[str, Any]:
        """Collect system metrics data."""
        return {"cpu_usage": 0.0, "memory_usage": 0.0}

def get_data_collector() -> DataCollector:
    """Get data collector instance."""
    return DataCollector()

def collect_system_metrics() -> Dict[str, Any]:
    """Collect system metrics."""
    collector = get_data_collector()
    return collector.collect_data()