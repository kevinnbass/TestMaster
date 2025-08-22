"""
Tracking Manager Module
=======================
Tracks operations and metrics for TestMaster components.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrackingManager:
    """Manages operation tracking and metrics."""
    
    def __init__(self):
        """Initialize tracking manager."""
        self.operations = []
        self.metrics = defaultdict(list)
        self.active_operations = {}
        logger.info("Tracking Manager initialized")
    
    def start_operation(self, operation_id: str, operation_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start tracking an operation."""
        operation = {
            'id': operation_id,
            'type': operation_type,
            'start_time': time.time(),
            'metadata': metadata or {},
            'status': 'running'
        }
        self.active_operations[operation_id] = operation
        return operation
    
    def end_operation(self, operation_id: str, status: str = 'completed', result: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """End tracking an operation."""
        if operation_id in self.active_operations:
            operation = self.active_operations.pop(operation_id)
            operation['end_time'] = time.time()
            operation['duration'] = operation['end_time'] - operation['start_time']
            operation['status'] = status
            operation['result'] = result
            self.operations.append(operation)
            
            # Track metrics
            self.metrics[operation['type']].append(operation['duration'])
            
            return operation
        return None
    
    def track_metric(self, metric_name: str, value: float) -> None:
        """Track a metric value."""
        self.metrics[metric_name].append(value)
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Get tracked metrics."""
        if metric_name:
            return {metric_name: self.metrics.get(metric_name, [])}
        return dict(self.metrics)
    
    def get_operations(self, operation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tracked operations."""
        if operation_type:
            return [op for op in self.operations if op['type'] == operation_type]
        return self.operations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tracking summary."""
        summary = {
            'total_operations': len(self.operations),
            'active_operations': len(self.active_operations),
            'metrics_tracked': len(self.metrics),
            'operation_types': {}
        }
        
        # Summarize by operation type
        for op in self.operations:
            op_type = op['type']
            if op_type not in summary['operation_types']:
                summary['operation_types'][op_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_duration': 0
                }
            summary['operation_types'][op_type]['count'] += 1
            summary['operation_types'][op_type]['total_duration'] += op.get('duration', 0)
        
        # Calculate averages
        for op_type, stats in summary['operation_types'].items():
            if stats['count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['count']
        
        return summary


# Global instance
_tracking_manager = None


def get_tracking_manager() -> TrackingManager:
    """Get the global tracking manager instance."""
    global _tracking_manager
    if _tracking_manager is None:
        _tracking_manager = TrackingManager()
    return _tracking_manager


# Convenience functions
def track_operation(operation_id: str, operation_type: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to track an operation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_tracking_manager()
            manager.start_operation(operation_id, operation_type, metadata)
            try:
                result = func(*args, **kwargs)
                manager.end_operation(operation_id, 'completed', result)
                return result
            except Exception as e:
                manager.end_operation(operation_id, 'failed', str(e))
                raise
        return wrapper
    return decorator


def track_metric(metric_name: str, value: float) -> None:
    """Track a metric value."""
    get_tracking_manager().track_metric(metric_name, value)
