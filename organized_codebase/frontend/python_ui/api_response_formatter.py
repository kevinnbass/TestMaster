#!/usr/bin/env python3
"""
API Response Formatter - Atomic Component
Frontend data formatting and transformation
Agent Z - STEELCLAD Frontend Atomization
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class FormatType(Enum):
    """Response format types"""
    JSON = "json"
    TABLE = "table"
    CHART = "chart"
    METRIC = "metric"
    LIST = "list"
    GRID = "grid"
    TIMELINE = "timeline"


class DataTransform(Enum):
    """Data transformation types"""
    AGGREGATE = "aggregate"
    FILTER = "filter"
    SORT = "sort"
    GROUP = "group"
    PIVOT = "pivot"
    NORMALIZE = "normalize"


@dataclass
class FormattedResponse:
    """Formatted response structure"""
    format_type: FormatType
    data: Any
    metadata: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'format': self.format_type.value,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class APIResponseFormatter:
    """
    API response formatting component for frontend
    Transforms and formats data for dashboard consumption
    """
    
    def __init__(self):
        self.format_templates: Dict[FormatType, Dict[str, Any]] = {}
        self.transform_functions: Dict[DataTransform, callable] = {}
        
        # Performance metrics
        self.metrics = {
            'formats_applied': 0,
            'transforms_applied': 0,
            'avg_format_time': 0.0,
            'format_errors': 0
        }
        
        # Initialize formatters and transformers
        self._initialize_formatters()
        self._initialize_transformers()
    
    def _initialize_formatters(self):
        """Initialize format templates"""
        self.format_templates = {
            FormatType.JSON: {'type': 'json', 'indent': 2},
            FormatType.TABLE: {'type': 'table', 'headers': [], 'rows': []},
            FormatType.CHART: {'type': 'chart', 'labels': [], 'datasets': []},
            FormatType.METRIC: {'type': 'metric', 'value': None, 'unit': ''},
            FormatType.LIST: {'type': 'list', 'items': []},
            FormatType.GRID: {'type': 'grid', 'columns': 3, 'items': []},
            FormatType.TIMELINE: {'type': 'timeline', 'events': []}
        }
    
    def _initialize_transformers(self):
        """Initialize data transformers"""
        self.transform_functions = {
            DataTransform.AGGREGATE: self._transform_aggregate,
            DataTransform.FILTER: self._transform_filter,
            DataTransform.SORT: self._transform_sort,
            DataTransform.GROUP: self._transform_group,
            DataTransform.PIVOT: self._transform_pivot,
            DataTransform.NORMALIZE: self._transform_normalize
        }
    
    def format_for_frontend(self, data: Any, format_type: FormatType,
                           options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format data for frontend consumption
        Main interface for response formatting
        """
        import time
        start_time = time.time()
        
        try:
            # Apply formatting based on type
            if format_type == FormatType.JSON:
                formatted = self._format_json(data, options)
            elif format_type == FormatType.TABLE:
                formatted = self._format_table(data, options)
            elif format_type == FormatType.CHART:
                formatted = self._format_chart(data, options)
            elif format_type == FormatType.METRIC:
                formatted = self._format_metric(data, options)
            elif format_type == FormatType.LIST:
                formatted = self._format_list(data, options)
            elif format_type == FormatType.GRID:
                formatted = self._format_grid(data, options)
            else:
                formatted = self._format_timeline(data, options)
            
            # Create response
            response = FormattedResponse(
                format_type=format_type,
                data=formatted,
                metadata=self._generate_metadata(data, format_type),
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics
            format_time = time.time() - start_time
            self.metrics['formats_applied'] += 1
            self.metrics['avg_format_time'] = (
                (self.metrics['avg_format_time'] * 0.9) + (format_time * 0.1)
            )
            
            return response.to_dict()
            
        except Exception as e:
            self.metrics['format_errors'] += 1
            return {
                'error': str(e),
                'format_type': format_type.value,
                'raw_data': data
            }
    
    def _format_json(self, data: Any, options: Dict[str, Any] = None) -> Any:
        """Format data as JSON"""
        options = options or {}
        indent = options.get('indent', 2)
        
        if isinstance(data, (dict, list)):
            return data
        elif hasattr(data, '__dict__'):
            return data.__dict__
        else:
            return {'value': data}
    
    def _format_table(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data as table"""
        options = options or {}
        
        if isinstance(data, list) and data:
            # Extract headers from first item
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [list(item.values()) for item in data]
            else:
                headers = ['Value']
                rows = [[item] for item in data]
        elif isinstance(data, dict):
            headers = ['Key', 'Value']
            rows = [[k, v] for k, v in data.items()]
        else:
            headers = ['Value']
            rows = [[data]]
        
        return {
            'headers': headers,
            'rows': rows,
            'total_rows': len(rows),
            'sortable': options.get('sortable', True),
            'filterable': options.get('filterable', True)
        }
    
    def _format_chart(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data for chart display"""
        options = options or {}
        chart_type = options.get('chart_type', 'line')
        
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
        elif isinstance(data, list):
            labels = [f"Item {i+1}" for i in range(len(data))]
            values = data
        else:
            labels = ['Value']
            values = [data]
        
        return {
            'type': chart_type,
            'labels': labels,
            'datasets': [{
                'label': options.get('label', 'Data'),
                'data': values,
                'backgroundColor': options.get('backgroundColor', 'rgba(75, 192, 192, 0.2)'),
                'borderColor': options.get('borderColor', 'rgba(75, 192, 192, 1)'),
                'borderWidth': 1
            }],
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'animation': {'duration': 500}
            }
        }
    
    def _format_metric(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data as metric display"""
        options = options or {}
        
        # Extract value
        if isinstance(data, dict):
            value = data.get('value', 0)
        elif isinstance(data, (int, float)):
            value = data
        else:
            value = len(data) if hasattr(data, '__len__') else 0
        
        return {
            'value': value,
            'label': options.get('label', 'Metric'),
            'unit': options.get('unit', ''),
            'trend': options.get('trend', 'stable'),
            'change': options.get('change', 0),
            'color': self._get_metric_color(value, options)
        }
    
    def _format_list(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data as list"""
        options = options or {}
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [{'key': k, 'value': v} for k, v in data.items()]
        else:
            items = [data]
        
        return {
            'items': items[:options.get('limit', 100)],
            'total_items': len(items),
            'has_more': len(items) > options.get('limit', 100),
            'item_template': options.get('template', 'default')
        }
    
    def _format_grid(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data as grid layout"""
        options = options or {}
        columns = options.get('columns', 3)
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [{'id': k, 'content': v} for k, v in data.items()]
        else:
            items = [data]
        
        # Organize into grid
        grid_rows = []
        for i in range(0, len(items), columns):
            grid_rows.append(items[i:i+columns])
        
        return {
            'columns': columns,
            'rows': grid_rows,
            'total_items': len(items),
            'responsive': options.get('responsive', True)
        }
    
    def _format_timeline(self, data: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format data as timeline"""
        options = options or {}
        
        events = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    events.append({
                        'timestamp': item.get('timestamp', datetime.now().isoformat()),
                        'title': item.get('title', f"Event {i+1}"),
                        'description': item.get('description', ''),
                        'type': item.get('type', 'default')
                    })
                else:
                    events.append({
                        'timestamp': datetime.now().isoformat(),
                        'title': f"Event {i+1}",
                        'description': str(item),
                        'type': 'default'
                    })
        
        return {
            'events': events,
            'orientation': options.get('orientation', 'vertical'),
            'show_dates': options.get('show_dates', True)
        }
    
    def transform_data(self, data: Any, transform: DataTransform,
                       params: Dict[str, Any] = None) -> Any:
        """Apply data transformation"""
        params = params or {}
        
        if transform in self.transform_functions:
            transformed = self.transform_functions[transform](data, params)
            self.metrics['transforms_applied'] += 1
            return transformed
        
        return data
    
    def _transform_aggregate(self, data: Any, params: Dict[str, Any]) -> Any:
        """Aggregate data"""
        operation = params.get('operation', 'sum')
        
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            if operation == 'sum':
                return sum(data)
            elif operation == 'avg':
                return sum(data) / len(data) if data else 0
            elif operation == 'max':
                return max(data) if data else 0
            elif operation == 'min':
                return min(data) if data else 0
        
        return data
    
    def _transform_filter(self, data: Any, params: Dict[str, Any]) -> Any:
        """Filter data"""
        filters = params.get('filters', {})
        
        if isinstance(data, list):
            filtered = []
            for item in data:
                if isinstance(item, dict):
                    match = all(
                        item.get(key) == value
                        for key, value in filters.items()
                    )
                    if match:
                        filtered.append(item)
                else:
                    filtered.append(item)
            return filtered
        
        return data
    
    def _transform_sort(self, data: Any, params: Dict[str, Any]) -> Any:
        """Sort data"""
        sort_key = params.get('key', None)
        reverse = params.get('reverse', False)
        
        if isinstance(data, list):
            if sort_key and all(isinstance(x, dict) for x in data):
                return sorted(data, key=lambda x: x.get(sort_key, ''), reverse=reverse)
            else:
                return sorted(data, reverse=reverse)
        
        return data
    
    def _transform_group(self, data: Any, params: Dict[str, Any]) -> Any:
        """Group data"""
        group_key = params.get('key', None)
        
        if isinstance(data, list) and group_key:
            grouped = {}
            for item in data:
                if isinstance(item, dict):
                    key_value = item.get(group_key, 'unknown')
                    if key_value not in grouped:
                        grouped[key_value] = []
                    grouped[key_value].append(item)
            return grouped
        
        return data
    
    def _transform_pivot(self, data: Any, params: Dict[str, Any]) -> Any:
        """Pivot data"""
        # Simple pivot implementation
        return data
    
    def _transform_normalize(self, data: Any, params: Dict[str, Any]) -> Any:
        """Normalize data"""
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            max_val = max(data) if data else 1
            return [x / max_val for x in data] if max_val != 0 else data
        
        return data
    
    def _generate_metadata(self, data: Any, format_type: FormatType) -> Dict[str, Any]:
        """Generate metadata for formatted response"""
        metadata = {
            'original_type': type(data).__name__,
            'format_applied': format_type.value,
            'formatted_at': datetime.now().isoformat()
        }
        
        if hasattr(data, '__len__'):
            metadata['original_size'] = len(data)
        
        return metadata
    
    def _get_metric_color(self, value: float, options: Dict[str, Any]) -> str:
        """Get color for metric based on value and thresholds"""
        thresholds = options.get('thresholds', {})
        
        if 'danger' in thresholds and value >= thresholds['danger']:
            return '#dc3545'
        elif 'warning' in thresholds and value >= thresholds['warning']:
            return '#ffc107'
        else:
            return '#28a745'
    
    def batch_format(self, data_items: List[Tuple[Any, FormatType, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Batch format multiple data items"""
        results = []
        
        for data, format_type, options in data_items:
            results.append(self.format_for_frontend(data, format_type, options))
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get formatter metrics"""
        return {
            **self.metrics,
            'available_formats': len(self.format_templates),
            'available_transforms': len(self.transform_functions),
            'latency_target_met': self.metrics['avg_format_time'] * 1000 < 50
        }