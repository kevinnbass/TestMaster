#!/usr/bin/env python3
"""
📊 ATOM: Dashboard Data Display Component
=========================================
Data display and visualization for dashboard.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

class DisplayMode(Enum):
    """Display modes for data"""
    TABLE = "table"
    CARDS = "cards"
    LIST = "list"
    GRID = "grid"
    TIMELINE = "timeline"
    CHART = "chart"

@dataclass
class DataDisplayConfig:
    """Data display configuration"""
    mode: DisplayMode
    columns: List[str]
    sorting: Dict[str, str]
    filtering: Dict[str, Any]
    pagination: Dict[str, int]
    grouping: Optional[str] = None
    aggregation: Optional[Dict[str, str]] = None

class DashboardDataDisplay:
    """Dashboard data display component"""
    
    def __init__(self):
        self.display_configs: Dict[str, DataDisplayConfig] = {}
        self.data_cache = {}
        self.display_preferences = self._initialize_preferences()
        self.formatters = self._initialize_formatters()
    
    def _initialize_preferences(self) -> Dict[str, Any]:
        """Initialize display preferences"""
        return {
            'default_mode': DisplayMode.TABLE,
            'items_per_page': 25,
            'enable_sorting': True,
            'enable_filtering': True,
            'enable_export': True,
            'date_format': 'YYYY-MM-DD HH:mm:ss',
            'number_format': '0,0.00',
            'highlight_changes': True,
            'compact_view': False
        }
    
    def _initialize_formatters(self) -> Dict[str, Any]:
        """Initialize data formatters"""
        return {
            'datetime': self._format_datetime,
            'number': self._format_number,
            'percentage': self._format_percentage,
            'currency': self._format_currency,
            'status': self._format_status,
            'boolean': self._format_boolean,
            'duration': self._format_duration
        }
    
    def render_data_display(self, data: List[Dict[str, Any]], config: DataDisplayConfig = None) -> Dict[str, Any]:
        """
        Render data display component
        
        Args:
            data: Data to display
            config: Display configuration
            
        Returns:
            Data display UI configuration
        """
        config = config or self._get_default_config()
        
        # Apply transformations
        processed_data = self._process_data(data, config)
        
        # Render based on display mode
        if config.mode == DisplayMode.TABLE:
            return self._render_table_display(processed_data, config)
        elif config.mode == DisplayMode.CARDS:
            return self._render_cards_display(processed_data, config)
        elif config.mode == DisplayMode.GRID:
            return self._render_grid_display(processed_data, config)
        elif config.mode == DisplayMode.TIMELINE:
            return self._render_timeline_display(processed_data, config)
        elif config.mode == DisplayMode.CHART:
            return self._render_chart_display(processed_data, config)
        else:
            return self._render_list_display(processed_data, config)
    
    def _get_default_config(self) -> DataDisplayConfig:
        """Get default display configuration"""
        return DataDisplayConfig(
            mode=self.display_preferences['default_mode'],
            columns=[],
            sorting={'field': 'timestamp', 'order': 'desc'},
            filtering={},
            pagination={'page': 1, 'size': self.display_preferences['items_per_page']}
        )
    
    def _process_data(self, data: List[Dict[str, Any]], config: DataDisplayConfig) -> List[Dict[str, Any]]:
        """Process data based on configuration"""
        processed = data.copy()
        
        # Apply filtering
        if config.filtering:
            processed = self._apply_filters(processed, config.filtering)
        
        # Apply grouping
        if config.grouping:
            processed = self._apply_grouping(processed, config.grouping)
        
        # Apply sorting
        if config.sorting:
            processed = self._apply_sorting(processed, config.sorting)
        
        # Apply pagination
        if config.pagination:
            processed = self._apply_pagination(processed, config.pagination)
        
        # Apply formatting
        processed = self._apply_formatting(processed)
        
        return processed
    
    def _render_table_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render table display"""
        columns = config.columns or self._infer_columns(data)
        
        return {
            'display_type': 'table',
            'config': {
                'columns': [
                    {
                        'field': col,
                        'header': self._humanize_column_name(col),
                        'sortable': self.display_preferences['enable_sorting'],
                        'filterable': self.display_preferences['enable_filtering'],
                        'resizable': True,
                        'formatter': self._get_column_formatter(col)
                    }
                    for col in columns
                ],
                'data': data,
                'features': {
                    'row_selection': True,
                    'column_reorder': True,
                    'export': self.display_preferences['enable_export'],
                    'search': True,
                    'pagination': {
                        'enabled': True,
                        'page_size': config.pagination['size'],
                        'current_page': config.pagination['page']
                    }
                },
                'styling': {
                    'striped': True,
                    'hover': True,
                    'compact': self.display_preferences['compact_view'],
                    'bordered': True
                }
            }
        }
    
    def _render_cards_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render cards display"""
        return {
            'display_type': 'cards',
            'config': {
                'cards': [
                    self._create_card(item, config) 
                    for item in data
                ],
                'layout': {
                    'columns': 3,
                    'gap': '20px',
                    'responsive': True
                },
                'features': {
                    'flip_animation': True,
                    'hover_effect': True,
                    'click_action': 'expand'
                }
            }
        }
    
    def _render_grid_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render grid display"""
        return {
            'display_type': 'grid',
            'config': {
                'items': data,
                'grid': {
                    'columns': 'auto-fill',
                    'min_width': '200px',
                    'gap': '15px'
                },
                'item_template': self._get_grid_item_template(),
                'features': {
                    'lazy_loading': True,
                    'infinite_scroll': len(data) > 50,
                    'item_animation': 'fade-in'
                }
            }
        }
    
    def _render_timeline_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render timeline display"""
        timeline_data = self._prepare_timeline_data(data)
        
        return {
            'display_type': 'timeline',
            'config': {
                'events': timeline_data,
                'orientation': 'vertical',
                'features': {
                    'zoom': True,
                    'filter_by_date': True,
                    'group_by_period': True,
                    'show_connections': True
                },
                'styling': {
                    'line_style': 'solid',
                    'marker_size': 'medium',
                    'color_by_category': True
                }
            }
        }
    
    def _render_chart_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render chart display"""
        chart_data = self._prepare_chart_data(data, config)
        
        return {
            'display_type': 'chart',
            'config': {
                'type': self._infer_chart_type(data),
                'data': chart_data,
                'options': {
                    'responsive': True,
                    'legend': {'position': 'bottom'},
                    'tooltips': {'enabled': True},
                    'animation': {'duration': 1000}
                },
                'interactions': {
                    'zoom': True,
                    'pan': True,
                    'export': True
                }
            }
        }
    
    def _render_list_display(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Render list display"""
        return {
            'display_type': 'list',
            'config': {
                'items': [
                    self._create_list_item(item)
                    for item in data
                ],
                'features': {
                    'collapsible': True,
                    'searchable': True,
                    'sortable': True,
                    'grouping': config.grouping is not None
                },
                'styling': {
                    'show_icons': True,
                    'show_badges': True,
                    'alternating_colors': True
                }
            }
        }
    
    def _apply_filters(self, data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply filters to data"""
        filtered = data
        
        for field, filter_value in filters.items():
            if isinstance(filter_value, dict):
                # Complex filter (e.g., range, contains)
                filtered = self._apply_complex_filter(filtered, field, filter_value)
            else:
                # Simple equality filter
                filtered = [item for item in filtered if item.get(field) == filter_value]
        
        return filtered
    
    def _apply_complex_filter(self, data: List[Dict], field: str, filter_spec: Dict) -> List[Dict]:
        """Apply complex filter to data"""
        filtered = data
        
        if 'min' in filter_spec:
            filtered = [item for item in filtered if item.get(field, 0) >= filter_spec['min']]
        if 'max' in filter_spec:
            filtered = [item for item in filtered if item.get(field, 0) <= filter_spec['max']]
        if 'contains' in filter_spec:
            filtered = [item for item in filtered if filter_spec['contains'] in str(item.get(field, ''))]
        if 'in' in filter_spec:
            filtered = [item for item in filtered if item.get(field) in filter_spec['in']]
        
        return filtered
    
    def _apply_grouping(self, data: List[Dict], group_by: str) -> List[Dict]:
        """Apply grouping to data"""
        groups = {}
        
        for item in data:
            group_key = item.get(group_by, 'Unknown')
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        
        # Return flattened grouped data with group headers
        grouped_data = []
        for group_key, items in groups.items():
            grouped_data.append({
                '_is_group_header': True,
                '_group_key': group_key,
                '_group_count': len(items)
            })
            grouped_data.extend(items)
        
        return grouped_data
    
    def _apply_sorting(self, data: List[Dict], sorting: Dict[str, str]) -> List[Dict]:
        """Apply sorting to data"""
        field = sorting.get('field')
        order = sorting.get('order', 'asc')
        
        if not field:
            return data
        
        return sorted(
            data,
            key=lambda x: x.get(field, ''),
            reverse=(order == 'desc')
        )
    
    def _apply_pagination(self, data: List[Dict], pagination: Dict[str, int]) -> List[Dict]:
        """Apply pagination to data"""
        page = pagination.get('page', 1)
        size = pagination.get('size', 25)
        
        start = (page - 1) * size
        end = start + size
        
        return data[start:end]
    
    def _apply_formatting(self, data: List[Dict]) -> List[Dict]:
        """Apply formatting to data"""
        formatted = []
        
        for item in data:
            formatted_item = {}
            for key, value in item.items():
                formatter = self._get_value_formatter(key, value)
                formatted_item[key] = formatter(value) if formatter else value
            formatted.append(formatted_item)
        
        return formatted
    
    def _infer_columns(self, data: List[Dict]) -> List[str]:
        """Infer columns from data"""
        if not data:
            return []
        
        columns = set()
        for item in data[:10]:  # Sample first 10 items
            columns.update(item.keys())
        
        # Remove internal fields
        columns = [col for col in columns if not col.startswith('_')]
        
        return sorted(columns)
    
    def _humanize_column_name(self, column: str) -> str:
        """Convert column name to human-readable format"""
        return column.replace('_', ' ').title()
    
    def _get_column_formatter(self, column: str) -> Optional[str]:
        """Get formatter for column"""
        if 'date' in column or 'time' in column:
            return 'datetime'
        elif 'percent' in column:
            return 'percentage'
        elif 'price' in column or 'cost' in column:
            return 'currency'
        elif 'status' in column:
            return 'status'
        return None
    
    def _get_value_formatter(self, key: str, value: Any) -> Optional[Any]:
        """Get formatter for value"""
        formatter_type = self._get_column_formatter(key)
        if formatter_type and formatter_type in self.formatters:
            return self.formatters[formatter_type]
        return None
    
    def _format_datetime(self, value: Any) -> str:
        """Format datetime value"""
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        return str(value)
    
    def _format_number(self, value: Any) -> str:
        """Format number value"""
        if isinstance(value, (int, float)):
            return f"{value:,.2f}"
        return str(value)
    
    def _format_percentage(self, value: Any) -> str:
        """Format percentage value"""
        if isinstance(value, (int, float)):
            return f"{value * 100:.1f}%"
        return str(value)
    
    def _format_currency(self, value: Any) -> str:
        """Format currency value"""
        if isinstance(value, (int, float)):
            return f"${value:,.2f}"
        return str(value)
    
    def _format_status(self, value: Any) -> Dict[str, str]:
        """Format status value"""
        status_map = {
            'active': {'text': 'Active', 'color': 'success'},
            'inactive': {'text': 'Inactive', 'color': 'secondary'},
            'error': {'text': 'Error', 'color': 'danger'},
            'warning': {'text': 'Warning', 'color': 'warning'}
        }
        return status_map.get(str(value).lower(), {'text': str(value), 'color': 'info'})
    
    def _format_boolean(self, value: Any) -> str:
        """Format boolean value"""
        return '✓' if value else '✗'
    
    def _format_duration(self, value: Any) -> str:
        """Format duration value"""
        if isinstance(value, timedelta):
            return str(value)
        elif isinstance(value, (int, float)):
            return f"{value}s"
        return str(value)
    
    # Helper methods for specific display types
    def _create_card(self, item: Dict, config: DataDisplayConfig) -> Dict[str, Any]:
        """Create card from data item"""
        return {
            'title': item.get('name', item.get('id', 'Item')),
            'content': item,
            'actions': ['view', 'edit']
        }
    
    def _get_grid_item_template(self) -> str:
        """Get grid item template"""
        return "grid-item-default"
    
    def _prepare_timeline_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for timeline display"""
        return [
            {
                'timestamp': item.get('timestamp', datetime.utcnow()),
                'title': item.get('title', 'Event'),
                'description': item.get('description', '')
            }
            for item in data
        ]
    
    def _prepare_chart_data(self, data: List[Dict], config: DataDisplayConfig) -> Dict[str, Any]:
        """Prepare data for chart display"""
        return {
            'labels': [item.get('label', '') for item in data],
            'datasets': [{
                'data': [item.get('value', 0) for item in data]
            }]
        }
    
    def _infer_chart_type(self, data: List[Dict]) -> str:
        """Infer appropriate chart type from data"""
        if len(data) < 5:
            return 'bar'
        elif any('timestamp' in item for item in data):
            return 'line'
        else:
            return 'pie'
    
    def _create_list_item(self, item: Dict) -> Dict[str, Any]:
        """Create list item from data"""
        return {
            'primary': item.get('title', item.get('name', 'Item')),
            'secondary': item.get('description', ''),
            'metadata': item
        }