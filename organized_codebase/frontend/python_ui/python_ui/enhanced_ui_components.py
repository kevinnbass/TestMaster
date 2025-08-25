#!/usr/bin/env python3
"""
🎨 ATOM: Enhanced UI Components
================================
Advanced UI components for enhanced dashboard functionality.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

class ComponentType(Enum):
    """Types of enhanced UI components"""
    CHART = "chart"
    GAUGE = "gauge"
    TABLE = "table"
    CARD = "card"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    NETWORK = "network"
    ALERT = "alert"

@dataclass
class ComponentConfig:
    """Component configuration"""
    id: str
    type: ComponentType
    title: str
    data_source: str
    refresh_interval: int = 5000
    interactive: bool = True
    customizable: bool = True
    export_enabled: bool = True

class EnhancedUIComponents:
    """Enhanced UI components for dashboards"""
    
    def __init__(self):
        self.components: Dict[str, ComponentConfig] = {}
        self.themes = self._initialize_themes()
        self.animations = self._initialize_animations()
        self.interaction_handlers: Dict[str, Callable] = {}
    
    def _initialize_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize component themes"""
        return {
            'light': {
                'background': '#ffffff',
                'text': '#2d3748',
                'primary': '#4299e1',
                'secondary': '#718096',
                'success': '#48bb78',
                'warning': '#ed8936',
                'danger': '#f56565'
            },
            'dark': {
                'background': '#1a202c',
                'text': '#f7fafc',
                'primary': '#63b3ed',
                'secondary': '#a0aec0',
                'success': '#68d391',
                'warning': '#f6ad55',
                'danger': '#fc8181'
            },
            'professional': {
                'background': '#f8f9fa',
                'text': '#212529',
                'primary': '#0066cc',
                'secondary': '#6c757d',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545'
            }
        }
    
    def _initialize_animations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize animation configurations"""
        return {
            'fade_in': {'duration': 300, 'easing': 'ease-in'},
            'slide_up': {'duration': 400, 'easing': 'ease-out'},
            'scale': {'duration': 200, 'easing': 'ease-in-out'},
            'rotate': {'duration': 500, 'easing': 'linear'},
            'pulse': {'duration': 1000, 'easing': 'ease-in-out', 'repeat': True}
        }
    
    def create_performance_gauge(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create performance gauge component"""
        return {
            'component': 'performance_gauge',
            'type': ComponentType.GAUGE.value,
            'config': {
                'style': 'radial',
                'segments': [
                    {'range': [0, 30], 'color': 'danger', 'label': 'Poor'},
                    {'range': [30, 70], 'color': 'warning', 'label': 'Fair'},
                    {'range': [70, 90], 'color': 'success', 'label': 'Good'},
                    {'range': [90, 100], 'color': 'primary', 'label': 'Excellent'}
                ],
                'value': metrics.get('performance_score', 0),
                'label': 'System Performance',
                'animation': self.animations['rotate'],
                'indicators': {
                    'needle': True,
                    'value_display': True,
                    'trend_arrow': True
                }
            }
        }
    
    def create_metrics_heatmap(self, data: List[List[float]], labels: Dict[str, List[str]]) -> Dict[str, Any]:
        """Create metrics heatmap component"""
        return {
            'component': 'metrics_heatmap',
            'type': ComponentType.HEATMAP.value,
            'config': {
                'data': data,
                'x_labels': labels.get('x', []),
                'y_labels': labels.get('y', []),
                'color_scale': {
                    'min': min(min(row) for row in data) if data else 0,
                    'max': max(max(row) for row in data) if data else 100,
                    'palette': 'viridis',
                    'reverse': False
                },
                'cell_config': {
                    'show_values': True,
                    'format': '.1f',
                    'hover_effect': True,
                    'click_action': 'show_details'
                },
                'legend': {
                    'show': True,
                    'position': 'right',
                    'title': 'Value Scale'
                }
            }
        }
    
    def create_alert_banner(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create alert banner component"""
        prioritized_alerts = sorted(alerts, key=lambda x: x.get('priority', 0), reverse=True)
        
        return {
            'component': 'alert_banner',
            'type': ComponentType.ALERT.value,
            'config': {
                'alerts': [
                    {
                        'id': alert.get('id', f"alert_{i}"),
                        'type': alert.get('type', 'info'),
                        'title': alert.get('title', 'Alert'),
                        'message': alert.get('message', ''),
                        'timestamp': alert.get('timestamp', datetime.utcnow().isoformat()),
                        'dismissible': alert.get('dismissible', True),
                        'actions': alert.get('actions', [])
                    }
                    for i, alert in enumerate(prioritized_alerts[:5])
                ],
                'display_mode': 'stack',  # stack, carousel, or single
                'animation': self.animations['slide_up'],
                'auto_dismiss': {
                    'enabled': True,
                    'duration': 10000  # 10 seconds
                }
            }
        }
    
    def create_interactive_timeline(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create interactive timeline component"""
        return {
            'component': 'interactive_timeline',
            'type': ComponentType.TIMELINE.value,
            'config': {
                'events': [
                    {
                        'id': event.get('id'),
                        'timestamp': event.get('timestamp'),
                        'title': event.get('title'),
                        'description': event.get('description'),
                        'category': event.get('category'),
                        'icon': self._get_event_icon(event.get('category')),
                        'color': self._get_event_color(event.get('category'))
                    }
                    for event in events
                ],
                'layout': 'vertical',  # vertical or horizontal
                'grouping': 'category',  # category, time, or none
                'zoom': {
                    'enabled': True,
                    'min_scale': '1hour',
                    'max_scale': '1year'
                },
                'filters': {
                    'categories': True,
                    'date_range': True,
                    'search': True
                },
                'interaction': {
                    'hover_preview': True,
                    'click_expand': True,
                    'drag_scroll': True
                }
            }
        }
    
    def create_network_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Create network graph component"""
        return {
            'component': 'network_graph',
            'type': ComponentType.NETWORK.value,
            'config': {
                'nodes': nodes,
                'edges': edges,
                'layout': {
                    'type': 'force-directed',
                    'physics': {
                        'charge': -300,
                        'link_distance': 100,
                        'gravity': 0.05
                    }
                },
                'styling': {
                    'node_size': 'dynamic',  # Based on connections
                    'edge_width': 'weighted',
                    'color_scheme': 'category',
                    'labels': {
                        'show': True,
                        'font_size': 12
                    }
                },
                'interaction': {
                    'zoom': True,
                    'pan': True,
                    'node_click': 'show_details',
                    'edge_click': 'highlight_path',
                    'hover': 'show_tooltip'
                },
                'controls': {
                    'zoom_controls': True,
                    'layout_selector': True,
                    'filter_panel': True
                }
            }
        }
    
    def create_data_table(self, data: List[Dict], columns: List[Dict]) -> Dict[str, Any]:
        """Create enhanced data table component"""
        return {
            'component': 'data_table',
            'type': ComponentType.TABLE.value,
            'config': {
                'data': data,
                'columns': columns,
                'features': {
                    'sorting': True,
                    'filtering': True,
                    'pagination': True,
                    'row_selection': True,
                    'column_resize': True,
                    'export': ['csv', 'json', 'excel']
                },
                'styling': {
                    'striped': True,
                    'hover_highlight': True,
                    'compact': False,
                    'bordered': True
                },
                'pagination_config': {
                    'page_size': 25,
                    'page_size_options': [10, 25, 50, 100],
                    'show_total': True
                },
                'row_actions': ['view', 'edit', 'delete'],
                'bulk_actions': ['export', 'delete_selected']
            }
        }
    
    def create_metric_card(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Create metric card component"""
        return {
            'component': 'metric_card',
            'type': ComponentType.CARD.value,
            'config': {
                'title': metric.get('title'),
                'value': metric.get('value'),
                'unit': metric.get('unit', ''),
                'icon': metric.get('icon'),
                'trend': {
                    'direction': metric.get('trend_direction', 'stable'),
                    'value': metric.get('trend_value', 0),
                    'percentage': metric.get('trend_percentage', 0)
                },
                'sparkline': metric.get('sparkline_data', []),
                'status': {
                    'level': self._calculate_status_level(metric),
                    'color': self._get_status_color(metric)
                },
                'footer': {
                    'text': metric.get('footer_text', ''),
                    'link': metric.get('footer_link', '')
                },
                'animation': self.animations['fade_in']
            }
        }
    
    def _get_event_icon(self, category: str) -> str:
        """Get icon for event category"""
        icons = {
            'system': '⚙️',
            'user': '👤',
            'alert': '🚨',
            'performance': '📊',
            'security': '🔒',
            'update': '🔄'
        }
        return icons.get(category, '📌')
    
    def _get_event_color(self, category: str) -> str:
        """Get color for event category"""
        colors = {
            'system': 'primary',
            'user': 'info',
            'alert': 'danger',
            'performance': 'warning',
            'security': 'danger',
            'update': 'success'
        }
        return colors.get(category, 'secondary')
    
    def _calculate_status_level(self, metric: Dict[str, Any]) -> str:
        """Calculate status level for metric"""
        value = metric.get('value', 0)
        thresholds = metric.get('thresholds', {})
        
        if value >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif value >= thresholds.get('warning', float('inf')):
            return 'warning'
        else:
            return 'normal'
    
    def _get_status_color(self, metric: Dict[str, Any]) -> str:
        """Get status color for metric"""
        level = self._calculate_status_level(metric)
        colors = {
            'critical': 'danger',
            'warning': 'warning',
            'normal': 'success'
        }
        return colors.get(level, 'secondary')
    
    def register_component(self, config: ComponentConfig):
        """Register a new component"""
        self.components[config.id] = config
    
    def register_interaction_handler(self, component_id: str, handler: Callable):
        """Register interaction handler for component"""
        self.interaction_handlers[component_id] = handler
    
    def apply_theme(self, theme_name: str) -> Dict[str, Any]:
        """Apply theme to components"""
        if theme_name in self.themes:
            return self.themes[theme_name]
        return self.themes['light']