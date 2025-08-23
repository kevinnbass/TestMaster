"""
Advanced Visualization Engine - EPSILON ENHANCEMENT Hour 5
===========================================================

Advanced visualization system with AI-powered chart selection, interactive drill-down 
capabilities, and context-aware adaptations - extracted from monolithic dashboard 
as part of STEELCLAD modularization protocol.

Created: 2025-08-23 20:00:00
Author: Agent Epsilon
Module: dashboard_modules.visualization.advanced_visualization
"""

from typing import Dict, List, Any, Optional


class AdvancedVisualizationEngine:
    """
    EPSILON ENHANCEMENT: Advanced visualization system with AI-powered chart selection,
    interactive drill-down capabilities, and context-aware adaptations.
    """
    
    def __init__(self):
        self.chart_intelligence = {}
        self.interaction_patterns = {}
        self.visualization_cache = {}
        self.context_adaptations = {}
    
    def select_optimal_visualization(self, data_characteristics: Dict[str, Any], 
                                   user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered visualization selection based on data characteristics and user context."""
        recommendations = []
        
        # Analyze data characteristics
        data_type = self._analyze_data_type(data_characteristics)
        data_volume = data_characteristics.get('volume', 0)
        temporal_nature = data_characteristics.get('has_time_series', False)
        correlation_density = data_characteristics.get('correlation_count', 0)
        
        # User context considerations
        user_role = user_context.get('role', 'general')
        device_type = user_context.get('device', 'desktop')
        
        # Chart recommendation logic
        if temporal_nature and data_volume > 10:
            if user_role in ['executive', 'financial']:
                recommendations.append({
                    'type': 'intelligent_line_chart',
                    'priority': 0.9,
                    'reason': 'Time series data optimal for trend analysis',
                    'enhancements': ['trend_lines', 'forecast_overlay', 'anomaly_detection']
                })
            else:
                recommendations.append({
                    'type': 'interactive_multi_line',
                    'priority': 0.85,
                    'reason': 'Technical users benefit from granular time series control',
                    'enhancements': ['zoom_controls', 'data_brushing', 'correlation_highlights']
                })
        
        if correlation_density > 3:
            recommendations.append({
                'type': 'correlation_matrix_heatmap',
                'priority': 0.8,
                'reason': 'High correlation density requires matrix visualization',
                'enhancements': ['interactive_drill_down', 'statistical_overlays', 'cluster_highlighting']
            })
        
        # Hierarchical data recommendations
        if data_characteristics.get('has_hierarchy'):
            recommendations.append({
                'type': 'intelligent_treemap',
                'priority': 0.75,
                'reason': 'Hierarchical data benefits from treemap visualization',
                'enhancements': ['zoom_navigation', 'breadcrumb_trail', 'dynamic_sizing']
            })
        
        # Categorical data recommendations
        if data_characteristics.get('has_categories'):
            if device_type == 'mobile':
                recommendations.append({
                    'type': 'mobile_optimized_bar',
                    'priority': 0.8,
                    'reason': 'Mobile-optimized visualization for categorical data',
                    'enhancements': ['touch_interactions', 'simplified_labels', 'gesture_navigation']
                })
            else:
                recommendations.append({
                    'type': 'interactive_bar_chart',
                    'priority': 0.7,
                    'reason': 'Standard bar chart with interactive enhancements',
                    'enhancements': ['sorting_controls', 'filter_options', 'drill_down']
                })
        
        # Network/relationship data
        if data_characteristics.get('has_relationships'):
            recommendations.append({
                'type': 'force_directed_network',
                'priority': 0.85,
                'reason': 'Network visualization optimal for relationship data',
                'enhancements': ['physics_simulation', 'clustering', 'interactive_exploration']
            })
        
        return sorted(recommendations, key=lambda x: x['priority'], reverse=True)
    
    def create_interactive_chart_config(self, chart_type: str, data: Any, 
                                      user_context: Dict[str, Any], 
                                      enhancements: List[str]) -> Dict[str, Any]:
        """Create intelligent chart configuration with interactive capabilities."""
        base_config = self._get_base_chart_config(chart_type)
        
        # Add intelligence enhancements
        if 'drill_down' in enhancements:
            base_config['plugins']['drill_down'] = {
                'enabled': True,
                'levels': self._generate_drill_down_levels(data),
                'transition_animation': 'smooth_zoom'
            }
        
        if 'smart_tooltips' in enhancements:
            base_config['plugins']['smart_tooltips'] = {
                'enabled': True,
                'context_aware': True,
                'relationship_hints': True,
                'prediction_overlay': user_context.get('role') in ['technical', 'analyst']
            }
        
        if 'trend_lines' in enhancements:
            base_config['plugins']['trend_analysis'] = {
                'enabled': True,
                'show_regression_line': True,
                'confidence_intervals': user_context.get('role') in ['analyst', 'technical'],
                'forecast_periods': 5
            }
        
        if 'anomaly_detection' in enhancements:
            base_config['plugins']['anomaly_detection'] = {
                'enabled': True,
                'highlight_outliers': True,
                'statistical_method': 'z_score',
                'sensitivity': 2.0
            }
        
        if 'correlation_highlights' in enhancements:
            base_config['plugins']['correlation_analysis'] = {
                'enabled': True,
                'show_correlation_strength': True,
                'highlight_strong_correlations': True,
                'correlation_threshold': 0.7
            }
        
        # Device-specific optimizations
        if user_context.get('device') == 'mobile':
            base_config = self._apply_mobile_optimizations(base_config)
        elif user_context.get('device') == 'tablet':
            base_config = self._apply_tablet_optimizations(base_config)
        
        return base_config
    
    def generate_contextual_interactions(self, chart_data: Any, 
                                       relationships: Dict[str, Any], 
                                       user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent contextual interactions for charts."""
        interactions = []
        
        # Hover interactions with intelligence
        interactions.append({
            'trigger': 'hover',
            'action': 'smart_tooltip',
            'intelligence': {
                'show_related_metrics': True,
                'correlation_indicators': relationships.get('correlations', []),
                'trend_analysis': True,
                'prediction_hints': user_context.get('role') in ['analyst', 'technical']
            }
        })
        
        # Click interactions for drill-down
        interactions.append({
            'trigger': 'click',
            'action': 'contextual_drill_down',
            'intelligence': {
                'determine_drill_target': True,
                'preserve_context': True,
                'smart_breadcrumbs': True,
                'related_data_suggestion': True
            }
        })
        
        # Double-click for zoom/focus
        interactions.append({
            'trigger': 'double_click',
            'action': 'intelligent_focus',
            'intelligence': {
                'auto_zoom_to_data': True,
                'filter_related_charts': True,
                'maintain_context_links': True
            }
        })
        
        # Right-click context menu
        interactions.append({
            'trigger': 'right_click',
            'action': 'context_menu',
            'intelligence': {
                'smart_menu_items': True,
                'role_based_options': user_context.get('role', 'general'),
                'data_export_options': True,
                'analysis_shortcuts': True
            }
        })
        
        # Touch gestures for mobile
        if user_context.get('device') in ['mobile', 'tablet']:
            interactions.extend(self._generate_touch_interactions(chart_data, user_context))
        
        return interactions
    
    def generate_responsive_layout(self, chart_configs: List[Dict[str, Any]], 
                                 screen_size: Dict[str, int], 
                                 user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate responsive layout based on screen size and user context."""
        layout = {
            'grid_system': 'flexible',
            'breakpoints': {
                'mobile': 768,
                'tablet': 1024,
                'desktop': 1200,
                'large_desktop': 1920
            },
            'chart_arrangements': []
        }
        
        screen_width = screen_size.get('width', 1200)
        screen_height = screen_size.get('height', 800)
        
        # Determine layout strategy
        if screen_width <= 768:  # Mobile
            layout['strategy'] = 'single_column'
            layout['chart_size'] = {'width': '100%', 'height': '300px'}
            layout['spacing'] = {'margin': '10px', 'padding': '5px'}
        elif screen_width <= 1024:  # Tablet
            layout['strategy'] = 'two_column'
            layout['chart_size'] = {'width': '48%', 'height': '400px'}
            layout['spacing'] = {'margin': '15px', 'padding': '10px'}
        else:  # Desktop and larger
            layout['strategy'] = 'grid_layout'
            layout['chart_size'] = {'width': '32%', 'height': '450px'}
            layout['spacing'] = {'margin': '20px', 'padding': '15px'}
        
        # Arrange charts based on priority and user role
        priority_charts = []
        secondary_charts = []
        
        for chart_config in chart_configs:
            priority = chart_config.get('priority', 0.5)
            if priority >= 0.8:
                priority_charts.append(chart_config)
            else:
                secondary_charts.append(chart_config)
        
        # Role-based prioritization
        user_role = user_context.get('role', 'general')
        if user_role in ['executive', 'manager']:
            layout['chart_arrangements'] = self._arrange_for_executives(priority_charts, secondary_charts, layout)
        elif user_role in ['analyst', 'technical']:
            layout['chart_arrangements'] = self._arrange_for_technical(priority_charts, secondary_charts, layout)
        else:
            layout['chart_arrangements'] = self._arrange_default(priority_charts, secondary_charts, layout)
        
        return layout
    
    def optimize_chart_performance(self, chart_config: Dict[str, Any], 
                                 data_size: int, 
                                 device_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize chart performance based on data size and device capabilities."""
        optimized_config = chart_config.copy()
        
        # Data size optimizations
        if data_size > 10000:
            optimized_config['data_sampling'] = {
                'enabled': True,
                'method': 'intelligent_sampling',
                'max_points': 5000,
                'preserve_trends': True
            }
        elif data_size > 1000:
            optimized_config['data_aggregation'] = {
                'enabled': True,
                'method': 'time_based_grouping',
                'group_size': 'auto'
            }
        
        # Device capability optimizations
        gpu_support = device_capabilities.get('webgl', False)
        memory_limit = device_capabilities.get('memory_mb', 4096)
        
        if not gpu_support or memory_limit < 2048:
            # Low-end device optimizations
            optimized_config['rendering'] = {
                'use_canvas_fallback': True,
                'reduce_animation_complexity': True,
                'limit_simultaneous_animations': 2,
                'use_data_decimation': True
            }
        else:
            # High-end device enhancements
            optimized_config['rendering'] = {
                'use_webgl_acceleration': True,
                'enable_advanced_animations': True,
                'high_dpi_support': True,
                'smooth_interactions': True
            }
        
        # Network optimization
        network_speed = device_capabilities.get('network_speed', 'broadband')
        if network_speed in ['slow', '3g']:
            optimized_config['data_loading'] = {
                'progressive_loading': True,
                'compression': True,
                'lazy_load_details': True
            }
        
        return optimized_config
    
    def _analyze_data_type(self, characteristics: Dict[str, Any]) -> str:
        """Analyze data type from characteristics."""
        if characteristics.get('has_hierarchy'):
            return 'hierarchical'
        elif characteristics.get('has_correlations'):
            return 'correlational'
        elif characteristics.get('has_time_series'):
            return 'temporal'
        elif characteristics.get('has_categories'):
            return 'categorical'
        elif characteristics.get('has_relationships'):
            return 'network'
        else:
            return 'numerical'
    
    def _get_base_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """Get base configuration for chart types."""
        configs = {
            'intelligent_line_chart': {
                'type': 'line',
                'responsive': True,
                'maintainAspectRatio': False,
                'interaction': {'intersect': False, 'mode': 'index'},
                'plugins': {
                    'legend': {'display': True, 'position': 'top'},
                    'tooltip': {'mode': 'index', 'intersect': False}
                },
                'scales': {
                    'x': {'type': 'time', 'display': True},
                    'y': {'beginAtZero': False, 'display': True}
                },
                'elements': {
                    'line': {'tension': 0.1},
                    'point': {'radius': 3, 'hoverRadius': 6}
                }
            },
            'interactive_multi_line': {
                'type': 'line',
                'responsive': True,
                'maintainAspectRatio': False,
                'interaction': {'intersect': False, 'mode': 'index'},
                'plugins': {
                    'legend': {'display': True, 'position': 'top'},
                    'tooltip': {'mode': 'index', 'intersect': False},
                    'zoom': {'enabled': True, 'mode': 'x'}
                },
                'scales': {
                    'x': {'type': 'time', 'display': True},
                    'y': {'beginAtZero': False, 'display': True}
                }
            },
            'correlation_matrix_heatmap': {
                'type': 'matrix',
                'responsive': True,
                'maintainAspectRatio': True,
                'interaction': {'intersect': True, 'mode': 'point'},
                'plugins': {
                    'legend': {'display': False},
                    'tooltip': {'mode': 'point', 'intersect': True}
                },
                'scales': {
                    'x': {'type': 'category', 'display': True},
                    'y': {'type': 'category', 'display': True}
                }
            },
            'intelligent_treemap': {
                'type': 'treemap',
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': False},
                    'tooltip': {'mode': 'point'}
                }
            },
            'interactive_bar_chart': {
                'type': 'bar',
                'responsive': True,
                'maintainAspectRatio': False,
                'interaction': {'intersect': False, 'mode': 'index'},
                'plugins': {
                    'legend': {'display': True, 'position': 'top'},
                    'tooltip': {'mode': 'index', 'intersect': False}
                },
                'scales': {
                    'x': {'type': 'category', 'display': True},
                    'y': {'beginAtZero': True, 'display': True}
                }
            },
            'force_directed_network': {
                'type': 'network',
                'responsive': True,
                'maintainAspectRatio': false,
                'layout': {
                    'algorithm': 'force_directed',
                    'iterations': 100,
                    'physics': {'enabled': True}
                },
                'plugins': {
                    'tooltip': {'mode': 'point'},
                    'zoom': {'enabled': True}
                }
            }
        }
        return configs.get(chart_type, configs['intelligent_line_chart'])
    
    def _generate_drill_down_levels(self, data: Any) -> List[str]:
        """Generate drill-down levels based on data structure."""
        levels = []
        
        if isinstance(data, dict):
            if 'daily' in data and 'hourly' in data:
                levels = ['daily', 'hourly', 'minute']
            elif 'categories' in data and 'subcategories' in data:
                levels = ['categories', 'subcategories', 'items']
            elif 'yearly' in data and 'monthly' in data:
                levels = ['yearly', 'monthly', 'weekly', 'daily']
            else:
                levels = ['overview', 'details', 'diagnostics']
        elif isinstance(data, list) and len(data) > 100:
            levels = ['aggregated', 'grouped', 'individual']
        else:
            levels = ['overview', 'details']
        
        return levels
    
    def _apply_mobile_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mobile-specific optimizations to chart configuration."""
        mobile_config = config.copy()
        
        # Optimize for touch interactions
        mobile_config['interaction'] = {
            'intersect': True,
            'mode': 'point'
        }
        
        # Increase touch target sizes
        if 'elements' in mobile_config:
            mobile_config['elements']['point']['radius'] = 6
            mobile_config['elements']['point']['hoverRadius'] = 10
        
        # Simplify legends and labels
        mobile_config['plugins']['legend']['position'] = 'bottom'
        mobile_config['plugins']['legend']['labels'] = {'boxWidth': 12, 'fontSize': 10}
        
        # Optimize scales for mobile viewing
        if 'scales' in mobile_config:
            mobile_config['scales']['x']['ticks'] = {'maxTicksLimit': 5}
            mobile_config['scales']['y']['ticks'] = {'maxTicksLimit': 5}
        
        return mobile_config
    
    def _apply_tablet_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tablet-specific optimizations to chart configuration."""
        tablet_config = config.copy()
        
        # Balance between desktop and mobile optimizations
        tablet_config['interaction'] = {
            'intersect': False,
            'mode': 'index'
        }
        
        # Medium-sized touch targets
        if 'elements' in tablet_config:
            tablet_config['elements']['point']['radius'] = 4
            tablet_config['elements']['point']['hoverRadius'] = 8
        
        # Optimize tick limits for tablet screens
        if 'scales' in tablet_config:
            tablet_config['scales']['x']['ticks'] = {'maxTicksLimit': 8}
            tablet_config['scales']['y']['ticks'] = {'maxTicksLimit': 8}
        
        return tablet_config
    
    def _generate_touch_interactions(self, chart_data: Any, 
                                   user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate touch-specific interactions for mobile devices."""
        touch_interactions = [
            {
                'trigger': 'tap',
                'action': 'select_point',
                'intelligence': {
                    'highlight_related_data': True,
                    'show_detailed_tooltip': True
                }
            },
            {
                'trigger': 'long_press',
                'action': 'context_menu',
                'intelligence': {
                    'touch_optimized_menu': True,
                    'quick_actions': ['zoom', 'filter', 'export']
                }
            },
            {
                'trigger': 'pinch',
                'action': 'zoom',
                'intelligence': {
                    'maintain_data_integrity': True,
                    'smooth_transitions': True
                }
            },
            {
                'trigger': 'swipe',
                'action': 'pan_or_navigate',
                'intelligence': {
                    'direction_aware': True,
                    'momentum_scrolling': True,
                    'boundary_detection': True
                }
            }
        ]
        
        return touch_interactions
    
    def _arrange_for_executives(self, priority_charts: List[Dict[str, Any]], 
                              secondary_charts: List[Dict[str, Any]], 
                              layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Arrange charts optimally for executive users."""
        arrangements = []
        
        # Executives prefer high-level overview charts first
        for chart in priority_charts[:2]:  # Top 2 priority charts
            arrangements.append({
                'chart': chart,
                'position': 'top',
                'size': 'large',
                'emphasis': 'high'
            })
        
        # Secondary charts in smaller format
        for chart in secondary_charts[:4]:  # Up to 4 secondary charts
            arrangements.append({
                'chart': chart,
                'position': 'bottom',
                'size': 'medium',
                'emphasis': 'medium'
            })
        
        return arrangements
    
    def _arrange_for_technical(self, priority_charts: List[Dict[str, Any]], 
                             secondary_charts: List[Dict[str, Any]], 
                             layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Arrange charts optimally for technical users."""
        arrangements = []
        
        # Technical users prefer detailed charts with equal emphasis
        all_charts = priority_charts + secondary_charts
        
        for i, chart in enumerate(all_charts):
            arrangements.append({
                'chart': chart,
                'position': f'grid_{i}',
                'size': 'uniform',
                'emphasis': 'equal'
            })
        
        return arrangements
    
    def _arrange_default(self, priority_charts: List[Dict[str, Any]], 
                        secondary_charts: List[Dict[str, Any]], 
                        layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default chart arrangement for general users."""
        arrangements = []
        
        # Balanced approach - priority first, then secondary
        for chart in priority_charts:
            arrangements.append({
                'chart': chart,
                'position': 'priority_section',
                'size': 'standard',
                'emphasis': 'high'
            })
        
        for chart in secondary_charts:
            arrangements.append({
                'chart': chart,
                'position': 'secondary_section',
                'size': 'standard',
                'emphasis': 'medium'
            })
        
        return arrangements