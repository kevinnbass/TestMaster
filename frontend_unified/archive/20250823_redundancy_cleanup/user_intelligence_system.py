#!/usr/bin/env python3
"""
STEELCLAD MODULE: User Intelligence System
==========================================

User intelligence classes extracted from unified_gamma_dashboard.py
Original: 3,634 lines → User Intelligence Module: ~400 lines

Complete functionality extraction with zero regression.

Author: Agent Epsilon (STEELCLAD Anti-Regression Modularization)
"""

from datetime import datetime
from collections import deque


class UserIntelligenceEngine:
    """
    Advanced user intelligence system that adapts interface and information
    delivery based on user behavior, role, and context.
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.behavior_patterns = deque(maxlen=1000)
    
    def personalize_information(self, raw_data, user_context):
        """Personalize information delivery based on user context."""
        if not user_context:
            return self._get_default_personalization(raw_data)
        
        user_role = user_context.get('role', 'general')
        user_preferences = user_context.get('preferences', {})
        
        personalization = {
            'priority_metrics': self._get_priority_metrics_for_role(user_role, raw_data),
            'information_density': self._determine_information_density(user_role, user_preferences),
            'visualization_preferences': self._get_visualization_preferences(user_role),
            'alert_preferences': self._get_alert_preferences(user_role, raw_data)
        }
        
        return personalization
    
    def _get_default_personalization(self, raw_data):
        """Get default personalization for unknown users."""
        return {
            'priority_metrics': ['system_health', 'api_usage', 'performance_metrics'],
            'information_density': 'medium',
            'visualization_preferences': 'standard_charts',
            'alert_preferences': 'standard'
        }
    
    def _get_priority_metrics_for_role(self, role, raw_data):
        """Get priority metrics based on user role."""
        role_priorities = {
            'executive': ['system_health', 'api_usage', 'agent_coordination'],
            'technical': ['performance_metrics', 'system_health', 'technical_insights'],
            'financial': ['api_usage', 'cost_analysis', 'budget_status'],
            'operations': ['agent_status', 'system_health', 'coordination_status']
        }
        
        return role_priorities.get(role, ['system_health', 'api_usage', 'performance_metrics'])
    
    def _determine_information_density(self, role, preferences):
        """Determine optimal information density for user."""
        role_density = {
            'executive': 'high',
            'technical': 'maximum',
            'financial': 'focused',
            'operations': 'detailed'
        }
        
        return preferences.get('density', role_density.get(role, 'medium'))
    
    def _get_visualization_preferences(self, role):
        """Get visualization preferences for user role."""
        role_viz = {
            'executive': 'executive_dashboard',
            'technical': 'detailed_charts',
            'financial': 'financial_charts',
            'operations': 'operational_dashboard'
        }
        
        return role_viz.get(role, 'standard_charts')
    
    def _get_alert_preferences(self, role, raw_data):
        """Get alert preferences based on role and current data."""
        if role == 'executive':
            return 'critical_only'
        elif role == 'technical':
            return 'detailed'
        elif role == 'financial':
            return 'cost_focused'
        else:
            return 'standard'


class PredictiveDataCache:
    """
    Intelligent caching system that predicts data needs and prefetches
    relevant information to optimize response times.
    """
    
    def __init__(self):
        self.cache = {}
        self.access_patterns = deque(maxlen=500)
        self.prediction_accuracy = 0.0
    
    def predict_and_cache(self, user_context, current_data):
        """Predict future data needs and cache accordingly."""
        predictions = self._generate_predictions(user_context, current_data)
        
        for prediction in predictions:
            cache_key = f"predicted_{prediction['type']}_{prediction['context']}"
            self.cache[cache_key] = {
                'data': prediction['data'],
                'timestamp': datetime.now(),
                'confidence': prediction['confidence']
            }
        
        return len(predictions)
    
    def _generate_predictions(self, user_context, current_data):
        """Generate predictions based on patterns."""
        predictions = []
        
        # Predict likely next requests based on current context
        if user_context and user_context.get('role') == 'technical':
            predictions.append({
                'type': 'detailed_performance',
                'context': 'technical_user',
                'data': current_data.get('performance_metrics', {}),
                'confidence': 0.8
            })
        
        return predictions


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
    
    def select_optimal_visualization(self, data_characteristics, user_context):
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
        
        return sorted(recommendations, key=lambda x: x['priority'], reverse=True)
    
    def create_interactive_chart_config(self, chart_type, data, user_context, enhancements):
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
        
        return base_config
    
    def generate_contextual_interactions(self, chart_data, relationships, user_context):
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
        
        return interactions
    
    def _analyze_data_type(self, characteristics):
        """Analyze data type from characteristics."""
        if characteristics.get('has_hierarchy'):
            return 'hierarchical'
        elif characteristics.get('has_correlations'):
            return 'correlational'
        elif characteristics.get('has_time_series'):
            return 'temporal'
        elif characteristics.get('has_categories'):
            return 'categorical'
        else:
            return 'numerical'
    
    def _get_base_chart_config(self, chart_type):
        """Get base configuration for chart types."""
        configs = {
            'intelligent_line_chart': {
                'type': 'line',
                'responsive': True,
                'interaction': {'intersect': False, 'mode': 'index'},
                'plugins': {'legend': {'display': True}, 'tooltip': {'mode': 'index'}},
                'scales': {'x': {'type': 'time'}, 'y': {'beginAtZero': False}}
            },
            'correlation_matrix_heatmap': {
                'type': 'matrix',
                'responsive': True,
                'interaction': {'intersect': True, 'mode': 'point'},
                'plugins': {'legend': {'display': False}, 'tooltip': {'mode': 'point'}},
                'scales': {'x': {'type': 'category'}, 'y': {'type': 'category'}}
            }
        }
        return configs.get(chart_type, configs['intelligent_line_chart'])
    
    def _generate_drill_down_levels(self, data):
        """Generate drill-down levels based on data structure."""
        levels = []
        
        if isinstance(data, dict):
            if 'daily' in data and 'hourly' in data:
                levels = ['daily', 'hourly', 'minute']
            elif 'categories' in data and 'subcategories' in data:
                levels = ['categories', 'subcategories', 'items']
            else:
                levels = ['overview', 'details', 'diagnostics']
        
        return levels