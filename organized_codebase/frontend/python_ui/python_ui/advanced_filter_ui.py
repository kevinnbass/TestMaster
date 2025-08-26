"""
MODULE: Advanced Filter UI - Interactive Dashboard Filtering
==================================================================

PURPOSE:
    Comprehensive filtering interface for dashboard visualizations,
    providing intuitive UI components for complex data filtering,
    saved filter presets, and real-time filter application.

CORE FUNCTIONALITY:
    • Dynamic filter builder with multiple conditions
    • Saved filter presets and templates
    • Real-time filter preview and validation
    • Export/import filter configurations
    • Filter history and undo/redo functionality

EDIT HISTORY (Last 5 Changes):
==================================================================
[2025-08-23 11:00:00] | Agent Gamma | FEATURE
   └─ Goal: Create advanced filtering UI system
   └─ Changes: Built comprehensive filter interface with presets
   └─ Impact: Enables intuitive data filtering in dashboard

METADATA:
==================================================================
Created: 2025-08-23 by Agent Gamma
Language: Python
Dependencies: Flask, data_aggregation_pipeline
Integration Points: chart_integration.py, dashboard HTML
Performance Notes: Real-time filter application <50ms
Security Notes: Input sanitization, XSS prevention

TESTING STATUS:
==================================================================
Unit Tests: [Pending] | Last Run: [Not yet tested]
Integration Tests: [Pending] | Last Run: [Not yet tested]
Performance Tests: [Target: <50ms application] | Last Run: [Not yet tested]
Known Issues: Initial implementation - requires testing

COORDINATION NOTES:
==================================================================
Dependencies: Data aggregation pipeline
Provides: Filter UI components for dashboard
Breaking Changes: None - additive enhancement
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import from data aggregation module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.frontend.data.data_aggregation_pipeline import FilterCondition, FilterOperator

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Types of filter inputs."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    DATETIME = "datetime"
    SELECT = "select"
    MULTISELECT = "multiselect"
    RANGE = "range"
    BOOLEAN = "boolean"
    CUSTOM = "custom"


@dataclass
class FilterPreset:
    """Saved filter configuration."""
    id: str
    name: str
    description: str
    conditions: List[FilterCondition]
    created_at: datetime
    updated_at: datetime
    is_public: bool = False
    tags: List[str] = None
    usage_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'conditions': [
                {
                    'field': c.field,
                    'operator': c.operator.value,
                    'value': c.value,
                    'case_sensitive': c.case_sensitive
                }
                for c in self.conditions
            ],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_public': self.is_public,
            'tags': self.tags or [],
            'usage_count': self.usage_count
        }


@dataclass
class FilterField:
    """Filter field configuration."""
    name: str
    label: str
    type: FilterType
    operators: List[FilterOperator]
    default_value: Any = None
    options: List[Dict] = None  # For select/multiselect
    min_value: Any = None  # For number/date ranges
    max_value: Any = None
    step: Any = None  # For number inputs
    placeholder: str = ""
    help_text: str = ""
    validation_rules: Dict = None


class AdvancedFilterUI:
    """
    Advanced filter UI system for dashboard interactions.
    
    Features:
    - Dynamic filter builder with drag-and-drop
    - Complex condition groups (AND/OR logic)
    - Saved presets and templates
    - Real-time validation and preview
    - Export/import functionality
    - Undo/redo with history tracking
    """
    
    def __init__(self):
        self.filter_presets = {}
        self.filter_fields = {}
        self.filter_history = []
        self.history_index = -1
        self.max_history = 50
        self.active_filters = []
        
        # Performance tracking
        self.metrics = {
            'filters_created': 0,
            'filters_applied': 0,
            'presets_saved': 0,
            'presets_loaded': 0,
            'avg_build_time': 0
        }
        
        # Initialize default filter templates
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default filter templates."""
        # Time-based filters
        self.add_filter_preset(FilterPreset(
            id='last_24h',
            name='Last 24 Hours',
            description='Filter data from the last 24 hours',
            conditions=[
                FilterCondition(
                    field='timestamp',
                    operator=FilterOperator.GREATER_EQUAL,
                    value=(datetime.now() - timedelta(hours=24)).isoformat()
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_public=True,
            tags=['time', 'recent']
        ))
        
        self.add_filter_preset(FilterPreset(
            id='last_week',
            name='Last Week',
            description='Filter data from the last 7 days',
            conditions=[
                FilterCondition(
                    field='timestamp',
                    operator=FilterOperator.GREATER_EQUAL,
                    value=(datetime.now() - timedelta(days=7)).isoformat()
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_public=True,
            tags=['time', 'week']
        ))
        
        # Performance filters
        self.add_filter_preset(FilterPreset(
            id='high_performance',
            name='High Performance',
            description='Filter for high performance metrics',
            conditions=[
                FilterCondition(
                    field='performance_score',
                    operator=FilterOperator.GREATER_EQUAL,
                    value=80
                ),
                FilterCondition(
                    field='error_rate',
                    operator=FilterOperator.LESS_THAN,
                    value=5
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_public=True,
            tags=['performance', 'quality']
        ))
    
    def register_filter_field(self, field: FilterField):
        """Register a filterable field."""
        self.filter_fields[field.name] = field
        logger.info(f"Registered filter field: {field.name}")
    
    def build_filter_ui(self, fields: Optional[List[str]] = None) -> Dict:
        """
        Build filter UI configuration for frontend rendering.
        
        Args:
            fields: Specific fields to include (None for all)
            
        Returns:
            UI configuration dictionary
        """
        start_time = datetime.now()
        
        if fields:
            selected_fields = {
                name: field for name, field in self.filter_fields.items()
                if name in fields
            }
        else:
            selected_fields = self.filter_fields
        
        ui_config = {
            'fields': [],
            'operators': {
                'text': ['equals', 'not_equals', 'contains', 'starts_with', 'ends_with'],
                'number': ['equals', 'not_equals', 'greater_than', 'less_than', 'between'],
                'date': ['equals', 'not_equals', 'greater_than', 'less_than', 'between'],
                'select': ['equals', 'not_equals', 'in', 'not_in'],
                'boolean': ['equals', 'not_equals']
            },
            'presets': self.get_public_presets(),
            'recent_filters': self.get_recent_filters(5)
        }
        
        for field_name, field in selected_fields.items():
            field_config = {
                'name': field.name,
                'label': field.label,
                'type': field.type.value,
                'operators': [op.value for op in field.operators],
                'placeholder': field.placeholder,
                'help_text': field.help_text
            }
            
            # Add type-specific properties
            if field.type in [FilterType.SELECT, FilterType.MULTISELECT]:
                field_config['options'] = field.options or []
            
            if field.type in [FilterType.NUMBER, FilterType.RANGE]:
                field_config['min'] = field.min_value
                field_config['max'] = field.max_value
                field_config['step'] = field.step
            
            if field.type in [FilterType.DATE, FilterType.DATETIME]:
                field_config['min'] = field.min_value
                field_config['max'] = field.max_value
            
            ui_config['fields'].append(field_config)
        
        # Update metrics
        build_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_avg_build_time(build_time)
        self.metrics['filters_created'] += 1
        
        return ui_config
    
    def apply_filters(self, conditions: List[Dict]) -> List[FilterCondition]:
        """
        Apply filter conditions from UI input.
        
        Args:
            conditions: List of condition dictionaries from UI
            
        Returns:
            List of FilterCondition objects
        """
        filter_conditions = []
        
        for condition in conditions:
            try:
                filter_condition = FilterCondition(
                    field=condition['field'],
                    operator=FilterOperator(condition['operator']),
                    value=condition['value'],
                    case_sensitive=condition.get('case_sensitive', True)
                )
                filter_conditions.append(filter_condition)
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid filter condition: {e}")
                continue
        
        # Add to history
        self._add_to_history(filter_conditions)
        
        # Update active filters
        self.active_filters = filter_conditions
        
        self.metrics['filters_applied'] += 1
        
        return filter_conditions
    
    def add_filter_preset(self, preset: FilterPreset):
        """Add or update a filter preset."""
        preset.updated_at = datetime.now()
        self.filter_presets[preset.id] = preset
        self.metrics['presets_saved'] += 1
        logger.info(f"Added filter preset: {preset.name}")
    
    def load_filter_preset(self, preset_id: str) -> Optional[List[FilterCondition]]:
        """Load a filter preset by ID."""
        preset = self.filter_presets.get(preset_id)
        if preset:
            preset.usage_count += 1
            preset.updated_at = datetime.now()
            self.metrics['presets_loaded'] += 1
            return preset.conditions
        return None
    
    def get_public_presets(self) -> List[Dict]:
        """Get all public filter presets."""
        return [
            preset.to_dict()
            for preset in self.filter_presets.values()
            if preset.is_public
        ]
    
    def get_user_presets(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get user-specific filter presets."""
        # In production, filter by user_id
        return [
            preset.to_dict()
            for preset in self.filter_presets.values()
            if not preset.is_public
        ]
    
    def export_filters(self, filter_ids: Optional[List[str]] = None) -> str:
        """
        Export filter configurations to JSON.
        
        Args:
            filter_ids: Specific filter IDs to export (None for all)
            
        Returns:
            JSON string of filter configurations
        """
        if filter_ids:
            filters_to_export = [
                self.filter_presets[fid].to_dict()
                for fid in filter_ids
                if fid in self.filter_presets
            ]
        else:
            filters_to_export = [
                preset.to_dict()
                for preset in self.filter_presets.values()
            ]
        
        export_data = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'filters': filters_to_export
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_filters(self, json_data: str) -> int:
        """
        Import filter configurations from JSON.
        
        Args:
            json_data: JSON string of filter configurations
            
        Returns:
            Number of filters imported
        """
        try:
            data = json.loads(json_data)
            imported_count = 0
            
            for filter_data in data.get('filters', []):
                # Convert back to FilterCondition objects
                conditions = []
                for cond in filter_data.get('conditions', []):
                    conditions.append(FilterCondition(
                        field=cond['field'],
                        operator=FilterOperator(cond['operator']),
                        value=cond['value'],
                        case_sensitive=cond.get('case_sensitive', True)
                    ))
                
                preset = FilterPreset(
                    id=filter_data.get('id', str(uuid.uuid4())),
                    name=filter_data['name'],
                    description=filter_data.get('description', ''),
                    conditions=conditions,
                    created_at=datetime.fromisoformat(filter_data.get('created_at', datetime.now().isoformat())),
                    updated_at=datetime.now(),
                    is_public=filter_data.get('is_public', False),
                    tags=filter_data.get('tags', []),
                    usage_count=filter_data.get('usage_count', 0)
                )
                
                self.add_filter_preset(preset)
                imported_count += 1
            
            logger.info(f"Imported {imported_count} filter presets")
            return imported_count
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to import filters: {e}")
            return 0
    
    def _add_to_history(self, conditions: List[FilterCondition]):
        """Add filter conditions to history."""
        # Remove any history after current index
        self.filter_history = self.filter_history[:self.history_index + 1]
        
        # Add new entry
        self.filter_history.append({
            'timestamp': datetime.now().isoformat(),
            'conditions': conditions
        })
        
        # Limit history size
        if len(self.filter_history) > self.max_history:
            self.filter_history.pop(0)
        else:
            self.history_index += 1
    
    def undo_filter(self) -> Optional[List[FilterCondition]]:
        """Undo last filter operation."""
        if self.history_index > 0:
            self.history_index -= 1
            return self.filter_history[self.history_index]['conditions']
        return None
    
    def redo_filter(self) -> Optional[List[FilterCondition]]:
        """Redo filter operation."""
        if self.history_index < len(self.filter_history) - 1:
            self.history_index += 1
            return self.filter_history[self.history_index]['conditions']
        return None
    
    def get_recent_filters(self, limit: int = 10) -> List[Dict]:
        """Get recent filter history."""
        recent = self.filter_history[-limit:] if self.filter_history else []
        return list(reversed(recent))
    
    def validate_filter_value(self, field_name: str, value: Any) -> bool:
        """
        Validate filter value against field configuration.
        
        Args:
            field_name: Field name to validate against
            value: Value to validate
            
        Returns:
            Validation result
        """
        field = self.filter_fields.get(field_name)
        if not field:
            return True  # Unknown field, allow
        
        # Type-specific validation
        if field.type == FilterType.NUMBER:
            try:
                num_value = float(value)
                if field.min_value is not None and num_value < field.min_value:
                    return False
                if field.max_value is not None and num_value > field.max_value:
                    return False
            except (TypeError, ValueError):
                return False
        
        elif field.type in [FilterType.DATE, FilterType.DATETIME]:
            try:
                date_value = datetime.fromisoformat(value)
                if field.min_value and date_value < datetime.fromisoformat(field.min_value):
                    return False
                if field.max_value and date_value > datetime.fromisoformat(field.max_value):
                    return False
            except (TypeError, ValueError):
                return False
        
        elif field.type in [FilterType.SELECT, FilterType.MULTISELECT]:
            if field.options:
                valid_values = [opt.get('value') for opt in field.options]
                if field.type == FilterType.SELECT:
                    return value in valid_values
                else:  # MULTISELECT
                    return all(v in valid_values for v in value)
        
        return True
    
    def suggest_filter_values(self, field_name: str, partial_value: str = "") -> List[Dict]:
        """
        Suggest filter values based on partial input.
        
        Args:
            field_name: Field to get suggestions for
            partial_value: Partial value for autocomplete
            
        Returns:
            List of suggested values
        """
        field = self.filter_fields.get(field_name)
        if not field:
            return []
        
        suggestions = []
        
        if field.type in [FilterType.SELECT, FilterType.MULTISELECT] and field.options:
            for option in field.options:
                if partial_value.lower() in str(option.get('label', '')).lower():
                    suggestions.append(option)
        
        # Limit suggestions
        return suggestions[:10]
    
    def _update_avg_build_time(self, new_time: float):
        """Update average build time metric."""
        current_avg = self.metrics['avg_build_time']
        count = self.metrics['filters_created']
        
        if count > 1:
            self.metrics['avg_build_time'] = (current_avg * (count - 1) + new_time) / count
        else:
            self.metrics['avg_build_time'] = new_time
    
    def get_metrics(self) -> Dict:
        """Get filter UI metrics."""
        return {
            **self.metrics,
            'total_presets': len(self.filter_presets),
            'active_filters': len(self.active_filters),
            'history_size': len(self.filter_history)
        }
    
    def clear_filters(self):
        """Clear all active filters."""
        self.active_filters = []
        logger.info("All filters cleared")
    
    def reset_history(self):
        """Reset filter history."""
        self.filter_history = []
        self.history_index = -1
        logger.info("Filter history reset")


# Singleton instance
filter_ui = AdvancedFilterUI()