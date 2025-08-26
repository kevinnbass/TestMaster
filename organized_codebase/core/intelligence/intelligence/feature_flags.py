"""
Enhanced Feature Flags System

Advanced feature flag management with conditional logic, A/B testing,
gradual rollouts, and integration with the intelligence system.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import random
from pathlib import Path


class FeatureStatus(Enum):
    """Feature flag status."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    CONDITIONAL = "conditional"
    TESTING = "testing"
    ROLLOUT = "rollout"


class ConditionType(Enum):
    """Types of feature conditions."""
    USER_ID = "user_id"
    USER_GROUP = "user_group"
    PERCENTAGE = "percentage"
    TIME_WINDOW = "time_window"
    SYSTEM_LOAD = "system_load"
    ENVIRONMENT = "environment"
    VERSION = "version"
    CUSTOM = "custom"


@dataclass
class FeatureCondition:
    """Condition for feature flag activation."""
    condition_type: ConditionType
    operator: str  # "eq", "ne", "in", "not_in", "gt", "lt", "gte", "lte", "between"
    value: Any
    secondary_value: Optional[Any] = None  # For "between" operator
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against provided context."""
        if self.condition_type == ConditionType.USER_ID:
            user_id = context.get("user_id")
            return self._apply_operator(user_id, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.USER_GROUP:
            user_groups = context.get("user_groups", [])
            return self._apply_operator(user_groups, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.PERCENTAGE:
            # Use user_id for consistent percentage-based rollouts
            user_id = context.get("user_id", "anonymous")
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
            percentage = (hash_value % 100) + 1
            return self._apply_operator(percentage, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.TIME_WINDOW:
            current_time = context.get("current_time", datetime.now())
            return self._apply_operator(current_time, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.SYSTEM_LOAD:
            system_load = context.get("system_load", 0.0)
            return self._apply_operator(system_load, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.ENVIRONMENT:
            environment = context.get("environment", "development")
            return self._apply_operator(environment, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.VERSION:
            version = context.get("version", "0.0.0")
            return self._apply_operator(version, self.operator, self.value, self.secondary_value)
        
        elif self.condition_type == ConditionType.CUSTOM:
            custom_evaluator = context.get("custom_evaluators", {}).get(str(self.value))
            if custom_evaluator and callable(custom_evaluator):
                return custom_evaluator(context)
            return False
        
        return False
    
    def _apply_operator(self, actual: Any, operator: str, expected: Any, secondary: Any = None) -> bool:
        """Apply comparison operator."""
        try:
            if operator == "eq":
                return actual == expected
            elif operator == "ne":
                return actual != expected
            elif operator == "in":
                return actual in expected if isinstance(expected, (list, tuple, set)) else str(actual) in str(expected)
            elif operator == "not_in":
                return actual not in expected if isinstance(expected, (list, tuple, set)) else str(actual) not in str(expected)
            elif operator == "gt":
                return actual > expected
            elif operator == "lt":
                return actual < expected
            elif operator == "gte":
                return actual >= expected
            elif operator == "lte":
                return actual <= expected
            elif operator == "between":
                return expected <= actual <= secondary if secondary is not None else False
            else:
                return False
        except (TypeError, ValueError):
            return False


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.DISABLED
    conditions: List[FeatureCondition] = field(default_factory=list)
    default_value: Any = False
    enabled_value: Any = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Rollout configuration
    rollout_percentage: float = 0.0  # 0-100
    rollout_start_time: Optional[datetime] = None
    rollout_end_time: Optional[datetime] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate the feature flag and return the appropriate value."""
        if self.status == FeatureStatus.DISABLED:
            return self.default_value
        
        elif self.status == FeatureStatus.ENABLED:
            return self.enabled_value
        
        elif self.status == FeatureStatus.CONDITIONAL:
            # All conditions must be true
            for condition in self.conditions:
                if not condition.evaluate(context):
                    return self.default_value
            return self.enabled_value
        
        elif self.status == FeatureStatus.TESTING:
            # A/B testing - use percentage-based rollout
            return self._evaluate_percentage_rollout(context)
        
        elif self.status == FeatureStatus.ROLLOUT:
            # Gradual rollout with time-based progression
            return self._evaluate_gradual_rollout(context)
        
        return self.default_value
    
    def _evaluate_percentage_rollout(self, context: Dict[str, Any]) -> Any:
        """Evaluate percentage-based rollout."""
        user_id = context.get("user_id", "anonymous")
        hash_value = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 100) + 1
        
        if percentage <= self.rollout_percentage:
            return self.enabled_value
        return self.default_value
    
    def _evaluate_gradual_rollout(self, context: Dict[str, Any]) -> Any:
        """Evaluate gradual rollout over time."""
        current_time = context.get("current_time", datetime.now())
        
        if not self.rollout_start_time or not self.rollout_end_time:
            return self.default_value
        
        if current_time < self.rollout_start_time:
            return self.default_value
        
        if current_time >= self.rollout_end_time:
            return self.enabled_value
        
        # Calculate progress through rollout period
        total_duration = (self.rollout_end_time - self.rollout_start_time).total_seconds()
        elapsed_duration = (current_time - self.rollout_start_time).total_seconds()
        progress = elapsed_duration / total_duration
        
        # Use progress to determine rollout percentage
        current_percentage = progress * 100
        
        user_id = context.get("user_id", "anonymous")
        hash_value = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest()[:8], 16)
        user_percentage = (hash_value % 100) + 1
        
        if user_percentage <= current_percentage:
            return self.enabled_value
        return self.default_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "conditions": [
                {
                    "condition_type": c.condition_type.value,
                    "operator": c.operator,
                    "value": c.value,
                    "secondary_value": c.secondary_value
                }
                for c in self.conditions
            ],
            "default_value": self.default_value,
            "enabled_value": self.enabled_value,
            "metadata": self.metadata,
            "rollout_percentage": self.rollout_percentage,
            "rollout_start_time": self.rollout_start_time.isoformat() if self.rollout_start_time else None,
            "rollout_end_time": self.rollout_end_time.isoformat() if self.rollout_end_time else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }


class FeatureFlagsManager:
    """
    Advanced feature flags management system.
    
    Features:
    - Dynamic feature flag evaluation
    - Conditional logic and A/B testing
    - Gradual rollouts and canary deployments
    - Real-time flag updates
    - Integration with intelligence system
    - Performance monitoring and analytics
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 enable_persistence: bool = True,
                 enable_analytics: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config_file = config_file
        self.enable_persistence = enable_persistence
        self.enable_analytics = enable_analytics
        
        # Feature flags storage
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.flag_history: List[Dict[str, Any]] = []
        
        # Context providers
        self.context_providers: Dict[str, Callable] = {}
        
        # Analytics
        self.evaluation_stats: Dict[str, Dict[str, int]] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Caching
        self.evaluation_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(seconds=60)  # 1 minute cache
        
        # Initialize default intelligence system flags
        self._initialize_default_flags()
        
        # Load from file if specified
        if self.config_file:
            self.load_from_file(self.config_file)
    
    def _initialize_default_flags(self):
        """Initialize default feature flags for intelligence system."""
        # Classical Analysis Features
        self.create_flag(
            name="classical_analysis.advanced_security",
            description="Enable advanced security analysis features",
            status=FeatureStatus.ENABLED,
            metadata={"category": "analysis", "impact": "high"}
        )
        
        self.create_flag(
            name="classical_analysis.ml_patterns",
            description="Enable ML-based pattern recognition",
            status=FeatureStatus.CONDITIONAL,
            conditions=[
                FeatureCondition(
                    condition_type=ConditionType.SYSTEM_LOAD,
                    operator="lt",
                    value=0.8
                )
            ],
            metadata={"category": "analysis", "impact": "medium"}
        )
        
        # Documentation Features
        self.create_flag(
            name="documentation.auto_generation",
            description="Enable automatic documentation generation",
            status=FeatureStatus.ENABLED,
            metadata={"category": "documentation", "impact": "high"}
        )
        
        self.create_flag(
            name="documentation.llm_enhancement",
            description="Use LLM for enhanced documentation quality",
            status=FeatureStatus.ROLLOUT,
            rollout_percentage=50.0,
            metadata={"category": "documentation", "impact": "medium"}
        )
        
        # Integration Features
        self.create_flag(
            name="integration.real_time_updates",
            description="Enable real-time system updates via WebSocket",
            status=FeatureStatus.TESTING,
            rollout_percentage=25.0,
            metadata={"category": "integration", "impact": "high"}
        )
        
        self.create_flag(
            name="integration.advanced_caching",
            description="Enable advanced caching mechanisms",
            status=FeatureStatus.ENABLED,
            metadata={"category": "integration", "impact": "medium"}
        )
        
        # Performance Features
        self.create_flag(
            name="performance.parallel_analysis",
            description="Enable parallel analysis processing",
            status=FeatureStatus.CONDITIONAL,
            conditions=[
                FeatureCondition(
                    condition_type=ConditionType.SYSTEM_LOAD,
                    operator="lt", 
                    value=0.7
                )
            ],
            metadata={"category": "performance", "impact": "high"}
        )
        
        # Experimental Features
        self.create_flag(
            name="experimental.ai_code_generation",
            description="Experimental AI-powered code generation",
            status=FeatureStatus.TESTING,
            rollout_percentage=10.0,
            metadata={"category": "experimental", "impact": "low"}
        )
    
    def create_flag(self, 
                    name: str, 
                    description: str,
                    status: FeatureStatus = FeatureStatus.DISABLED,
                    conditions: List[FeatureCondition] = None,
                    default_value: Any = False,
                    enabled_value: Any = True,
                    **kwargs) -> FeatureFlag:
        """Create a new feature flag."""
        flag = FeatureFlag(
            name=name,
            description=description,
            status=status,
            conditions=conditions or [],
            default_value=default_value,
            enabled_value=enabled_value,
            **kwargs
        )
        
        self.feature_flags[name] = flag
        
        # Initialize stats
        self.evaluation_stats[name] = {
            "total_evaluations": 0,
            "enabled_count": 0,
            "disabled_count": 0
        }
        
        self.logger.info(f"Created feature flag: {name}")
        
        if self.enable_persistence and self.config_file:
            self.save_to_file(self.config_file)
        
        return flag
    
    def update_flag(self, 
                    name: str,
                    status: Optional[FeatureStatus] = None,
                    conditions: Optional[List[FeatureCondition]] = None,
                    rollout_percentage: Optional[float] = None,
                    **kwargs):
        """Update an existing feature flag."""
        if name not in self.feature_flags:
            raise ValueError(f"Feature flag not found: {name}")
        
        flag = self.feature_flags[name]
        
        # Track change history
        old_values = flag.to_dict()
        
        # Update fields
        if status is not None:
            flag.status = status
        if conditions is not None:
            flag.conditions = conditions
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        
        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        flag.updated_at = datetime.now()
        
        # Record history
        self.flag_history.append({
            "flag_name": name,
            "timestamp": datetime.now().isoformat(),
            "old_values": old_values,
            "new_values": flag.to_dict(),
            "action": "update"
        })
        
        # Clear cache for this flag
        self._clear_cache_for_flag(name)
        
        self.logger.info(f"Updated feature flag: {name}")
        
        if self.enable_persistence and self.config_file:
            self.save_to_file(self.config_file)
    
    def delete_flag(self, name: str):
        """Delete a feature flag."""
        if name not in self.feature_flags:
            raise ValueError(f"Feature flag not found: {name}")
        
        # Record deletion in history
        self.flag_history.append({
            "flag_name": name,
            "timestamp": datetime.now().isoformat(),
            "old_values": self.feature_flags[name].to_dict(),
            "new_values": None,
            "action": "delete"
        })
        
        del self.feature_flags[name]
        if name in self.evaluation_stats:
            del self.evaluation_stats[name]
        
        self._clear_cache_for_flag(name)
        
        self.logger.info(f"Deleted feature flag: {name}")
        
        if self.enable_persistence and self.config_file:
            self.save_to_file(self.config_file)
    
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled."""
        value = self.get_flag_value(flag_name, context)
        return bool(value)
    
    def get_flag_value(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Get the value of a feature flag."""
        if flag_name not in self.feature_flags:
            self.logger.warning(f"Feature flag not found: {flag_name}")
            return False
        
        # Build evaluation context
        eval_context = self._build_evaluation_context(context)
        
        # Check cache first
        cache_key = self._get_cache_key(flag_name, eval_context)
        if cache_key in self.evaluation_cache:
            cached_value, cached_time = self.evaluation_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_value
        
        # Evaluate flag
        start_time = datetime.now()
        flag = self.feature_flags[flag_name]
        value = flag.evaluate(eval_context)
        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cache result
        self.evaluation_cache[cache_key] = (value, datetime.now())
        
        # Update analytics
        if self.enable_analytics:
            self._update_analytics(flag_name, value, evaluation_time)
        
        return value
    
    def _build_evaluation_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive evaluation context."""
        eval_context = {
            "current_time": datetime.now(),
            "user_id": "anonymous",
            "environment": "development",
            "version": "1.0.0",
            "system_load": 0.5  # Default values
        }
        
        # Add provided context
        if context:
            eval_context.update(context)
        
        # Add context from providers
        for provider_name, provider_func in self.context_providers.items():
            try:
                provider_context = provider_func()
                if isinstance(provider_context, dict):
                    eval_context.update(provider_context)
            except Exception as e:
                self.logger.warning(f"Context provider {provider_name} failed: {e}")
        
        return eval_context
    
    def _get_cache_key(self, flag_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for flag evaluation."""
        # Use relevant context fields for caching
        relevant_fields = ["user_id", "environment", "version", "user_groups"]
        context_str = "|".join(
            f"{field}:{context.get(field, 'none')}" 
            for field in relevant_fields
        )
        return f"{flag_name}:{context_str}"
    
    def _clear_cache_for_flag(self, flag_name: str):
        """Clear cache entries for a specific flag."""
        keys_to_remove = [key for key in self.evaluation_cache.keys() if key.startswith(f"{flag_name}:")]
        for key in keys_to_remove:
            del self.evaluation_cache[key]
    
    def _update_analytics(self, flag_name: str, value: Any, evaluation_time: float):
        """Update analytics for flag evaluation."""
        stats = self.evaluation_stats[flag_name]
        stats["total_evaluations"] += 1
        
        if value:
            stats["enabled_count"] += 1
        else:
            stats["disabled_count"] += 1
        
        # Update performance metrics
        if flag_name not in self.performance_metrics:
            self.performance_metrics[flag_name] = evaluation_time
        else:
            # Moving average
            current_avg = self.performance_metrics[flag_name]
            total_evaluations = stats["total_evaluations"]
            self.performance_metrics[flag_name] = (
                (current_avg * (total_evaluations - 1) + evaluation_time) / total_evaluations
            )
    
    def register_context_provider(self, name: str, provider: Callable):
        """Register a context provider function."""
        self.context_providers[name] = provider
        self.logger.info(f"Registered context provider: {name}")
    
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags and their current status."""
        return {name: flag.to_dict() for name, flag in self.feature_flags.items()}
    
    def get_flag_analytics(self) -> Dict[str, Any]:
        """Get analytics data for all flags."""
        return {
            "evaluation_stats": self.evaluation_stats.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "cache_stats": {
                "total_entries": len(self.evaluation_cache),
                "cache_hit_rate": self._calculate_cache_hit_rate()
            }
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This is a simplified implementation
        # Real implementation would track hits vs misses
        return 0.85  # Placeholder
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export complete configuration."""
        return {
            "feature_flags": {name: flag.to_dict() for name, flag in self.feature_flags.items()},
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_flags": len(self.feature_flags),
                "version": "1.0.0"
            }
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import configuration from dictionary."""
        if "feature_flags" not in config:
            raise ValueError("Invalid configuration format")
        
        for flag_name, flag_data in config["feature_flags"].items():
            # Convert conditions
            conditions = []
            for cond_data in flag_data.get("conditions", []):
                condition = FeatureCondition(
                    condition_type=ConditionType(cond_data["condition_type"]),
                    operator=cond_data["operator"],
                    value=cond_data["value"],
                    secondary_value=cond_data.get("secondary_value")
                )
                conditions.append(condition)
            
            # Create/update flag
            flag = FeatureFlag(
                name=flag_data["name"],
                description=flag_data["description"],
                status=FeatureStatus(flag_data["status"]),
                conditions=conditions,
                default_value=flag_data["default_value"],
                enabled_value=flag_data["enabled_value"],
                metadata=flag_data.get("metadata", {}),
                rollout_percentage=flag_data.get("rollout_percentage", 0.0)
            )
            
            # Handle datetime fields
            if flag_data.get("rollout_start_time"):
                flag.rollout_start_time = datetime.fromisoformat(flag_data["rollout_start_time"])
            if flag_data.get("rollout_end_time"):
                flag.rollout_end_time = datetime.fromisoformat(flag_data["rollout_end_time"])
            
            self.feature_flags[flag_name] = flag
            
            # Initialize stats
            if flag_name not in self.evaluation_stats:
                self.evaluation_stats[flag_name] = {
                    "total_evaluations": 0,
                    "enabled_count": 0,
                    "disabled_count": 0
                }
        
        self.logger.info(f"Imported {len(config['feature_flags'])} feature flags")
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        try:
            config = self.export_configuration()
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_from_file(self, file_path: str):
        """Load configuration from file."""
        try:
            if not Path(file_path).exists():
                self.logger.warning(f"Configuration file not found: {file_path}")
                return
            
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            self.import_configuration(config)
            self.logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def start_periodic_cleanup(self, interval: int = 3600):
        """Start periodic cleanup of cache and analytics."""
        while True:
            try:
                # Clean expired cache entries
                current_time = datetime.now()
                expired_keys = [
                    key for key, (_, cached_time) in self.evaluation_cache.items()
                    if current_time - cached_time > self.cache_ttl
                ]
                
                for key in expired_keys:
                    del self.evaluation_cache[key]
                
                # Limit history size
                if len(self.flag_history) > 10000:
                    self.flag_history = self.flag_history[-5000:]
                
                self.logger.debug(f"Cleaned up {len(expired_keys)} cache entries")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(interval)


# Convenience functions
def create_simple_flag(name: str, 
                      description: str, 
                      enabled: bool = False) -> FeatureFlag:
    """Create a simple enabled/disabled feature flag."""
    return FeatureFlag(
        name=name,
        description=description,
        status=FeatureStatus.ENABLED if enabled else FeatureStatus.DISABLED
    )


def create_percentage_flag(name: str, 
                          description: str, 
                          percentage: float) -> FeatureFlag:
    """Create a percentage-based rollout flag."""
    return FeatureFlag(
        name=name,
        description=description,
        status=FeatureStatus.TESTING,
        rollout_percentage=percentage
    )


def create_conditional_flag(name: str, 
                           description: str, 
                           conditions: List[FeatureCondition]) -> FeatureFlag:
    """Create a conditional feature flag."""
    return FeatureFlag(
        name=name,
        description=description,
        status=FeatureStatus.CONDITIONAL,
        conditions=conditions
    )