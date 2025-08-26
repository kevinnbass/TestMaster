"""
Adaptive Integration Engine
==========================

Revolutionary adaptive behavior learning and dynamic integration system.
Extracted from meta_intelligence_orchestrator.py for enterprise modular architecture.

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import (
    CapabilityType, IntelligenceBehaviorType, SystemBehaviorModel,
    SystemIntegrationStatus, OrchestrationEvent, CapabilityProfile
)


class AdaptiveIntegrationEngine:
    """
    Revolutionary Adaptive Integration Engine
    
    Learns and adapts to intelligence system behaviors, implementing dynamic
    integration strategies that evolve based on observed patterns and performance.
    """
    
    def __init__(self, learning_rate: float = 0.1, adaptation_threshold: float = 0.8):
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        
        # Core data stores
        self.behavior_models: Dict[str, SystemBehaviorModel] = {}
        self.integration_statuses: Dict[str, SystemIntegrationStatus] = {}
        self.adaptation_history: List[OrchestrationEvent] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # ML components
        self.behavior_classifier = None
        self.performance_predictor = None
        self.scaler = StandardScaler()
        
        # Integration patterns
        self.integration_patterns: Dict[str, Any] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        
        # Adaptive learning
        self.learning_weights: Dict[str, float] = {
            'response_time': 0.3,
            'accuracy': 0.4,
            'reliability': 0.2,
            'resource_efficiency': 0.1
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def learn_system_behavior(self, system_id: str, 
                                  interaction_data: Dict[str, Any]) -> SystemBehaviorModel:
        """Learn and update behavior model for an intelligence system"""
        
        # Get or create behavior model
        if system_id not in self.behavior_models:
            self.behavior_models[system_id] = SystemBehaviorModel(
                system_id=system_id,
                behavior_type=IntelligenceBehaviorType.ADAPTIVE,
                behavior_patterns={},
                response_time_distribution={},
                success_rate_over_time=[],
                failure_patterns={},
                adaptation_rate=0.5,
                learning_curve_data=[],
                resource_usage_patterns={},
                interaction_preferences={}
            )
        
        model = self.behavior_models[system_id]
        
        # Update behavior patterns
        await self._update_behavior_patterns(model, interaction_data)
        
        # Update performance metrics
        await self._update_performance_metrics(model, interaction_data)
        
        # Classify behavior type
        model.behavior_type = await self._classify_behavior_type(model)
        
        # Calculate adaptation rate
        model.adaptation_rate = await self._calculate_adaptation_rate(model)
        
        # Update confidence
        model.confidence_in_model = await self._calculate_model_confidence(model)
        
        model.last_behavior_update = datetime.now()
        
        self.logger.info(f"Updated behavior model for {system_id} - "
                        f"Type: {model.behavior_type.value}, "
                        f"Confidence: {model.confidence_in_model:.2f}")
        
        return model
    
    async def _update_behavior_patterns(self, model: SystemBehaviorModel, 
                                      interaction_data: Dict[str, Any]):
        """Update behavior patterns based on new interaction data"""
        
        # Extract behavioral indicators
        response_time = interaction_data.get('response_time', 0)
        success = interaction_data.get('success', False)
        resource_usage = interaction_data.get('resource_usage', {})
        
        # Update response time distribution
        percentile_key = self._calculate_percentile_key(response_time)
        if percentile_key not in model.response_time_distribution:
            model.response_time_distribution[percentile_key] = 0
        model.response_time_distribution[percentile_key] += 1
        
        # Update success rate over time
        current_time = datetime.now()
        model.success_rate_over_time.append((current_time, 1.0 if success else 0.0))
        
        # Keep only recent data (last 100 interactions)
        if len(model.success_rate_over_time) > 100:
            model.success_rate_over_time = model.success_rate_over_time[-100:]
        
        # Update failure patterns
        if not success:
            failure_type = interaction_data.get('failure_type', 'unknown')
            if failure_type not in model.failure_patterns:
                model.failure_patterns[failure_type] = 0
            model.failure_patterns[failure_type] += 1
        
        # Update resource usage patterns
        for resource, usage in resource_usage.items():
            if resource not in model.resource_usage_patterns:
                model.resource_usage_patterns[resource] = []
            model.resource_usage_patterns[resource].append(usage)
            
            # Keep only recent data
            if len(model.resource_usage_patterns[resource]) > 50:
                model.resource_usage_patterns[resource] = \
                    model.resource_usage_patterns[resource][-50:]
    
    async def _update_performance_metrics(self, model: SystemBehaviorModel, 
                                        interaction_data: Dict[str, Any]):
        """Update performance metrics and learning curve"""
        
        # Calculate performance score
        performance_score = await self._calculate_performance_score(interaction_data)
        
        # Update learning curve
        current_time = datetime.now()
        model.learning_curve_data.append((current_time, performance_score))
        
        # Keep only recent data (last 50 interactions)
        if len(model.learning_curve_data) > 50:
            model.learning_curve_data = model.learning_curve_data[-50:]
        
        # Update interaction preferences
        interaction_type = interaction_data.get('interaction_type', 'default')
        if interaction_type not in model.interaction_preferences:
            model.interaction_preferences[interaction_type] = 0.5
        
        # Adjust preference based on success
        success = interaction_data.get('success', False)
        adjustment = self.learning_rate * (1.0 if success else -0.5)
        model.interaction_preferences[interaction_type] = max(0.0, min(1.0,
            model.interaction_preferences[interaction_type] + adjustment))
    
    async def _classify_behavior_type(self, model: SystemBehaviorModel) -> IntelligenceBehaviorType:
        """Classify the system's behavior type based on observed patterns"""
        
        if len(model.learning_curve_data) < 5:
            return IntelligenceBehaviorType.ADAPTIVE
        
        # Analyze learning curve for behavior classification
        recent_performance = [perf for _, perf in model.learning_curve_data[-10:]]
        
        # Calculate variance to determine behavior type
        variance = np.var(recent_performance) if recent_performance else 0
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0] \
                if len(recent_performance) > 1 else 0
        
        # Classification logic
        if variance < 0.01 and abs(trend) < 0.01:
            return IntelligenceBehaviorType.DETERMINISTIC
        elif variance > 0.1:
            return IntelligenceBehaviorType.PROBABILISTIC
        elif trend > 0.05:
            return IntelligenceBehaviorType.LEARNING
        elif model.adaptation_rate > 0.7:
            return IntelligenceBehaviorType.ADAPTIVE
        else:
            return IntelligenceBehaviorType.REACTIVE
    
    async def _calculate_adaptation_rate(self, model: SystemBehaviorModel) -> float:
        """Calculate how quickly the system adapts to new conditions"""
        
        if len(model.learning_curve_data) < 10:
            return 0.5
        
        # Analyze adaptation speed from learning curve
        performance_data = [perf for _, perf in model.learning_curve_data]
        
        # Calculate moving average to smooth data
        window_size = min(5, len(performance_data) // 2)
        moving_avg = []
        for i in range(window_size, len(performance_data)):
            avg = np.mean(performance_data[i-window_size:i])
            moving_avg.append(avg)
        
        if len(moving_avg) < 2:
            return 0.5
        
        # Calculate rate of change in performance
        changes = [moving_avg[i] - moving_avg[i-1] for i in range(1, len(moving_avg))]
        adaptation_rate = np.mean([abs(change) for change in changes])
        
        return min(1.0, adaptation_rate * 2)  # Scale to 0-1 range
    
    async def _calculate_model_confidence(self, model: SystemBehaviorModel) -> float:
        """Calculate confidence in the behavior model"""
        
        # Base confidence on amount of data and consistency
        data_points = len(model.learning_curve_data)
        data_confidence = min(1.0, data_points / 50)
        
        # Calculate consistency in recent performance
        if len(model.learning_curve_data) >= 5:
            recent_performance = [perf for _, perf in model.learning_curve_data[-10:]]
            consistency = 1.0 - np.std(recent_performance)
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5
        
        # Combined confidence score
        confidence = (data_confidence * 0.6) + (consistency * 0.4)
        return confidence
    
    async def _calculate_performance_score(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate overall performance score for an interaction"""
        
        response_time = interaction_data.get('response_time', float('inf'))
        accuracy = interaction_data.get('accuracy', 0.0)
        success = interaction_data.get('success', False)
        resource_efficiency = interaction_data.get('resource_efficiency', 0.5)
        
        # Normalize response time (lower is better)
        time_score = 1.0 / (1.0 + response_time) if response_time > 0 else 1.0
        
        # Combine weighted scores
        performance_score = (
            self.learning_weights['response_time'] * time_score +
            self.learning_weights['accuracy'] * accuracy +
            self.learning_weights['reliability'] * (1.0 if success else 0.0) +
            self.learning_weights['resource_efficiency'] * resource_efficiency
        )
        
        return min(1.0, max(0.0, performance_score))
    
    def _calculate_percentile_key(self, response_time: float) -> str:
        """Calculate percentile key for response time distribution"""
        
        if response_time < 0.1:
            return "p10"
        elif response_time < 0.5:
            return "p25"
        elif response_time < 1.0:
            return "p50"
        elif response_time < 2.0:
            return "p75"
        elif response_time < 5.0:
            return "p90"
        else:
            return "p95"
    
    async def integrate_system(self, system_id: str, 
                             capability_profile: CapabilityProfile) -> SystemIntegrationStatus:
        """Integrate a new intelligence system with adaptive strategies"""
        
        # Create integration status
        integration_status = SystemIntegrationStatus(
            system_id=system_id,
            integration_stage="discovered",
            integration_timestamp=datetime.now(),
            last_interaction=datetime.now(),
            total_interactions=0,
            successful_interactions=0,
            integration_health=1.0,
            performance_trend="stable"
        )
        
        # Determine integration strategy based on capability profile
        strategy = await self._select_integration_strategy(capability_profile)
        
        # Execute integration
        success = await self._execute_integration(system_id, strategy)
        
        if success:
            integration_status.integration_stage = "integrated"
            integration_status.integration_health = 0.8
            self.logger.info(f"Successfully integrated system {system_id}")
        else:
            integration_status.issues.append("Integration failed")
            integration_status.integration_health = 0.3
            self.logger.error(f"Failed to integrate system {system_id}")
        
        self.integration_statuses[system_id] = integration_status
        return integration_status
    
    async def _select_integration_strategy(self, 
                                         capability_profile: CapabilityProfile) -> Dict[str, Any]:
        """Select optimal integration strategy based on system capabilities"""
        
        strategy = {
            'approach': 'gradual',
            'monitoring_level': 'high',
            'fallback_enabled': True,
            'optimization_target': 'performance'
        }
        
        # Analyze capabilities to determine strategy
        total_capability_score = sum(capability_profile.capabilities.values())
        
        if total_capability_score > 7.0:  # High capability system
            strategy['approach'] = 'direct'
            strategy['monitoring_level'] = 'medium'
            strategy['optimization_target'] = 'efficiency'
        elif capability_profile.reliability < 0.5:  # Low reliability
            strategy['approach'] = 'sandboxed'
            strategy['monitoring_level'] = 'maximum'
            strategy['fallback_enabled'] = True
        
        return strategy
    
    async def _execute_integration(self, system_id: str, 
                                 strategy: Dict[str, Any]) -> bool:
        """Execute the integration process"""
        
        try:
            # Simulate integration process based on strategy
            if strategy['approach'] == 'direct':
                # Direct integration - fastest but highest risk
                await asyncio.sleep(0.1)
                success_rate = 0.9
            elif strategy['approach'] == 'gradual':
                # Gradual integration - balanced approach
                await asyncio.sleep(0.2)
                success_rate = 0.95
            else:  # sandboxed
                # Sandboxed integration - slowest but safest
                await asyncio.sleep(0.3)
                success_rate = 0.98
            
            # Simulate success/failure
            import random
            return random.random() < success_rate
            
        except Exception as e:
            self.logger.error(f"Integration execution failed for {system_id}: {e}")
            return False
    
    async def adapt_integration(self, system_id: str, 
                              performance_data: Dict[str, Any]) -> bool:
        """Adapt integration strategy based on performance feedback"""
        
        if system_id not in self.integration_statuses:
            self.logger.warning(f"No integration status found for {system_id}")
            return False
        
        status = self.integration_statuses[system_id]
        
        # Analyze performance trends
        trend = await self._analyze_performance_trend(system_id, performance_data)
        
        # Update integration status
        status.last_interaction = datetime.now()
        status.total_interactions += 1
        
        if performance_data.get('success', False):
            status.successful_interactions += 1
        
        # Determine if adaptation is needed
        if trend == 'degrading' and status.integration_health > 0.5:
            # Apply optimization
            optimization = await self._select_optimization(system_id, performance_data)
            success = await self._apply_optimization(system_id, optimization)
            
            if success:
                status.optimizations_applied.append(optimization['type'])
                status.performance_trend = 'improving'
                status.integration_health = min(1.0, status.integration_health + 0.1)
                
                self.logger.info(f"Applied optimization {optimization['type']} to {system_id}")
                return True
        
        # Update health based on success rate
        success_rate = status.interaction_success_rate()
        status.integration_health = success_rate
        status.performance_trend = trend
        
        return False
    
    async def _analyze_performance_trend(self, system_id: str, 
                                       performance_data: Dict[str, Any]) -> str:
        """Analyze performance trend for a system"""
        
        if system_id not in self.behavior_models:
            return "stable"
        
        model = self.behavior_models[system_id]
        
        if len(model.learning_curve_data) < 5:
            return "stable"
        
        # Analyze recent performance trend
        recent_performance = [perf for _, perf in model.learning_curve_data[-5:]]
        
        # Calculate trend using linear regression
        x = list(range(len(recent_performance)))
        trend_slope = np.polyfit(x, recent_performance, 1)[0]
        
        if trend_slope > 0.02:
            return "improving"
        elif trend_slope < -0.02:
            return "degrading"
        else:
            return "stable"
    
    async def _select_optimization(self, system_id: str, 
                                 performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate optimization based on performance issues"""
        
        optimization = {
            'type': 'performance_tuning',
            'parameters': {},
            'expected_improvement': 0.1
        }
        
        # Analyze specific performance issues
        if performance_data.get('response_time', 0) > 2.0:
            optimization['type'] = 'caching_optimization'
            optimization['parameters'] = {'cache_size': 1000, 'ttl': 300}
            optimization['expected_improvement'] = 0.3
        elif performance_data.get('accuracy', 1.0) < 0.8:
            optimization['type'] = 'accuracy_enhancement'
            optimization['parameters'] = {'validation_level': 'high'}
            optimization['expected_improvement'] = 0.2
        elif performance_data.get('resource_efficiency', 1.0) < 0.5:
            optimization['type'] = 'resource_optimization'
            optimization['parameters'] = {'memory_limit': '512MB', 'cpu_limit': '1'}
            optimization['expected_improvement'] = 0.25
        
        return optimization
    
    async def _apply_optimization(self, system_id: str, 
                                optimization: Dict[str, Any]) -> bool:
        """Apply optimization to the system"""
        
        try:
            # Simulate optimization application
            self.logger.info(f"Applying {optimization['type']} to {system_id}")
            
            # Update integration status to track optimization
            if system_id in self.integration_statuses:
                status = self.integration_statuses[system_id]
                status.next_optimization_due = datetime.now() + timedelta(hours=24)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization to {system_id}: {e}")
            return False
    
    def get_integration_insights(self) -> Dict[str, Any]:
        """Get insights about integration patterns and performance"""
        
        insights = {
            'total_systems': len(self.integration_statuses),
            'integration_success_rate': 0.0,
            'average_health': 0.0,
            'common_issues': {},
            'optimization_effectiveness': {},
            'behavior_distribution': {}
        }
        
        if not self.integration_statuses:
            return insights
        
        # Calculate success rate
        successful = sum(1 for status in self.integration_statuses.values() 
                        if status.integration_stage in ['integrated', 'optimized'])
        insights['integration_success_rate'] = successful / len(self.integration_statuses)
        
        # Calculate average health
        total_health = sum(status.integration_health for status in self.integration_statuses.values())
        insights['average_health'] = total_health / len(self.integration_statuses)
        
        # Analyze common issues
        for status in self.integration_statuses.values():
            for issue in status.issues:
                if issue not in insights['common_issues']:
                    insights['common_issues'][issue] = 0
                insights['common_issues'][issue] += 1
        
        # Analyze behavior distribution
        for model in self.behavior_models.values():
            behavior_type = model.behavior_type.value
            if behavior_type not in insights['behavior_distribution']:
                insights['behavior_distribution'][behavior_type] = 0
            insights['behavior_distribution'][behavior_type] += 1
        
        return insights


def create_adaptive_integration_engine(learning_rate: float = 0.1, 
                                     adaptation_threshold: float = 0.8) -> AdaptiveIntegrationEngine:
    """Factory function to create AdaptiveIntegrationEngine instance"""
    
    return AdaptiveIntegrationEngine(
        learning_rate=learning_rate,
        adaptation_threshold=adaptation_threshold
    )