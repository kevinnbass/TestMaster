"""
Orchestration Strategy Selector
==============================

Revolutionary orchestration strategy selection and optimization engine.
Extracted from meta_intelligence_orchestrator.py for enterprise modular architecture.

Agent D Implementation - Hour 14-15: Revolutionary Intelligence Modularization
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

from .data_models import (
    OrchestrationStrategy, CapabilityType, CapabilityProfile,
    SystemBehaviorModel, OrchestrationPlan, OrchestrationEvent
)


@dataclass
class StrategyPerformanceRecord:
    """Record of strategy performance for learning"""
    strategy: OrchestrationStrategy
    objective_type: str
    system_count: int
    capability_mix: Dict[str, float]
    performance_metrics: Dict[str, float]
    execution_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class OrchestrationStrategySelector:
    """
    Revolutionary Orchestration Strategy Selector
    
    Uses machine learning and performance history to intelligently select
    the optimal orchestration strategy for any given objective and system combination.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        # Strategy performance tracking
        self.strategy_performance: Dict[OrchestrationStrategy, List[StrategyPerformanceRecord]] = {
            strategy: [] for strategy in OrchestrationStrategy
        }
        
        # System registry
        self.system_profiles: Dict[str, CapabilityProfile] = {}
        self.system_behaviors: Dict[str, SystemBehaviorModel] = {}
        
        # ML components
        self.strategy_classifier: Optional[RandomForestClassifier] = None
        self.performance_predictor: Optional[RandomForestClassifier] = None
        self.feature_scaler = StandardScaler()
        
        # Strategy rules and heuristics
        self.strategy_rules: Dict[str, Any] = self._initialize_strategy_rules()
        
        # Performance baselines
        self.performance_baselines: Dict[OrchestrationStrategy, Dict[str, float]] = {}
        
        # Adaptation tracking
        self.strategy_adaptations: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_strategy_rules(self) -> Dict[str, Any]:
        """Initialize strategy selection rules and heuristics"""
        
        return {
            OrchestrationStrategy.SEQUENTIAL: {
                'best_for': ['data_pipeline', 'processing_chain', 'transformation'],
                'system_count_range': (2, 5),
                'latency_tolerance': 'high',
                'accuracy_priority': 'medium',
                'resource_efficiency': 'high'
            },
            OrchestrationStrategy.PARALLEL: {
                'best_for': ['comparison', 'consensus', 'redundancy'],
                'system_count_range': (2, 10),
                'latency_tolerance': 'low',
                'accuracy_priority': 'high',
                'resource_efficiency': 'medium'
            },
            OrchestrationStrategy.PIPELINE: {
                'best_for': ['complex_processing', 'multi_stage', 'specialization'],
                'system_count_range': (3, 8),
                'latency_tolerance': 'medium',
                'accuracy_priority': 'high',
                'resource_efficiency': 'medium'
            },
            OrchestrationStrategy.ENSEMBLE: {
                'best_for': ['prediction', 'classification', 'decision_making'],
                'system_count_range': (3, 7),
                'latency_tolerance': 'medium',
                'accuracy_priority': 'very_high',
                'resource_efficiency': 'low'
            },
            OrchestrationStrategy.HIERARCHICAL: {
                'best_for': ['coordination', 'delegation', 'management'],
                'system_count_range': (4, 15),
                'latency_tolerance': 'medium',
                'accuracy_priority': 'medium',
                'resource_efficiency': 'high'
            },
            OrchestrationStrategy.ADAPTIVE: {
                'best_for': ['dynamic', 'uncertain', 'experimental'],
                'system_count_range': (2, 12),
                'latency_tolerance': 'variable',
                'accuracy_priority': 'variable',
                'resource_efficiency': 'variable'
            },
            OrchestrationStrategy.COMPETITIVE: {
                'best_for': ['optimization', 'best_result', 'racing'],
                'system_count_range': (2, 6),
                'latency_tolerance': 'low',
                'accuracy_priority': 'high',
                'resource_efficiency': 'low'
            },
            OrchestrationStrategy.COLLABORATIVE: {
                'best_for': ['synthesis', 'combination', 'cooperation'],
                'system_count_range': (3, 10),
                'latency_tolerance': 'medium',
                'accuracy_priority': 'high',
                'resource_efficiency': 'medium'
            }
        }
    
    async def register_system(self, capability_profile: CapabilityProfile,
                            behavior_model: Optional[SystemBehaviorModel] = None):
        """Register a system for strategy selection"""
        
        system_id = capability_profile.system_id
        self.system_profiles[system_id] = capability_profile
        
        if behavior_model:
            self.system_behaviors[system_id] = behavior_model
        
        self.logger.debug(f"Registered system {system_id} for strategy selection")
    
    async def select_strategy(self, participating_systems: List[str],
                            objective: str,
                            requirements: Dict[str, Any],
                            constraints: Optional[Dict[str, Any]] = None) -> OrchestrationStrategy:
        """Select optimal orchestration strategy"""
        
        self.logger.info(f"🎯 Selecting strategy for {len(participating_systems)} systems: {objective}")
        
        # Analyze context
        context = await self._analyze_orchestration_context(
            participating_systems, objective, requirements, constraints
        )
        
        # Get strategy candidates
        candidates = await self._get_strategy_candidates(context)
        
        # Score strategies
        strategy_scores = {}
        for strategy in candidates:
            score = await self._score_strategy(strategy, context)
            strategy_scores[strategy] = score
        
        # Select best strategy
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            best_score = strategy_scores[best_strategy]
            
            self.logger.info(f"✅ Selected strategy: {best_strategy.value} (score: {best_score:.2f})")
            return best_strategy
        else:
            # Fallback to adaptive strategy
            self.logger.warning("No suitable strategy found, defaulting to ADAPTIVE")
            return OrchestrationStrategy.ADAPTIVE
    
    async def _analyze_orchestration_context(self, participating_systems: List[str],
                                           objective: str,
                                           requirements: Dict[str, Any],
                                           constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the context for orchestration strategy selection"""
        
        context = {
            'system_count': len(participating_systems),
            'objective_type': await self._classify_objective_type(objective),
            'capability_mix': await self._analyze_capability_mix(participating_systems),
            'performance_requirements': await self._extract_performance_requirements(requirements),
            'resource_constraints': await self._extract_resource_constraints(constraints),
            'system_characteristics': await self._analyze_system_characteristics(participating_systems),
            'complexity_level': await self._assess_complexity_level(objective, requirements)
        }
        
        return context
    
    async def _classify_objective_type(self, objective: str) -> str:
        """Classify the type of objective"""
        
        objective_lower = objective.lower()
        
        # Keyword-based classification
        if any(word in objective_lower for word in ['analyze', 'process', 'transform']):
            return 'processing'
        elif any(word in objective_lower for word in ['predict', 'forecast', 'estimate']):
            return 'prediction'
        elif any(word in objective_lower for word in ['classify', 'categorize', 'identify']):
            return 'classification'
        elif any(word in objective_lower for word in ['optimize', 'improve', 'enhance']):
            return 'optimization'
        elif any(word in objective_lower for word in ['compare', 'evaluate', 'assess']):
            return 'comparison'
        elif any(word in objective_lower for word in ['generate', 'create', 'produce']):
            return 'generation'
        elif any(word in objective_lower for word in ['coordinate', 'manage', 'orchestrate']):
            return 'coordination'
        else:
            return 'general'
    
    async def _analyze_capability_mix(self, participating_systems: List[str]) -> Dict[str, float]:
        """Analyze the mix of capabilities across participating systems"""
        
        capability_totals = {cap.value: 0.0 for cap in CapabilityType}
        capability_counts = {cap.value: 0 for cap in CapabilityType}
        
        for system_id in participating_systems:
            if system_id in self.system_profiles:
                profile = self.system_profiles[system_id]
                for cap_type, score in profile.capabilities.items():
                    if score > 0.5:  # Consider only significant capabilities
                        capability_totals[cap_type.value] += score
                        capability_counts[cap_type.value] += 1
        
        # Calculate averages and distribution
        capability_mix = {}
        total_systems = len(participating_systems)
        
        for cap_type in CapabilityType:
            cap_value = cap_type.value
            if capability_counts[cap_value] > 0:
                avg_score = capability_totals[cap_value] / capability_counts[cap_value]
                coverage = capability_counts[cap_value] / total_systems
                capability_mix[cap_value] = avg_score * coverage
            else:
                capability_mix[cap_value] = 0.0
        
        return capability_mix
    
    async def _extract_performance_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance requirements from requirements specification"""
        
        perf_requirements = {
            'max_latency': requirements.get('max_latency', float('inf')),
            'min_accuracy': requirements.get('min_accuracy', 0.0),
            'min_throughput': requirements.get('min_throughput', 0.0),
            'max_cost': requirements.get('max_cost', float('inf')),
            'reliability_requirement': requirements.get('reliability', 0.9),
            'priority': requirements.get('priority', 'medium')
        }
        
        # Infer implicit requirements
        if 'real_time' in str(requirements).lower():
            perf_requirements['max_latency'] = min(perf_requirements['max_latency'], 1.0)
        
        if 'high_accuracy' in str(requirements).lower():
            perf_requirements['min_accuracy'] = max(perf_requirements['min_accuracy'], 0.9)
        
        if 'critical' in str(requirements).lower():
            perf_requirements['reliability_requirement'] = max(
                perf_requirements['reliability_requirement'], 0.99
            )
        
        return perf_requirements
    
    async def _extract_resource_constraints(self, 
                                          constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract resource constraints"""
        
        if not constraints:
            constraints = {}
        
        resource_constraints = {
            'max_cpu': constraints.get('max_cpu', float('inf')),
            'max_memory': constraints.get('max_memory', float('inf')),
            'max_cost': constraints.get('max_cost', float('inf')),
            'max_execution_time': constraints.get('max_execution_time', float('inf')),
            'allowed_strategies': constraints.get('allowed_strategies', list(OrchestrationStrategy))
        }
        
        return resource_constraints
    
    async def _analyze_system_characteristics(self, participating_systems: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of participating systems"""
        
        characteristics = {
            'avg_processing_time': 0.0,
            'avg_accuracy': 0.0,
            'avg_reliability': 0.0,
            'total_cost': 0.0,
            'capability_diversity': 0.0,
            'input_output_compatibility': 0.0
        }
        
        if not participating_systems:
            return characteristics
        
        # Analyze system profiles
        total_processing_time = 0.0
        total_accuracy = 0.0
        total_reliability = 0.0
        total_cost = 0.0
        
        all_capabilities = set()
        all_input_types = set()
        all_output_types = set()
        
        for system_id in participating_systems:
            if system_id in self.system_profiles:
                profile = self.system_profiles[system_id]
                
                total_processing_time += profile.processing_time
                total_accuracy += profile.accuracy
                total_reliability += profile.reliability
                total_cost += profile.cost_per_operation
                
                all_capabilities.update(profile.capabilities.keys())
                all_input_types.update(profile.input_types)
                all_output_types.update(profile.output_types)
        
        count = len(participating_systems)
        characteristics['avg_processing_time'] = total_processing_time / count
        characteristics['avg_accuracy'] = total_accuracy / count
        characteristics['avg_reliability'] = total_reliability / count
        characteristics['total_cost'] = total_cost
        
        # Calculate capability diversity
        unique_capabilities = len(all_capabilities)
        max_possible_capabilities = len(CapabilityType)
        characteristics['capability_diversity'] = unique_capabilities / max_possible_capabilities
        
        # Calculate input/output compatibility
        compatibility_score = len(all_input_types & all_output_types) / max(1, len(all_input_types | all_output_types))
        characteristics['input_output_compatibility'] = compatibility_score
        
        return characteristics
    
    async def _assess_complexity_level(self, objective: str, requirements: Dict[str, Any]) -> str:
        """Assess the complexity level of the orchestration"""
        
        complexity_indicators = 0
        
        # Objective complexity
        if len(objective.split()) > 10:
            complexity_indicators += 1
        
        if any(word in objective.lower() for word in ['complex', 'advanced', 'sophisticated']):
            complexity_indicators += 2
        
        # Requirements complexity
        if len(requirements) > 5:
            complexity_indicators += 1
        
        if any(isinstance(v, dict) for v in requirements.values()):
            complexity_indicators += 1
        
        if requirements.get('min_accuracy', 0) > 0.9:
            complexity_indicators += 1
        
        if requirements.get('max_latency', float('inf')) < 1.0:
            complexity_indicators += 1
        
        # Classify complexity
        if complexity_indicators <= 2:
            return 'low'
        elif complexity_indicators <= 4:
            return 'medium'
        else:
            return 'high'
    
    async def _get_strategy_candidates(self, context: Dict[str, Any]) -> List[OrchestrationStrategy]:
        """Get candidate strategies based on context"""
        
        candidates = []
        
        system_count = context['system_count']
        objective_type = context['objective_type']
        complexity_level = context['complexity_level']
        
        # Apply rule-based filtering
        for strategy, rules in self.strategy_rules.items():
            # Check system count range
            min_systems, max_systems = rules['system_count_range']
            if not (min_systems <= system_count <= max_systems):
                continue
            
            # Check if strategy is good for objective type
            if objective_type in rules['best_for']:
                candidates.append(strategy)
                continue
            
            # Check complexity compatibility
            if complexity_level == 'high' and strategy in [
                OrchestrationStrategy.PIPELINE,
                OrchestrationStrategy.HIERARCHICAL,
                OrchestrationStrategy.ADAPTIVE
            ]:
                candidates.append(strategy)
            elif complexity_level == 'low' and strategy in [
                OrchestrationStrategy.SEQUENTIAL,
                OrchestrationStrategy.PARALLEL
            ]:
                candidates.append(strategy)
            elif complexity_level == 'medium':
                candidates.append(strategy)
        
        # Ensure we have at least some candidates
        if not candidates:
            candidates = [
                OrchestrationStrategy.SEQUENTIAL,
                OrchestrationStrategy.PARALLEL,
                OrchestrationStrategy.ADAPTIVE
            ]
        
        return candidates
    
    async def _score_strategy(self, strategy: OrchestrationStrategy, 
                            context: Dict[str, Any]) -> float:
        """Score a strategy based on context and historical performance"""
        
        score = 0.0
        
        # Rule-based scoring
        rule_score = await self._calculate_rule_based_score(strategy, context)
        score += rule_score * 0.4
        
        # Historical performance scoring
        historical_score = await self._calculate_historical_score(strategy, context)
        score += historical_score * 0.3
        
        # ML-based scoring (if trained)
        ml_score = await self._calculate_ml_score(strategy, context)
        score += ml_score * 0.2
        
        # Resource efficiency scoring
        efficiency_score = await self._calculate_efficiency_score(strategy, context)
        score += efficiency_score * 0.1
        
        return score
    
    async def _calculate_rule_based_score(self, strategy: OrchestrationStrategy,
                                        context: Dict[str, Any]) -> float:
        """Calculate rule-based score for a strategy"""
        
        if strategy not in self.strategy_rules:
            return 0.5
        
        rules = self.strategy_rules[strategy]
        score = 0.0
        
        # System count compatibility
        min_systems, max_systems = rules['system_count_range']
        system_count = context['system_count']
        
        if min_systems <= system_count <= max_systems:
            score += 0.3
        elif system_count < min_systems:
            score += 0.1
        else:  # system_count > max_systems
            score += 0.2
        
        # Objective type compatibility
        objective_type = context['objective_type']
        if objective_type in rules['best_for']:
            score += 0.4
        
        # Performance requirements compatibility
        perf_requirements = context['performance_requirements']
        
        if rules['latency_tolerance'] == 'low' and perf_requirements['max_latency'] < 2.0:
            score += 0.2
        elif rules['latency_tolerance'] == 'high' and perf_requirements['max_latency'] >= 5.0:
            score += 0.2
        
        if rules['accuracy_priority'] == 'very_high' and perf_requirements['min_accuracy'] > 0.9:
            score += 0.1
        elif rules['accuracy_priority'] == 'high' and perf_requirements['min_accuracy'] > 0.8:
            score += 0.1
        
        return min(1.0, score)
    
    async def _calculate_historical_score(self, strategy: OrchestrationStrategy,
                                        context: Dict[str, Any]) -> float:
        """Calculate score based on historical performance"""
        
        if strategy not in self.strategy_performance:
            return 0.5
        
        performance_records = self.strategy_performance[strategy]
        
        if not performance_records:
            return 0.5
        
        # Find similar contexts in history
        similar_records = []
        objective_type = context['objective_type']
        system_count = context['system_count']
        
        for record in performance_records:
            if (record.objective_type == objective_type and
                abs(record.system_count - system_count) <= 2):
                similar_records.append(record)
        
        if not similar_records:
            # Use all records as fallback
            similar_records = performance_records[-10:]  # Recent records
        
        # Calculate average success rate and performance
        success_rate = sum(1 for r in similar_records if r.success) / len(similar_records)
        avg_performance = np.mean([
            np.mean(list(r.performance_metrics.values())) for r in similar_records
        ])
        
        # Combine success rate and performance
        historical_score = (success_rate * 0.6) + (avg_performance * 0.4)
        
        return historical_score
    
    async def _calculate_ml_score(self, strategy: OrchestrationStrategy,
                                context: Dict[str, Any]) -> float:
        """Calculate ML-based score (if model is trained)"""
        
        if not self.strategy_classifier:
            return 0.5
        
        try:
            # Prepare features
            features = await self._extract_ml_features(context)
            features_scaled = self.feature_scaler.transform([features])
            
            # Get prediction probabilities
            probabilities = self.strategy_classifier.predict_proba(features_scaled)[0]
            
            # Find probability for this strategy
            strategy_classes = self.strategy_classifier.classes_
            if strategy in strategy_classes:
                strategy_index = list(strategy_classes).index(strategy)
                return probabilities[strategy_index]
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error in ML scoring: {e}")
            return 0.5
    
    async def _calculate_efficiency_score(self, strategy: OrchestrationStrategy,
                                        context: Dict[str, Any]) -> float:
        """Calculate resource efficiency score"""
        
        rules = self.strategy_rules.get(strategy, {})
        resource_efficiency = rules.get('resource_efficiency', 'medium')
        
        system_characteristics = context['system_characteristics']
        total_cost = system_characteristics['total_cost']
        
        # Base efficiency score from rules
        efficiency_map = {
            'very_high': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'very_low': 0.2
        }
        
        base_efficiency = efficiency_map.get(resource_efficiency, 0.6)
        
        # Adjust based on actual resource usage
        cost_factor = 1.0 / (1.0 + total_cost)  # Lower cost = higher efficiency
        
        efficiency_score = (base_efficiency * 0.7) + (cost_factor * 0.3)
        
        return efficiency_score
    
    async def _extract_ml_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features for ML models"""
        
        features = []
        
        # Basic context features
        features.append(context['system_count'])
        features.append(hash(context['objective_type']) % 1000 / 1000.0)  # Normalized hash
        features.append({'low': 0.2, 'medium': 0.5, 'high': 0.8}.get(context['complexity_level'], 0.5))
        
        # Capability mix features
        capability_mix = context['capability_mix']
        for cap_type in CapabilityType:
            features.append(capability_mix.get(cap_type.value, 0.0))
        
        # Performance requirements
        perf_req = context['performance_requirements']
        features.extend([
            1.0 / (1.0 + perf_req.get('max_latency', 1.0)),  # Normalized latency
            perf_req.get('min_accuracy', 0.0),
            perf_req.get('reliability_requirement', 0.0)
        ])
        
        # System characteristics
        sys_char = context['system_characteristics']
        features.extend([
            1.0 / (1.0 + sys_char.get('avg_processing_time', 1.0)),
            sys_char.get('avg_accuracy', 0.0),
            sys_char.get('avg_reliability', 0.0),
            sys_char.get('capability_diversity', 0.0),
            sys_char.get('input_output_compatibility', 0.0)
        ])
        
        return features
    
    async def record_strategy_performance(self, orchestration_plan: OrchestrationPlan,
                                        performance_metrics: Dict[str, float],
                                        execution_time: float,
                                        success: bool):
        """Record strategy performance for learning"""
        
        strategy = orchestration_plan.orchestration_strategy
        
        # Extract capability mix from participating systems
        capability_mix = {}
        for system_id in orchestration_plan.participating_systems:
            if system_id in self.system_profiles:
                profile = self.system_profiles[system_id]
                for cap_type, score in profile.capabilities.items():
                    if cap_type.value not in capability_mix:
                        capability_mix[cap_type.value] = 0.0
                    capability_mix[cap_type.value] += score
        
        # Normalize by system count
        system_count = len(orchestration_plan.participating_systems)
        for cap_type in capability_mix:
            capability_mix[cap_type] /= system_count
        
        # Create performance record
        record = StrategyPerformanceRecord(
            strategy=strategy,
            objective_type='general',  # Would need to be extracted from plan
            system_count=system_count,
            capability_mix=capability_mix,
            performance_metrics=performance_metrics,
            execution_time=execution_time,
            success=success
        )
        
        # Store record
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        self.strategy_performance[strategy].append(record)
        
        # Limit history size
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
        
        # Update baselines
        await self._update_performance_baselines(strategy, record)
        
        self.logger.debug(f"Recorded performance for strategy {strategy.value}: "
                         f"Success={success}, Time={execution_time:.2f}s")
    
    async def _update_performance_baselines(self, strategy: OrchestrationStrategy,
                                          record: StrategyPerformanceRecord):
        """Update performance baselines for a strategy"""
        
        if strategy not in self.performance_baselines:
            self.performance_baselines[strategy] = {}
        
        baselines = self.performance_baselines[strategy]
        
        # Update baselines with exponential moving average
        alpha = 0.1  # Learning rate
        
        for metric, value in record.performance_metrics.items():
            if metric not in baselines:
                baselines[metric] = value
            else:
                baselines[metric] = (1 - alpha) * baselines[metric] + alpha * value
        
        # Update execution time baseline
        if 'execution_time' not in baselines:
            baselines['execution_time'] = record.execution_time
        else:
            baselines['execution_time'] = (1 - alpha) * baselines['execution_time'] + \
                                        alpha * record.execution_time
        
        # Update success rate baseline
        if 'success_rate' not in baselines:
            baselines['success_rate'] = 1.0 if record.success else 0.0
        else:
            success_value = 1.0 if record.success else 0.0
            baselines['success_rate'] = (1 - alpha) * baselines['success_rate'] + \
                                      alpha * success_value
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get insights about strategy selection and performance"""
        
        insights = {
            'strategy_usage_count': {},
            'strategy_success_rates': {},
            'average_execution_times': {},
            'performance_baselines': self.performance_baselines.copy(),
            'total_records': 0,
            'most_successful_strategy': None,
            'fastest_strategy': None
        }
        
        # Analyze strategy usage and performance
        strategy_stats = {}
        
        for strategy, records in self.strategy_performance.items():
            if not records:
                continue
            
            insights['strategy_usage_count'][strategy.value] = len(records)
            insights['total_records'] += len(records)
            
            # Calculate success rate
            successes = sum(1 for r in records if r.success)
            success_rate = successes / len(records)
            insights['strategy_success_rates'][strategy.value] = success_rate
            
            # Calculate average execution time
            avg_time = np.mean([r.execution_time for r in records])
            insights['average_execution_times'][strategy.value] = avg_time
            
            strategy_stats[strategy] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'usage_count': len(records)
            }
        
        # Find best strategies
        if strategy_stats:
            # Most successful strategy
            best_success = max(strategy_stats.items(), key=lambda x: x[1]['success_rate'])
            insights['most_successful_strategy'] = best_success[0].value
            
            # Fastest strategy
            fastest = min(strategy_stats.items(), key=lambda x: x[1]['avg_time'])
            insights['fastest_strategy'] = fastest[0].value
        
        return insights


def create_strategy_selector(learning_rate: float = 0.1) -> OrchestrationStrategySelector:
    """Factory function to create OrchestrationStrategySelector instance"""
    
    return OrchestrationStrategySelector(learning_rate=learning_rate)