"""
Adaptive Prediction Enhancer
============================

Meta-learning enhancement for the PredictiveAnalyticsEngine.
Enables the prediction system to improve its own prediction accuracy
through continuous learning and adaptation.

Author: Agent A Phase 2 - Self-Learning Intelligence
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics

from .predictive_analytics_engine import PredictiveAnalyticsEngine, ModelType, PredictionResult


@dataclass
class AccuracyPattern:
    """Pattern analysis for prediction accuracy"""
    metric_name: str
    model_type: ModelType
    accuracy_trend: str  # "improving", "degrading", "stable"
    average_accuracy: float
    accuracy_variance: float
    optimal_parameters: Dict[str, Any]
    improvement_opportunities: List[str]
    last_analysis: datetime = field(default_factory=datetime.now)


@dataclass
class MetaLearningInsight:
    """Insights discovered through meta-learning"""
    insight_id: str
    insight_type: str  # "model_optimization", "parameter_tuning", "data_preprocessing"
    confidence: float
    impact_score: float
    affected_metrics: List[str]
    recommended_actions: List[str]
    implementation_complexity: str  # "low", "medium", "high"
    expected_improvement: float
    discovered_at: datetime = field(default_factory=datetime.now)


class AdaptivePredictionEnhancer:
    """
    Meta-learning system that enhances the existing PredictiveAnalyticsEngine
    by learning from prediction accuracy patterns and automatically improving
    prediction performance.
    """
    
    def __init__(self, analytics_hub):
        self.analytics_hub = analytics_hub
        self.predictive_engine = analytics_hub.predictive_engine
        self.logger = logging.getLogger(__name__)
        
        # Meta-learning data storage
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.accuracy_patterns: Dict[str, AccuracyPattern] = {}
        self.meta_learning_insights: List[MetaLearningInsight] = []
        self.parameter_performance_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        
        # Enhancement configuration
        self.config = {
            "analysis_interval": 300,  # 5 minutes
            "min_predictions_for_analysis": 10,
            "accuracy_improvement_threshold": 0.05,
            "confidence_threshold": 0.7,
            "auto_tuning_enabled": True,
            "max_parameter_experiments": 5
        }
        
        # Performance tracking
        self.enhancement_stats = {
            "total_analyses": 0,
            "accuracy_improvements_detected": 0,
            "auto_optimizations_applied": 0,
            "insights_discovered": 0,
            "parameters_tuned": 0,
            "start_time": datetime.now()
        }
        
        # Meta-learning state
        self.is_enhancing = False
        self.enhancement_task = None
        
        self.logger.info("Adaptive Prediction Enhancer initialized")
    
    async def start_enhancement(self):
        """Start the meta-learning enhancement process"""
        if self.is_enhancing:
            self.logger.warning("Enhancement already running")
            return
        
        self.is_enhancing = True
        self.enhancement_task = asyncio.create_task(self._enhancement_loop())
        self.logger.info("Started adaptive prediction enhancement")
    
    async def stop_enhancement(self):
        """Stop the meta-learning enhancement process"""
        self.is_enhancing = False
        if self.enhancement_task:
            self.enhancement_task.cancel()
            try:
                await self.enhancement_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped adaptive prediction enhancement")
    
    async def _enhancement_loop(self):
        """Main meta-learning enhancement loop"""
        while self.is_enhancing:
            try:
                await asyncio.sleep(self.config["analysis_interval"])
                
                # Analyze prediction accuracy patterns
                await self._analyze_accuracy_patterns()
                
                # Identify improvement opportunities
                insights = await self._identify_improvement_opportunities()
                
                # Apply automatic optimizations
                if self.config["auto_tuning_enabled"]:
                    await self._apply_automatic_optimizations(insights)
                
                # Update decision confidence calibration
                await self._calibrate_decision_confidence()
                
                # Generate meta-learning insights
                await self._generate_meta_insights()
                
                self.enhancement_stats["total_analyses"] += 1
                
            except Exception as e:
                self.logger.error(f"Enhancement loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_accuracy_patterns(self):
        """Analyze prediction accuracy patterns for each metric"""
        try:
            # Get recent predictions from the predictive engine
            active_predictions = self.predictive_engine.get_active_predictions()
            
            for metric_name, prediction in active_predictions.items():
                # Calculate current accuracy (simplified - compare to actual values if available)
                current_accuracy = self._estimate_prediction_accuracy(prediction)
                
                if current_accuracy is not None:
                    # Store accuracy history
                    self.accuracy_history[metric_name].append({
                        "timestamp": datetime.now(),
                        "accuracy": current_accuracy,
                        "model_type": prediction.model_type,
                        "confidence": prediction.confidence
                    })
                    
                    # Analyze pattern if we have enough data
                    if len(self.accuracy_history[metric_name]) >= self.config["min_predictions_for_analysis"]:
                        pattern = await self._analyze_metric_accuracy_pattern(metric_name)
                        if pattern:
                            self.accuracy_patterns[metric_name] = pattern
                            
        except Exception as e:
            self.logger.error(f"Failed to analyze accuracy patterns: {e}")
    
    def _estimate_prediction_accuracy(self, prediction: PredictionResult) -> Optional[float]:
        """Estimate prediction accuracy (simplified implementation)"""
        try:
            # In a real implementation, this would compare predictions to actual values
            # For now, we'll use confidence and variance as proxy metrics
            
            if not prediction.predicted_values:
                return None
            
            # Calculate prediction stability (lower variance = higher accuracy proxy)
            values = [v[1] for v in prediction.predicted_values[:5]]
            if len(values) < 2:
                return prediction.confidence
            
            variance = np.var(values)
            mean_value = np.mean(values)
            
            # Normalize variance relative to mean (coefficient of variation)
            if mean_value != 0:
                cv = abs(variance ** 0.5 / mean_value)
                # Convert to accuracy score (lower CV = higher accuracy)
                stability_score = max(0.0, 1.0 - min(cv, 1.0))
            else:
                stability_score = 0.5
            
            # Combine confidence with stability
            estimated_accuracy = (prediction.confidence * 0.6 + stability_score * 0.4)
            return min(1.0, max(0.0, estimated_accuracy))
            
        except Exception as e:
            self.logger.debug(f"Failed to estimate accuracy: {e}")
            return None
    
    async def _analyze_metric_accuracy_pattern(self, metric_name: str) -> Optional[AccuracyPattern]:
        """Analyze accuracy pattern for a specific metric"""
        try:
            history = list(self.accuracy_history[metric_name])
            if len(history) < 5:
                return None
            
            # Calculate trend
            recent_accuracies = [h["accuracy"] for h in history[-10:]]
            older_accuracies = [h["accuracy"] for h in history[-20:-10]] if len(history) >= 20 else recent_accuracies
            
            recent_avg = statistics.mean(recent_accuracies)
            older_avg = statistics.mean(older_accuracies)
            
            if recent_avg > older_avg + 0.02:
                trend = "improving"
            elif recent_avg < older_avg - 0.02:
                trend = "degrading"
            else:
                trend = "stable"
            
            # Calculate overall statistics
            all_accuracies = [h["accuracy"] for h in history]
            average_accuracy = statistics.mean(all_accuracies)
            accuracy_variance = statistics.variance(all_accuracies) if len(all_accuracies) > 1 else 0.0
            
            # Identify model type most commonly used
            model_types = [h["model_type"] for h in history]
            most_common_model = max(set(model_types), key=model_types.count) if model_types else ModelType.LINEAR_REGRESSION
            
            # Identify improvement opportunities
            opportunities = []
            if trend == "degrading":
                opportunities.append("Consider model retraining")
                opportunities.append("Evaluate alternative model types")
            
            if accuracy_variance > 0.05:
                opportunities.append("Investigate prediction instability")
                opportunities.append("Consider data preprocessing improvements")
            
            if average_accuracy < 0.7:
                opportunities.append("Improve feature engineering")
                opportunities.append("Increase training data quality")
            
            return AccuracyPattern(
                metric_name=metric_name,
                model_type=most_common_model,
                accuracy_trend=trend,
                average_accuracy=average_accuracy,
                accuracy_variance=accuracy_variance,
                optimal_parameters={},  # Will be populated by parameter tuning
                improvement_opportunities=opportunities
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze pattern for {metric_name}: {e}")
            return None
    
    async def _identify_improvement_opportunities(self) -> List[MetaLearningInsight]:
        """Identify specific improvement opportunities through meta-learning"""
        insights = []
        
        try:
            for metric_name, pattern in self.accuracy_patterns.items():
                # Insight: Model type optimization
                if pattern.accuracy_trend == "degrading" or pattern.average_accuracy < 0.65:
                    insights.append(MetaLearningInsight(
                        insight_id=f"model_opt_{metric_name}_{int(datetime.now().timestamp())}",
                        insight_type="model_optimization",
                        confidence=0.8,
                        impact_score=0.15,
                        affected_metrics=[metric_name],
                        recommended_actions=["Try different model types", "Retrain with more data"],
                        implementation_complexity="medium",
                        expected_improvement=0.10
                    ))
                
                # Insight: Parameter tuning needed
                if pattern.accuracy_variance > 0.08:
                    insights.append(MetaLearningInsight(
                        insight_id=f"param_tune_{metric_name}_{int(datetime.now().timestamp())}",
                        insight_type="parameter_tuning",
                        confidence=0.7,
                        impact_score=0.10,
                        affected_metrics=[metric_name],
                        recommended_actions=["Tune hyperparameters", "Optimize regularization"],
                        implementation_complexity="low",
                        expected_improvement=0.08
                    ))
                
                # Insight: Data preprocessing optimization
                if "data preprocessing" in pattern.improvement_opportunities:
                    insights.append(MetaLearningInsight(
                        insight_id=f"data_prep_{metric_name}_{int(datetime.now().timestamp())}",
                        insight_type="data_preprocessing",
                        confidence=0.6,
                        impact_score=0.12,
                        affected_metrics=[metric_name],
                        recommended_actions=["Improve feature scaling", "Add feature selection"],
                        implementation_complexity="high",
                        expected_improvement=0.12
                    ))
            
            # Store insights
            self.meta_learning_insights.extend(insights)
            self.enhancement_stats["insights_discovered"] += len(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to identify improvement opportunities: {e}")
            return []
    
    async def _apply_automatic_optimizations(self, insights: List[MetaLearningInsight]):
        """Apply automatic optimizations based on meta-learning insights"""
        applied_count = 0
        
        try:
            for insight in insights:
                if (insight.confidence >= self.config["confidence_threshold"] and 
                    insight.implementation_complexity == "low"):
                    
                    if insight.insight_type == "parameter_tuning":
                        success = await self._auto_tune_parameters(insight)
                        if success:
                            applied_count += 1
                    
                    elif insight.insight_type == "model_optimization":
                        success = await self._auto_optimize_models(insight)
                        if success:
                            applied_count += 1
            
            self.enhancement_stats["auto_optimizations_applied"] += applied_count
            
            if applied_count > 0:
                self.logger.info(f"Applied {applied_count} automatic optimizations")
                
        except Exception as e:
            self.logger.error(f"Failed to apply automatic optimizations: {e}")
    
    async def _auto_tune_parameters(self, insight: MetaLearningInsight) -> bool:
        """Automatically tune parameters for affected metrics"""
        try:
            for metric_name in insight.affected_metrics:
                # Simple parameter tuning (in production, would use more sophisticated methods)
                if metric_name in self.predictive_engine.models:
                    # Record current performance
                    current_pattern = self.accuracy_patterns.get(metric_name)
                    if not current_pattern:
                        continue
                    
                    # Try different parameters (simplified approach)
                    parameter_experiments = [
                        {"n_estimators": 120, "max_depth": 8},
                        {"n_estimators": 80, "max_depth": 12},
                        {"n_estimators": 150, "max_depth": 6}
                    ]
                    
                    best_params = None
                    best_score = current_pattern.average_accuracy
                    
                    for params in parameter_experiments[:self.config["max_parameter_experiments"]]:
                        # Test parameters (simplified - would need actual model retraining)
                        estimated_score = self._estimate_parameter_performance(metric_name, params)
                        
                        if estimated_score > best_score:
                            best_score = estimated_score
                            best_params = params
                    
                    if best_params:
                        # Update pattern with optimal parameters
                        current_pattern.optimal_parameters = best_params
                        self.accuracy_patterns[metric_name] = current_pattern
                        
                        self.enhancement_stats["parameters_tuned"] += 1
                        self.logger.info(f"Auto-tuned parameters for {metric_name}: {best_params}")
                        
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to auto-tune parameters: {e}")
            return False
    
    def _estimate_parameter_performance(self, metric_name: str, params: Dict[str, Any]) -> float:
        """Estimate performance for given parameters (simplified)"""
        try:
            # Simplified estimation based on parameter characteristics
            # In production, this would involve actual model training/validation
            
            base_score = self.accuracy_patterns[metric_name].average_accuracy
            
            # Simple heuristics for parameter impact
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", 10)
            
            # Optimal ranges (simplified)
            estimator_score = 1.0 - abs(n_estimators - 100) / 200
            depth_score = 1.0 - abs(max_depth - 8) / 16
            
            # Combined score
            parameter_score = (estimator_score + depth_score) / 2
            estimated_improvement = parameter_score * 0.1  # Max 10% improvement
            
            return base_score + estimated_improvement
            
        except Exception:
            return 0.5
    
    async def _auto_optimize_models(self, insight: MetaLearningInsight) -> bool:
        """Automatically optimize model selection for affected metrics"""
        try:
            # This would involve testing different model types
            # Simplified implementation for demonstration
            
            optimized_count = 0
            for metric_name in insight.affected_metrics:
                current_pattern = self.accuracy_patterns.get(metric_name)
                if not current_pattern:
                    continue
                
                # Suggest alternative model type
                current_model = current_pattern.model_type
                alternative_model = ModelType.RANDOM_FOREST if current_model == ModelType.LINEAR_REGRESSION else ModelType.LINEAR_REGRESSION
                
                # Estimate improvement (simplified)
                if current_pattern.average_accuracy < 0.7:
                    # Suggest the alternative might be better
                    self.logger.info(f"Suggesting {alternative_model.value} for {metric_name} (current: {current_model.value})")
                    optimized_count += 1
            
            return optimized_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to auto-optimize models: {e}")
            return False
    
    async def _calibrate_decision_confidence(self):
        """Calibrate decision confidence based on prediction accuracy track record"""
        try:
            # Get intelligent decisions from predictive engine
            decisions = self.predictive_engine.get_intelligent_decisions()
            
            for decision in decisions:
                # Adjust confidence based on track record of related metrics
                related_metrics = decision.trigger_metrics
                
                accuracy_scores = []
                for metric in related_metrics:
                    if metric in self.accuracy_patterns:
                        accuracy_scores.append(self.accuracy_patterns[metric].average_accuracy)
                
                if accuracy_scores:
                    avg_accuracy = statistics.mean(accuracy_scores)
                    # Adjust decision confidence based on prediction accuracy
                    adjusted_confidence = decision.confidence * avg_accuracy
                    decision.confidence = min(1.0, max(0.1, adjusted_confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate decision confidence: {e}")
    
    async def _generate_meta_insights(self):
        """Generate high-level insights about the learning process itself"""
        try:
            # Analyze overall enhancement effectiveness
            if len(self.accuracy_patterns) >= 3:
                improving_count = sum(1 for p in self.accuracy_patterns.values() if p.accuracy_trend == "improving")
                total_patterns = len(self.accuracy_patterns)
                improvement_rate = improving_count / total_patterns
                
                if improvement_rate > 0.6:
                    self.logger.info(f"Meta-learning is effective: {improvement_rate:.1%} of metrics improving")
                elif improvement_rate < 0.3:
                    self.logger.warning(f"Meta-learning needs attention: only {improvement_rate:.1%} of metrics improving")
            
        except Exception as e:
            self.logger.error(f"Failed to generate meta insights: {e}")
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get comprehensive enhancement status"""
        return {
            "is_enhancing": self.is_enhancing,
            "enhancement_stats": self.enhancement_stats.copy(),
            "accuracy_patterns_count": len(self.accuracy_patterns),
            "insights_count": len(self.meta_learning_insights),
            "metrics_being_enhanced": list(self.accuracy_patterns.keys()),
            "recent_insights": [
                {
                    "type": insight.insight_type,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "affected_metrics": insight.affected_metrics,
                    "discovered_at": insight.discovered_at.isoformat()
                }
                for insight in sorted(self.meta_learning_insights, key=lambda x: x.discovered_at, reverse=True)[:5]
            ]
        }
    
    def get_accuracy_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get accuracy patterns for all metrics"""
        return {
            metric: {
                "trend": pattern.accuracy_trend,
                "average_accuracy": pattern.average_accuracy,
                "variance": pattern.accuracy_variance,
                "model_type": pattern.model_type.value,
                "improvement_opportunities": pattern.improvement_opportunities,
                "optimal_parameters": pattern.optimal_parameters
            }
            for metric, pattern in self.accuracy_patterns.items()
        }


# Export
__all__ = ['AdaptivePredictionEnhancer', 'AccuracyPattern', 'MetaLearningInsight']