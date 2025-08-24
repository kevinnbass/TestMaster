"""
Self-Optimizing ML Orchestrator
===============================

Meta-learning enhancement for the MLOrchestrator that enables autonomous
optimization of integration flows, resource allocation, and module interactions.

Author: Agent A Phase 2 - Self-Learning Intelligence
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics

from .ml_orchestrator import MLOrchestrator, IntegrationFlow, IntegrationPattern, OrchestrationMode


@dataclass
class FlowPerformanceMetrics:
    """Performance metrics for integration flows"""
    flow_id: str
    average_latency: float
    throughput: float  # messages per second
    error_rate: float
    resource_efficiency: float
    optimization_score: float
    bottlenecks: List[str]
    last_optimized: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationInsight:
    """Insights for ML orchestration optimization"""
    insight_id: str
    optimization_type: str  # "flow_reconfiguration", "resource_reallocation", "pattern_adjustment"
    target_flows: List[str]
    target_modules: List[str]
    current_performance: float
    predicted_improvement: float
    implementation_cost: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    recommended_actions: List[str]
    confidence: float
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModuleSynergy:
    """Discovered synergies between ML modules"""
    module_a: str
    module_b: str
    synergy_type: str  # "complementary", "reinforcing", "sequential"
    synergy_strength: float  # 0-1 score
    optimal_flow_pattern: IntegrationPattern
    performance_boost: float
    discovered_patterns: List[str]


class SelfOptimizingOrchestrator:
    """
    Meta-learning system that enhances the existing MLOrchestrator
    with autonomous optimization capabilities for flows, resources, and synergies.
    """
    
    def __init__(self, ml_orchestrator: MLOrchestrator):
        self.ml_orchestrator = ml_orchestrator
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.flow_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.module_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Optimization insights and discoveries
        self.optimization_insights: List[OptimizationInsight] = []
        self.discovered_synergies: List[ModuleSynergy] = []
        self.flow_performance_cache: Dict[str, FlowPerformanceMetrics] = {}
        
        # Learning configuration
        self.config = {
            "analysis_interval": 180,  # 3 minutes
            "optimization_interval": 600,  # 10 minutes
            "min_data_points_for_analysis": 20,
            "performance_improvement_threshold": 0.10,
            "auto_optimization_enabled": True,
            "max_concurrent_optimizations": 3,
            "risk_tolerance": "medium"  # "low", "medium", "high"
        }
        
        # Optimization state
        self.optimization_stats = {
            "total_analyses": 0,
            "optimizations_discovered": 0,
            "optimizations_applied": 0,
            "synergies_discovered": 0,
            "performance_improvements": 0,
            "start_time": datetime.now()
        }
        
        # Active optimization processes
        self.active_optimizations: Set[str] = set()
        self.is_optimizing = False
        self.optimization_task = None
        
        self.logger.info("Self-Optimizing Orchestrator initialized")
    
    async def start_optimization(self):
        """Start the self-optimization process"""
        if self.is_optimizing:
            self.logger.warning("Optimization already running")
            return
        
        self.is_optimizing = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Started self-optimizing orchestration")
    
    async def stop_optimization(self):
        """Stop the self-optimization process"""
        self.is_optimizing = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped self-optimizing orchestration")
    
    async def _optimization_loop(self):
        """Main self-optimization loop"""
        while self.is_optimizing:
            try:
                await asyncio.sleep(self.config["analysis_interval"])
                
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Analyze flow performance patterns
                await self._analyze_flow_performance()
                
                # Discover module synergies
                await self._discover_module_synergies()
                
                # Identify optimization opportunities
                insights = await self._identify_optimization_opportunities()
                
                # Apply automatic optimizations
                if self.config["auto_optimization_enabled"]:
                    await self._apply_autonomous_optimizations(insights)
                
                # Learn resource allocation patterns
                await self._learn_resource_patterns()
                
                self.optimization_stats["total_analyses"] += 1
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_performance_metrics(self):
        """Collect current performance metrics from the orchestrator"""
        try:
            # Get current orchestration status
            status = self.ml_orchestrator.get_orchestration_status()
            
            # Collect flow metrics
            for flow_id, flow_status in status.get("integration_flows", {}).items():
                metrics = {
                    "timestamp": datetime.now(),
                    "latency": flow_status.get("average_latency", 0.0),
                    "message_count": flow_status.get("message_count", 0),
                    "error_count": flow_status.get("error_count", 0),
                    "throughput": self._calculate_throughput(flow_status),
                    "enabled": flow_status.get("enabled", False)
                }
                self.flow_metrics_history[flow_id].append(metrics)
            
            # Collect module metrics
            for module_name, module_status in status.get("module_status", {}).items():
                metrics = {
                    "timestamp": datetime.now(),
                    "processing_time": module_status.get("processing_time", 0.0),
                    "success_rate": module_status.get("success_rate", 1.0),
                    "resource_usage": module_status.get("resource_usage", {}),
                    "status": module_status.get("status", "unknown")
                }
                self.module_performance_history[module_name].append(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
    
    def _calculate_throughput(self, flow_status: Dict[str, Any]) -> float:
        """Calculate throughput for a flow"""
        try:
            message_count = flow_status.get("message_count", 0)
            # Estimate throughput based on message count (simplified)
            return message_count / max(1, self.config["analysis_interval"])
        except Exception:
            return 0.0
    
    async def _analyze_flow_performance(self):
        """Analyze performance patterns for integration flows"""
        try:
            for flow_id, metrics_history in self.flow_metrics_history.items():
                if len(metrics_history) >= self.config["min_data_points_for_analysis"]:
                    performance_metrics = await self._calculate_flow_performance_metrics(flow_id, metrics_history)
                    if performance_metrics:
                        self.flow_performance_cache[flow_id] = performance_metrics
                        
        except Exception as e:
            self.logger.error(f"Failed to analyze flow performance: {e}")
    
    async def _calculate_flow_performance_metrics(self, flow_id: str, metrics_history: deque) -> Optional[FlowPerformanceMetrics]:
        """Calculate comprehensive performance metrics for a flow"""
        try:
            recent_metrics = list(metrics_history)[-50:]  # Last 50 data points
            
            if not recent_metrics:
                return None
            
            # Calculate averages
            latencies = [m["latency"] for m in recent_metrics if m["latency"] > 0]
            throughputs = [m["throughput"] for m in recent_metrics if m["throughput"] > 0]
            error_counts = [m["error_count"] for m in recent_metrics]
            message_counts = [m["message_count"] for m in recent_metrics if m["message_count"] > 0]
            
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            avg_throughput = statistics.mean(throughputs) if throughputs else 0.0
            total_errors = sum(error_counts)
            total_messages = sum(message_counts)
            
            # Calculate error rate
            error_rate = total_errors / max(1, total_messages)
            
            # Calculate resource efficiency (lower latency + higher throughput = better efficiency)
            resource_efficiency = 0.0
            if avg_latency > 0 and avg_throughput > 0:
                resource_efficiency = avg_throughput / (avg_latency + 1.0)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(avg_latency, avg_throughput, error_rate, resource_efficiency)
            
            # Identify bottlenecks
            bottlenecks = self._identify_flow_bottlenecks(flow_id, recent_metrics)
            
            return FlowPerformanceMetrics(
                flow_id=flow_id,
                average_latency=avg_latency,
                throughput=avg_throughput,
                error_rate=error_rate,
                resource_efficiency=resource_efficiency,
                optimization_score=optimization_score,
                bottlenecks=bottlenecks
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate flow performance metrics for {flow_id}: {e}")
            return None
    
    def _calculate_optimization_score(self, latency: float, throughput: float, error_rate: float, efficiency: float) -> float:
        """Calculate overall optimization score for a flow"""
        try:
            # Normalize metrics (0-1 scale, higher is better)
            latency_score = max(0.0, 1.0 - min(latency / 1000.0, 1.0))  # Normalize to 1 second
            throughput_score = min(throughput / 10.0, 1.0)  # Normalize to 10 msgs/sec
            error_score = max(0.0, 1.0 - min(error_rate * 10, 1.0))  # Penalize errors heavily
            efficiency_score = min(efficiency / 5.0, 1.0)  # Normalize efficiency
            
            # Weighted combination
            score = (latency_score * 0.3 + throughput_score * 0.2 + 
                    error_score * 0.4 + efficiency_score * 0.1)
            
            return score
            
        except Exception:
            return 0.5
    
    def _identify_flow_bottlenecks(self, flow_id: str, recent_metrics: List[Dict[str, Any]]) -> List[str]:
        """Identify bottlenecks in a flow"""
        bottlenecks = []
        
        try:
            # Analyze patterns in the metrics
            latencies = [m["latency"] for m in recent_metrics if m["latency"] > 0]
            error_counts = [m["error_count"] for m in recent_metrics]
            
            # High latency bottleneck
            if latencies and statistics.mean(latencies) > 500:  # 500ms threshold
                bottlenecks.append("high_latency")
            
            # Error rate bottleneck
            if error_counts and sum(error_counts) > len(error_counts) * 0.1:  # 10% error rate
                bottlenecks.append("high_error_rate")
            
            # Throughput bottleneck
            throughputs = [m["throughput"] for m in recent_metrics if m["throughput"] > 0]
            if throughputs and statistics.mean(throughputs) < 1.0:  # Less than 1 msg/sec
                bottlenecks.append("low_throughput")
            
        except Exception as e:
            self.logger.debug(f"Failed to identify bottlenecks for {flow_id}: {e}")
        
        return bottlenecks
    
    async def _discover_module_synergies(self):
        """Discover synergies between ML modules"""
        try:
            module_names = list(self.module_performance_history.keys())
            
            # Analyze pairs of modules for potential synergies
            for i, module_a in enumerate(module_names):
                for module_b in module_names[i+1:]:
                    synergy = await self._analyze_module_pair_synergy(module_a, module_b)
                    if synergy and synergy.synergy_strength > 0.6:
                        # Check if synergy already exists
                        existing = any(s.module_a == module_a and s.module_b == module_b 
                                     for s in self.discovered_synergies)
                        if not existing:
                            self.discovered_synergies.append(synergy)
                            self.optimization_stats["synergies_discovered"] += 1
                            self.logger.info(f"Discovered synergy between {module_a} and {module_b}: {synergy.synergy_strength:.2f}")
                            
        except Exception as e:
            self.logger.error(f"Failed to discover module synergies: {e}")
    
    async def _analyze_module_pair_synergy(self, module_a: str, module_b: str) -> Optional[ModuleSynergy]:
        """Analyze potential synergy between two modules"""
        try:
            # Get performance data for both modules
            history_a = list(self.module_performance_history[module_a])
            history_b = list(self.module_performance_history[module_b])
            
            if len(history_a) < 10 or len(history_b) < 10:
                return None
            
            # Calculate correlation in performance patterns
            times_a = [m["processing_time"] for m in history_a[-20:]]
            times_b = [m["processing_time"] for m in history_b[-20:]]
            
            if len(times_a) != len(times_b):
                min_len = min(len(times_a), len(times_b))
                times_a = times_a[:min_len]
                times_b = times_b[:min_len]
            
            # Calculate correlation
            correlation = np.corrcoef(times_a, times_b)[0, 1] if len(times_a) > 1 else 0.0
            
            # Determine synergy type and strength
            synergy_strength = 0.0
            synergy_type = "complementary"
            optimal_pattern = IntegrationPattern.PIPELINE
            
            if correlation > 0.7:
                # Positive correlation - reinforcing synergy
                synergy_type = "reinforcing"
                synergy_strength = correlation
                optimal_pattern = IntegrationPattern.COORDINATION
            elif correlation < -0.3:
                # Negative correlation - complementary synergy
                synergy_type = "complementary"
                synergy_strength = abs(correlation)
                optimal_pattern = IntegrationPattern.PIPELINE
            else:
                # Look for sequential patterns
                success_rates_a = [m["success_rate"] for m in history_a[-20:]]
                success_rates_b = [m["success_rate"] for m in history_b[-20:]]
                
                if (statistics.mean(success_rates_a) > 0.9 and 
                    statistics.mean(success_rates_b) > 0.9):
                    synergy_type = "sequential"
                    synergy_strength = 0.7
                    optimal_pattern = IntegrationPattern.PIPELINE
            
            if synergy_strength > 0.5:
                # Calculate performance boost potential
                perf_boost = min(synergy_strength * 0.2, 0.3)  # Max 30% boost
                
                return ModuleSynergy(
                    module_a=module_a,
                    module_b=module_b,
                    synergy_type=synergy_type,
                    synergy_strength=synergy_strength,
                    optimal_flow_pattern=optimal_pattern,
                    performance_boost=perf_boost,
                    discovered_patterns=[f"correlation_{correlation:.2f}"]
                )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to analyze synergy between {module_a} and {module_b}: {e}")
            return None
    
    async def _identify_optimization_opportunities(self) -> List[OptimizationInsight]:
        """Identify specific optimization opportunities"""
        insights = []
        
        try:
            # Flow optimization opportunities
            for flow_id, performance in self.flow_performance_cache.items():
                if performance.optimization_score < 0.6:  # Poor performance
                    insights.append(OptimizationInsight(
                        insight_id=f"flow_opt_{flow_id}_{int(datetime.now().timestamp())}",
                        optimization_type="flow_reconfiguration",
                        target_flows=[flow_id],
                        target_modules=[],
                        current_performance=performance.optimization_score,
                        predicted_improvement=0.25,
                        implementation_cost="medium",
                        risk_level="low",
                        recommended_actions=[
                            f"Optimize {flow_id} flow configuration",
                            f"Address bottlenecks: {', '.join(performance.bottlenecks)}"
                        ],
                        confidence=0.8
                    ))
            
            # Resource reallocation opportunities
            underutilized_modules = []
            overutilized_modules = []
            
            for module_name, history in self.module_performance_history.items():
                if len(history) >= 10:
                    recent_usage = [m["resource_usage"].get("cpu", 0.5) for m in list(history)[-10:]]
                    avg_usage = statistics.mean(recent_usage)
                    
                    if avg_usage < 0.3:
                        underutilized_modules.append(module_name)
                    elif avg_usage > 0.8:
                        overutilized_modules.append(module_name)
            
            if underutilized_modules and overutilized_modules:
                insights.append(OptimizationInsight(
                    insight_id=f"resource_opt_{int(datetime.now().timestamp())}",
                    optimization_type="resource_reallocation",
                    target_flows=[],
                    target_modules=underutilized_modules + overutilized_modules,
                    current_performance=0.5,
                    predicted_improvement=0.20,
                    implementation_cost="low",
                    risk_level="low",
                    recommended_actions=[
                        "Reallocate resources from underutilized to overutilized modules",
                        f"Scale down: {', '.join(underutilized_modules[:3])}",
                        f"Scale up: {', '.join(overutilized_modules[:3])}"
                    ],
                    confidence=0.7
                ))
            
            # Synergy-based optimization opportunities
            for synergy in self.discovered_synergies:
                if synergy.performance_boost > 0.15:  # Significant boost potential
                    insights.append(OptimizationInsight(
                        insight_id=f"synergy_opt_{synergy.module_a}_{synergy.module_b}_{int(datetime.now().timestamp())}",
                        optimization_type="pattern_adjustment",
                        target_flows=[],
                        target_modules=[synergy.module_a, synergy.module_b],
                        current_performance=0.6,
                        predicted_improvement=synergy.performance_boost,
                        implementation_cost="medium",
                        risk_level="medium",
                        recommended_actions=[
                            f"Create {synergy.optimal_flow_pattern.value} flow between {synergy.module_a} and {synergy.module_b}",
                            f"Leverage {synergy.synergy_type} synergy pattern"
                        ],
                        confidence=synergy.synergy_strength
                    ))
            
            # Store insights
            self.optimization_insights.extend(insights)
            self.optimization_stats["optimizations_discovered"] += len(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to identify optimization opportunities: {e}")
            return []
    
    async def _apply_autonomous_optimizations(self, insights: List[OptimizationInsight]):
        """Apply autonomous optimizations based on insights"""
        applied_count = 0
        
        try:
            # Filter insights based on risk tolerance and confidence
            safe_insights = [
                insight for insight in insights
                if (insight.confidence > 0.7 and 
                    self._is_safe_to_apply(insight))
            ]
            
            for insight in safe_insights[:self.config["max_concurrent_optimizations"]]:
                if insight.insight_id not in self.active_optimizations:
                    success = await self._apply_optimization(insight)
                    if success:
                        applied_count += 1
                        self.active_optimizations.add(insight.insight_id)
            
            self.optimization_stats["optimizations_applied"] += applied_count
            
            if applied_count > 0:
                self.logger.info(f"Applied {applied_count} autonomous optimizations")
                
        except Exception as e:
            self.logger.error(f"Failed to apply autonomous optimizations: {e}")
    
    def _is_safe_to_apply(self, insight: OptimizationInsight) -> bool:
        """Check if optimization is safe to apply based on risk tolerance"""
        risk_levels = {"low": 0, "medium": 1, "high": 2}
        tolerance_levels = {"low": 0, "medium": 1, "high": 2}
        
        return risk_levels[insight.risk_level] <= tolerance_levels[self.config["risk_tolerance"]]
    
    async def _apply_optimization(self, insight: OptimizationInsight) -> bool:
        """Apply a specific optimization"""
        try:
            if insight.optimization_type == "resource_reallocation":
                return await self._apply_resource_optimization(insight)
            elif insight.optimization_type == "flow_reconfiguration":
                return await self._apply_flow_optimization(insight)
            elif insight.optimization_type == "pattern_adjustment":
                return await self._apply_pattern_optimization(insight)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {insight.insight_id}: {e}")
            return False
        finally:
            # Remove from active optimizations after processing
            self.active_optimizations.discard(insight.insight_id)
    
    async def _apply_resource_optimization(self, insight: OptimizationInsight) -> bool:
        """Apply resource reallocation optimization"""
        try:
            # This would implement actual resource reallocation
            # For now, log the optimization for demonstration
            self.logger.info(f"Applied resource optimization: {', '.join(insight.recommended_actions)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply resource optimization: {e}")
            return False
    
    async def _apply_flow_optimization(self, insight: OptimizationInsight) -> bool:
        """Apply flow reconfiguration optimization"""
        try:
            # This would implement actual flow reconfiguration
            # For now, log the optimization for demonstration
            self.logger.info(f"Applied flow optimization: {', '.join(insight.recommended_actions)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply flow optimization: {e}")
            return False
    
    async def _apply_pattern_optimization(self, insight: OptimizationInsight) -> bool:
        """Apply integration pattern optimization"""
        try:
            # This would implement actual pattern adjustments
            # For now, log the optimization for demonstration
            self.logger.info(f"Applied pattern optimization: {', '.join(insight.recommended_actions)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply pattern optimization: {e}")
            return False
    
    async def _learn_resource_patterns(self):
        """Learn optimal resource allocation patterns"""
        try:
            # Analyze resource usage patterns across all modules
            resource_patterns = {}
            
            for module_name, history in self.module_performance_history.items():
                if len(history) >= 20:
                    recent_metrics = list(history)[-20:]
                    
                    # Analyze resource usage patterns
                    cpu_usage = [m["resource_usage"].get("cpu", 0.5) for m in recent_metrics]
                    processing_times = [m["processing_time"] for m in recent_metrics]
                    success_rates = [m["success_rate"] for m in recent_metrics]
                    
                    # Find optimal resource usage
                    optimal_cpu = self._find_optimal_resource_usage(cpu_usage, processing_times, success_rates)
                    
                    resource_patterns[module_name] = {
                        "optimal_cpu": optimal_cpu,
                        "current_avg_cpu": statistics.mean(cpu_usage),
                        "performance_correlation": np.corrcoef(cpu_usage, processing_times)[0, 1] if len(cpu_usage) > 1 else 0.0
                    }
            
            # Store learned patterns for future optimization
            self._store_resource_patterns(resource_patterns)
            
        except Exception as e:
            self.logger.error(f"Failed to learn resource patterns: {e}")
    
    def _find_optimal_resource_usage(self, cpu_usage: List[float], processing_times: List[float], success_rates: List[float]) -> float:
        """Find optimal CPU usage based on performance metrics"""
        try:
            if len(cpu_usage) != len(processing_times) or len(cpu_usage) != len(success_rates):
                return 0.5  # Default
            
            # Find CPU usage that minimizes processing time while maintaining high success rate
            best_cpu = 0.5
            best_score = 0.0
            
            for i, cpu in enumerate(cpu_usage):
                if success_rates[i] > 0.9:  # Only consider high success rate points
                    # Score based on inverse processing time (faster is better)
                    score = 1.0 / (processing_times[i] + 1.0)
                    if score > best_score:
                        best_score = score
                        best_cpu = cpu
            
            return best_cpu
            
        except Exception:
            return 0.5  # Default
    
    def _store_resource_patterns(self, patterns: Dict[str, Any]):
        """Store learned resource patterns for future use"""
        try:
            # This would store patterns in a persistent storage
            # For now, just log significant findings
            for module, pattern in patterns.items():
                if abs(pattern["optimal_cpu"] - pattern["current_avg_cpu"]) > 0.2:
                    self.logger.info(f"Resource learning: {module} optimal CPU {pattern['optimal_cpu']:.2f} vs current {pattern['current_avg_cpu']:.2f}")
                    
        except Exception as e:
            self.logger.debug(f"Failed to store resource patterns: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        return {
            "is_optimizing": self.is_optimizing,
            "optimization_stats": self.optimization_stats.copy(),
            "active_optimizations": len(self.active_optimizations),
            "discovered_insights_count": len(self.optimization_insights),
            "discovered_synergies_count": len(self.discovered_synergies),
            "flow_performance_cache_size": len(self.flow_performance_cache),
            "recent_insights": [
                {
                    "type": insight.optimization_type,
                    "confidence": insight.confidence,
                    "predicted_improvement": insight.predicted_improvement,
                    "target_flows": insight.target_flows,
                    "target_modules": insight.target_modules,
                    "discovered_at": insight.discovered_at.isoformat()
                }
                for insight in sorted(self.optimization_insights, key=lambda x: x.discovered_at, reverse=True)[:5]
            ],
            "top_synergies": [
                {
                    "modules": [s.module_a, s.module_b],
                    "type": s.synergy_type,
                    "strength": s.synergy_strength,
                    "performance_boost": s.performance_boost,
                    "optimal_pattern": s.optimal_flow_pattern.value
                }
                for s in sorted(self.discovered_synergies, key=lambda x: x.synergy_strength, reverse=True)[:5]
            ]
        }


# Export
__all__ = ['SelfOptimizingOrchestrator', 'FlowPerformanceMetrics', 'OptimizationInsight', 'ModuleSynergy']