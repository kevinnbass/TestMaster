"""
LLM Provider Optimization

Optimizes provider selection based on cost, latency, and quality.
Adapted from Agency Swarm's optimization patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import statistics
import time
from datetime import datetime, timedelta

from .universal_llm_provider import LLMProviderManager, LLMMessage, MessageRole


class OptimizationObjective(Enum):
    """Optimization objectives for provider selection."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_QUALITY = "maximize_quality"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    BALANCED = "balanced"


@dataclass
class ProviderMetrics:
    """Metrics for a provider."""
    provider_name: str
    
    # Performance metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    throughput: float = 0.0  # requests per minute
    
    # Cost metrics
    avg_cost_per_request: float = 0.0
    total_cost: float = 0.0
    cost_per_token: float = 0.0
    
    # Quality metrics (would need evaluation dataset)
    quality_score: float = 0.0
    accuracy_score: float = 0.0
    
    # Reliability metrics
    success_rate: float = 1.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    
    # Usage metrics
    total_requests: int = 0
    total_tokens: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_provider(self, provider):
        """Update metrics from provider statistics."""
        provider_metrics = provider.get_metrics()
        
        self.avg_response_time = provider_metrics.get('avg_response_time', 0.0)
        self.total_cost = provider_metrics.get('total_cost', 0.0)
        self.total_requests = provider_metrics.get('request_count', 0)
        self.total_tokens = provider_metrics.get('total_tokens', 0)
        self.error_rate = provider_metrics.get('error_rate', 0.0)
        self.success_rate = 1.0 - self.error_rate
        
        if self.total_requests > 0:
            self.avg_cost_per_request = self.total_cost / self.total_requests
            self.throughput = self.total_requests / max(1, (datetime.now() - self.last_updated).total_seconds() / 60)
        
        if self.total_tokens > 0:
            self.cost_per_token = self.total_cost / self.total_tokens
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'provider_name': self.provider_name,
            'avg_response_time': self.avg_response_time,
            'p95_response_time': self.p95_response_time,
            'throughput': self.throughput,
            'avg_cost_per_request': self.avg_cost_per_request,
            'total_cost': self.total_cost,
            'cost_per_token': self.cost_per_token,
            'quality_score': self.quality_score,
            'accuracy_score': self.accuracy_score,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'uptime_percentage': self.uptime_percentage,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'last_updated': self.last_updated.isoformat()
        }


class ProviderOptimizer:
    """Optimizes provider selection based on metrics and objectives."""
    
    def __init__(self, provider_manager: LLMProviderManager):
        self.provider_manager = provider_manager
        self.metrics: Dict[str, ProviderMetrics] = {}
        self.optimization_objective = OptimizationObjective.BALANCED
        
        # Optimization weights for balanced approach
        self.weights = {
            'cost': 0.3,
            'latency': 0.3,
            'quality': 0.2,
            'reliability': 0.2
        }
        
        # Historical performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        
        print("Provider Optimizer initialized")
    
    def set_optimization_objective(self, objective: OptimizationObjective, weights: Dict[str, float] = None):
        """Set optimization objective and weights."""
        self.optimization_objective = objective
        
        if weights:
            self.weights = weights
        elif objective == OptimizationObjective.MINIMIZE_COST:
            self.weights = {'cost': 0.8, 'latency': 0.1, 'quality': 0.05, 'reliability': 0.05}
        elif objective == OptimizationObjective.MINIMIZE_LATENCY:
            self.weights = {'cost': 0.1, 'latency': 0.8, 'quality': 0.05, 'reliability': 0.05}
        elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
            self.weights = {'cost': 0.1, 'latency': 0.2, 'quality': 0.6, 'reliability': 0.1}
        elif objective == OptimizationObjective.MAXIMIZE_RELIABILITY:
            self.weights = {'cost': 0.1, 'latency': 0.2, 'quality': 0.1, 'reliability': 0.6}
        else:  # BALANCED
            self.weights = {'cost': 0.25, 'latency': 0.25, 'quality': 0.25, 'reliability': 0.25}
        
        print(f"   Optimization objective: {objective.value}")
        print(f"   Weights: {self.weights}")
    
    def update_provider_metrics(self):
        """Update metrics for all providers."""
        for provider_name, provider in self.provider_manager.providers.items():
            if provider_name not in self.metrics:
                self.metrics[provider_name] = ProviderMetrics(provider_name=provider_name)
            
            self.metrics[provider_name].update_from_provider(provider)
    
    def benchmark_providers(self, test_messages: List[LLMMessage] = None, iterations: int = 5) -> Dict[str, ProviderMetrics]:
        """Benchmark all providers with test requests."""
        if not test_messages:
            test_messages = [
                LLMMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=MessageRole.USER, content="Write a simple Python function that adds two numbers.")
            ]
        
        print(f"\nBenchmarking {len(self.provider_manager.providers)} providers...")
        
        benchmark_results = {}
        
        for provider_name, provider in self.provider_manager.providers.items():
            print(f"   Benchmarking {provider_name}...")
            
            response_times = []
            costs = []
            success_count = 0
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    response = provider.generate_sync(test_messages)
                    response_time = time.time() - start_time
                    
                    response_times.append(response_time)
                    costs.append(response.cost_estimate)
                    success_count += 1
                    
                except Exception as e:
                    print(f"      Iteration {i+1} failed: {str(e)}")
                    continue
            
            if response_times:
                # Calculate metrics
                metrics = ProviderMetrics(provider_name=provider_name)
                metrics.avg_response_time = statistics.mean(response_times)
                metrics.p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
                metrics.avg_cost_per_request = statistics.mean(costs) if costs else 0.0
                metrics.success_rate = success_count / iterations
                metrics.error_rate = 1.0 - metrics.success_rate
                metrics.total_requests = iterations
                
                benchmark_results[provider_name] = metrics
                self.metrics[provider_name] = metrics
                
                print(f"      Avg response time: {metrics.avg_response_time:.3f}s")
                print(f"      Avg cost: ${metrics.avg_cost_per_request:.6f}")
                print(f"      Success rate: {metrics.success_rate:.1%}")
            else:
                print(f"      All requests failed")
        
        return benchmark_results
    
    def get_optimal_provider(self, context: Dict[str, Any] = None) -> str:
        """Get optimal provider based on current objective."""
        if not self.metrics:
            self.update_provider_metrics()
        
        if not self.metrics:
            # No metrics available, return primary provider
            return self.provider_manager.primary_provider
        
        # Calculate scores for each provider
        provider_scores = {}
        
        for provider_name, metrics in self.metrics.items():
            score = self._calculate_provider_score(metrics, context)
            provider_scores[provider_name] = score
        
        # Return provider with best score
        best_provider = max(provider_scores, key=provider_scores.get)
        
        print(f"   Optimal provider: {best_provider} (score: {provider_scores[best_provider]:.3f})")
        
        return best_provider
    
    def _calculate_provider_score(self, metrics: ProviderMetrics, context: Dict[str, Any] = None) -> float:
        """Calculate weighted score for a provider."""
        # Normalize metrics (0-1 scale, higher is better)
        
        # Cost score (lower cost = higher score)
        max_cost = max(m.avg_cost_per_request for m in self.metrics.values()) or 1.0
        cost_score = 1.0 - (metrics.avg_cost_per_request / max_cost) if max_cost > 0 else 1.0
        
        # Latency score (lower latency = higher score)
        max_latency = max(m.avg_response_time for m in self.metrics.values()) or 1.0
        latency_score = 1.0 - (metrics.avg_response_time / max_latency) if max_latency > 0 else 1.0
        
        # Quality score (higher quality = higher score)
        quality_score = metrics.quality_score / 100.0 if metrics.quality_score > 0 else 0.5
        
        # Reliability score (higher success rate = higher score)
        reliability_score = metrics.success_rate
        
        # Calculate weighted score
        total_score = (
            cost_score * self.weights['cost'] +
            latency_score * self.weights['latency'] +
            quality_score * self.weights['quality'] +
            reliability_score * self.weights['reliability']
        )
        
        # Apply context adjustments
        if context:
            # Adjust for urgency
            if context.get('urgent', False):
                total_score = total_score * 0.7 + latency_score * 0.3
            
            # Adjust for budget constraints
            if context.get('budget_constrained', False):
                total_score = total_score * 0.7 + cost_score * 0.3
            
            # Adjust for quality requirements
            if context.get('high_quality_required', False):
                total_score = total_score * 0.7 + quality_score * 0.3
        
        return total_score
    
    def optimize_provider_allocation(self, expected_requests: int = 100) -> Dict[str, float]:
        """Optimize allocation of requests across providers."""
        if not self.metrics:
            self.update_provider_metrics()
        
        if len(self.metrics) <= 1:
            # Only one provider, allocate 100%
            return {list(self.metrics.keys())[0]: 1.0} if self.metrics else {}
        
        # Calculate optimal allocation based on objective
        allocation = {}
        
        if self.optimization_objective == OptimizationObjective.MINIMIZE_COST:
            # Allocate to cheapest providers
            sorted_providers = sorted(self.metrics.items(), 
                                    key=lambda x: x[1].avg_cost_per_request)
            
            # Use top 3 cheapest providers
            top_providers = sorted_providers[:3]
            total_weight = sum(1.0 / (m.avg_cost_per_request + 0.001) for _, m in top_providers)
            
            for provider_name, metrics in top_providers:
                weight = (1.0 / (metrics.avg_cost_per_request + 0.001)) / total_weight
                allocation[provider_name] = weight
        
        elif self.optimization_objective == OptimizationObjective.MINIMIZE_LATENCY:
            # Allocate to fastest providers
            sorted_providers = sorted(self.metrics.items(),
                                    key=lambda x: x[1].avg_response_time)
            
            top_providers = sorted_providers[:3]
            total_weight = sum(1.0 / (m.avg_response_time + 0.001) for _, m in top_providers)
            
            for provider_name, metrics in top_providers:
                weight = (1.0 / (metrics.avg_response_time + 0.001)) / total_weight
                allocation[provider_name] = weight
        
        else:
            # Balanced allocation using provider scores
            provider_scores = {}
            for provider_name, metrics in self.metrics.items():
                provider_scores[provider_name] = self._calculate_provider_score(metrics)
            
            total_score = sum(provider_scores.values())
            
            for provider_name, score in provider_scores.items():
                allocation[provider_name] = score / total_score if total_score > 0 else 1.0 / len(provider_scores)
        
        print(f"\nOptimal allocation for {expected_requests} requests:")
        for provider, percent in allocation.items():
            requests = int(expected_requests * percent)
            print(f"   {provider}: {percent:.1%} ({requests} requests)")
        
        return allocation
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all provider metrics."""
        if not self.metrics:
            self.update_provider_metrics()
        
        summary = {
            'total_providers': len(self.metrics),
            'optimization_objective': self.optimization_objective.value,
            'weights': self.weights,
            'providers': {}
        }
        
        for provider_name, metrics in self.metrics.items():
            summary['providers'][provider_name] = metrics.to_dict()
        
        # Calculate aggregated metrics
        if self.metrics:
            all_metrics = list(self.metrics.values())
            summary['aggregated'] = {
                'avg_response_time': statistics.mean(m.avg_response_time for m in all_metrics),
                'avg_cost_per_request': statistics.mean(m.avg_cost_per_request for m in all_metrics),
                'avg_success_rate': statistics.mean(m.success_rate for m in all_metrics),
                'total_requests': sum(m.total_requests for m in all_metrics),
                'total_cost': sum(m.total_cost for m in all_metrics)
            }
        
        return summary


class CostOptimizer(ProviderOptimizer):
    """Specialized optimizer for cost minimization."""
    
    def __init__(self, provider_manager: LLMProviderManager):
        super().__init__(provider_manager)
        self.set_optimization_objective(OptimizationObjective.MINIMIZE_COST)
    
    def get_cheapest_provider(self) -> str:
        """Get the cheapest provider."""
        if not self.metrics:
            self.update_provider_metrics()
        
        if not self.metrics:
            return self.provider_manager.primary_provider
        
        cheapest = min(self.metrics.items(), 
                      key=lambda x: x[1].avg_cost_per_request)
        
        return cheapest[0]
    
    def estimate_cost(self, provider_name: str, num_requests: int, avg_tokens: int = 1000) -> float:
        """Estimate cost for a number of requests."""
        if provider_name not in self.metrics:
            return 0.0
        
        metrics = self.metrics[provider_name]
        
        if metrics.cost_per_token > 0:
            return num_requests * avg_tokens * metrics.cost_per_token
        else:
            return num_requests * metrics.avg_cost_per_request


class LatencyOptimizer(ProviderOptimizer):
    """Specialized optimizer for latency minimization."""
    
    def __init__(self, provider_manager: LLMProviderManager):
        super().__init__(provider_manager)
        self.set_optimization_objective(OptimizationObjective.MINIMIZE_LATENCY)
    
    def get_fastest_provider(self) -> str:
        """Get the fastest provider."""
        if not self.metrics:
            self.update_provider_metrics()
        
        if not self.metrics:
            return self.provider_manager.primary_provider
        
        fastest = min(self.metrics.items(),
                     key=lambda x: x[1].avg_response_time)
        
        return fastest[0]


class QualityOptimizer(ProviderOptimizer):
    """Specialized optimizer for quality maximization."""
    
    def __init__(self, provider_manager: LLMProviderManager):
        super().__init__(provider_manager)
        self.set_optimization_objective(OptimizationObjective.MAXIMIZE_QUALITY)
    
    def get_highest_quality_provider(self) -> str:
        """Get the highest quality provider."""
        if not self.metrics:
            self.update_provider_metrics()
        
        if not self.metrics:
            return self.provider_manager.primary_provider
        
        highest_quality = max(self.metrics.items(),
                             key=lambda x: x[1].quality_score)
        
        return highest_quality[0]
    
    def evaluate_response_quality(self, response: str, expected_criteria: List[str] = None) -> float:
        """Evaluate response quality (simplified heuristic)."""
        score = 0.0
        
        # Length score (reasonable length)
        if 50 <= len(response) <= 2000:
            score += 20
        elif len(response) > 2000:
            score += 10
        
        # Structure score (paragraphs, sentences)
        sentences = response.count('.') + response.count('!') + response.count('?')
        if sentences >= 2:
            score += 20
        
        # Code quality (if contains code)
        if 'def ' in response or 'function' in response or 'class ' in response:
            score += 30
        
        # Completeness (if contains expected criteria)
        if expected_criteria:
            found_criteria = sum(1 for criteria in expected_criteria if criteria.lower() in response.lower())
            score += (found_criteria / len(expected_criteria)) * 30
        
        return min(score, 100.0)