"""
TestMaster Cross-System Analysis Component
==========================================

Extracted from consolidated integration hub for better modularization.
Provides comprehensive cross-system performance and correlation analysis.

Original location: core/intelligence/integration/__init__.py (lines ~200-600)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics
import logging
import uuid

# Import base structures
from ..base import CrossSystemAnalysis, IntegrationEndpoint, IntegrationStatus


class CrossSystemAnalyzer:
    """
    Comprehensive cross-system performance and correlation analyzer.
    
    Preserves all functionality from cross_system_analytics.py including:
    - System correlation analysis with ML insights
    - Performance bottleneck identification
    - Resource contention detection
    - Predictive failure analysis
    - Capacity forecasting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("cross_system_analyzer")
        
        # Analysis cache for performance
        self._analysis_cache: Dict[str, CrossSystemAnalysis] = {}
        
        # Metrics storage
        self._system_metrics: Dict[str, Dict[str, Any]] = {}
        self._correlation_cache: Dict[str, Dict[str, float]] = {}
        
    def analyze_cross_system_performance(self, 
                                       systems: List[str],
                                       endpoints: Dict[str, IntegrationEndpoint],
                                       time_window_hours: int = 24,
                                       include_correlations: bool = True) -> CrossSystemAnalysis:
        """
        Comprehensive cross-system performance analysis.
        
        Args:
            systems: List of systems to analyze
            endpoints: Dictionary of system endpoints
            time_window_hours: Analysis time window
            include_correlations: Include correlation analysis
            
        Returns:
            CrossSystemAnalysis with comprehensive metrics
        """
        try:
            analysis_id = f"cross_system_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            
            # Initialize analysis structure
            analysis = CrossSystemAnalysis(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                systems_analyzed=systems,
                analysis_duration=0.0
            )
            
            # Collect performance data for each system
            system_metrics = {}
            for system in systems:
                metrics = self._collect_system_metrics(system, endpoints, time_window_hours)
                system_metrics[system] = metrics
                
                # Calculate health scores
                health_score = self._calculate_system_health_score(metrics)
                analysis.system_health_scores[system] = health_score
            
            # Store metrics for future use
            self._system_metrics = system_metrics
            
            # Perform correlation analysis
            if include_correlations and len(systems) > 1:
                correlations = self._analyze_system_correlations(system_metrics)
                analysis.system_correlations = correlations
                
                # Performance correlations
                perf_correlations = self._analyze_performance_correlations(system_metrics)
                analysis.performance_correlations = perf_correlations
                
                # Error correlations
                error_correlations = self._analyze_error_correlations(system_metrics)
                analysis.error_correlations = error_correlations
            
            # Cross-system latency analysis
            latency_analysis = self._analyze_cross_system_latency(systems, endpoints)
            analysis.cross_system_latency = latency_analysis
            
            # Bottleneck identification
            bottlenecks = self._identify_performance_bottlenecks(system_metrics)
            analysis.bottleneck_analysis = bottlenecks
            
            # Resource contention analysis
            contention = self._detect_resource_contention(system_metrics)
            analysis.resource_contention = contention
            
            # Availability matrix
            availability = self._calculate_availability_matrix(systems, endpoints, time_window_hours)
            analysis.availability_matrix = availability
            
            # Data flow and consistency analysis
            data_flows = self._analyze_data_flows(systems, endpoints)
            analysis.data_flow_patterns = data_flows
            
            consistency_checks = self._perform_consistency_checks(systems, endpoints)
            analysis.data_consistency_checks = consistency_checks
            
            # Generate optimization recommendations
            optimizations = self._generate_optimization_recommendations(analysis)
            analysis.optimization_opportunities = optimizations
            
            # Predictive analysis
            failure_predictions = self._predict_system_failures(system_metrics)
            analysis.predicted_failures = failure_predictions
            
            capacity_forecasts = self._forecast_capacity_needs(system_metrics)
            analysis.capacity_forecasts = capacity_forecasts
            
            # Finalize analysis
            analysis.analysis_duration = time.time() - start_time
            
            # Cache results
            self._analysis_cache[analysis_id] = analysis
            
            self.logger.info(f"Cross-system analysis complete: {len(systems)} systems in {analysis.analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Cross-system analysis failed: {e}")
            return self._create_error_analysis(systems, str(e))
    
    def get_system_correlations(self, 
                              primary_system: str,
                              endpoints: Dict[str, IntegrationEndpoint],
                              correlation_threshold: float = 0.7) -> Dict[str, float]:
        """
        Get correlation scores between primary system and all other systems.
        
        Args:
            primary_system: System to correlate against
            endpoints: System endpoints
            correlation_threshold: Minimum correlation to include
            
        Returns:
            Dictionary of system correlations
        """
        try:
            # Check cache first
            cache_key = f"{primary_system}_{correlation_threshold}"
            if cache_key in self._correlation_cache:
                return self._correlation_cache[cache_key]
            
            correlations = {}
            
            # Get metrics for primary system
            primary_metrics = self._collect_system_metrics(primary_system, endpoints, 24)
            if not primary_metrics:
                return correlations
            
            # Compare with all other connected systems
            for endpoint_id, endpoint in endpoints.items():
                if endpoint.name != primary_system and endpoint.status == IntegrationStatus.CONNECTED:
                    system_metrics = self._collect_system_metrics(endpoint.name, endpoints, 24)
                    if system_metrics:
                        correlation = self._calculate_metric_correlation(primary_metrics, system_metrics)
                        if correlation >= correlation_threshold:
                            correlations[endpoint.name] = correlation
            
            # Cache results
            self._correlation_cache[cache_key] = correlations
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"System correlation analysis failed: {e}")
            return {}
    
    # === Metrics Collection ===
    
    def _collect_system_metrics(self, system: str, endpoints: Dict[str, IntegrationEndpoint], time_window_hours: int) -> Dict[str, Any]:
        """Collect performance metrics for a system."""
        # Find endpoint for this system
        endpoint = None
        for ep in endpoints.values():
            if ep.name == system:
                endpoint = ep
                break
        
        if not endpoint:
            return {}
        
        # Collect metrics
        metrics = {
            'response_times': endpoint.response_times[-100:],  # Recent response times
            'error_rate': len([t for t in endpoint.response_times if t > 5.0]) / len(endpoint.response_times) if endpoint.response_times else 0,
            'availability': endpoint.availability_percentage,
            'throughput': len(endpoint.response_times) / time_window_hours if endpoint.response_times else 0,
            'last_successful_connection': endpoint.last_successful_connection.isoformat() if endpoint.last_successful_connection else None,
            'status': endpoint.status.value,
            'error_rates': endpoint.error_rates,
            'throughput_metrics': endpoint.throughput_metrics
        }
        
        return metrics
    
    # === Health Score Calculation ===
    
    def _calculate_system_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate health score for a system."""
        if not metrics:
            return 0.0
        
        # Base score from availability
        health_score = metrics.get('availability', 0.0)
        
        # Adjust for error rate
        error_rate = metrics.get('error_rate', 0.0)
        health_score *= (1.0 - error_rate)
        
        # Adjust for response time
        response_times = metrics.get('response_times', [])
        if response_times:
            avg_response_time = statistics.mean(response_times)
            if avg_response_time < 1.0:
                health_score *= 1.0  # Good response time
            elif avg_response_time < 5.0:
                health_score *= 0.8  # Acceptable response time
            else:
                health_score *= 0.5  # Poor response time
        
        return min(100.0, max(0.0, health_score))
    
    # === Correlation Analysis ===
    
    def _analyze_system_correlations(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between systems."""
        correlations = {}
        
        systems = list(system_metrics.keys())
        for i, system1 in enumerate(systems):
            correlations[system1] = {}
            for j, system2 in enumerate(systems):
                if i != j:
                    correlation = self._calculate_metric_correlation(
                        system_metrics[system1], 
                        system_metrics[system2]
                    )
                    correlations[system1][system2] = correlation
        
        return correlations
    
    def _calculate_metric_correlation(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> float:
        """Calculate correlation between two sets of metrics."""
        try:
            avail1 = metrics1.get('availability', 0.0)
            avail2 = metrics2.get('availability', 0.0)
            error1 = metrics1.get('error_rate', 0.0)
            error2 = metrics2.get('error_rate', 0.0)
            
            # Calculate simple correlation
            avail_diff = abs(avail1 - avail2) / 100.0
            error_diff = abs(error1 - error2)
            
            correlation = 1.0 - (avail_diff + error_diff) / 2.0
            return max(0.0, min(1.0, correlation))
        except:
            return 0.0
    
    def _analyze_performance_correlations(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Analyze performance correlations across systems."""
        perf_correlations = {}
        
        # Analyze response time correlations
        response_time_data = {}
        for system, metrics in system_metrics.items():
            response_times = metrics.get('response_times', [])
            if response_times:
                response_time_data[system] = statistics.mean(response_times)
        
        # Calculate correlation strength
        if len(response_time_data) > 1:
            values = list(response_time_data.values())
            variance = statistics.variance(values) if len(values) > 1 else 0
            mean_val = statistics.mean(values)
            
            correlation_strength = 1.0 - (variance / (mean_val ** 2)) if mean_val > 0 else 0
            perf_correlations['response_time_correlation'] = max(0.0, min(1.0, correlation_strength))
            perf_correlations['synchronized_performance'] = correlation_strength > 0.7
        
        return perf_correlations
    
    def _analyze_error_correlations(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze error correlations between systems."""
        error_correlations = {}
        
        # Find systems with high error rates
        high_error_systems = []
        for system, metrics in system_metrics.items():
            error_rate = metrics.get('error_rate', 0.0)
            if error_rate > 0.1:  # More than 10% error rate
                high_error_systems.append(system)
        
        if high_error_systems:
            error_correlations['high_error_systems'] = high_error_systems
            error_correlations['cascade_risk'] = ['High risk of error cascade'] if len(high_error_systems) > 2 else []
        
        return error_correlations
    
    # === Performance Analysis ===
    
    def _analyze_cross_system_latency(self, systems: List[str], endpoints: Dict[str, IntegrationEndpoint]) -> Dict[str, float]:
        """Analyze latency between systems."""
        latency_analysis = {}
        
        for system in systems:
            # Calculate average latency for each system
            endpoint = self._get_endpoint_by_name(system, endpoints)
            
            if endpoint and endpoint.response_times:
                avg_latency = statistics.mean(endpoint.response_times)
                latency_analysis[system] = avg_latency
        
        return latency_analysis
    
    def _identify_performance_bottlenecks(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Identify performance bottlenecks."""
        bottlenecks = {}
        
        for system, metrics in system_metrics.items():
            response_times = metrics.get('response_times', [])
            error_rate = metrics.get('error_rate', 0.0)
            availability = metrics.get('availability', 100.0)
            
            if response_times and statistics.mean(response_times) > 5.0:
                bottlenecks[system] = "High response time"
            elif error_rate > 0.15:
                bottlenecks[system] = "High error rate"
            elif availability < 95.0:
                bottlenecks[system] = "Low availability"
        
        return bottlenecks
    
    def _detect_resource_contention(self, system_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Detect resource contention issues."""
        contention_issues = []
        
        # Look for systems with simultaneous performance degradation
        degraded_systems = []
        for system, metrics in system_metrics.items():
            response_times = metrics.get('response_times', [])
            if response_times and statistics.mean(response_times) > 3.0:
                degraded_systems.append(system)
        
        if len(degraded_systems) > 1:
            contention_issues.append(f"Potential resource contention among: {', '.join(degraded_systems)}")
        
        # Check for throughput limitations
        throughput_values = []
        for metrics in system_metrics.values():
            throughput = metrics.get('throughput', 0)
            if throughput > 0:
                throughput_values.append(throughput)
        
        if throughput_values and max(throughput_values) / min(throughput_values) > 10:
            contention_issues.append("Significant throughput imbalance detected")
        
        return contention_issues
    
    # === Availability Analysis ===
    
    def _calculate_availability_matrix(self, systems: List[str], endpoints: Dict[str, IntegrationEndpoint], time_window_hours: int) -> Dict[str, Dict[str, float]]:
        """Calculate availability matrix between systems."""
        availability_matrix = {}
        
        for system in systems:
            availability_matrix[system] = {}
            for other_system in systems:
                if system != other_system:
                    # Calculate cross-system availability
                    availability = self._calculate_cross_system_availability(
                        system, other_system, endpoints
                    )
                    availability_matrix[system][other_system] = availability
        
        return availability_matrix
    
    def _calculate_cross_system_availability(self, system1: str, system2: str, endpoints: Dict[str, IntegrationEndpoint]) -> float:
        """Calculate availability between two systems."""
        endpoint1 = self._get_endpoint_by_name(system1, endpoints)
        endpoint2 = self._get_endpoint_by_name(system2, endpoints)
        
        if not endpoint1 or not endpoint2:
            return 0.0
        
        # Base availability on both systems being connected
        if (endpoint1.status == IntegrationStatus.CONNECTED and 
            endpoint2.status == IntegrationStatus.CONNECTED):
            return min(endpoint1.availability_percentage, endpoint2.availability_percentage)
        
        return 0.0
    
    # === Data Flow Analysis ===
    
    def _analyze_data_flows(self, systems: List[str], endpoints: Dict[str, IntegrationEndpoint]) -> Dict[str, List[str]]:
        """Analyze data flow patterns between systems."""
        data_flows = {}
        
        for system in systems:
            endpoint = self._get_endpoint_by_name(system, endpoints)
            if endpoint:
                # Analyze based on endpoint configuration
                flows = []
                
                # Check for event streaming
                if endpoint.event_streaming:
                    flows.append("event_streaming")
                
                # Check for real-time sync
                if endpoint.real_time_sync:
                    flows.append("real_time_sync")
                
                # Check for API connections
                if endpoint.websocket_enabled:
                    flows.append("websocket")
                
                data_flows[system] = flows
        
        return data_flows
    
    def _perform_consistency_checks(self, systems: List[str], endpoints: Dict[str, IntegrationEndpoint]) -> Dict[str, bool]:
        """Perform data consistency checks across systems."""
        consistency_checks = {}
        
        for system in systems:
            endpoint = self._get_endpoint_by_name(system, endpoints)
            consistency_checks[system] = (
                endpoint is not None and 
                endpoint.status == IntegrationStatus.CONNECTED and
                endpoint.last_successful_connection is not None
            )
        
        return consistency_checks
    
    # === Predictive Analysis ===
    
    def _predict_system_failures(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Predict system failure probabilities."""
        failure_predictions = {}
        
        for system, metrics in system_metrics.items():
            failure_risk = 0.0
            
            # Risk from error rate
            error_rate = metrics.get('error_rate', 0.0)
            failure_risk += error_rate * 0.5
            
            # Risk from availability
            availability = metrics.get('availability', 100.0)
            if availability < 95.0:
                failure_risk += (100.0 - availability) / 100.0 * 0.3
            
            # Risk from response time
            response_times = metrics.get('response_times', [])
            if response_times:
                avg_response_time = statistics.mean(response_times)
                if avg_response_time > 5.0:
                    failure_risk += 0.2
            
            failure_predictions[system] = min(1.0, failure_risk)
        
        return failure_predictions
    
    def _forecast_capacity_needs(self, system_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Forecast capacity needs for systems."""
        capacity_forecasts = {}
        
        for system, metrics in system_metrics.items():
            forecast = {
                'current_utilization': 0.0,
                'predicted_growth': 0.0,
                'recommended_scaling': 0.0
            }
            
            # Estimate utilization from throughput and response times
            throughput = metrics.get('throughput', 0.0)
            response_times = metrics.get('response_times', [])
            
            if response_times and throughput > 0:
                avg_response_time = statistics.mean(response_times)
                # Simple utilization estimate
                utilization = min(100.0, (avg_response_time * throughput * 10))
                forecast['current_utilization'] = utilization
                
                # Predict growth based on current trends
                if utilization > 70:
                    forecast['predicted_growth'] = 20.0  # 20% growth expected
                    forecast['recommended_scaling'] = 1.5  # 50% scale up
                elif utilization > 50:
                    forecast['predicted_growth'] = 10.0
                    forecast['recommended_scaling'] = 1.2
            
            capacity_forecasts[system] = forecast
        
        return capacity_forecasts
    
    # === Recommendations ===
    
    def _generate_optimization_recommendations(self, analysis: CrossSystemAnalysis) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Analyze bottlenecks
        for system, bottleneck in analysis.bottleneck_analysis.items():
            recommendations.append(f"Optimize {system}: {bottleneck}")
        
        # Analyze low health scores
        for system, health_score in analysis.system_health_scores.items():
            if health_score < 70:
                recommendations.append(f"Improve health of {system} (current score: {health_score:.1f})")
        
        # Analyze correlations
        for system, correlations in analysis.system_correlations.items():
            high_correlations = [s for s, c in correlations.items() if c > 0.8]
            if len(high_correlations) > 2:
                recommendations.append(f"Consider load balancing for {system} due to high correlation")
        
        # Resource contention
        if analysis.resource_contention:
            recommendations.append("Address resource contention issues")
        
        return recommendations[:10]  # Return top 10 recommendations
    
    # === Helper Methods ===
    
    def _get_endpoint_by_name(self, system_name: str, endpoints: Dict[str, IntegrationEndpoint]) -> Optional[IntegrationEndpoint]:
        """Get endpoint by system name."""
        for endpoint in endpoints.values():
            if endpoint.name == system_name:
                return endpoint
        return None
    
    def _create_error_analysis(self, systems: List[str], error: str) -> CrossSystemAnalysis:
        """Create minimal analysis on error."""
        return CrossSystemAnalysis(
            analysis_id=f"error_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            systems_analyzed=systems,
            analysis_duration=0.0
        )
    
    def clear_cache(self):
        """Clear analysis and correlation caches."""
        self._analysis_cache.clear()
        self._correlation_cache.clear()
        self._system_metrics.clear()
        self.logger.info("Analysis cache cleared")


# Public API exports
__all__ = ['CrossSystemAnalyzer']