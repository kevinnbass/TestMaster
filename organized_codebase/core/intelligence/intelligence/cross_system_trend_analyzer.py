"""
Cross-System Trend Analyzer
===========================

Advanced trend analysis engine that identifies and predicts long-term trends
across all intelligence frameworks in the TestMaster ecosystem.

Agent A - Hour 19-21: Predictive Intelligence Enhancement
Building upon CrossSystemSemanticLearner and AdvancedPatternRecognizer.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
import logging
import json
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod

# Statistical and ML imports
try:
    from scipy import stats, signal
    from scipy.optimize import minimize_scalar
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    HAS_ADVANCED_STATS = True
except ImportError:
    HAS_ADVANCED_STATS = False
    logging.warning("Advanced statistical libraries not available. Using simplified methods.")


@dataclass
class TrendPoint:
    """A single point in a trend"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]
    source_system: str
    confidence: float = 1.0


@dataclass
class EvolutionaryTrend:
    """Represents an evolutionary trend across systems"""
    trend_id: str
    trend_type: str  # 'linear', 'exponential', 'logarithmic', 'cyclical', 'polynomial'
    strength: float  # 0-1, strength of the trend
    direction: str   # 'increasing', 'decreasing', 'stable', 'oscillating'
    slope: float
    r_squared: float
    start_timestamp: datetime
    end_timestamp: datetime
    affected_systems: List[str]
    trend_parameters: Dict[str, Any]
    confidence: float
    prediction_horizon: timedelta
    statistical_significance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            **asdict(self),
            'start_timestamp': self.start_timestamp.isoformat(),
            'end_timestamp': self.end_timestamp.isoformat(),
            'prediction_horizon': str(self.prediction_horizon)
        }


@dataclass
class SystemEvolutionPrediction:
    """Prediction of how a system will evolve"""
    system_name: str
    evolution_type: str  # 'growth', 'decline', 'stability', 'transformation'
    predicted_changes: Dict[str, Any]
    timeline: timedelta
    confidence: float
    driving_factors: List[str]
    risk_factors: List[str]
    recommended_actions: List[str]
    impact_assessment: Dict[str, float]


@dataclass
class CapacityForecast:
    """Forecast of future capacity needs"""
    resource_type: str
    current_utilization: float
    predicted_peak: float
    time_to_capacity: Optional[timedelta]
    growth_rate: float
    seasonal_patterns: Dict[str, float]
    bottleneck_probability: float
    scaling_recommendations: List[str]


class TrendDetector(ABC):
    """Abstract base class for trend detection algorithms"""
    
    @abstractmethod
    def detect_trends(self, data: List[TrendPoint]) -> List[EvolutionaryTrend]:
        """Detect trends in the given data"""
        pass
    
    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the detector"""
        pass


class LinearTrendDetector(TrendDetector):
    """Detects linear trends using regression analysis"""
    
    def __init__(self, min_r_squared: float = 0.5, min_points: int = 10):
        self.min_r_squared = min_r_squared
        self.min_points = min_points
    
    def detect_trends(self, data: List[TrendPoint]) -> List[EvolutionaryTrend]:
        """Detect linear trends"""
        trends = []
        
        if len(data) < self.min_points:
            return trends
        
        try:
            # Group data by system
            system_data = defaultdict(list)
            for point in data:
                system_data[point.source_system].append(point)
            
            # Analyze trends for each system
            for system, points in system_data.items():
                if len(points) < self.min_points:
                    continue
                
                # Prepare data for regression
                timestamps = [(p.timestamp - points[0].timestamp).total_seconds() for p in points]
                values = [p.value for p in points]
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
                r_squared = r_value ** 2
                
                if r_squared >= self.min_r_squared and p_value < 0.05:
                    # Determine trend direction
                    if abs(slope) < std_err:
                        direction = 'stable'
                    elif slope > 0:
                        direction = 'increasing'
                    else:
                        direction = 'decreasing'
                    
                    trend = EvolutionaryTrend(
                        trend_id=self._generate_trend_id(system, 'linear'),
                        trend_type='linear',
                        strength=r_squared,
                        direction=direction,
                        slope=slope,
                        r_squared=r_squared,
                        start_timestamp=points[0].timestamp,
                        end_timestamp=points[-1].timestamp,
                        affected_systems=[system],
                        trend_parameters={
                            'intercept': intercept,
                            'p_value': p_value,
                            'std_err': std_err
                        },
                        confidence=1 - p_value,
                        prediction_horizon=timedelta(hours=24),
                        statistical_significance=1 - p_value
                    )
                    
                    trends.append(trend)
            
        except Exception as e:
            logging.error(f"Linear trend detection failed: {e}")
        
        return trends
    
    def _generate_trend_id(self, system: str, trend_type: str) -> str:
        """Generate unique trend ID"""
        data = f"{system}_{trend_type}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information"""
        return {
            'detector_type': 'LinearTrendDetector',
            'min_r_squared': self.min_r_squared,
            'min_points': self.min_points
        }


class ExponentialTrendDetector(TrendDetector):
    """Detects exponential growth/decay trends"""
    
    def __init__(self, min_r_squared: float = 0.6, min_points: int = 15):
        self.min_r_squared = min_r_squared
        self.min_points = min_points
    
    def detect_trends(self, data: List[TrendPoint]) -> List[EvolutionaryTrend]:
        """Detect exponential trends"""
        trends = []
        
        if len(data) < self.min_points:
            return trends
        
        try:
            # Group data by system
            system_data = defaultdict(list)
            for point in data:
                system_data[point.source_system].append(point)
            
            for system, points in system_data.items():
                if len(points) < self.min_points:
                    continue
                
                # Prepare data for exponential fitting
                timestamps = [(p.timestamp - points[0].timestamp).total_seconds() for p in points]
                values = [max(p.value, 0.001) for p in points]  # Avoid log(0)
                
                # Try logarithmic transformation
                try:
                    log_values = np.log(values)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, log_values)
                    r_squared = r_value ** 2
                    
                    if r_squared >= self.min_r_squared and p_value < 0.05:
                        # This indicates exponential trend
                        growth_rate = slope
                        direction = 'increasing' if growth_rate > 0 else 'decreasing'
                        
                        trend = EvolutionaryTrend(
                            trend_id=self._generate_trend_id(system, 'exponential'),
                            trend_type='exponential',
                            strength=r_squared,
                            direction=direction,
                            slope=growth_rate,
                            r_squared=r_squared,
                            start_timestamp=points[0].timestamp,
                            end_timestamp=points[-1].timestamp,
                            affected_systems=[system],
                            trend_parameters={
                                'base': np.exp(intercept),
                                'growth_rate': growth_rate,
                                'p_value': p_value,
                                'doubling_time': np.log(2) / abs(growth_rate) if growth_rate != 0 else float('inf')
                            },
                            confidence=1 - p_value,
                            prediction_horizon=timedelta(hours=12),
                            statistical_significance=1 - p_value
                        )
                        
                        trends.append(trend)
                
                except (ValueError, RuntimeWarning):
                    # Log transformation failed, skip exponential detection
                    continue
            
        except Exception as e:
            logging.error(f"Exponential trend detection failed: {e}")
        
        return trends
    
    def _generate_trend_id(self, system: str, trend_type: str) -> str:
        """Generate unique trend ID"""
        data = f"{system}_{trend_type}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information"""
        return {
            'detector_type': 'ExponentialTrendDetector',
            'min_r_squared': self.min_r_squared,
            'min_points': self.min_points
        }


class CyclicalTrendDetector(TrendDetector):
    """Detects cyclical/periodic trends"""
    
    def __init__(self, min_correlation: float = 0.6, min_points: int = 48):
        self.min_correlation = min_correlation
        self.min_points = min_points
    
    def detect_trends(self, data: List[TrendPoint]) -> List[EvolutionaryTrend]:
        """Detect cyclical trends"""
        trends = []
        
        if len(data) < self.min_points:
            return trends
        
        try:
            # Group data by system
            system_data = defaultdict(list)
            for point in data:
                system_data[point.source_system].append(point)
            
            for system, points in system_data.items():
                if len(points) < self.min_points:
                    continue
                
                # Sort by timestamp
                points = sorted(points, key=lambda p: p.timestamp)
                values = np.array([p.value for p in points])
                
                # Detrend the data
                x = np.arange(len(values))
                detrended = values - np.polyval(np.polyfit(x, values, 1), x)
                
                # Find dominant frequency using autocorrelation
                periods_found = self._find_periods(detrended)
                
                for period, correlation in periods_found:
                    if correlation >= self.min_correlation:
                        # Calculate cycle parameters
                        cycle_duration = self._estimate_cycle_duration(points, period)
                        
                        trend = EvolutionaryTrend(
                            trend_id=self._generate_trend_id(system, 'cyclical'),
                            trend_type='cyclical',
                            strength=correlation,
                            direction='oscillating',
                            slope=0.0,  # Cyclical trends have no net slope
                            r_squared=correlation**2,
                            start_timestamp=points[0].timestamp,
                            end_timestamp=points[-1].timestamp,
                            affected_systems=[system],
                            trend_parameters={
                                'period': period,
                                'cycle_duration': str(cycle_duration),
                                'correlation': correlation,
                                'amplitude': np.std(detrended)
                            },
                            confidence=correlation,
                            prediction_horizon=cycle_duration,
                            statistical_significance=correlation
                        )
                        
                        trends.append(trend)
                        break  # Only take the strongest period
            
        except Exception as e:
            logging.error(f"Cyclical trend detection failed: {e}")
        
        return trends
    
    def _find_periods(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """Find periodic patterns using autocorrelation"""
        periods = []
        
        try:
            # Calculate autocorrelation for different lags
            max_lag = min(len(data) // 4, 100)  # Don't check beyond 1/4 of data length
            
            for lag in range(2, max_lag):
                if lag < len(data):
                    correlation = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(correlation) and correlation > 0:
                        periods.append((lag, abs(correlation)))
            
            # Sort by correlation strength
            periods.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logging.error(f"Period detection failed: {e}")
        
        return periods[:5]  # Return top 5 periods
    
    def _estimate_cycle_duration(self, points: List[TrendPoint], period_points: int) -> timedelta:
        """Estimate the duration of one cycle"""
        if len(points) <= period_points:
            return timedelta(hours=24)  # Default to daily cycle
        
        time_diff = points[period_points].timestamp - points[0].timestamp
        return time_diff
    
    def _generate_trend_id(self, system: str, trend_type: str) -> str:
        """Generate unique trend ID"""
        data = f"{system}_{trend_type}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information"""
        return {
            'detector_type': 'CyclicalTrendDetector',
            'min_correlation': self.min_correlation,
            'min_points': self.min_points
        }


class CrossSystemTrendAnalyzer:
    """
    Advanced trend analyzer that identifies and predicts long-term trends
    across all intelligence frameworks in the TestMaster ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Cross-System Trend Analyzer"""
        self.config = config or self._get_default_config()
        
        # Core components
        self.trend_detectors: Dict[str, TrendDetector] = {}
        self.detected_trends: Dict[str, List[EvolutionaryTrend]] = defaultdict(list)
        self.system_evolution_history: Dict[str, List[TrendPoint]] = defaultdict(list)
        self.capacity_forecasts: Dict[str, CapacityForecast] = {}
        
        # Data storage
        self.trend_data: deque = deque(maxlen=10000)
        self.cross_system_correlations: Dict[Tuple[str, str], float] = {}
        
        # Initialize trend detectors
        self._initialize_detectors()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Performance tracking
        self.analysis_metrics = {
            'trends_detected': 0,
            'systems_analyzed': 0,
            'cross_correlations_found': 0,
            'predictions_made': 0,
            'accuracy_scores': []
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'trend_detection_window': timedelta(hours=48),
            'min_data_points': 20,
            'correlation_threshold': 0.6,
            'trend_strength_threshold': 0.5,
            'prediction_horizons': [
                timedelta(hours=6),
                timedelta(hours=24),
                timedelta(days=7)
            ],
            'capacity_warning_threshold': 0.8,
            'cross_system_analysis_interval': timedelta(hours=1),
            'evolution_tracking_enabled': True,
            'statistical_significance_threshold': 0.05
        }
    
    def _initialize_detectors(self) -> None:
        """Initialize trend detection algorithms"""
        self.trend_detectors = {
            'linear': LinearTrendDetector(
                min_r_squared=self.config['trend_strength_threshold'],
                min_points=self.config['min_data_points']
            ),
            'exponential': ExponentialTrendDetector(
                min_r_squared=self.config['trend_strength_threshold'] + 0.1,
                min_points=self.config['min_data_points'] + 5
            ),
            'cyclical': CyclicalTrendDetector(
                min_correlation=self.config['correlation_threshold'],
                min_points=self.config['min_data_points'] * 2
            )
        }
    
    def _setup_logging(self) -> None:
        """Setup logging for the analyzer"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def add_trend_data(self, system_name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Add new data point for trend analysis"""
        try:
            trend_point = TrendPoint(
                timestamp=datetime.now(),
                value=value,
                metadata=metadata or {},
                source_system=system_name,
                confidence=1.0
            )
            
            self.system_evolution_history[system_name].append(trend_point)
            self.trend_data.append(trend_point)
            
            # Trigger analysis if we have enough data
            if len(self.system_evolution_history[system_name]) >= self.config['min_data_points']:
                await self._analyze_system_trends(system_name)
            
        except Exception as e:
            self.logger.error(f"Failed to add trend data for {system_name}: {e}")
    
    async def analyze_cross_system_trends(self) -> Dict[str, List[EvolutionaryTrend]]:
        """Analyze trends across all systems"""
        all_trends = {}
        
        try:
            # Analyze trends for each system
            for system_name in self.system_evolution_history.keys():
                await self._analyze_system_trends(system_name)
            
            # Find cross-system correlations
            await self._analyze_cross_system_correlations()
            
            # Generate evolution predictions
            await self._generate_evolution_predictions()
            
            # Update capacity forecasts
            await self._update_capacity_forecasts()
            
            all_trends = dict(self.detected_trends)
            self.analysis_metrics['systems_analyzed'] = len(all_trends)
            
        except Exception as e:
            self.logger.error(f"Cross-system trend analysis failed: {e}")
        
        return all_trends
    
    async def _analyze_system_trends(self, system_name: str) -> None:
        """Analyze trends for a specific system"""
        try:
            system_data = self.system_evolution_history[system_name]
            
            if len(system_data) < self.config['min_data_points']:
                return
            
            # Apply trend detection algorithms
            detected_trends = []
            
            for detector_name, detector in self.trend_detectors.items():
                trends = detector.detect_trends(system_data[-200:])  # Use recent data
                detected_trends.extend(trends)
            
            # Filter and rank trends
            significant_trends = [
                trend for trend in detected_trends
                if trend.statistical_significance >= (1 - self.config['statistical_significance_threshold'])
                and trend.strength >= self.config['trend_strength_threshold']
            ]
            
            # Sort by strength
            significant_trends.sort(key=lambda t: t.strength, reverse=True)
            
            # Update detected trends
            self.detected_trends[system_name] = significant_trends[:5]  # Keep top 5
            self.analysis_metrics['trends_detected'] += len(significant_trends)
            
        except Exception as e:
            self.logger.error(f"System trend analysis failed for {system_name}: {e}")
    
    async def _analyze_cross_system_correlations(self) -> None:
        """Analyze correlations between different systems"""
        try:
            systems = list(self.system_evolution_history.keys())
            
            for i, system_a in enumerate(systems):
                for system_b in systems[i+1:]:
                    correlation = await self._calculate_system_correlation(system_a, system_b)
                    
                    if abs(correlation) >= self.config['correlation_threshold']:
                        self.cross_system_correlations[(system_a, system_b)] = correlation
                        self.analysis_metrics['cross_correlations_found'] += 1
            
        except Exception as e:
            self.logger.error(f"Cross-system correlation analysis failed: {e}")
    
    async def _calculate_system_correlation(self, system_a: str, system_b: str) -> float:
        """Calculate correlation between two systems"""
        try:
            data_a = self.system_evolution_history[system_a]
            data_b = self.system_evolution_history[system_b]
            
            if len(data_a) < 10 or len(data_b) < 10:
                return 0.0
            
            # Align timestamps and calculate correlation
            # This is a simplified approach - in practice, would use more sophisticated alignment
            min_len = min(len(data_a), len(data_b))
            values_a = [point.value for point in data_a[-min_len:]]
            values_b = [point.value for point in data_b[-min_len:]]
            
            if HAS_ADVANCED_STATS:
                correlation, p_value = stats.pearsonr(values_a, values_b)
                return correlation if p_value < 0.05 else 0.0
            else:
                # Simple correlation calculation
                return np.corrcoef(values_a, values_b)[0, 1] if min_len > 1 else 0.0
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed for {system_a} vs {system_b}: {e}")
            return 0.0
    
    async def _generate_evolution_predictions(self) -> None:
        """Generate predictions for system evolution"""
        try:
            for system_name, trends in self.detected_trends.items():
                if not trends:
                    continue
                
                # Use the strongest trend for prediction
                primary_trend = trends[0]
                
                prediction = await self._predict_system_evolution(system_name, primary_trend)
                
                if prediction:
                    self.analysis_metrics['predictions_made'] += 1
                    
                    # Store prediction for future validation
                    # (In production, would store these for accuracy tracking)
            
        except Exception as e:
            self.logger.error(f"Evolution prediction generation failed: {e}")
    
    async def _predict_system_evolution(self, system_name: str, trend: EvolutionaryTrend) -> Optional[SystemEvolutionPrediction]:
        """Predict how a system will evolve based on its trends"""
        try:
            # Determine evolution type based on trend characteristics
            evolution_type = self._classify_evolution_type(trend)
            
            # Predict specific changes
            predicted_changes = self._predict_specific_changes(system_name, trend)
            
            # Assess risks and recommendations
            risk_factors = self._assess_risk_factors(trend)
            recommendations = self._generate_recommendations(evolution_type, trend)
            
            # Calculate impact assessment
            impact_assessment = self._assess_impact(system_name, trend)
            
            prediction = SystemEvolutionPrediction(
                system_name=system_name,
                evolution_type=evolution_type,
                predicted_changes=predicted_changes,
                timeline=trend.prediction_horizon,
                confidence=trend.confidence,
                driving_factors=self._identify_driving_factors(trend),
                risk_factors=risk_factors,
                recommended_actions=recommendations,
                impact_assessment=impact_assessment
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"System evolution prediction failed for {system_name}: {e}")
            return None
    
    def _classify_evolution_type(self, trend: EvolutionaryTrend) -> str:
        """Classify the type of evolution based on trend characteristics"""
        if trend.trend_type == 'exponential' and trend.direction == 'increasing':
            return 'rapid_growth'
        elif trend.trend_type == 'exponential' and trend.direction == 'decreasing':
            return 'rapid_decline'
        elif trend.trend_type == 'linear' and trend.direction == 'increasing':
            return 'steady_growth'
        elif trend.trend_type == 'linear' and trend.direction == 'decreasing':
            return 'gradual_decline'
        elif trend.trend_type == 'cyclical':
            return 'cyclical_stability'
        elif trend.direction == 'stable':
            return 'stability'
        else:
            return 'transformation'
    
    def _predict_specific_changes(self, system_name: str, trend: EvolutionaryTrend) -> Dict[str, Any]:
        """Predict specific changes in the system"""
        changes = {}
        
        try:
            if trend.trend_type == 'linear':
                # Linear extrapolation
                time_horizon = trend.prediction_horizon.total_seconds()
                predicted_change = trend.slope * time_horizon
                changes['value_change'] = predicted_change
                changes['percentage_change'] = predicted_change / max(abs(trend.trend_parameters.get('intercept', 1)), 1)
                
            elif trend.trend_type == 'exponential':
                # Exponential extrapolation
                growth_rate = trend.trend_parameters.get('growth_rate', 0)
                time_horizon = trend.prediction_horizon.total_seconds()
                changes['growth_multiplier'] = np.exp(growth_rate * time_horizon)
                changes['doubling_time'] = trend.trend_parameters.get('doubling_time', float('inf'))
                
            elif trend.trend_type == 'cyclical':
                # Cyclical prediction
                changes['next_peak_time'] = trend.trend_parameters.get('period', 24) / 4  # Quarter cycle
                changes['amplitude'] = trend.trend_parameters.get('amplitude', 1)
                
        except Exception as e:
            self.logger.error(f"Specific change prediction failed: {e}")
        
        return changes
    
    def _assess_risk_factors(self, trend: EvolutionaryTrend) -> List[str]:
        """Assess risk factors based on trend characteristics"""
        risks = []
        
        if trend.trend_type == 'exponential' and trend.direction == 'increasing':
            risks.append('Resource exhaustion risk')
            risks.append('Scalability bottleneck risk')
            
        elif trend.trend_type == 'exponential' and trend.direction == 'decreasing':
            risks.append('System failure risk')
            risks.append('Performance degradation risk')
            
        elif trend.strength > 0.9:
            risks.append('High dependency risk')
            
        elif trend.statistical_significance < 0.9:
            risks.append('Prediction uncertainty risk')
            
        if trend.r_squared < 0.7:
            risks.append('Model accuracy risk')
            
        return risks
    
    def _generate_recommendations(self, evolution_type: str, trend: EvolutionaryTrend) -> List[str]:
        """Generate recommendations based on evolution type"""
        recommendations = []
        
        if evolution_type == 'rapid_growth':
            recommendations.append('Scale infrastructure proactively')
            recommendations.append('Monitor resource utilization closely')
            recommendations.append('Implement auto-scaling mechanisms')
            
        elif evolution_type == 'rapid_decline':
            recommendations.append('Investigate root causes immediately')
            recommendations.append('Implement performance recovery measures')
            recommendations.append('Consider system health diagnostics')
            
        elif evolution_type == 'steady_growth':
            recommendations.append('Plan for gradual capacity increases')
            recommendations.append('Monitor trend continuation')
            
        elif evolution_type == 'cyclical_stability':
            recommendations.append('Optimize for cyclical patterns')
            recommendations.append('Implement predictive resource allocation')
            
        return recommendations
    
    def _identify_driving_factors(self, trend: EvolutionaryTrend) -> List[str]:
        """Identify factors driving the trend"""
        factors = []
        
        # This would be enhanced with actual system analysis
        factors.append(f"System behavior pattern: {trend.trend_type}")
        factors.append(f"Statistical strength: {trend.strength:.2f}")
        
        if trend.trend_type == 'cyclical':
            factors.append("Periodic system behavior")
        elif trend.direction == 'increasing':
            factors.append("Increasing demand or load")
        elif trend.direction == 'decreasing':
            factors.append("Decreasing efficiency or performance")
            
        return factors
    
    def _assess_impact(self, system_name: str, trend: EvolutionaryTrend) -> Dict[str, float]:
        """Assess the impact of the trend on system performance"""
        impact = {
            'performance_impact': 0.0,
            'resource_impact': 0.0,
            'reliability_impact': 0.0,
            'cost_impact': 0.0
        }
        
        try:
            # Base impact on trend strength and direction
            base_impact = trend.strength
            
            if trend.direction == 'increasing' and trend.trend_type == 'exponential':
                impact['resource_impact'] = min(base_impact * 2, 1.0)
                impact['cost_impact'] = min(base_impact * 1.5, 1.0)
                
            elif trend.direction == 'decreasing':
                impact['performance_impact'] = min(base_impact * 1.8, 1.0)
                impact['reliability_impact'] = min(base_impact * 1.5, 1.0)
                
        except Exception as e:
            self.logger.error(f"Impact assessment failed: {e}")
        
        return impact
    
    async def _update_capacity_forecasts(self) -> None:
        """Update capacity forecasts based on trend analysis"""
        try:
            for system_name, trends in self.detected_trends.items():
                if not trends:
                    continue
                
                # Find resource utilization trends
                primary_trend = trends[0]
                
                if primary_trend.direction in ['increasing', 'exponential']:
                    forecast = await self._generate_capacity_forecast(system_name, primary_trend)
                    if forecast:
                        self.capacity_forecasts[system_name] = forecast
            
        except Exception as e:
            self.logger.error(f"Capacity forecast update failed: {e}")
    
    async def _generate_capacity_forecast(self, system_name: str, trend: EvolutionaryTrend) -> Optional[CapacityForecast]:
        """Generate capacity forecast for a system"""
        try:
            # Estimate current utilization (simplified)
            recent_data = self.system_evolution_history[system_name][-10:]
            current_utilization = np.mean([p.value for p in recent_data]) if recent_data else 0.5
            
            # Predict peak utilization
            if trend.trend_type == 'linear':
                predicted_peak = current_utilization + trend.slope * 3600  # 1 hour ahead
            elif trend.trend_type == 'exponential':
                growth_rate = trend.trend_parameters.get('growth_rate', 0.01)
                predicted_peak = current_utilization * np.exp(growth_rate * 3600)
            else:
                predicted_peak = current_utilization * 1.2  # Conservative estimate
            
            # Calculate time to capacity
            time_to_capacity = None
            if predicted_peak > self.config['capacity_warning_threshold']:
                # Rough estimate based on trend
                if trend.slope > 0:
                    remaining_capacity = 1.0 - current_utilization
                    time_to_capacity = timedelta(seconds=remaining_capacity / trend.slope)
                elif trend.trend_type == 'exponential':
                    growth_rate = trend.trend_parameters.get('growth_rate', 0.01)
                    if growth_rate > 0:
                        time_to_capacity = timedelta(seconds=np.log(1.0 / current_utilization) / growth_rate)
            
            forecast = CapacityForecast(
                resource_type=system_name,
                current_utilization=current_utilization,
                predicted_peak=min(predicted_peak, 1.0),
                time_to_capacity=time_to_capacity,
                growth_rate=trend.slope,
                seasonal_patterns={},  # Would be populated from cyclical analysis
                bottleneck_probability=min(predicted_peak, 1.0),
                scaling_recommendations=self._generate_scaling_recommendations(predicted_peak)
            )
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Capacity forecast generation failed for {system_name}: {e}")
            return None
    
    def _generate_scaling_recommendations(self, predicted_peak: float) -> List[str]:
        """Generate scaling recommendations based on predicted peak"""
        recommendations = []
        
        if predicted_peak > 0.9:
            recommendations.append('Urgent scaling required')
            recommendations.append('Implement emergency capacity measures')
        elif predicted_peak > 0.8:
            recommendations.append('Proactive scaling recommended')
            recommendations.append('Monitor resource allocation closely')
        elif predicted_peak > 0.7:
            recommendations.append('Plan for capacity expansion')
        
        return recommendations
    
    async def get_trend_summary(self, system_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive trend summary"""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'systems_analyzed': list(self.detected_trends.keys()),
            'total_trends_detected': sum(len(trends) for trends in self.detected_trends.values()),
            'cross_system_correlations': len(self.cross_system_correlations),
            'capacity_warnings': [],
            'performance_metrics': dict(self.analysis_metrics)
        }
        
        try:
            if system_name:
                # System-specific summary
                if system_name in self.detected_trends:
                    summary['system_trends'] = [trend.to_dict() for trend in self.detected_trends[system_name]]
                    
                if system_name in self.capacity_forecasts:
                    summary['capacity_forecast'] = asdict(self.capacity_forecasts[system_name])
            else:
                # Global summary
                summary['all_trends'] = {
                    system: [trend.to_dict() for trend in trends]
                    for system, trends in self.detected_trends.items()
                }
                
                summary['all_capacity_forecasts'] = {
                    system: asdict(forecast)
                    for system, forecast in self.capacity_forecasts.items()
                }
            
            # Add capacity warnings
            for system, forecast in self.capacity_forecasts.items():
                if forecast.predicted_peak > self.config['capacity_warning_threshold']:
                    summary['capacity_warnings'].append({
                        'system': system,
                        'predicted_peak': forecast.predicted_peak,
                        'time_to_capacity': str(forecast.time_to_capacity) if forecast.time_to_capacity else None
                    })
            
        except Exception as e:
            self.logger.error(f"Trend summary generation failed: {e}")
            summary['error'] = str(e)
        
        return summary
    
    async def predict_future_trends(self, horizon: timedelta) -> Dict[str, Any]:
        """Predict future trends within the given horizon"""
        predictions = {
            'prediction_horizon': str(horizon),
            'system_predictions': {},
            'cross_system_impacts': {},
            'risk_assessment': {},
            'confidence_scores': {}
        }
        
        try:
            for system_name, trends in self.detected_trends.items():
                if not trends:
                    continue
                
                primary_trend = trends[0]
                
                # Extrapolate trend into the future
                future_values = await self._extrapolate_trend(primary_trend, horizon)
                
                predictions['system_predictions'][system_name] = {
                    'trend_type': primary_trend.trend_type,
                    'predicted_values': future_values,
                    'confidence': primary_trend.confidence
                }
                
                predictions['confidence_scores'][system_name] = primary_trend.confidence
                
                # Assess risks for this prediction
                predictions['risk_assessment'][system_name] = self._assess_prediction_risks(primary_trend, horizon)
            
            # Analyze cross-system impacts
            for (system_a, system_b), correlation in self.cross_system_correlations.items():
                if system_a in predictions['system_predictions']:
                    predictions['cross_system_impacts'][f"{system_a}->{system_b}"] = {
                        'correlation': correlation,
                        'predicted_impact': correlation * predictions['confidence_scores'].get(system_a, 0.5)
                    }
            
        except Exception as e:
            self.logger.error(f"Future trend prediction failed: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    async def _extrapolate_trend(self, trend: EvolutionaryTrend, horizon: timedelta) -> List[float]:
        """Extrapolate a trend into the future"""
        future_values = []
        
        try:
            # Get current value (simplified - would use actual current system state)
            current_value = 1.0  # Placeholder
            
            # Generate predictions for multiple points in the future
            num_points = min(int(horizon.total_seconds() / 3600), 24)  # Hourly points, max 24
            
            for i in range(1, num_points + 1):
                time_offset = timedelta(hours=i)
                
                if trend.trend_type == 'linear':
                    predicted_value = current_value + trend.slope * time_offset.total_seconds()
                elif trend.trend_type == 'exponential':
                    growth_rate = trend.trend_parameters.get('growth_rate', 0.01)
                    predicted_value = current_value * np.exp(growth_rate * time_offset.total_seconds())
                elif trend.trend_type == 'cyclical':
                    period = trend.trend_parameters.get('period', 24)
                    amplitude = trend.trend_parameters.get('amplitude', 0.1)
                    phase_offset = (time_offset.total_seconds() / 3600) % period
                    predicted_value = current_value + amplitude * np.sin(2 * np.pi * phase_offset / period)
                else:
                    predicted_value = current_value  # Stable prediction
                
                future_values.append(float(predicted_value))
            
        except Exception as e:
            self.logger.error(f"Trend extrapolation failed: {e}")
            future_values = [1.0] * 10  # Fallback stable prediction
        
        return future_values
    
    def _assess_prediction_risks(self, trend: EvolutionaryTrend, horizon: timedelta) -> Dict[str, float]:
        """Assess risks associated with a prediction"""
        risks = {
            'model_uncertainty': 1.0 - trend.confidence,
            'extrapolation_risk': min(horizon.total_seconds() / trend.prediction_horizon.total_seconds(), 2.0),
            'trend_stability_risk': 1.0 - trend.r_squared,
            'data_quality_risk': 1.0 - trend.statistical_significance
        }
        
        # Overall risk score (higher is riskier)
        risks['overall_risk'] = np.mean(list(risks.values()))
        
        return risks
    
    def save_analysis_state(self, filepath: str) -> None:
        """Save analyzer state to file"""
        try:
            state = {
                'config': self.config,
                'detected_trends': {
                    system: [trend.to_dict() for trend in trends]
                    for system, trends in self.detected_trends.items()
                },
                'capacity_forecasts': {
                    system: asdict(forecast)
                    for system, forecast in self.capacity_forecasts.items()
                },
                'cross_system_correlations': {
                    f"{sys_a}|{sys_b}": corr
                    for (sys_a, sys_b), corr in self.cross_system_correlations.items()
                },
                'analysis_metrics': dict(self.analysis_metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert timedelta objects to strings
            def convert_timedeltas(obj):
                if isinstance(obj, timedelta):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_timedeltas(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timedeltas(item) for item in obj]
                return obj
            
            state = convert_timedeltas(state)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            self.logger.info(f"Analysis state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis state: {e}")


# Factory function for easy instantiation
def create_cross_system_trend_analyzer(config: Dict[str, Any] = None) -> CrossSystemTrendAnalyzer:
    """Create and return a configured Cross-System Trend Analyzer"""
    return CrossSystemTrendAnalyzer(config)


# Export main classes
__all__ = [
    'CrossSystemTrendAnalyzer',
    'EvolutionaryTrend', 
    'SystemEvolutionPrediction', 
    'CapacityForecast',
    'TrendPoint',
    'create_cross_system_trend_analyzer'
]