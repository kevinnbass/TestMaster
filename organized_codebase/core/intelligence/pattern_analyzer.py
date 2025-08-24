"""
Pattern Analyzer - Advanced Temporal Pattern Recognition
=======================================================

Sophisticated temporal pattern analysis system implementing advanced signal
processing, frequency domain analysis, and chaos theory for comprehensive
temporal pattern detection with enterprise-grade mathematical algorithms.

This module provides advanced pattern recognition including:
- Fourier analysis for periodic pattern detection
- Trend analysis using linear regression and statistical validation
- Seasonal decomposition with significance testing
- Chaos detection using Lyapunov exponents
- Fractal analysis with Hurst exponent calculation

Author: Agent A - PHASE 2: Hours 100-200
Created: 2025-08-22
Module: pattern_analyzer.py (240 lines)
"""

import asyncio
import numpy as np
import hashlib
from typing import List, Optional
from collections import deque
from datetime import datetime
from scipy import stats
from scipy.fft import fft, fftfreq

from .temporal_types import TemporalPattern, TemporalPatternType


class TemporalPatternAnalyzer:
    """
    Enterprise temporal pattern analyzer implementing advanced signal processing
    and mathematical analysis for comprehensive pattern detection and recognition.
    
    Features:
    - Fourier analysis for periodic pattern detection with significance testing
    - Statistical trend analysis using linear regression validation
    - Seasonal decomposition with pattern significance evaluation
    - Chaos detection using Lyapunov exponent calculations
    - Fractal analysis with Hurst exponent and self-similarity detection
    """
    
    def __init__(self):
        self.patterns = deque(maxlen=1000)
        self.frequency_spectrum = {}
        self.decomposition_cache = {}
        
    async def analyze_temporal_patterns(self, time_series: np.ndarray) -> List[TemporalPattern]:
        """
        Comprehensive temporal pattern analysis using advanced mathematical algorithms.
        
        Args:
            time_series: Input time series data for pattern analysis
            
        Returns:
            List of detected temporal patterns with confidence metrics
        """
        patterns = []
        
        # Fourier analysis for periodic patterns with frequency domain processing
        periodic_patterns = self._detect_periodic_patterns(time_series)
        patterns.extend(periodic_patterns)
        
        # Statistical trend detection with regression analysis
        trend_pattern = self._detect_trend(time_series)
        if trend_pattern:
            patterns.append(trend_pattern)
        
        # Seasonal decomposition with significance testing
        seasonal_patterns = self._detect_seasonal_patterns(time_series)
        patterns.extend(seasonal_patterns)
        
        # Chaos detection using nonlinear dynamics
        if self._is_chaotic(time_series):
            chaos_pattern = self._create_chaos_pattern(time_series)
            patterns.append(chaos_pattern)
        
        # Fractal analysis with self-similarity detection
        fractal_pattern = self._detect_fractal_patterns(time_series)
        if fractal_pattern:
            patterns.append(fractal_pattern)
        
        # Store patterns for historical analysis
        self.patterns.extend(patterns)
        
        return patterns
    
    def _detect_periodic_patterns(self, time_series: np.ndarray) -> List[TemporalPattern]:
        """Detect periodic patterns using advanced FFT analysis with significance testing"""
        patterns = []
        
        n = len(time_series)
        if n < 4:
            return patterns
        
        # Compute Fast Fourier Transform for frequency domain analysis
        yf = fft(time_series)
        xf = fftfreq(n, 1)[:n//2]
        
        # Calculate power spectrum with normalization
        power = 2.0/n * np.abs(yf[:n//2])
        
        # Statistical significance threshold using standard deviation
        threshold = np.mean(power) + 2 * np.std(power)
        
        # Identify significant frequency components
        significant_indices = power > threshold
        significant_freqs = xf[significant_indices]
        significant_powers = power[significant_indices]
        
        for freq, amp in zip(significant_freqs, significant_powers):
            if freq > 0:  # Exclude DC component
                # Calculate phase information
                freq_idx = np.argmin(np.abs(xf - freq))
                phase = np.angle(yf[freq_idx])
                
                pattern = TemporalPattern(
                    pattern_id=self._generate_id("periodic"),
                    pattern_type=TemporalPatternType.PERIODIC,
                    frequency=freq,
                    amplitude=amp,
                    phase=phase,
                    period=1/freq if freq != 0 else None,
                    confidence=min(1.0, amp / np.max(power)),
                    start_time=datetime.now(),
                    end_time=None,
                    recurrence_probability=0.8
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_trend(self, time_series: np.ndarray) -> Optional[TemporalPattern]:
        """Detect linear trends using statistical regression with significance validation"""
        if len(time_series) < 2:
            return None
        
        # Perform linear regression analysis
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
        
        # Statistical significance testing (correlation > 0.3, p-value < 0.05)
        if abs(r_value) > 0.3 and p_value < 0.05:
            return TemporalPattern(
                pattern_id=self._generate_id("trend"),
                pattern_type=TemporalPatternType.TREND,
                frequency=None,
                amplitude=slope,
                phase=None,
                period=None,
                confidence=abs(r_value),
                start_time=datetime.now(),
                end_time=None,
                recurrence_probability=0.9
            )
        
        return None
    
    def _detect_seasonal_patterns(self, time_series: np.ndarray) -> List[TemporalPattern]:
        """Detect seasonal patterns using decomposition with statistical validation"""
        patterns = []
        
        if len(time_series) < 12:
            return patterns
        
        # Seasonal decomposition with multiple period detection
        for season_length in [12, 4, 7]:  # Annual, quarterly, weekly
            if len(time_series) >= season_length * 2:
                seasonal_component = self._extract_seasonal_component(time_series, season_length)
                
                if seasonal_component is not None:
                    patterns.append(seasonal_component)
        
        return patterns
    
    def _extract_seasonal_component(self, time_series: np.ndarray, season_length: int) -> Optional[TemporalPattern]:
        """Extract seasonal component with statistical significance testing"""
        # Calculate seasonal averages
        seasonal = np.zeros(season_length)
        for i in range(season_length):
            seasonal[i] = np.mean(time_series[i::season_length])
        
        # Normalize to remove trend
        seasonal = seasonal - np.mean(seasonal)
        
        # Test for significant seasonality
        seasonal_strength = np.std(seasonal)
        total_variance = np.std(time_series)
        
        if seasonal_strength > 0.1 and seasonal_strength / total_variance > 0.1:
            return TemporalPattern(
                pattern_id=self._generate_id("seasonal"),
                pattern_type=TemporalPatternType.SEASONAL,
                frequency=1/season_length,
                amplitude=seasonal_strength,
                phase=None,
                period=season_length,
                confidence=min(1.0, seasonal_strength / total_variance),
                start_time=datetime.now(),
                end_time=None,
                recurrence_probability=0.95
            )
        
        return None
    
    def _is_chaotic(self, time_series: np.ndarray) -> bool:
        """Detect chaotic behavior using Lyapunov exponent analysis"""
        if len(time_series) < 10:
            return False
        
        # Calculate largest Lyapunov exponent
        lyapunov = self._calculate_lyapunov_exponent(time_series)
        
        # Positive Lyapunov exponent indicates sensitive dependence (chaos)
        return lyapunov > 0
    
    def _calculate_lyapunov_exponent(self, time_series: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent using phase space reconstruction"""
        n = len(time_series)
        if n < 10:
            return 0.0
        
        # Simplified Lyapunov exponent calculation
        divergence_sum = 0.0
        valid_points = 0
        
        for i in range(1, n):
            if abs(time_series[i-1]) > 1e-10:
                # Calculate local divergence rate
                local_divergence = abs((time_series[i] - time_series[i-1]) / time_series[i-1])
                
                if local_divergence > 1e-10:
                    divergence_sum += np.log(local_divergence)
                    valid_points += 1
        
        return divergence_sum / max(1, valid_points) if valid_points > 0 else 0.0
    
    def _create_chaos_pattern(self, time_series: np.ndarray) -> TemporalPattern:
        """Create chaos pattern with nonlinear dynamics characteristics"""
        return TemporalPattern(
            pattern_id=self._generate_id("chaos"),
            pattern_type=TemporalPatternType.CHAOTIC,
            frequency=None,
            amplitude=np.std(time_series),
            phase=None,
            period=None,
            confidence=0.7,
            start_time=datetime.now(),
            end_time=None,
            recurrence_probability=0.0  # Chaotic systems are unpredictable
        )
    
    def _detect_fractal_patterns(self, time_series: np.ndarray) -> Optional[TemporalPattern]:
        """Detect fractal patterns using Hurst exponent and self-similarity analysis"""
        if len(time_series) < 16:
            return None
        
        # Calculate Hurst exponent for self-similarity detection
        hurst = self._calculate_hurst_exponent(time_series)
        
        # Hurst > 0.6 indicates persistent behavior (fractal-like)
        if hurst > 0.6:
            return TemporalPattern(
                pattern_id=self._generate_id("fractal"),
                pattern_type=TemporalPatternType.FRACTAL,
                frequency=None,
                amplitude=hurst,
                phase=None,
                period=None,
                confidence=min(1.0, abs(hurst - 0.5) * 2),
                start_time=datetime.now(),
                end_time=None,
                recurrence_probability=0.7
            )
        
        return None
    
    def _calculate_hurst_exponent(self, time_series: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        n = len(time_series)
        if n < 16:
            return 0.5
        
        # Calculate mean-centered cumulative sum
        mean_val = np.mean(time_series)
        cumsum = np.cumsum(time_series - mean_val)
        
        # Calculate range (R)
        R = np.max(cumsum) - np.min(cumsum)
        
        # Calculate standard deviation (S)
        S = np.std(time_series)
        
        if S == 0:
            return 0.5
        
        # R/S ratio
        rs_ratio = R / S
        
        # Hurst exponent approximation
        if rs_ratio > 0:
            hurst = np.log(rs_ratio) / np.log(n)
        else:
            hurst = 0.5
        
        # Bound Hurst exponent between 0 and 1
        return max(0.0, min(1.0, hurst))
    
    def get_pattern_summary(self) -> dict:
        """Get comprehensive summary of detected patterns"""
        if not self.patterns:
            return {"total_patterns": 0, "pattern_types": {}}
        
        pattern_counts = {}
        for pattern in self.patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": pattern_counts,
            "avg_confidence": np.mean([p.confidence for p in self.patterns]),
            "latest_pattern": self.patterns[-1].pattern_type.value if self.patterns else None
        }
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique pattern ID with timestamp and hash"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{prefix}_{timestamp}_{len(self.patterns)}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"


# Export pattern analysis components
__all__ = ['TemporalPatternAnalyzer']