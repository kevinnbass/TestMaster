"""
Pattern Detection Engine
========================

Detects patterns in streaming data for the intelligence system.

Author: Agent A - Integration Fix
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Pattern:
    """Represents a detected pattern."""
    pattern_type: str
    confidence: float
    start_index: int
    end_index: int
    metadata: Dict[str, Any]


class PatternDetector:
    """Base pattern detector class."""
    
    def __init__(self, window_size: int = 100):
        """Initialize pattern detector."""
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.pattern_history = []
        self.index = 0
    
    def process_stream(self, value: float) -> Optional[Pattern]:
        """
        Process a single value from the stream.
        
        Args:
            value: The value to process
            
        Returns:
            Detected pattern if any, None otherwise
        """
        self.data_buffer.append(value)
        self.index += 1
        
        if len(self.data_buffer) < 3:
            return None
        
        # Check for various patterns
        pattern = self._detect_pattern()
        
        if pattern:
            self.pattern_history.append(pattern)
            
        return pattern
    
    def _detect_pattern(self) -> Optional[Pattern]:
        """Detect patterns in the current buffer."""
        data = np.array(self.data_buffer)
        
        # Check for spike pattern
        spike = self._detect_spike(data)
        if spike:
            return spike
        
        # Check for trend pattern
        trend = self._detect_trend(data)
        if trend:
            return trend
        
        # Check for cycle pattern
        cycle = self._detect_cycle(data)
        if cycle:
            return cycle
        
        return None
    
    def _detect_spike(self, data: np.ndarray) -> Optional[Pattern]:
        """Detect spike patterns."""
        if len(data) < 3:
            return None
        
        mean = np.mean(data[:-1])
        std = np.std(data[:-1])
        
        if std == 0:
            return None
        
        z_score = abs((data[-1] - mean) / std)
        
        if z_score > 3:  # 3 standard deviations
            return Pattern(
                pattern_type='spike',
                confidence=min(z_score / 5, 1.0),
                start_index=self.index - 1,
                end_index=self.index,
                metadata={'z_score': float(z_score), 'direction': 'up' if data[-1] > mean else 'down'}
            )
        
        return None
    
    def _detect_trend(self, data: np.ndarray) -> Optional[Pattern]:
        """Detect trend patterns."""
        if len(data) < 10:
            return None
        
        # Check last 10 points for consistent trend
        recent_data = data[-10:]
        diffs = np.diff(recent_data)
        
        # Check if mostly increasing or decreasing
        positive_ratio = np.sum(diffs > 0) / len(diffs)
        
        if positive_ratio > 0.8:
            return Pattern(
                pattern_type='trend',
                confidence=positive_ratio,
                start_index=self.index - 10,
                end_index=self.index,
                metadata={'direction': 'increasing', 'strength': float(np.mean(diffs))}
            )
        elif positive_ratio < 0.2:
            return Pattern(
                pattern_type='trend',
                confidence=1 - positive_ratio,
                start_index=self.index - 10,
                end_index=self.index,
                metadata={'direction': 'decreasing', 'strength': float(np.mean(diffs))}
            )
        
        return None
    
    def _detect_cycle(self, data: np.ndarray) -> Optional[Pattern]:
        """Detect cyclical patterns."""
        if len(data) < 20:
            return None
        
        # Simple autocorrelation check
        for lag in range(2, min(10, len(data) // 2)):
            correlation = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            
            if correlation > 0.7:
                return Pattern(
                    pattern_type='cycle',
                    confidence=correlation,
                    start_index=self.index - len(data),
                    end_index=self.index,
                    metadata={'period': lag, 'correlation': float(correlation)}
                )
        
        return None
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        if not self.pattern_history:
            return {'total_patterns': 0, 'patterns_by_type': {}}
        
        patterns_by_type = {}
        for pattern in self.pattern_history:
            if pattern.pattern_type not in patterns_by_type:
                patterns_by_type[pattern.pattern_type] = 0
            patterns_by_type[pattern.pattern_type] += 1
        
        return {
            'total_patterns': len(self.pattern_history),
            'patterns_by_type': patterns_by_type,
            'average_confidence': np.mean([p.confidence for p in self.pattern_history])
        }


def create_pattern_detector(config: Optional[Dict[str, Any]] = None) -> PatternDetector:
    """
    Create a pattern detector with the given configuration.
    
    Args:
        config: Configuration for the pattern detector
        
    Returns:
        Configured pattern detector instance
    """
    if config is None:
        config = {}
    
    window_size = config.get('window_size', 100)
    return PatternDetector(window_size=window_size)