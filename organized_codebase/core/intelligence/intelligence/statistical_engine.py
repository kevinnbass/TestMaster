"""
Statistical Analysis Engine
===========================

Provides statistical analysis capabilities for the intelligence system.

Author: Agent A - Integration Fix
"""

import numpy as np
from typing import Dict, Any, Union


def run_statistical_analysis(data: Union[np.ndarray, Dict], analysis_type: str) -> Dict[str, Any]:
    """
    Run statistical analysis on data.
    
    Args:
        data: Input data for analysis
        analysis_type: Type of statistical analysis to perform
    
    Returns:
        Dictionary containing analysis results
    """
    # Convert data to numpy array if needed
    if isinstance(data, dict):
        data_array = np.array(list(data.values()) if data else [])
    else:
        data_array = np.asarray(data)
    
    # Handle empty data
    if data_array.size == 0:
        return {
            'status': 'no_data',
            'analysis_type': analysis_type,
            'results': {}
        }
    
    # Perform analysis based on type
    results = {
        'analysis_type': analysis_type,
        'data_points': len(data_array),
    }
    
    if analysis_type == 'descriptive':
        results.update({
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array))
        })
    elif analysis_type == 'distribution':
        results.update({
            'skewness': float(calculate_skewness(data_array)),
            'kurtosis': float(calculate_kurtosis(data_array)),
            'percentiles': {
                '25': float(np.percentile(data_array, 25)),
                '50': float(np.percentile(data_array, 50)),
                '75': float(np.percentile(data_array, 75))
            }
        })
    elif analysis_type == 'trend':
        results.update({
            'trend_direction': detect_trend(data_array),
            'volatility': float(np.std(np.diff(data_array))) if len(data_array) > 1 else 0,
            'change_points': detect_change_points(data_array)
        })
    else:
        # Default analysis
        results.update({
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array))
        })
    
    return results


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data distribution."""
    n = len(data)
    if n < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data distribution."""
    n = len(data)
    if n < 4:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3
    return kurt


def detect_trend(data: np.ndarray) -> str:
    """Detect trend direction in data."""
    if len(data) < 2:
        return 'insufficient_data'
    
    # Simple linear regression
    x = np.arange(len(data))
    coefficients = np.polyfit(x, data, 1)
    slope = coefficients[0]
    
    if abs(slope) < 0.01:
        return 'stable'
    elif slope > 0:
        return 'increasing'
    else:
        return 'decreasing'


def detect_change_points(data: np.ndarray) -> list:
    """Detect significant change points in data."""
    if len(data) < 3:
        return []
    
    change_points = []
    threshold = np.std(data) * 2  # 2 standard deviations
    
    for i in range(1, len(data) - 1):
        if abs(data[i] - data[i-1]) > threshold:
            change_points.append(i)
    
    return change_points