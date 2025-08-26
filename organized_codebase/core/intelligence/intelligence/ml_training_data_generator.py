#!/usr/bin/env python3
"""
ML Training Data Generator - Agent A Hour 9
Generates synthetic training data for ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
from pathlib import Path

def generate_synthetic_training_data(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for ML models
    
    Creates realistic system metrics with patterns and relationships
    """
    np.random.seed(42)  # For reproducibility
    
    data = []
    base_time = datetime.now() - timedelta(hours=num_samples/6)  # 6 samples per hour
    
    # Generate patterns
    for i in range(num_samples):
        timestamp = base_time + timedelta(minutes=i*10)
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base patterns
        daily_pattern = np.sin(2 * np.pi * hour_of_day / 24) * 20 + 50
        weekly_pattern = 10 if day_of_week < 5 else -10  # Weekday vs weekend
        
        # CPU usage with daily pattern and some noise
        cpu_usage = daily_pattern + weekly_pattern + np.random.normal(0, 5)
        cpu_usage = np.clip(cpu_usage, 10, 95)
        
        # Memory usage correlated with CPU
        memory_usage = cpu_usage * 0.8 + np.random.normal(0, 3)
        memory_usage = np.clip(memory_usage, 20, 90)
        
        # Response time inversely related to available resources
        response_time = 500 - (100 - cpu_usage) * 3 + np.random.normal(0, 20)
        response_time = max(50, response_time)
        
        # Error rate increases with high load
        error_rate = 0
        if cpu_usage > 80:
            error_rate = (cpu_usage - 80) / 10 + np.random.uniform(0, 1)
        error_rate = min(5, error_rate)
        
        # Service count varies by time of day
        service_count = 10 + hour_of_day // 2 + np.random.randint(-2, 3)
        
        # Dependency health decreases with errors
        dependency_health = 100 - error_rate * 10 - np.random.uniform(0, 5)
        dependency_health = max(70, dependency_health)
        
        # Import success rate
        import_success_rate = 100 - error_rate * 2 - np.random.uniform(0, 2)
        import_success_rate = max(90, import_success_rate)
        
        # Overall health (target variable for prediction)
        overall_health = (
            (100 - cpu_usage) * 0.3 +
            (100 - memory_usage) * 0.2 +
            (500 - response_time) / 5 * 0.2 +
            (100 - error_rate * 20) * 0.15 +
            dependency_health * 0.15
        )
        overall_health = np.clip(overall_health, 20, 100)
        
        # Future health (for supervised learning)
        future_health = overall_health + np.random.normal(0, 5)
        if cpu_usage > 85:  # Degradation pattern
            future_health -= 10
        future_health = np.clip(future_health, 15, 100)
        
        # Additional features for other models
        disk_usage = 30 + day_of_week * 2 + np.random.normal(0, 5)
        disk_usage = np.clip(disk_usage, 10, 80)
        
        request_rate = 100 + hour_of_day * 5 + np.random.normal(0, 10)
        request_rate = max(10, request_rate)
        
        active_connections = request_rate / 10 + np.random.normal(0, 2)
        active_connections = max(1, int(active_connections))
        
        # Performance features
        response_time_trend = (response_time - 250) / 100
        throughput_change = np.random.normal(0, 0.1)
        error_rate_trend = error_rate / 5
        cpu_trend = (cpu_usage - 50) / 50
        memory_pressure = memory_usage / 100
        queue_depth = max(0, int(request_rate / 20 + np.random.normal(0, 2)))
        cache_hit_rate = 90 - error_rate * 5 + np.random.normal(0, 3)
        cache_hit_rate = np.clip(cache_hit_rate, 60, 99)
        
        # Anomaly features (variance calculations)
        cpu_variance = np.random.exponential(5) if np.random.random() < 0.1 else np.random.normal(2, 0.5)
        memory_variance = np.random.exponential(3) if np.random.random() < 0.1 else np.random.normal(1.5, 0.3)
        response_time_spike = 1 if response_time > 400 else 0
        error_rate_change = np.random.uniform(-0.5, 0.5)
        service_failures = np.random.poisson(0.1)
        dependency_changes = np.random.poisson(0.2)
        
        # Create record
        record = {
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'avg_response_time': response_time,
            'error_rate': error_rate,
            'active_services': service_count,
            'dependency_health': dependency_health,
            'import_success_rate': import_success_rate,
            'overall_health': overall_health,
            'future_health': future_health,
            'disk_usage': disk_usage,
            'request_rate': request_rate,
            'active_connections': active_connections,
            'response_time_trend': response_time_trend,
            'throughput_change': throughput_change,
            'error_rate_trend': error_rate_trend,
            'cpu_trend': cpu_trend,
            'memory_pressure': memory_pressure,
            'queue_depth': queue_depth,
            'cache_hit_rate': cache_hit_rate,
            'cpu_variance': cpu_variance,
            'memory_variance': memory_variance,
            'response_time_spike': response_time_spike,
            'error_rate_change': error_rate_change,
            'service_failures': service_failures,
            'dependency_changes': dependency_changes,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Add some anomalies (10% of data)
    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.1), replace=False)
    for idx in anomaly_indices:
        # Create anomalous patterns
        if np.random.random() < 0.3:
            # CPU spike anomaly
            df.loc[idx, 'cpu_usage'] = 95 + np.random.uniform(-2, 2)
            df.loc[idx, 'cpu_variance'] = 20 + np.random.exponential(5)
        elif np.random.random() < 0.6:
            # Memory leak anomaly
            df.loc[idx, 'memory_usage'] = 88 + np.random.uniform(-2, 5)
            df.loc[idx, 'memory_variance'] = 15 + np.random.exponential(3)
        else:
            # Response time anomaly
            df.loc[idx, 'avg_response_time'] = 800 + np.random.uniform(0, 200)
            df.loc[idx, 'response_time_spike'] = 1
    
    return df


def save_training_data(df: pd.DataFrame, output_dir: str = "data/predictive_analytics"):
    """Save training data to CSV and JSON formats"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_file = output_path / "training_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved training data to {csv_file}")
    
    # Save as JSON for API usage
    json_file = output_path / "training_data.json"
    df_json = df.copy()
    df_json['timestamp'] = df_json['timestamp'].astype(str)
    
    with open(json_file, 'w') as f:
        json.dump(df_json.to_dict('records'), f, indent=2)
    print(f"Saved training data to {json_file}")
    
    # Save summary statistics
    stats_file = output_path / "training_stats.json"
    stats = {
        'num_samples': len(df),
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        },
        'feature_stats': {}
    }
    
    for col in df.select_dtypes(include=[np.number]).columns:
        stats['feature_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved training statistics to {stats_file}")
    
    return csv_file, json_file, stats_file


def generate_realtime_metrics():
    """Generate a single set of real-time metrics for testing"""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # Generate some historical data
    cpu_history = [cpu_percent + np.random.normal(0, 5) for _ in range(10)]
    memory_history = [memory.percent + np.random.normal(0, 3) for _ in range(10)]
    
    metrics = {
        'cpu_usage': cpu_percent,
        'memory_usage': memory.percent,
        'avg_response_time': 100 + cpu_percent * 2 + np.random.normal(0, 10),
        'error_rate': max(0, (cpu_percent - 70) / 20) if cpu_percent > 70 else 0,
        'active_services': 15 + np.random.randint(-2, 3),
        'dependency_health': 95.0 + np.random.normal(0, 2),
        'import_success_rate': 98.0 + np.random.normal(0, 1),
        'cpu_history': cpu_history,
        'memory_history': memory_history,
        'response_time_spike': 1 if cpu_percent > 80 else 0,
        'error_rate_change': np.random.uniform(-0.1, 0.1),
        'service_failures': 0 if cpu_percent < 85 else np.random.poisson(0.5),
        'dependency_changes': np.random.poisson(0.1),
        'response_time_trend': np.random.uniform(-0.1, 0.1),
        'throughput_change': np.random.uniform(-0.05, 0.05),
        'error_rate_trend': 0.02 if cpu_percent > 60 else 0,
        'cpu_trend': (cpu_percent - 50) / 100,
        'memory_pressure': memory.percent / 100,
        'queue_depth': max(0, int(5 + np.random.normal(0, 2))),
        'cache_hit_rate': 85.0 + np.random.normal(0, 5),
        'disk_usage': psutil.disk_usage('/').percent,
        'request_rate': 100 + np.random.normal(0, 20),
        'active_connections': 10 + np.random.randint(-3, 4),
        'cpu_variance': np.var(cpu_history),
        'memory_variance': np.var(memory_history),
        'overall_health': 100 - cpu_percent * 0.3 - memory.percent * 0.2
    }
    
    return metrics


if __name__ == "__main__":
    print("=== ML Training Data Generator ===\n")
    
    # Generate training data
    print("Generating synthetic training data...")
    df = generate_synthetic_training_data(num_samples=2000)
    
    print(f"\nGenerated {len(df)} training samples")
    print(f"Features: {list(df.columns)}")
    print(f"\nData shape: {df.shape}")
    
    # Show sample statistics
    print("\nSample statistics:")
    print(df[['cpu_usage', 'memory_usage', 'overall_health', 'error_rate']].describe())
    
    # Check for anomalies
    high_cpu = df[df['cpu_usage'] > 90]
    print(f"\nHigh CPU samples (anomalies): {len(high_cpu)} ({len(high_cpu)/len(df)*100:.1f}%)")
    
    # Save data
    print("\nSaving training data...")
    csv_file, json_file, stats_file = save_training_data(df)
    
    print("\n=== Training Data Generation Complete ===")
    print(f"Files created:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    print(f"  - {stats_file}")
    
    # Generate real-time metrics sample
    print("\n=== Real-time Metrics Sample ===")
    metrics = generate_realtime_metrics()
    print(f"CPU Usage: {metrics['cpu_usage']:.1f}%")
    print(f"Memory Usage: {metrics['memory_usage']:.1f}%")
    print(f"Overall Health: {metrics['overall_health']:.1f}%")