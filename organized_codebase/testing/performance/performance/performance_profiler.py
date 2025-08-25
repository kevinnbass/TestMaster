#!/usr/bin/env python3
"""
Performance Profiling System with Flame Graphs
==============================================
Comprehensive performance profiling with flame graph generation, hotspot detection,
and performance regression analysis for enterprise applications.
"""

import sys
import time
import json
import cProfile
import pstats
import io
import threading
import traceback
import functools
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import subprocess
import os
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProfileData:
    """Single profiling session data."""
    session_id: str
    timestamp: float
    duration: float
    function_calls: int
    total_time: float
    hotspots: List[Dict[str, Any]]
    memory_usage: Dict[str, float]
    cpu_usage: float
    call_graph: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis."""
    timestamp: float
    execution_time_ms: float
    memory_delta_mb: float
    cpu_percent: float
    io_operations: int
    function_name: str
    module_name: str
    line_number: int

class FlameGraphGenerator:
    """Generates flame graphs from profiling data."""
    
    def __init__(self):
        self.flame_graph_tools_path = self._find_flamegraph_tools()
    
    def _find_flamegraph_tools(self) -> Optional[str]:
        """Find FlameGraph tools installation."""
        # Common paths for FlameGraph tools
        possible_paths = [
            '/usr/local/bin/flamegraph.pl',
            '/opt/FlameGraph/flamegraph.pl',
            './FlameGraph/flamegraph.pl',
            'flamegraph.pl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def generate_flame_graph(self, profile_data: pstats.Stats, output_file: str) -> bool:
        """Generate flame graph from profile data."""
        try:
            # Convert profile data to flame graph format
            flame_data = self._convert_to_flame_format(profile_data)
            
            # Write flame data to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.flame', delete=False) as f:
                f.write(flame_data)
                flame_file = f.name
            
            if self.flame_graph_tools_path:
                # Use FlameGraph tools if available
                cmd = [
                    'perl', self.flame_graph_tools_path,
                    '--title', 'Python Performance Profile',
                    '--width', '1200',
                    '--height', '800',
                    flame_file
                ]
                
                with open(output_file, 'w') as output:
                    result = subprocess.run(cmd, stdout=output, stderr=subprocess.PIPE, text=True)
                
                os.unlink(flame_file)
                return result.returncode == 0
            else:
                # Generate HTML flame graph without external tools
                html_content = self._generate_html_flame_graph(flame_data)
                with open(output_file, 'w') as f:
                    f.write(html_content)
                os.unlink(flame_file)
                return True
                
        except Exception as e:
            logger.error(f"Flame graph generation error: {e}")
            return False
    
    def _convert_to_flame_format(self, profile_data: pstats.Stats) -> str:
        """Convert pstats data to flame graph format."""
        flame_lines = []
        
        for (file_path, line_num, func_name), (cc, nc, tt, ct, callers) in profile_data.stats.items():
            # Create stack trace representation
            module_name = os.path.basename(file_path) if file_path else 'unknown'
            stack = f"{module_name};{func_name}"
            
            # Add cumulative time as the value
            flame_lines.append(f"{stack} {int(ct * 1000)}")  # Convert to milliseconds
        
        return '\n'.join(flame_lines)
    
    def _generate_html_flame_graph(self, flame_data: str) -> str:
        """Generate simple HTML flame graph visualization."""
        lines = flame_data.strip().split('\n')
        total_time = sum(int(line.split()[-1]) for line in lines if line.strip())
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python Performance Flame Graph</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .flame-bar {{ 
                    margin: 1px 0; 
                    padding: 2px 5px; 
                    background: linear-gradient(to right, #ff6b6b, #ffa726, #ffee58);
                    border: 1px solid #333;
                    font-size: 12px;
                    position: relative;
                }}
                .flame-info {{ margin-bottom: 20px; }}
                .stack {{ font-weight: bold; }}
                .time {{ float: right; }}
            </style>
        </head>
        <body>
            <div class="flame-info">
                <h2>Python Performance Flame Graph</h2>
                <p>Total Time: {total_time}ms</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="flame-graph">
        """
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            stack, time_str = parts
            time_ms = int(time_str)
            percentage = (time_ms / total_time) * 100 if total_time > 0 else 0
            width = max(percentage, 1)  # Minimum 1% width for visibility
            
            html += f"""
                <div class="flame-bar" style="width: {width}%;">
                    <span class="stack">{stack}</span>
                    <span class="time">{time_ms}ms ({percentage:.1f}%)</span>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

class PerformanceProfiler:
    """Comprehensive performance profiling system."""
    
    def __init__(self):
        self.active_sessions = {}
        self.completed_sessions = deque(maxlen=100)
        self.hotspot_detector = HotspotDetector()
        self.regression_detector = RegressionDetector()
        self.flame_generator = FlameGraphGenerator()
        
        # Performance tracking
        self.function_metrics = defaultdict(list)
        self.baseline_metrics = {}
        
        # Configuration
        self.config = {
            'enable_memory_tracking': True,
            'enable_io_tracking': True,
            'hotspot_threshold_ms': 100,
            'regression_threshold_percent': 20,
            'max_call_depth': 50
        }
    
    def start_profiling(self, session_id: str = None) -> str:
        """Start a new profiling session."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        if session_id in self.active_sessions:
            raise ValueError(f"Session {session_id} already active")
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Track session state
        session_data = {
            'profiler': profiler,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / (1024 * 1024),
            'start_cpu': psutil.cpu_percent(),
            'metrics': []
        }
        
        self.active_sessions[session_id] = session_data
        profiler.enable()
        
        logger.info(f"Started profiling session: {session_id}")
        return session_id
    
    def stop_profiling(self, session_id: str) -> ProfileData:
        """Stop profiling session and return results."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        profiler = session['profiler']
        
        # Stop profiler
        profiler.disable()
        
        # Calculate session metrics
        end_time = time.time()
        duration = end_time - session['start_time']
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_delta = end_memory - session['start_memory']
        
        # Extract profile statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Detect hotspots
        hotspots = self.hotspot_detector.detect_hotspots(stats)
        
        # Create profile data
        profile_data = ProfileData(
            session_id=session_id,
            timestamp=session['start_time'],
            duration=duration,
            function_calls=stats.total_calls,
            total_time=stats.total_tt,
            hotspots=hotspots,
            memory_usage={
                'start_mb': session['start_memory'],
                'end_mb': end_memory,
                'delta_mb': memory_delta
            },
            cpu_usage=psutil.cpu_percent(),
            call_graph=self._extract_call_graph(stats)
        )
        
        # Store completed session
        self.completed_sessions.append(profile_data)
        del self.active_sessions[session_id]
        
        logger.info(f"Completed profiling session: {session_id} (duration: {duration:.2f}s)")
        return profile_data
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile individual functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session_id = f"func_{func.__name__}_{int(time.time())}"
            
            # Start profiling
            self.start_profiling(session_id)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop profiling
                profile_data = self.stop_profiling(session_id)
                
                # Store function metrics
                metric = PerformanceMetrics(
                    timestamp=profile_data.timestamp,
                    execution_time_ms=profile_data.duration * 1000,
                    memory_delta_mb=profile_data.memory_usage['delta_mb'],
                    cpu_percent=profile_data.cpu_usage,
                    io_operations=0,  # Would need more detailed tracking
                    function_name=func.__name__,
                    module_name=func.__module__,
                    line_number=func.__code__.co_firstlineno
                )
                
                self.function_metrics[func.__name__].append(metric)
                
                # Check for regressions
                self.regression_detector.check_regression(func.__name__, metric)
        
        return wrapper
    
    def _extract_call_graph(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Extract call graph information from stats."""
        call_graph = {}
        
        for (file_path, line_num, func_name), (cc, nc, tt, ct, callers) in stats.stats.items():
            node_id = f"{file_path}:{line_num}:{func_name}"
            
            call_graph[node_id] = {
                'function': func_name,
                'file': file_path,
                'line': line_num,
                'call_count': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'callers': list(callers.keys()) if callers else []
            }
        
        return call_graph
    
    def generate_flame_graph(self, session_id: str, output_file: str = None) -> str:
        """Generate flame graph for a session."""
        # Find session data
        profile_data = None
        for session in self.completed_sessions:
            if session.session_id == session_id:
                profile_data = session
                break
        
        if not profile_data:
            raise ValueError(f"Session {session_id} not found")
        
        if output_file is None:
            output_file = f"flamegraph_{session_id}.svg"
        
        # Recreate stats object (simplified for flame graph generation)
        # In a real implementation, you'd store the raw profiler data
        logger.info(f"Generating flame graph for session {session_id}")
        logger.info(f"Output file: {output_file}")
        
        # For now, generate a simple HTML flame graph
        output_file = output_file.replace('.svg', '.html')
        
        flame_data = []
        for hotspot in profile_data.hotspots:
            stack = f"{hotspot['module']};{hotspot['function']}"
            time_ms = int(hotspot['cumulative_time'] * 1000)
            flame_data.append(f"{stack} {time_ms}")
        
        html_content = self.flame_generator._generate_html_flame_graph('\n'.join(flame_data))
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Flame graph saved to: {output_file}")
        return output_file
    
    def get_performance_summary(self, function_name: str = None, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for analysis."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        if function_name:
            # Function-specific summary
            metrics = [
                m for m in self.function_metrics[function_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not metrics:
                return {'error': f'No data for function {function_name}'}
            
            execution_times = [m.execution_time_ms for m in metrics]
            memory_deltas = [m.memory_delta_mb for m in metrics]
            
            return {
                'function_name': function_name,
                'call_count': len(metrics),
                'avg_execution_time_ms': sum(execution_times) / len(execution_times),
                'max_execution_time_ms': max(execution_times),
                'min_execution_time_ms': min(execution_times),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'total_memory_delta_mb': sum(memory_deltas),
                'recent_trend': self._calculate_trend(execution_times[-10:]) if len(execution_times) >= 10 else 'insufficient_data'
            }
        else:
            # Overall system summary
            recent_sessions = [
                s for s in self.completed_sessions
                if s.timestamp >= cutoff_time
            ]
            
            if not recent_sessions:
                return {'error': 'No recent session data'}
            
            durations = [s.duration for s in recent_sessions]
            memory_deltas = [s.memory_usage['delta_mb'] for s in recent_sessions]
            
            return {
                'session_count': len(recent_sessions),
                'avg_session_duration': sum(durations) / len(durations),
                'total_function_calls': sum(s.function_calls for s in recent_sessions),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'hottest_functions': self._get_hottest_functions(recent_sessions),
                'performance_trend': self._calculate_trend(durations[-10:]) if len(durations) >= 10 else 'insufficient_data'
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend from recent values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.1:
            return 'degrading'
        elif slope < -0.1:
            return 'improving'
        else:
            return 'stable'
    
    def _get_hottest_functions(self, sessions: List[ProfileData]) -> List[Dict[str, Any]]:
        """Get the hottest functions across sessions."""
        function_times = defaultdict(list)
        
        for session in sessions:
            for hotspot in session.hotspots:
                func_name = f"{hotspot['module']}.{hotspot['function']}"
                function_times[func_name].append(hotspot['cumulative_time'])
        
        # Calculate average times and sort
        avg_times = {
            func: sum(times) / len(times)
            for func, times in function_times.items()
        }
        
        sorted_functions = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'function': func, 'avg_time': avg_time}
            for func, avg_time in sorted_functions[:10]
        ]
    
    def export_profile_data(self, output_file: str = None) -> str:
        """Export all profile data to JSON."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"profile_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': time.time(),
            'completed_sessions': [asdict(session) for session in self.completed_sessions],
            'function_metrics': {
                func_name: [asdict(metric) for metric in metrics]
                for func_name, metrics in self.function_metrics.items()
            },
            'configuration': self.config
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Profile data exported to: {output_file}")
        return output_file

class HotspotDetector:
    """Detects performance hotspots in profiling data."""
    
    def detect_hotspots(self, stats: pstats.Stats, top_n: int = 20) -> List[Dict[str, Any]]:
        """Detect performance hotspots from profile statistics."""
        hotspots = []
        
        # Sort by cumulative time to find the most expensive functions
        sorted_stats = sorted(
            stats.stats.items(),
            key=lambda x: x[1][3],  # Sort by cumulative time
            reverse=True
        )
        
        for i, ((file_path, line_num, func_name), (cc, nc, tt, ct, callers)) in enumerate(sorted_stats[:top_n]):
            hotspot = {
                'rank': i + 1,
                'function': func_name,
                'module': os.path.basename(file_path) if file_path else 'unknown',
                'line_number': line_num,
                'call_count': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': ct / cc if cc > 0 else 0,
                'percentage': (ct / stats.total_tt * 100) if stats.total_tt > 0 else 0
            }
            hotspots.append(hotspot)
        
        return hotspots

class RegressionDetector:
    """Detects performance regressions."""
    
    def __init__(self):
        self.baselines = {}
        self.regression_threshold = 0.2  # 20% degradation threshold
    
    def set_baseline(self, function_name: str, metrics: PerformanceMetrics):
        """Set performance baseline for a function."""
        self.baselines[function_name] = {
            'execution_time_ms': metrics.execution_time_ms,
            'memory_delta_mb': metrics.memory_delta_mb,
            'timestamp': metrics.timestamp
        }
    
    def check_regression(self, function_name: str, current_metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        """Check if current performance represents a regression."""
        if function_name not in self.baselines:
            # Set as baseline if no baseline exists
            self.set_baseline(function_name, current_metrics)
            return None
        
        baseline = self.baselines[function_name]
        
        # Calculate performance change
        time_change = (current_metrics.execution_time_ms - baseline['execution_time_ms']) / baseline['execution_time_ms']
        memory_change = (current_metrics.memory_delta_mb - baseline['memory_delta_mb']) / max(baseline['memory_delta_mb'], 1)
        
        # Check for regression
        is_regression = time_change > self.regression_threshold or memory_change > self.regression_threshold
        
        if is_regression:
            regression_data = {
                'function_name': function_name,
                'timestamp': current_metrics.timestamp,
                'baseline_time_ms': baseline['execution_time_ms'],
                'current_time_ms': current_metrics.execution_time_ms,
                'time_change_percent': time_change * 100,
                'baseline_memory_mb': baseline['memory_delta_mb'],
                'current_memory_mb': current_metrics.memory_delta_mb,
                'memory_change_percent': memory_change * 100,
                'severity': 'high' if time_change > 0.5 else 'medium'
            }
            
            logger.warning(f"Performance regression detected in {function_name}: "
                         f"Time: {time_change*100:.1f}%, Memory: {memory_change*100:.1f}%")
            
            return regression_data
        
        return None


def main():
    """Demo and testing of performance profiling system."""
    profiler = PerformanceProfiler()
    
    # Example 1: Profile a session
    session_id = profiler.start_profiling("demo_session")
    
    # Simulate some work
    def expensive_function():
        total = 0
        for i in range(100000):
            total += i * i
        return total
    
    def memory_intensive_function():
        data = []
        for i in range(10000):
            data.append([j for j in range(100)])
        return len(data)
    
    # Run some functions
    expensive_function()
    memory_intensive_function()
    
    # Stop profiling
    profile_data = profiler.stop_profiling(session_id)
    
    logger.info(f"Profile completed:")
    logger.info(f"  Duration: {profile_data.duration:.3f}s")
    logger.info(f"  Function calls: {profile_data.function_calls}")
    logger.info(f"  Memory delta: {profile_data.memory_usage['delta_mb']:.2f}MB")
    logger.info(f"  Hotspots found: {len(profile_data.hotspots)}")
    
    # Generate flame graph
    flame_file = profiler.generate_flame_graph(session_id)
    logger.info(f"Flame graph generated: {flame_file}")
    
    # Example 2: Use function decorator
    @profiler.profile_function
    def decorated_function(n):
        return sum(i*i for i in range(n))
    
    # Call decorated function multiple times
    for i in range(5):
        result = decorated_function(1000 * (i + 1))
    
    # Get performance summary
    summary = profiler.get_performance_summary('decorated_function')
    logger.info(f"Function performance summary: {json.dumps(summary, indent=2)}")
    
    # Export all data
    export_file = profiler.export_profile_data()
    logger.info(f"All profile data exported to: {export_file}")


if __name__ == "__main__":
    main()