"""
Load Testing and Performance Analysis Framework for TestMaster
Generates synthetic load and analyzes system performance
"""

import time
import threading
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


class LoadPattern(Enum):
    """Load testing patterns"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STEP = "step"
    WAVE = "wave"
    RANDOM = "random"


class MetricType(Enum):
    """Performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    QUEUE_SIZE = "queue_size"


@dataclass
class LoadConfig:
    """Load test configuration"""
    max_users: int
    duration: int  # seconds
    pattern: LoadPattern
    ramp_time: int = 0
    think_time: float = 1.0
    timeout: float = 30.0
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Single request result"""
    timestamp: float
    response_time: float
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    payload_size: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Performance analysis results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    throughput: float  # requests per second
    error_rate: float
    concurrency_achieved: float


@dataclass
class LoadTestReport:
    """Complete load test report"""
    config: LoadConfig
    metrics: PerformanceMetrics
    timeline: List[Tuple[float, Dict[str, float]]]
    bottlenecks: List[str]
    recommendations: List[str]
    success: bool


class VirtualUser:
    """Simulates a virtual user"""
    
    def __init__(self, user_id: int, config: LoadConfig):
        self.user_id = user_id
        self.config = config
        self.results = []
        self.active = False
        
    def run(self, target_function: Callable, payload_generator: Callable = None):
        """Execute virtual user scenario"""
        self.active = True
        start_time = time.time()
        
        while self.active and (time.time() - start_time) < self.config.duration:
            try:
                # Generate payload
                payload = payload_generator() if payload_generator else None
                
                # Execute request
                request_start = time.time()
                
                if payload:
                    response = target_function(payload)
                else:
                    response = target_function()
                    
                response_time = time.time() - request_start
                
                # Record result
                result = TestResult(
                    timestamp=time.time(),
                    response_time=response_time,
                    success=True,
                    status_code=getattr(response, 'status_code', 200)
                )
                self.results.append(result)
                
            except Exception as e:
                result = TestResult(
                    timestamp=time.time(),
                    response_time=time.time() - request_start if 'request_start' in locals() else 0,
                    success=False,
                    error=str(e)
                )
                self.results.append(result)
                
            # Think time
            if self.config.think_time > 0:
                time.sleep(random.uniform(0, self.config.think_time * 2))
                
        self.active = False
    
    def stop(self):
        """Stop virtual user"""
        self.active = False


class LoadGenerator:
    """Main load testing framework"""
    
    def __init__(self):
        self.users = []
        self.metrics_timeline = []
        self.executor = None
        
    def execute_load_test(self, target_function: Callable, config: LoadConfig,
                         payload_generator: Optional[Callable] = None) -> LoadTestReport:
        """Execute load test with specified configuration"""
        
        # Initialize
        self.users = []
        self.metrics_timeline = []
        all_results = []
        
        # Create virtual users based on pattern
        user_schedule = self._generate_user_schedule(config)
        
        try:
            with ThreadPoolExecutor(max_workers=config.max_users) as executor:
                self.executor = executor
                
                # Submit user tasks according to schedule
                futures = []
                for start_time, user_count in user_schedule:
                    # Wait for scheduled time
                    time.sleep(max(0, start_time - time.time()))
                    
                    # Launch users
                    for i in range(user_count):
                        user = VirtualUser(len(self.users), config)
                        self.users.append(user)
                        
                        future = executor.submit(user.run, target_function, payload_generator)
                        futures.append((user, future))
                
                # Monitor and collect results
                start_time = time.time()
                
                # Wait for completion or timeout
                for user, future in futures:
                    try:
                        future.result(timeout=config.duration + 10)
                        all_results.extend(user.results)
                    except Exception as e:
                        # User failed, but continue with others
                        continue
                        
        except KeyboardInterrupt:
            # Graceful shutdown
            for user in self.users:
                user.stop()
                
        # Analyze results
        metrics = self._analyze_performance(all_results, config.duration)
        timeline = self._generate_timeline(all_results)
        bottlenecks = self._identify_bottlenecks(metrics, timeline)
        recommendations = self._generate_recommendations(metrics, config)
        success = self._evaluate_success(metrics, config)
        
        return LoadTestReport(
            config=config,
            metrics=metrics,
            timeline=timeline,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            success=success
        )
    
    def _generate_user_schedule(self, config: LoadConfig) -> List[Tuple[float, int]]:
        """Generate user launch schedule based on pattern"""
        schedule = []
        start_time = time.time()
        
        if config.pattern == LoadPattern.CONSTANT:
            # Launch all users at once
            schedule.append((start_time, config.max_users))
            
        elif config.pattern == LoadPattern.RAMP_UP:
            # Gradually increase users
            interval = config.ramp_time / config.max_users
            for i in range(config.max_users):
                schedule.append((start_time + i * interval, 1))
                
        elif config.pattern == LoadPattern.SPIKE:
            # Quick ramp up to max, then maintain
            ramp_time = min(10, config.ramp_time)
            interval = ramp_time / config.max_users
            for i in range(config.max_users):
                schedule.append((start_time + i * interval, 1))
                
        elif config.pattern == LoadPattern.STEP:
            # Step increases
            step_size = config.max_users // 5
            step_duration = config.duration // 5
            for step in range(5):
                step_time = start_time + step * step_duration
                schedule.append((step_time, step_size))
                
        elif config.pattern == LoadPattern.WAVE:
            # Sine wave pattern
            for i in range(config.max_users):
                wave_time = config.duration * (i / config.max_users)
                delay = (1 + math.sin(wave_time * 2 * math.pi / config.duration)) / 2
                schedule.append((start_time + delay * config.ramp_time, 1))
                
        elif config.pattern == LoadPattern.RANDOM:
            # Random arrival times
            for i in range(config.max_users):
                random_delay = random.uniform(0, config.ramp_time)
                schedule.append((start_time + random_delay, 1))
                
        return sorted(schedule)
    
    def _analyze_performance(self, results: List[TestResult], 
                           duration: float) -> PerformanceMetrics:
        """Analyze performance from results"""
        if not results:
            return PerformanceMetrics(
                total_requests=0, successful_requests=0, failed_requests=0,
                avg_response_time=0, p50_response_time=0, p95_response_time=0,
                p99_response_time=0, max_response_time=0, min_response_time=0,
                throughput=0, error_rate=0, concurrency_achieved=0
            )
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        response_times = [r.response_time for r in successful]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50 = statistics.median(response_times)
            p95 = self._percentile(response_times, 95)
            p99 = self._percentile(response_times, 99)
            max_time = max(response_times)
            min_time = min(response_times)
        else:
            avg_response_time = p50 = p95 = p99 = max_time = min_time = 0
        
        throughput = len(results) / duration if duration > 0 else 0
        error_rate = len(failed) / len(results) if results else 0
        
        # Calculate peak concurrency
        concurrency = self._calculate_peak_concurrency(results)
        
        return PerformanceMetrics(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_response_time=avg_response_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            max_response_time=max_time,
            min_response_time=min_time,
            throughput=throughput,
            error_rate=error_rate,
            concurrency_achieved=concurrency
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_peak_concurrency(self, results: List[TestResult]) -> float:
        """Calculate peak concurrent requests"""
        if not results:
            return 0
            
        # Create timeline of request starts and ends
        events = []
        for result in results:
            start_time = result.timestamp - result.response_time
            end_time = result.timestamp
            events.append((start_time, 1))  # Request start
            events.append((end_time, -1))   # Request end
            
        # Sort by time
        events.sort()
        
        # Calculate peak concurrency
        current_concurrency = 0
        peak_concurrency = 0
        
        for timestamp, delta in events:
            current_concurrency += delta
            peak_concurrency = max(peak_concurrency, current_concurrency)
            
        return peak_concurrency
    
    def _generate_timeline(self, results: List[TestResult]) -> List[Tuple[float, Dict[str, float]]]:
        """Generate performance timeline"""
        if not results:
            return []
            
        # Group results by time windows (1 second intervals)
        start_time = min(r.timestamp for r in results)
        end_time = max(r.timestamp for r in results)
        
        timeline = []
        current_time = start_time
        
        while current_time <= end_time:
            window_end = current_time + 1.0
            window_results = [r for r in results 
                            if current_time <= r.timestamp < window_end]
            
            if window_results:
                successful = [r for r in window_results if r.success]
                response_times = [r.response_time for r in successful]
                
                window_metrics = {
                    'timestamp': current_time,
                    'requests': len(window_results),
                    'successful': len(successful),
                    'avg_response_time': statistics.mean(response_times) if response_times else 0,
                    'error_rate': (len(window_results) - len(successful)) / len(window_results)
                }
                timeline.append((current_time, window_metrics))
                
            current_time += 1.0
            
        return timeline
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics,
                             timeline: List[Tuple[float, Dict[str, float]]]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # High error rate
        if metrics.error_rate > 0.05:  # > 5%
            bottlenecks.append(f"High error rate: {metrics.error_rate*100:.1f}%")
            
        # High response times
        if metrics.p95_response_time > 5.0:  # > 5 seconds
            bottlenecks.append(f"Slow P95 response time: {metrics.p95_response_time:.2f}s")
            
        # Low throughput relative to concurrency
        if metrics.concurrency_achieved > 0:
            efficiency = metrics.throughput / metrics.concurrency_achieved
            if efficiency < 0.5:  # Less than 0.5 RPS per concurrent user
                bottlenecks.append(f"Low throughput efficiency: {efficiency:.2f} RPS/user")
        
        # Degradation over time
        if len(timeline) > 10:
            early_avg = statistics.mean(m['avg_response_time'] for t, m in timeline[:5])
            late_avg = statistics.mean(m['avg_response_time'] for t, m in timeline[-5:])
            
            if late_avg > early_avg * 1.5:
                bottlenecks.append("Response time degradation over time")
                
        return bottlenecks
    
    def _generate_recommendations(self, metrics: PerformanceMetrics,
                                 config: LoadConfig) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if metrics.error_rate > 0.01:
            recommendations.append("Investigate and fix errors causing failures")
            
        if metrics.avg_response_time > 2.0:
            recommendations.append("Optimize application response time")
            
        if metrics.throughput < config.max_users * 0.5:
            recommendations.append("Scale infrastructure or optimize bottlenecks")
            
        if metrics.p99_response_time > metrics.avg_response_time * 5:
            recommendations.append("Address response time outliers")
            
        if metrics.concurrency_achieved < config.max_users * 0.8:
            recommendations.append("Check resource limits and connection pools")
            
        return recommendations[:5]
    
    def _evaluate_success(self, metrics: PerformanceMetrics, 
                         config: LoadConfig) -> bool:
        """Evaluate if load test passed success criteria"""
        criteria = config.success_criteria
        
        if not criteria:
            # Default criteria
            return (metrics.error_rate < 0.05 and 
                   metrics.avg_response_time < 5.0 and
                   metrics.throughput > 0)
        
        # Check custom criteria
        for metric, threshold in criteria.items():
            actual_value = getattr(metrics, metric, None)
            if actual_value is None:
                continue
                
            if isinstance(threshold, dict):
                # Range check
                if 'min' in threshold and actual_value < threshold['min']:
                    return False
                if 'max' in threshold and actual_value > threshold['max']:
                    return False
            else:
                # Simple threshold
                if actual_value > threshold:
                    return False
                    
        return True