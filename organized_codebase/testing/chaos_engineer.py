"""
Chaos Engineering Framework for TestMaster
Introduces controlled failures to test system resilience
"""

import random
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager


class ChaosType(Enum):
    """Types of chaos experiments"""
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE = "resource"
    NETWORK = "network"
    CORRUPTION = "corruption"
    THROTTLE = "throttle"
    TIMEOUT = "timeout"


@dataclass
class ChaosExperiment:
    """Defines a chaos experiment"""
    name: str
    type: ChaosType
    target: str
    intensity: float  # 0.0 to 1.0
    duration: float  # seconds
    config: Dict[str, Any]
    hypothesis: str
    rollback_plan: Optional[Callable] = None


@dataclass
class ChaosResult:
    """Result of chaos experiment"""
    experiment: str
    success: bool
    impact_metrics: Dict[str, float]
    failures: List[str]
    recovery_time: float
    hypothesis_validated: bool
    recommendations: List[str]


class ChaosMonkey:
    """Chaos injection utilities"""
    
    def __init__(self):
        self.active_chaos = {}
        self.original_functions = {}
        
    @contextmanager
    def inject_latency(self, target: Callable, delay: float, probability: float = 1.0):
        """Inject latency into function calls"""
        def wrapped(*args, **kwargs):
            if random.random() < probability:
                time.sleep(delay)
            return target(*args, **kwargs)
        
        yield wrapped
    
    @contextmanager
    def inject_errors(self, target: Callable, error_type: Exception, 
                     probability: float = 0.1, message: str = "Chaos induced error"):
        """Inject errors into function calls"""
        def wrapped(*args, **kwargs):
            if random.random() < probability:
                raise error_type(message)
            return target(*args, **kwargs)
        
        yield wrapped
    
    @contextmanager
    def inject_corruption(self, target: Callable, corruption_func: Callable,
                         probability: float = 0.1):
        """Corrupt function return values"""
        def wrapped(*args, **kwargs):
            result = target(*args, **kwargs)
            if random.random() < probability:
                return corruption_func(result)
            return result
        
        yield wrapped
    
    def corrupt_data(self, data: Any) -> Any:
        """Corrupt data in various ways"""
        if isinstance(data, str):
            # String corruption
            if len(data) > 0:
                pos = random.randint(0, len(data) - 1)
                return data[:pos] + '?' + data[pos+1:]
            return ""
        elif isinstance(data, (int, float)):
            # Numeric corruption
            return data * random.choice([0, -1, 2, 1000000])
        elif isinstance(data, list):
            # List corruption
            if data:
                return data[:-1] if random.random() > 0.5 else data + data
            return []
        elif isinstance(data, dict):
            # Dict corruption
            if data:
                keys = list(data.keys())
                if keys:
                    del data[random.choice(keys)]
            return data
        return None
    
    @contextmanager  
    def throttle_calls(self, target: Callable, max_calls: int, 
                      window: float = 1.0):
        """Throttle function calls"""
        calls = []
        lock = threading.Lock()
        
        def wrapped(*args, **kwargs):
            with lock:
                now = time.time()
                # Remove old calls outside window
                calls[:] = [t for t in calls if now - t < window]
                
                if len(calls) >= max_calls:
                    raise RuntimeError(f"Throttled: max {max_calls} calls per {window}s")
                    
                calls.append(now)
                return target(*args, **kwargs)
        
        yield wrapped


class ChaosEngineer:
    """Main chaos engineering framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monkey = ChaosMonkey()
        self.experiments = []
        self.results = []
        self.steady_state_metrics = {}
        
    def define_steady_state(self, metrics: Dict[str, Callable]) -> None:
        """Define steady state metrics"""
        self.steady_state_metrics = {}
        for name, metric_func in metrics.items():
            try:
                self.steady_state_metrics[name] = metric_func()
            except Exception as e:
                self.steady_state_metrics[name] = None
                
    def run_experiment(self, experiment: ChaosExperiment, 
                      target_system: Any) -> ChaosResult:
        """Run a chaos experiment"""
        
        # Measure baseline
        baseline_metrics = self._measure_metrics(target_system)
        
        # Inject chaos
        start_time = time.time()
        failures = []
        
        try:
            if experiment.type == ChaosType.LATENCY:
                impact = self._inject_latency(experiment, target_system)
            elif experiment.type == ChaosType.ERROR:
                impact = self._inject_errors(experiment, target_system)
            elif experiment.type == ChaosType.RESOURCE:
                impact = self._inject_resource_pressure(experiment, target_system)
            elif experiment.type == ChaosType.CORRUPTION:
                impact = self._inject_corruption(experiment, target_system)
            elif experiment.type == ChaosType.THROTTLE:
                impact = self._inject_throttling(experiment, target_system)
            else:
                impact = {}
                
        except Exception as e:
            failures.append(str(e))
            impact = {}
            
        # Wait for duration
        time.sleep(experiment.duration)
        
        # Measure impact
        chaos_metrics = self._measure_metrics(target_system)
        recovery_start = time.time()
        
        # Rollback if provided
        if experiment.rollback_plan:
            try:
                experiment.rollback_plan()
            except Exception as e:
                failures.append(f"Rollback failed: {e}")
        
        # Measure recovery
        recovery_metrics = self._measure_metrics(target_system)
        recovery_time = time.time() - recovery_start
        
        # Analyze results
        impact_analysis = self._analyze_impact(
            baseline_metrics, chaos_metrics, recovery_metrics
        )
        
        hypothesis_validated = self._validate_hypothesis(
            experiment, impact_analysis
        )
        
        recommendations = self._generate_recommendations(
            experiment, impact_analysis, failures
        )
        
        result = ChaosResult(
            experiment=experiment.name,
            success=len(failures) == 0,
            impact_metrics=impact_analysis,
            failures=failures,
            recovery_time=recovery_time,
            hypothesis_validated=hypothesis_validated,
            recommendations=recommendations
        )
        
        self.results.append(result)
        return result
    
    def _inject_latency(self, exp: ChaosExperiment, system: Any) -> Dict:
        """Inject latency chaos"""
        delay = exp.config.get('delay', 1.0)
        probability = exp.intensity
        
        # Would modify target functions to add latency
        # This is a simulation
        return {'latency_added': delay, 'probability': probability}
    
    def _inject_errors(self, exp: ChaosExperiment, system: Any) -> Dict:
        """Inject error chaos"""
        error_rate = exp.intensity
        error_type = exp.config.get('error_type', 'RuntimeError')
        
        # Would modify target functions to throw errors
        return {'error_rate': error_rate, 'error_type': error_type}
    
    def _inject_resource_pressure(self, exp: ChaosExperiment, system: Any) -> Dict:
        """Simulate resource pressure"""
        resource_type = exp.config.get('resource', 'memory')
        pressure_level = exp.intensity
        
        if resource_type == 'memory':
            # Allocate memory to create pressure
            size = int(1024 * 1024 * 100 * pressure_level)  # Up to 100MB
            _ = bytearray(size)
        elif resource_type == 'cpu':
            # CPU intensive operation
            end_time = time.time() + exp.duration
            while time.time() < end_time:
                _ = sum(i**2 for i in range(1000))
                
        return {'resource': resource_type, 'pressure': pressure_level}
    
    def _inject_corruption(self, exp: ChaosExperiment, system: Any) -> Dict:
        """Inject data corruption"""
        corruption_rate = exp.intensity
        
        # Would modify data handling to corrupt values
        return {'corruption_rate': corruption_rate}
    
    def _inject_throttling(self, exp: ChaosExperiment, system: Any) -> Dict:
        """Inject throttling"""
        max_calls = exp.config.get('max_calls', 10)
        window = exp.config.get('window', 1.0)
        
        # Would wrap functions with throttling
        return {'max_calls': max_calls, 'window': window}
    
    def _measure_metrics(self, system: Any) -> Dict[str, float]:
        """Measure system metrics"""
        metrics = {}
        
        # Simulate metric collection
        metrics['response_time'] = random.uniform(0.1, 1.0)
        metrics['error_rate'] = random.uniform(0, 0.1)
        metrics['throughput'] = random.uniform(100, 1000)
        metrics['availability'] = random.uniform(0.9, 1.0)
        
        return metrics
    
    def _analyze_impact(self, baseline: Dict, chaos: Dict, 
                       recovery: Dict) -> Dict[str, float]:
        """Analyze impact of chaos"""
        impact = {}
        
        for key in baseline:
            if key in chaos:
                # Calculate percentage change
                if baseline[key] != 0:
                    change = ((chaos[key] - baseline[key]) / baseline[key]) * 100
                else:
                    change = 100.0 if chaos[key] > 0 else 0.0
                impact[f"{key}_change"] = change
                
                # Check recovery
                if key in recovery:
                    if baseline[key] != 0:
                        recovery_pct = (recovery[key] / baseline[key]) * 100
                    else:
                        recovery_pct = 100.0
                    impact[f"{key}_recovery"] = recovery_pct
                    
        return impact
    
    def _validate_hypothesis(self, exp: ChaosExperiment, 
                           impact: Dict[str, float]) -> bool:
        """Validate experiment hypothesis"""
        # Simple validation based on acceptable degradation
        max_degradation = exp.config.get('max_degradation', 50)  # 50% max
        
        for key, value in impact.items():
            if '_change' in key and abs(value) > max_degradation:
                return False
                
        return True
    
    def _generate_recommendations(self, exp: ChaosExperiment,
                                 impact: Dict, failures: List[str]) -> List[str]:
        """Generate recommendations from experiment"""
        recommendations = []
        
        # Check for high impact metrics
        for key, value in impact.items():
            if '_change' in key and abs(value) > 30:
                metric = key.replace('_change', '')
                recommendations.append(
                    f"Improve resilience for {metric} - degraded by {value:.1f}%"
                )
                
        # Check recovery metrics
        for key, value in impact.items():
            if '_recovery' in key and value < 90:
                metric = key.replace('_recovery', '')
                recommendations.append(
                    f"Improve recovery for {metric} - only {value:.1f}% recovered"
                )
                
        # Add failure-specific recommendations
        if failures:
            recommendations.append("Implement better error handling")
            
        if not recommendations:
            recommendations.append("System showed good resilience")
            
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate chaos engineering report"""
        if not self.results:
            return {"message": "No experiments run"}
            
        return {
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r.success),
            'failed': sum(1 for r in self.results if not r.success),
            'hypotheses_validated': sum(1 for r in self.results if r.hypothesis_validated),
            'avg_recovery_time': sum(r.recovery_time for r in self.results) / len(self.results),
            'key_findings': self._summarize_findings(),
            'recommendations': self._consolidate_recommendations()
        }
    
    def _summarize_findings(self) -> List[str]:
        """Summarize key findings"""
        findings = []
        
        # Find weakest areas
        all_impacts = {}
        for result in self.results:
            for key, value in result.impact_metrics.items():
                if '_change' in key:
                    if key not in all_impacts:
                        all_impacts[key] = []
                    all_impacts[key].append(abs(value))
                    
        for key, values in all_impacts.items():
            avg_impact = sum(values) / len(values)
            if avg_impact > 20:
                findings.append(f"{key}: average {avg_impact:.1f}% degradation")
                
        return findings
    
    def _consolidate_recommendations(self) -> List[str]:
        """Consolidate all recommendations"""
        all_recs = []
        for result in self.results:
            all_recs.extend(result.recommendations)
            
        # Deduplicate and prioritize
        unique_recs = list(set(all_recs))
        return unique_recs[:10]  # Top 10