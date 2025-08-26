"""
Watchdog Process Manager (Part 3/3) - TestMaster Advanced ML
Advanced process monitoring and management with ML-driven insights
Extracted from analytics_watchdog.py (674 lines) → 3 coordinated ML modules
"""

import asyncio
import logging
import os
import psutil
import signal
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge

from .watchdog_ml_monitor import ComponentState, WatchdogAction
from .watchdog_recovery_system import AdvancedRecoverySystem


@dataclass
class ProcessInfo:
    """Advanced process information with ML insights"""
    
    process_name: str
    process_id: Optional[int]
    command: List[str]
    working_directory: str
    environment: Dict[str, str]
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    
    # State tracking
    status: str = "unknown"  # running, stopped, failed, restarting
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Resource monitoring
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    open_files: int = 0
    threads: int = 0
    
    # ML Enhancement Fields
    performance_score: float = 1.0
    anomaly_score: float = 0.0
    failure_probability: float = 0.0
    resource_trend: str = "stable"
    optimization_suggestions: List[str] = field(default_factory=list)
    ml_insights: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessEvent:
    """Process lifecycle event with ML context"""
    
    event_id: str
    process_name: str
    event_type: str  # start, stop, restart, crash, anomaly
    timestamp: datetime
    process_id: Optional[int]
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    resource_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # ML Enhancement
    predicted_event: bool = False
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)


class AdvancedProcessManager:
    """
    ML-enhanced process monitoring and management system
    Part 3/3 of the complete watchdog system
    """
    
    def __init__(self,
                 recovery_system: AdvancedRecoverySystem,
                 monitoring_interval: int = 15,
                 enable_ml_analysis: bool = True,
                 process_history_limit: int = 1000):
        """Initialize advanced process manager"""
        
        self.recovery_system = recovery_system
        self.monitoring_interval = monitoring_interval
        self.enable_ml_analysis = enable_ml_analysis
        self.process_history_limit = process_history_limit
        
        # ML Models for Process Intelligence
        self.failure_predictor: Optional[RandomForestRegressor] = None
        self.resource_anomaly_detector: Optional[IsolationForest] = None
        self.performance_clusterer: Optional[DBSCAN] = None
        self.resource_optimizer: Optional[Ridge] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.resource_scaler = RobustScaler()
        self.process_feature_history: deque = deque(maxlen=2000)
        
        # Process Management
        self.monitored_processes: Dict[str, ProcessInfo] = {}
        self.process_events: deque = deque(maxlen=process_history_limit)
        self.process_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Process Control
        self.active_processes: Dict[int, psutil.Process] = {}
        self.process_restart_counts: defaultdict = defaultdict(int)
        self.process_configurations: Dict[str, Dict[str, Any]] = {}
        
        # ML Insights and Predictions
        self.ml_process_insights: Dict[str, Dict[str, Any]] = {}
        self.resource_predictions: Dict[str, Dict[str, float]] = {}
        self.anomaly_patterns: List[Dict[str, Any]] = []
        
        # System Resource Baseline
        self.system_baseline: Dict[str, float] = {}
        self.resource_thresholds: Dict[str, Dict[str, float]] = {
            'cpu': {'warning': 80.0, 'critical': 95.0},
            'memory': {'warning': 75.0, 'critical': 90.0},
            'io': {'warning': 100.0, 'critical': 200.0}  # MB/s
        }
        
        # Configuration
        self.max_restart_attempts = 5
        self.restart_cooldown = 30
        self.cascade_detection_enabled = True
        self.auto_optimization_enabled = True
        
        # Statistics
        self.process_stats = {
            'processes_monitored': 0,
            'process_restarts': 0,
            'process_failures': 0,
            'ml_predictions_made': 0,
            'anomalies_detected': 0,
            'optimizations_applied': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.process_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start monitoring
        if enable_ml_analysis:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_analysis_loop())
        
        self._establish_system_baseline()
        asyncio.create_task(self._process_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for process intelligence"""
        
        try:
            # Process failure prediction
            self.failure_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                min_samples_split=5
            )
            
            # Resource anomaly detection
            self.resource_anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Process performance clustering
            self.performance_clusterer = DBSCAN(
                eps=0.3,
                min_samples=5,
                metric='euclidean'
            )
            
            # Resource usage optimization
            self.resource_optimizer = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            self.logger.info("Process ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Process ML model initialization failed: {e}")
            self.enable_ml_analysis = False
    
    def register_process(self,
                        process_name: str,
                        command: List[str],
                        working_directory: str = None,
                        environment: Dict[str, str] = None,
                        auto_start: bool = True,
                        restart_on_failure: bool = True) -> bool:
        """Register process for advanced monitoring"""
        
        try:
            with self.process_lock:
                # Create process info
                process_info = ProcessInfo(
                    process_name=process_name,
                    process_id=None,
                    command=command,
                    working_directory=working_directory or os.getcwd(),
                    environment=environment or dict(os.environ)
                )
                
                self.monitored_processes[process_name] = process_info
                self.process_configurations[process_name] = {
                    'auto_start': auto_start,
                    'restart_on_failure': restart_on_failure,
                    'max_restarts': self.max_restart_attempts,
                    'restart_cooldown': self.restart_cooldown
                }
                
                self.process_stats['processes_monitored'] += 1
            
            # Start process if auto_start is enabled
            if auto_start:
                await self.start_process(process_name)
            
            self.logger.info(f"Process registered: {process_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Process registration failed for {process_name}: {e}")
            return False
    
    async def start_process(self, process_name: str) -> bool:
        """Start monitored process with ML tracking"""
        
        try:
            with self.process_lock:
                if process_name not in self.monitored_processes:
                    self.logger.error(f"Process not registered: {process_name}")
                    return False
                
                process_info = self.monitored_processes[process_name]
                
                # Check if already running
                if process_info.process_id and await self._is_process_running(process_info.process_id):
                    self.logger.warning(f"Process already running: {process_name}")
                    return True
                
                # Start the process
                process = subprocess.Popen(
                    process_info.command,
                    cwd=process_info.working_directory,
                    env=process_info.environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )
                
                # Update process info
                process_info.process_id = process.pid
                process_info.status = "running"
                process_info.start_time = datetime.now()
                process_info.restart_count += 1
                process_info.last_restart = datetime.now()
                
                # Add to active processes
                self.active_processes[process.pid] = psutil.Process(process.pid)
                
                # Record process event
                await self._record_process_event(
                    process_name, "start", process.pid
                )
                
                self.logger.info(f"Process started: {process_name} (PID: {process.pid})")
                return True
                
        except Exception as e:
            self.logger.error(f"Process start failed for {process_name}: {e}")
            return False
    
    async def stop_process(self, process_name: str, force: bool = False) -> bool:
        """Stop monitored process gracefully or forcefully"""
        
        try:
            with self.process_lock:
                if process_name not in self.monitored_processes:
                    self.logger.error(f"Process not registered: {process_name}")
                    return False
                
                process_info = self.monitored_processes[process_name]
                
                if not process_info.process_id:
                    self.logger.warning(f"Process not running: {process_name}")
                    return True
                
                # Get process handle
                if process_info.process_id in self.active_processes:
                    process = self.active_processes[process_info.process_id]
                else:
                    try:
                        process = psutil.Process(process_info.process_id)
                    except psutil.NoSuchProcess:
                        process_info.process_id = None
                        process_info.status = "stopped"
                        return True
                
                # Stop the process
                if force:
                    process.kill()
                    await self._record_process_event(process_name, "kill", process_info.process_id)
                else:
                    process.terminate()
                    await self._record_process_event(process_name, "stop", process_info.process_id)
                
                # Wait for process to stop
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    if not force:
                        # Force kill if graceful termination failed
                        process.kill()
                        process.wait(timeout=5)
                
                # Update process info
                process_info.process_id = None
                process_info.status = "stopped"
                
                # Remove from active processes
                if process.pid in self.active_processes:
                    del self.active_processes[process.pid]
                
                self.logger.info(f"Process stopped: {process_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Process stop failed for {process_name}: {e}")
            return False
    
    async def restart_process(self, process_name: str) -> bool:
        """Restart process with ML optimization"""
        
        try:
            # Stop the process first
            stop_success = await self.stop_process(process_name)
            
            if not stop_success:
                self.logger.error(f"Failed to stop process for restart: {process_name}")
                return False
            
            # Brief pause before restart
            await asyncio.sleep(2)
            
            # Apply ML optimizations before restart
            if self.enable_ml_analysis and self.auto_optimization_enabled:
                await self._apply_ml_optimizations(process_name)
            
            # Start the process
            start_success = await self.start_process(process_name)
            
            if start_success:
                self.process_stats['process_restarts'] += 1
                await self._record_process_event(process_name, "restart", None)
            
            return start_success
            
        except Exception as e:
            self.logger.error(f"Process restart failed for {process_name}: {e}")
            return False
    
    async def _process_monitoring_loop(self):
        """Main process monitoring loop with ML analysis"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Monitor all registered processes
                for process_name in list(self.monitored_processes.keys()):
                    if self.shutdown_event.is_set():
                        break
                    
                    await self._monitor_process_health(process_name)
                
                # Check for process failures and handle restarts
                await self._handle_failed_processes()
                
                # Update system baseline
                await self._update_system_baseline()
                
            except Exception as e:
                self.logger.error(f"Process monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_process_health(self, process_name: str):
        """Monitor individual process health with ML insights"""
        
        try:
            with self.process_lock:
                process_info = self.monitored_processes[process_name]
                
                # Check if process is running
                if process_info.process_id:
                    if not await self._is_process_running(process_info.process_id):
                        # Process died unexpectedly
                        process_info.status = "failed"
                        process_info.process_id = None
                        self.process_stats['process_failures'] += 1
                        
                        await self._record_process_event(
                            process_name, "crash", process_info.process_id
                        )
                        
                        return
                    
                    # Collect resource metrics
                    try:
                        process = self.active_processes.get(process_info.process_id)
                        if not process:
                            process = psutil.Process(process_info.process_id)
                            self.active_processes[process_info.process_id] = process
                        
                        # Update resource metrics
                        await self._update_process_metrics(process_info, process)
                        
                        # ML analysis
                        if self.enable_ml_analysis:
                            await self._analyze_process_with_ml(process_info)
                        
                        # Store metrics history
                        self.process_metrics_history[process_name].append({
                            'timestamp': datetime.now(),
                            'cpu': process_info.cpu_usage,
                            'memory': process_info.memory_usage,
                            'memory_percent': process_info.memory_percent,
                            'performance_score': process_info.performance_score
                        })
                        
                    except psutil.NoSuchProcess:
                        process_info.status = "failed"
                        process_info.process_id = None
                        
                        if process_info.process_id in self.active_processes:
                            del self.active_processes[process_info.process_id]
                
        except Exception as e:
            self.logger.error(f"Process health monitoring failed for {process_name}: {e}")
    
    async def _update_process_metrics(self, process_info: ProcessInfo, process: psutil.Process):
        """Update comprehensive process metrics"""
        
        try:
            # CPU and memory usage
            process_info.cpu_usage = process.cpu_percent()
            
            memory_info = process.memory_info()
            process_info.memory_usage = memory_info.rss / 1024 / 1024  # MB
            process_info.memory_percent = process.memory_percent()
            
            # I/O statistics
            try:
                io_counters = process.io_counters()
                process_info.io_read_bytes = io_counters.read_bytes
                process_info.io_write_bytes = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                pass
            
            # Thread and file handle count
            try:
                process_info.threads = process.num_threads()
                process_info.open_files = len(process.open_files())
            except (psutil.AccessDenied, AttributeError):
                pass
            
            # Update uptime
            if process_info.start_time:
                process_info.uptime_seconds = (datetime.now() - process_info.start_time).total_seconds()
            
            # Update status
            process_info.status = process.status()
            
        except Exception as e:
            self.logger.error(f"Process metrics update failed: {e}")
    
    async def _analyze_process_with_ml(self, process_info: ProcessInfo):
        """Perform ML analysis on process metrics"""
        
        try:
            with self.ml_lock:
                # Extract features for ML analysis
                features = await self._extract_process_features(process_info)
                
                if len(self.process_feature_history) >= 30:
                    # Anomaly detection
                    if self.resource_anomaly_detector:
                        anomaly_score = self.resource_anomaly_detector.decision_function([features])[0]
                        process_info.anomaly_score = float(anomaly_score)
                        
                        if anomaly_score < -0.5:  # Anomaly threshold
                            process_info.ml_insights['anomaly_detected'] = True
                            self.process_stats['anomalies_detected'] += 1
                            
                            await self._record_process_event(
                                process_info.process_name, "anomaly", process_info.process_id
                            )
                    
                    # Failure prediction
                    if self.failure_predictor:
                        failure_score = await self._predict_process_failure(features)
                        process_info.failure_probability = failure_score
                        
                        if failure_score > 0.8:  # High failure probability
                            process_info.ml_insights['failure_risk'] = 'high'
                            self.process_stats['ml_predictions_made'] += 1
                    
                    # Performance clustering
                    if self.performance_clusterer:
                        cluster_id = await self._assign_performance_cluster(features)
                        process_info.ml_insights['performance_cluster'] = cluster_id
                    
                    # Generate optimization suggestions
                    process_info.optimization_suggestions = await self._generate_optimization_suggestions(
                        process_info
                    )
                
                # Add features to history
                self.process_feature_history.append(features)
                
        except Exception as e:
            self.logger.error(f"ML process analysis failed: {e}")
    
    def _extract_process_features(self, process_info: ProcessInfo) -> np.ndarray:
        """Extract ML features from process information"""
        
        features = np.array([
            process_info.cpu_usage,
            process_info.memory_usage,
            process_info.memory_percent,
            process_info.io_read_bytes / 1024 / 1024,  # MB
            process_info.io_write_bytes / 1024 / 1024,  # MB
            process_info.threads,
            process_info.open_files,
            process_info.uptime_seconds / 3600,  # hours
            process_info.restart_count,
            process_info.performance_score,
            datetime.now().hour,
            datetime.now().weekday()
        ], dtype=np.float64)
        
        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    async def _ml_analysis_loop(self):
        """ML analysis and model maintenance loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(120)  # Run every 2 minutes
                
                if len(self.process_feature_history) >= 50:
                    # Retrain models periodically
                    await self._retrain_ml_models()
                    
                    # Update resource predictions
                    await self._update_resource_predictions()
                    
                    # Generate system insights
                    await self._generate_system_insights()
                
            except Exception as e:
                self.logger.error(f"ML analysis loop error: {e}")
                await asyncio.sleep(10)
    
    def get_process_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive process management dashboard"""
        
        # Process summary
        process_summary = {}
        for name, info in self.monitored_processes.items():
            process_summary[name] = {
                'status': info.status,
                'pid': info.process_id,
                'cpu_usage': info.cpu_usage,
                'memory_usage': info.memory_usage,
                'uptime_hours': info.uptime_seconds / 3600 if info.uptime_seconds else 0,
                'restart_count': info.restart_count,
                'performance_score': info.performance_score,
                'failure_probability': info.failure_probability,
                'anomaly_score': info.anomaly_score,
                'optimization_suggestions': info.optimization_suggestions[:3]
            }
        
        # System resource overview
        system_resources = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'active_processes': len([p for p in self.monitored_processes.values() if p.status == 'running'])
        }
        
        return {
            'process_overview': {
                'total_processes': len(self.monitored_processes),
                'running_processes': len([p for p in self.monitored_processes.values() if p.status == 'running']),
                'failed_processes': len([p for p in self.monitored_processes.values() if p.status == 'failed']),
            },
            'processes': process_summary,
            'system_resources': system_resources,
            'statistics': self.process_stats.copy(),
            'ml_insights': {
                'feature_history_size': len(self.process_feature_history),
                'predictions_made': self.process_stats['ml_predictions_made'],
                'anomalies_detected': self.process_stats['anomalies_detected'],
                'optimizations_applied': self.process_stats['optimizations_applied']
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of process manager"""
        
        self.logger.info("Shutting down process manager...")
        
        # Stop all monitored processes
        for process_name in list(self.monitored_processes.keys()):
            await self.stop_process(process_name)
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Process manager shutdown complete")