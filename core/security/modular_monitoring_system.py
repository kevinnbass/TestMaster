#!/usr/bin/env python3
"""
Modular Security Monitoring System
Agent D Hour 5 - STEELCLAD Modularized Security Monitoring

Orchestrates all security monitoring components in a modular architecture
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import modular components
from .monitoring_modules.security_events import (
    SecurityEvent, ThreatLevel, ResponseAction, SecurityMetrics,
    SecurityEventProcessor
)
from .monitoring_modules.threat_detector import (
    ThreatDetectionEngine, VulnerabilityAssessment
)
from .monitoring_modules.database_manager import DatabaseManager
from .monitoring_modules.response_orchestrator import ResponseOrchestrator
from .monitoring_modules.performance_monitor import PerformanceMonitor

@dataclass
class SystemStatus:
    """Overall system status aggregation"""
    monitoring_active: bool
    components_status: Dict[str, str]
    total_events: int
    unresolved_events: int
    system_health_score: float
    last_update: str

class ModularSecurityMonitoringSystem:
    """
    STEELCLAD-compliant modular security monitoring system that orchestrates
    all security monitoring components while maintaining clear separation of concerns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize modular security monitoring system"""
        self.config = config or self._get_default_config()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize modular components
        self.database_manager = DatabaseManager(
            db_path=self.config.get('database_path')
        )
        
        self.threat_detector = ThreatDetectionEngine()
        
        self.response_orchestrator = ResponseOrchestrator(
            max_workers=self.config.get('response_workers', 5),
            response_timeout=self.config.get('response_timeout', 300)
        )
        
        self.performance_monitor = PerformanceMonitor(
            monitoring_interval=self.config.get('performance_interval', 30)
        )
        
        # System state
        self.monitoring_active = False
        self.component_tasks = {}
        self.event_processing_queue = asyncio.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            'system_start_time': datetime.datetime.now(),
            'total_events_processed': 0,
            'total_threats_detected': 0,
            'total_responses_executed': 0,
            'system_health_checks': 0
        }
        
        # Register performance alert callback
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        self.logger.info("Modular Security Monitoring System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the monitoring system"""
        return {
            'database_path': str(Path(__file__).parent / "security_monitoring.db"),
            'response_workers': 5,
            'response_timeout': 300,
            'performance_interval': 30,
            'scan_intervals': {
                'security_scan': 300,      # 5 minutes
                'vulnerability_scan': 3600, # 1 hour
                'integrity_check': 1800    # 30 minutes
            },
            'file_extensions_to_monitor': ['.py', '.js', '.sql', '.json', '.xml', '.yaml', '.yml'],
            'max_concurrent_scans': 3,
            'event_retention_days': 30,
            'log_level': 'INFO'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitoring system"""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            # Create logs directory
            logs_dir = Path(__file__).parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Setup file handler
            log_file = logs_dir / f"modular_monitoring_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            
            # Setup console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        return logger
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        if self.monitoring_active:
            self.logger.warning("Monitoring system already active")
            return
        
        try:
            self.monitoring_active = True
            
            # Start performance monitoring
            await self.performance_monitor.start_monitoring()
            self.component_tasks['performance'] = self.performance_monitor.monitoring_task
            
            # Start response orchestrator
            await self.response_orchestrator.start_processing()
            self.component_tasks['response'] = self.response_orchestrator.processing_task
            
            # Start event processing
            self.component_tasks['events'] = asyncio.create_task(self._event_processing_loop())
            
            # Start security scanning
            self.component_tasks['security_scan'] = asyncio.create_task(self._security_scan_loop())
            
            # Start system health monitoring
            self.component_tasks['health'] = asyncio.create_task(self._health_monitoring_loop())
            
            self.logger.info("Modular Security Monitoring System started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring system: {e}")
            await self.stop_monitoring()
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        try:
            # Stop performance monitor
            await self.performance_monitor.stop_monitoring()
            
            # Stop response orchestrator
            await self.response_orchestrator.stop_processing()
            
            # Cancel all component tasks
            for task_name, task in self.component_tasks.items():
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.component_tasks.clear()
            
            self.logger.info("Modular Security Monitoring System stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.monitoring_active:
            try:
                # Get event from queue with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_processing_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the event
                await self._process_security_event(event)
                
                # Mark task done
                self.event_processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _security_scan_loop(self):
        """Security scanning loop"""
        scan_interval = self.config['scan_intervals']['security_scan']
        
        while self.monitoring_active:
            try:
                # Start security operation tracking
                operation_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                perf_key = self.performance_monitor.security_tracker.start_operation(
                    'scan', operation_id
                )
                
                # Perform security scan
                await self._perform_security_scan()
                
                # End performance tracking
                self.performance_monitor.security_tracker.end_operation(perf_key)
                
                self.logger.info(f"Security scan completed: {operation_id}")
                
                # Wait for next scan
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in security scan loop: {e}")
                await asyncio.sleep(scan_interval)
    
    async def _perform_security_scan(self):
        """Perform comprehensive security scan"""
        try:
            # Scan current directory and subdirectories
            scan_path = str(Path(__file__).parent.parent.parent)
            
            # Get threat detection results
            detection_results = self.threat_detector.scan_directory(
                directory_path=scan_path,
                file_extensions=self.config['file_extensions_to_monitor'],
                max_files=500  # Limit for performance
            )
            
            # Process detection results
            for result in detection_results:
                # Create security event
                event = SecurityEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type=result.pattern_name,
                    threat_level=result.threat_level,
                    source_file=result.evidence.get('file_path', 'unknown'),
                    description=f"Threat detected: {result.pattern_name}",
                    evidence=result.evidence,
                    response_action=result.suggested_action
                )
                
                # Queue event for processing
                await self.event_processing_queue.put(event)
            
            self.stats['total_threats_detected'] += len(detection_results)
            
        except Exception as e:
            self.logger.error(f"Error performing security scan: {e}")
    
    async def _process_security_event(self, event: SecurityEvent):
        """Process individual security event"""
        try:
            # Validate event
            is_valid, validation_msg = SecurityEventProcessor.validate_event(event)
            if not is_valid:
                self.logger.warning(f"Invalid security event: {validation_msg}")
                return
            
            # Store event in database
            self.database_manager.store_security_event(event)
            
            # Queue response if needed
            if event.response_action != ResponseAction.LOG_ONLY:
                await self.response_orchestrator.queue_response(event)
            
            # Update statistics
            self.stats['total_events_processed'] += 1
            
            self.logger.info(f"Processed security event: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing security event: {e}")
    
    async def _health_monitoring_loop(self):
        """System health monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform system health check
                health_status = await self._perform_health_check()
                
                # Store health metrics
                health_metrics = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'health_score': health_status.system_health_score,
                    'components_healthy': sum(
                        1 for status in health_status.components_status.values() 
                        if status == 'healthy'
                    ),
                    'total_components': len(health_status.components_status)
                }
                
                self.database_manager.store_system_metrics(health_metrics)
                
                self.stats['system_health_checks'] += 1
                
                # Wait before next health check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> SystemStatus:
        """Perform comprehensive system health check"""
        try:
            components_status = {}
            
            # Check database health
            try:
                db_stats = self.database_manager.get_database_stats()
                components_status['database'] = 'healthy' if db_stats else 'unhealthy'
            except:
                components_status['database'] = 'unhealthy'
            
            # Check threat detector health
            try:
                detector_stats = self.threat_detector.get_statistics()
                components_status['threat_detector'] = 'healthy' if detector_stats else 'unhealthy'
            except:
                components_status['threat_detector'] = 'unhealthy'
            
            # Check response orchestrator health
            try:
                response_stats = self.response_orchestrator.get_statistics()
                components_status['response_orchestrator'] = 'healthy' if response_stats else 'unhealthy'
            except:
                components_status['response_orchestrator'] = 'unhealthy'
            
            # Check performance monitor health
            try:
                perf_data = self.performance_monitor.get_current_performance()
                components_status['performance_monitor'] = 'healthy' if perf_data else 'unhealthy'
            except:
                components_status['performance_monitor'] = 'unhealthy'
            
            # Calculate health score
            healthy_components = sum(1 for status in components_status.values() if status == 'healthy')
            total_components = len(components_status)
            health_score = (healthy_components / total_components) * 100 if total_components > 0 else 0
            
            # Get event counts
            recent_events = self.database_manager.get_security_events(limit=1000)
            unresolved_events = sum(1 for event in recent_events if not event.resolved)
            
            return SystemStatus(
                monitoring_active=self.monitoring_active,
                components_status=components_status,
                total_events=len(recent_events),
                unresolved_events=unresolved_events,
                system_health_score=health_score,
                last_update=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            return SystemStatus(
                monitoring_active=self.monitoring_active,
                components_status={'error': 'unhealthy'},
                total_events=0,
                unresolved_events=0,
                system_health_score=0.0,
                last_update=datetime.datetime.now().isoformat()
            )
    
    async def _handle_performance_alert(self, alert):
        """Handle performance alerts from the performance monitor"""
        try:
            # Create security event for performance alert
            event = SecurityEvent(
                timestamp=alert.timestamp,
                event_type="performance_alert",
                threat_level=ThreatLevel.MEDIUM if alert.severity == 'warning' else ThreatLevel.HIGH,
                source_file="performance_monitor",
                description=f"Performance alert: {alert.description}",
                evidence={
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'severity': alert.severity,
                    'suggested_actions': alert.suggested_actions
                },
                response_action=ResponseAction.ALERT
            )
            
            # Queue for processing
            await self.event_processing_queue.put(event)
            
        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")
    
    async def manual_scan(self, target_path: str = None) -> Dict[str, Any]:
        """Perform manual security scan on specified path"""
        try:
            if target_path is None:
                target_path = str(Path(__file__).parent.parent.parent)
            
            # Start performance tracking
            operation_id = f"manual_{datetime.datetime.now().strftime('%H%M%S')}"
            perf_key = self.performance_monitor.security_tracker.start_operation(
                'scan', operation_id
            )
            
            # Perform scan
            detection_results = self.threat_detector.scan_directory(
                directory_path=target_path,
                file_extensions=self.config['file_extensions_to_monitor']
            )
            
            # End performance tracking
            performance_impact = self.performance_monitor.security_tracker.end_operation(perf_key)
            
            # Process results
            events_created = 0
            for result in detection_results:
                event = SecurityEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    event_type=result.pattern_name,
                    threat_level=result.threat_level,
                    source_file=result.evidence.get('file_path', 'unknown'),
                    description=f"Manual scan threat: {result.pattern_name}",
                    evidence=result.evidence,
                    response_action=result.suggested_action
                )
                
                await self.event_processing_queue.put(event)
                events_created += 1
            
            return {
                'scan_completed': True,
                'target_path': target_path,
                'threats_detected': len(detection_results),
                'events_created': events_created,
                'performance_impact': performance_impact,
                'scan_timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in manual scan: {e}")
            return {
                'scan_completed': False,
                'error': str(e),
                'scan_timestamp': datetime.datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get component statistics
            db_stats = self.database_manager.get_database_stats()
            detector_stats = self.threat_detector.get_statistics()
            response_stats = self.response_orchestrator.get_statistics()
            performance_data = self.performance_monitor.get_current_performance()
            
            # Get recent metrics
            security_metrics = self.database_manager.get_security_metrics(hours=24)
            
            return {
                'system_info': {
                    'monitoring_active': self.monitoring_active,
                    'start_time': self.stats['system_start_time'].isoformat(),
                    'uptime_seconds': (datetime.datetime.now() - self.stats['system_start_time']).total_seconds()
                },
                'statistics': self.stats.copy(),
                'component_status': {
                    'database': db_stats,
                    'threat_detector': detector_stats,
                    'response_orchestrator': response_stats,
                    'performance_monitor': performance_data
                },
                'security_metrics': security_metrics.__dict__ if security_metrics else {},
                'generated_at': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.datetime.now().isoformat()
            }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current system configuration"""
        return self.config.copy()
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update system configuration (requires restart to take effect)"""
        try:
            self.config.update(new_config)
            self.logger.info("Configuration updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old data from all components"""
        try:
            # Cleanup database
            db_cleanup = self.database_manager.cleanup_old_data(days_to_keep)
            
            # Cleanup threat detector cache
            self.threat_detector.clear_cache()
            
            # Cleanup response orchestrator tasks
            self.response_orchestrator.cleanup_completed_tasks(hours=days_to_keep * 24)
            
            # Cleanup performance monitor data (files older than retention period)
            performance_dir = self.performance_monitor.storage_path
            cleaned_files = 0
            if performance_dir.exists():
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
                for file_path in performance_dir.glob("metrics_*.json"):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        cleaned_files += 1
            
            cleanup_result = {
                'database_cleanup': db_cleanup,
                'cache_cleared': True,
                'tasks_cleaned': True,
                'files_cleaned': cleaned_files,
                'cleanup_date': datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"System cleanup completed: {cleanup_result}")
            return cleanup_result
            
        except Exception as e:
            self.logger.error(f"Error during system cleanup: {e}")
            return {'error': str(e)}

# Factory function for easy instantiation
def create_monitoring_system(config: Dict[str, Any] = None) -> ModularSecurityMonitoringSystem:
    """Factory function to create a configured monitoring system"""
    return ModularSecurityMonitoringSystem(config)

# Main execution for standalone operation
async def main():
    """Main function for standalone operation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modular Security Monitoring System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--scan-path', type=str, help='Path to scan')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Create monitoring system
    monitoring_system = create_monitoring_system(config)
    
    try:
        if args.daemon:
            # Run as daemon
            await monitoring_system.start_monitoring()
            print("Monitoring system started. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        elif args.scan_path:
            # Perform one-time scan
            print(f"Performing security scan on: {args.scan_path}")
            result = await monitoring_system.manual_scan(args.scan_path)
            print(json.dumps(result, indent=2))
            
        else:
            # Show system status
            status = monitoring_system.get_system_status()
            print(json.dumps(status, indent=2))
            
    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
        await monitoring_system.stop_monitoring()
        print("Monitoring system stopped.")

if __name__ == "__main__":
    asyncio.run(main())