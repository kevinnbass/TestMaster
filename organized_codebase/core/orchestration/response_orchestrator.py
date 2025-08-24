#!/usr/bin/env python3
"""
Response Orchestration Module
Agent D Hour 5 - Modularized Automated Response System

Handles automated response actions and incident orchestration
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from .security_events import SecurityEvent, ThreatLevel, ResponseAction

@dataclass
class ResponseResult:
    """Result of automated response execution"""
    success: bool
    action_type: str
    execution_time: datetime.datetime
    duration_ms: int
    details: Dict[str, Any]
    error_message: Optional[str] = None

class ResponseStatus(Enum):
    """Status of response execution"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

@dataclass
class ResponseTask:
    """Response task with execution context"""
    event: SecurityEvent
    action: ResponseAction
    priority: int
    created_at: datetime.datetime
    status: ResponseStatus = ResponseStatus.PENDING
    result: Optional[ResponseResult] = None
    retry_count: int = 0
    max_retries: int = 3

class ResponseOrchestrator:
    """Central orchestrator for automated security responses"""
    
    def __init__(self, max_workers: int = 5, response_timeout: int = 300):
        """Initialize response orchestrator"""
        self.max_workers = max_workers
        self.response_timeout = response_timeout
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.response_queue = asyncio.Queue(maxsize=1000)
        self.active_tasks = {}
        self.response_handlers = {}
        
        self.quarantine_dir = Path(__file__).parent.parent / "QUARANTINE"
        self.alerts_dir = Path(__file__).parent.parent / "ALERTS"
        
        # Create directories
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'responses_executed': 0,
            'responses_successful': 0,
            'responses_failed': 0,
            'average_response_time': 0.0,
            'quarantine_actions': 0,
            'alert_actions': 0,
            'block_actions': 0,
            'restart_actions': 0,
            'emergency_shutdowns': 0
        }
        
        # Initialize default response handlers
        self._register_default_handlers()
        
        # Start background processing
        self.processing_active = True
        self.processing_task = None
    
    def _register_default_handlers(self):
        """Register default response action handlers"""
        self.response_handlers = {
            ResponseAction.LOG_ONLY: self._handle_log_only,
            ResponseAction.ALERT: self._handle_alert,
            ResponseAction.QUARANTINE: self._handle_quarantine,
            ResponseAction.BLOCK: self._handle_block,
            ResponseAction.RESTART_SERVICE: self._handle_restart_service,
            ResponseAction.EMERGENCY_SHUTDOWN: self._handle_emergency_shutdown
        }
    
    async def start_processing(self):
        """Start background response processing"""
        if self.processing_task is None or self.processing_task.done():
            self.processing_active = True
            self.processing_task = asyncio.create_task(self._process_response_queue())
            self.logger.info("Response orchestrator started")
    
    async def stop_processing(self):
        """Stop background response processing"""
        self.processing_active = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        self.logger.info("Response orchestrator stopped")
    
    async def queue_response(self, event: SecurityEvent) -> str:
        """Queue security event for automated response"""
        # Calculate priority based on threat level and urgency
        priority = self._calculate_priority(event)
        
        task = ResponseTask(
            event=event,
            action=event.response_action,
            priority=priority,
            created_at=datetime.datetime.now()
        )
        
        task_id = f"{event.event_id}_{event.response_action.value}"
        self.active_tasks[task_id] = task
        
        await self.response_queue.put(task)
        
        self.logger.info(f"Queued response task {task_id} with priority {priority}")
        return task_id
    
    def _calculate_priority(self, event: SecurityEvent) -> int:
        """Calculate response priority based on event characteristics"""
        base_priority = event.threat_level.severity_score * 10
        
        # Increase priority for critical/emergency events
        if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            base_priority += 50
        
        # Increase priority for escalated response actions
        if event.response_action.escalation_level >= 4:
            base_priority += 20
        
        # Increase priority for unresolved events
        if not event.resolved:
            base_priority += 10
        
        return base_priority
    
    async def _process_response_queue(self):
        """Background task to process response queue"""
        pending_tasks = []
        
        while self.processing_active:
            try:
                # Get next task with timeout
                try:
                    task = await asyncio.wait_for(self.response_queue.get(), timeout=1.0)
                    pending_tasks.append(task)
                except asyncio.TimeoutError:
                    # Check for completed tasks
                    if pending_tasks:
                        await self._check_completed_tasks(pending_tasks)
                    continue
                
                # Sort tasks by priority
                pending_tasks.sort(key=lambda t: t.priority, reverse=True)
                
                # Execute high priority tasks
                tasks_to_execute = []
                for task in pending_tasks[:self.max_workers]:
                    if task.status == ResponseStatus.PENDING:
                        tasks_to_execute.append(task)
                
                # Submit tasks for execution
                for task in tasks_to_execute:
                    future = self.executor.submit(self._execute_response_task, task)
                    task.status = ResponseStatus.IN_PROGRESS
                    task.future = future
                
                # Check for completed tasks
                await self._check_completed_tasks(pending_tasks)
                
                # Remove completed tasks
                pending_tasks = [t for t in pending_tasks 
                               if t.status in [ResponseStatus.PENDING, ResponseStatus.IN_PROGRESS]]
                
            except Exception as e:
                self.logger.error(f"Error in response queue processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_completed_tasks(self, pending_tasks: List[ResponseTask]):
        """Check for completed response tasks"""
        for task in pending_tasks:
            if hasattr(task, 'future') and task.future.done():
                try:
                    result = task.future.result()
                    task.result = result
                    task.status = ResponseStatus.COMPLETED if result.success else ResponseStatus.FAILED
                    
                    # Update statistics
                    self._update_stats(task, result)
                    
                    # Handle failed tasks for retry
                    if not result.success and task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = ResponseStatus.PENDING
                        await self.response_queue.put(task)
                        self.logger.warning(f"Retrying task {task.event.event_id}, "
                                          f"attempt {task.retry_count}")
                    
                except Exception as e:
                    task.status = ResponseStatus.FAILED
                    self.logger.error(f"Task execution error: {e}")
    
    def _execute_response_task(self, task: ResponseTask) -> ResponseResult:
        """Execute individual response task"""
        start_time = datetime.datetime.now()
        
        try:
            handler = self.response_handlers.get(task.action)
            if not handler:
                raise ValueError(f"No handler for action: {task.action}")
            
            # Execute handler with timeout
            result = handler(task.event)
            
            duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            return ResponseResult(
                success=True,
                action_type=task.action.value,
                execution_time=start_time,
                duration_ms=int(duration),
                details=result
            )
            
        except Exception as e:
            duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            return ResponseResult(
                success=False,
                action_type=task.action.value,
                execution_time=start_time,
                duration_ms=int(duration),
                details={},
                error_message=str(e)
            )
    
    def _handle_log_only(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle LOG_ONLY response action"""
        log_message = f"Security Event: {event.event_type} - {event.description}"
        self.logger.warning(log_message)
        
        return {
            'action': 'logged',
            'message': log_message,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def _handle_alert(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle ALERT response action"""
        alert_file = self.alerts_dir / f"alert_{event.event_id}.json"
        
        alert_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp,
            'threat_level': event.threat_level.value,
            'event_type': event.event_type,
            'source_file': event.source_file,
            'description': event.description,
            'evidence': event.evidence,
            'alert_generated': datetime.datetime.now().isoformat()
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        self.stats['alert_actions'] += 1
        
        return {
            'action': 'alert_generated',
            'alert_file': str(alert_file),
            'alert_data': alert_data
        }
    
    def _handle_quarantine(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle QUARANTINE response action"""
        source_file = Path(event.source_file)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Create quarantine subdirectory
        quarantine_subdir = self.quarantine_dir / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        quarantine_subdir.mkdir(exist_ok=True)
        
        # Create quarantine filename
        quarantine_file = quarantine_subdir / f"{event.event_id}_{source_file.name}"
        
        # Move file to quarantine
        shutil.move(str(source_file), str(quarantine_file))
        
        # Create quarantine metadata
        metadata_file = quarantine_subdir / f"{event.event_id}_metadata.json"
        metadata = {
            'event_id': event.event_id,
            'original_path': str(source_file),
            'quarantine_path': str(quarantine_file),
            'quarantine_time': datetime.datetime.now().isoformat(),
            'threat_level': event.threat_level.value,
            'event_type': event.event_type,
            'description': event.description,
            'evidence': event.evidence
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.stats['quarantine_actions'] += 1
        
        return {
            'action': 'quarantined',
            'original_path': str(source_file),
            'quarantine_path': str(quarantine_file),
            'metadata_path': str(metadata_file)
        }
    
    def _handle_block(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle BLOCK response action"""
        # Extract IP or process information from evidence
        evidence = event.evidence
        blocked_items = []
        
        # Block IP addresses if found
        if 'ip_addresses' in evidence:
            for ip in evidence['ip_addresses']:
                try:
                    # Use Windows netsh command to block IP
                    cmd = f'netsh advfirewall firewall add rule name="Security_Block_{ip}" dir=in action=block remoteip={ip}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        blocked_items.append(f"IP:{ip}")
                    else:
                        self.logger.warning(f"Failed to block IP {ip}: {result.stderr}")
                        
                except Exception as e:
                    self.logger.error(f"Error blocking IP {ip}: {e}")
        
        # Block processes if found
        if 'process_names' in evidence:
            for process in evidence['process_names']:
                try:
                    # Kill process by name
                    cmd = f'taskkill /F /IM "{process}"'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        blocked_items.append(f"Process:{process}")
                    else:
                        self.logger.warning(f"Failed to kill process {process}: {result.stderr}")
                        
                except Exception as e:
                    self.logger.error(f"Error killing process {process}: {e}")
        
        self.stats['block_actions'] += 1
        
        return {
            'action': 'blocked',
            'blocked_items': blocked_items,
            'block_time': datetime.datetime.now().isoformat()
        }
    
    def _handle_restart_service(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle RESTART_SERVICE response action"""
        # Extract service information from evidence
        evidence = event.evidence
        service_name = evidence.get('service_name', 'unknown')
        
        restarted_services = []
        
        try:
            # Windows service restart
            cmd = f'net stop "{service_name}" && net start "{service_name}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                restarted_services.append(service_name)
            else:
                self.logger.warning(f"Failed to restart service {service_name}: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error restarting service {service_name}: {e}")
        
        self.stats['restart_actions'] += 1
        
        return {
            'action': 'service_restarted',
            'services': restarted_services,
            'restart_time': datetime.datetime.now().isoformat()
        }
    
    def _handle_emergency_shutdown(self, event: SecurityEvent) -> Dict[str, Any]:
        """Handle EMERGENCY_SHUTDOWN response action"""
        # Log emergency shutdown
        emergency_log = f"EMERGENCY SHUTDOWN TRIGGERED: {event.description}"
        self.logger.critical(emergency_log)
        
        # Create emergency shutdown record
        emergency_file = self.alerts_dir / f"emergency_{event.event_id}.json"
        emergency_data = {
            'event_id': event.event_id,
            'shutdown_time': datetime.datetime.now().isoformat(),
            'trigger_event': event.to_dict(),
            'emergency_level': 'CRITICAL',
            'action_taken': 'EMERGENCY_SHUTDOWN'
        }
        
        with open(emergency_file, 'w') as f:
            json.dump(emergency_data, f, indent=2)
        
        self.stats['emergency_shutdowns'] += 1
        
        # Note: Actual system shutdown would be implemented based on requirements
        # For safety, we log instead of actually shutting down
        
        return {
            'action': 'emergency_shutdown_logged',
            'emergency_file': str(emergency_file),
            'shutdown_time': datetime.datetime.now().isoformat()
        }
    
    def register_custom_handler(self, action: ResponseAction, 
                              handler: Callable[[SecurityEvent], Dict[str, Any]]):
        """Register custom response handler"""
        self.response_handlers[action] = handler
        self.logger.info(f"Registered custom handler for {action.value}")
    
    def get_task_status(self, task_id: str) -> Optional[ResponseTask]:
        """Get status of response task"""
        return self.active_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel pending response task"""
        task = self.active_tasks.get(task_id)
        if task and task.status == ResponseStatus.PENDING:
            task.status = ResponseStatus.CANCELLED
            return True
        return False
    
    def _update_stats(self, task: ResponseTask, result: ResponseResult):
        """Update response statistics"""
        self.stats['responses_executed'] += 1
        
        if result.success:
            self.stats['responses_successful'] += 1
        else:
            self.stats['responses_failed'] += 1
        
        # Update average response time
        total_time = (self.stats['average_response_time'] * 
                     (self.stats['responses_executed'] - 1) + 
                     result.duration_ms)
        self.stats['average_response_time'] = total_time / self.stats['responses_executed']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get response orchestrator statistics"""
        stats = self.stats.copy()
        stats['active_tasks'] = len(self.active_tasks)
        stats['queue_size'] = self.response_queue.qsize()
        stats['processing_active'] = self.processing_active
        
        # Calculate success rate
        if stats['responses_executed'] > 0:
            stats['success_rate'] = (stats['responses_successful'] / 
                                   stats['responses_executed']) * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def cleanup_completed_tasks(self, hours: int = 24):
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        tasks_to_remove = []
        for task_id, task in self.active_tasks.items():
            if (task.status in [ResponseStatus.COMPLETED, ResponseStatus.FAILED, ResponseStatus.CANCELLED] 
                and task.created_at < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
        
        self.logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")