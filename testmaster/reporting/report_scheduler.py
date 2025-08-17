"""
Report Scheduler for TestMaster

Automated report generation and distribution system.
"""

import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.feature_flags import FeatureFlags

class DeliveryMethod(Enum):
    """Report delivery methods."""
    FILE_SYSTEM = "file_system"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

@dataclass
class ScheduleConfig:
    """Schedule configuration for automated reports."""
    name: str
    interval_hours: int
    report_type: str
    delivery_method: DeliveryMethod
    recipients: List[str]

@dataclass
class ReportDelivery:
    """Report delivery configuration."""
    delivery_id: str
    method: DeliveryMethod
    target: str
    metadata: Dict[str, Any]

class ReportScheduler:
    """Automated report scheduler."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'performance_reporting')
        self.lock = threading.RLock()
        self.scheduled_reports: Dict[str, ScheduleConfig] = {}
        self.is_running = False
        
        if not self.enabled:
            return
    
    def start_scheduler(self):
        """Start report scheduler."""
        if self.enabled:
            self.is_running = True
    
    def schedule_report(self, config: ScheduleConfig) -> str:
        """Schedule automated report generation."""
        if not self.enabled:
            return ""
        
        schedule_id = f"schedule_{len(self.scheduled_reports)}"
        
        with self.lock:
            self.scheduled_reports[schedule_id] = config
        
        return schedule_id
    
    def shutdown(self):
        """Shutdown report scheduler."""
        self.is_running = False

def get_report_scheduler() -> ReportScheduler:
    """Get report scheduler instance."""
    return ReportScheduler()

def schedule_automated_reports(name: str, interval_hours: int = 24) -> str:
    """Schedule automated reports."""
    scheduler = get_report_scheduler()
    config = ScheduleConfig(
        name=name,
        interval_hours=interval_hours,
        report_type="system_overview",
        delivery_method=DeliveryMethod.FILE_SYSTEM,
        recipients=[]
    )
    return scheduler.schedule_report(config)