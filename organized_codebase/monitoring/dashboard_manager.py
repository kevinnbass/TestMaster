"""
Unified Dashboard Manager

Main orchestration module for the dashboard system that coordinates
widgets, layouts, themes, and real-time updates.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .dashboard_types import (
    DashboardConfig, DashboardLayout, DashboardWidget,
    DashboardTheme, InteractionMode, DashboardAlert,
    DashboardEvent, AlertLevel
)
from .widget_manager import WidgetManager


class UnifiedDashboardManager:
    """
    Main dashboard manager that orchestrates all dashboard functionality
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core managers
        self.widget_manager = WidgetManager()
        
        # Dashboard storage
        self.dashboards: Dict[str, DashboardConfig] = {}
        self.active_dashboards: Dict[str, str] = {}  # user_id -> dashboard_id
        
        # Real-time features
        self.event_listeners: Dict[str, List[Callable]] = {}
        self.alerts: Dict[str, DashboardAlert] = {}
        
        # Themes and layouts
        self.custom_themes: Dict[str, Dict[str, Any]] = {}
        self.layout_templates: Dict[str, DashboardLayout] = {}
        
        # State management
        self.is_running = False
        
        # Initialize default components
        self._initialize_defaults()
        
        self.logger.info("UnifiedDashboardManager initialized")
    
    def create_dashboard(self, name: str, description: str = "",
                        created_by: str = "system") -> DashboardConfig:
        """Create a new dashboard"""
        dashboard = DashboardConfig(
            name=name,
            description=description,
            created_by=created_by
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        
        self.logger.info(f"Created dashboard: {name} ({dashboard.dashboard_id})")
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """Update dashboard configuration"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Update dashboard properties
        for key, value in updates.items():
            if hasattr(dashboard, key):
                setattr(dashboard, key, value)
        
        dashboard.updated_at = datetime.now()
        
        self.logger.debug(f"Updated dashboard: {dashboard_id}")
        return True
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Remove all widgets
        for widget in dashboard.widgets:
            self.widget_manager.unregister_widget(widget.widget_id)
        
        # Remove dashboard
        del self.dashboards[dashboard_id]
        
        # Update active dashboards
        for user_id, active_id in list(self.active_dashboards.items()):
            if active_id == dashboard_id:
                del self.active_dashboards[user_id]
        
        self.logger.info(f"Deleted dashboard: {dashboard_id}")
        return True
    
    def add_widget_to_dashboard(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Register widget with manager
        self.widget_manager.register_widget(widget)
        
        # Add to dashboard
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now()
        
        self.logger.debug(f"Added widget to dashboard {dashboard_id}: {widget.title}")
        return True
    
    def remove_widget_from_dashboard(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove widget from dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Find and remove widget
        for i, widget in enumerate(dashboard.widgets):
            if widget.widget_id == widget_id:
                # Remove from dashboard
                del dashboard.widgets[i]
                
                # Unregister from manager
                self.widget_manager.unregister_widget(widget_id)
                
                dashboard.updated_at = datetime.now()
                self.logger.debug(f"Removed widget {widget_id} from dashboard {dashboard_id}")
                return True
        
        return False
    
    def set_user_dashboard(self, user_id: str, dashboard_id: str) -> bool:
        """Set active dashboard for user"""
        if dashboard_id not in self.dashboards:
            return False
        
        self.active_dashboards[user_id] = dashboard_id
        return True
    
    def get_user_dashboard(self, user_id: str) -> Optional[DashboardConfig]:
        """Get active dashboard for user"""
        dashboard_id = self.active_dashboards.get(user_id)
        if dashboard_id:
            return self.dashboards.get(dashboard_id)
        return None
    
    def create_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO,
                    target_dashboard: str = None, **kwargs) -> DashboardAlert:
        """Create a dashboard alert"""
        alert = DashboardAlert(
            title=title,
            message=message,
            level=level,
            **kwargs
        )
        
        self.alerts[alert.alert_id] = alert
        
        # Broadcast alert event
        event = DashboardEvent(
            event_type="alert_created",
            data=alert.__dict__,
            target_widget=target_dashboard
        )
        self._broadcast_event(event)
        
        self.logger.info(f"Created alert: {title} ({level.value})")
        return alert
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        if not alert.is_dismissible:
            return False
        
        del self.alerts[alert_id]
        
        # Broadcast dismissal event
        event = DashboardEvent(
            event_type="alert_dismissed",
            data={"alert_id": alert_id}
        )
        self._broadcast_event(event)
        
        return True
    
    def get_active_alerts(self, filter_level: AlertLevel = None) -> List[DashboardAlert]:
        """Get active alerts"""
        alerts = []
        current_time = datetime.now()
        
        for alert in self.alerts.values():
            # Skip expired alerts
            if alert.is_expired():
                continue
            
            # Apply level filter
            if filter_level and alert.level != filter_level:
                continue
            
            alerts.append(alert)
        
        # Sort by creation time (newest first)
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def register_event_listener(self, event_type: str, callback: Callable):
        """Register event listener"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(callback)
    
    def create_layout_template(self, name: str, layout: DashboardLayout) -> str:
        """Create a layout template"""
        template_id = f"template_{len(self.layout_templates)}"
        layout.name = name
        self.layout_templates[template_id] = layout
        
        self.logger.debug(f"Created layout template: {name}")
        return template_id
    
    def apply_layout_template(self, dashboard_id: str, template_id: str) -> bool:
        """Apply layout template to dashboard"""
        if dashboard_id not in self.dashboards or template_id not in self.layout_templates:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        template = self.layout_templates[template_id]
        
        dashboard.layout = template
        dashboard.updated_at = datetime.now()
        
        return True
    
    def create_custom_theme(self, name: str, theme_config: Dict[str, Any]) -> str:
        """Create custom theme"""
        theme_id = f"theme_{len(self.custom_themes)}"
        self.custom_themes[theme_id] = {
            "name": name,
            "config": theme_config,
            "created_at": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Created custom theme: {name}")
        return theme_id
    
    def apply_theme(self, dashboard_id: str, theme: DashboardTheme,
                   custom_theme_id: str = None) -> bool:
        """Apply theme to dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        dashboard.layout.theme = theme
        
        if theme == DashboardTheme.CUSTOM and custom_theme_id:
            if custom_theme_id in self.custom_themes:
                dashboard.layout.custom_theme_id = custom_theme_id
        
        dashboard.updated_at = datetime.now()
        return True
    
    async def start_dashboard_system(self):
        """Start the dashboard system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start widget manager
        await self.widget_manager.start_updates()
        
        # Start background tasks
        asyncio.create_task(self._alert_cleanup_loop())
        
        self.logger.info("Dashboard system started")
    
    async def stop_dashboard_system(self):
        """Stop the dashboard system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop widget manager
        await self.widget_manager.stop_updates()
        
        self.logger.info("Dashboard system stopped")
    
    def export_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Export dashboard configuration"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        return dashboard.to_dict()
    
    def import_dashboard(self, config_data: Dict[str, Any]) -> Optional[str]:
        """Import dashboard from configuration"""
        try:
            # Create dashboard from config
            dashboard = DashboardConfig(**config_data)
            
            # Register all widgets
            for widget_data in config_data.get("widgets", []):
                widget = DashboardWidget(**widget_data)
                self.widget_manager.register_widget(widget)
            
            # Store dashboard
            self.dashboards[dashboard.dashboard_id] = dashboard
            
            self.logger.info(f"Imported dashboard: {dashboard.name}")
            return dashboard.dashboard_id
            
        except Exception as e:
            self.logger.error(f"Failed to import dashboard: {e}")
            return None
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard system statistics"""
        widget_stats = self.widget_manager.get_manager_stats()
        
        return {
            "total_dashboards": len(self.dashboards),
            "active_users": len(self.active_dashboards),
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.get_active_alerts()),
            "custom_themes": len(self.custom_themes),
            "layout_templates": len(self.layout_templates),
            "event_listeners": sum(len(listeners) for listeners in self.event_listeners.values()),
            "widget_stats": widget_stats,
            "is_running": self.is_running
        }
    
    def _initialize_defaults(self):
        """Initialize default themes and layouts"""
        # Create default layout template
        default_layout = DashboardLayout(
            name="Default Grid",
            columns=12,
            row_height=60
        )
        self.layout_templates["default"] = default_layout
        
        # Create responsive layout template
        responsive_layout = DashboardLayout(
            name="Responsive Layout",
            columns=12,
            row_height=50,
            is_responsive=True
        )
        self.layout_templates["responsive"] = responsive_layout
    
    def _broadcast_event(self, event: DashboardEvent):
        """Broadcast event to listeners"""
        listeners = self.event_listeners.get(event.event_type, [])
        
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    asyncio.create_task(listener(event))
                else:
                    listener(event)
            except Exception as e:
                self.logger.error(f"Event listener failed: {e}")
    
    async def _alert_cleanup_loop(self):
        """Background loop to clean up expired alerts"""
        while self.is_running:
            try:
                expired_alerts = []
                current_time = datetime.now()
                
                for alert_id, alert in self.alerts.items():
                    if alert.is_expired():
                        expired_alerts.append(alert_id)
                
                for alert_id in expired_alerts:
                    del self.alerts[alert_id]
                
                if expired_alerts:
                    self.logger.debug(f"Cleaned up {len(expired_alerts)} expired alerts")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Alert cleanup loop failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error