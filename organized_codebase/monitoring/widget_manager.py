"""
Dashboard Widget Manager

This module manages dashboard widgets including creation, update,
rendering, and data binding functionality.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .dashboard_types import (
    DashboardWidget, ChartWidget, MetricWidget, WidgetType,
    ChartType, UpdateFrequency, DataSourceType, DashboardEvent
)


class WidgetManager:
    """Manages dashboard widgets and their lifecycle"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.widgets: Dict[str, DashboardWidget] = {}
        self.widget_data: Dict[str, Any] = {}
        self.data_sources: Dict[str, Callable] = {}
        self.update_callbacks: Dict[str, List[Callable]] = {}
        self.widget_cache: Dict[str, Dict[str, Any]] = {}
        
        # Background update tasks
        self.update_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        self.logger.info("WidgetManager initialized")
    
    def register_widget(self, widget: DashboardWidget) -> str:
        """Register a widget with the manager"""
        widget.updated_at = datetime.now()
        self.widgets[widget.widget_id] = widget
        
        # Initialize widget data
        self.widget_data[widget.widget_id] = None
        
        # Start update task if needed
        if widget.update_frequency != UpdateFrequency.MANUAL:
            self._start_widget_updates(widget)
        
        self.logger.debug(f"Registered widget: {widget.title} ({widget.widget_id})")
        return widget.widget_id
    
    def unregister_widget(self, widget_id: str) -> bool:
        """Unregister a widget"""
        if widget_id not in self.widgets:
            return False
        
        # Stop update task
        self._stop_widget_updates(widget_id)
        
        # Clean up
        del self.widgets[widget_id]
        self.widget_data.pop(widget_id, None)
        self.widget_cache.pop(widget_id, None)
        self.update_callbacks.pop(widget_id, None)
        
        self.logger.debug(f"Unregistered widget: {widget_id}")
        return True
    
    def update_widget(self, widget_id: str, updates: Dict[str, Any]) -> bool:
        """Update widget configuration"""
        if widget_id not in self.widgets:
            return False
        
        widget = self.widgets[widget_id]
        
        # Update widget properties
        for key, value in updates.items():
            if hasattr(widget, key):
                setattr(widget, key, value)
        
        widget.updated_at = datetime.now()
        
        # Restart update task if frequency changed
        if "update_frequency" in updates:
            self._stop_widget_updates(widget_id)
            if widget.update_frequency != UpdateFrequency.MANUAL:
                self._start_widget_updates(widget)
        
        self.logger.debug(f"Updated widget: {widget_id}")
        return True
    
    def get_widget(self, widget_id: str) -> Optional[DashboardWidget]:
        """Get widget by ID"""
        return self.widgets.get(widget_id)
    
    def get_widget_data(self, widget_id: str) -> Any:
        """Get current data for widget"""
        return self.widget_data.get(widget_id)
    
    def set_widget_data(self, widget_id: str, data: Any) -> bool:
        """Set data for widget"""
        if widget_id not in self.widgets:
            return False
        
        self.widget_data[widget_id] = data
        
        # Trigger update callbacks
        self._trigger_update_callbacks(widget_id, data)
        
        return True
    
    def register_data_source(self, source_name: str, data_function: Callable) -> None:
        """Register a data source function"""
        self.data_sources[source_name] = data_function
        self.logger.debug(f"Registered data source: {source_name}")
    
    def register_update_callback(self, widget_id: str, callback: Callable) -> None:
        """Register callback for widget updates"""
        if widget_id not in self.update_callbacks:
            self.update_callbacks[widget_id] = []
        
        self.update_callbacks[widget_id].append(callback)
    
    async def refresh_widget(self, widget_id: str) -> bool:
        """Manually refresh widget data"""
        if widget_id not in self.widgets:
            return False
        
        widget = self.widgets[widget_id]
        
        try:
            # Get fresh data
            data = await self._fetch_widget_data(widget)
            if data is not None:
                self.set_widget_data(widget_id, data)
                return True
        except Exception as e:
            self.logger.error(f"Failed to refresh widget {widget_id}: {e}")
        
        return False
    
    async def refresh_all_widgets(self) -> Dict[str, bool]:
        """Refresh all widgets"""
        results = {}
        
        for widget_id in self.widgets:
            results[widget_id] = await self.refresh_widget(widget_id)
        
        return results
    
    def create_chart_widget(self, title: str, chart_type: ChartType,
                           data_source: str = None, **kwargs) -> ChartWidget:
        """Create a chart widget"""
        widget = ChartWidget(
            title=title,
            chart_type=chart_type,
            data_source=data_source,
            **kwargs
        )
        
        self.register_widget(widget)
        return widget
    
    def create_metric_widget(self, title: str, value: Any = 0,
                           unit: str = "", **kwargs) -> MetricWidget:
        """Create a metric widget"""
        widget = MetricWidget(
            title=title,
            value=value,
            unit=unit,
            **kwargs
        )
        
        self.register_widget(widget)
        return widget
    
    def get_widgets_by_type(self, widget_type: WidgetType) -> List[DashboardWidget]:
        """Get all widgets of a specific type"""
        return [w for w in self.widgets.values() if w.widget_type == widget_type]
    
    def search_widgets(self, query: str) -> List[DashboardWidget]:
        """Search widgets by title or configuration"""
        query_lower = query.lower()
        results = []
        
        for widget in self.widgets.values():
            if (query_lower in widget.title.lower() or
                query_lower in str(widget.configuration).lower()):
                results.append(widget)
        
        return results
    
    async def start_updates(self):
        """Start background widget updates"""
        self.is_running = True
        
        for widget in self.widgets.values():
            if widget.update_frequency != UpdateFrequency.MANUAL:
                self._start_widget_updates(widget)
        
        self.logger.info("Started widget update manager")
    
    async def stop_updates(self):
        """Stop background widget updates"""
        self.is_running = False
        
        # Cancel all update tasks
        for task in self.update_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.update_tasks:
            await asyncio.gather(*self.update_tasks.values(), return_exceptions=True)
        
        self.update_tasks.clear()
        self.logger.info("Stopped widget update manager")
    
    def _start_widget_updates(self, widget: DashboardWidget):
        """Start background updates for a widget"""
        if not self.is_running:
            return
        
        if widget.widget_id in self.update_tasks:
            self.update_tasks[widget.widget_id].cancel()
        
        task = asyncio.create_task(self._widget_update_loop(widget))
        self.update_tasks[widget.widget_id] = task
    
    def _stop_widget_updates(self, widget_id: str):
        """Stop background updates for a widget"""
        if widget_id in self.update_tasks:
            self.update_tasks[widget_id].cancel()
            del self.update_tasks[widget_id]
    
    async def _widget_update_loop(self, widget: DashboardWidget):
        """Background update loop for a widget"""
        try:
            while self.is_running and widget.widget_id in self.widgets:
                # Fetch and update data
                data = await self._fetch_widget_data(widget)
                if data is not None:
                    self.set_widget_data(widget.widget_id, data)
                
                # Wait for next update
                interval = self._get_update_interval(widget.update_frequency)
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Widget update loop failed for {widget.widget_id}: {e}")
    
    async def _fetch_widget_data(self, widget: DashboardWidget) -> Any:
        """Fetch data for a widget"""
        if not widget.data_source:
            return None
        
        try:
            if widget.data_source_type == DataSourceType.STATIC:
                return widget.configuration.get("static_data")
            
            elif widget.data_source_type == DataSourceType.API:
                return await self._fetch_api_data(widget.data_source)
            
            elif widget.data_source in self.data_sources:
                data_func = self.data_sources[widget.data_source]
                if asyncio.iscoroutinefunction(data_func):
                    return await data_func(widget)
                else:
                    return data_func(widget)
            
            else:
                self.logger.warning(f"Unknown data source: {widget.data_source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to fetch data for widget {widget.widget_id}: {e}")
            return None
    
    async def _fetch_api_data(self, api_endpoint: str) -> Any:
        """Fetch data from API endpoint"""
        # In production, implement actual HTTP client
        # For now, return mock data
        return {"mock": "data", "timestamp": datetime.now().isoformat()}
    
    def _get_update_interval(self, frequency: UpdateFrequency) -> float:
        """Get update interval in seconds"""
        intervals = {
            UpdateFrequency.REAL_TIME: 0.1,
            UpdateFrequency.FAST: 1.0,
            UpdateFrequency.NORMAL: 5.0,
            UpdateFrequency.SLOW: 30.0,
            UpdateFrequency.MANUAL: float('inf')
        }
        return intervals.get(frequency, 5.0)
    
    def _trigger_update_callbacks(self, widget_id: str, data: Any):
        """Trigger update callbacks for a widget"""
        callbacks = self.update_callbacks.get(widget_id, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(widget_id, data))
                else:
                    callback(widget_id, data)
            except Exception as e:
                self.logger.error(f"Update callback failed: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get widget manager statistics"""
        widget_types = {}
        for widget in self.widgets.values():
            widget_type = widget.widget_type.value
            widget_types[widget_type] = widget_types.get(widget_type, 0) + 1
        
        return {
            "total_widgets": len(self.widgets),
            "widget_types": widget_types,
            "active_update_tasks": len(self.update_tasks),
            "registered_data_sources": len(self.data_sources),
            "widgets_with_callbacks": len(self.update_callbacks),
            "is_running": self.is_running
        }