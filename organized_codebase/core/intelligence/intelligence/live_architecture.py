"""
Live Architecture Documentation

Real-time system topology mapping with interactive diagrams and performance overlay.
"""

import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Thread, Lock
import logging

from .diagram_creator import DiagramCreator, Component, Relationship

logger = logging.getLogger(__name__)


@dataclass
class LiveComponent:
    """Live component with performance metrics."""
    name: str
    type: str
    file_path: str
    cpu_usage: float
    memory_usage: float
    last_accessed: str
    health_status: str
    dependencies: List[str]
    

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    component: str
    metric_type: str  # cpu, memory, response_time, throughput
    value: float
    timestamp: str
    

class LiveArchitectureDocumentation:
    """
    Real-time architecture documentation with live performance metrics.
    Provides interactive topology mapping and system health visualization.
    """
    
    def __init__(self, update_interval: int = 10):
        """
        Initialize live architecture documentation.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.diagram_creator = DiagramCreator()
        self.live_components = {}
        self.performance_history = []
        self.monitoring_active = False
        self.metrics_lock = Lock()
        
        # System monitoring
        self.system_info = {}
        self.process_map = {}
        
        logger.info("Live Architecture Documentation initialized")
        
    def start_live_monitoring(self, project_path: str) -> None:
        """
        Start live architecture monitoring.
        
        Args:
            project_path: Path to monitor
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.project_path = project_path
        
        # Initial architecture analysis
        self.diagram_creator.analyze_architecture(project_path)
        
        # Start monitoring thread
        monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("Live architecture monitoring started")
        
    def stop_live_monitoring(self) -> None:
        """Stop live architecture monitoring."""
        self.monitoring_active = False
        logger.info("Live architecture monitoring stopped")
        
    def get_live_topology(self) -> Dict[str, Any]:
        """
        Get current live topology with performance data.
        
        Returns:
            Live topology data
        """
        with self.metrics_lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'components': [asdict(comp) for comp in self.live_components.values()],
                'system_info': self.system_info,
                'performance_summary': self._calculate_performance_summary(),
                'health_overview': self._calculate_health_overview(),
                'topology_graph': self.diagram_creator.generate_component_graph()
            }
            
    def get_component_details(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a component.
        
        Args:
            component_name: Name of component
            
        Returns:
            Component details
        """
        if component_name in self.live_components:
            component = self.live_components[component_name]
            
            # Get performance history for this component
            component_metrics = [
                m for m in self.performance_history
                if m.component == component_name
            ]
            
            return {
                'component': asdict(component),
                'performance_history': [asdict(m) for m in component_metrics[-100:]],  # Last 100 points
                'dependencies': component.dependencies,
                'dependents': self._find_dependents(component_name),
                'file_info': self._get_file_info(component.file_path)
            }
            
        return None
        
    def generate_interactive_diagram(self, format: str = "mermaid") -> str:
        """
        Generate interactive architecture diagram with live data.
        
        Args:
            format: Diagram format (mermaid, d3, cytoscape)
            
        Returns:
            Interactive diagram definition
        """
        if format == "mermaid":
            return self._generate_live_mermaid()
        elif format == "d3":
            return self._generate_d3_diagram()
        elif format == "cytoscape":
            return self._generate_cytoscape_diagram()
        else:
            return self.diagram_creator.generate_mermaid_diagram()
            
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[PerformanceMetric]]:
        """
        Get performance trends over time.
        
        Args:
            hours: Hours of history
            
        Returns:
            Performance trends by component
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        trends = {}
        for metric in self.performance_history:
            if datetime.fromisoformat(metric.timestamp) >= cutoff_time:
                if metric.component not in trends:
                    trends[metric.component] = []
                trends[metric.component].append(metric)
                
        return trends
        
    def detect_architecture_changes(self) -> List[Dict[str, Any]]:
        """
        Detect changes in architecture.
        
        Returns:
            List of detected changes
        """
        changes = []
        
        # Re-analyze architecture
        current_components = self.diagram_creator.components
        previous_components = set(self.live_components.keys())
        current_component_names = set(current_components.keys())
        
        # Detect new components
        new_components = current_component_names - previous_components
        for comp_name in new_components:
            changes.append({
                'type': 'component_added',
                'component': comp_name,
                'timestamp': datetime.now().isoformat()
            })
            
        # Detect removed components
        removed_components = previous_components - current_component_names
        for comp_name in removed_components:
            changes.append({
                'type': 'component_removed',
                'component': comp_name,
                'timestamp': datetime.now().isoformat()
            })
            
        return changes
        
    def generate_architecture_report(self) -> str:
        """
        Generate live architecture report.
        
        Returns:
            Architecture report in markdown
        """
        report = [
            "# Live Architecture Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Overview",
            f"- Total Components: {len(self.live_components)}",
            f"- Monitoring Status: {'Active' if self.monitoring_active else 'Inactive'}",
            f"- System CPU: {self.system_info.get('cpu_percent', 0):.1f}%",
            f"- System Memory: {self.system_info.get('memory_percent', 0):.1f}%",
            "",
            "## Component Health",
            ""
        ]
        
        for comp_name, component in self.live_components.items():
            status_emoji = "🟢" if component.health_status == "healthy" else "🔴"
            report.append(f"- {status_emoji} **{comp_name}**: {component.health_status}")
            
        report.extend([
            "",
            "## Performance Summary",
            ""
        ])
        
        perf_summary = self._calculate_performance_summary()
        for metric, value in perf_summary.items():
            report.append(f"- {metric}: {value}")
            
        return "\n".join(report)
        
    # Private methods
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._update_component_metrics()
                self._cleanup_old_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
                
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics."""
        with self.metrics_lock:
            self.system_info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'boot_time': psutil.boot_time(),
                'timestamp': datetime.now().isoformat()
            }
            
    def _update_component_metrics(self) -> None:
        """Update component-specific metrics."""
        timestamp = datetime.now().isoformat()
        
        # Update existing components
        for comp_name, diagram_component in self.diagram_creator.components.items():
            # Simulate component metrics (in real implementation, this would
            # connect to actual monitoring systems)
            cpu_usage = psutil.cpu_percent() / len(self.diagram_creator.components)
            memory_usage = psutil.virtual_memory().percent / len(self.diagram_creator.components)
            
            live_comp = LiveComponent(
                name=comp_name,
                type=diagram_component.type,
                file_path=diagram_component.file_path,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                last_accessed=timestamp,
                health_status="healthy" if cpu_usage < 80 and memory_usage < 80 else "warning",
                dependencies=diagram_component.dependencies
            )
            
            self.live_components[comp_name] = live_comp
            
            # Record performance metrics
            self.performance_history.append(PerformanceMetric(
                component=comp_name,
                metric_type="cpu",
                value=cpu_usage,
                timestamp=timestamp
            ))
            
            self.performance_history.append(PerformanceMetric(
                component=comp_name,
                metric_type="memory",
                value=memory_usage,
                timestamp=timestamp
            ))
            
    def _cleanup_old_metrics(self) -> None:
        """Clean up old performance metrics."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.performance_history = [
            metric for metric in self.performance_history
            if datetime.fromisoformat(metric.timestamp) >= cutoff_time
        ]
        
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary."""
        if not self.live_components:
            return {}
            
        total_cpu = sum(comp.cpu_usage for comp in self.live_components.values())
        total_memory = sum(comp.memory_usage for comp in self.live_components.values())
        
        return {
            'average_cpu_usage': total_cpu / len(self.live_components),
            'average_memory_usage': total_memory / len(self.live_components),
            'healthy_components': len([c for c in self.live_components.values() if c.health_status == "healthy"]),
            'total_components': len(self.live_components)
        }
        
    def _calculate_health_overview(self) -> Dict[str, int]:
        """Calculate health overview."""
        health_counts = {'healthy': 0, 'warning': 0, 'critical': 0}
        
        for component in self.live_components.values():
            health_counts[component.health_status] = health_counts.get(component.health_status, 0) + 1
            
        return health_counts
        
    def _find_dependents(self, component_name: str) -> List[str]:
        """Find components that depend on this component."""
        dependents = []
        
        for comp_name, component in self.live_components.items():
            if component_name in component.dependencies:
                dependents.append(comp_name)
                
        return dependents
        
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        try:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                return {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'lines': len(path.read_text().splitlines()) if path.suffix == '.py' else 0
                }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            
        return {}
        
    def _generate_live_mermaid(self) -> str:
        """Generate Mermaid diagram with live performance data."""
        lines = ["graph TD"]
        
        # Add components with performance indicators
        for comp_name, component in self.live_components.items():
            status_color = "green" if component.health_status == "healthy" else "red"
            cpu_indicator = f"CPU:{component.cpu_usage:.1f}%"
            
            lines.append(f"    {comp_name}[{component.name}<br/>{cpu_indicator}]")
            lines.append(f"    style {comp_name} fill:{status_color}")
            
        # Add relationships
        for relationship in self.diagram_creator.relationships:
            lines.append(f"    {relationship.source} --> {relationship.target}")
            
        return "\n".join(lines)
        
    def _generate_d3_diagram(self) -> str:
        """Generate D3.js diagram definition."""
        nodes = []
        links = []
        
        for comp_name, component in self.live_components.items():
            nodes.append({
                'id': comp_name,
                'name': component.name,
                'type': component.type,
                'cpu_usage': component.cpu_usage,
                'memory_usage': component.memory_usage,
                'health': component.health_status
            })
            
        for rel in self.diagram_creator.relationships:
            links.append({
                'source': rel.source,
                'target': rel.target,
                'type': rel.type
            })
            
        return json.dumps({'nodes': nodes, 'links': links}, indent=2)
        
    def _generate_cytoscape_diagram(self) -> str:
        """Generate Cytoscape.js diagram definition."""
        elements = []
        
        # Add nodes
        for comp_name, component in self.live_components.items():
            elements.append({
                'data': {
                    'id': comp_name,
                    'label': component.name,
                    'cpu': component.cpu_usage,
                    'memory': component.memory_usage,
                    'health': component.health_status
                }
            })
            
        # Add edges
        for rel in self.diagram_creator.relationships:
            elements.append({
                'data': {
                    'id': f"{rel.source}-{rel.target}",
                    'source': rel.source,
                    'target': rel.target,
                    'type': rel.type
                }
            })
            
        return json.dumps(elements, indent=2)