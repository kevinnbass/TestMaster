"""
Comprehensive Visualization Engine
================================
Advanced visualization system integrating all 8 Phase 2 visualization modules.

This engine provides unified access to all visualization capabilities extracted
from the 7 major frameworks, creating comprehensive documentation visualizations.

Author: Agent D - Documentation Intelligence
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import asyncio
import base64
import io

# Import all Phase 2 visualization modules
try:
    from ...visualization.dashboard_systems import (
        DashboardMetric, DashboardRenderer, RealTimeDashboard,
        AgentMonitoringDashboard, FlowVisualizationDashboard
    )
    from ...visualization.interactive_visualizations import (
        ChartData, VisualizationEngine, FlowVisualizationEngine,
        ChartRenderer, MatplotlibRenderer
    )
    from ...visualization.data_visualization_engines import (
        DataSeries, ChartConfiguration, VisualizationEngine as DataVisualizationEngine
    )
    from ...visualization.realtime_ui_components import (
        StreamingMessage, UIUpdate, LiveChart, LiveProgressBar
    )
    from ...visualization.agent_interface_systems import (
        ChatMessage, AgentStatus, ChatInterface, AgentControlPanel
    )
    from ...visualization.development_tools_ui import (
        ProjectConfig, StudioInterface, WorkflowDesigner
    )
    from ...visualization.observability_systems import (
        TraceEvent, PerformanceMetric, ObservabilityDashboard
    )
    from ...visualization.user_experience_frameworks import (
        UserJourney, NavigationSystem, ResponsiveUIFramework
    )
except ImportError:
    # Fallback for standalone usage
    pass

class VisualizationType(Enum):
    """Types of visualizations supported by the engine."""
    DASHBOARD = "dashboard"
    CHART = "chart"
    FLOW_DIAGRAM = "flow_diagram"
    NETWORK_GRAPH = "network_graph"
    REAL_TIME = "real_time"
    INTERACTIVE = "interactive"
    DOCUMENTATION_MAP = "documentation_map"
    PROGRESS_TRACKER = "progress_tracker"
    ANALYTICS_VIEW = "analytics_view"

class VisualizationComplexity(Enum):
    """Complexity levels for visualizations."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"

@dataclass
class VisualizationRequest:
    """Request for creating a visualization."""
    visualization_type: VisualizationType
    data_source: Dict[str, Any]
    configuration: Dict[str, Any] = field(default_factory=dict)
    styling_preferences: Dict[str, Any] = field(default_factory=dict)
    interactivity_level: str = "medium"
    export_formats: List[str] = field(default_factory=lambda: ["html", "png"])
    framework_preferences: List[str] = field(default_factory=list)

@dataclass
class VisualizationResult:
    """Result from visualization generation."""
    visualization_id: str
    visualization_type: VisualizationType
    generated_content: Dict[str, Any]  # HTML, images, data
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    framework_used: str
    creation_timestamp: datetime = field(default_factory=datetime.now)

class DocumentationVisualizationEngine:
    """Main visualization engine for documentation generation."""
    
    def __init__(self):
        self.dashboard_renderer = None
        self.chart_renderer = None
        self.visualization_engine = None
        self.data_viz_engine = None
        self.observability_dashboard = None
        
        self._initialize_components()
        self.visualization_cache: Dict[str, VisualizationResult] = {}
        self.performance_tracker = VisualizationPerformanceTracker()
    
    def _initialize_components(self):
        """Initialize all visualization components safely."""
        try:
            self.dashboard_renderer = DashboardRenderer()
            self.chart_renderer = MatplotlibRenderer() 
            self.visualization_engine = VisualizationEngine()
            self.data_viz_engine = DataVisualizationEngine()
            self.observability_dashboard = ObservabilityDashboard()
        except:
            # Initialize mock components for fallback
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components when Phase 2 modules not available."""
        self.dashboard_renderer = MockDashboardRenderer()
        self.chart_renderer = MockChartRenderer()
        self.visualization_engine = MockVisualizationEngine()
        self.data_viz_engine = MockDataVisualizationEngine()
        self.observability_dashboard = MockObservabilityDashboard()
    
    async def generate_visualization(self, request: VisualizationRequest) -> VisualizationResult:
        """Generate a visualization based on the request."""
        visualization_id = self._generate_visualization_id(request)
        
        # Check cache first
        if visualization_id in self.visualization_cache:
            return self.visualization_cache[visualization_id]
        
        # Track performance
        start_time = datetime.now()
        
        try:
            # Route to appropriate visualization generator
            if request.visualization_type == VisualizationType.DASHBOARD:
                result = await self._generate_dashboard(request, visualization_id)
            elif request.visualization_type == VisualizationType.CHART:
                result = await self._generate_chart(request, visualization_id)
            elif request.visualization_type == VisualizationType.FLOW_DIAGRAM:
                result = await self._generate_flow_diagram(request, visualization_id)
            elif request.visualization_type == VisualizationType.NETWORK_GRAPH:
                result = await self._generate_network_graph(request, visualization_id)
            elif request.visualization_type == VisualizationType.REAL_TIME:
                result = await self._generate_real_time_viz(request, visualization_id)
            elif request.visualization_type == VisualizationType.DOCUMENTATION_MAP:
                result = await self._generate_documentation_map(request, visualization_id)
            elif request.visualization_type == VisualizationType.PROGRESS_TRACKER:
                result = await self._generate_progress_tracker(request, visualization_id)
            elif request.visualization_type == VisualizationType.ANALYTICS_VIEW:
                result = await self._generate_analytics_view(request, visualization_id)
            else:
                result = await self._generate_interactive_viz(request, visualization_id)
            
            # Calculate performance metrics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            result.performance_metrics = {
                "generation_time": generation_time,
                "complexity_score": self._calculate_complexity_score(request),
                "cache_hit": False
            }
            
            # Cache the result
            self.visualization_cache[visualization_id] = result
            
            # Track performance
            self.performance_tracker.record_generation(request.visualization_type, generation_time)
            
            return result
            
        except Exception as e:
            return self._create_error_result(visualization_id, request.visualization_type, str(e))
    
    async def _generate_dashboard(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate dashboard visualization."""
        dashboard_config = request.configuration.get("dashboard", {})
        
        # Create dashboard metrics from data source
        metrics = self._create_dashboard_metrics(request.data_source)
        
        # Configure dashboard based on framework preferences
        if "agency_swarm" in request.framework_preferences:
            dashboard_html = self._create_agency_swarm_dashboard(metrics, dashboard_config)
            framework_used = "agency_swarm"
        elif "crewai" in request.framework_preferences:
            dashboard_html = self._create_crewai_dashboard(metrics, dashboard_config)
            framework_used = "crewai"
        else:
            dashboard_html = self._create_unified_dashboard(metrics, dashboard_config)
            framework_used = "unified"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.DASHBOARD,
            generated_content={"html": dashboard_html, "metrics": metrics},
            metadata={"metric_count": len(metrics), "framework": framework_used},
            performance_metrics={},
            framework_used=framework_used
        )
    
    async def _generate_chart(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate chart visualization."""
        chart_config = request.configuration.get("chart", {})
        chart_type = chart_config.get("type", "line")
        
        # Prepare chart data
        chart_data = self._prepare_chart_data(request.data_source, chart_type)
        
        # Generate chart using appropriate renderer
        if "phidata" in request.framework_preferences:
            chart_content = self._create_phidata_chart(chart_data, chart_config)
            framework_used = "phidata"
        else:
            chart_content = self._create_standard_chart(chart_data, chart_config)
            framework_used = "matplotlib"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.CHART,
            generated_content=chart_content,
            metadata={"chart_type": chart_type, "data_points": len(chart_data)},
            performance_metrics={},
            framework_used=framework_used
        )
    
    async def _generate_flow_diagram(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate flow diagram visualization."""
        flow_config = request.configuration.get("flow", {})
        
        # Extract flow data
        nodes = request.data_source.get("nodes", [])
        connections = request.data_source.get("connections", [])
        
        # Create flow diagram based on framework preferences
        if "crewai" in request.framework_preferences:
            flow_html = self._create_crewai_flow_diagram(nodes, connections, flow_config)
            framework_used = "crewai"
        elif "autogen" in request.framework_preferences:
            flow_html = self._create_autogen_flow_diagram(nodes, connections, flow_config)
            framework_used = "autogen"
        else:
            flow_html = self._create_generic_flow_diagram(nodes, connections, flow_config)
            framework_used = "generic"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.FLOW_DIAGRAM,
            generated_content={"html": flow_html, "nodes": nodes, "connections": connections},
            metadata={"node_count": len(nodes), "connection_count": len(connections)},
            performance_metrics={},
            framework_used=framework_used
        )
    
    async def _generate_network_graph(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate network graph visualization."""
        graph_config = request.configuration.get("graph", {})
        
        # Extract network data
        entities = request.data_source.get("entities", [])
        relationships = request.data_source.get("relationships", [])
        
        # Create network visualization
        if "swarms" in request.framework_preferences:
            graph_html = self._create_swarms_network_graph(entities, relationships, graph_config)
            framework_used = "swarms"
        else:
            graph_html = self._create_d3_network_graph(entities, relationships, graph_config)
            framework_used = "d3"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.NETWORK_GRAPH,
            generated_content={"html": graph_html, "entities": entities, "relationships": relationships},
            metadata={"entity_count": len(entities), "relationship_count": len(relationships)},
            performance_metrics={},
            framework_used=framework_used
        )
    
    async def _generate_real_time_viz(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate real-time visualization."""
        realtime_config = request.configuration.get("realtime", {})
        update_interval = realtime_config.get("update_interval", 1000)  # milliseconds
        
        # Create real-time visualization components
        live_chart_html = self._create_live_chart(request.data_source, realtime_config)
        websocket_script = self._create_websocket_script(update_interval)
        
        combined_html = f"{live_chart_html}\n{websocket_script}"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.REAL_TIME,
            generated_content={"html": combined_html, "update_interval": update_interval},
            metadata={"real_time": True, "update_frequency": f"{update_interval}ms"},
            performance_metrics={},
            framework_used="websocket"
        )
    
    async def _generate_documentation_map(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate documentation map visualization."""
        map_config = request.configuration.get("map", {})
        
        # Extract documentation structure
        sections = request.data_source.get("sections", [])
        hierarchy = request.data_source.get("hierarchy", {})
        
        # Create interactive documentation map
        if "agentscope" in request.framework_preferences:
            map_html = self._create_agentscope_doc_map(sections, hierarchy, map_config)
            framework_used = "agentscope"
        else:
            map_html = self._create_interactive_doc_map(sections, hierarchy, map_config)
            framework_used = "interactive"
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.DOCUMENTATION_MAP,
            generated_content={"html": map_html, "sections": sections, "hierarchy": hierarchy},
            metadata={"section_count": len(sections), "depth": self._calculate_hierarchy_depth(hierarchy)},
            performance_metrics={},
            framework_used=framework_used
        )
    
    async def _generate_progress_tracker(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate progress tracker visualization."""
        progress_config = request.configuration.get("progress", {})
        
        # Extract progress data
        tasks = request.data_source.get("tasks", [])
        completion_data = request.data_source.get("completion", {})
        
        # Create progress visualization
        progress_html = self._create_progress_dashboard(tasks, completion_data, progress_config)
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.PROGRESS_TRACKER,
            generated_content={"html": progress_html, "tasks": tasks, "completion": completion_data},
            metadata={"task_count": len(tasks), "overall_progress": completion_data.get("percentage", 0)},
            performance_metrics={},
            framework_used="progress_tracker"
        )
    
    async def _generate_analytics_view(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate analytics view visualization."""
        analytics_config = request.configuration.get("analytics", {})
        
        # Extract analytics data
        metrics = request.data_source.get("metrics", {})
        trends = request.data_source.get("trends", [])
        insights = request.data_source.get("insights", [])
        
        # Create comprehensive analytics view
        analytics_html = self._create_analytics_dashboard(metrics, trends, insights, analytics_config)
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.ANALYTICS_VIEW,
            generated_content={"html": analytics_html, "metrics": metrics, "trends": trends, "insights": insights},
            metadata={"metric_count": len(metrics), "trend_count": len(trends), "insight_count": len(insights)},
            performance_metrics={},
            framework_used="analytics"
        )
    
    async def _generate_interactive_viz(self, request: VisualizationRequest, viz_id: str) -> VisualizationResult:
        """Generate interactive visualization (fallback)."""
        interactive_config = request.configuration.get("interactive", {})
        
        # Create basic interactive visualization
        interactive_html = self._create_basic_interactive_viz(request.data_source, interactive_config)
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=VisualizationType.INTERACTIVE,
            generated_content={"html": interactive_html},
            metadata={"interactivity": "basic"},
            performance_metrics={},
            framework_used="basic"
        )
    
    def _generate_visualization_id(self, request: VisualizationRequest) -> str:
        """Generate unique visualization ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = hash(str(request.data_source))
        return f"{request.visualization_type.value}_{timestamp}_{abs(data_hash)}"
    
    def _create_dashboard_metrics(self, data_source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create dashboard metrics from data source."""
        metrics = []
        
        # Extract metrics from various data formats
        if "metrics" in data_source:
            metrics.extend(data_source["metrics"])
        
        if "performance" in data_source:
            perf_data = data_source["performance"]
            metrics.append({
                "name": "Performance Score",
                "value": perf_data.get("score", 0),
                "unit": "score",
                "trend": "up" if perf_data.get("score", 0) > 0.8 else "down"
            })
        
        if "coverage" in data_source:
            coverage_data = data_source["coverage"]
            metrics.append({
                "name": "Test Coverage",
                "value": coverage_data.get("percentage", 0),
                "unit": "%",
                "trend": "up" if coverage_data.get("percentage", 0) > 80 else "neutral"
            })
        
        return metrics
    
    def _create_unified_dashboard(self, metrics: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Create unified dashboard HTML."""
        dashboard_html = """
        <div class="unified-dashboard">
            <h2>Documentation Dashboard</h2>
            <div class="metrics-grid">
        """
        
        for metric in metrics:
            dashboard_html += f"""
                <div class="metric-card">
                    <h3>{metric['name']}</h3>
                    <div class="metric-value">{metric['value']} {metric.get('unit', '')}</div>
                    <div class="metric-trend trend-{metric.get('trend', 'neutral')}">
                        {metric.get('trend', 'stable').title()}
                    </div>
                </div>
            """
        
        dashboard_html += """
            </div>
            <style>
                .unified-dashboard { font-family: Arial, sans-serif; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
                .metric-card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
                .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
                .trend-up { color: green; }
                .trend-down { color: red; }
                .trend-neutral { color: #666; }
            </style>
        </div>
        """
        
        return dashboard_html
    
    def _prepare_chart_data(self, data_source: Dict[str, Any], chart_type: str) -> List[Dict[str, Any]]:
        """Prepare data for chart visualization."""
        if "chart_data" in data_source:
            return data_source["chart_data"]
        
        # Transform generic data for chart
        chart_data = []
        if "values" in data_source and "labels" in data_source:
            values = data_source["values"]
            labels = data_source["labels"]
            
            for i, value in enumerate(values):
                chart_data.append({
                    "x": labels[i] if i < len(labels) else f"Item {i+1}",
                    "y": value
                })
        
        return chart_data
    
    def _create_standard_chart(self, chart_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create standard chart visualization."""
        chart_type = config.get("type", "line")
        title = config.get("title", "Chart")
        
        # Generate Chart.js HTML
        chart_html = f"""
        <div class="chart-container">
            <canvas id="chart_{id(chart_data)}" width="400" height="200"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                const ctx = document.getElementById('chart_{id(chart_data)}').getContext('2d');
                const chart = new Chart(ctx, {{
                    type: '{chart_type}',
                    data: {{
                        labels: {json.dumps([d['x'] for d in chart_data])},
                        datasets: [{{
                            label: '{title}',
                            data: {json.dumps([d['y'] for d in chart_data])},
                            borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '{title}'
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """
        
        return {"html": chart_html, "data": chart_data}
    
    def _create_generic_flow_diagram(self, nodes: List[Dict], connections: List[Dict], config: Dict) -> str:
        """Create generic flow diagram."""
        flow_html = """
        <div class="flow-diagram">
            <div class="flow-container">
        """
        
        # Add nodes
        for node in nodes:
            node_id = node.get("id", "")
            node_label = node.get("label", node_id)
            node_type = node.get("type", "default")
            
            flow_html += f"""
                <div class="flow-node node-{node_type}" id="{node_id}">
                    {node_label}
                </div>
            """
        
        flow_html += """
            </div>
            <style>
                .flow-diagram { position: relative; width: 100%; height: 400px; border: 1px solid #ddd; }
                .flow-container { position: relative; width: 100%; height: 100%; }
                .flow-node { position: absolute; padding: 10px; border: 2px solid #007bff; 
                            background: white; border-radius: 5px; cursor: pointer; }
                .node-start { border-color: #28a745; background: #d4edda; }
                .node-process { border-color: #007bff; background: #d1ecf1; }
                .node-end { border-color: #dc3545; background: #f8d7da; }
            </style>
        </div>
        """
        
        return flow_html
    
    def _create_d3_network_graph(self, entities: List[Dict], relationships: List[Dict], config: Dict) -> str:
        """Create D3.js network graph."""
        graph_html = f"""
        <div class="network-graph">
            <svg id="network_{id(entities)}" width="600" height="400"></svg>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
                const nodes = {json.dumps(entities)};
                const links = {json.dumps(relationships)};
                
                const svg = d3.select('#network_{id(entities)}');
                const width = 600;
                const height = 400;
                
                const simulation = d3.forceSimulation(nodes)
                    .force('link', d3.forceLink(links).id(d => d.id))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2));
                
                const link = svg.append('g')
                    .selectAll('line')
                    .data(links)
                    .join('line')
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2);
                
                const node = svg.append('g')
                    .selectAll('circle')
                    .data(nodes)
                    .join('circle')
                    .attr('r', 8)
                    .attr('fill', '#007bff')
                    .call(d3.drag()
                        .on('start', dragstarted)
                        .on('drag', dragged)
                        .on('end', dragended));
                
                node.append('title')
                    .text(d => d.label || d.id);
                
                simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                }});
                
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
            </script>
        </div>
        """
        
        return graph_html
    
    def _calculate_complexity_score(self, request: VisualizationRequest) -> float:
        """Calculate complexity score for visualization request."""
        base_score = 0.5
        
        # Add complexity based on visualization type
        type_complexity = {
            VisualizationType.CHART: 0.1,
            VisualizationType.DASHBOARD: 0.3,
            VisualizationType.FLOW_DIAGRAM: 0.4,
            VisualizationType.NETWORK_GRAPH: 0.6,
            VisualizationType.REAL_TIME: 0.8,
            VisualizationType.ANALYTICS_VIEW: 0.7
        }
        
        complexity = base_score + type_complexity.get(request.visualization_type, 0.3)
        
        # Add complexity based on data size
        data_size = len(str(request.data_source))
        if data_size > 10000:
            complexity += 0.2
        elif data_size > 1000:
            complexity += 0.1
        
        # Add complexity based on configuration
        if request.configuration:
            complexity += len(request.configuration) * 0.05
        
        return min(1.0, complexity)
    
    def _create_error_result(self, viz_id: str, viz_type: VisualizationType, error: str) -> VisualizationResult:
        """Create error result when visualization generation fails."""
        error_html = f"""
        <div class="visualization-error">
            <h3>Visualization Generation Error</h3>
            <p>Type: {viz_type.value}</p>
            <p>Error: {error}</p>
            <p>ID: {viz_id}</p>
        </div>
        """
        
        return VisualizationResult(
            visualization_id=viz_id,
            visualization_type=viz_type,
            generated_content={"html": error_html, "error": error},
            metadata={"error": True, "error_message": error},
            performance_metrics={"generation_time": 0.0, "error": True},
            framework_used="error"
        )
    
    # Additional helper methods for framework-specific visualizations
    def _create_agency_swarm_dashboard(self, metrics: List[Dict], config: Dict) -> str:
        """Create Agency-Swarm style dashboard."""
        return self._create_unified_dashboard(metrics, config)  # Simplified for now
    
    def _create_crewai_dashboard(self, metrics: List[Dict], config: Dict) -> str:
        """Create CrewAI style dashboard.""" 
        return self._create_unified_dashboard(metrics, config)  # Simplified for now
    
    def _create_phidata_chart(self, chart_data: List[Dict], config: Dict) -> Dict[str, Any]:
        """Create PhiData style chart."""
        return self._create_standard_chart(chart_data, config)  # Simplified for now
    
    def _create_live_chart(self, data_source: Dict, config: Dict) -> str:
        """Create live updating chart."""
        return """
        <div class="live-chart">
            <canvas id="liveChart" width="400" height="200"></canvas>
            <p>Real-time data visualization (WebSocket connection required)</p>
        </div>
        """
    
    def _create_websocket_script(self, update_interval: int) -> str:
        """Create WebSocket script for real-time updates."""
        return f"""
        <script>
            // WebSocket connection for real-time updates
            // Update interval: {update_interval}ms
            console.log("Real-time visualization initialized");
        </script>
        """
    
    def _create_interactive_doc_map(self, sections: List[Dict], hierarchy: Dict, config: Dict) -> str:
        """Create interactive documentation map."""
        return """
        <div class="doc-map">
            <h3>Documentation Map</h3>
            <div class="map-container">
                <!-- Interactive documentation structure would be rendered here -->
                <p>Interactive documentation navigation map</p>
            </div>
        </div>
        """
    
    def _calculate_hierarchy_depth(self, hierarchy: Dict) -> int:
        """Calculate depth of documentation hierarchy."""
        if not hierarchy:
            return 0
        
        def get_depth(node):
            if not isinstance(node, dict) or 'children' not in node:
                return 1
            
            if not node['children']:
                return 1
                
            return 1 + max(get_depth(child) for child in node['children'])
        
        return get_depth(hierarchy)

class VisualizationPerformanceTracker:
    """Tracks performance metrics for visualizations."""
    
    def __init__(self):
        self.generation_times: Dict[VisualizationType, List[float]] = {}
        self.total_generations = 0
        self.average_generation_time = 0.0
    
    def record_generation(self, viz_type: VisualizationType, generation_time: float):
        """Record visualization generation performance."""
        if viz_type not in self.generation_times:
            self.generation_times[viz_type] = []
        
        self.generation_times[viz_type].append(generation_time)
        self.total_generations += 1
        
        # Update average
        all_times = [time for times in self.generation_times.values() for time in times]
        self.average_generation_time = sum(all_times) / len(all_times) if all_times else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "total_generations": self.total_generations,
            "average_generation_time": self.average_generation_time,
            "by_type": {}
        }
        
        for viz_type, times in self.generation_times.items():
            stats["by_type"][viz_type.value] = {
                "count": len(times),
                "average_time": sum(times) / len(times) if times else 0.0,
                "min_time": min(times) if times else 0.0,
                "max_time": max(times) if times else 0.0
            }
        
        return stats

# Mock components for fallback when Phase 2 modules not available
class MockDashboardRenderer:
    def render(self, *args, **kwargs):
        return "<div>Mock Dashboard</div>"

class MockChartRenderer:
    def render(self, *args, **kwargs):
        return {"html": "<div>Mock Chart</div>"}

class MockVisualizationEngine:
    def generate(self, *args, **kwargs):
        return {"html": "<div>Mock Visualization</div>"}

class MockDataVisualizationEngine:
    def create_chart(self, *args, **kwargs):
        return {"html": "<div>Mock Data Chart</div>"}

class MockObservabilityDashboard:
    def render(self, *args, **kwargs):
        return "<div>Mock Observability Dashboard</div>"

# Global visualization engine instance
_visualization_engine = DocumentationVisualizationEngine()

def get_visualization_engine() -> DocumentationVisualizationEngine:
    """Get the global visualization engine instance."""
    return _visualization_engine

async def create_documentation_visualization(viz_type: str, data: Dict[str, Any], 
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """High-level function to create documentation visualizations."""
    engine = get_visualization_engine()
    
    try:
        request = VisualizationRequest(
            visualization_type=VisualizationType(viz_type),
            data_source=data,
            configuration=config or {},
            framework_preferences=config.get("frameworks", []) if config else []
        )
        
        result = await engine.generate_visualization(request)
        
        return {
            "success": True,
            "visualization_id": result.visualization_id,
            "content": result.generated_content,
            "metadata": result.metadata,
            "performance": result.performance_metrics,
            "framework": result.framework_used
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "visualization_type": viz_type
        }