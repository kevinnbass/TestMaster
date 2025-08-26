"""
Phase 3 UI/UX Test
==================

Validates advanced UI/UX system components.
"""

import sys
import asyncio
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_studio_interface():
    """Test Studio Interface system"""
    print("Testing Studio Interface...")
    
    try:
        from ui_ux import (
            StudioInterface,
            InteractionMode,
            VisualizationType,
            StudioSession,
            StudioWorkflow
        )
        
        # Create studio
        studio = StudioInterface()
        print(f"[OK] Created studio: {studio.studio_id}")
        
        # Create session
        session_id = studio.create_session("test_user", InteractionMode.VISUAL)
        print(f"[OK] Created session: {session_id}")
        
        # Create workflow
        workflow_id = studio.create_workflow(session_id, "Test Workflow")
        print(f"[OK] Created workflow: {workflow_id}")
        
        # Add agent node
        node_id = studio.add_agent_node(session_id, "test_executor", {"x": 100, "y": 100})
        print(f"[OK] Added agent node: {node_id}")
        
        # Generate code
        code = studio.generate_code(session_id)
        print(f"[OK] Generated code: {len(code)} characters")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Studio Interface test failed: {e}")
        return False

def test_agentverse_ui():
    """Test AgentVerse UI system"""
    print("\nTesting AgentVerse UI...")
    
    try:
        from ui_ux import (
            AgentVerseUI,
            AgentVisualization,
            InteractionGraph,
            AgentCard,
            WorkflowDiagram
        )
        from ui_ux.agent_verse_ui import InteractionType, LayoutAlgorithm
        
        # Create UI
        ui = AgentVerseUI()
        print(f"[OK] Created AgentVerse UI: {ui.ui_id}")
        
        # Create visualization
        viz_id = ui.create_visualization("Test Viz")
        print(f"[OK] Created visualization: {viz_id}")
        
        # Add agents
        agent1 = ui.add_agent("agent1", "Test Agent 1", "executor")
        agent2 = ui.add_agent("agent2", "Test Agent 2", "analyzer")
        print(f"[OK] Added agents: {agent1.agent_id}, {agent2.agent_id}")
        
        # Track interaction
        edge_id = ui.track_interaction("agent1", "agent2", InteractionType.MESSAGE)
        print(f"[OK] Tracked interaction: {edge_id}")
        
        # Set layout
        ui.set_layout(LayoutAlgorithm.FORCE_DIRECTED)
        print(f"[OK] Set layout algorithm")
        
        # Get status
        status = ui.get_ui_status()
        print(f"[OK] UI status: {status['total_agents']} agents")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] AgentVerse UI test failed: {e}")
        return False

async def test_interactive_dashboard():
    """Test Interactive Dashboard system"""
    print("\nTesting Interactive Dashboard...")
    
    try:
        from ui_ux import (
            InteractiveDashboard,
            DashboardWidget,
            RealtimeChart,
            MetricsPanel,
            ControlPanel
        )
        from ui_ux.interactive_dashboard import ChartType
        
        # Create dashboard
        dashboard = InteractiveDashboard("Test Dashboard")
        print(f"[OK] Created dashboard: {dashboard.dashboard_id}")
        
        # Add chart widget
        chart = RealtimeChart(
            title="Performance",
            chart_type=ChartType.LINE
        )
        chart.add_point(75.5, "cpu")
        chart.add_point(82.3, "cpu")
        widget_id = dashboard.add_widget(chart)
        print(f"[OK] Added chart widget: {widget_id}")
        
        # Add metrics panel
        metrics = MetricsPanel(title="System Metrics")
        metrics.update_metric("Temperature", 65, "°C")
        metrics.update_metric("Requests", 1250, "req/s")
        dashboard.add_widget(metrics)
        print(f"[OK] Added metrics panel")
        
        # Add control panel
        controls = ControlPanel(title="Controls")
        control_id = controls.add_control("Power", "switch")
        controls.trigger_control(control_id, True)
        dashboard.add_widget(controls)
        print(f"[OK] Added control panel")
        
        # Add alert
        alert_id = dashboard.add_alert("warning", "High CPU usage detected")
        print(f"[OK] Added alert: {alert_id}")
        
        # Export layout
        layout = dashboard.export_layout()
        print(f"[OK] Exported layout: {len(layout['widgets'])} widgets")
        
        # Get dashboard data
        data = dashboard.get_dashboard_data()
        widget_count = len(data.get('widgets', {}))
        alert_count = len(data.get('alerts', []))
        print(f"[OK] Dashboard data: {widget_count} widgets, {alert_count} alerts")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Interactive Dashboard test failed: {e}")
        return False

async def main():
    """Main test runner"""
    print("Phase 3 Advanced UI/UX Validation")
    print("=" * 50)
    
    # Test components
    studio_success = test_studio_interface()
    agentverse_success = test_agentverse_ui()
    dashboard_success = await test_interactive_dashboard()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Studio Interface: {'PASS' if studio_success else 'FAIL'}")
    print(f"AgentVerse UI: {'PASS' if agentverse_success else 'FAIL'}")
    print(f"Interactive Dashboard: {'PASS' if dashboard_success else 'FAIL'}")
    
    overall_success = studio_success and agentverse_success and dashboard_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\nPhase 3 UI/UX components are working correctly!")
    else:
        print("\nSome Phase 3 UI/UX components need attention.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)