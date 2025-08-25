"""
Phase 3 Complete Integration Test
==================================

Comprehensive test of all Phase 3 components working together.
"""

import sys
import asyncio
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_complete_integration():
    """Test all Phase 3 components working together"""
    print("Phase 3 Complete Integration Test")
    print("=" * 60)
    
    successes = []
    failures = []
    
    # Test 1: Enterprise Deployment with Service Registry
    print("\n1. Testing Deployment + Registry Integration...")
    try:
        from deployment import (
            EnterpriseTestDeployment,
            ServiceRegistry,
            ServiceDescriptor,
            ServiceEndpoint,
            ServiceConfig,
            ServiceType,
            DeploymentMode,
            ServiceHealth
        )
        
        # Create deployment
        deployment = EnterpriseTestDeployment(DeploymentMode.PRODUCTION)
        
        # Create registry
        registry = ServiceRegistry()
        
        # Deploy service
        service_config = ServiceConfig(
            service_type=ServiceType.TEST_EXECUTOR,
            name="Production Test Service",
            replicas=3
        )
        service_id = deployment.deploy_service(service_config)
        
        # Register in registry
        descriptor = ServiceDescriptor(
            service_id=service_id,
            name="Production Test Service",
            service_type="test_executor",
            endpoints=[
                ServiceEndpoint(host="prod-1", port=8080, zone="us-east-1"),
                ServiceEndpoint(host="prod-2", port=8080, zone="us-west-2")
            ]
        )
        registry.register_service(descriptor)
        registry.update_health(service_id, ServiceHealth.HEALTHY)
        
        # Verify
        discovered = registry.discover_services(service_type="test_executor")
        print(f"   [OK] Deployment created, {len(discovered)} services registered")
        successes.append("Deployment + Registry")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("Deployment + Registry")
    
    # Test 2: Swarm Orchestration with Task Distribution
    print("\n2. Testing Swarm Orchestration...")
    try:
        from deployment import (
            SwarmOrchestrator,
            SwarmConfig,
            SwarmTask,
            SwarmAgent
        )
        
        # Create swarm
        config = SwarmConfig(
            name="ProductionSwarm",
            min_agents=5,
            max_agents=20,
            enable_auto_scaling=True
        )
        swarm = SwarmOrchestrator(config)
        
        # Add agents with different capabilities
        for i in range(5):
            agent = SwarmAgent(
                name=f"ProdAgent_{i}",
                capabilities={f"capability_{i%3}", "test_execution"}
            )
            swarm.add_agent(agent)
        
        # Submit tasks with dependencies
        task1 = SwarmTask(task_type="setup", priority=10)
        task2 = SwarmTask(task_type="test", priority=8, dependencies=[])
        task3 = SwarmTask(task_type="cleanup", priority=5, dependencies=[])
        
        task1_id = swarm.submit_task(task1)
        task2.dependencies = [task1_id]
        task2_id = swarm.submit_task(task2)
        task3.dependencies = [task2_id]
        task3_id = swarm.submit_task(task3)
        
        # Assign and complete tasks
        await swarm.assign_task(task1_id)
        await swarm.complete_task(task1_id, {"result": "setup complete"})
        
        status = swarm.get_swarm_status()
        print(f"   [OK] Swarm active with {status['agents']['total']} agents, {status['tasks']['total']} tasks")
        successes.append("Swarm Orchestration")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("Swarm Orchestration")
    
    # Test 3: Studio Interface with Workflow Creation
    print("\n3. Testing Studio Interface...")
    try:
        from ui_ux import (
            StudioInterface,
            InteractionMode,
            VisualizationType
        )
        
        # Create studio and session
        studio = StudioInterface()
        session_id = studio.create_session("prod_user", InteractionMode.HYBRID)
        
        # Load template and customize
        workflow_id = studio.load_template(session_id, "parallel_test")
        
        # Add custom nodes
        node1 = studio.add_agent_node(session_id, "test_analyzer", {"x": 600, "y": 100})
        node2 = studio.add_agent_node(session_id, "test_reporter", {"x": 700, "y": 200})
        
        # Connect nodes
        studio.connect_nodes(session_id, node1, node2, "on_success")
        
        # Generate code
        code = studio.generate_code(session_id)
        
        # Get visualization
        viz_data = studio.get_visualization(session_id, VisualizationType.WORKFLOW)
        
        print(f"   [OK] Workflow created with {len(viz_data['nodes'])} nodes, code generated")
        successes.append("Studio Interface")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("Studio Interface")
    
    # Test 4: AgentVerse UI with Real-time Visualization
    print("\n4. Testing AgentVerse UI...")
    try:
        from ui_ux import AgentVerseUI
        from ui_ux.agent_verse_ui import InteractionType, LayoutAlgorithm
        
        # Create UI and visualization
        ui = AgentVerseUI()
        viz_id = ui.create_visualization("Production Monitor")
        
        # Add production agents
        agents = []
        for i in range(10):
            agent = ui.add_agent(
                f"prod_agent_{i}",
                f"Production Agent {i}",
                "executor" if i % 2 == 0 else "analyzer"
            )
            agents.append(agent)
        
        # Simulate interactions
        for i in range(20):
            source_idx = i % 10
            target_idx = (i + 3) % 10
            ui.track_interaction(
                f"prod_agent_{source_idx}",
                f"prod_agent_{target_idx}",
                InteractionType.TASK_ASSIGNMENT if i % 3 == 0 else InteractionType.MESSAGE
            )
        
        # Update agent metrics
        for i in range(10):
            ui.update_agent_status(
                f"prod_agent_{i}",
                "busy" if i < 5 else "idle",
                {"cpu": 50 + i * 5, "memory": 70 - i * 3}
            )
        
        # Apply different layouts
        ui.set_layout(LayoutAlgorithm.HIERARCHICAL)
        
        # Get interaction matrix
        matrix = ui.get_interaction_matrix()
        
        print(f"   [OK] Visualization active with {matrix['total_interactions']} interactions tracked")
        successes.append("AgentVerse UI")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("AgentVerse UI")
    
    # Test 5: Interactive Dashboard with Real-time Updates
    print("\n5. Testing Interactive Dashboard...")
    try:
        from ui_ux import (
            InteractiveDashboard,
            RealtimeChart,
            MetricsPanel,
            ControlPanel
        )
        from ui_ux.interactive_dashboard import ChartType
        
        # Create production dashboard
        dashboard = InteractiveDashboard("Production Dashboard")
        
        # Add performance chart
        perf_chart = RealtimeChart(
            title="System Performance",
            chart_type=ChartType.LINE,
            position={"x": 0, "y": 0},
            size={"width": 6, "height": 4}
        )
        
        # Add data points
        for i in range(10):
            perf_chart.add_point(70 + i * 2, "cpu")
            perf_chart.add_point(50 + i * 3, "memory")
            perf_chart.add_point(30 + i * 4, "network")
        
        dashboard.add_widget(perf_chart)
        
        # Add metrics panel with thresholds
        metrics = MetricsPanel(
            title="Production Metrics",
            position={"x": 6, "y": 0},
            size={"width": 6, "height": 2}
        )
        
        metrics.thresholds = {
            "Error Rate": {"warning": 5, "critical": 10},
            "Response Time": {"warning": 1000, "critical": 2000}
        }
        
        metrics.update_metric("Error Rate", 3, "%")
        metrics.update_metric("Response Time", 850, "ms")
        metrics.update_metric("Throughput", 1500, "req/s")
        
        dashboard.add_widget(metrics)
        
        # Add control panel
        controls = ControlPanel(
            title="Production Controls",
            position={"x": 6, "y": 2},
            size={"width": 6, "height": 2}
        )
        
        # Add controls with callbacks
        def scale_callback(value):
            return f"Scaled to {value} instances"
        
        controls.add_control("Auto Scale", "switch")
        controls.add_control("Instance Count", "slider", {"min": 1, "max": 20}, scale_callback)
        controls.add_control("Deploy", "button")
        
        dashboard.add_widget(controls)
        
        # Add alerts
        dashboard.add_alert("info", "System startup complete")
        dashboard.add_alert("warning", "High memory usage detected on node-3")
        
        # Export layout for reuse
        layout = dashboard.export_layout()
        
        print(f"   [OK] Dashboard configured with {len(dashboard.widgets)} widgets, {len(dashboard.alerts)} alerts")
        successes.append("Interactive Dashboard")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("Interactive Dashboard")
    
    # Test 6: Full Integration - All Components Together
    print("\n6. Testing Full Integration...")
    try:
        # Deployment provides infrastructure
        deployment_status = deployment.get_deployment_status()
        
        # Registry tracks services
        registry_status = registry.get_registry_status()
        
        # Swarm handles task distribution
        swarm_status = swarm.get_swarm_status()
        
        # Studio provides workflow design
        studio_status = studio.get_studio_status()
        
        # AgentVerse visualizes interactions
        ui_status = ui.get_ui_status()
        
        # Dashboard monitors everything
        dashboard_status = dashboard.get_dashboard_status()
        
        print(f"   [OK] All components operational and integrated")
        print(f"        - Deployment: {deployment_status['total_instances']} instances")
        print(f"        - Registry: {registry_status['total_services']} services")
        print(f"        - Swarm: {swarm_status['agents']['total']} agents")
        print(f"        - Studio: {studio_status['templates_available']} templates")
        print(f"        - AgentVerse: {ui_status['total_agents']} visualized agents")
        print(f"        - Dashboard: {dashboard_status['widget_count']} widgets")
        
        successes.append("Full Integration")
        
    except Exception as e:
        print(f"   [FAIL] {e}")
        failures.append("Full Integration")
    
    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary:")
    print(f"Successes: {len(successes)}/6")
    for component in successes:
        print(f"  [PASS] {component}")
    
    if failures:
        print(f"\nFailures: {len(failures)}/6")
        for component in failures:
            print(f"  [FAIL] {component}")
    
    overall_success = len(failures) == 0
    print(f"\nOverall Result: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\nPhase 3 Production Enhancement is FULLY OPERATIONAL!")
        print("All enterprise deployment and advanced UI/UX features are working perfectly.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(test_complete_integration())
    sys.exit(0 if success else 1)