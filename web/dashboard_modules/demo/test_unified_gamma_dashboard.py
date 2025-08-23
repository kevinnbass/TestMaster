#!/usr/bin/env python3
"""
Test Runner for Unified Gamma Dashboard
=======================================

Validates the unified dashboard implementation without starting the full server.
Tests component initialization, data integration, and API endpoints.

Author: Agent Gamma (Greek Swarm)
Created: 2025-08-23 14:00:00
"""

import sys
import os
from pathlib import Path
import json

# Add web directory to path
sys.path.insert(0, str(Path(__file__).parent / "web"))

def test_unified_dashboard_import():
    """Test that the unified dashboard can be imported successfully."""
    try:
        from unified_gamma_dashboard import (
            UnifiedDashboardEngine, 
            APIUsageTracker, 
            AgentCoordinator,
            DataIntegrator,
            PerformanceMonitor
        )
        print("✅ Successfully imported UnifiedDashboardEngine and components")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_component_initialization():
    """Test that all components can be initialized."""
    try:
        from unified_gamma_dashboard import (
            APIUsageTracker, 
            AgentCoordinator,
            DataIntegrator,
            PerformanceMonitor
        )
        
        # Test individual component initialization
        api_tracker = APIUsageTracker()
        print("✅ APIUsageTracker initialized")
        
        agent_coordinator = AgentCoordinator()
        print("✅ AgentCoordinator initialized")
        
        data_integrator = DataIntegrator()
        print("✅ DataIntegrator initialized")
        
        performance_monitor = PerformanceMonitor()
        print("✅ PerformanceMonitor initialized")
        
        return True
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

def test_data_methods():
    """Test that data methods work correctly."""
    try:
        from unified_gamma_dashboard import (
            APIUsageTracker, 
            AgentCoordinator,
            DataIntegrator,
            PerformanceMonitor
        )
        
        # Test API tracker
        api_tracker = APIUsageTracker()
        api_tracker.track_request("test_service", "test_endpoint")
        stats = api_tracker.get_usage_stats()
        assert "timestamp" in stats
        assert "total_requests" in stats
        print("✅ APIUsageTracker methods work")
        
        # Test agent coordinator
        agent_coordinator = AgentCoordinator()
        status = agent_coordinator.get_coordination_status()
        assert "timestamp" in status
        assert "agents" in status
        print("✅ AgentCoordinator methods work")
        
        # Test data integrator
        data_integrator = DataIntegrator()
        unified_data = data_integrator.get_unified_data()
        assert "timestamp" in unified_data
        assert "system_health" in unified_data
        print("✅ DataIntegrator methods work")
        
        # Test performance monitor
        performance_monitor = PerformanceMonitor()
        metrics = performance_monitor.get_metrics()
        assert "timestamp" in metrics
        assert "cpu_usage" in metrics
        print("✅ PerformanceMonitor methods work")
        
        return True
    except Exception as e:
        print(f"❌ Data method testing failed: {e}")
        return False

def test_dashboard_engine_init():
    """Test that the main dashboard engine can be initialized."""
    try:
        from unified_gamma_dashboard import UnifiedDashboardEngine
        
        # Initialize dashboard engine (without starting server)
        dashboard = UnifiedDashboardEngine(port=5999)  # Use test port
        print("✅ UnifiedDashboardEngine initialized successfully")
        
        # Test that subsystems are initialized
        assert dashboard.api_tracker is not None
        assert dashboard.agent_coordinator is not None  
        assert dashboard.data_integrator is not None
        assert dashboard.performance_monitor is not None
        print("✅ All subsystems initialized")
        
        # Test backend service configuration
        assert len(dashboard.backend_services) == 5
        assert 'port_5000' in dashboard.backend_services
        assert 'port_5002' in dashboard.backend_services
        print("✅ Backend service configuration correct")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard engine initialization failed: {e}")
        return False

def test_html_template():
    """Test that the HTML template is properly defined."""
    try:
        from unified_gamma_dashboard import UNIFIED_GAMMA_DASHBOARD_HTML
        
        # Check that template exists and has key components
        assert len(UNIFIED_GAMMA_DASHBOARD_HTML) > 1000  # Substantial template
        assert "Unified Gamma Dashboard" in UNIFIED_GAMMA_DASHBOARD_HTML
        assert "three.js" in UNIFIED_GAMMA_DASHBOARD_HTML.lower()
        assert "socket.io" in UNIFIED_GAMMA_DASHBOARD_HTML.lower()
        assert "chart.js" in UNIFIED_GAMMA_DASHBOARD_HTML.lower()
        print("✅ HTML template is complete and includes required libraries")
        
        return True
    except Exception as e:
        print(f"❌ HTML template testing failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("UNIFIED GAMMA DASHBOARD - TEST SUITE")
    print("=" * 60)
    print("Testing dashboard implementation and components...")
    print()
    
    tests = [
        ("Import Test", test_unified_dashboard_import),
        ("Component Initialization", test_component_initialization), 
        ("Data Methods", test_data_methods),
        ("Dashboard Engine", test_dashboard_engine_init),
        ("HTML Template", test_html_template)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("=" * 60)
    print("TEST SUMMARY")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print()
    
    if passed == total:
        print("ALL TESTS PASSED! Unified Gamma Dashboard is ready for deployment.")
        print()
        print("Next Steps:")
        print("1. Start the dashboard: python web/unified_gamma_dashboard.py")
        print("2. Access at: http://localhost:5015")
        print("3. Test real-time features with backend services")
    else:
        print("Some tests failed. Review implementation before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)