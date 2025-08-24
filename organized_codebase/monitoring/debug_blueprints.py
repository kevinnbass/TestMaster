#!/usr/bin/env python3
"""
Debug script to test blueprint imports and route registration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_blueprint_imports():
    """Test all blueprint imports"""
    print("Testing blueprint imports...")
    
    try:
        from api.performance import performance_bp, init_performance_api
        print("[OK] Performance blueprint imported successfully")
        print(f"   Routes: {[rule.rule for rule in performance_bp.url_map.iter_rules()]}")
    except Exception as e:
        print(f"[FAIL] Performance blueprint import failed: {e}")
    
    try:
        from api.analytics import analytics_bp, init_analytics_api
        print("[OK] Analytics blueprint imported successfully")
        print(f"   Routes: {[rule.rule for rule in analytics_bp.url_map.iter_rules()]}")
    except Exception as e:
        print(f"[FAIL] Analytics blueprint import failed: {e}")
    
    try:
        from api.workflow import workflow_bp, init_workflow_api
        print("[OK] Workflow blueprint imported successfully")
        print(f"   Routes: {[rule.rule for rule in workflow_bp.url_map.iter_rules()]}")
    except Exception as e:
        print(f"[FAIL] Workflow blueprint import failed: {e}")
    
    try:
        from api.llm import llm_bp, init_llm_api
        print("[OK] LLM blueprint imported successfully")
        print(f"   Routes: {[rule.rule for rule in llm_bp.url_map.iter_rules()]}")
    except Exception as e:
        print(f"[FAIL] LLM blueprint import failed: {e}")

def test_server_blueprint_registration():
    """Test server blueprint registration"""
    print("\nTesting server blueprint registration...")
    
    try:
        from server import DashboardServer
        server = DashboardServer()
        
        print("[OK] Server created successfully")
        print("Registered routes:")
        for rule in server.app.url_map.iter_rules():
            print(f"   {rule.methods} {rule.rule} -> {rule.endpoint}")
            
    except Exception as e:
        print(f"[FAIL] Server creation failed: {e}")

if __name__ == '__main__':
    test_blueprint_imports()
    test_server_blueprint_registration()