#!/usr/bin/env python3
"""
Test analytics blueprint import and functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_analytics_import():
    """Test analytics blueprint import"""
    try:
        print("Testing analytics import...")
        from api.analytics import analytics_bp, init_analytics_api, get_current_metrics
        print("[OK] Analytics blueprint imported successfully")
        
        # Test initialization
        init_analytics_api(None)
        print("[OK] Analytics API initialized")
        
        # Test direct function call
        with analytics_bp.test_client() as client:
            print("[WARNING] Cannot test blueprint directly - needs Flask app context")
            
        print(f"Blueprint name: {analytics_bp.name}")
        print(f"Blueprint import name: {analytics_bp.import_name}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Analytics import/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_with_analytics():
    """Test server with analytics explicitly"""
    try:
        from flask import Flask
        from api.analytics import analytics_bp, init_analytics_api
        
        # Create minimal Flask app
        app = Flask(__name__)
        init_analytics_api(None)
        app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
        
        # Test with Flask test client
        with app.test_client() as client:
            print("Testing analytics endpoint with test client...")
            response = client.get('/api/analytics/metrics')
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.get_data(as_text=True)}")
            
        return True
    except Exception as e:
        print(f"[FAIL] Server analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Analytics Blueprint Import Test")
    print("=" * 40)
    
    success1 = test_analytics_import()
    print()
    success2 = test_server_with_analytics()
    
    if success1 and success2:
        print("\n[SUCCESS] All analytics tests passed!")
    else:
        print("\n[FAILED] Some analytics tests failed!")