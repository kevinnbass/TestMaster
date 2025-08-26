"""
Quick test to verify Security API server can start
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_security_api_instantiation():
    """Test that Security API can be instantiated"""
    try:
        from core.intelligence.security.security_api import SecurityAPI
        
        # Create API instance
        api = SecurityAPI()
        
        # Get Flask app
        app = api.get_app()
        
        print("[SUCCESS] Security API instantiated successfully!")
        print(f"[INFO] Flask app name: {app.name}")
        print(f"[INFO] Available endpoints:")
        
        # List all routes
        for rule in app.url_map.iter_rules():
            print(f"  - {rule.endpoint}: {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
        
        print("\n[INFO] Security API is ready to serve at:")
        print("  http://localhost:5002/api/security/health")
        print("\nTo start the server, run:")
        print("  python -c \"from core.intelligence.security.security_api import SecurityAPI; SecurityAPI().run()\"")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Could not instantiate Security API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_security_api_instantiation()
    sys.exit(0 if success else 1)