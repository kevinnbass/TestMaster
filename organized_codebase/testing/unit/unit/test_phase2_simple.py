"""
Simple Phase 2 Integration Test
===============================

Basic validation test for Phase 2 multi-agent framework components.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that Phase 2 components can be imported"""
    print("Testing Phase 2 Component Imports...")
    
    try:
        from agents.roles import TestArchitect, TestEngineer
        print("[OK] Multi-agent roles imported")
    except Exception as e:
        print(f"[FAIL] Roles import failed: {e}")
        return False
    
    try:
        from agents.supervisor import TestingSupervisor
        print("[OK] Supervisor imported")
    except Exception as e:
        print(f"[FAIL] Supervisor import failed: {e}")
        return False
    
    try:
        from agents.team import TestingTeam
        print("[OK] Team management imported")
    except Exception as e:
        print(f"[FAIL] Team management import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of Phase 2 components"""
    print("\nTesting Basic Functionality...")
    
    try:
        # Test team creation
        from agents.team import TestingTeam
        team = TestingTeam.create_standard_team()
        print(f"[OK] Standard team created: {team.team_id}")
        
        # Test team status
        status = team.get_team_status()
        print(f"[OK] Team status obtained: {len(status['role_statuses'])} roles")
        
        # Test supervisor
        from agents.supervisor import TestingSupervisor
        supervisor = TestingSupervisor()
        supervisor_status = supervisor.get_supervision_status()
        print(f"[OK] Supervisor created: {supervisor.supervisor_id}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("Phase 2 Integration Validation (Simple)")
    print("=" * 45)
    
    # Test imports
    import_success = test_imports()
    
    # Test functionality if imports succeeded
    func_success = False
    if import_success:
        func_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 45)
    print("Test Summary:")
    print(f"Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"Functionality: {'PASS' if func_success else 'FAIL'}")
    
    overall_success = import_success and func_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\nPhase 2 components are working correctly!")
    else:
        print("\nSome Phase 2 components need attention.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)