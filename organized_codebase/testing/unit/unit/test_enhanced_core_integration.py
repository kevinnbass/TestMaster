#!/usr/bin/env python3
"""
Enhanced Core Integration Test
=============================

Comprehensive test to verify all high-value utilities are properly
integrated and working together in the TestMaster core framework.
"""

import time
import sys
from pathlib import Path
from typing import Dict, Any

def test_enhanced_core_integration() -> Dict[str, Any]:
    """Test all enhanced core utilities integration."""
    
    print("=" * 80)
    print("ENHANCED CORE INTEGRATION TEST")
    print("=" * 80)
    
    results = {
        'test_start_time': time.time(),
        'modules_tested': [],
        'successes': 0,
        'failures': 0,
        'capabilities': {},
        'integration_status': 'unknown'
    }
    
    # Test 1: Core framework import and capabilities
    print("\n1. Testing core framework capabilities...")
    try:
        import core
        
        results['capabilities'] = getattr(core, '__enhanced_capabilities__', {})
        version = getattr(core, '__version__', 'unknown')
        
        print(f"   Core framework version: {version}")
        print(f"   Enhanced capabilities: {results['capabilities']}")
        
        if any(results['capabilities'].values()):
            print("   [PASS] Enhanced capabilities detected")
            results['successes'] += 1
        else:
            print("   [WARN] No enhanced capabilities available")
            
        results['modules_tested'].append('core_framework')
        
    except Exception as e:
        print(f"   [FAIL] Core framework import failed: {e}")
        results['failures'] += 1
    
    # Test 2: Emergency Backup & Recovery
    print("\n2. Testing Emergency Backup & Recovery...")
    if results['capabilities'].get('emergency_backup_recovery', False):
        try:
            from core.reliability.emergency_backup_recovery import CoreEmergencyBackupRecovery
            
            # Initialize backup system
            backup_system = CoreEmergencyBackupRecovery()
            
            # Test backup creation
            backup_id = backup_system.create_emergency_backup(
                components=['test_component'],
                backup_type='emergency'
            )
            
            if backup_id:
                print(f"   [PASS] Emergency backup created: {backup_id}")
                
                # Test backup status
                health = backup_system.get_system_health()
                print(f"   [INFO] Backup system health: {health['backup_system']['active']}")
                
                results['successes'] += 1
            else:
                print("   [FAIL] Emergency backup creation failed")
                results['failures'] += 1
                
            backup_system.shutdown()
            results['modules_tested'].append('emergency_backup')
            
        except Exception as e:
            print(f"   [FAIL] Emergency backup test failed: {e}")
            results['failures'] += 1
    else:
        print("   [SKIP] Emergency backup not available")
    
    # Test 3: Advanced Testing Intelligence
    print("\n3. Testing Advanced Testing Intelligence...")
    if results['capabilities'].get('advanced_testing_intelligence', False):
        try:
            from core.testing.advanced_testing_intelligence import AdvancedTestingIntelligence
            
            # Initialize testing intelligence
            testing_ai = AdvancedTestingIntelligence(
                src_dirs=[Path('core')],
                test_dirs=[Path('tests')]
            )
            
            # Test dashboard data
            dashboard_data = testing_ai.get_testing_dashboard_data()
            
            if 'coverage_summary' in dashboard_data:
                print(f"   [PASS] Testing intelligence active")
                print(f"   [INFO] Dashboard data keys: {list(dashboard_data.keys())}")
                results['successes'] += 1
            else:
                print("   [FAIL] Testing intelligence dashboard data invalid")
                results['failures'] += 1
                
            testing_ai.shutdown()
            results['modules_tested'].append('testing_intelligence')
            
        except Exception as e:
            print(f"   [FAIL] Testing intelligence test failed: {e}")
            results['failures'] += 1
    else:
        print("   [SKIP] Testing intelligence not available")
    
    # Test 4: Enhanced Orchestration
    print("\n4. Testing Enhanced Orchestration...")
    if results['capabilities'].get('enhanced_orchestration', False):
        try:
            from core.orchestration.enhanced_agent_orchestrator import (
                EnhancedAgentOrchestrator, SwarmAgent, EnhancedTask, TaskPriority
            )
            
            # Initialize orchestrator
            orchestrator = EnhancedAgentOrchestrator(max_agents=5)
            
            # Register test agent
            test_agent = SwarmAgent(
                agent_id="test_agent_001",
                name="Test Agent",
                capabilities=["test_execution", "analysis"]
            )
            
            registered = orchestrator.register_agent(test_agent)
            if registered:
                print("   [PASS] Agent registration successful")
                
                # Test task submission
                test_task = EnhancedTask(
                    task_type="test_execution",
                    priority=TaskPriority.HIGH,
                    required_capabilities=["test_execution"]
                )
                
                task_id = orchestrator.submit_task(test_task)
                if task_id:
                    print(f"   [PASS] Task submitted: {task_id}")
                    
                    # Get metrics
                    metrics = orchestrator.get_orchestration_metrics()
                    print(f"   [INFO] Agents: {metrics['agents']['total']}, Tasks: {metrics['tasks']['queued']}")
                    
                    results['successes'] += 1
                else:
                    print("   [FAIL] Task submission failed")
                    results['failures'] += 1
            else:
                print("   [FAIL] Agent registration failed")
                results['failures'] += 1
                
            orchestrator.shutdown()
            results['modules_tested'].append('enhanced_orchestration')
            
        except Exception as e:
            print(f"   [FAIL] Enhanced orchestration test failed: {e}")
            results['failures'] += 1
    else:
        print("   [SKIP] Enhanced orchestration not available")
    
    # Test 5: Enhanced State Management
    print("\n5. Testing Enhanced State Management...")
    if results['capabilities'].get('enhanced_state_management', False):
        try:
            from core.state.enhanced_state_manager import (
                EnhancedStateManager, StateScope, StatePersistence, TeamConfiguration, TeamRole
            )
            
            # Initialize state manager
            state_manager = EnhancedStateManager()
            
            # Test state operations
            success = state_manager.set_state(
                key = os.getenv('KEY'),
                value={"test": "data", "timestamp": time.time()},
                scope=StateScope.GLOBAL,
                persistence=StatePersistence.MEMORY_ONLY
            )
            
            if success:
                print("   [PASS] State set operation successful")
                
                # Test state retrieval
                retrieved_value = state_manager.get_state("test_key")
                if retrieved_value and "test" in retrieved_value:
                    print("   [PASS] State retrieval successful")
                    
                    # Test team registration
                    team_config = TeamConfiguration(
                        team_id="test_team_001",
                        name="Test Team",
                        roles=[TeamRole.ENGINEER, TeamRole.QA_AGENT]
                    )
                    
                    team_registered = state_manager.register_team(team_config)
                    if team_registered:
                        print("   [PASS] Team registration successful")
                        
                        # Get system summary
                        summary = state_manager.get_system_state_summary()
                        print(f"   [INFO] State entries: {summary['state_entries']['total']}")
                        print(f"   [INFO] Teams: {summary['teams']['total']}")
                        
                        results['successes'] += 1
                    else:
                        print("   [FAIL] Team registration failed")
                        results['failures'] += 1
                else:
                    print("   [FAIL] State retrieval failed")
                    results['failures'] += 1
            else:
                print("   [FAIL] State set operation failed")
                results['failures'] += 1
                
            state_manager.shutdown()
            results['modules_tested'].append('enhanced_state_management')
            
        except Exception as e:
            print(f"   [FAIL] Enhanced state management test failed: {e}")
            results['failures'] += 1
    else:
        print("   [SKIP] Enhanced state management not available")
    
    # Test 6: Integration between systems
    print("\n6. Testing Cross-System Integration...")
    try:
        # Test if enhanced utilities can work together
        available_utilities = sum(results['capabilities'].values())
        
        if available_utilities >= 2:
            print(f"   [PASS] {available_utilities} utilities available for integration")
            
            # Test framework import with all utilities
            try:
                import core
                all_exports = dir(core)
                enhanced_exports = [
                    export for export in all_exports 
                    if any(keyword in export.lower() for keyword in ['enhanced', 'advanced', 'core'])
                ]
                
                print(f"   [INFO] Enhanced exports: {len(enhanced_exports)}")
                
                if enhanced_exports:
                    print("   [PASS] Cross-system integration successful")
                    results['successes'] += 1
                else:
                    print("   [WARN] Limited cross-system integration")
                    
            except Exception as e:
                print(f"   [FAIL] Cross-system integration test failed: {e}")
                results['failures'] += 1
                
        else:
            print("   [SKIP] Insufficient utilities for integration test")
            
        results['modules_tested'].append('cross_system_integration')
        
    except Exception as e:
        print(f"   [FAIL] Cross-system integration test failed: {e}")
        results['failures'] += 1
    
    # Calculate final results
    results['test_duration'] = time.time() - results['test_start_time']
    total_tests = results['successes'] + results['failures']
    results['success_rate'] = (results['successes'] / total_tests * 100) if total_tests > 0 else 0
    
    # Determine integration status
    if results['success_rate'] >= 80:
        results['integration_status'] = 'excellent'
    elif results['success_rate'] >= 60:
        results['integration_status'] = 'good'
    elif results['success_rate'] >= 40:
        results['integration_status'] = 'fair'
    else:
        results['integration_status'] = 'poor'
    
    # Print final summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Modules tested: {len(results['modules_tested'])}")
    print(f"Tests passed: {results['successes']}")
    print(f"Tests failed: {results['failures']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Integration status: {results['integration_status'].upper()}")
    print(f"Test duration: {results['test_duration']:.2f}s")
    
    # Enhanced capabilities summary
    print(f"\nEnhanced Capabilities Available:")
    for capability, available in results['capabilities'].items():
        status = "[YES]" if available else "[NO]"
        print(f"  {status} {capability}")
    
    if results['integration_status'] in ['excellent', 'good']:
        print(f"\n[OK] SUCCESS: Enhanced core utilities are properly integrated!")
    else:
        print(f"\n[WARN] WARNING: Integration issues detected")
    
    return results

if __name__ == '__main__':
    test_results = test_enhanced_core_integration()
    
    # Exit with appropriate code
    if test_results['integration_status'] in ['excellent', 'good']:
        sys.exit(0)
    else:
        sys.exit(1)