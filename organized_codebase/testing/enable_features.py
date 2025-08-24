#!/usr/bin/env python3
"""
Enable Deep Integration Features - Simple Version
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def enable_integration_features():
    """Enable deep integration features."""
    
    print("Enabling Deep Integration Features...")
    print("="*60)
    
    try:
        from testmaster.core.feature_flags import FeatureFlags
        
        # Initialize feature flags
        FeatureFlags.initialize()
        
        # Enable core features
        print("Phase 1A: Core Infrastructure")
        FeatureFlags.enable('layer1_test_foundation', 'shared_state')
        print("  - Shared state enabled")
        
        FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
        print("  - Context preservation enabled")
        
        FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
        print("  - Performance monitoring enabled")
        
        print("Phase 1B: Generator Integration")
        FeatureFlags.enable('layer1_test_foundation', 'streaming_generation')
        print("  - Streaming generation enabled")
        
        FeatureFlags.enable('layer1_test_foundation', 'agent_qa')
        print("  - Agent Q&A enabled")
        
        print("Phase 3: Flow Optimization")
        FeatureFlags.enable('layer3_orchestration', 'flow_optimizer')
        print("  - Flow optimizer enabled")
        
        print("\nTesting connections...")
        
        # Test shared state
        from testmaster.core.shared_state import get_shared_state
        shared_state = get_shared_state()
        shared_state.set("test_key", "test_value")
        value = shared_state.get("test_key")
        
        if value == "test_value":
            print("  - Shared state working")
            
            # Check intelligent cache
            if hasattr(shared_state, 'intelligent_cache') and shared_state.intelligent_cache:
                print("  - Intelligent cache connected")
            else:
                print("  - Using memory backend")
        
        # Test orchestrator
        from testmaster.core.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        status = orchestrator.get_status()
        print(f"  - Orchestrator working (DAG tasks: {status['dag_tasks']})")
        
        print("\nDEEP INTEGRATION ENABLED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    enable_integration_features()