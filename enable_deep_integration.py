#!/usr/bin/env python3
"""
Enable Deep Integration Features

This script activates all the deep integration features that have been
connected in Phase 1A and 1B of the integration roadmap.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from testmaster.core.feature_flags import FeatureFlags
from testmaster.core.shared_state import get_shared_state
from testmaster.core.orchestrator import get_orchestrator


def enable_all_integration_features():
    """Enable all deep integration features."""
    
    print("🚀 Enabling Deep Integration Features...")
    print("="*60)
    
    # Initialize feature flags
    FeatureFlags.initialize()
    
    # Enable Phase 1A features
    print("\n📦 Phase 1A: Core Infrastructure")
    print("-"*40)
    
    # Enable shared state with intelligent cache
    FeatureFlags.enable('layer1_test_foundation', 'shared_state')
    print("✅ Shared state with intelligent cache enabled")
    
    # Enable context preservation
    FeatureFlags.enable('layer1_test_foundation', 'context_preservation')
    print("✅ Context preservation enabled")
    
    # Enable performance monitoring
    FeatureFlags.enable('layer1_test_foundation', 'performance_monitoring')
    print("✅ Performance monitoring enabled")
    
    # Enable advanced config
    FeatureFlags.enable('layer1_test_foundation', 'advanced_config')
    print("✅ Advanced configuration enabled")
    
    # Enable Phase 1B features
    print("\n🔧 Phase 1B: Generator Integration")
    print("-"*40)
    
    # Enable streaming generation
    FeatureFlags.enable('layer1_test_foundation', 'streaming_generation')
    print("✅ Streaming generation enabled")
    
    # Enable agent QA
    FeatureFlags.enable('layer1_test_foundation', 'agent_qa')
    print("✅ Agent Q&A enabled")
    
    # Enable Phase 2A features (partial)
    print("\n🧠 Phase 2A: Intelligence Layer")
    print("-"*40)
    
    # Enable tracking manager
    FeatureFlags.enable('layer2_monitoring', 'tracking_manager')
    print("✅ Tracking manager enabled")
    
    # Enable graph workflows
    FeatureFlags.enable('layer2_monitoring', 'graph_workflows')
    print("✅ Graph workflows enabled")
    
    # Enable Phase 3 features
    print("\n⚡ Phase 3: Flow Optimization")
    print("-"*40)
    
    # Enable flow optimizer
    FeatureFlags.enable('layer3_orchestration', 'flow_optimizer')
    print("✅ Flow optimizer enabled")
    
    # Enable adaptive strategies
    FeatureFlags.enable('layer3_orchestration', 'adaptive_strategies')
    print("✅ Adaptive strategies enabled")
    
    # Test connections
    print("\n🔌 Testing Integrations...")
    print("-"*40)
    
    # Test shared state
    try:
        shared_state = get_shared_state()
        shared_state.set("test_key", "test_value")
        value = shared_state.get("test_key")
        if value == "test_value":
            print("✅ Shared state working")
            
            # Check if intelligent cache is connected
            if hasattr(shared_state, 'intelligent_cache') and shared_state.intelligent_cache:
                print("✅ Intelligent cache connected")
                cache_stats = shared_state.intelligent_cache.get_stats()
                print(f"   Cache: {cache_stats['entries']} entries, {cache_stats['size_mb']:.2f} MB")
            else:
                print("⚠️ Intelligent cache not connected (fallback to memory)")
        else:
            print("❌ Shared state test failed")
    except Exception as e:
        print(f"❌ Shared state error: {e}")
    
    # Test orchestrator
    try:
        orchestrator = get_orchestrator()
        status = orchestrator.get_status()
        print(f"✅ Orchestrator working (DAG tasks: {status['dag_tasks']})")
        if status.get('cache_connected'):
            print("✅ Orchestrator connected to cache")
        if status.get('flow_analyzer_connected'):
            print("✅ Orchestrator connected to flow analyzer")
    except Exception as e:
        print(f"❌ Orchestrator error: {e}")
    
    # Test LLM caching
    try:
        from testmaster.core.shared_state import cache_llm_response, get_cached_llm_response
        
        # Cache a test response
        cache_llm_response(
            prompt="test prompt",
            response="test response",
            model="test"
        )
        
        # Retrieve it
        cached = get_cached_llm_response("test prompt", "test")
        if cached == "test response":
            print("✅ LLM response caching working")
        else:
            print("❌ LLM response caching failed")
    except Exception as e:
        print(f"❌ LLM caching error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Deep Integration Features Enabled!")
    print("="*60)
    
    # Get final statistics
    shared_state = get_shared_state()
    stats = shared_state.get_stats()
    
    print("\n📊 System Status:")
    print(f"  Active workflows: {stats.get('active_workflows', 0)}")
    print(f"  Cache hit rate: {stats.get('cache_stats', {}).get('hit_rate', 0):.1f}%")
    print(f"  Features enabled: {len([f for f in stats.get('feature_flags', {}).values() if f])}")
    
    # Configuration recommendations
    print("\n💡 Next Steps:")
    print("1. Run: python testmaster/core/orchestrator.py")
    print("   To test DAG-based workflow execution")
    print("2. Run: python cache/intelligent_cache.py --test")
    print("   To test intelligent caching")
    print("3. Run: python -m testmaster orchestrate --target . --mode comprehensive")
    print("   To test full orchestration with all features")
    
    return True


def test_integration_workflow():
    """Test a complete integration workflow."""
    print("\n" + "="*60)
    print("🧪 Testing Integration Workflow")
    print("="*60)
    
    from testmaster.core.orchestrator import Orchestrator
    from testmaster.core.shared_state import WorkflowContext
    from pathlib import Path
    import asyncio
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Find a test module
    test_modules = list(Path("testmaster").glob("**/*.py"))[:2]
    if not test_modules:
        print("No test modules found")
        return
    
    print(f"\nTesting with modules: {[m.name for m in test_modules]}")
    
    # Create workflow
    with WorkflowContext("test_integration_001") as context:
        print(f"Created workflow context: {context.workflow_id}")
        
        # Create test generation workflow
        workflow_id = orchestrator.create_test_generation_workflow(test_modules)
        print(f"Created workflow: {workflow_id}")
        
        # Execute workflow (async)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(orchestrator.execute_workflow())
            
            print("\n📈 Workflow Results:")
            metrics = results.get('workflow_metrics', {})
            print(f"  Total tasks: {metrics.get('total_tasks', 0)}")
            print(f"  Completed: {metrics.get('completed', 0)}")
            print(f"  Failed: {metrics.get('failed', 0)}")
            print(f"  Execution time: {metrics.get('execution_time', 0):.2f}s")
            print(f"  Parallel efficiency: {metrics.get('parallel_efficiency', 0):.1%}")
            
            # Check flow analysis
            if 'flow_analysis' in results:
                flow = results['flow_analysis']
                print(f"\n🔍 Flow Analysis:")
                print(f"  Efficiency score: {flow.efficiency_score:.2f}")
                print(f"  Bottlenecks found: {len(flow.bottlenecks)}")
                if flow.optimization_recommendations:
                    print(f"  Recommendations:")
                    for rec in flow.optimization_recommendations[:3]:
                        print(f"    - {rec}")
        
        except Exception as e:
            print(f"Workflow execution error: {e}")
        finally:
            loop.close()
    
    print("\n✅ Integration workflow test completed!")


if __name__ == "__main__":
    # Enable all features
    success = enable_all_integration_features()
    
    # Run integration test if features enabled successfully
    if success:
        response = input("\n🔧 Run integration workflow test? (y/n): ")
        if response.lower() == 'y':
            test_integration_workflow()
    
    print("\n✨ Deep integration setup complete!")