#!/usr/bin/env python3
"""
Quick test of enhanced endpoint
"""

try:
    from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer
    print("Testing enhanced analyzer...")
    
    analyzer = EnhancedLinkageAnalyzer()
    print("Analyzer created")
    
    # Test with a very limited analysis first
    results = analyzer.analyze_codebase("TestMaster", max_files=5)
    print(f"Analysis completed. Keys: {list(results.keys())}")
    
    if "semantic_dimensions" in results:
        print(f"Semantic analysis found: {len(results['semantic_dimensions'].get('intent_classifications', {}))} classifications")
    
    if "multi_layer_graph" in results:
        print(f"Multi-layer graph has: {len(results['multi_layer_graph'].get('nodes', []))} nodes")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()