#!/usr/bin/env python3
"""
Agent Gamma Test - Simple Version
================================

Testing Agent Gamma visualization enhancements without unicode issues.
"""

# Import Agent Alpha and Beta systems
try:
    from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer
    from hybrid_dashboard_integration import hybrid_dashboard, quick_hybrid_analysis
    AGENT_INTEGRATION = True
    print("Agent Alpha + Beta integration available")
except ImportError:
    AGENT_INTEGRATION = False
    print("Agent systems not available, using fallback")

class GammaVisualizationEngine:
    def __init__(self):
        self.performance_stats = {
            "visualizations_rendered": 0,
            "graph_interactions": 0,
            "semantic_queries": 0
        }
        
        self.alpha_integration = AGENT_INTEGRATION
        self.beta_integration = AGENT_INTEGRATION
        
        print("Agent Gamma Visualization Engine initialized")
        if AGENT_INTEGRATION:
            print("   Agent Alpha semantic analysis: ACTIVE")
            print("   Agent Beta performance optimization: ACTIVE")
        else:
            print("   Standalone visualization mode")
    
    def get_enhanced_graph_data(self, analysis_mode="auto"):
        if self.alpha_integration and self.beta_integration:
            try:
                result = quick_hybrid_analysis("TestMaster", analysis_mode)
                result["gamma_enhancements"] = {
                    "visualization_layers": ["semantic", "performance", "security", "quality"],
                    "interaction_features": ["filtering", "search", "clustering"]
                }
                self.performance_stats["semantic_queries"] += 1
                return result
            except Exception as e:
                print(f"Agent integration error: {e}")
                return self._fallback_analysis()
        else:
            return self._fallback_analysis()
    
    def _fallback_analysis(self):
        return {
            "analysis_mode": "fallback",
            "total_files": 0,
            "gamma_note": "Visualization-only mode"
        }
    
    def get_gamma_stats(self):
        return {
            **self.performance_stats,
            "agent_coordination": {
                "alpha_integration": self.alpha_integration,
                "beta_integration": self.beta_integration
            }
        }

if __name__ == "__main__":
    print("Testing Agent Gamma Visualization System")
    print("=" * 50)
    
    engine = GammaVisualizationEngine()
    
    # Test enhanced graph data
    print("\nTesting enhanced graph data generation...")
    graph_data = engine.get_enhanced_graph_data("intelligent")
    print(f"   Graph data generated: {len(str(graph_data))} chars")
    print(f"   Agent Alpha integration: {engine.alpha_integration}")
    print(f"   Agent Beta integration: {engine.beta_integration}")
    
    # Test statistics
    print("\nAgent Gamma Statistics:")
    stats = engine.get_gamma_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    print("\nAgent Gamma Visualization System test completed!")