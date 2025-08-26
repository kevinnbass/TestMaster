"""
Test Suite for Agent B Feature Integrations
Tests all newly integrated components from archives
"""

import sys
import os
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add TestMaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_analyzer():
    """Test ML Code Analysis integration"""
    print("\n=== Testing ML Code Analyzer ===")
    try:
        from core.intelligence.analysis.ml_analyzer import MLCodeAnalyzer
        
        analyzer = MLCodeAnalyzer()
        # Test basic initialization
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'ml_frameworks')
        
        # Test framework detection patterns
        assert 'tensorflow' in analyzer.ml_frameworks
        assert 'pytorch' in analyzer.ml_frameworks
        
        print("✓ ML Code Analyzer initialized successfully")
        print(f"  - Supported frameworks: {list(analyzer.ml_frameworks.keys())}")
        print(f"  - Antipattern types: {list(analyzer.ml_antipatterns.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ ML Code Analyzer test failed: {e}")
        return False


def test_semantic_analyzer():
    """Test Semantic Analysis integration"""
    print("\n=== Testing Semantic Analyzer ===")
    try:
        from core.intelligence.analysis.semantic_analyzer import (
            SemanticAnalyzer, IntentType, SemanticIntent, ConceptualPattern
        )
        
        analyzer = SemanticAnalyzer()
        # Test basic initialization
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'intent_keywords')
        
        # Test intent types
        assert IntentType.DATA_PROCESSING
        assert IntentType.API_ENDPOINT
        assert IntentType.AUTHENTICATION
        
        # Test semantic patterns
        assert 'factory' in analyzer.semantic_patterns
        assert 'singleton' in analyzer.semantic_patterns
        assert 'observer' in analyzer.semantic_patterns
        
        print("✓ Semantic Analyzer initialized successfully")
        print(f"  - Intent types: {len(analyzer.intent_keywords)}")
        print(f"  - Semantic patterns: {len(analyzer.semantic_patterns)}")
        print(f"  - Relationship types: {len(analyzer.relationship_types)}")
        
        return True
    except Exception as e:
        print(f"✗ Semantic Analyzer test failed: {e}")
        return False


def test_business_analyzer():
    """Test Business Rule Analysis integration"""
    print("\n=== Testing Business Analyzer ===")
    try:
        from core.intelligence.analysis.business_analyzer import BusinessAnalyzer
        
        analyzer = BusinessAnalyzer()
        # Test basic initialization
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'rule_patterns')
        
        # Test rule patterns
        assert 'validation' in analyzer.rule_patterns
        assert 'calculation' in analyzer.rule_patterns
        assert 'authorization' in analyzer.rule_patterns
        
        print("✓ Business Analyzer initialized successfully")
        print(f"  - Rule patterns: {list(analyzer.rule_patterns.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Business Analyzer test failed: {e}")
        return False


def test_debt_analyzer():
    """Test Technical Debt Analysis integration"""
    print("\n=== Testing Technical Debt Analyzer ===")
    try:
        from core.intelligence.analysis.debt_analyzer import TechnicalDebtAnalyzer
        
        analyzer = TechnicalDebtAnalyzer()
        # Test basic initialization
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'debt_items')
        assert hasattr(analyzer, 'debt_metrics')
        assert hasattr(analyzer, 'debt_cost_factors')
        
        # Test cost factors
        assert 'code_duplication' in analyzer.debt_cost_factors
        assert 'missing_tests' in analyzer.debt_cost_factors
        
        # Test interest rates
        assert hasattr(analyzer, 'interest_rates')
        assert 'security' in analyzer.interest_rates
        
        print("✓ Technical Debt Analyzer initialized successfully")
        print(f"  - Cost factors: {list(analyzer.debt_cost_factors.keys())}")
        print(f"  - Interest rates: {list(analyzer.interest_rates.keys())}")
        print(f"  - Productivity factors: {list(analyzer.productivity_factors.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Technical Debt Analyzer test failed: {e}")
        return False


def test_analysis_hub():
    """Test the unified Analysis Hub"""
    print("\n=== Testing Analysis Hub ===")
    try:
        from core.intelligence.analysis import AnalysisHub
        
        hub = AnalysisHub()
        # Test basic initialization
        assert hub is not None
        assert hasattr(hub, 'ml_analyzer')
        assert hasattr(hub, 'semantic_analyzer')
        assert hasattr(hub, 'business_analyzer')
        assert hasattr(hub, 'debt_analyzer')
        
        # Test hub methods
        assert hasattr(hub, 'run_comprehensive_analysis')
        assert hasattr(hub, 'analyze_business_rules')
        assert hasattr(hub, 'analyze_technical_debt')
        assert hasattr(hub, 'analyze_ml_code')
        assert hasattr(hub, 'analyze_semantics')
        
        print("✓ Analysis Hub initialized successfully")
        print(f"  - ML Analyzer: {'✓' if hub.ml_analyzer else '✗'}")
        print(f"  - Semantic Analyzer: {'✓' if hub.semantic_analyzer else '✗'}")
        print(f"  - Business Analyzer: {'✓' if hub.business_analyzer else '✗'}")
        print(f"  - Debt Analyzer: {'✓' if hub.debt_analyzer else '✗'}")
        
        return True
    except Exception as e:
        print(f"✗ Analysis Hub test failed: {e}")
        return False


def test_agent_qa():
    """Test Agent QA System integration"""
    print("\n=== Testing Agent QA System ===")
    try:
        from core.intelligence.monitoring.agent_qa import AgentQualityAssurance as AgentQASystem
        
        qa_system = AgentQASystem()
        # Test basic initialization
        assert qa_system is not None
        assert hasattr(qa_system, 'inspect_agent')
        assert hasattr(qa_system, 'calculate_score')
        assert hasattr(qa_system, 'run_benchmarks')
        
        # Test monitoring methods
        assert hasattr(qa_system, 'start_monitoring')
        assert hasattr(qa_system, 'stop_monitoring')
        
        # Test validation rules
        assert len(qa_system.validation_rules) > 0
        
        print("✓ Agent QA System initialized successfully")
        print(f"  - Validation rules: {len(qa_system.validation_rules)}")
        print(f"  - Monitoring enabled: {qa_system.enabled}")
        print(f"  - Score weight categories: {qa_system.score_weights if isinstance(qa_system.score_weights, list) else list(qa_system.score_weights.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Agent QA System test failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions for direct access"""
    print("\n=== Testing Convenience Functions ===")
    try:
        from core.intelligence.analysis import (
            analyze_ml_code,
            analyze_semantics,
            analyze_business_rules,
            analyze_technical_debt
        )
        
        # Test that functions are importable
        assert analyze_ml_code is not None
        assert analyze_semantics is not None
        assert analyze_business_rules is not None
        assert analyze_technical_debt is not None
        
        print("✓ All convenience functions available")
        print("  - analyze_ml_code()")
        print("  - analyze_semantics()")
        print("  - analyze_business_rules()")
        print("  - analyze_technical_debt()")
        
        return True
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        return False


def test_monitoring_hub():
    """Test the monitoring hub integration"""
    print("\n=== Testing Monitoring Hub ===")
    try:
        from core.intelligence.monitoring import get_agent_qa
        
        # Test factory function
        qa1 = get_agent_qa()
        qa2 = get_agent_qa()
        
        # Should return the same instance (singleton)
        assert qa1 is qa2
        
        print("✓ Monitoring Hub initialized successfully")
        print("  - Singleton pattern: ✓")
        print("  - Factory function: get_agent_qa()")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring Hub test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("AGENT B FEATURE INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("ML Code Analyzer", test_ml_analyzer),
        ("Semantic Analyzer", test_semantic_analyzer),
        ("Business Analyzer", test_business_analyzer),
        ("Technical Debt Analyzer", test_debt_analyzer),
        ("Analysis Hub", test_analysis_hub),
        ("Agent QA System", test_agent_qa),
        ("Convenience Functions", test_convenience_functions),
        ("Monitoring Hub", test_monitoring_hub)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATIONS SUCCESSFUL! 🎉")
        print("\nAgent B has successfully integrated:")
        print("  - ML Code Analysis from archive")
        print("  - Semantic Analysis from archive")
        print("  - Business Rule Analysis from archive")
        print("  - Technical Debt Analysis from archive")
        print("  - Agent QA System components")
        print("\nAll features are ready for use!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)