"""
Simple Phase 2 Validation Test
Test key components functionality without Unicode characters
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_component_imports():
    """Test that all Phase 2 components can be imported"""
    print("Testing Phase 2 Component Imports...")
    
    try:
        # AI Generation Components
        from core.intelligence.testing.ai_generation.claude_test_generator import ClaudeTestGenerator
        from core.intelligence.testing.ai_generation.gemini_test_generator import GeminiTestGenerator  
        from core.intelligence.testing.ai_generation.universal_ai_generator import UniversalAIGenerator
        print("OK - AI Generation components import successfully")
        
        # Advanced Analytics Components
        from core.intelligence.testing.advanced.statistical_coverage_analyzer import StatisticalCoverageAnalyzer
        from core.intelligence.testing.advanced.ml_test_optimizer import MLTestOptimizer
        print("OK - Advanced analytics components import successfully")
        
        # Enterprise Quality Components
        from core.intelligence.testing.enterprise.quality_analytics_engine import QualityAnalyticsEngine
        from core.intelligence.testing.enterprise.quality_gate_automation import QualityGateAutomation
        from core.intelligence.testing.enterprise.predictive_test_failure import PredictiveTestFailureSystem
        print("OK - Enterprise quality components import successfully")
        
        # Security Components
        from core.intelligence.testing.security.advanced_owasp_tester import AdvancedOWASPTester
        from core.intelligence.testing.security.api_security_tester import APISecurityTester
        from core.intelligence.testing.security.compliance_validator import ComplianceValidator
        print("OK - Security components import successfully")
        
        # Automation Components
        from core.intelligence.testing.automation.test_maintenance_system import TestMaintenanceSystem
        from core.intelligence.testing.automation.continuous_testing_engine import ContinuousTestingEngine
        from core.intelligence.testing.automation.enterprise_test_orchestrator import EnterpriseTestOrchestrator
        print("OK - Automation components import successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL - Import failed: {e}")
        traceback.print_exc()
        return False

def test_component_initialization():
    """Test that components can be initialized"""
    print("\nTesting Phase 2 Component Initialization...")
    
    try:
        # Test key component initialization
        from core.intelligence.testing.ai_generation.claude_test_generator import ClaudeTestGenerator
        from core.intelligence.testing.advanced.statistical_coverage_analyzer import StatisticalCoverageAnalyzer
        from core.intelligence.testing.enterprise.quality_analytics_engine import QualityAnalyticsEngine
        from core.intelligence.testing.security.advanced_owasp_tester import AdvancedOWASPTester
        from core.intelligence.testing.automation.test_maintenance_system import TestMaintenanceSystem
        
        # Initialize components
        claude_gen = ClaudeTestGenerator()
        coverage_analyzer = StatisticalCoverageAnalyzer()
        quality_engine = QualityAnalyticsEngine()
        owasp_tester = AdvancedOWASPTester("http://localhost:8080")
        maintenance_system = TestMaintenanceSystem("./tests")
        
        print("OK - All key components initialized successfully")
        return True
        
    except Exception as e:
        print(f"FAIL - Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting Phase 2 Basic Functionality...")
    
    try:
        # Test coverage analyzer basic functionality
        from core.intelligence.testing.advanced.statistical_coverage_analyzer import StatisticalCoverageAnalyzer
        
        analyzer = StatisticalCoverageAnalyzer()
        assert analyzer.confidence_level == 0.95
        
        # Test quality engine basic functionality
        from core.intelligence.testing.enterprise.quality_analytics_engine import QualityAnalyticsEngine
        
        engine = QualityAnalyticsEngine()
        assert len(engine.metrics_definitions) > 0
        
        # Test basic quality analysis
        mock_results = {
            'total_tests': 100,
            'passed_tests': 95,
            'failed_tests': 5,
            'coverage_percentage': 87.5,
            'execution_time': 45.2
        }
        
        result = engine.analyze_quality("test_project", mock_results)
        assert result is not None
        assert result.overall_score > 0
        assert len(result.metrics) > 0
        
        print("OK - Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"FAIL - Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("PHASE 2 ENHANCED TESTING FRAMEWORK - VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_component_imports),
        ("Component Initialization", test_component_initialization),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for name, test in tests:
        try:
            result = test()
            results.append((name, result))
            print(f"Test '{name}': {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"Test '{name}': CRASHED - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for i, (name, result) in enumerate(results):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {name}: {status}")
    
    success_rate = (passed / total) * 100
    print(f"\nOVERALL: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\nPHASE 2 VALIDATION: SUCCESS!")
        print("The enhanced testing framework is ready for production use.")
    elif success_rate >= 60:
        print("\nPHASE 2 VALIDATION: PARTIAL SUCCESS")
        print("Most components are working but some issues need attention.")
    else:
        print("\nPHASE 2 VALIDATION: NEEDS WORK")
        print("Significant issues detected that require fixing.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)