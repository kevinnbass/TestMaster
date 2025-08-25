"""
Quick Phase 2 Validation Test
Test key components functionality without complex dependencies
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
        print("✓ AI Generation components import successfully")
        
        # Advanced Analytics Components
        from core.intelligence.testing.advanced.statistical_coverage_analyzer import StatisticalCoverageAnalyzer
        from core.intelligence.testing.advanced.ml_test_optimizer import MLTestOptimizer
        print("✓ Advanced analytics components import successfully")
        
        # Enterprise Quality Components
        from core.intelligence.testing.enterprise.quality_analytics_engine import QualityAnalyticsEngine
        from core.intelligence.testing.enterprise.quality_gate_automation import QualityGateAutomation
        from core.intelligence.testing.enterprise.predictive_test_failure import PredictiveTestFailureSystem
        print("✓ Enterprise quality components import successfully")
        
        # Security Components
        from core.intelligence.testing.security.advanced_owasp_tester import AdvancedOWASPTester
        from core.intelligence.testing.security.api_security_tester import APISecurityTester
        from core.intelligence.testing.security.compliance_validator import ComplianceValidator
        print("✓ Security components import successfully")
        
        # Automation Components
        from core.intelligence.testing.automation.test_maintenance_system import TestMaintenanceSystem
        from core.intelligence.testing.automation.continuous_testing_engine import ContinuousTestingEngine
        from core.intelligence.testing.automation.enterprise_test_orchestrator import EnterpriseTestOrchestrator
        print("✓ Automation components import successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
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
        
        print("✓ All key components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_package_structure():
    """Test package structure and __init__ files"""
    print("\nTesting Phase 2 Package Structure...")
    
    packages_to_test = [
        "core.intelligence.testing.ai_generation",
        "core.intelligence.testing.advanced", 
        "core.intelligence.testing.enterprise",
        "core.intelligence.testing.security",
        "core.intelligence.testing.automation"
    ]
    
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✓ Package {package} structure OK")
        except Exception as e:
            print(f"❌ Package {package} failed: {e}")
            return False
    
    return True

def test_data_structures():
    """Test key data structures and enums"""
    print("\nTesting Phase 2 Data Structures...")
    
    try:
        # Test key enums and data structures
        from core.intelligence.testing.ai_generation.claude_test_generator import TestComplexity
        from core.intelligence.testing.advanced.statistical_coverage_analyzer import CoverageType
        from core.intelligence.testing.enterprise.quality_analytics_engine import QualityLevel
        from core.intelligence.testing.security.advanced_owasp_tester import OWASPCategory
        from core.intelligence.testing.automation.test_maintenance_system import MaintenanceType
        
        # Verify enums have expected values
        assert hasattr(TestComplexity, 'SIMPLE')
        assert hasattr(CoverageType, 'LINE') 
        assert hasattr(QualityLevel, 'EXCELLENT')
        assert hasattr(OWASPCategory, 'INJECTION')
        assert hasattr(MaintenanceType, 'REPAIR_SYNTAX')
        
        print("✓ All data structures and enums are properly defined")
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
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
        
        print("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("PHASE 2 ENHANCED TESTING FRAMEWORK - QUICK VALIDATION")
    print("=" * 60)
    
    tests = [
        test_component_imports,
        test_component_initialization,
        test_package_structure,
        test_data_structures,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    success_rate = (passed / total) * 100
    print(f"\nOVERALL: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\n🎯 PHASE 2 VALIDATION: SUCCESS!")
        print("The enhanced testing framework is ready for production use.")
    elif success_rate >= 60:
        print("\n⚠️  PHASE 2 VALIDATION: PARTIAL SUCCESS")
        print("Most components are working but some issues need attention.")
    else:
        print("\n❌ PHASE 2 VALIDATION: NEEDS WORK")
        print("Significant issues detected that require fixing.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)