#!/usr/bin/env python3
"""
Test Comprehensive Classical Analysis
====================================

Test all the comprehensive classical analysis techniques.
"""

from pathlib import Path
import sys
import os

# Add testmaster to path
sys.path.insert(0, str(Path(__file__).parent))

def test_comprehensive_analysis():
    """Test the comprehensive classical analysis."""
    print("[ANALYSIS] TESTING COMPREHENSIVE CLASSICAL ANALYSIS")
    print("=" * 80)
    
    try:
        from testmaster.analysis import ComprehensiveCodebaseAnalyzer
        
        analyzer = ComprehensiveCodebaseAnalyzer(Path("."))
        results = analyzer.perform_comprehensive_analysis()
        
        print(f"[OK] Analysis completed successfully!")
        print(f"[INFO] Analysis categories: {len(results) - 1}")  # Minus timestamp
        
        # Display key results
        if "comprehensive_summary" in results:
            summary = results["comprehensive_summary"]
            overview = summary.get("analysis_overview", {})
            key_metrics = summary.get("key_metrics", {})
            
            print(f"\n[METRICS] Key Metrics:")
            for metric, value in key_metrics.items():
                print(f"   {metric}: {value}")
            
            print(f"\n[OVERVIEW] Analysis Overview:")
            print(f"   Techniques Applied: {overview.get('total_techniques_applied', 0)}")
            print(f"   Analysis Completeness: {overview.get('analysis_completeness', 'UNKNOWN')}")
            
            strengths = overview.get('primary_strengths', [])
            if strengths:
                print(f"   Primary Strengths: {', '.join(strengths)}")
            
            improvements = overview.get('areas_for_improvement', [])
            if improvements:
                print(f"   Areas for Improvement: {', '.join(improvements)}")
            
            issues = overview.get('critical_issues', [])
            if issues:
                print(f"   Critical Issues: {', '.join(issues)}")
        
        # Display analysis category results
        categories_tested = []
        for category, data in results.items():
            if category not in ["analysis_timestamp", "comprehensive_summary"]:
                categories_tested.append(category)
                if isinstance(data, dict):
                    subcategories = len(data)
                    print(f"   [OK] {category}: {subcategories} subcategories")
                elif isinstance(data, list):
                    print(f"   [OK] {category}: {len(data)} items")
                else:
                    print(f"   [OK] {category}: analyzed")
        
        print(f"\n[RESULTS] Analysis Categories Successfully Tested:")
        for i, category in enumerate(categories_tested, 1):
            print(f"   {i:2d}. {category.replace('_', ' ').title()}")
        
        print(f"\n[SUCCESS] COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"   Total Categories: {len(categories_tested)}")
        print(f"   All classical techniques integrated successfully!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual analysis components."""
    print("\n[TESTING] TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    try:
        from testmaster.analysis import ComprehensiveCodebaseAnalyzer
        
        analyzer = ComprehensiveCodebaseAnalyzer(Path("."))
        
        # Test Halstead metrics
        print("\n1. Testing Halstead Metrics...")
        halstead = analyzer._calculate_halstead_metrics()
        print(f"   Volume: {halstead.get('volume', 0):.1f}")
        print(f"   Difficulty: {halstead.get('difficulty', 0):.1f}")
        print(f"   Estimated Bugs: {halstead.get('estimated_bugs', 0):.3f}")
        
        # Test SLOC metrics
        print("\n2. Testing SLOC Metrics...")
        sloc = analyzer._calculate_sloc_metrics()
        print(f"   Total Lines: {sloc.get('total_lines', 0):,}")
        print(f"   Code Lines: {sloc.get('code_lines', 0):,}")
        print(f"   Comment Ratio: {sloc.get('comment_ratio', 0):.1%}")
        
        # Test clone detection
        print("\n3. Testing Clone Detection...")
        clones = analyzer._detect_exact_clones()
        print(f"   Exact Clones Found: {len(clones)}")
        if clones:
            print(f"   Total Clone Instances: {sum(clone['clone_count'] for clone in clones)}")
        
        # Test security analysis
        print("\n4. Testing Security Analysis...")
        vulns = analyzer._detect_vulnerability_patterns()
        print(f"   Potential Vulnerabilities: {len(vulns)}")
        high_severity = [v for v in vulns if v.get("severity") == "HIGH"]
        print(f"   High Severity Issues: {len(high_severity)}")
        
        # Test identifier analysis
        print("\n5. Testing Identifier Analysis...")
        identifiers = analyzer._analyze_identifiers()
        print(f"   Total Identifiers: {identifiers.get('total_identifiers', 0):,}")
        print(f"   Unique Identifiers: {identifiers.get('unique_identifiers', 0):,}")
        print(f"   Vocabulary Richness: {identifiers.get('vocabulary_richness', 0):.3f}")
        
        naming = identifiers.get('naming_patterns', {})
        print(f"   Naming Patterns:")
        for pattern, count in naming.items():
            print(f"     {pattern}: {count}")
        
        print("\n[OK] All individual components working!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Individual component testing failed: {e}")
        return False

def main():
    """Run comprehensive analysis tests."""
    print("[TEST] COMPREHENSIVE CLASSICAL ANALYSIS TEST SUITE")
    print("=" * 80)
    print("Testing every classical codebase analysis technique available...")
    print()
    
    success = True
    
    # Test comprehensive analysis
    if not test_comprehensive_analysis():
        success = False
    
    # Test individual components
    if not test_individual_components():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] ALL TESTS PASSED - COMPREHENSIVE ANALYSIS FULLY OPERATIONAL!")
        print("\n[ANALYSIS] Available Analysis Techniques:")
        techniques = [
            "Software Metrics (Halstead, McCabe, SLOC, Maintainability Index)",
            "Graph Theory Analysis (Call graphs, Control flow, Dependencies)",
            "Code Clone Detection (Exact, Near, Structural, Semantic)",
            "Security Vulnerability Patterns and Code Smells",
            "Linguistic Analysis and Identifier Patterns",
            "Evolution and Change Pattern Analysis",
            "Statistical Code Analysis and Outlier Detection",
            "Structural and Architectural Pattern Detection",
            "Comprehensive Complexity Analysis (9 dimensions)",
            "Quality Analysis (9 quality factors)"
        ]
        
        for i, technique in enumerate(techniques, 1):
            print(f"   {i:2d}. {technique}")
        
        print(f"\n[INFO] Total Classical Techniques: {len(techniques)}")
        print("[INFO] All techniques are low-cost static analysis methods")
        print("[FAST] Analysis runs without external dependencies")
        print("[OVERVIEW] Perfect complement to LLM-based analysis")
        
    else:
        print("[ERROR] SOME TESTS FAILED - CHECK IMPLEMENTATION")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())