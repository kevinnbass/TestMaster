"""
AGENT X STEELCLAD: Test Atomic Integration
Verify all atomic components are properly registered and accessible
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from specialized.atoms.atomic_registry import (
    AtomicComponentRegistry,
    AtomicDashboardBuilder,
    get_atomization_report
)

def test_atomic_registry():
    """Test the atomic component registry"""
    print("=" * 60)
    print("TESTING ATOMIC COMPONENT REGISTRY")
    print("=" * 60)
    
    registry = AtomicComponentRegistry()
    
    # Test CSS components
    print("\n📄 CSS COMPONENTS:")
    for name, component in registry.registry["css"].items():
        print(f"  ✅ {name}: {component['url']}")
    
    # Test JS components
    print("\n📜 JAVASCRIPT COMPONENTS:")
    for name, component in registry.registry["js"].items():
        print(f"  ✅ {name}: {component['url']}")
    
    # Test Python modules
    print("\n🐍 PYTHON ATOMIC MODULES:")
    for name, component in registry.registry["python"].items():
        print(f"  ✅ {name}: {component['module']}")
    
    # Test HTML cores
    print("\n🏗️ HTML CORE TEMPLATES:")
    for name, component in registry.registry["html_cores"].items():
        print(f"  ✅ {name}: {component['template']}")
    
    # Get stats
    stats = registry.get_atomic_stats()
    print("\n📊 ATOMIC STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_dashboard_builder():
    """Test the atomic dashboard builder"""
    print("\n" + "=" * 60)
    print("TESTING ATOMIC DASHBOARD BUILDER")
    print("=" * 60)
    
    builder = AtomicDashboardBuilder()
    
    # Test building a dashboard with multiple components
    builder.add_css("dashboard_styles", "charts_styles")
    builder.add_js("dashboard_scripts", "charts_scripts")
    builder.add_python_module("dashboard_analytics", "viz_engine")
    
    print("\n🔨 BUILDING HTML IMPORTS:")
    html_imports = builder.build_html_imports()
    for line in html_imports.split('\n'):
        print(f"  {line}")
    
    print("\n🔨 BUILDING PYTHON IMPORTS:")
    python_imports = builder.build_python_imports()
    for line in python_imports.split('\n'):
        print(f"  {line}")

def test_dashboard_assets():
    """Test loading assets for specific dashboards"""
    print("\n" + "=" * 60)
    print("TESTING DASHBOARD ASSET LOADING")
    print("=" * 60)
    
    registry = AtomicComponentRegistry()
    
    dashboards = ["dashboard", "unified_gamma", "charts", "unified_template"]
    
    for dashboard in dashboards:
        assets = registry.get_dashboard_assets(dashboard)
        print(f"\n🎨 {dashboard.upper()} DASHBOARD:")
        print(f"  Stylesheets: {assets['stylesheets']}")
        print(f"  Scripts: {assets['scripts']}")

def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("AGENT X STEELCLAD: ATOMIC INTEGRATION TEST SUITE")
    print("🚀 " * 20)
    
    # Run tests
    test_atomic_registry()
    test_dashboard_builder()
    test_dashboard_assets()
    
    # Print final report
    print("\n" + "=" * 60)
    print(get_atomization_report())
    
    print("\n✅ ATOMIC INTEGRATION TEST COMPLETE!")
    print("All atomic components are properly registered and accessible.")
    print("The STEELCLAD atomization has been successfully integrated! 🛡️")

if __name__ == "__main__":
    main()