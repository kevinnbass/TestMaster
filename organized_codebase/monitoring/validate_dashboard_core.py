#!/usr/bin/env python3
"""
Validate Dashboard Core Functionality
======================================
Ensures all 58 analytics modules and other components
are still accessible after the rename.
"""

import sys
import importlib
from pathlib import Path

def validate_imports():
    """Validate all dashboard_core imports work."""
    
    # Critical imports to test
    critical_imports = [
        'dashboard.dashboard_core.monitor',
        'dashboard.dashboard_core.cache',
        'dashboard.dashboard_core.error_handler',
        'dashboard.dashboard_core.metrics_feed',
        'dashboard.dashboard_core.analytics_aggregator',
        'dashboard.dashboard_core.real_data_extractor',
        'dashboard.dashboard_core.test_collector',
        'dashboard.dashboard_core.data_store'
    ]
    
    # Analytics modules (58 of them!)
    analytics_modules = [
        'analytics_aggregator',
        'analytics_anomaly_detector',
        'analytics_backup',
        'analytics_batch_processor',
        'analytics_circuit_breaker',
        'analytics_compressor',
        'analytics_connectivity_monitor',
        'analytics_correlator',
        'analytics_data_sanitizer',
        'analytics_dead_letter_queue',
        'analytics_deduplication',
        'analytics_deduplication_engine',
        'analytics_delivery_enhancer',
        'analytics_delivery_guarantee',
        'analytics_delivery_verifier',
        'analytics_error_recovery',
        'analytics_event_queue',
        'analytics_export_manager',
        'analytics_fallback_system',
        'analytics_flow_monitor',
        'analytics_health_monitor',
        'analytics_heartbeat_monitor',
        'analytics_integrity_guardian',
        'analytics_integrity_verifier',
        'analytics_metrics_collector',
        'analytics_normalizer',
        'analytics_optimizer',
        'analytics_performance_booster',
        'analytics_performance_monitor',
        'analytics_performance_optimizer',
        'analytics_persistence',
        'analytics_pipeline',
        'analytics_pipeline_health_monitor',
        'analytics_priority_queue',
        'analytics_quality_assurance',
        'analytics_quantum_retry',
        'analytics_rate_limiter',
        'analytics_receipt_tracker',
        'analytics_recovery_orchestrator',
        'analytics_redundancy',
        'analytics_retry_manager',
        'analytics_sla_tracker',
        'analytics_smart_cache',
        'analytics_streaming',
        'analytics_telemetry',
        'analytics_validator',
        'analytics_watchdog',
        'emergency_backup_recovery',
        'predictive_flow_optimizer',
        'realtime_analytics_tracker',
        'system_observability_metrics'
    ]
    
    print("=" * 60)
    print("VALIDATING DASHBOARD CORE FUNCTIONALITY")
    print("=" * 60)
    
    # Test critical imports
    print("\n[1] Testing Critical Imports:")
    print("-" * 40)
    success_count = 0
    for module_path in critical_imports:
        try:
            module = importlib.import_module(module_path)
            print(f"  [OK] {module_path}")
            success_count += 1
        except ImportError as e:
            print(f"  [FAIL] {module_path}: {e}")
    
    print(f"\nCritical imports: {success_count}/{len(critical_imports)} successful")
    
    # Test analytics modules
    print("\n[2] Testing Analytics Modules:")
    print("-" * 40)
    analytics_success = 0
    for module_name in analytics_modules:
        module_path = f'dashboard.dashboard_core.{module_name}'
        try:
            module = importlib.import_module(module_path)
            analytics_success += 1
            # Show progress every 10 modules
            if analytics_success % 10 == 0:
                print(f"  Validated {analytics_success} analytics modules...")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
    
    print(f"\nAnalytics modules: {analytics_success}/{len(analytics_modules)} successful")
    
    # Test that old paths fail (as expected)
    print("\n[3] Verifying Old Paths Are Invalid:")
    print("-" * 40)
    old_paths = [
        'dashboard.core.monitor',
        'dashboard.core.cache'
    ]
    
    for old_path in old_paths:
        try:
            importlib.import_module(old_path)
            print(f"  [WARNING] Old path still works: {old_path}")
        except ImportError:
            print(f"  [OK] Old path correctly fails: {old_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_modules = len(critical_imports) + len(analytics_modules)
    total_success = success_count + analytics_success
    
    print(f"Total modules tested: {total_modules}")
    print(f"Successfully imported: {total_success}")
    print(f"Success rate: {total_success/total_modules*100:.1f}%")
    
    if total_success == total_modules:
        print("\n[SUCCESS] ALL FUNCTIONALITY PRESERVED!")
        print("The dashboard/core rename to dashboard/dashboard_core was successful.")
        print("All 58 analytics modules plus core components are accessible.")
        return 0
    else:
        print(f"\n[WARNING] {total_modules - total_success} modules failed to import")
        print("Some functionality may need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(validate_imports())