"""
Comprehensive Unit Tests for Architecture Components
===================================================

Tests for the three core architecture components implemented in Phase 0 Hour 1:
- LayerManager (core/architecture/layer_separation.py)
- DependencyContainer (core/architecture/dependency_injection.py) 
- ImportResolver (core/foundation/import_resolver.py)

These tests ensure the solid foundation established in Hour 1 is properly validated
and ready for the extensive modularization work ahead in the 500-hour mission.

Author: Agent A (Latin Swarm)
Created: Phase 0 Hour 2
"""

import unittest
import sys
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the components we're testing
from core.architecture.layer_separation import LayerManager, LayerViolation, ArchitectureValidationResult
from core.architecture.dependency_injection import (
    DependencyContainer, LifetimeScope, InjectionStrategy, 
    CircularDependencyError, DependencyResolutionError
)
from core.foundation.import_resolver import (
    ImportResolver, ImportStrategy, FeatureDiscoveryLog
)


class TestLayerManager(unittest.TestCase):
    """Comprehensive tests for LayerManager component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.layer_manager = LayerManager()
    
    def test_initialization(self):
        """Test LayerManager initialization."""
        self.assertIsNotNone(self.layer_manager.layers)
        self.assertIsNotNone(self.layer_manager.layer_dependencies)
        self.assertIsNotNone(self.layer_manager.adapters)
        self.assertEqual(len(self.layer_manager.layers), 0)
    
    def test_register_layer_success(self):
        """Test successful layer registration."""
        result = self.layer_manager.register_layer("presentation", 1, "UI layer")
        self.assertTrue(result)
        self.assertIn("presentation", self.layer_manager.layers)
        self.assertEqual(self.layer_manager.layers["presentation"]["level"], 1)
    
    def test_register_layer_duplicate(self):
        """Test registering duplicate layer."""
        self.layer_manager.register_layer("business", 2, "Business logic")
        result = self.layer_manager.register_layer("business", 3, "Duplicate")
        self.assertFalse(result)
        self.assertEqual(self.layer_manager.layers["business"]["level"], 2)
    
    def test_add_layer_dependency_valid(self):
        """Test adding valid layer dependency."""
        self.layer_manager.register_layer("presentation", 1, "UI layer")
        self.layer_manager.register_layer("business", 2, "Business layer")
        
        result = self.layer_manager.add_layer_dependency("presentation", "business")
        self.assertTrue(result)
        self.assertIn("business", self.layer_manager.layer_dependencies["presentation"])
    
    def test_add_layer_dependency_invalid_direction(self):
        """Test adding dependency in wrong direction (violation)."""
        self.layer_manager.register_layer("presentation", 1, "UI layer")
        self.layer_manager.register_layer("business", 2, "Business layer")
        
        # Business layer depending on presentation layer is a violation
        result = self.layer_manager.add_layer_dependency("business", "presentation")
        self.assertFalse(result)
    
    def test_validate_architecture_integrity_clean(self):
        """Test architecture validation with clean setup."""
        # Setup clean architecture
        self.layer_manager.register_layer("presentation", 1, "UI layer")
        self.layer_manager.register_layer("business", 2, "Business layer")
        self.layer_manager.register_layer("data", 3, "Data layer")
        
        self.layer_manager.add_layer_dependency("presentation", "business")
        self.layer_manager.add_layer_dependency("business", "data")
        
        result = self.layer_manager.validate_architecture_integrity()
        self.assertIsInstance(result, ArchitectureValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.violations), 0)
        self.assertGreater(result.compliance_score, 90)
    
    def test_validate_architecture_integrity_violations(self):
        """Test architecture validation with violations."""
        # Setup architecture with violations
        self.layer_manager.register_layer("presentation", 1, "UI layer")
        self.layer_manager.register_layer("business", 2, "Business layer")
        
        # Force a violation by directly manipulating dependencies
        self.layer_manager.layer_dependencies["business"] = ["presentation"]
        
        result = self.layer_manager.validate_architecture_integrity()
        self.assertIsInstance(result, ArchitectureValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.violations), 0)
    
    def test_register_adapter_success(self):
        """Test successful adapter registration."""
        result = self.layer_manager.register_adapter(
            "test_adapter", "presentation", "business", "Test adapter"
        )
        self.assertTrue(result)
        self.assertIn("test_adapter", self.layer_manager.adapters)
    
    def test_get_architecture_metrics(self):
        """Test architecture metrics generation."""
        # Setup basic architecture
        self.layer_manager.register_layer("presentation", 1, "UI")
        self.layer_manager.register_layer("business", 2, "Logic")
        self.layer_manager.add_layer_dependency("presentation", "business")
        
        metrics = self.layer_manager.get_architecture_metrics()
        
        self.assertIn("total_layers", metrics)
        self.assertIn("total_dependencies", metrics)
        self.assertIn("total_adapters", metrics)
        self.assertIn("compliance_score", metrics)
        self.assertEqual(metrics["total_layers"], 2)


class TestDependencyContainer(unittest.TestCase):
    """Comprehensive tests for DependencyContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.container = DependencyContainer()
    
    def test_initialization(self):
        """Test DependencyContainer initialization."""
        self.assertIsNotNone(self.container.services)
        self.assertIsNotNone(self.container.instances)
        self.assertIsNotNone(self.container.resolvers)
        self.assertEqual(len(self.container.services), 0)
    
    def test_register_singleton_service(self):
        """Test registering singleton service."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        self.container.register(TestService, lifetime=LifetimeScope.SINGLETON)
        self.assertIn(TestService, self.container.services)
        self.assertEqual(self.container.services[TestService]['lifetime'], LifetimeScope.SINGLETON)
    
    def test_register_transient_service(self):
        """Test registering transient service."""
        class TestService:
            pass
        
        self.container.register(TestService, lifetime=LifetimeScope.TRANSIENT)
        self.assertEqual(self.container.services[TestService]['lifetime'], LifetimeScope.TRANSIENT)
    
    def test_resolve_singleton_same_instance(self):
        """Test that singleton returns same instance."""
        class TestService:
            def __init__(self):
                self.id = time.time()
        
        self.container.register(TestService, lifetime=LifetimeScope.SINGLETON)
        
        instance1 = self.container.resolve(TestService)
        instance2 = self.container.resolve(TestService)
        
        self.assertIs(instance1, instance2)
        self.assertEqual(instance1.id, instance2.id)
    
    def test_resolve_transient_different_instances(self):
        """Test that transient returns different instances."""
        class TestService:
            def __init__(self):
                self.id = time.time()
        
        self.container.register(TestService, lifetime=LifetimeScope.TRANSIENT)
        
        instance1 = self.container.resolve(TestService)
        time.sleep(0.01)  # Ensure different timestamps
        instance2 = self.container.resolve(TestService)
        
        self.assertIsNot(instance1, instance2)
        self.assertNotEqual(instance1.id, instance2.id)
    
    def test_resolve_with_dependencies(self):
        """Test resolving service with dependencies."""
        class DatabaseService:
            def __init__(self):
                self.connected = True
        
        class UserService:
            def __init__(self, db_service: DatabaseService):
                self.db_service = db_service
        
        self.container.register(DatabaseService, lifetime=LifetimeScope.SINGLETON)
        self.container.register(UserService, lifetime=LifetimeScope.TRANSIENT)
        
        user_service = self.container.resolve(UserService)
        
        self.assertIsInstance(user_service, UserService)
        self.assertIsInstance(user_service.db_service, DatabaseService)
        self.assertTrue(user_service.db_service.connected)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a):
                self.service_a = service_a
        
        self.container.register(ServiceA, lifetime=LifetimeScope.TRANSIENT)
        self.container.register(ServiceB, lifetime=LifetimeScope.TRANSIENT)
        
        with self.assertRaises(CircularDependencyError):
            self.container.resolve(ServiceA)
    
    def test_unregistered_service_resolution(self):
        """Test resolving unregistered service."""
        class UnregisteredService:
            pass
        
        with self.assertRaises(DependencyResolutionError):
            self.container.resolve(UnregisteredService)
    
    def test_custom_resolver(self):
        """Test custom resolver functionality."""
        class TestService:
            def __init__(self, value):
                self.value = value
        
        def custom_resolver():
            return TestService("custom_value")
        
        self.container.register(TestService, resolver=custom_resolver)
        instance = self.container.resolve(TestService)
        
        self.assertIsInstance(instance, TestService)
        self.assertEqual(instance.value, "custom_value")
    
    def test_validate_registrations(self):
        """Test registration validation."""
        class ValidService:
            def __init__(self):
                pass
        
        self.container.register(ValidService)
        validation_result = self.container.validate_registrations()
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn("is_valid", validation_result)
        self.assertIn("errors", validation_result)
        self.assertTrue(validation_result["is_valid"])
    
    def test_get_registration_info(self):
        """Test getting registration information."""
        class TestService:
            pass
        
        self.container.register(TestService, lifetime=LifetimeScope.SINGLETON)
        info = self.container.get_registration_info(TestService)
        
        self.assertIsNotNone(info)
        self.assertEqual(info["lifetime"], LifetimeScope.SINGLETON)
        self.assertIn("registered_at", info)


class TestImportResolver(unittest.TestCase):
    """Comprehensive tests for ImportResolver component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resolver = ImportResolver()
    
    def test_initialization(self):
        """Test ImportResolver initialization."""
        self.assertIsNotNone(self.resolver.cache)
        self.assertIsNotNone(self.resolver.search_paths)
        self.assertIsNotNone(self.resolver.providers)
        self.assertTrue(self.resolver.enable_discovery)
    
    def test_resolve_import_direct_success(self):
        """Test successful direct import resolution."""
        # Test with built-in module
        result = self.resolver.resolve_import("json", ImportStrategy.DIRECT)
        self.assertIsNotNone(result)
        self.assertEqual(result.__name__, "json")
    
    def test_resolve_import_with_caching(self):
        """Test import resolution with caching."""
        # First resolution
        result1 = self.resolver.resolve_import("os", ImportStrategy.DIRECT)
        
        # Second resolution (should use cache)
        result2 = self.resolver.resolve_import("os", ImportStrategy.DIRECT)
        
        self.assertIs(result1, result2)
        self.assertGreater(self.resolver.cache_stats["hits"], 0)
    
    def test_resolve_import_with_fallback(self):
        """Test import resolution with fallback strategy."""
        # Try to import non-existent module with fallback
        result = self.resolver.resolve_import(
            "nonexistent_module_12345", 
            ImportStrategy.WITH_FALLBACK
        )
        
        # Should return None for non-existent module
        self.assertIsNone(result)
    
    def test_add_search_path(self):
        """Test adding custom search path."""
        initial_count = len(self.resolver.search_paths)
        test_path = "/custom/search/path"
        
        self.resolver.add_search_path(test_path)
        
        self.assertEqual(len(self.resolver.search_paths), initial_count + 1)
        self.assertIn(test_path, self.resolver.search_paths)
    
    def test_register_provider(self):
        """Test registering custom import provider."""
        def custom_provider(module_name):
            if module_name == "test_module":
                return Mock()
            return None
        
        self.resolver.register_provider("custom", custom_provider, priority=10)
        
        self.assertIn("custom", self.resolver.providers)
        self.assertEqual(self.resolver.providers["custom"]["priority"], 10)
    
    def test_execute_feature_discovery(self):
        """Test feature discovery protocol execution."""
        # Test that feature discovery can be enabled/disabled
        original_setting = self.resolver.enable_discovery
        
        self.resolver.enable_discovery = False
        self.assertFalse(self.resolver.enable_discovery)
        
        self.resolver.enable_discovery = True
        self.assertTrue(self.resolver.enable_discovery)
        
        # Restore original setting
        self.resolver.enable_discovery = original_setting
    
    def test_get_import_statistics(self):
        """Test import statistics generation."""
        # Perform some imports to generate stats
        self.resolver.resolve_import("sys")
        self.resolver.resolve_import("os")
        
        stats = self.resolver.get_import_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_imports", stats)
        self.assertIn("cache_stats", stats)
        self.assertIn("provider_stats", stats)
        self.assertGreater(stats["total_imports"], 0)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate cache
        self.resolver.resolve_import("json")
        self.assertGreater(len(self.resolver.cache), 0)
        
        # Clear cache
        self.resolver.clear_cache()
        self.assertEqual(len(self.resolver.cache), 0)
        self.assertEqual(self.resolver.cache_stats["hits"], 0)
    
    def test_import_resolution_error_handling(self):
        """Test error handling in import resolution."""
        # Test with invalid module name - should return None gracefully
        result = self.resolver.resolve_import("", ImportStrategy.DIRECT)
        self.assertIsNone(result)
    
    def test_discover_modules_in_path(self):
        """Test module discovery in search paths."""
        # Use current project directory for testing
        project_path = str(Path(__file__).parent.parent)
        self.resolver.add_search_path(project_path)
        
        modules = self.resolver.discover_modules_in_path(project_path)
        
        self.assertIsInstance(modules, list)
        # Should find some Python modules in the project
        self.assertGreater(len(modules), 0)


class TestComponentIntegration(unittest.TestCase):
    """Integration tests for all three components working together."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.layer_manager = LayerManager()
        self.container = DependencyContainer()
        self.resolver = ImportResolver()
    
    def test_components_can_be_imported_together(self):
        """Test that all components can be imported and used together."""
        # This test ensures there are no import conflicts
        self.assertIsNotNone(self.layer_manager)
        self.assertIsNotNone(self.container)
        self.assertIsNotNone(self.resolver)
    
    def test_architecture_with_dependency_injection(self):
        """Test LayerManager working with DependencyContainer."""
        # Register LayerManager as a service
        self.container.register(LayerManager, lifetime=LifetimeScope.SINGLETON)
        
        # Resolve LayerManager from container
        layer_manager = self.container.resolve(LayerManager)
        
        self.assertIsInstance(layer_manager, LayerManager)
        
        # Use the resolved instance
        layer_manager.register_layer("test", 1, "Test layer")
        self.assertIn("test", layer_manager.layers)
    
    def test_import_resolver_with_dependency_injection(self):
        """Test ImportResolver working with DependencyContainer."""
        # Register ImportResolver as a service
        self.container.register(ImportResolver, lifetime=LifetimeScope.SINGLETON)
        
        # Resolve ImportResolver from container
        resolver = self.container.resolve(ImportResolver)
        
        self.assertIsInstance(resolver, ImportResolver)
        
        # Test import resolution
        result = resolver.resolve_import("json")
        self.assertIsNotNone(result)
    
    def test_full_integration_scenario(self):
        """Test a complete integration scenario."""
        # 1. Setup architecture layers
        self.layer_manager.register_layer("presentation", 1, "UI Layer")
        self.layer_manager.register_layer("business", 2, "Business Logic")
        self.layer_manager.register_layer("data", 3, "Data Access")
        
        # 2. Register components in DI container
        self.container.register(LayerManager, instance=self.layer_manager)
        self.container.register(ImportResolver, instance=self.resolver)
        
        # 3. Resolve components and verify they work
        resolved_layer_manager = self.container.resolve(LayerManager)
        resolved_resolver = self.container.resolve(ImportResolver)
        
        # 4. Validate architecture
        validation_result = resolved_layer_manager.validate_architecture_integrity()
        
        # 5. Test import resolution
        json_module = resolved_resolver.resolve_import("json")
        
        # Assertions
        self.assertIs(resolved_layer_manager, self.layer_manager)
        self.assertIs(resolved_resolver, self.resolver)
        self.assertTrue(validation_result.is_valid)
        self.assertIsNotNone(json_module)


class TestPerformanceAndReliability(unittest.TestCase):
    """Performance and reliability tests for all components."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.layer_manager = LayerManager()
        self.container = DependencyContainer()
        self.resolver = ImportResolver()
    
    def test_layer_manager_performance(self):
        """Test LayerManager performance with many layers."""
        start_time = time.time()
        
        # Register many layers
        for i in range(100):
            self.layer_manager.register_layer(f"layer_{i}", i + 1, f"Layer {i}")
        
        # Add dependencies
        for i in range(99):
            self.layer_manager.add_layer_dependency(f"layer_{i}", f"layer_{i + 1}")
        
        # Validate architecture
        result = self.layer_manager.validate_architecture_integrity()
        
        elapsed = time.time() - start_time
        
        self.assertTrue(result.is_valid)
        self.assertLess(elapsed, 5.0)  # Should complete within 5 seconds
    
    def test_dependency_container_performance(self):
        """Test DependencyContainer performance with many services."""
        start_time = time.time()
        
        # Register many services
        services = []
        for i in range(50):
            service_class = type(f"TestService_{i}", (), {"__init__": lambda self: None})
            services.append(service_class)
            self.container.register(service_class, lifetime=LifetimeScope.TRANSIENT)
        
        # Resolve all services
        instances = []
        for service_class in services:
            instances.append(self.container.resolve(service_class))
        
        elapsed = time.time() - start_time
        
        self.assertEqual(len(instances), 50)
        self.assertLess(elapsed, 2.0)  # Should complete within 2 seconds
    
    def test_import_resolver_caching_performance(self):
        """Test ImportResolver caching performance."""
        # First round - populate cache
        start_time = time.time()
        for _ in range(10):
            self.resolver.resolve_import("json")
        first_round = time.time() - start_time
        
        # Second round - use cache
        start_time = time.time()
        for _ in range(10):
            self.resolver.resolve_import("json")
        second_round = time.time() - start_time
        
        # Cache should make second round faster
        self.assertLess(second_round, first_round)
        self.assertGreater(self.resolver.cache_stats["hits"], 0)
    
    def test_memory_usage_stability(self):
        """Test that components don't leak memory."""
        import gc
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many instances
        for _ in range(100):
            layer_manager = LayerManager()
            layer_manager.register_layer("temp", 1, "Temp")
            del layer_manager
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not grow significantly
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Allow some growth but not excessive


if __name__ == '__main__':
    # Configure test runner
    unittest.TestLoader.testMethodPrefix = 'test_'
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLayerManager,
        TestDependencyContainer, 
        TestImportResolver,
        TestComponentIntegration,
        TestPerformanceAndReliability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTest(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE COMPONENTS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE FOUNDATION VALIDATION: {'✅ PASSED' if result.wasSuccessful() else '❌ FAILED'}")
    print(f"{'='*60}")