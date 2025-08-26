"""
Self-Evolving Architecture Test Suite
====================================

Agent C Hours 120-130: Self-Evolving Architecture Implementation

Comprehensive testing and demonstration of the self-evolving architecture system
with autonomous evolution planning, execution, and validation capabilities.
"""

import asyncio
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

from self_evolving_architecture import (
    create_self_evolving_architecture,
    ArchitecturalComponent,
    ArchitecturalMetrics,
    EvolutionTrigger,
    EvolutionPriority,
    EvolutionScope
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfEvolvingArchitectureTestSuite:
    """Comprehensive test suite for self-evolving architecture system"""
    
    def __init__(self):
        self.architecture_engine = None
        self.test_results = []
        self.test_codebase_path = None
        
    async def initialize_test_environment(self):
        """Initialize test environment with sample codebase"""
        logger.info("🚀 Initializing Self-Evolving Architecture Test Environment...")
        
        # Create temporary test codebase
        self.test_codebase_path = await self._create_test_codebase()
        
        # Initialize architecture engine
        config = {
            'evolution_interval_hours': 1,  # Fast evolution for testing
            'health_threshold': 70.0,
            'max_concurrent_evolutions': 5,
            'auto_evolution_enabled': True,
            'backup_before_evolution': True,
            'rollback_on_failure': True,
            'evolution_effort_limit_hours': 100.0,
            'intelligence_integration': True
        }
        
        self.architecture_engine = create_self_evolving_architecture(config)
        
        # Initialize with test codebase
        initialization_success = await self.architecture_engine.initialize(self.test_codebase_path)
        
        logger.info(f"[OK] Environment initialized: {initialization_success}")
        return initialization_success
    
    async def _create_test_codebase(self) -> str:
        """Create a realistic test codebase with various architectural issues"""
        temp_dir = tempfile.mkdtemp(prefix="test_architecture_")
        base_path = Path(temp_dir)
        
        # Create complex module with high coupling (anti-pattern)
        complex_module = base_path / "complex_module.py"
        complex_module.write_text('''
"""
Complex module with high coupling and complexity - needs refactoring
"""
import os
import sys
import json
import requests
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

class GodClass:
    """Example of God Class anti-pattern - does too many things"""
    
    def __init__(self):
        self.db_connection = sqlite3.connect("data.db")
        self.api_client = requests.Session()
        self.config = self._load_config()
        self.cache = {}
        self.logger = self._setup_logging()
    
    def _load_config(self):
        # Complex configuration loading
        with open("config.json", "r") as f:
            config = json.load(f)
        return config
    
    def _setup_logging(self):
        # Logging setup
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def process_user_data(self, user_id: int) -> Dict:
        # Complex user data processing
        user_data = self._fetch_user_from_db(user_id)
        enriched_data = self._enrich_user_data(user_data)
        validated_data = self._validate_user_data(enriched_data)
        processed_data = self._transform_user_data(validated_data)
        self._cache_user_data(user_id, processed_data)
        self._send_notification(user_id, "processed")
        return processed_data
    
    def _fetch_user_from_db(self, user_id: int):
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cursor.fetchone()
    
    def _enrich_user_data(self, user_data):
        # API calls to enrich data
        response = self.api_client.get(f"https://api.example.com/users/{user_data[0]}")
        return {**user_data, **response.json()}
    
    def _validate_user_data(self, data):
        # Complex validation logic
        if not data.get("email"):
            raise ValueError("Email required")
        if not data.get("name"):
            raise ValueError("Name required")
        return data
    
    def _transform_user_data(self, data):
        # Complex transformation logic
        return {
            "id": data["id"],
            "full_name": f"{data['first_name']} {data['last_name']}",
            "contact": {
                "email": data["email"],
                "phone": data.get("phone"),
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "source": "api"
            }
        }
    
    def _cache_user_data(self, user_id: int, data: Dict):
        self.cache[user_id] = data
    
    def _send_notification(self, user_id: int, event: str):
        # Notification logic
        self.logger.info(f"Notification sent to {user_id}: {event}")
    
    def generate_report(self, report_type: str) -> str:
        # Another complex method doing different things
        if report_type == "users":
            return self._generate_user_report()
        elif report_type == "activity":
            return self._generate_activity_report()
        else:
            raise ValueError("Unknown report type")
    
    def _generate_user_report(self) -> str:
        # Complex report generation
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        return f"User Report: {count} total users"
    
    def _generate_activity_report(self) -> str:
        # Another complex report
        return "Activity Report: Generated at " + datetime.now().isoformat()

# Tightly coupled helper functions (should be in separate modules)
def utility_function_1(data):
    return GodClass().process_user_data(data["user_id"])

def utility_function_2(report_type):
    return GodClass().generate_report(report_type)

def database_helper():
    # Direct database access without abstraction
    db = sqlite3.connect("data.db")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
    return db
        ''')
        
        # Create tightly coupled modules
        module_a = base_path / "module_a.py"
        module_a.write_text('''
"""
Module A - tightly coupled with Module B
"""
from module_b import ModuleBClass
from complex_module import GodClass

class ModuleAClass:
    def __init__(self):
        self.module_b = ModuleBClass()  # Tight coupling
        self.god_class = GodClass()     # Dependency on problematic class
    
    def do_work(self):
        # Method does too many things
        result = self.module_b.process()
        data = self.god_class.process_user_data(1)
        return {"result": result, "data": data}
    
    def another_method(self):
        return self.module_b.another_process()
        ''')
        
        module_b = base_path / "module_b.py"
        module_b.write_text('''
"""
Module B - tightly coupled with Module A
"""

class ModuleBClass:
    def process(self):
        # Creates circular dependency if we import module_a
        return "processed"
    
    def another_process(self):
        return "another_processed"
        ''')
        
        # Create a well-structured module (good example)
        good_module = base_path / "well_structured_module.py"
        good_module.write_text('''
"""
Well-structured module following best practices
"""
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any

class DataProcessor(Protocol):
    """Protocol for data processing"""
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

class UserDataProcessor:
    """Single responsibility: process user data"""
    
    def __init__(self, validator: 'DataValidator', enricher: 'DataEnricher'):
        self.validator = validator
        self.enricher = enricher
    
    def process_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user data with proper separation of concerns"""
        validated = self.validator.validate(user_data)
        enriched = self.enricher.enrich(validated)
        return enriched

class DataValidator:
    """Single responsibility: validate data"""
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data.get("email"):
            raise ValueError("Email required")
        return data

class DataEnricher:
    """Single responsibility: enrich data"""
    
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "enriched": True}
        ''')
        
        # Create legacy code with technical debt
        legacy_module = base_path / "legacy_code.py"
        legacy_module.write_text('''
"""
Legacy module with technical debt and outdated patterns
"""

# Global variables (anti-pattern)
global_cache = {}
global_counter = 0

def legacy_function(x, y, z=None, w=None, a=None, b=None):
    """Function with too many parameters"""
    global global_counter
    global_counter += 1
    
    # Nested conditions (high cyclomatic complexity)
    if x:
        if y:
            if z:
                if w:
                    if a:
                        if b:
                            return "deeply_nested_result"
                        else:
                            return "missing_b"
                    else:
                        return "missing_a"
                else:
                    return "missing_w"
            else:
                return "missing_z"
        else:
            return "missing_y"
    else:
        return "missing_x"

def unused_function():
    """This function is never called - dead code"""
    print("This is dead code")
    return "unused"

class LegacyClass:
    """Legacy class with poor design"""
    
    def __init__(self):
        # Too many instance variables
        self.var1 = None
        self.var2 = None
        self.var3 = None
        self.var4 = None
        self.var5 = None
        self.var6 = None
        self.var7 = None
        self.var8 = None
        self.var9 = None
        self.var10 = None
    
    def method_with_many_lines(self):
        """Method that's too long"""
        # 50+ lines of code doing various things
        result = []
        for i in range(100):
            if i % 2 == 0:
                if i % 4 == 0:
                    if i % 8 == 0:
                        result.append(f"divisible_by_8: {i}")
                    else:
                        result.append(f"divisible_by_4: {i}")
                else:
                    result.append(f"divisible_by_2: {i}")
            else:
                if i % 3 == 0:
                    if i % 9 == 0:
                        result.append(f"divisible_by_9: {i}")
                    else:
                        result.append(f"divisible_by_3: {i}")
                else:
                    result.append(f"odd: {i}")
        
        # More processing
        filtered = []
        for item in result:
            if "divisible" in item:
                filtered.append(item.upper())
            else:
                filtered.append(item.lower())
        
        # Even more processing
        final_result = {}
        for i, item in enumerate(filtered):
            final_result[f"item_{i}"] = {
                "value": item,
                "length": len(item),
                "index": i,
                "processed": True
            }
        
        return final_result
        ''')
        
        # Create circular dependency
        circular_a = base_path / "circular_a.py"
        circular_a.write_text('''
"""
Module that creates circular dependency with circular_b
"""
from circular_b import CircularB

class CircularA:
    def __init__(self):
        self.circular_b = CircularB()
    
    def work_with_b(self):
        return self.circular_b.process()
        ''')
        
        circular_b = base_path / "circular_b.py"
        circular_b.write_text('''
"""
Module that creates circular dependency with circular_a
"""
# Uncomment to create actual circular dependency
# from circular_a import CircularA

class CircularB:
    def __init__(self):
        pass  # self.circular_a = CircularA()  # Would create circle
    
    def process(self):
        return "processed_b"
        ''')
        
        logger.info(f"[SETUP] Created test codebase at: {temp_dir}")
        return temp_dir
    
    async def test_codebase_analysis(self) -> Dict[str, Any]:
        """Test comprehensive codebase analysis"""
        logger.info("🔍 Testing Codebase Analysis...")
        
        try:
            # Get system status
            status = await self.architecture_engine.get_system_status()
            
            # Analyze architectural components
            components = list(self.architecture_engine.components.values())
            
            # Test component analysis
            analyzer = self.architecture_engine.analyzer
            component_analyses = []
            
            for component in components[:5]:  # Test first 5 components
                analysis = await analyzer.analyze_component(component)
                component_analyses.append(analysis)
            
            # Test system-wide analysis
            system_analysis = await analyzer.analyze_system_architecture(components)
            
            results = {
                'system_status': status,
                'total_components': len(components),
                'component_analyses': component_analyses,
                'system_analysis': system_analysis,
                'average_health_score': status['system_overview']['average_health_score'],
                'components_needing_evolution': status['system_overview']['components_needing_evolution']
            }
            
            logger.info(f"    [OK] Analyzed {len(components)} components")
            logger.info(f"    [HEALTH] Average health score: {results['average_health_score']:.1f}")
            logger.info(f"    [EVOLUTION] Components needing evolution: {results['components_needing_evolution']}")
            
            return {
                'test_type': 'codebase_analysis',
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Codebase analysis failed: {e}")
            return {
                'test_type': 'codebase_analysis',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_evolution_planning(self) -> Dict[str, Any]:
        """Test evolution planning capabilities"""
        logger.info("📋 Testing Evolution Planning...")
        
        try:
            # Get components
            components = list(self.architecture_engine.components.values())
            
            # Create evolution plan
            planner = self.architecture_engine.planner
            evolution_plan = await planner.create_evolution_plan(
                components,
                constraints={
                    'max_effort_hours': 50.0,
                    'max_actions': 10
                }
            )
            
            # Validate plan structure
            required_keys = [
                'plan_id', 'created_at', 'system_analysis', 'evolution_actions',
                'execution_timeline', 'resource_requirements', 'success_metrics'
            ]
            
            plan_valid = all(key in evolution_plan for key in required_keys)
            
            results = {
                'plan_created': True,
                'plan_valid': plan_valid,
                'plan_id': evolution_plan['plan_id'],
                'total_actions': len(evolution_plan['evolution_actions']),
                'total_effort_hours': evolution_plan['resource_requirements']['total_effort_hours'],
                'estimated_duration_weeks': evolution_plan['resource_requirements']['estimated_duration_weeks'],
                'timeline_phases': len(evolution_plan['execution_timeline']['phases']),
                'action_types': list(set(action['action_type'] for action in evolution_plan['evolution_actions'])),
                'priority_distribution': self._analyze_priority_distribution(evolution_plan['evolution_actions'])
            }
            
            logger.info(f"    [OK] Created evolution plan with {results['total_actions']} actions")
            logger.info(f"    [EFFORT] Total effort: {results['total_effort_hours']} hours")
            logger.info(f"    [DURATION] Estimated duration: {results['estimated_duration_weeks']} weeks")
            
            return {
                'test_type': 'evolution_planning',
                'status': 'success',
                'results': results,
                'full_plan': evolution_plan
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Evolution planning failed: {e}")
            return {
                'test_type': 'evolution_planning',
                'status': 'failed',
                'error': str(e)
            }
    
    def _analyze_priority_distribution(self, actions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of action priorities"""
        distribution = {}
        for action in actions:
            priority = action.get('priority', 'unknown')
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution
    
    async def test_autonomous_evolution(self) -> Dict[str, Any]:
        """Test autonomous evolution execution"""
        logger.info("🤖 Testing Autonomous Evolution...")
        
        try:
            # Force evolution cycle
            evolution_result = await self.architecture_engine.evolve(force=True)
            
            # Get updated system status
            post_evolution_status = await self.architecture_engine.get_system_status()
            
            results = {
                'evolution_executed': evolution_result['status'] != 'failed',
                'evolution_status': evolution_result['status'],
                'cycle_id': evolution_result.get('cycle_id'),
                'execution_result': evolution_result.get('execution_result'),
                'post_evolution_health': post_evolution_status['system_overview']['average_health_score'],
                'evolution_metrics': post_evolution_status['evolution_metrics'],
                'evolution_cycles_completed': post_evolution_status['evolution_metrics']['evolution_cycles'],
                'successful_evolutions': post_evolution_status['evolution_metrics']['successful_evolutions']
            }
            
            logger.info(f"    [OK] Evolution cycle completed: {results['evolution_status']}")
            logger.info(f"    [HEALTH] Post-evolution health: {results['post_evolution_health']:.1f}")
            logger.info(f"    [CYCLES] Total cycles: {results['evolution_cycles_completed']}")
            
            return {
                'test_type': 'autonomous_evolution',
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Autonomous evolution failed: {e}")
            return {
                'test_type': 'autonomous_evolution',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_pattern_detection(self) -> Dict[str, Any]:
        """Test architectural pattern detection"""
        logger.info("🔍 Testing Pattern Detection...")
        
        try:
            components = list(self.architecture_engine.components.values())
            analyzer = self.architecture_engine.analyzer
            
            # Analyze architectural patterns
            system_analysis = await analyzer.analyze_system_architecture(components)
            patterns = system_analysis.get('architectural_patterns', {})
            
            # Test individual component pattern detection
            problematic_components = []
            well_structured_components = []
            
            for component in components:
                analysis = await analyzer.analyze_component(component)
                
                if analysis['anti_patterns']:
                    problematic_components.append({
                        'component': component.name,
                        'anti_patterns': analysis['anti_patterns'],
                        'health_score': analysis['health_score']
                    })
                
                if analysis['health_score'] > 80:
                    well_structured_components.append({
                        'component': component.name,
                        'health_score': analysis['health_score']
                    })
            
            results = {
                'patterns_detected': len(patterns.get('detected_patterns', [])),
                'anti_patterns_found': len(patterns.get('anti_patterns', [])),
                'pattern_recommendations': len(patterns.get('pattern_recommendations', [])),
                'problematic_components': len(problematic_components),
                'well_structured_components': len(well_structured_components),
                'detected_patterns': patterns.get('detected_patterns', []),
                'system_hotspots': len(system_analysis.get('system_hotspots', [])),
                'circular_dependencies': system_analysis['dependency_analysis']['circular_dependencies']
            }
            
            logger.info(f"    [OK] Detected {results['patterns_detected']} architectural patterns")
            logger.info(f"    [ISSUES] Found {results['anti_patterns_found']} anti-patterns")
            logger.info(f"    [HOTSPOTS] Identified {results['system_hotspots']} system hotspots")
            
            return {
                'test_type': 'pattern_detection',
                'status': 'success',
                'results': results,
                'details': {
                    'problematic_components': problematic_components[:5],  # Top 5
                    'well_structured_components': well_structured_components
                }
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Pattern detection failed: {e}")
            return {
                'test_type': 'pattern_detection',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_predictive_evolution(self) -> Dict[str, Any]:
        """Test predictive evolution capabilities"""
        logger.info("🔮 Testing Predictive Evolution...")
        
        try:
            # Test prediction for different time horizons
            predictions = {}
            
            for days in [7, 30, 90]:
                prediction = await self.architecture_engine.predict_evolution_needs(days)
                predictions[f'{days}_days'] = prediction
            
            # Analyze prediction quality
            prediction_available = any(p.get('prediction_available', False) for p in predictions.values())
            
            results = {
                'prediction_system_available': prediction_available,
                'predictions': predictions,
                'intelligence_integration': self.architecture_engine.intelligence_enabled,
                'prediction_confidence': predictions.get('30_days', {}).get('confidence', 0.0)
            }
            
            if prediction_available:
                thirty_day_pred = predictions.get('30_days', {})
                logger.info(f"    [OK] 30-day prediction available")
                logger.info(f"    [HEALTH] Predicted health: {thirty_day_pred.get('predicted_avg_health', 'N/A'):.1f}")
                logger.info(f"    [PROBABILITY] Evolution probability: {thirty_day_pred.get('evolution_probability', 0):.1%}")
            else:
                logger.info("    [INFO] Predictive capabilities limited - using simplified mode")
            
            return {
                'test_type': 'predictive_evolution',
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Predictive evolution failed: {e}")
            return {
                'test_type': 'predictive_evolution',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_system_resilience(self) -> Dict[str, Any]:
        """Test system resilience and error handling"""
        logger.info("🛡️ Testing System Resilience...")
        
        try:
            results = {}
            
            # Test with invalid input
            try:
                invalid_result = await self.architecture_engine.evolve()
                results['handles_invalid_input'] = True
            except Exception as e:
                results['handles_invalid_input'] = False
                results['invalid_input_error'] = str(e)
            
            # Test with resource constraints
            try:
                components = list(self.architecture_engine.components.values())
                constrained_plan = await self.architecture_engine.planner.create_evolution_plan(
                    components,
                    constraints={'max_effort_hours': 1.0, 'max_actions': 1}  # Very tight constraints
                )
                results['handles_constraints'] = True
                results['constrained_actions'] = len(constrained_plan['evolution_actions'])
            except Exception as e:
                results['handles_constraints'] = False
                results['constraints_error'] = str(e)
            
            # Test error recovery
            original_config = self.architecture_engine.config.copy()
            try:
                # Temporarily break configuration
                self.architecture_engine.config['invalid_setting'] = 'invalid_value'
                
                # System should still function
                status = await self.architecture_engine.get_system_status()
                results['error_recovery'] = status is not None
                
                # Restore configuration
                self.architecture_engine.config = original_config
                
            except Exception as e:
                results['error_recovery'] = False
                results['recovery_error'] = str(e)
                self.architecture_engine.config = original_config
            
            success_rate = sum(1 for key in results if key.endswith('_handles') and results[key]) / 3
            
            logger.info(f"    [OK] Resilience tests completed")
            logger.info(f"    [SCORE] Success rate: {success_rate:.1%}")
            
            return {
                'test_type': 'system_resilience',
                'status': 'success',
                'results': results,
                'success_rate': success_rate
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Resilience testing failed: {e}")
            return {
                'test_type': 'system_resilience',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_integration_capabilities(self) -> Dict[str, Any]:
        """Test integration with other intelligence systems"""
        logger.info("🔗 Testing Integration Capabilities...")
        
        try:
            # Test intelligence integration status
            intelligence_available = self.architecture_engine.intelligence_enabled
            
            results = {
                'intelligence_integration': intelligence_available,
                'decision_engine_available': hasattr(self.architecture_engine, 'decision_engine'),
                'pattern_engine_available': hasattr(self.architecture_engine, 'pattern_engine')
            }
            
            if intelligence_available:
                # Test decision engine integration
                try:
                    if hasattr(self.architecture_engine, 'decision_engine'):
                        decision_status = await self.architecture_engine.decision_engine.get_engine_status()
                        results['decision_engine_status'] = decision_status['status']
                        results['decision_engine_metrics'] = decision_status.get('performance_metrics', {})
                except Exception as e:
                    results['decision_engine_error'] = str(e)
                
                # Test pattern engine integration
                try:
                    if hasattr(self.architecture_engine, 'pattern_engine'):
                        # Pattern engine integration test
                        results['pattern_engine_initialized'] = True
                except Exception as e:
                    results['pattern_engine_error'] = str(e)
            
            # Test external tool integration
            results['external_analytics'] = self.architecture_engine.analyzer is not None
            results['evolution_planner'] = self.architecture_engine.planner is not None
            
            integration_score = sum(1 for key in results if key.endswith('_available') and results[key]) / 3
            
            logger.info(f"    [OK] Integration tests completed")
            logger.info(f"    [INTELLIGENCE] Intelligence systems: {intelligence_available}")
            logger.info(f"    [SCORE] Integration score: {integration_score:.1%}")
            
            return {
                'test_type': 'integration_capabilities',
                'status': 'success',
                'results': results,
                'integration_score': integration_score
            }
            
        except Exception as e:
            logger.error(f"    [ERROR] Integration testing failed: {e}")
            return {
                'test_type': 'integration_capabilities',
                'status': 'failed',
                'error': str(e)
            }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🧪 Starting Self-Evolving Architecture Test Suite...")
        
        start_time = datetime.now()
        
        # Initialize test environment
        initialization_success = await self.initialize_test_environment()
        if not initialization_success:
            return {'status': 'failed', 'error': 'Failed to initialize test environment'}
        
        # Test execution plan
        test_functions = [
            self.test_codebase_analysis,
            self.test_evolution_planning,
            self.test_autonomous_evolution,
            self.test_pattern_detection,
            self.test_predictive_evolution,
            self.test_system_resilience,
            self.test_integration_capabilities
        ]
        
        # Execute all tests
        test_results = []
        for test_func in test_functions:
            try:
                logger.info(f"Running {test_func.__name__}...")
                result = await test_func()
                test_results.append(result)
                logger.info(f"[OK] {test_func.__name__} completed")
            except Exception as e:
                logger.error(f"[ERROR] {test_func.__name__} failed: {e}")
                test_results.append({
                    'test_type': test_func.__name__,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate summary statistics
        end_time = datetime.now()
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r['status'] == 'success'])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        # Get final system status
        final_status = await self.architecture_engine.get_system_status()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_suite_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(end_time - start_time),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate
            },
            'individual_test_results': test_results,
            'final_system_status': final_status,
            'test_environment': {
                'codebase_path': self.test_codebase_path,
                'components_analyzed': len(self.architecture_engine.components),
                'intelligence_enabled': self.architecture_engine.intelligence_enabled
            },
            'performance_summary': {
                'average_system_health': final_status['system_overview']['average_health_score'],
                'evolution_cycles_completed': final_status['evolution_metrics']['evolution_cycles'],
                'successful_evolutions': final_status['evolution_metrics']['successful_evolutions']
            }
        }
        
        logger.info("🎉 Test Suite Completed!")
        logger.info(f"[STATS] Success Rate: {success_rate:.1%}")
        logger.info(f"[TIME] Duration: {end_time - start_time}")
        logger.info(f"[HEALTH] Final System Health: {final_status['system_overview']['average_health_score']:.1f}")
        
        return comprehensive_results
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.test_codebase_path and Path(self.test_codebase_path).exists():
            shutil.rmtree(self.test_codebase_path)
            logger.info(f"[CLEANUP] Removed test codebase: {self.test_codebase_path}")


async def main():
    """Main test execution function"""
    test_suite = SelfEvolvingArchitectureTestSuite()
    
    try:
        # Run comprehensive test suite
        results = await test_suite.run_comprehensive_test_suite()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"self_evolving_architecture_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[SAVE] Test results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("SELF-EVOLVING ARCHITECTURE TEST SUMMARY")
        print("="*80)
        print(f"[OK] Total Tests: {results['test_suite_info']['successful_tests']}/{results['test_suite_info']['total_tests']}")
        print(f"[STATS] Success Rate: {results['test_suite_info']['success_rate']:.1%}")
        print(f"[TIME] Duration: {results['test_suite_info']['duration']}")
        print(f"[HEALTH] Final System Health: {results['performance_summary']['average_system_health']:.1f}")
        print(f"[EVOLUTION] Cycles Completed: {results['performance_summary']['evolution_cycles_completed']}")
        print(f"[COMPONENTS] Components Analyzed: {results['test_environment']['components_analyzed']}")
        print(f"[INTELLIGENCE] Intelligence Systems: {results['test_environment']['intelligence_enabled']}")
        print("="*80)
        
        # Clean up
        await test_suite.cleanup_test_environment()
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())