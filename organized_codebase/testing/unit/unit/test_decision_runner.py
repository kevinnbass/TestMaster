#!/usr/bin/env python3
"""
Test Decision Engine Runner - Main Coordinator
==============================================

Main coordinator for the enhanced autonomous decision engine test suite.
This module orchestrates all specialized test modules.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from test_decision_models import TestConfiguration
from test_decision_scaling import ScalingDecisionTester
from test_decision_performance import PerformanceOptimizationTester
from test_decision_emergency import EmergencyResponseTester
from test_decision_cognitive import CognitiveEnhancementTester
from test_decision_pattern import PatternRecognitionTester
from test_decision_ml import MLEnsembleTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDecisionEngineTestSuite:
    """Comprehensive test suite for enhanced autonomous decision engine"""

    def __init__(self, engine=None):
        """Initialize the test suite"""
        self.engine = engine
        self.test_results = []
        self.performance_metrics = {}

        # Initialize test modules
        if engine:
            self.scaling_tester = ScalingDecisionTester(engine)
            self.performance_tester = PerformanceOptimizationTester(engine)
            self.emergency_tester = EmergencyResponseTester(engine)
            self.cognitive_tester = CognitiveEnhancementTester(engine)
            self.pattern_tester = PatternRecognitionTester(engine)
            self.ml_tester = MLEnsembleTester(engine)

    def set_engine(self, engine):
        """Set the decision engine for testing"""
        self.engine = engine

        # Initialize test modules with engine
        self.scaling_tester = ScalingDecisionTester(engine)
        self.performance_tester = PerformanceOptimizationTester(engine)
        self.emergency_tester = EmergencyResponseTester(engine)
        self.cognitive_tester = CognitiveEnhancementTester(engine)
        self.pattern_tester = PatternRecognitionTester(engine)
        self.ml_tester = MLEnsembleTester(engine)

    async def initialize_engine(self):
        """Initialize the enhanced decision engine"""
        logger.info("🚀 Initializing Enhanced Autonomous Decision Engine...")

        config = TestConfiguration.get_default_config()

        # Mock engine initialization for testing
        if self.engine is None:
            # Create a mock engine status for testing
            engine_status = {
                'status': 'initialized',
                'cognitive_enabled': config['cognitive_enhancement'],
                'model_trained': config['ensemble_ml'],
                'pattern_recognition_enabled': config['pattern_recognition']
            }
        else:
            # Real engine status
            try:
                engine_status = await self.engine.get_engine_status()
            except:
                engine_status = {
                    'status': 'initialized',
                    'cognitive_enabled': config['cognitive_enhancement'],
                    'model_trained': config['ensemble_ml'],
                    'pattern_recognition_enabled': config['pattern_recognition']
                }

        logger.info(f"✅ Engine initialized with status: {engine_status['status']}")
        logger.info(f"📊 Cognitive enhancement: {engine_status.get('cognitive_enabled', False)}")
        logger.info(f"🧠 ML models available: {engine_status.get('model_trained', False)}")

        return engine_status

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🧪 Starting Enhanced Autonomous Decision Engine Test Suite...")

        start_time = datetime.now()

        # Initialize engine
        engine_status = await self.initialize_engine()

        # Run all test categories
        test_results = {
            'engine_status': engine_status,
            'test_execution': {
                'start_time': start_time.isoformat(),
                'test_categories': []
            }
        }

        # Test categories - run each specialized tester
        test_categories = [
            ('scaling_decisions', self._run_scaling_tests),
            ('performance_optimization', self._run_performance_tests),
            ('emergency_response', self._run_emergency_tests),
            ('cognitive_enhancement', self._run_cognitive_tests),
            ('pattern_recognition', self._run_pattern_tests),
            ('ml_ensemble_learning', self._run_ml_tests)
        ]

        for test_name, test_func in test_categories:
            try:
                logger.info(f"Running {test_name} tests...")
                result = await test_func()
                test_results['test_execution']['test_categories'].append(result)
                logger.info(f"✅ {test_name} tests completed")
            except Exception as e:
                logger.error(f"❌ {test_name} tests failed: {e}")
                test_results['test_execution']['test_categories'].append({
                    'test_type': test_name,
                    'error': str(e),
                    'status': 'failed'
                })

        # Final performance metrics
        end_time = datetime.now()

        test_results['test_execution']['end_time'] = end_time.isoformat()
        test_results['test_execution']['duration'] = str(end_time - start_time)

        # Generate summary statistics
        test_results['summary'] = self._generate_test_summary(test_results['test_execution']['test_categories'])

        logger.info("🎉 Test Suite Completed!")
        logger.info(f"📊 Overall Success Rate: {test_results['summary']['success_rate']:.1%}")
        logger.info(f"⏱️ Total Duration: {test_results['test_execution']['duration']}")

        return test_results

    async def _run_scaling_tests(self) -> Dict[str, Any]:
        """Run scaling decision tests"""
        if not hasattr(self, 'scaling_tester'):
            return {'test_type': 'scaling_decisions', 'error': 'Engine not initialized', 'status': 'failed'}

        scaling_results = await self.scaling_tester.test_scaling_decisions()
        load_results = await self.scaling_tester.test_scaling_under_load()
        cost_results = await self.scaling_tester.test_cost_optimization_scaling()

        # Combine all scaling test results
        combined_results = {
            'test_type': 'scaling_decisions',
            'sub_tests': [scaling_results, load_results, cost_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [scaling_results, load_results, cost_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [scaling_results, load_results, cost_results])
        }

        return combined_results

    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance optimization tests"""
        if not hasattr(self, 'performance_tester'):
            return {'test_type': 'performance_optimization', 'error': 'Engine not initialized', 'status': 'failed'}

        perf_results = await self.performance_tester.test_performance_optimization_decisions()
        db_results = await self.performance_tester.test_database_optimization()
        cache_results = await self.performance_tester.test_cache_optimization()

        # Combine all performance test results
        combined_results = {
            'test_type': 'performance_optimization',
            'sub_tests': [perf_results, db_results, cache_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [perf_results, db_results, cache_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [perf_results, db_results, cache_results])
        }

        return combined_results

    async def _run_emergency_tests(self) -> Dict[str, Any]:
        """Run emergency response tests"""
        if not hasattr(self, 'emergency_tester'):
            return {'test_type': 'emergency_response', 'error': 'Engine not initialized', 'status': 'failed'}

        emergency_results = await self.emergency_tester.test_emergency_response_decisions()
        disaster_results = await self.emergency_tester.test_disaster_recovery()
        incident_results = await self.emergency_tester.test_incident_response()

        # Combine all emergency test results
        combined_results = {
            'test_type': 'emergency_response',
            'sub_tests': [emergency_results, disaster_results, incident_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [emergency_results, disaster_results, incident_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [emergency_results, disaster_results, incident_results])
        }

        return combined_results

    async def _run_cognitive_tests(self) -> Dict[str, Any]:
        """Run cognitive enhancement tests"""
        if not hasattr(self, 'cognitive_tester'):
            return {'test_type': 'cognitive_enhancement', 'error': 'Engine not initialized', 'status': 'failed'}

        cognitive_results = await self.cognitive_tester.test_cognitive_enhancement()
        business_results = await self.cognitive_tester.test_business_logic_cognition()

        # Combine all cognitive test results
        combined_results = {
            'test_type': 'cognitive_enhancement',
            'sub_tests': [cognitive_results, business_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [cognitive_results, business_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [cognitive_results, business_results])
        }

        return combined_results

    async def _run_pattern_tests(self) -> Dict[str, Any]:
        """Run pattern recognition tests"""
        if not hasattr(self, 'pattern_tester'):
            return {'test_type': 'pattern_recognition', 'error': 'Engine not initialized', 'status': 'failed'}

        pattern_results = await self.pattern_tester.test_pattern_recognition()
        temporal_results = await self.pattern_tester.test_temporal_patterns()
        behavioral_results = await self.pattern_tester.test_behavioral_patterns()

        # Combine all pattern test results
        combined_results = {
            'test_type': 'pattern_recognition',
            'sub_tests': [pattern_results, temporal_results, behavioral_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [pattern_results, temporal_results, behavioral_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [pattern_results, temporal_results, behavioral_results])
        }

        return combined_results

    async def _run_ml_tests(self) -> Dict[str, Any]:
        """Run ML ensemble learning tests"""
        if not hasattr(self, 'ml_tester'):
            return {'test_type': 'ml_ensemble_learning', 'error': 'Engine not initialized', 'status': 'failed'}

        ml_results = await self.ml_tester.test_ml_ensemble_learning()
        training_results = await self.ml_tester.test_model_training()
        predictive_results = await self.ml_tester.test_predictive_accuracy()

        # Combine all ML test results
        combined_results = {
            'test_type': 'ml_ensemble_learning',
            'sub_tests': [ml_results, training_results, predictive_results],
            'total_tests': sum(r.get('total_tests', 0) for r in [ml_results, training_results, predictive_results]),
            'successful_tests': sum(r.get('successful_tests', 0) for r in [ml_results, training_results, predictive_results])
        }

        return combined_results

    def _generate_test_summary(self, test_categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from test results"""
        total_tests = 0
        successful_tests = 0
        successful_categories = 0

        for category in test_categories:
            if 'error' not in category:
                successful_categories += 1
                total_tests += category.get('total_tests', 0)
                successful_tests += category.get('successful_tests', 0)

        return {
            'total_test_categories': len(test_categories),
            'successful_categories': successful_categories,
            'total_individual_tests': total_tests,
            'successful_individual_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0
        }

    def save_test_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_decision_engine_test_results_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"📁 Test results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return None

    def print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "="*80)
        print("ENHANCED AUTONOMOUS DECISION ENGINE TEST SUMMARY")
        print("="*80)

        summary = results.get('summary', {})
        execution = results.get('test_execution', {})

        print(f"[OK] Test Categories: {summary.get('successful_categories', 0)}/{summary.get('total_test_categories', 0)}")
        print(f"[OK] Individual Tests: {summary.get('successful_individual_tests', 0)}/{summary.get('total_individual_tests', 0)}")
        print(f"[STATS] Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"[TIME] Duration: {execution.get('duration', 'Unknown')}")
        print("="*80)


async def main():
    """Main test execution function"""
    test_suite = EnhancedDecisionEngineTestSuite()

    try:
        # Run comprehensive test suite
        results = await test_suite.run_comprehensive_test_suite()

        # Save results to file
        results_file = test_suite.save_test_results(results)

        # Print summary
        test_suite.print_summary(results)

        return results

    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return {'error': str(e), 'status': 'failed'}


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
