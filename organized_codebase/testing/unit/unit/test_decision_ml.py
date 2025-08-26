#!/usr/bin/env python3
"""
Test Decision Engine ML Ensemble Tests
======================================

ML ensemble learning decision testing functionality for the enhanced autonomous decision engine.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import logging
from typing import Dict, List, Any

from autonomous_decision_engine import (
    DecisionType,
    DecisionStatus
)
from test_decision_models import TestConfiguration, TestResultProcessor

logger = logging.getLogger(__name__)


class MLEnsembleTester:
    """Handles ML ensemble learning decision testing"""

    def __init__(self, engine):
        """Initialize the ML ensemble tester"""
        self.engine = engine

    async def test_ml_ensemble_learning(self) -> Dict[str, Any]:
        """Test ML ensemble learning capabilities"""
        logger.info("🤖 Testing ML Ensemble Learning...")

        test_contexts = TestConfiguration.get_ml_test_contexts()
        results = []

        for test_case in test_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                # Test with ML ensemble enabled
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_ml_ensemble=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_ensemble': test_case['expected_ensemble'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'models_available': test_case['context']['models_available'],
                    'training_data_size': test_case['context']['training_data_size'],
                    'ml_enhanced': 'ml_success_probability' in decision.selected_option.expected_outcomes,
                    'ml_probability': decision.selected_option.expected_outcomes.get('ml_success_probability', 0.0),
                    'ensemble_used': hasattr(self.engine, 'decision_classifiers') and self.engine.decision_classifiers is not None
                })

                logger.info(f"    ✅ ML Ensemble: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ ML ensemble test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'ml_ensemble_learning',
            'total_tests': len(test_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_model_training(self) -> Dict[str, Any]:
        """Test ML model training and learning"""
        logger.info("🎯 Testing ML Model Training...")

        training_contexts = [
            {
                'name': 'Supervised Learning Training',
                'context': {
                    'models_available': 3,
                    'training_data_size': 1000,
                    'feature_count': 15,
                    'target_accuracy': 0.85,
                    'learning_algorithm': 'gradient_boosting',
                    'source': 'model_trainer'
                },
                'expected_training_success': True
            },
            {
                'name': 'Unsupervised Learning Training',
                'context': {
                    'models_available': 2,
                    'training_data_size': 5000,
                    'feature_count': 20,
                    'target_accuracy': 0.75,
                    'learning_algorithm': 'clustering',
                    'source': 'model_trainer'
                },
                'expected_training_success': True
            },
            {
                'name': 'Reinforcement Learning Training',
                'context': {
                    'models_available': 1,
                    'training_data_size': 100,
                    'feature_count': 8,
                    'target_accuracy': 0.65,
                    'learning_algorithm': 'q_learning',
                    'source': 'model_trainer'
                },
                'expected_training_success': False  # RL needs more data
            }
        ]

        results = []

        for test_case in training_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                # Test training capability
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.PERFORMANCE_OPTIMIZATION,
                    test_case['context'],
                    enable_ml_ensemble=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_training_success': test_case['expected_training_success'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'training_data_size': test_case['context']['training_data_size'],
                    'learning_algorithm': test_case['context']['learning_algorithm'],
                    'model_training_successful': 'model_accuracy' in decision.selected_option.expected_outcomes,
                    'achieved_accuracy': decision.selected_option.expected_outcomes.get('model_accuracy', 0.0)
                })

                logger.info(f"    ✅ Model Training: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Model training test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'model_training',
            'total_tests': len(training_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_predictive_accuracy(self) -> Dict[str, Any]:
        """Test predictive accuracy of ML models"""
        logger.info("🔮 Testing Predictive Accuracy...")

        prediction_contexts = [
            {
                'name': 'Short-term Prediction',
                'context': {
                    'prediction_horizon': '1_hour',
                    'historical_data_points': 720,
                    'prediction_confidence_threshold': 0.8,
                    'model_type': 'time_series',
                    'feature_importance_available': True,
                    'source': 'predictor_tester'
                },
                'expected_accuracy': 0.85
            },
            {
                'name': 'Long-term Prediction',
                'context': {
                    'prediction_horizon': '24_hours',
                    'historical_data_points': 8760,
                    'prediction_confidence_threshold': 0.6,
                    'model_type': 'trend_analysis',
                    'feature_importance_available': True,
                    'source': 'predictor_tester'
                },
                'expected_accuracy': 0.65
            },
            {
                'name': 'Real-time Prediction',
                'context': {
                    'prediction_horizon': '5_minutes',
                    'historical_data_points': 60,
                    'prediction_confidence_threshold': 0.9,
                    'model_type': 'real_time',
                    'feature_importance_available': False,
                    'source': 'predictor_tester'
                },
                'expected_accuracy': 0.95
            }
        ]

        results = []

        for test_case in prediction_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_ml_ensemble=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_accuracy': test_case['expected_accuracy'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'prediction_horizon': test_case['context']['prediction_horizon'],
                    'historical_data_points': test_case['context']['historical_data_points'],
                    'predictive_accuracy': decision.selected_option.expected_outcomes.get('prediction_accuracy', 0.0),
                    'feature_importance_used': test_case['context']['feature_importance_available']
                })

                logger.info(f"    ✅ Predictive Accuracy: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Predictive accuracy test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'predictive_accuracy',
            'total_tests': len(prediction_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }
