#!/usr/bin/env python3
"""
Test Decision Engine Pattern Recognition Tests
==============================================

Pattern recognition decision testing functionality for the enhanced autonomous decision engine.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import logging
from typing import Dict, List, Any

from autonomous_decision_engine import (
    DecisionType
)
from test_decision_models import TestConfiguration, TestResultProcessor

logger = logging.getLogger(__name__)


class PatternRecognitionTester:
    """Handles pattern recognition decision testing"""

    def __init__(self, engine):
        """Initialize the pattern recognition tester"""
        self.engine = engine

    async def test_pattern_recognition(self) -> Dict[str, Any]:
        """Test pattern recognition capabilities"""
        logger.info("🔍 Testing Pattern Recognition...")

        test_contexts = TestConfiguration.get_pattern_test_contexts()
        results = []

        for test_case in test_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                # Make decision with pattern recognition enabled
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_pattern_recognition=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_pattern': test_case['expected_pattern'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'pattern_detected': 'pattern_insights' in decision.context,
                    'pattern_confidence': decision.context.get('pattern_confidence', 0.0),
                    'similarity_score': test_case['context']['similarity_score'],
                    'historical_data_points': test_case['context']['historical_data_points'],
                    'pattern_type': self._determine_pattern_type(decision)
                })

                logger.info(f"    ✅ Pattern Recognition: {result['pattern_type']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Pattern recognition test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'pattern_recognition',
            'total_tests': len(test_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    def _determine_pattern_type(self, decision) -> str:
        """Determine the type of pattern detected"""
        context = decision.context

        if 'pattern_insights' in context:
            pattern_data = context['pattern_insights']

            if pattern_data.get('pattern_type') == 'cyclical':
                return 'daily_cycle'
            elif pattern_data.get('pattern_type') == 'anomaly':
                return 'anomaly_detected'
            elif pattern_data.get('pattern_type') == 'trend':
                return 'trend_analysis'
            else:
                return 'pattern_recognized'

        return 'no_pattern_detected'

    async def test_temporal_patterns(self) -> Dict[str, Any]:
        """Test temporal pattern recognition"""
        logger.info("⏰ Testing Temporal Pattern Recognition...")

        temporal_contexts = [
            {
                'name': 'Business Hours Pattern',
                'context': {
                    'time_of_day': '09:00',
                    'day_of_week': 'monday',
                    'traffic_pattern': 'business_hours_start',
                    'similarity_score': 0.9,
                    'historical_data_points': 200,
                    'peak_hours': True,
                    'source': 'temporal_analyzer'
                },
                'expected_pattern': 'business_hours_start'
            },
            {
                'name': 'Weekend Pattern',
                'context': {
                    'time_of_day': '14:00',
                    'day_of_week': 'saturday',
                    'traffic_pattern': 'weekend_activity',
                    'similarity_score': 0.7,
                    'historical_data_points': 50,
                    'off_peak_hours': True,
                    'source': 'temporal_analyzer'
                },
                'expected_pattern': 'weekend_activity'
            },
            {
                'name': 'Maintenance Window',
                'context': {
                    'time_of_day': '02:00',
                    'day_of_week': 'sunday',
                    'traffic_pattern': 'maintenance_window',
                    'similarity_score': 0.95,
                    'historical_data_points': 150,
                    'maintenance_window': True,
                    'source': 'temporal_analyzer'
                },
                'expected_pattern': 'maintenance_window'
            }
        ]

        results = []

        for test_case in temporal_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_pattern_recognition=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_pattern': test_case['expected_pattern'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'time_of_day': test_case['context']['time_of_day'],
                    'day_of_week': test_case['context']['day_of_week'],
                    'temporal_pattern_detected': 'temporal_insights' in decision.context,
                    'business_hours_detected': test_case['context'].get('peak_hours', False),
                    'maintenance_window_detected': test_case['context'].get('maintenance_window', False)
                })

                logger.info(f"    ✅ Temporal Pattern: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Temporal pattern test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'temporal_patterns',
            'total_tests': len(temporal_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_behavioral_patterns(self) -> Dict[str, Any]:
        """Test user behavior pattern recognition"""
        logger.info("👥 Testing Behavioral Pattern Recognition...")

        behavioral_contexts = [
            {
                'name': 'User Behavior Spike',
                'context': {
                    'user_activity': 'high_engagement',
                    'session_duration': 450,
                    'page_views': 25,
                    'interaction_rate': 0.8,
                    'behavior_pattern': 'power_user',
                    'source': 'behavior_analyzer'
                },
                'expected_pattern': 'power_user_behavior'
            },
            {
                'name': 'Abnormal User Behavior',
                'context': {
                    'user_activity': 'suspicious',
                    'session_duration': 120,
                    'page_views': 50,
                    'interaction_rate': 0.1,
                    'behavior_pattern': 'bot_activity',
                    'source': 'behavior_analyzer'
                },
                'expected_pattern': 'suspicious_activity'
            },
            {
                'name': 'Normal User Behavior',
                'context': {
                    'user_activity': 'normal',
                    'session_duration': 180,
                    'page_views': 8,
                    'interaction_rate': 0.4,
                    'behavior_pattern': 'casual_user',
                    'source': 'behavior_analyzer'
                },
                'expected_pattern': 'normal_usage'
            }
        ]

        results = []

        for test_case in behavioral_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_pattern_recognition=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_pattern': test_case['expected_pattern'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'user_activity': test_case['context']['user_activity'],
                    'behavior_pattern': test_case['context']['behavior_pattern'],
                    'behavioral_insights': 'behavioral_insights' in decision.context,
                    'engagement_score': test_case['context']['interaction_rate']
                })

                logger.info(f"    ✅ Behavioral Pattern: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Behavioral pattern test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'behavioral_patterns',
            'total_tests': len(behavioral_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }
