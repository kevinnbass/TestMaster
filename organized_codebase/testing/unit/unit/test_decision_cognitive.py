#!/usr/bin/env python3
"""
Test Decision Engine Cognitive Tests
====================================

Cognitive enhancement decision testing functionality for the enhanced autonomous decision engine.

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


class CognitiveEnhancementTester:
    """Handles cognitive enhancement decision testing"""

    def __init__(self, engine):
        """Initialize the cognitive enhancement tester"""
        self.engine = engine

    async def test_cognitive_enhancement(self) -> Dict[str, Any]:
        """Test cognitive enhancement features"""
        logger.info("🧠 Testing Cognitive Enhancement...")

        test_contexts = TestConfiguration.get_cognitive_test_contexts()
        results = []

        for test_case in test_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                # Test with cognitive enhancement enabled
                cognitive_decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=True
                )

                # Test without cognitive enhancement
                standard_decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=False
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'cognitive_decision_id': cognitive_decision.decision_id,
                    'standard_decision_id': standard_decision.decision_id,
                    'cognitive_confidence': cognitive_decision.confidence_score,
                    'standard_confidence': standard_decision.confidence_score,
                    'cognitive_risk': cognitive_decision.selected_option.risk_score,
                    'standard_risk': standard_decision.selected_option.risk_score,
                    'cognitive_selected_option': cognitive_decision.selected_option.name,
                    'standard_selected_option': standard_decision.selected_option.name,
                    'expected_enhancement': test_case['expected_enhancement'],
                    'confidence_improvement': cognitive_decision.confidence_score - standard_decision.confidence_score,
                    'risk_reduction': standard_decision.selected_option.risk_score - cognitive_decision.selected_option.risk_score,
                    'cognitive_features_detected': 'cognitive_insights' in cognitive_decision.context
                })

                logger.info(f"    ✅ Cognitive enhancement improved confidence by: {result['confidence_improvement']:.3f}")
                logger.info(f"    ✅ Risk reduction achieved: {result['risk_reduction']:.3f}")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Cognitive enhancement test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'cognitive_enhancement',
            'total_tests': len(test_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_business_logic_cognition(self) -> Dict[str, Any]:
        """Test cognitive understanding of business logic"""
        logger.info("💼 Testing Business Logic Cognition...")

        business_contexts = [
            {
                'name': 'Complex Business Rules',
                'context': {
                    'business_rules_count': 150,
                    'decision_complexity': 0.8,
                    'historical_accuracy': 0.75,
                    'pattern_similarity': 0.6,
                    'stakeholder_requirements': 25,
                    'compliance_constraints': 10,
                    'source': 'business_analyzer'
                },
                'expected_cognitive_benefit': True
            },
            {
                'name': 'Simple Operational Rules',
                'context': {
                    'business_rules_count': 5,
                    'decision_complexity': 0.2,
                    'historical_accuracy': 0.95,
                    'pattern_similarity': 0.9,
                    'stakeholder_requirements': 2,
                    'compliance_constraints': 1,
                    'source': 'operations_monitor'
                },
                'expected_cognitive_benefit': False
            },
            {
                'name': 'Regulatory Compliance',
                'context': {
                    'business_rules_count': 75,
                    'decision_complexity': 0.6,
                    'historical_accuracy': 0.85,
                    'pattern_similarity': 0.4,
                    'stakeholder_requirements': 15,
                    'compliance_constraints': 25,
                    'source': 'compliance_monitor'
                },
                'expected_cognitive_benefit': True
            }
        ]

        results = []

        for test_case in business_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                # Test with cognitive enhancement
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'expected_cognitive_benefit': test_case['expected_cognitive_benefit'],
                    'business_rules_processed': test_case['context']['business_rules_count'],
                    'decision_complexity': test_case['context']['decision_complexity'],
                    'cognitive_insights_generated': 'cognitive_insights' in decision.context,
                    'compliance_considered': test_case['context']['compliance_constraints'] > 5
                })

                logger.info(f"    ✅ Business Logic Cognition: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Business logic cognition test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'business_logic_cognition',
            'total_tests': len(business_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }
