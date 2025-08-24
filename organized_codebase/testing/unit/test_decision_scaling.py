#!/usr/bin/env python3
"""
Test Decision Engine Scaling Tests
==================================

Scaling decision testing functionality for the enhanced autonomous decision engine.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import logging
from typing import Dict, List, Any

from autonomous_decision_engine import (
    DecisionType,
)
from test_decision_models import TestConfiguration, TestResultProcessor

logger = logging.getLogger(__name__)


class ScalingDecisionTester:
    """Handles scaling decision testing"""

    def __init__(self, engine):
        """Initialize the scaling decision tester"""
        self.engine = engine

    async def test_scaling_decisions(self) -> Dict[str, Any]:
        """Test autonomous scaling decisions"""
        logger.info("🔧 Testing Scaling Decision Making...")

        test_contexts = TestConfiguration.get_scaling_test_contexts()
        results = []

        for test_case in test_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.parameters.get('action'),
                    'expected_action': test_case['expected_action'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'cognitive_enhanced': 'cognitive_insights' in test_case['context'],
                    'execution_time': decision.execution_results.get('execution_time', 0) if decision.execution_results else 0
                })

                logger.info(f"    ✅ Decision: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'scaling_decisions',
            'total_tests': len(test_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_scaling_under_load(self) -> Dict[str, Any]:
        """Test scaling decisions under high load conditions"""
        logger.info("🔥 Testing Scaling Under Load...")

        # High load test contexts
        high_load_contexts = [
            {
                'name': 'Extreme CPU Load',
                'context': {
                    'cpu_usage': 0.95,
                    'memory_usage': 0.88,
                    'avg_response_time': 500,
                    'requests_per_second': 500,
                    'error_percentage': 0.15,
                    'concurrent_users': 10000,
                    'source': 'load_testing'
                },
                'expected_action': 'emergency_scale'
            },
            {
                'name': 'Memory Exhaustion',
                'context': {
                    'cpu_usage': 0.75,
                    'memory_usage': 0.96,
                    'avg_response_time': 300,
                    'requests_per_second': 200,
                    'error_percentage': 0.08,
                    'memory_leaks_detected': True,
                    'source': 'memory_monitor'
                },
                'expected_action': 'memory_optimization'
            }
        ]

        results = []

        for test_case in high_load_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=True
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.parameters.get('action'),
                    'expected_action': test_case['expected_action'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'cognitive_enhanced': True,
                    'execution_time': decision.execution_results.get('execution_time', 0) if decision.execution_results else 0
                })

                logger.info(f"    ✅ Decision: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'scaling_under_load',
            'total_tests': len(high_load_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_cost_optimization_scaling(self) -> Dict[str, Any]:
        """Test cost-optimized scaling decisions"""
        logger.info("💰 Testing Cost-Optimized Scaling...")

        cost_contexts = [
            {
                'name': 'Over-Provisioned Resources',
                'context': {
                    'cpu_usage': 0.15,
                    'memory_usage': 0.20,
                    'cost_per_hour': 25.50,
                    'utilization_efficiency': 0.25,
                    'idle_time_percentage': 85,
                    'source': 'cost_analyzer'
                },
                'expected_action': 'scale_down_aggressive'
            },
            {
                'name': 'Optimal Resource Usage',
                'context': {
                    'cpu_usage': 0.65,
                    'memory_usage': 0.70,
                    'cost_per_hour': 18.75,
                    'utilization_efficiency': 0.85,
                    'idle_time_percentage': 15,
                    'source': 'efficiency_monitor'
                },
                'expected_action': 'maintain_current'
            }
        ]

        results = []

        for test_case in cost_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.SCALING_DECISION,
                    test_case['context'],
                    enable_cognitive_reasoning=False  # Cost decisions might not need cognitive enhancement
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.parameters.get('action'),
                    'expected_action': test_case['expected_action'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'cognitive_enhanced': False,
                    'execution_time': decision.execution_results.get('execution_time', 0) if decision.execution_results else 0
                })

                logger.info(f"    ✅ Decision: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'cost_optimization_scaling',
            'total_tests': len(cost_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }
