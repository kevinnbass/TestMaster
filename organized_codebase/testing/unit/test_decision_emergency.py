#!/usr/bin/env python3
"""
Test Decision Engine Emergency Tests
====================================

Emergency response decision testing functionality for the enhanced autonomous decision engine.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import logging
from typing import Dict, List, Any

from autonomous_decision_engine import (
    DecisionType,
    DecisionUrgency
)
from test_decision_models import TestConfiguration, TestResultProcessor

logger = logging.getLogger(__name__)


class EmergencyResponseTester:
    """Handles emergency response decision testing"""

    def __init__(self, engine):
        """Initialize the emergency response tester"""
        self.engine = engine

    async def test_emergency_response_decisions(self) -> Dict[str, Any]:
        """Test emergency response decisions"""
        logger.info("🚨 Testing Emergency Response Decisions...")

        test_contexts = TestConfiguration.get_emergency_test_contexts()
        results = []

        for test_case in test_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.EMERGENCY_RESPONSE,
                    test_case['context'],
                    DecisionUrgency.EMERGENCY
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_action': test_case.get('expected_action'),
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'urgency': decision.urgency.value,
                    'risk_assessment': decision.risk_assessment,
                    'rollback_feasible': decision.rollback_plan.get('rollback_feasible', False) if decision.rollback_plan else False
                })

                logger.info(f"    🚨 Emergency Action: {result['selected_action']} (urgency: {result.get('urgency', 'unknown')})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Emergency test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'emergency_response',
            'total_tests': len(test_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_disaster_recovery(self) -> Dict[str, Any]:
        """Test disaster recovery scenarios"""
        logger.info("🌪️ Testing Disaster Recovery Scenarios...")

        disaster_contexts = [
            {
                'name': 'Data Center Outage',
                'context': {
                    'primary_dc_down': True,
                    'secondary_dc_available': True,
                    'data_loss_risk': 0.05,
                    'rto_requirement': 300,  # 5 minutes
                    'rpo_requirement': 60,   # 1 minute data loss acceptable
                    'affected_services': ['web', 'api', 'database'],
                    'source': 'infrastructure_monitor'
                },
                'expected_action': 'failover_to_secondary'
            },
            {
                'name': 'Complete System Crash',
                'context': {
                    'all_systems_down': True,
                    'backup_available': True,
                    'data_backup_age': 1800,  # 30 minutes old
                    'estimated_recovery_time': 3600,  # 1 hour
                    'business_impact': 'critical',
                    'source': 'system_monitor'
                },
                'expected_action': 'full_system_restore'
            },
            {
                'name': 'Network Partition',
                'context': {
                    'network_partition': True,
                    'split_brain_risk': True,
                    'inconsistent_data': True,
                    'service_isolation': True,
                    'communication_failure': True,
                    'source': 'network_monitor'
                },
                'expected_action': 'network_healing'
            }
        ]

        results = []

        for test_case in disaster_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.EMERGENCY_RESPONSE,
                    test_case['context'],
                    DecisionUrgency.CRITICAL
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_action': test_case['expected_action'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'urgency': decision.urgency.value,
                    'data_loss_risk': test_case['context'].get('data_loss_risk', 0),
                    'recovery_time_estimate': test_case['context'].get('estimated_recovery_time', 0)
                })

                logger.info(f"    🌪️ Disaster Recovery: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Disaster recovery test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'disaster_recovery',
            'total_tests': len(disaster_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }

    async def test_incident_response(self) -> Dict[str, Any]:
        """Test incident response and mitigation"""
        logger.info("🔍 Testing Incident Response...")

        incident_contexts = [
            {
                'name': 'DDoS Attack',
                'context': {
                    'traffic_spike': True,
                    'traffic_pattern': 'suspicious',
                    'request_rate': 10000,
                    'normal_rate': 1000,
                    'attack_vector': 'UDP_flood',
                    'mitigation_available': True,
                    'source': 'security_monitor'
                },
                'expected_action': 'activate_ddos_protection'
            },
            {
                'name': 'Data Exfiltration',
                'context': {
                    'unusual_data_access': True,
                    'large_data_transfers': True,
                    'unauthorized_access': True,
                    'data_sensitivity': 'high',
                    'containment_status': 'detected',
                    'source': 'data_loss_prevention'
                },
                'expected_action': 'data_exfiltration_response'
            },
            {
                'name': 'Service Degradation',
                'context': {
                    'response_time_degradation': True,
                    'error_rate_increase': True,
                    'resource_exhaustion': True,
                    'cascading_failure': False,
                    'auto_recovery_available': True,
                    'source': 'service_monitor'
                },
                'expected_action': 'service_restoration'
            }
        ]

        results = []

        for test_case in incident_contexts:
            logger.info(f"  Testing: {test_case['name']}")

            try:
                decision = await self.engine.make_enhanced_decision(
                    DecisionType.EMERGENCY_RESPONSE,
                    test_case['context'],
                    DecisionUrgency.HIGH
                )

                result = TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'decision_id': decision.decision_id,
                    'selected_action': decision.selected_option.name,
                    'expected_action': test_case['expected_action'],
                    'confidence': decision.confidence_score,
                    'risk_score': decision.selected_option.risk_score,
                    'status': decision.status.value,
                    'urgency': decision.urgency.value,
                    'attack_vector': test_case['context'].get('attack_vector'),
                    'mitigation_available': test_case['context'].get('mitigation_available', False)
                })

                logger.info(f"    🔍 Incident Response: {result['selected_action']} (confidence: {result['confidence']:.2f})")
                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Incident response test failed: {e}")
                results.append(TestResultProcessor.format_test_result({
                    'test_name': test_case['name'],
                    'error': str(e),
                    'status': 'failed'
                }))

        return {
            'test_type': 'incident_response',
            'total_tests': len(incident_contexts),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'results': results,
            'performance_summary': TestResultProcessor.generate_performance_summary(results)
        }
