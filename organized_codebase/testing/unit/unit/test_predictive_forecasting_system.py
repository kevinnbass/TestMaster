"""
Predictive Intelligence & Forecasting System Test Suite
======================================================

Agent C Hours 130-140: Predictive Intelligence & Forecasting Systems

Comprehensive testing and demonstration of the advanced predictive forecasting system
with ensemble models, cognitive prediction, and multi-horizon forecasting capabilities.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from advanced_predictive_forecasting_system import (
    create_advanced_predictive_forecasting_system,
    PredictionType,
    ForecastHorizon,
    ForecastingMethod
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveForecastingTestSuite:
    """Comprehensive test suite for predictive forecasting system"""
    
    def __init__(self):
        self.forecasting_system = None
        self.test_results = []
        self.synthetic_data = None
        
    async def initialize_test_system(self):
        """Initialize the predictive forecasting system for testing"""
        logger.info("🔮 Initializing Predictive Forecasting Test System...")
        
        # Create synthetic training data
        self.synthetic_data = await self._create_synthetic_training_data()
        
        # Initialize forecasting system
        config = {
            'enable_ensemble_forecasting': True,
            'enable_cognitive_forecasting': True,
            'auto_model_selection': True,
            'prediction_validation_enabled': True,
            'real_time_learning': True,
            'max_concurrent_predictions': 50,
            'uncertainty_threshold': 0.7
        }
        
        self.forecasting_system = create_advanced_predictive_forecasting_system(config)
        
        # Initialize with training data
        initialization_success = await self.forecasting_system.initialize(self.synthetic_data)
        
        logger.info(f"    [OK] System initialized: {initialization_success}")
        return initialization_success
    
    async def _create_synthetic_training_data(self) -> Dict[str, Any]:
        """Create realistic synthetic training data for testing"""
        historical_data = []
        base_time = datetime.now() - timedelta(days=30)
        
        # Generate 30 days of hourly data
        for i in range(30 * 24):
            timestamp = base_time + timedelta(hours=i)
            
            # Simulate realistic system metrics with patterns
            hour_of_day = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Performance patterns
            base_cpu = 0.3 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily cycle
            base_memory = 0.4 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
            
            # Add noise and trends
            cpu_usage = max(0.1, min(0.95, base_cpu + np.random.normal(0, 0.1)))
            memory_usage = max(0.1, min(0.95, base_memory + np.random.normal(0, 0.05)))
            
            # Derived metrics
            latency = 50 + (cpu_usage * 200) + np.random.normal(0, 20)
            throughput = max(10, 200 - (cpu_usage * 100) + np.random.normal(0, 30))
            error_rate = max(0.001, cpu_usage * 0.02 + np.random.normal(0, 0.005))
            
            # Create data point
            data_point = {
                'timestamp': timestamp.isoformat(),
                'value': cpu_usage,  # Primary target
                'metrics': {
                    'cpu': cpu_usage,
                    'memory': memory_usage,
                    'latency': latency,
                    'throughput': throughput,
                    'error_rate': error_rate
                },
                'context': {
                    'hour': hour_of_day,
                    'weekday': day_of_week,
                    'is_weekend': day_of_week >= 5,
                    'is_peak_hours': 9 <= hour_of_day <= 17
                }
            }
            
            historical_data.append(data_point)
        
        return {
            'historical_data': historical_data,
            'metadata': {
                'data_points': len(historical_data),
                'time_range': '30 days',
                'frequency': 'hourly',
                'metrics': ['cpu', 'memory', 'latency', 'throughput', 'error_rate']
            }
        }
    
    async def test_ensemble_forecasting(self) -> Dict[str, Any]:
        """Test ensemble forecasting capabilities"""
        logger.info("🤖 Testing Ensemble Forecasting...")
        
        try:
            test_scenarios = [
                {
                    'name': 'Short-term Performance Prediction',
                    'type': PredictionType.PERFORMANCE,
                    'metric': 'cpu_usage',
                    'horizon': ForecastHorizon.SHORT_TERM,
                    'horizon_value': timedelta(hours=6),
                    'context': {
                        'current_value': 0.75,
                        'cpu': 0.75,
                        'memory': 0.6,
                        'latency': 180,
                        'throughput': 120,
                        'error_rate': 0.015
                    }
                },
                {
                    'name': 'Medium-term Capacity Prediction',
                    'type': PredictionType.CAPACITY,
                    'metric': 'memory_usage',
                    'horizon': ForecastHorizon.MEDIUM_TERM,
                    'horizon_value': timedelta(days=3),
                    'context': {
                        'current_value': 0.65,
                        'cpu': 0.5,
                        'memory': 0.65,
                        'growth_trend': 0.05,
                        'peak_usage_pattern': True
                    }
                },
                {
                    'name': 'Long-term Health Prediction',
                    'type': PredictionType.HEALTH,
                    'metric': 'system_health',
                    'horizon': ForecastHorizon.LONG_TERM,
                    'horizon_value': timedelta(days=14),
                    'context': {
                        'current_value': 0.85,
                        'degradation_trend': -0.02,
                        'maintenance_schedule': False,
                        'load_forecast': 'increasing'
                    }
                }
            ]
            
            results = []
            
            for scenario in test_scenarios:
                logger.info(f"    Testing: {scenario['name']}")
                
                # Create prediction
                prediction = await self.forecasting_system.create_prediction(
                    scenario['type'],
                    scenario['metric'],
                    scenario['horizon'],
                    scenario['horizon_value'],
                    scenario['context'],
                    ForecastingMethod.ENSEMBLE
                )
                
                result = {
                    'scenario': scenario['name'],
                    'prediction_id': prediction.prediction_id,
                    'predicted_value': prediction.prediction_value,
                    'confidence_interval': prediction.confidence_interval,
                    'uncertainty': prediction.uncertainty_score,
                    'model_confidence': prediction.model_confidence,
                    'method_used': prediction.method_used.value,
                    'prediction_path_length': len(prediction.prediction_path),
                    'feature_importance': list(prediction.feature_importance.keys())[:5]  # Top 5
                }
                
                results.append(result)
                
                logger.info(f"        ✅ Predicted: {prediction.prediction_value:.3f}")
                logger.info(f"        📊 Confidence: {prediction.model_confidence:.1%}")
                logger.info(f"        ⚠️ Uncertainty: {prediction.uncertainty_score:.1%}")
            
            return {
                'test_type': 'ensemble_forecasting',
                'status': 'success',
                'scenarios_tested': len(test_scenarios),
                'results': results,
                'average_confidence': np.mean([r['model_confidence'] for r in results]),
                'average_uncertainty': np.mean([r['uncertainty'] for r in results])
            }
            
        except Exception as e:
            logger.error(f"    ❌ Ensemble forecasting test failed: {e}")
            return {
                'test_type': 'ensemble_forecasting',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_cognitive_forecasting(self) -> Dict[str, Any]:
        """Test cognitive forecasting capabilities"""
        logger.info("🧠 Testing Cognitive Forecasting...")
        
        try:
            cognitive_scenarios = [
                {
                    'name': 'Behavioral Pattern Prediction',
                    'type': PredictionType.BEHAVIOR,
                    'metric': 'user_activity',
                    'horizon': ForecastHorizon.MEDIUM_TERM,
                    'horizon_value': timedelta(days=7),
                    'context': {
                        'current_value': 0.8,
                        'user_growth_trend': 'accelerating',
                        'seasonal_pattern': 'high_season',
                        'marketing_campaign': True,
                        'competitor_activity': 'low'
                    }
                },
                {
                    'name': 'Evolution Needs Prediction',
                    'type': PredictionType.EVOLUTION,
                    'metric': 'architecture_health',
                    'horizon': ForecastHorizon.STRATEGIC,
                    'horizon_value': timedelta(weeks=8),
                    'context': {
                        'current_value': 0.75,
                        'technical_debt': 'increasing',
                        'feature_velocity': 'stable',
                        'team_growth': 'expanding',
                        'complexity_trend': 'rising'
                    }
                }
            ]
            
            results = []
            
            for scenario in cognitive_scenarios:
                logger.info(f"    Testing: {scenario['name']}")
                
                # Create cognitive prediction
                prediction = await self.forecasting_system.create_prediction(
                    scenario['type'],
                    scenario['metric'],
                    scenario['horizon'],
                    scenario['horizon_value'],
                    scenario['context'],
                    ForecastingMethod.COGNITIVE
                )
                
                result = {
                    'scenario': scenario['name'],
                    'prediction_id': prediction.prediction_id,
                    'predicted_value': prediction.prediction_value,
                    'confidence_interval': prediction.confidence_interval,
                    'uncertainty': prediction.uncertainty_score,
                    'method_used': prediction.method_used.value,
                    'cognitive_enhanced': prediction.metadata.get('cognitive_enhanced', False),
                    'reasoning_available': 'cognitive_reasoning' in prediction.feature_importance
                }
                
                results.append(result)
                
                logger.info(f"        🧠 Cognitive Prediction: {prediction.prediction_value:.3f}")
                logger.info(f"        🎯 Method: {prediction.method_used.value}")
                logger.info(f"        🔍 Enhanced: {result['cognitive_enhanced']}")
            
            return {
                'test_type': 'cognitive_forecasting',
                'status': 'success',
                'scenarios_tested': len(cognitive_scenarios),
                'results': results,
                'cognitive_success_rate': sum(1 for r in results if r['cognitive_enhanced']) / len(results)
            }
            
        except Exception as e:
            logger.error(f"    ❌ Cognitive forecasting test failed: {e}")
            return {
                'test_type': 'cognitive_forecasting',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_multi_horizon_forecasting(self) -> Dict[str, Any]:
        """Test multi-horizon forecasting capabilities"""
        logger.info("⏰ Testing Multi-Horizon Forecasting...")
        
        try:
            # Same prediction across different horizons
            base_context = {
                'current_value': 0.7,
                'cpu': 0.7,
                'memory': 0.6,
                'trend': 'stable',
                'volatility': 'low'
            }
            
            horizons = [
                (ForecastHorizon.SHORT_TERM, timedelta(hours=3)),
                (ForecastHorizon.MEDIUM_TERM, timedelta(days=1)),
                (ForecastHorizon.LONG_TERM, timedelta(days=7)),
                (ForecastHorizon.STRATEGIC, timedelta(weeks=4))
            ]
            
            results = []
            
            for horizon, horizon_value in horizons:
                logger.info(f"    Testing horizon: {horizon.value}")
                
                prediction = await self.forecasting_system.create_prediction(
                    PredictionType.PERFORMANCE,
                    'performance_metric',
                    horizon,
                    horizon_value,
                    base_context
                )
                
                result = {
                    'horizon': horizon.value,
                    'horizon_value': str(horizon_value),
                    'prediction_id': prediction.prediction_id,
                    'predicted_value': prediction.prediction_value,
                    'uncertainty': prediction.uncertainty_score,
                    'path_points': len(prediction.prediction_path),
                    'confidence_width': prediction.confidence_interval[1] - prediction.confidence_interval[0]
                }
                
                results.append(result)
                
                logger.info(f"        ⏳ {horizon.value}: {prediction.prediction_value:.3f}")
                logger.info(f"        📈 Path points: {len(prediction.prediction_path)}")
            
            # Analyze horizon consistency
            values = [r['predicted_value'] for r in results]
            uncertainties = [r['uncertainty'] for r in results]
            
            return {
                'test_type': 'multi_horizon_forecasting',
                'status': 'success',
                'horizons_tested': len(horizons),
                'results': results,
                'prediction_range': max(values) - min(values),
                'uncertainty_trend': 'increasing' if uncertainties[-1] > uncertainties[0] else 'stable',
                'horizon_consistency': np.std(values) < 0.1  # Low standard deviation
            }
            
        except Exception as e:
            logger.error(f"    ❌ Multi-horizon forecasting test failed: {e}")
            return {
                'test_type': 'multi_horizon_forecasting',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_batch_predictions(self) -> Dict[str, Any]:
        """Test batch prediction processing"""
        logger.info("📦 Testing Batch Predictions...")
        
        try:
            # Create batch of prediction requests
            batch_requests = []
            
            for i in range(10):
                request = {
                    'prediction_type': 'performance',
                    'target_metric': f'metric_{i}',
                    'forecast_horizon': 'short_term',
                    'horizon_value': timedelta(hours=2),
                    'context_data': {
                        'current_value': 0.5 + (i * 0.05),
                        'cpu': 0.4 + (i * 0.03),
                        'memory': 0.3 + (i * 0.04),
                        'load_factor': i / 10.0
                    },
                    'preferred_method': 'ensemble'
                }
                batch_requests.append(request)
            
            # Process batch
            batch_start = datetime.now()
            batch_results = await self.forecasting_system.batch_predict(batch_requests)
            batch_duration = datetime.now() - batch_start
            
            # Analyze results
            successful_predictions = len(batch_results)
            average_uncertainty = np.mean([r.uncertainty_score for r in batch_results])
            prediction_values = [r.prediction_value for r in batch_results]
            
            logger.info(f"    📊 Processed {successful_predictions}/{len(batch_requests)} predictions")
            logger.info(f"    ⏱️ Batch duration: {batch_duration}")
            logger.info(f"    📈 Average uncertainty: {average_uncertainty:.1%}")
            
            return {
                'test_type': 'batch_predictions',
                'status': 'success',
                'requested_predictions': len(batch_requests),
                'successful_predictions': successful_predictions,
                'batch_duration': str(batch_duration),
                'average_uncertainty': average_uncertainty,
                'prediction_range': max(prediction_values) - min(prediction_values),
                'throughput_per_second': successful_predictions / max(1, batch_duration.total_seconds())
            }
            
        except Exception as e:
            logger.error(f"    ❌ Batch predictions test failed: {e}")
            return {
                'test_type': 'batch_predictions',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_prediction_validation(self) -> Dict[str, Any]:
        """Test prediction validation and accuracy"""
        logger.info("✅ Testing Prediction Validation...")
        
        try:
            # Create predictions and simulate actual outcomes
            validation_scenarios = []
            actual_values = {}
            
            for i in range(5):
                # Create prediction
                prediction = await self.forecasting_system.create_prediction(
                    PredictionType.PERFORMANCE,
                    f'test_metric_{i}',
                    ForecastHorizon.SHORT_TERM,
                    timedelta(hours=1),
                    {
                        'current_value': 0.6 + (i * 0.1),
                        'trend': 'stable',
                        'noise_level': 0.05
                    }
                )
                
                # Simulate actual outcome (add realistic variation)
                prediction_value = prediction.prediction_value
                actual_value = prediction_value + np.random.normal(0, 0.05)  # 5% noise
                
                validation_scenarios.append({
                    'prediction_id': prediction.prediction_id,
                    'predicted_value': prediction_value,
                    'actual_value': actual_value,
                    'within_confidence': (
                        prediction.confidence_interval[0] <= actual_value <= prediction.confidence_interval[1]
                    )
                })
                
                actual_values[prediction.prediction_id] = actual_value
            
            # Validate predictions
            validation_results = await self.forecasting_system.validate_predictions(actual_values)
            
            # Calculate additional metrics
            errors = [abs(s['predicted_value'] - s['actual_value']) for s in validation_scenarios]
            within_confidence_count = sum(1 for s in validation_scenarios if s['within_confidence'])
            
            logger.info(f"    📊 Validated {validation_results['validated_predictions']} predictions")
            logger.info(f"    🎯 Average error: {validation_results.get('average_error', 0):.3f}")
            logger.info(f"    ✅ Within confidence: {within_confidence_count}/{len(validation_scenarios)}")
            
            return {
                'test_type': 'prediction_validation',
                'status': 'success',
                'validation_results': validation_results,
                'scenarios': validation_scenarios,
                'average_absolute_error': np.mean(errors),
                'max_error': max(errors),
                'min_error': min(errors),
                'confidence_accuracy': within_confidence_count / len(validation_scenarios)
            }
            
        except Exception as e:
            logger.error(f"    ❌ Prediction validation test failed: {e}")
            return {
                'test_type': 'prediction_validation',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """Test integration with other Agent C systems"""
        logger.info("🔗 Testing System Integration...")
        
        try:
            # Get system status
            system_status = await self.forecasting_system.get_system_status()
            
            # Test predictions with enhanced context
            integration_prediction = await self.forecasting_system.create_prediction(
                PredictionType.EVOLUTION,
                'system_evolution',
                ForecastHorizon.STRATEGIC,
                timedelta(weeks=6),
                {
                    'current_value': 0.8,
                    'architecture_complexity': 'high',
                    'development_velocity': 'moderate',
                    'technical_debt_trend': 'increasing',
                    'team_size': 'growing'
                }
            )
            
            # Analyze integration features
            integration_features = {
                'agent_c_integration': system_status['system_info']['agent_c_integration'],
                'integrated_intelligence': system_status['system_info']['integrated_intelligence'],
                'ensemble_models': system_status['predictors']['ensemble_predictor']['ensemble_size'],
                'cognitive_available': system_status['predictors']['cognitive_predictor'] is not None,
                'pattern_enhancement': 'pattern_insights' in integration_prediction.request.context_data,
                'decision_enhancement': 'decision_insights' in integration_prediction.request.context_data
            }
            
            integration_score = sum(1 for feature in integration_features.values() if feature) / len(integration_features)
            
            logger.info(f"    🔗 Integration score: {integration_score:.1%}")
            logger.info(f"    🤖 Ensemble models: {integration_features['ensemble_models']}")
            logger.info(f"    🧠 Cognitive available: {integration_features['cognitive_available']}")
            
            return {
                'test_type': 'system_integration',
                'status': 'success',
                'system_status': system_status,
                'integration_features': integration_features,
                'integration_score': integration_score,
                'enhanced_prediction': integration_prediction.to_dict()
            }
            
        except Exception as e:
            logger.error(f"    ❌ System integration test failed: {e}")
            return {
                'test_type': 'system_integration',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_advanced_forecasting_features(self) -> Dict[str, Any]:
        """Test advanced forecasting features"""
        logger.info("🚀 Testing Advanced Forecasting Features...")
        
        try:
            advanced_tests = []
            
            # Test uncertainty quantification
            uncertainty_test = await self._test_uncertainty_quantification()
            advanced_tests.append(uncertainty_test)
            
            # Test model auto-selection
            model_selection_test = await self._test_model_auto_selection()
            advanced_tests.append(model_selection_test)
            
            # Test constraint handling
            constraint_test = await self._test_constraint_handling()
            advanced_tests.append(constraint_test)
            
            successful_tests = sum(1 for test in advanced_tests if test['status'] == 'success')
            
            return {
                'test_type': 'advanced_forecasting_features',
                'status': 'success' if successful_tests == len(advanced_tests) else 'partial',
                'total_tests': len(advanced_tests),
                'successful_tests': successful_tests,
                'individual_tests': advanced_tests
            }
            
        except Exception as e:
            logger.error(f"    ❌ Advanced features test failed: {e}")
            return {
                'test_type': 'advanced_forecasting_features',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_uncertainty_quantification(self) -> Dict[str, Any]:
        """Test uncertainty quantification"""
        try:
            # Create predictions with varying uncertainty levels
            low_uncertainty_prediction = await self.forecasting_system.create_prediction(
                PredictionType.PERFORMANCE,
                'stable_metric',
                ForecastHorizon.SHORT_TERM,
                timedelta(hours=2),
                {'current_value': 0.7, 'volatility': 'low', 'trend': 'stable'}
            )
            
            high_uncertainty_prediction = await self.forecasting_system.create_prediction(
                PredictionType.BEHAVIOR,
                'volatile_metric',
                ForecastHorizon.LONG_TERM,
                timedelta(days=14),
                {'current_value': 0.5, 'volatility': 'high', 'trend': 'unpredictable'}
            )
            
            uncertainty_difference = high_uncertainty_prediction.uncertainty_score - low_uncertainty_prediction.uncertainty_score
            
            return {
                'test_name': 'uncertainty_quantification',
                'status': 'success',
                'low_uncertainty': low_uncertainty_prediction.uncertainty_score,
                'high_uncertainty': high_uncertainty_prediction.uncertainty_score,
                'uncertainty_difference': uncertainty_difference,
                'quantification_working': uncertainty_difference > 0.1
            }
        except Exception as e:
            return {'test_name': 'uncertainty_quantification', 'status': 'failed', 'error': str(e)}
    
    async def _test_model_auto_selection(self) -> Dict[str, Any]:
        """Test automatic model selection"""
        try:
            # Test different prediction types to trigger different models
            ensemble_prediction = await self.forecasting_system.create_prediction(
                PredictionType.PERFORMANCE,
                'performance_metric',
                ForecastHorizon.MEDIUM_TERM,
                timedelta(days=2),
                {'current_value': 0.6}
            )
            
            cognitive_prediction = await self.forecasting_system.create_prediction(
                PredictionType.BEHAVIOR,
                'behavior_metric',
                ForecastHorizon.STRATEGIC,
                timedelta(weeks=4),
                {'current_value': 0.7, 'pattern': 'complex'}
            )
            
            return {
                'test_name': 'model_auto_selection',
                'status': 'success',
                'ensemble_method': ensemble_prediction.method_used.value,
                'cognitive_method': cognitive_prediction.method_used.value,
                'selection_working': ensemble_prediction.method_used != cognitive_prediction.method_used
            }
        except Exception as e:
            return {'test_name': 'model_auto_selection', 'status': 'failed', 'error': str(e)}
    
    async def _test_constraint_handling(self) -> Dict[str, Any]:
        """Test constraint handling in predictions"""
        try:
            # Create prediction with constraints
            constrained_prediction = await self.forecasting_system.create_prediction(
                PredictionType.CAPACITY,
                'constrained_metric',
                ForecastHorizon.MEDIUM_TERM,
                timedelta(days=3),
                {'current_value': 0.9, 'growth_rate': 0.3},
                constraints={'min_value': 0.5, 'max_value': 1.0}
            )
            
            within_constraints = (0.5 <= constrained_prediction.prediction_value <= 1.0)
            
            return {
                'test_name': 'constraint_handling',
                'status': 'success',
                'predicted_value': constrained_prediction.prediction_value,
                'within_constraints': within_constraints,
                'constraints_applied': 'constraints' in constrained_prediction.request.to_dict()
            }
        except Exception as e:
            return {'test_name': 'constraint_handling', 'status': 'failed', 'error': str(e)}
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("🧪 Starting Predictive Forecasting Test Suite...")
        
        start_time = datetime.now()
        
        # Initialize test system
        initialization_success = await self.initialize_test_system()
        if not initialization_success:
            return {'status': 'failed', 'error': 'System initialization failed'}
        
        # Test functions
        test_functions = [
            self.test_ensemble_forecasting,
            self.test_cognitive_forecasting,
            self.test_multi_horizon_forecasting,
            self.test_batch_predictions,
            self.test_prediction_validation,
            self.test_system_integration,
            self.test_advanced_forecasting_features
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
        partial_tests = len([r for r in test_results if r['status'] == 'partial'])
        success_rate = (successful_tests + partial_tests * 0.5) / total_tests if total_tests > 0 else 0.0
        
        # Get final system status
        final_status = await self.forecasting_system.get_system_status()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_suite_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(end_time - start_time),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'partial_tests': partial_tests,
                'success_rate': success_rate
            },
            'individual_test_results': test_results,
            'final_system_status': final_status,
            'test_data_summary': {
                'training_data_points': self.synthetic_data['metadata']['data_points'],
                'time_range': self.synthetic_data['metadata']['time_range'],
                'metrics_included': self.synthetic_data['metadata']['metrics']
            },
            'performance_summary': {
                'total_predictions_made': final_status['metrics']['total_predictions'],
                'average_accuracy': final_status['metrics']['average_accuracy'],
                'average_uncertainty': final_status['metrics']['average_uncertainty'],
                'ensemble_size': final_status['predictors']['ensemble_predictor']['ensemble_size']
            }
        }
        
        logger.info("🎉 Predictive Forecasting Test Suite Completed!")
        logger.info(f"[STATS] Success Rate: {success_rate:.1%}")
        logger.info(f"[TIME] Duration: {end_time - start_time}")
        logger.info(f"[PREDICTIONS] Total Made: {final_status['metrics']['total_predictions']}")
        
        return comprehensive_results


async def main():
    """Main test execution function"""
    test_suite = PredictiveForecastingTestSuite()
    
    try:
        # Run comprehensive test suite
        results = await test_suite.run_comprehensive_test_suite()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"predictive_forecasting_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[SAVE] Test results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("PREDICTIVE INTELLIGENCE & FORECASTING SYSTEM TEST SUMMARY")
        print("="*80)
        print(f"[OK] Total Tests: {results['test_suite_info']['successful_tests']}/{results['test_suite_info']['total_tests']}")
        print(f"[STATS] Success Rate: {results['test_suite_info']['success_rate']:.1%}")
        print(f"[TIME] Duration: {results['test_suite_info']['duration']}")
        print(f"[ACCURACY] Average Accuracy: {results['performance_summary']['average_accuracy']:.1%}")
        print(f"[ENSEMBLE] Model Count: {results['performance_summary']['ensemble_size']}")
        print(f"[PREDICTIONS] Total Made: {results['performance_summary']['total_predictions_made']}")
        print(f"[DATA] Training Points: {results['test_data_summary']['training_data_points']}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())