"""
Analytics Aggregation Pipeline
==============================

Advanced data aggregation pipeline with transformation, enrichment,
and intelligent data flow management.

Author: TestMaster Team
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import copy

logger = logging.getLogger(__name__)

class DataTransformer:
    """Base class for data transformers."""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution = None
    
    def transform(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the data. Override in subclasses."""
        start_time = time.time()
        try:
            result = self._transform_impl(data, context)
            self.execution_count += 1
            self.total_execution_time += time.time() - start_time
            self.last_execution = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Error in transformer {self.name}: {e}")
            return data  # Return original data on error
    
    def _transform_impl(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation method to override."""
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformer statistics."""
        avg_time = self.total_execution_time / max(self.execution_count, 1)
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_time,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None
        }

class DataEnricher(DataTransformer):
    """Enriches analytics data with additional computed metrics."""
    
    def _transform_impl(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        enriched = copy.deepcopy(data)
        
        # Add computed metrics
        enriched['_enrichment'] = {
            'computed_at': datetime.now().isoformat(),
            'pipeline_stage': 'enrichment',
            'enricher_version': '1.0'
        }
        
        # Calculate derived metrics
        if 'system_metrics' in enriched:
            self._enrich_system_metrics(enriched['system_metrics'])
        
        if 'test_analytics' in enriched:
            self._enrich_test_metrics(enriched['test_analytics'])
        
        if 'workflow_analytics' in enriched:
            self._enrich_workflow_metrics(enriched['workflow_analytics'])
        
        # Add cross-component correlations
        self._add_cross_correlations(enriched)
        
        return enriched
    
    def _enrich_system_metrics(self, system_metrics: Dict[str, Any]):
        """Enrich system metrics with derived values."""
        if 'cpu' in system_metrics and 'memory' in system_metrics:
            # Calculate system load score
            cpu_usage = system_metrics['cpu'].get('usage_percent', 0)
            memory_usage = system_metrics['memory'].get('percent', 0)
            
            # Weighted system load (CPU has higher weight)
            system_load = (cpu_usage * 0.7) + (memory_usage * 0.3)
            
            system_metrics['derived'] = {
                'system_load_score': system_load,
                'performance_grade': self._calculate_performance_grade(system_load),
                'resource_pressure': 'high' if system_load > 80 else 'medium' if system_load > 60 else 'low'
            }
    
    def _enrich_test_metrics(self, test_metrics: Dict[str, Any]):
        """Enrich test metrics with quality indicators."""
        total_tests = test_metrics.get('total_tests', 0)
        passed_tests = test_metrics.get('passed', 0)
        failed_tests = test_metrics.get('failed', 0)
        
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            fail_rate = (failed_tests / total_tests) * 100
            
            # Calculate test health score
            coverage = test_metrics.get('coverage_percent', 0)
            health_score = (pass_rate * 0.4) + (coverage * 0.4) + (min(100, total_tests / 10) * 0.2)
            
            test_metrics['derived'] = {
                'pass_rate': pass_rate,
                'fail_rate': fail_rate,
                'health_score': health_score,
                'quality_tier': self._calculate_quality_tier(health_score),
                'recommendation': self._get_test_recommendation(pass_rate, coverage, total_tests)
            }
    
    def _enrich_workflow_metrics(self, workflow_metrics: Dict[str, Any]):
        """Enrich workflow metrics with efficiency indicators."""
        active_workflows = workflow_metrics.get('active_workflows', 0)
        success_rate = workflow_metrics.get('success_rate', 100)
        avg_duration = workflow_metrics.get('average_duration_seconds', 0)
        
        # Calculate workflow efficiency score
        efficiency_score = self._calculate_workflow_efficiency(success_rate, avg_duration, active_workflows)
        
        workflow_metrics['derived'] = {
            'efficiency_score': efficiency_score,
            'performance_rating': self._get_performance_rating(efficiency_score),
            'optimization_potential': 100 - efficiency_score if efficiency_score < 90 else 0
        }
    
    def _add_cross_correlations(self, data: Dict[str, Any]):
        """Add cross-component correlation analysis."""
        correlations = {}
        
        # System load vs test performance
        if 'system_metrics' in data and 'test_analytics' in data:
            system_load = data['system_metrics'].get('derived', {}).get('system_load_score', 0)
            test_health = data['test_analytics'].get('derived', {}).get('health_score', 0)
            
            correlations['system_test_correlation'] = {
                'system_load': system_load,
                'test_health': test_health,
                'correlation_strength': self._calculate_correlation_strength(system_load, test_health)
            }
        
        if correlations:
            data['cross_correlations'] = correlations
    
    def _calculate_performance_grade(self, system_load: float) -> str:
        """Calculate performance grade based on system load."""
        if system_load <= 30:
            return 'A'
        elif system_load <= 50:
            return 'B'
        elif system_load <= 70:
            return 'C'
        elif system_load <= 85:
            return 'D'
        else:
            return 'F'
    
    def _calculate_quality_tier(self, health_score: float) -> str:
        """Calculate quality tier based on health score."""
        if health_score >= 90:
            return 'excellent'
        elif health_score >= 75:
            return 'good'
        elif health_score >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _get_test_recommendation(self, pass_rate: float, coverage: float, total_tests: int) -> str:
        """Get test improvement recommendation."""
        if pass_rate < 80:
            return 'Focus on fixing failing tests'
        elif coverage < 70:
            return 'Increase test coverage'
        elif total_tests < 50:
            return 'Add more comprehensive tests'
        else:
            return 'Tests are in good shape'
    
    def _calculate_workflow_efficiency(self, success_rate: float, avg_duration: float, active_count: int) -> float:
        """Calculate workflow efficiency score."""
        # Base score from success rate
        score = success_rate
        
        # Adjust for duration (penalty for slow workflows)
        if avg_duration > 300:  # 5 minutes
            score *= 0.8
        elif avg_duration > 120:  # 2 minutes
            score *= 0.9
        
        # Adjust for load (penalty for too many active workflows)
        if active_count > 10:
            score *= 0.9
        elif active_count > 20:
            score *= 0.8
        
        return min(100, max(0, score))
    
    def _get_performance_rating(self, efficiency_score: float) -> str:
        """Get performance rating based on efficiency score."""
        if efficiency_score >= 95:
            return 'outstanding'
        elif efficiency_score >= 85:
            return 'excellent'
        elif efficiency_score >= 75:
            return 'good'
        elif efficiency_score >= 60:
            return 'fair'
        else:
            return 'needs_improvement'
    
    def _calculate_correlation_strength(self, value1: float, value2: float) -> str:
        """Calculate correlation strength between two values."""
        # Simple correlation based on value ranges
        diff = abs(value1 - value2)
        if diff < 10:
            return 'strong'
        elif diff < 25:
            return 'moderate'
        else:
            return 'weak'

class DataNormalizer(DataTransformer):
    """Normalizes data formats and units across different components."""
    
    def _transform_impl(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(data)
        
        # Normalize timestamps
        self._normalize_timestamps(normalized)
        
        # Normalize units
        self._normalize_units(normalized)
        
        # Standardize field names
        self._standardize_fields(normalized)
        
        # Add normalization metadata
        normalized['_normalization'] = {
            'normalized_at': datetime.now().isoformat(),
            'normalizer_version': '1.0',
            'applied_transforms': ['timestamps', 'units', 'fields']
        }
        
        return normalized
    
    def _normalize_timestamps(self, data: Dict[str, Any]):
        """Ensure all timestamps are in ISO format."""
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'last_scan']
        
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in timestamp_fields and isinstance(value, str):
                        try:
                            # Try to parse and reformat timestamp
                            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            obj[key] = dt.isoformat()
                        except ValueError:
                            pass  # Keep original if parsing fails
                    elif isinstance(value, (dict, list)):
                        normalize_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    normalize_recursive(item)
        
        normalize_recursive(data)
    
    def _normalize_units(self, data: Dict[str, Any]):
        """Normalize units across metrics."""
        # Memory values to MB
        if 'system_metrics' in data and 'memory' in data['system_metrics']:
            memory = data['system_metrics']['memory']
            for key in ['total_mb', 'used_mb', 'available_mb']:
                if key in memory:
                    # Ensure values are in MB
                    memory[key] = round(memory[key], 2)
        
        # Duration values to seconds
        if 'workflow_analytics' in data:
            workflow = data['workflow_analytics']
            if 'average_duration_seconds' in workflow:
                workflow['average_duration_seconds'] = round(workflow['average_duration_seconds'], 2)
    
    def _standardize_fields(self, data: Dict[str, Any]):
        """Standardize field names and structures."""
        # Ensure consistent boolean representations
        def standardize_booleans(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.lower() in ['true', 'false']:
                        obj[key] = value.lower() == 'true'
                    elif isinstance(value, (dict, list)):
                        standardize_booleans(value)
            elif isinstance(obj, list):
                for item in obj:
                    standardize_booleans(item)
        
        standardize_booleans(data)

class DataAggregator(DataTransformer):
    """Aggregates data from multiple sources with conflict resolution."""
    
    def __init__(self, name: str, aggregation_rules: Dict[str, str] = None):
        super().__init__(name)
        self.aggregation_rules = aggregation_rules or {}
        self.data_sources = defaultdict(list)
    
    def add_data_source(self, source_name: str, data: Dict[str, Any]):
        """Add data from a specific source."""
        self.data_sources[source_name].append({
            'timestamp': datetime.now(),
            'data': data
        })
        
        # Keep only recent data (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.data_sources[source_name] = [
            item for item in self.data_sources[source_name]
            if item['timestamp'] > cutoff
        ]
    
    def _transform_impl(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        aggregated = copy.deepcopy(data)
        
        # Add aggregation metadata
        aggregated['_aggregation'] = {
            'aggregated_at': datetime.now().isoformat(),
            'source_count': len(self.data_sources),
            'aggregation_method': 'weighted_average'
        }
        
        # Aggregate numeric metrics
        self._aggregate_numeric_metrics(aggregated)
        
        # Resolve conflicts in categorical data
        self._resolve_categorical_conflicts(aggregated)
        
        return aggregated
    
    def _aggregate_numeric_metrics(self, data: Dict[str, Any]):
        """Aggregate numeric metrics from multiple sources."""
        numeric_paths = [
            'system_metrics.cpu.usage_percent',
            'system_metrics.memory.percent',
            'test_analytics.coverage_percent',
            'workflow_analytics.success_rate'
        ]
        
        for path in numeric_paths:
            values = []
            for source_data in self.data_sources.values():
                for item in source_data:
                    value = self._get_nested_value(item['data'], path)
                    if value is not None and isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                # Use weighted average (more recent values have higher weight)
                aggregated_value = sum(values) / len(values)
                self._set_nested_value(data, path, round(aggregated_value, 2))
    
    def _resolve_categorical_conflicts(self, data: Dict[str, Any]):
        """Resolve conflicts in categorical data using voting."""
        categorical_paths = [
            'system_metrics.derived.performance_grade',
            'test_analytics.health_status'
        ]
        
        for path in categorical_paths:
            values = []
            for source_data in self.data_sources.values():
                for item in source_data:
                    value = self._get_nested_value(item['data'], path)
                    if value is not None:
                        values.append(value)
            
            if values:
                # Use majority vote
                from collections import Counter
                most_common = Counter(values).most_common(1)
                if most_common:
                    self._set_nested_value(data, path, most_common[0][0])
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value

class AnalyticsPipeline:
    """
    Main analytics pipeline that orchestrates data flow through transformers.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the analytics pipeline.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.transformers = []
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'start_time': datetime.now()
        }
        
        # Error tracking
        self.recent_errors = deque(maxlen=100)
        
        # Default transformers
        self._setup_default_pipeline()
        
        logger.info("Analytics Pipeline initialized")
    
    def _setup_default_pipeline(self):
        """Set up the default transformation pipeline."""
        # Add transformers in order of execution
        self.add_transformer(DataNormalizer('normalizer'))
        self.add_transformer(DataEnricher('enricher'))
        self.add_transformer(DataAggregator('aggregator'))
    
    def add_transformer(self, transformer: DataTransformer):
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
        logger.info(f"Added transformer to pipeline: {transformer.name}")
    
    def remove_transformer(self, transformer_name: str) -> bool:
        """Remove a transformer from the pipeline."""
        for i, transformer in enumerate(self.transformers):
            if transformer.name == transformer_name:
                del self.transformers[i]
                logger.info(f"Removed transformer from pipeline: {transformer_name}")
                return True
        return False
    
    def process(self, analytics_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process analytics data through the pipeline.
        
        Args:
            analytics_data: The analytics data to process
            context: Additional context for processing
        
        Returns:
            Processed analytics data
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Add pipeline context
            context.update({
                'pipeline_start_time': start_time,
                'pipeline_id': f"pipeline_{int(start_time * 1000)}",
                'transformer_count': len(self.transformers)
            })
            
            # Process through each transformer
            processed_data = analytics_data
            transformer_results = []
            
            for transformer in self.transformers:
                transformer_start = time.time()
                try:
                    processed_data = transformer.transform(processed_data, context)
                    transformer_duration = time.time() - transformer_start
                    
                    transformer_results.append({
                        'name': transformer.name,
                        'duration': transformer_duration,
                        'success': True
                    })
                    
                except Exception as e:
                    transformer_duration = time.time() - transformer_start
                    error_info = {
                        'transformer': transformer.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.recent_errors.append(error_info)
                    
                    transformer_results.append({
                        'name': transformer.name,
                        'duration': transformer_duration,
                        'success': False,
                        'error': str(e)
                    })
                    
                    logger.error(f"Transformer {transformer.name} failed: {e}")
            
            # Add pipeline metadata to processed data
            total_duration = time.time() - start_time
            processed_data['_pipeline'] = {
                'processed_at': datetime.now().isoformat(),
                'total_duration': total_duration,
                'transformer_results': transformer_results,
                'success': all(result['success'] for result in transformer_results)
            }
            
            # Update statistics
            self.pipeline_stats['total_executions'] += 1
            self.pipeline_stats['total_execution_time'] += total_duration
            
            if processed_data['_pipeline']['success']:
                self.pipeline_stats['successful_executions'] += 1
            else:
                self.pipeline_stats['failed_executions'] += 1
            
            return processed_data
            
        except Exception as e:
            # Pipeline-level error
            error_info = {
                'type': 'pipeline_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.recent_errors.append(error_info)
            
            self.pipeline_stats['total_executions'] += 1
            self.pipeline_stats['failed_executions'] += 1
            
            logger.error(f"Pipeline processing failed: {e}")
            
            # Return original data with error information
            error_data = copy.deepcopy(analytics_data)
            error_data['_pipeline'] = {
                'processed_at': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
            return error_data
    
    def process_async(self, analytics_data: Dict[str, Any], context: Dict[str, Any] = None) -> 'Future':
        """Process analytics data asynchronously."""
        return self.executor.submit(self.process, analytics_data, context)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        uptime = (datetime.now() - self.pipeline_stats['start_time']).total_seconds()
        avg_execution_time = (self.pipeline_stats['total_execution_time'] / 
                            max(self.pipeline_stats['total_executions'], 1))
        
        return {
            'transformer_count': len(self.transformers),
            'total_executions': self.pipeline_stats['total_executions'],
            'successful_executions': self.pipeline_stats['successful_executions'],
            'failed_executions': self.pipeline_stats['failed_executions'],
            'success_rate': (self.pipeline_stats['successful_executions'] / 
                           max(self.pipeline_stats['total_executions'], 1)) * 100,
            'average_execution_time': avg_execution_time,
            'total_execution_time': self.pipeline_stats['total_execution_time'],
            'uptime_seconds': uptime,
            'executions_per_second': self.pipeline_stats['total_executions'] / max(uptime, 1),
            'recent_errors': list(self.recent_errors),
            'transformer_stats': [transformer.get_stats() for transformer in self.transformers]
        }
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self.pipeline_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'start_time': datetime.now()
        }
        self.recent_errors.clear()
        
        # Reset transformer stats
        for transformer in self.transformers:
            transformer.execution_count = 0
            transformer.total_execution_time = 0.0
            transformer.last_execution = None
    
    def shutdown(self):
        """Shutdown the pipeline."""
        self.executor.shutdown(wait=True)
        logger.info("Analytics Pipeline shutdown")