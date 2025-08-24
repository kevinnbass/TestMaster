#!/usr/bin/env python3
"""
🏗️ MODULE: Garbage Collection Optimizer - GC Performance Tuning
==============================================================

📋 PURPOSE:
    Python garbage collection optimization system for performance tuning.
    Analyzes GC behavior and optimizes settings for memory or speed targets.

🎯 CORE FUNCTIONALITY:
    • Garbage collection performance analysis and statistics tracking
    • GC threshold optimization based on target performance profiles
    • GC benchmarking with different configuration presets
    • Historical optimization tracking and rollback capabilities

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 09:15:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract GC optimization functionality from memory_management_optimizer.py via STEELCLAD
   └─ Changes: Created dedicated module for garbage collection optimization
   └─ Impact: Clean separation of GC tuning logic from memory management orchestration

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: gc, time, psutil, logging, dataclasses
🎯 Integration Points: MemoryManager orchestration and monitoring systems
⚡ Performance Notes: Efficient GC analysis with minimal impact on running system
🔒 Security Notes: Safe GC configuration changes with rollback capability

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via memory management validation | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: gc, psutil, memory_management_models
📤 Provides: Garbage collection optimization and tuning capabilities
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import gc
import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import asdict

# Import models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.memory_management_models import GCStats, GC_THRESHOLDS_PRESETS

class GarbageCollectionOptimizer:
    """Optimizes Python garbage collection settings"""
    
    def __init__(self):
        self.original_thresholds = gc.get_threshold()
        self.gc_stats_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        self.logger = logging.getLogger('GarbageCollectionOptimizer')
    
    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze current garbage collection performance"""
        # Get current GC statistics
        gc_stats = []
        for i in range(3):
            if i < len(gc.get_stats()):
                stats = gc.get_stats()[i]
                gc_stats.append(GCStats(
                    generation=i,
                    collections=stats.get('collections', 0),
                    collected=stats.get('collected', 0),
                    uncollectable=stats.get('uncollectable', 0),
                    threshold=gc.get_threshold()
                ))
        
        # Count objects by generation
        object_counts = [0, 0, 0]
        try:
            for obj in gc.get_objects():
                gen = gc.get_referents(obj)
                if len(gen) < 3:
                    object_counts[0] += 1
                elif len(gen) < 10:
                    object_counts[1] += 1
                else:
                    object_counts[2] += 1
        except Exception:
            # Fallback to simple counting
            object_counts = [len(gc.get_objects()), 0, 0]
        
        analysis = {
            'gc_statistics': [asdict(stat) for stat in gc_stats],
            'object_counts_by_generation': object_counts,
            'total_objects': len(gc.get_objects()),
            'gc_enabled': gc.isenabled(),
            'current_thresholds': gc.get_threshold(),
            'original_thresholds': self.original_thresholds
        }
        
        return analysis
    
    def optimize_gc_settings(self, target_performance: str = 'balanced') -> Dict[str, Any]:
        """Optimize garbage collection settings"""
        current_analysis = self.analyze_gc_performance()
        
        # Record baseline
        baseline = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis': current_analysis,
            'target': target_performance
        }
        
        # Determine optimal thresholds based on target
        if target_performance == 'memory':
            # Optimize for low memory usage (more frequent GC)
            new_thresholds = GC_THRESHOLDS_PRESETS['memory']
        elif target_performance == 'speed':
            # Optimize for speed (less frequent GC)
            new_thresholds = GC_THRESHOLDS_PRESETS['speed']
        else:  # balanced
            # Balanced approach based on object count
            total_objects = current_analysis['total_objects']
            if total_objects < 10000:
                new_thresholds = GC_THRESHOLDS_PRESETS['balanced_small']
            elif total_objects < 50000:
                new_thresholds = GC_THRESHOLDS_PRESETS['balanced_medium']
            else:
                new_thresholds = GC_THRESHOLDS_PRESETS['balanced_large']
        
        # Apply new thresholds
        gc.set_threshold(*new_thresholds)
        
        # Force garbage collection to clear current state
        collected = gc.collect()
        
        optimization_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'baseline': baseline,
            'applied_thresholds': new_thresholds,
            'objects_collected': collected,
            'target_performance': target_performance
        }
        
        self.optimization_history.append(optimization_record)
        
        self.logger.info(f"Optimized GC settings for {target_performance}: {new_thresholds}")
        self.logger.info(f"Collected {collected} objects during optimization")
        
        return optimization_record
    
    def benchmark_gc_performance(self) -> Dict[str, Any]:
        """Benchmark garbage collection performance"""
        results = {}
        
        # Test different threshold settings
        test_configurations = [
            ('conservative', GC_THRESHOLDS_PRESETS['memory']),
            ('default', GC_THRESHOLDS_PRESETS['balanced_small']),
            ('aggressive', GC_THRESHOLDS_PRESETS['speed'])
        ]
        
        original_thresholds = gc.get_threshold()
        
        for config_name, thresholds in test_configurations:
            # Set test thresholds
            gc.set_threshold(*thresholds)
            gc.collect()  # Clear state
            
            # Create test objects and measure GC performance
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss
            
            # Create objects that will need garbage collection
            test_objects = []
            for i in range(1000):
                obj = {'data': [j for j in range(100)], 'refs': []}
                # Create circular references
                obj['refs'].append(obj)
                test_objects.append(obj)
            
            # Force garbage collection and measure
            gc_start = time.perf_counter()
            collected = gc.collect()
            gc_time = time.perf_counter() - gc_start
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            results[config_name] = {
                'thresholds': thresholds,
                'total_time': end_time - start_time,
                'gc_time': gc_time,
                'objects_collected': collected,
                'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                'gc_overhead_percent': (gc_time / (end_time - start_time)) * 100
            }
            
            # Cleanup
            del test_objects
            gc.collect()
        
        # Restore original thresholds
        gc.set_threshold(*original_thresholds)
        
        self.logger.info("GC performance benchmark completed")
        return results
    
    def reset_gc_settings(self):
        """Reset GC settings to original values"""
        gc.set_threshold(*self.original_thresholds)
        self.logger.info(f"Reset GC settings to original: {self.original_thresholds}")
    
    def get_gc_health_assessment(self) -> Dict[str, Any]:
        """Assess current GC health and provide recommendations"""
        analysis = self.analyze_gc_performance()
        current_thresholds = analysis['current_thresholds']
        total_objects = analysis['total_objects']
        
        # Calculate collection efficiency
        total_collections = sum(stat['collections'] for stat in analysis['gc_statistics'])
        total_collected = sum(stat['collected'] for stat in analysis['gc_statistics'])
        
        collection_efficiency = 0
        if total_collections > 0:
            collection_efficiency = total_collected / total_collections
        
        # Assess threshold appropriateness
        threshold_assessment = self._assess_thresholds(current_thresholds, total_objects)
        
        # Generate health score
        health_score = self._calculate_gc_health_score(analysis, collection_efficiency)
        
        recommendations = []
        if health_score < 60:
            recommendations.extend(self._generate_gc_recommendations(analysis, threshold_assessment))
        
        return {
            'health_score': health_score,
            'collection_efficiency': collection_efficiency,
            'threshold_assessment': threshold_assessment,
            'recommendations': recommendations,
            'current_status': self._get_health_status(health_score),
            'optimization_opportunities': self._identify_optimization_opportunities(analysis)
        }
    
    def _assess_thresholds(self, thresholds, total_objects) -> Dict[str, str]:
        """Assess if current thresholds are appropriate for object count"""
        gen0_threshold = thresholds[0]
        
        if total_objects < 5000:
            if gen0_threshold > 1000:
                return {'assessment': 'high', 'reason': 'Thresholds too high for small object count'}
            else:
                return {'assessment': 'appropriate', 'reason': 'Thresholds suitable for small object count'}
        elif total_objects < 25000:
            if gen0_threshold < 500:
                return {'assessment': 'low', 'reason': 'Thresholds too low for medium object count'}
            elif gen0_threshold > 1500:
                return {'assessment': 'high', 'reason': 'Thresholds too high for medium object count'}
            else:
                return {'assessment': 'appropriate', 'reason': 'Thresholds suitable for medium object count'}
        else:
            if gen0_threshold < 1000:
                return {'assessment': 'low', 'reason': 'Thresholds too low for large object count'}
            else:
                return {'assessment': 'appropriate', 'reason': 'Thresholds suitable for large object count'}
    
    def _calculate_gc_health_score(self, analysis, collection_efficiency) -> float:
        """Calculate overall GC health score"""
        base_score = 100.0
        
        # Penalize low collection efficiency
        if collection_efficiency < 50:
            base_score -= 30
        elif collection_efficiency < 100:
            base_score -= 15
        
        # Penalize high uncollectable objects
        total_uncollectable = sum(stat['uncollectable'] for stat in analysis['gc_statistics'])
        if total_uncollectable > 0:
            base_score -= min(20, total_uncollectable * 2)
        
        # Bonus for GC being enabled
        if analysis['gc_enabled']:
            base_score += 10
        
        return max(0, min(100, base_score))
    
    def _get_health_status(self, health_score) -> str:
        """Get health status based on score"""
        if health_score >= 80:
            return 'excellent'
        elif health_score >= 60:
            return 'good'
        elif health_score >= 40:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_gc_recommendations(self, analysis, threshold_assessment) -> List[str]:
        """Generate GC optimization recommendations"""
        recommendations = []
        
        if threshold_assessment['assessment'] == 'high':
            recommendations.append("Consider reducing GC thresholds to collect garbage more frequently")
        elif threshold_assessment['assessment'] == 'low':
            recommendations.append("Consider increasing GC thresholds to reduce GC overhead")
        
        total_uncollectable = sum(stat['uncollectable'] for stat in analysis['gc_statistics'])
        if total_uncollectable > 0:
            recommendations.append("Investigate uncollectable objects - possible circular references")
        
        if not analysis['gc_enabled']:
            recommendations.append("Enable garbage collection for automatic memory management")
        
        return recommendations
    
    def _identify_optimization_opportunities(self, analysis) -> List[str]:
        """Identify potential optimization opportunities"""
        opportunities = []
        
        total_objects = analysis['total_objects']
        if total_objects > 50000:
            opportunities.append("High object count - consider object pooling or caching strategies")
        
        gen_collections = [stat['collections'] for stat in analysis['gc_statistics']]
        if len(gen_collections) >= 2 and gen_collections[0] > gen_collections[1] * 10:
            opportunities.append("High generation 0 collections - optimize object lifecycle management")
        
        return opportunities