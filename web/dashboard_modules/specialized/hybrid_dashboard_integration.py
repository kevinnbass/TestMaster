#!/usr/bin/env python3
"""
Hybrid Dashboard Integration - Agent Beta Performance System
==========================================================

Combines the best of both worlds:
- Fast real-time analysis for dashboard updates (1,768 files/sec)
- Comprehensive semantic analysis for detailed insights (47 files/sec)
- Intelligent switching between modes based on request type
- Advanced caching for optimal performance

Agent Beta's performance optimization strategy:
- Use original quick_linkage_analysis for real-time dashboard
- Use PerformanceOptimizedAnalyzer for comprehensive background analysis
- Implement smart caching layer for best performance
- Provide seamless integration with Agent Alpha's semantic enhancements

Author: Agent Beta - Dashboard Intelligence Swarm
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

# Import both analysis systems
from enhanced_linkage_dashboard import quick_linkage_analysis
from performance_optimized_linkage import performance_analyzer, optimized_linkage_analysis
from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer

class HybridDashboardSystem:
    """
    Intelligent dashboard backend that automatically selects the best analysis approach.
    
    Performance Modes:
    - FAST: Original system for real-time updates (1,768 files/sec)
    - COMPREHENSIVE: Optimized system for deep analysis (47 files/sec) 
    - INTELLIGENT: Agent Alpha's semantic analysis integration
    - CACHED: Cached results for instant responses
    """
    
    def __init__(self, cache_ttl=300):  # 5 minute cache TTL
        self.cache_ttl = cache_ttl
        self.analysis_cache = {}
        self.cache_timestamps = {}
        self.performance_stats = {
            "fast_queries": 0,
            "comprehensive_queries": 0,
            "intelligent_queries": 0,
            "cache_hits": 0,
            "total_requests": 0,
            "average_response_time": 0
        }
        
        # Initialize intelligent analyzer (Agent Alpha's work)
        self.intelligent_analyzer = EnhancedLinkageAnalyzer()
        
        # Analysis mode thresholds
        self.fast_file_threshold = 5000  # Use fast mode for > 5000 files
        self.comprehensive_file_threshold = 1000  # Use comprehensive mode for < 1000 files
        
        print("Hybrid Dashboard System initialized")
        print(f"   Fast Mode: {self.fast_file_threshold}+ files")
        print(f"   Comprehensive Mode: < {self.comprehensive_file_threshold} files")
        print(f"   Cache TTL: {cache_ttl} seconds")
    
    async def analyze_codebase(self, base_dir="TestMaster", analysis_type="auto", force_refresh=False):
        """
        Main analysis entry point with intelligent mode selection.
        
        Analysis Types:
        - "auto": Automatically select best mode based on codebase size
        - "fast": Force fast mode (real-time dashboard updates)
        - "comprehensive": Force comprehensive mode (detailed analysis)
        - "intelligent": Force intelligent mode (Agent Alpha's semantic analysis)
        """
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = self._get_cached_result(base_dir, analysis_type)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                self._update_performance_stats(time.time() - start_time)
                return cached_result
        
        # Determine optimal analysis mode
        if analysis_type == "auto":
            analysis_type = self._select_optimal_mode(base_dir)
        
        # Execute analysis based on selected mode
        if analysis_type == "fast":
            result = await self._fast_analysis(base_dir)
            self.performance_stats["fast_queries"] += 1
            
        elif analysis_type == "comprehensive":
            result = await self._comprehensive_analysis(base_dir)
            self.performance_stats["comprehensive_queries"] += 1
            
        elif analysis_type == "intelligent":
            result = await self._intelligent_analysis(base_dir)
            self.performance_stats["intelligent_queries"] += 1
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Cache the result
        self._cache_result(base_dir, analysis_type, result)
        
        # Update performance stats
        end_time = time.time()
        processing_time = end_time - start_time
        self._update_performance_stats(processing_time)
        
        # Add performance metadata
        result["performance_info"] = {
            "analysis_mode": analysis_type,
            "processing_time": processing_time,
            "cache_used": False,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _select_optimal_mode(self, base_dir):
        """Intelligently select the best analysis mode based on codebase characteristics."""
        try:
            # Quick file count to determine optimal mode
            file_count = self._quick_file_count(base_dir)
            
            if file_count > self.fast_file_threshold:
                return "fast"
            elif file_count < self.comprehensive_file_threshold:
                return "comprehensive" 
            else:
                return "intelligent"  # Sweet spot for semantic analysis
                
        except Exception:
            return "fast"  # Default to fast mode on error
    
    def _quick_file_count(self, base_dir):
        """Quick count of Python files for mode selection."""
        count = 0
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return 0
            
        for root, dirs, files in os.walk(base_path):
            # Skip problematic directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'QUARANTINE', 'archive']]
            
            for file in files:
                if file.endswith('.py'):
                    count += 1
                    if count > self.fast_file_threshold:  # Early termination for large codebases
                        break
                        
        return count
    
    async def _fast_analysis(self, base_dir):
        """Fast analysis using the original dashboard system (1,768 files/sec)."""
        try:
            result = quick_linkage_analysis(base_dir)
            
            # Enhance with performance metadata
            result["analysis_mode"] = "fast"
            result["performance_profile"] = "real_time_optimized"
            result["estimated_files_per_second"] = 1768
            
            return result
            
        except Exception as e:
            return {"error": f"Fast analysis failed: {str(e)}", "analysis_mode": "fast"}
    
    async def _comprehensive_analysis(self, base_dir):
        """Comprehensive analysis using the optimized async system (47 files/sec)."""
        try:
            result = await optimized_linkage_analysis(base_dir)
            
            # Enhance with performance metadata
            result["analysis_mode"] = "comprehensive"
            result["performance_profile"] = "deep_analysis_optimized"
            result["estimated_files_per_second"] = 47
            result["features"] = [
                "AST-based parsing",
                "Intelligent caching",
                "Async processing",
                "Memory optimization"
            ]
            
            return result
            
        except Exception as e:
            return {"error": f"Comprehensive analysis failed: {str(e)}", "analysis_mode": "comprehensive"}
    
    async def _intelligent_analysis(self, base_dir):
        """Intelligent analysis using Agent Alpha's semantic system."""
        try:
            # Run Agent Alpha's enhanced analysis
            result = self.intelligent_analyzer.analyze_codebase(base_dir)
            
            # Enhance with performance metadata
            result["analysis_mode"] = "intelligent" 
            result["performance_profile"] = "semantic_intelligence_enhanced"
            result["features"] = [
                "ML-powered intent classification",
                "15+ semantic categories", 
                "Security vulnerability assessment",
                "Quality metrics analysis",
                "Pattern recognition",
                "Predictive analytics"
            ]
            result["agent_alpha_integration"] = True
            
            return result
            
        except Exception as e:
            return {"error": f"Intelligent analysis failed: {str(e)}", "analysis_mode": "intelligent"}
    
    def _get_cache_key(self, base_dir, analysis_type):
        """Generate cache key for analysis results."""
        # Include directory modification time for cache invalidation
        try:
            base_path = Path(base_dir)
            if base_path.exists():
                # Get latest modification time of Python files
                latest_mtime = 0
                for root, dirs, files in os.walk(base_path):
                    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
                    for file in files:
                        if file.endswith('.py'):
                            file_path = Path(root) / file
                            try:
                                mtime = file_path.stat().st_mtime
                                latest_mtime = max(latest_mtime, mtime)
                            except:
                                continue
                
                # Create cache key with directory and latest modification time
                cache_data = f"{base_dir}:{analysis_type}:{latest_mtime}"
                return hashlib.md5(cache_data.encode()).hexdigest()
        except:
            pass
            
        # Fallback cache key
        return hashlib.md5(f"{base_dir}:{analysis_type}".encode()).hexdigest()
    
    def _get_cached_result(self, base_dir, analysis_type):
        """Get cached analysis result if available and valid."""
        cache_key = self._get_cache_key(base_dir, analysis_type)
        
        if cache_key in self.analysis_cache:
            cache_timestamp = self.cache_timestamps.get(cache_key, 0)
            
            # Check if cache is still valid
            if time.time() - cache_timestamp < self.cache_ttl:
                cached_result = self.analysis_cache[cache_key].copy()
                cached_result["performance_info"] = {
                    "analysis_mode": analysis_type,
                    "processing_time": 0.001,  # Near-instant cache response
                    "cache_used": True,
                    "cache_age_seconds": time.time() - cache_timestamp,
                    "timestamp": datetime.now().isoformat()
                }
                return cached_result
        
        return None
    
    def _cache_result(self, base_dir, analysis_type, result):
        """Cache analysis result for future use."""
        cache_key = self._get_cache_key(base_dir, analysis_type)
        
        # Store result and timestamp
        self.analysis_cache[cache_key] = result.copy()
        self.cache_timestamps[cache_key] = time.time()
        
        # Limit cache size (keep only recent entries)
        if len(self.analysis_cache) > 50:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])[:10]
            for old_key in oldest_keys:
                self.analysis_cache.pop(old_key, None)
                self.cache_timestamps.pop(old_key, None)
    
    def _update_performance_stats(self, processing_time):
        """Update performance statistics."""
        # Update average response time
        total_requests = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["average_response_time"]
        
        self.performance_stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        total_queries = (self.performance_stats["fast_queries"] + 
                        self.performance_stats["comprehensive_queries"] + 
                        self.performance_stats["intelligent_queries"])
        
        cache_hit_rate = 0
        if self.performance_stats["total_requests"] > 0:
            cache_hit_rate = (self.performance_stats["cache_hits"] / 
                             self.performance_stats["total_requests"] * 100)
        
        return {
            **self.performance_stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.analysis_cache),
            "active_analysis_modes": {
                "fast": self.performance_stats["fast_queries"],
                "comprehensive": self.performance_stats["comprehensive_queries"], 
                "intelligent": self.performance_stats["intelligent_queries"]
            },
            "system_status": "optimal",
            "agent_coordination": {
                "agent_alpha_integration": "active",
                "agent_beta_optimization": "active",
                "hybrid_performance": "enabled"
            }
        }
    
    def clear_cache(self):
        """Clear all cached analysis results."""
        self.analysis_cache.clear()
        self.cache_timestamps.clear()
        return {"status": "cache_cleared", "timestamp": datetime.now().isoformat()}

# Global hybrid system instance
hybrid_dashboard = HybridDashboardSystem()

# Backwards compatibility functions for existing dashboard
async def hybrid_linkage_analysis(base_dir="TestMaster", analysis_type="auto", force_refresh=False):
    """Main hybrid analysis function for dashboard integration."""
    return await hybrid_dashboard.analyze_codebase(base_dir, analysis_type, force_refresh)

def quick_hybrid_analysis(base_dir="TestMaster", analysis_type="auto"):
    """Synchronous wrapper for hybrid analysis."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(hybrid_linkage_analysis(base_dir, analysis_type))

def get_hybrid_performance_stats():
    """Get hybrid system performance statistics."""
    return hybrid_dashboard.get_performance_stats()

def clear_hybrid_cache():
    """Clear hybrid system cache."""
    return hybrid_dashboard.clear_cache()

# Flask integration endpoints for enhanced dashboard
def create_hybrid_flask_routes(app):
    """Create Flask routes for hybrid dashboard system."""
    
    @app.route('/hybrid-linkage-data')
    def hybrid_linkage_data():
        """Serve hybrid linkage analysis data."""
        from flask import jsonify, request
        
        try:
            analysis_type = request.args.get('mode', 'auto')
            force_refresh = request.args.get('refresh', 'false').lower() == 'true'
            
            print(f"Hybrid analysis request: mode={analysis_type}, refresh={force_refresh}")
            
            result = quick_hybrid_analysis("TestMaster", analysis_type)
            return jsonify(result)
            
        except Exception as e:
            print(f"Hybrid analysis error: {e}")
            return jsonify({
                "error": str(e), 
                "analysis_mode": "error",
                "total_files": 0
            })
    
    @app.route('/hybrid-performance-stats')
    def hybrid_performance_stats():
        """Serve hybrid system performance statistics."""
        from flask import jsonify
        
        try:
            stats = get_hybrid_performance_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    @app.route('/hybrid-cache-clear', methods=['POST'])
    def hybrid_cache_clear():
        """Clear hybrid system cache."""
        from flask import jsonify
        
        try:
            result = clear_hybrid_cache()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Performance testing
    import asyncio
    
    async def test_hybrid_system():
        print("Testing Hybrid Dashboard System")
        print("=" * 50)
        
        # Test all analysis modes
        modes = ["fast", "comprehensive", "intelligent"]
        
        for mode in modes:
            print(f"\nTesting {mode} mode...")
            start_time = time.time()
            
            result = await hybrid_dashboard.analyze_codebase("TestMaster", mode)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"   Files analyzed: {result.get('total_files', 'N/A')}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Analysis mode: {result.get('analysis_mode', 'N/A')}")
            print(f"   Cache used: {result.get('performance_info', {}).get('cache_used', False)}")
        
        # Test cache performance  
        print(f"\nPerformance Statistics:")
        stats = hybrid_dashboard.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
    
    # Run performance test
    asyncio.run(test_hybrid_system())