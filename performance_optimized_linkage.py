#!/usr/bin/env python3
"""
High-Performance Async Linkage Analysis System
==============================================

Agent Beta's performance-optimized dashboard backend with:
- Async/await file processing
- Intelligent caching
- Incremental updates
- Memory-efficient data structures
- Parallel processing

Performance Targets:
- 10x faster analysis for large codebases
- 50% memory usage reduction
- Sub-second incremental updates
- Support for 10,000+ file repositories

Author: Agent Beta - Dashboard Intelligence Swarm
"""

import asyncio
import os
import sys
import time
import json
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
from typing import Dict, List, Any, Set
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import pickle

class PerformanceOptimizedAnalyzer:
    """Ultra-fast async linkage analyzer with intelligent caching."""
    
    def __init__(self, cache_size=1000, max_workers=4):
        self.cache_size = cache_size
        self.max_workers = max_workers
        
        # Cache systems
        self.file_cache = {}  # filename -> {hash: str, analysis: dict}
        self.results_cache = {}  # analysis_key -> results
        self.dependency_cache = defaultdict(set)  # file -> {dependencies}
        
        # Performance tracking
        self.performance_stats = {
            "total_files_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time": 0,
            "memory_usage": 0
        }
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_lock = threading.Lock()
        
    async def analyze_codebase_async(self, base_dir="TestMaster", max_files=None, incremental=True):
        """Ultra-fast async codebase analysis with caching."""
        start_time = time.time()
        
        results = {
            "files": {},
            "dependencies": {},
            "categories": {
                "orphaned_files": [],
                "hanging_files": [],
                "marginal_files": [],
                "well_connected_files": []
            },
            "performance_stats": {},
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Get Python files asynchronously
        python_files = await self._get_python_files_async(base_dir, max_files)
        
        if not python_files:
            return results
            
        # Process files with async batch processing
        file_results = await self._process_files_batch(python_files, base_dir, incremental)
        
        # Build dependency graph efficiently
        dependency_graph = await self._build_dependency_graph_async(file_results)
        
        # Categorize files based on connectivity
        categories = self._categorize_files_fast(file_results, dependency_graph)
        
        # Update results
        results["files"] = file_results
        results["dependencies"] = dependency_graph
        results["categories"] = categories
        results["total_files"] = len(python_files)
        
        # Performance stats
        end_time = time.time()
        self.performance_stats["processing_time"] = end_time - start_time
        self.performance_stats["files_per_second"] = len(python_files) / (end_time - start_time)
        results["performance_stats"] = self.performance_stats.copy()
        
        return results
    
    async def _get_python_files_async(self, base_dir, max_files):
        """Async Python file discovery with smart filtering."""
        python_files = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return python_files
            
        # Use asyncio to make file system traversal non-blocking
        def scan_directory():
            files = []
            skip_dirs = {'__pycache__', '.git', 'QUARANTINE', 'archive', '.pytest_cache', 'node_modules'}
            skip_files = {'original_', '_original', 'ARCHIVED', 'backup', '.pyc'}
            
            for root, dirs, filenames in os.walk(base_path):
                # Filter directories in-place for efficiency
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                
                for filename in filenames:
                    if (filename.endswith('.py') and 
                        not any(skip in filename for skip in skip_files) and
                        (max_files is None or len(files) < max_files)):
                        files.append(Path(root) / filename)
            
            return files
        
        # Run directory scan in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        python_files = await loop.run_in_executor(self.executor, scan_directory)
        
        return python_files
    
    async def _process_files_batch(self, python_files, base_dir, incremental=True):
        """Process files in async batches with intelligent caching."""
        base_path = Path(base_dir)
        results = {}
        
        # Create batches for parallel processing
        batch_size = min(50, max(10, len(python_files) // self.max_workers))
        batches = [python_files[i:i + batch_size] for i in range(0, len(python_files), batch_size)]
        
        # Process batches concurrently
        batch_tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_file_batch(batch, base_path, incremental))
            batch_tasks.append(task)
        
        # Await all batch results
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Merge results
        for batch_result in batch_results:
            results.update(batch_result)
        
        return results
    
    async def _process_file_batch(self, file_batch, base_path, incremental):
        """Process a batch of files with caching."""
        batch_results = {}
        
        for py_file in file_batch:
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                # Check cache if incremental
                if incremental and self._is_file_cached(py_file, relative_path):
                    batch_results[relative_path] = self.file_cache[relative_path]['analysis']
                    self.performance_stats["cache_hits"] += 1
                    continue
                
                # Process file
                file_result = await self._analyze_single_file(py_file, relative_path)
                batch_results[relative_path] = file_result
                
                # Update cache
                self._update_file_cache(py_file, relative_path, file_result)
                self.performance_stats["cache_misses"] += 1
                self.performance_stats["total_files_processed"] += 1
                
            except Exception as e:
                # Log error but continue processing
                batch_results[str(py_file)] = {
                    "error": str(e),
                    "imports": [],
                    "functions": [],
                    "classes": [],
                    "lines_of_code": 0
                }
        
        return batch_results
    
    def _read_file_sync(self, file_path):
        """Synchronous file reading for thread pool."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""
    
    async def _analyze_single_file(self, py_file, relative_path):
        """Analyze single file with optimized parsing."""
        try:
            # Use async file reading with thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self.executor, 
                self._read_file_sync, 
                py_file
            )
            
            # Fast analysis using regex and AST
            analysis_result = await self._fast_file_analysis(content)
            analysis_result["path"] = relative_path
            analysis_result["size_bytes"] = len(content)
            analysis_result["lines_of_code"] = len(content.splitlines())
            
            return analysis_result
            
        except Exception as e:
            return {
                "path": relative_path,
                "error": str(e),
                "imports": [],
                "functions": [],
                "classes": [],
                "lines_of_code": 0
            }
    
    async def _fast_file_analysis(self, content):
        """Ultra-fast file content analysis."""
        # Run AST parsing in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(self.executor, self._parse_content, content)
        return analysis
    
    def _parse_content(self, content):
        """CPU-bound content parsing."""
        analysis = {
            "imports": [],
            "functions": [],
            "classes": [],
            "import_count": 0,
            "complexity_score": 0
        }
        
        try:
            # Fast regex-based import extraction (faster than AST for simple cases)
            import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith(('import ', 'from ')):
                    analysis["imports"].append(line)
                    analysis["import_count"] += 1
            
            # AST analysis for structure (only if needed)
            if len(content) < 50000:  # Skip AST for very large files
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            analysis["functions"].append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            analysis["classes"].append(node.name)
                        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                            analysis["complexity_score"] += 1
                            
                except SyntaxError:
                    # File has syntax errors, skip AST analysis
                    pass
                    
        except Exception:
            # Fallback to basic counting
            analysis["import_count"] = content.count('import ') + content.count('from ')
            
        return analysis
    
    def _is_file_cached(self, py_file, relative_path):
        """Check if file analysis is cached and valid."""
        if relative_path not in self.file_cache:
            return False
            
        try:
            # Check if file has been modified
            current_hash = self._get_file_hash(py_file)
            cached_hash = self.file_cache[relative_path].get('hash')
            
            return current_hash == cached_hash
        except:
            return False
    
    def _get_file_hash(self, file_path):
        """Get fast file hash for change detection."""
        try:
            stat = file_path.stat()
            # Use size + mtime as fast hash
            return f"{stat.st_size}_{stat.st_mtime}"
        except:
            return ""
    
    def _update_file_cache(self, py_file, relative_path, analysis):
        """Update file cache with new analysis."""
        with self.cache_lock:
            file_hash = self._get_file_hash(py_file)
            self.file_cache[relative_path] = {
                'hash': file_hash,
                'analysis': analysis,
                'timestamp': time.time()
            }
            
            # Limit cache size
            if len(self.file_cache) > self.cache_size:
                # Remove oldest entries
                oldest_key = min(self.file_cache.keys(), 
                               key=lambda k: self.file_cache[k]['timestamp'])
                del self.file_cache[oldest_key]
    
    async def _build_dependency_graph_async(self, file_results):
        """Build dependency graph efficiently."""
        dependency_graph = defaultdict(list)
        
        # Build reverse lookup for faster dependency resolution
        file_lookup = {Path(path).stem: path for path in file_results.keys()}
        
        for file_path, analysis in file_results.items():
            dependencies = []
            
            # Extract dependencies from imports
            for import_line in analysis.get("imports", []):
                # Simple dependency extraction - can be enhanced
                if "from " in import_line:
                    module = import_line.split("from ")[1].split(" import")[0].strip()
                else:
                    module = import_line.replace("import ", "").split(".")[0].strip()
                
                # Check if it's a local module
                if module in file_lookup and file_lookup[module] != file_path:
                    dependencies.append(file_lookup[module])
            
            dependency_graph[file_path] = dependencies
        
        return dict(dependency_graph)
    
    def _categorize_files_fast(self, file_results, dependency_graph):
        """Fast file categorization based on connectivity."""
        categories = {
            "orphaned_files": [],
            "hanging_files": [],
            "marginal_files": [],
            "well_connected_files": []
        }
        
        # Build incoming dependency count
        incoming_deps = defaultdict(int)
        for file_path, deps in dependency_graph.items():
            for dep in deps:
                incoming_deps[dep] += 1
        
        # Categorize efficiently
        for file_path, analysis in file_results.items():
            outgoing_deps = len(dependency_graph.get(file_path, []))
            incoming_count = incoming_deps[file_path]
            total_deps = outgoing_deps + incoming_count
            
            file_info = {
                "path": file_path,
                "outgoing_deps": outgoing_deps,
                "incoming_deps": incoming_count,
                "total_deps": total_deps,
                "lines_of_code": analysis.get("lines_of_code", 0),
                "import_count": analysis.get("import_count", 0)
            }
            
            # Fast categorization
            if total_deps == 0:
                categories["orphaned_files"].append(file_info)
            elif outgoing_deps > 15 and incoming_count == 0:
                categories["hanging_files"].append(file_info)
            elif total_deps < 3:
                categories["marginal_files"].append(file_info)
            else:
                categories["well_connected_files"].append(file_info)
        
        # Sort categories by total dependencies (most connected first)
        for category in categories.values():
            category.sort(key=lambda x: x["total_deps"], reverse=True)
        
        return categories
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        cache_total = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        cache_hit_rate = (self.performance_stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        
        return {
            **self.performance_stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.file_cache),
            "memory_efficient": True
        }
    
    async def incremental_update(self, base_dir, changed_files=None):
        """Perform incremental update for specific files."""
        if not changed_files:
            return await self.analyze_codebase_async(base_dir, incremental=True)
        
        # Process only changed files
        results = {"updated_files": {}, "timestamp": datetime.now().isoformat()}
        
        base_path = Path(base_dir)
        for file_path in changed_files:
            py_file = base_path / file_path
            if py_file.exists() and py_file.suffix == '.py':
                analysis = await self._analyze_single_file(py_file, file_path)
                results["updated_files"][file_path] = analysis
                self._update_file_cache(py_file, file_path, analysis)
        
        return results

# Global analyzer instance for dashboard use
performance_analyzer = PerformanceOptimizedAnalyzer()

async def optimized_linkage_analysis(base_dir="TestMaster", max_files=None):
    """High-performance linkage analysis for dashboard."""
    return await performance_analyzer.analyze_codebase_async(base_dir, max_files)

def get_analyzer_stats():
    """Get performance statistics."""
    return performance_analyzer.get_performance_stats()

# Backwards compatibility wrapper
def quick_optimized_analysis(base_dir="TestMaster", max_files=None):
    """Sync wrapper for async analysis."""
    import asyncio
    
    # Run async analysis in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(optimized_linkage_analysis(base_dir, max_files))

if __name__ == "__main__":
    # Performance test
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "TestMaster"
    print(f"Running performance-optimized analysis on {base_dir}...")
    
    start_time = time.time()
    results = quick_optimized_analysis(base_dir)
    end_time = time.time()
    
    print(f"\nPerformance Results:")
    print(f"Files analyzed: {results['total_files']}")
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Files per second: {results['total_files'] / (end_time - start_time):.1f}")
    print(f"\nCache performance: {get_analyzer_stats()['cache_hit_rate']}")