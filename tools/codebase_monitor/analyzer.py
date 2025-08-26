#!/usr/bin/env python3
"""
Codebase Monitor Analyzer
========================

Generates comprehensive reports about codebase health including:
- File statistics by extension
- Directory summaries 
- Duplicate file detection
- Code quality hotspots
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any


class CodebaseAnalyzer:
    def __init__(self, root_path: str, output_dir: str):
        self.root = Path(root_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.total_files = 0
        self.total_size = 0
        self.total_lines = 0
        self.extensions = defaultdict(lambda: {"files": 0, "lines": 0, "bytes": 0})
        self.duplicates = []
        self.hotspots = defaultdict(list)
        self.directory_summaries = []
        
        # Hash tracking for duplicates
        self.file_hashes = defaultdict(list)
        
        # Ignore patterns
        self.ignore_patterns = {
            ".git", "__pycache__", ".venv", "node_modules", 
            ".pytest_cache", ".mypy_cache", ".vscode", ".idea",
            "logs", "*.log", "*.pyc", "*.pyo", "*.pyd",
            "PRODUCTION_PACKAGES", "organized_codebase/monitoring",
            "tools/codebase_monitor/reports"
        }
    
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored based on patterns."""
        path_str = str(path).replace('\\', '/')
        for pattern in self.ignore_patterns:
            if pattern in path_str or path.name == pattern:
                return True
            if pattern.startswith("*.") and path.suffix == pattern[1:]:
                return True
        return False
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file for basic metrics."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            
            # Count lines for text files
            lines = 0
            if size < 10 * 1024 * 1024:  # Only read files < 10MB
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = sum(1 for _ in f)
                except (UnicodeDecodeError, PermissionError):
                    # Binary file or no permission
                    pass
            
            # Calculate file hash for duplicate detection
            file_hash = ""
            if size > 0 and size < 50 * 1024 * 1024:  # Hash files < 50MB
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        self.file_hashes[file_hash].append(str(file_path.relative_to(self.root)))
                except (PermissionError, OSError):
                    pass
            
            return {
                "path": file_path,
                "size": size,
                "lines": lines,
                "hash": file_hash,
                "extension": file_path.suffix.lower()
            }
        except (OSError, PermissionError):
            return None
    
    def detect_hotspots(self, file_path: Path, file_info: Dict[str, Any]):
        """Detect various code quality hotspots."""
        rel_path = str(file_path.relative_to(self.root))
        
        # Large files
        if file_info["size"] > 1024 * 1024:  # > 1MB
            self.hotspots["large_files"].append(rel_path)
        
        # Long files
        if file_info["lines"] > 1000:
            self.hotspots["long_files"].append(rel_path)
        
        # For Python files, check for specific patterns
        if file_path.suffix == '.py' and file_info["lines"] > 0:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # High complexity indicators
                if content.count('if ') + content.count('for ') + content.count('while ') > 50:
                    self.hotspots["high_branching_python"].append(rel_path)
                
                # Long average function length (heuristic)
                func_count = content.count('def ')
                if func_count > 0 and file_info["lines"] / func_count > 60:
                    self.hotspots["long_avg_function_lines"].append(rel_path)
                    
            except (UnicodeDecodeError, PermissionError):
                pass
        
        # For TypeScript/JavaScript files
        if file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # ": any" usage
                any_count = content.count(': any')
                if any_count > 20:
                    self.hotspots["ts_any_overuse"].append(rel_path)
                
                # ESLint disable comments
                disable_count = content.count('eslint-disable') + content.count('ts-ignore')
                if disable_count > 10:
                    self.hotspots["eslint_ts_ignored_heavy"].append(rel_path)
                    
            except (UnicodeDecodeError, PermissionError):
                pass
    
    def scan_directory(self) -> None:
        """Scan the entire directory tree."""
        print(f"Scanning {self.root}...")
        
        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)
            
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(root_path / d)]
            
            # Analyze directory
            dir_size = 0
            dir_files = 0
            
            for file in files:
                file_path = root_path / file
                
                if self.should_ignore(file_path):
                    continue
                
                file_info = self.analyze_file(file_path)
                if file_info is None:
                    continue
                
                # Update totals
                self.total_files += 1
                self.total_size += file_info["size"]
                self.total_lines += file_info["lines"]
                
                # Update by extension
                ext = file_info["extension"] or ""
                self.extensions[ext]["files"] += 1
                self.extensions[ext]["lines"] += file_info["lines"]
                self.extensions[ext]["bytes"] += file_info["size"]
                
                # Directory stats
                dir_size += file_info["size"]
                dir_files += 1
                
                # Detect hotspots
                self.detect_hotspots(file_path, file_info)
            
            # Add directory summary
            if dir_files > 0:
                rel_dir = str(root_path.relative_to(self.root))
                self.directory_summaries.append({
                    "rel_dir": rel_dir,
                    "total_files": dir_files,
                    "total_size_bytes": dir_size
                })
    
    def find_duplicates(self) -> None:
        """Identify duplicate files based on content hash."""
        for file_hash, paths in self.file_hashes.items():
            if len(paths) > 1:
                self.duplicates.append(paths)
        
        # Sort by number of duplicates (descending)
        self.duplicates.sort(key=len, reverse=True)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate the main analysis report."""
        timestamp = time.time()
        
        report = {
            "root": str(self.root),
            "generated_at_epoch": timestamp,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size,
            "total_code_lines": self.total_lines,
            "extensions": dict(self.extensions),
            "directory_summaries": self.directory_summaries,
            "duplicates": self.duplicates,
            "hotspots": dict(self.hotspots)
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save report to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_filename = f"scan_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save summary markdown
        summary_path = self.output_dir / f"summary_{timestamp}.md"
        self.save_summary(report, summary_path)
        
        print(json.dumps({"json_report": str(json_path), "summary_report": str(summary_path)}))
        return str(json_path)
    
    def save_summary(self, report: Dict[str, Any], summary_path: Path):
        """Save human-readable summary."""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Codebase Health Summary\\n\\n")
            f.write(f"**Generated:** {time.ctime(report['generated_at_epoch'])}\\n\\n")
            f.write(f"## Overview\\n")
            f.write(f"- **Total Files:** {report['total_files']:,}\\n")
            f.write(f"- **Total Size:** {report['total_size_bytes']:,} bytes\\n")
            f.write(f"- **Total Lines:** {report['total_code_lines']:,}\\n\\n")
            
            f.write(f"## Top Extensions\\n")
            top_exts = sorted(report['extensions'].items(), 
                            key=lambda x: x[1]['files'], reverse=True)[:10]
            for ext, data in top_exts:
                ext_name = ext or "(no extension)"
                f.write(f"- **{ext_name}**: {data['files']} files, {data['bytes']:,} bytes\\n")
            
            f.write(f"\\n## Hotspots\\n")
            for hotspot_type, paths in report['hotspots'].items():
                if paths:
                    f.write(f"- **{hotspot_type}**: {len(paths)} files\\n")
            
            f.write(f"\\n## Duplicates\\n")
            f.write(f"- **Duplicate Groups:** {len(report['duplicates'])}\\n")
            if report['duplicates']:
                total_dupes = sum(len(group) for group in report['duplicates'])
                f.write(f"- **Total Duplicate Files:** {total_dupes}\\n")
    
    def run(self) -> str:
        """Run the complete analysis."""
        self.scan_directory()
        self.find_duplicates()
        report = self.generate_report()
        return self.save_report(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze codebase health')
    parser.add_argument('--root', default='.', help='Root directory to analyze')
    parser.add_argument('--output-dir', default='./reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    analyzer = CodebaseAnalyzer(args.root, args.output_dir)
    report_path = analyzer.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())