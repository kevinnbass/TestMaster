#!/usr/bin/env python3
"""
LLM Intelligence Scanner Storage Module
=======================================

Handles caching, file operations, and storage for the LLM intelligence scanner.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.sandbox.scanner_models import LLMIntelligenceEntry, LLMIntelligenceMap


class CacheManager:
    """Manages caching of LLM analysis results"""

    def __init__(self, cache_dir: Path, logger):
        """Initialize the cache manager"""
        self.cache_dir = cache_dir
        self.logger = logger
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load existing cache from previous scans"""
        cache_file = self.cache_dir / "llm_intelligence_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}

    def save_cache(self) -> None:
        """Save current cache to disk"""
        cache_file = self.cache_dir / "llm_intelligence_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def get_cached_entry(self, file_path: Path, root_dir: Path) -> Optional[Dict[str, Any]]:
        """Get cached entry if it exists and is still valid"""
        cache_key = str(file_path.relative_to(root_dir))
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            current_hash = self.calculate_file_hash(file_path)

            if cached_data.get('file_hash') == current_hash:
                self.logger.info(f"Using cached analysis for: {file_path}")
                return cached_data
        return None

    def update_cache(self, entries: List[LLMIntelligenceEntry]) -> None:
        """Update cache with new entries"""
        for entry in entries:
            cache_key = entry.relative_path
            cache_data = {
                'full_path': entry.full_path,
                'relative_path': entry.relative_path,
                'file_hash': entry.file_hash,
                'analysis_timestamp': entry.analysis_timestamp,
                'module_summary': entry.module_summary,
                'functionality_details': entry.functionality_details,
                'dependencies_analysis': entry.dependencies_analysis,
                'security_implications': entry.security_implications,
                'testing_requirements': entry.testing_requirements,
                'architectural_role': entry.architectural_role,
                'primary_classification': entry.primary_classification,
                'secondary_classifications': entry.secondary_classifications,
                'reorganization_recommendations': entry.reorganization_recommendations,
                'confidence_score': entry.confidence_score,
                'key_features': entry.key_features,
                'integration_points': entry.integration_points,
                'complexity_assessment': entry.complexity_assessment,
                'maintainability_notes': entry.maintainability_notes,
                'file_size': entry.file_size,
                'line_count': entry.line_count,
                'class_count': entry.class_count,
                'function_count': entry.function_count
            }
            self.cache[cache_key] = cache_data

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class FileOperations:
    """Handles file operations for the scanner"""

    def __init__(self, root_dir: Path, exclusions: Set[str], logger):
        """Initialize file operations"""
        self.root_dir = root_dir
        self.exclusions = exclusions
        self.logger = logger

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded from scanning"""
        if not path.exists() or not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
            return any(part in self.exclusions for part in rel_path.parts)
        except ValueError:
            return True

    def find_python_files(self) -> List[Path]:
        """Find all Python files to analyze"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclusions)]

            for file in files:
                if file.endswith('.py') and not any(excl in file for excl in ['test_', 'setup.py']):
                    file_path = Path(root) / file
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)

        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        return python_files

    def calculate_file_statistics(self, content: str) -> Dict[str, int]:
        """Calculate basic file statistics"""
        lines = content.split('\n')
        class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))

        return {
            'line_count': len(lines),
            'class_count': class_count,
            'function_count': function_count,
            'file_size': len(content.encode('utf-8'))
        }


class IntelligenceMapBuilder:
    """Builds the final intelligence map from analyzed entries"""

    def __init__(self, logger):
        """Initialize the map builder"""
        self.logger = logger

    def build_intelligence_map(self, entries: List[LLMIntelligenceEntry],
                              insights_generator) -> LLMIntelligenceMap:
        """Build complete intelligence map from entries"""

        # Calculate totals
        total_lines = sum(entry.line_count for entry in entries)

        # Create classification summary
        classification_summary = insights_generator.create_classification_summary(entries)

        # Generate reorganization insights
        reorganization_insights = insights_generator.generate_reorganization_insights(
            entries, classification_summary
        )

        # Build directory structure
        directory_structure = self._build_directory_structure(entries)

        # Create intelligence map
        intelligence_map = LLMIntelligenceMap(
            scan_timestamp=datetime.now().isoformat(),
            total_files_scanned=len(entries),
            total_lines_analyzed=total_lines,
            directory_structure=directory_structure,
            intelligence_entries=entries,
            classification_summary=classification_summary,
            reorganization_insights=reorganization_insights,
            scan_metadata=self._get_scan_metadata(len(entries), total_lines)
        )

        return intelligence_map

    def _build_directory_structure(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Build directory structure representation"""
        structure = {}

        for entry in entries:
            path_parts = Path(entry.relative_path).parts
            current_level = structure

            for i, part in enumerate(path_parts[:-1]):  # All parts except filename
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Add file info at the leaf
            if '__files__' not in current_level:
                current_level['__files__'] = []
            current_level['__files__'].append(path_parts[-1])

        return structure

    def _get_scan_metadata(self, file_count: int, line_count: int) -> Dict[str, Any]:
        """Get scan metadata"""
        return {
            'scan_completed': datetime.now().isoformat(),
            'files_analyzed': file_count,
            'lines_analyzed': line_count,
            'scanner_version': '4.0'
        }

    def save_intelligence_map(self, intelligence_map: LLMIntelligenceMap, output_file: Path) -> None:
        """Save intelligence map to file"""
        try:
            # Convert to dictionary for JSON serialization
            map_dict = {
                'scan_timestamp': intelligence_map.scan_timestamp,
                'total_files_scanned': intelligence_map.total_files_scanned,
                'total_lines_analyzed': intelligence_map.total_lines_analyzed,
                'directory_structure': intelligence_map.directory_structure,
                'intelligence_entries': [entry.__dict__ for entry in intelligence_map.intelligence_entries],
                'classification_summary': intelligence_map.classification_summary,
                'reorganization_insights': intelligence_map.reorganization_insights,
                'scan_metadata': intelligence_map.scan_metadata
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(map_dict, f, indent=2, default=str, ensure_ascii=False)

            self.logger.info(f"Intelligence map saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save intelligence map: {e}")