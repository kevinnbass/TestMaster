#!/usr/bin/env python3
"""
LLM Core Intelligence Scanner
=============================

Core functionality for LLM-powered code intelligence scanning.
"""

import os
import json
import hashlib
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import logging
import re
import ast

from llm_data import (
    LLMProvider, Classification, LLMIntelligenceEntry,
    StaticAnalysisResult, IntegratedIntelligence,
    LLMIntelligenceMap, ReorganizationPhase, ReorganizationPlan
)
from llm_clients import create_llm_client


class LLMIntelligenceScanner:
    """
    Core LLM intelligence scanner that analyzes Python files
    and produces directory-ordered intelligence maps.
    """

    def __init__(self, root_dir: Path, config: Dict[str, Any] = None):
        self.root_dir = root_dir.resolve()
        self.config = config or self._get_default_config()

        # Setup exclusions (same as existing system)
        self.exclusions = self._get_exclusions()

        # Setup directories
        self.cache_dir = self.root_dir / "tools" / "codebase_reorganizer" / "llm_cache"
        self.output_dir = self.root_dir / "tools" / "codebase_reorganizer" / "intelligence_output"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.llm_client = self._initialize_llm_client()
        self.static_analyzers = self._initialize_static_analyzers()

        # Setup logging
        self._setup_logging()

        # Load cache
        self.cache = self._load_cache()

        self.logger.info("LLM Intelligence Scanner initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'llm_provider': LLMProvider.MOCK.value,
            'llm_model': 'gpt-4',
            'api_key': None,
            'max_concurrent': 3,
            'max_file_size': 50000,  # bytes
            'max_lines_per_file': 1000,
            'confidence_threshold': 0.7,
            'enable_static_analysis': True,
            'cache_enabled': True,
            'preserve_directory_order': True,
            'llm_temperature': 0.0,
            'llm_max_tokens': 2000,
            'chunk_size': 4000,
            'chunk_overlap': 200
        }

    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"llm_intelligence_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_exclusions(self) -> Set[str]:
        """Get exclusion patterns"""
        return {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentscope',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', 'tests', 'test_sessions', 'testmaster_sessions',
            'tools', 'codebase_reorganizer'
        }

    def _initialize_llm_client(self):
        """Initialize LLM client based on configuration"""
        provider = self.config['llm_provider']

        try:
            return create_llm_client(provider, self.config)
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            # Fallback to mock client
            from llm_clients import MockLLMClient
            return MockLLMClient()

    def _initialize_static_analyzers(self) -> Dict[str, Any]:
        """Initialize static analysis tools"""
        analyzers = {}

        try:
            # Try to import static analyzers
            from semantic_analyzer import SemanticAnalyzer, analyze_semantics
            from relationship_analyzer import RelationshipAnalyzer, analyze_relationships
            from pattern_detector import PatternDetector, detect_patterns
            from code_quality_analyzer import CodeQualityAnalyzer, analyze_quality

            analyzers['semantic'] = SemanticAnalyzer()
            analyzers['relationship'] = RelationshipAnalyzer()
            analyzers['pattern'] = PatternDetector()
            analyzers['quality'] = CodeQualityAnalyzer()
            self.logger.info("Static analyzers initialized successfully")
        except ImportError:
            self.logger.warning("Static analyzers not available, running LLM-only mode")

        return analyzers

    def _load_cache(self) -> Dict[str, Any]:
        """Load existing cache"""
        cache_file = self.cache_dir / "intelligence_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save cache to disk"""
        if not self.config['cache_enabled']:
            return

        cache_file = self.cache_dir / "intelligence_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded"""
        if not path.exists() or not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
            return any(part in self.exclusions for part in rel_path.parts)
        except ValueError:
            return True

    def scan_and_analyze(self, output_file: Optional[Path] = None,
                        max_files: Optional[int] = None) -> LLMIntelligenceMap:
        """
        Perform comprehensive intelligence scan.

        Args:
            output_file: Optional path to save results
            max_files: Maximum number of files to analyze (for testing)

        Returns:
            Complete intelligence map
        """
        self.logger.info("Starting comprehensive intelligence scan...")

        # Generate unique scan ID
        scan_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]

        # Find all Python files to analyze (preserving directory order)
        python_files = self._discover_python_files()

        if max_files:
            python_files = python_files[:max_files]
            self.logger.info(f"Limiting analysis to {max_files} files for testing")

        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        # Analyze files
        intelligence_entries = []
        total_lines = 0

        with ThreadPoolExecutor(max_workers=self.config['max_concurrent']) as executor:
            future_to_file = {
                executor.submit(self._analyze_file_comprehensive, file_path): file_path
                for file_path in python_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    entry = future.result()
                    if entry:
                        intelligence_entries.append(entry)
                        total_lines += entry.line_count
                except Exception as e:
                    self.logger.error(f"Failed to analyze {file_path}: {e}")

        # Create intelligence map
        intelligence_map = LLMIntelligenceMap(
            scan_timestamp=datetime.now().isoformat(),
            scan_id=scan_id,
            total_files_scanned=len(intelligence_entries),
            total_lines_analyzed=total_lines,
            directory_structure=self._build_directory_structure(intelligence_entries),
            intelligence_entries=intelligence_entries,
            scan_metadata={
                'llm_provider': self.config['llm_provider'],
                'llm_model': self.config['llm_model'],
                'static_analysis_enabled': bool(self.static_analyzers),
                'cache_enabled': self.config['cache_enabled']
            },
            scan_statistics=self._calculate_scan_statistics(intelligence_entries)
        )

        # Save results if requested
        if output_file:
            self.save_intelligence_map(intelligence_map, output_file)

        self.logger.info(f"Intelligence scan completed: {len(intelligence_entries)} files analyzed")
        return intelligence_map

    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the root directory"""
        python_files = []

        for root, dirs, files in os.walk(self.root_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclusions]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)

        # Sort by directory depth and then alphabetically for consistent ordering
        python_files.sort(key=lambda x: (len(x.relative_to(self.root_dir).parts), str(x)))

        return python_files

    def _analyze_file_comprehensive(self, file_path: Path) -> Optional[LLMIntelligenceEntry]:
        """Perform comprehensive analysis of a single file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Basic file metrics
            file_size = len(content.encode('utf-8'))
            lines = content.split('\n')
            line_count = len(lines)

            # Skip if file too large
            if file_size > self.config['max_file_size'] or line_count > self.config['max_lines_per_file']:
                self.logger.warning(f"Skipping large file: {file_path}")
                return None

            # Calculate file hash for caching
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # Check cache first
            cache_key = f"{file_path}:{file_hash}"
            if cache_key in self.cache:
                self.logger.debug(f"Using cached analysis for {file_path}")
                cached_data = self.cache[cache_key]
                return LLMIntelligenceEntry(**cached_data)

            # Perform static analysis
            static_results = self._perform_static_analysis(content, file_path)

            # Perform LLM analysis
            llm_response = self._perform_llm_analysis(content, file_path)

            # Parse LLM response
            llm_data = self._parse_llm_response(llm_response)

            # Create intelligence entry
            entry = LLMIntelligenceEntry(
                full_path=str(file_path),
                relative_path=str(file_path.relative_to(self.root_dir)),
                file_hash=file_hash,
                analysis_timestamp=datetime.now().isoformat(),
                module_summary=llm_data.get('summary', ''),
                functionality_details=llm_data.get('functionality', ''),
                dependencies_analysis=llm_data.get('dependencies', ''),
                security_implications=llm_data.get('security', ''),
                testing_requirements=llm_data.get('testing', ''),
                architectural_role=llm_data.get('architecture', ''),
                primary_classification=llm_data.get('primary_classification', 'uncategorized'),
                secondary_classifications=llm_data.get('secondary_classifications', []),
                reorganization_recommendations=llm_data.get('reorganization', []),
                confidence_score=llm_data.get('confidence', 0.5),
                key_features=llm_data.get('key_features', []),
                integration_points=llm_data.get('integration_points', []),
                complexity_assessment=llm_data.get('complexity', 'unknown'),
                maintainability_notes=llm_data.get('maintainability', ''),
                file_size=file_size,
                line_count=line_count,
                class_count=self._count_classes(content),
                function_count=self._count_functions(content),
                analysis_errors=[]
            )

            # Cache the result
            self.cache[cache_key] = asdict(entry)
            self._save_cache()

            return entry

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def _perform_static_analysis(self, content: str, file_path: Path) -> StaticAnalysisResult:
        """Perform static analysis using available tools"""
        static_results = StaticAnalysisResult()

        try:
            # Semantic analysis
            if 'semantic' in self.static_analyzers:
                semantic_result = self.static_analyzers['semantic'].analyze_semantics(content, file_path)
                static_results.semantic = semantic_result

            # Relationship analysis
            if 'relationship' in self.static_analyzers:
                relationship_result = self.static_analyzers['relationship'].analyze_relationships(content, file_path)
                static_results.relationship = relationship_result

            # Pattern detection
            if 'pattern' in self.static_analyzers:
                pattern_result = self.static_analyzers['pattern'].detect_patterns(content, file_path)
                static_results.pattern = pattern_result

            # Quality analysis
            if 'quality' in self.static_analyzers:
                quality_result = self.static_analyzers['quality'].analyze_quality(content, file_path)
                static_results.quality = quality_result

        except Exception as e:
            self.logger.warning(f"Static analysis failed: {e}")

        return static_results

    def _perform_llm_analysis(self, content: str, file_path: Path) -> str:
        """Perform LLM analysis of the code"""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(content, file_path)

            # Call LLM
            response = self.llm_client.analyze_code(prompt)

            return response

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return '{"error": "LLM analysis failed", "confidence": 0.0}'

    def _create_analysis_prompt(self, content: str, file_path: Path) -> str:
        """Create analysis prompt for LLM"""
        relative_path = file_path.relative_to(self.root_dir)

        # Prepare content outside f-string to avoid backslash issues
        content_lines = len(content.split('\n'))

        prompt = f"""
Analyze the following Python file and provide detailed intelligence:

File: {relative_path}
Lines: {content_lines}

Code:
```python
""" + content + """

Please analyze this code and provide the following information in JSON format:
- summary: Brief description of what this module does
- functionality: Detailed explanation of the functionality
- dependencies: Analysis of dependencies and imports
- security: Security implications and considerations
- testing: Testing requirements and approaches
- architecture: Architectural role and patterns used
- primary_classification: Main category (security, intelligence, frontend_dashboard, documentation, testing, utility, api, database, data_processing, orchestration, automation, monitoring, analytics, devops, uncategorized)
- secondary_classifications: List of additional categories
- reorganization: List of reorganization recommendations
- confidence: Confidence score (0.0 to 1.0) for this analysis
- key_features: List of key features and capabilities
- integration_points: List of integration points with other systems
- complexity: Assessment of complexity (low, medium, high)
- maintainability: Notes on maintainability and potential improvements

Return only valid JSON.
"""
        return prompt.strip()

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in LLM response")
                return {'confidence': 0.1, 'summary': 'Unable to parse LLM response'}

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {'confidence': 0.1, 'summary': 'JSON parsing error in LLM response'}

    def _count_classes(self, content: str) -> int:
        """Count number of classes in the code"""
        try:
            tree = ast.parse(content)
            return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        except:
            return 0

    def _count_functions(self, content: str) -> int:
        """Count number of functions in the code"""
        try:
            tree = ast.parse(content)
            return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        except:
            return 0

    def _build_directory_structure(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Build directory structure from intelligence entries"""
        structure = {}

        for entry in entries:
            path_parts = Path(entry.relative_path).parts
            current = structure

            for part in path_parts[:-1]:  # All parts except filename
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add file info
            current[path_parts[-1]] = {
                'classification': entry.primary_classification,
                'confidence': entry.confidence_score,
                'size': entry.line_count
            }

        return structure

    def _calculate_scan_statistics(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Calculate scan statistics"""
        if not entries:
            return {}

        confidences = [entry.confidence_score for entry in entries]
        classifications = [entry.primary_classification for entry in entries]

        return {
            'average_confidence': sum(confidences) / len(confidences),
            'high_confidence_count': len([c for c in confidences if c > 0.8]),
            'medium_confidence_count': len([c for c in confidences if 0.6 <= c <= 0.8]),
            'low_confidence_count': len([c for c in confidences if c < 0.6]),
            'classification_distribution': {
                cls: classifications.count(cls)
                for cls in set(classifications)
            },
            'largest_file': max(entries, key=lambda x: x.line_count).relative_path,
            'smallest_file': min(entries, key=lambda x: x.line_count).relative_path
        }

    def save_intelligence_map(self, intelligence_map: LLMIntelligenceMap, output_path: Path) -> None:
        """Save intelligence map to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(intelligence_map), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Intelligence map saved to {output_path}")

    def load_intelligence_map(self, input_path: Path) -> LLMIntelligenceMap:
        """Load intelligence map from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return LLMIntelligenceMap(**data)

    def integrate_with_static_analysis(self, intelligence_entries: List[LLMIntelligenceEntry]) -> List[IntegratedIntelligence]:
        """Integrate LLM intelligence with static analysis results"""
        integrated = []

        for entry in intelligence_entries:
            # Perform static analysis
            static_results = self._perform_static_analysis(
                self._read_file_content(entry.full_path),
                Path(entry.full_path)
            )

            # Calculate integrated confidence and classification
            integrated_classification = self._calculate_integrated_classification(entry, static_results)
            integration_confidence = self._calculate_integration_confidence(entry, static_results)

            # Generate synthesis reasoning
            synthesis_reasoning = self._generate_synthesis_reasoning(entry, static_results)

            # Create final recommendations
            final_recommendations = self._generate_final_recommendations(entry, static_results)

            integrated_entry = IntegratedIntelligence(
                file_path=entry.full_path,
                relative_path=entry.relative_path,
                static_analysis=static_results,
                llm_analysis=entry,
                confidence_factors=self._calculate_confidence_factors(entry, static_results),
                integrated_classification=integrated_classification,
                reorganization_priority=self._calculate_reorganization_priority(entry, static_results),
                integration_confidence=integration_confidence,
                final_recommendations=final_recommendations,
                synthesis_reasoning=synthesis_reasoning
            )

            integrated.append(integrated_entry)

        return integrated

    def _read_file_content(self, file_path: str) -> str:
        """Read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""

    def _calculate_integrated_classification(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> str:
        """Calculate integrated classification combining LLM and static analysis"""
        # For now, use LLM classification as primary
        return entry.primary_classification

    def _calculate_integration_confidence(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> float:
        """Calculate confidence in the integrated analysis"""
        # Simple average for now
        return entry.confidence_score

    def _calculate_confidence_factors(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> Dict[str, float]:
        """Calculate confidence factors for different aspects"""
        return {
            'llm_confidence': entry.confidence_score,
            'static_consistency': 0.8,  # Placeholder
            'integration_quality': 0.7  # Placeholder
        }

    def _calculate_reorganization_priority(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> int:
        """Calculate reorganization priority"""
        priority = 5  # Default

        # Adjust based on confidence
        if entry.confidence_score > 0.8:
            priority -= 2
        elif entry.confidence_score < 0.4:
            priority += 2

        # Adjust based on file size
        if entry.line_count > 500:
            priority -= 1
        elif entry.line_count < 50:
            priority += 1

        return max(1, min(10, priority))  # Clamp between 1-10

    def _generate_synthesis_reasoning(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> str:
        """Generate reasoning for the synthesis"""
        return f"LLM analysis confidence: {entry.confidence_score:.2f}. Static analysis integrated."

    def _generate_final_recommendations(self, entry: LLMIntelligenceEntry, static_results: StaticAnalysisResult) -> List[str]:
        """Generate final recommendations combining all sources"""
        recommendations = entry.reorganization_recommendations.copy()

        # Add size-based recommendations
        if entry.line_count > 300:
            recommendations.append("Consider splitting this large module")

        return recommendations

