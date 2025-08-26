#!/usr/bin/env python3
"""
LLM Intelligence System
========================

A comprehensive LLM-powered code intelligence system that:

1. Scans Python files while preserving directory structure
2. Uses LLM analysis to understand code purpose and functionality
3. Integrates with existing static analysis tools
4. Generates phased reorganization plans based on confidence and impact
5. Provides comprehensive JSON intelligence maps for codebase understanding

Author: LLM Intelligence System
Version: 1.0.0
"""

import os
import json
import hashlib
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import logging
import re
import ast
import sys

# Import existing intelligence modules
try:
    from semantic_analyzer import SemanticAnalyzer, analyze_semantics
    from relationship_analyzer import RelationshipAnalyzer, analyze_relationships
    from pattern_detector import PatternDetector, detect_patterns
    from code_quality_analyzer import CodeQualityAnalyzer, analyze_quality
    HAS_STATIC_ANALYZERS = True
except ImportError:
    HAS_STATIC_ANALYZERS = False
    print("Warning: Static analyzers not available, running LLM-only mode")


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    MOCK = "mock"


class Classification(Enum):
    """Standard classification categories"""
    SECURITY = "security"
    INTELLIGENCE = "intelligence"
    FRONTEND_DASHBOARD = "frontend_dashboard"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    UTILITY = "utility"
    API = "api"
    DATABASE = "database"
    DATA_PROCESSING = "data_processing"
    ORCHESTRATION = "orchestration"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"
    DEVOPS = "devops"
    UNCATEGORIZED = "uncategorized"


@dataclass
class LLMIntelligenceEntry:
    """Single entry in the LLM intelligence map"""
    full_path: str
    relative_path: str
    file_hash: str
    analysis_timestamp: str
    module_summary: str = ""
    functionality_details: str = ""
    dependencies_analysis: str = ""
    security_implications: str = ""
    testing_requirements: str = ""
    architectural_role: str = ""
    primary_classification: str = "uncategorized"
    secondary_classifications: List[str] = field(default_factory=list)
    reorganization_recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    key_features: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    complexity_assessment: str = "unknown"
    maintainability_notes: str = ""
    file_size: int = 0
    line_count: int = 0
    class_count: int = 0
    function_count: int = 0
    analysis_errors: List[str] = field(default_factory=list)


@dataclass
class StaticAnalysisResult:
    """Results from static analysis tools"""
    semantic: Dict[str, Any] = field(default_factory=dict)
    relationship: Dict[str, Any] = field(default_factory=dict)
    pattern: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IntegratedIntelligence:
    """Integrated intelligence from multiple sources"""
    file_path: str
    relative_path: str
    static_analysis: StaticAnalysisResult = field(default_factory=StaticAnalysisResult)
    llm_analysis: LLMIntelligenceEntry = field(default_factory=LLMIntelligenceEntry)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    integrated_classification: str = "uncategorized"
    reorganization_priority: int = 5
    integration_confidence: float = 0.5
    final_recommendations: List[str] = field(default_factory=list)
    synthesis_reasoning: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LLMIntelligenceMap:
    """Complete LLM intelligence map"""
    scan_timestamp: str
    scan_id: str
    total_files_scanned: int = 0
    total_lines_analyzed: int = 0
    directory_structure: Dict[str, Any] = field(default_factory=dict)
    intelligence_entries: List[LLMIntelligenceEntry] = field(default_factory=list)
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    scan_statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReorganizationPhase:
    """Single phase in the reorganization plan"""
    phase_number: int
    phase_name: str
    description: str
    modules: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time_minutes: int = 0
    risk_level: str = "medium"
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ReorganizationPlan:
    """Complete reorganization plan"""
    plan_timestamp: str
    total_modules: int = 0
    reorganization_phases: List[ReorganizationPhase] = field(default_factory=list)
    estimated_total_time_hours: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    implementation_guidelines: List[str] = field(default_factory=list)


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
        self.logger.info(f"Root directory: {self.root_dir}")
        self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

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
            'enable_static_analysis': HAS_STATIC_ANALYZERS,
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
        provider = LLMProvider(self.config['llm_provider'])

        if provider == LLMProvider.MOCK:
            return MockLLMClient()
        elif provider == LLMProvider.OPENAI:
            try:
                return OpenAIClient(self.config['api_key'], self.config['llm_model'])
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                return MockLLMClient()
        elif provider == LLMProvider.OLLAMA:
            try:
                return OllamaClient(self.config['llm_model'])
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama client: {e}")
                return MockLLMClient()
        else:
            self.logger.warning(f"Unknown provider {provider}, using mock")
            return MockLLMClient()

    def _initialize_static_analyzers(self) -> Dict[str, Any]:
        """Initialize static analysis tools"""
        analyzers = {}

        if not self.config['enable_static_analysis'] or not HAS_STATIC_ANALYZERS:
            return analyzers

        try:
            analyzers['semantic'] = SemanticAnalyzer()
            analyzers['relationship'] = RelationshipAnalyzer()
            analyzers['pattern'] = PatternDetector()
            analyzers['quality'] = CodeQualityAnalyzer()
            self.logger.info("Static analyzers initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize static analyzers: {e}")

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
                        self.logger.info(f"Analyzed: {entry.relative_path}")
                    else:
                        self.logger.warning(f"Failed to analyze: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")

        # Build directory structure
        directory_structure = self._build_directory_structure(python_files)

        # Calculate scan statistics
        scan_statistics = self._calculate_scan_statistics(intelligence_entries)

        # Create intelligence map
        intelligence_map = LLMIntelligenceMap(
            scan_timestamp=datetime.now().isoformat(),
            scan_id=scan_id,
            total_files_scanned=len(intelligence_entries),
            total_lines_analyzed=total_lines,
            directory_structure=directory_structure,
            intelligence_entries=intelligence_entries,
            scan_metadata=self._get_scan_metadata(),
            scan_statistics=scan_statistics
        )

        # Save results
        if output_file:
            self._save_intelligence_map(intelligence_map, output_file)

        # Update cache
        self._update_cache(intelligence_entries)
        self._save_cache()

        self.logger.info("Intelligence scan completed successfully!")
        self.logger.info(f"Analyzed {len(intelligence_entries)} files with {total_lines} total lines")

        return intelligence_map

    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files while preserving directory order"""
        python_files = []

        # Walk directory tree in order
        for root, dirs, files in os.walk(self.root_dir):
            # Sort directories and files for consistent ordering
            dirs.sort()
            files.sort()

            # Filter directories
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclusions)]

            for file in files:
                if file.endswith('.py') and not any(excl in file for excl in ['test_', 'setup.py']):
                    file_path = Path(root) / file
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)

        return python_files

    def _analyze_file_comprehensive(self, file_path: Path) -> Optional[LLMIntelligenceEntry]:
        """Perform comprehensive analysis of a single file"""
        try:
            # Calculate file hash for caching
            file_hash = self._calculate_file_hash(file_path)

            # Check cache first
            cache_key = str(file_path.relative_to(self.root_dir))
            if (self.config['cache_enabled'] and
                cache_key in self.cache and
                self.cache[cache_key].get('file_hash') == file_hash):
                self.logger.info(f"Using cached analysis for: {file_path}")
                cached_data = self.cache[cache_key]
                return LLMIntelligenceEntry(**cached_data)

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Get basic file statistics
            file_stats = self._get_file_statistics(content, file_path)

            # Perform static analysis if enabled
            static_analysis = StaticAnalysisResult()
            if self.config['enable_static_analysis'] and self.static_analyzers:
                static_analysis = self._perform_static_analysis(file_path, content)

            # Perform LLM analysis
            llm_analysis = self._perform_llm_analysis(file_path, content, file_stats, static_analysis)

            # Create intelligence entry
            entry = LLMIntelligenceEntry(
                full_path=str(file_path),
                relative_path=str(file_path.relative_to(self.root_dir)),
                file_hash=file_hash,
                analysis_timestamp=datetime.now().isoformat(),
                module_summary=llm_analysis.get('summary', ''),
                functionality_details=llm_analysis.get('functionality', ''),
                dependencies_analysis=llm_analysis.get('dependencies', ''),
                security_implications=llm_analysis.get('security', ''),
                testing_requirements=llm_analysis.get('testing', ''),
                architectural_role=llm_analysis.get('architecture', ''),
                primary_classification=llm_analysis.get('primary_classification', 'uncategorized'),
                secondary_classifications=llm_analysis.get('secondary_classifications', []),
                reorganization_recommendations=llm_analysis.get('reorganization', []),
                confidence_score=llm_analysis.get('confidence', 0.5),
                key_features=llm_analysis.get('key_features', []),
                integration_points=llm_analysis.get('integration_points', []),
                complexity_assessment=llm_analysis.get('complexity', 'unknown'),
                maintainability_notes=llm_analysis.get('maintainability', ''),
                file_size=file_stats['size'],
                line_count=file_stats['lines'],
                class_count=file_stats['classes'],
                function_count=file_stats['functions']
            )

            return entry

        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed for {file_path}: {e}")
            return None

    def _get_file_statistics(self, content: str, file_path: Path) -> Dict[str, int]:
        """Get basic file statistics"""
        lines = content.split('\n')
        line_count = len(lines)

        try:
            tree = ast.parse(content)
            class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            function_count = len([node for node in ast.walk(tree)
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and not self._is_method_in_class(tree, node)])
        except SyntaxError:
            class_count = 0
            function_count = 0

        return {
            'size': len(content.encode('utf-8')),
            'lines': line_count,
            'classes': class_count,
            'functions': function_count
        }

    def _is_method_in_class(self, tree: ast.AST, func_node: ast.AST) -> bool:
        """Check if function is a method in a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in ast.walk(node):
                    if item == func_node:
                        return True
        return False

    def _perform_static_analysis(self, file_path: Path, content: str) -> StaticAnalysisResult:
        """Perform static analysis using existing tools"""
        analysis = StaticAnalysisResult()

        try:
            # Semantic analysis
            if 'semantic' in self.static_analyzers:
                analysis.semantic = self.static_analyzers['semantic'].analyze_semantics(content, file_path)

            # Relationship analysis
            if 'relationship' in self.static_analyzers:
                analysis.relationship = self.static_analyzers['relationship'].analyze_relationships(content, str(file_path))

            # Pattern analysis
            if 'pattern' in self.static_analyzers:
                analysis.pattern = self.static_analyzers['pattern'].detect_patterns(content, file_path)

            # Quality analysis
            if 'quality' in self.static_analyzers:
                analysis.quality = self.static_analyzers['quality'].analyze_quality(content, file_path)

        except Exception as e:
            self.logger.warning(f"Static analysis failed for {file_path}: {e}")

        return analysis

    def _perform_llm_analysis(self, file_path: Path, content: str,
                            file_stats: Dict[str, int],
                            static_analysis: StaticAnalysisResult) -> Dict[str, Any]:
        """Perform LLM analysis with comprehensive context"""

        # Prepare analysis context
        context = self._prepare_analysis_context(file_path, content, file_stats, static_analysis)

        # Generate analysis prompt
        prompt = self._generate_llm_prompt(context)

        # Get LLM response
        llm_response = self.llm_client.analyze_code(prompt)

        # Parse and validate response
        analysis_data = self._parse_llm_response(llm_response)

        return analysis_data

    def _prepare_analysis_context(self, file_path: Path, content: str,
                                file_stats: Dict[str, int],
                                static_analysis: StaticAnalysisResult) -> Dict[str, Any]:
        """Prepare comprehensive analysis context"""
        relative_path = file_path.relative_to(self.root_dir)

        # Extract static signals
        static_signals = {}
        if static_analysis.semantic:
            static_signals['semantic_purpose'] = static_analysis.semantic.get('primary_purpose', 'unknown')
            static_signals['semantic_confidence'] = static_analysis.semantic.get('semantic_confidence', 0.0)

        if static_analysis.pattern:
            patterns = static_analysis.pattern.get('patterns', [])
            static_signals['detected_patterns'] = [p.get('pattern_name', '') for p in patterns if patterns]
            static_signals['pattern_confidence'] = static_analysis.pattern.get('high_confidence_patterns', 0)

        if static_analysis.quality:
            static_signals['quality_score'] = static_analysis.quality.get('overall_score', 0.5)

        # Extract basic code structure
        try:
            tree = ast.parse(content)
            imports = []
            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not self._is_method_in_class(tree, node):
                        functions.append(node.name)

        except SyntaxError:
            imports = []
            classes = []
            functions = []

        return {
            'full_path': str(file_path),
            'relative_path': str(relative_path),
            'file_stats': file_stats,
            'static_signals': static_signals,
            'imports': imports[:20],  # Limit for prompt size
            'classes': classes[:10],
            'functions': functions[:20],
            'content_preview': content[:2000] if len(content) > 2000 else content
        }

    def _generate_llm_prompt(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive LLM analysis prompt"""
        prompt = f"""
You are an expert Python code analyst. Analyze the provided Python module and return a comprehensive JSON assessment.

FILE CONTEXT:
- Full path: {context['full_path']}
- Relative path: {context['relative_path']}
- Size: {context['file_stats']['size']} bytes
- Lines: {context['file_stats']['lines']}
- Classes: {context['file_stats']['classes']}
- Functions: {context['file_stats']['functions']}

STATIC ANALYSIS SIGNALS:
- Semantic purpose: {context['static_signals'].get('semantic_purpose', 'unknown')}
- Semantic confidence: {context['static_signals'].get('semantic_confidence', 0.0):.2f}
- Detected patterns: {', '.join(context['static_signals'].get('detected_patterns', []))}
- Pattern confidence: {context['static_signals'].get('pattern_confidence', 0)}
- Quality score: {context['static_signals'].get('quality_score', 0.5):.2f}

CODE STRUCTURE:
- Imports: {', '.join(context['imports'])}
- Classes: {', '.join(context['classes'])}
- Functions: {', '.join(context['functions'])}

CODE CONTENT:
```
{context['content_preview']}
```

TASK: Return a JSON object with these exact keys and provide detailed, specific analysis:

{{
    "summary": "2-3 sentence overview of what this module does and its primary purpose",
    "functionality": "Detailed breakdown of key functionality, classes, methods, and their purposes",
    "dependencies": "Analysis of imports, external dependencies, and coupling relationships",
    "security": "Security implications, authentication, encryption, or sensitive data handling",
    "testing": "Testing requirements, mock objects needed, or test coverage considerations",
    "architecture": "Architectural role - service, utility, model, controller, etc.",
    "primary_classification": "Choose from: security, intelligence, frontend_dashboard, documentation, testing, utility, api, database, data_processing, orchestration, automation, monitoring, analytics, devops, uncategorized",
    "secondary_classifications": ["Array of secondary categories that apply"],
    "reorganization": ["Specific recommendations for where this should be moved or reorganized"],
    "confidence": 0.0-1.0,
    "key_features": ["List of key functionality features"],
    "integration_points": ["How this integrates with other modules or systems"],
    "complexity": "Assessment: low, medium, high, very_high",
    "maintainability": "Notes on maintainability, potential issues, or refactoring needs"
}}

GUIDELINES:
- Be specific and actionable in your analysis
- Use the static signals to validate your assessment
- If uncertain about classification, lower confidence score
- Focus on what the code DOES, not just what it CONTAINS
- Consider security implications carefully
- Provide practical reorganization suggestions

Return ONLY valid JSON. No additional text.
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate required fields
                required_fields = [
                    'summary', 'functionality', 'dependencies', 'security',
                    'testing', 'architecture', 'primary_classification',
                    'secondary_classifications', 'reorganization', 'confidence',
                    'key_features', 'integration_points', 'complexity', 'maintainability'
                ]

                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = self._get_default_value(field)

                # Validate classification
                valid_classifications = {c.value for c in Classification}
                if parsed['primary_classification'] not in valid_classifications:
                    parsed['primary_classification'] = 'uncategorized'

                # Ensure confidence is a number
                try:
                    parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
                except (ValueError, TypeError):
                    parsed['confidence'] = 0.5

                return parsed
            else:
                return self._get_default_response()

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return self._get_default_response()
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._get_default_response()

    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing field"""
        defaults = {
            'summary': 'Analysis not available',
            'functionality': 'Functionality details not determined',
            'dependencies': 'Dependency analysis not available',
            'security': 'No security implications identified',
            'testing': 'Testing requirements not specified',
            'architecture': 'Architectural role not determined',
            'primary_classification': 'uncategorized',
            'secondary_classifications': [],
            'reorganization': [],
            'confidence': 0.5,
            'key_features': [],
            'integration_points': [],
            'complexity': 'unknown',
            'maintainability': 'Maintainability assessment not available'
        }
        return defaults.get(field, '')

    def _get_default_response(self) -> Dict[str, Any]:
        """Get default response when parsing fails"""
        return {
            'summary': 'Analysis failed - unable to process response',
            'functionality': 'Functionality analysis not available',
            'dependencies': 'Dependency analysis not available',
            'security': 'Security analysis not available',
            'testing': 'Testing analysis not available',
            'architecture': 'Architecture analysis not available',
            'primary_classification': 'uncategorized',
            'secondary_classifications': [],
            'reorganization': ['Manual review required'],
            'confidence': 0.1,
            'key_features': [],
            'integration_points': [],
            'complexity': 'unknown',
            'maintainability': 'Manual review required due to analysis failure'
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return "unknown"

    def _build_directory_structure(self, python_files: List[Path]) -> Dict[str, Any]:
        """Build directory structure representation"""
        structure = {}

        for file_path in python_files:
            try:
                relative_path = file_path.relative_to(self.root_dir)
                parts = list(relative_path.parts)

                if parts:
                    current = structure
                    for i, part in enumerate(parts[:-1]):  # All parts except filename
                        if part not in current:
                            current[part] = {}
                        current = current[part]

                    # Add file info
                    file_name = parts[-1]
                    if file_name not in current:
                        current[file_name] = {
                            'type': 'file',
                            'size': file_path.stat().st_size,
                            'path': str(relative_path)
                        }

            except Exception as e:
                self.logger.warning(f"Error building structure for {file_path}: {e}")

        return structure

    def _calculate_scan_statistics(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Calculate comprehensive scan statistics"""
        if not entries:
            return {}

        # Classification distribution
        primary_classifications = {}
        secondary_classifications = {}

        for entry in entries:
            primary = entry.primary_classification
            primary_classifications[primary] = primary_classifications.get(primary, 0) + 1

            for secondary in entry.secondary_classifications:
                secondary_classifications[secondary] = secondary_classifications.get(secondary, 0) + 1

        # Confidence statistics
        confidences = [entry.confidence_score for entry in entries]

        # Size statistics
        sizes = [entry.file_size for entry in entries]
        lines = [entry.line_count for entry in entries]

        # Complexity distribution
        complexities = {}
        for entry in entries:
            complexity = entry.complexity_assessment
            complexities[complexity] = complexities.get(complexity, 0) + 1

        return {
            'classification_distribution': {
                'primary': primary_classifications,
                'secondary': secondary_classifications
            },
            'confidence_stats': {
                'mean': sum(confidences) / len(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'high_confidence_count': len([c for c in confidences if c >= 0.8])
            },
            'size_stats': {
                'total_size_bytes': sum(sizes),
                'total_lines': sum(lines),
                'avg_file_size': sum(sizes) / len(sizes) if sizes else 0,
                'avg_lines_per_file': sum(lines) / len(lines) if lines else 0
            },
            'complexity_distribution': complexities,
            'scan_efficiency': {
                'files_per_minute': len(entries) / max(1, (datetime.now() - datetime.fromisoformat(entries[0].analysis_timestamp)).total_seconds() / 60) if entries else 0
            }
        }

    def _get_scan_metadata(self) -> Dict[str, Any]:
        """Get scan metadata"""
        return {
            'scanner_version': '1.0.0',
            'llm_provider': self.config['llm_provider'],
            'llm_model': self.config['llm_model'],
            'enable_static_analysis': self.config['enable_static_analysis'],
            'cache_enabled': self.config['cache_enabled'],
            'root_directory': str(self.root_dir),
            'exclusions_applied': list(self.exclusions),
            'scan_configuration': self.config
        }

    def _save_intelligence_map(self, intelligence_map: LLMIntelligenceMap, output_file: Path) -> None:
        """Save intelligence map to file"""
        try:
            # Convert to dictionary (handling dataclass serialization)
            map_dict = {
                'scan_timestamp': intelligence_map.scan_timestamp,
                'scan_id': intelligence_map.scan_id,
                'total_files_scanned': intelligence_map.total_files_scanned,
                'total_lines_analyzed': intelligence_map.total_lines_analyzed,
                'directory_structure': intelligence_map.directory_structure,
                'intelligence_entries': [entry.__dict__ if hasattr(entry, '__dict__') else entry for entry in intelligence_map.intelligence_entries],
                'scan_metadata': intelligence_map.scan_metadata,
                'scan_statistics': intelligence_map.scan_statistics
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(map_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Intelligence map saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save intelligence map: {e}")

    def _update_cache(self, entries: List[LLMIntelligenceEntry]) -> None:
        """Update cache with new entries"""
        for entry in entries:
            cache_key = entry.relative_path
            self.cache[cache_key] = entry.__dict__ if hasattr(entry, '__dict__') else asdict(entry)


# LLM Client implementations
class MockLLMClient:
    """Mock LLM client for testing and development"""

    def analyze_code(self, prompt: str) -> str:
        """Return mock analysis response"""
        time.sleep(0.1)  # Simulate API delay

        # Extract basic info from prompt
        lines = prompt.split('\n')
        file_path = "unknown"
        for line in lines:
            if 'Relative path:' in line:
                file_path = line.split(':')[1].strip()
                break

        # Generate appropriate mock response based on file path
        if 'security' in file_path.lower() or 'auth' in file_path.lower():
            return self._mock_security_response()
        elif 'test' in file_path.lower():
            return self._mock_test_response()
        elif 'api' in file_path.lower():
            return self._mock_api_response()
        else:
            return self._mock_general_response()

    def _mock_security_response(self) -> str:
        return '''{
    "summary": "This module handles authentication and authorization for the system, implementing secure token-based access control with encryption.",
    "functionality": "Provides JWT token generation and validation, user authentication, role-based permissions, and secure password hashing using bcrypt.",
    "dependencies": "Uses cryptography library for encryption, JWT for tokens, and integrates with user management systems.",
    "security": "Handles sensitive authentication data, implements secure token storage, and provides protection against common attacks like CSRF and XSS.",
    "testing": "Requires comprehensive security testing, mock authentication, and penetration testing scenarios.",
    "architecture": "Security service layer providing authentication and authorization services to the application.",
    "primary_classification": "security",
    "secondary_classifications": ["authentication", "encryption"],
    "reorganization": ["Move to src/core/security/", "Group with other security modules", "Ensure access control"],
    "confidence": 0.9,
    "key_features": ["JWT tokens", "role-based access", "password hashing", "CSRF protection"],
    "integration_points": ["User management", "API endpoints", "Database layer", "Frontend authentication"],
    "complexity": "medium",
    "maintainability": "Regular security updates required, follow OWASP guidelines, consider adding rate limiting"
}'''

    def _mock_test_response(self) -> str:
        return '''{
    "summary": "This module contains unit tests and integration tests for system components, ensuring code reliability and functionality.",
    "functionality": "Provides test cases for core functionality, mock objects, test fixtures, and assertion helpers for various system components.",
    "dependencies": "Uses pytest framework, mock library, and testing utilities with minimal external dependencies.",
    "security": "No direct security implications, but tests security features of other modules.",
    "testing": "Self-testing module that defines testing patterns and practices for the codebase.",
    "architecture": "Testing infrastructure supporting the development and validation process across all system layers.",
    "primary_classification": "testing",
    "secondary_classifications": ["unit_tests", "integration_tests", "test_framework"],
    "reorganization": ["Move to tests/ directory", "Organize by component tested", "Separate unit and integration tests"],
    "confidence": 0.85,
    "key_features": ["Test fixtures", "Mock objects", "Assertion helpers", "Coverage reporting"],
    "integration_points": ["All testable components", "CI/CD pipeline", "Coverage reporting tools"],
    "complexity": "low",
    "maintainability": "Tests should be maintained alongside code changes, consider test data management"
}'''

    def _mock_api_response(self) -> str:
        return '''{
    "summary": "This module implements REST API endpoints and request handling for the system, providing external interfaces.",
    "functionality": "Defines API routes, request validation, response formatting, and error handling for HTTP endpoints using FastAPI.",
    "dependencies": "Uses FastAPI framework, integrates with business logic and data layers through dependency injection.",
    "security": "Implements input validation, authentication middleware, and secure response handling with CORS support.",
    "testing": "Requires API testing, integration testing, and load testing scenarios with tools like Postman or pytest.",
    "architecture": "Presentation layer providing external API interfaces to the system with clear separation of concerns.",
    "primary_classification": "api",
    "secondary_classifications": ["web_framework", "request_handling", "rest_api"],
    "reorganization": ["Move to src/api/", "Group related endpoints", "Separate API from business logic"],
    "confidence": 0.88,
    "key_features": ["REST endpoints", "Request validation", "Error handling", "CORS support", "API documentation"],
    "integration_points": ["Business logic", "Database layer", "Authentication", "Frontend applications"],
    "complexity": "medium",
    "maintainability": "API versioning and documentation required, consider OpenAPI specification"
}'''

    def _mock_general_response(self) -> str:
        return '''{
    "summary": "This module provides utility functions and helper classes for common operations across the system.",
    "functionality": "Contains shared utilities, helper functions, and common data structures used by multiple components.",
    "dependencies": "Minimal dependencies, primarily standard library and basic external packages.",
    "security": "No direct security implications, but should follow secure coding practices.",
    "testing": "Requires unit testing for individual functions and integration testing for usage patterns.",
    "architecture": "Utility layer providing common functionality to other system components.",
    "primary_classification": "utility",
    "secondary_classifications": ["helpers", "common"],
    "reorganization": ["Move to src/utils/", "Group similar utilities", "Consider shared library"],
    "confidence": 0.7,
    "key_features": ["Helper functions", "Common utilities", "Shared code", "Data structures"],
    "integration_points": ["Multiple system components", "Cross-cutting concerns"],
    "complexity": "low",
    "maintainability": "Should remain simple and focused on utility functions, avoid feature creep"
}'''


class OpenAIClient:
    """OpenAI LLM client"""

    def __init__(self, api_key: str, model: str):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise Exception("OpenAI library not installed")

    def analyze_code(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


class OllamaClient:
    """Ollama local LLM client"""

    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://localhost:11434"

    def analyze_code(self, prompt: str) -> str:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )
            return response.json().get("response", "No response")
        except Exception as e:
            return f"Error: {e}"


def main():
    """Main function to run the LLM intelligence scanner"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Intelligence Scanner")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory to scan")
    parser.add_argument("--provider", type=str, default="mock",
                      choices=["openai", "anthropic", "groq", "ollama", "mock"],
                      help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="Model name to use")
    parser.add_argument("--output", type=str, default="llm_intelligence_map.json",
                      help="Output file path")
    parser.add_argument("--max-files", type=int,
                      help="Maximum number of files to analyze (for testing)")
    parser.add_argument("--max-concurrent", type=int, default=3,
                      help="Maximum concurrent requests")
    parser.add_argument("--api-key", type=str,
                      help="API key for the LLM provider")
    parser.add_argument("--no-cache", action="store_true",
                      help="Disable caching")
    parser.add_argument("--no-static", action="store_true",
                      help="Disable static analysis")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    output_file = Path(args.output)

    # Create configuration
    config = {
        'llm_provider': args.provider,
        'llm_model': args.model,
        'api_key': args.api_key,
        'max_concurrent': args.max_concurrent,
        'confidence_threshold': 0.7,
        'enable_static_analysis': not args.no_static,
        'cache_enabled': not args.no_cache,
        'preserve_directory_order': True,
        'llm_temperature': 0.0,
        'llm_max_tokens': 2000,
        'chunk_size': 4000,
        'chunk_overlap': 200
    }

    # Initialize scanner
    scanner = LLMIntelligenceScanner(root_dir, config)

    print("🧠 Starting LLM Intelligence Scan...")
    print(f"Root directory: {root_dir}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Static analysis: {'enabled' if not args.no_static else 'disabled'}")
    print(f"Caching: {'enabled' if not args.no_cache else 'disabled'}")
    print(f"Output: {output_file}")

    # Run scan
    intelligence_map = scanner.scan_and_analyze(output_file, args.max_files)

    print("
✅ Scan completed!"    print(f"Files analyzed: {intelligence_map.total_files_scanned}")
    print(f"Total lines: {intelligence_map.total_lines_analyzed}")
    print(f"Output saved to: {output_file}")

    # Print summary
    if intelligence_map.scan_statistics:
        stats = intelligence_map.scan_statistics
        print("
📊 Classification Summary:"        for category, count in sorted(stats['classification_distribution']['primary'].items(),
                               key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")

        confidence_stats = stats['confidence_stats']
        print("
🎯 Confidence Statistics:"        print(".2f"        print(".2f"        print(f"  High confidence (>0.8): {confidence_stats['high_confidence_count']}")


if __name__ == "__main__":
    main()

