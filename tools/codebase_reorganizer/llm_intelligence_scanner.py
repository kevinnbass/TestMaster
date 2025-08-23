#!/usr/bin/env python3
"""
LLM Intelligence Scanner
========================

An LLM-powered system that creates comprehensive intelligence about Python modules
for informed reorganization decisions. Uses advanced prompting to understand code
semantics and functionality beyond traditional static analysis.

This system creates a JSON intelligence map that can be used to supplement
existing reorganization intelligence with deep semantic understanding.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
import logging
from enum import Enum

# Import the existing exclusions logic
try:
    from reorganizer import CodebaseReorganizer
    HAS_REORGANIZER = True
except ImportError:
    HAS_REORGANIZER = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"  # Local LLM
    MOCK = "mock"  # For testing without API


@dataclass
class LLMIntelligenceEntry:
    """Single entry in the LLM intelligence map"""
    full_path: str
    relative_path: str
    file_hash: str
    analysis_timestamp: str
    module_summary: str
    functionality_details: str
    dependencies_analysis: str
    security_implications: str
    testing_requirements: str
    architectural_role: str
    primary_classification: str
    secondary_classifications: List[str]
    reorganization_recommendations: List[str]
    confidence_score: float
    key_features: List[str]
    integration_points: List[str]
    complexity_assessment: str
    maintainability_notes: str
    file_size: int
    line_count: int
    class_count: int
    function_count: int


@dataclass
class LLMIntelligenceMap:
    """Complete LLM intelligence map"""
    scan_timestamp: str
    total_files_scanned: int
    total_lines_analyzed: int
    directory_structure: Dict[str, Any]
    intelligence_entries: List[LLMIntelligenceEntry]
    classification_summary: Dict[str, int]
    reorganization_insights: Dict[str, Any]
    scan_metadata: Dict[str, Any]


class LLMIntelligenceScanner:
    """
    LLM-powered intelligence scanner that analyzes Python files
    to create comprehensive reorganization intelligence.
    """

    def __init__(self, root_dir: Path, provider: LLMProvider = LLMProvider.MOCK,
                 api_key: Optional[str] = None, model: str = "gpt-4",
                 max_concurrent: int = 3, cache_dir: Optional[Path] = None):
        """
        Initialize the LLM intelligence scanner.

        Args:
            root_dir: Root directory to scan
            provider: LLM provider to use
            api_key: API key for the provider
            model: Model name to use
            max_concurrent: Maximum concurrent LLM requests
            cache_dir: Directory for caching results
        """
        self.root_dir = root_dir.resolve()
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.cache_dir = cache_dir or self.root_dir / "tools" / "codebase_reorganizer" / "llm_cache"

        # Setup exclusions (same as existing system)
        self.exclusions = self._get_exclusions()

        # Setup logging
        self._setup_logging()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()

        # Load existing cache
        self.cache = self._load_cache()

        self.logger.info("LLM Intelligence Scanner initialized")
        self.logger.info(f"Using provider: {provider.value}, model: {model}")
        self.logger.info(f"Cache directory: {self.cache_dir}")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"llm_intelligence_scan_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_exclusions(self) -> set[str]:
        """Get the same exclusions as the existing reorganizer"""
        if HAS_REORGANIZER:
            try:
                # Use the existing reorganizer's exclusion logic
                temp_reorganizer = CodebaseReorganizer(self.root_dir)
                return temp_reorganizer.exclude_patterns
            except:
                pass

        # Fallback exclusions
        return {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentscope',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', 'tests', 'test_sessions', 'testmaster_sessions',
            'tools', 'codebase_reorganizer'  # Don't scan our own tools
        }

    def _initialize_llm_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == LLMProvider.MOCK:
            return MockLLMClient()
        elif self.provider == LLMProvider.OPENAI:
            try:
                import openai
                return OpenAIClient(self.api_key, self.model)
            except ImportError:
                self.logger.error("OpenAI library not installed, falling back to mock")
                return MockLLMClient()
        elif self.provider == LLMProvider.OLLAMA:
            try:
                return OllamaClient(self.model)
            except Exception as e:
                self.logger.error(f"Ollama client failed: {e}, falling back to mock")
                return MockLLMClient()
        else:
            self.logger.warning(f"Provider {self.provider} not implemented, using mock")
            return MockLLMClient()

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

    def _save_cache(self) -> None:
        """Save current cache to disk"""
        cache_file = self.cache_dir / "llm_intelligence_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded from scanning"""
        if not path.exists() or not path.is_file() or path.suffix != '.py':
            return True

        try:
            rel_path = path.relative_to(self.root_dir)
            return any(part in self.exclusions for part in rel_path.parts)
        except ValueError:
            return True

    def scan_and_analyze(self, output_file: Optional[Path] = None) -> LLMIntelligenceMap:
        """
        Perform comprehensive LLM-powered intelligence scan.

        Args:
            output_file: Optional path to save the intelligence map

        Returns:
            Complete LLM intelligence map
        """
        self.logger.info("Starting LLM intelligence scan...")

        # Find all Python files to analyze
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclusions)]
            for file in files:
                if file.endswith('.py') and not any(excl in file for excl in ['test_', 'setup.py']):
                    file_path = Path(root) / file
                    if not self.should_exclude(file_path):
                        python_files.append(file_path)

        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        # Analyze files with LLM (with concurrency control)
        intelligence_entries = []
        total_lines = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_file = {
                executor.submit(self._analyze_file_with_llm, file_path): file_path
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

        # Create classification summary
        classification_summary = self._create_classification_summary(intelligence_entries)

        # Generate reorganization insights
        reorganization_insights = self._generate_reorganization_insights(
            intelligence_entries, classification_summary
        )

        # Build directory structure representation
        directory_structure = self._build_directory_structure(intelligence_entries)

        # Create comprehensive intelligence map
        intelligence_map = LLMIntelligenceMap(
            scan_timestamp=datetime.now().isoformat(),
            total_files_scanned=len(intelligence_entries),
            total_lines_analyzed=total_lines,
            directory_structure=directory_structure,
            intelligence_entries=intelligence_entries,
            classification_summary=classification_summary,
            reorganization_insights=reorganization_insights,
            scan_metadata=self._get_scan_metadata()
        )

        # Save to file if requested
        if output_file:
            self._save_intelligence_map(intelligence_map, output_file)

        # Update cache
        self._update_cache(intelligence_entries)
        self._save_cache()

        self.logger.info("LLM intelligence scan completed!")
        self.logger.info(f"Analyzed {len(intelligence_entries)} files with {total_lines} total lines")

        return intelligence_map

    def _analyze_file_with_llm(self, file_path: Path) -> Optional[LLMIntelligenceEntry]:
        """Analyze a single file using LLM"""
        try:
            # Calculate file hash for caching
            file_hash = self._calculate_file_hash(file_path)

            # Check cache first
            cache_key = str(file_path.relative_to(self.root_dir))
            if cache_key in self.cache and self.cache[cache_key].get('file_hash') == file_hash:
                self.logger.info(f"Using cached analysis for: {file_path}")
                cached_data = self.cache[cache_key]
                return LLMIntelligenceEntry(**cached_data)

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Basic file statistics
            lines = content.split('\n')
            line_count = len(lines)
            file_size = len(content.encode('utf-8'))

            # Count classes and functions
            class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
            function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))

            # Generate comprehensive analysis prompt
            prompt = self._generate_analysis_prompt(content, file_path)

            # Get LLM analysis
            llm_response = self.llm_client.analyze_code(prompt)

            # Parse LLM response
            analysis_data = self._parse_llm_response(llm_response)

            # Create intelligence entry
            entry = LLMIntelligenceEntry(
                full_path=str(file_path),
                relative_path=str(file_path.relative_to(self.root_dir)),
                file_hash=file_hash,
                analysis_timestamp=datetime.now().isoformat(),
                module_summary=analysis_data.get('summary', 'No summary available'),
                functionality_details=analysis_data.get('functionality', 'No functionality details'),
                dependencies_analysis=analysis_data.get('dependencies', 'No dependency analysis'),
                security_implications=analysis_data.get('security', 'No security implications identified'),
                testing_requirements=analysis_data.get('testing', 'No testing requirements specified'),
                architectural_role=analysis_data.get('architecture', 'No architectural role identified'),
                primary_classification=analysis_data.get('primary_classification', 'uncategorized'),
                secondary_classifications=analysis_data.get('secondary_classifications', []),
                reorganization_recommendations=analysis_data.get('reorganization', []),
                confidence_score=analysis_data.get('confidence', 0.5),
                key_features=analysis_data.get('key_features', []),
                integration_points=analysis_data.get('integration_points', []),
                complexity_assessment=analysis_data.get('complexity', 'Unknown complexity'),
                maintainability_notes=analysis_data.get('maintainability', 'No maintainability notes'),
                file_size=file_size,
                line_count=line_count,
                class_count=class_count,
                function_count=function_count
            )

            return entry

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def _generate_analysis_prompt(self, content: str, file_path: Path) -> str:
        """Generate a comprehensive analysis prompt for the LLM"""
        relative_path = file_path.relative_to(self.root_dir)

        prompt = f"""
You are an expert Python code analyst tasked with deeply understanding a Python module for reorganization purposes.

FILE TO ANALYZE:
- Full path: {file_path}
- Relative path: {relative_path}
- File size: {len(content.encode('utf-8'))} bytes
- Lines of code: {len(content.split('\n'))}

CODE CONTENT:
```
{content[:4000]}...  # Truncated for brevity
```

Please provide a comprehensive analysis in the following JSON format:

{{
    "summary": "A concise but exhaustive 2-3 sentence description of what this module does and its primary purpose",
    "functionality": "Detailed breakdown of the module's functionality, key classes, functions, and their purposes",
    "dependencies": "Analysis of imports, external dependencies, and coupling relationships",
    "security": "Security implications, authentication, encryption, or sensitive data handling",
    "testing": "Testing requirements, mock objects needed, or test coverage considerations",
    "architecture": "Architectural role - is this a service, utility, model, controller, etc.?",
    "primary_classification": "Primary category: security/intelligence/frontend_dashboard/documentation/testing/utility/api/database/automation/orchestration",
    "secondary_classifications": ["List", "of", "secondary", "categories"],
    "reorganization": ["Specific reorganization recommendations", "where this should be moved", "related modules it should be grouped with"],
    "confidence": 0.0-1.0,
    "key_features": ["List", "of", "key", "functionality", "features"],
    "integration_points": ["How", "this", "integrates", "with", "other", "modules"],
    "complexity": "Assessment of code complexity and maintainability challenges",
    "maintainability": "Notes on maintainability, potential refactoring needs, or code quality issues"
}}

Focus on:
- Deep semantic understanding of what the code DOES, not just what it CONTAINS
- How this module fits into the larger system architecture
- Security and operational considerations
- How this should be reorganized for better cohesion and coupling
- Practical insights for developers who need to understand and maintain this code

Be specific and actionable in your analysis. If you're uncertain about any aspect, note that in your confidence score.
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: create structured data from text response
                return self._fallback_parse_response(response)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_parse_response(response)

    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON LLM responses"""
        # Extract what we can from the text response
        return {
            'summary': response[:500] + '...' if len(response) > 500 else response,
            'functionality': 'Analysis provided in summary',
            'dependencies': 'Dependencies mentioned in analysis',
            'security': 'Security aspects mentioned in analysis',
            'testing': 'Testing aspects mentioned in analysis',
            'architecture': 'Architecture mentioned in analysis',
            'primary_classification': 'uncategorized',
            'secondary_classifications': [],
            'reorganization': ['Manual review required'],
            'confidence': 0.3,
            'key_features': [],
            'integration_points': [],
            'complexity': 'Unknown complexity',
            'maintainability': 'Manual review required'
        }

    def _create_classification_summary(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, int]:
        """Create a summary of classifications across all entries"""
        summary = {}
        for entry in entries:
            primary = entry.primary_classification
            summary[primary] = summary.get(primary, 0) + 1

            for secondary in entry.secondary_classifications:
                summary[secondary] = summary.get(secondary, 0) + 1

        return summary

    def _generate_reorganization_insights(self, entries: List[LLMIntelligenceEntry],
                                        classification_summary: Dict[str, int]) -> Dict[str, Any]:
        """Generate high-level reorganization insights"""
        insights = {
            'classification_distribution': classification_summary,
            'recommended_structure': self._generate_recommended_structure(entries),
            'problematic_modules': self._identify_problematic_modules(entries),
            'integration_opportunities': self._identify_integration_opportunities(entries),
            'security_concerns': self._analyze_security_concerns(entries),
            'testing_gaps': self._analyze_testing_gaps(entries)
        }

        return insights

    def _generate_recommended_structure(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Generate recommended directory structure based on analysis"""
        # Group entries by primary classification
        by_classification = {}
        for entry in entries:
            primary = entry.primary_classification
            if primary not in by_classification:
                by_classification[primary] = []
            by_classification[primary].append(entry.relative_path)

        return {
            'core_directories': list(by_classification.keys()),
            'directory_contents': by_classification,
            'estimated_reduction': self._calculate_organization_benefit(entries)
        }

    def _identify_problematic_modules(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Identify modules that need special attention"""
        problematic = []

        for entry in entries:
            issues = []

            if entry.confidence_score < 0.6:
                issues.append("Low analysis confidence")

            if "security" in entry.security_implications.lower() and "none" not in entry.security_implications.lower():
                issues.append("Security concerns identified")

            if entry.complexity_assessment.lower() in ['high', 'very high', 'complex']:
                issues.append("High complexity")

            if issues:
                problematic.append({
                    'path': entry.relative_path,
                    'issues': issues,
                    'confidence': entry.confidence_score
                })

        return problematic

    def _identify_integration_opportunities(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Identify opportunities for better module integration"""
        opportunities = []

        # Group by classification
        by_classification = {}
        for entry in entries:
            primary = entry.primary_classification
            if primary not in by_classification:
                by_classification[primary] = []
            by_classification[primary].append(entry)

        # Look for scattered modules that could be consolidated
        for classification, modules in by_classification.items():
            if len(modules) > 3:  # If many modules in same classification
                current_paths = [m.relative_path for m in modules]
                opportunities.append({
                    'classification': classification,
                    'module_count': len(modules),
                    'current_paths': current_paths,
                    'consolidation_potential': 'high' if len(modules) > 5 else 'medium'
                })

        return opportunities

    def _analyze_security_concerns(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Analyze security concerns across modules"""
        concerns = []

        for entry in entries:
            if entry.security_implications and "none" not in entry.security_implications.lower():
                concerns.append({
                    'module': entry.relative_path,
                    'security_notes': entry.security_implications,
                    'classification': entry.primary_classification
                })

        return concerns

    def _analyze_testing_gaps(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Analyze testing requirements and gaps"""
        testing_needs = []

        for entry in entries:
            if entry.testing_requirements and "none" not in entry.testing_requirements.lower():
                testing_needs.append({
                    'module': entry.relative_path,
                    'testing_requirements': entry.testing_requirements,
                    'complexity': entry.complexity_assessment
                })

        return testing_needs

    def _calculate_organization_benefit(self, entries: List[LLMIntelligenceEntry]) -> float:
        """Calculate estimated benefit of reorganization"""
        # Simple heuristic: more scattered modules = higher benefit
        total_modules = len(entries)
        classifications = len(set(e.primary_classification for e in entries))

        # Estimate reduction in cognitive load
        if total_modules > 20:
            return min(0.8, (total_modules - 20) * 0.1)
        else:
            return 0.2

    def _build_directory_structure(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Build a representation of the directory structure"""
        structure = {}

        for entry in entries:
            parts = entry.relative_path.split('/')
            current = structure

            for i, part in enumerate(parts[:-1]):  # All parts except filename
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add file with its classification
            if parts[-1] not in current:
                current[parts[-1]] = {
                    'classification': entry.primary_classification,
                    'size': entry.file_size,
                    'lines': entry.line_count
                }

        return structure

    def _get_scan_metadata(self) -> Dict[str, Any]:
        """Get metadata about the scan"""
        return {
            'scanner_version': '1.0.0',
            'llm_provider': self.provider.value,
            'llm_model': self.model,
            'scan_root': str(self.root_dir),
            'exclusions_applied': list(self.exclusions),
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }

    def _save_intelligence_map(self, intelligence_map: LLMIntelligenceMap, output_file: Path) -> None:
        """Save the intelligence map to a JSON file"""
        try:
            # Convert to dictionary
            map_dict = asdict(intelligence_map)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(map_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Intelligence map saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save intelligence map: {e}")

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

    def _update_cache(self, entries: List[LLMIntelligenceEntry]) -> None:
        """Update the cache with new entries"""
        for entry in entries:
            cache_key = entry.relative_path
            self.cache[cache_key] = asdict(entry)


class MockLLMClient:
    """Mock LLM client for testing without API calls"""

    def analyze_code(self, prompt: str) -> str:
        """Mock analysis response"""
        time.sleep(0.1)  # Simulate API delay

        # Extract file path from prompt
        lines = prompt.split('\n')
        file_path = "unknown"
        for line in lines:
            if line.startswith('- Full path:'):
                file_path = line.split(':')[1].strip()
                break

        # Generate mock analysis based on file path
        if 'security' in file_path.lower() or 'auth' in file_path.lower():
            return self._mock_security_response()
        elif 'test' in file_path.lower():
            return self._mock_test_response()
        elif 'api' in file_path.lower():
            return self._mock_api_response()
        else:
            return self._mock_general_response()

    def _mock_security_response(self) -> str:
        return '''
{
    "summary": "This module handles authentication and authorization for the system, implementing secure token-based access control with encryption.",
    "functionality": "Provides JWT token generation and validation, user authentication, role-based permissions, and secure password hashing.",
    "dependencies": "Uses cryptography library for encryption, JWT for tokens, and integrates with user management systems.",
    "security": "Handles sensitive authentication data, implements secure token storage, and provides protection against common attacks.",
    "testing": "Requires comprehensive security testing, mock authentication, and penetration testing scenarios.",
    "architecture": "Security service layer providing authentication and authorization services to the application.",
    "primary_classification": "security",
    "secondary_classifications": ["authentication", "encryption"],
    "reorganization": ["Move to src/core/security/", "Group with other security modules", "Ensure access control"],
    "confidence": 0.9,
    "key_features": ["JWT tokens", "Password hashing", "Role-based access"],
    "integration_points": ["User management", "API endpoints", "Database layer"],
    "complexity": "Medium complexity with security considerations",
    "maintainability": "Regular security updates required, follow security best practices"
}
'''

    def _mock_test_response(self) -> str:
        return '''
{
    "summary": "This module contains unit tests and integration tests for system components, ensuring code reliability and functionality.",
    "functionality": "Provides test cases for core functionality, mock objects, test fixtures, and assertion helpers.",
    "dependencies": "Uses pytest framework, mock library, and testing utilities.",
    "security": "No direct security implications, but tests security features of other modules.",
    "testing": "Self-testing module that defines testing patterns and practices.",
    "architecture": "Testing infrastructure supporting the development and validation process.",
    "primary_classification": "testing",
    "secondary_classifications": ["unit_tests", "integration_tests"],
    "reorganization": ["Move to tests/ directory", "Organize by component tested", "Separate unit and integration tests"],
    "confidence": 0.8,
    "key_features": ["Test fixtures", "Mock objects", "Assertion helpers"],
    "integration_points": ["All testable components", "CI/CD pipeline", "Coverage reporting"],
    "complexity": "Low to medium complexity depending on test scenarios",
    "maintainability": "Tests should be maintained alongside code changes"
}
'''

    def _mock_api_response(self) -> str:
        return '''
{
    "summary": "This module implements REST API endpoints and request handling for the system, providing external interfaces.",
    "functionality": "Defines API routes, request validation, response formatting, and error handling for HTTP endpoints.",
    "dependencies": "Uses FastAPI or Flask framework, integrates with business logic and data layers.",
    "security": "Implements input validation, authentication middleware, and secure response handling.",
    "testing": "Requires API testing, integration testing, and load testing scenarios.",
    "architecture": "Presentation layer providing external API interfaces to the system.",
    "primary_classification": "api",
    "secondary_classifications": ["web_framework", "request_handling"],
    "reorganization": ["Move to src/api/", "Group related endpoints", "Separate API from business logic"],
    "confidence": 0.85,
    "key_features": ["REST endpoints", "Request validation", "Error handling"],
    "integration_points": ["Business logic", "Database layer", "Authentication"],
    "complexity": "Medium complexity with API design considerations",
    "maintainability": "API versioning and documentation required"
}
'''

    def _mock_general_response(self) -> str:
        return '''
{
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
    "key_features": ["Helper functions", "Common utilities", "Shared code"],
    "integration_points": ["Multiple system components", "Cross-cutting concerns"],
    "complexity": "Low complexity with focused responsibilities",
    "maintainability": "Should remain simple and focused on utility functions"
}
'''


class OpenAIClient:
    """OpenAI LLM client"""

    def __init__(self, api_key: str, model: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_code(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
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
            import json
            import urllib.request
            import urllib.error

            # Use built-in urllib instead of external requests library
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }).encode('utf-8')

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Python-URLLib'
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "No response")

        except urllib.error.URLError as e:
            return f"Network Error: {e}"
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
    parser.add_argument("--max-concurrent", type=int, default=3,
                      help="Maximum concurrent requests")
    parser.add_argument("--api-key", type=str,
                      help="API key for the LLM provider")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    output_file = Path(args.output)

    provider = LLMProvider(args.provider)

    scanner = LLMIntelligenceScanner(
        root_dir=root_dir,
        provider=provider,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )

    print("🧠 Starting LLM Intelligence Scan...")
    print(f"Root directory: {root_dir}")
    print(f"Provider: {provider.value}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")

    intelligence_map = scanner.scan_and_analyze(output_file)

    print("
✅ Scan completed!"    print(f"Files analyzed: {intelligence_map.total_files_scanned}")
    print(f"Total lines: {intelligence_map.total_lines_analyzed}")
    print(f"Output saved to: {output_file}")

    # Print classification summary
    print("
📊 Classification Summary:"    for category, count in sorted(intelligence_map.classification_summary.items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")

    if intelligence_map.reorganization_insights['problematic_modules']:
        print("
⚠️  Problematic modules identified:"        for module in intelligence_map.reorganization_insights['problematic_modules'][:5]:
            print(f"  {module['path']}: {', '.join(module['issues'])}")


if __name__ == "__main__":
    main()
