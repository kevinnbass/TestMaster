"""
Cross-Repository Pattern Mining System

This module provides advanced pattern mining capabilities across multiple
repositories to identify common design patterns, anti-patterns, architectural
patterns, and best practices through large-scale code analysis.
"""

import ast
import os
import json
import time
import hashlib
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse
import tempfile
import shutil
import threading
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseAnalyzer


class PatternType(Enum):
    """Types of code patterns"""
    DESIGN_PATTERN = "design_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    ANTI_PATTERN = "anti_pattern"
    IDIOM = "idiom"
    BEST_PRACTICE = "best_practice"
    SECURITY_PATTERN = "security_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"


class RepositoryType(Enum):
    """Types of repositories"""
    GIT = "git"
    LOCAL = "local"
    GITHUB = "github"
    GITLAB = "gitlab"


@dataclass
class CodePattern:
    """Represents a discovered code pattern"""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    repositories: Set[str] = field(default_factory=set)
    code_snippets: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'name': self.name,
            'description': self.description,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'examples': self.examples,
            'repositories': list(self.repositories),
            'code_snippets': self.code_snippets,
            'metrics': self.metrics,
            'tags': list(self.tags)
        }


@dataclass
class RepositoryInfo:
    """Information about a repository"""
    repo_id: str
    repo_type: RepositoryType
    url: str
    local_path: str
    language: str = "python"
    files_analyzed: int = 0
    patterns_found: int = 0
    last_analyzed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternExtractor:
    """Extracts various types of patterns from code"""
    
    def __init__(self):
        self.design_pattern_signatures = self._initialize_design_patterns()
        self.anti_pattern_signatures = self._initialize_anti_patterns()
        self.architectural_patterns = self._initialize_architectural_patterns()
        
    def _initialize_design_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize design pattern signatures"""
        return {
            'singleton': {
                'description': 'Singleton pattern implementation',
                'signatures': [
                    r'class\s+\w+.*:\s*\n\s*_instance\s*=\s*None',
                    r'def\s+__new__\s*\(cls.*\)',
                    r'if\s+cls\._instance\s+is\s+None'
                ],
                'ast_patterns': ['single_instance_check', 'new_override']
            },
            'factory': {
                'description': 'Factory pattern implementation',
                'signatures': [
                    r'def\s+create_\w+\(',
                    r'def\s+make_\w+\(',
                    r'class\s+\w+Factory'
                ],
                'ast_patterns': ['factory_method', 'product_creation']
            },
            'observer': {
                'description': 'Observer pattern implementation',
                'signatures': [
                    r'def\s+add_observer\(',
                    r'def\s+notify\(\s*\)',
                    r'def\s+update\('
                ],
                'ast_patterns': ['observer_list', 'notification_method']
            },
            'decorator': {
                'description': 'Decorator pattern implementation',
                'signatures': [
                    r'@\w+',
                    r'def\s+\w+\(.*\)\s*:\s*\n\s*def\s+wrapper',
                    r'functools\.wraps'
                ],
                'ast_patterns': ['decorator_function', 'wrapper_function']
            },
            'strategy': {
                'description': 'Strategy pattern implementation',
                'signatures': [
                    r'class\s+\w+Strategy',
                    r'def\s+execute\(',
                    r'def\s+set_strategy\('
                ],
                'ast_patterns': ['strategy_interface', 'strategy_context']
            },
            'builder': {
                'description': 'Builder pattern implementation',
                'signatures': [
                    r'class\s+\w+Builder',
                    r'def\s+build\(\s*\)',
                    r'def\s+with_\w+\('
                ],
                'ast_patterns': ['builder_methods', 'fluent_interface']
            },
            'adapter': {
                'description': 'Adapter pattern implementation',
                'signatures': [
                    r'class\s+\w+Adapter',
                    r'def\s+adapt\(',
                    r'self\._adaptee'
                ],
                'ast_patterns': ['adapter_interface', 'adaptee_delegation']
            },
            'facade': {
                'description': 'Facade pattern implementation',
                'signatures': [
                    r'class\s+\w+Facade',
                    r'def\s+\w+\(.*\):\s*\n.*\w+\.\w+\(',
                    r'self\._subsystem'
                ],
                'ast_patterns': ['facade_methods', 'subsystem_coordination']
            }
        }
    
    def _initialize_anti_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize anti-pattern signatures"""
        return {
            'god_class': {
                'description': 'God class anti-pattern (class with too many responsibilities)',
                'metrics': ['method_count > 20', 'line_count > 500'],
                'signatures': [r'class\s+\w+.*:\s*\n(.*\n){500,}']
            },
            'long_method': {
                'description': 'Long method anti-pattern',
                'metrics': ['method_lines > 50'],
                'signatures': [r'def\s+\w+.*:\s*\n(.*\n){50,}']
            },
            'duplicate_code': {
                'description': 'Code duplication anti-pattern',
                'metrics': ['similarity > 0.8'],
                'signatures': []  # Detected by similarity analysis
            },
            'magic_numbers': {
                'description': 'Magic numbers anti-pattern',
                'signatures': [r'\b\d{2,}\b(?!\s*[)}\]])'],
                'ast_patterns': ['numeric_literals']
            },
            'deep_inheritance': {
                'description': 'Deep inheritance hierarchy',
                'metrics': ['inheritance_depth > 6'],
                'ast_patterns': ['inheritance_chain']
            },
            'circular_dependency': {
                'description': 'Circular import dependencies',
                'ast_patterns': ['import_cycles']
            },
            'feature_envy': {
                'description': 'Feature envy (class using another class excessively)',
                'metrics': ['external_method_calls > 10'],
                'ast_patterns': ['excessive_coupling']
            }
        }
    
    def _initialize_architectural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize architectural pattern signatures"""
        return {
            'mvc': {
                'description': 'Model-View-Controller architecture',
                'structure_patterns': ['models/', 'views/', 'controllers/'],
                'file_patterns': ['*model*.py', '*view*.py', '*controller*.py']
            },
            'mvp': {
                'description': 'Model-View-Presenter architecture',
                'structure_patterns': ['models/', 'views/', 'presenters/'],
                'file_patterns': ['*model*.py', '*view*.py', '*presenter*.py']
            },
            'mvvm': {
                'description': 'Model-View-ViewModel architecture',
                'structure_patterns': ['models/', 'views/', 'viewmodels/'],
                'file_patterns': ['*model*.py', '*view*.py', '*viewmodel*.py']
            },
            'layered': {
                'description': 'Layered architecture',
                'structure_patterns': ['presentation/', 'business/', 'data/', 'domain/'],
                'file_patterns': ['*service*.py', '*repository*.py', '*dto*.py']
            },
            'microservices': {
                'description': 'Microservices architecture',
                'structure_patterns': ['services/', 'api/'],
                'file_patterns': ['*service*.py', '*api*.py', 'requirements.txt', 'Dockerfile']
            },
            'hexagonal': {
                'description': 'Hexagonal (Ports and Adapters) architecture',
                'structure_patterns': ['domain/', 'ports/', 'adapters/'],
                'file_patterns': ['*port*.py', '*adapter*.py', '*domain*.py']
            },
            'cqrs': {
                'description': 'Command Query Responsibility Segregation',
                'structure_patterns': ['commands/', 'queries/', 'handlers/'],
                'file_patterns': ['*command*.py', '*query*.py', '*handler*.py']
            }
        }
    
    def extract_patterns(self, file_path: str, content: str, tree: ast.AST) -> List[CodePattern]:
        """Extract all types of patterns from code"""
        patterns = []
        
        # Extract design patterns
        patterns.extend(self._extract_design_patterns(file_path, content, tree))
        
        # Extract anti-patterns
        patterns.extend(self._extract_anti_patterns(file_path, content, tree))
        
        # Extract idioms and best practices
        patterns.extend(self._extract_idioms(file_path, content, tree))
        
        # Extract security patterns
        patterns.extend(self._extract_security_patterns(file_path, content, tree))
        
        return patterns
    
    def _extract_design_patterns(self, file_path: str, content: str, tree: ast.AST) -> List[CodePattern]:
        """Extract design patterns"""
        patterns = []
        
        for pattern_name, pattern_info in self.design_pattern_signatures.items():
            confidence = self._calculate_pattern_confidence(
                content, tree, pattern_info['signatures'], pattern_info.get('ast_patterns', [])
            )
            
            if confidence > 0.6:  # Threshold for pattern detection
                pattern = CodePattern(
                    pattern_id=f"design_{pattern_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    pattern_type=PatternType.DESIGN_PATTERN,
                    name=pattern_name.title() + " Pattern",
                    description=pattern_info['description'],
                    frequency=1,
                    confidence=confidence,
                    examples=[{
                        'file_path': file_path,
                        'pattern_locations': self._find_pattern_locations(content, pattern_info['signatures'])
                    }],
                    repositories={self._extract_repo_name(file_path)},
                    code_snippets=[self._extract_relevant_code(content, tree, pattern_name)],
                    tags={pattern_name, 'design-pattern'}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_anti_patterns(self, file_path: str, content: str, tree: ast.AST) -> List[CodePattern]:
        """Extract anti-patterns"""
        patterns = []
        
        for pattern_name, pattern_info in self.anti_pattern_signatures.items():
            is_anti_pattern, confidence, metrics = self._detect_anti_pattern(
                file_path, content, tree, pattern_name, pattern_info
            )
            
            if is_anti_pattern:
                pattern = CodePattern(
                    pattern_id=f"anti_{pattern_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    pattern_type=PatternType.ANTI_PATTERN,
                    name=pattern_name.replace('_', ' ').title() + " Anti-Pattern",
                    description=pattern_info['description'],
                    frequency=1,
                    confidence=confidence,
                    examples=[{
                        'file_path': file_path,
                        'metrics': metrics
                    }],
                    repositories={self._extract_repo_name(file_path)},
                    metrics=metrics,
                    tags={pattern_name, 'anti-pattern'}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_idioms(self, file_path: str, content: str, tree: ast.AST) -> List[CodePattern]:
        """Extract Python idioms and best practices"""
        patterns = []
        
        # Check for common Python idioms
        idioms = {
            'list_comprehension': {
                'description': 'List comprehension usage',
                'ast_check': lambda node: isinstance(node, ast.ListComp)
            },
            'context_manager': {
                'description': 'Context manager usage (with statement)',
                'ast_check': lambda node: isinstance(node, ast.With)
            },
            'generator_expression': {
                'description': 'Generator expression usage',
                'ast_check': lambda node: isinstance(node, ast.GeneratorExp)
            },
            'enumerate_usage': {
                'description': 'Enumerate function usage',
                'ast_check': lambda node: (isinstance(node, ast.Call) and 
                                         isinstance(node.func, ast.Name) and 
                                         node.func.id == 'enumerate')
            },
            'zip_usage': {
                'description': 'Zip function usage for parallel iteration',
                'ast_check': lambda node: (isinstance(node, ast.Call) and 
                                         isinstance(node.func, ast.Name) and 
                                         node.func.id == 'zip')
            }
        }
        
        for idiom_name, idiom_info in idioms.items():
            count = 0
            examples = []
            
            for node in ast.walk(tree):
                if idiom_info['ast_check'](node):
                    count += 1
                    if hasattr(node, 'lineno'):
                        examples.append({'line': node.lineno})
            
            if count > 0:
                pattern = CodePattern(
                    pattern_id=f"idiom_{idiom_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    pattern_type=PatternType.IDIOM,
                    name=idiom_name.replace('_', ' ').title(),
                    description=idiom_info['description'],
                    frequency=count,
                    confidence=0.9,  # High confidence for AST-based detection
                    examples=examples,
                    repositories={self._extract_repo_name(file_path)},
                    metrics={'usage_count': count},
                    tags={idiom_name, 'python-idiom'}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_security_patterns(self, file_path: str, content: str, tree: ast.AST) -> List[CodePattern]:
        """Extract security-related patterns"""
        patterns = []
        
        security_patterns = {
            'input_validation': {
                'description': 'Input validation pattern',
                'signatures': [r'validate_\w+\(', r'sanitize_\w+\(', r'assert\s+isinstance\(']
            },
            'authentication': {
                'description': 'Authentication pattern',
                'signatures': [r'@login_required', r'authenticate\(', r'check_auth\(']
            },
            'authorization': {
                'description': 'Authorization pattern',
                'signatures': [r'@require_permission', r'has_permission\(', r'check_access\(']
            },
            'secure_random': {
                'description': 'Secure random number generation',
                'signatures': [r'secrets\.', r'os\.urandom\(', r'random\.SystemRandom\(']
            },
            'password_hashing': {
                'description': 'Password hashing pattern',
                'signatures': [r'bcrypt\.', r'scrypt\.', r'pbkdf2_hmac\(', r'hashlib\.sha256\(']
            }
        }
        
        for pattern_name, pattern_info in security_patterns.items():
            confidence = self._calculate_pattern_confidence(
                content, tree, pattern_info['signatures'], []
            )
            
            if confidence > 0.3:  # Lower threshold for security patterns
                pattern = CodePattern(
                    pattern_id=f"security_{pattern_name}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    pattern_type=PatternType.SECURITY_PATTERN,
                    name=pattern_name.replace('_', ' ').title() + " Security Pattern",
                    description=pattern_info['description'],
                    frequency=1,
                    confidence=confidence,
                    examples=[{
                        'file_path': file_path,
                        'pattern_locations': self._find_pattern_locations(content, pattern_info['signatures'])
                    }],
                    repositories={self._extract_repo_name(file_path)},
                    tags={pattern_name, 'security-pattern'}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_confidence(self, content: str, tree: ast.AST, 
                                    signatures: List[str], ast_patterns: List[str]) -> float:
        """Calculate confidence score for pattern detection"""
        import re
        
        total_score = 0.0
        max_score = len(signatures) + len(ast_patterns)
        
        if max_score == 0:
            return 0.0
        
        # Check regex signatures
        for signature in signatures:
            if re.search(signature, content, re.MULTILINE):
                total_score += 1.0
        
        # Check AST patterns (simplified)
        for ast_pattern in ast_patterns:
            if self._check_ast_pattern(tree, ast_pattern):
                total_score += 1.0
        
        return total_score / max_score
    
    def _check_ast_pattern(self, tree: ast.AST, pattern: str) -> bool:
        """Check for specific AST patterns"""
        # Simplified AST pattern checking
        pattern_checks = {
            'single_instance_check': lambda: any(
                isinstance(node, ast.Compare) and 
                isinstance(node.left, ast.Attribute) and
                node.left.attr == '_instance'
                for node in ast.walk(tree)
            ),
            'new_override': lambda: any(
                isinstance(node, ast.FunctionDef) and node.name == '__new__'
                for node in ast.walk(tree)
            ),
            'factory_method': lambda: any(
                isinstance(node, ast.FunctionDef) and 
                (node.name.startswith('create_') or node.name.startswith('make_'))
                for node in ast.walk(tree)
            ),
            'observer_list': lambda: any(
                isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and 'observer' in target.id.lower()
                    for target in node.targets)
                for node in ast.walk(tree)
            ),
            'decorator_function': lambda: any(
                len(node.decorator_list) > 0
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            )
        }
        
        check_func = pattern_checks.get(pattern)
        return check_func() if check_func else False
    
    def _detect_anti_pattern(self, file_path: str, content: str, tree: ast.AST, 
                           pattern_name: str, pattern_info: Dict[str, Any]) -> Tuple[bool, float, Dict[str, float]]:
        """Detect specific anti-pattern"""
        metrics = {}
        
        if pattern_name == 'god_class':
            # Analyze class complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    
                    metrics[f'{node.name}_methods'] = method_count
                    metrics[f'{node.name}_lines'] = line_count
                    
                    if method_count > 20 or line_count > 500:
                        return True, 0.8, metrics
        
        elif pattern_name == 'long_method':
            # Analyze method length
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    metrics[f'{node.name}_lines'] = line_count
                    
                    if line_count > 50:
                        return True, 0.7, metrics
        
        elif pattern_name == 'magic_numbers':
            # Count magic numbers
            magic_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.Constant, ast.Num)):
                    value = getattr(node, 'value', getattr(node, 'n', None))
                    if isinstance(value, (int, float)) and abs(value) > 1 and abs(value) != 100:
                        magic_count += 1
            
            metrics['magic_numbers'] = magic_count
            if magic_count > 5:
                return True, 0.6, metrics
        
        return False, 0.0, metrics
    
    def _find_pattern_locations(self, content: str, signatures: List[str]) -> List[Dict[str, Any]]:
        """Find locations where patterns occur"""
        import re
        locations = []
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for signature in signatures:
                if re.search(signature, line):
                    locations.append({
                        'line_number': i + 1,
                        'line_content': line.strip(),
                        'pattern': signature
                    })
        
        return locations
    
    def _extract_relevant_code(self, content: str, tree: ast.AST, pattern_name: str) -> str:
        """Extract relevant code snippet for pattern"""
        # Simplified - return first 10 lines of file
        lines = content.split('\n')
        return '\n'.join(lines[:10])
    
    def _extract_repo_name(self, file_path: str) -> str:
        """Extract repository name from file path"""
        parts = Path(file_path).parts
        # Try to find a reasonable repository name
        for part in reversed(parts):
            if not part.startswith('.') and part not in ['src', 'lib', 'python']:
                return part
        return 'unknown'


class RepositoryManager:
    """Manages repository access and cloning"""
    
    def __init__(self, cache_dir: str = ".repo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.repositories = {}  # repo_id -> RepositoryInfo
        
    def add_repository(self, repo_url: str, repo_type: RepositoryType = RepositoryType.GIT) -> str:
        """Add a repository for analysis"""
        repo_id = hashlib.md5(repo_url.encode()).hexdigest()
        
        if repo_type == RepositoryType.LOCAL:
            local_path = repo_url
        else:
            # Clone repository to cache
            local_path = self._clone_repository(repo_url, repo_id)
        
        repo_info = RepositoryInfo(
            repo_id=repo_id,
            repo_type=repo_type,
            url=repo_url,
            local_path=local_path
        )
        
        self.repositories[repo_id] = repo_info
        return repo_id
    
    def get_repository_files(self, repo_id: str, pattern: str = "*.py") -> List[str]:
        """Get list of Python files in repository"""
        if repo_id not in self.repositories:
            return []
        
        repo_info = self.repositories[repo_id]
        repo_path = Path(repo_info.local_path)
        
        if not repo_path.exists():
            return []
        
        # Find all Python files
        python_files = list(repo_path.rglob(pattern))
        return [str(f) for f in python_files if f.is_file()]
    
    def _clone_repository(self, repo_url: str, repo_id: str) -> str:
        """Clone repository to local cache"""
        local_path = self.cache_dir / repo_id
        
        if local_path.exists():
            return str(local_path)
        
        try:
            # Use git clone
            subprocess.run([
                'git', 'clone', '--depth', '1', repo_url, str(local_path)
            ], check=True, capture_output=True)
            
            return str(local_path)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to clone repository {repo_url}: {e}")
            return ""


class PatternDatabase:
    """Database for storing and querying patterns"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    name TEXT,
                    description TEXT,
                    frequency INTEGER,
                    confidence REAL,
                    repositories TEXT,
                    tags TEXT,
                    metrics TEXT,
                    created_at REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT,
                    file_path TEXT,
                    line_number INTEGER,
                    code_snippet TEXT,
                    FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
                )
            ''')
            
            conn.commit()
    
    def store_pattern(self, pattern: CodePattern) -> None:
        """Store pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.pattern_type.value,
                pattern.name,
                pattern.description,
                pattern.frequency,
                pattern.confidence,
                json.dumps(list(pattern.repositories)),
                json.dumps(list(pattern.tags)),
                json.dumps(pattern.metrics),
                time.time()
            ))
            
            # Store examples
            for example in pattern.examples:
                conn.execute('''
                    INSERT INTO pattern_examples (pattern_id, file_path, line_number, code_snippet)
                    VALUES (?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    example.get('file_path', ''),
                    example.get('line_number', 0),
                    example.get('code_snippet', '')
                ))
            
            conn.commit()
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Dict[str, Any]]:
        """Get patterns by type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM patterns WHERE pattern_type = ?
            ''', (pattern_type.value,))
            
            columns = [desc[0] for desc in cursor.description]
            patterns = []
            
            for row in cursor.fetchall():
                pattern_dict = dict(zip(columns, row))
                pattern_dict['repositories'] = json.loads(pattern_dict['repositories'])
                pattern_dict['tags'] = json.loads(pattern_dict['tags'])
                pattern_dict['metrics'] = json.loads(pattern_dict['metrics'])
                patterns.append(pattern_dict)
            
            return patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total patterns by type
            cursor = conn.execute('''
                SELECT pattern_type, COUNT(*) FROM patterns GROUP BY pattern_type
            ''')
            patterns_by_type = dict(cursor.fetchall())
            
            # Total repositories
            cursor = conn.execute('SELECT COUNT(DISTINCT repositories) FROM patterns')
            total_repos = cursor.fetchone()[0]
            
            # Most common patterns
            cursor = conn.execute('''
                SELECT name, COUNT(*) as count FROM patterns 
                GROUP BY name ORDER BY count DESC LIMIT 10
            ''')
            common_patterns = cursor.fetchall()
            
            return {
                'patterns_by_type': patterns_by_type,
                'total_repositories': total_repos,
                'most_common_patterns': common_patterns
            }


class CrossRepositoryPatternMiner(BaseAnalyzer):
    """
    Cross-Repository Pattern Mining System
    
    Analyzes patterns across multiple repositories to identify common
    design patterns, anti-patterns, and architectural patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.config = config or {}
        self.cache_dir = self.config.get('cache_dir', '.repo_cache')
        self.db_path = self.config.get('db_path', 'patterns.db')
        self.max_workers = self.config.get('max_workers', 4)
        
        # Core components
        self.pattern_extractor = PatternExtractor()
        self.repo_manager = RepositoryManager(self.cache_dir)
        self.pattern_db = PatternDatabase(self.db_path)
        
        # Analysis state
        self.discovered_patterns = {}  # pattern_id -> CodePattern
        self.repository_stats = {}     # repo_id -> stats
        
        self.logger = logging.getLogger(__name__)
    
    def add_repository(self, repo_url: str, repo_type: RepositoryType = RepositoryType.GIT) -> str:
        """Add repository for pattern mining"""
        return self.repo_manager.add_repository(repo_url, repo_type)
    
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform cross-repository pattern analysis
        
        If file_path is provided, analyzes patterns in that specific file.
        Otherwise, analyzes all added repositories.
        """
        if file_path:
            return self._analyze_single_file(file_path)
        else:
            return self._analyze_all_repositories()
    
    def mine_patterns_across_repositories(self) -> Dict[str, Any]:
        """Mine patterns across all added repositories"""
        self.logger.info("Starting cross-repository pattern mining...")
        
        start_time = time.time()
        total_files = 0
        total_patterns = 0
        
        # Process each repository
        for repo_id, repo_info in self.repo_manager.repositories.items():
            self.logger.info(f"Mining patterns in repository: {repo_info.url}")
            
            repo_patterns = self._mine_repository_patterns(repo_id)
            files_analyzed = len(self.repo_manager.get_repository_files(repo_id))
            
            # Update stats
            self.repository_stats[repo_id] = {
                'files_analyzed': files_analyzed,
                'patterns_found': len(repo_patterns),
                'analysis_time': time.time()
            }
            
            total_files += files_analyzed
            total_patterns += len(repo_patterns)
        
        # Aggregate and cluster similar patterns
        clustered_patterns = self._cluster_similar_patterns()
        
        # Generate insights
        insights = self._generate_pattern_insights()
        
        analysis_time = time.time() - start_time
        
        self.logger.info(f"Pattern mining completed: {total_files} files, {total_patterns} patterns, {analysis_time:.2f}s")
        
        return {
            'mining_summary': {
                'repositories_analyzed': len(self.repo_manager.repositories),
                'total_files': total_files,
                'total_patterns': total_patterns,
                'unique_patterns': len(clustered_patterns),
                'analysis_time': analysis_time
            },
            'pattern_clusters': clustered_patterns,
            'repository_stats': self.repository_stats,
            'insights': insights,
            'pattern_statistics': self.pattern_db.get_pattern_statistics()
        }
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Dict[str, Any]]:
        """Get discovered patterns by type"""
        return self.pattern_db.get_patterns_by_type(pattern_type)
    
    def get_repository_patterns(self, repo_id: str) -> List[CodePattern]:
        """Get patterns discovered in a specific repository"""
        return [
            pattern for pattern in self.discovered_patterns.values()
            if repo_id in pattern.repositories
        ]
    
    def find_similar_patterns(self, pattern_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find patterns similar to the given pattern"""
        if pattern_id not in self.discovered_patterns:
            return []
        
        target_pattern = self.discovered_patterns[pattern_id]
        similar_patterns = []
        
        for pid, pattern in self.discovered_patterns.items():
            if pid != pattern_id and pattern.pattern_type == target_pattern.pattern_type:
                similarity = self._calculate_pattern_similarity(target_pattern, pattern)
                if similarity >= threshold:
                    similar_patterns.append((pid, similarity))
        
        return sorted(similar_patterns, key=lambda x: x[1], reverse=True)
    
    def generate_pattern_report(self) -> Dict[str, Any]:
        """Generate comprehensive pattern analysis report"""
        # Pattern distribution by type
        type_distribution = defaultdict(int)
        confidence_scores = []
        
        for pattern in self.discovered_patterns.values():
            type_distribution[pattern.pattern_type.value] += 1
            confidence_scores.append(pattern.confidence)
        
        # Repository coverage
        repo_coverage = {}
        for repo_id, repo_info in self.repo_manager.repositories.items():
            patterns_in_repo = len(self.get_repository_patterns(repo_id))
            repo_coverage[repo_info.url] = patterns_in_repo
        
        # Most common patterns
        pattern_frequency = Counter()
        for pattern in self.discovered_patterns.values():
            pattern_frequency[pattern.name] += pattern.frequency
        
        return {
            'summary': {
                'total_patterns': len(self.discovered_patterns),
                'total_repositories': len(self.repo_manager.repositories),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'pattern_types': len(type_distribution)
            },
            'distribution': {
                'by_type': dict(type_distribution),
                'by_repository': repo_coverage,
                'most_common': pattern_frequency.most_common(10)
            },
            'quality_metrics': {
                'high_confidence_patterns': len([p for p in self.discovered_patterns.values() if p.confidence > 0.8]),
                'cross_repo_patterns': len([p for p in self.discovered_patterns.values() if len(p.repositories) > 1]),
                'anti_patterns_found': len([p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.ANTI_PATTERN])
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze patterns in a single file"""
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            patterns = self.pattern_extractor.extract_patterns(file_path, content, tree)
            
            # Store patterns
            for pattern in patterns:
                self.discovered_patterns[pattern.pattern_id] = pattern
                self.pattern_db.store_pattern(pattern)
            
            return {
                'file_path': file_path,
                'patterns_found': len(patterns),
                'patterns': [pattern.to_dict() for pattern in patterns]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_all_repositories(self) -> Dict[str, Any]:
        """Analyze patterns across all repositories"""
        return self.mine_patterns_across_repositories()
    
    def _mine_repository_patterns(self, repo_id: str) -> List[CodePattern]:
        """Mine patterns in a specific repository"""
        files = self.repo_manager.get_repository_files(repo_id)
        patterns = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._analyze_file_patterns, file_path): file_path
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_patterns = future.result()
                    patterns.extend(file_patterns)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
        
        # Store patterns
        for pattern in patterns:
            self.discovered_patterns[pattern.pattern_id] = pattern
            self.pattern_db.store_pattern(pattern)
        
        return patterns
    
    def _analyze_file_patterns(self, file_path: str) -> List[CodePattern]:
        """Analyze patterns in a single file"""
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            tree = ast.parse(content)
            return self.pattern_extractor.extract_patterns(file_path, content, tree)
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return []
    
    def _cluster_similar_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster similar patterns together"""
        if not self.discovered_patterns:
            return {}
        
        # Group patterns by type first
        patterns_by_type = defaultdict(list)
        for pattern in self.discovered_patterns.values():
            patterns_by_type[pattern.pattern_type.value].append(pattern)
        
        clustered = {}
        
        for pattern_type, patterns in patterns_by_type.items():
            if len(patterns) < 2:
                clustered[pattern_type] = [p.to_dict() for p in patterns]
                continue
            
            # Create feature vectors for clustering
            feature_vectors = []
            pattern_list = []
            
            for pattern in patterns:
                # Simple feature vector based on name similarity and metrics
                features = [
                    pattern.confidence,
                    pattern.frequency,
                    len(pattern.repositories),
                    len(pattern.tags)
                ]
                feature_vectors.append(features)
                pattern_list.append(pattern)
            
            # Perform clustering
            if len(feature_vectors) > 1:
                try:
                    features_array = np.array(feature_vectors)
                    n_clusters = min(len(patterns) // 2, 5)  # Reasonable number of clusters
                    
                    if n_clusters > 1:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(features_array)
                    else:
                        cluster_labels = [0] * len(patterns)
                    
                    # Group by clusters
                    clusters = defaultdict(list)
                    for i, label in enumerate(cluster_labels):
                        clusters[f"cluster_{label}"].append(pattern_list[i].to_dict())
                    
                    clustered[pattern_type] = dict(clusters)
                    
                except Exception as e:
                    self.logger.error(f"Error clustering patterns: {e}")
                    clustered[pattern_type] = [p.to_dict() for p in patterns]
            else:
                clustered[pattern_type] = [p.to_dict() for p in patterns]
        
        return clustered
    
    def _generate_pattern_insights(self) -> List[str]:
        """Generate insights from discovered patterns"""
        insights = []
        
        if not self.discovered_patterns:
            return ["No patterns discovered yet."]
        
        # Anti-pattern insights
        anti_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.ANTI_PATTERN]
        if anti_patterns:
            insights.append(f"Found {len(anti_patterns)} anti-patterns that should be refactored")
        
        # Design pattern insights
        design_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.DESIGN_PATTERN]
        if design_patterns:
            most_common = Counter(p.name for p in design_patterns).most_common(1)
            if most_common:
                insights.append(f"Most commonly used design pattern: {most_common[0][0]}")
        
        # Cross-repository patterns
        cross_repo_patterns = [p for p in self.discovered_patterns.values() if len(p.repositories) > 1]
        if cross_repo_patterns:
            insights.append(f"Found {len(cross_repo_patterns)} patterns that appear across multiple repositories")
        
        # Security patterns
        security_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.SECURITY_PATTERN]
        if security_patterns:
            insights.append(f"Identified {len(security_patterns)} security-related patterns")
        else:
            insights.append("No security patterns detected - consider implementing security best practices")
        
        # High confidence patterns
        high_confidence = [p for p in self.discovered_patterns.values() if p.confidence > 0.8]
        if high_confidence:
            insights.append(f"{len(high_confidence)} patterns detected with high confidence (>80%)")
        
        return insights
    
    def _calculate_pattern_similarity(self, pattern1: CodePattern, pattern2: CodePattern) -> float:
        """Calculate similarity between two patterns"""
        # Simple similarity based on name, tags, and type
        similarity = 0.0
        
        # Name similarity (simple word overlap)
        words1 = set(pattern1.name.lower().split())
        words2 = set(pattern2.name.lower().split())
        name_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        
        # Tag similarity
        tag_similarity = len(pattern1.tags & pattern2.tags) / max(len(pattern1.tags | pattern2.tags), 1)
        
        # Type match
        type_match = 1.0 if pattern1.pattern_type == pattern2.pattern_type else 0.0
        
        # Weighted average
        similarity = (name_similarity * 0.4 + tag_similarity * 0.3 + type_match * 0.3)
        
        return similarity
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pattern analysis"""
        recommendations = []
        
        # Anti-pattern recommendations
        anti_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.ANTI_PATTERN]
        if anti_patterns:
            god_classes = [p for p in anti_patterns if 'god_class' in p.tags]
            if god_classes:
                recommendations.append("Refactor god classes by applying Single Responsibility Principle")
            
            long_methods = [p for p in anti_patterns if 'long_method' in p.tags]
            if long_methods:
                recommendations.append("Break down long methods into smaller, focused functions")
        
        # Security recommendations
        security_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.SECURITY_PATTERN]
        if len(security_patterns) < 3:
            recommendations.append("Implement more security patterns for better application security")
        
        # Design pattern recommendations
        design_patterns = [p for p in self.discovered_patterns.values() if p.pattern_type == PatternType.DESIGN_PATTERN]
        pattern_types = set(p.name for p in design_patterns)
        
        if 'Singleton Pattern' not in pattern_types:
            recommendations.append("Consider using Singleton pattern for global state management")
        
        if 'Factory Pattern' not in pattern_types:
            recommendations.append("Consider using Factory pattern for object creation")
        
        # General recommendations
        if len(self.repo_manager.repositories) < 3:
            recommendations.append("Add more repositories for better pattern analysis coverage")
        
        return recommendations