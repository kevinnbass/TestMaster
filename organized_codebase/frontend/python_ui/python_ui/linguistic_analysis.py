"""
Linguistic Analysis Module
==========================

Implements comprehensive linguistic analysis:
- Identifier naming conventions analysis
- Vocabulary metrics and diversity
- Comment and documentation quality
- Natural language patterns in code
- Readability and comprehension metrics
"""

import ast
import re
import string
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import statistics

from .base_analyzer import BaseAnalyzer


class LinguisticAnalyzer(BaseAnalyzer):
    """Analyzer for linguistic patterns in code."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self._init_linguistic_patterns()
    
    def _init_linguistic_patterns(self):
        """Initialize linguistic analysis patterns."""
        # Common English words (simplified set)
        self.common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
            'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'data', 'file',
            'name', 'path', 'size', 'type', 'value', 'item', 'list', 'dict', 'key', 'result', 'error'
        }
        
        # Technical abbreviations and their meanings
        self.abbreviations = {
            'id': 'identifier', 'num': 'number', 'str': 'string', 'obj': 'object', 'func': 'function',
            'var': 'variable', 'tmp': 'temporary', 'temp': 'temporary', 'val': 'value', 'param': 'parameter',
            'arg': 'argument', 'cfg': 'configuration', 'config': 'configuration', 'info': 'information',
            'msg': 'message', 'err': 'error', 'exc': 'exception', 'ctx': 'context', 'db': 'database',
            'api': 'application programming interface', 'url': 'uniform resource locator', 'uri': 'uniform resource identifier',
            'http': 'hypertext transfer protocol', 'json': 'javascript object notation', 'xml': 'extensible markup language'
        }
        
        # Domain-specific terms (programming)
        self.programming_terms = {
            'class', 'function', 'method', 'variable', 'parameter', 'argument', 'return', 'import',
            'module', 'package', 'library', 'framework', 'algorithm', 'data', 'structure', 'array',
            'list', 'dictionary', 'tuple', 'set', 'string', 'integer', 'float', 'boolean', 'null',
            'true', 'false', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'with', 'yield', 'lambda', 'async', 'await', 'def', 'class', 'self', 'super', 'init'
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive linguistic analysis."""
        print("[INFO] Analyzing Linguistic Patterns...")
        
        results = {
            "naming_conventions": self._analyze_naming_conventions(),
            "vocabulary_metrics": self._calculate_vocabulary_metrics(),
            "comment_analysis": self._analyze_comments(),
            "documentation_quality": self._assess_documentation_quality(),
            "readability_metrics": self._calculate_readability_metrics(),
            "abbreviation_analysis": self._analyze_abbreviations(),
            "domain_terminology": self._extract_domain_terms(),
            "natural_language_patterns": self._analyze_nl_patterns()
        }
        
        print(f"  [OK] Analyzed {len(results)} linguistic categories")
        return results
    
    def _analyze_naming_conventions(self) -> Dict[str, Any]:
        """Analyze identifier naming conventions and consistency."""
        naming_data = {
            'identifiers': [],
            'convention_violations': [],
            'consistency_analysis': {},
            'naming_patterns': defaultdict(int)
        }
        
        # Naming convention patterns
        conventions = {
            'snake_case': r'^[a-z_][a-z0-9_]*$',
            'camelCase': r'^[a-z][a-zA-Z0-9]*$',
            'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
            'UPPER_CASE': r'^[A-Z_][A-Z0-9_]*$',
            'kebab-case': r'^[a-z-][a-z0-9-]*$',
            'mixed_case': r'^[a-zA-Z][a-zA-Z0-9_]*$'
        }
        
        identifier_types = {
            'variables': [],
            'functions': [],
            'classes': [],
            'constants': [],
            'modules': []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    # Variables and attributes
                    if isinstance(node, ast.Name):
                        identifier_types['variables'].append({
                            'name': node.id,
                            'file': file_key,
                            'line': node.lineno,
                            'type': 'variable'
                        })
                    
                    # Functions
                    elif isinstance(node, ast.FunctionDef):
                        identifier_types['functions'].append({
                            'name': node.name,
                            'file': file_key,
                            'line': node.lineno,
                            'type': 'function'
                        })
                    
                    # Classes
                    elif isinstance(node, ast.ClassDef):
                        identifier_types['classes'].append({
                            'name': node.name,
                            'file': file_key,
                            'line': node.lineno,
                            'type': 'class'
                        })
                    
                    # Constants (uppercase variables)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                identifier_types['constants'].append({
                                    'name': target.id,
                                    'file': file_key,
                                    'line': node.lineno,
                                    'type': 'constant'
                                })
                
            except Exception:
                continue
        
        # Analyze naming patterns for each identifier type
        for id_type, identifiers in identifier_types.items():
            if not identifiers:
                continue
                
            type_conventions = defaultdict(int)
            
            for identifier in identifiers:
                name = identifier['name']
                
                # Skip special Python names
                if name.startswith('__') and name.endswith('__'):
                    continue
                
                # Check against each convention
                matched_convention = None
                for conv_name, pattern in conventions.items():
                    if re.match(pattern, name):
                        type_conventions[conv_name] += 1
                        matched_convention = conv_name
                        break
                
                naming_data['identifiers'].append({
                    **identifier,
                    'convention': matched_convention,
                    'length': len(name),
                    'word_count': len(re.split(r'[_\-]|(?=[A-Z])', name)),
                    'has_numbers': bool(re.search(r'\d', name)),
                    'abbreviation_ratio': self._calculate_abbreviation_ratio(name)
                })
                
                # Check for violations
                expected_convention = self._get_expected_convention(id_type)
                if matched_convention and matched_convention != expected_convention:
                    naming_data['convention_violations'].append({
                        **identifier,
                        'expected': expected_convention,
                        'actual': matched_convention
                    })
            
            naming_data['naming_patterns'][id_type] = dict(type_conventions)
        
        # Calculate consistency metrics
        total_identifiers = sum(len(ids) for ids in identifier_types.values())
        total_violations = len(naming_data['convention_violations'])
        
        naming_data['consistency_analysis'] = {
            'total_identifiers': total_identifiers,
            'convention_violations': total_violations,
            'consistency_score': (total_identifiers - total_violations) / max(total_identifiers, 1),
            'most_common_violation': self._get_most_common_violation(naming_data['convention_violations']),
            'identifier_length_stats': self._calculate_identifier_length_stats(naming_data['identifiers'])
        }
        
        return naming_data
    
    def _get_expected_convention(self, identifier_type: str) -> str:
        """Get expected naming convention for identifier type."""
        conventions_map = {
            'variables': 'snake_case',
            'functions': 'snake_case', 
            'classes': 'PascalCase',
            'constants': 'UPPER_CASE',
            'modules': 'snake_case'
        }
        return conventions_map.get(identifier_type, 'snake_case')
    
    def _calculate_abbreviation_ratio(self, name: str) -> float:
        """Calculate ratio of abbreviations in an identifier."""
        words = re.split(r'[_\-]|(?=[A-Z])', name.lower())
        abbreviations = [word for word in words if word in self.abbreviations]
        return len(abbreviations) / max(len(words), 1)
    
    def _get_most_common_violation(self, violations: List[Dict]) -> Optional[str]:
        """Get most common naming convention violation."""
        if not violations:
            return None
        violation_types = [v['actual'] for v in violations if v['actual']]
        if violation_types:
            return Counter(violation_types).most_common(1)[0][0]
        return None
    
    def _calculate_identifier_length_stats(self, identifiers: List[Dict]) -> Dict[str, float]:
        """Calculate identifier length statistics."""
        lengths = [id['length'] for id in identifiers]
        if not lengths:
            return {'mean': 0, 'median': 0, 'max': 0, 'min': 0}
        
        return {
            'mean': statistics.mean(lengths),
            'median': statistics.median(lengths),
            'max': max(lengths),
            'min': min(lengths),
            'std_dev': statistics.stdev(lengths) if len(lengths) > 1 else 0
        }
    
    def _calculate_vocabulary_metrics(self) -> Dict[str, Any]:
        """Calculate vocabulary richness and diversity metrics."""
        vocabulary_data = {
            'unique_words': set(),
            'word_frequencies': Counter(),
            'vocabulary_by_file': {},
            'programming_terms': set(),
            'domain_terms': set(),
            'abbreviations': set()
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Extract words from identifiers, comments, and strings
                words = self._extract_words_from_content(content)
                file_vocabulary = set(words)
                vocabulary_data['vocabulary_by_file'][file_key] = {
                    'word_count': len(words),
                    'unique_words': len(file_vocabulary),
                    'vocabulary_richness': len(file_vocabulary) / max(len(words), 1)
                }
                
                # Update global vocabulary
                vocabulary_data['unique_words'].update(file_vocabulary)
                vocabulary_data['word_frequencies'].update(words)
                
                # Categorize words
                for word in file_vocabulary:
                    if word.lower() in self.programming_terms:
                        vocabulary_data['programming_terms'].add(word.lower())
                    elif word.lower() in self.abbreviations:
                        vocabulary_data['abbreviations'].add(word.lower())
                    elif len(word) > 3 and word.lower() not in self.common_words:
                        vocabulary_data['domain_terms'].add(word.lower())
                
            except Exception:
                continue
        
        # Calculate metrics
        total_words = sum(vocabulary_data['word_frequencies'].values())
        unique_words = len(vocabulary_data['unique_words'])
        
        # Type-Token Ratio (TTR)
        ttr = unique_words / max(total_words, 1)
        
        # Hapax Legomena (words occurring only once)
        hapax_count = sum(1 for count in vocabulary_data['word_frequencies'].values() if count == 1)
        hapax_ratio = hapax_count / max(unique_words, 1)
        
        # Most frequent words
        most_frequent = vocabulary_data['word_frequencies'].most_common(10)
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'type_token_ratio': ttr,
            'hapax_legomena': hapax_count,
            'hapax_ratio': hapax_ratio,
            'programming_terms_count': len(vocabulary_data['programming_terms']),
            'domain_terms_count': len(vocabulary_data['domain_terms']),
            'abbreviations_count': len(vocabulary_data['abbreviations']),
            'vocabulary_diversity': self._calculate_vocabulary_diversity(vocabulary_data['word_frequencies']),
            'most_frequent_words': most_frequent,
            'per_file_metrics': vocabulary_data['vocabulary_by_file']
        }
    
    def _extract_words_from_content(self, content: str) -> List[str]:
        """Extract words from code content."""
        words = []
        
        try:
            tree = ast.parse(content)
            
            # Extract from identifiers
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.FunctionDef, ast.ClassDef)):
                    name = node.id if isinstance(node, ast.Name) else node.name
                    # Split camelCase and snake_case
                    name_words = re.split(r'[_\-]|(?=[A-Z])', name)
                    words.extend([w.lower() for w in name_words if w and w.isalpha()])
                
                # Extract from string literals
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Simple word extraction from strings
                    string_words = re.findall(r'\b[a-zA-Z]{2,}\b', node.value)
                    words.extend([w.lower() for w in string_words])
            
            # Extract from comments
            lines = content.split('\n')
            for line in lines:
                if '#' in line:
                    comment = line[line.index('#')+1:].strip()
                    comment_words = re.findall(r'\b[a-zA-Z]{2,}\b', comment)
                    words.extend([w.lower() for w in comment_words])
                    
        except Exception:
            # Fallback: simple regex extraction
            words = re.findall(r'\b[a-zA-Z]{2,}\b', content.lower())
        
        return words
    
    def _calculate_vocabulary_diversity(self, word_frequencies: Counter) -> float:
        """Calculate vocabulary diversity using Shannon entropy."""
        if not word_frequencies:
            return 0.0
        
        total = sum(word_frequencies.values())
        entropy = 0.0
        
        for count in word_frequencies.values():
            probability = count / total
            entropy += probability * (-1) * (probability ** 0.5 if probability > 0 else 0)
        
        return entropy
    
    def _analyze_comments(self) -> Dict[str, Any]:
        """Analyze comment quality and patterns."""
        comment_data = {
            'total_comments': 0,
            'comment_lines': 0,
            'comments_by_type': defaultdict(int),
            'comment_quality_indicators': [],
            'files_with_comments': 0,
            'avg_comment_length': 0
        }
        
        comment_lengths = []
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                file_has_comments = False
                file_comments = []
                
                for line_num, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        comment_data['total_comments'] += 1
                        comment_data['comment_lines'] += 1
                        file_has_comments = True
                        
                        comment_text = stripped[1:].strip()
                        file_comments.append(comment_text)
                        comment_lengths.append(len(comment_text))
                        
                        # Classify comment type
                        if comment_text.upper().startswith(('TODO', 'FIXME', 'HACK', 'BUG')):
                            comment_data['comments_by_type']['task'] += 1
                        elif len(comment_text) > 50:
                            comment_data['comments_by_type']['explanatory'] += 1
                        elif any(word in comment_text.lower() for word in ['param', 'return', 'arg']):
                            comment_data['comments_by_type']['documentation'] += 1
                        else:
                            comment_data['comments_by_type']['general'] += 1
                
                if file_has_comments:
                    comment_data['files_with_comments'] += 1
                
                # Analyze comment quality for this file
                if file_comments:
                    avg_length = statistics.mean([len(c) for c in file_comments])
                    quality_score = self._calculate_comment_quality_score(file_comments)
                    
                    comment_data['comment_quality_indicators'].append({
                        'file': file_key,
                        'comment_count': len(file_comments),
                        'avg_length': avg_length,
                        'quality_score': quality_score
                    })
                
            except Exception:
                continue
        
        # Calculate summary statistics
        total_files = len(list(self._get_python_files()))
        comment_data['avg_comment_length'] = statistics.mean(comment_lengths) if comment_lengths else 0
        comment_data['comment_density'] = comment_data['comment_lines'] / max(total_files, 1)
        comment_data['files_with_comments_ratio'] = comment_data['files_with_comments'] / max(total_files, 1)
        
        overall_quality = statistics.mean([
            ind['quality_score'] for ind in comment_data['comment_quality_indicators']
        ]) if comment_data['comment_quality_indicators'] else 0
        
        comment_data['overall_quality_score'] = overall_quality
        comment_data['comment_types_distribution'] = dict(comment_data['comments_by_type'])
        
        return comment_data
    
    def _calculate_comment_quality_score(self, comments: List[str]) -> float:
        """Calculate quality score for comments."""
        if not comments:
            return 0.0
        
        quality_indicators = {
            'has_punctuation': 0,
            'proper_length': 0,
            'descriptive_words': 0,
            'technical_terms': 0,
            'complete_sentences': 0
        }
        
        for comment in comments:
            # Check punctuation
            if any(p in comment for p in '.!?'):
                quality_indicators['has_punctuation'] += 1
            
            # Check length (not too short, not too long)
            if 10 <= len(comment) <= 100:
                quality_indicators['proper_length'] += 1
            
            # Check for descriptive words
            words = comment.lower().split()
            descriptive_words = {'explain', 'calculate', 'process', 'handle', 'check', 'validate', 'ensure'}
            if any(word in words for word in descriptive_words):
                quality_indicators['descriptive_words'] += 1
            
            # Check for technical terms
            if any(term in words for term in self.programming_terms):
                quality_indicators['technical_terms'] += 1
            
            # Check for complete sentences (basic heuristic)
            if len(comment) > 15 and comment[0].isupper() and comment.endswith('.'):
                quality_indicators['complete_sentences'] += 1
        
        # Calculate weighted score
        total_comments = len(comments)
        score = (
            (quality_indicators['has_punctuation'] * 0.2) +
            (quality_indicators['proper_length'] * 0.3) +
            (quality_indicators['descriptive_words'] * 0.2) +
            (quality_indicators['technical_terms'] * 0.2) +
            (quality_indicators['complete_sentences'] * 0.1)
        ) / total_comments
        
        return min(score, 1.0)
    
    def _assess_documentation_quality(self) -> Dict[str, Any]:
        """Assess documentation quality (docstrings, README, etc.)."""
        doc_data = {
            'docstring_coverage': {},
            'docstring_quality': [],
            'documentation_files': [],
            'api_documentation': []
        }
        
        # Analyze docstring coverage
        total_functions = 0
        functions_with_docstrings = 0
        total_classes = 0
        classes_with_docstrings = 0
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                file_functions = 0
                file_docstring_functions = 0
                file_classes = 0
                file_docstring_classes = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        file_functions += 1
                        
                        docstring = ast.get_docstring(node)
                        if docstring:
                            functions_with_docstrings += 1
                            file_docstring_functions += 1
                            
                            # Analyze docstring quality
                            quality_score = self._analyze_docstring_quality(docstring)
                            doc_data['docstring_quality'].append({
                                'file': file_key,
                                'function': node.name,
                                'line': node.lineno,
                                'length': len(docstring),
                                'quality_score': quality_score
                            })
                    
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        file_classes += 1
                        
                        docstring = ast.get_docstring(node)
                        if docstring:
                            classes_with_docstrings += 1
                            file_docstring_classes += 1
                
                # File-level docstring coverage
                if file_functions > 0 or file_classes > 0:
                    doc_data['docstring_coverage'][file_key] = {
                        'function_coverage': file_docstring_functions / max(file_functions, 1),
                        'class_coverage': file_docstring_classes / max(file_classes, 1),
                        'total_functions': file_functions,
                        'total_classes': file_classes
                    }
                
            except Exception:
                continue
        
        # Look for documentation files
        doc_patterns = ['README*', '*.md', '*.rst', '*.txt', 'docs/*']
        for pattern in doc_patterns:
            for doc_file in self.base_path.glob(pattern):
                if doc_file.is_file():
                    doc_data['documentation_files'].append({
                        'file': str(doc_file.relative_to(self.base_path)),
                        'size': doc_file.stat().st_size,
                        'type': doc_file.suffix
                    })
        
        # Calculate overall metrics
        function_coverage = functions_with_docstrings / max(total_functions, 1)
        class_coverage = classes_with_docstrings / max(total_classes, 1)
        
        avg_quality = statistics.mean([
            item['quality_score'] for item in doc_data['docstring_quality']
        ]) if doc_data['docstring_quality'] else 0
        
        doc_data['summary'] = {
            'function_docstring_coverage': function_coverage,
            'class_docstring_coverage': class_coverage,
            'overall_docstring_coverage': (function_coverage + class_coverage) / 2,
            'average_docstring_quality': avg_quality,
            'documentation_files_count': len(doc_data['documentation_files']),
            'total_functions_analyzed': total_functions,
            'total_classes_analyzed': total_classes
        }
        
        return doc_data
    
    def _analyze_docstring_quality(self, docstring: str) -> float:
        """Analyze the quality of a docstring."""
        if not docstring:
            return 0.0
        
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        if 20 <= len(docstring) <= 500:
            quality_score += 0.2
        
        # Check for proper formatting
        lines = docstring.strip().split('\n')
        if len(lines) > 1:  # Multi-line
            quality_score += 0.1
        
        # Check for parameter documentation
        if re.search(r'(param|arg|parameter)', docstring, re.IGNORECASE):
            quality_score += 0.2
        
        # Check for return documentation
        if re.search(r'(return|returns)', docstring, re.IGNORECASE):
            quality_score += 0.2
        
        # Check for examples
        if re.search(r'(example|usage)', docstring, re.IGNORECASE):
            quality_score += 0.1
        
        # Check for proper grammar (basic)
        if docstring[0].isupper() and docstring.endswith('.'):
            quality_score += 0.1
        
        # Check for technical terms
        words = docstring.lower().split()
        if any(term in words for term in self.programming_terms):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _calculate_readability_metrics(self) -> Dict[str, Any]:
        """Calculate code readability metrics."""
        readability_data = {
            'avg_line_length': 0,
            'complex_expressions': 0,
            'nesting_depth_distribution': defaultdict(int),
            'identifier_readability': []
        }
        
        line_lengths = []
        complex_expr_count = 0
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Line length analysis
                for line in lines:
                    if line.strip():  # Skip empty lines
                        line_lengths.append(len(line))
                
                # Complex expression analysis
                for node in ast.walk(tree):
                    if isinstance(node, ast.BoolOp) and len(node.values) > 2:
                        complex_expr_count += 1
                    elif isinstance(node, ast.Compare) and len(node.comparators) > 1:
                        complex_expr_count += 1
                
                # Nesting depth analysis
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        max_depth = self._calculate_max_nesting_depth(node)
                        readability_data['nesting_depth_distribution'][max_depth] += 1
                
            except Exception:
                continue
        
        readability_data['avg_line_length'] = statistics.mean(line_lengths) if line_lengths else 0
        readability_data['complex_expressions'] = complex_expr_count
        readability_data['line_length_distribution'] = self._calculate_line_length_distribution(line_lengths)
        
        # Calculate overall readability score
        readability_score = self._calculate_readability_score(readability_data)
        readability_data['readability_score'] = readability_score
        
        return readability_data
    
    def _calculate_max_nesting_depth(self, func_node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth in a function."""
        max_depth = 0
        
        def get_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nesting_nodes):
                    get_depth(child, current_depth + 1)
                else:
                    get_depth(child, current_depth)
        
        get_depth(func_node)
        return max_depth
    
    def _calculate_line_length_distribution(self, line_lengths: List[int]) -> Dict[str, int]:
        """Calculate distribution of line lengths."""
        distribution = {
            '0-50': 0,
            '51-80': 0,
            '81-120': 0,
            '>120': 0
        }
        
        for length in line_lengths:
            if length <= 50:
                distribution['0-50'] += 1
            elif length <= 80:
                distribution['51-80'] += 1
            elif length <= 120:
                distribution['81-120'] += 1
            else:
                distribution['>120'] += 1
        
        return distribution
    
    def _calculate_readability_score(self, readability_data: Dict) -> float:
        """Calculate overall readability score."""
        score = 100.0
        
        # Penalize long lines
        avg_line_length = readability_data['avg_line_length']
        if avg_line_length > 120:
            score -= 20
        elif avg_line_length > 80:
            score -= 10
        
        # Penalize complex expressions
        if readability_data['complex_expressions'] > 10:
            score -= 15
        
        # Penalize deep nesting
        deep_nesting = sum(count for depth, count in readability_data['nesting_depth_distribution'].items() if depth > 3)
        if deep_nesting > 5:
            score -= 20
        
        return max(score, 0.0)
    
    def _analyze_abbreviations(self) -> Dict[str, Any]:
        """Analyze abbreviation usage patterns."""
        abbrev_data = {
            'abbreviations_found': [],
            'unclear_abbreviations': [],
            'abbreviation_consistency': {},
            'expansion_suggestions': []
        }
        
        # Extract identifiers and check for abbreviations
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Name, ast.FunctionDef, ast.ClassDef)):
                        name = node.id if isinstance(node, ast.Name) else node.name
                        
                        # Split identifier into words
                        words = re.split(r'[_\-]|(?=[A-Z])', name.lower())
                        
                        for word in words:
                            if word in self.abbreviations:
                                abbrev_data['abbreviations_found'].append({
                                    'abbreviation': word,
                                    'expansion': self.abbreviations[word],
                                    'identifier': name,
                                    'file': file_key,
                                    'line': getattr(node, 'lineno', 0)
                                })
                            elif len(word) <= 3 and word.isalpha() and word not in self.common_words:
                                # Potential unclear abbreviation
                                abbrev_data['unclear_abbreviations'].append({
                                    'abbreviation': word,
                                    'identifier': name,
                                    'file': file_key,
                                    'line': getattr(node, 'lineno', 0),
                                    'suggestion': 'Consider using full word'
                                })
                
            except Exception:
                continue
        
        # Analyze consistency
        abbrev_usage = Counter([item['abbreviation'] for item in abbrev_data['abbreviations_found']])
        
        abbrev_data['summary'] = {
            'total_abbreviations': len(abbrev_data['abbreviations_found']),
            'unclear_abbreviations': len(abbrev_data['unclear_abbreviations']),
            'abbreviation_ratio': len(abbrev_data['abbreviations_found']) / max(
                len(abbrev_data['abbreviations_found']) + len(abbrev_data['unclear_abbreviations']), 1
            ),
            'most_used_abbreviations': abbrev_usage.most_common(5)
        }
        
        return abbrev_data
    
    def _extract_domain_terms(self) -> Dict[str, Any]:
        """Extract domain-specific terminology."""
        domain_data = {
            'technical_terms': set(),
            'business_terms': set(),
            'domain_vocabulary': Counter(),
            'term_categories': defaultdict(set)
        }
        
        # Define domain categories
        web_terms = {'http', 'url', 'api', 'json', 'xml', 'html', 'css', 'javascript', 'server', 'client'}
        data_terms = {'database', 'query', 'table', 'record', 'field', 'sql', 'nosql', 'mongodb', 'redis'}
        ml_terms = {'model', 'train', 'predict', 'algorithm', 'neural', 'learning', 'classification', 'regression'}
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                words = self._extract_words_from_content(content)
                
                for word in words:
                    word_lower = word.lower()
                    
                    # Categorize terms
                    if word_lower in web_terms:
                        domain_data['term_categories']['web'].add(word_lower)
                    elif word_lower in data_terms:
                        domain_data['term_categories']['data'].add(word_lower)
                    elif word_lower in ml_terms:
                        domain_data['term_categories']['machine_learning'].add(word_lower)
                    elif word_lower in self.programming_terms:
                        domain_data['technical_terms'].add(word_lower)
                    elif len(word) > 4 and word_lower not in self.common_words:
                        domain_data['business_terms'].add(word_lower)
                    
                    domain_data['domain_vocabulary'][word_lower] += 1
                
            except Exception:
                continue
        
        return {
            'technical_terms_count': len(domain_data['technical_terms']),
            'business_terms_count': len(domain_data['business_terms']),
            'domain_categories': {k: len(v) for k, v in domain_data['term_categories'].items()},
            'most_frequent_domain_terms': domain_data['domain_vocabulary'].most_common(10),
            'vocabulary_specialization': len(domain_data['technical_terms']) / max(
                len(domain_data['technical_terms']) + len(domain_data['business_terms']), 1
            )
        }
    
    def _analyze_nl_patterns(self) -> Dict[str, Any]:
        """Analyze natural language patterns in code."""
        nl_data = {
            'sentence_structures': defaultdict(int),
            'grammar_patterns': defaultdict(int),
            'language_consistency': 0.0,
            'readability_indicators': []
        }
        
        # Analyze comments and docstrings for natural language patterns
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                
                # Analyze docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            self._analyze_text_patterns(docstring, nl_data)
                
                # Analyze comments
                lines = content.split('\n')
                for line in lines:
                    if '#' in line:
                        comment = line[line.index('#')+1:].strip()
                        if len(comment) > 10:  # Skip very short comments
                            self._analyze_text_patterns(comment, nl_data)
                
            except Exception:
                continue
        
        # Calculate language consistency
        total_patterns = sum(nl_data['grammar_patterns'].values())
        if total_patterns > 0:
            # Simple consistency metric based on pattern distribution
            pattern_ratios = [count/total_patterns for count in nl_data['grammar_patterns'].values()]
            consistency = 1.0 - (statistics.stdev(pattern_ratios) if len(pattern_ratios) > 1 else 0)
            nl_data['language_consistency'] = consistency
        
        return dict(nl_data)
    
    def _analyze_text_patterns(self, text: str, nl_data: Dict):
        """Analyze patterns in natural language text."""
        # Simple sentence structure analysis
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                # Count sentence types
                if sentence.endswith('?'):
                    nl_data['sentence_structures']['question'] += 1
                elif sentence.endswith('!'):
                    nl_data['sentence_structures']['exclamation'] += 1
                else:
                    nl_data['sentence_structures']['statement'] += 1
                
                # Basic grammar pattern detection
                words = sentence.lower().split()
                if len(words) > 0:
                    if words[0] in ['this', 'the', 'a', 'an']:
                        nl_data['grammar_patterns']['article_start'] += 1
                    if words[0] in ['return', 'returns', 'get', 'set', 'create', 'delete']:
                        nl_data['grammar_patterns']['verb_start'] += 1
                    if any(word in words for word in ['and', 'or', 'but', 'however']):
                        nl_data['grammar_patterns']['conjunctions'] += 1