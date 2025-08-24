"""
Business Logic Analyzer - Core Implementation
==============================================

Complete business logic analysis including domain model extraction,
business rule detection, and process flow analysis.
"""

import ast
import re
import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib

from .data_models import BusinessRule, DomainEntity, BusinessProcess

logger = logging.getLogger(__name__)


class BusinessLogicAnalyzer:
    """
    Complete Business Logic Analyzer - Enhanced Implementation
    
    Comprehensive business logic analysis including:
    - Domain model extraction
    - Business rule detection  
    - Process flow analysis
    - Decision logic mapping
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Business Logic Analyzer"""
        self.config = config or self._get_default_config()
        
        # Analysis results storage
        self.business_rules: List[BusinessRule] = []
        self.domain_entities: List[DomainEntity] = []
        self.business_processes: List[BusinessProcess] = []
        
        # Pattern definitions for business logic detection
        self.business_patterns = self._initialize_business_patterns()
        
        # Analysis caching
        self.analysis_cache: Dict[str, Any] = {}
        self.processed_files: Set[str] = set()
        
        # Statistics
        self.analysis_stats = {
            'files_processed': 0,
            'rules_found': 0,
            'entities_found': 0,
            'processes_found': 0,
            'total_analysis_time': 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'include_patterns': ['*.py', '*.js', '*.ts', '*.java'],
            'exclude_patterns': ['test_*', '*_test*', '*/tests/*'],
            'min_rule_confidence': 0.6,
            'extract_domain_entities': True,
            'extract_business_processes': True,
            'extract_business_rules': True,
            'cache_analysis': True,
            'detailed_analysis': True
        }
    
    def _initialize_business_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize business logic detection patterns"""
        return {
            'validation_rules': {
                'patterns': [
                    r'def\s+validate_\w+',
                    r'if.*(?:len|length|size).*[<>=]',
                    r'if.*(?:email|phone|postal|zip).*match',
                    r'ValidationError|ValueError',
                    r'(?:min|max)_(?:length|value|amount)',
                ],
                'rule_type': 'validation',
                'confidence_boost': 0.8
            },
            'business_calculation': {
                'patterns': [
                    r'def\s+calculate_\w+',
                    r'(?:price|cost|fee|rate|amount|total)\s*[*+\-/=]',
                    r'(?:tax|discount|commission|interest)',
                    r'def\s+(?:get_|compute_)?(?:price|cost|total)',
                ],
                'rule_type': 'calculation',
                'confidence_boost': 0.7
            },
            'workflow_rules': {
                'patterns': [
                    r'def\s+(?:process|handle|execute)_\w+',
                    r'if.*status.*==.*["\'](?:pending|approved|rejected)',
                    r'state\s*=\s*["\'](?:active|inactive|completed)',
                    r'workflow|process|step|stage',
                ],
                'rule_type': 'workflow',
                'confidence_boost': 0.6
            },
            'authorization_rules': {
                'patterns': [
                    r'def\s+(?:can_|is_|has_|check_)\w+',
                    r'@(?:requires?|permission|authorize)',
                    r'if.*(?:user|role|permission)',
                    r'(?:admin|owner|member).*(?:only|required)',
                ],
                'rule_type': 'authorization',
                'confidence_boost': 0.9
            },
            'data_transformation': {
                'patterns': [
                    r'def\s+(?:transform|convert|map|serialize)',
                    r'json\.(?:loads|dumps)',
                    r'\.to_dict\(\)|\.from_dict\(',
                    r'(?:format|parse)_\w+',
                ],
                'rule_type': 'transformation',
                'confidence_boost': 0.5
            }
        }
    
    def analyze_codebase(self, root_path: str) -> Dict[str, Any]:
        """Analyze entire codebase for business logic"""
        start_time = datetime.now()
        logger.info(f"Starting business logic analysis of {root_path}")
        
        # Reset analysis state
        self.business_rules.clear()
        self.domain_entities.clear()
        self.business_processes.clear()
        self.processed_files.clear()
        
        # Find and analyze files
        files_to_analyze = self._find_files_to_analyze(root_path)
        
        for file_path in files_to_analyze:
            try:
                self._analyze_file(file_path)
                self.analysis_stats['files_processed'] += 1
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Post-process results
        self._post_process_analysis()
        
        # Update statistics
        analysis_time = (datetime.now() - start_time).total_seconds()
        self.analysis_stats['total_analysis_time'] = analysis_time
        self.analysis_stats['rules_found'] = len(self.business_rules)
        self.analysis_stats['entities_found'] = len(self.domain_entities)
        self.analysis_stats['processes_found'] = len(self.business_processes)
        
        logger.info(f"Business logic analysis completed in {analysis_time:.2f}s")
        logger.info(f"Found {len(self.business_rules)} rules, {len(self.domain_entities)} entities, {len(self.business_processes)} processes")
        
        return self._generate_analysis_report()
    
    def _find_files_to_analyze(self, root_path: str) -> List[str]:
        """Find files to analyze based on patterns"""
        files_to_analyze = []
        root = Path(root_path)
        
        for include_pattern in self.config['include_patterns']:
            for file_path in root.rglob(include_pattern):
                if self._should_analyze_file(str(file_path)):
                    files_to_analyze.append(str(file_path))
        
        return files_to_analyze
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Check if file should be analyzed"""
        # Check exclude patterns
        for exclude_pattern in self.config['exclude_patterns']:
            if re.search(exclude_pattern.replace('*', '.*'), file_path):
                return False
        
        # Check if already processed
        if file_path in self.processed_files:
            return False
        
        return True
    
    def _analyze_file(self, file_path: str) -> None:
        """Analyze a single file for business logic"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.processed_files.add(file_path)
            
            # Extract business rules
            if self.config['extract_business_rules']:
                rules = self._extract_business_rules(content, file_path)
                self.business_rules.extend(rules)
            
            # Extract domain entities
            if self.config['extract_domain_entities']:
                entities = self._extract_domain_entities(content, file_path)
                self.domain_entities.extend(entities)
            
            # Extract business processes
            if self.config['extract_business_processes']:
                processes = self._extract_business_processes(content, file_path)
                self.business_processes.extend(processes)
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    def _extract_business_rules(self, content: str, file_path: str) -> List[BusinessRule]:
        """Extract business rules from code content"""
        rules = []
        
        try:
            # Try to parse as Python AST
            tree = ast.parse(content)
            
            # Analyze functions for business logic patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_rules = self._analyze_function_for_rules(node, content, file_path)
                    rules.extend(function_rules)
                    
        except SyntaxError:
            # Fallback to pattern matching for non-Python files
            rules.extend(self._extract_rules_by_patterns(content, file_path))
        
        return rules
    
    def _analyze_function_for_rules(self, func_node: ast.FunctionDef, content: str, file_path: str) -> List[BusinessRule]:
        """Analyze a function for business rules"""
        rules = []
        func_name = func_node.name
        
        # Get function source
        try:
            func_lines = content.split('\n')[func_node.lineno-1:func_node.end_lineno]
            func_source = '\n'.join(func_lines)
        except:
            func_source = func_name
        
        # Check against business patterns
        for pattern_category, pattern_info in self.business_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, func_source, re.IGNORECASE):
                    rule = BusinessRule(
                        rule_type=pattern_info['rule_type'],
                        description=f"Business rule detected in function '{func_name}'",
                        location=f"{file_path}:{func_node.lineno}",
                        implementation=func_source[:200] + "..." if len(func_source) > 200 else func_source,
                        priority="medium",
                        business_impact="medium"
                    )
                    
                    # Enhance rule details based on pattern type
                    self._enhance_rule_details(rule, func_source, pattern_info['rule_type'])
                    rules.append(rule)
                    break  # Only one rule per function per category
        
        return rules
    
    def _extract_rules_by_patterns(self, content: str, file_path: str) -> List[BusinessRule]:
        """Extract rules using pattern matching"""
        rules = []
        
        for pattern_category, pattern_info in self.business_patterns.items():
            for pattern in pattern_info['patterns']:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    rule = BusinessRule(
                        rule_type=pattern_info['rule_type'],
                        description=f"Business rule pattern detected: {pattern_category}",
                        location=f"{file_path}:{line_num}",
                        implementation=match.group()[:200],
                        priority="medium",
                        business_impact="medium"
                    )
                    rules.append(rule)
        
        return rules
    
    def _enhance_rule_details(self, rule: BusinessRule, source: str, rule_type: str) -> None:
        """Enhance rule details based on source analysis"""
        if rule_type == 'validation':
            if 'email' in source.lower():
                rule.description += " - Email validation"
            elif 'password' in source.lower():
                rule.description += " - Password validation"
            elif 'length' in source.lower():
                rule.description += " - Length validation"
        
        elif rule_type == 'calculation':
            if any(word in source.lower() for word in ['price', 'cost', 'total']):
                rule.priority = "high"
                rule.business_impact = "high"
        
        elif rule_type == 'authorization':
            rule.priority = "high"
            rule.business_impact = "high"
    
    def _extract_domain_entities(self, content: str, file_path: str) -> List[DomainEntity]:
        """Extract domain entities from code"""
        entities = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entity = self._analyze_class_as_entity(node, file_path)
                    if entity:
                        entities.append(entity)
                        
        except SyntaxError:
            # Fallback for non-Python files
            pass
        
        return entities
    
    def _analyze_class_as_entity(self, class_node: ast.ClassDef, file_path: str) -> Optional[DomainEntity]:
        """Analyze a class as potential domain entity"""
        class_name = class_node.name
        
        # Skip obvious non-entity classes
        skip_patterns = ['test', 'mock', 'base', 'abstract', 'helper', 'util', 'config']
        if any(pattern in class_name.lower() for pattern in skip_patterns):
            return None
        
        # Extract attributes and methods
        attributes = []
        methods = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Public methods only
                    methods.append({
                        'name': node.name,
                        'type': 'method',
                        'line': node.lineno
                    })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': 'attribute',
                            'line': node.lineno
                        })
        
        # Only consider classes with both attributes and methods as entities
        if len(attributes) > 0 and len(methods) > 0:
            return DomainEntity(
                name=class_name,
                entity_type="domain_entity",
                attributes=attributes,
                methods=methods,
                location=f"{file_path}:{class_node.lineno}",
                business_value="medium"
            )
        
        return None
    
    def _extract_business_processes(self, content: str, file_path: str) -> List[BusinessProcess]:
        """Extract business processes from code"""
        processes = []
        
        # Look for process-like functions
        process_patterns = [
            r'def\s+(process|handle|execute|workflow|run)_\w+',
            r'def\s+\w*(?:process|workflow|procedure)\w*',
        ]
        
        for pattern in process_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                
                process = BusinessProcess(
                    name=match.group().replace('def ', '').split('(')[0],
                    steps=self._extract_process_steps(content, match.start()),
                    location=f"{file_path}:{line_num}",
                    business_value="medium"
                )
                processes.append(process)
        
        return processes
    
    def _extract_process_steps(self, content: str, start_pos: int) -> List[Dict[str, str]]:
        """Extract steps from a process function"""
        # Simple step extraction - look for comments or specific patterns
        steps = []
        
        # Get the function content (simplified)
        lines = content[start_pos:].split('\n')[:20]  # First 20 lines
        
        step_num = 1
        for line in lines:
            line = line.strip()
            if line.startswith('#') and any(word in line.lower() for word in ['step', 'stage', 'phase']):
                steps.append({
                    'step_number': str(step_num),
                    'description': line.replace('#', '').strip(),
                    'type': 'process_step'
                })
                step_num += 1
        
        return steps
    
    def _post_process_analysis(self) -> None:
        """Post-process analysis results"""
        # Remove duplicates
        self._remove_duplicate_rules()
        self._remove_duplicate_entities()
        
        # Enhance relationships
        self._discover_entity_relationships()
    
    def _remove_duplicate_rules(self) -> None:
        """Remove duplicate business rules"""
        seen_rules = set()
        unique_rules = []
        
        for rule in self.business_rules:
            rule_key = f"{rule.rule_type}:{rule.location}:{rule.description}"
            if rule_key not in seen_rules:
                seen_rules.add(rule_key)
                unique_rules.append(rule)
        
        self.business_rules = unique_rules
    
    def _remove_duplicate_entities(self) -> None:
        """Remove duplicate domain entities"""
        seen_entities = set()
        unique_entities = []
        
        for entity in self.domain_entities:
            entity_key = f"{entity.name}:{entity.location}"
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
        
        self.domain_entities = unique_entities
    
    def _discover_entity_relationships(self) -> None:
        """Discover relationships between entities"""
        entity_names = {entity.name.lower() for entity in self.domain_entities}
        
        for entity in self.domain_entities:
            # Look for references to other entities in methods
            for method in entity.methods:
                method_name = method['name'].lower()
                for other_entity_name in entity_names:
                    if other_entity_name != entity.name.lower() and other_entity_name in method_name:
                        relationship = {
                            'type': 'reference',
                            'target_entity': other_entity_name,
                            'context': f"Method {method['name']} references {other_entity_name}"
                        }
                        if relationship not in entity.relationships:
                            entity.relationships.append(relationship)
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        return {
            'summary': {
                'total_files_processed': self.analysis_stats['files_processed'],
                'business_rules_found': len(self.business_rules),
                'domain_entities_found': len(self.domain_entities),
                'business_processes_found': len(self.business_processes),
                'analysis_time_seconds': self.analysis_stats['total_analysis_time']
            },
            'business_rules': [rule.to_dict() for rule in self.business_rules],
            'domain_entities': [entity.to_dict() for entity in self.domain_entities],
            'business_processes': [process.to_dict() for process in self.business_processes],
            'statistics': self.analysis_stats,
            'rule_types_distribution': self._get_rule_types_distribution(),
            'entity_types_distribution': self._get_entity_types_distribution()
        }
    
    def _get_rule_types_distribution(self) -> Dict[str, int]:
        """Get distribution of rule types"""
        distribution = defaultdict(int)
        for rule in self.business_rules:
            distribution[rule.rule_type] += 1
        return dict(distribution)
    
    def _get_entity_types_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types"""
        distribution = defaultdict(int)
        for entity in self.domain_entities:
            distribution[entity.entity_type] += 1
        return dict(distribution)
    
    def export_analysis(self, output_path: str, format: str = 'json') -> None:
        """Export analysis results"""
        report = self._generate_analysis_report()
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Analysis exported to {output_path}")


def create_business_logic_analyzer(config: Optional[Dict[str, Any]] = None) -> BusinessLogicAnalyzer:
    """Factory function to create a business logic analyzer"""
    return BusinessLogicAnalyzer(config)


__all__ = ['BusinessLogicAnalyzer', 'create_business_logic_analyzer']