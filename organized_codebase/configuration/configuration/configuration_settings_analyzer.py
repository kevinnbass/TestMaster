#!/usr/bin/env python3
"""
Configuration & Settings Analysis Tool - Agent C Hours 41-43
Analyzes configuration files, settings patterns, and environment variables across the codebase.
Identifies configuration consolidation opportunities and security patterns.
"""

import ast
import json
import argparse
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys

class ConfigurationSettingsAnalyzer(ast.NodeVisitor):
    """Analyzes configuration and settings patterns across the codebase."""
    
    def __init__(self):
        self.config_patterns = defaultdict(list)
        self.environment_variables = defaultdict(set)
        self.config_files = []
        self.settings_classes = []
        self.hardcoded_values = []
        self.config_usage_patterns = defaultdict(int)
        self.security_concerns = []
        self.consolidation_opportunities = []
        self.current_file = None
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for configuration patterns."""
        self.current_file = str(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if this is a configuration file
            if self._is_config_file(file_path, content):
                self.config_files.append({
                    'file': str(file_path),
                    'type': self._determine_config_type(file_path, content),
                    'patterns': self._extract_config_patterns(content)
                })
            
            # Parse AST for configuration usage
            tree = ast.parse(content, filename=str(file_path))
            self.visit(tree)
            
            # Analyze raw content for additional patterns
            self._analyze_raw_content(content)
            
            return {
                'file': str(file_path),
                'is_config_file': self._is_config_file(file_path, content),
                'analyzed': True
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'analyzed': False
            }
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find settings classes."""
        if self._is_settings_class(node):
            self.settings_classes.append({
                'name': node.name,
                'file': self.current_file,
                'line': node.lineno,
                'attributes': self._extract_class_attributes(node),
                'methods': self._extract_class_methods(node)
            })
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to find configuration patterns."""
        # Check for environment variable usage
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if (isinstance(node.value.func.value, ast.Name) and 
                    node.value.func.value.id == 'os' and 
                    node.value.func.attr in ['getenv', 'environ']):
                    self._process_env_var_usage(node)
        
        # Check for hardcoded configuration values
        if isinstance(node.value, (ast.Str, ast.Constant, ast.Num)):
            self._check_hardcoded_config(node)
        
        # Check for configuration dictionaries
        if isinstance(node.value, ast.Dict):
            self._analyze_config_dict(node)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to find configuration usage."""
        # Look for config loading patterns
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['load', 'read', 'get', 'set']:
                self._analyze_config_call(node)
        
        # Look for logging configuration
        if isinstance(node.func, ast.Name):
            if 'config' in node.func.id.lower() or 'setup' in node.func.id.lower():
                self.config_usage_patterns['function_calls'] += 1
        
        self.generic_visit(node)
    
    def _is_config_file(self, file_path: Path, content: str) -> bool:
        """Determine if file is a configuration file."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        config_patterns = [
            'config', 'settings', 'configuration', 'setup', 'env',
            'constants', 'defaults', 'params', 'options'
        ]
        
        for pattern in config_patterns:
            if pattern in filename:
                return True
        
        # Check file extensions
        if file_path.suffix.lower() in ['.ini', '.cfg', '.conf', '.yaml', '.yml', '.toml', '.json']:
            return True
        
        # Check content patterns
        content_patterns = [
            'class.*Config', 'class.*Settings', 'DATABASE_URL', 'SECRET_KEY',
            'API_KEY', 'DEBUG.*=', 'ALLOWED_HOSTS', 'DATABASES.*=',
            'os.environ', 'getenv', 'configuration', 'settings'
        ]
        
        for pattern in content_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _determine_config_type(self, file_path: Path, content: str) -> str:
        """Determine the type of configuration file."""
        filename = file_path.name.lower()
        
        if file_path.suffix.lower() in ['.ini', '.cfg', '.conf']:
            return 'ini_config'
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            return 'yaml_config'
        elif file_path.suffix.lower() == '.toml':
            return 'toml_config'
        elif file_path.suffix.lower() == '.json':
            return 'json_config'
        elif 'django' in content.lower() and 'settings' in filename:
            return 'django_settings'
        elif 'flask' in content.lower() and 'config' in filename:
            return 'flask_config'
        elif 'fastapi' in content.lower():
            return 'fastapi_config'
        elif re.search(r'class.*Config', content):
            return 'class_based_config'
        elif 'environ' in content or 'getenv' in content:
            return 'environment_config'
        else:
            return 'python_config'
    
    def _extract_config_patterns(self, content: str) -> List[str]:
        """Extract configuration patterns from content."""
        patterns = []
        
        # Environment variable patterns
        env_patterns = re.findall(r'os\.environ\[[\'"](.*?)[\'"]\]', content)
        patterns.extend([f"env_var:{var}" for var in env_patterns])
        
        # getenv patterns
        getenv_patterns = re.findall(r'os\.getenv\([\'"](.*?)[\'"]', content)
        patterns.extend([f"getenv:{var}" for var in getenv_patterns])
        
        # Database URL patterns
        if re.search(r'DATABASE_URL|DB_URL|database.*url', content, re.IGNORECASE):
            patterns.append('database_url')
        
        # Secret key patterns
        if re.search(r'SECRET_KEY|secret.*key|api.*key', content, re.IGNORECASE):
            patterns.append('secret_key')
        
        # Debug patterns
        if re.search(r'DEBUG.*=|debug.*mode', content, re.IGNORECASE):
            patterns.append('debug_config')
        
        return patterns
    
    def _is_settings_class(self, node: ast.ClassDef) -> bool:
        """Check if class is a settings/configuration class."""
        class_indicators = [
            'config' in node.name.lower(),
            'settings' in node.name.lower(),
            'configuration' in node.name.lower(),
            'options' in node.name.lower(),
            'params' in node.name.lower()
        ]
        
        return any(class_indicators)
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract attributes from a class definition."""
        attributes = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attr_info = {
                            'name': target.id,
                            'line': child.lineno,
                            'type': self._get_value_type(child.value),
                            'is_constant': target.id.isupper()
                        }
                        attributes.append(attr_info)
        
        return attributes
    
    def _extract_class_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract method names from a class definition."""
        methods = []
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(child.name)
        
        return methods
    
    def _process_env_var_usage(self, node: ast.Assign) -> None:
        """Process environment variable usage."""
        if isinstance(node.value, ast.Call) and node.value.args:
            if isinstance(node.value.args[0], (ast.Str, ast.Constant)):
                env_var = node.value.args[0].s if hasattr(node.value.args[0], 's') else node.value.args[0].value
                
                # Get variable name being assigned to
                var_name = None
                if node.targets and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                
                self.environment_variables[env_var].add(self.current_file)
                
                # Check for default values
                default_value = None
                if len(node.value.args) > 1:
                    default_arg = node.value.args[1]
                    if isinstance(default_arg, (ast.Str, ast.Constant)):
                        default_value = default_arg.s if hasattr(default_arg, 's') else default_arg.value
                
                self.config_patterns['environment_variables'].append({
                    'env_var': env_var,
                    'file': self.current_file,
                    'line': node.lineno,
                    'assigned_to': var_name,
                    'default_value': default_value
                })
    
    def _check_hardcoded_config(self, node: ast.Assign) -> None:
        """Check for hardcoded configuration values."""
        value = None
        if isinstance(node.value, ast.Str):
            value = node.value.s
        elif isinstance(node.value, ast.Constant):
            value = node.value.value
        elif isinstance(node.value, ast.Num):
            value = node.value.n
        
        if value is not None and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Check if this looks like a configuration value
                config_indicators = [
                    'url', 'host', 'port', 'key', 'secret', 'token',
                    'path', 'dir', 'file', 'timeout', 'limit', 'max',
                    'debug', 'verbose', 'config', 'setting'
                ]
                
                if any(indicator in var_name.lower() for indicator in config_indicators):
                    self.hardcoded_values.append({
                        'variable': var_name,
                        'value': str(value)[:100],  # Truncate long values
                        'file': self.current_file,
                        'line': node.lineno,
                        'type': type(value).__name__
                    })
    
    def _analyze_config_dict(self, node: ast.Assign) -> None:
        """Analyze configuration dictionaries."""
        if node.targets and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            
            if any(word in var_name.lower() for word in ['config', 'settings', 'options', 'params']):
                self.config_patterns['config_dictionaries'].append({
                    'variable': var_name,
                    'file': self.current_file,
                    'line': node.lineno,
                    'key_count': len(node.value.keys) if node.value.keys else 0
                })
    
    def _analyze_config_call(self, node: ast.Call) -> None:
        """Analyze configuration-related function calls."""
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            if hasattr(node.func, 'value') and isinstance(node.func.value, ast.Name):
                object_name = node.func.value.id
                
                self.config_usage_patterns[f'{object_name}.{method_name}'] += 1
    
    def _analyze_raw_content(self, content: str) -> None:
        """Analyze raw file content for configuration patterns."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for security concerns
            if re.search(r'password.*=.*[\'"][^\'"]*[\'"]', line_stripped, re.IGNORECASE):
                self.security_concerns.append({
                    'type': 'hardcoded_password',
                    'file': self.current_file,
                    'line': i + 1,
                    'content': line_stripped[:100]
                })
            
            if re.search(r'secret.*=.*[\'"][^\'"]*[\'"]', line_stripped, re.IGNORECASE):
                self.security_concerns.append({
                    'type': 'hardcoded_secret',
                    'file': self.current_file,
                    'line': i + 1,
                    'content': line_stripped[:100]
                })
            
            # Check for TODO/FIXME in config
            if re.search(r'(TODO|FIXME|HACK).*config', line_stripped, re.IGNORECASE):
                self.config_patterns['config_todos'].append({
                    'file': self.current_file,
                    'line': i + 1,
                    'content': line_stripped
                })
    
    def _get_value_type(self, node) -> str:
        """Get the type of a value node."""
        if isinstance(node, ast.Str):
            return 'string'
        elif isinstance(node, ast.Num):
            return 'number'
        elif isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Dict):
            return 'dict'
        elif isinstance(node, ast.Call):
            return 'function_call'
        else:
            return 'unknown'
    
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze all files in directory for configuration patterns."""
        results = {
            'files_analyzed': [],
            'total_files': 0,
            'successful_analyses': 0,
            'errors': []
        }
        
        # Analyze Python files
        python_files = list(root_path.rglob('*.py'))
        
        # Also analyze common config file types
        config_extensions = ['*.ini', '*.cfg', '*.conf', '*.yaml', '*.yml', '*.toml', '*.json']
        config_files = []
        for ext in config_extensions:
            config_files.extend(root_path.rglob(ext))
        
        all_files = python_files + config_files
        results['total_files'] = len(all_files)
        
        print(f"Analyzing {len(all_files)} files for configuration patterns...")
        print(f"  - Python files: {len(python_files)}")
        print(f"  - Config files: {len(config_files)}")
        
        for i, file_path in enumerate(all_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(all_files)} files analyzed")
            
            if file_path.suffix.lower() == '.py':
                analysis = self.analyze_file(file_path)
            else:
                analysis = self._analyze_non_python_config(file_path)
            
            results['files_analyzed'].append(analysis)
            
            if analysis.get('analyzed', False):
                results['successful_analyses'] += 1
            else:
                results['errors'].append(analysis)
        
        print(f"Analysis complete: {results['successful_analyses']}/{results['total_files']} files analyzed successfully")
        return results
    
    def _analyze_non_python_config(self, file_path: Path) -> Dict[str, Any]:
        """Analyze non-Python configuration files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            config_info = {
                'file': str(file_path),
                'type': self._determine_config_type(file_path, content),
                'patterns': self._extract_config_patterns(content),
                'analyzed': True,
                'is_config_file': True
            }
            
            self.config_files.append(config_info)
            return config_info
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'analyzed': False
            }
    
    def detect_consolidation_opportunities(self) -> Dict[str, Any]:
        """Detect configuration consolidation opportunities."""
        opportunities = {
            'duplicate_env_vars': self._find_duplicate_env_vars(),
            'scattered_configs': self._find_scattered_configs(),
            'hardcoded_consolidation': self._find_hardcoded_consolidation(),
            'config_file_consolidation': self._analyze_config_file_spread(),
            'security_improvements': self._identify_security_improvements()
        }
        
        return opportunities
    
    def _find_duplicate_env_vars(self) -> List[Dict[str, Any]]:
        """Find environment variables used in multiple files."""
        duplicates = []
        
        for env_var, files in self.environment_variables.items():
            if len(files) > 1:
                duplicates.append({
                    'env_var': env_var,
                    'file_count': len(files),
                    'files': list(files),
                    'consolidation_benefit': 'centralize_env_var_access'
                })
        
        return sorted(duplicates, key=lambda x: x['file_count'], reverse=True)[:20]
    
    def _find_scattered_configs(self) -> List[Dict[str, Any]]:
        """Find configuration scattered across multiple files."""
        config_groups = defaultdict(list)
        
        # Group by configuration patterns
        for pattern_type, patterns in self.config_patterns.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if isinstance(pattern, dict) and 'file' in pattern:
                        config_groups[pattern_type].append(pattern['file'])
        
        scattered = []
        for config_type, files in config_groups.items():
            unique_files = set(files)
            if len(unique_files) > 3:  # More than 3 files have this config type
                scattered.append({
                    'config_type': config_type,
                    'file_count': len(unique_files),
                    'total_instances': len(files),
                    'files': list(unique_files)[:10],  # Limit for readability
                    'consolidation_benefit': f'centralize_{config_type}'
                })
        
        return sorted(scattered, key=lambda x: x['file_count'], reverse=True)
    
    def _find_hardcoded_consolidation(self) -> List[Dict[str, Any]]:
        """Find hardcoded values that could be consolidated."""
        value_groups = defaultdict(list)
        
        # Group hardcoded values by their content
        for hardcoded in self.hardcoded_values:
            value_groups[hardcoded['value']].append(hardcoded)
        
        consolidation_candidates = []
        for value, instances in value_groups.items():
            if len(instances) > 1:
                consolidation_candidates.append({
                    'value': value,
                    'instance_count': len(instances),
                    'files': list(set(inst['file'] for inst in instances)),
                    'variables': [inst['variable'] for inst in instances],
                    'consolidation_benefit': 'extract_to_constant'
                })
        
        return sorted(consolidation_candidates, key=lambda x: x['instance_count'], reverse=True)[:15]
    
    def _analyze_config_file_spread(self) -> Dict[str, Any]:
        """Analyze spread of configuration across different file types."""
        config_by_type = defaultdict(list)
        
        for config_file in self.config_files:
            config_by_type[config_file['type']].append(config_file['file'])
        
        return {
            'config_file_types': dict(config_by_type),
            'total_config_files': len(self.config_files),
            'consolidation_recommendation': self._suggest_config_consolidation(config_by_type)
        }
    
    def _suggest_config_consolidation(self, config_by_type: Dict[str, List[str]]) -> List[str]:
        """Suggest configuration consolidation strategies."""
        suggestions = []
        
        if len(config_by_type) > 3:
            suggestions.append("Consider consolidating to fewer configuration file types")
        
        if 'environment_config' in config_by_type and len(config_by_type['environment_config']) > 1:
            suggestions.append("Centralize environment variable handling")
        
        if len([f for files in config_by_type.values() for f in files]) > 10:
            suggestions.append("High number of config files - consider configuration hierarchy")
        
        return suggestions
    
    def _identify_security_improvements(self) -> List[Dict[str, Any]]:
        """Identify security improvements for configuration."""
        improvements = []
        
        # Add security concerns
        improvements.extend(self.security_concerns)
        
        # Check for unencrypted secrets
        for hardcoded in self.hardcoded_values:
            if any(keyword in hardcoded['variable'].lower() 
                   for keyword in ['password', 'secret', 'key', 'token']):
                improvements.append({
                    'type': 'unencrypted_secret',
                    'file': hardcoded['file'],
                    'line': hardcoded.get('line', 0),
                    'variable': hardcoded['variable'],
                    'recommendation': 'Move to environment variable or secure vault'
                })
        
        return improvements
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive configuration analysis summary."""
        consolidation_opportunities = self.detect_consolidation_opportunities()
        
        summary = {
            'configuration_statistics': {
                'total_config_files': len(self.config_files),
                'settings_classes': len(self.settings_classes),
                'environment_variables': len(self.environment_variables),
                'hardcoded_values': len(self.hardcoded_values),
                'config_patterns': {k: len(v) if isinstance(v, list) else v 
                                   for k, v in self.config_patterns.items()},
                'config_usage_patterns': dict(self.config_usage_patterns)
            },
            'security_analysis': {
                'security_concerns': len(self.security_concerns),
                'hardcoded_secrets': len([c for c in self.security_concerns 
                                        if c['type'] in ['hardcoded_password', 'hardcoded_secret']]),
                'unencrypted_config_values': len([h for h in self.hardcoded_values 
                                                if any(kw in h['variable'].lower() 
                                                      for kw in ['password', 'secret', 'key'])])
            },
            'consolidation_analysis': {
                'duplicate_env_vars': len(consolidation_opportunities['duplicate_env_vars']),
                'scattered_configs': len(consolidation_opportunities['scattered_configs']),
                'hardcoded_consolidation': len(consolidation_opportunities['hardcoded_consolidation']),
                'config_file_spread': consolidation_opportunities['config_file_consolidation']['total_config_files']
            },
            'consolidation_opportunities': consolidation_opportunities,
            'configuration_health_score': self._calculate_config_health_score(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _calculate_config_health_score(self) -> float:
        """Calculate overall configuration health score (0-100)."""
        score = 100.0
        
        # Penalize security concerns
        security_penalty = min(len(self.security_concerns) * 10, 40)
        score -= security_penalty
        
        # Penalize scattered configuration
        if len(self.config_files) > 15:
            score -= min((len(self.config_files) - 15) * 2, 20)
        
        # Penalize hardcoded values
        hardcoded_penalty = min(len(self.hardcoded_values) * 0.5, 20)
        score -= hardcoded_penalty
        
        # Bonus for environment variable usage
        if len(self.environment_variables) > 0:
            score += min(len(self.environment_variables) * 0.5, 10)
        
        # Bonus for settings classes
        if len(self.settings_classes) > 0:
            score += min(len(self.settings_classes) * 2, 10)
        
        return max(score, 0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate configuration improvement recommendations."""
        recommendations = []
        
        if len(self.security_concerns) > 0:
            recommendations.append(f"Address {len(self.security_concerns)} security concerns in configuration")
        
        if len(self.hardcoded_values) > 10:
            recommendations.append("Consider extracting hardcoded values to configuration files")
        
        if len(self.environment_variables) < 5:
            recommendations.append("Consider using more environment variables for configuration")
        
        if len(self.config_files) > 10:
            recommendations.append("Consider consolidating configuration files")
        
        if len(self.settings_classes) == 0:
            recommendations.append("Consider creating settings classes for better organization")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Configuration & Settings Analysis Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 41-43: Configuration & Settings Analysis ===")
    print(f"Analyzing directory: {args.root}")
    
    analyzer = ConfigurationSettingsAnalyzer()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = analyzer.analyze_directory(root_path)
    
    # Generate summary
    summary = analyzer.generate_summary()
    
    # Combine results
    final_results = {
        'analysis_metadata': {
            'tool': 'configuration_settings_analyzer',
            'version': '1.0',
            'agent': 'Agent_C',
            'hours': '41-43',
            'phase': 'Utility_Component_Extraction'
        },
        'analysis_results': analysis_results,
        'summary': summary,
        'raw_data': {
            'config_patterns': {k: list(v) if isinstance(v, list) else v 
                               for k, v in analyzer.config_patterns.items()},
            'environment_variables': {k: list(v) for k, v in analyzer.environment_variables.items()},
            'config_files': analyzer.config_files,
            'settings_classes': analyzer.settings_classes,
            'hardcoded_values': analyzer.hardcoded_values,
            'security_concerns': analyzer.security_concerns
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== CONFIGURATION & SETTINGS ANALYSIS COMPLETE ===")
    print(f"Files analyzed: {analysis_results['successful_analyses']}/{analysis_results['total_files']}")
    print(f"Configuration files found: {summary['configuration_statistics']['total_config_files']}")
    print(f"Environment variables: {summary['configuration_statistics']['environment_variables']}")
    print(f"Settings classes: {summary['configuration_statistics']['settings_classes']}")
    print(f"Security concerns: {summary['security_analysis']['security_concerns']}")
    print(f"Configuration health score: {summary['configuration_health_score']:.1f}/100")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()