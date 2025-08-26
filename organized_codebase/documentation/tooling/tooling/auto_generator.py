"""
Automated Documentation Generator

AI-powered documentation generation that analyzes code structure,
extracts meaningful patterns, and generates comprehensive documentation.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocSection:
    """Represents a documentation section."""
    title: str
    content: str
    code_refs: List[str]
    importance: str  # high, medium, low
    
    
@dataclass 
class ModuleDoc:
    """Complete module documentation."""
    module_path: str
    summary: str
    detailed_description: str
    functions: Dict[str, str]
    classes: Dict[str, str]
    examples: List[str]
    security_notes: List[str]
    performance_notes: List[str]
    

class DocumentationAutoGenerator:
    """
    Automated documentation generator using AI and code analysis.
    Generates comprehensive documentation with minimal human intervention.
    """
    
    def __init__(self):
        """Initialize the auto generator."""
        self.generated_count = 0
        self.cache = {}
        logger.info("Documentation Auto Generator initialized")
        
    def analyze_module(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module for documentation generation.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Analysis results with structure and patterns
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            analysis = {
                'module': file_path,
                'functions': [],
                'classes': [],
                'imports': [],
                'constants': [],
                'complexity': 0,
                'lines': len(source.splitlines()),
                'has_tests': 'test' in file_path.lower()
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'returns': ast.unparse(node.returns) if node.returns else None,
                        'docstring': ast.get_docstring(node),
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'complexity': self._calculate_complexity(node)
                    }
                    analysis['functions'].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [ast.unparse(b) for b in node.bases],
                        'methods': [],
                        'docstring': ast.get_docstring(node),
                        'decorators': [ast.unparse(d) for d in node.decorator_list]
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                            
                    analysis['classes'].append(class_info)
                    
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(f"{node.module}.{node.names[0].name}")
                    
            analysis['complexity'] = sum(f['complexity'] for f in analysis['functions'])
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {}
            
    def generate_documentation(self, file_path: str) -> ModuleDoc:
        """
        Generate complete documentation for a module.
        
        Args:
            file_path: Path to module
            
        Returns:
            Complete module documentation
        """
        analysis = self.analyze_module(file_path)
        
        if not analysis:
            return None
            
        doc = ModuleDoc(
            module_path=file_path,
            summary=self._generate_summary(analysis),
            detailed_description=self._generate_description(analysis),
            functions=self._document_functions(analysis['functions']),
            classes=self._document_classes(analysis['classes']),
            examples=self._generate_examples(analysis),
            security_notes=self._analyze_security(analysis),
            performance_notes=self._analyze_performance(analysis)
        )
        
        self.generated_count += 1
        self.cache[file_path] = doc
        
        return doc
        
    def generate_api_docs(self, module_path: str) -> str:
        """
        Generate API documentation in Markdown format.
        
        Args:
            module_path: Path to module or package
            
        Returns:
            Markdown formatted API documentation
        """
        doc = self.generate_documentation(module_path)
        
        if not doc:
            return ""
            
        md = [
            f"# API Documentation: {Path(module_path).stem}",
            "",
            f"## Summary",
            doc.summary,
            "",
            f"## Description", 
            doc.detailed_description,
            ""
        ]
        
        if doc.functions:
            md.extend(["## Functions", ""])
            for name, desc in doc.functions.items():
                md.extend([f"### {name}", desc, ""])
                
        if doc.classes:
            md.extend(["## Classes", ""])
            for name, desc in doc.classes.items():
                md.extend([f"### {name}", desc, ""])
                
        if doc.examples:
            md.extend(["## Examples", ""])
            for example in doc.examples:
                md.extend(["```python", example, "```", ""])
                
        if doc.security_notes:
            md.extend(["## Security Considerations", ""])
            for note in doc.security_notes:
                md.append(f"- {note}")
                
        if doc.performance_notes:
            md.extend(["", "## Performance Notes", ""])
            for note in doc.performance_notes:
                md.append(f"- {note}")
                
        return "\n".join(md)
        
    def batch_generate(self, directory: str, pattern: str = "*.py") -> Dict[str, ModuleDoc]:
        """
        Generate documentation for multiple files.
        
        Args:
            directory: Directory to scan
            pattern: File pattern to match
            
        Returns:
            Dict of file paths to documentation
        """
        results = {}
        path = Path(directory)
        
        for file_path in path.rglob(pattern):
            if not any(skip in str(file_path) for skip in ['__pycache__', '.git', 'test']):
                doc = self.generate_documentation(str(file_path))
                if doc:
                    results[str(file_path)] = doc
                    
        logger.info(f"Generated documentation for {len(results)} files")
        return results
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get documentation generation metrics."""
        return {
            'total_generated': self.generated_count,
            'cached_docs': len(self.cache),
            'timestamp': datetime.now().isoformat()
        }
        
    # Helper methods
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
        return complexity
        
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate module summary."""
        return f"Module with {len(analysis['functions'])} functions and {len(analysis['classes'])} classes"
        
    def _generate_description(self, analysis: Dict) -> str:
        """Generate detailed description."""
        return f"This module contains {analysis['lines']} lines with complexity score of {analysis['complexity']}"
        
    def _document_functions(self, functions: List[Dict]) -> Dict[str, str]:
        """Document all functions."""
        docs = {}
        for func in functions:
            docs[func['name']] = func.get('docstring', 'No documentation available')
        return docs
        
    def _document_classes(self, classes: List[Dict]) -> Dict[str, str]:
        """Document all classes."""
        docs = {}
        for cls in classes:
            docs[cls['name']] = cls.get('docstring', 'No documentation available')
        return docs
        
    def _generate_examples(self, analysis: Dict) -> List[str]:
        """Generate usage examples."""
        examples = []
        for func in analysis['functions'][:3]:  # Top 3 functions
            examples.append(f"{func['name']}({', '.join(func['args'])})")
        return examples
        
    def _analyze_security(self, analysis: Dict) -> List[str]:
        """Analyze security considerations."""
        notes = []
        for func in analysis['functions']:
            if any(sec in func['name'].lower() for sec in ['password', 'token', 'auth']):
                notes.append(f"Function {func['name']} handles sensitive data")
        return notes
        
    def _analyze_performance(self, analysis: Dict) -> List[str]:
        """Analyze performance considerations."""
        notes = []
        if analysis['complexity'] > 10:
            notes.append(f"High complexity score: {analysis['complexity']}")
        if analysis['lines'] > 500:
            notes.append(f"Large module: {analysis['lines']} lines")
        return notes