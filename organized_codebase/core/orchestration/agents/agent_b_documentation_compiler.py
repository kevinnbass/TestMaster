#!/usr/bin/env python3
"""
Agent B Documentation Compiler - Hours 81-85
Comprehensive documentation compilation system for TestMaster framework
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path
import ast
import re

class DocumentationCompiler:
    def __init__(self):
        self.base_path = Path(".")
        self.documentation = {
            'modules': {},
            'functions': {},
            'classes': {},
            'patterns': {},
            'examples': {},
            'guides': {},
            'metadata': {
                'compiled_at': datetime.now().isoformat(),
                'agent': 'Agent_B',
                'hours': '81-85',
                'phase': 'Documentation_Compilation'
            }
        }
        self.analysis_results = []
        
    def load_existing_analysis(self):
        """Load all existing Agent B analysis results"""
        print("Loading existing Agent B analysis results...")
        
        # Load hour-based analysis files
        hour_files = list(glob.glob("*hour*.json"))
        for file in hour_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.analysis_results.append({
                        'file': file,
                        'data': data,
                        'type': 'hourly_analysis'
                    })
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        # Load GRAPH.json if available
        if os.path.exists("GRAPH.json"):
            try:
                with open("GRAPH.json", 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    self.analysis_results.append({
                        'file': 'GRAPH.json',
                        'data': graph_data,
                        'type': 'neo4j_graph'
                    })
            except Exception as e:
                print(f"⚠️  Could not load GRAPH.json: {e}")
        
        # Load Agent B specific files
        agent_b_files = glob.glob("*agent_b*.py") + glob.glob("*Agent_B*.md")
        for file in agent_b_files:
            try:
                if file.endswith('.py'):
                    self.analysis_results.append({
                        'file': file,
                        'type': 'python_analysis',
                        'content': open(file, 'r', encoding='utf-8').read()
                    })
                elif file.endswith('.md'):
                    self.analysis_results.append({
                        'file': file,
                        'type': 'markdown_report',
                        'content': open(file, 'r', encoding='utf-8').read()
                    })
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        print(f"Loaded {len(self.analysis_results)} analysis result files")
        return self.analysis_results
    
    def compile_module_documentation(self):
        """Compile comprehensive module documentation"""
        print("Compiling module documentation...")
        
        # Find all Python modules in the codebase
        python_files = list(glob.glob("**/*.py", recursive=True))
        
        for file_path in python_files:
            if self._should_process_file(file_path):
                try:
                    module_doc = self._extract_module_documentation(file_path)
                    if module_doc:
                        self.documentation['modules'][file_path] = module_doc
                except Exception as e:
                    print(f"Warning: Error processing {file_path}: {e}")
        
        print(f"Compiled documentation for {len(self.documentation['modules'])} modules")
        
    def _should_process_file(self, file_path):
        """Determine if file should be processed"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache',
            'venv',
            'env'
        ]
        
        for pattern in skip_patterns:
            if pattern in file_path:
                return False
        return True
    
    def _extract_module_documentation(self, file_path):
        """Extract documentation from a Python module"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to extract docstrings and structure
            tree = ast.parse(content)
            
            module_doc = {
                'path': file_path,
                'overview': ast.get_docstring(tree) or "No module docstring",
                'functions': [],
                'classes': [],
                'imports': [],
                'exports': [],
                'line_count': len(content.splitlines()),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_doc = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or "No docstring",
                        'line_number': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'is_private': node.name.startswith('_'),
                        'complexity_estimate': self._estimate_complexity(node)
                    }
                    module_doc['functions'].append(func_doc)
                    
                elif isinstance(node, ast.ClassDef):
                    class_doc = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or "No docstring",
                        'line_number': node.lineno,
                        'methods': [],
                        'is_private': node.name.startswith('_'),
                        'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_doc = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item) or "No docstring",
                                'line_number': item.lineno,
                                'is_private': item.name.startswith('_'),
                                'is_static': any(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in item.decorator_list),
                                'is_class_method': any(isinstance(d, ast.Name) and d.id == 'classmethod' for d in item.decorator_list)
                            }
                            class_doc['methods'].append(method_doc)
                    
                    module_doc['classes'].append(class_doc)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_doc['imports'].append({
                            'module': alias.name,
                            'alias': alias.asname,
                            'type': 'import'
                        })
                        
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        module_doc['imports'].append({
                            'module': node.module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'type': 'from_import'
                        })
            
            return module_doc
            
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
            return None
    
    def _estimate_complexity(self, node):
        """Estimate function complexity based on AST structure"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def compile_function_documentation(self):
        """Generate comprehensive function documentation"""
        print("Compiling function documentation...")
        
        function_count = 0
        for module_path, module_data in self.documentation['modules'].items():
            for func in module_data['functions']:
                func_key = f"{module_path}::{func['name']}"
                self.documentation['functions'][func_key] = {
                    'module': module_path,
                    'name': func['name'],
                    'docstring': func['docstring'],
                    'signature': self._generate_function_signature(func),
                    'complexity': func['complexity_estimate'],
                    'usage_example': self._generate_usage_example(func),
                    'line_number': func['line_number'],
                    'is_private': func['is_private']
                }
                function_count += 1
        
        print(f"Compiled documentation for {function_count} functions")
    
    def _generate_function_signature(self, func):
        """Generate function signature string"""
        args = ', '.join(func['args'])
        return f"{func['name']}({args})"
    
    def _generate_usage_example(self, func):
        """Generate basic usage example for function"""
        if func['name'].startswith('test_'):
            return f"# Test function - run with pytest\npytest -v -k {func['name']}"
        elif func['name'] == '__init__':
            return f"# Constructor method"
        else:
            args = ', '.join([f"<{arg}>" for arg in func['args'] if arg != 'self'])
            return f"# Example usage:\nresult = {func['name']}({args})"
    
    def compile_class_documentation(self):
        """Generate comprehensive class documentation"""
        print("Compiling class documentation...")
        
        class_count = 0
        for module_path, module_data in self.documentation['modules'].items():
            for cls in module_data['classes']:
                cls_key = f"{module_path}::{cls['name']}"
                self.documentation['classes'][cls_key] = {
                    'module': module_path,
                    'name': cls['name'],
                    'docstring': cls['docstring'],
                    'methods': cls['methods'],
                    'inheritance': cls['bases'],
                    'method_count': len(cls['methods']),
                    'public_methods': [m for m in cls['methods'] if not m['is_private']],
                    'private_methods': [m for m in cls['methods'] if m['is_private']],
                    'usage_example': self._generate_class_usage_example(cls),
                    'line_number': cls['line_number']
                }
                class_count += 1
        
        print(f"Compiled documentation for {class_count} classes")
    
    def _generate_class_usage_example(self, cls):
        """Generate basic usage example for class"""
        return f"""# Example usage:
{cls['name'].lower()} = {cls['name']}()
# Use methods: {', '.join([m['name'] for m in cls['methods'][:3] if not m['is_private']])}"""
    
    def compile_pattern_documentation(self):
        """Compile design pattern documentation from analysis"""
        print("Compiling pattern documentation...")
        
        # Look for pattern analysis in loaded results
        pattern_count = 0
        for result in self.analysis_results:
            if 'pattern' in str(result.get('file', '')).lower():
                try:
                    data = result['data']
                    if 'patterns' in data:
                        for pattern_name, pattern_info in data['patterns'].items():
                            self.documentation['patterns'][pattern_name] = {
                                'name': pattern_name,
                                'description': pattern_info.get('description', 'No description'),
                                'implementation': pattern_info.get('implementation', {}),
                                'examples': pattern_info.get('examples', []),
                                'benefits': pattern_info.get('benefits', []),
                                'usage_scenarios': pattern_info.get('usage_scenarios', [])
                            }
                            pattern_count += 1
                except Exception as e:
                    print(f"Warning: Error processing pattern data: {e}")
        
        # Add common patterns found in codebase
        common_patterns = self._identify_common_patterns()
        for pattern_name, pattern_info in common_patterns.items():
            if pattern_name not in self.documentation['patterns']:
                self.documentation['patterns'][pattern_name] = pattern_info
                pattern_count += 1
        
        print(f"Compiled documentation for {pattern_count} patterns")
    
    def _identify_common_patterns(self):
        """Identify common design patterns in the codebase"""
        patterns = {}
        
        # Look for common patterns based on class and function names
        for module_path, module_data in self.documentation['modules'].items():
            for cls in module_data['classes']:
                name = cls['name'].lower()
                
                if 'factory' in name:
                    patterns['Factory Pattern'] = {
                        'name': 'Factory Pattern',
                        'description': 'Creates objects without specifying exact classes',
                        'implementation': {'found_in': module_path, 'class': cls['name']},
                        'examples': [f"Use {cls['name']} to create objects dynamically"]
                    }
                
                elif 'builder' in name:
                    patterns['Builder Pattern'] = {
                        'name': 'Builder Pattern', 
                        'description': 'Constructs complex objects step by step',
                        'implementation': {'found_in': module_path, 'class': cls['name']},
                        'examples': [f"Use {cls['name']} to build complex objects"]
                    }
                
                elif 'manager' in name or 'handler' in name:
                    patterns['Manager Pattern'] = {
                        'name': 'Manager Pattern',
                        'description': 'Manages lifecycle and coordination of related objects',
                        'implementation': {'found_in': module_path, 'class': cls['name']},
                        'examples': [f"Use {cls['name']} to coordinate system components"]
                    }
        
        return patterns
    
    def generate_documentation_toc(self):
        """Generate table of contents for documentation"""
        print("Generating documentation table of contents...")
        
        toc = {
            'sections': [
                {
                    'title': 'Module Documentation',
                    'count': len(self.documentation['modules']),
                    'items': list(self.documentation['modules'].keys())[:10]  # First 10 for TOC
                },
                {
                    'title': 'Function Reference',
                    'count': len(self.documentation['functions']),
                    'items': list(self.documentation['functions'].keys())[:10]
                },
                {
                    'title': 'Class Reference', 
                    'count': len(self.documentation['classes']),
                    'items': list(self.documentation['classes'].keys())[:10]
                },
                {
                    'title': 'Design Patterns',
                    'count': len(self.documentation['patterns']),
                    'items': list(self.documentation['patterns'].keys())
                }
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        return toc
    
    def create_search_index(self):
        """Create searchable index of documentation"""
        print("Creating searchable documentation index...")
        
        search_index = {
            'modules': {},
            'functions': {},
            'classes': {},
            'keywords': set()
        }
        
        # Index modules
        for path, module in self.documentation['modules'].items():
            search_terms = [
                module['overview'].lower(),
                path.lower(),
                ' '.join([f['name'] for f in module['functions']]),
                ' '.join([c['name'] for c in module['classes']])
            ]
            search_index['modules'][path] = ' '.join(search_terms)
            
        # Index functions
        for func_key, func in self.documentation['functions'].items():
            search_terms = [
                func['name'].lower(),
                func['docstring'].lower(),
                func['signature'].lower()
            ]
            search_index['functions'][func_key] = ' '.join(search_terms)
            
        # Index classes
        for cls_key, cls in self.documentation['classes'].items():
            search_terms = [
                cls['name'].lower(),
                cls['docstring'].lower(),
                ' '.join([m['name'] for m in cls['methods']])
            ]
            search_index['classes'][cls_key] = ' '.join(search_terms)
        
        # Extract keywords
        all_text = ' '.join([
            ' '.join(search_index['modules'].values()),
            ' '.join(search_index['functions'].values()),
            ' '.join(search_index['classes'].values())
        ])
        
        # Simple keyword extraction (words longer than 3 chars, appearing multiple times)
        words = re.findall(r'\b\w{4,}\b', all_text.lower())
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        search_index['keywords'] = [word for word, count in word_counts.items() if count > 2][:100]
        
        return search_index
    
    def compile_complete_documentation(self):
        """Main compilation method - orchestrates entire process"""
        print("Starting comprehensive documentation compilation...")
        
        # Load existing analysis
        self.load_existing_analysis()
        
        # Compile all documentation types
        self.compile_module_documentation()
        self.compile_function_documentation()
        self.compile_class_documentation()
        self.compile_pattern_documentation()
        
        # Generate supporting materials
        toc = self.generate_documentation_toc()
        search_index = self.create_search_index()
        
        # Add metadata
        self.documentation['toc'] = toc
        self.documentation['search_index'] = search_index
        self.documentation['stats'] = {
            'total_modules': len(self.documentation['modules']),
            'total_functions': len(self.documentation['functions']),
            'total_classes': len(self.documentation['classes']),
            'total_patterns': len(self.documentation['patterns']),
            'analysis_files_processed': len(self.analysis_results)
        }
        
        return self.documentation
    
    def save_documentation(self, filename="agent_b_compiled_documentation.json"):
        """Save compiled documentation to file"""
        print(f"Saving compiled documentation to {filename}...")
        
        # Convert sets to lists for JSON serialization
        if 'search_index' in self.documentation and 'keywords' in self.documentation['search_index']:
            self.documentation['search_index']['keywords'] = list(self.documentation['search_index']['keywords'])
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.documentation, f, indent=2, ensure_ascii=False)
        
        print(f"Documentation saved to {filename}")
        return filename

def main():
    """Main execution function"""
    print("Agent B Hours 81-85: Documentation Compilation Starting...")
    
    compiler = DocumentationCompiler()
    
    try:
        # Compile all documentation
        documentation = compiler.compile_complete_documentation()
        
        # Save results
        output_file = compiler.save_documentation()
        
        # Print summary
        stats = documentation['stats']
        print(f"""
DOCUMENTATION COMPILATION COMPLETE:
   Modules documented: {stats['total_modules']}
   Functions documented: {stats['total_functions']}  
   Classes documented: {stats['total_classes']}
   Patterns documented: {stats['total_patterns']}
   Analysis files processed: {stats['analysis_files_processed']}
   Output file: {output_file}
        """)
        
        return True
        
    except Exception as e:
        print(f"Error during documentation compilation: {e}")
        return False

if __name__ == "__main__":
    main()