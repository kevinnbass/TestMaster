"""
AI-Powered Documentation Generator
=================================

Revolutionary documentation generation system with quality assessment.
Extracted from predictive_code_intelligence.py for enterprise modular architecture.

Agent D Implementation - Hour 16-17: Predictive Intelligence Modularization
"""

import ast
import re
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .data_models import GeneratedDocumentation, DocumentationType


class DocumentationGenerator:
    """
    Revolutionary AI-Powered Documentation Generator
    
    Generates comprehensive, high-quality documentation for code elements
    with intelligent quality assessment and multiple format support.
    """
    
    def __init__(self):
        self.docstring_templates = {
            'function': '''"""
{summary}

{description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Example:
{example}
"""''',
            'class': '''"""
{summary}

{description}

Attributes:
{attributes}

Methods:
{methods}

Example:
{example}
"""''',
            'module': '''"""
{summary}

{description}

Classes:
{classes}

Functions:
{functions}

Usage:
{usage}
"""'''
        }
        
        self.quality_weights = {
            'completeness': 0.3,
            'clarity': 0.25,
            'technical_accuracy': 0.2,
            'style_consistency': 0.15,
            'example_quality': 0.1
        }
        
        self.style_patterns = {
            'google': {
                'args_format': 'Args:\n    param_name (type): Description',
                'returns_format': 'Returns:\n    type: Description',
                'raises_format': 'Raises:\n    ExceptionType: Description'
            },
            'numpy': {
                'args_format': 'Parameters\n----------\nparam_name : type\n    Description',
                'returns_format': 'Returns\n-------\ntype\n    Description',
                'raises_format': 'Raises\n------\nExceptionType\n    Description'
            },
            'sphinx': {
                'args_format': ':param param_name: Description\n:type param_name: type',
                'returns_format': ':returns: Description\n:rtype: type',
                'raises_format': ':raises ExceptionType: Description'
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_documentation(self, code: str, doc_type: DocumentationType,
                             target_element: str = "", style: str = "google") -> GeneratedDocumentation:
        """Generate documentation for code element with quality assessment"""
        
        try:
            doc = GeneratedDocumentation(
                documentation_type=doc_type,
                target_element=target_element
            )
            
            # Parse code
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                doc.generated_content = f"Unable to generate documentation due to syntax errors: {str(e)}"
                doc.documentation_quality = 0.0
                return doc
            
            # Generate based on type
            if doc_type == DocumentationType.FUNCTION_DOCSTRING:
                doc.generated_content = self._generate_function_docstring(tree, target_element, style)
            elif doc_type == DocumentationType.CLASS_DOCSTRING:
                doc.generated_content = self._generate_class_docstring(tree, target_element, style)
            elif doc_type == DocumentationType.MODULE_DOCSTRING:
                doc.generated_content = self._generate_module_docstring(tree, code, style)
            elif doc_type == DocumentationType.API_DOCUMENTATION:
                doc.generated_content = self._generate_api_documentation(tree, code, style)
            elif doc_type == DocumentationType.INLINE_COMMENTS:
                doc.generated_content = self._generate_inline_comments(code)
            elif doc_type == DocumentationType.README_SECTION:
                doc.generated_content = self._generate_readme_section(tree, code)
            elif doc_type == DocumentationType.ARCHITECTURE_DOCUMENTATION:
                doc.generated_content = self._generate_architecture_documentation(tree, code)
            else:
                doc.generated_content = self._generate_general_documentation(tree, code)
            
            # Assess documentation quality
            self._assess_documentation_quality(doc)
            
            # Set generation metadata
            doc.generation_metadata = {
                'style': style,
                'generation_timestamp': str(ast.literal_eval("__import__('datetime').datetime.now()")),
                'code_lines': len(code.split('\n')),
                'ast_nodes': len(list(ast.walk(tree)))
            }
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Error generating documentation: {e}")
            return GeneratedDocumentation(
                documentation_type=doc_type,
                generated_content=f"Documentation generation error: {str(e)}",
                documentation_quality=0.0
            )
    
    def _generate_function_docstring(self, tree: ast.AST, function_name: str, style: str) -> str:
        """Generate docstring for specific function"""
        
        try:
            # Find the function
            target_function = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_function = node
                    break
            
            if not target_function:
                return f"Function '{function_name}' not found."
            
            # Extract function information
            summary = self._generate_function_summary(target_function)
            description = self._generate_function_description(target_function)
            args = self._generate_args_documentation(target_function, style)
            returns = self._generate_returns_documentation(target_function, style)
            raises = self._generate_raises_documentation(target_function, style)
            example = self._generate_function_example(target_function)
            
            # Fill template
            docstring = self.docstring_templates['function'].format(
                summary=summary,
                description=description,
                args=args,
                returns=returns,
                raises=raises,
                example=example
            )
            
            return docstring.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating function docstring: {e}")
            return f"Error generating docstring for function '{function_name}'"
    
    def _generate_function_summary(self, node: ast.FunctionDef) -> str:
        """Generate concise summary for function"""
        
        try:
            name = node.name
            
            # Generate summary based on function name patterns
            if name.startswith('get_'):
                return f"Retrieve {name[4:].replace('_', ' ')}"
            elif name.startswith('set_'):
                return f"Set {name[4:].replace('_', ' ')}"
            elif name.startswith('create_'):
                return f"Create {name[7:].replace('_', ' ')}"
            elif name.startswith('delete_'):
                return f"Delete {name[7:].replace('_', ' ')}"
            elif name.startswith('validate_'):
                return f"Validate {name[9:].replace('_', ' ')}"
            elif name.startswith('process_'):
                return f"Process {name[8:].replace('_', ' ')}"
            elif name.startswith('calculate_'):
                return f"Calculate {name[10:].replace('_', ' ')}"
            elif name.startswith('analyze_'):
                return f"Analyze {name[8:].replace('_', ' ')}"
            elif name.startswith('generate_'):
                return f"Generate {name[9:].replace('_', ' ')}"
            elif name.startswith('parse_'):
                return f"Parse {name[6:].replace('_', ' ')}"
            elif name.startswith('format_'):
                return f"Format {name[7:].replace('_', ' ')}"
            elif name.startswith('convert_'):
                return f"Convert {name[8:].replace('_', ' ')}"
            elif name.startswith('transform_'):
                return f"Transform {name[10:].replace('_', ' ')}"
            elif name.startswith('filter_'):
                return f"Filter {name[7:].replace('_', ' ')}"
            elif name.startswith('sort_'):
                return f"Sort {name[5:].replace('_', ' ')}"
            elif name.startswith('search_'):
                return f"Search {name[7:].replace('_', ' ')}"
            elif name.startswith('find_'):
                return f"Find {name[5:].replace('_', ' ')}"
            elif name.startswith('load_'):
                return f"Load {name[5:].replace('_', ' ')}"
            elif name.startswith('save_'):
                return f"Save {name[5:].replace('_', ' ')}"
            elif name.startswith('update_'):
                return f"Update {name[7:].replace('_', ' ')}"
            elif name.startswith('remove_'):
                return f"Remove {name[7:].replace('_', ' ')}"
            elif name.startswith('add_'):
                return f"Add {name[4:].replace('_', ' ')}"
            elif name.startswith('check_'):
                return f"Check {name[6:].replace('_', ' ')}"
            elif name.startswith('verify_'):
                return f"Verify {name[7:].replace('_', ' ')}"
            elif name.startswith('build_'):
                return f"Build {name[6:].replace('_', ' ')}"
            elif name.startswith('render_'):
                return f"Render {name[7:].replace('_', ' ')}"
            elif name.startswith('execute_'):
                return f"Execute {name[8:].replace('_', ' ')}"
            elif name.startswith('run_'):
                return f"Run {name[4:].replace('_', ' ')}"
            elif name.startswith('start_'):
                return f"Start {name[6:].replace('_', ' ')}"
            elif name.startswith('stop_'):
                return f"Stop {name[5:].replace('_', ' ')}"
            elif name.startswith('init'):
                return f"Initialize {name[4:].replace('_', ' ')}" if len(name) > 4 else "Initialize the object"
            elif name.startswith('setup_'):
                return f"Set up {name[6:].replace('_', ' ')}"
            elif name.startswith('cleanup_'):
                return f"Clean up {name[8:].replace('_', ' ')}"
            elif name.startswith('reset_'):
                return f"Reset {name[6:].replace('_', ' ')}"
            elif name.startswith('clear_'):
                return f"Clear {name[6:].replace('_', ' ')}"
            else:
                # Generic based on verb patterns
                if any(verb in name.lower() for verb in ['handle', 'manage', 'control']):
                    return f"Handle {name.replace('_', ' ').replace('handle', '').strip()}"
                else:
                    return f"Execute {name.replace('_', ' ')} operation"
                
        except Exception as e:
            self.logger.error(f"Error generating function summary: {e}")
            return f"Function {node.name}"
    
    def _generate_function_description(self, node: ast.FunctionDef) -> str:
        """Generate detailed description for function"""
        
        try:
            # Analyze function body for more context
            has_loops = any(isinstance(child, (ast.For, ast.While)) for child in ast.walk(node))
            has_conditionals = any(isinstance(child, ast.If) for child in ast.walk(node))
            has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
            has_async = any(isinstance(child, (ast.AsyncWith, ast.Await)) for child in ast.walk(node))
            
            # Count different types of operations
            assignments = len([n for n in ast.walk(node) if isinstance(n, ast.Assign)])
            function_calls = len([n for n in ast.walk(node) if isinstance(n, ast.Call)])
            
            description_parts = []
            
            # Add behavioral descriptions
            if has_async:
                description_parts.append("performs asynchronous operations")
            if has_loops:
                description_parts.append("iterates through data structures")
            if has_conditionals:
                description_parts.append("includes conditional logic for different scenarios")
            if has_try_except:
                description_parts.append("handles potential exceptions gracefully")
            
            # Add complexity descriptions
            if assignments > 5:
                description_parts.append("performs multiple data transformations")
            if function_calls > 10:
                description_parts.append("coordinates with multiple other functions")
            
            # Analyze docstring if present
            existing_docstring = ast.get_docstring(node)
            if existing_docstring:
                # Extract meaningful parts from existing docstring
                sentences = existing_docstring.split('.')
                if len(sentences) > 1:
                    description_parts.append(f"implements {sentences[0].lower().strip()}")
            
            if description_parts:
                return f"This function {', '.join(description_parts)}."
            else:
                return "This function performs the specified operation with appropriate error handling."
                
        except Exception as e:
            self.logger.error(f"Error generating function description: {e}")
            return "Detailed description not available."
    
    def _generate_args_documentation(self, node: ast.FunctionDef, style: str) -> str:
        """Generate arguments documentation in specified style"""
        
        try:
            if not node.args.args:
                return "    None"
            
            args_doc = []
            style_config = self.style_patterns.get(style, self.style_patterns['google'])
            
            for arg in node.args.args:
                if arg.arg == 'self':
                    continue
                
                # Infer type and description from name
                arg_type = self._infer_argument_type(arg.arg)
                arg_desc = self._infer_argument_description(arg.arg)
                
                if style == 'google':
                    args_doc.append(f"    {arg.arg} ({arg_type}): {arg_desc}")
                elif style == 'numpy':
                    args_doc.append(f"{arg.arg} : {arg_type}\n    {arg_desc}")
                elif style == 'sphinx':
                    args_doc.append(f":param {arg.arg}: {arg_desc}")
                    args_doc.append(f":type {arg.arg}: {arg_type}")
                else:
                    args_doc.append(f"    {arg.arg} ({arg_type}): {arg_desc}")
            
            return "\n".join(args_doc) if args_doc else "    None"
            
        except Exception as e:
            self.logger.error(f"Error generating args documentation: {e}")
            return "    Arguments documentation not available."
    
    def _infer_argument_type(self, arg_name: str) -> str:
        """Infer argument type from name with enhanced patterns"""
        
        name_lower = arg_name.lower()
        
        # Enhanced type inference patterns
        type_patterns = {
            # Identifiers
            'id': 'Union[int, str]',
            'uuid': 'str',
            'key': 'str',
            
            # Text and strings
            'name': 'str',
            'title': 'str',
            'description': 'str',
            'text': 'str',
            'message': 'str',
            'content': 'str',
            'value': 'str',
            'label': 'str',
            
            # Numbers
            'count': 'int',
            'size': 'int',
            'length': 'int',
            'width': 'int',
            'height': 'int',
            'index': 'int',
            'number': 'Union[int, float]',
            'amount': 'Union[int, float]',
            'price': 'float',
            'rate': 'float',
            'percentage': 'float',
            'ratio': 'float',
            
            # Collections
            'data': 'Union[List, Dict, Any]',
            'items': 'List[Any]',
            'list': 'List[Any]',
            'array': 'List[Any]',
            'dict': 'Dict[str, Any]',
            'mapping': 'Dict[str, Any]',
            'params': 'Dict[str, Any]',
            'kwargs': 'Dict[str, Any]',
            'args': 'List[Any]',
            
            # Configuration
            'config': 'Dict[str, Any]',
            'settings': 'Dict[str, Any]',
            'options': 'Dict[str, Any]',
            'preferences': 'Dict[str, Any]',
            
            # File system
            'file': 'Union[str, Path]',
            'path': 'Union[str, Path]',
            'filename': 'str',
            'directory': 'Union[str, Path]',
            'folder': 'Union[str, Path]',
            
            # Boolean flags
            'enabled': 'bool',
            'active': 'bool',
            'visible': 'bool',
            'valid': 'bool',
            'ready': 'bool',
            'done': 'bool',
            'success': 'bool',
            'failed': 'bool',
            'debug': 'bool',
            'verbose': 'bool',
            
            # Time and dates
            'date': 'datetime',
            'time': 'Union[datetime, float]',
            'timestamp': 'Union[datetime, float]',
            'duration': 'Union[timedelta, float]',
            'timeout': 'Union[int, float]',
            
            # Callbacks and functions
            'callback': 'Callable',
            'handler': 'Callable',
            'func': 'Callable',
            'function': 'Callable',
            'method': 'Callable',
            
            # Network and URLs
            'url': 'str',
            'uri': 'str',
            'endpoint': 'str',
            'host': 'str',
            'port': 'int',
            
            # Database
            'query': 'str',
            'sql': 'str',
            'table': 'str',
            'column': 'str',
            'row': 'Dict[str, Any]',
            'cursor': 'Any',
            'connection': 'Any'
        }
        
        # Check for exact matches first
        for pattern, type_hint in type_patterns.items():
            if pattern in name_lower:
                return type_hint
        
        # Check for suffixes
        if name_lower.endswith('_id'):
            return 'Union[int, str]'
        elif name_lower.endswith('_list'):
            return 'List[Any]'
        elif name_lower.endswith('_dict'):
            return 'Dict[str, Any]'
        elif name_lower.endswith('_count'):
            return 'int'
        elif name_lower.endswith('_flag'):
            return 'bool'
        elif name_lower.endswith('_path'):
            return 'Union[str, Path]'
        elif name_lower.endswith('_url'):
            return 'str'
        elif name_lower.endswith('_time'):
            return 'Union[datetime, float]'
        
        return 'Any'
    
    def _infer_argument_description(self, arg_name: str) -> str:
        """Infer argument description from name with enhanced patterns"""
        
        name_lower = arg_name.lower()
        
        # Enhanced description patterns
        desc_patterns = {
            'id': 'Unique identifier',
            'uuid': 'Universally unique identifier',
            'key': 'Key for identification or lookup',
            'name': 'Name or identifier string',
            'title': 'Title or heading text',
            'description': 'Descriptive text',
            'text': 'Text content',
            'message': 'Message content',
            'content': 'Content data',
            'value': 'Value to be processed',
            'label': 'Label or tag',
            'data': 'Data to be processed or analyzed',
            'items': 'Collection of items',
            'config': 'Configuration parameters',
            'settings': 'Settings dictionary',
            'options': 'Options or preferences',
            'params': 'Parameters dictionary',
            'file': 'File path or file object',
            'path': 'File or directory path',
            'filename': 'Name of the file',
            'directory': 'Directory path',
            'enabled': 'Whether feature is enabled',
            'active': 'Whether item is active',
            'verbose': 'Whether to show detailed output',
            'debug': 'Whether to enable debug mode',
            'timeout': 'Timeout duration in seconds',
            'count': 'Number of items',
            'size': 'Size measurement',
            'length': 'Length measurement',
            'index': 'Index position',
            'callback': 'Function to call back',
            'handler': 'Event handler function',
            'url': 'URL string',
            'host': 'Host address',
            'port': 'Port number',
            'query': 'Query string or object',
            'table': 'Database table name'
        }
        
        # Check for exact matches
        for pattern, description in desc_patterns.items():
            if pattern in name_lower:
                return description
        
        # Check for prefixes
        if name_lower.startswith('is_'):
            return f"Whether {arg_name[3:].replace('_', ' ')}"
        elif name_lower.startswith('has_'):
            return f"Whether has {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('can_'):
            return f"Whether can {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('should_'):
            return f"Whether should {arg_name[7:].replace('_', ' ')}"
        elif name_lower.startswith('will_'):
            return f"Whether will {arg_name[5:].replace('_', ' ')}"
        elif name_lower.startswith('max_'):
            return f"Maximum {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('min_'):
            return f"Minimum {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('num_'):
            return f"Number of {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('total_'):
            return f"Total {arg_name[6:].replace('_', ' ')}"
        elif name_lower.startswith('current_'):
            return f"Current {arg_name[8:].replace('_', ' ')}"
        elif name_lower.startswith('new_'):
            return f"New {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('old_'):
            return f"Old {arg_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('source_'):
            return f"Source {arg_name[7:].replace('_', ' ')}"
        elif name_lower.startswith('target_'):
            return f"Target {arg_name[7:].replace('_', ' ')}"
        elif name_lower.startswith('input_'):
            return f"Input {arg_name[6:].replace('_', ' ')}"
        elif name_lower.startswith('output_'):
            return f"Output {arg_name[7:].replace('_', ' ')}"
        
        # Check for suffixes
        if name_lower.endswith('_list'):
            return f"List of {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_dict'):
            return f"Dictionary of {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_set'):
            return f"Set of {arg_name[:-4].replace('_', ' ')}"
        elif name_lower.endswith('_tuple'):
            return f"Tuple of {arg_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_id'):
            return f"ID of {arg_name[:-3].replace('_', ' ')}"
        elif name_lower.endswith('_name'):
            return f"Name of {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_path'):
            return f"Path to {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_file'):
            return f"File for {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_dir'):
            return f"Directory for {arg_name[:-4].replace('_', ' ')}"
        elif name_lower.endswith('_url'):
            return f"URL for {arg_name[:-4].replace('_', ' ')}"
        elif name_lower.endswith('_count'):
            return f"Count of {arg_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_size'):
            return f"Size of {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_length'):
            return f"Length of {arg_name[:-7].replace('_', ' ')}"
        elif name_lower.endswith('_flag'):
            return f"Flag for {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_mode'):
            return f"Mode for {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_type'):
            return f"Type of {arg_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_format'):
            return f"Format for {arg_name[:-7].replace('_', ' ')}"
        elif name_lower.endswith('_style'):
            return f"Style for {arg_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_level'):
            return f"Level of {arg_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_status'):
            return f"Status of {arg_name[:-7].replace('_', ' ')}"
        elif name_lower.endswith('_state'):
            return f"State of {arg_name[:-6].replace('_', ' ')}"
        
        # Default description
        return f"The {arg_name.replace('_', ' ')}"
    
    def _generate_returns_documentation(self, node: ast.FunctionDef, style: str) -> str:
        """Generate returns documentation in specified style"""
        
        try:
            # Check if function has return statements
            return_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            
            if not return_nodes:
                return "    None"
            
            # Try to infer return type from function name and return statements
            name_lower = node.name.lower()
            
            # Analyze return statements
            return_types = []
            for ret_node in return_nodes:
                if ret_node.value is None:
                    return_types.append("None")
                elif isinstance(ret_node.value, ast.Constant):
                    if isinstance(ret_node.value.value, bool):
                        return_types.append("bool")
                    elif isinstance(ret_node.value.value, int):
                        return_types.append("int")
                    elif isinstance(ret_node.value.value, float):
                        return_types.append("float")
                    elif isinstance(ret_node.value.value, str):
                        return_types.append("str")
                elif isinstance(ret_node.value, ast.List):
                    return_types.append("List")
                elif isinstance(ret_node.value, ast.Dict):
                    return_types.append("Dict")
                elif isinstance(ret_node.value, ast.Name):
                    return_types.append("Any")
            
            # Infer from function name
            if name_lower.startswith('get_'):
                description = f"The requested {node.name[4:].replace('_', ' ')}"
                inferred_type = "Any"
            elif name_lower.startswith('is_') or name_lower.startswith('has_') or name_lower.startswith('can_'):
                description = "True if condition is met, False otherwise"
                inferred_type = "bool"
            elif name_lower.startswith('calculate_') or name_lower.startswith('compute_'):
                description = "The calculated result"
                inferred_type = "Union[int, float]"
            elif name_lower.startswith('create_') or name_lower.startswith('build_'):
                description = f"The created {node.name.split('_', 1)[1].replace('_', ' ')}"
                inferred_type = "Any"
            elif name_lower.startswith('find_') or name_lower.startswith('search_'):
                description = f"The found {node.name.split('_', 1)[1].replace('_', ' ')} or None if not found"
                inferred_type = "Optional[Any]"
            elif 'list' in name_lower:
                description = "List of items"
                inferred_type = "List[Any]"
            elif 'dict' in name_lower:
                description = "Dictionary of key-value pairs"
                inferred_type = "Dict[str, Any]"
            elif name_lower.startswith('count_'):
                description = f"Number of {node.name[6:].replace('_', ' ')}"
                inferred_type = "int"
            elif name_lower.startswith('parse_'):
                description = f"Parsed {node.name[6:].replace('_', ' ')}"
                inferred_type = "Any"
            elif name_lower.startswith('format_'):
                description = f"Formatted {node.name[7:].replace('_', ' ')}"
                inferred_type = "str"
            elif name_lower.startswith('convert_'):
                description = f"Converted {node.name[8:].replace('_', ' ')}"
                inferred_type = "Any"
            elif name_lower.startswith('validate_'):
                description = "True if validation passes, False otherwise"
                inferred_type = "bool"
            else:
                description = "The result of the operation"
                inferred_type = "Any"
            
            # Use detected types if available
            if return_types:
                unique_types = list(set(return_types))
                if len(unique_types) == 1:
                    inferred_type = unique_types[0]
                else:
                    inferred_type = f"Union[{', '.join(unique_types)}]"
            
            # Format according to style
            if style == 'google':
                return f"    {inferred_type}: {description}"
            elif style == 'numpy':
                return f"{inferred_type}\n    {description}"
            elif style == 'sphinx':
                return f":returns: {description}\n:rtype: {inferred_type}"
            else:
                return f"    {inferred_type}: {description}"
                
        except Exception as e:
            self.logger.error(f"Error generating returns documentation: {e}")
            return "    Return value documentation not available."
    
    def _generate_raises_documentation(self, node: ast.FunctionDef, style: str) -> str:
        """Generate raises documentation in specified style"""
        
        try:
            raises = []
            
            # Check for explicit raise statements
            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                        exception_name = child.exc.func.id
                        raises.append(f"{exception_name}: When specific conditions are not met")
                    elif isinstance(child.exc, ast.Name):
                        exception_name = child.exc.id
                        raises.append(f"{exception_name}: When error conditions occur")
                
                # Check for try-except blocks
                elif isinstance(child, ast.Try):
                    for handler in child.handlers:
                        if handler.type and isinstance(handler.type, ast.Name):
                            exception_name = handler.type.id
                            raises.append(f"{exception_name}: When specific error conditions are encountered")
            
            # Add common exceptions based on function patterns
            name_lower = node.name.lower()
            
            if any(pattern in name_lower for pattern in ['file', 'read', 'write', 'open']):
                raises.append("FileNotFoundError: When file cannot be found or accessed")
                raises.append("PermissionError: When insufficient permissions to access file")
            
            if any(pattern in name_lower for pattern in ['parse', 'decode', 'convert']):
                raises.append("ValueError: When input data is invalid or malformed")
            
            if any(pattern in name_lower for pattern in ['http', 'request', 'api', 'url']):
                raises.append("ConnectionError: When network connection fails")
                raises.append("TimeoutError: When request times out")
            
            if any(pattern in name_lower for pattern in ['validate', 'check']):
                raises.append("ValidationError: When validation fails")
            
            # Format according to style
            if raises:
                unique_raises = list(dict.fromkeys(raises))  # Remove duplicates while preserving order
                
                if style == 'google':
                    return "\n".join([f"    {exc}" for exc in unique_raises])
                elif style == 'numpy':
                    formatted = []
                    for exc in unique_raises:
                        exc_type, desc = exc.split(': ', 1)
                        formatted.append(f"{exc_type}\n    {desc}")
                    return "\n".join(formatted)
                elif style == 'sphinx':
                    formatted = []
                    for exc in unique_raises:
                        exc_type, desc = exc.split(': ', 1)
                        formatted.append(f":raises {exc_type}: {desc}")
                    return "\n".join(formatted)
                else:
                    return "\n".join([f"    {exc}" for exc in unique_raises])
            else:
                return "    None"
                
        except Exception as e:
            self.logger.error(f"Error generating raises documentation: {e}")
            return "    Exception documentation not available."
    
    def _generate_function_example(self, node: ast.FunctionDef) -> str:
        """Generate comprehensive usage example for function"""
        
        try:
            function_name = node.name
            
            # Generate example based on parameters
            if not node.args.args or (len(node.args.args) == 1 and node.args.args[0].arg == 'self'):
                # No parameters
                example = f"    >>> {function_name}()\n    # Expected output based on function purpose"
            else:
                # Generate sample parameters
                params = []
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue
                    
                    param_example = self._generate_parameter_example(arg.arg)
                    params.append(param_example)
                
                params_str = ", ".join(params)
                
                # Create multi-line example for complex functions
                if len(params) > 3:
                    example = f"    >>> result = {function_name}(\n"
                    for i, (arg, param_val) in enumerate(zip([a.arg for a in node.args.args if a.arg != 'self'], params)):
                        example += f"    ...     {arg}={param_val}"
                        if i < len(params) - 1:
                            example += ","
                        example += "\n"
                    example += "    ... )\n    >>> print(result)\n    # Expected output"
                else:
                    example = f"    >>> result = {function_name}({params_str})\n    >>> print(result)\n    # Expected output"
            
            # Add context-specific example details
            name_lower = function_name.lower()
            if 'validate' in name_lower:
                example += "\n    True  # if validation passes"
            elif 'calculate' in name_lower or 'compute' in name_lower:
                example += "\n    42.5  # calculated result"
            elif 'get' in name_lower:
                example += "\n    'retrieved_value'  # the requested data"
            elif 'create' in name_lower:
                example += "\n    <CreatedObject>  # newly created instance"
            elif 'parse' in name_lower:
                example += "\n    {'parsed': 'data'}  # parsed structure"
            elif 'list' in name_lower:
                example += "\n    ['item1', 'item2', 'item3']  # list of items"
            
            return example
            
        except Exception as e:
            self.logger.error(f"Error generating function example: {e}")
            return "    # Example usage would be provided here"
    
    def _generate_parameter_example(self, param_name: str) -> str:
        """Generate realistic example value for parameter"""
        
        name_lower = param_name.lower()
        
        # Enhanced parameter example patterns
        if 'id' in name_lower:
            return "123" if name_lower.endswith('_id') else "'user_123'"
        elif 'name' in name_lower:
            if 'file' in name_lower:
                return "'document.txt'"
            elif 'user' in name_lower:
                return "'john_doe'"
            else:
                return "'example_name'"
        elif 'email' in name_lower:
            return "'user@example.com'"
        elif 'url' in name_lower:
            return "'https://api.example.com'"
        elif 'path' in name_lower:
            return "'/path/to/file.txt'"
        elif 'count' in name_lower or 'size' in name_lower or 'limit' in name_lower:
            return "10"
        elif 'timeout' in name_lower:
            return "30.0"
        elif 'port' in name_lower:
            return "8080"
        elif 'data' in name_lower:
            if 'json' in name_lower:
                return "{'key': 'value'}"
            else:
                return "[1, 2, 3, 4, 5]"
        elif 'config' in name_lower or 'settings' in name_lower:
            return "{'debug': True, 'timeout': 30}"
        elif 'params' in name_lower or 'kwargs' in name_lower:
            return "{'param1': 'value1', 'param2': 'value2'}"
        elif 'headers' in name_lower:
            return "{'Content-Type': 'application/json'}"
        elif 'query' in name_lower:
            return "'SELECT * FROM users'"
        elif 'message' in name_lower or 'text' in name_lower:
            return "'Hello, World!'"
        elif 'password' in name_lower or 'secret' in name_lower:
            return "'secure_password'"
        elif 'token' in name_lower:
            return "'abc123def456'"
        elif 'key' in name_lower:
            return "'api_key_here'"
        elif 'host' in name_lower:
            return "'localhost'"
        elif 'enabled' in name_lower or 'active' in name_lower or 'debug' in name_lower:
            return "True"
        elif 'verbose' in name_lower:
            return "False"
        elif 'callback' in name_lower or 'handler' in name_lower:
            return "my_callback_function"
        elif 'format' in name_lower:
            return "'json'"
        elif 'encoding' in name_lower:
            return "'utf-8'"
        elif 'mode' in name_lower:
            return "'read'"
        elif 'type' in name_lower:
            return "'default'"
        elif 'level' in name_lower:
            return "1"
        elif 'index' in name_lower:
            return "0"
        elif 'version' in name_lower:
            return "'1.0.0'"
        elif 'status' in name_lower:
            return "'active'"
        elif 'priority' in name_lower:
            return "5"
        elif 'weight' in name_lower:
            return "0.8"
        elif 'rate' in name_lower:
            return "0.95"
        elif 'threshold' in name_lower:
            return "0.5"
        elif 'percentage' in name_lower:
            return "75.0"
        elif 'amount' in name_lower:
            return "100.0"
        elif 'price' in name_lower:
            return "19.99"
        elif 'date' in name_lower:
            return "datetime(2023, 12, 25)"
        elif 'time' in name_lower:
            return "datetime.now()"
        else:
            return "'example_value'"
    
    def _generate_class_docstring(self, tree: ast.AST, class_name: str, style: str) -> str:
        """Generate comprehensive docstring for class"""
        
        try:
            # Find the class
            target_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    target_class = node
                    break
            
            if not target_class:
                return f"Class '{class_name}' not found."
            
            # Generate components
            summary = self._generate_class_summary(target_class)
            description = self._generate_class_description(target_class)
            attributes = self._extract_class_attributes(target_class, style)
            methods = self._extract_class_methods(target_class, style)
            example = self._generate_class_example(target_class)
            
            # Fill template
            docstring = self.docstring_templates['class'].format(
                summary=summary,
                description=description,
                attributes=attributes,
                methods=methods,
                example=example
            )
            
            return docstring.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating class docstring: {e}")
            return f"Error generating docstring for class '{class_name}'"
    
    def _generate_class_summary(self, node: ast.ClassDef) -> str:
        """Generate concise summary for class"""
        
        name = node.name
        
        # Enhanced class summary patterns
        name_lower = name.lower()
        
        if 'manager' in name_lower:
            return f"{name} - Manages and coordinates {name.replace('Manager', '').lower()} operations"
        elif 'handler' in name_lower:
            return f"{name} - Handles {name.replace('Handler', '').lower()} events and requests"
        elif 'processor' in name_lower:
            return f"{name} - Processes {name.replace('Processor', '').lower()} data and operations"
        elif 'analyzer' in name_lower or 'analyser' in name_lower:
            return f"{name} - Analyzes {name.replace('Analyzer', '').replace('Analyser', '').lower()} data"
        elif 'generator' in name_lower:
            return f"{name} - Generates {name.replace('Generator', '').lower()} content"
        elif 'builder' in name_lower:
            return f"{name} - Builds and constructs {name.replace('Builder', '').lower()} objects"
        elif 'factory' in name_lower:
            return f"{name} - Factory for creating {name.replace('Factory', '').lower()} instances"
        elif 'adapter' in name_lower:
            return f"{name} - Adapts {name.replace('Adapter', '').lower()} interfaces"
        elif 'controller' in name_lower:
            return f"{name} - Controls {name.replace('Controller', '').lower()} behavior"
        elif 'service' in name_lower:
            return f"{name} - Provides {name.replace('Service', '').lower()} services"
        elif 'client' in name_lower:
            return f"{name} - Client for {name.replace('Client', '').lower()} interactions"
        elif 'server' in name_lower:
            return f"{name} - Server for {name.replace('Server', '').lower()} operations"
        elif 'validator' in name_lower:
            return f"{name} - Validates {name.replace('Validator', '').lower()} data"
        elif 'parser' in name_lower:
            return f"{name} - Parses {name.replace('Parser', '').lower()} data"
        elif 'formatter' in name_lower:
            return f"{name} - Formats {name.replace('Formatter', '').lower()} output"
        elif 'converter' in name_lower:
            return f"{name} - Converts {name.replace('Converter', '').lower()} data"
        elif 'transformer' in name_lower:
            return f"{name} - Transforms {name.replace('Transformer', '').lower()} data"
        elif 'monitor' in name_lower:
            return f"{name} - Monitors {name.replace('Monitor', '').lower()} status"
        elif 'scheduler' in name_lower:
            return f"{name} - Schedules {name.replace('Scheduler', '').lower()} operations"
        elif 'engine' in name_lower:
            return f"{name} - Core engine for {name.replace('Engine', '').lower()} processing"
        elif 'helper' in name_lower or 'util' in name_lower:
            return f"{name} - Utility class for {name.replace('Helper', '').replace('Util', '').lower()} operations"
        else:
            return f"{name} - Represents and manages {name.lower()} functionality"
    
    def _generate_class_description(self, node: ast.ClassDef) -> str:
        """Generate detailed description for class"""
        
        # Analyze class structure
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        properties = len([n for n in methods if n.name.startswith('_') and not n.name.startswith('__')])
        public_methods = len([n for n in methods if not n.name.startswith('_')])
        special_methods = len([n for n in methods if n.name.startswith('__')])
        
        description_parts = []
        
        if len(methods) > 10:
            description_parts.append("This is a comprehensive class with extensive functionality")
        elif len(methods) > 5:
            description_parts.append("This class provides a complete set of operations")
        else:
            description_parts.append("This class encapsulates core functionality")
        
        if public_methods > 0:
            description_parts.append(f"with {public_methods} public methods for external interaction")
        
        if properties > 0:
            description_parts.append(f"and {properties} internal methods for state management")
        
        if special_methods > 2:  # More than just __init__
            description_parts.append("It implements special methods for enhanced Python integration")
        
        # Check for inheritance
        if node.bases:
            base_count = len(node.bases)
            if base_count == 1:
                description_parts.append("The class extends a base class to inherit core functionality")
            else:
                description_parts.append(f"The class inherits from {base_count} base classes for mixed functionality")
        
        return ". ".join(description_parts) + "."
    
    def _extract_class_attributes(self, node: ast.ClassDef, style: str) -> str:
        """Extract and document class attributes"""
        
        try:
            attributes = []
            
            # Look for assignments in __init__ method
            init_method = None
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    init_method = child
                    break
            
            if init_method:
                for child in ast.walk(init_method):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                attr_name = target.attr
                                attr_desc = self._infer_attribute_description(attr_name)
                                attr_type = self._infer_attribute_type(attr_name, child.value)
                                
                                if style == 'google':
                                    attributes.append(f"    {attr_name} ({attr_type}): {attr_desc}")
                                elif style == 'numpy':
                                    attributes.append(f"{attr_name} : {attr_type}\n    {attr_desc}")
                                elif style == 'sphinx':
                                    attributes.append(f":ivar {attr_name}: {attr_desc}")
                                    attributes.append(f":vartype {attr_name}: {attr_type}")
                                else:
                                    attributes.append(f"    {attr_name} ({attr_type}): {attr_desc}")
            
            # Look for class variables
            for child in node.body:
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            attr_desc = self._infer_attribute_description(attr_name)
                            attr_type = self._infer_attribute_type(attr_name, child.value)
                            
                            if style == 'google':
                                attributes.append(f"    {attr_name} ({attr_type}): {attr_desc} (class variable)")
                            elif style == 'numpy':
                                attributes.append(f"{attr_name} : {attr_type}\n    {attr_desc} (class variable)")
                            elif style == 'sphinx':
                                attributes.append(f":cvar {attr_name}: {attr_desc}")
                                attributes.append(f":vartype {attr_name}: {attr_type}")
                            else:
                                attributes.append(f"    {attr_name} ({attr_type}): {attr_desc} (class variable)")
            
            return "\n".join(attributes) if attributes else "    None"
            
        except Exception as e:
            self.logger.error(f"Error extracting class attributes: {e}")
            return "    Attributes documentation not available."
    
    def _extract_class_methods(self, node: ast.ClassDef, style: str) -> str:
        """Extract and document class methods"""
        
        try:
            methods = []
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    method_name = child.name
                    if method_name.startswith('__') and method_name.endswith('__'):
                        continue  # Skip magic methods
                    
                    method_desc = self._infer_method_description(method_name, child)
                    
                    if style == 'google':
                        methods.append(f"    {method_name}(): {method_desc}")
                    elif style == 'numpy':
                        methods.append(f"{method_name}\n    {method_desc}")
                    elif style == 'sphinx':
                        methods.append(f":meth:`{method_name}`: {method_desc}")
                    else:
                        methods.append(f"    {method_name}(): {method_desc}")
            
            return "\n".join(methods) if methods else "    None"
            
        except Exception as e:
            self.logger.error(f"Error extracting class methods: {e}")
            return "    Methods documentation not available."
    
    def _infer_attribute_description(self, attr_name: str) -> str:
        """Infer description for class attribute"""
        
        name_lower = attr_name.lower()
        
        # Common attribute patterns
        attr_patterns = {
            'id': 'Unique identifier',
            'name': 'Name or identifier',
            'title': 'Title string',
            'description': 'Description text',
            'status': 'Current status',
            'state': 'Current state',
            'config': 'Configuration settings',
            'settings': 'Settings dictionary',
            'data': 'Data storage',
            'cache': 'Cached data',
            'buffer': 'Data buffer',
            'queue': 'Processing queue',
            'stack': 'Data stack',
            'list': 'List of items',
            'dict': 'Dictionary mapping',
            'set': 'Set of unique items',
            'count': 'Item count',
            'size': 'Size measurement',
            'length': 'Length value',
            'index': 'Current index',
            'position': 'Current position',
            'offset': 'Offset value',
            'timestamp': 'Timestamp value',
            'timeout': 'Timeout duration',
            'enabled': 'Whether feature is enabled',
            'active': 'Whether item is active',
            'ready': 'Whether system is ready',
            'running': 'Whether process is running',
            'finished': 'Whether task is finished',
            'verbose': 'Whether verbose output is enabled',
            'debug': 'Whether debug mode is enabled'
        }
        
        for pattern, description in attr_patterns.items():
            if pattern in name_lower:
                return description
        
        # Check for common prefixes/suffixes
        if name_lower.startswith('is_'):
            return f"Whether {attr_name[3:].replace('_', ' ')}"
        elif name_lower.startswith('has_'):
            return f"Whether has {attr_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('can_'):
            return f"Whether can {attr_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('should_'):
            return f"Whether should {attr_name[7:].replace('_', ' ')}"
        elif name_lower.startswith('max_'):
            return f"Maximum {attr_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('min_'):
            return f"Minimum {attr_name[4:].replace('_', ' ')}"
        elif name_lower.startswith('total_'):
            return f"Total {attr_name[6:].replace('_', ' ')}"
        elif name_lower.startswith('current_'):
            return f"Current {attr_name[8:].replace('_', ' ')}"
        elif name_lower.endswith('_count'):
            return f"Number of {attr_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_list'):
            return f"List of {attr_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_dict'):
            return f"Dictionary of {attr_name[:-5].replace('_', ' ')}"
        elif name_lower.endswith('_cache'):
            return f"Cache for {attr_name[:-6].replace('_', ' ')}"
        elif name_lower.endswith('_buffer'):
            return f"Buffer for {attr_name[:-7].replace('_', ' ')}"
        
        return f"The {attr_name.replace('_', ' ')}"
    
    def _infer_attribute_type(self, attr_name: str, value_node: ast.AST) -> str:
        """Infer type for class attribute"""
        
        # First try to infer from the assigned value
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "bool"
            elif isinstance(value_node.value, int):
                return "int"
            elif isinstance(value_node.value, float):
                return "float"
            elif isinstance(value_node.value, str):
                return "str"
        elif isinstance(value_node, ast.List):
            return "List[Any]"
        elif isinstance(value_node, ast.Dict):
            return "Dict[str, Any]"
        elif isinstance(value_node, ast.Set):
            return "Set[Any]"
        elif isinstance(value_node, ast.Tuple):
            return "Tuple[Any, ...]"
        
        # Fall back to name-based inference
        return self._infer_argument_type(attr_name)
    
    def _infer_method_description(self, method_name: str, method_node: ast.FunctionDef) -> str:
        """Infer description for class method"""
        
        # Get existing docstring if available
        existing_docstring = ast.get_docstring(method_node)
        if existing_docstring:
            # Use first sentence of existing docstring
            first_sentence = existing_docstring.split('.')[0].strip()
            if first_sentence:
                return first_sentence
        
        # Use function purpose inference
        return self._infer_function_purpose(method_name, method_node)
    
    def _generate_class_example(self, node: ast.ClassDef) -> str:
        """Generate comprehensive usage example for class"""
        
        try:
            class_name = node.name
            
            # Find __init__ method to determine constructor parameters
            init_method = None
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    init_method = child
                    break
            
            # Generate constructor example
            if init_method and len(init_method.args.args) > 1:  # More than just 'self'
                params = []
                for arg in init_method.args.args:
                    if arg.arg == 'self':
                        continue
                    param_example = self._generate_parameter_example(arg.arg)
                    params.append(f"{arg.arg}={param_example}")
                
                constructor_call = f"{class_name}({', '.join(params)})"
            else:
                constructor_call = f"{class_name}()"
            
            # Find some public methods for example
            public_methods = [n.name for n in node.body 
                            if isinstance(n, ast.FunctionDef) and 
                            not n.name.startswith('_') and n.name != '__init__']
            
            example = f"    >>> obj = {constructor_call}\n"
            
            if public_methods:
                # Use first few public methods
                for method in public_methods[:3]:
                    if 'get' in method.lower():
                        example += f"    >>> result = obj.{method}()\n"
                    elif 'set' in method.lower():
                        example += f"    >>> obj.{method}('new_value')\n"
                    elif 'process' in method.lower():
                        example += f"    >>> obj.{method}(data)\n"
                    else:
                        example += f"    >>> obj.{method}()\n"
                
                example += "    >>> print(obj)\n    <{} object at 0x...>".format(class_name)
            else:
                example += f"    >>> print(obj)\n    <{class_name} object at 0x...>"
            
            return example
            
        except Exception as e:
            self.logger.error(f"Error generating class example: {e}")
            return "    # Example usage would be provided here"
    
    def _generate_module_docstring(self, tree: ast.AST, code: str, style: str) -> str:
        """Generate module-level docstring"""
        
        try:
            # Analyze module contents
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) 
                        and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                                   if n in ast.walk(parent))]
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            
            # Generate module summary
            if classes and functions:
                summary = f"Module providing {len(classes)} classes and {len(functions)} functions"
            elif classes:
                summary = f"Module defining {len(classes)} classes"
            elif functions:
                summary = f"Module providing {len(functions)} utility functions"
            else:
                summary = "Utility module"
            
            # Generate description based on content
            description = self._infer_module_purpose(classes, functions, code)
            
            # Document classes
            if classes:
                classes_doc = "\n".join([f"    {cls}: {self._infer_class_purpose_from_name(cls)}" 
                                       for cls in classes])
            else:
                classes_doc = "    None"
            
            # Document functions
            if functions:
                functions_doc = "\n".join([f"    {func}: {self._infer_function_purpose_from_name(func)}" 
                                         for func in functions])
            else:
                functions_doc = "    None"
            
            # Generate usage example
            usage = self._generate_module_usage_example(classes, functions)
            
            # Fill template
            docstring = self.docstring_templates['module'].format(
                summary=summary,
                description=description,
                classes=classes_doc,
                functions=functions_doc,
                usage=usage
            )
            
            return docstring.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating module docstring: {e}")
            return "Module documentation not available."
    
    def _infer_module_purpose(self, classes: List[str], functions: List[str], code: str) -> str:
        """Infer the purpose of the module"""
        
        # Analyze names for domain clues
        all_names = classes + functions
        combined_text = ' '.join(all_names).lower() + ' ' + code.lower()
        
        domain_indicators = {
            'data processing': ['process', 'parse', 'transform', 'convert', 'filter', 'analyze'],
            'web development': ['http', 'request', 'response', 'server', 'client', 'api', 'web'],
            'database operations': ['database', 'db', 'sql', 'query', 'table', 'record'],
            'file operations': ['file', 'read', 'write', 'path', 'directory', 'io'],
            'testing utilities': ['test', 'mock', 'assert', 'verify', 'check'],
            'configuration management': ['config', 'setting', 'option', 'parameter'],
            'utility functions': ['util', 'helper', 'tool', 'common'],
            'mathematical operations': ['math', 'calculate', 'compute', 'algorithm'],
            'networking': ['network', 'socket', 'connection', 'protocol'],
            'security': ['auth', 'security', 'encrypt', 'hash', 'token'],
            'logging and monitoring': ['log', 'monitor', 'track', 'debug'],
            'user interface': ['ui', 'interface', 'display', 'view', 'render']
        }
        
        scores = {}
        for domain, keywords in domain_indicators.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[domain] = score
        
        if scores:
            primary_domain = max(scores, key=scores.get)
            return f"This module provides functionality for {primary_domain}."
        else:
            return "This module contains various utility functions and classes."
    
    def _infer_class_purpose_from_name(self, class_name: str) -> str:
        """Infer class purpose from name only"""
        
        name_lower = class_name.lower()
        
        purpose_map = {
            'manager': 'manages operations',
            'handler': 'handles events',
            'processor': 'processes data',
            'analyzer': 'analyzes data',
            'generator': 'generates content',
            'builder': 'builds objects',
            'factory': 'creates instances',
            'adapter': 'adapts interfaces',
            'controller': 'controls behavior',
            'service': 'provides services',
            'client': 'client functionality',
            'server': 'server functionality',
            'validator': 'validates data',
            'parser': 'parses data',
            'formatter': 'formats output',
            'converter': 'converts data'
        }
        
        for pattern, purpose in purpose_map.items():
            if pattern in name_lower:
                return purpose
        
        return 'provides functionality'
    
    def _infer_function_purpose_from_name(self, func_name: str) -> str:
        """Infer function purpose from name only"""
        
        name_lower = func_name.lower()
        
        if name_lower.startswith('get_'):
            return 'retrieves data'
        elif name_lower.startswith('set_'):
            return 'sets values'
        elif name_lower.startswith('create_'):
            return 'creates objects'
        elif name_lower.startswith('delete_'):
            return 'deletes items'
        elif name_lower.startswith('process_'):
            return 'processes data'
        elif name_lower.startswith('calculate_'):
            return 'performs calculations'
        elif name_lower.startswith('validate_'):
            return 'validates input'
        elif name_lower.startswith('parse_'):
            return 'parses data'
        elif name_lower.startswith('format_'):
            return 'formats output'
        else:
            return 'utility function'
    
    def _generate_module_usage_example(self, classes: List[str], functions: List[str]) -> str:
        """Generate usage example for module"""
        
        examples = []
        
        if classes:
            class_name = classes[0]
            examples.append(f"    >>> from module import {class_name}")
            examples.append(f"    >>> obj = {class_name}()")
            examples.append("    >>> obj.method()")
        
        if functions:
            func_name = functions[0]
            examples.append(f"    >>> from module import {func_name}")
            examples.append(f"    >>> result = {func_name}()")
        
        return "\n".join(examples) if examples else "    # Usage examples would be provided here"
    
    def _generate_api_documentation(self, tree: ast.AST, code: str, style: str) -> str:
        """Generate comprehensive API documentation"""
        
        try:
            documentation = "# API Documentation\n\n"
            
            # Add module overview
            documentation += "## Overview\n\n"
            documentation += self._generate_module_overview(tree, code)
            documentation += "\n\n"
            
            # Document all public classes
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) 
                      and not n.name.startswith('_')]
            
            if classes:
                documentation += "## Classes\n\n"
                for cls_node in classes:
                    documentation += f"### {cls_node.name}\n\n"
                    class_doc = self._generate_class_docstring(tree, cls_node.name, style)
                    documentation += class_doc + "\n\n"
            
            # Document all public functions
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) 
                        and not n.name.startswith('_') and
                        not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                               if n in ast.walk(parent))]
            
            if functions:
                documentation += "## Functions\n\n"
                for func_node in functions:
                    documentation += f"### {func_node.name}\n\n"
                    func_doc = self._generate_function_docstring(tree, func_node.name, style)
                    documentation += func_doc + "\n\n"
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Error generating API documentation: {e}")
            return "API documentation not available."
    
    def _generate_module_overview(self, tree: ast.AST, code: str) -> str:
        """Generate module overview for API documentation"""
        
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        overview = "This module provides "
        
        if classes and functions:
            overview += f"{len(classes)} classes and {len(functions)} functions "
        elif classes:
            overview += f"{len(classes)} classes "
        elif functions:
            overview += f"{len(functions)} functions "
        
        overview += "for " + self._infer_module_purpose(classes, functions, code)
        
        return overview
    
    def _generate_inline_comments(self, code: str) -> str:
        """Generate intelligent inline comments for code"""
        
        try:
            lines = code.split('\n')
            commented_lines = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    commented_lines.append(line)
                    continue
                
                # Add intelligent comments for specific patterns
                comment = self._generate_line_comment(stripped)
                if comment:
                    # Preserve original indentation
                    indent = len(line) - len(line.lstrip())
                    commented_lines.append(f"{line}  {comment}")
                else:
                    commented_lines.append(line)
            
            return '\n'.join(commented_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating inline comments: {e}")
            return code
    
    def _generate_line_comment(self, line: str) -> Optional[str]:
        """Generate comment for a specific line of code"""
        
        # Pattern-based comment generation
        if re.match(r'def\s+\w+.*:', line):
            return "# Function definition"
        elif re.match(r'class\s+\w+.*:', line):
            return "# Class definition"
        elif re.match(r'if\s+.*:', line):
            return "# Conditional check"
        elif re.match(r'elif\s+.*:', line):
            return "# Alternative condition"
        elif re.match(r'else\s*:', line):
            return "# Default case"
        elif re.match(r'for\s+.*:', line):
            return "# Loop iteration"
        elif re.match(r'while\s+.*:', line):
            return "# Conditional loop"
        elif re.match(r'try\s*:', line):
            return "# Exception handling block"
        elif re.match(r'except\s*.*:', line):
            return "# Handle exceptions"
        elif re.match(r'finally\s*:', line):
            return "# Cleanup block"
        elif re.match(r'with\s+.*:', line):
            return "# Context manager"
        elif 'return' in line:
            return "# Return result"
        elif re.match(r'\w+\s*=\s*.*', line) and not line.startswith('='):
            return "# Variable assignment"
        elif 'import' in line:
            return "# Import dependencies"
        elif 'print(' in line:
            return "# Debug output"
        elif 'raise' in line:
            return "# Raise exception"
        elif 'assert' in line:
            return "# Assertion check"
        elif 'yield' in line:
            return "# Generator yield"
        elif 'await' in line:
            return "# Async operation"
        elif re.match(r'.*\(\)', line) and '=' not in line:
            return "# Function call"
        
        return None
    
    def _generate_readme_section(self, tree: ast.AST, code: str) -> str:
        """Generate README section for the code"""
        
        try:
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            readme = "## Code Overview\n\n"
            
            # Description
            purpose = self._infer_module_purpose(classes, functions, code)
            readme += f"{purpose}\n\n"
            
            # Features
            readme += "### Features\n\n"
            if classes:
                readme += f"- {len(classes)} classes for object-oriented functionality\n"
            if functions:
                readme += f"- {len(functions)} utility functions\n"
            
            readme += "- Comprehensive error handling\n"
            readme += "- Well-documented API\n\n"
            
            # Usage
            readme += "### Usage\n\n"
            readme += "```python\n"
            readme += self._generate_module_usage_example(classes, functions)
            readme += "\n```\n\n"
            
            # Requirements
            readme += "### Requirements\n\n"
            readme += "- Python 3.6+\n"
            
            # Add discovered imports as requirements
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            external_imports = [imp for imp in set(imports) 
                              if not imp.startswith('_') and imp not in ['os', 'sys', 'json', 're']]
            
            for imp in external_imports:
                readme += f"- {imp}\n"
            
            return readme
            
        except Exception as e:
            self.logger.error(f"Error generating README section: {e}")
            return "README section not available."
    
    def _generate_architecture_documentation(self, tree: ast.AST, code: str) -> str:
        """Generate architecture documentation"""
        
        try:
            arch_doc = "# Architecture Documentation\n\n"
            
            # Overview
            arch_doc += "## System Overview\n\n"
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            arch_doc += f"The system consists of {len(classes)} classes and {len(functions)} functions "
            arch_doc += "organized in a modular architecture.\n\n"
            
            # Component Analysis
            if classes:
                arch_doc += "## Components\n\n"
                for cls_node in classes:
                    methods = [n for n in cls_node.body if isinstance(n, ast.FunctionDef)]
                    arch_doc += f"### {cls_node.name}\n\n"
                    arch_doc += f"- **Type**: Core Component\n"
                    arch_doc += f"- **Methods**: {len(methods)}\n"
                    arch_doc += f"- **Responsibility**: {self._infer_class_purpose_from_name(cls_node.name)}\n\n"
            
            # Dependencies
            arch_doc += "## Dependencies\n\n"
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            if imports:
                arch_doc += "External dependencies:\n\n"
                for imp in set(imports):
                    arch_doc += f"- `{imp}`\n"
            else:
                arch_doc += "No external dependencies.\n"
            
            arch_doc += "\n"
            
            # Design Patterns
            arch_doc += "## Design Patterns\n\n"
            patterns = self._identify_design_patterns(classes, functions)
            if patterns:
                for pattern in patterns:
                    arch_doc += f"- **{pattern}**: Identified in the codebase\n"
            else:
                arch_doc += "- Standard procedural/object-oriented design\n"
            
            return arch_doc
            
        except Exception as e:
            self.logger.error(f"Error generating architecture documentation: {e}")
            return "Architecture documentation not available."
    
    def _identify_design_patterns(self, classes: List, functions: List) -> List[str]:
        """Identify design patterns in the code"""
        
        patterns = []
        class_names = [cls.name.lower() for cls in classes]
        
        # Common pattern detection
        if any('factory' in name for name in class_names):
            patterns.append("Factory Pattern")
        if any('builder' in name for name in class_names):
            patterns.append("Builder Pattern")
        if any('adapter' in name for name in class_names):
            patterns.append("Adapter Pattern")
        if any('observer' in name for name in class_names):
            patterns.append("Observer Pattern")
        if any('singleton' in name for name in class_names):
            patterns.append("Singleton Pattern")
        if any('strategy' in name for name in class_names):
            patterns.append("Strategy Pattern")
        if any('decorator' in name for name in class_names):
            patterns.append("Decorator Pattern")
        if any('proxy' in name for name in class_names):
            patterns.append("Proxy Pattern")
        if any('facade' in name for name in class_names):
            patterns.append("Facade Pattern")
        
        return patterns
    
    def _generate_general_documentation(self, tree: ast.AST, code: str) -> str:
        """Generate general documentation when specific type not specified"""
        
        try:
            doc = "# Code Documentation\n\n"
            
            # Overview
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            doc += "## Overview\n\n"
            doc += f"This code contains {len(classes)} classes and {len(functions)} functions.\n\n"
            
            # Main components
            if classes:
                doc += "## Classes\n\n"
                for cls in classes:
                    doc += f"- **{cls.name}**: {self._infer_class_purpose_from_name(cls.name)}\n"
                doc += "\n"
            
            if functions:
                doc += "## Functions\n\n"
                standalone_functions = [f for f in functions 
                                      if not any(f in ast.walk(cls) for cls in classes)]
                for func in standalone_functions:
                    doc += f"- **{func.name}**: {self._infer_function_purpose_from_name(func.name)}\n"
                doc += "\n"
            
            # Usage note
            doc += "## Usage\n\n"
            doc += "Refer to individual function and class documentation for detailed usage instructions.\n"
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Error generating general documentation: {e}")
            return "Documentation not available."
    
    def _assess_documentation_quality(self, doc: GeneratedDocumentation):
        """Comprehensive quality assessment of generated documentation"""
        
        try:
            # Assess individual quality metrics
            doc.completeness_score = self._assess_completeness(doc)
            doc.clarity_score = self._assess_clarity(doc)
            doc.technical_accuracy = self._assess_technical_accuracy(doc)
            doc.style_consistency = self._assess_style_consistency(doc)
            
            # Calculate overall quality using weighted average
            quality_scores = {
                'completeness': doc.completeness_score,
                'clarity': doc.clarity_score,
                'technical_accuracy': doc.technical_accuracy,
                'style_consistency': doc.style_consistency
            }
            
            weighted_quality = sum(score * self.quality_weights[metric] 
                                 for metric, score in quality_scores.items())
            
            # Add example quality bonus
            if doc.includes_examples:
                weighted_quality += self.quality_weights['example_quality']
            
            doc.documentation_quality = min(1.0, weighted_quality)
            
        except Exception as e:
            self.logger.error(f"Error assessing documentation quality: {e}")
            doc.documentation_quality = 0.5
    
    def _assess_completeness(self, doc: GeneratedDocumentation) -> float:
        """Assess completeness of documentation"""
        
        try:
            completeness_score = 0.0
            content_lower = doc.generated_content.lower()
            
            # Check for essential components
            if any(keyword in content_lower for keyword in ['args', 'parameters', 'param']):
                completeness_score += 0.2
                doc.includes_parameters = True
            
            if any(keyword in content_lower for keyword in ['returns', 'return']):
                completeness_score += 0.2
                doc.includes_return_values = True
            
            if any(keyword in content_lower for keyword in ['example', 'usage']):
                completeness_score += 0.2
                doc.includes_examples = True
            
            if any(keyword in content_lower for keyword in ['raises', 'exception', 'error']):
                completeness_score += 0.2
                doc.includes_exceptions = True
            
            # Check for description quality
            if len(doc.generated_content.split('.')) >= 3:
                completeness_score += 0.2  # Multiple sentences
            
            return completeness_score
            
        except Exception as e:
            self.logger.error(f"Error assessing completeness: {e}")
            return 0.0
    
    def _assess_clarity(self, doc: GeneratedDocumentation) -> float:
        """Assess clarity of documentation"""
        
        try:
            clarity_factors = []
            content = doc.generated_content
            
            # Sentence structure and readability
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if len(sentences) >= 2:
                clarity_factors.append(0.8)
            else:
                clarity_factors.append(0.4)
            
            # Use of clear language indicators
            clarity_indicators = ['this', 'the', 'it', 'when', 'if', 'whether', 'how']
            indicator_count = sum(1 for indicator in clarity_indicators 
                                if indicator in content.lower())
            clarity_factors.append(min(1.0, indicator_count / len(clarity_indicators)))
            
            # Appropriate length
            word_count = len(content.split())
            if 20 <= word_count <= 200:
                clarity_factors.append(1.0)
            elif 10 <= word_count < 20 or 200 < word_count <= 400:
                clarity_factors.append(0.7)
            else:
                clarity_factors.append(0.4)
            
            # Structure and formatting
            if any(marker in content for marker in ['Args:', 'Returns:', 'Example:']):
                clarity_factors.append(0.9)
            else:
                clarity_factors.append(0.5)
            
            return np.mean(clarity_factors)
            
        except Exception as e:
            self.logger.error(f"Error assessing clarity: {e}")
            return 0.5
    
    def _assess_technical_accuracy(self, doc: GeneratedDocumentation) -> float:
        """Assess technical accuracy of documentation"""
        
        try:
            accuracy_factors = []
            content = doc.generated_content.lower()
            
            # Check for technical terms appropriateness
            if doc.documentation_type == DocumentationType.FUNCTION_DOCSTRING:
                if 'function' in content:
                    accuracy_factors.append(0.8)
                else:
                    accuracy_factors.append(0.6)
            elif doc.documentation_type == DocumentationType.CLASS_DOCSTRING:
                if 'class' in content:
                    accuracy_factors.append(0.8)
                else:
                    accuracy_factors.append(0.6)
            
            # Check for type information
            type_indicators = ['int', 'str', 'bool', 'float', 'list', 'dict', 'any', 'optional']
            if any(indicator in content for indicator in type_indicators):
                accuracy_factors.append(0.9)
            else:
                accuracy_factors.append(0.5)
            
            # Check for behavioral descriptions
            behavior_terms = ['returns', 'performs', 'calculates', 'processes', 'validates']
            if any(term in content for term in behavior_terms):
                accuracy_factors.append(0.8)
            else:
                accuracy_factors.append(0.6)
            
            # Error handling mentions
            if any(term in content for term in ['error', 'exception', 'raises']):
                accuracy_factors.append(0.7)
            else:
                accuracy_factors.append(0.5)
            
            return np.mean(accuracy_factors)
            
        except Exception as e:
            self.logger.error(f"Error assessing technical accuracy: {e}")
            return 0.5
    
    def _assess_style_consistency(self, doc: GeneratedDocumentation) -> float:
        """Assess style consistency of documentation"""
        
        try:
            consistency_factors = []
            content = doc.generated_content
            
            # Check for consistent formatting
            if '"""' in content or '```' in content:
                consistency_factors.append(0.8)
            else:
                consistency_factors.append(0.6)
            
            # Consistent capitalization
            lines = content.split('\n')
            consistent_caps = True
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped[0].isupper() and not stripped.startswith((':', '-', '*')):
                    consistent_caps = False
                    break
            
            consistency_factors.append(0.8 if consistent_caps else 0.4)
            
            # Consistent section formatting
            section_markers = ['Args:', 'Returns:', 'Raises:', 'Example:']
            found_markers = [marker for marker in section_markers if marker in content]
            if len(found_markers) >= 2:
                consistency_factors.append(0.9)
            elif len(found_markers) == 1:
                consistency_factors.append(0.7)
            else:
                consistency_factors.append(0.5)
            
            return np.mean(consistency_factors)
            
        except Exception as e:
            self.logger.error(f"Error assessing style consistency: {e}")
            return 0.5


def create_documentation_generator() -> DocumentationGenerator:
    """Factory function to create DocumentationGenerator instance"""
    
    return DocumentationGenerator()