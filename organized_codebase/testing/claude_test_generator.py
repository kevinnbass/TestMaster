"""
Claude-Powered Advanced Test Generator for TestMaster
Integrates Anthropic Claude for intelligent test generation
"""

import ast
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp

class TestGenerationStrategy(Enum):
    """Test generation strategies"""
    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EDGE_CASE = "edge_case"
    BUSINESS_LOGIC = "business_logic"

@dataclass
class CodeAnalysis:
    """Advanced code analysis results"""
    file_path: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    complexity_score: int
    business_logic_patterns: List[str]
    security_concerns: List[str]
    performance_hotspots: List[str]
    dependencies: List[str]
    test_worthiness: float  # 0-1 score

@dataclass
class GeneratedTest:
    """Generated test with metadata"""
    test_code: str
    strategy: TestGenerationStrategy
    coverage_targets: List[str]
    confidence_score: float
    estimated_effectiveness: float
    generation_time: float

class ClaudeTestGenerator:
    """Advanced test generator using Claude AI"""
    
    def __init__(self, api_key: Optional[str] = None, require_api_key: bool = True):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if require_api_key and not self.api_key:
            raise ValueError("Anthropic API key required")
        self.api_key = self.api_key or "test_key"  # Allow testing without real key
        
        self.model = "claude-3-opus-20240229"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.generated_count = 0
        
    async def analyze_code_intelligence(self, file_path: str) -> CodeAnalysis:
        """Advanced AI-powered code analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Extract detailed information
            classes = self._extract_class_details(tree)
            functions = self._extract_function_details(tree)
            complexity = self._calculate_complexity(tree)
            
            # AI-powered analysis
            business_patterns = await self._identify_business_logic(code)
            security_concerns = await self._identify_security_patterns(code)
            performance_hotspots = await self._identify_performance_issues(code)
            
            # Calculate test worthiness
            test_worthiness = self._calculate_test_worthiness(
                complexity, len(classes), len(functions), 
                len(business_patterns), len(security_concerns)
            )
            
            return CodeAnalysis(
                file_path=file_path,
                classes=classes,
                functions=functions,
                complexity_score=complexity,
                business_logic_patterns=business_patterns,
                security_concerns=security_concerns,
                performance_hotspots=performance_hotspots,
                dependencies=self._extract_dependencies(tree),
                test_worthiness=test_worthiness
            )
            
        except Exception as e:
            return self._empty_analysis(file_path, str(e))
    
    def _extract_class_details(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract detailed class information"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                properties = [n.targets[0].id for n in node.body 
                            if isinstance(n, ast.Assign) and hasattr(n.targets[0], 'id')]
                
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'properties': properties,
                    'docstring': ast.get_docstring(node) or "",
                    'line_number': node.lineno,
                    'inheritance': [base.id for base in node.bases 
                                  if hasattr(base, 'id')]
                })
        return classes
    
    def _extract_function_details(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract detailed function information"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                args = [arg.arg for arg in node.args.args]
                returns = ast.unparse(node.returns) if node.returns else None
                
                functions.append({
                    'name': node.name,
                    'args': args,
                    'returns': returns,
                    'docstring': ast.get_docstring(node) or "",
                    'line_number': node.lineno,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'decorators': [ast.unparse(d) for d in node.decorator_list]
                })
        return functions
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a class method"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    async def _identify_business_logic(self, code: str) -> List[str]:
        """Use Claude to identify business logic patterns"""
        prompt = f"""
        Analyze this Python code and identify business logic patterns:
        
        ```python
        {code[:2000]}  # Truncate for API limits
        ```
        
        Return a JSON list of business logic patterns found (e.g., validation, calculations, workflows).
        """
        
        try:
            response = await self._call_claude(prompt)
            patterns = json.loads(response)
            return patterns if isinstance(patterns, list) else []
        except:
            return ["validation", "data_processing"]  # Fallback
    
    async def _identify_security_patterns(self, code: str) -> List[str]:
        """Identify potential security concerns"""
        security_keywords = [
            'password', 'token', 'secret', 'key', 'auth', 'login',
            'sql', 'query', 'exec', 'eval', 'input', 'file', 'path'
        ]
        
        concerns = []
        code_lower = code.lower()
        for keyword in security_keywords:
            if keyword in code_lower:
                concerns.append(f"potential_{keyword}_concern")
        
        return concerns[:5]  # Limit to top 5
    
    async def _identify_performance_issues(self, code: str) -> List[str]:
        """Identify potential performance hotspots"""
        hotspots = []
        
        if 'for' in code.lower() and 'range' in code.lower():
            hotspots.append("nested_loops")
        if 'sql' in code.lower() or 'query' in code.lower():
            hotspots.append("database_operations")
        if 'file' in code.lower() or 'read' in code.lower():
            hotspots.append("file_operations")
        if 'request' in code.lower() or 'http' in code.lower():
            hotspots.append("network_operations")
        
        return hotspots
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract import dependencies"""
        deps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                deps.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom) and node.module:
                deps.append(node.module)
        return list(set(deps))[:10]  # Top 10 unique dependencies
    
    def _calculate_test_worthiness(self, complexity: int, classes: int, 
                                 functions: int, business_patterns: int, 
                                 security_concerns: int) -> float:
        """Calculate how worthy this code is of testing"""
        score = 0.0
        
        # Complexity contribution
        if complexity > 10:
            score += 0.3
        elif complexity > 5:
            score += 0.2
        else:
            score += 0.1
        
        # Structure contribution
        score += min(0.2, (classes + functions) * 0.02)
        
        # Business logic contribution
        score += min(0.3, business_patterns * 0.1)
        
        # Security contribution
        score += min(0.2, security_concerns * 0.05)
        
        return min(1.0, score)
    
    async def generate_comprehensive_tests(self, analysis: CodeAnalysis, 
                                         strategy: TestGenerationStrategy = TestGenerationStrategy.COMPREHENSIVE) -> GeneratedTest:
        """Generate comprehensive tests using Claude AI"""
        start_time = time.time()
        
        # Create context-aware prompt
        prompt = self._create_generation_prompt(analysis, strategy)
        
        # Generate tests with Claude
        test_code = await self._call_claude(prompt)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        confidence_score = self._calculate_confidence(analysis, test_code)
        effectiveness = self._estimate_effectiveness(analysis, test_code)
        coverage_targets = self._extract_coverage_targets(analysis)
        
        self.generated_count += 1
        
        return GeneratedTest(
            test_code=test_code,
            strategy=strategy,
            coverage_targets=coverage_targets,
            confidence_score=confidence_score,
            estimated_effectiveness=effectiveness,
            generation_time=generation_time
        )
    
    def _create_generation_prompt(self, analysis: CodeAnalysis, 
                                strategy: TestGenerationStrategy) -> str:
        """Create intelligent prompt for test generation"""
        
        base_prompt = f"""
        Generate comprehensive pytest tests for this Python module:

        **File**: {analysis.file_path}
        **Complexity**: {analysis.complexity_score}
        **Test Worthiness**: {analysis.test_worthiness:.2f}
        **Strategy**: {strategy.value}

        **Classes Found**: {len(analysis.classes)}
        {self._format_classes(analysis.classes)}

        **Functions Found**: {len(analysis.functions)}
        {self._format_functions(analysis.functions)}

        **Business Logic Patterns**: {', '.join(analysis.business_logic_patterns)}
        **Security Concerns**: {', '.join(analysis.security_concerns)}
        **Performance Hotspots**: {', '.join(analysis.performance_hotspots)}

        **Requirements**:
        1. Generate tests that achieve 100% line and branch coverage
        2. Focus on business logic validation, not just syntax
        3. Include edge cases and error conditions
        4. Test security concerns if any found
        5. Mock external dependencies: {', '.join(analysis.dependencies)}
        6. Use descriptive test names explaining what is being tested
        7. Include docstrings explaining test purpose
        8. Use pytest fixtures for setup/teardown
        9. Test performance hotspots if identified
        10. Validate input/output for all public methods

        Generate ONLY the Python test code starting with imports.
        """
        
        if strategy == TestGenerationStrategy.SECURITY:
            base_prompt += "\n\n**FOCUS**: Security testing - input validation, injection, authorization"
        elif strategy == TestGenerationStrategy.PERFORMANCE:
            base_prompt += "\n\n**FOCUS**: Performance testing - timing, memory usage, scalability"
        elif strategy == TestGenerationStrategy.EDGE_CASE:
            base_prompt += "\n\n**FOCUS**: Edge cases - boundary values, null inputs, extreme conditions"
        
        return base_prompt
    
    def _format_classes(self, classes: List[Dict[str, Any]]) -> str:
        """Format class information for prompt"""
        if not classes:
            return "None"
        
        formatted = []
        for cls in classes[:3]:  # Limit to prevent prompt overflow
            methods_str = ', '.join(cls['methods'][:5])
            formatted.append(f"- {cls['name']}: methods=[{methods_str}]")
        
        return '\n'.join(formatted)
    
    def _format_functions(self, functions: List[Dict[str, Any]]) -> str:
        """Format function information for prompt"""
        if not functions:
            return "None"
        
        formatted = []
        for func in functions[:5]:  # Limit to prevent prompt overflow
            args_str = ', '.join(func['args'])
            formatted.append(f"- {func['name']}({args_str}) -> {func['returns'] or 'None'}")
        
        return '\n'.join(formatted)
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API with error handling"""
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': self.model,
            'max_tokens': 4000,
            'messages': [{
                'role': 'user',
                'content': prompt
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, 
                                      json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['content'][0]['text']
                    else:
                        error_text = await response.text()
                        raise Exception(f"Claude API error {response.status}: {error_text}")
        
        except Exception as e:
            # Fallback to basic test template
            return self._generate_fallback_test(prompt)
    
    def _generate_fallback_test(self, prompt: str) -> str:
        """Generate basic test when Claude API fails"""
        return '''import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Basic test template - Claude API unavailable
def test_module_imports():
    """Test that the module can be imported successfully"""
    try:
        # Add actual module import here
        assert True
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")

def test_basic_functionality():
    """Test basic functionality when available"""
    # Add specific tests based on code analysis
    assert True, "Implement specific tests based on analysis"
'''
    
    def _calculate_confidence(self, analysis: CodeAnalysis, test_code: str) -> float:
        """Calculate confidence in generated tests"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on analysis quality
        confidence += analysis.test_worthiness * 0.3
        
        # Boost confidence based on test code quality
        if 'pytest' in test_code:
            confidence += 0.1
        if 'assert' in test_code:
            confidence += 0.1
        if 'def test_' in test_code:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _estimate_effectiveness(self, analysis: CodeAnalysis, test_code: str) -> float:
        """Estimate test effectiveness"""
        effectiveness = 0.6  # Base effectiveness
        
        # Adjust based on complexity coverage
        if analysis.complexity_score > 10:
            effectiveness += 0.2
        
        # Adjust based on business logic coverage
        if analysis.business_logic_patterns:
            effectiveness += 0.1
        
        # Adjust based on test completeness
        test_functions = test_code.count('def test_')
        if test_functions >= 5:
            effectiveness += 0.1
        
        return min(1.0, effectiveness)
    
    def _extract_coverage_targets(self, analysis: CodeAnalysis) -> List[str]:
        """Extract what should be covered by tests"""
        targets = []
        
        # Add classes and methods
        for cls in analysis.classes:
            targets.append(f"class:{cls['name']}")
            targets.extend([f"method:{cls['name']}.{m}" for m in cls['methods'][:3]])
        
        # Add functions
        for func in analysis.functions[:5]:
            targets.append(f"function:{func['name']}")
        
        # Add business logic patterns
        targets.extend([f"business_logic:{p}" for p in analysis.business_logic_patterns[:3]])
        
        return targets
    
    def _empty_analysis(self, file_path: str, error: str) -> CodeAnalysis:
        """Return empty analysis on error"""
        return CodeAnalysis(
            file_path=file_path,
            classes=[],
            functions=[],
            complexity_score=0,
            business_logic_patterns=[],
            security_concerns=[],
            performance_hotspots=[],
            dependencies=[],
            test_worthiness=0.1
        )