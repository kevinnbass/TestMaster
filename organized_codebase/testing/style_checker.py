"""
Documentation Style Checker

This module provides comprehensive style checking for documentation compliance
across different formats and style guides.
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json


class StyleGuide(Enum):
    """Supported documentation style guides."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    EPYTEXT = "epytext"
    PEP257 = "pep257"
    MARKDOWN = "markdown"
    RESTRUCTURED_TEXT = "rst"
    CUSTOM = "custom"


class SeverityLevel(Enum):
    """Severity levels for style violations."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class StyleViolation:
    """Represents a style violation in documentation."""
    rule_id: str
    severity: SeverityLevel
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class StyleCheckResult:
    """Results of style checking."""
    file_path: str
    style_guide: StyleGuide
    violations: List[StyleViolation]
    total_lines: int
    compliant_lines: int
    compliance_score: float
    
    def __post_init__(self):
        if not self.violations:
            self.compliance_score = 100.0
        else:
            # Calculate compliance score based on violations
            error_count = sum(1 for v in self.violations if v.severity == SeverityLevel.ERROR)
            warning_count = sum(1 for v in self.violations if v.severity == SeverityLevel.WARNING)
            info_count = sum(1 for v in self.violations if v.severity == SeverityLevel.INFO)
            
            # Weight violations by severity
            total_weight = error_count * 3 + warning_count * 2 + info_count * 1
            max_weight = self.total_lines * 3  # Maximum possible violations
            
            self.compliance_score = max(0, (1 - total_weight / max_weight) * 100) if max_weight > 0 else 100.0


class DocumentationStyleChecker:
    """
    Comprehensive style checker for documentation compliance.
    """
    
    def __init__(self):
        """Initialize the documentation style checker."""
        self.rules = {}
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize style checking rules for different guides."""
        self.rules = {
            StyleGuide.GOOGLE: self._get_google_rules(),
            StyleGuide.NUMPY: self._get_numpy_rules(),
            StyleGuide.SPHINX: self._get_sphinx_rules(),
            StyleGuide.PEP257: self._get_pep257_rules(),
            StyleGuide.MARKDOWN: self._get_markdown_rules(),
            StyleGuide.RESTRUCTURED_TEXT: self._get_rst_rules(),
        }
    
    def _get_google_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get Google style guide rules."""
        return {
            "G001": {
                "description": "Docstring should start with a brief one-line summary",
                "severity": SeverityLevel.ERROR,
                "pattern": r'^"""[^.]*\.$',
                "check_function": self._check_google_summary
            },
            "G002": {
                "description": "Args section should use proper format",
                "severity": SeverityLevel.ERROR,
                "pattern": r'Args:\s*\n(\s+\w+\s*\([^)]+\):\s*.+\n)*',
                "check_function": self._check_google_args
            },
            "G003": {
                "description": "Returns section should specify type and description",
                "severity": SeverityLevel.ERROR,
                "pattern": r'Returns:\s*\n\s+[^:]+:\s*.+',
                "check_function": self._check_google_returns
            },
            "G004": {
                "description": "Raises section should list exceptions",
                "severity": SeverityLevel.WARNING,
                "pattern": r'Raises:\s*\n(\s+\w+:\s*.+\n)*',
                "check_function": self._check_google_raises
            },
            "G005": {
                "description": "Examples should be properly formatted",
                "severity": SeverityLevel.INFO,
                "pattern": r'Examples?:\s*\n(\s+>>> .+\n)*',
                "check_function": self._check_google_examples
            },
            "G006": {
                "description": "Proper indentation in docstring sections",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_google_indentation
            },
            "G007": {
                "description": "No trailing whitespace in docstring",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_trailing_whitespace
            }
        }
    
    def _get_numpy_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get NumPy style guide rules."""
        return {
            "N001": {
                "description": "Section headers should be underlined with dashes",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_numpy_headers
            },
            "N002": {
                "description": "Parameters section should use proper format",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_numpy_parameters
            },
            "N003": {
                "description": "Returns section should specify type",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_numpy_returns
            },
            "N004": {
                "description": "Notes section should be properly formatted",
                "severity": SeverityLevel.INFO,
                "check_function": self._check_numpy_notes
            },
            "N005": {
                "description": "References section should be properly formatted",
                "severity": SeverityLevel.INFO,
                "check_function": self._check_numpy_references
            }
        }
    
    def _get_sphinx_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get Sphinx style guide rules."""
        return {
            "S001": {
                "description": "Parameters should use :param: directive",
                "severity": SeverityLevel.ERROR,
                "pattern": r':param\s+\w+:\s*.+',
                "check_function": self._check_sphinx_params
            },
            "S002": {
                "description": "Types should use :type: directive",
                "severity": SeverityLevel.WARNING,
                "pattern": r':type\s+\w+:\s*.+',
                "check_function": self._check_sphinx_types
            },
            "S003": {
                "description": "Return value should use :returns: directive",
                "severity": SeverityLevel.ERROR,
                "pattern": r':returns?:\s*.+',
                "check_function": self._check_sphinx_returns
            },
            "S004": {
                "description": "Return type should use :rtype: directive",
                "severity": SeverityLevel.WARNING,
                "pattern": r':rtype:\s*.+',
                "check_function": self._check_sphinx_rtype
            },
            "S005": {
                "description": "Exceptions should use :raises: directive",
                "severity": SeverityLevel.WARNING,
                "pattern": r':raises?\s+\w+:\s*.+',
                "check_function": self._check_sphinx_raises
            }
        }
    
    def _get_pep257_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get PEP 257 style guide rules."""
        return {
            "P001": {
                "description": "Docstring should be a triple-quoted string",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_pep257_quotes
            },
            "P002": {
                "description": "One-line docstring should be on one line",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_pep257_oneline
            },
            "P003": {
                "description": "Multi-line docstring summary should be on first line",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_pep257_multiline
            },
            "P004": {
                "description": "Docstring should end with a period",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_pep257_period
            },
            "P005": {
                "description": "Blank line after summary in multi-line docstring",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_pep257_blank_line
            }
        }
    
    def _get_markdown_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get Markdown style guide rules."""
        return {
            "M001": {
                "description": "Headers should use ATX style (#)",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_markdown_headers
            },
            "M002": {
                "description": "Lists should be consistent (- or *)",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_markdown_lists
            },
            "M003": {
                "description": "Code blocks should be fenced with ```",
                "severity": SeverityLevel.INFO,
                "check_function": self._check_markdown_code_blocks
            },
            "M004": {
                "description": "Links should use reference style when repeated",
                "severity": SeverityLevel.INFO,
                "check_function": self._check_markdown_links
            },
            "M005": {
                "description": "No trailing spaces on lines",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_trailing_whitespace
            }
        }
    
    def _get_rst_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get reStructuredText style guide rules."""
        return {
            "R001": {
                "description": "Section headers should be properly underlined",
                "severity": SeverityLevel.ERROR,
                "check_function": self._check_rst_headers
            },
            "R002": {
                "description": "Code blocks should use proper directives",
                "severity": SeverityLevel.WARNING,
                "check_function": self._check_rst_code_blocks
            },
            "R003": {
                "description": "Cross-references should use proper syntax",
                "severity": SeverityLevel.INFO,
                "check_function": self._check_rst_references
            }
        }
    
    def check_file(self, file_path: str, style_guide: StyleGuide) -> StyleCheckResult:
        """
        Check a file for style compliance.
        
        Args:
            file_path: Path to the file to check
            style_guide: Style guide to check against
            
        Returns:
            Style check results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, UnicodeDecodeError) as e:
            return StyleCheckResult(
                file_path=file_path,
                style_guide=style_guide,
                violations=[StyleViolation(
                    rule_id="FILE_ERROR",
                    severity=SeverityLevel.ERROR,
                    message=f"Could not read file: {e}"
                )],
                total_lines=0,
                compliant_lines=0,
                compliance_score=0.0
            )
        
        lines = content.split('\n')
        violations = []
        
        # Get rules for the style guide
        rules = self.rules.get(style_guide, {})
        
        # Check each rule
        for rule_id, rule_config in rules.items():
            rule_violations = self._check_rule(content, lines, rule_id, rule_config)
            violations.extend(rule_violations)
        
        return StyleCheckResult(
            file_path=file_path,
            style_guide=style_guide,
            violations=violations,
            total_lines=len(lines),
            compliant_lines=len(lines) - len(violations)
        )
    
    def check_docstring(self, docstring: str, style_guide: StyleGuide) -> StyleCheckResult:
        """
        Check a docstring for style compliance.
        
        Args:
            docstring: Docstring content to check
            style_guide: Style guide to check against
            
        Returns:
            Style check results
        """
        lines = docstring.split('\n')
        violations = []
        
        # Get rules for the style guide
        rules = self.rules.get(style_guide, {})
        
        # Check each rule
        for rule_id, rule_config in rules.items():
            rule_violations = self._check_rule(docstring, lines, rule_id, rule_config)
            violations.extend(rule_violations)
        
        return StyleCheckResult(
            file_path="<docstring>",
            style_guide=style_guide,
            violations=violations,
            total_lines=len(lines),
            compliant_lines=len(lines) - len(violations)
        )
    
    def _check_rule(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check a specific rule against content."""
        violations = []
        
        # Use check function if available
        if 'check_function' in rule_config:
            check_func = rule_config['check_function']
            rule_violations = check_func(content, lines, rule_id, rule_config)
            violations.extend(rule_violations)
        
        # Use pattern if available
        elif 'pattern' in rule_config:
            pattern = rule_config['pattern']
            if not re.search(pattern, content, re.MULTILINE):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=rule_config['description']
                ))
        
        return violations
    
    # Google Style Checking Functions
    
    def _check_google_summary(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style summary format."""
        violations = []
        
        if not content.strip():
            return violations
        
        # Find docstring start
        docstring_match = re.search(r'"""([^"]*(?:"[^"]*"[^"]*)*)"""', content, re.DOTALL)
        if not docstring_match:
            return violations
        
        docstring_content = docstring_match.group(1).strip()
        if not docstring_content:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="Docstring is empty",
                line_number=1
            ))
            return violations
        
        first_line = docstring_content.split('\n')[0].strip()
        
        # Check if first line is a proper summary
        if not first_line:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="First line should be a brief summary",
                line_number=1
            ))
        elif not first_line.endswith('.'):
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=SeverityLevel.WARNING,
                message="Summary should end with a period",
                line_number=1,
                suggestion=f"{first_line}."
            ))
        elif len(first_line) > 79:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=SeverityLevel.WARNING,
                message="Summary line is too long (>79 characters)",
                line_number=1
            ))
        
        return violations
    
    def _check_google_args(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style Args section format."""
        violations = []
        
        # Look for Args section
        args_match = re.search(r'Args?:\s*\n((?:\s+.+\n)*)', content, re.MULTILINE)
        if not args_match:
            return violations  # No Args section found, might be intentional
        
        args_section = args_match.group(1)
        args_lines = args_section.strip().split('\n')
        
        for i, line in enumerate(args_lines):
            line = line.strip()
            if not line:
                continue
            
            # Check parameter format: name (type): description
            if not re.match(r'\w+\s*\([^)]+\):\s*.+', line):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=f"Invalid parameter format: {line}",
                    line_number=i + 1,
                    suggestion="Use format: param_name (type): description"
                ))
        
        return violations
    
    def _check_google_returns(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style Returns section format."""
        violations = []
        
        # Look for Returns section
        returns_match = re.search(r'Returns?:\s*\n((?:\s+.+\n)*)', content, re.MULTILINE)
        if not returns_match:
            # Check if function actually returns something
            if 'return ' in content or 'yield ' in content:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=SeverityLevel.WARNING,
                    message="Function returns value but has no Returns section"
                ))
            return violations
        
        returns_section = returns_match.group(1).strip()
        
        # Check format: type: description
        if not re.match(r'\s*[^:]+:\s*.+', returns_section):
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="Returns section should specify type and description",
                suggestion="Use format: type: description"
            ))
        
        return violations
    
    def _check_google_raises(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style Raises section format."""
        violations = []
        
        # Look for Raises section
        raises_match = re.search(r'Raises?:\s*\n((?:\s+.+\n)*)', content, re.MULTILINE)
        if not raises_match:
            # Check if function raises exceptions
            if 'raise ' in content:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=SeverityLevel.INFO,
                    message="Function raises exceptions but has no Raises section"
                ))
            return violations
        
        raises_section = raises_match.group(1)
        raises_lines = raises_section.strip().split('\n')
        
        for i, line in enumerate(raises_lines):
            line = line.strip()
            if not line:
                continue
            
            # Check exception format: ExceptionType: description
            if not re.match(r'\w+:\s*.+', line):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=f"Invalid exception format: {line}",
                    line_number=i + 1,
                    suggestion="Use format: ExceptionType: description"
                ))
        
        return violations
    
    def _check_google_examples(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style Examples section format."""
        violations = []
        
        # Look for Examples section
        examples_match = re.search(r'Examples?:\s*\n((?:\s+.+\n)*)', content, re.MULTILINE)
        if not examples_match:
            return violations  # No Examples section found
        
        examples_section = examples_match.group(1)
        
        # Check for proper doctest format
        if '>>>' in examples_section:
            # Validate doctest syntax
            example_lines = examples_section.split('\n')
            for i, line in enumerate(example_lines):
                line = line.strip()
                if line.startswith('>>>'):
                    # Check if it's valid Python
                    try:
                        code = line[3:].strip()
                        if code:
                            compile(code, '<doctest>', 'eval')
                    except SyntaxError:
                        try:
                            compile(code, '<doctest>', 'exec')
                        except SyntaxError:
                            violations.append(StyleViolation(
                                rule_id=rule_id,
                                severity=SeverityLevel.WARNING,
                                message=f"Invalid Python syntax in example: {line}",
                                line_number=i + 1
                            ))
        
        return violations
    
    def _check_google_indentation(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Google style indentation."""
        violations = []
        
        # Find docstring boundaries
        docstring_match = re.search(r'"""([^"]*(?:"[^"]*"[^"]*)*)"""', content, re.DOTALL)
        if not docstring_match:
            return violations
        
        docstring_lines = docstring_match.group(1).split('\n')
        
        # Check indentation consistency
        section_patterns = [
            r'Args?:\s*$',
            r'Returns?:\s*$',
            r'Raises?:\s*$',
            r'Examples?:\s*$',
            r'Notes?:\s*$'
        ]
        
        in_section = False
        expected_indent = None
        
        for i, line in enumerate(docstring_lines[1:], 2):  # Skip first line
            if not line.strip():
                continue
            
            # Check if this is a section header
            is_section_header = any(re.match(pattern, line.strip()) for pattern in section_patterns)
            
            if is_section_header:
                in_section = True
                expected_indent = None
                continue
            
            if in_section:
                current_indent = len(line) - len(line.lstrip())
                
                if expected_indent is None:
                    if current_indent == 0:
                        violations.append(StyleViolation(
                            rule_id=rule_id,
                            severity=rule_config['severity'],
                            message="Section content should be indented",
                            line_number=i
                        ))
                    else:
                        expected_indent = current_indent
                elif current_indent != expected_indent and line.strip():
                    violations.append(StyleViolation(
                        rule_id=rule_id,
                        severity=rule_config['severity'],
                        message=f"Inconsistent indentation (expected {expected_indent}, got {current_indent})",
                        line_number=i
                    ))
        
        return violations
    
    # NumPy Style Checking Functions
    
    def _check_numpy_headers(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check NumPy style section headers."""
        violations = []
        
        # Look for section headers
        section_headers = ['Parameters', 'Returns', 'Yields', 'Raises', 'See Also', 'Notes', 'References', 'Examples']
        
        for header in section_headers:
            header_pattern = f'^{header}\\s*$'
            header_matches = list(re.finditer(header_pattern, content, re.MULTILINE))
            
            for match in header_matches:
                line_num = content[:match.start()].count('\n') + 1
                
                # Check if next line is underlined with dashes
                if line_num < len(lines):
                    next_line = lines[line_num] if line_num < len(lines) else ""
                    if not re.match(r'^-+\s*$', next_line):
                        violations.append(StyleViolation(
                            rule_id=rule_id,
                            severity=rule_config['severity'],
                            message=f"Section '{header}' should be underlined with dashes",
                            line_number=line_num + 1,
                            suggestion="-" * len(header)
                        ))
                    elif len(next_line.strip()) != len(header):
                        violations.append(StyleViolation(
                            rule_id=rule_id,
                            severity=SeverityLevel.WARNING,
                            message=f"Underline length should match header length",
                            line_number=line_num + 1,
                            suggestion="-" * len(header)
                        ))
        
        return violations
    
    def _check_numpy_parameters(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check NumPy style Parameters section."""
        violations = []
        
        # Look for Parameters section
        params_match = re.search(r'Parameters\s*\n-+\s*\n((?:(?!^\w+\s*\n-+).*\n)*)', content, re.MULTILINE)
        if not params_match:
            return violations
        
        params_section = params_match.group(1)
        param_blocks = re.split(r'\n(?=\w+\s*:)', params_section)
        
        for block in param_blocks:
            block = block.strip()
            if not block:
                continue
            
            # Check parameter format: name : type
            first_line = block.split('\n')[0]
            if ':' not in first_line:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=f"Parameter should specify type: {first_line}",
                    suggestion="Use format: param_name : type"
                ))
        
        return violations
    
    def _check_numpy_returns(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check NumPy style Returns section."""
        violations = []
        
        # Look for Returns section
        returns_match = re.search(r'Returns\s*\n-+\s*\n((?:(?!^\w+\s*\n-+).*\n)*)', content, re.MULTILINE)
        if not returns_match:
            return violations
        
        returns_section = returns_match.group(1).strip()
        
        # Check format
        if not re.match(r'\w+\s*\n\s+.*', returns_section):
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="Returns section should specify type and description",
                suggestion="Format:\ntype\n    description"
            ))
        
        return violations
    
    def _check_numpy_notes(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check NumPy style Notes section."""
        violations = []
        
        # Look for Notes section
        notes_match = re.search(r'Notes\s*\n-+\s*\n((?:(?!^\w+\s*\n-+).*\n)*)', content, re.MULTILINE)
        if not notes_match:
            return violations
        
        notes_section = notes_match.group(1)
        
        # Check for proper formatting (general guidelines)
        if not notes_section.strip():
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=SeverityLevel.WARNING,
                message="Notes section is empty"
            ))
        
        return violations
    
    def _check_numpy_references(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check NumPy style References section."""
        violations = []
        
        # Look for References section
        refs_match = re.search(r'References\s*\n-+\s*\n((?:(?!^\w+\s*\n-+).*\n)*)', content, re.MULTILINE)
        if not refs_match:
            return violations
        
        refs_section = refs_match.group(1)
        ref_lines = [line.strip() for line in refs_section.split('\n') if line.strip()]
        
        # Check reference format
        for i, ref_line in enumerate(ref_lines):
            # Basic check for citation format
            if not (ref_line.startswith('..') or re.match(r'\[\d+\]', ref_line) or ref_line.startswith('http')):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=SeverityLevel.INFO,
                    message=f"Reference format may be incorrect: {ref_line}",
                    line_number=i + 1
                ))
        
        return violations
    
    # Sphinx Style Checking Functions
    
    def _check_sphinx_params(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Sphinx style parameter documentation."""
        violations = []
        
        # Look for :param: directives
        param_matches = re.findall(r':param\s+(\w+):\s*(.+)', content)
        
        for param_name, param_desc in param_matches:
            if not param_desc.strip():
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=f"Parameter '{param_name}' has no description"
                ))
        
        return violations
    
    def _check_sphinx_types(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Sphinx style type documentation."""
        violations = []
        
        # Find all :param: directives
        param_names = set(re.findall(r':param\s+(\w+):', content))
        
        # Find all :type: directives
        type_names = set(re.findall(r':type\s+(\w+):', content))
        
        # Check for missing type information
        missing_types = param_names - type_names
        for param_name in missing_types:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message=f"Parameter '{param_name}' missing type information",
                suggestion=f":type {param_name}: type_here"
            ))
        
        return violations
    
    def _check_sphinx_returns(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Sphinx style return documentation."""
        violations = []
        
        # Look for :returns: or :return: directive
        returns_match = re.search(r':returns?:\s*(.+)', content)
        if not returns_match:
            if 'return ' in content:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=SeverityLevel.WARNING,
                    message="Function returns value but has no :returns: directive"
                ))
            return violations
        
        return_desc = returns_match.group(1).strip()
        if not return_desc:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message=":returns: directive has no description"
            ))
        
        return violations
    
    def _check_sphinx_rtype(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Sphinx style return type documentation."""
        violations = []
        
        # If there's a :returns: directive, check for :rtype:
        if re.search(r':returns?:', content):
            if not re.search(r':rtype:', content):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message="Return value should have type specified with :rtype:"
                ))
        
        return violations
    
    def _check_sphinx_raises(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Sphinx style exception documentation."""
        violations = []
        
        # Look for :raises: directives
        raises_matches = re.findall(r':raises?\s+(\w+):\s*(.+)', content)
        
        for exception_type, exception_desc in raises_matches:
            if not exception_desc.strip():
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message=f"Exception '{exception_type}' has no description"
                ))
        
        return violations
    
    # PEP 257 Style Checking Functions
    
    def _check_pep257_quotes(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check PEP 257 triple quote requirement."""
        violations = []
        
        # Check for proper triple quotes
        if not (content.strip().startswith('"""') and content.strip().endswith('"""')):
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="Docstring should use triple double quotes"
            ))
        
        return violations
    
    def _check_pep257_oneline(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check PEP 257 one-line docstring format."""
        violations = []
        
        # Check if it's a one-line docstring
        docstring_content = content.strip()[3:-3].strip()  # Remove triple quotes
        
        if '\n' not in docstring_content:  # One-line docstring
            # Should fit on one line with quotes
            full_line = f'"""{docstring_content}"""'
            if len(full_line) <= 79:  # Standard line length
                # Check if it's actually formatted as one line
                if content.count('\n') > 0:
                    violations.append(StyleViolation(
                        rule_id=rule_id,
                        severity=rule_config['severity'],
                        message="One-line docstring should be on a single line"
                    ))
        
        return violations
    
    def _check_pep257_multiline(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check PEP 257 multi-line docstring format."""
        violations = []
        
        lines = content.split('\n')
        if len(lines) > 1:  # Multi-line docstring
            first_line = lines[0].strip()
            
            # First line should contain the opening quotes and summary
            if not first_line.startswith('"""'):
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message="Multi-line docstring should start with summary on first line"
                ))
            
            # Check for summary on first line
            summary = first_line[3:].strip()  # Remove opening quotes
            if not summary:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message="Summary should be on the first line"
                ))
        
        return violations
    
    def _check_pep257_period(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check PEP 257 period at end of summary."""
        violations = []
        
        # Extract summary (first sentence)
        docstring_content = content.strip()[3:-3].strip()
        if not docstring_content:
            return violations
        
        first_line = docstring_content.split('\n')[0].strip()
        if first_line and not first_line.endswith('.'):
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message="Docstring summary should end with a period"
            ))
        
        return violations
    
    def _check_pep257_blank_line(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check PEP 257 blank line after summary."""
        violations = []
        
        lines = content.split('\n')
        if len(lines) > 2:  # Multi-line with content after summary
            # Check if there's a blank line after the summary
            if lines[1].strip() != '':
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message="Multi-line docstring should have blank line after summary"
                ))
        
        return violations
    
    # Markdown Style Checking Functions
    
    def _check_markdown_headers(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Markdown header style (ATX vs Setext)."""
        violations = []
        
        for i, line in enumerate(lines):
            # Check for Setext-style headers (underlined with = or -)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^[=-]+\s*$', next_line):
                    violations.append(StyleViolation(
                        rule_id=rule_id,
                        severity=rule_config['severity'],
                        message="Use ATX-style headers (#) instead of Setext-style",
                        line_number=i + 1,
                        suggestion=f"# {line}"
                    ))
        
        return violations
    
    def _check_markdown_lists(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Markdown list consistency."""
        violations = []
        
        list_markers = set()
        
        for i, line in enumerate(lines):
            # Check for list items
            list_match = re.match(r'^(\s*)([*+-])\s+', line)
            if list_match:
                marker = list_match.group(2)
                list_markers.add(marker)
        
        # Check for consistency
        if len(list_markers) > 1:
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message=f"Inconsistent list markers: {', '.join(list_markers)}",
                suggestion="Use consistent list markers (- or *)"
            ))
        
        return violations
    
    def _check_markdown_code_blocks(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Markdown code block style."""
        violations = []
        
        for i, line in enumerate(lines):
            # Check for indented code blocks (4 spaces)
            if re.match(r'^    \S', line):
                # Look for surrounding fenced code blocks
                has_fenced_blocks = any('```' in l for l in lines)
                if has_fenced_blocks:
                    violations.append(StyleViolation(
                        rule_id=rule_id,
                        severity=rule_config['severity'],
                        message="Use fenced code blocks (```) instead of indented blocks",
                        line_number=i + 1
                    ))
        
        return violations
    
    def _check_markdown_links(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check Markdown link style."""
        violations = []
        
        # Find all inline links
        inline_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        # Check for repeated URLs
        url_counts = {}
        for text, url in inline_links:
            url_counts[url] = url_counts.get(url, 0) + 1
        
        repeated_urls = {url: count for url, count in url_counts.items() if count > 1}
        
        for url, count in repeated_urls.items():
            violations.append(StyleViolation(
                rule_id=rule_id,
                severity=rule_config['severity'],
                message=f"URL '{url}' is repeated {count} times, consider using reference-style links"
            ))
        
        return violations
    
    # reStructuredText Style Checking Functions
    
    def _check_rst_headers(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check reStructuredText header format."""
        violations = []
        
        for i, line in enumerate(lines):
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Check for header underlines
                if re.match(r'^[=\-`\':\"~\^_\*\+#<>]+\s*$', next_line):
                    # Check if underline length matches header length
                    if len(next_line.strip()) != len(line.strip()):
                        violations.append(StyleViolation(
                            rule_id=rule_id,
                            severity=rule_config['severity'],
                            message="Header underline length should match header text length",
                            line_number=i + 2,
                            suggestion="=" * len(line.strip())
                        ))
        
        return violations
    
    def _check_rst_code_blocks(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check reStructuredText code block directives."""
        violations = []
        
        # Look for code blocks
        code_block_pattern = r'::\s*$'
        
        for i, line in enumerate(lines):
            if re.search(code_block_pattern, line):
                # Check if it's a proper directive
                if not line.strip().startswith('.. code'):
                    violations.append(StyleViolation(
                        rule_id=rule_id,
                        severity=rule_config['severity'],
                        message="Consider using explicit code directive (.. code-block::)",
                        line_number=i + 1,
                        suggestion=".. code-block:: python"
                    ))
        
        return violations
    
    def _check_rst_references(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check reStructuredText cross-reference syntax."""
        violations = []
        
        # Look for potential cross-references
        ref_patterns = [
            r':ref:`[^`]+`',
            r':doc:`[^`]+`',
            r':class:`[^`]+`',
            r':func:`[^`]+`',
            r':meth:`[^`]+`'
        ]
        
        # This is a basic check - in practice, you'd validate against actual targets
        for pattern in ref_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Check for valid reference format
                ref_content = match.group(0)
                if '|' in ref_content:  # Custom text
                    parts = ref_content.split('|')
                    if len(parts) != 2:
                        violations.append(StyleViolation(
                            rule_id=rule_id,
                            severity=rule_config['severity'],
                            message=f"Invalid reference format: {ref_content}"
                        ))
        
        return violations
    
    # Common Style Checking Functions
    
    def _check_trailing_whitespace(
        self, 
        content: str, 
        lines: List[str], 
        rule_id: str, 
        rule_config: Dict[str, Any]
    ) -> List[StyleViolation]:
        """Check for trailing whitespace."""
        violations = []
        
        for i, line in enumerate(lines):
            if line.rstrip() != line:
                violations.append(StyleViolation(
                    rule_id=rule_id,
                    severity=rule_config['severity'],
                    message="Line has trailing whitespace",
                    line_number=i + 1
                ))
        
        return violations
    
    def generate_style_report(self, results: List[StyleCheckResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive style report from multiple check results.
        
        Args:
            results: List of style check results
            
        Returns:
            Comprehensive style report
        """
        total_files = len(results)
        total_violations = sum(len(result.violations) for result in results)
        
        # Calculate overall compliance score
        if total_files > 0:
            overall_score = sum(result.compliance_score for result in results) / total_files
        else:
            overall_score = 100.0
        
        # Count violations by severity
        severity_counts = {
            SeverityLevel.ERROR: 0,
            SeverityLevel.WARNING: 0,
            SeverityLevel.INFO: 0,
            SeverityLevel.HINT: 0
        }
        
        for result in results:
            for violation in result.violations:
                severity_counts[violation.severity] += 1
        
        # Most common violations
        rule_counts = {}
        for result in results:
            for violation in result.violations:
                rule_counts[violation.rule_id] = rule_counts.get(violation.rule_id, 0) + 1
        
        most_common_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Files with most violations
        files_by_violations = sorted(results, key=lambda x: len(x.violations), reverse=True)[:10]
        
        return {
            'summary': {
                'total_files': total_files,
                'total_violations': total_violations,
                'overall_compliance_score': round(overall_score, 2),
                'files_with_violations': sum(1 for r in results if r.violations)
            },
            'violations_by_severity': {
                severity.value: count for severity, count in severity_counts.items()
            },
            'most_common_violations': [
                {'rule_id': rule_id, 'count': count} for rule_id, count in most_common_rules
            ],
            'files_with_most_violations': [
                {
                    'file_path': result.file_path,
                    'violation_count': len(result.violations),
                    'compliance_score': round(result.compliance_score, 2)
                }
                for result in files_by_violations
            ],
            'compliance_distribution': {
                'excellent': sum(1 for r in results if r.compliance_score >= 90),
                'good': sum(1 for r in results if 70 <= r.compliance_score < 90),
                'fair': sum(1 for r in results if 50 <= r.compliance_score < 70),
                'poor': sum(1 for r in results if r.compliance_score < 50)
            }
        }
    
    def suggest_fixes(self, violations: List[StyleViolation]) -> List[Dict[str, Any]]:
        """
        Suggest automatic fixes for style violations.
        
        Args:
            violations: List of style violations
            
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        for violation in violations:
            fix = {
                'rule_id': violation.rule_id,
                'line_number': violation.line_number,
                'original_issue': violation.message,
                'suggested_fix': violation.suggestion,
                'auto_fixable': violation.suggestion is not None,
                'severity': violation.severity.value
            }
            
            # Add specific fix instructions based on rule type
            if violation.rule_id.startswith('G'):  # Google style
                fix['style_guide'] = 'Google'
                fix['reference_url'] = 'https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings'
            elif violation.rule_id.startswith('N'):  # NumPy style
                fix['style_guide'] = 'NumPy'
                fix['reference_url'] = 'https://numpydoc.readthedocs.io/en/latest/format.html'
            elif violation.rule_id.startswith('S'):  # Sphinx style
                fix['style_guide'] = 'Sphinx'
                fix['reference_url'] = 'https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html'
            elif violation.rule_id.startswith('P'):  # PEP 257
                fix['style_guide'] = 'PEP 257'
                fix['reference_url'] = 'https://www.python.org/dev/peps/pep-0257/'
            
            fixes.append(fix)
        
        return fixes