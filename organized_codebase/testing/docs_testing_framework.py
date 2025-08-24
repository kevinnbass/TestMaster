"""
Documentation Testing Framework

Comprehensive testing system for documentation quality, accuracy,
and consistency based on AgentScope's robust documentation patterns.
"""

import os
import re
import ast
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of documentation tests."""
    LINK_VALIDATION = "link_validation"
    CODE_EXECUTION = "code_execution"
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    FORMATTING = "formatting"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    ACCESSIBILITY = "accessibility"


class Severity(Enum):
    """Test issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


@dataclass
class TestIssue:
    """Represents a documentation test issue."""
    test_type: TestType
    severity: Severity
    message: str
    file_path: str
    line_number: int = 0
    column: int = 0
    suggestion: str = ""
    context: str = ""


@dataclass
class TestResult:
    """Represents test results for a document."""
    file_path: str
    test_type: TestType
    passed: bool
    issues: List[TestIssue] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocsTestingFramework:
    """
    Documentation testing framework inspired by AgentScope's
    comprehensive testing approach for documentation quality.
    """
    
    def __init__(self, docs_path: str = "docs"):
        """Initialize documentation testing framework."""
        self.docs_path = Path(docs_path)
        self.test_results = []
        self.config = self._load_default_config()
        self.spell_checker = None
        self.grammar_checker = None
        logger.info(f"Docs testing framework initialized at {docs_path}")
        
    def run_all_tests(self, file_patterns: List[str] = None) -> List[TestResult]:
        """Run all documentation tests."""
        if not file_patterns:
            file_patterns = ["*.md", "*.rst", "*.txt"]
            
        # Find all documentation files
        doc_files = []
        for pattern in file_patterns:
            doc_files.extend(self.docs_path.rglob(pattern))
            
        all_results = []
        
        # Run each test type
        for test_type in TestType:
            if self.config.get(f"enable_{test_type.value}", True):
                for doc_file in doc_files:
                    result = self.run_test(doc_file, test_type)
                    if result:
                        all_results.append(result)
                        
        self.test_results = all_results
        return all_results
        
    def run_test(self, file_path: Path, test_type: TestType) -> Optional[TestResult]:
        """Run a specific test on a file."""
        import time
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            issues = []
            
            if test_type == TestType.LINK_VALIDATION:
                issues = self._test_links(content, str(file_path))
            elif test_type == TestType.CODE_EXECUTION:
                issues = self._test_code_blocks(content, str(file_path))
            elif test_type == TestType.SPELLING:
                issues = self._test_spelling(content, str(file_path))
            elif test_type == TestType.GRAMMAR:
                issues = self._test_grammar(content, str(file_path))
            elif test_type == TestType.FORMATTING:
                issues = self._test_formatting(content, str(file_path))
            elif test_type == TestType.CONSISTENCY:
                issues = self._test_consistency(content, str(file_path))
            elif test_type == TestType.COMPLETENESS:
                issues = self._test_completeness(content, str(file_path))
            elif test_type == TestType.ACCESSIBILITY:
                issues = self._test_accessibility(content, str(file_path))
                
            execution_time = time.time() - start_time
            passed = all(issue.severity != Severity.ERROR for issue in issues)
            
            return TestResult(
                file_path=str(file_path),
                test_type=test_type,
                passed=passed,
                issues=issues,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error running test {test_type.value} on {file_path}: {e}")
            return None
            
    def _test_links(self, content: str, file_path: str) -> List[TestIssue]:
        """Test link validity."""
        issues = []
        
        # Find all links
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        links = re.finditer(link_pattern, content)
        
        for match in links:
            link_text = match.group(1)
            link_url = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            # Test different link types
            if link_url.startswith('http'):
                # External link - would need actual HTTP validation
                if self.config.get('validate_external_links', False):
                    issues.extend(self._validate_external_link(link_url, line_num, file_path))
            elif link_url.startswith('#'):
                # Internal anchor
                issues.extend(self._validate_internal_anchor(link_url, content, line_num, file_path))
            elif not link_url.startswith('mailto:'):
                # Relative file link
                issues.extend(self._validate_file_link(link_url, file_path, line_num))
                
        return issues
        
    def _test_code_blocks(self, content: str, file_path: str) -> List[TestIssue]:
        """Test code block execution."""
        issues = []
        
        # Find code blocks
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        code_blocks = re.finditer(code_pattern, content, re.DOTALL)
        
        for match in code_blocks:
            language = match.group(1) or 'text'
            code = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            if language.lower() == 'python':
                issues.extend(self._validate_python_code(code, line_num, file_path))
            elif language.lower() in ['bash', 'shell', 'sh']:
                issues.extend(self._validate_shell_code(code, line_num, file_path))
            elif language.lower() == 'json':
                issues.extend(self._validate_json_code(code, line_num, file_path))
                
        return issues
        
    def _test_spelling(self, content: str, file_path: str) -> List[TestIssue]:
        """Test spelling accuracy."""
        issues = []
        
        # Extract text content (skip code blocks)
        text_content = self._extract_text_content(content)
        
        # Simple word validation (would integrate with spell checker)
        words = re.findall(r'\b[a-zA-Z]+\b', text_content)
        
        # Check for common technical terms
        technical_terms = {
            'API', 'SDK', 'JSON', 'YAML', 'HTTP', 'URL', 'CLI', 'GUI',
            'async', 'await', 'config', 'auth', 'repo', 'docs'
        }
        
        for word in words:
            if len(word) > 15:  # Very long words might be typos
                issues.append(TestIssue(
                    test_type=TestType.SPELLING,
                    severity=Severity.WARNING,
                    message=f"Very long word '{word}' - possible typo",
                    file_path=file_path,
                    line_number=self._find_word_line(content, word),
                    suggestion="Check if this word is spelled correctly"
                ))
                
        return issues
        
    def _test_grammar(self, content: str, file_path: str) -> List[TestIssue]:
        """Test grammar and style."""
        issues = []
        
        # Extract sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            line_num = self._find_sentence_line(content, sentence)
            
            # Check for very long sentences
            if len(sentence.split()) > 30:
                issues.append(TestIssue(
                    test_type=TestType.GRAMMAR,
                    severity=Severity.WARNING,
                    message="Very long sentence - consider breaking up",
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Split into shorter sentences for better readability"
                ))
                
            # Check for passive voice indicators
            passive_indicators = ['is being', 'was being', 'will be', 'has been', 'have been']
            for indicator in passive_indicators:
                if indicator in sentence.lower():
                    issues.append(TestIssue(
                        test_type=TestType.GRAMMAR,
                        severity=Severity.SUGGESTION,
                        message=f"Possible passive voice: '{indicator}'",
                        file_path=file_path,
                        line_number=line_num,
                        suggestion="Consider using active voice for clarity"
                    ))
                    break
                    
        return issues
        
    def _test_formatting(self, content: str, file_path: str) -> List[TestIssue]:
        """Test formatting consistency."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(TestIssue(
                    test_type=TestType.FORMATTING,
                    severity=Severity.WARNING,
                    message="Trailing whitespace",
                    file_path=file_path,
                    line_number=i,
                    suggestion="Remove trailing spaces"
                ))
                
            # Check for inconsistent heading styles
            if line.startswith('#'):
                if not line.startswith('# ') and len(line) > 1:
                    issues.append(TestIssue(
                        test_type=TestType.FORMATTING,
                        severity=Severity.WARNING,
                        message="Missing space after hash in heading",
                        file_path=file_path,
                        line_number=i,
                        suggestion="Add space after # in headings"
                    ))
                    
        # Check for consistent code block formatting
        code_blocks = re.findall(r'```\w*\n.*?\n```', content, re.DOTALL)
        for block in code_blocks:
            if not block.endswith('\n```'):
                issues.append(TestIssue(
                    test_type=TestType.FORMATTING,
                    severity=Severity.WARNING,
                    message="Code block not properly closed",
                    file_path=file_path,
                    line_number=self._find_text_line(content, block),
                    suggestion="Ensure code blocks end with ``` on new line"
                ))
                
        return issues
        
    def _test_consistency(self, content: str, file_path: str) -> List[TestIssue]:
        """Test consistency across documentation."""
        issues = []
        
        # Check for consistent terminology
        terms_to_check = {
            'API': ['api', 'Api'],
            'JSON': ['json', 'Json'],
            'URL': ['url', 'Url'],
            'CLI': ['cli', 'Cli']
        }
        
        for standard_term, variations in terms_to_check.items():
            for variation in variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', content):
                    line_num = self._find_word_line(content, variation)
                    issues.append(TestIssue(
                        test_type=TestType.CONSISTENCY,
                        severity=Severity.SUGGESTION,
                        message=f"Inconsistent term '{variation}' - use '{standard_term}'",
                        file_path=file_path,
                        line_number=line_num,
                        suggestion=f"Replace '{variation}' with '{standard_term}' for consistency"
                    ))
                    
        return issues
        
    def _test_completeness(self, content: str, file_path: str) -> List[TestIssue]:
        """Test documentation completeness."""
        issues = []
        
        # Check for required sections in README files
        if 'readme' in Path(file_path).name.lower():
            required_sections = ['installation', 'usage', 'examples', 'contributing']
            content_lower = content.lower()
            
            for section in required_sections:
                if section not in content_lower:
                    issues.append(TestIssue(
                        test_type=TestType.COMPLETENESS,
                        severity=Severity.WARNING,
                        message=f"Missing recommended section: {section}",
                        file_path=file_path,
                        line_number=1,
                        suggestion=f"Consider adding a '{section.title()}' section"
                    ))
                    
        # Check for empty code blocks
        empty_code_pattern = r'```\w*\n\s*\n```'
        empty_blocks = re.finditer(empty_code_pattern, content)
        
        for match in empty_blocks:
            line_num = content[:match.start()].count('\n') + 1
            issues.append(TestIssue(
                test_type=TestType.COMPLETENESS,
                severity=Severity.ERROR,
                message="Empty code block",
                file_path=file_path,
                line_number=line_num,
                suggestion="Add code example or remove empty code block"
            ))
            
        return issues
        
    def _test_accessibility(self, content: str, file_path: str) -> List[TestIssue]:
        """Test accessibility features."""
        issues = []
        
        # Check for alt text in images
        image_pattern = r'!\[([^\]]*)\]\([^\)]+\)'
        images = re.finditer(image_pattern, content)
        
        for match in images:
            alt_text = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            if not alt_text or alt_text.strip() == '':
                issues.append(TestIssue(
                    test_type=TestType.ACCESSIBILITY,
                    severity=Severity.WARNING,
                    message="Image missing alt text",
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Add descriptive alt text for accessibility"
                ))
            elif len(alt_text) < 3:
                issues.append(TestIssue(
                    test_type=TestType.ACCESSIBILITY,
                    severity=Severity.WARNING,
                    message="Alt text too short",
                    file_path=file_path,
                    line_number=line_num,
                    suggestion="Use more descriptive alt text"
                ))
                
        return issues
        
    def _validate_python_code(self, code: str, line_num: int, file_path: str) -> List[TestIssue]:
        """Validate Python code syntax."""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(TestIssue(
                test_type=TestType.CODE_EXECUTION,
                severity=Severity.ERROR,
                message=f"Python syntax error: {e.msg}",
                file_path=file_path,
                line_number=line_num + (e.lineno or 1) - 1,
                column=e.offset or 0,
                suggestion="Fix Python syntax error"
            ))
            
        return issues
        
    def _validate_json_code(self, code: str, line_num: int, file_path: str) -> List[TestIssue]:
        """Validate JSON syntax."""
        issues = []
        
        try:
            import json
            json.loads(code)
        except json.JSONDecodeError as e:
            issues.append(TestIssue(
                test_type=TestType.CODE_EXECUTION,
                severity=Severity.ERROR,
                message=f"JSON syntax error: {e.msg}",
                file_path=file_path,
                line_number=line_num + e.lineno - 1,
                column=e.colno,
                suggestion="Fix JSON syntax error"
            ))
            
        return issues
        
    def _validate_shell_code(self, code: str, line_num: int, file_path: str) -> List[TestIssue]:
        """Validate shell code."""
        issues = []
        
        # Basic shell validation
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for dangerous commands
            dangerous_commands = ['rm -rf /', 'dd if=', 'format c:', 'del /s']
            for cmd in dangerous_commands:
                if cmd in line.lower():
                    issues.append(TestIssue(
                        test_type=TestType.CODE_EXECUTION,
                        severity=Severity.ERROR,
                        message=f"Potentially dangerous command: {cmd}",
                        file_path=file_path,
                        line_number=line_num + i,
                        suggestion="Remove or modify dangerous command"
                    ))
                    
        return issues
        
    def _extract_text_content(self, content: str) -> str:
        """Extract text content, excluding code blocks."""
        # Remove code blocks
        content = re.sub(r'```.*?\n```', '', content, flags=re.DOTALL)
        
        # Remove inline code
        content = re.sub(r'`[^`]+`', '', content)
        
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        return content
        
    def _find_word_line(self, content: str, word: str) -> int:
        """Find line number of word in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if word in line:
                return i
        return 1
        
    def _find_sentence_line(self, content: str, sentence: str) -> int:
        """Find line number of sentence in content."""
        return self._find_word_line(content, sentence[:20])  # Use first 20 chars
        
    def _find_text_line(self, content: str, text: str) -> int:
        """Find line number of text in content."""
        return content[:content.find(text)].count('\n') + 1 if text in content else 1
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'enable_link_validation': True,
            'enable_code_execution': True,
            'enable_spelling': True,
            'enable_grammar': True,
            'enable_formatting': True,
            'enable_consistency': True,
            'enable_completeness': True,
            'enable_accessibility': True,
            'validate_external_links': False,  # Disabled by default (slow)
            'max_line_length': 100,
            'required_readme_sections': ['Installation', 'Usage', 'Examples'],
        }
        
    def generate_test_report(self, output_format: str = "markdown") -> str:
        """Generate test report."""
        if output_format == "markdown":
            return self._generate_markdown_report()
        elif output_format == "json":
            return self._generate_json_report()
        else:
            return self._generate_text_report()
            
    def _generate_markdown_report(self) -> str:
        """Generate markdown test report."""
        if not self.test_results:
            return "# Documentation Test Report\n\nNo tests run yet."
            
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        report = [
            "# Documentation Test Report",
            "",
            f"**Summary:** {passed_tests}/{total_tests} tests passed",
            "",
            "## Test Results",
            ""
        ]
        
        # Group by test type
        by_type = {}
        for result in self.test_results:
            if result.test_type not in by_type:
                by_type[result.test_type] = []
            by_type[result.test_type].append(result)
            
        for test_type, results in by_type.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            report.extend([
                f"### {test_type.value.title()} Tests",
                f"**Status:** {passed}/{total} passed",
                ""
            ])
            
            # Show failures
            failures = [r for r in results if not r.passed]
            if failures:
                report.append("**Issues:**")
                for result in failures:
                    for issue in result.issues:
                        if issue.severity == Severity.ERROR:
                            icon = "❌"
                        elif issue.severity == Severity.WARNING:
                            icon = "⚠️"
                        else:
                            icon = "ℹ️"
                            
                        report.append(f"- {icon} {issue.message} ({Path(issue.file_path).name}:{issue.line_number})")
                        
                report.append("")
                
        return "\n".join(report)
        
    def _generate_json_report(self) -> str:
        """Generate JSON test report."""
        import json
        
        report_data = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r.passed),
                "failed": sum(1 for r in self.test_results if not r.passed)
            },
            "results": []
        }
        
        for result in self.test_results:
            result_data = {
                "file_path": result.file_path,
                "test_type": result.test_type.value,
                "passed": result.passed,
                "execution_time": result.execution_time,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "line_number": issue.line_number,
                        "suggestion": issue.suggestion
                    }
                    for issue in result.issues
                ]
            }
            report_data["results"].append(result_data)
            
        return json.dumps(report_data, indent=2)
        
    def _generate_text_report(self) -> str:
        """Generate plain text test report."""
        lines = ["Documentation Test Report", "=" * 30, ""]
        
        if not self.test_results:
            lines.append("No tests run yet.")
            return "\n".join(lines)
            
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        
        lines.extend([
            f"Total Tests: {total}",
            f"Passed: {passed}",
            f"Failed: {total - passed}",
            ""
        ])
        
        # List failures
        failures = [r for r in self.test_results if not r.passed]
        if failures:
            lines.append("FAILURES:")
            for result in failures:
                lines.append(f"  {result.file_path} ({result.test_type.value})")
                for issue in result.issues:
                    lines.append(f"    - {issue.message}")
                    
        return "\n".join(lines)