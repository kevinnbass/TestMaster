"""
Language Detection Module
=========================
Detects programming languages from code snippets.
"""

from typing import Dict, Any, List

class UniversalLanguageDetector:
    """Universal language detector for all programming languages."""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'c++', 'go', 'rust', 'typescript']
    
    def detect(self, code: str) -> str:
        """Detect language from code."""
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code:
            return 'javascript'
        return 'unknown'

class LanguageDetector:
    """Detects programming language from code."""
    
    def __init__(self):
        self.languages = ['python', 'javascript', 'java', 'c++', 'go', 'rust']
    
    def detect(self, code: str) -> str:
        """Detect language from code snippet."""
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code:
            return 'javascript'
        elif 'public class' in code:
            return 'java'
        return 'unknown'
    
    def get_confidence(self, code: str) -> float:
        """Get confidence score for detection."""
        return 0.95 if self.detect(code) != 'unknown' else 0.0

# Global detector
detector = LanguageDetector()

def detect_language(code: str) -> str:
    """Detect programming language from code."""
    return detector.detect(code)

class CodebaseProfile:
    """Profile of a codebase including language distribution and metrics."""
    
    def __init__(self):
        self.languages: Dict[str, int] = {}
        self.total_files: int = 0
        self.total_lines: int = 0
        self.file_types: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_file(self, language: str, lines: int, file_type: str = None):
        """Add a file to the profile."""
        if language not in self.languages:
            self.languages[language] = 0
        self.languages[language] += lines
        self.total_files += 1
        self.total_lines += lines
        
        if file_type:
            if file_type not in self.file_types:
                self.file_types[file_type] = 0
            self.file_types[file_type] += 1
    
    def get_language_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of languages."""
        if self.total_lines == 0:
            return {}
        
        return {
            lang: (lines / self.total_lines) * 100 
            for lang, lines in self.languages.items()
        }
    
    def get_dominant_language(self) -> str:
        """Get the most prevalent language."""
        if not self.languages:
            return 'unknown'
        return max(self.languages.items(), key=lambda x: x[1])[0]