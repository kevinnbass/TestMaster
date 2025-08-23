#!/usr/bin/env python3
"""
Pattern Detector Coordinator
============================

Coordinates the pattern detection using specialized modules.
Detects common code patterns, architectural patterns, and design patterns.
"""

# Import specialized modules
from pattern_core import PatternDetector
from pattern_data import PatternAnalysis


def detect_patterns(content: str, file_path=None) -> dict:
    """
    Detect patterns in code content using the core detector.

    Args:
        content: The code content to analyze
        file_path: Optional path to the file for context

    Returns:
        Dictionary containing pattern detection results
    """
    detector = PatternDetector()
    analysis = detector.detect_patterns(content, file_path)

    # Convert to dictionary format for compatibility
    return {
        'total_patterns': len(analysis.patterns),
        'high_confidence_patterns': sum(1 for p in analysis.patterns if p.confidence > 0.7),
        'architectural_style': analysis.architectural_style,
        'coding_style': analysis.coding_style,
        'pattern_categories': analysis.pattern_categories,
        'confidence_distribution': analysis.confidence_distribution,
        'recommendations': analysis.recommendations,
        'patterns': [
            {
                'pattern_name': p.pattern_name,
                'pattern_type': p.pattern_type,
                'confidence': p.confidence,
                'location': p.location,
                'evidence': p.evidence,
                'category': p.category,
                'recommendations': p.recommendations
            }
            for p in analysis.patterns
        ]
    }


def get_pattern_confidence(content: str) -> float:
    """
    Calculate overall pattern confidence for the given content.

    Args:
        content: The code content to analyze

    Returns:
        Average confidence score across all detected patterns
    """
    result = detect_patterns(content)
    patterns = result.get('patterns', [])

    if not patterns:
        return 0.0

    total_confidence = sum(p['confidence'] for p in patterns)
    return total_confidence / len(patterns)


def identify_architectural_style(content: str) -> str:
    """
    Identify the architectural style of the given content.

    Args:
        content: The code content to analyze

    Returns:
        String describing the architectural style
    """
    result = detect_patterns(content)
    return result.get('architectural_style', 'unknown')


def main():
    """
    Main function demonstrating pattern detection capabilities.

    This function showcases the pattern detection functionality
    and serves as an example of how to use the pattern detector.
    """
    print("🔍 PATTERN DETECTOR")
    print("===================")
    print("Detects common code patterns, architectural patterns, and design patterns")
    print()

    # Example code for demonstration
    sample_code = '''
import asyncio
from typing import List

class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email

class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def save(self, user: User) -> bool:
        return True

    def find_by_id(self, user_id: int) -> User:
        return User(id=user_id, name="John", email="john@example.com")

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, name: str, email: str) -> User:
        user = User(id=123, name=name, email=email)
        self.repository.save(user)
        return user

    async def get_user_async(self, user_id: int) -> User:
        await asyncio.sleep(0.1)
        return self.repository.find_by_id(user_id)

# Singleton pattern for configuration
class Config:
    _instance = None

    def __init__(self):
        if Config._instance is not None:
            raise Exception("Config is a singleton class")
        self.settings = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

def main():
    config = Config.get_instance()
    repo = UserRepository(None)
    service = UserService(repo)

    user = service.create_user("Alice", "alice@example.com")
    print(f"Created user: {user.name} ({user.email})")

if __name__ == "__main__":
    main()
'''

    print("📝 Analyzing sample code for patterns...")
    result = detect_patterns(sample_code, "sample.py")

    print("
🔍 Analysis Results:"    print(f"📊 Total Patterns: {result['total_patterns']}")
    print(f"🎯 High Confidence Patterns: {result['high_confidence_patterns']}")
    print(f"🏗️  Architectural Style: {result['architectural_style']}")
    print(f"💻 Coding Style: {result['coding_style']}")
    print(f"📂 Pattern Categories: {result['pattern_categories']}")
    print(f"📈 Confidence Distribution: {result['confidence_distribution']}")

    if result['patterns']:
        print("
🎨 Detected Patterns:"        for i, pattern in enumerate(result['patterns'][:3], 1):  # Show first 3
            print(f"   {i}. {pattern['pattern_name']} ({pattern['pattern_type']})")
            print(".3f")
            print(f"      📍 {pattern['location']}")
            print()

    if result['recommendations']:
        print("
💡 Recommendations:"        for rec in result['recommendations'][:2]:  # Show first 2
            print(f"   • {rec}")

    print("
✅ Pattern analysis complete!"    print()
    print("This detector helps identify design patterns, architectural styles,")
    print("and coding patterns that can inform code organization and improvement.")


if __name__ == "__main__":
    main()
