#!/usr/bin/env python3
"""
Relationship Analyzer Coordinator
================================

Coordinates the relationship analysis using specialized modules.
Analyzes relationships between modules, classes, and functions to understand
how components interact within the codebase.

Key capabilities:
- Import dependency analysis
- Function call relationship mapping
- Data flow analysis
- Module coupling measurement
- Interface analysis
"""

# Import specialized modules
from relationship_core import RelationshipAnalyzer as CoreRelationshipAnalyzer
from relationship_data import RelationshipAnalysis
from relationship_ml import MLManager


def analyze_relationships(content: str, file_path: Optional[str] = None) -> dict:
    """
    Analyze relationships in code content using the core analyzer.

    Args:
        content: The code content to analyze
        file_path: Optional path to the file for context

    Returns:
        Dictionary containing relationship analysis results
    """
    analyzer = CoreRelationshipAnalyzer()
    return analyzer.analyze_relationships(content, file_path)


def calculate_coupling_score(content: str) -> float:
    """
    Calculate coupling score for the given content.

    Args:
        content: The code content to analyze

    Returns:
        Coupling score between 0.0 and 1.0
    """
    result = analyze_relationships(content)
    return result.get('coupling_metrics', {}).get('coupling_score', 0.0)


def find_highly_coupled_components(content: str) -> list:
    """
    Find highly coupled components in the code.

    Args:
        content: The code content to analyze

    Returns:
        List of highly coupled component names
    """
    result = analyze_relationships(content)
    return result.get('coupling_metrics', {}).get('highly_coupled_components', [])


def main():
    """
    Main function demonstrating relationship analysis capabilities.

    This function showcases the relationship analysis functionality
    and serves as an example of how to use the relationship analyzer.
    """
    print("🔗 RELATIONSHIP ANALYZER")
    print("=======================")
    print("Analyzes relationships between modules, classes, and functions")
    print()

    # Example code for demonstration
    sample_code = '''
import os
import sys
from pathlib import Path

class FileProcessor:
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator()

    def process_file(self, file_path):
        data = self.load_data(file_path)
        validated_data = self.validator.validate(data)
        return self.format_output(validated_data)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            return f.read()

class DataValidator:
    def validate(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        return data.strip()

def main():
    processor = FileProcessor({"input_dir": "/data"})
    result = processor.process_file("input.txt")
    print(f"Processed: {result}")

if __name__ == "__main__":
    main()
'''

    print("📝 Analyzing sample code for relationships...")
    result = analyze_relationships(sample_code, "sample.py")

    print("\n🔍 Analysis Results:")
    print(f"📊 Coupling Score: {result['coupling_metrics']['coupling_score']:.3f}")
    print(f"📦 Import Relationships: {len(result['import_relationships'])}")
    print(f"🏗️  Class Relationships: {len(result['class_relationships'])}")
    print(f"🔄 Function Calls: {len(result['function_relationships'])}")
    print(f"📈 Graph Nodes: {result['relationship_graph']['node_count']}")
    print(f"🔗 Graph Edges: {result['relationship_graph']['edge_count']}")

    if result['recommendations']:
        print("\n💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"   • {rec}")

    print("\n✅ Analysis complete!")
    print()
    print("This analyzer helps understand how components interact")
    print("and identifies opportunities for improving modularity.")


if __name__ == "__main__":
    main()