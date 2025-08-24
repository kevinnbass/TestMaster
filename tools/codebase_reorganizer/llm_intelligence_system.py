#!/usr/bin/env python3
"""
LLM Intelligence System Coordinator
====================================

Coordinates the LLM-powered intelligence system using specialized modules.
"""

# Import specialized modules
from llm_core import LLMIntelligenceScanner
from llm_data import LLMIntelligenceMap
from pathlib import Path


def scan_and_analyze(root_dir, config=None, output_file=None, max_files=None):
    """Perform comprehensive intelligence scan and analysis."""
    scanner = LLMIntelligenceScanner(root_dir, config)
    return scanner.scan_and_analyze(output_file, max_files)


def main():
    """Main function demonstrating LLM intelligence system capabilities."""
    print(" LLM INTELLIGENCE SYSTEM")
    print("=========================")
    print("Comprehensive LLM-powered code intelligence and analysis")
    print()
    print(" System ready for use!")


if __name__ == "__main__":
    main()
