#!/usr/bin/env python3
"""
Personal Codebase Toolkit
=========================

A simple command-line interface to run all your codebase analysis tools
in one place. Perfect for understanding your personal projects.
"""

import sys
import argparse
from pathlib import Path

def run_focus_analysis():
    """Run the focused file analysis"""
    try:
        from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.focus_analyzer import analyze_core_files
        print("Running focused file analysis...")
        print()
        analyze_core_files()
    except ImportError:
        print("analyzers/focus_analyzer.py not found. Please make sure the analyzers directory exists.")
    except Exception as e:
        print(f"Error running focus analysis: {e}")

def run_dependency_analysis():
    """Run the dependency analysis"""
    try:
        from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.dependency_analyzer import DependencyAnalyzer
        print("Running dependency analysis...")
        print()
        analyzer = DependencyAnalyzer()
        analyzer.analyze_dependencies()
        analyzer.suggest_refactoring()
    except ImportError:
        print("analyzers/dependency_analyzer.py not found. Please make sure the analyzers directory exists.")
    except Exception as e:
        print(f"Error running dependency analysis: {e}")

def run_breakdown_analysis():
    """Run the file breakdown analysis"""
    try:
        from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.file_breakdown_advisor import FileBreakdownAdvisor
        print("Running file breakdown analysis...")
        print()
        advisor = FileBreakdownAdvisor()
        advisor.analyze_large_files(min_lines=150)  # Lower threshold for personal projects
    except ImportError:
        print("analyzers/file_breakdown_advisor.py not found. Please make sure the analyzers directory exists.")
    except Exception as e:
        print(f"Error running breakdown analysis: {e}")

def run_organization_analysis():
    """Run the organization analysis"""
    try:
        from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.organization_advisor import CodeOrganizationAdvisor
        print("Running organization analysis...")
        print()
        advisor = CodeOrganizationAdvisor()
        advisor.analyze_organization()
    except ImportError:
        print("analyzers/organization_advisor.py not found. Please make sure the analyzers directory exists.")
    except Exception as e:
        print(f"Error running organization analysis: {e}")

def run_simple_analysis():
    """Run the simple codebase analysis"""
    try:
        from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.simple_codebase_analyzer import CodebaseAnalyzer
        print("Running simple codebase analysis...")
        print()
        analyzer = CodebaseAnalyzer(".")
        analyzer.analyze()
        analyzer.find_potential_issues()
        analyzer.suggest_improvements()
    except ImportError:
        print("analyzers/simple_codebase_analyzer.py not found. Please make sure the analyzers directory exists.")
    except Exception as e:
        print(f"Error running simple analysis: {e}")

def run_all_analyses():
    """Run all available analyses"""
    analyses = [
        ("Focus Analysis", run_focus_analysis),
        ("Dependency Analysis", run_dependency_analysis), 
        ("Breakdown Analysis", run_breakdown_analysis),
        ("Organization Analysis", run_organization_analysis)
    ]
    
    print("Personal Codebase Analysis Toolkit")
    print("=" * 35)
    print("Running all analyses on your codebase...")
    print()
    
    for name, func in analyses:
        print("=" * 60)
        print(f"{name}")
        print("=" * 60)
        func()
        print("\n" + "=" * 60)
        print()

def show_help():
    """Show available commands and what they do"""
    print("Personal Codebase Toolkit - Available Commands")
    print("=" * 46)
    print()
    print("focus      - Analyze files in current directory only")
    print("           - Good for getting a quick overview")
    print()
    print("deps       - Analyze dependencies between your files") 
    print("           - Shows which files depend on which")
    print()
    print("breakdown  - Suggest how to break down large files")
    print("           - Helps identify files that might be too big")
    print()
    print("organize   - Suggest better file organization")
    print("           - Recommends directory structure improvements")
    print()
    print("simple     - Basic codebase analysis")
    print("           - General overview with suggestions")
    print()
    print("all        - Run all analyses")
    print("           - Complete codebase analysis")
    print()
    print("help       - Show this help message")
    print()
    print("Examples:")
    print("  python codebase_toolkit.py focus")
    print("  python codebase_toolkit.py all")
    print()

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Personal Codebase Toolkit")
        print("Use 'python codebase_toolkit.py help' for available commands")
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'focus': run_focus_analysis,
        'deps': run_dependency_analysis,
        'breakdown': run_breakdown_analysis,
        'organize': run_organization_analysis,
        'simple': run_simple_analysis,
        'all': run_all_analyses,
        'help': show_help
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python codebase_toolkit.py help' for available commands")

if __name__ == "__main__":
    main()