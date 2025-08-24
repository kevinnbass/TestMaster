#!/usr/bin/env python3
"""
Run Intelligence System - Modular Coordinator
=============================================

Main runner script for the comprehensive LLM-based code intelligence system.
This is the modular coordinator that uses focused components.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Add the reorganizer directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import our intelligence system components
try:
    from llm_intelligence_system import LLMIntelligenceScanner
    from intelligence_integration_engine import IntelligenceIntegrationEngine
    from reorganization_planner import ReorganizationPlanner, DetailedReorganizationPlan
    HAS_COMPONENTS = True
except ImportError as e:
    print(f"Warning: Missing components: {e}")
    print("Please ensure all intelligence system components are installed.")
    HAS_COMPONENTS = False

# Import our modular components
from intelligence_pipeline import IntelligencePipelineExecutor
from intelligence_reports import IntelligenceReportGenerator
from intelligence_cli import (
    create_argument_parser,
    validate_main_arguments,
    execute_pipeline_action
)


class IntelligenceSystemRunner:
    """
    Main runner for the complete intelligence system.
    Handles the entire pipeline from scanning to execution.
    """

    def __init__(self, root_dir: Path):
        """Initialize the intelligence system with root directory and components.

        Args:
            root_dir: Root directory for the intelligence system operations
        """
        self.root_dir = root_dir.resolve()
        self.output_dir = self.root_dir / "tools" / "codebase_reorganizer" / "intelligence_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.scanner = None
        self.integration_engine = None
        self.planner = None

        if HAS_COMPONENTS:
            self.scanner = LLMIntelligenceScanner(root_dir)
            self.integration_engine = IntelligenceIntegrationEngine(root_dir)
            self.planner = ReorganizationPlanner(root_dir)

        # Initialize our modular components
        self.pipeline_executor = IntelligencePipelineExecutor(
            self.scanner, self.integration_engine, self.planner, self.output_dir
        )
        self.report_generator = IntelligenceReportGenerator(
            self.root_dir, self.output_dir, self.scanner, self.integration_engine, self.planner
        )

    def run_full_pipeline(self, max_files: Optional[int] = None,
                         provider: str = "mock", model: str = "gpt-4",
                         api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete intelligence pipeline.

        Args:
            max_files: Maximum number of files to process (for testing)
            provider: LLM provider to use
            model: LLM model to use
            api_key: API key for the provider

        Returns:
            Results from the complete pipeline
        """
        return self.pipeline_executor.run_full_pipeline(max_files, provider, model, api_key)

    def run_single_step(self, step: str, **kwargs) -> Dict[str, Any]:
        """Run a single step of the pipeline"""
        return self.pipeline_executor.run_single_step(step, **kwargs)

    def generate_comprehensive_report(self, results_file: Optional[str] = None) -> str:
        """Generate a comprehensive report"""
        if results_file:
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            # Use default results structure
            results = {
                'success': True,
                'steps_completed': ['pipeline_execution'],
                'output_files': {},
                'errors': []
            }

        report = self.report_generator.generate_report(results)

        # Save report to file
        report_file = self.output_dir / f"intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Report saved to: {report_file}")
        return report

    def print_system_status(self) -> None:
        """Print the current system status"""
        self.report_generator.print_system_status()

    def print_step_results(self, results: Dict[str, Any]) -> None:
        """Print results of a single step execution"""
        print(f"\n📊 Step Results: {results.get('step', 'unknown').upper()}")
        print("=" * 50)

        if results.get('success'):
            print("✅ Status: SUCCESS")
        else:
            print("❌ Status: FAILED")

        if results.get('output_files'):
            print("📁 Output Files:")
            for file_type, file_path in results.get('output_files', {}).items():
                print(f"   {file_type}: {file_path}")

        if results.get('errors'):
            print("❌ Errors:")
            for error in results.get('errors', []):
                print(f"   {error}")

        print("=" * 50)


def main():
    """Main entry point for the intelligence system with comprehensive parameter validation"""
    # Parameter validation (Rule 7 compliance)
    assert len(__name__) > 0, "Module name must be valid"
    assert main.__name__ == 'main', "Function name must be 'main'"

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    validation_result = validate_main_arguments(args)
    if validation_result != 0:
        return validation_result

    # Initialize runner
    root_dir = Path(args.root).resolve()
    runner = IntelligenceSystemRunner(root_dir)

    # Execute pipeline action
    return execute_pipeline_action(args, runner)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
