#!/usr/bin/env python3
"""
Intelligence System CLI Module
==============================

Handles command-line interface functionality for the intelligence system.
Contains argument parsing, validation, and CLI execution logic.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any


def get_parser_epilog() -> str:
    """Get the parser epilog with examples"""
    return """
Examples:
  # Run complete pipeline with mock LLM
  python run_intelligence_system.py --full-pipeline --max-files 10

  # Run complete pipeline with OpenAI
  python run_intelligence_system.py --full-pipeline --provider openai --api-key YOUR_KEY

  # Run single scanning step
  python run_intelligence_system.py --step scan --max-files 5

  # Show system status
  python run_intelligence_system.py --status

  # Generate report from existing results
  python run_intelligence_system.py --generate-report --results-file path/to/results.json
    """


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Add pipeline-related arguments to parser"""
    # Pipeline options
    pipeline_group = parser.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument("--full-pipeline", action="store_true",
                              help="Run the complete intelligence pipeline")
    pipeline_group.add_argument("--step", choices=["scan", "integrate", "plan", "execute"],
                              help="Run a single step of the pipeline")
    pipeline_group.add_argument("--status", action="store_true",
                              help="Show system status")
    pipeline_group.add_argument("--generate-report", action="store_true",
                              help="Generate a report from existing results")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser"""
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory to analyze (default: current)")

    # Common options
    parser.add_argument("--max-files", type=int,
                      help="Maximum number of files to process (for testing)")
    parser.add_argument("--provider", type=str, default="mock",
                      choices=["openai", "anthropic", "groq", "ollama", "mock"],
                      help="LLM provider to use (default: mock)")
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="LLM model to use (default: gpt-4)")
    parser.add_argument("--api-key", type=str,
                      help="API key for LLM provider")


def add_step_specific_arguments(parser: argparse.ArgumentParser) -> None:
    """Add step-specific arguments to parser"""
    # Step-specific options
    parser.add_argument("--llm-map", type=str,
                      help="Path to LLM intelligence map (for integrate/plan steps)")
    parser.add_argument("--integrated", type=str,
                      help="Path to integrated intelligence (for plan step)")
    parser.add_argument("--plan", type=str,
                      help="Path to reorganization plan (for execute step)")
    parser.add_argument("--batch-id", type=str,
                      help="Batch ID to execute")
    parser.add_argument("--dry-run", action="store_true",
                      help="Perform dry run (no actual changes)")
    parser.add_argument("--results-file", type=str,
                      help="Path to results file for report generation")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM-Based Code Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_parser_epilog()
    )

    add_common_arguments(parser)
    add_pipeline_arguments(parser)
    add_step_specific_arguments(parser)

    return parser


def validate_main_arguments(args: argparse.Namespace) -> int:
    """Validate main function arguments (Rule 7 compliance)"""
    if args.root and not Path(args.root).exists():
        print(f"❌ Error: Root directory '{args.root}' does not exist")
        return 1

    if args.root and not Path(args.root).is_dir():
        print(f"❌ Error: Root path '{args.root}' is not a directory")
        return 1

    # Validate file paths if provided
    if args.llm_map and not Path(args.llm_map).exists():
        print(f"❌ Error: LLM map file '{args.llm_map}' does not exist")
        return 1

    if args.integrated and not Path(args.integrated).exists():
        print(f"❌ Error: Integrated intelligence file '{args.integrated}' does not exist")
        return 1

    if args.plan and not Path(args.plan).exists():
        print(f"❌ Error: Plan file '{args.plan}' does not exist")
        return 1

    if args.results_file and not Path(args.results_file).exists():
        print(f"❌ Error: Results file '{args.results_file}' does not exist")
        return 1

    # Validate numeric parameters
    if args.max_files and args.max_files < 1:
        print("❌ Error: max-files must be a positive integer")
        return 1

    if args.max_files and args.max_files > 10000:  # Safety bound
        print("❌ Error: max-files cannot exceed 10,000")
        return 1

    # Validate provider requires API key for non-mock providers
    if args.provider != "mock" and not args.api_key:
        print(f"❌ Error: API key required for provider '{args.provider}'")
        return 1

    # Validate step-specific requirements
    if args.step == "integrate" and not args.llm_map:
        print("❌ Error: --llm-map required for integrate step")
        return 1

    if args.step == "plan" and not args.integrated:
        print("❌ Error: --integrated required for plan step")
        return 1

    if args.step == "execute" and not args.plan:
        print("❌ Error: --plan required for execute step")
        return 1

    return 0


def execute_pipeline_action(args: argparse.Namespace, runner) -> int:
    """Execute the main pipeline action based on arguments"""
    try:
        if args.status:
            # Show system status
            runner.print_system_status()
            return 0

        if args.generate_report:
            # Generate report from existing results
            if not args.results_file:
                print("❌ --results-file required for report generation")
                return 1

            runner.generate_comprehensive_report(args.results_file)
            return 0

        if args.full_pipeline:
            # Run complete pipeline
            results = runner.run_full_pipeline(
                max_files=args.max_files,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key
            )
            runner.generate_comprehensive_report()
            return 0

        if args.step:
            # Run single step
            if args.step == "scan":
                results = runner._run_scan_step(max_files=args.max_files)
            elif args.step == "integrate":
                results = runner._run_integrate_step(args.llm_map)
            elif args.step == "plan":
                results = runner._run_plan_step(args.integrated)
            elif args.step == "execute":
                results = runner._run_execute_step(args.plan, args.batch_id, args.dry_run)
            else:
                print(f"❌ Unknown step: {args.step}")
                return 1

            runner.print_step_results(results)
            return 0

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1

    return 0
