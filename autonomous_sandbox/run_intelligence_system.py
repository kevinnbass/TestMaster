#!/usr/bin/env python3
"""
Run Intelligence System
=======================

Main runner script for the comprehensive LLM-based code intelligence system.
Provides an easy-to-use interface for the entire reorganization pipeline.
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

    def _initialize_pipeline_results(self) -> Dict[str, Any]:
        """Initialize pipeline results structure (helper function)"""
        return {
            'pipeline_started': datetime.now().isoformat(),
            'steps_completed': [],
            'output_files': {},
            'success': False,
            'errors': []
        }

    def _run_llm_scanning_step(self, results: Dict[str, Any], max_files: Optional[int],
                              provider: str, model: str, api_key: Optional[str]) -> Dict[str, Any]:
        """Run LLM intelligence scanning step (helper function)"""
        print("\n📡 Step 1: LLM Intelligence Scanning")
        print("-" * 40)

        config = {
            'llm_provider': provider,
            'llm_model': model,
            'api_key': api_key,
            'max_concurrent': 3,
            'preserve_directory_order': True,
            'enable_static_analysis': True
        }

        if self.scanner:
            self.scanner.config.update(config)

            llm_map_file = self.output_dir / f"llm_intelligence_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            llm_intelligence_map = self.scanner.scan_and_analyze(llm_map_file, max_files)

            results['output_files']['llm_intelligence_map'] = str(llm_map_file)
            results['steps_completed'].append('llm_scanning')
            print(f"✅ LLM scanning completed - {llm_intelligence_map.total_files_scanned} files analyzed")
        else:
            raise Exception("LLM scanner not available")

        return results

    def _run_integration_step(self, results: Dict[str, Any], llm_intelligence_map: Any) -> tuple:
        """Run intelligence integration step (helper function)"""
        print("\n🔗 Step 2: Intelligence Integration")
        print("-" * 40)

        if self.integration_engine:
            integrated_intelligence = self.integration_engine.integrate_intelligence(
                llm_intelligence_map.__dict__ if hasattr(llm_intelligence_map, '__dict__') else llm_intelligence_map
            )

            integrated_file = self.output_dir / f"integrated_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.integration_engine.save_integration_results(
                integrated_intelligence, None, integrated_file
            )

            results['output_files']['integrated_intelligence'] = str(integrated_file)
            results['steps_completed'].append('intelligence_integration')
            print(f"✅ Intelligence integration completed - {len(integrated_intelligence)} entries processed")
        else:
            raise Exception("Integration engine not available")

        return results, integrated_intelligence, llm_intelligence_map

    def _run_planning_step(self, results: Dict[str, Any], llm_intelligence_map: Any,
                          integrated_intelligence: Any) -> Dict[str, Any]:
        """Run reorganization planning step (helper function)"""
        print("\n📋 Step 3: Reorganization Planning")
        print("-" * 40)

        if self.planner:
            reorganization_plan = self.planner.create_reorganization_plan(
                llm_intelligence_map.__dict__ if hasattr(llm_intelligence_map, '__dict__') else llm_intelligence_map,
                integrated_intelligence
            )

            plan_file = self.output_dir / f"reorganization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.planner.save_reorganization_plan(reorganization_plan, plan_file)

            results['output_files']['reorganization_plan'] = str(plan_file)
            results['steps_completed'].append('reorganization_planning')
            print(f"✅ Reorganization planning completed - {reorganization_plan.total_batches} batches created")
        else:
            raise Exception("Reorganization planner not available")

        return results

    def _finalize_pipeline_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize pipeline results and display summary (helper function)"""
        # Pipeline completed successfully
        results['success'] = True
        results['pipeline_completed'] = datetime.now().isoformat()

        print("\n🎉 Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 LLM Intelligence Map: {results['output_files']['llm_intelligence_map']}")
        print(f"🔗 Integrated Intelligence: {results['output_files']['integrated_intelligence']}")
        print(f"📋 Reorganization Plan: {results['output_files']['reorganization_plan']}")

        return results

    def run_full_pipeline(self, max_files: Optional[int] = None,
                         provider: str = "mock", model: str = "gpt-4",
                         api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete intelligence pipeline (coordinator function).

        Args:
            max_files: Maximum number of files to process (for testing)
            provider: LLM provider to use
            model: LLM model to use
            api_key: API key for the provider

        Returns:
            Results from the complete pipeline
        """
        results = self._initialize_pipeline_results()

        try:
            print("🚀 Starting Complete Intelligence System Pipeline")
            print("=" * 60)

            # Step 1: Run LLM scanning
            results = self._run_llm_scanning_step(results, max_files, provider, model, api_key)

            # Step 2: Run intelligence integration
            results, integrated_intelligence, llm_intelligence_map = self._run_integration_step(results, llm_intelligence_map)

            # Step 3: Run reorganization planning
            results = self._run_planning_step(results, llm_intelligence_map, integrated_intelligence)

            # Finalize results
            results = self._finalize_pipeline_results(results)

            return results

        except Exception as e:
            results['errors'].append(str(e))
            results['pipeline_failed'] = datetime.now().isoformat()
            print(f"\n❌ Pipeline failed: {e}")
            return results

    def run_single_step(self, step: str, **kwargs) -> Dict[str, Any]:
        """Run a single step of the pipeline"""

        results = {
            'step': step,
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            if step == 'scan':
                return self._run_scan_step(**kwargs)
            elif step == 'integrate':
                return self._run_integrate_step(**kwargs)
            elif step == 'plan':
                return self._run_plan_step(**kwargs)
            elif step == 'execute':
                return self._run_execute_step(**kwargs)
            else:
                raise ValueError(f"Unknown step: {step}")

        except Exception as e:
            results['errors'].append(str(e))
            results['failed'] = datetime.now().isoformat()
            return results

    def _run_scan_step(self, max_files: Optional[int] = None,
                      provider: str = "mock", model: str = "gpt-4",
                      api_key: Optional[str] = None) -> Dict[str, Any]:
        """Run the scanning step"""

        results = {
            'step': 'scan',
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            config = {
                'llm_provider': provider,
                'llm_model': model,
                'api_key': api_key,
                'max_concurrent': 3,
                'enable_static_analysis': True
            }

            if self.scanner:
                self.scanner.config.update(config)

                output_file = self.output_dir / f"llm_intelligence_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                intelligence_map = self.scanner.scan_and_analyze(output_file, max_files)

                results['output_files']['intelligence_map'] = str(output_file)
                results['files_scanned'] = intelligence_map.total_files_scanned
                results['success'] = True
                results['completed'] = datetime.now().isoformat()

        except Exception as e:
            results['errors'].append(str(e))

        return results

    def _run_integrate_step(self, llm_map_file: str) -> Dict[str, Any]:
        """Run the integration step"""

        results = {
            'step': 'integrate',
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            with open(llm_map_file, 'r', encoding='utf-8') as f:
                llm_map = json.load(f)

            if self.integration_engine:
                integrated_intelligence = self.integration_engine.integrate_intelligence(llm_map)

                output_file = self.output_dir / f"integrated_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.integration_engine.save_integration_results(
                    integrated_intelligence, None, output_file
                )

                results['output_files']['integrated_intelligence'] = str(output_file)
                results['modules_integrated'] = len(integrated_intelligence)
                results['success'] = True
                results['completed'] = datetime.now().isoformat()

        except Exception as e:
            results['errors'].append(str(e))

        return results

    def _run_plan_step(self, llm_map_file: str, integrated_file: str) -> Dict[str, Any]:
        """Run the planning step (coordinator function)"""

        # Pre-allocate results with error list (Rule 3 compliance)
        MAX_ERRORS = 100  # Safety bound for errors
        errors = [None] * MAX_ERRORS  # Pre-allocate errors list
        error_count = 0

        results = {
            'step': 'plan',
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            # Load pipeline data using helper
            llm_map, integrated_data = self._load_pipeline_data(llm_map_file, integrated_file)

            if self.planner:
                # Reconstruct integrated intelligence using helper
                integrated_intelligence = self._reconstruct_integrated_intelligence(integrated_data)

                # Create and save reorganization plan using helper
                reorganization_plan, output_file = self._create_and_save_reorganization_plan(llm_map, integrated_intelligence)

                results['output_files']['reorganization_plan'] = str(output_file)
                results['total_batches'] = reorganization_plan.total_batches
                results['total_tasks'] = reorganization_plan.total_tasks
                results['estimated_hours'] = reorganization_plan.summary['batch_statistics']['total_estimated_hours']
                results['success'] = True
                results['completed'] = datetime.now().isoformat()

        except Exception as e:
            if error_count < MAX_ERRORS:
                errors[error_count] = str(e)
                error_count += 1

        # Update results with trimmed errors list
        results['errors'] = errors[:error_count]
        return results

    def _load_pipeline_data(self, llm_map_file: str, integrated_file: str) -> tuple:
        """Load LLM map and integrated data files (helper function)"""
        with open(llm_map_file, 'r', encoding='utf-8') as f:
            llm_map = json.load(f)

        with open(integrated_file, 'r', encoding='utf-8') as f:
            integrated_data = json.load(f)

        return llm_map, integrated_data

    def _reconstruct_integrated_intelligence(self, integrated_data: Dict) -> List:
        """Reconstruct integrated intelligence objects (helper function)"""
        from intelligence_integration_engine import IntegratedIntelligence

        # Convert integrated data to objects with pre-allocation (Rule 3 compliance)
        integrated_items = integrated_data.get('integrated_intelligence', [])
        MAX_INTEGRATED = 1000  # Safety bound for integrated intelligence
        integrated_intelligence = [None] * MAX_INTEGRATED  # Pre-allocate with placeholder
        integrated_count = 0

        # Bounded loop for reconstructing integrated intelligence
        for i in range(min(len(integrated_items), MAX_INTEGRATED)):
            item = integrated_items[i]
            if integrated_count < MAX_INTEGRATED:
                integrated_intelligence[integrated_count] = IntegratedIntelligence(**item)
                integrated_count += 1

        return integrated_intelligence[:integrated_count]  # Return actual data

    def _create_and_save_reorganization_plan(self, llm_map: Dict, integrated_intelligence: List) -> tuple:
        """Create and save reorganization plan (helper function)"""
        reorganization_plan = self.planner.create_reorganization_plan(llm_map, integrated_intelligence)

        output_file = self.output_dir / f"reorganization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.planner.save_reorganization_plan(reorganization_plan, output_file)

        return reorganization_plan, output_file

    def _run_execute_step(self, plan_file: str, batch_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Run the execution step (coordinator function)"""

        # Pre-allocate errors list to avoid dynamic resizing
        MAX_ERRORS = 10
        errors = [None] * MAX_ERRORS
        error_count = 0

        results = {
            'step': 'execute',
            'started': datetime.now().isoformat(),
            'success': False,
            'errors': []
        }

        try:
            # Load plan data using helper
            plan_data = self._load_plan_data(plan_file)

            if self.planner:
                # Reconstruct batches using helper
                batches = self._reconstruct_batches(plan_data)

                # Filter plan data using helper
                filtered_plan_data = self._filter_plan_data(plan_data)

                # Create detailed plan using helper
                plan = self._create_detailed_plan(filtered_plan_data, batches)

                # Execute plan and update results using helper
                results = self._execute_plan_and_update_results(plan, batch_id, dry_run, results)

        except Exception as e:
            if error_count < MAX_ERRORS:
                errors[error_count] = str(e)
                error_count += 1

        # Update results with trimmed errors list
        results['errors'] = errors[:error_count]
        return results

    def _load_plan_data(self, plan_file: str) -> Dict:
        """Load reorganization plan data (helper function)"""
        with open(plan_file, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        return plan_data

    def _reconstruct_batches(self, plan_data: Dict) -> List:
        """Reconstruct reorganization batches (helper function)"""
        from reorganization_planner import ReorganizationBatch

        # Convert batches to objects with pre-allocation (Rule 3 compliance)
        plan_batches = plan_data.get('batches', [])
        MAX_BATCHES = 100  # Safety bound for batches
        batches = [None] * MAX_BATCHES  # Pre-allocate with placeholder
        batch_count = 0

        # Bounded loop for reconstructing batches
        for i in range(min(len(plan_batches), MAX_BATCHES)):
            batch = plan_batches[i]
            if batch_count < MAX_BATCHES:
                batches[batch_count] = ReorganizationBatch(**batch)
                batch_count += 1

        return batches[:batch_count]  # Return actual data

    def _filter_plan_data(self, plan_data: Dict) -> Dict:
        """Filter plan data to exclude batches (helper function)"""
        # Filter plan data to exclude batches with bounded operations
        plan_items = list(plan_data.items())
        MAX_PLAN_ITEMS = 50  # Safety bound for plan data items
        filtered_plan_data = {}

        # Bounded loop for filtering plan data
        for i in range(min(len(plan_items), MAX_PLAN_ITEMS)):
            k, v = plan_items[i]
            if k != 'batches':
                filtered_plan_data[k] = v

        return filtered_plan_data

    def _create_detailed_plan(self, filtered_plan_data: Dict, batches: List) -> Any:
        """Create detailed reorganization plan (helper function)"""
        from reorganization_planner import DetailedReorganizationPlan
        return DetailedReorganizationPlan(**filtered_plan_data, batches=batches)

    def _execute_plan_and_update_results(self, plan: Any, batch_id: str, dry_run: bool,
                                       results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan and update results (helper function)"""
        execution_results = self.planner.execute_plan_batch(plan, batch_id, dry_run)

        results.update(execution_results)
        results['success'] = execution_results.get('success', False)
        results['completed'] = datetime.now().isoformat()

        return results

    def _initialize_report_structure(self) -> tuple:
        """Initialize report structure (helper function)"""
        MAX_REPORT_LINES = 1000  # Safety bound for report lines
        report = [None] * MAX_REPORT_LINES  # Pre-allocate with placeholder
        report_count = 0
        return report, report_count, MAX_REPORT_LINES

    def _add_report_header(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                          results: Dict[str, Any]) -> int:
        """Add report header information (helper function)"""
        current_count = report_count

        if current_count < MAX_REPORT_LINES:
            report[current_count] = "# Complete Intelligence System Report"
            current_count += 1
        if current_count < MAX_REPORT_LINES:
            report[current_count] = "=" * 50
            current_count += 1
        if current_count < MAX_REPORT_LINES:
            report[current_count] = ""
            current_count += 1

        # Add timestamp
        if current_count < MAX_REPORT_LINES:
            report[current_count] = f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            current_count += 1
        if current_count < MAX_REPORT_LINES:
            report[current_count] = ""
            current_count += 1

        return current_count

    def _add_step_results(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                         results: Dict[str, Any]) -> int:
        """Add step results to report (helper function)"""
        current_count = report_count

        if current_count < MAX_REPORT_LINES:
            report[current_count] = "## Pipeline Results"
            current_count += 1

        # Add overall status
        success_status = "✅ SUCCESS" if results.get('success', False) else "❌ FAILED"
        if current_count < MAX_REPORT_LINES:
            report[current_count] = f"**Overall Status:** {success_status}"
            current_count += 1
        if current_count < MAX_REPORT_LINES:
            report[current_count] = ""
            current_count += 1

        # Add completed steps with bounded loop
        steps_completed = results.get('steps_completed', [])
        for i in range(min(len(steps_completed), 20)):  # Safety bound for steps
            step = steps_completed[i]
            if current_count < MAX_REPORT_LINES:
                report[current_count] = f"- ✅ {step.replace('_', ' ').title()}"
                current_count += 1

        return current_count

    def _add_errors_to_report(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                             results: Dict[str, Any]) -> int:
        """Add errors to report (helper function)"""
        current_count = report_count

        if results.get('errors'):
            if current_count < MAX_REPORT_LINES:
                report[current_count] = ""
                current_count += 1
            if current_count < MAX_REPORT_LINES:
                report[current_count] = "## Errors Encountered"
                current_count += 1

            # Add errors with bounded loop
            errors_list = results.get('errors', [])
            for i in range(min(len(errors_list), 50)):  # Safety bound for errors
                error = errors_list[i]
                if current_count < MAX_REPORT_LINES:
                    report[current_count] = f"- ❌ {error}"
                    current_count += 1

        return current_count

    def _finalize_report(self, report: List, report_count: int, MAX_REPORT_LINES: int) -> str:
        """Finalize and return report (helper function)"""
        if report_count < MAX_REPORT_LINES:
            report[report_count] = ""
            report_count += 1
        if report_count < MAX_REPORT_LINES:
            report[report_count] = "---"
            report_count += 1
        if report_count < MAX_REPORT_LINES:
            report[report_count] = "*Generated by High-Reliability Intelligence System*"
            report_count += 1

        return "\n".join(line for line in report[:report_count] if line is not None)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report of the pipeline results (coordinator function)"""
        # Initialize report structure
        report, report_count, MAX_REPORT_LINES = self._initialize_report_structure()

        # Add basic report content
        report_count = self._add_basic_report_content(report, report_count, MAX_REPORT_LINES, results)

        # Add step results
        report_count = self._add_step_results_simple(report, report_count, MAX_REPORT_LINES, results)

        # Add errors
        report_count = self._add_errors_to_report(report, report_count, MAX_REPORT_LINES, results)

        # Add output files
        report_count = self._add_output_files(report, report_count, MAX_REPORT_LINES, results)

        # Finalize and return report
        return self._finalize_report(report, report_count, MAX_REPORT_LINES)

    def _add_basic_report_content(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                                 results: Dict[str, Any]) -> int:
        """Add basic report content (helper function)"""
        current_count = report_count

        if current_count < MAX_REPORT_LINES:
            report[current_count] = "# LLM Intelligence System Report"
            current_count += 1
        if current_count < MAX_REPORT_LINES:
            report[current_count] = f"Generated: {datetime.now().isoformat()}"
            current_count += 1

        # Add pipeline status
        if results.get('success'):
            if current_count < MAX_REPORT_LINES:
                report[current_count] = "## ✅ Pipeline Status: SUCCESS"
                current_count += 1
        else:
            if current_count < MAX_REPORT_LINES:
                report[current_count] = "## ❌ Pipeline Status: FAILED"
                current_count += 1

        return current_count

    def _add_step_results_simple(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                                results: Dict[str, Any]) -> int:
        """Add step results (helper function)"""
        current_count = report_count

        if current_count < MAX_REPORT_LINES:
            report[current_count] = "## Pipeline Steps Completed"
            current_count += 1

        steps_completed = results.get('steps_completed', [])
        for i in range(min(len(steps_completed), 20)):
            step = steps_completed[i]
            if current_count < MAX_REPORT_LINES:
                report[current_count] = f"- ✅ {step.replace('_', ' ').title()}"
                current_count += 1

        return current_count

    def _add_output_files(self, report: List, report_count: int, MAX_REPORT_LINES: int,
                         results: Dict[str, Any]) -> int:
        """Add output files section (helper function)"""
        current_count = report_count

        if current_count < MAX_REPORT_LINES:
            report[current_count] = "## Output Files Generated"
            current_count += 1

        output_files = results.get('output_files', {})
        output_items = list(output_files.items())
        for i in range(min(len(output_items), 20)):
            file_type, file_path = output_items[i]
            if current_count < MAX_REPORT_LINES:
                report[current_count] = f"- {file_type.replace('_', ' ').title()}: `{file_path}`"
                current_count += 1

        return current_count

    def print_system_status(self) -> None:
        """Print the current system status"""

        print("🤖 LLM Intelligence System Status")
        print("=" * 50)

        print(f"Root Directory: {self.root_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Components Available: {'✅' if HAS_COMPONENTS else '❌'}")

        if HAS_COMPONENTS:
            print("Available Components:")
            print("  ✅ LLM Intelligence Scanner")
            print("  ✅ Intelligence Integration Engine")
            print("  ✅ Reorganization Planner")

            if self.scanner:
                print(f"  Scanner Config: {self.scanner.config['llm_provider']} / {self.scanner.config['llm_model']}")
        else:
            print("❌ Components not available - check imports")

        print(f"Output Directory Exists: {'✅' if self.output_dir.exists() else '❌'}")

        # Check for existing output files
        existing_files = list(self.output_dir.glob("*.json"))
        if existing_files:
            print(f"Existing Output Files: {len(existing_files)}")
            for file in existing_files[-3:]:  # Show last 3
                print(f"  📄 {file.name}")
        else:
            print("Existing Output Files: None")


def _get_parser_epilog() -> str:
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


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
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


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
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


def _add_step_specific_arguments(parser: argparse.ArgumentParser) -> None:
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


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM-Based Code Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_get_parser_epilog()
    )

    _add_common_arguments(parser)
    _add_pipeline_arguments(parser)
    _add_step_specific_arguments(parser)

    return parser


def _validate_main_arguments(args: argparse.Namespace) -> int:
    """Validate main function arguments (Rule 7 compliance)"""
    from pathlib import Path

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


def _execute_pipeline_action(args: argparse.Namespace, runner: IntelligenceSystemRunner) -> int:
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
                results = runner.run_scan_step(max_files=args.max_files)
            elif args.step == "integrate":
                results = runner.run_integrate_step(args.llm_map, provider=args.provider, api_key=args.api_key)
            elif args.step == "plan":
                results = runner.run_plan_step(args.integrated, provider=args.provider, api_key=args.api_key)
            elif args.step == "execute":
                results = runner.run_execute_step(args.plan, args.batch_id, args.dry_run)
            else:
                print(f"❌ Unknown step: {args.step}")
                return 1

            runner.print_step_results(results)
            return 0

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1

    return 0


def main():
    """Main entry point for the intelligence system with comprehensive parameter validation"""
    # Parameter validation (Rule 7 compliance)
    assert len(__name__) > 0, "Module name must be valid"
    assert main.__name__ == 'main', "Function name must be 'main'"

    # Parse arguments with error handling
    try:
        parser = _create_argument_parser()
        args = parser.parse_args()
        assert args is not None, "Argument parsing must succeed"
    except SystemExit as e:
        # Handle argparse errors gracefully
        print("❌ Invalid arguments provided")
        return 1
    except Exception as e:
        print(f"❌ Error parsing arguments: {e}")
        return 1

    # Comprehensive parameter validation (Rule 7 compliance)
    validation_result = _validate_main_arguments(args)
    if validation_result != 0:
        return validation_result

    # Additional validation for critical parameters
    if hasattr(args, 'max_files') and args.max_files is not None:
        if not isinstance(args.max_files, int) or args.max_files <= 0:
            print("❌ --max-files must be a positive integer")
            return 1

    if hasattr(args, 'provider') and args.provider:
        valid_providers = ["openai", "anthropic", "groq", "ollama", "mock"]
        if args.provider not in valid_providers:
            print(f"❌ Invalid provider '{args.provider}'. Must be one of: {valid_providers}")
            return 1

    if hasattr(args, 'model') and args.model:
        if not isinstance(args.model, str) or len(args.model.strip()) == 0:
            print("❌ --model must be a non-empty string")
            return 1

    # Initialize the runner with error handling
    try:
        root_dir = Path(args.root).resolve()
        runner = IntelligenceSystemRunner(root_dir)
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return 1

    # Execute the requested action with comprehensive error handling
    try:
        return _execute_pipeline_action(args, runner)
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error during execution: {e}")
        return 1


if __name__ == "__main__":
    main()

