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

        results = {
            'pipeline_started': datetime.now().isoformat(),
            'steps_completed': [],
            'output_files': {},
            'success': False,
            'errors': []
        }

        try:
            print("🚀 Starting Complete Intelligence System Pipeline")
            print("=" * 60)

            # Step 1: LLM Intelligence Scanning
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

            # Step 2: Intelligence Integration
            print("\n🔗 Step 2: Intelligence Integration")
            print("-" * 40)

            if self.integration_engine:
                integrated_intelligence = self.integration_engine.integrate_intelligence(llm_intelligence_map.__dict__ if hasattr(llm_intelligence_map, '__dict__') else llm_intelligence_map)

                integrated_file = self.output_dir / f"integrated_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.integration_engine.save_integration_results(
                    integrated_intelligence, None, integrated_file
                )

                results['output_files']['integrated_intelligence'] = str(integrated_file)
                results['steps_completed'].append('intelligence_integration')
                print(f"✅ Intelligence integration completed - {len(integrated_intelligence)} entries processed")
            else:
                raise Exception("Integration engine not available")

            # Step 3: Reorganization Planning
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
        """Run the planning step"""

        results = {
            'step': 'plan',
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            with open(llm_map_file, 'r', encoding='utf-8') as f:
                llm_map = json.load(f)

            with open(integrated_file, 'r', encoding='utf-8') as f:
                integrated_data = json.load(f)

            if self.planner:
                # Reconstruct integrated intelligence objects
                from intelligence_integration_engine import IntegratedIntelligence
                integrated_intelligence = [IntegratedIntelligence(**item) for item in integrated_data.get('integrated_intelligence', [])]

                reorganization_plan = self.planner.create_reorganization_plan(llm_map, integrated_intelligence)

                output_file = self.output_dir / f"reorganization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.planner.save_reorganization_plan(reorganization_plan, output_file)

                results['output_files']['reorganization_plan'] = str(output_file)
                results['total_batches'] = reorganization_plan.total_batches
                results['total_tasks'] = reorganization_plan.total_tasks
                results['estimated_hours'] = reorganization_plan.summary['batch_statistics']['total_estimated_hours']
                results['success'] = True
                results['completed'] = datetime.now().isoformat()

        except Exception as e:
            results['errors'].append(str(e))

        return results

    def _run_execute_step(self, plan_file: str, batch_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Run the execution step"""

        results = {
            'step': 'execute',
            'started': datetime.now().isoformat(),
            'success': False,
            'errors': []
        }

        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)

            if self.planner:
                # Reconstruct plan object
                from reorganization_planner import ReorganizationBatch
                batches = [ReorganizationBatch(**batch) for batch in plan_data['batches']]
                from reorganization_planner import DetailedReorganizationPlan
                plan = DetailedReorganizationPlan(**{k: v for k, v in plan_data.items() if k != 'batches'}, batches=batches)

                execution_results = self.planner.execute_plan_batch(plan, batch_id, dry_run)

                results.update(execution_results)
                results['success'] = execution_results.get('success', False)
                results['completed'] = datetime.now().isoformat()

        except Exception as e:
            results['errors'].append(str(e))

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report of the pipeline results"""

        report = []
        report.append("# LLM Intelligence System Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        if results.get('success'):
            report.append("## ✅ Pipeline Status: SUCCESS")
        else:
            report.append("## ❌ Pipeline Status: FAILED")

        report.append("")
        report.append("## Pipeline Steps Completed")
        for step in results.get('steps_completed', []):
            report.append(f"- ✅ {step.replace('_', ' ').title()}")

        if results.get('errors'):
            report.append("")
            report.append("## Errors Encountered")
            for error in results['errors']:
                report.append(f"- ❌ {error}")

        report.append("")
        report.append("## Output Files Generated")
        for file_type, file_path in results.get('output_files', {}).items():
            report.append(f"- {file_type.replace('_', ' ').title()}: `{file_path}`")

        # Try to load and summarize the reorganization plan
        if 'reorganization_plan' in results.get('output_files', {}):
            plan_file = results['output_files']['reorganization_plan']
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)

                report.append("")
                report.append("## Reorganization Plan Summary")
                report.append(f"- Total Tasks: {plan_data.get('total_tasks', 0)}")
                report.append(f"- Total Batches: {plan_data.get('total_batches', 0)}")
                report.append(".1f")

                summary = plan_data.get('summary', {})
                if 'task_statistics' in summary:
                    task_stats = summary['task_statistics']
                    report.append(f"- High Priority Tasks: {task_stats.get('high_priority', 0)}")
                    report.append(f"- Security Modules: {task_stats.get('security_modules', 0)}")

                report.append("")
                report.append("### Reorganization Batches")
                for i, batch in enumerate(plan_data.get('batches', []), 1):
                    report.append(f"{i}. **{batch['batch_name']}** ({batch['risk_level']}) - {len(batch['tasks'])} tasks")

            except Exception as e:
                report.append(f"Could not load plan details: {e}")

        report.append("")
        report.append("## Next Steps")
        report.append("1. Review the generated intelligence files")
        report.append("2. Examine the reorganization plan")
        report.append("3. Execute batches in order (starting with low-risk)")
        report.append("4. Run tests after each batch completion")
        report.append("5. Monitor for any import or functionality issues")

        return "\n".join(report)

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


def main():
    """Main entry point for the intelligence system"""

    parser = argparse.ArgumentParser(
        description="LLM-Based Code Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    )

    parser.add_argument("--root", type=str, default=".",
                      help="Root directory to analyze (default: current)")

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

    args = parser.parse_args()

    # Initialize the runner
    root_dir = Path(args.root).resolve()
    runner = IntelligenceSystemRunner(root_dir)

    if args.status:
        # Show system status
        runner.print_system_status()
        return

    if args.generate_report:
        # Generate report from existing results
        if not args.results_file:
            print("❌ --results-file required for report generation")
            return

        try:
            with open(args.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            report = runner.generate_report(results)
            print(report)

            # Save report to file
            report_file = root_dir / "tools" / "codebase_reorganizer" / "intelligence_output" / f"intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"\n📄 Report saved to: {report_file}")

        except Exception as e:
            print(f"❌ Error generating report: {e}")
        return

    if args.full_pipeline:
        # Run the complete pipeline
        results = runner.run_full_pipeline(
            max_files=args.max_files,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key
        )

        # Save results
        results_file = root_dir / "tools" / "codebase_reorganizer" / "intelligence_output" / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Pipeline results saved to: {results_file}")

        # Generate and display report
        if results['success']:
            report = runner.generate_report(results)
            print("\n" + "="*60)
            print("📋 PIPELINE REPORT")
            print("="*60)
            print(report)

    elif args.step:
        # Run a single step
        step_kwargs = {}

        if args.step == 'scan':
            step_kwargs.update({
                'max_files': args.max_files,
                'provider': args.provider,
                'model': args.model,
                'api_key': args.api_key
            })

        elif args.step == 'integrate':
            if not args.llm_map:
                print("❌ --llm-map required for integrate step")
                return
            step_kwargs['llm_map_file'] = args.llm_map

        elif args.step == 'plan':
            if not args.llm_map or not args.integrated:
                print("❌ --llm-map and --integrated required for plan step")
                return
            step_kwargs.update({
                'llm_map_file': args.llm_map,
                'integrated_file': args.integrated
            })

        elif args.step == 'execute':
            if not args.plan or not args.batch_id:
                print("❌ --plan and --batch-id required for execute step")
                return
            step_kwargs.update({
                'plan_file': args.plan,
                'batch_id': args.batch_id,
                'dry_run': args.dry_run
            })

        results = runner.run_single_step(args.step, **step_kwargs)

        if results['success']:
            print(f"✅ Step '{args.step}' completed successfully")
            if 'output_files' in results:
                for file_type, file_path in results['output_files'].items():
                    print(f"  📄 {file_type}: {file_path}")
        else:
            print(f"❌ Step '{args.step}' failed")
            for error in results.get('errors', []):
                print(f"  Error: {error}")


if __name__ == "__main__":
    main()
