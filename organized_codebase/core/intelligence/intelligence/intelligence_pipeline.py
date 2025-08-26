#!/usr/bin/env python3
"""
Intelligence Pipeline Execution Module
======================================

Handles the execution of the complete intelligence system pipeline.
Contains all pipeline running, step execution, and data handling logic.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class IntelligencePipelineExecutor:
    """Handles execution of intelligence system pipeline"""

    def __init__(self, scanner, integration_engine, planner, output_dir: Path):
        """Initialize the pipeline executor with components"""
        self.scanner = scanner
        self.integration_engine = integration_engine
        self.planner = planner
        self.output_dir = output_dir

    def initialize_pipeline_results(self) -> Dict[str, Any]:
        """Initialize pipeline results structure"""
        return {
            'pipeline_started': datetime.now().isoformat(),
            'steps_completed': [],
            'output_files': {},
            'success': False,
            'errors': []
        }

    def run_llm_scanning_step(self, results: Dict[str, Any], max_files: Optional[int],
                            provider: str, model: str, api_key: Optional[str]) -> Dict[str, Any]:
        """Run LLM intelligence scanning step"""
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

    def run_integration_step(self, results: Dict[str, Any], llm_intelligence_map: Any) -> tuple:
        """Run intelligence integration step"""
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

    def run_planning_step(self, results: Dict[str, Any], llm_intelligence_map: Any,
                        integrated_intelligence: Any) -> Dict[str, Any]:
        """Run reorganization planning step"""
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

    def finalize_pipeline_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize pipeline results and display summary"""
        # Pipeline completed successfully
        results['success'] = True
        results['pipeline_completed'] = datetime.now().isoformat()

        print("\n🎉 Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📊 Results saved to: {self.output_dir}")
        print(f"📁 Output files: {len(results['output_files'])}")
        print(f"✅ Steps completed: {', '.join(results['steps_completed'])}")

        return results

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
        results = self.initialize_pipeline_results()

        try:
            print("🚀 Starting Complete Intelligence System Pipeline")
            print("=" * 60)

            # Step 1: Run LLM scanning
            results = self.run_llm_scanning_step(results, max_files, provider, model, api_key)

            # Step 2: Run intelligence integration
            results, integrated_intelligence, llm_intelligence_map = self.run_integration_step(results, llm_intelligence_map)

            # Step 3: Run reorganization planning
            results = self.run_planning_step(results, llm_intelligence_map, integrated_intelligence)

            # Finalize results
            results = self.finalize_pipeline_results(results)

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
                print(f"✅ Scan step completed - {intelligence_map.total_files_scanned} files analyzed")
            else:
                raise Exception("Scanner not available")

        except Exception as e:
            results['errors'].append(str(e))
            results['failed'] = datetime.now().isoformat()
            print(f"❌ Scan step failed: {e}")

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
            # Load LLM map data
            with open(llm_map_file, 'r') as f:
                llm_map_data = json.load(f)

            # Reconstruct LLM intelligence map object
            llm_map = self._reconstruct_llm_map(llm_map_data)

            if self.integration_engine:
                integrated_intelligence = self.integration_engine.integrate_intelligence(llm_map)

                output_file = self.output_dir / f"integrated_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.integration_engine.save_integration_results(integrated_intelligence, None, output_file)

                results['output_files']['integrated_intelligence'] = str(output_file)
                results['entries_processed'] = len(integrated_intelligence)
                results['success'] = True
                results['completed'] = datetime.now().isoformat()
                print(f"✅ Integration step completed - {len(integrated_intelligence)} entries processed")
            else:
                raise Exception("Integration engine not available")

        except Exception as e:
            results['errors'].append(str(e))
            results['failed'] = datetime.now().isoformat()
            print(f"❌ Integration step failed: {e}")

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
            # Load data files
            with open(llm_map_file, 'r') as f:
                llm_map_data = json.load(f)

            with open(integrated_file, 'r') as f:
                integrated_data = json.load(f)

            # Reconstruct objects
            llm_map = self._reconstruct_llm_map(llm_map_data)
            integrated_intelligence = self._reconstruct_integrated_intelligence(integrated_data)

            if self.planner:
                reorganization_plan = self.planner.create_reorganization_plan(llm_map, integrated_intelligence)

                output_file = self.output_dir / f"reorganization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.planner.save_reorganization_plan(reorganization_plan, output_file)

                results['output_files']['reorganization_plan'] = str(output_file)
                results['batches_created'] = reorganization_plan.total_batches
                results['success'] = True
                results['completed'] = datetime.now().isoformat()
                print(f"✅ Planning step completed - {reorganization_plan.total_batches} batches created")
            else:
                raise Exception("Planner not available")

        except Exception as e:
            results['errors'].append(str(e))
            results['failed'] = datetime.now().isoformat()
            print(f"❌ Planning step failed: {e}")

        return results

    def _run_execute_step(self, plan_file: str, batch_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Run the execution step"""
        results = {
            'step': 'execute',
            'started': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'errors': []
        }

        try:
            # Load plan data
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)

            # Reconstruct plan object
            plan = self._reconstruct_plan(plan_data)

            if self.planner:
                execution_results = self.planner.execute_reorganization_batch(plan, batch_id, dry_run)

                results['execution_results'] = execution_results
                results['dry_run'] = dry_run
                results['batch_id'] = batch_id
                results['success'] = True
                results['completed'] = datetime.now().isoformat()
                print(f"✅ Execution step completed - Batch {batch_id} {'simulated' if dry_run else 'executed'}")
            else:
                raise Exception("Planner not available")

        except Exception as e:
            results['errors'].append(str(e))
            results['failed'] = datetime.now().isoformat()
            print(f"❌ Execution step failed: {e}")

        return results

    def _reconstruct_llm_map(self, llm_map_data: Dict) -> Any:
        """Reconstruct LLM intelligence map from data"""
        # This would reconstruct the LLM map object from saved data
        # For now, return the data as-is
        return llm_map_data

    def _reconstruct_integrated_intelligence(self, integrated_data: Dict) -> List:
        """Reconstruct integrated intelligence from data"""
        # This would reconstruct the integrated intelligence list from saved data
        # For now, return the data as-is
        return integrated_data.get('integrated_intelligence', [])

    def _reconstruct_plan(self, plan_data: Dict) -> Any:
        """Reconstruct reorganization plan from data"""
        # This would reconstruct the plan object from saved data
        # For now, return the data as-is
        return plan_data
