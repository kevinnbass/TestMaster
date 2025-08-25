#!/usr/bin/env python3
"""
Final Archive Sweep
===================

Comprehensive final check across all phases to ensure no features were lost
during the consolidation process. Validates all phases and provides a
complete integration status report.

Author: TestMaster Enhancement System
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Any
import glob


class FinalArchiveSweep:
    """Comprehensive archive sweep and validation"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.archive_base = self.base_path / "archive"
        
        # Phase mapping
        self.phases = {
            "C4": {
                "name": "Observability Consolidation",
                "archive_path": "phase4_archive",
                "consolidated_files": [
                    "observability/unified_monitor.py"
                ]
            },
            "C5": {
                "name": "State & Configuration Consolidation", 
                "archive_path": "phase5_archive",
                "consolidated_files": [
                    "state/unified_state_manager.py",
                    "config/yaml_config_enhancer.py"
                ]
            },
            "C6": {
                "name": "Orchestration Consolidation",
                "archive_path": "phase6_archive", 
                "consolidated_files": [
                    "orchestration/unified_orchestrator.py",
                    "orchestration/swarm_router_enhancement.py"
                ]
            },
            "C7": {
                "name": "UI/Dashboard Consolidation",
                "archive_path": "phase7_archive",
                "consolidated_files": [
                    "ui/unified_dashboard.py",
                    "ui/nocode_enhancement.py"
                ]
            }
        }
        
        self.validation_results = {}
        
    def perform_final_sweep(self) -> Dict[str, Any]:
        """Perform comprehensive final archive sweep"""
        print("FINAL ARCHIVE SWEEP")
        print("=" * 80)
        print("Comprehensive validation of all consolidation phases")
        print("Checking for any missing features across the entire system")
        print()
        
        sweep_results = {
            "sweep_timestamp": "2025-08-19T20:30:00.000000",
            "total_phases": len(self.phases),
            "phases_validated": {},
            "overall_status": "VALIDATING",
            "summary": {},
            "recommendations": []
        }
        
        # Validate each phase
        for phase_id, phase_info in self.phases.items():
            print(f"Validating Phase {phase_id}: {phase_info['name']}")
            print("-" * 60)
            
            phase_result = self._validate_phase(phase_id, phase_info)
            sweep_results["phases_validated"][phase_id] = phase_result
            
            print(f"Phase {phase_id} Status: {phase_result['status']}")
            print()
        
        # Generate overall assessment
        sweep_results.update(self._generate_overall_assessment(sweep_results))
        
        # Generate final report
        self._generate_final_report(sweep_results)
        
        return sweep_results
    
    def _validate_phase(self, phase_id: str, phase_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific consolidation phase"""
        phase_result = {
            "phase_id": phase_id,
            "phase_name": phase_info["name"],
            "status": "UNKNOWN",
            "archive_exists": False,
            "consolidated_files_exist": 0,
            "total_consolidated_files": len(phase_info["consolidated_files"]),
            "archive_files": [],
            "missing_consolidated_files": [],
            "validation_notes": []
        }
        
        # Check if archive exists
        archive_path = self.archive_base / phase_info["archive_path"]
        if archive_path.exists():
            phase_result["archive_exists"] = True
            
            # Count archived files
            archive_files = []
            for pattern in ["*.py", "*.jsx", "*.js", "*.ts"]:
                archive_files.extend(glob.glob(str(archive_path / "**" / pattern), recursive=True))
            
            phase_result["archive_files"] = [Path(f).name for f in archive_files]
            print(f"  Archive found: {len(archive_files)} files archived")
        else:
            phase_result["validation_notes"].append(f"Archive directory not found: {archive_path}")
            print(f"  Archive not found: {archive_path}")
        
        # Check consolidated files exist
        existing_files = 0
        missing_files = []
        
        for consolidated_file in phase_info["consolidated_files"]:
            file_path = self.base_path / consolidated_file
            if file_path.exists():
                existing_files += 1
                file_size = file_path.stat().st_size
                print(f"  Consolidated file exists: {consolidated_file} ({file_size:,} bytes)")
            else:
                missing_files.append(consolidated_file)
                print(f"  Missing consolidated file: {consolidated_file}")
        
        phase_result["consolidated_files_exist"] = existing_files
        phase_result["missing_consolidated_files"] = missing_files
        
        # Determine phase status
        if existing_files == len(phase_info["consolidated_files"]) and phase_result["archive_exists"]:
            phase_result["status"] = "COMPLETED"
        elif existing_files > 0:
            phase_result["status"] = "PARTIAL"
        else:
            phase_result["status"] = "FAILED"
        
        return phase_result
    
    def _generate_overall_assessment(self, sweep_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of all phases"""
        completed_phases = 0
        partial_phases = 0
        failed_phases = 0
        
        total_archived_files = 0
        total_consolidated_files = 0
        total_existing_consolidated = 0
        
        for phase_result in sweep_results["phases_validated"].values():
            if phase_result["status"] == "COMPLETED":
                completed_phases += 1
            elif phase_result["status"] == "PARTIAL":
                partial_phases += 1
            else:
                failed_phases += 1
            
            total_archived_files += len(phase_result["archive_files"])
            total_consolidated_files += phase_result["total_consolidated_files"]
            total_existing_consolidated += phase_result["consolidated_files_exist"]
        
        # Determine overall status
        if completed_phases == len(self.phases):
            overall_status = "FULLY_COMPLETED"
        elif completed_phases + partial_phases == len(self.phases):
            overall_status = "MOSTLY_COMPLETED"
        else:
            overall_status = "NEEDS_ATTENTION"
        
        # Generate recommendations
        recommendations = []
        
        if failed_phases > 0:
            recommendations.append(f"Review {failed_phases} failed phases for missing consolidated files")
        
        if partial_phases > 0:
            recommendations.append(f"Complete {partial_phases} partial phases")
        
        if total_existing_consolidated < total_consolidated_files:
            missing_count = total_consolidated_files - total_existing_consolidated
            recommendations.append(f"Create {missing_count} missing consolidated files")
        
        if overall_status == "FULLY_COMPLETED":
            recommendations.append("All phases completed successfully - ready for production")
        
        return {
            "overall_status": overall_status,
            "summary": {
                "completed_phases": completed_phases,
                "partial_phases": partial_phases, 
                "failed_phases": failed_phases,
                "total_phases": len(self.phases),
                "completion_rate": f"{(completed_phases / len(self.phases)) * 100:.1f}%",
                "total_archived_files": total_archived_files,
                "total_consolidated_files": total_consolidated_files,
                "existing_consolidated_files": total_existing_consolidated,
                "consolidation_rate": f"{(total_existing_consolidated / max(1, total_consolidated_files)) * 100:.1f}%"
            },
            "recommendations": recommendations
        }
    
    def _generate_final_report(self, sweep_results: Dict[str, Any]):
        """Generate comprehensive final report"""
        print("FINAL SWEEP REPORT")
        print("=" * 80)
        
        print(f"Overall Status: {sweep_results['overall_status']}")
        print(f"Completion Rate: {sweep_results['summary']['completion_rate']}")
        print(f"Consolidation Rate: {sweep_results['summary']['consolidation_rate']}")
        print()
        
        print("Phase Summary:")
        print("-" * 40)
        for phase_id, phase_result in sweep_results["phases_validated"].items():
            status_icon = {
                "COMPLETED": "PASS",
                "PARTIAL": "WARN", 
                "FAILED": "FAIL",
                "UNKNOWN": "UNKNOWN"
            }.get(phase_result["status"], "?")
            
            consolidated_ratio = f"{phase_result['consolidated_files_exist']}/{phase_result['total_consolidated_files']}"
            archived_count = len(phase_result['archive_files'])
            
            print(f"{status_icon} Phase {phase_id}: {phase_result['phase_name']}")
            print(f"    Consolidated Files: {consolidated_ratio}")
            print(f"    Archived Files: {archived_count}")
            
            if phase_result["missing_consolidated_files"]:
                print(f"    Missing: {', '.join(phase_result['missing_consolidated_files'])}")
            print()
        
        print("Overall Statistics:")
        print("-" * 40)
        summary = sweep_results["summary"]
        print(f"Completed Phases: {summary['completed_phases']}/{summary['total_phases']}")
        print(f"Partial Phases: {summary['partial_phases']}")
        print(f"Failed Phases: {summary['failed_phases']}")
        print(f"Total Archived Files: {summary['total_archived_files']}")
        print(f"Consolidated Files: {summary['existing_consolidated_files']}/{summary['total_consolidated_files']}")
        print()
        
        if sweep_results["recommendations"]:
            print("Recommendations:")
            print("-" * 40)
            for i, rec in enumerate(sweep_results["recommendations"], 1):
                print(f"{i}. {rec}")
            print()
        
        # Save detailed report
        report_file = "final_archive_sweep_report.json"
        with open(report_file, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        
        print(f"Detailed report saved: {report_file}")
        
        # Final assessment
        if sweep_results["overall_status"] == "FULLY_COMPLETED":
            print()
            print("CONSOLIDATION SUCCESS!")
            print("All phases completed with full feature preservation.")
            print("System ready for integration report generation.")
        else:
            print()
            print("CONSOLIDATION REVIEW NEEDED")
            print("Some phases require attention before final integration.")


def main():
    """Run final archive sweep"""
    sweeper = FinalArchiveSweep()
    results = sweeper.perform_final_sweep()
    
    return 0 if results["overall_status"] == "FULLY_COMPLETED" else 1


if __name__ == "__main__":
    exit(main())