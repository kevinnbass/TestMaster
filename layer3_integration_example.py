"""
Layer 3: Intelligent Orchestration - Integration Example

Demonstrates the complete Layer 3 system working together:
- File tagging and classification
- Work distribution logic
- Automated investigation
- Smart handoff system
- Functional structure mapping
- Coverage intelligence
- Regression tracking

This example shows how TestMaster coordinates intelligent analysis
and seamless handoffs with Claude Code.
"""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Layer 3 imports
from testmaster.orchestrator import (
    FileTagger, WorkDistributor, AutoInvestigator, HandoffManager
)
from testmaster.overview import (
    StructureMapper, CoverageIntelligence, RegressionTracker
)

# Work distribution types
from testmaster.orchestrator.work_distributor import WorkType, HandoffTarget
from testmaster.orchestrator.handoff_manager import HandoffType

# Layer management
from testmaster.core.layer_manager import LayerManager


class Layer3OrchestrationDemo:
    """
    Complete Layer 3 orchestration demonstration.
    
    Shows how TestMaster intelligently analyzes a codebase,
    identifies issues, and coordinates with Claude Code.
    """
    
    def __init__(self, project_path: str):
        """Initialize the orchestration demo."""
        self.project_path = Path(project_path)
        
        # Verify Layer 3 is enabled
        self.layer_manager = LayerManager()
        if not self.layer_manager.is_enabled("layer3_orchestration"):
            raise RuntimeError("Layer 3: Intelligent Orchestration must be enabled")
        
        # Initialize all Layer 3 components
        print("🚀 Initializing Layer 3: Intelligent Orchestration...")
        
        self.file_tagger = FileTagger()
        self.work_distributor = WorkDistributor()
        self.auto_investigator = AutoInvestigator()
        self.handoff_manager = HandoffManager()
        
        self.structure_mapper = StructureMapper(str(self.project_path))
        self.coverage_intelligence = CoverageIntelligence(str(self.project_path))
        self.regression_tracker = RegressionTracker()
        
        print("✅ Layer 3 components initialized")
        print()
    
    async def run_complete_orchestration(self):
        """Run complete orchestration workflow."""
        print("🎯 Starting Complete Layer 3 Orchestration Workflow")
        print("=" * 60)
        
        # Step 1: Analyze codebase structure
        await self._analyze_codebase_structure()
        
        # Step 2: Analyze coverage intelligence
        await self._analyze_coverage_intelligence()
        
        # Step 3: Tag and classify files
        await self._tag_and_classify_files()
        
        # Step 4: Simulate some test failures and track regressions
        await self._simulate_test_failures()
        
        # Step 5: Investigate issues automatically
        await self._investigate_issues()
        
        # Step 6: Distribute work intelligently
        await self._distribute_work()
        
        # Step 7: Create intelligent handoffs
        await self._create_handoffs()
        
        # Step 8: Generate comprehensive reports
        await self._generate_reports()
        
        print("\n🎉 Layer 3 Orchestration Complete!")
        print("TestMaster is now intelligently coordinating with Claude Code")
    
    async def _analyze_codebase_structure(self):
        """Step 1: Analyze codebase structure."""
        print("📊 Step 1: Analyzing Codebase Structure")
        print("-" * 40)
        
        # Analyze functional structure
        functional_map = self.structure_mapper.analyze_structure()
        
        print(f"   🗺️ Analyzed {len(functional_map.modules)} modules")
        print(f"   🔗 Found {len(functional_map.relationships)} relationships")
        print(f"   🏗️ Identified {len(functional_map.core_modules)} core modules")
        print(f"   📋 Detected {len(functional_map.architectural_patterns)} patterns")
        
        if functional_map.design_issues:
            print(f"   ⚠️ Found {len(functional_map.design_issues)} design issues:")
            for issue in functional_map.design_issues[:3]:
                print(f"      • {issue}")
        
        # Get critical modules for priority attention
        critical_modules = self.structure_mapper.get_critical_modules()
        if critical_modules:
            print(f"   🎯 Critical modules requiring attention:")
            for module in critical_modules[:3]:
                print(f"      • {Path(module).name}")
        
        print()
    
    async def _analyze_coverage_intelligence(self):
        """Step 2: Analyze coverage intelligence."""
        print("🎯 Step 2: Analyzing Coverage Intelligence")
        print("-" * 40)
        
        # Simulate some coverage data
        coverage_data = {
            "src/auth/login.py": 45.0,
            "src/payment/processor.py": 60.0,
            "src/api/endpoints.py": 30.0,
            "src/utils/helpers.py": 85.0
        }
        
        # Analyze coverage intelligence
        intelligence_report = self.coverage_intelligence.analyze_coverage_intelligence(coverage_data)
        
        print(f"   📈 Overall coverage: {intelligence_report.overall_coverage:.1f}%")
        print(f"   🛤️ Critical path coverage: {intelligence_report.critical_path_coverage:.1f}%")
        print(f"   🔍 Found {intelligence_report.total_coverage_gaps} coverage gaps")
        print(f"   ⚠️ High-risk modules: {len(intelligence_report.high_risk_modules)}")
        
        # Show top priority gaps
        if intelligence_report.top_priority_gaps:
            print(f"   🚨 Top priority coverage gaps:")
            for gap in intelligence_report.top_priority_gaps[:3]:
                print(f"      • {gap.gap_type.value} in {Path(gap.module_path).name} (score: {gap.priority_score:.1f})")
        
        # Show critical uncovered paths
        if intelligence_report.critical_uncovered_paths:
            print(f"   🔴 Critical uncovered paths:")
            for path in intelligence_report.critical_uncovered_paths[:3]:
                print(f"      • {path.function_name} in {Path(path.module_path).name} ({path.criticality.value})")
        
        print()
    
    async def _tag_and_classify_files(self):
        """Step 3: Tag and classify files."""
        print("🏷️ Step 3: Tagging and Classifying Files")
        print("-" * 40)
        
        # Get some sample files to tag
        python_files = list(self.project_path.rglob("*.py"))[:10]  # Limit for demo
        
        for file_path in python_files:
            if file_path.exists():
                # Classify and tag file
                classification = self.file_tagger.classify_file(file_path)
                
                if classification:
                    print(f"   📁 {file_path.name}:")
                    print(f"      Type: {classification.module_type.value}")
                    print(f"      Status: {classification.status.value}")
                    print(f"      Priority: {classification.priority.value}")
                    print(f"      Complexity: {classification.complexity_score:.1f}")
                    
                    # Update tags based on analysis
                    self.file_tagger.update_file_tags(
                        str(file_path), 
                        {"analysis_date": datetime.now().isoformat()}
                    )
        
        print(f"   ✅ Tagged {len(python_files)} files")
        print()
    
    async def _simulate_test_failures(self):
        """Step 4: Simulate test failures and track regressions."""
        print("📈 Step 4: Simulating Test Failures for Regression Tracking")
        print("-" * 40)
        
        # Simulate some test failures
        test_failures = [
            {
                "test_file": "tests/test_auth.py",
                "test_function": "test_login_success",
                "error_message": "AssertionError: Expected status 200, got 401",
                "module_under_test": "src/auth/login.py"
            },
            {
                "test_file": "tests/test_payment.py", 
                "test_function": "test_process_payment",
                "error_message": "TimeoutError: Payment processing timeout after 30s",
                "module_under_test": "src/payment/processor.py"
            },
            {
                "test_file": "tests/test_api.py",
                "test_function": "test_endpoint_validation",
                "error_message": "ImportError: No module named 'validators'",
                "module_under_test": "src/api/endpoints.py"
            }
        ]
        
        failure_ids = []
        for failure in test_failures:
            failure_id = self.regression_tracker.record_failure(
                test_file=failure["test_file"],
                test_function=failure["test_function"],
                error_message=failure["error_message"],
                module_under_test=failure["module_under_test"],
                environment="ci_pipeline"
            )
            failure_ids.append(failure_id)
            print(f"   📝 Recorded failure: {failure['test_function']}")
        
        # Analyze regression trends
        regression_summary = self.regression_tracker.analyze_regression_trends(days=7)
        print(f"   📊 Regression Analysis:")
        print(f"      Total failures: {regression_summary.total_failures}")
        print(f"      Failure trend: {regression_summary.failure_trend}")
        print(f"      High-risk modules: {len(regression_summary.high_risk_modules)}")
        
        # Generate failure predictions
        predictions = self.regression_tracker.predict_potential_failures()
        if predictions:
            print(f"   🔮 Generated {len(predictions)} failure predictions")
            for pred in predictions[:2]:
                print(f"      • {pred.target_module}: {pred.predicted_failure_type.value} ({pred.confidence.value})")
        
        print()
    
    async def _investigate_issues(self):
        """Step 5: Investigate issues automatically."""
        print("🔍 Step 5: Automated Investigation")
        print("-" * 40)
        
        # Investigate some files that need attention
        investigation_targets = [
            "src/auth/login.py",
            "src/payment/processor.py", 
            "src/api/endpoints.py"
        ]
        
        investigation_ids = []
        for target in investigation_targets:
            if Path(target).exists():
                # Start investigation
                investigation_id = self.auto_investigator.start_investigation(
                    investigation_type="idle_analysis",
                    target=target,
                    priority="normal"
                )
                investigation_ids.append(investigation_id)
                print(f"   🔍 Started investigation: {Path(target).name}")
        
        # Simulate investigation completion and get results
        for inv_id in investigation_ids:
            investigation = self.auto_investigator.get_investigation_status(inv_id)
            if investigation:
                print(f"   📋 Investigation {inv_id[:8]}:")
                print(f"      Status: {investigation.status.value}")
                print(f"      Evidence count: {len(investigation.evidence_collected)}")
                
                # Simulate some findings
                if investigation.evidence_collected:
                    print(f"      Key findings: {investigation.evidence_collected[0][:50]}...")
        
        print()
    
    async def _distribute_work(self):
        """Step 6: Distribute work intelligently."""
        print("🎯 Step 6: Intelligent Work Distribution")
        print("-" * 40)
        
        # Create various work items
        work_items = [
            {
                "work_type": WorkType.TEST_FAILURE,
                "title": "Fix Login Test Failure", 
                "description": "AssertionError in test_login_success",
                "source_file": "src/auth/login.py",
                "test_file": "tests/test_auth.py",
                "error_message": "AssertionError: Expected status 200, got 401"
            },
            {
                "work_type": WorkType.COVERAGE_GAP,
                "title": "Improve Payment Processor Coverage",
                "description": "Coverage is only 60%, needs improvement",
                "source_file": "src/payment/processor.py"
            },
            {
                "work_type": WorkType.IDLE_MODULE,
                "title": "Review Idle API Endpoints",
                "description": "Module has been idle for 2+ hours",
                "source_file": "src/api/endpoints.py"
            },
            {
                "work_type": WorkType.BREAKING_CHANGE,
                "title": "Critical Import Error",
                "description": "ImportError breaking test suite",
                "source_file": "src/api/endpoints.py",
                "error_message": "ImportError: No module named 'validators'"
            }
        ]
        
        # Add work items and make distribution decisions
        work_decisions = []
        for item in work_items:
            # Add work item
            item_id = self.work_distributor.add_work_item(**item)
            
            # Make handoff decision
            decision = self.work_distributor.make_handoff_decision(item_id)
            if decision:
                work_decisions.append(decision)
                print(f"   📋 {item['title']}:")
                print(f"      → {decision.target.value} ({decision.reason.value})")
                print(f"      Confidence: {decision.confidence}%")
        
        # Batch similar items
        batches = self.work_distributor.batch_similar_items()
        if batches:
            print(f"   📦 Created {len(batches)} work batches for efficiency")
        
        # Get distribution statistics
        stats = self.work_distributor.get_distribution_statistics()
        print(f"   📊 Work Distribution Summary:")
        print(f"      Total items: {stats['total_work_items']}")
        print(f"      TestMaster auto: {stats['target_distribution'].get('testmaster_auto', 0)}")
        print(f"      Claude Code: {stats['target_distribution'].get('claude_code', 0)}")
        
        print()
    
    async def _create_handoffs(self):
        """Step 7: Create intelligent handoffs."""
        print("🤝 Step 7: Creating Smart Handoffs")
        print("-" * 40)
        
        # Create investigation handoff
        investigation_results = {
            "total_findings": 3,
            "critical_findings": 1,
            "high_findings": 1,
            "evidence_count": 5
        }
        
        investigation_handoff = self.handoff_manager.create_investigation_handoff(
            target="src/auth/login.py",
            investigation_results=investigation_results,
            priority="high"
        )
        print(f"   🔍 Investigation handoff: {investigation_handoff[:8]}")
        
        # Create work delegation handoff
        work_item = {
            "title": "Fix Critical Import Error",
            "work_type": "breaking_change",
            "complexity_level": "high",
            "estimated_effort_minutes": 45,
            "source_file": "src/api/endpoints.py",
            "priority": "critical"
        }
        
        delegation_handoff = self.handoff_manager.create_work_delegation_handoff(work_item)
        print(f"   📋 Work delegation handoff: {delegation_handoff[:8]}")
        
        # Create escalation handoff
        escalation_handoff = self.handoff_manager.create_escalation_handoff(
            issue_description="Persistent test failures in authentication",
            failed_attempts=[
                "Automated test repair",
                "Dependency update",
                "Configuration reset"
            ],
            target="src/auth/login.py",
            error_details="Multiple authentication tests failing after recent changes"
        )
        print(f"   🚨 Escalation handoff: {escalation_handoff[:8]}")
        
        # Get handoff statistics
        handoff_stats = self.handoff_manager.get_handoff_statistics()
        print(f"   📊 Handoff Summary:")
        print(f"      Total handoffs: {handoff_stats['total_handoffs']}")
        print(f"      Active handoffs: {handoff_stats['active_handoffs']}")
        print(f"      Patterns learned: {handoff_stats['patterns_learned']}")
        
        print()
    
    async def _generate_reports(self):
        """Step 8: Generate comprehensive reports."""
        print("📄 Step 8: Generating Comprehensive Reports")
        print("-" * 40)
        
        # Generate structure report
        structure_stats = self.structure_mapper.get_structure_statistics()
        print(f"   🗺️ Structure Analysis:")
        print(f"      Modules analyzed: {structure_stats['total_modules']}")
        print(f"      Architectural patterns: {structure_stats['architectural_patterns']}")
        print(f"      Design issues: {structure_stats['design_issues']}")
        
        # Generate coverage report
        coverage_stats = self.coverage_intelligence.get_coverage_statistics()
        print(f"   🎯 Coverage Intelligence:")
        print(f"      Overall coverage: {coverage_stats['overall_coverage']:.1f}%")
        print(f"      Critical paths: {coverage_stats['total_critical_paths']}")
        print(f"      Coverage gaps: {coverage_stats['total_coverage_gaps']}")
        
        # Generate regression report
        regression_stats = self.regression_tracker.get_regression_statistics()
        print(f"   📈 Regression Tracking:")
        print(f"      Total failures tracked: {regression_stats['total_failures']}")
        print(f"      Resolution rate: {regression_stats['resolution_rate']:.1f}%")
        print(f"      Prediction accuracy: {regression_stats['prediction_accuracy']:.1f}%")
        
        # Export detailed reports
        try:
            self.structure_mapper.export_structure_report("layer3_structure_report.json")
            self.coverage_intelligence.export_coverage_report("layer3_coverage_report.json")
            self.regression_tracker.export_regression_report("layer3_regression_report.json")
            self.handoff_manager.export_handoff_report("layer3_handoff_report.json")
            self.work_distributor.export_distribution_report("layer3_distribution_report.json")
            
            print(f"   💾 Exported 5 detailed analysis reports")
        except Exception as e:
            print(f"   ⚠️ Error exporting reports: {e}")
        
        print()


def demonstrate_layer3_patterns():
    """Demonstrate the key Layer 3 patterns and capabilities."""
    print("🎯 Layer 3: Intelligent Orchestration Patterns")
    print("=" * 50)
    
    patterns = [
        {
            "name": "Agent-Squad Configuration-Driven Classification",
            "component": "FileTagger",
            "description": "Automatic file classification using configurable rules",
            "benefit": "Consistent, scalable file categorization"
        },
        {
            "name": "OpenAI Swarm Function-Based Handoff",
            "component": "WorkDistributor", 
            "description": "Dynamic work routing based on complexity and capability",
            "benefit": "Optimal work distribution between automated and manual systems"
        },
        {
            "name": "LangGraph Supervisor Delegation",
            "component": "AutoInvestigator",
            "description": "Automated investigation with task delegation patterns",
            "benefit": "Systematic issue analysis and evidence collection"
        },
        {
            "name": "OpenAI Swarm Context Preservation",
            "component": "HandoffManager",
            "description": "Rich context packaging for seamless handoffs",
            "benefit": "Intelligent communication with Claude Code"
        }
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern['name']}")
        print(f"   Component: {pattern['component']}")
        print(f"   Pattern: {pattern['description']}")
        print(f"   Benefit: {pattern['benefit']}")
        print()


async def main():
    """Main demonstration function."""
    print("🚀 TestMaster Layer 3: Intelligent Orchestration")
    print("=" * 60)
    print()
    
    # Show pattern demonstrations
    demonstrate_layer3_patterns()
    
    # Run the complete orchestration demo
    project_path = "."  # Current directory for demo
    demo = Layer3OrchestrationDemo(project_path)
    
    try:
        await demo.run_complete_orchestration()
        
        print("\n" + "=" * 60)
        print("🎉 Layer 3 Orchestration Demo Complete!")
        print()
        print("TestMaster now provides:")
        print("• 🏷️ Intelligent file tagging and classification")
        print("• 🎯 Smart work distribution between TestMaster and Claude Code")
        print("• 🔍 Automated investigation with evidence collection")
        print("• 🤝 Context-rich handoffs with pattern learning")
        print("• 🗺️ Comprehensive codebase structure analysis")
        print("• 🎯 Critical path identification and coverage intelligence")
        print("• 📈 Predictive failure detection and regression tracking")
        print()
        print("The system is ready for intelligent test orchestration!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure Layer 3 is properly configured and enabled.")


if __name__ == "__main__":
    asyncio.run(main())