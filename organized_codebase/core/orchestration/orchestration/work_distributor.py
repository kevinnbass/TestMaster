"""
Work Distribution Logic

Inspired by OpenAI Swarm's function-based handoff patterns for
dynamic routing and Agent-Squad's configuration-driven classification.

Features:
- Decide: TestMaster fix vs Claude Code fix
- Complexity assessment for handoff decisions  
- Batch similar issues for efficiency
- Dynamic routing based on issue type and complexity
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import json

from core.layer_manager import requires_layer


class WorkType(Enum):
    """Types of work items."""
    TEST_FAILURE = "test_failure"
    COVERAGE_GAP = "coverage_gap"
    IDLE_MODULE = "idle_module"
    BREAKING_CHANGE = "breaking_change"
    REFACTORING = "refactoring"
    NEW_FEATURE = "new_feature"
    BUG_FIX = "bug_fix"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"


class ComplexityLevel(IntEnum):
    """Complexity levels for work assessment."""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


class HandoffTarget(Enum):
    """Targets for work handoff."""
    TESTMASTER_AUTO = "testmaster_auto"
    TESTMASTER_MANUAL = "testmaster_manual"
    CLAUDE_CODE = "claude_code"
    HUMAN_REVIEW = "human_review"
    EXTERNAL_TOOL = "external_tool"


class HandoffReason(Enum):
    """Reasons for handoff decisions."""
    COMPLEXITY_TOO_HIGH = "complexity_too_high"
    REQUIRES_UNDERSTANDING = "requires_understanding"
    AUTOMATED_SOLUTION = "automated_solution"
    BATCH_PROCESSING = "batch_processing"
    HUMAN_JUDGMENT = "human_judgment"
    DOMAIN_EXPERTISE = "domain_expertise"


@dataclass
class WorkItem:
    """A work item that needs to be handled."""
    item_id: str
    work_type: WorkType
    title: str
    description: str
    
    # Source information
    source_file: Optional[str] = None
    test_file: Optional[str] = None
    related_files: List[str] = field(default_factory=list)
    
    # Assessment results
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    estimated_effort_minutes: int = 30
    
    # Context information
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    
    # Work tracking
    assigned_to: Optional[HandoffTarget] = None
    handoff_reason: Optional[HandoffReason] = None
    progress_status: str = "pending"
    resolution_notes: Optional[str] = None


@dataclass
class HandoffDecision:
    """Decision result for work handoff."""
    work_item: WorkItem
    target: HandoffTarget
    reason: HandoffReason
    confidence: float  # 0-100
    
    # Decision context
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_approach: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    
    # Batching information
    batch_id: Optional[str] = None
    batch_priority: int = 0
    
    # Timing
    decision_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


@dataclass
class BatchContext:
    """Context for batching similar work items."""
    batch_id: str
    work_items: List[WorkItem]
    batch_type: str
    total_estimated_effort: int
    recommended_target: HandoffTarget
    batch_reason: str
    created_at: datetime = field(default_factory=datetime.now)


class WorkDistributor:
    """
    Work distribution system using OpenAI Swarm handoff patterns.
    
    Analyzes work items and decides optimal routing between
    TestMaster automated fixes and Claude Code manual intervention.
    """
    
    @requires_layer("layer3_orchestration", "work_distribution")
    def __init__(self):
        """Initialize work distributor."""
        
        # Work item storage
        self._work_items: Dict[str, WorkItem] = {}
        self._handoff_decisions: Dict[str, HandoffDecision] = {}
        self._batch_contexts: Dict[str, BatchContext] = {}
        
        # Decision rules and patterns
        self._decision_rules: List[Callable[[WorkItem], Optional[HandoffDecision]]] = []
        self._complexity_assessors: Dict[WorkType, Callable[[WorkItem], ComplexityLevel]] = {}
        
        # Statistics
        self._stats = {
            'total_items': 0,
            'automated_assignments': 0,
            'manual_assignments': 0,
            'batched_items': 0,
            'decision_accuracy': 0.0
        }
        
        # Setup default decision logic
        self._setup_default_decision_rules()
        self._setup_complexity_assessors()
        
        print("🎯 Work distributor initialized")
        print("   📊 Ready for intelligent work routing")
    
    def add_work_item(self, work_type: WorkType, title: str, description: str,
                     source_file: str = None, test_file: str = None,
                     error_message: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a new work item for distribution.
        
        Args:
            work_type: Type of work
            title: Work item title
            description: Detailed description
            source_file: Source file involved
            test_file: Test file involved
            error_message: Error message if applicable
            metadata: Additional metadata
            
        Returns:
            Work item ID
        """
        item_id = self._generate_work_id()
        
        work_item = WorkItem(
            item_id=item_id,
            work_type=work_type,
            title=title,
            description=description,
            source_file=source_file,
            test_file=test_file,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        # Assess complexity
        work_item.complexity_level = self._assess_complexity(work_item)
        
        # Estimate effort
        work_item.estimated_effort_minutes = self._estimate_effort(work_item)
        
        # Store work item
        self._work_items[item_id] = work_item
        self._stats['total_items'] += 1
        
        print(f"📝 Added work item: {work_type.value} - {title}")
        return item_id
    
    def make_handoff_decision(self, item_id: str) -> Optional[HandoffDecision]:
        """
        Make handoff decision for a work item.
        
        Args:
            item_id: Work item ID
            
        Returns:
            HandoffDecision if successful
        """
        if item_id not in self._work_items:
            return None
        
        work_item = self._work_items[item_id]
        
        # Try each decision rule
        for rule in self._decision_rules:
            try:
                decision = rule(work_item)
                if decision:
                    # Store decision
                    self._handoff_decisions[item_id] = decision
                    work_item.assigned_to = decision.target
                    work_item.handoff_reason = decision.reason
                    
                    # Update statistics
                    if decision.target in [HandoffTarget.TESTMASTER_AUTO, HandoffTarget.TESTMASTER_MANUAL]:
                        self._stats['automated_assignments'] += 1
                    else:
                        self._stats['manual_assignments'] += 1
                    
                    print(f"🎯 Decision: {work_item.title} → {decision.target.value} ({decision.reason.value})")
                    return decision
                    
            except Exception as e:
                print(f"⚠️ Error in decision rule: {e}")
        
        # Default decision if no rules matched
        default_decision = self._make_default_decision(work_item)
        self._handoff_decisions[item_id] = default_decision
        work_item.assigned_to = default_decision.target
        work_item.handoff_reason = default_decision.reason
        
        return default_decision
    
    def batch_similar_items(self, max_batch_size: int = 10) -> List[BatchContext]:
        """
        Batch similar work items for efficient processing.
        
        Args:
            max_batch_size: Maximum items per batch
            
        Returns:
            List of batch contexts
        """
        print("📦 Batching similar work items...")
        
        # Group items by type and target
        grouped_items = {}
        
        for item in self._work_items.values():
            if item.assigned_to is None:
                continue  # Skip unassigned items
            
            # Create grouping key
            group_key = f"{item.work_type.value}_{item.assigned_to.value}_{item.complexity_level}"
            
            if group_key not in grouped_items:
                grouped_items[group_key] = []
            grouped_items[group_key].append(item)
        
        # Create batches
        batches = []
        batch_count = 0
        
        for group_key, items in grouped_items.items():
            if len(items) >= 2:  # Only batch if 2+ items
                # Split into batches of max_batch_size
                for i in range(0, len(items), max_batch_size):
                    batch_items = items[i:i + max_batch_size]
                    
                    batch_id = f"batch_{int(datetime.now().timestamp())}_{batch_count}"
                    batch_count += 1
                    
                    # Calculate total effort
                    total_effort = sum(item.estimated_effort_minutes for item in batch_items)
                    
                    # Determine batch properties
                    work_type = batch_items[0].work_type.value
                    target = batch_items[0].assigned_to
                    
                    batch_context = BatchContext(
                        batch_id=batch_id,
                        work_items=batch_items,
                        batch_type=f"{work_type}_batch",
                        total_estimated_effort=total_effort,
                        recommended_target=target,
                        batch_reason=f"Batched {len(batch_items)} similar {work_type} items"
                    )
                    
                    # Assign batch ID to items
                    for item in batch_items:
                        if item.item_id in self._handoff_decisions:
                            self._handoff_decisions[item.item_id].batch_id = batch_id
                    
                    self._batch_contexts[batch_id] = batch_context
                    batches.append(batch_context)
                    
                    self._stats['batched_items'] += len(batch_items)
        
        print(f"📦 Created {len(batches)} batches with {self._stats['batched_items']} items")
        return batches
    
    def get_work_for_target(self, target: HandoffTarget) -> List[WorkItem]:
        """Get work items assigned to a specific target."""
        return [
            item for item in self._work_items.values()
            if item.assigned_to == target
        ]
    
    def get_high_priority_work(self) -> List[WorkItem]:
        """Get high priority work items."""
        high_priority = []
        
        for item in self._work_items.values():
            # High priority criteria
            if (item.work_type in [WorkType.TEST_FAILURE, WorkType.BREAKING_CHANGE, WorkType.SECURITY] or
                item.complexity_level >= ComplexityLevel.COMPLEX or
                (item.due_date and datetime.now() > item.due_date) or
                "critical" in item.metadata.get("tags", [])):
                high_priority.append(item)
        
        # Sort by urgency
        high_priority.sort(key=lambda item: (
            -item.work_type.value.count("break"),  # Breaking changes first
            -item.complexity_level,
            item.created_at
        ))
        
        return high_priority
    
    def _setup_default_decision_rules(self):
        """Setup default decision rules using function-based handoff patterns."""
        
        # Rule 1: Automated test failures
        def test_failure_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.TEST_FAILURE:
                # Simple test failures can be auto-fixed
                if (item.complexity_level <= ComplexityLevel.SIMPLE and
                    item.error_message and
                    any(keyword in item.error_message.lower() 
                        for keyword in ["assertion", "import", "syntax"])):
                    
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.TESTMASTER_AUTO,
                        reason=HandoffReason.AUTOMATED_SOLUTION,
                        confidence=85.0,
                        suggested_approach="Automated test repair using self-healing patterns"
                    )
                
                # Complex test failures need manual review
                elif item.complexity_level >= ComplexityLevel.COMPLEX:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.CLAUDE_CODE,
                        reason=HandoffReason.COMPLEXITY_TOO_HIGH,
                        confidence=90.0,
                        suggested_approach="Manual analysis and debugging required"
                    )
            
            return None
        
        # Rule 2: Coverage gaps
        def coverage_gap_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.COVERAGE_GAP:
                # Can generate tests automatically for simple cases
                if item.complexity_level <= ComplexityLevel.MODERATE:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.TESTMASTER_AUTO,
                        reason=HandoffReason.AUTOMATED_SOLUTION,
                        confidence=75.0,
                        suggested_approach="Generate missing tests using AI patterns"
                    )
                else:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.CLAUDE_CODE,
                        reason=HandoffReason.REQUIRES_UNDERSTANDING,
                        confidence=80.0,
                        suggested_approach="Manual test design for complex logic"
                    )
            
            return None
        
        # Rule 3: Idle modules
        def idle_module_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.IDLE_MODULE:
                # Always route to Claude Code for investigation
                return HandoffDecision(
                    work_item=item,
                    target=HandoffTarget.CLAUDE_CODE,
                    reason=HandoffReason.HUMAN_JUDGMENT,
                    confidence=95.0,
                    suggested_approach="Investigate module relevance and update needs"
                )
            
            return None
        
        # Rule 4: Breaking changes
        def breaking_change_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.BREAKING_CHANGE:
                # Always high priority to Claude Code
                return HandoffDecision(
                    work_item=item,
                    target=HandoffTarget.CLAUDE_CODE,
                    reason=HandoffReason.COMPLEXITY_TOO_HIGH,
                    confidence=100.0,
                    suggested_approach="Immediate manual intervention required",
                    estimated_completion=datetime.now() + timedelta(hours=2)
                )
            
            return None
        
        # Rule 5: Refactoring
        def refactoring_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.REFACTORING:
                # Simple refactoring can be automated
                if item.complexity_level <= ComplexityLevel.SIMPLE:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.TESTMASTER_MANUAL,
                        reason=HandoffReason.AUTOMATED_SOLUTION,
                        confidence=70.0,
                        suggested_approach="Use automated refactoring tools"
                    )
                else:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.CLAUDE_CODE,
                        reason=HandoffReason.DOMAIN_EXPERTISE,
                        confidence=85.0,
                        suggested_approach="Manual refactoring with design considerations"
                    )
            
            return None
        
        # Rule 6: New features
        def new_feature_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.NEW_FEATURE:
                # Always to Claude Code for features
                return HandoffDecision(
                    work_item=item,
                    target=HandoffTarget.CLAUDE_CODE,
                    reason=HandoffReason.REQUIRES_UNDERSTANDING,
                    confidence=95.0,
                    suggested_approach="Design and implement new functionality"
                )
            
            return None
        
        # Rule 7: Documentation
        def documentation_rule(item: WorkItem) -> Optional[HandoffDecision]:
            if item.work_type == WorkType.DOCUMENTATION:
                # Can generate basic documentation automatically
                if item.complexity_level <= ComplexityLevel.MODERATE:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.TESTMASTER_AUTO,
                        reason=HandoffReason.AUTOMATED_SOLUTION,
                        confidence=60.0,
                        suggested_approach="Generate documentation from code analysis"
                    )
                else:
                    return HandoffDecision(
                        work_item=item,
                        target=HandoffTarget.CLAUDE_CODE,
                        reason=HandoffReason.DOMAIN_EXPERTISE,
                        confidence=80.0,
                        suggested_approach="Manual documentation with domain context"
                    )
            
            return None
        
        # Add all rules
        self._decision_rules.extend([
            test_failure_rule,
            coverage_gap_rule,
            idle_module_rule,
            breaking_change_rule,
            refactoring_rule,
            new_feature_rule,
            documentation_rule
        ])
    
    def _setup_complexity_assessors(self):
        """Setup complexity assessment functions for each work type."""
        
        def assess_test_failure_complexity(item: WorkItem) -> ComplexityLevel:
            if not item.error_message:
                return ComplexityLevel.MODERATE
            
            error_lower = item.error_message.lower()
            
            # Simple errors
            if any(keyword in error_lower for keyword in [
                "assertion", "importerror", "syntaxerror", "namenotdefined"
            ]):
                return ComplexityLevel.SIMPLE
            
            # Complex errors
            elif any(keyword in error_lower for keyword in [
                "segmentation", "memory", "timeout", "deadlock", "race"
            ]):
                return ComplexityLevel.VERY_COMPLEX
            
            # Moderate errors
            elif any(keyword in error_lower for keyword in [
                "attributeerror", "typeerror", "valueerror"
            ]):
                return ComplexityLevel.MODERATE
            
            return ComplexityLevel.MODERATE
        
        def assess_coverage_gap_complexity(item: WorkItem) -> ComplexityLevel:
            # Check file size and structure
            if item.source_file:
                try:
                    file_path = Path(item.source_file)
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            lines = len(f.readlines())
                        
                        if lines > 500:
                            return ComplexityLevel.COMPLEX
                        elif lines > 200:
                            return ComplexityLevel.MODERATE
                        else:
                            return ComplexityLevel.SIMPLE
                except:
                    pass
            
            return ComplexityLevel.MODERATE
        
        def assess_idle_module_complexity(item: WorkItem) -> ComplexityLevel:
            # Idle modules always need human judgment
            return ComplexityLevel.MODERATE
        
        def assess_breaking_change_complexity(item: WorkItem) -> ComplexityLevel:
            # Breaking changes are always complex
            return ComplexityLevel.VERY_COMPLEX
        
        def assess_refactoring_complexity(item: WorkItem) -> ComplexityLevel:
            # Determine by scope
            scope = item.metadata.get("scope", "unknown")
            
            if scope in ["function", "method"]:
                return ComplexityLevel.SIMPLE
            elif scope in ["class", "module"]:
                return ComplexityLevel.MODERATE
            elif scope in ["package", "architecture"]:
                return ComplexityLevel.VERY_COMPLEX
            
            return ComplexityLevel.MODERATE
        
        def assess_default_complexity(item: WorkItem) -> ComplexityLevel:
            # Default assessment based on metadata
            if "simple" in item.metadata.get("tags", []):
                return ComplexityLevel.SIMPLE
            elif "complex" in item.metadata.get("tags", []):
                return ComplexityLevel.COMPLEX
            
            return ComplexityLevel.MODERATE
        
        # Register assessors
        self._complexity_assessors = {
            WorkType.TEST_FAILURE: assess_test_failure_complexity,
            WorkType.COVERAGE_GAP: assess_coverage_gap_complexity,
            WorkType.IDLE_MODULE: assess_idle_module_complexity,
            WorkType.BREAKING_CHANGE: assess_breaking_change_complexity,
            WorkType.REFACTORING: assess_refactoring_complexity,
            WorkType.NEW_FEATURE: assess_default_complexity,
            WorkType.BUG_FIX: assess_default_complexity,
            WorkType.DOCUMENTATION: assess_default_complexity,
            WorkType.PERFORMANCE: assess_default_complexity,
            WorkType.SECURITY: lambda x: ComplexityLevel.VERY_COMPLEX  # Always complex
        }
    
    def _assess_complexity(self, work_item: WorkItem) -> ComplexityLevel:
        """Assess complexity of a work item."""
        assessor = self._complexity_assessors.get(work_item.work_type)
        
        if assessor:
            try:
                return assessor(work_item)
            except Exception as e:
                print(f"⚠️ Error assessing complexity: {e}")
        
        return ComplexityLevel.MODERATE
    
    def _estimate_effort(self, work_item: WorkItem) -> int:
        """Estimate effort in minutes for a work item."""
        # Base effort by complexity
        base_minutes = {
            ComplexityLevel.TRIVIAL: 5,
            ComplexityLevel.SIMPLE: 15,
            ComplexityLevel.MODERATE: 30,
            ComplexityLevel.COMPLEX: 60,
            ComplexityLevel.VERY_COMPLEX: 120
        }
        
        effort = base_minutes.get(work_item.complexity_level, 30)
        
        # Adjust by work type
        type_multipliers = {
            WorkType.TEST_FAILURE: 0.8,
            WorkType.COVERAGE_GAP: 1.0,
            WorkType.IDLE_MODULE: 1.5,
            WorkType.BREAKING_CHANGE: 2.0,
            WorkType.REFACTORING: 1.5,
            WorkType.NEW_FEATURE: 3.0,
            WorkType.BUG_FIX: 1.2,
            WorkType.DOCUMENTATION: 0.7,
            WorkType.PERFORMANCE: 2.5,
            WorkType.SECURITY: 3.0
        }
        
        multiplier = type_multipliers.get(work_item.work_type, 1.0)
        
        return int(effort * multiplier)
    
    def _make_default_decision(self, work_item: WorkItem) -> HandoffDecision:
        """Make default decision when no rules match."""
        # Default to Claude Code for manual review
        return HandoffDecision(
            work_item=work_item,
            target=HandoffTarget.CLAUDE_CODE,
            reason=HandoffReason.HUMAN_JUDGMENT,
            confidence=50.0,
            suggested_approach="Manual review and assessment needed"
        )
    
    def _generate_work_id(self) -> str:
        """Generate unique work item ID."""
        import time
        return f"work_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def get_distribution_statistics(self) -> Dict[str, Any]:
        """Get work distribution statistics."""
        total_items = len(self._work_items)
        
        # Count by target
        target_counts = {}
        for target in HandoffTarget:
            target_counts[target.value] = len(self.get_work_for_target(target))
        
        # Count by work type
        type_counts = {}
        for work_type in WorkType:
            type_counts[work_type.value] = len([
                item for item in self._work_items.values()
                if item.work_type == work_type
            ])
        
        # Count by complexity
        complexity_counts = {}
        for complexity in ComplexityLevel:
            complexity_counts[complexity.name] = len([
                item for item in self._work_items.values()
                if item.complexity_level == complexity
            ])
        
        # Calculate average effort
        total_effort = sum(item.estimated_effort_minutes for item in self._work_items.values())
        avg_effort = total_effort / max(total_items, 1)
        
        return {
            "total_work_items": total_items,
            "target_distribution": target_counts,
            "work_type_distribution": type_counts,
            "complexity_distribution": complexity_counts,
            "batches_created": len(self._batch_contexts),
            "average_effort_minutes": avg_effort,
            "total_estimated_effort_hours": total_effort / 60,
            "high_priority_items": len(self.get_high_priority_work()),
            "statistics": dict(self._stats)
        }
    
    def export_distribution_report(self, output_path: str = "work_distribution_report.json"):
        """Export work distribution report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_distribution_statistics(),
            "work_items": [],
            "handoff_decisions": [],
            "batch_contexts": []
        }
        
        # Add work items
        for item in self._work_items.values():
            item_data = {
                "item_id": item.item_id,
                "work_type": item.work_type.value,
                "title": item.title,
                "description": item.description,
                "complexity_level": item.complexity_level.name,
                "estimated_effort_minutes": item.estimated_effort_minutes,
                "assigned_to": item.assigned_to.value if item.assigned_to else None,
                "handoff_reason": item.handoff_reason.value if item.handoff_reason else None,
                "created_at": item.created_at.isoformat(),
                "metadata": item.metadata
            }
            report["work_items"].append(item_data)
        
        # Add decisions
        for decision in self._handoff_decisions.values():
            decision_data = {
                "work_item_id": decision.work_item.item_id,
                "target": decision.target.value,
                "reason": decision.reason.value,
                "confidence": decision.confidence,
                "suggested_approach": decision.suggested_approach,
                "batch_id": decision.batch_id,
                "decision_time": decision.decision_time.isoformat()
            }
            report["handoff_decisions"].append(decision_data)
        
        # Add batch contexts
        for batch in self._batch_contexts.values():
            batch_data = {
                "batch_id": batch.batch_id,
                "batch_type": batch.batch_type,
                "item_count": len(batch.work_items),
                "total_estimated_effort": batch.total_estimated_effort,
                "recommended_target": batch.recommended_target.value,
                "batch_reason": batch.batch_reason,
                "created_at": batch.created_at.isoformat()
            }
            report["batch_contexts"].append(batch_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Work distribution report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting work distribution report: {e}")


# Convenience functions for work distribution
def create_test_failure_work(test_path: str, error_message: str, distributor: WorkDistributor) -> str:
    """Create a test failure work item."""
    return distributor.add_work_item(
        work_type=WorkType.TEST_FAILURE,
        title=f"Test Failure: {Path(test_path).name}",
        description=f"Test failed with error: {error_message}",
        test_file=test_path,
        error_message=error_message
    )


def create_coverage_gap_work(source_file: str, coverage_percentage: float, 
                           distributor: WorkDistributor) -> str:
    """Create a coverage gap work item."""
    return distributor.add_work_item(
        work_type=WorkType.COVERAGE_GAP,
        title=f"Low Coverage: {Path(source_file).name}",
        description=f"Coverage is {coverage_percentage:.1f}%, needs improvement",
        source_file=source_file,
        metadata={"coverage": coverage_percentage}
    )


def create_idle_module_work(module_path: str, idle_hours: float,
                          distributor: WorkDistributor) -> str:
    """Create an idle module work item."""
    return distributor.add_work_item(
        work_type=WorkType.IDLE_MODULE,
        title=f"Idle Module: {Path(module_path).name}",
        description=f"Module has been idle for {idle_hours:.1f} hours",
        source_file=module_path,
        metadata={"idle_hours": idle_hours}
    )