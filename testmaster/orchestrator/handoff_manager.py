"""
Smart Handoff System

Inspired by OpenAI Swarm's context preservation patterns
for intelligent handoffs between TestMaster and Claude Code.

Features:
- Package context for Claude Code handoffs
- Track handoff status and responses
- Learn from resolution patterns
- Context preservation and enrichment
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import yaml

from ..core.layer_manager import requires_layer
from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..core.monitoring_decorators import monitor_performance


class HandoffType(Enum):
    """Types of handoffs."""
    INVESTIGATION_REQUEST = "investigation_request"
    ISSUE_ESCALATION = "issue_escalation"
    CONTEXT_SHARING = "context_sharing"
    WORK_DELEGATION = "work_delegation"
    STATUS_UPDATE = "status_update"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"


class HandoffStatus(Enum):
    """Status of handoffs."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContextType(Enum):
    """Types of context information."""
    FILE_ANALYSIS = "file_analysis"
    ERROR_DETAILS = "error_details"
    INVESTIGATION_RESULTS = "investigation_results"
    HISTORICAL_DATA = "historical_data"
    DEPENDENCY_INFO = "dependency_info"
    TEST_COVERAGE = "test_coverage"
    PERFORMANCE_DATA = "performance_data"
    RECOMMENDATIONS = "recommendations"


@dataclass
class ContextItem:
    """A piece of context information."""
    context_type: ContextType
    title: str
    content: Any
    confidence: float  # 0-100
    relevance: float  # 0-100
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffContext:
    """Complete context package for handoff."""
    handoff_id: str
    handoff_type: HandoffType
    title: str
    description: str
    
    # Target information
    primary_target: str  # Main file/module/issue
    related_targets: List[str] = field(default_factory=list)
    
    # Context items
    context_items: List[ContextItem] = field(default_factory=list)
    
    # Request details
    requested_action: str = ""
    priority_level: str = "normal"  # low, normal, high, critical
    urgency: str = "normal"  # low, normal, high, emergency
    
    # Constraints and preferences
    constraints: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    
    # Background information
    background: Optional[str] = None
    previous_attempts: List[str] = field(default_factory=list)
    related_issues: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Status tracking
    status: HandoffStatus = HandoffStatus.PENDING
    assigned_to: Optional[str] = None
    
    # Response tracking
    acknowledgment_received: bool = False
    response_received: bool = False
    completion_confirmed: bool = False


@dataclass
class HandoffResponse:
    """Response to a handoff."""
    response_id: str
    handoff_id: str
    response_type: str  # acknowledgment, progress_update, completion, question
    
    # Response content
    message: str
    status_update: Optional[str] = None
    progress_percent: Optional[float] = None
    
    # Results (if completion)
    results: Dict[str, Any] = field(default_factory=dict)
    files_modified: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    
    # Follow-up
    follow_up_needed: bool = False
    follow_up_requests: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
    # Response metadata
    responder: str = "Claude Code"
    confidence: float = 0.0
    satisfaction_rating: Optional[float] = None


@dataclass
class HandoffPattern:
    """Learned pattern from handoff history."""
    pattern_id: str
    pattern_type: str
    description: str
    
    # Pattern conditions
    triggers: List[str] = field(default_factory=list)
    context_indicators: List[str] = field(default_factory=list)
    
    # Pattern outcomes
    typical_resolution_time: Optional[float] = None  # hours
    success_rate: float = 0.0
    common_actions: List[str] = field(default_factory=list)
    
    # Pattern metadata
    examples_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


class HandoffManager:
    """
    Smart handoff system using OpenAI Swarm context preservation.
    
    Manages intelligent handoffs between TestMaster and Claude Code
    with rich context packaging and pattern learning.
    
    Enhanced with toggleable advanced handoff tools:
    - Validation tools for handoff integrity
    - Context compression for efficiency  
    - Priority queue management
    - Automatic retry with exponential backoff
    - Handoff batching for related items
    """
    
    def __init__(self, handoff_dir: str = ".testmaster_handoffs"):
        """
        Initialize handoff manager.
        
        Args:
            handoff_dir: Directory for handoff persistence
        """
        self.handoff_dir = Path(handoff_dir)
        self.handoff_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.handoff_dir / "pending").mkdir(exist_ok=True)
        (self.handoff_dir / "active").mkdir(exist_ok=True)
        (self.handoff_dir / "completed").mkdir(exist_ok=True)
        (self.handoff_dir / "responses").mkdir(exist_ok=True)
        
        # Handoff storage
        self._handoffs: Dict[str, HandoffContext] = {}
        self._responses: Dict[str, HandoffResponse] = {}
        self._patterns: Dict[str, HandoffPattern] = {}
        
        # Context enrichment functions
        self._context_enrichers: Dict[ContextType, Callable] = {}
        
        # Statistics
        self._stats = {
            'total_handoffs': 0,
            'successful_handoffs': 0,
            'failed_handoffs': 0,
            'avg_resolution_time': 0.0,
            'patterns_learned': 0
        }
        
        # NEW: Advanced handoff tools (toggleable)
        if FeatureFlags.is_enabled('layer2_monitoring', 'handoff_tools'):
            self._setup_advanced_tools()
            print("   Advanced handoff tools enabled")
        else:
            self._validation_enabled = False
            self._compression_enabled = False
            self._priority_queue = None
            self._retry_manager = None
            self._batch_processor = None
        
        # Setup context enrichers
        self._setup_context_enrichers()
        
        print("Handoff manager initialized")
        print(f"   Handoff directory: {self.handoff_dir}")
        
        # Initialize shared state if enabled
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
    
    @monitor_performance(name="handoff_creation")
    def create_handoff(self, handoff_type: HandoffType, title: str, description: str,
                      primary_target: str, requested_action: str,
                      priority_level: str = "normal",
                      urgency: str = "normal",
                      context_data: Dict[str, Any] = None,
                      constraints: List[str] = None,
                      background: str = None) -> str:
        """
        Create a new handoff with rich context.
        
        Args:
            handoff_type: Type of handoff
            title: Handoff title
            description: Detailed description
            primary_target: Main target (file, module, etc.)
            requested_action: What action is requested
            priority_level: Priority level
            urgency: Urgency level
            context_data: Additional context data
            constraints: Any constraints
            background: Background information
            
        Returns:
            Handoff ID
        """
        handoff_id = self._generate_handoff_id()
        
        # Create handoff context
        handoff_context = HandoffContext(
            handoff_id=handoff_id,
            handoff_type=handoff_type,
            title=title,
            description=description,
            primary_target=primary_target,
            requested_action=requested_action,
            priority_level=priority_level,
            urgency=urgency,
            constraints=constraints or [],
            background=background
        )
        
        # Set expiration based on urgency
        if urgency == "emergency":
            handoff_context.expires_at = datetime.now() + timedelta(hours=2)
        elif urgency == "high":
            handoff_context.expires_at = datetime.now() + timedelta(hours=8)
        else:
            handoff_context.expires_at = datetime.now() + timedelta(days=1)
        
        # Enrich with context
        self._enrich_handoff_context(handoff_context, context_data or {})
        
        # NEW: Validate handoff if tools enabled
        if self._validation_enabled:
            validation_result = self._validate_handoff(handoff_context)
            if not validation_result['is_valid']:
                print(f"Handoff validation warnings: {validation_result['warnings']}")
                # Add validation warnings to context
                if validation_result['warnings']:
                    warning_context = ContextItem(
                        context_type=ContextType.RECOMMENDATIONS,
                        title="Validation Warnings",
                        content={"warnings": validation_result['warnings']},
                        confidence=100.0,
                        relevance=95.0,
                        source="HandoffValidator"
                    )
                    handoff_context.context_items.append(warning_context)
        
        # NEW: Compress context if enabled
        if self._compression_enabled:
            handoff_context = self._compress_context(handoff_context)
        
        # NEW: Add to priority queue if enabled
        if self._priority_queue:
            self._priority_queue.add(handoff_context)
        
        # Store handoff
        self._handoffs[handoff_id] = handoff_context
        self._stats['total_handoffs'] += 1
        
        # NEW: Update shared state if enabled
        if self.shared_state:
            self.shared_state.increment("handoffs_created")
            self.shared_state.set(f"handoff_{handoff_id}_status", "created")
            self.shared_state.append("recent_handoffs", {
                "id": handoff_id,
                "type": handoff_type.value,
                "title": title,
                "timestamp": datetime.now().isoformat()
            })
        
        # Persist handoff
        self._persist_handoff(handoff_context)
        
        print(f"Created handoff: {title} (ID: {handoff_id})")
        return handoff_id
    
    def create_investigation_handoff(self, target: str, investigation_results: Dict[str, Any],
                                   priority: str = "normal") -> str:
        """
        Create handoff for investigation results.
        
        Args:
            target: Investigation target
            investigation_results: Results from automated investigation
            priority: Handoff priority
            
        Returns:
            Handoff ID
        """
        findings_count = investigation_results.get('total_findings', 0)
        critical_count = investigation_results.get('critical_findings', 0)
        
        if critical_count > 0:
            urgency = "high"
            title = f"Critical Issues Found in {Path(target).name}"
        elif findings_count > 0:
            urgency = "normal"
            title = f"Investigation Results for {Path(target).name}"
        else:
            urgency = "low"
            title = f"Investigation Complete: {Path(target).name}"
        
        description = f"""
Automated investigation of {target} has been completed.

Investigation Summary:
- Total findings: {findings_count}
- Critical findings: {critical_count}
- High priority findings: {investigation_results.get('high_findings', 0)}
- Evidence collected: {investigation_results.get('evidence_count', 0)}

Please review the findings and take appropriate action.
        """.strip()
        
        return self.create_handoff(
            handoff_type=HandoffType.INVESTIGATION_REQUEST,
            title=title,
            description=description,
            primary_target=target,
            requested_action="Review investigation findings and implement fixes",
            priority_level=priority,
            urgency=urgency,
            context_data={"investigation_results": investigation_results},
            background="Automated investigation detected issues requiring human review"
        )
    
    def create_work_delegation_handoff(self, work_item: Dict[str, Any]) -> str:
        """
        Create handoff for work delegation.
        
        Args:
            work_item: Work item to delegate
            
        Returns:
            Handoff ID
        """
        item_title = work_item.get('title', 'Work Item')
        complexity = work_item.get('complexity_level', 'moderate')
        
        title = f"Work Delegation: {item_title}"
        description = f"""
A work item has been analyzed and determined to require manual intervention.

Work Item Details:
- Type: {work_item.get('work_type', 'unknown')}
- Complexity: {complexity}
- Estimated Effort: {work_item.get('estimated_effort_minutes', 'unknown')} minutes
- Target: {work_item.get('source_file', work_item.get('test_file', 'unknown'))}

The automated analysis determined this requires human expertise due to complexity or domain knowledge requirements.
        """.strip()
        
        urgency = "high" if work_item.get('work_type') == "breaking_change" else "normal"
        
        return self.create_handoff(
            handoff_type=HandoffType.WORK_DELEGATION,
            title=title,
            description=description,
            primary_target=work_item.get('source_file', work_item.get('test_file', '')),
            requested_action=f"Handle {work_item.get('work_type', 'work item')} as delegated",
            priority_level=work_item.get('priority', 'normal'),
            urgency=urgency,
            context_data={"work_item": work_item},
            constraints=work_item.get('constraints', [])
        )
    
    def create_escalation_handoff(self, issue_description: str, failed_attempts: List[str],
                                target: str, error_details: str = None) -> str:
        """
        Create handoff for issue escalation.
        
        Args:
            issue_description: Description of the issue
            failed_attempts: List of failed automated attempts
            target: Target file/module
            error_details: Error details if available
            
        Returns:
            Handoff ID
        """
        title = f"Escalation: {issue_description}"
        description = f"""
An issue has been escalated after automated resolution attempts failed.

Issue: {issue_description}
Target: {target}

Failed Attempts:
{chr(10).join(f'- {attempt}' for attempt in failed_attempts)}

Manual intervention is required to resolve this issue.
        """.strip()
        
        context_data = {"failed_attempts": failed_attempts}
        if error_details:
            context_data["error_details"] = error_details
        
        return self.create_handoff(
            handoff_type=HandoffType.ISSUE_ESCALATION,
            title=title,
            description=description,
            primary_target=target,
            requested_action="Resolve escalated issue that automated systems could not handle",
            priority_level="high",
            urgency="high",
            context_data=context_data,
            background="Multiple automated resolution attempts have failed"
        )
    
    def receive_response(self, handoff_id: str, response_type: str, message: str,
                        status_update: str = None, progress_percent: float = None,
                        results: Dict[str, Any] = None,
                        files_modified: List[str] = None,
                        actions_taken: List[str] = None) -> str:
        """
        Receive and process a response to a handoff.
        
        Args:
            handoff_id: Handoff ID
            response_type: Type of response
            message: Response message
            status_update: Status update
            progress_percent: Progress percentage
            results: Results if completion
            files_modified: Files that were modified
            actions_taken: Actions that were taken
            
        Returns:
            Response ID
        """
        if handoff_id not in self._handoffs:
            raise ValueError(f"Handoff {handoff_id} not found")
        
        response_id = self._generate_response_id()
        handoff = self._handoffs[handoff_id]
        
        response = HandoffResponse(
            response_id=response_id,
            handoff_id=handoff_id,
            response_type=response_type,
            message=message,
            status_update=status_update,
            progress_percent=progress_percent,
            results=results or {},
            files_modified=files_modified or [],
            actions_taken=actions_taken or []
        )
        
        # Update handoff status
        if response_type == "acknowledgment":
            handoff.acknowledgment_received = True
            handoff.status = HandoffStatus.ACKNOWLEDGED
        elif response_type == "progress_update":
            handoff.status = HandoffStatus.IN_PROGRESS
        elif response_type == "completion":
            handoff.status = HandoffStatus.COMPLETED
            handoff.completion_confirmed = True
            self._stats['successful_handoffs'] += 1
            
            # Learn from this handoff
            self._learn_from_handoff(handoff, response)
        
        # Store response
        self._responses[response_id] = response
        
        # Persist response
        self._persist_response(response)
        
        # Update handoff file
        self._persist_handoff(handoff)
        
        print(f"Received response for handoff {handoff_id}: {response_type}")
        return response_id
    
    def get_active_handoffs(self) -> List[HandoffContext]:
        """Get all active handoffs."""
        return [
            handoff for handoff in self._handoffs.values()
            if handoff.status in [HandoffStatus.PENDING, HandoffStatus.SENT, 
                                HandoffStatus.ACKNOWLEDGED, HandoffStatus.IN_PROGRESS]
        ]
    
    def get_handoff_by_id(self, handoff_id: str) -> Optional[HandoffContext]:
        """Get handoff by ID."""
        return self._handoffs.get(handoff_id)
    
    def get_overdue_handoffs(self) -> List[HandoffContext]:
        """Get handoffs that are overdue."""
        current_time = datetime.now()
        return [
            handoff for handoff in self._handoffs.values()
            if (handoff.expires_at and current_time > handoff.expires_at and
                handoff.status not in [HandoffStatus.COMPLETED, HandoffStatus.CANCELLED])
        ]
    
    def cancel_handoff(self, handoff_id: str, reason: str = None) -> bool:
        """
        Cancel a handoff.
        
        Args:
            handoff_id: Handoff ID
            reason: Cancellation reason
            
        Returns:
            True if cancelled successfully
        """
        if handoff_id not in self._handoffs:
            return False
        
        handoff = self._handoffs[handoff_id]
        handoff.status = HandoffStatus.CANCELLED
        
        # Add cancellation note
        if reason:
            cancellation_context = ContextItem(
                context_type=ContextType.RECOMMENDATIONS,
                title="Cancellation Reason",
                content=reason,
                confidence=100.0,
                relevance=100.0,
                source="HandoffManager"
            )
            handoff.context_items.append(cancellation_context)
        
        self._persist_handoff(handoff)
        print(f"Cancelled handoff: {handoff_id}")
        return True
    
    def _enrich_handoff_context(self, handoff: HandoffContext, context_data: Dict[str, Any]):
        """Enrich handoff with additional context."""
        # Add file analysis if target is a file
        if Path(handoff.primary_target).exists():
            file_context = self._enrich_file_analysis(handoff.primary_target)
            if file_context:
                handoff.context_items.append(file_context)
        
        # Add investigation results if provided
        if "investigation_results" in context_data:
            inv_context = ContextItem(
                context_type=ContextType.INVESTIGATION_RESULTS,
                title="Investigation Results",
                content=context_data["investigation_results"],
                confidence=90.0,
                relevance=100.0,
                source="AutoInvestigator"
            )
            handoff.context_items.append(inv_context)
        
        # Add work item details if provided
        if "work_item" in context_data:
            work_context = ContextItem(
                context_type=ContextType.RECOMMENDATIONS,
                title="Work Item Analysis",
                content=context_data["work_item"],
                confidence=85.0,
                relevance=90.0,
                source="WorkDistributor"
            )
            handoff.context_items.append(work_context)
        
        # Add error details if provided
        if "error_details" in context_data:
            error_context = ContextItem(
                context_type=ContextType.ERROR_DETAILS,
                title="Error Information",
                content=context_data["error_details"],
                confidence=95.0,
                relevance=100.0,
                source="ErrorAnalyzer"
            )
            handoff.context_items.append(error_context)
        
        # Add failed attempts if provided
        if "failed_attempts" in context_data:
            attempts_context = ContextItem(
                context_type=ContextType.HISTORICAL_DATA,
                title="Previous Attempts",
                content={"failed_attempts": context_data["failed_attempts"]},
                confidence=100.0,
                relevance=95.0,
                source="AutomatedSystems"
            )
            handoff.context_items.append(attempts_context)
        
        # Add recommendations from patterns
        pattern_recommendations = self._get_pattern_recommendations(handoff)
        if pattern_recommendations:
            pattern_context = ContextItem(
                context_type=ContextType.RECOMMENDATIONS,
                title="Pattern-Based Recommendations",
                content=pattern_recommendations,
                confidence=70.0,
                relevance=80.0,
                source="PatternLearning"
            )
            handoff.context_items.append(pattern_context)
    
    def _enrich_file_analysis(self, file_path: str) -> Optional[ContextItem]:
        """Enrich with file analysis context."""
        try:
            enricher = self._context_enrichers.get(ContextType.FILE_ANALYSIS)
            if enricher:
                return enricher(file_path)
        except Exception as e:
            print(f"Error enriching file analysis: {e}")
        return None
    
    def _setup_context_enrichers(self):
        """Setup context enrichment functions."""
        
        def enrich_file_analysis(file_path: str) -> ContextItem:
            """Enrich with file analysis."""
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic analysis
                lines = content.split('\\n')
                code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                
                # Import analysis
                imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
                
                analysis = {
                    'file_size': file_path.stat().st_size,
                    'total_lines': len(lines),
                    'code_lines': len(code_lines),
                    'import_count': len(imports),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'file_type': file_path.suffix,
                    'relative_path': str(file_path.relative_to(Path.cwd())) if file_path.is_relative_to(Path.cwd()) else str(file_path)
                }
                
                return ContextItem(
                    context_type=ContextType.FILE_ANALYSIS,
                    title=f"File Analysis: {file_path.name}",
                    content=analysis,
                    confidence=95.0,
                    relevance=90.0,
                    source="FileAnalyzer"
                )
                
            except Exception as e:
                return ContextItem(
                    context_type=ContextType.FILE_ANALYSIS,
                    title=f"File Analysis Error: {file_path.name}",
                    content={"error": str(e)},
                    confidence=50.0,
                    relevance=60.0,
                    source="FileAnalyzer"
                )
        
        self._context_enrichers[ContextType.FILE_ANALYSIS] = enrich_file_analysis
    
    def _get_pattern_recommendations(self, handoff: HandoffContext) -> Optional[Dict[str, Any]]:
        """Get recommendations based on learned patterns."""
        recommendations = []
        
        # Find matching patterns
        for pattern in self._patterns.values():
            # Simple pattern matching
            if (handoff.handoff_type.value in pattern.triggers or
                any(trigger in handoff.description.lower() for trigger in pattern.triggers)):
                
                if pattern.success_rate > 0.7:  # Only high-success patterns
                    recommendations.extend(pattern.common_actions)
        
        if recommendations:
            return {
                "recommended_actions": list(set(recommendations)),
                "based_on_patterns": len(self._patterns),
                "confidence": "medium"
            }
        
        return None
    
    def _learn_from_handoff(self, handoff: HandoffContext, response: HandoffResponse):
        """Learn patterns from completed handoff."""
        # Extract pattern triggers
        triggers = [handoff.handoff_type.value]
        
        # Add keywords from description
        description_words = handoff.description.lower().split()
        key_terms = ["error", "failure", "issue", "bug", "coverage", "test", "idle"]
        triggers.extend([word for word in description_words if word in key_terms])
        
        # Create or update pattern
        pattern_key = f"{handoff.handoff_type.value}_{hash('_'.join(sorted(triggers))) % 1000}"
        
        if pattern_key in self._patterns:
            pattern = self._patterns[pattern_key]
            pattern.examples_count += 1
            
            # Update success rate
            if handoff.status == HandoffStatus.COMPLETED:
                pattern.success_rate = (pattern.success_rate * (pattern.examples_count - 1) + 1) / pattern.examples_count
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.examples_count - 1)) / pattern.examples_count
            
            # Update common actions
            if response.actions_taken:
                for action in response.actions_taken:
                    if action not in pattern.common_actions:
                        pattern.common_actions.append(action)
            
            pattern.last_updated = datetime.now()
            
        else:
            # Create new pattern
            pattern = HandoffPattern(
                pattern_id=pattern_key,
                pattern_type=handoff.handoff_type.value,
                description=f"Pattern for {handoff.handoff_type.value} handoffs",
                triggers=triggers,
                success_rate=1.0 if handoff.status == HandoffStatus.COMPLETED else 0.0,
                common_actions=response.actions_taken or [],
                examples_count=1,
                confidence=50.0  # Start with medium confidence
            )
            
            self._patterns[pattern_key] = pattern
            self._stats['patterns_learned'] += 1
        
        # Update confidence based on examples
        pattern.confidence = min(95.0, 50.0 + (pattern.examples_count * 5))
    
    def _persist_handoff(self, handoff: HandoffContext):
        """Persist handoff to file system."""
        try:
            # Convert to dictionary
            handoff_dict = asdict(handoff)
            
            # Convert datetime objects
            handoff_dict['created_at'] = handoff.created_at.isoformat()
            if handoff.expires_at:
                handoff_dict['expires_at'] = handoff.expires_at.isoformat()
            
            # Convert enums
            handoff_dict['handoff_type'] = handoff.handoff_type.value
            handoff_dict['status'] = handoff.status.value
            
            # Convert context items
            for i, item in enumerate(handoff_dict['context_items']):
                item['context_type'] = handoff.context_items[i].context_type.value
                item['timestamp'] = handoff.context_items[i].timestamp.isoformat()
            
            # Determine target directory
            if handoff.status in [HandoffStatus.PENDING, HandoffStatus.SENT]:
                target_dir = self.handoff_dir / "pending"
            elif handoff.status in [HandoffStatus.ACKNOWLEDGED, HandoffStatus.IN_PROGRESS]:
                target_dir = self.handoff_dir / "active"
            else:
                target_dir = self.handoff_dir / "completed"
            
            # Write to file
            file_path = target_dir / f"handoff_{handoff.handoff_id}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(handoff_dict, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e:
            print(f"Error persisting handoff: {e}")
    
    def _persist_response(self, response: HandoffResponse):
        """Persist response to file system."""
        try:
            # Convert to dictionary
            response_dict = asdict(response)
            response_dict['created_at'] = response.created_at.isoformat()
            
            # Write to file
            file_path = self.handoff_dir / "responses" / f"response_{response.response_id}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(response_dict, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e:
            print(f"Error persisting response: {e}")
    
    def _setup_advanced_tools(self):
        """Setup advanced handoff tools when enabled."""
        config = FeatureFlags.get_config('layer2_monitoring', 'handoff_tools')
        
        # Setup validation
        self._validation_enabled = config.get('validation', True)
        self._validation_rules = {
            'title_max_length': 200,
            'description_max_length': 5000,
            'context_items_max': 20,
            'required_fields': ['title', 'description', 'primary_target', 'requested_action']
        }
        
        # Setup compression
        self._compression_enabled = config.get('compression', True)
        self._compression_threshold = config.get('compression_threshold', 10000)  # bytes
        
        # Setup priority queue
        self._priority_queue = PriorityQueue() if config.get('priority_queue', True) else None
        
        # Setup retry manager
        self._retry_manager = RetryManager(
            max_retries=config.get('max_retries', 3),
            backoff_base=config.get('backoff_base', 2)
        ) if config.get('retry', True) else None
        
        # Setup batch processor
        self._batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 5),
            batch_timeout=config.get('batch_timeout', 60)
        ) if config.get('batching', True) else None
    
    def _validate_handoff(self, handoff: HandoffContext) -> Dict[str, Any]:
        """Validate handoff context for completeness and correctness."""
        warnings = []
        errors = []
        
        # Check required fields
        for field in self._validation_rules['required_fields']:
            if not getattr(handoff, field, None):
                errors.append(f"Missing required field: {field}")
        
        # Check length constraints
        if len(handoff.title) > self._validation_rules['title_max_length']:
            warnings.append(f"Title exceeds {self._validation_rules['title_max_length']} characters")
        
        if len(handoff.description) > self._validation_rules['description_max_length']:
            warnings.append(f"Description exceeds {self._validation_rules['description_max_length']} characters")
        
        # Check context items
        if len(handoff.context_items) > self._validation_rules['context_items_max']:
            warnings.append(f"Too many context items ({len(handoff.context_items)} > {self._validation_rules['context_items_max']})")
        
        # Check target validity
        if handoff.primary_target:
            target_path = Path(handoff.primary_target)
            if not target_path.exists() and not target_path.is_absolute():
                warnings.append(f"Target may not exist: {handoff.primary_target}")
        
        # Check priority/urgency consistency
        if handoff.urgency == "emergency" and handoff.priority_level == "low":
            warnings.append("Inconsistent: emergency urgency with low priority")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'score': max(0, 100 - len(errors) * 50 - len(warnings) * 10)
        }
    
    def _compress_context(self, handoff: HandoffContext) -> HandoffContext:
        """Compress large context items for efficiency."""
        import gzip
        import base64
        
        for item in handoff.context_items:
            # Check if content is large
            content_str = str(item.content)
            if len(content_str) > self._compression_threshold:
                # Compress the content
                compressed = gzip.compress(content_str.encode('utf-8'))
                encoded = base64.b64encode(compressed).decode('utf-8')
                
                # Replace with compressed version
                item.content = {
                    '_compressed': True,
                    'data': encoded,
                    'original_size': len(content_str),
                    'compressed_size': len(encoded)
                }
                item.metadata['compression'] = 'gzip+base64'
                
                print(f"   Compressed context item: {len(content_str)} -> {len(encoded)} bytes")
        
        return handoff
    
    def _decompress_context(self, item_content: Dict[str, Any]) -> str:
        """Decompress a compressed context item."""
        if isinstance(item_content, dict) and item_content.get('_compressed'):
            import gzip
            import base64
            
            encoded = item_content['data']
            compressed = base64.b64decode(encoded)
            decompressed = gzip.decompress(compressed).decode('utf-8')
            return decompressed
        
        return str(item_content)
    
    def _generate_handoff_id(self) -> str:
        """Generate unique handoff ID."""
        import time
        return f"handoff_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def _generate_response_id(self) -> str:
        """Generate unique response ID."""
        import time
        return f"response_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def get_handoff_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        total_handoffs = len(self._handoffs)
        active_handoffs = len(self.get_active_handoffs())
        overdue_handoffs = len(self.get_overdue_handoffs())
        
        # Count by type
        type_counts = {}
        for handoff_type in HandoffType:
            type_counts[handoff_type.value] = len([
                h for h in self._handoffs.values()
                if h.handoff_type == handoff_type
            ])
        
        # Count by status
        status_counts = {}
        for status in HandoffStatus:
            status_counts[status.value] = len([
                h for h in self._handoffs.values()
                if h.status == status
            ])
        
        # Calculate average resolution time
        completed_handoffs = [
            h for h in self._handoffs.values()
            if h.status == HandoffStatus.COMPLETED
        ]
        
        avg_resolution_time = 0.0
        if completed_handoffs:
            resolution_times = []
            for handoff in completed_handoffs:
                # Find completion response
                completion_response = None
                for response in self._responses.values():
                    if (response.handoff_id == handoff.handoff_id and 
                        response.response_type == "completion"):
                        completion_response = response
                        break
                
                if completion_response:
                    resolution_time = (completion_response.created_at - handoff.created_at).total_seconds() / 3600
                    resolution_times.append(resolution_time)
            
            if resolution_times:
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        return {
            "total_handoffs": total_handoffs,
            "active_handoffs": active_handoffs,
            "overdue_handoffs": overdue_handoffs,
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "average_resolution_time_hours": avg_resolution_time,
            "patterns_learned": len(self._patterns),
            "response_count": len(self._responses),
            "statistics": dict(self._stats)
        }
    
    def export_handoff_report(self, output_path: str = "handoff_report.json"):
        """Export comprehensive handoff report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_handoff_statistics(),
            "handoffs": [],
            "responses": [],
            "patterns": []
        }
        
        # Add handoffs
        for handoff in self._handoffs.values():
            handoff_data = {
                "handoff_id": handoff.handoff_id,
                "handoff_type": handoff.handoff_type.value,
                "title": handoff.title,
                "description": handoff.description,
                "primary_target": handoff.primary_target,
                "priority_level": handoff.priority_level,
                "urgency": handoff.urgency,
                "status": handoff.status.value,
                "created_at": handoff.created_at.isoformat(),
                "expires_at": handoff.expires_at.isoformat() if handoff.expires_at else None,
                "context_items_count": len(handoff.context_items),
                "acknowledgment_received": handoff.acknowledgment_received,
                "completion_confirmed": handoff.completion_confirmed
            }
            report["handoffs"].append(handoff_data)
        
        # Add responses
        for response in self._responses.values():
            response_data = {
                "response_id": response.response_id,
                "handoff_id": response.handoff_id,
                "response_type": response.response_type,
                "message": response.message,
                "progress_percent": response.progress_percent,
                "created_at": response.created_at.isoformat(),
                "files_modified_count": len(response.files_modified),
                "actions_taken_count": len(response.actions_taken)
            }
            report["responses"].append(response_data)
        
        # Add patterns
        for pattern in self._patterns.values():
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "success_rate": pattern.success_rate,
                "examples_count": pattern.examples_count,
                "confidence": pattern.confidence,
                "common_actions": pattern.common_actions,
                "last_updated": pattern.last_updated.isoformat()
            }
            report["patterns"].append(pattern_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Handoff report exported to {output_path}")
        except Exception as e:
            print(f"Error exporting handoff report: {e}")


class PriorityQueue:
    """Priority queue for handoff management."""
    
    def __init__(self):
        """Initialize priority queue."""
        import heapq
        self._queue = []
        self._index = 0
        
    def add(self, handoff: HandoffContext):
        """Add handoff to priority queue."""
        import heapq
        
        # Calculate priority score
        priority = self._calculate_priority(handoff)
        
        # Add to queue (negative for max-heap behavior)
        heapq.heappush(self._queue, (-priority, self._index, handoff))
        self._index += 1
    
    def get_next(self) -> Optional[HandoffContext]:
        """Get next highest priority handoff."""
        import heapq
        
        if self._queue:
            _, _, handoff = heapq.heappop(self._queue)
            return handoff
        return None
    
    def _calculate_priority(self, handoff: HandoffContext) -> float:
        """Calculate priority score for handoff."""
        score = 0.0
        
        # Priority level scoring
        priority_scores = {'critical': 100, 'high': 75, 'normal': 50, 'low': 25}
        score += priority_scores.get(handoff.priority_level, 50)
        
        # Urgency scoring
        urgency_scores = {'emergency': 50, 'high': 30, 'normal': 10, 'low': 0}
        score += urgency_scores.get(handoff.urgency, 10)
        
        # Age factor (older = higher priority)
        age_minutes = (datetime.now() - handoff.created_at).total_seconds() / 60
        score += min(20, age_minutes / 10)  # Max 20 points for age
        
        # Context richness (more context = higher priority)
        score += min(10, len(handoff.context_items) * 2)
        
        return score


class RetryManager:
    """Manage retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, backoff_base: float = 2):
        """Initialize retry manager."""
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._retry_counts = {}
        self._last_attempts = {}
    
    def should_retry(self, handoff_id: str) -> bool:
        """Check if handoff should be retried."""
        return self._retry_counts.get(handoff_id, 0) < self.max_retries
    
    def get_retry_delay(self, handoff_id: str) -> float:
        """Get delay before next retry."""
        retry_count = self._retry_counts.get(handoff_id, 0)
        return self.backoff_base ** retry_count
    
    def record_attempt(self, handoff_id: str):
        """Record a retry attempt."""
        self._retry_counts[handoff_id] = self._retry_counts.get(handoff_id, 0) + 1
        self._last_attempts[handoff_id] = datetime.now()
    
    def reset(self, handoff_id: str):
        """Reset retry count for successful handoff."""
        self._retry_counts.pop(handoff_id, None)
        self._last_attempts.pop(handoff_id, None)


class BatchProcessor:
    """Process related handoffs in batches."""
    
    def __init__(self, batch_size: int = 5, batch_timeout: int = 60):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout  # seconds
        self._batches = {}
        self._batch_timers = {}
    
    def add_to_batch(self, handoff: HandoffContext) -> Optional[str]:
        """Add handoff to batch, return batch ID if ready."""
        # Group by target and type
        batch_key = f"{handoff.handoff_type.value}_{Path(handoff.primary_target).parent}"
        
        if batch_key not in self._batches:
            self._batches[batch_key] = []
            self._batch_timers[batch_key] = datetime.now()
        
        self._batches[batch_key].append(handoff)
        
        # Check if batch is ready
        if len(self._batches[batch_key]) >= self.batch_size:
            return self._create_batch(batch_key)
        
        # Check if batch timeout
        if (datetime.now() - self._batch_timers[batch_key]).total_seconds() > self.batch_timeout:
            return self._create_batch(batch_key)
        
        return None
    
    def _create_batch(self, batch_key: str) -> str:
        """Create batch from accumulated handoffs."""
        batch_id = f"batch_{batch_key}_{int(datetime.now().timestamp())}"
        batch = self._batches.pop(batch_key, [])
        self._batch_timers.pop(batch_key, None)
        
        print(f"Created batch {batch_id} with {len(batch)} handoffs")
        return batch_id


# Convenience functions for common handoffs
def create_test_failure_handoff(test_file: str, error_message: str, 
                              manager: HandoffManager) -> str:
    """Create handoff for test failure."""
    return manager.create_escalation_handoff(
        issue_description=f"Test failure in {Path(test_file).name}",
        failed_attempts=["Automated test repair"],
        target=test_file,
        error_details=error_message
    )


def create_coverage_improvement_handoff(source_file: str, current_coverage: float,
                                      target_coverage: float, manager: HandoffManager) -> str:
    """Create handoff for coverage improvement."""
    return manager.create_handoff(
        handoff_type=HandoffType.WORK_DELEGATION,
        title=f"Improve Coverage: {Path(source_file).name}",
        description=f"Coverage is {current_coverage:.1f}%, needs to reach {target_coverage:.1f}%",
        primary_target=source_file,
        requested_action=f"Improve test coverage from {current_coverage:.1f}% to {target_coverage:.1f}%",
        priority_level="normal",
        context_data={"current_coverage": current_coverage, "target_coverage": target_coverage}
    )


def batch_handoffs(handoffs: List[Dict[str, Any]], manager: HandoffManager) -> List[str]:
    """Create a batch of related handoffs."""
    handoff_ids = []
    
    # Group by type and target
    for handoff_data in handoffs:
        handoff_id = manager.create_handoff(**handoff_data)
        handoff_ids.append(handoff_id)
    
    print(f"Created batch of {len(handoff_ids)} handoffs")
    return handoff_ids