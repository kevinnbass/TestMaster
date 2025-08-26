"""
TestMaster Orchestrator Module - Layer 3

Intelligent orchestration inspired by framework analysis:
- OpenAI Swarm: Function-based handoff for dynamic routing
- Agent-Squad: Configuration-driven classification  
- LangGraph: Supervisor delegation for task distribution
- Agency-Swarm: Context preservation in handoffs

Provides:
- Automatic file tagging and classification
- Work distribution logic between TestMaster and Claude Code
- Automated investigation and analysis
- Smart handoff system with context preservation
"""

from .file_tagger import FileTagger, FileClassification, TaggingRule
from .work_distributor import WorkDistributor, WorkItem, HandoffDecision
from .investigator import AutoInvestigator, Investigation, InvestigationResult
from .handoff_manager import HandoffManager, HandoffContext, HandoffResponse

__all__ = [
    "FileTagger",
    "FileClassification", 
    "TaggingRule",
    "WorkDistributor",
    "WorkItem",
    "HandoffDecision",
    "AutoInvestigator",
    "Investigation",
    "InvestigationResult",
    "HandoffManager",
    "HandoffContext",
    "HandoffResponse"
]