"""
Enterprise Governance Framework

Comprehensive governance automation for enterprise documentation and security
with policy management, audit trails, and stakeholder workflow orchestration.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class GovernanceLevel(Enum):
    """Governance authority levels."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    ADMINISTRATIVE = "administrative"


class PolicyType(Enum):
    """Types of governance policies."""
    DOCUMENTATION_STANDARDS = "documentation_standards"
    SECURITY_REQUIREMENTS = "security_requirements"
    COMPLIANCE_MANDATES = "compliance_mandates"
    QUALITY_GATES = "quality_gates"
    APPROVAL_WORKFLOWS = "approval_workflows"
    RISK_MANAGEMENT = "risk_management"
    DATA_GOVERNANCE = "data_governance"
    CHANGE_MANAGEMENT = "change_management"


class ApprovalStatus(Enum):
    """Approval workflow status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


@dataclass
class GovernancePolicy:
    """Enterprise governance policy definition."""
    policy_id: str
    policy_type: PolicyType
    title: str
    description: str
    authority_level: GovernanceLevel
    effective_date: datetime
    expiration_date: Optional[datetime]
    
    # Policy rules and enforcement
    rules: List[Dict[str, Any]] = field(default_factory=list)
    enforcement_actions: List[str] = field(default_factory=list)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stakeholders and workflow
    policy_owner: str = ""
    approvers: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    
    # Monitoring and compliance
    compliance_metrics: Dict[str, str] = field(default_factory=dict)
    monitoring_frequency: str = "monthly"
    automated_enforcement: bool = True
    
    # Audit and reporting
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    last_review_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None


@dataclass
class ApprovalRequest:
    """Governance approval request."""
    request_id: str
    request_type: str
    title: str
    description: str
    requester: str
    created_date: datetime
    target_date: Optional[datetime]
    
    # Approval workflow
    current_approver: str
    approval_chain: List[str] = field(default_factory=list)
    approvals_received: List[Dict[str, Any]] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Request details
    affected_components: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    business_justification: str = ""
    technical_details: Dict[str, Any] = field(default_factory=dict)
    
    # Audit
    audit_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GovernanceMetrics:
    """Governance framework metrics."""
    total_policies: int = 0
    active_policies: int = 0
    compliance_rate: float = 0.0
    pending_approvals: int = 0
    policy_violations: int = 0
    governance_score: float = 0.0
    
    # Performance metrics
    average_approval_time: float = 0.0  # hours
    policy_effectiveness: float = 0.0
    stakeholder_satisfaction: float = 0.0
    audit_findings: int = 0


class EnterpriseGovernanceFramework:
    """
    Comprehensive enterprise governance framework for documentation and security
    with automated policy enforcement, approval workflows, and compliance monitoring.
    """
    
    def __init__(self):
        """Initialize enterprise governance framework."""
        self.policies = {}
        self.approval_requests = {}
        self.stakeholder_matrix = {}
        self.workflow_engines = {}
        self.compliance_monitors = {}
        
        # Governance configuration
        self.governance_config = {
            'auto_enforcement': True,
            'escalation_enabled': True,
            'audit_logging': True,
            'notification_channels': ['email', 'slack', 'dashboard']
        }
        
        # Initialize standard policies
        self._initialize_standard_policies()
        
        # Metrics tracking
        self.metrics = GovernanceMetrics()
        
        logger.info("Enterprise Governance Framework initialized")
        
    def _initialize_standard_policies(self) -> None:
        """Initialize standard enterprise governance policies."""
        # Documentation Standards Policy
        doc_policy = GovernancePolicy(
            policy_id="GOV-DOC-001",
            policy_type=PolicyType.DOCUMENTATION_STANDARDS,
            title="Enterprise Documentation Standards",
            description="Mandatory documentation requirements for all enterprise systems",
            authority_level=GovernanceLevel.STRATEGIC,
            effective_date=datetime.now(),
            expiration_date=datetime.now() + timedelta(days=365),
            rules=[
                {
                    'rule_id': 'DOC-001',
                    'description': 'All APIs must have OpenAPI 3.0 specifications',
                    'enforcement': 'blocking',
                    'validation': 'automated'
                },
                {
                    'rule_id': 'DOC-002',
                    'description': 'All public functions must have docstrings',
                    'enforcement': 'warning',
                    'validation': 'automated'
                },
                {
                    'rule_id': 'DOC-003',
                    'description': 'Architecture decisions must be documented in ADRs',
                    'enforcement': 'blocking',
                    'validation': 'manual'
                }
            ],
            policy_owner="chief_architect",
            approvers=["chief_architect", "engineering_director"],
            automated_enforcement=True
        )
        
        # Security Requirements Policy
        security_policy = GovernancePolicy(
            policy_id="GOV-SEC-001",
            policy_type=PolicyType.SECURITY_REQUIREMENTS,
            title="Enterprise Security Requirements",
            description="Mandatory security controls for all enterprise applications",
            authority_level=GovernanceLevel.STRATEGIC,
            effective_date=datetime.now(),
            expiration_date=datetime.now() + timedelta(days=365),
            rules=[
                {
                    'rule_id': 'SEC-001',
                    'description': 'All endpoints must implement authentication',
                    'enforcement': 'blocking',
                    'validation': 'automated'
                },
                {
                    'rule_id': 'SEC-002',
                    'description': 'Sensitive data must be encrypted at rest',
                    'enforcement': 'blocking',
                    'validation': 'automated'
                },
                {
                    'rule_id': 'SEC-003',
                    'description': 'Security scans must pass before deployment',
                    'enforcement': 'blocking',
                    'validation': 'automated'
                }
            ],
            policy_owner="ciso",
            approvers=["ciso", "security_architect"],
            automated_enforcement=True
        )
        
        self.policies[doc_policy.policy_id] = doc_policy
        self.policies[security_policy.policy_id] = security_policy
        
    async def create_approval_request(self, request_data: Dict[str, Any]) -> str:
        """
        Create new governance approval request.
        
        Args:
            request_data: Request details
            
        Returns:
            Request ID
        """
        request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
        
        # Determine approval chain based on request type and impact
        approval_chain = await self._determine_approval_chain(request_data)
        
        request = ApprovalRequest(
            request_id=request_id,
            request_type=request_data.get('type', 'general'),
            title=request_data.get('title', ''),
            description=request_data.get('description', ''),
            requester=request_data.get('requester', ''),
            created_date=datetime.now(),
            target_date=request_data.get('target_date'),
            current_approver=approval_chain[0] if approval_chain else '',
            approval_chain=approval_chain,
            affected_components=request_data.get('affected_components', []),
            risk_assessment=request_data.get('risk_assessment', {}),
            business_justification=request_data.get('business_justification', ''),
            technical_details=request_data.get('technical_details', {})
        )
        
        # Add to audit log
        request.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'request_created',
            'actor': request_data.get('requester', ''),
            'details': 'Approval request created'
        })
        
        self.approval_requests[request_id] = request
        
        # Start approval workflow
        await self._start_approval_workflow(request)
        
        logger.info(f"Created approval request {request_id}")
        return request_id
        
    async def process_approval(self, request_id: str, approver: str, decision: str, comments: str = "") -> Dict[str, Any]:
        """
        Process approval decision.
        
        Args:
            request_id: Request identifier
            approver: Approver identifier
            decision: 'approve', 'reject', 'escalate'
            comments: Optional comments
            
        Returns:
            Processing result
        """
        if request_id not in self.approval_requests:
            raise ValueError(f"Request {request_id} not found")
            
        request = self.approval_requests[request_id]
        
        # Validate approver authority
        if approver != request.current_approver:
            raise ValueError(f"Approver {approver} not authorized for this request")
            
        # Record approval
        approval_record = {
            'approver': approver,
            'decision': decision,
            'timestamp': datetime.now().isoformat(),
            'comments': comments
        }
        request.approvals_received.append(approval_record)
        
        # Add to audit log
        request.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': f'approval_{decision}',
            'actor': approver,
            'details': f"Approval decision: {decision}. Comments: {comments}"
        })
        
        result = {'status': 'processed', 'next_action': ''}
        
        if decision == 'approve':
            # Move to next approver or complete
            current_index = request.approval_chain.index(approver)
            if current_index + 1 < len(request.approval_chain):
                # More approvers in chain
                request.current_approver = request.approval_chain[current_index + 1]
                result['next_action'] = f"Waiting for approval from {request.current_approver}"
                await self._notify_next_approver(request)
            else:
                # All approvals received
                request.status = ApprovalStatus.APPROVED
                result['next_action'] = "Request fully approved"
                await self._complete_approval_workflow(request)
                
        elif decision == 'reject':
            request.status = ApprovalStatus.REJECTED
            result['next_action'] = "Request rejected"
            await self._handle_rejection(request)
            
        elif decision == 'escalate':
            request.status = ApprovalStatus.ESCALATED
            await self._handle_escalation(request)
            result['next_action'] = "Request escalated"
            
        return result
        
    async def enforce_policy(self, policy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce governance policy in given context.
        
        Args:
            policy_id: Policy identifier
            context: Enforcement context
            
        Returns:
            Enforcement result
        """
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
            
        policy = self.policies[policy_id]
        enforcement_result = {
            'policy_id': policy_id,
            'compliant': True,
            'violations': [],
            'warnings': [],
            'actions_taken': []
        }
        
        # Check each rule in the policy
        for rule in policy.rules:
            rule_result = await self._evaluate_rule(rule, context)
            
            if not rule_result['compliant']:
                enforcement_result['compliant'] = False
                
                violation = {
                    'rule_id': rule['rule_id'],
                    'description': rule['description'],
                    'evidence': rule_result.get('evidence', []),
                    'severity': rule_result.get('severity', 'medium')
                }
                
                if rule.get('enforcement') == 'blocking':
                    enforcement_result['violations'].append(violation)
                else:
                    enforcement_result['warnings'].append(violation)
                    
                # Take enforcement actions if automated
                if policy.automated_enforcement and rule.get('enforcement') == 'blocking':
                    actions = await self._take_enforcement_actions(rule, context)
                    enforcement_result['actions_taken'].extend(actions)
                    
        # Update policy audit trail
        policy.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'context': context.get('component', 'unknown'),
            'result': enforcement_result,
            'enforcer': 'automated_system'
        })
        
        return enforcement_result
        
    async def generate_governance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive governance dashboard."""
        # Calculate current metrics
        await self._update_metrics()
        
        # Get pending items
        pending_approvals = [req for req in self.approval_requests.values() 
                           if req.status == ApprovalStatus.PENDING]
        
        # Get recent violations
        recent_violations = []
        for policy in self.policies.values():
            recent_entries = [entry for entry in policy.audit_trail 
                            if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=7)]
            for entry in recent_entries:
                if not entry['result']['compliant']:
                    recent_violations.extend(entry['result']['violations'])
                    
        # Calculate trends
        trends = await self._calculate_governance_trends()
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': {
                'governance_score': self.metrics.governance_score,
                'compliance_rate': self.metrics.compliance_rate,
                'pending_approvals': len(pending_approvals),
                'recent_violations': len(recent_violations),
                'policies_active': self.metrics.active_policies
            },
            'pending_approvals': [
                {
                    'request_id': req.request_id,
                    'title': req.title,
                    'requester': req.requester,
                    'current_approver': req.current_approver,
                    'days_pending': (datetime.now() - req.created_date).days,
                    'risk_level': req.risk_assessment.get('level', 'medium')
                }
                for req in pending_approvals[:10]  # Top 10
            ],
            'policy_compliance': {
                policy_id: {
                    'title': policy.title,
                    'compliance_rate': self._calculate_policy_compliance(policy),
                    'last_violation': self._get_last_violation_date(policy),
                    'effectiveness': self._calculate_policy_effectiveness(policy)
                }
                for policy_id, policy in self.policies.items()
            },
            'trends': trends,
            'recommendations': await self._generate_governance_recommendations()
        }
        
        return dashboard
        
    async def automated_policy_review(self) -> Dict[str, Any]:
        """Perform automated policy review and optimization."""
        review_results = {
            'timestamp': datetime.now().isoformat(),
            'policies_reviewed': 0,
            'optimization_recommendations': [],
            'policy_conflicts': [],
            'coverage_gaps': [],
            'effectiveness_scores': {}
        }
        
        for policy_id, policy in self.policies.items():
            review_results['policies_reviewed'] += 1
            
            # Calculate effectiveness
            effectiveness = self._calculate_policy_effectiveness(policy)
            review_results['effectiveness_scores'][policy_id] = effectiveness
            
            # Generate optimization recommendations
            if effectiveness < 0.7:
                review_results['optimization_recommendations'].append({
                    'policy_id': policy_id,
                    'issue': 'Low effectiveness',
                    'recommendation': 'Review policy rules and enforcement mechanisms',
                    'priority': 'high'
                })
                
            # Check for upcoming expiration
            if policy.expiration_date and policy.expiration_date < datetime.now() + timedelta(days=30):
                review_results['optimization_recommendations'].append({
                    'policy_id': policy_id,
                    'issue': 'Expiring soon',
                    'recommendation': 'Schedule policy renewal review',
                    'priority': 'medium'
                })
                
        return review_results
        
    # Helper methods
    async def _determine_approval_chain(self, request_data: Dict[str, Any]) -> List[str]:
        """Determine approval chain based on request type and impact."""
        request_type = request_data.get('type', 'general')
        risk_level = request_data.get('risk_assessment', {}).get('level', 'medium')
        
        # Default approval chains by type and risk
        approval_chains = {
            ('security_change', 'high'): ['security_architect', 'ciso', 'engineering_director'],
            ('security_change', 'medium'): ['security_architect', 'ciso'],
            ('documentation_change', 'high'): ['tech_writer', 'chief_architect'],
            ('documentation_change', 'medium'): ['tech_writer'],
            ('policy_change', 'high'): ['policy_owner', 'governance_board', 'executive_sponsor'],
            ('general', 'high'): ['manager', 'director'],
            ('general', 'medium'): ['manager']
        }
        
        key = (request_type, risk_level)
        return approval_chains.get(key, ['manager'])
        
    def _calculate_policy_effectiveness(self, policy: GovernancePolicy) -> float:
        """Calculate policy effectiveness score."""
        if not policy.audit_trail:
            return 0.5  # No data
            
        recent_entries = [entry for entry in policy.audit_trail 
                         if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=30)]
        
        if not recent_entries:
            return 0.5
            
        compliant_count = sum(1 for entry in recent_entries if entry['result']['compliant'])
        return compliant_count / len(recent_entries)
        
    async def _update_metrics(self) -> None:
        """Update governance metrics."""
        self.metrics.total_policies = len(self.policies)
        self.metrics.active_policies = sum(1 for p in self.policies.values() 
                                         if not p.expiration_date or p.expiration_date > datetime.now())
        
        # Calculate overall compliance rate
        compliance_scores = [self._calculate_policy_effectiveness(p) for p in self.policies.values()]
        self.metrics.compliance_rate = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        # Pending approvals
        self.metrics.pending_approvals = sum(1 for req in self.approval_requests.values() 
                                           if req.status == ApprovalStatus.PENDING)
        
        # Calculate governance score (weighted average of multiple factors)
        self.metrics.governance_score = (
            self.metrics.compliance_rate * 0.4 +
            (1 - min(self.metrics.pending_approvals / 100, 1.0)) * 0.3 +  # Penalty for too many pending
            (self.metrics.active_policies / max(self.metrics.total_policies, 1)) * 0.3
        ) * 100