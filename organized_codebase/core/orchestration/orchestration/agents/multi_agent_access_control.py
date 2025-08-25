"""
MetaGPT Derived Multi-Agent Access Control
Extracted from MetaGPT team collaboration patterns and agent management
Enhanced for comprehensive multi-agent security and access control
"""

import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from .error_handler import SecurityError, AuthorizationError, ValidationError, security_error_handler


class AgentType(Enum):
    """Types of agents in the multi-agent system"""
    PRODUCT_MANAGER = "product_manager"
    ARCHITECT = "architect"
    PROJECT_MANAGER = "project_manager"
    ENGINEER = "engineer"
    QA_ENGINEER = "qa_engineer"
    RESEARCHER = "researcher"
    DESIGNER = "designer"
    ANALYST = "analyst"
    COORDINATOR = "coordinator"
    SUPERVISOR = "supervisor"


class PermissionLevel(Enum):
    """Permission levels for agent actions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    OWNER = "owner"


class ResourceType(Enum):
    """Types of resources in the system"""
    CODE = "code"
    DOCUMENT = "document"
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    WORKFLOW = "workflow"
    PROJECT = "project"
    TEAM = "team"
    SYSTEM = "system"


class AccessDecision(Enum):
    """Access control decisions"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"
    ESCALATE = "escalate"


@dataclass
class AgentCredentials:
    """Agent credentials and identity information"""
    agent_id: str
    agent_name: str
    agent_type: AgentType
    team_id: str
    owner_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    api_key: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    trust_score: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if agent has been active recently"""
        return (datetime.utcnow() - self.last_active).total_seconds() < 3600
    
    @property
    def is_trusted(self) -> bool:
        """Check if agent meets trust threshold"""
        return self.trust_score >= 70.0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_active = datetime.utcnow()


@dataclass
class AccessPolicy:
    """Access control policy definition"""
    policy_id: str
    name: str
    resource_type: ResourceType
    required_permission: PermissionLevel
    allowed_agent_types: Set[AgentType] = field(default_factory=set)
    denied_agent_types: Set[AgentType] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_request(self, agent_type: AgentType, resource_type: ResourceType, 
                       permission: PermissionLevel) -> bool:
        """Check if policy matches access request"""
        return (self.active and 
                self.resource_type == resource_type and
                self.required_permission == permission)


@dataclass
class AccessRequest:
    """Access control request"""
    request_id: str
    agent_id: str
    resource_id: str
    resource_type: ResourceType
    permission_requested: PermissionLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    justification: Optional[str] = None


@dataclass
class AccessResult:
    """Result of access control evaluation"""
    request_id: str
    agent_id: str
    decision: AccessDecision
    reason: str
    policies_applied: List[str] = field(default_factory=list)
    conditions_met: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    audit_info: Dict[str, Any] = field(default_factory=dict)


class TeamManager:
    """Team-based access control management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.teams: Dict[str, Dict[str, Any]] = {}
        self.team_memberships: Dict[str, str] = {}  # agent_id -> team_id
        self.team_hierarchies: Dict[str, Set[str]] = {}  # parent_team -> child_teams
    
    def create_team(self, team_name: str, team_owner: str, 
                   parent_team: str = None) -> str:
        """Create new team with hierarchical structure"""
        try:
            team_id = str(uuid.uuid4())
            
            team_config = {
                'team_id': team_id,
                'name': team_name,
                'owner': team_owner,
                'parent_team': parent_team,
                'members': {team_owner},
                'created_at': datetime.utcnow(),
                'permissions': set(),
                'resources': set()
            }
            
            self.teams[team_id] = team_config
            self.team_memberships[team_owner] = team_id
            
            # Update hierarchy
            if parent_team and parent_team in self.teams:
                if parent_team not in self.team_hierarchies:
                    self.team_hierarchies[parent_team] = set()
                self.team_hierarchies[parent_team].add(team_id)
            
            self.logger.info(f"Created team: {team_name} (ID: {team_id})")
            return team_id
            
        except Exception as e:
            error = SecurityError(f"Failed to create team: {str(e)}", "TEAM_CREATE_001")
            security_error_handler.handle_error(error)
            raise error
    
    def add_agent_to_team(self, agent_id: str, team_id: str, 
                         added_by: str) -> bool:
        """Add agent to team with authorization check"""
        try:
            if team_id not in self.teams:
                raise ValidationError(f"Team not found: {team_id}")
            
            team = self.teams[team_id]
            
            # Check if requester has authority to add members
            if added_by != team['owner'] and added_by not in team['members']:
                raise AuthorizationError("Insufficient permissions to add team member")
            
            # Add agent to team
            team['members'].add(agent_id)
            self.team_memberships[agent_id] = team_id
            
            self.logger.info(f"Added agent {agent_id} to team {team_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add agent to team: {e}")
            return False
    
    def get_agent_team(self, agent_id: str) -> Optional[str]:
        """Get team ID for agent"""
        return self.team_memberships.get(agent_id)
    
    def get_team_permissions(self, team_id: str) -> Set[str]:
        """Get permissions for team"""
        if team_id not in self.teams:
            return set()
        
        permissions = self.teams[team_id]['permissions'].copy()
        
        # Inherit permissions from parent teams
        parent_team = self.teams[team_id].get('parent_team')
        if parent_team and parent_team in self.teams:
            permissions.update(self.get_team_permissions(parent_team))
        
        return permissions
    
    def is_team_member(self, agent_id: str, team_id: str) -> bool:
        """Check if agent is member of team (including hierarchy)"""
        if team_id not in self.teams:
            return False
        
        # Direct membership
        if agent_id in self.teams[team_id]['members']:
            return True
        
        # Check child teams
        if team_id in self.team_hierarchies:
            for child_team in self.team_hierarchies[team_id]:
                if self.is_team_member(agent_id, child_team):
                    return True
        
        return False


class PermissionManager:
    """Fine-grained permission management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agent_permissions: Dict[str, Set[str]] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[PermissionLevel]]] = {}
        self.permission_templates: Dict[AgentType, Set[str]] = {}
        
        # Initialize permission templates
        self._initialize_permission_templates()
    
    def grant_permission(self, agent_id: str, permission: str, 
                        granted_by: str = None) -> bool:
        """Grant permission to agent"""
        try:
            if agent_id not in self.agent_permissions:
                self.agent_permissions[agent_id] = set()
            
            self.agent_permissions[agent_id].add(permission)
            
            self.logger.info(f"Granted permission '{permission}' to agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to grant permission: {e}")
            return False
    
    def revoke_permission(self, agent_id: str, permission: str,
                         revoked_by: str = None) -> bool:
        """Revoke permission from agent"""
        try:
            if agent_id in self.agent_permissions:
                self.agent_permissions[agent_id].discard(permission)
                
                self.logger.info(f"Revoked permission '{permission}' from agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to revoke permission: {e}")
            return False
    
    def has_permission(self, agent_id: str, permission: str) -> bool:
        """Check if agent has specific permission"""
        if agent_id not in self.agent_permissions:
            return False
        
        return permission in self.agent_permissions[agent_id]
    
    def get_agent_permissions(self, agent_id: str) -> Set[str]:
        """Get all permissions for agent"""
        return self.agent_permissions.get(agent_id, set()).copy()
    
    def apply_permission_template(self, agent_id: str, agent_type: AgentType) -> bool:
        """Apply permission template based on agent type"""
        try:
            if agent_type not in self.permission_templates:
                return False
            
            template_permissions = self.permission_templates[agent_type]
            
            if agent_id not in self.agent_permissions:
                self.agent_permissions[agent_id] = set()
            
            self.agent_permissions[agent_id].update(template_permissions)
            
            self.logger.info(f"Applied {agent_type.value} template to agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply permission template: {e}")
            return False
    
    def set_resource_permissions(self, resource_id: str, agent_id: str, 
                               permissions: Set[PermissionLevel]) -> bool:
        """Set specific permissions for agent on resource"""
        try:
            if resource_id not in self.resource_permissions:
                self.resource_permissions[resource_id] = {}
            
            self.resource_permissions[resource_id][agent_id] = permissions
            
            self.logger.info(f"Set resource permissions for {agent_id} on {resource_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set resource permissions: {e}")
            return False
    
    def get_resource_permissions(self, resource_id: str, agent_id: str) -> Set[PermissionLevel]:
        """Get agent's permissions on specific resource"""
        if (resource_id not in self.resource_permissions or 
            agent_id not in self.resource_permissions[resource_id]):
            return set()
        
        return self.resource_permissions[resource_id][agent_id].copy()
    
    def _initialize_permission_templates(self):
        """Initialize default permission templates for agent types"""
        self.permission_templates = {
            AgentType.PRODUCT_MANAGER: {
                'project.read', 'project.write', 'requirements.write',
                'team.view', 'documents.write', 'workflow.approve'
            },
            AgentType.ARCHITECT: {
                'project.read', 'code.read', 'code.write', 'design.write',
                'documents.write', 'model.create', 'config.write'
            },
            AgentType.PROJECT_MANAGER: {
                'project.read', 'project.write', 'team.manage', 'workflow.manage',
                'documents.write', 'reports.generate'
            },
            AgentType.ENGINEER: {
                'code.read', 'code.write', 'code.execute', 'documents.read',
                'model.use', 'data.read'
            },
            AgentType.QA_ENGINEER: {
                'code.read', 'code.execute', 'documents.read', 'documents.write',
                'workflow.test', 'reports.generate'
            },
            AgentType.RESEARCHER: {
                'data.read', 'data.analyze', 'documents.read', 'documents.write',
                'model.use', 'reports.generate'
            },
            AgentType.DESIGNER: {
                'documents.read', 'documents.write', 'design.write',
                'model.use', 'resources.create'
            },
            AgentType.ANALYST: {
                'data.read', 'data.analyze', 'documents.read', 'documents.write',
                'reports.generate', 'model.use'
            },
            AgentType.COORDINATOR: {
                'team.view', 'workflow.view', 'documents.read',
                'communication.manage', 'reports.view'
            },
            AgentType.SUPERVISOR: {
                'team.manage', 'project.read', 'workflow.manage',
                'documents.read', 'reports.view', 'audit.view'
            }
        }


class MultiAgentAccessController:
    """Comprehensive multi-agent access control system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, AgentCredentials] = {}
        self.policies: Dict[str, AccessPolicy] = {}
        self.team_manager = TeamManager()
        self.permission_manager = PermissionManager()
        self.access_history: List[AccessResult] = []
        self.max_history = 10000
        self._lock = threading.RLock()
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def register_agent(self, agent_name: str, agent_type: AgentType,
                      team_id: str = None, owner_id: str = None) -> str:
        """Register new agent in the access control system"""
        try:
            with self._lock:
                agent_id = str(uuid.uuid4())
                
                credentials = AgentCredentials(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    agent_type=agent_type,
                    team_id=team_id or "default",
                    owner_id=owner_id or "system"
                )
                
                # Generate API key
                credentials.api_key = f"ak_{agent_id[:8]}_{int(time.time())}"
                
                # Apply permission template
                self.permission_manager.apply_permission_template(agent_id, agent_type)
                
                # Store agent
                self.agents[agent_id] = credentials
                
                self.logger.info(f"Registered agent: {agent_name} ({agent_type.value})")
                return agent_id
                
        except Exception as e:
            error = SecurityError(f"Failed to register agent: {str(e)}", "AGENT_REG_001")
            security_error_handler.handle_error(error)
            raise error
    
    def evaluate_access(self, agent_id: str, resource_id: str, 
                       resource_type: ResourceType,
                       permission_requested: PermissionLevel,
                       context: Dict[str, Any] = None) -> AccessResult:
        """Evaluate access request against policies"""
        try:
            with self._lock:
                request = AccessRequest(
                    request_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    permission_requested=permission_requested,
                    context=context or {}
                )
                
                # Check if agent exists and is active
                if agent_id not in self.agents:
                    result = AccessResult(
                        request_id=request.request_id,
                        agent_id=agent_id,
                        decision=AccessDecision.DENY,
                        reason="Agent not found",
                        audit_info={'error': 'agent_not_found'}
                    )
                    self._add_to_access_history(result)
                    return result
                
                agent = self.agents[agent_id]
                agent.update_activity()
                
                # Check agent trust score
                if not agent.is_trusted:
                    result = AccessResult(
                        request_id=request.request_id,
                        agent_id=agent_id,
                        decision=AccessDecision.DENY,
                        reason=f"Agent trust score too low: {agent.trust_score}",
                        audit_info={'trust_score': agent.trust_score}
                    )
                    self._add_to_access_history(result)
                    return result
                
                # Evaluate policies
                applicable_policies = self._find_applicable_policies(
                    agent.agent_type, resource_type, permission_requested
                )
                
                if not applicable_policies:
                    # No policies found - default deny
                    result = AccessResult(
                        request_id=request.request_id,
                        agent_id=agent_id,
                        decision=AccessDecision.DENY,
                        reason="No applicable policies found",
                        audit_info={'default_action': 'deny'}
                    )
                    self._add_to_access_history(result)
                    return result
                
                # Check policies in priority order
                for policy in sorted(applicable_policies, key=lambda p: p.priority, reverse=True):
                    decision = self._evaluate_policy(agent, request, policy)
                    
                    if decision != AccessDecision.CONDITIONAL:
                        result = AccessResult(
                            request_id=request.request_id,
                            agent_id=agent_id,
                            decision=decision,
                            reason=f"Policy '{policy.name}' applied",
                            policies_applied=[policy.policy_id],
                            audit_info={
                                'policy_name': policy.name,
                                'policy_id': policy.policy_id,
                                'agent_type': agent.agent_type.value
                            }
                        )
                        self._add_to_access_history(result)
                        return result
                
                # All policies were conditional - escalate
                result = AccessResult(
                    request_id=request.request_id,
                    agent_id=agent_id,
                    decision=AccessDecision.ESCALATE,
                    reason="All applicable policies require escalation",
                    policies_applied=[p.policy_id for p in applicable_policies],
                    audit_info={'escalation_required': True}
                )
                self._add_to_access_history(result)
                return result
                
        except Exception as e:
            self.logger.error(f"Access evaluation failed: {e}")
            
            error_result = AccessResult(
                request_id=getattr(request, 'request_id', 'error'),
                agent_id=agent_id,
                decision=AccessDecision.DENY,
                reason=f"Evaluation error: {str(e)}",
                audit_info={'error': str(e)}
            )
            return error_result
    
    def add_access_policy(self, policy: AccessPolicy) -> bool:
        """Add new access policy"""
        try:
            with self._lock:
                self.policies[policy.policy_id] = policy
                self.logger.info(f"Added access policy: {policy.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add access policy: {e}")
            return False
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent access control statistics"""
        try:
            with self._lock:
                total_agents = len(self.agents)
                active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
                trusted_agents = sum(1 for agent in self.agents.values() if agent.is_trusted)
                
                # Agent type distribution
                type_distribution = {}
                for agent in self.agents.values():
                    agent_type = agent.agent_type.value
                    type_distribution[agent_type] = type_distribution.get(agent_type, 0) + 1
                
                # Access decision statistics
                recent_access = [result for result in self.access_history 
                               if (datetime.utcnow() - result.timestamp).total_seconds() < 3600]
                
                decision_counts = {}
                for result in recent_access:
                    decision = result.decision.value
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
                
                # Team statistics
                total_teams = len(self.team_manager.teams)
                
                return {
                    'agent_stats': {
                        'total_agents': total_agents,
                        'active_agents': active_agents,
                        'trusted_agents': trusted_agents,
                        'activity_rate_pct': (active_agents / max(total_agents, 1)) * 100,
                        'trust_rate_pct': (trusted_agents / max(total_agents, 1)) * 100,
                        'type_distribution': type_distribution
                    },
                    'access_stats_1h': {
                        'total_requests': len(recent_access),
                        'decision_distribution': decision_counts,
                        'approval_rate_pct': (decision_counts.get('allow', 0) / max(len(recent_access), 1)) * 100
                    },
                    'policy_stats': {
                        'total_policies': len(self.policies),
                        'active_policies': sum(1 for p in self.policies.values() if p.active)
                    },
                    'team_stats': {
                        'total_teams': total_teams
                    },
                    'security_score': self._calculate_access_security_score()
                }
                
        except Exception as e:
            self.logger.error(f"Error generating agent statistics: {e}")
            return {'error': str(e)}
    
    def _find_applicable_policies(self, agent_type: AgentType, 
                                resource_type: ResourceType,
                                permission: PermissionLevel) -> List[AccessPolicy]:
        """Find policies applicable to the access request"""
        applicable = []
        
        for policy in self.policies.values():
            if not policy.active:
                continue
            
            if policy.resource_type != resource_type:
                continue
            
            if policy.required_permission != permission:
                continue
            
            # Check agent type restrictions
            if policy.allowed_agent_types and agent_type not in policy.allowed_agent_types:
                continue
            
            if policy.denied_agent_types and agent_type in policy.denied_agent_types:
                continue
            
            applicable.append(policy)
        
        return applicable
    
    def _evaluate_policy(self, agent: AgentCredentials, 
                        request: AccessRequest, 
                        policy: AccessPolicy) -> AccessDecision:
        """Evaluate specific policy against request"""
        try:
            # Basic checks
            if agent.agent_type in policy.denied_agent_types:
                return AccessDecision.DENY
            
            if policy.allowed_agent_types and agent.agent_type not in policy.allowed_agent_types:
                return AccessDecision.DENY
            
            # Check conditions
            if policy.conditions:
                if not self._check_policy_conditions(agent, request, policy.conditions):
                    return AccessDecision.CONDITIONAL
            
            # Check resource-specific permissions
            resource_perms = self.permission_manager.get_resource_permissions(
                request.resource_id, request.agent_id
            )
            
            if resource_perms and request.permission_requested in resource_perms:
                return AccessDecision.ALLOW
            
            # Check agent permissions
            required_permission = f"{request.resource_type.value}.{request.permission_requested.value}"
            if self.permission_manager.has_permission(request.agent_id, required_permission):
                return AccessDecision.ALLOW
            
            return AccessDecision.DENY
            
        except Exception as e:
            self.logger.error(f"Policy evaluation error: {e}")
            return AccessDecision.DENY
    
    def _check_policy_conditions(self, agent: AgentCredentials, 
                               request: AccessRequest,
                               conditions: Dict[str, Any]) -> bool:
        """Check if policy conditions are met"""
        try:
            # Time-based conditions
            if 'time_range' in conditions:
                current_hour = datetime.utcnow().hour
                start_hour, end_hour = conditions['time_range']
                if not (start_hour <= current_hour <= end_hour):
                    return False
            
            # Team-based conditions
            if 'requires_team_membership' in conditions:
                required_team = conditions['requires_team_membership']
                if not self.team_manager.is_team_member(agent.agent_id, required_team):
                    return False
            
            # Trust score conditions
            if 'min_trust_score' in conditions:
                min_trust = conditions['min_trust_score']
                if agent.trust_score < min_trust:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Condition check error: {e}")
            return False
    
    def _calculate_access_security_score(self) -> float:
        """Calculate overall access security score"""
        try:
            score = 0.0
            
            # Agent security
            if self.agents:
                trusted_rate = sum(1 for a in self.agents.values() if a.is_trusted) / len(self.agents)
                score += trusted_rate * 30
            
            # Policy coverage
            active_policies = sum(1 for p in self.policies.values() if p.active)
            if active_policies > 0:
                score += min(30, active_policies * 5)
            
            # Access pattern analysis
            if self.access_history:
                recent_access = [r for r in self.access_history[-100:]]
                if recent_access:
                    approval_rate = sum(1 for r in recent_access if r.decision == AccessDecision.ALLOW) / len(recent_access)
                    # Moderate approval rate indicates good security (not too permissive, not too restrictive)
                    if 0.3 <= approval_rate <= 0.7:
                        score += 20
                    elif 0.1 <= approval_rate < 0.3 or 0.7 < approval_rate <= 0.9:
                        score += 10
            
            # Team organization
            if len(self.team_manager.teams) > 0:
                score += 20
            
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating access security score: {e}")
            return 0.0
    
    def _add_to_access_history(self, result: AccessResult):
        """Add access result to history"""
        self.access_history.append(result)
        
        # Limit history size
        if len(self.access_history) > self.max_history:
            self.access_history = self.access_history[-self.max_history // 2:]
    
    def _initialize_default_policies(self):
        """Initialize default access control policies"""
        try:
            # Code access policy
            code_policy = AccessPolicy(
                policy_id="default_code_access",
                name="Default Code Access",
                resource_type=ResourceType.CODE,
                required_permission=PermissionLevel.WRITE,
                allowed_agent_types={AgentType.ENGINEER, AgentType.ARCHITECT},
                priority=100
            )
            
            # Document access policy
            doc_policy = AccessPolicy(
                policy_id="default_doc_access",
                name="Default Document Access",
                resource_type=ResourceType.DOCUMENT,
                required_permission=PermissionLevel.READ,
                allowed_agent_types=set(AgentType),  # All agent types can read docs
                priority=90
            )
            
            # System access policy
            system_policy = AccessPolicy(
                policy_id="system_access",
                name="System Access Control",
                resource_type=ResourceType.SYSTEM,
                required_permission=PermissionLevel.ADMIN,
                allowed_agent_types={AgentType.SUPERVISOR},
                priority=200
            )
            
            self.policies[code_policy.policy_id] = code_policy
            self.policies[doc_policy.policy_id] = doc_policy
            self.policies[system_policy.policy_id] = system_policy
            
            self.logger.info("Default access policies initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default policies: {e}")


# Global multi-agent access controller
multi_agent_access_control = MultiAgentAccessController()


# Convenience functions
def register_agent_with_access_control(agent_name: str, agent_type: AgentType) -> str:
    """Convenience function to register agent with access control"""
    return multi_agent_access_control.register_agent(agent_name, agent_type)


def check_agent_access(agent_id: str, resource_id: str, resource_type: ResourceType,
                      permission: PermissionLevel) -> bool:
    """Convenience function to check agent access"""
    result = multi_agent_access_control.evaluate_access(agent_id, resource_id, resource_type, permission)
    return result.decision == AccessDecision.ALLOW