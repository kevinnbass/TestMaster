"""
Focused Incident Responder

Handles incident response coordination, automated remediation, and response workflows.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class IncidentStatus(Enum):
    """Incident status levels."""
    NEW = "new"
    INVESTIGATING = "investigating"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    CLOSED = "closed"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResponseAction(Enum):
    """Types of response actions."""
    ISOLATE_SYSTEM = "isolate_system"
    BLOCK_IP = "block_ip"
    QUARANTINE_FILE = "quarantine_file"
    RESET_CREDENTIALS = "reset_credentials"
    PATCH_SYSTEM = "patch_system"
    NOTIFY_STAKEHOLDERS = "notify_stakeholders"
    COLLECT_EVIDENCE = "collect_evidence"
    ANALYZE_LOGS = "analyze_logs"


@dataclass
class SecurityIncident:
    """Security incident definition."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsePlaybook:
    """Incident response playbook."""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    response_steps: List[Dict[str, Any]]
    automation_level: str  # manual, semi_automated, fully_automated
    estimated_duration: int  # minutes
    required_roles: List[str] = field(default_factory=list)


class IncidentResponder:
    """
    Focused incident response engine.
    Handles incident coordination, automated remediation, and response workflows.
    """
    
    def __init__(self):
        """Initialize incident responder with playbooks and response capabilities."""
        try:
            # Incident management
            self.active_incidents = {}  # incident_id -> SecurityIncident
            self.incident_history = []
            self.incident_counter = 0
            
            # Response playbooks
            self.response_playbooks = {}  # playbook_id -> ResponsePlaybook
            self.automation_enabled = True
            self.auto_containment_enabled = True
            
            # Response teams and roles
            self.response_teams = {
                'security_analyst': {'available': True, 'current_load': 0},
                'incident_commander': {'available': True, 'current_load': 0},
                'forensics_expert': {'available': True, 'current_load': 0},
                'communications': {'available': True, 'current_load': 0}
            }
            
            # Response metrics
            self.response_metrics = {
                'total_incidents': 0,
                'incidents_resolved': 0,
                'average_response_time': 0.0,
                'average_resolution_time': 0.0,
                'automated_responses': 0,
                'manual_responses': 0
            }
            
            # Initialize default playbooks
            self._initialize_default_playbooks()
            
            logger.info("Incident Responder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize incident responder: {e}")
            raise
    
    async def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new security incident and initiate response.
        
        Args:
            incident_data: Incident information and context
            
        Returns:
            Incident creation result with assigned ID and initial response
        """
        try:
            self.incident_counter += 1
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{self.incident_counter:04d}"
            
            # Create incident object
            incident = SecurityIncident(
                incident_id=incident_id,
                title=incident_data.get('title', 'Security Incident'),
                description=incident_data.get('description', ''),
                severity=IncidentSeverity(incident_data.get('severity', 'medium')),
                status=IncidentStatus.NEW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                affected_systems=incident_data.get('affected_systems', []),
                indicators=incident_data.get('indicators', []),
                metadata=incident_data.get('metadata', {})
            )
            
            # Store incident
            self.active_incidents[incident_id] = incident
            self.response_metrics['total_incidents'] += 1
            
            # Assign to response team
            assigned_to = await self._assign_incident(incident)
            incident.assigned_to = assigned_to
            
            # Trigger automated response if enabled
            initial_response = []
            if self.automation_enabled:
                initial_response = await self._trigger_automated_response(incident)
            
            # Select appropriate playbook
            playbook = self._select_response_playbook(incident)
            
            result = {
                'incident_id': incident_id,
                'status': 'created',
                'severity': incident.severity.value,
                'assigned_to': assigned_to,
                'initial_response_actions': initial_response,
                'recommended_playbook': playbook.playbook_id if playbook else None,
                'estimated_resolution_time': playbook.estimated_duration if playbook else 120,
                'creation_timestamp': incident.created_at.isoformat()
            }
            
            logger.info(f"Incident created: {incident_id} (severity: {incident.severity.value})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            return {
                'status': 'creation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def update_incident_status(self, incident_id: str, new_status: str, 
                                   update_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Update incident status and trigger appropriate response actions.
        
        Args:
            incident_id: Incident ID to update
            new_status: New status for the incident
            update_notes: Optional notes about the status change
            
        Returns:
            Status update result
        """
        try:
            if incident_id not in self.active_incidents:
                return {
                    'status': 'update_failed',
                    'error': f'Incident {incident_id} not found',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            incident = self.active_incidents[incident_id]
            old_status = incident.status
            incident.status = IncidentStatus(new_status)
            incident.updated_at = datetime.utcnow()
            
            # Add status change to metadata
            if 'status_history' not in incident.metadata:
                incident.metadata['status_history'] = []
            
            incident.metadata['status_history'].append({
                'from_status': old_status.value,
                'to_status': new_status,
                'timestamp': datetime.utcnow().isoformat(),
                'notes': update_notes
            })
            
            # Trigger status-specific actions
            status_actions = await self._handle_status_change(incident, old_status)
            
            # Check if incident should be closed
            if new_status == 'closed':
                await self._close_incident(incident)
            
            return {
                'incident_id': incident_id,
                'status': 'updated',
                'old_status': old_status.value,
                'new_status': new_status,
                'triggered_actions': status_actions,
                'update_timestamp': incident.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update incident status {incident_id}: {e}")
            return {
                'status': 'update_failed',
                'error': str(e),
                'incident_id': incident_id
            }
    
    async def execute_response_action(self, incident_id: str, action_type: str, 
                                    action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific response action for an incident.
        
        Args:
            incident_id: Incident ID
            action_type: Type of response action to execute
            action_params: Parameters for the action
            
        Returns:
            Action execution result
        """
        try:
            if incident_id not in self.active_incidents:
                return {
                    'status': 'execution_failed',
                    'error': f'Incident {incident_id} not found'
                }
            
            incident = self.active_incidents[incident_id]
            action_start = datetime.utcnow()
            
            # Execute the specific action
            execution_result = await self._execute_action(action_type, action_params)
            
            # Record action in incident
            action_record = {
                'action_type': action_type,
                'parameters': action_params,
                'execution_time': action_start.isoformat(),
                'result': execution_result,
                'executed_by': 'automated_system'  # In real system, would be actual user
            }
            
            incident.response_actions.append(action_record)
            incident.updated_at = datetime.utcnow()
            
            # Update metrics
            if execution_result.get('status') == 'success':
                self.response_metrics['automated_responses'] += 1
            
            return {
                'incident_id': incident_id,
                'action_type': action_type,
                'execution_status': execution_result.get('status', 'unknown'),
                'execution_result': execution_result,
                'execution_timestamp': action_start.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute response action {action_type}: {e}")
            return {
                'status': 'execution_failed',
                'error': str(e),
                'action_type': action_type
            }
    
    async def get_incident_details(self, incident_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific incident.
        
        Args:
            incident_id: Incident ID to retrieve
            
        Returns:
            Detailed incident information
        """
        try:
            if incident_id not in self.active_incidents:
                return {
                    'status': 'not_found',
                    'error': f'Incident {incident_id} not found'
                }
            
            incident = self.active_incidents[incident_id]
            
            # Calculate incident duration
            duration = (datetime.utcnow() - incident.created_at).total_seconds()
            
            # Get response statistics
            successful_actions = sum(
                1 for action in incident.response_actions 
                if action.get('result', {}).get('status') == 'success'
            )
            
            return {
                'incident_id': incident.incident_id,
                'title': incident.title,
                'description': incident.description,
                'severity': incident.severity.value,
                'status': incident.status.value,
                'assigned_to': incident.assigned_to,
                'created_at': incident.created_at.isoformat(),
                'updated_at': incident.updated_at.isoformat(),
                'duration_seconds': duration,
                'affected_systems': incident.affected_systems,
                'indicators': incident.indicators,
                'response_actions_count': len(incident.response_actions),
                'successful_actions': successful_actions,
                'response_actions': incident.response_actions,
                'metadata': incident.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get incident details {incident_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'incident_id': incident_id
            }
    
    def list_active_incidents(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all active incidents with optional severity filtering.
        
        Args:
            severity_filter: Optional severity level filter
            
        Returns:
            List of active incidents
        """
        try:
            incidents = []
            
            for incident in self.active_incidents.values():
                # Apply severity filter if specified
                if severity_filter and incident.severity.value != severity_filter:
                    continue
                
                incident_summary = {
                    'incident_id': incident.incident_id,
                    'title': incident.title,
                    'severity': incident.severity.value,
                    'status': incident.status.value,
                    'assigned_to': incident.assigned_to,
                    'created_at': incident.created_at.isoformat(),
                    'affected_systems_count': len(incident.affected_systems),
                    'response_actions_count': len(incident.response_actions)
                }
                incidents.append(incident_summary)
            
            # Sort by severity and creation time
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            incidents.sort(key=lambda x: (severity_order.get(x['severity'], 4), x['created_at']))
            
            return incidents
            
        except Exception as e:
            logger.error(f"Failed to list active incidents: {e}")
            return []
    
    def get_response_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive incident response metrics.
        
        Returns:
            Response performance metrics
        """
        try:
            # Calculate additional metrics
            total_incidents = self.response_metrics['total_incidents']
            resolved_incidents = self.response_metrics['incidents_resolved']
            
            resolution_rate = (resolved_incidents / total_incidents * 100) if total_incidents > 0 else 0.0
            
            # Current incident statistics
            active_count = len(self.active_incidents)
            critical_active = sum(
                1 for incident in self.active_incidents.values() 
                if incident.severity == IncidentSeverity.CRITICAL
            )
            
            # Team utilization
            team_utilization = {}
            for role, info in self.response_teams.items():
                team_utilization[role] = {
                    'available': info['available'],
                    'current_load': info['current_load']
                }
            
            return {
                'response_metrics': self.response_metrics,
                'current_statistics': {
                    'active_incidents': active_count,
                    'critical_active': critical_active,
                    'resolution_rate_percent': resolution_rate
                },
                'team_utilization': team_utilization,
                'system_status': {
                    'automation_enabled': self.automation_enabled,
                    'auto_containment_enabled': self.auto_containment_enabled,
                    'playbooks_available': len(self.response_playbooks)
                },
                'metrics_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get response metrics: {e}")
            return {
                'error': str(e),
                'metrics_timestamp': datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    def _initialize_default_playbooks(self) -> None:
        """Initialize default incident response playbooks."""
        try:
            # Malware incident playbook
            malware_playbook = ResponsePlaybook(
                playbook_id="PB001_MALWARE",
                name="Malware Incident Response",
                description="Standard response for malware incidents",
                trigger_conditions=["malware_detected", "suspicious_file"],
                response_steps=[
                    {"step": 1, "action": "isolate_system", "description": "Isolate affected systems"},
                    {"step": 2, "action": "collect_evidence", "description": "Collect forensic evidence"},
                    {"step": 3, "action": "analyze_logs", "description": "Analyze security logs"},
                    {"step": 4, "action": "quarantine_file", "description": "Quarantine malicious files"},
                    {"step": 5, "action": "patch_system", "description": "Apply security patches"}
                ],
                automation_level="semi_automated",
                estimated_duration=120,
                required_roles=["security_analyst", "forensics_expert"]
            )
            
            # Data breach playbook
            breach_playbook = ResponsePlaybook(
                playbook_id="PB002_BREACH",
                name="Data Breach Response",
                description="Response for potential data breaches",
                trigger_conditions=["data_exfiltration", "unauthorized_access"],
                response_steps=[
                    {"step": 1, "action": "isolate_system", "description": "Isolate compromised systems"},
                    {"step": 2, "action": "reset_credentials", "description": "Reset affected user credentials"},
                    {"step": 3, "action": "notify_stakeholders", "description": "Notify legal and compliance teams"},
                    {"step": 4, "action": "collect_evidence", "description": "Preserve evidence for investigation"},
                    {"step": 5, "action": "analyze_logs", "description": "Determine scope of breach"}
                ],
                automation_level="manual",
                estimated_duration=240,
                required_roles=["incident_commander", "forensics_expert", "communications"]
            )
            
            self.response_playbooks[malware_playbook.playbook_id] = malware_playbook
            self.response_playbooks[breach_playbook.playbook_id] = breach_playbook
            
        except Exception as e:
            logger.error(f"Failed to initialize default playbooks: {e}")
    
    async def _assign_incident(self, incident: SecurityIncident) -> str:
        """Assign incident to appropriate response team member."""
        try:
            # Simple assignment logic - in real implementation would be more sophisticated
            if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                return "incident_commander"
            else:
                return "security_analyst"
        except Exception as e:
            logger.error(f"Incident assignment failed: {e}")
            return "security_analyst"
    
    async def _trigger_automated_response(self, incident: SecurityIncident) -> List[Dict[str, Any]]:
        """Trigger automated response actions."""
        try:
            actions = []
            
            # Automatic isolation for critical incidents
            if incident.severity == IncidentSeverity.CRITICAL and self.auto_containment_enabled:
                for system in incident.affected_systems:
                    action_result = await self._execute_action(
                        'isolate_system', 
                        {'system_id': system, 'isolation_type': 'network'}
                    )
                    actions.append({
                        'action': 'isolate_system',
                        'target': system,
                        'result': action_result
                    })
            
            # Automatic evidence collection
            evidence_result = await self._execute_action(
                'collect_evidence',
                {'systems': incident.affected_systems, 'evidence_type': 'logs'}
            )
            actions.append({
                'action': 'collect_evidence',
                'result': evidence_result
            })
            
            return actions
            
        except Exception as e:
            logger.error(f"Automated response trigger failed: {e}")
            return []
    
    def _select_response_playbook(self, incident: SecurityIncident) -> Optional[ResponsePlaybook]:
        """Select appropriate response playbook based on incident characteristics."""
        try:
            # Match playbook based on incident indicators and metadata
            incident_type = incident.metadata.get('incident_type', '').lower()
            
            for playbook in self.response_playbooks.values():
                for condition in playbook.trigger_conditions:
                    if condition.lower() in incident_type or condition.lower() in incident.description.lower():
                        return playbook
            
            # Default to first available playbook
            return list(self.response_playbooks.values())[0] if self.response_playbooks else None
            
        except Exception as e:
            logger.error(f"Playbook selection failed: {e}")
            return None
    
    async def _handle_status_change(self, incident: SecurityIncident, 
                                  old_status: IncidentStatus) -> List[str]:
        """Handle incident status change and trigger appropriate actions."""
        try:
            triggered_actions = []
            
            if incident.status == IncidentStatus.INVESTIGATING:
                # Start investigation procedures
                triggered_actions.append("investigation_started")
                
            elif incident.status == IncidentStatus.CONTAINMENT:
                # Ensure containment measures are in place
                for system in incident.affected_systems:
                    await self._execute_action('isolate_system', {'system_id': system})
                triggered_actions.append("containment_measures_applied")
                
            elif incident.status == IncidentStatus.RECOVERY:
                # Begin recovery procedures
                triggered_actions.append("recovery_procedures_initiated")
                
            return triggered_actions
            
        except Exception as e:
            logger.error(f"Status change handling failed: {e}")
            return []
    
    async def _close_incident(self, incident: SecurityIncident) -> None:
        """Close incident and update metrics."""
        try:
            # Move to incident history
            self.incident_history.append(incident)
            
            # Remove from active incidents
            if incident.incident_id in self.active_incidents:
                del self.active_incidents[incident.incident_id]
            
            # Update metrics
            self.response_metrics['incidents_resolved'] += 1
            
            # Calculate resolution time
            resolution_time = (incident.updated_at - incident.created_at).total_seconds()
            
            # Update average resolution time
            resolved_count = self.response_metrics['incidents_resolved']
            current_avg = self.response_metrics['average_resolution_time']
            
            if resolved_count > 1:
                self.response_metrics['average_resolution_time'] = (
                    (current_avg * (resolved_count - 1) + resolution_time) / resolved_count
                )
            else:
                self.response_metrics['average_resolution_time'] = resolution_time
                
        except Exception as e:
            logger.error(f"Incident closure failed: {e}")
    
    async def _execute_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific response action."""
        try:
            # Mock action execution - in real implementation would perform actual actions
            await asyncio.sleep(0.1)  # Simulate action execution time
            
            action_results = {
                'isolate_system': {'status': 'success', 'message': f"System {params.get('system_id', 'unknown')} isolated"},
                'block_ip': {'status': 'success', 'message': f"IP {params.get('ip_address', 'unknown')} blocked"},
                'quarantine_file': {'status': 'success', 'message': f"File {params.get('file_path', 'unknown')} quarantined"},
                'reset_credentials': {'status': 'success', 'message': f"Credentials reset for {params.get('user_id', 'unknown')}"},
                'collect_evidence': {'status': 'success', 'message': 'Evidence collection initiated'},
                'analyze_logs': {'status': 'success', 'message': 'Log analysis completed'},
                'notify_stakeholders': {'status': 'success', 'message': 'Stakeholders notified'},
                'patch_system': {'status': 'success', 'message': 'Security patches applied'}
            }
            
            return action_results.get(action_type, {
                'status': 'unknown_action',
                'message': f'Unknown action type: {action_type}'
            })
            
        except Exception as e:
            logger.error(f"Action execution failed for {action_type}: {e}")
            return {
                'status': 'execution_failed',
                'error': str(e)
            }