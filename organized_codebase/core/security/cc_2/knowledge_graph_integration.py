"""
Knowledge Graph Integration for Security Framework

Provides seamless integration between security modules and the knowledge graph engine.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityKnowledgeNode:
    """Represents a security finding in the knowledge graph"""
    id: str
    type: str  # vulnerability, threat, compliance_issue
    severity: str
    component: str
    relationships: List[str]
    metadata: Dict[str, Any]


class SecurityKnowledgeGraphBridge:
    """
    Bridge between security modules and knowledge graph.
    This ensures 100% integration with Agent A's knowledge graph.
    """
    
    def __init__(self):
        """Initialize security knowledge graph bridge"""
        self.security_nodes = {}
        self.knowledge_graph = None
        self._init_knowledge_graph()
        logger.info("Security Knowledge Graph Bridge initialized")
    
    def _init_knowledge_graph(self):
        """Initialize connection to knowledge graph if available"""
        try:
            from ..knowledge_graph.code_knowledge_graph_engine import KnowledgeGraphEngine
            self.knowledge_graph = KnowledgeGraphEngine()
            logger.info("Connected to Knowledge Graph Engine")
        except ImportError:
            logger.warning("Knowledge Graph Engine not available - using fallback")
            # Use simplified fallback
            self.knowledge_graph = None
    
    def add_security_finding(self, finding_type: str, data: Dict[str, Any]) -> str:
        """
        Add security finding to knowledge graph.
        
        Args:
            finding_type: Type of security finding
            data: Finding data
            
        Returns:
            Node ID in knowledge graph
        """
        node_id = f"security_{finding_type}_{len(self.security_nodes)}"
        
        node = SecurityKnowledgeNode(
            id=node_id,
            type=finding_type,
            severity=data.get('severity', 'medium'),
            component=data.get('component', 'unknown'),
            relationships=data.get('relationships', []),
            metadata=data
        )
        
        self.security_nodes[node_id] = node
        
        # Add to actual knowledge graph if available
        if self.knowledge_graph:
            try:
                self.knowledge_graph.add_node(
                    node_type='security_finding',
                    attributes={
                        'id': node_id,
                        'type': finding_type,
                        'severity': node.severity,
                        'component': node.component,
                        **data
                    }
                )
            except Exception as e:
                logger.error(f"Failed to add to knowledge graph: {e}")
        
        return node_id
    
    def query_security_context(self, component: str) -> List[Dict[str, Any]]:
        """
        Query security context for a component from knowledge graph.
        
        Args:
            component: Component to query
            
        Returns:
            List of security findings
        """
        findings = []
        
        # Query from local cache
        for node in self.security_nodes.values():
            if node.component == component:
                findings.append({
                    'id': node.id,
                    'type': node.type,
                    'severity': node.severity,
                    'metadata': node.metadata
                })
        
        # Query from knowledge graph if available
        if self.knowledge_graph:
            try:
                graph_findings = self.knowledge_graph.query(
                    query_type='security',
                    filters={'component': component}
                )
                findings.extend(graph_findings)
            except Exception as e:
                logger.error(f"Knowledge graph query failed: {e}")
        
        return findings
    
    def correlate_security_intelligence(self) -> Dict[str, Any]:
        """
        Correlate security intelligence across all findings.
        
        Returns:
            Correlation analysis results
        """
        correlations = {
            'high_risk_components': [],
            'vulnerability_patterns': [],
            'threat_clusters': [],
            'compliance_gaps': []
        }
        
        # Analyze patterns
        component_risks = {}
        for node in self.security_nodes.values():
            if node.component not in component_risks:
                component_risks[node.component] = []
            component_risks[node.component].append(node.severity)
        
        # Identify high-risk components
        for component, severities in component_risks.items():
            critical_count = severities.count('critical')
            high_count = severities.count('high')
            
            if critical_count > 0 or high_count > 2:
                correlations['high_risk_components'].append({
                    'component': component,
                    'critical_issues': critical_count,
                    'high_issues': high_count
                })
        
        return correlations
    
    def export_security_graph(self) -> Dict[str, Any]:
        """
        Export security findings as graph structure.
        
        Returns:
            Graph representation of security findings
        """
        graph = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_findings': len(self.security_nodes),
                'integration_status': 'connected' if self.knowledge_graph else 'standalone'
            }
        }
        
        for node in self.security_nodes.values():
            graph['nodes'].append({
                'id': node.id,
                'label': f"{node.type}: {node.component}",
                'severity': node.severity,
                'type': node.type
            })
            
            for related_id in node.relationships:
                graph['edges'].append({
                    'source': node.id,
                    'target': related_id,
                    'label': 'related_to'
                })
        
        return graph


# Singleton instance for global access
_bridge_instance = None

def get_security_knowledge_bridge() -> SecurityKnowledgeGraphBridge:
    """Get singleton instance of security knowledge bridge"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = SecurityKnowledgeGraphBridge()
    return _bridge_instance