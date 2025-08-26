"""
AI Security Integration Layer

Provides seamless integration between security modules and AI Code Explorer.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AISecurityExplorer:
    """
    Integration layer between AI Code Explorer and Security modules.
    Allows natural language queries about security findings.
    """
    
    def __init__(self):
        """Initialize AI Security Explorer with fallback support"""
        self.ai_explorer = None
        self.knowledge_graph = None
        self.security_context = {}
        self._init_ai_components()
        logger.info("AI Security Explorer initialized")
    
    def _init_ai_components(self):
        """Initialize AI components with graceful fallback"""
        try:
            # Try to import knowledge graph
            from ..knowledge_graph.code_knowledge_graph_engine import CodeKnowledgeGraphEngine
            self.knowledge_graph = CodeKnowledgeGraphEngine()
            
            # Try to import AI explorer with knowledge graph
            from ..knowledge_graph.ai_code_explorer import AICodeExplorer
            self.ai_explorer = AICodeExplorer(self.knowledge_graph)
            logger.info("AI Code Explorer connected successfully")
        except ImportError as e:
            logger.warning(f"AI Code Explorer not fully available: {e}")
            # Use fallback implementation
            self.ai_explorer = None
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            self.ai_explorer = None
    
    async def query_security_insights(self, query: str) -> Dict[str, Any]:
        """
        Query security insights using natural language.
        
        Args:
            query: Natural language security query
            
        Returns:
            Security insights and recommendations
        """
        insights = {
            'query': query,
            'findings': [],
            'recommendations': [],
            'risk_assessment': None
        }
        
        # Extract security context from query
        if 'vulnerability' in query.lower() or 'vuln' in query.lower():
            insights['findings'].append({
                'type': 'vulnerability_analysis',
                'message': 'Analyzing vulnerability patterns in codebase'
            })
        
        if 'threat' in query.lower():
            insights['findings'].append({
                'type': 'threat_analysis',
                'message': 'Evaluating threat landscape'
            })
        
        if 'compliance' in query.lower():
            insights['findings'].append({
                'type': 'compliance_check',
                'message': 'Checking compliance requirements'
            })
        
        # Use AI explorer if available
        if self.ai_explorer:
            try:
                ai_response = await self.ai_explorer.converse(query)
                insights['ai_analysis'] = ai_response
            except Exception as e:
                logger.error(f"AI conversation failed: {e}")
        
        # Add security-specific insights
        insights['recommendations'] = self._generate_security_recommendations(query)
        insights['risk_assessment'] = self._assess_query_risk(query)
        
        return insights
    
    def _generate_security_recommendations(self, query: str) -> List[str]:
        """Generate security recommendations based on query"""
        recommendations = []
        
        query_lower = query.lower()
        
        if 'sql' in query_lower or 'database' in query_lower:
            recommendations.append("Use parameterized queries to prevent SQL injection")
        
        if 'password' in query_lower or 'auth' in query_lower:
            recommendations.append("Implement secure password hashing with bcrypt or Argon2")
        
        if 'api' in query_lower:
            recommendations.append("Implement rate limiting and authentication for API endpoints")
        
        if 'input' in query_lower or 'user' in query_lower:
            recommendations.append("Validate and sanitize all user inputs")
        
        if 'encrypt' in query_lower or 'crypto' in query_lower:
            recommendations.append("Use industry-standard encryption algorithms")
        
        return recommendations
    
    def _assess_query_risk(self, query: str) -> str:
        """Assess risk level based on query content"""
        high_risk_keywords = ['vulnerability', 'exploit', 'injection', 'overflow', 'privilege']
        medium_risk_keywords = ['security', 'authentication', 'authorization', 'encryption']
        
        query_lower = query.lower()
        
        for keyword in high_risk_keywords:
            if keyword in query_lower:
                return 'HIGH'
        
        for keyword in medium_risk_keywords:
            if keyword in query_lower:
                return 'MEDIUM'
        
        return 'LOW'
    
    def integrate_security_findings(self, findings: List[Dict[str, Any]]) -> None:
        """
        Integrate security findings into AI context.
        
        Args:
            findings: List of security findings to integrate
        """
        for finding in findings:
            finding_id = finding.get('id', f"finding_{len(self.security_context)}")
            self.security_context[finding_id] = finding
            
            # Add to knowledge graph if available
            if self.knowledge_graph:
                try:
                    self.knowledge_graph.add_security_node(finding)
                except Exception as e:
                    logger.error(f"Failed to add security node: {e}")
    
    def get_security_context_summary(self) -> Dict[str, Any]:
        """Get summary of current security context"""
        return {
            'total_findings': len(self.security_context),
            'integration_status': 'connected' if self.ai_explorer else 'fallback',
            'high_risk_findings': sum(
                1 for f in self.security_context.values() 
                if f.get('severity') in ['critical', 'high']
            ),
            'categories': list(set(
                f.get('type', 'unknown') 
                for f in self.security_context.values()
            ))
        }


# Singleton instance
_ai_security_instance = None

def get_ai_security_explorer() -> AISecurityExplorer:
    """Get singleton instance of AI Security Explorer"""
    global _ai_security_instance
    if _ai_security_instance is None:
        _ai_security_instance = AISecurityExplorer()
    return _ai_security_instance