"""
Conversational Interface Module
=============================

Conversational monitoring interface inspired by AutoGen patterns.
RESTORED from enhanced_monitor.py - this functionality was missing in consolidation.

Author: TestMaster Phase 1C Consolidation (RESTORATION)
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .event_monitoring import MonitoringAgent

class ConversationalMonitor:
    """
    Conversational interface for monitoring system.
    Inspired by AutoGen's conversational patterns.
    RESTORED from enhanced_monitor.py
    """
    
    def __init__(self):
        self.conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {}
        self.agents: Dict[str, MonitoringAgent] = {}
        self.logger = logging.getLogger('ConversationalMonitor')
    
    def add_agent(self, agent: MonitoringAgent):
        """Add a monitoring agent to the conversation"""
        self.agents[agent.name] = agent
        self.logger.info(f"Added agent {agent.name} to conversation")
    
    async def process_message(self, message: str, sender: str = "user") -> str:
        """Process a conversational message and generate response"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "message": message
        })
        
        # Analyze message intent
        intent = await self._analyze_intent(message)
        
        # Route to appropriate agent or provide general response
        response = await self._generate_response(message, intent)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "sender": "assistant",
            "message": response
        })
        
        return response
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user message intent"""
        intent = {
            "type": "general",
            "target_agent": None,
            "action": "query",
            "keywords": []
        }
        
        message_lower = message.lower()
        
        # Simple intent classification
        if any(word in message_lower for word in ["performance", "speed", "slow", "fast"]):
            intent["type"] = "performance"
            intent["target_agent"] = "performance_monitor"
        elif any(word in message_lower for word in ["quality", "coverage", "tests", "bugs"]):
            intent["type"] = "quality"
            intent["target_agent"] = "quality_monitor"
        elif any(word in message_lower for word in ["security", "vulnerability", "threat"]):
            intent["type"] = "security"
            intent["target_agent"] = "security_monitor"
        elif any(word in message_lower for word in ["collaboration", "agents", "communication"]):
            intent["type"] = "collaboration"
            intent["target_agent"] = "collaboration_monitor"
        
        # Detect action type
        if any(word in message_lower for word in ["show", "display", "get", "what"]):
            intent["action"] = "query"
        elif any(word in message_lower for word in ["alert", "notify", "warn"]):
            intent["action"] = "alert"
        elif any(word in message_lower for word in ["analyze", "check", "investigate"]):
            intent["action"] = "analyze"
        
        return intent
    
    async def _generate_response(self, message: str, intent: Dict[str, Any]) -> str:
        """Generate response based on message and intent"""
        if intent["target_agent"] and intent["target_agent"] in self.agents:
            # Route to specific agent
            agent = self.agents[intent["target_agent"]]
            return await agent.respond_to_query(message, self.context)
        else:
            # General monitoring response
            return await self._generate_general_response(message, intent)
    
    async def _generate_general_response(self, message: str, intent: Dict[str, Any]) -> str:
        """Generate general monitoring response"""
        if intent["action"] == "query":
            if "status" in message.lower():
                return self._get_system_status()
            elif "help" in message.lower():
                return self._get_help_message()
            else:
                return "I can help you monitor test performance, quality, security, and collaboration. What would you like to know?"
        else:
            return "I understand you want to monitor the testing system. How can I assist you?"
    
    def _get_system_status(self) -> str:
        """Get overall system status"""
        active_agents = sum(1 for agent in self.agents.values() if agent.active)
        total_events = sum(len(agent.events) for agent in self.agents.values())
        
        return f"""System Status:
- Active monitoring agents: {active_agents}/{len(self.agents)}
- Total monitoring events: {total_events}
- Conversation ID: {self.conversation_id}
- Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    def _get_help_message(self) -> str:
        """Get help message"""
        capabilities = []
        for agent in self.agents.values():
            capabilities.extend(agent.capabilities)
        
        return f"""TestMaster Enhanced Monitoring Help:

Available monitoring capabilities:
{chr(10).join(f"- {cap}" for cap in set(capabilities))}

You can ask about:
- System performance and speed
- Test quality and coverage
- Security vulnerabilities
- Agent collaboration
- Overall system status

Example queries:
- "How is the system performing?"
- "Show me quality metrics"
- "Any security issues?"
- "What's the collaboration status?"
"""

# Export key components
__all__ = [
    'ConversationalMonitor'
]