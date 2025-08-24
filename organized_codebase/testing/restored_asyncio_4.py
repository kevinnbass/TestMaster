"""
Enhanced Test Monitor
=====================

Advanced monitoring system combining Phidata's multi-modal capabilities
with AutoGen's conversational patterns for comprehensive test monitoring.

Author: TestMaster Team
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

from core.observability import global_observability
from core.monitor import RealTimeMonitor

class MonitoringMode(Enum):
    """Monitoring operation modes"""
    PASSIVE = "passive"           # Observe only
    INTERACTIVE = "interactive"   # Respond to queries
    PROACTIVE = "proactive"      # Generate insights and alerts
    CONVERSATIONAL = "conversational"  # Full conversational interface

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MonitoringEvent:
    """Represents a monitoring event"""
    id: str = field(default_factory=lambda: f"event_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    source: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    resolved: bool = False

class MonitoringAgent(ABC):
    """Base class for monitoring agents"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.agent_id = f"monitor_{uuid.uuid4().hex[:12]}"
        self.active = False
        self.events: List[MonitoringEvent] = []
        self.logger = logging.getLogger(f'MonitoringAgent.{name}')
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze monitoring data and generate events"""
        pass
    
    @abstractmethod
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to monitoring queries"""
        pass
    
    async def start_monitoring(self):
        """Start the monitoring agent"""
        self.active = True
        self.logger.info(f"Monitoring agent {self.name} started")
    
    async def stop_monitoring(self):
        """Stop the monitoring agent"""
        self.active = False
        self.logger.info(f"Monitoring agent {self.name} stopped")

class ConversationalMonitor:
    """
    Conversational interface for monitoring system.
    Inspired by AutoGen's conversational patterns.
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

class MultiModalAnalyzer:
    """
    Multi-modal data analyzer inspired by Phidata's capabilities.
    Handles various data types and formats for comprehensive analysis.
    """
    
    def __init__(self):
        self.analyzer_id = f"analyzer_{uuid.uuid4().hex[:12]}"
        self.supported_formats = {
            "json": self._analyze_json,
            "csv": self._analyze_csv,
            "log": self._analyze_logs,
            "metrics": self._analyze_metrics,
            "image": self._analyze_image,
            "text": self._analyze_text
        }
        self.logger = logging.getLogger('MultiModalAnalyzer')
    
    async def analyze_data(self, data: Any, data_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data of various types and formats"""
        if data_type not in self.supported_formats:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        analysis_start = time.time()
        
        try:
            analyzer_func = self.supported_formats[data_type]
            result = await analyzer_func(data, context or {})
            
            analysis_time = time.time() - analysis_start
            
            return {
                "analyzer_id": self.analyzer_id,
                "data_type": data_type,
                "analysis_time": analysis_time,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {data_type}: {e}")
            return {
                "analyzer_id": self.analyzer_id,
                "data_type": data_type,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_json(self, data: Union[dict, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON data"""
        if isinstance(data, str):
            data = json.loads(data)
        
        analysis = {
            "structure": {},
            "insights": [],
            "anomalies": [],
            "recommendations": []
        }
        
        # Analyze structure
        analysis["structure"] = {
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "depth": self._calculate_dict_depth(data) if isinstance(data, dict) else 0,
            "size": len(str(data))
        }
        
        # Look for performance data
        if "performance" in data or "metrics" in data:
            analysis["insights"].append("Performance metrics detected")
        
        # Look for error indicators
        if "error" in str(data).lower() or "fail" in str(data).lower():
            analysis["anomalies"].append("Error indicators found in data")
        
        return analysis
    
    async def _analyze_csv(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CSV data"""
        lines = data.strip().split('\n')
        
        analysis = {
            "rows": len(lines),
            "columns": len(lines[0].split(',')) if lines else 0,
            "insights": [],
            "patterns": []
        }
        
        if analysis["rows"] > 1000:
            analysis["insights"].append("Large dataset detected - consider sampling")
        
        return analysis
    
    async def _analyze_logs(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log data"""
        lines = data.strip().split('\n')
        
        analysis = {
            "total_lines": len(lines),
            "error_count": 0,
            "warning_count": 0,
            "patterns": [],
            "timeline": {}
        }
        
        for line in lines:
            line_lower = line.lower()
            if "error" in line_lower:
                analysis["error_count"] += 1
            elif "warning" in line_lower or "warn" in line_lower:
                analysis["warning_count"] += 1
        
        # Calculate error rate
        if analysis["total_lines"] > 0:
            error_rate = analysis["error_count"] / analysis["total_lines"] * 100
            if error_rate > 5:
                analysis["patterns"].append(f"High error rate: {error_rate:.1f}%")
        
        return analysis
    
    async def _analyze_metrics(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics data"""
        analysis = {
            "metrics_count": len(data),
            "trends": {},
            "alerts": [],
            "summary": {}
        }
        
        for metric_name, value in data.items():
            if isinstance(value, (int, float)):
                # Simple threshold checking
                if "response_time" in metric_name.lower() and value > 1000:
                    analysis["alerts"].append(f"High response time: {metric_name} = {value}ms")
                elif "error_rate" in metric_name.lower() and value > 5:
                    analysis["alerts"].append(f"High error rate: {metric_name} = {value}%")
                elif "cpu" in metric_name.lower() and value > 80:
                    analysis["alerts"].append(f"High CPU usage: {metric_name} = {value}%")
        
        return analysis
    
    async def _analyze_image(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image data (placeholder for actual image analysis)"""
        return {
            "type": "image",
            "format": "unknown",
            "insights": ["Image analysis not fully implemented"],
            "recommendations": ["Consider integrating computer vision capabilities"]
        }
    
    async def _analyze_text(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text data"""
        analysis = {
            "length": len(data),
            "word_count": len(data.split()),
            "sentiment": "neutral",
            "keywords": [],
            "insights": []
        }
        
        # Simple keyword extraction
        important_words = ["error", "performance", "quality", "security", "test", "failed", "success"]
        for word in important_words:
            if word in data.lower():
                analysis["keywords"].append(word)
        
        # Simple sentiment analysis
        positive_words = ["success", "pass", "good", "excellent", "improved"]
        negative_words = ["error", "fail", "bad", "poor", "degraded"]
        
        positive_count = sum(1 for word in positive_words if word in data.lower())
        negative_count = sum(1 for word in negative_words if word in data.lower())
        
        if positive_count > negative_count:
            analysis["sentiment"] = "positive"
        elif negative_count > positive_count:
            analysis["sentiment"] = "negative"
        
        return analysis
    
    def _calculate_dict_depth(self, d: dict, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a nested dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth
        
        return max(
            self._calculate_dict_depth(value, current_depth + 1)
            for value in d.values()
        )

class EnhancedTestMonitor:
    """
    Enhanced monitoring system combining Phidata's multi-modal capabilities
    with AutoGen's conversational patterns for comprehensive test monitoring.
    """
    
    def __init__(self, mode: MonitoringMode = MonitoringMode.INTERACTIVE):
        self.monitor_id = f"enhanced_monitor_{uuid.uuid4().hex[:12]}"
        self.mode = mode
        
        # Core components
        self.base_monitor = RealTimeMonitor()
        self.conversational_monitor = ConversationalMonitor()
        self.multimodal_analyzer = MultiModalAnalyzer()
        
        # Monitoring agents
        self.monitoring_agents: Dict[str, MonitoringAgent] = {}
        
        # Event management
        self.events: List[MonitoringEvent] = []
        self.event_handlers: Dict[str, Callable] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.monitor_metrics = {
            "events_processed": 0,
            "conversations_handled": 0,
            "analyses_performed": 0,
            "alerts_generated": 0,
            "uptime": 0.0
        }
        
        # Configuration
        self.config = {
            "alert_thresholds": {
                "response_time": 1000,  # ms
                "error_rate": 5,        # %
                "cpu_usage": 80,        # %
                "memory_usage": 85      # %
            },
            "retention_days": 7,
            "analysis_interval": 30  # seconds
        }
        
        self.logger = logging.getLogger('EnhancedTestMonitor')
        self.start_time = datetime.now()
    
    def add_monitoring_agent(self, agent: MonitoringAgent):
        """Add a monitoring agent to the system"""
        self.monitoring_agents[agent.name] = agent
        self.conversational_monitor.add_agent(agent)
        self.logger.info(f"Added monitoring agent: {agent.name}")
    
    async def start_monitoring(self, session_id: Optional[str] = None):
        """Start the enhanced monitoring system"""
        # Start base monitor
        self.base_monitor.start_monitoring()
        
        # Start all monitoring agents
        for agent in self.monitoring_agents.values():
            await agent.start_monitoring()
        
        # Track in observability
        if session_id:
            global_observability.track_agent_action(
                session_id,
                "EnhancedTestMonitor",
                "monitoring_started",
                {
                    "monitor_id": self.monitor_id,
                    "mode": self.mode.value,
                    "agent_count": len(self.monitoring_agents)
                }
            )
        
        # Start background monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Enhanced monitoring started in {self.mode.value} mode")
    
    async def stop_monitoring(self):
        """Stop the enhanced monitoring system"""
        # Stop all monitoring agents
        for agent in self.monitoring_agents.values():
            await agent.stop_monitoring()
        
        # Stop base monitor
        self.base_monitor.stop_monitoring()
        
        # Calculate final metrics
        self.monitor_metrics["uptime"] = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info("Enhanced monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect data from base monitor
                current_data = self.base_monitor.get_current_metrics()
                
                # Process with monitoring agents
                await self._process_monitoring_data(current_data)
                
                # Perform periodic analysis
                await self._perform_periodic_analysis()
                
                # Clean up old events
                await self._cleanup_old_events()
                
                # Wait for next iteration
                await asyncio.sleep(self.config["analysis_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_monitoring_data(self, data: Dict[str, Any]):
        """Process monitoring data with all agents"""
        for agent in self.monitoring_agents.values():
            try:
                events = await agent.analyze(data)
                self.events.extend(events)
                self.monitor_metrics["events_processed"] += len(events)
                
                # Generate alerts for high-severity events
                for event in events:
                    if event.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                        await self._generate_alert(event)
                        
            except Exception as e:
                self.logger.error(f"Error processing data with agent {agent.name}: {e}")
    
    async def _perform_periodic_analysis(self):
        """Perform periodic multi-modal analysis"""
        try:
            # Analyze recent events
            recent_events = [
                event for event in self.events
                if event.timestamp > datetime.now() - timedelta(minutes=10)
            ]
            
            if recent_events:
                event_data = {
                    "event_count": len(recent_events),
                    "event_types": [event.event_type for event in recent_events],
                    "alert_levels": [event.level.value for event in recent_events]
                }
                
                analysis = await self.multimodal_analyzer.analyze_data(
                    event_data,
                    "json",
                    {"context": "periodic_analysis"}
                )
                
                self.monitor_metrics["analyses_performed"] += 1
                
        except Exception as e:
            self.logger.error(f"Error in periodic analysis: {e}")
    
    async def _generate_alert(self, event: MonitoringEvent):
        """Generate alert for high-severity events"""
        alert_message = f"ALERT [{event.level.value.upper()}]: {event.message}"
        
        # In a real implementation, you'd send to notification systems
        self.logger.warning(alert_message)
        self.monitor_metrics["alerts_generated"] += 1
    
    async def _cleanup_old_events(self):
        """Clean up old events based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.config["retention_days"])
        
        initial_count = len(self.events)
        self.events = [event for event in self.events if event.timestamp > cutoff_date]
        
        cleaned_count = initial_count - len(self.events)
        if cleaned_count > 0:
            self.logger.debug(f"Cleaned up {cleaned_count} old events")
    
    async def process_conversation(self, message: str, sender: str = "user") -> str:
        """Process conversational monitoring query"""
        response = await self.conversational_monitor.process_message(message, sender)
        self.monitor_metrics["conversations_handled"] += 1
        return response
    
    async def analyze_data(self, data: Any, data_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data using multi-modal analyzer"""
        return await self.multimodal_analyzer.analyze_data(data, data_type, context)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        agent_statuses = {
            name: {
                "active": agent.active,
                "events_count": len(agent.events),
                "capabilities": agent.capabilities
            }
            for name, agent in self.monitoring_agents.items()
        }
        
        recent_events = [
            {
                "id": event.id,
                "type": event.event_type,
                "level": event.level.value,
                "message": event.message,
                "timestamp": event.timestamp.isoformat()
            }
            for event in self.events[-10:]  # Last 10 events
        ]
        
        return {
            "monitor_id": self.monitor_id,
            "mode": self.mode.value,
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "metrics": self.monitor_metrics,
            "agent_statuses": agent_statuses,
            "recent_events": recent_events,
            "total_events": len(self.events),
            "active_conversations": len(self.conversational_monitor.conversation_history)
        }

# Export components
__all__ = [
    'EnhancedTestMonitor',
    'MonitoringAgent',
    'ConversationalMonitor',
    'MultiModalAnalyzer',
    'MonitoringMode',
    'AlertLevel',
    'MonitoringEvent'
]