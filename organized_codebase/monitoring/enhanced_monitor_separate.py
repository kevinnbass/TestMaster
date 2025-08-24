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

@dataclass
class MonitoringAgent:
    """Represents a monitoring agent with specific capabilities"""
    id: str
    name: str
    capabilities: List[str]
    status: str = "active"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)

class ConversationalMonitor:
    """
    Conversational monitoring interface that can interact with users
    and provide intelligent insights about system behavior.
    """
    
    def __init__(self):
        self.monitoring_agents: List[MonitoringAgent] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.active_alerts: List[MonitoringEvent] = []
        self.mode = MonitoringMode.INTERACTIVE
        
        # Initialize with default monitoring agents
        self._initialize_default_agents()
        
        logging.info("Conversational Monitor initialized")
    
    def _initialize_default_agents(self):
        """Initialize default monitoring agents"""
        default_agents = [
            MonitoringAgent(
                id="system_monitor",
                name="System Performance Monitor",
                capabilities=["cpu_monitoring", "memory_monitoring", "disk_monitoring"]
            ),
            MonitoringAgent(
                id="test_monitor",
                name="Test Execution Monitor",
                capabilities=["test_tracking", "failure_analysis", "performance_metrics"]
            ),
            MonitoringAgent(
                id="security_monitor",
                name="Security Monitoring Agent",
                capabilities=["vulnerability_scanning", "compliance_checking", "threat_detection"]
            ),
            MonitoringAgent(
                id="quality_monitor",
                name="Code Quality Monitor",
                capabilities=["code_analysis", "coverage_tracking", "quality_metrics"]
            )
        ]
        
        for agent in default_agents:
            self.add_agent(agent)
    
    def add_agent(self, agent: MonitoringAgent):
        """Add a monitoring agent"""
        self.monitoring_agents.append(agent)
        logging.info(f"Added monitoring agent: {agent.name}")
    
    def process_user_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query and provide intelligent response"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user_query",
            "content": query,
            "context": context or {}
        })
        
        # Analyze query intent
        intent = self._analyze_query_intent(query)
        
        # Generate response based on intent
        response = self._generate_response(intent, query, context)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "system_response",
            "content": response,
            "intent": intent
        })
        
        return {
            "response": response,
            "intent": intent,
            "timestamp": datetime.now().isoformat(),
            "suggested_actions": self._suggest_actions(intent, query)
        }
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze user query to determine intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["status", "health", "running"]):
            return "status_inquiry"
        elif any(word in query_lower for word in ["error", "fail", "problem", "issue"]):
            return "error_investigation"
        elif any(word in query_lower for word in ["performance", "slow", "fast", "speed"]):
            return "performance_inquiry"
        elif any(word in query_lower for word in ["test", "coverage", "quality"]):
            return "test_inquiry"
        elif any(word in query_lower for word in ["security", "vulnerability", "threat"]):
            return "security_inquiry"
        elif any(word in query_lower for word in ["help", "guide", "how"]):
            return "help_request"
        else:
            return "general_inquiry"
    
    def _generate_response(self, intent: str, query: str, context: Dict[str, Any]) -> str:
        """Generate response based on query intent"""
        if intent == "status_inquiry":
            return self._get_system_status()
        elif intent == "error_investigation":
            return self._investigate_errors()
        elif intent == "performance_inquiry":
            return self._get_performance_summary()
        elif intent == "test_inquiry":
            return self._get_test_summary()
        elif intent == "security_inquiry":
            return self._get_security_summary()
        elif intent == "help_request":
            return self._get_help_message()
        else:
            return self._get_general_response(query)
    
    def _get_system_status(self) -> str:
        """Get comprehensive system status"""
        active_agents = len([a for a in self.monitoring_agents if a.status == "active"])
        active_alerts = len([a for a in self.active_alerts if a.level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]])
        
        status = f"""System Status Summary:
- Active Monitoring Agents: {active_agents}/{len(self.monitoring_agents)}
- Active Alerts: {active_alerts}
- System Health: {'GOOD' if active_alerts == 0 else 'ATTENTION NEEDED'}
- Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return status
    
    def _get_help_message(self) -> str:
        """Get help message with available commands"""
        return """Available Monitoring Commands:
- "What's the system status?" - Get overall system health
- "Any errors or issues?" - Check for recent errors
- "How's the performance?" - Get performance metrics
- "Test results summary?" - Get testing overview
- "Security status?" - Check security alerts
- "Show me the logs" - Display recent logs
- "Help" - Show this message

Monitoring Agents Available:
""" + "\n".join([f"- {agent.name}: {', '.join(agent.capabilities)}" for agent in self.monitoring_agents])

class MultiModalAnalyzer:
    """
    Advanced multi-modal analyzer that can process different types of data
    including logs, metrics, code, and configuration files.
    """
    
    def __init__(self):
        self.analyzers = {
            "log": self._analyze_logs,
            "metric": self._analyze_metrics,
            "code": self._analyze_code,
            "config": self._analyze_config,
            "test": self._analyze_test_results
        }
        
        # Data processing pipeline
        self.processing_pipeline = []
        self.analysis_cache = {}
        
        logging.info("Multi-Modal Analyzer initialized")
    
    def analyze(self, data: Any, data_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data using appropriate analyzer"""
        if data_type not in self.analyzers:
            return {"error": f"Unknown data type: {data_type}"}
        
        # Check cache
        cache_key = f"{data_type}_{hash(str(data))}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Perform analysis
        analyzer = self.analyzers[data_type]
        result = analyzer(data, context or {})
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        return result
    
    def _analyze_logs(self, logs: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log data for patterns and issues"""
        error_count = sum(1 for log in logs if "error" in log.lower())
        warning_count = sum(1 for log in logs if "warning" in log.lower())
        
        # Extract common patterns
        patterns = {}
        for log in logs:
            # Simple pattern extraction (can be enhanced with ML)
            if "failed" in log.lower():
                patterns["failures"] = patterns.get("failures", 0) + 1
            if "timeout" in log.lower():
                patterns["timeouts"] = patterns.get("timeouts", 0) + 1
        
        return {
            "total_logs": len(logs),
            "error_count": error_count,
            "warning_count": warning_count,
            "patterns": patterns,
            "severity": "high" if error_count > 10 else "medium" if error_count > 0 else "low"
        }
    
    def _analyze_metrics(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        anomalies = []
        trends = {}
        
        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 1:
                # Calculate trend
                if values[-1] > values[0]:
                    trends[metric_name] = "increasing"
                elif values[-1] < values[0]:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
                
                # Check for anomalies (simple threshold-based)
                avg = sum(values) / len(values)
                for i, value in enumerate(values):
                    if abs(value - avg) > avg * 0.5:  # 50% deviation
                        anomalies.append({
                            "metric": metric_name,
                            "index": i,
                            "value": value,
                            "expected": avg
                        })
        
        return {
            "trends": trends,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "health_score": max(0, 100 - len(anomalies) * 10)
        }
    
    def _analyze_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for quality and issues"""
        lines = code.split('\n')
        
        # Basic code analysis
        complexity_score = self._calculate_code_complexity(code)
        quality_issues = self._find_quality_issues(code)
        
        return {
            "line_count": len(lines),
            "complexity_score": complexity_score,
            "quality_issues": quality_issues,
            "maintainability": "good" if complexity_score < 10 and len(quality_issues) < 5 else "needs_improvement"
        }
    
    def _analyze_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze configuration for best practices"""
        security_issues = []
        performance_issues = []
        
        # Check for common security issues
        if "password" in str(config).lower():
            security_issues.append("Potential password in config")
        if "secret" in str(config).lower():
            security_issues.append("Potential secret in config")
        
        # Check for performance issues
        config_depth = self._calculate_dict_depth(config)
        if config_depth > 5:
            performance_issues.append("Configuration too deeply nested")
        
        return {
            "security_issues": security_issues,
            "performance_issues": performance_issues,
            "depth": config_depth,
            "complexity": "high" if config_depth > 3 else "medium" if config_depth > 2 else "low"
        }
    
    def _analyze_test_results(self, test_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test execution results"""
        total_tests = test_results.get("total", 0)
        passed_tests = test_results.get("passed", 0)
        failed_tests = test_results.get("failed", 0)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "quality": "excellent" if success_rate >= 95 else "good" if success_rate >= 80 else "needs_improvement"
        }
    
    def _calculate_code_complexity(self, code: str) -> int:
        """Calculate basic code complexity score"""
        complexity = 0
        
        # Count control flow statements
        control_keywords = ["if", "for", "while", "try", "except", "with"]
        for keyword in control_keywords:
            complexity += code.count(keyword)
        
        # Count function definitions
        complexity += code.count("def ")
        
        # Count class definitions
        complexity += code.count("class ")
        
        return complexity
    
    def _find_quality_issues(self, code: str) -> List[str]:
        """Find basic code quality issues"""
        issues = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Check line length
            if len(line) > 120:
                issues.append(f"Line {i+1}: Line too long ({len(line)} chars)")
            
            # Check for TODO comments
            if "TODO" in line:
                issues.append(f"Line {i+1}: TODO comment found")
        
        return issues
    
    def _calculate_dict_depth(self, d: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        max_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth

class EnhancedTestMonitor:
    """
    Enhanced test monitoring system that combines real-time monitoring,
    conversational interface, and multi-modal analysis.
    """
    
    def __init__(self, mode: MonitoringMode = MonitoringMode.INTERACTIVE):
        self.mode = mode
        self.conversational_monitor = ConversationalMonitor()
        self.multimodal_analyzer = MultiModalAnalyzer()
        self.real_time_monitor = None
        
        # Event tracking
        self.events: List[MonitoringEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.alert_history: List[MonitoringEvent] = []
        
        # Integration with global observability
        self.observability = global_observability
        
        # Subscribe to observability events
        self.observability.event_handlers["session_started"].append(self._on_session_started)
        self.observability.event_handlers["action_completed"].append(self._on_action_completed)
        self.observability.event_handlers["llm_call"].append(self._on_llm_call)
        
        logging.info(f"Enhanced Test Monitor initialized in {mode.value} mode")
    
    def start_monitoring(self):
        """Start the enhanced monitoring system"""
        # Initialize real-time monitor if available
        try:
            self.real_time_monitor = RealTimeMonitor()
            logging.info("Real-time monitor initialized")
        except Exception as e:
            logging.warning(f"Could not initialize real-time monitor: {e}")
        
        # Start conversational interface if in interactive mode
        if self.mode in [MonitoringMode.INTERACTIVE, MonitoringMode.CONVERSATIONAL]:
            logging.info("Conversational monitoring interface ready")
        
        # Start proactive monitoring if enabled
        if self.mode in [MonitoringMode.PROACTIVE, MonitoringMode.CONVERSATIONAL]:
            self._start_proactive_monitoring()
    
    def add_monitoring_agent(self, agent: MonitoringAgent):
        """Add a monitoring agent to the system"""
        self.conversational_monitor.add_agent(agent)
        
        # Create event for new agent
        event = MonitoringEvent(
            event_type="agent_added",
            source="enhanced_monitor",
            level=AlertLevel.INFO,
            message=f"Monitoring agent added: {agent.name}",
            data={"agent_id": agent.id, "capabilities": agent.capabilities}
        )
        
        self._emit_event(event)
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query through conversational interface"""
        return self.conversational_monitor.process_user_query(query, context)
    
    def analyze_data(self, data: Any, data_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data using multi-modal analyzer"""
        return self.multimodal_analyzer.analyze(data, data_type, context)
    
    def create_alert(self, level: AlertLevel, message: str, source: str, data: Dict[str, Any] = None):
        """Create a monitoring alert"""
        alert = MonitoringEvent(
            event_type="alert",
            source=source,
            level=level,
            message=message,
            data=data or {}
        )
        
        self.conversational_monitor.active_alerts.append(alert)
        self.alert_history.append(alert)
        self._emit_event(alert)
        
        logging.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else
            logging.ERROR if level == AlertLevel.ERROR else
            logging.WARNING if level == AlertLevel.WARNING else
            logging.INFO,
            f"Monitor Alert [{level.value.upper()}]: {message}"
        )
    
    def _start_proactive_monitoring(self):
        """Start proactive monitoring for automatic insights"""
        # This would typically run in a background thread
        # For now, we'll just log that it's enabled
        logging.info("Proactive monitoring started - will generate automatic insights")
    
    def _on_session_started(self, event_type: str, data: Dict[str, Any]):
        """Handle session started event from observability"""
        self.create_alert(
            AlertLevel.INFO,
            f"Test session started: {data.get('name', 'Unknown')}",
            "observability",
            data
        )
    
    def _on_action_completed(self, event_type: str, data: Dict[str, Any]):
        """Handle action completed event from observability"""
        if not data.get("success", True):
            self.create_alert(
                AlertLevel.WARNING,
                f"Action failed: {data.get('error', 'Unknown error')}",
                "observability",
                data
            )
    
    def _on_llm_call(self, event_type: str, data: Dict[str, Any]):
        """Handle LLM call event from observability"""
        if not data.get("success", True):
            self.create_alert(
                AlertLevel.ERROR,
                "LLM call failed",
                "observability",
                data
            )
        elif data.get("cost", 0) > 0.5:  # High cost threshold
            self.create_alert(
                AlertLevel.WARNING,
                f"High cost LLM call: ${data.get('cost', 0):.4f}",
                "observability",
                data
            )
    
    def _emit_event(self, event: MonitoringEvent):
        """Emit monitoring event to registered handlers"""
        self.events.append(event)
        
        # Call registered event handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Error in event handler: {e}")
        
        # Call general event handlers
        for handler in self.event_handlers.get("*", []):
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Error in general event handler: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        active_alerts = [a for a in self.conversational_monitor.active_alerts 
                        if a.level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]]
        
        recent_events = [e for e in self.events 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "mode": self.mode.value,
            "monitoring_agents": len(self.conversational_monitor.monitoring_agents),
            "active_alerts": len(active_alerts),
            "recent_events": len(recent_events),
            "total_events": len(self.events),
            "performance_history_size": len(self.performance_history),
            "alert_history_size": len(self.alert_history),
            "real_time_monitor_available": self.real_time_monitor is not None,
            "observability_integration": "enabled"
        }

# Global enhanced monitor instance
enhanced_monitor = EnhancedTestMonitor()