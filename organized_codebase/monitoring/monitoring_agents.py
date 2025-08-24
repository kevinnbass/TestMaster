"""
Specialized Monitoring Agents
=============================

Specialized monitoring agents for different aspects of the testing system.
Each agent focuses on specific monitoring domains with conversational capabilities.

Author: TestMaster Team
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# UPDATED: Import from new modular observability system
from ..observability.core.event_monitoring import MonitoringAgent, MonitoringEvent, AlertLevel

class PerformanceMonitoringAgent(MonitoringAgent):
    """
    Monitors system and test performance metrics.
    Tracks response times, throughput, resource usage, and performance trends.
    """
    
    def __init__(self):
        super().__init__(
            name="performance_monitor",
            capabilities=[
                "response_time_analysis",
                "throughput_monitoring", 
                "resource_usage_tracking",
                "performance_trend_analysis",
                "bottleneck_detection"
            ]
        )
        
        self.performance_history: List[Dict[str, Any]] = []
        self.thresholds = {
            "response_time_ms": 1000,
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
            "throughput_min": 10  # requests per minute
        }
        
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze performance data and generate events"""
        events = []
        timestamp = datetime.now()
        
        # Store for trend analysis
        self.performance_history.append({
            "timestamp": timestamp,
            "data": data.copy()
        })
        
        # Keep only last hour of data
        cutoff = timestamp - timedelta(hours=1)
        self.performance_history = [
            entry for entry in self.performance_history
            if entry["timestamp"] > cutoff
        ]
        
        # Analyze response times
        if "response_time" in data:
            response_time = data["response_time"]
            if response_time > self.thresholds["response_time_ms"]:
                events.append(MonitoringEvent(
                    event_type="high_response_time",
                    source="performance_monitor",
                    level=AlertLevel.WARNING if response_time < 2000 else AlertLevel.ERROR,
                    message=f"High response time detected: {response_time}ms",
                    data={"response_time": response_time, "threshold": self.thresholds["response_time_ms"]}
                ))
        
        # Analyze resource usage
        if "cpu_usage" in data:
            cpu_usage = data["cpu_usage"]
            if cpu_usage > self.thresholds["cpu_usage_percent"]:
                events.append(MonitoringEvent(
                    event_type="high_cpu_usage",
                    source="performance_monitor",
                    level=AlertLevel.WARNING if cpu_usage < 90 else AlertLevel.CRITICAL,
                    message=f"High CPU usage: {cpu_usage}%",
                    data={"cpu_usage": cpu_usage, "threshold": self.thresholds["cpu_usage_percent"]}
                ))
        
        if "memory_usage" in data:
            memory_usage = data["memory_usage"]
            if memory_usage > self.thresholds["memory_usage_percent"]:
                events.append(MonitoringEvent(
                    event_type="high_memory_usage",
                    source="performance_monitor",
                    level=AlertLevel.WARNING if memory_usage < 95 else AlertLevel.CRITICAL,
                    message=f"High memory usage: {memory_usage}%",
                    data={"memory_usage": memory_usage, "threshold": self.thresholds["memory_usage_percent"]}
                ))
        
        # Analyze performance trends
        trend_events = await self._analyze_trends()
        events.extend(trend_events)
        
        return events
    
    async def _analyze_trends(self) -> List[MonitoringEvent]:
        """Analyze performance trends over time"""
        events = []
        
        if len(self.performance_history) < 10:
            return events  # Need more data for trend analysis
        
        # Analyze response time trend
        response_times = [
            entry["data"].get("response_time", 0)
            for entry in self.performance_history[-10:]
            if "response_time" in entry["data"]
        ]
        
        if len(response_times) >= 5:
            recent_avg = statistics.mean(response_times[-5:])
            older_avg = statistics.mean(response_times[:-5])
            
            if recent_avg > older_avg * 1.5:  # 50% increase
                events.append(MonitoringEvent(
                    event_type="performance_degradation",
                    source="performance_monitor",
                    level=AlertLevel.WARNING,
                    message=f"Performance degradation detected: {recent_avg:.1f}ms vs {older_avg:.1f}ms",
                    data={
                        "recent_average": recent_avg,
                        "previous_average": older_avg,
                        "degradation_percent": ((recent_avg - older_avg) / older_avg) * 100
                    }
                ))
        
        return events
    
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to performance-related queries"""
        query_lower = query.lower()
        
        if "current" in query_lower and "performance" in query_lower:
            return self._get_current_performance_status()
        elif "trend" in query_lower or "history" in query_lower:
            return self._get_performance_trends()
        elif "bottleneck" in query_lower:
            return self._identify_bottlenecks()
        elif "resource" in query_lower:
            return self._get_resource_usage()
        else:
            return self._get_performance_summary()
    
    def _get_current_performance_status(self) -> str:
        """Get current performance status"""
        if not self.performance_history:
            return "No performance data available yet."
        
        latest = self.performance_history[-1]["data"]
        
        status_parts = ["Current Performance Status:"]
        
        if "response_time" in latest:
            rt = latest["response_time"]
            status = "🟢 Good" if rt < 500 else "🟡 Fair" if rt < 1000 else "🔴 Poor"
            status_parts.append(f"Response Time: {rt}ms ({status})")
        
        if "cpu_usage" in latest:
            cpu = latest["cpu_usage"]
            status = "🟢 Good" if cpu < 60 else "🟡 Fair" if cpu < 80 else "🔴 High"
            status_parts.append(f"CPU Usage: {cpu}% ({status})")
        
        if "memory_usage" in latest:
            mem = latest["memory_usage"]
            status = "🟢 Good" if mem < 70 else "🟡 Fair" if mem < 85 else "🔴 High"
            status_parts.append(f"Memory Usage: {mem}% ({status})")
        
        return "\n".join(status_parts)
    
    def _get_performance_trends(self) -> str:
        """Get performance trend analysis"""
        if len(self.performance_history) < 5:
            return "Insufficient data for trend analysis."
        
        # Calculate trends for last 10 data points
        recent_data = self.performance_history[-10:]
        
        trends = []
        
        # Response time trend
        response_times = [entry["data"].get("response_time") for entry in recent_data]
        response_times = [rt for rt in response_times if rt is not None]
        
        if len(response_times) >= 3:
            if response_times[-1] > response_times[0]:
                trend = "📈 Increasing"
            elif response_times[-1] < response_times[0]:
                trend = "📉 Decreasing"
            else:
                trend = "➡️ Stable"
            
            avg_rt = statistics.mean(response_times)
            trends.append(f"Response Time: {avg_rt:.1f}ms average ({trend})")
        
        if trends:
            return "Performance Trends (last 10 measurements):\n" + "\n".join(trends)
        else:
            return "No trend data available for key metrics."
    
    def _identify_bottlenecks(self) -> str:
        """Identify potential performance bottlenecks"""
        if not self.performance_history:
            return "No data available for bottleneck analysis."
        
        latest = self.performance_history[-1]["data"]
        bottlenecks = []
        
        if latest.get("cpu_usage", 0) > 80:
            bottlenecks.append("🔴 CPU bottleneck detected (high CPU usage)")
        
        if latest.get("memory_usage", 0) > 85:
            bottlenecks.append("🔴 Memory bottleneck detected (high memory usage)")
        
        if latest.get("response_time", 0) > 1000:
            bottlenecks.append("🔴 Response time bottleneck detected")
        
        if latest.get("disk_io", 0) > 80:
            bottlenecks.append("🔴 Disk I/O bottleneck detected")
        
        if bottlenecks:
            return "Potential Bottlenecks:\n" + "\n".join(bottlenecks)
        else:
            return "🟢 No significant bottlenecks detected."
    
    def _get_resource_usage(self) -> str:
        """Get current resource usage summary"""
        if not self.performance_history:
            return "No resource usage data available."
        
        latest = self.performance_history[-1]["data"]
        
        resources = []
        if "cpu_usage" in latest:
            resources.append(f"CPU: {latest['cpu_usage']}%")
        if "memory_usage" in latest:
            resources.append(f"Memory: {latest['memory_usage']}%")
        if "disk_usage" in latest:
            resources.append(f"Disk: {latest['disk_usage']}%")
        if "network_usage" in latest:
            resources.append(f"Network: {latest['network_usage']}%")
        
        return "Current Resource Usage:\n" + "\n".join(resources) if resources else "No resource data available."
    
    def _get_performance_summary(self) -> str:
        """Get overall performance summary"""
        return """Performance Monitoring Capabilities:

🔍 Current Metrics:
- Response time monitoring
- CPU and memory usage tracking
- Throughput analysis
- Resource utilization

📊 Trend Analysis:
- Performance degradation detection
- Historical comparison
- Bottleneck identification

💡 Ask me about:
- "What's the current performance status?"
- "Show me performance trends"
- "Are there any bottlenecks?"
- "What's the resource usage?"
"""

class QualityMonitoringAgent(MonitoringAgent):
    """
    Monitors test quality metrics and code coverage.
    Tracks test results, coverage percentages, and quality trends.
    """
    
    def __init__(self):
        super().__init__(
            name="quality_monitor",
            capabilities=[
                "test_result_analysis",
                "coverage_monitoring",
                "quality_score_tracking",
                "test_reliability_analysis",
                "quality_trend_detection"
            ]
        )
        
        self.quality_history: List[Dict[str, Any]] = []
        self.thresholds = {
            "min_coverage_percent": 80,
            "max_failure_rate": 5,
            "min_quality_score": 75
        }
    
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze quality data and generate events"""
        events = []
        
        # Store quality data
        self.quality_history.append({
            "timestamp": datetime.now(),
            "data": data.copy()
        })
        
        # Analyze test coverage
        if "test_coverage" in data:
            coverage = data["test_coverage"]
            if coverage < self.thresholds["min_coverage_percent"]:
                events.append(MonitoringEvent(
                    event_type="low_test_coverage",
                    source="quality_monitor",
                    level=AlertLevel.WARNING if coverage > 60 else AlertLevel.ERROR,
                    message=f"Test coverage below threshold: {coverage}%",
                    data={"coverage": coverage, "threshold": self.thresholds["min_coverage_percent"]}
                ))
        
        # Analyze test failure rate
        if "test_results" in data:
            results = data["test_results"]
            total = results.get("total", 0)
            failed = results.get("failed", 0)
            
            if total > 0:
                failure_rate = (failed / total) * 100
                if failure_rate > self.thresholds["max_failure_rate"]:
                    events.append(MonitoringEvent(
                        event_type="high_test_failure_rate",
                        source="quality_monitor",
                        level=AlertLevel.ERROR if failure_rate > 20 else AlertLevel.WARNING,
                        message=f"High test failure rate: {failure_rate:.1f}%",
                        data={"failure_rate": failure_rate, "failed": failed, "total": total}
                    ))
        
        return events
    
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to quality-related queries"""
        query_lower = query.lower()
        
        if "coverage" in query_lower:
            return self._get_coverage_status()
        elif "test" in query_lower and ("result" in query_lower or "status" in query_lower):
            return self._get_test_results()
        elif "quality" in query_lower and "score" in query_lower:
            return self._get_quality_score()
        else:
            return self._get_quality_summary()
    
    def _get_coverage_status(self) -> str:
        """Get test coverage status"""
        if not self.quality_history:
            return "No coverage data available."
        
        latest = self.quality_history[-1]["data"]
        coverage = latest.get("test_coverage")
        
        if coverage is not None:
            status = "🟢 Excellent" if coverage >= 90 else "🟡 Good" if coverage >= 80 else "🔴 Needs Improvement"
            return f"Test Coverage: {coverage}% ({status})\nTarget: {self.thresholds['min_coverage_percent']}%"
        else:
            return "No coverage data in latest metrics."
    
    def _get_test_results(self) -> str:
        """Get latest test results"""
        if not self.quality_history:
            return "No test results available."
        
        latest = self.quality_history[-1]["data"]
        results = latest.get("test_results")
        
        if results:
            total = results.get("total", 0)
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            skipped = results.get("skipped", 0)
            
            if total > 0:
                pass_rate = (passed / total) * 100
                status = "🟢 Excellent" if pass_rate >= 95 else "🟡 Good" if pass_rate >= 90 else "🔴 Needs Attention"
                
                return f"""Latest Test Results:
Total Tests: {total}
Passed: {passed} ({pass_rate:.1f}%)
Failed: {failed}
Skipped: {skipped}
Status: {status}"""
            else:
                return "No tests have been run yet."
        else:
            return "No test result data available."
    
    def _get_quality_score(self) -> str:
        """Get overall quality score"""
        return "Quality score calculation not yet implemented."
    
    def _get_quality_summary(self) -> str:
        """Get quality monitoring summary"""
        return """Quality Monitoring Capabilities:

📋 Test Analysis:
- Test result tracking
- Failure rate monitoring
- Test reliability analysis

📊 Coverage Monitoring:
- Line coverage tracking
- Branch coverage analysis
- Coverage trend detection

🎯 Quality Metrics:
- Overall quality score
- Code quality assessment
- Quality trend analysis

💡 Ask me about:
- "What's the test coverage?"
- "Show me test results"
- "What's the quality score?"
"""

class SecurityMonitoringAgent(MonitoringAgent):
    """
    Monitors security-related metrics and vulnerabilities.
    Tracks security scans, vulnerability reports, and security trends.
    """
    
    def __init__(self):
        super().__init__(
            name="security_monitor",
            capabilities=[
                "vulnerability_scanning",
                "security_alert_monitoring",
                "compliance_checking",
                "threat_detection",
                "security_trend_analysis"
            ]
        )
        
        self.security_history: List[Dict[str, Any]] = []
        self.severity_thresholds = {
            "critical": 0,  # Any critical vulnerability is an alert
            "high": 5,
            "medium": 20
        }
    
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze security data and generate events"""
        events = []
        
        # Store security data
        self.security_history.append({
            "timestamp": datetime.now(),
            "data": data.copy()
        })
        
        # Analyze vulnerability counts
        if "vulnerabilities" in data:
            vulns = data["vulnerabilities"]
            
            for severity, count in vulns.items():
                threshold = self.severity_thresholds.get(severity, float('inf'))
                
                if count > threshold:
                    level = AlertLevel.CRITICAL if severity == "critical" else \
                           AlertLevel.ERROR if severity == "high" else \
                           AlertLevel.WARNING
                    
                    events.append(MonitoringEvent(
                        event_type=f"{severity}_vulnerabilities",
                        source="security_monitor",
                        level=level,
                        message=f"{severity.title()} vulnerabilities detected: {count}",
                        data={"severity": severity, "count": count, "threshold": threshold}
                    ))
        
        return events
    
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to security-related queries"""
        query_lower = query.lower()
        
        if "vulnerability" in query_lower or "vuln" in query_lower:
            return self._get_vulnerability_status()
        elif "compliance" in query_lower:
            return self._get_compliance_status()
        elif "threat" in query_lower:
            return self._get_threat_status()
        else:
            return self._get_security_summary()
    
    def _get_vulnerability_status(self) -> str:
        """Get vulnerability status"""
        if not self.security_history:
            return "No security scan data available."
        
        latest = self.security_history[-1]["data"]
        vulns = latest.get("vulnerabilities", {})
        
        if vulns:
            total = sum(vulns.values())
            critical = vulns.get("critical", 0)
            high = vulns.get("high", 0)
            medium = vulns.get("medium", 0)
            low = vulns.get("low", 0)
            
            status = "🔴 Critical" if critical > 0 else \
                    "🟡 Moderate" if high > 0 else \
                    "🟢 Good"
            
            return f"""Vulnerability Status: {status}
Total Vulnerabilities: {total}
Critical: {critical}
High: {high}
Medium: {medium}
Low: {low}"""
        else:
            return "No vulnerability data available."
    
    def _get_compliance_status(self) -> str:
        """Get compliance status"""
        return "Compliance monitoring not yet implemented."
    
    def _get_threat_status(self) -> str:
        """Get threat detection status"""
        return "Threat detection monitoring not yet implemented."
    
    def _get_security_summary(self) -> str:
        """Get security monitoring summary"""
        return """Security Monitoring Capabilities:

🔒 Vulnerability Management:
- Vulnerability scanning
- Severity assessment
- Remediation tracking

📋 Compliance Monitoring:
- Standards compliance checking
- Policy enforcement
- Audit trail maintenance

🛡️ Threat Detection:
- Security event monitoring
- Anomaly detection
- Incident response

💡 Ask me about:
- "What vulnerabilities were found?"
- "What's the compliance status?"
- "Any security threats detected?"
"""

class CollaborationMonitoringAgent(MonitoringAgent):
    """
    Monitors collaboration and communication between agents.
    Tracks message patterns, response times, and collaboration effectiveness.
    """
    
    def __init__(self):
        super().__init__(
            name="collaboration_monitor",
            capabilities=[
                "message_pattern_analysis",
                "response_time_tracking",
                "collaboration_scoring",
                "communication_efficiency",
                "team_coordination_analysis"
            ]
        )
        
        self.collaboration_history: List[Dict[str, Any]] = []
        self.thresholds = {
            "max_response_time": 5000,  # 5 seconds
            "min_collaboration_score": 70
        }
    
    async def analyze(self, data: Dict[str, Any]) -> List[MonitoringEvent]:
        """Analyze collaboration data and generate events"""
        events = []
        
        # Store collaboration data
        self.collaboration_history.append({
            "timestamp": datetime.now(),
            "data": data.copy()
        })
        
        # Analyze agent communication patterns
        if "agent_communications" in data:
            comms = data["agent_communications"]
            
            # Check for communication delays
            for comm in comms:
                response_time = comm.get("response_time", 0)
                if response_time > self.thresholds["max_response_time"]:
                    events.append(MonitoringEvent(
                        event_type="slow_agent_communication",
                        source="collaboration_monitor",
                        level=AlertLevel.WARNING,
                        message=f"Slow agent communication: {response_time}ms",
                        data={"response_time": response_time, "agents": comm.get("agents", [])}
                    ))
        
        return events
    
    async def respond_to_query(self, query: str, context: Dict[str, Any]) -> str:
        """Respond to collaboration-related queries"""
        query_lower = query.lower()
        
        if "communication" in query_lower:
            return self._get_communication_status()
        elif "collaboration" in query_lower and "score" in query_lower:
            return self._get_collaboration_score()
        elif "agent" in query_lower and ("interaction" in query_lower or "activity" in query_lower):
            return self._get_agent_interactions()
        else:
            return self._get_collaboration_summary()
    
    def _get_communication_status(self) -> str:
        """Get agent communication status"""
        return "Agent communication analysis not yet fully implemented."
    
    def _get_collaboration_score(self) -> str:
        """Get collaboration effectiveness score"""
        return "Collaboration scoring not yet implemented."
    
    def _get_agent_interactions(self) -> str:
        """Get agent interaction patterns"""
        return "Agent interaction analysis not yet implemented."
    
    def _get_collaboration_summary(self) -> str:
        """Get collaboration monitoring summary"""
        return """Collaboration Monitoring Capabilities:

🤝 Communication Analysis:
- Message pattern tracking
- Response time monitoring
- Communication efficiency

📊 Collaboration Scoring:
- Team effectiveness metrics
- Coordination assessment
- Collaboration trends

🔗 Interaction Patterns:
- Agent-to-agent communication
- Workflow coordination
- Dependency tracking

💡 Ask me about:
- "How is agent communication?"
- "What's the collaboration score?"
- "Show me agent interactions"
"""

# Export all monitoring agents
__all__ = [
    'PerformanceMonitoringAgent',
    'QualityMonitoringAgent',
    'SecurityMonitoringAgent',
    'CollaborationMonitoringAgent'
]