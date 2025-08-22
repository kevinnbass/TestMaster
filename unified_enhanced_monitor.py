#!/usr/bin/env python3
"""
Unified Enhanced Monitor
Agent B Hours 100-110: Advanced User Experience & Enhancement

Unified system integrating enhanced dashboard, advanced alerts, and query optimization.
"""

import json
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

# Import our enhanced components
from enhanced_dashboard import EnhancedDashboardServer
from advanced_alert_system import AdvancedAlertSystem
from query_optimization_engine import QueryOptimizationEngine

class UnifiedEnhancedMonitor:
    """Unified enhanced monitoring system with all advanced features"""
    
    def __init__(self, config_file: str = "db_monitor_config.json"):
        self.config_file = Path(config_file)
        self.running = False
        
        # Initialize components
        print("[INIT] Starting Unified Enhanced Monitor...")
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize enhanced components
        self.dashboard = EnhancedDashboardServer(config_file)
        self.alert_system = AdvancedAlertSystem()
        
        # Initialize query optimizer for each database
        self.query_optimizers = {}
        for db_name, db_config in self.config.get("databases", {}).items():
            if db_config.get("enabled", False):
                db_path = db_config.get("path")
                if db_path:
                    self.query_optimizers[db_name] = QueryOptimizationEngine(db_path)
        
        # Enhanced metrics storage
        self.enhanced_metrics_history = []
        self.system_insights = {}
        
        print(f"[OK] Initialized with {len(self.query_optimizers)} database optimizers")
        print(f"[OK] Alert system loaded with {len(self.alert_system.alert_rules)} rules")
    
    def load_config(self):
        """Load monitoring configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"databases": {}}
    
    def collect_unified_metrics(self):
        """Collect unified metrics from all components"""
        # Collect base metrics from dashboard
        metrics = self.dashboard.collect_enhanced_metrics()
        
        # Add alert system metrics
        alert_stats = self.alert_system.get_alert_statistics()
        metrics["alerts"] = {
            "active_count": alert_stats["active_alerts_count"],
            "total_generated": alert_stats["total_alerts"],
            "critical_count": alert_stats["critical_alerts"],
            "resolution_rate": (alert_stats["resolved_alerts"] / max(alert_stats["total_alerts"], 1)) * 100
        }
        
        # Add query optimization insights
        optimization_insights = {}
        for db_name, optimizer in self.query_optimizers.items():
            report = optimizer.get_optimization_report()
            optimization_insights[db_name] = {
                "total_profiles": report["query_profiles"]["total_queries"],
                "slow_queries": report["query_profiles"]["slow_queries"],
                "optimization_opportunities": report["query_profiles"]["slow_queries"]
            }
        
        metrics["query_optimization"] = optimization_insights
        
        # Check metrics against alert system
        self.alert_system.check_metrics(metrics)
        
        # Store enhanced metrics
        self.enhanced_metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.enhanced_metrics_history) > 1000:
            self.enhanced_metrics_history = self.enhanced_metrics_history[-1000:]
        
        return metrics
    
    def generate_system_insights(self):
        """Generate comprehensive system insights"""
        if not self.enhanced_metrics_history:
            return {}
        
        latest_metrics = self.enhanced_metrics_history[-1]
        
        insights = {
            "system_health_score": self.calculate_unified_health_score(),
            "performance_trends": self.analyze_performance_trends(),
            "optimization_opportunities": self.identify_optimization_opportunities(),
            "alert_patterns": self.analyze_alert_patterns(),
            "capacity_predictions": self.predict_capacity_needs(),
            "recommendations": self.generate_unified_recommendations()
        }
        
        return insights
    
    def calculate_unified_health_score(self):
        """Calculate unified system health score"""
        if not self.enhanced_metrics_history:
            return 50
        
        latest = self.enhanced_metrics_history[-1]
        score = 100
        
        # System performance factors
        system = latest.get("system", {})
        score -= max(0, (system.get("cpu_percent", 0) - 70) * 2)  # -2 points per % over 70%
        score -= max(0, (system.get("memory_percent", 0) - 80) * 3)  # -3 points per % over 80%
        
        # Alert factors
        alert_info = latest.get("alerts", {})
        score -= alert_info.get("critical_count", 0) * 15  # -15 points per critical alert
        score -= alert_info.get("active_count", 0) * 5    # -5 points per active alert
        
        # Query performance factors
        query_info = latest.get("query_optimization", {})
        total_slow_queries = sum(db.get("slow_queries", 0) for db in query_info.values())
        score -= total_slow_queries * 2  # -2 points per slow query
        
        return max(0, min(100, score))
    
    def analyze_performance_trends(self):
        """Analyze performance trends over time"""
        if len(self.enhanced_metrics_history) < 2:
            return {}
        
        recent = self.enhanced_metrics_history[-5:]  # Last 5 measurements
        trends = {}
        
        # CPU trend
        cpu_values = [m["system"]["cpu_percent"] for m in recent if "system" in m]
        if len(cpu_values) >= 2:
            cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
            trends["cpu"] = {
                "direction": cpu_trend,
                "change": cpu_values[-1] - cpu_values[0],
                "current": cpu_values[-1]
            }
        
        # Memory trend
        memory_values = [m["system"]["memory_percent"] for m in recent if "system" in m]
        if len(memory_values) >= 2:
            memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            trends["memory"] = {
                "direction": memory_trend,
                "change": memory_values[-1] - memory_values[0],
                "current": memory_values[-1]
            }
        
        # Database growth trend
        db_sizes = [m["totals"]["database_size_mb"] for m in recent if "totals" in m]
        if len(db_sizes) >= 2:
            db_trend = "growing" if db_sizes[-1] > db_sizes[0] else "stable"
            trends["database_growth"] = {
                "direction": db_trend,
                "change_mb": db_sizes[-1] - db_sizes[0],
                "current_mb": db_sizes[-1]
            }
        
        return trends
    
    def identify_optimization_opportunities(self):
        """Identify system optimization opportunities"""
        opportunities = []
        
        if not self.enhanced_metrics_history:
            return opportunities
        
        latest = self.enhanced_metrics_history[-1]
        
        # High resource usage opportunities
        system = latest.get("system", {})
        if system.get("cpu_percent", 0) > 80:
            opportunities.append({
                "type": "cpu_optimization",
                "priority": "high",
                "description": "CPU usage is high - investigate running processes",
                "impact": "performance"
            })
        
        if system.get("memory_percent", 0) > 85:
            opportunities.append({
                "type": "memory_optimization",
                "priority": "high",
                "description": "Memory usage is high - consider memory cleanup or upgrade",
                "impact": "stability"
            })
        
        # Query optimization opportunities
        query_info = latest.get("query_optimization", {})
        for db_name, db_info in query_info.items():
            if db_info.get("slow_queries", 0) > 0:
                opportunities.append({
                    "type": "query_optimization",
                    "priority": "medium",
                    "description": f"Database {db_name} has {db_info['slow_queries']} slow queries",
                    "impact": "performance",
                    "database": db_name
                })
        
        # Alert pattern opportunities
        alert_info = latest.get("alerts", {})
        if alert_info.get("active_count", 0) > 10:
            opportunities.append({
                "type": "alert_tuning",
                "priority": "medium", 
                "description": "Many active alerts - consider adjusting thresholds",
                "impact": "monitoring"
            })
        
        return opportunities
    
    def analyze_alert_patterns(self):
        """Analyze alert patterns and trends"""
        alert_history = self.alert_system.get_alert_history(24)  # Last 24 hours
        
        if not alert_history:
            return {"message": "No recent alerts to analyze"}
        
        # Alert frequency by hour
        hourly_counts = [0] * 24
        for alert in alert_history:
            hour = alert.timestamp.hour
            hourly_counts[hour] += 1
        
        # Most common alert types
        type_counts = {}
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        
        for alert in alert_history:
            type_counts[alert.type] = type_counts.get(alert.type, 0) + 1
            severity_counts[alert.severity] += 1
        
        return {
            "total_alerts_24h": len(alert_history),
            "peak_hour": hourly_counts.index(max(hourly_counts)),
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "none",
            "severity_distribution": severity_counts,
            "alert_frequency": len(alert_history) / 24  # Alerts per hour
        }
    
    def predict_capacity_needs(self):
        """Predict future capacity needs based on trends"""
        if len(self.enhanced_metrics_history) < 5:
            return {"message": "Insufficient data for predictions"}
        
        recent = self.enhanced_metrics_history[-10:]  # Last 10 measurements
        predictions = {}
        
        # Memory usage prediction
        memory_values = [m["system"]["memory_used_gb"] for m in recent if "system" in m]
        if len(memory_values) >= 5:
            # Simple linear trend
            avg_change = (memory_values[-1] - memory_values[0]) / len(memory_values)
            predicted_30_days = memory_values[-1] + (avg_change * 30 * 24 * 2)  # Assuming 2 measurements per hour
            
            predictions["memory_gb_30_days"] = max(0, predicted_30_days)
        
        # Database size prediction
        db_sizes = [m["totals"]["database_size_mb"] for m in recent if "totals" in m]
        if len(db_sizes) >= 5:
            avg_growth = (db_sizes[-1] - db_sizes[0]) / len(db_sizes)
            predicted_db_size = db_sizes[-1] + (avg_growth * 30 * 24 * 2)
            
            predictions["database_mb_30_days"] = max(0, predicted_db_size)
        
        return predictions
    
    def generate_unified_recommendations(self):
        """Generate unified system recommendations"""
        recommendations = []
        
        if not self.enhanced_metrics_history:
            return ["Start monitoring to generate recommendations"]
        
        latest = self.enhanced_metrics_history[-1]
        insights = self.system_insights
        
        # Performance recommendations
        system = latest.get("system", {})
        if system.get("cpu_percent", 0) > 80:
            recommendations.append("High CPU usage detected - consider identifying and optimizing resource-intensive processes")
        
        if system.get("memory_percent", 0) > 85:
            recommendations.append("High memory usage detected - consider restarting services or adding more RAM")
        
        # Query optimization recommendations
        query_info = latest.get("query_optimization", {})
        slow_query_count = sum(db.get("slow_queries", 0) for db in query_info.values())
        if slow_query_count > 0:
            recommendations.append(f"Found {slow_query_count} slow queries - run query optimization analysis")
        
        # Alert system recommendations
        alert_info = latest.get("alerts", {})
        if alert_info.get("active_count", 0) > 5:
            recommendations.append("Multiple active alerts - review system health and alert thresholds")
        
        # Capacity planning recommendations
        predictions = self.predict_capacity_needs()
        if "memory_gb_30_days" in predictions and predictions["memory_gb_30_days"] > 60:
            recommendations.append("Memory usage trending upward - consider capacity planning")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def get_unified_dashboard_data(self):
        """Get comprehensive data for unified dashboard"""
        # Collect current metrics
        current_metrics = self.collect_unified_metrics()
        
        # Generate insights
        self.system_insights = self.generate_system_insights()
        
        # Get alert information
        active_alerts = self.alert_system.get_active_alerts()
        recent_alerts = self.alert_system.get_alert_history(1)  # Last hour
        
        # Get optimization opportunities
        optimization_opportunities = []
        for db_name, optimizer in self.query_optimizers.items():
            report = optimizer.get_optimization_report()
            if report["query_profiles"]["slow_queries"] > 0:
                optimization_opportunities.append({
                    "database": db_name,
                    "slow_queries": report["query_profiles"]["slow_queries"],
                    "recommendations": report["recommendations"]
                })
        
        return {
            "current_metrics": current_metrics,
            "system_insights": self.system_insights,
            "alerts": {
                "active": [{"id": a.id, "title": a.title, "message": a.message, 
                           "severity": a.severity, "timestamp": a.timestamp.isoformat()} 
                          for a in active_alerts],
                "recent_count": len(recent_alerts),
                "alert_patterns": self.analyze_alert_patterns()
            },
            "optimization": {
                "opportunities": optimization_opportunities,
                "total_databases": len(self.query_optimizers),
                "total_slow_queries": sum(len(opp.get("slow_queries", [])) for opp in optimization_opportunities)
            },
            "recommendations": self.system_insights.get("recommendations", [])
        }
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        print("[OK] Starting unified monitoring loop...")
        
        while self.running:
            try:
                # Collect metrics
                self.collect_unified_metrics()
                
                # Update insights
                self.system_insights = self.generate_system_insights()
                
                # Sleep for interval
                time.sleep(30)  # 30-second intervals
                
            except Exception as e:
                print(f"[ERROR] Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_monitoring(self):
        """Start the unified monitoring system"""
        self.running = True
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        print("[OK] Unified Enhanced Monitor started")
        print("[OK] Features: Enhanced dashboard, advanced alerts, query optimization")
    
    def stop_monitoring(self):
        """Stop the unified monitoring system"""
        self.running = False
        print("[OK] Unified Enhanced Monitor stopped")

def main():
    """Main function to run unified enhanced monitor"""
    monitor = UnifiedEnhancedMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        print("\n[OK] Unified Enhanced Monitor is running")
        print("[OK] Enhanced features active:")
        print("     - Real-time metrics collection")
        print("     - Advanced alert system")
        print("     - Query optimization analysis")
        print("     - Performance trend analysis")
        print("     - Predictive capacity planning")
        print("     - Unified dashboard with insights")
        print("\n[OK] Access enhanced dashboard at http://localhost:8080")
        print("[OK] Press Ctrl+C to stop")
        
        # Keep main thread running
        while True:
            time.sleep(10)
            
            # Display periodic status
            data = monitor.get_unified_dashboard_data()
            health_score = data["system_insights"].get("system_health_score", 0)
            active_alerts = len(data["alerts"]["active"])
            
            print(f"\r[STATUS] Health: {health_score}/100 | Active Alerts: {active_alerts}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\n[OK] Shutting down Unified Enhanced Monitor...")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()