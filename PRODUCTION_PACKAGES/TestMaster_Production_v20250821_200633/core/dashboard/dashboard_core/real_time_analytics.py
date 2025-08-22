"""
Real-Time Analytics Integration
================================
Enhanced analytics for dashboard.
"""

from enhance_analytics import RealTimeAnalyticsCollector

# Global analytics instance
analytics_collector = RealTimeAnalyticsCollector()

def get_analytics_collector():
    """Get the global analytics collector instance."""
    return analytics_collector

def start_analytics():
    """Start analytics collection."""
    analytics_collector.start_collection()

def stop_analytics():
    """Stop analytics collection."""
    analytics_collector.stop_collection()

def get_real_time_metrics():
    """Get real-time metrics for dashboard."""
    return analytics_collector.get_real_time_data()

def get_test_metrics():
    """Get test execution metrics."""
    return analytics_collector.get_test_analytics()

# Auto-start on import
start_analytics()
