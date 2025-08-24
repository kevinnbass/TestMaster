#!/usr/bin/env python3
"""
API Usage Tracking System
=========================

Comprehensive tracking of all API calls, models used, costs, and usage patterns
for the TestMaster dashboard system. This ensures we monitor LLM API costs
before running AI-powered analysis tools.

Author: Multi-Agent Coordination System
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APICall:
    """Individual API call record"""
    timestamp: str
    endpoint: str
    model: str
    token_count: int
    estimated_cost: float
    purpose: str
    response_time: float
    success: bool
    agent: str  # alpha, beta, gamma
    request_size: int
    response_size: int

@dataclass
class APIBudget:
    """API usage budget configuration"""
    daily_limit: float = 10.0  # $10 daily limit
    hourly_limit: float = 2.0   # $2 hourly limit
    call_limit: int = 1000      # 1000 calls per day
    warning_threshold: float = 0.8  # 80% warning

class APIUsageTracker:
    """Comprehensive API usage tracking and budgeting system"""
    
    def __init__(self):
        self.api_calls = deque(maxlen=10000)  # Store last 10k calls
        self.budget = APIBudget()
        self.current_costs = {
            'today': 0.0,
            'this_hour': 0.0,
            'this_session': 0.0
        }
        self.call_counts = {
            'today': 0,
            'this_hour': 0,
            'this_session': 0
        }
        self.model_pricing = {
            # OpenAI pricing (per 1K tokens)
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            # Add other models as needed
        }
        self.alerts = deque(maxlen=100)
        self._lock = threading.RLock()
        
        # Load existing data
        self.load_usage_data()
        
        logger.info("API Usage Tracker initialized")
    
    def track_api_call(self, endpoint: str, model: str, token_count: int, 
                       purpose: str, agent: str, response_time: float = 0.0,
                       success: bool = True, request_size: int = 0, 
                       response_size: int = 0) -> Dict[str, Any]:
        """Track an API call and check budget limits"""
        
        with self._lock:
            # Calculate estimated cost
            estimated_cost = self.calculate_cost(model, token_count)
            
            # Create API call record
            api_call = APICall(
                timestamp=datetime.now().isoformat(),
                endpoint=endpoint,
                model=model,
                token_count=token_count,
                estimated_cost=estimated_cost,
                purpose=purpose,
                response_time=response_time,
                success=success,
                agent=agent,
                request_size=request_size,
                response_size=response_size
            )
            
            # Add to tracking
            self.api_calls.append(api_call)
            
            # Update current costs and counts
            self.current_costs['this_session'] += estimated_cost
            self.call_counts['this_session'] += 1
            
            # Update daily/hourly costs (simplified - in production would use proper time windows)
            self.current_costs['today'] += estimated_cost
            self.current_costs['this_hour'] += estimated_cost
            self.call_counts['today'] += 1
            self.call_counts['this_hour'] += 1
            
            # Check budget limits
            budget_status = self.check_budget_limits()
            
            # Log the call
            logger.info(f"API Call tracked: {agent} | {model} | {token_count} tokens | ${estimated_cost:.4f}")
            
            # Save data periodically
            if len(self.api_calls) % 10 == 0:
                self.save_usage_data()
            
            return {
                'call_id': len(self.api_calls),
                'estimated_cost': estimated_cost,
                'budget_status': budget_status,
                'remaining_budget': self.get_remaining_budget(),
                'total_calls_today': self.call_counts['today'],
                'total_cost_today': self.current_costs['today']
            }
    
    def calculate_cost(self, model: str, token_count: int) -> float:
        """Calculate estimated cost for API call"""
        if model not in self.model_pricing:
            # Default pricing for unknown models
            return (token_count / 1000) * 0.01
        
        pricing = self.model_pricing[model]
        # Simplified: assume equal input/output tokens
        input_tokens = token_count // 2
        output_tokens = token_count // 2
        
        cost = (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']
        return cost
    
    def check_budget_limits(self) -> Dict[str, Any]:
        """Check if we're approaching or exceeding budget limits"""
        status = {
            'daily_status': 'ok',
            'hourly_status': 'ok',
            'call_limit_status': 'ok',
            'warnings': []
        }
        
        # Check daily budget
        daily_usage = self.current_costs['today'] / self.budget.daily_limit
        if daily_usage >= 1.0:
            status['daily_status'] = 'exceeded'
            status['warnings'].append(f"Daily budget exceeded: ${self.current_costs['today']:.2f} / ${self.budget.daily_limit:.2f}")
        elif daily_usage >= self.budget.warning_threshold:
            status['daily_status'] = 'warning'
            status['warnings'].append(f"Daily budget warning: {daily_usage:.1%} used")
        
        # Check hourly budget
        hourly_usage = self.current_costs['this_hour'] / self.budget.hourly_limit
        if hourly_usage >= 1.0:
            status['hourly_status'] = 'exceeded'
            status['warnings'].append(f"Hourly budget exceeded: ${self.current_costs['this_hour']:.2f} / ${self.budget.hourly_limit:.2f}")
        elif hourly_usage >= self.budget.warning_threshold:
            status['hourly_status'] = 'warning'
            status['warnings'].append(f"Hourly budget warning: {hourly_usage:.1%} used")
        
        # Check call limits
        call_usage = self.call_counts['today'] / self.budget.call_limit
        if call_usage >= 1.0:
            status['call_limit_status'] = 'exceeded'
            status['warnings'].append(f"Daily call limit exceeded: {self.call_counts['today']} / {self.budget.call_limit}")
        elif call_usage >= self.budget.warning_threshold:
            status['call_limit_status'] = 'warning'
            status['warnings'].append(f"Daily call limit warning: {call_usage:.1%} used")
        
        # Add alerts for warnings/exceeded limits
        for warning in status['warnings']:
            self.add_alert('budget', warning, 'warning' if 'warning' in warning else 'critical')
        
        return status
    
    def add_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Add a budget alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)
        logger.warning(f"Budget Alert: {message}")
    
    def get_remaining_budget(self) -> Dict[str, Any]:
        """Get remaining budget information"""
        return {
            'daily_remaining': max(0, self.budget.daily_limit - self.current_costs['today']),
            'hourly_remaining': max(0, self.budget.hourly_limit - self.current_costs['this_hour']),
            'calls_remaining': max(0, self.budget.call_limit - self.call_counts['today']),
            'daily_percentage_used': (self.current_costs['today'] / self.budget.daily_limit) * 100,
            'hourly_percentage_used': (self.current_costs['this_hour'] / self.budget.hourly_limit) * 100,
            'calls_percentage_used': (self.call_counts['today'] / self.budget.call_limit) * 100
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        if not self.api_calls:
            return {'total_calls': 0, 'total_cost': 0}
        
        # Calculate statistics
        total_calls = len(self.api_calls)
        total_cost = sum(call.estimated_cost for call in self.api_calls)
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0
        
        # Agent breakdown
        agent_stats = defaultdict(lambda: {'calls': 0, 'cost': 0.0, 'tokens': 0})
        model_stats = defaultdict(lambda: {'calls': 0, 'cost': 0.0, 'tokens': 0})
        
        for call in self.api_calls:
            agent_stats[call.agent]['calls'] += 1
            agent_stats[call.agent]['cost'] += call.estimated_cost
            agent_stats[call.agent]['tokens'] += call.token_count
            
            model_stats[call.model]['calls'] += 1
            model_stats[call.model]['cost'] += call.estimated_cost
            model_stats[call.model]['tokens'] += call.token_count
        
        # Recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_calls = [
            call for call in self.api_calls 
            if datetime.fromisoformat(call.timestamp) > one_hour_ago
        ]
        
        return {
            'total_calls': total_calls,
            'total_cost': total_cost,
            'average_cost_per_call': avg_cost_per_call,
            'current_costs': self.current_costs,
            'current_counts': self.call_counts,
            'agent_breakdown': dict(agent_stats),
            'model_breakdown': dict(model_stats),
            'recent_activity': {
                'calls_last_hour': len(recent_calls),
                'cost_last_hour': sum(call.estimated_cost for call in recent_calls)
            },
            'budget_status': self.check_budget_limits(),
            'remaining_budget': self.get_remaining_budget(),
            'alerts': list(self.alerts)[-10:],  # Last 10 alerts
            'top_endpoints': self.get_top_endpoints(),
            'cost_trends': self.get_cost_trends()
        }
    
    def get_top_endpoints(self) -> List[Dict[str, Any]]:
        """Get top endpoints by usage"""
        endpoint_stats = defaultdict(lambda: {'calls': 0, 'cost': 0.0})
        
        for call in self.api_calls:
            endpoint_stats[call.endpoint]['calls'] += 1
            endpoint_stats[call.endpoint]['cost'] += call.estimated_cost
        
        # Sort by cost
        sorted_endpoints = sorted(
            endpoint_stats.items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        )
        
        return [
            {
                'endpoint': endpoint,
                'calls': stats['calls'],
                'cost': stats['cost'],
                'avg_cost': stats['cost'] / stats['calls'] if stats['calls'] > 0 else 0
            }
            for endpoint, stats in sorted_endpoints[:10]
        ]
    
    def get_cost_trends(self) -> Dict[str, List[float]]:
        """Get cost trends over time"""
        # Simplified trend calculation (last 24 hours)
        trends = {
            'hourly_costs': [0.0] * 24,
            'hourly_calls': [0] * 24
        }
        
        # In a real implementation, this would calculate actual hourly costs
        # For now, return sample data
        return trends
    
    def can_afford_operation(self, estimated_tokens: int, model: str) -> Dict[str, Any]:
        """Check if we can afford a specific operation"""
        estimated_cost = self.calculate_cost(model, estimated_tokens)
        
        # Check against remaining budget
        remaining = self.get_remaining_budget()
        
        can_afford = (
            estimated_cost <= remaining['daily_remaining'] and
            estimated_cost <= remaining['hourly_remaining'] and
            self.call_counts['today'] < self.budget.call_limit
        )
        
        return {
            'can_afford': can_afford,
            'estimated_cost': estimated_cost,
            'remaining_daily': remaining['daily_remaining'],
            'remaining_hourly': remaining['hourly_remaining'],
            'calls_remaining': remaining['calls_remaining'],
            'recommendation': 'proceed' if can_afford else 'delay_or_optimize'
        }
    
    def save_usage_data(self) -> None:
        """Save usage data to file"""
        try:
            data = {
                'api_calls': [asdict(call) for call in list(self.api_calls)[-1000:]],  # Save last 1000
                'current_costs': self.current_costs,
                'current_counts': self.call_counts,
                'budget_config': asdict(self.budget),
                'last_updated': datetime.now().isoformat()
            }
            
            with open('api_usage_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")
    
    def load_usage_data(self) -> None:
        """Load existing usage data"""
        try:
            usage_file = Path('api_usage_data.json')
            if usage_file.exists():
                with open(usage_file, 'r') as f:
                    data = json.load(f)
                
                # Load recent calls
                if 'api_calls' in data:
                    for call_data in data['api_calls']:
                        call = APICall(**call_data)
                        self.api_calls.append(call)
                
                # Load current costs (would need to reset based on time in production)
                if 'current_costs' in data:
                    self.current_costs.update(data['current_costs'])
                
                if 'current_counts' in data:
                    self.call_counts.update(data['current_counts'])
                
                logger.info(f"Loaded {len(self.api_calls)} API call records")
                
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")

# Global tracker instance
api_tracker = APIUsageTracker()

def track_api_call(endpoint: str, model: str, token_count: int, purpose: str, 
                   agent: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to track API calls"""
    return api_tracker.track_api_call(endpoint, model, token_count, purpose, agent, **kwargs)

def check_budget_before_call(estimated_tokens: int, model: str) -> Dict[str, Any]:
    """Check if we can afford an operation before making the call"""
    return api_tracker.can_afford_operation(estimated_tokens, model)

def get_usage_dashboard_data() -> Dict[str, Any]:
    """Get data for dashboard display"""
    return api_tracker.get_usage_statistics()

if __name__ == '__main__':
    # Demo the tracker
    print("API Usage Tracker Demo")
    print("=" * 40)
    
    # Simulate some API calls
    track_api_call("/intelligence-backend", "gpt-4", 1500, "semantic_analysis", "alpha")
    track_api_call("/performance-engine", "claude-3-sonnet", 800, "optimization", "beta")
    track_api_call("/3d-visualization", "gpt-3.5-turbo", 600, "graph_generation", "gamma")
    
    # Get statistics
    stats = get_usage_dashboard_data()
    print(f"Total Calls: {stats['total_calls']}")
    print(f"Total Cost: ${stats['total_cost']:.4f}")
    print(f"Budget Status: {stats['budget_status']}")
    
    # Check if we can afford a large operation
    can_afford = check_budget_before_call(5000, "gpt-4")
    print(f"Can afford 5K token GPT-4 call: {can_afford['can_afford']}")