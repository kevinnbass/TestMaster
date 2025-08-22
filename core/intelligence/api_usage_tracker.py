"""
API Usage Tracker - Comprehensive Cost Monitoring System
Agent Alpha Implementation - Hours 0-2
"""

import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from threading import Lock
import os

class APIUsageTracker:
    """Comprehensive API usage tracking with cost monitoring and budget controls."""
    
    # Model pricing per 1K tokens (input/output)
    MODEL_PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-32k': {'input': 0.06, 'output': 0.12},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'claude-sonnet-4': {'input': 0.003, 'output': 0.015},  # Current model
        'text-davinci-003': {'input': 0.02, 'output': 0.02},
        'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0001},
        'dall-e-3': {'per_image': 0.040},  # Standard quality
        'dall-e-2': {'per_image': 0.020},  # 1024x1024
    }
    
    def __init__(self, db_path: str = "api_usage.db", daily_budget: float = 50.0):
        self.db_path = db_path
        self.daily_budget = daily_budget
        self.lock = Lock()
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    actual_cost REAL DEFAULT NULL,
                    request_data TEXT,
                    response_data TEXT,
                    execution_time REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT DEFAULT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_budgets (
                    date TEXT PRIMARY KEY,
                    budget_limit REAL NOT NULL,
                    total_spent REAL DEFAULT 0.0,
                    call_count INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON api_calls(model);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_purpose ON api_calls(purpose);
            """)
            
    def log_api_call(self, 
                    model: str,
                    provider: str,
                    purpose: str,
                    input_tokens: int = 0,
                    output_tokens: int = 0,
                    request_data: Dict = None,
                    response_data: Dict = None,
                    execution_time: float = 0.0,
                    success: bool = True,
                    error_message: str = None) -> int:
        """Log an API call with comprehensive tracking."""
        
        with self.lock:
            total_tokens = input_tokens + output_tokens
            estimated_cost = self.calculate_cost(model, input_tokens, output_tokens)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO api_calls 
                    (model, provider, purpose, input_tokens, output_tokens, total_tokens,
                     estimated_cost, request_data, response_data, execution_time, 
                     success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model, provider, purpose, input_tokens, output_tokens, total_tokens,
                    estimated_cost, json.dumps(request_data) if request_data else None,
                    json.dumps(response_data) if response_data else None,
                    execution_time, success, error_message
                ))
                
                call_id = cursor.lastrowid
                
                # Update daily budget tracking
                today = datetime.now().strftime('%Y-%m-%d')
                conn.execute("""
                    INSERT OR REPLACE INTO daily_budgets 
                    (date, budget_limit, total_spent, call_count, last_updated)
                    VALUES (?, ?, 
                           COALESCE((SELECT total_spent FROM daily_budgets WHERE date = ?), 0) + ?,
                           COALESCE((SELECT call_count FROM daily_budgets WHERE date = ?), 0) + 1,
                           CURRENT_TIMESTAMP)
                """, (today, self.daily_budget, today, estimated_cost, today))
                
                return call_id
                
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for API call."""
        if model not in self.MODEL_PRICING:
            return 0.0
            
        pricing = self.MODEL_PRICING[model]
        
        if 'per_image' in pricing:
            return pricing['per_image']
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
        
    def check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status and warnings."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT budget_limit, total_spent, call_count 
                FROM daily_budgets WHERE date = ?
            """, (today,))
            
            result = cursor.fetchone()
            
            if result:
                budget_limit, total_spent, call_count = result
            else:
                budget_limit, total_spent, call_count = self.daily_budget, 0.0, 0
                
            remaining_budget = budget_limit - total_spent
            budget_used_pct = (total_spent / budget_limit) * 100 if budget_limit > 0 else 0
            
            status = {
                'date': today,
                'budget_limit': budget_limit,
                'total_spent': total_spent,
                'remaining_budget': remaining_budget,
                'budget_used_percentage': budget_used_pct,
                'call_count': call_count,
                'status': self._get_budget_status_level(budget_used_pct),
                'warning_message': self._get_budget_warning(budget_used_pct, remaining_budget)
            }
            
            return status
            
    def _get_budget_status_level(self, usage_pct: float) -> str:
        """Get budget status level."""
        if usage_pct >= 100:
            return 'EXCEEDED'
        elif usage_pct >= 90:
            return 'CRITICAL'
        elif usage_pct >= 75:
            return 'WARNING'
        elif usage_pct >= 50:
            return 'MODERATE'
        else:
            return 'SAFE'
            
    def _get_budget_warning(self, usage_pct: float, remaining: float) -> str:
        """Generate appropriate warning message."""
        if usage_pct >= 100:
            return f"🚨 BUDGET EXCEEDED! No remaining budget. Stop all API calls."
        elif usage_pct >= 90:
            return f"🔴 CRITICAL: Only ${remaining:.2f} remaining ({100-usage_pct:.1f}% left)"
        elif usage_pct >= 75:
            return f"🟡 WARNING: ${remaining:.2f} remaining ({100-usage_pct:.1f}% left)"
        elif usage_pct >= 50:
            return f"🟢 MODERATE: ${remaining:.2f} remaining ({100-usage_pct:.1f}% left)"
        else:
            return f"✅ SAFE: ${remaining:.2f} remaining ({100-usage_pct:.1f}% left)"
            
    def get_usage_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive usage analytics."""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Total usage stats
            cursor = conn.execute("""
                SELECT COUNT(*) as total_calls,
                       SUM(estimated_cost) as total_cost,
                       SUM(input_tokens) as total_input_tokens,
                       SUM(output_tokens) as total_output_tokens,
                       AVG(execution_time) as avg_execution_time
                FROM api_calls 
                WHERE DATE(timestamp) >= ?
            """, (start_date,))
            
            total_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Model breakdown
            cursor = conn.execute("""
                SELECT model, 
                       COUNT(*) as calls,
                       SUM(estimated_cost) as cost,
                       SUM(total_tokens) as tokens
                FROM api_calls 
                WHERE DATE(timestamp) >= ?
                GROUP BY model
                ORDER BY cost DESC
            """, (start_date,))
            
            model_breakdown = [dict(zip([col[0] for col in cursor.description], row)) 
                             for row in cursor.fetchall()]
            
            # Purpose breakdown
            cursor = conn.execute("""
                SELECT purpose,
                       COUNT(*) as calls,
                       SUM(estimated_cost) as cost,
                       AVG(execution_time) as avg_time
                FROM api_calls 
                WHERE DATE(timestamp) >= ?
                GROUP BY purpose
                ORDER BY calls DESC
            """, (start_date,))
            
            purpose_breakdown = [dict(zip([col[0] for col in cursor.description], row)) 
                               for row in cursor.fetchall()]
            
            # Daily trends
            cursor = conn.execute("""
                SELECT DATE(timestamp) as date,
                       COUNT(*) as calls,
                       SUM(estimated_cost) as cost
                FROM api_calls 
                WHERE DATE(timestamp) >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (start_date,))
            
            daily_trends = [dict(zip([col[0] for col in cursor.description], row)) 
                          for row in cursor.fetchall()]
            
            return {
                'period_days': days,
                'start_date': start_date,
                'total_stats': total_stats,
                'model_breakdown': model_breakdown,
                'purpose_breakdown': purpose_breakdown,
                'daily_trends': daily_trends,
                'current_budget_status': self.check_budget_status()
            }
            
    def pre_call_budget_check(self, estimated_tokens: int = 1000, model: str = 'claude-sonnet-4') -> Dict[str, Any]:
        """Check budget before making API call."""
        estimated_cost = self.calculate_cost(model, estimated_tokens, estimated_tokens)
        budget_status = self.check_budget_status()
        
        can_afford = budget_status['remaining_budget'] >= estimated_cost
        
        return {
            'can_afford': can_afford,
            'estimated_cost': estimated_cost,
            'remaining_budget': budget_status['remaining_budget'],
            'budget_status': budget_status['status'],
            'warning_message': budget_status['warning_message'],
            'recommendation': 'PROCEED' if can_afford else 'ABORT - INSUFFICIENT BUDGET'
        }
        
    def export_usage_report(self, output_path: str = None) -> str:
        """Export comprehensive usage report."""
        if not output_path:
            output_path = f"api_usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        analytics = self.get_usage_analytics(30)  # 30-day report
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_period': '30_days',
            'analytics': analytics,
            'model_pricing': self.MODEL_PRICING,
            'database_path': self.db_path
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return output_path

# Global tracker instance
api_tracker = APIUsageTracker()

def track_api_call(func):
    """Decorator for automatic API call tracking."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract tracking parameters from kwargs
        model = kwargs.pop('_track_model', 'unknown')
        provider = kwargs.pop('_track_provider', 'unknown') 
        purpose = kwargs.pop('_track_purpose', func.__name__)
        input_tokens = kwargs.pop('_track_input_tokens', 0)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Try to extract output tokens from result
            output_tokens = 0
            if isinstance(result, dict) and 'usage' in result:
                output_tokens = result.get('usage', {}).get('completion_tokens', 0)
                input_tokens = result.get('usage', {}).get('prompt_tokens', input_tokens)
            
            api_tracker.log_api_call(
                model=model,
                provider=provider,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_data={'args': str(args)[:500], 'kwargs': str(kwargs)[:500]},
                response_data={'result_type': type(result).__name__},
                execution_time=execution_time,
                success=True
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            api_tracker.log_api_call(
                model=model,
                provider=provider,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=0,
                request_data={'args': str(args)[:500], 'kwargs': str(kwargs)[:500]},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
            
            raise
            
    return wrapper

if __name__ == "__main__":
    # Test the tracker
    tracker = APIUsageTracker()
    
    # Simulate some API calls
    tracker.log_api_call('claude-sonnet-4', 'anthropic', 'test_analysis', 1000, 500)
    tracker.log_api_call('gpt-4', 'openai', 'code_review', 2000, 1000)
    
    # Check budget
    status = tracker.check_budget_status()
    print(f"Budget Status: {status}")
    
    # Get analytics
    analytics = tracker.get_usage_analytics()
    print(f"Analytics: {analytics}")