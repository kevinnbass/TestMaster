#!/usr/bin/env python3
"""
STEELCLAD MODULE: Advanced API Usage Tracking System
===================================================

API tracking classes extracted from unified_dashboard_modular.py
Original: 3,977 lines → Tracking Module: ~350 lines

Complete functionality extraction with zero regression.

Author: Agent X (STEELCLAD Anti-Regression Modularization)
"""

import sqlite3
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque


class ComprehensiveAPIUsageTracker:
    """
    Advanced API usage tracking system with budget monitoring and cost estimation.
    Tracks AI model usage, token consumption, and daily spending limits.
    """
    
    def __init__(self):
        self.api_calls = defaultdict(int)
        self.api_costs = {}
        self.model_usage = defaultdict(int) 
        self.api_history = deque(maxlen=10000)
        self.cost_estimates = {
            "gpt-4": 0.03,  # per 1k tokens
            "gpt-4-turbo": 0.01,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
            "gemini-pro": 0.0005,
            "llama-2": 0.0002
        }
        self.daily_budget = 50.0  # $50 daily budget
        self.daily_spending = 0.0
        self.last_reset = datetime.now().date()
    
    def track_api_call(self, endpoint: str, model: str = None, tokens: int = 0, purpose: str = "analysis"):
        """Track an API call with comprehensive cost estimation."""
        current_time = datetime.now()
        
        # Reset daily spending if new day
        if current_time.date() > self.last_reset:
            self.daily_spending = 0.0
            self.last_reset = current_time.date()
        
        # Estimate cost
        cost = 0.0
        if model and model in self.cost_estimates and tokens > 0:
            cost = (tokens / 1000) * self.cost_estimates[model]
            self.daily_spending += cost
        
        # Record the call
        call_record = {
            "timestamp": current_time.isoformat(),
            "endpoint": endpoint,
            "model": model,
            "tokens": tokens,
            "cost_usd": cost,
            "purpose": purpose,
            "daily_total": self.daily_spending
        }
        
        self.api_calls[endpoint] += 1
        if model:
            self.model_usage[model] += 1
        self.api_history.append(call_record)
        
        return call_record
    
    def check_budget_availability(self, estimated_cost: float):
        """Check if we can afford a planned API operation."""
        if self.daily_spending + estimated_cost > self.daily_budget:
            return False, f"Would exceed daily budget. Current: ${self.daily_spending:.2f}, Estimated: ${estimated_cost:.2f}, Budget: ${self.daily_budget}"
        return True, "Within budget"
    
    def get_usage_statistics(self):
        """Get comprehensive usage statistics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "daily_spending": round(self.daily_spending, 4),
            "daily_budget": self.daily_budget,
            "budget_remaining": round(self.daily_budget - self.daily_spending, 4),
            "total_api_calls": sum(self.api_calls.values()),
            "calls_by_endpoint": dict(self.api_calls),
            "model_usage": dict(self.model_usage),
            "budget_status": "OK" if self.daily_spending < self.daily_budget * 0.8 else "WARNING",
            "recent_calls": list(self.api_history)[-10:] if self.api_history else []
        }
    
    def get_cost_estimate(self, model: str, tokens: int):
        """Estimate cost for a planned API operation."""
        if model in self.cost_estimates:
            return (tokens / 1000) * self.cost_estimates[model]
        return 0.0


class DatabaseAPITracker:
    """
    Persistent SQLite database-backed API usage tracking system.
    Provides long-term storage and comprehensive analytics.
    """
    
    def __init__(self):
        self.db_path = "unified_api_usage_tracking.db"
        self.init_database()
        
    def init_database(self):
        """Initialize API usage tracking database with comprehensive schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT NOT NULL,
                model_used TEXT,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                purpose TEXT,
                agent TEXT,
                success BOOLEAN DEFAULT TRUE,
                response_time_ms INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                daily_budget_usd REAL DEFAULT 10.0,
                monthly_budget_usd REAL DEFAULT 300.0,
                current_daily_spend REAL DEFAULT 0.0,
                current_monthly_spend REAL DEFAULT 0.0,
                last_reset_date DATE DEFAULT CURRENT_DATE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_api_call(self, endpoint, model_used=None, tokens_used=0, cost_usd=0.0, 
                     purpose="analysis", agent="unknown", success=True, response_time_ms=0):
        """Log an API call with comprehensive metrics to persistent database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_calls 
            (endpoint, model_used, tokens_used, cost_usd, purpose, agent, success, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (endpoint, model_used, tokens_used, cost_usd, purpose, agent, success, response_time_ms))
        
        conn.commit()
        conn.close()
        
    def get_usage_stats(self, hours=24):
        """Get comprehensive API usage statistics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_calls,
                SUM(tokens_used) as total_tokens,
                SUM(cost_usd) as total_cost,
                AVG(response_time_ms) as avg_response_time,
                COUNT(DISTINCT model_used) as unique_models,
                agent,
                COUNT(*) as agent_calls,
                SUM(cost_usd) as agent_cost
            FROM api_calls 
            WHERE timestamp >= ?
            GROUP BY agent
            ORDER BY agent_cost DESC
        ''', (since_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "hours_analyzed": hours,
            "agent_stats": [
                {
                    "agent": row[5],
                    "calls": row[6],
                    "cost": row[7],
                    "avg_response_time": row[3]
                } for row in results
            ],
            "total_stats": {
                "calls": sum(row[6] for row in results),
                "cost": sum(row[7] for row in results),
                "tokens": sum(row[1] for row in results if row[1]),
                "unique_models": len(set(row[4] for row in results if row[4]))
            }
        }
    
    def get_agent_budgets(self):
        """Get budget information for all agents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT agent, daily_budget_usd, monthly_budget_usd, 
                   current_daily_spend, current_monthly_spend, last_reset_date
            FROM agent_budgets
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "budgets": [
                {
                    "agent": row[0],
                    "daily_budget": row[1],
                    "monthly_budget": row[2],
                    "daily_spend": row[3],
                    "monthly_spend": row[4],
                    "last_reset": row[5]
                } for row in results
            ]
        }