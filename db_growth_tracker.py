#!/usr/bin/env python3
"""
Database Growth Tracker
Agent B Hours 90-100: Database Growth Analysis

Tracks database size changes over time and predicts growth patterns.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class GrowthPoint:
    """Single database growth measurement"""
    timestamp: datetime
    size_mb: float
    table_count: int
    record_count: int
    index_count: int

@dataclass
class GrowthTrend:
    """Database growth trend analysis"""
    database_name: str
    current_size_mb: float
    daily_growth_mb: float
    weekly_growth_mb: float
    monthly_growth_mb: float
    growth_rate_percent: float
    predicted_size_30_days_mb: float
    largest_tables: List[Dict[str, Any]]
    growth_pattern: str  # 'linear', 'exponential', 'stable', 'declining'

class DatabaseGrowthTracker:
    """Tracks database growth patterns over time"""
    
    def __init__(self, data_file: str = "db_growth_data.json"):
        self.data_file = Path(data_file)
        self.growth_data: Dict[str, List[GrowthPoint]] = {}
        self.load_data()
    
    def load_data(self):
        """Load existing growth data"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to GrowthPoint objects
                for db_name, points in data.items():
                    self.growth_data[db_name] = []
                    for point in points:
                        growth_point = GrowthPoint(
                            timestamp=datetime.fromisoformat(point['timestamp']),
                            size_mb=point['size_mb'],
                            table_count=point['table_count'],
                            record_count=point['record_count'],
                            index_count=point['index_count']
                        )
                        self.growth_data[db_name].append(growth_point)
            except Exception as e:
                print(f"[WARNING] Failed to load growth data: {e}")
    
    def save_data(self):
        """Save growth data to file"""
        try:
            # Convert to serializable format
            serializable_data = {}
            for db_name, points in self.growth_data.items():
                serializable_data[db_name] = [asdict(point) for point in points]
                # Convert datetime to string
                for point in serializable_data[db_name]:
                    point['timestamp'] = point['timestamp'].isoformat()
            
            with open(self.data_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save growth data: {e}")
    
    def record_growth_point(self, db_name: str, db_path: str):
        """Record a growth measurement point"""
        try:
            if not Path(db_path).exists():
                print(f"[ERROR] Database not found: {db_path}")
                return
            
            # Get database size
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            
            # Connect and get database statistics
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Count indexes
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
            index_count = cursor.fetchone()[0]
            
            # Estimate total record count (rough approximation)
            record_count = 0
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    record_count += cursor.fetchone()[0]
                except:
                    pass  # Skip tables that can't be counted
            
            conn.close()
            
            # Create growth point
            growth_point = GrowthPoint(
                timestamp=datetime.now(),
                size_mb=size_mb,
                table_count=table_count,
                record_count=record_count,
                index_count=index_count
            )
            
            # Add to data
            if db_name not in self.growth_data:
                self.growth_data[db_name] = []
            
            self.growth_data[db_name].append(growth_point)
            
            # Keep only recent data (last 90 days)
            cutoff_date = datetime.now() - timedelta(days=90)
            self.growth_data[db_name] = [
                point for point in self.growth_data[db_name] 
                if point.timestamp > cutoff_date
            ]
            
            self.save_data()
            
            print(f"[OK] Recorded growth point for {db_name}: {size_mb:.3f}MB, {table_count} tables, {record_count} records")
            
        except Exception as e:
            print(f"[ERROR] Failed to record growth point for {db_name}: {e}")
    
    def analyze_growth_trend(self, db_name: str) -> Optional[GrowthTrend]:
        """Analyze growth trend for a database"""
        if db_name not in self.growth_data or len(self.growth_data[db_name]) < 2:
            return None
        
        points = sorted(self.growth_data[db_name], key=lambda x: x.timestamp)
        
        # Current stats
        current_point = points[-1]
        current_size = current_point.size_mb
        
        # Calculate growth rates
        daily_growth = self._calculate_growth_rate(points, days=1)
        weekly_growth = self._calculate_growth_rate(points, days=7)
        monthly_growth = self._calculate_growth_rate(points, days=30)
        
        # Calculate growth rate percentage
        if len(points) >= 2 and points[0].size_mb > 0:
            total_growth = current_size - points[0].size_mb
            days_elapsed = (points[-1].timestamp - points[0].timestamp).days
            if days_elapsed > 0:
                growth_rate_percent = (total_growth / points[0].size_mb) * 100 / days_elapsed * 30  # Monthly rate
            else:
                growth_rate_percent = 0.0
        else:
            growth_rate_percent = 0.0
        
        # Predict future size
        predicted_size = current_size + (daily_growth * 30)  # 30 days ahead
        
        # Determine growth pattern
        growth_pattern = self._determine_growth_pattern(points)
        
        # Get largest tables (placeholder - would need more detailed analysis)
        largest_tables = [
            {"table": "main_table", "size_mb": current_size * 0.6, "growth": "moderate"},
            {"table": "index_table", "size_mb": current_size * 0.3, "growth": "slow"},
            {"table": "metadata", "size_mb": current_size * 0.1, "growth": "stable"}
        ]
        
        return GrowthTrend(
            database_name=db_name,
            current_size_mb=current_size,
            daily_growth_mb=daily_growth,
            weekly_growth_mb=weekly_growth,
            monthly_growth_mb=monthly_growth,
            growth_rate_percent=growth_rate_percent,
            predicted_size_30_days_mb=predicted_size,
            largest_tables=largest_tables,
            growth_pattern=growth_pattern
        )
    
    def _calculate_growth_rate(self, points: List[GrowthPoint], days: int) -> float:
        """Calculate growth rate over specified number of days"""
        if len(points) < 2:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_points = [p for p in points if p.timestamp > cutoff_date]
        
        if len(recent_points) < 2:
            return 0.0
        
        # Simple linear growth calculation
        start_size = recent_points[0].size_mb
        end_size = recent_points[-1].size_mb
        time_diff = (recent_points[-1].timestamp - recent_points[0].timestamp).days
        
        if time_diff > 0:
            return (end_size - start_size) / time_diff * days
        else:
            return 0.0
    
    def _determine_growth_pattern(self, points: List[GrowthPoint]) -> str:
        """Determine the growth pattern"""
        if len(points) < 3:
            return "insufficient_data"
        
        sizes = [p.size_mb for p in points]
        
        # Calculate variance in growth rate
        growth_rates = []
        for i in range(1, len(sizes)):
            if i > 0:
                rate = sizes[i] - sizes[i-1]
                growth_rates.append(rate)
        
        if not growth_rates:
            return "stable"
        
        avg_growth = np.mean(growth_rates)
        growth_variance = np.var(growth_rates)
        
        # Determine pattern based on growth characteristics
        if abs(avg_growth) < 0.01:  # Very small growth
            return "stable"
        elif avg_growth < 0:
            return "declining"
        elif growth_variance < 0.01:  # Consistent growth
            return "linear"
        else:
            return "variable"
    
    def get_growth_summary(self) -> Dict[str, Any]:
        """Get summary of all database growth trends"""
        summary = {
            "databases_tracked": len(self.growth_data),
            "total_data_points": sum(len(points) for points in self.growth_data.values()),
            "databases": {}
        }
        
        for db_name in self.growth_data.keys():
            trend = self.analyze_growth_trend(db_name)
            if trend:
                summary["databases"][db_name] = {
                    "current_size_mb": round(trend.current_size_mb, 3),
                    "daily_growth_mb": round(trend.daily_growth_mb, 3),
                    "monthly_growth_mb": round(trend.monthly_growth_mb, 3),
                    "growth_pattern": trend.growth_pattern,
                    "predicted_size_30_days": round(trend.predicted_size_30_days_mb, 3)
                }
        
        return summary
    
    def generate_growth_report(self) -> str:
        """Generate a human-readable growth report"""
        summary = self.get_growth_summary()
        
        if summary["databases_tracked"] == 0:
            return "No database growth data available. Start tracking with record_growth_point()"
        
        report = f"""
DATABASE GROWTH REPORT
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Databases Tracked: {summary['databases_tracked']}
- Total Data Points: {summary['total_data_points']}

DATABASE DETAILS:
"""
        
        for db_name, data in summary["databases"].items():
            report += f"""
{db_name.upper()}:
  Current Size: {data['current_size_mb']:.3f} MB
  Daily Growth: {data['daily_growth_mb']:.3f} MB/day
  Monthly Growth: {data['monthly_growth_mb']:.3f} MB/month
  Growth Pattern: {data['growth_pattern'].replace('_', ' ').title()}
  Predicted Size (30 days): {data['predicted_size_30_days']:.3f} MB
"""
        
        return report

def track_databases_from_config(config_file: str = "db_monitor_config.json"):
    """Track all databases from monitor configuration"""
    tracker = DatabaseGrowthTracker()
    
    try:
        if not Path(config_file).exists():
            print(f"[ERROR] Config file not found: {config_file}")
            return
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        databases = config.get("databases", {})
        if not databases:
            print("[WARNING] No databases configured")
            return
        
        print(f"[OK] Tracking growth for {len(databases)} databases...")
        
        for db_name, db_config in databases.items():
            if db_config.get("enabled", False) and db_config.get("type") == "sqlite":
                db_path = db_config.get("path")
                if db_path:
                    tracker.record_growth_point(db_name, db_path)
        
        # Generate and display report
        report = tracker.generate_growth_report()
        print(report)
        
        return tracker
        
    except Exception as e:
        print(f"[ERROR] Failed to track databases: {e}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "track":
            # Track all configured databases
            track_databases_from_config()
        elif sys.argv[1] == "report":
            # Show growth report
            tracker = DatabaseGrowthTracker()
            print(tracker.generate_growth_report())
        elif len(sys.argv) >= 4 and sys.argv[1] == "add":
            # Add single database point
            db_name = sys.argv[2]
            db_path = sys.argv[3]
            tracker = DatabaseGrowthTracker()
            tracker.record_growth_point(db_name, db_path)
        else:
            print("Usage:")
            print("  python db_growth_tracker.py track    # Track all configured databases")
            print("  python db_growth_tracker.py report   # Show growth report")
            print("  python db_growth_tracker.py add <name> <path>  # Add single measurement")
    else:
        # Default: track configured databases
        track_databases_from_config()