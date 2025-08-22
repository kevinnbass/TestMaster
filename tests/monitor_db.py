#!/usr/bin/env python3
"""
Database Monitor CLI
Simple command-line tool to monitor your database

Usage:
  python monitor_db.py setup          # Create config file
  python monitor_db.py add <name> <type> <connection_info>  # Add database
  python monitor_db.py start          # Start monitoring
  python monitor_db.py status         # Show current status  
  python monitor_db.py report         # Generate report
  python monitor_db.py dashboard      # Start web dashboard
"""

import sys
import json
import time
from pathlib import Path

# Add the TestMaster modules to path
sys.path.append(str(Path(__file__).parent))

from TestMaster.core.orchestration.monitoring.personal_database_monitor import PersonalDatabaseMonitor
from TestMaster.core.orchestration.monitoring.simple_dashboard import SimpleDashboard

def setup_config():
    """Create initial configuration"""
    config = {
        "databases": {},
        "monitoring_interval": 30,
        "enable_alerts": True,
        "log_slow_queries": True,
        "dashboard_port": 8080
    }
    
    config_file = Path("database_monitor_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: database_monitor_config.json")
    print("\nNext steps:")
    print("1. Add a database: python monitor_db.py add mydb sqlite path/to/database.db")
    print("2. Start monitoring: python monitor_db.py start")
    print("3. View dashboard: python monitor_db.py dashboard")

def add_database():
    """Add a database to monitor"""
    if len(sys.argv) < 5:
        print("Usage: python monitor_db.py add <name> <type> <connection_string>")
        print("\nExamples:")
        print("  SQLite:     python monitor_db.py add myapp sqlite /path/to/app.db")
        print("  MySQL:      python monitor_db.py add mysql mysql 'host=localhost user=root password=pass database=mydb'")
        print("  PostgreSQL: python monitor_db.py add postgres postgresql 'host=localhost user=postgres password=pass database=mydb'")
        return
    
    name = sys.argv[2]
    db_type = sys.argv[3]
    connection_info = sys.argv[4]
    
    monitor = PersonalDatabaseMonitor()
    
    if db_type == "sqlite":
        success = monitor.add_database(name, "sqlite", path=connection_info)
    elif db_type == "mysql":
        # Parse connection string
        params = {}
        for param in connection_info.split():
            key, value = param.split('=', 1)
            params[key] = value
        success = monitor.add_database(name, "mysql", **params)
    elif db_type == "postgresql":
        # Parse connection string
        params = {}
        for param in connection_info.split():
            key, value = param.split('=', 1)
            params[key] = value
        success = monitor.add_database(name, "postgresql", **params)
    else:
        print(f"❌ Unsupported database type: {db_type}")
        print("Supported types: sqlite, mysql, postgresql")
        return
    
    if success:
        print(f"✅ Database '{name}' added successfully")
    else:
        print(f"❌ Failed to add database '{name}' - check connection parameters")

def show_status():
    """Show current monitoring status"""
    monitor = PersonalDatabaseMonitor()
    status = monitor.get_current_status()
    
    print("📊 DATABASE MONITOR STATUS")
    print("=" * 30)
    print(f"Monitoring Active: {'✅ Yes' if status['monitoring_active'] else '❌ No'}")
    print(f"Databases Configured: {status['databases_configured']}")
    print(f"Metrics Collected: {status['metrics_collected']}")
    print(f"Active Alerts: {status['total_alerts']}")
    
    if status['latest_metrics']:
        metrics = status['latest_metrics']
        print(f"\nLatest Metrics ({metrics['timestamp'][:19]}):")
        print(f"  CPU Usage: {metrics['cpu_usage']:.1f}%")
        print(f"  Memory Usage: {metrics['memory_usage_mb']:.1f} MB")
        print(f"  Database Size: {metrics['database_size_mb']:.1f} MB")
        print(f"  Connections: {metrics['connection_count']}")

def generate_report():
    """Generate and display monitoring report"""
    monitor = PersonalDatabaseMonitor()
    report = monitor.generate_simple_report()
    print(report)

def start_monitoring():
    """Start continuous monitoring"""
    monitor = PersonalDatabaseMonitor()
    
    print("🚀 Starting database monitoring...")
    print("Press Ctrl+C to stop")
    
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(10)
            status = monitor.get_current_status()
            
            if status['latest_metrics']:
                metrics = status['latest_metrics']
                timestamp = metrics['timestamp'][:19]
                cpu = metrics['cpu_usage']
                memory = metrics['memory_usage_mb']
                connections = metrics['connection_count']
                
                print(f"[{timestamp}] CPU: {cpu:.1f}% | Memory: {memory:.1f}MB | Connections: {connections}")
            
            # Show any new alerts
            alerts = monitor.get_active_alerts()
            for alert in alerts[-3:]:  # Show last 3 alerts
                if not alert.get('shown', False):
                    print(f"🚨 ALERT: {alert['message']}")
                    alert['shown'] = True
                    
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        monitor.stop_monitoring()
        print("✅ Monitoring stopped")

def start_dashboard():
    """Start web dashboard"""
    monitor = PersonalDatabaseMonitor()
    dashboard = SimpleDashboard(monitor)
    
    print("🚀 Starting database monitoring and dashboard...")
    monitor.start_monitoring()
    
    try:
        dashboard.start()  # This blocks
    finally:
        monitor.stop_monitoring()

def show_help():
    """Show help information"""
    print("📊 Personal Database Monitor")
    print("=" * 30)
    print("Commands:")
    print("  setup           Create configuration file")
    print("  add <name> <type> <info>  Add database to monitor")
    print("  start           Start continuous monitoring")
    print("  status          Show current status")
    print("  report          Generate monitoring report")
    print("  dashboard       Start web dashboard")
    print("  help            Show this help")
    print("\nExamples:")
    print("  python monitor_db.py setup")
    print("  python monitor_db.py add myapp sqlite /path/to/app.db")
    print("  python monitor_db.py start")
    print("  python monitor_db.py dashboard")

def main():
    """Main CLI function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        setup_config()
    elif command == "add":
        add_database()
    elif command == "start":
        start_monitoring()
    elif command == "status":
        show_status()
    elif command == "report":
        generate_report()
    elif command == "dashboard":
        start_dashboard()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()