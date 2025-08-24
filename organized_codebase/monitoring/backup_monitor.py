#!/usr/bin/env python3
"""
Backup Monitor
Agent B Hours 90-100: Backup Health Monitoring

Monitors database backup files, schedules, and integrity.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
import hashlib

@dataclass
class BackupFile:
    """Information about a backup file"""
    path: str
    size_mb: float
    created_date: datetime
    age_days: int
    is_valid: bool
    checksum: str
    database_name: str
    backup_type: str  # 'auto', 'manual', 'scheduled'

@dataclass
class BackupHealth:
    """Overall backup health status"""
    database_name: str
    total_backups: int
    latest_backup_age_days: int
    total_backup_size_mb: float
    backup_frequency_days: float
    oldest_backup_date: datetime
    newest_backup_date: datetime
    health_score: int  # 1-100
    issues: List[str]
    recommendations: List[str]

class BackupMonitor:
    """Monitors database backup health and integrity"""
    
    def __init__(self, backup_directories: List[str] = None):
        self.backup_directories = backup_directories or [
            "./large_files_temp",
            "./backups",
            "./auto_backups",
            "."
        ]
        
        # Backup file patterns
        self.backup_patterns = [
            r'.*backup.*\.db$',
            r'.*backup.*\.sqlite.*$', 
            r'.*\.backup$',
            r'auto_backup_.*\.db$',
            r'.*_\d{8}_\d{6}\.db$',  # date_time pattern
        ]
        
        self.backup_files: List[BackupFile] = []
        self.health_reports: Dict[str, BackupHealth] = {}
    
    def scan_for_backups(self):
        """Scan directories for backup files"""
        print("[SCANNING] Looking for backup files...")
        
        self.backup_files = []
        
        for directory in self.backup_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            
            print(f"[SCANNING] {directory}")
            
            for file_path in dir_path.rglob("*.db"):
                if self._is_backup_file(file_path):
                    backup_file = self._analyze_backup_file(file_path)
                    if backup_file:
                        self.backup_files.append(backup_file)
        
        print(f"[OK] Found {len(self.backup_files)} backup files")
    
    def _is_backup_file(self, file_path: Path) -> bool:
        """Check if a file appears to be a backup"""
        file_name = file_path.name.lower()
        
        for pattern in self.backup_patterns:
            if re.match(pattern, file_name):
                return True
        
        # Additional checks
        if 'backup' in file_name or 'bak' in file_name:
            return True
        
        return False
    
    def _analyze_backup_file(self, file_path: Path) -> Optional[BackupFile]:
        """Analyze a backup file"""
        try:
            # Get file stats
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created_date = datetime.fromtimestamp(stat.st_mtime)
            age_days = (datetime.now() - created_date).days
            
            # Determine database name and backup type
            database_name, backup_type = self._extract_backup_info(file_path)
            
            # Check if backup is valid SQLite
            is_valid = self._validate_backup(file_path)
            
            # Generate checksum
            checksum = self._calculate_checksum(file_path)
            
            return BackupFile(
                path=str(file_path),
                size_mb=size_mb,
                created_date=created_date,
                age_days=age_days,
                is_valid=is_valid,
                checksum=checksum,
                database_name=database_name,
                backup_type=backup_type
            )
            
        except Exception as e:
            print(f"[WARNING] Failed to analyze backup {file_path}: {e}")
            return None
    
    def _extract_backup_info(self, file_path: Path) -> tuple:
        """Extract database name and backup type from filename"""
        filename = file_path.name
        
        # Extract database name
        database_name = "unknown"
        if "cache" in filename.lower():
            database_name = "cache"
        elif "deduplication" in filename.lower():
            database_name = "deduplication"
        elif "auto_backup" in filename.lower():
            database_name = "auto_backup"
        else:
            # Try to extract from filename pattern
            match = re.search(r'([^_\-\.]+)', filename)
            if match:
                database_name = match.group(1)
        
        # Determine backup type
        backup_type = "manual"
        if "auto" in filename.lower():
            backup_type = "auto"
        elif re.search(r'\d{8}_\d{6}', filename):
            backup_type = "scheduled"
        
        return database_name, backup_type
    
    def _validate_backup(self, file_path: Path) -> bool:
        """Check if backup file is a valid SQLite database"""
        try:
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except:
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of backup file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "unknown"
    
    def analyze_backup_health(self) -> Dict[str, BackupHealth]:
        """Analyze backup health for each database"""
        if not self.backup_files:
            self.scan_for_backups()
        
        # Group backups by database
        db_backups: Dict[str, List[BackupFile]] = {}
        for backup in self.backup_files:
            if backup.database_name not in db_backups:
                db_backups[backup.database_name] = []
            db_backups[backup.database_name].append(backup)
        
        # Analyze each database's backup health
        self.health_reports = {}
        for db_name, backups in db_backups.items():
            health = self._calculate_backup_health(db_name, backups)
            self.health_reports[db_name] = health
        
        return self.health_reports
    
    def _calculate_backup_health(self, db_name: str, backups: List[BackupFile]) -> BackupHealth:
        """Calculate backup health for a specific database"""
        # Sort backups by date
        sorted_backups = sorted(backups, key=lambda x: x.created_date)
        
        # Basic stats
        total_backups = len(backups)
        total_size = sum(b.size_mb for b in backups)
        
        # Date analysis
        oldest_date = sorted_backups[0].created_date
        newest_date = sorted_backups[-1].created_date
        latest_age = (datetime.now() - newest_date).days
        
        # Calculate backup frequency
        if total_backups > 1:
            total_days = (newest_date - oldest_date).days
            frequency = total_days / (total_backups - 1) if total_backups > 1 else 0
        else:
            frequency = float('inf')
        
        # Identify issues
        issues = []
        recommendations = []
        
        # Check for old backups
        if latest_age > 7:
            issues.append(f"Latest backup is {latest_age} days old")
            recommendations.append("Create more recent backups")
        
        # Check for invalid backups
        invalid_backups = [b for b in backups if not b.is_valid]
        if invalid_backups:
            issues.append(f"{len(invalid_backups)} invalid backup files")
            recommendations.append("Remove or fix corrupted backup files")
        
        # Check backup frequency
        if frequency > 30:
            issues.append(f"Low backup frequency ({frequency:.1f} days between backups)")
            recommendations.append("Increase backup frequency")
        
        # Check for very old backups taking up space
        very_old_backups = [b for b in backups if b.age_days > 90]
        if len(very_old_backups) > 5:
            issues.append(f"{len(very_old_backups)} very old backups (>90 days)")
            recommendations.append("Consider archiving or removing old backups")
        
        # Calculate health score (1-100)
        health_score = 100
        
        # Deduct for issues
        if latest_age > 1:
            health_score -= min(50, latest_age * 2)  # Max 50 points for age
        
        if invalid_backups:
            health_score -= len(invalid_backups) * 10  # 10 points per invalid backup
        
        if frequency > 7:
            health_score -= min(30, (frequency - 7) * 2)  # Deduct for infrequent backups
        
        health_score = max(1, health_score)  # Minimum score of 1
        
        return BackupHealth(
            database_name=db_name,
            total_backups=total_backups,
            latest_backup_age_days=latest_age,
            total_backup_size_mb=total_size,
            backup_frequency_days=frequency,
            oldest_backup_date=oldest_date,
            newest_backup_date=newest_date,
            health_score=health_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def generate_backup_report(self) -> str:
        """Generate comprehensive backup report"""
        if not self.health_reports:
            self.analyze_backup_health()
        
        report = f"""
BACKUP HEALTH REPORT
====================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Backup Files Found: {len(self.backup_files)}
Databases Analyzed: {len(self.health_reports)}

"""
        
        if not self.backup_files:
            report += "No backup files found. Consider setting up automated backups.\n"
            return report
        
        # Overall summary
        avg_health = sum(h.health_score for h in self.health_reports.values()) / len(self.health_reports)
        total_size = sum(h.total_backup_size_mb for h in self.health_reports.values())
        
        report += f"OVERALL SUMMARY:\n"
        report += f"- Average Health Score: {avg_health:.1f}/100\n"
        report += f"- Total Backup Storage: {total_size:.2f} MB\n"
        report += f"- Health Status: {'Good' if avg_health > 80 else 'Fair' if avg_health > 60 else 'Poor'}\n\n"
        
        # Database details
        for db_name, health in self.health_reports.items():
            report += f"{db_name.upper()} DATABASE BACKUPS:\n"
            report += f"  Health Score: {health.health_score}/100\n"
            report += f"  Total Backups: {health.total_backups}\n"
            report += f"  Latest Backup: {health.latest_backup_age_days} days ago\n"
            report += f"  Backup Frequency: {health.backup_frequency_days:.1f} days\n"
            report += f"  Total Size: {health.total_backup_size_mb:.2f} MB\n"
            
            if health.issues:
                report += f"  Issues:\n"
                for issue in health.issues:
                    report += f"    - {issue}\n"
            
            if health.recommendations:
                report += f"  Recommendations:\n"
                for rec in health.recommendations:
                    report += f"    - {rec}\n"
            
            report += "\n"
        
        # File listing
        report += "BACKUP FILES:\n"
        for backup in sorted(self.backup_files, key=lambda x: x.created_date, reverse=True):
            status = "Valid" if backup.is_valid else "INVALID"
            report += f"  {backup.path}\n"
            report += f"    Size: {backup.size_mb:.2f}MB | Age: {backup.age_days} days | Status: {status}\n"
            report += f"    Type: {backup.backup_type} | DB: {backup.database_name}\n"
        
        return report
    
    def get_backup_summary(self) -> Dict[str, Any]:
        """Get backup summary for dashboard"""
        if not self.health_reports:
            self.analyze_backup_health()
        
        if not self.backup_files:
            return {
                "total_backups": 0,
                "databases": 0,
                "avg_health_score": 0,
                "total_size_mb": 0,
                "latest_backup_age_days": float('inf'),
                "status": "no_backups"
            }
        
        avg_health = sum(h.health_score for h in self.health_reports.values()) / len(self.health_reports)
        total_size = sum(h.total_backup_size_mb for h in self.health_reports.values())
        latest_age = min(h.latest_backup_age_days for h in self.health_reports.values())
        
        return {
            "total_backups": len(self.backup_files),
            "databases": len(self.health_reports),
            "avg_health_score": round(avg_health, 1),
            "total_size_mb": round(total_size, 2),
            "latest_backup_age_days": latest_age,
            "status": "good" if avg_health > 80 else "fair" if avg_health > 60 else "poor"
        }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        # Scan specific directories
        directories = sys.argv[2:] if len(sys.argv) > 2 else None
        monitor = BackupMonitor(directories)
        monitor.scan_for_backups()
        report = monitor.generate_backup_report()
        print(report)
    else:
        # Default: scan default directories
        monitor = BackupMonitor()
        report = monitor.generate_backup_report()
        print(report)