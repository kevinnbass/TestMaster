"""
Analytics Backup and Recovery System
====================================

Provides backup, recovery, and data integrity features for analytics data.
Ensures data durability and disaster recovery capabilities.

Author: TestMaster Team
"""

import logging
import os
import shutil
import json
import sqlite3
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import hashlib

logger = logging.getLogger(__name__)

class AnalyticsBackupManager:
    """
    Manages backup and recovery operations for analytics data.
    """
    
    def __init__(self, backup_dir: str = None, retention_days: int = 30, auto_backup_hours: int = 6):
        """
        Initialize the backup manager.
        
        Args:
            backup_dir: Directory to store backups
            retention_days: Days to retain backups
            auto_backup_hours: Hours between automatic backups
        """
        if backup_dir is None:
            backup_dir = os.path.join(os.path.dirname(__file__), '..', 'backups')
        
        self.backup_dir = backup_dir
        self.retention_days = retention_days
        self.auto_backup_hours = auto_backup_hours
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Backup metadata
        self.backup_metadata = {}
        self.last_backup_time = None
        
        # Threading
        self._lock = threading.Lock()
        self._auto_backup_timer = None
        
        logger.info(f"Analytics Backup Manager initialized: {backup_dir}")
    
    def create_backup(self, data_store_path: str, backup_name: str = None) -> Dict[str, Any]:
        """
        Create a backup of the analytics database.
        
        Args:
            data_store_path: Path to the SQLite database
            backup_name: Optional backup name
            
        Returns:
            Backup metadata
        """
        if backup_name is None:
            backup_name = f"analytics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.db")
        compressed_path = f"{backup_path}.gz"
        
        try:
            with self._lock:
                # Copy database file
                if os.path.exists(data_store_path):
                    shutil.copy2(data_store_path, backup_path)
                    
                    # Compress backup
                    with open(backup_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove uncompressed backup
                    os.remove(backup_path)
                    
                    # Calculate checksum
                    checksum = self._calculate_file_checksum(compressed_path)
                    
                    # Create metadata
                    metadata = {
                        'backup_name': backup_name,
                        'backup_path': compressed_path,
                        'original_path': data_store_path,
                        'timestamp': datetime.now().isoformat(),
                        'size_bytes': os.path.getsize(compressed_path),
                        'checksum': checksum,
                        'compression': 'gzip',
                        'version': '1.0'
                    }
                    
                    # Verify backup integrity
                    if self._verify_backup_integrity(compressed_path, checksum):
                        # Store metadata
                        self.backup_metadata[backup_name] = metadata
                        self._save_backup_metadata()
                        
                        self.last_backup_time = datetime.now()
                        
                        logger.info(f"Backup created successfully: {backup_name}")
                        return metadata
                    else:
                        raise Exception("Backup integrity verification failed")
                else:
                    raise FileNotFoundError(f"Database file not found: {data_store_path}")
                    
        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            # Cleanup on failure
            if os.path.exists(backup_path):
                os.remove(backup_path)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            raise
    
    def restore_backup(self, backup_name: str, restore_path: str) -> bool:
        """
        Restore a backup to the specified path.
        
        Args:
            backup_name: Name of the backup to restore
            restore_path: Path to restore the database to
            
        Returns:
            True if restore successful, False otherwise
        """
        if backup_name not in self.backup_metadata:
            logger.error(f"Backup {backup_name} not found")
            return False
        
        metadata = self.backup_metadata[backup_name]
        backup_path = metadata['backup_path']
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            with self._lock:
                # Verify backup integrity before restore
                if not self._verify_backup_integrity(backup_path, metadata['checksum']):
                    logger.error(f"Backup integrity check failed for {backup_name}")
                    return False
                
                # Create backup of current database if it exists
                if os.path.exists(restore_path):
                    current_backup = f"{restore_path}.pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(restore_path, current_backup)
                    logger.info(f"Current database backed up to: {current_backup}")
                
                # Decompress and restore
                temp_path = f"{restore_path}.temp"
                
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Verify restored database
                if self._verify_database_integrity(temp_path):
                    # Move temp file to final location
                    if os.path.exists(restore_path):
                        os.remove(restore_path)
                    os.rename(temp_path, restore_path)
                    
                    logger.info(f"Successfully restored backup {backup_name} to {restore_path}")
                    return True
                else:
                    logger.error("Restored database failed integrity check")
                    os.remove(temp_path)
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup metadata
        """
        self._load_backup_metadata()
        return list(self.backup_metadata.values())
    
    def cleanup_old_backups(self) -> int:
        """
        Remove backups older than retention period.
        
        Returns:
            Number of backups removed
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        
        with self._lock:
            backups_to_remove = []
            
            for backup_name, metadata in self.backup_metadata.items():
                backup_date = datetime.fromisoformat(metadata['timestamp'])
                if backup_date < cutoff_date:
                    backups_to_remove.append(backup_name)
            
            for backup_name in backups_to_remove:
                if self._remove_backup(backup_name):
                    removed_count += 1
        
        logger.info(f"Cleanup completed: {removed_count} old backups removed")
        return removed_count
    
    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get backup system status.
        
        Returns:
            Backup system status information
        """
        self._load_backup_metadata()
        
        total_backups = len(self.backup_metadata)
        total_size = sum(metadata.get('size_bytes', 0) for metadata in self.backup_metadata.values())
        
        # Find most recent backup
        most_recent = None
        if self.backup_metadata:
            most_recent = max(
                self.backup_metadata.values(),
                key=lambda x: datetime.fromisoformat(x['timestamp'])
            )
        
        return {
            'total_backups': total_backups,
            'total_size_mb': total_size / (1024 * 1024),
            'backup_directory': self.backup_dir,
            'retention_days': self.retention_days,
            'last_backup': most_recent['timestamp'] if most_recent else None,
            'auto_backup_enabled': self._auto_backup_timer is not None,
            'auto_backup_interval_hours': self.auto_backup_hours,
            'next_auto_backup': self._calculate_next_backup_time()
        }
    
    def start_auto_backup(self, data_store_path: str):
        """
        Start automatic backup scheduling.
        
        Args:
            data_store_path: Path to the database to backup
        """
        self.stop_auto_backup()  # Stop any existing timer
        
        def auto_backup_task():
            try:
                backup_name = f"auto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_backup(data_store_path, backup_name)
                self.cleanup_old_backups()
            except Exception as e:
                logger.error(f"Auto backup failed: {e}")
            
            # Schedule next backup
            self._auto_backup_timer = threading.Timer(
                self.auto_backup_hours * 3600,  # Convert hours to seconds
                auto_backup_task
            )
            self._auto_backup_timer.daemon = True
            self._auto_backup_timer.start()
        
        # Start first backup
        self._auto_backup_timer = threading.Timer(5, auto_backup_task)  # Start after 5 seconds
        self._auto_backup_timer.daemon = True
        self._auto_backup_timer.start()
        
        logger.info(f"Auto backup started: every {self.auto_backup_hours} hours")
    
    def stop_auto_backup(self):
        """Stop automatic backup scheduling."""
        if self._auto_backup_timer:
            self._auto_backup_timer.cancel()
            self._auto_backup_timer = None
            logger.info("Auto backup stopped")
    
    def export_analytics_data(self, data_store_path: str, export_format: str = 'json') -> str:
        """
        Export analytics data to a portable format.
        
        Args:
            data_store_path: Path to the SQLite database
            export_format: Export format ('json', 'csv')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f"analytics_export_{timestamp}.{export_format}"
        export_path = os.path.join(self.backup_dir, export_filename)
        
        try:
            with sqlite3.connect(data_store_path) as conn:
                if export_format == 'json':
                    self._export_to_json(conn, export_path)
                elif export_format == 'csv':
                    self._export_to_csv(conn, export_path)
                else:
                    raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Analytics data exported to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")
            raise
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_backup_integrity(self, backup_path: str, expected_checksum: str) -> bool:
        """Verify backup file integrity using checksum."""
        try:
            actual_checksum = self._calculate_file_checksum(backup_path)
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False
    
    def _verify_database_integrity(self, db_path: str) -> bool:
        """Verify SQLite database integrity."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                return result[0] == 'ok'
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return False
    
    def _save_backup_metadata(self):
        """Save backup metadata to file."""
        metadata_path = os.path.join(self.backup_dir, 'backup_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.backup_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    def _load_backup_metadata(self):
        """Load backup metadata from file."""
        metadata_path = os.path.join(self.backup_dir, 'backup_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.backup_metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
                self.backup_metadata = {}
    
    def _remove_backup(self, backup_name: str) -> bool:
        """Remove a backup and its metadata."""
        if backup_name not in self.backup_metadata:
            return False
        
        metadata = self.backup_metadata[backup_name]
        backup_path = metadata['backup_path']
        
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            del self.backup_metadata[backup_name]
            self._save_backup_metadata()
            
            logger.info(f"Backup {backup_name} removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove backup {backup_name}: {e}")
            return False
    
    def _calculate_next_backup_time(self) -> Optional[str]:
        """Calculate when the next auto backup will occur."""
        if not self._auto_backup_timer or not self.last_backup_time:
            return None
        
        next_backup = self.last_backup_time + timedelta(hours=self.auto_backup_hours)
        return next_backup.isoformat()
    
    def _export_to_json(self, conn: sqlite3.Connection, export_path: str):
        """Export database to JSON format."""
        data = {}
        
        # Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table_name in [t[0] for t in tables]:
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            data[table_name] = [
                dict(zip(columns, row)) for row in rows
            ]
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_to_csv(self, conn: sqlite3.Connection, export_path: str):
        """Export database to CSV format."""
        import csv
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        # Create a directory for CSV files
        csv_dir = export_path.replace('.csv', '_csv_export')
        os.makedirs(csv_dir, exist_ok=True)
        
        for table_name in [t[0] for t in tables]:
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            csv_path = os.path.join(csv_dir, f"{table_name}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_auto_backup()