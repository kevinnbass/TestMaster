"""
CrewAI Derived Flow Persistence Security
Extracted from CrewAI flow persistence patterns and state management
Enhanced for secure state persistence and transaction management
"""

import logging
import sqlite3
import json
import uuid
import threading
from typing import Dict, Any, Optional, List, Union, Type
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from .error_handler import SecurityError, ValidationError, security_error_handler

@dataclass
class FlowState:
    """Flow state representation based on CrewAI patterns"""
    flow_id: str
    state_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flow state to dictionary for serialization"""
        return {
            'flow_id': self.flow_id,
            'state_data': self.state_data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlowState':
        """Create flow state from dictionary"""
        return cls(
            flow_id=data['flow_id'],
            state_data=data['state_data'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', 1),
            metadata=data.get('metadata', {})
        )


@dataclass
class PersistenceConfig:
    """Persistence configuration settings"""
    database_path: str = "flow_persistence.db"
    enable_encryption: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_state_size_mb: int = 10
    compression_enabled: bool = True
    transaction_timeout: int = 30
    
    def __post_init__(self):
        if self.max_state_size_mb <= 0:
            raise ValidationError("max_state_size_mb must be positive")
        if self.transaction_timeout <= 0:
            raise ValidationError("transaction_timeout must be positive")


class SecureSQLitePersistence:
    """Secure SQLite-based persistence system based on CrewAI patterns"""
    
    def __init__(self, config: PersistenceConfig = None):
        self.config = config or PersistenceConfig()
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(self.config.database_path)
        self._lock = threading.RLock()
        self._connection_cache = threading.local()
        
        # Initialize database
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._connection_cache, 'connection'):
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.config.transaction_timeout,
                check_same_thread=False
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
            self._connection_cache.connection = conn
        
        return self._connection_cache.connection
    
    def _initialize_database(self):
        """Initialize database schema"""
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Create flows table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS flows (
                        flow_id TEXT PRIMARY KEY,
                        state_data TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        metadata TEXT,
                        checksum TEXT,
                        size_bytes INTEGER
                    )
                """)
                
                # Create flow history table for versioning
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS flow_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        flow_id TEXT NOT NULL,
                        state_data TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        operation TEXT NOT NULL,
                        checksum TEXT,
                        FOREIGN KEY (flow_id) REFERENCES flows (flow_id)
                    )
                """)
                
                # Create indices for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_flows_updated_at ON flows (updated_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_flow_id ON flow_history (flow_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_version ON flow_history (flow_id, version)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            error = SecurityError(f"Failed to initialize database: {str(e)}", "PERSISTENCE_INIT_001")
            security_error_handler.handle_error(error)
            raise error
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction rolled back: {str(e)}")
                raise
    
    def save_flow_state(self, flow_state: FlowState) -> bool:
        """Save flow state with security validation"""
        try:
            # Validate state size
            state_json = json.dumps(flow_state.state_data)
            state_size_mb = len(state_json.encode('utf-8')) / (1024 * 1024)
            
            if state_size_mb > self.config.max_state_size_mb:
                raise ValidationError(
                    f"State size {state_size_mb:.2f}MB exceeds limit {self.config.max_state_size_mb}MB"
                )
            
            # Generate checksum for integrity
            checksum = self._calculate_checksum(state_json)
            
            # Serialize metadata
            metadata_json = json.dumps(flow_state.metadata) if flow_state.metadata else "{}"
            
            current_time = datetime.now(timezone.utc)
            
            with self.transaction() as conn:
                # Check if flow exists
                existing = conn.execute(
                    "SELECT version FROM flows WHERE flow_id = ?",
                    (flow_state.flow_id,)
                ).fetchone()
                
                if existing:
                    # Update existing flow
                    new_version = existing[0] + 1
                    
                    # Save to history first
                    conn.execute("""
                        INSERT INTO flow_history 
                        (flow_id, state_data, version, created_at, operation, checksum)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        flow_state.flow_id, state_json, new_version,
                        current_time, "update", checksum
                    ))
                    
                    # Update main table
                    conn.execute("""
                        UPDATE flows 
                        SET state_data = ?, updated_at = ?, version = ?, 
                            metadata = ?, checksum = ?, size_bytes = ?
                        WHERE flow_id = ?
                    """, (
                        state_json, current_time, new_version,
                        metadata_json, checksum, len(state_json.encode('utf-8')),
                        flow_state.flow_id
                    ))
                    
                    flow_state.version = new_version
                    
                else:
                    # Insert new flow
                    conn.execute("""
                        INSERT INTO flows 
                        (flow_id, state_data, created_at, updated_at, version, 
                         metadata, checksum, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        flow_state.flow_id, state_json, current_time, current_time,
                        1, metadata_json, checksum, len(state_json.encode('utf-8'))
                    ))
                    
                    # Save to history
                    conn.execute("""
                        INSERT INTO flow_history 
                        (flow_id, state_data, version, created_at, operation, checksum)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        flow_state.flow_id, state_json, 1,
                        current_time, "create", checksum
                    ))
                
                flow_state.updated_at = current_time
                
                self.logger.info(f"Flow state saved: {flow_state.flow_id} (version {flow_state.version})")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to save flow state: {str(e)}", "PERSISTENCE_SAVE_001")
            security_error_handler.handle_error(error)
            return False
    
    def load_flow_state(self, flow_id: str, version: Optional[int] = None) -> Optional[FlowState]:
        """Load flow state with integrity verification"""
        try:
            conn = self._get_connection()
            
            if version is None:
                # Load latest version
                row = conn.execute("""
                    SELECT state_data, created_at, updated_at, version, metadata, checksum
                    FROM flows WHERE flow_id = ?
                """, (flow_id,)).fetchone()
            else:
                # Load specific version from history
                row = conn.execute("""
                    SELECT state_data, created_at, created_at, version, '{}', checksum
                    FROM flow_history 
                    WHERE flow_id = ? AND version = ?
                """, (flow_id, version)).fetchone()
            
            if not row:
                self.logger.warning(f"Flow state not found: {flow_id}")
                return None
            
            state_json, created_at, updated_at, version_num, metadata_json, stored_checksum = row
            
            # Verify integrity
            calculated_checksum = self._calculate_checksum(state_json)
            if calculated_checksum != stored_checksum:
                error = SecurityError(
                    f"Flow state integrity check failed for {flow_id}",
                    "PERSISTENCE_INTEGRITY_001"
                )
                security_error_handler.handle_error(error)
                return None
            
            # Deserialize data
            try:
                state_data = json.loads(state_json)
                metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError as e:
                error = SecurityError(
                    f"Failed to deserialize flow state {flow_id}: {str(e)}",
                    "PERSISTENCE_DESERIALIZE_001"
                )
                security_error_handler.handle_error(error)
                return None
            
            flow_state = FlowState(
                flow_id=flow_id,
                state_data=state_data,
                created_at=datetime.fromisoformat(created_at.replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(updated_at.replace('Z', '+00:00')),
                version=version_num,
                metadata=metadata
            )
            
            self.logger.info(f"Flow state loaded: {flow_id} (version {version_num})")
            return flow_state
            
        except Exception as e:
            error = SecurityError(f"Failed to load flow state: {str(e)}", "PERSISTENCE_LOAD_001")
            security_error_handler.handle_error(error)
            return None
    
    def delete_flow_state(self, flow_id: str) -> bool:
        """Delete flow state with history preservation"""
        try:
            with self.transaction() as conn:
                # Check if flow exists
                existing = conn.execute(
                    "SELECT 1 FROM flows WHERE flow_id = ?",
                    (flow_id,)
                ).fetchone()
                
                if not existing:
                    self.logger.warning(f"Flow not found for deletion: {flow_id}")
                    return False
                
                # Archive in history before deletion
                conn.execute("""
                    INSERT INTO flow_history 
                    (flow_id, state_data, version, created_at, operation, checksum)
                    SELECT flow_id, state_data, version, ?, 'delete', checksum
                    FROM flows WHERE flow_id = ?
                """, (datetime.now(timezone.utc), flow_id))
                
                # Delete from main table
                conn.execute("DELETE FROM flows WHERE flow_id = ?", (flow_id,))
                
                self.logger.info(f"Flow state deleted: {flow_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Failed to delete flow state: {str(e)}", "PERSISTENCE_DELETE_001")
            security_error_handler.handle_error(error)
            return False
    
    def list_flows(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List flows with pagination"""
        try:
            conn = self._get_connection()
            
            rows = conn.execute("""
                SELECT flow_id, created_at, updated_at, version, size_bytes
                FROM flows 
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
            
            flows = []
            for row in rows:
                flows.append({
                    'flow_id': row[0],
                    'created_at': row[1],
                    'updated_at': row[2],
                    'version': row[3],
                    'size_bytes': row[4]
                })
            
            return flows
            
        except Exception as e:
            self.logger.error(f"Failed to list flows: {str(e)}")
            return []
    
    def get_flow_history(self, flow_id: str) -> List[Dict[str, Any]]:
        """Get flow version history"""
        try:
            conn = self._get_connection()
            
            rows = conn.execute("""
                SELECT version, created_at, operation, checksum
                FROM flow_history 
                WHERE flow_id = ?
                ORDER BY version DESC
            """, (flow_id,)).fetchall()
            
            history = []
            for row in rows:
                history.append({
                    'version': row[0],
                    'created_at': row[1],
                    'operation': row[2],
                    'checksum': row[3]
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get flow history: {str(e)}")
            return []
    
    def cleanup_old_history(self, days_to_keep: int = 30) -> int:
        """Clean up old history entries"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            with self.transaction() as conn:
                result = conn.execute("""
                    DELETE FROM flow_history 
                    WHERE created_at < ? AND operation != 'delete'
                """, (cutoff_date,))
                
                deleted_count = result.rowcount
                
                self.logger.info(f"Cleaned up {deleted_count} old history entries")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup history: {str(e)}")
            return 0
    
    def get_persistence_statistics(self) -> Dict[str, Any]:
        """Get persistence system statistics"""
        try:
            conn = self._get_connection()
            
            # Flow statistics
            flow_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_flows,
                    AVG(size_bytes) as avg_size_bytes,
                    MAX(size_bytes) as max_size_bytes,
                    SUM(size_bytes) as total_size_bytes,
                    MAX(version) as max_version
                FROM flows
            """).fetchone()
            
            # History statistics
            history_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_history_entries,
                    COUNT(DISTINCT flow_id) as flows_with_history
                FROM flow_history
            """).fetchone()
            
            # Database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'total_flows': flow_stats[0] or 0,
                'average_size_bytes': flow_stats[1] or 0,
                'maximum_size_bytes': flow_stats[2] or 0,
                'total_storage_bytes': flow_stats[3] or 0,
                'maximum_version': flow_stats[4] or 0,
                'total_history_entries': history_stats[0] or 0,
                'flows_with_history': history_stats[1] or 0,
                'database_size_bytes': db_size,
                'database_path': str(self.db_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA-256 checksum for integrity verification"""
        import hashlib
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def close(self):
        """Close database connections"""
        try:
            if hasattr(self._connection_cache, 'connection'):
                self._connection_cache.connection.close()
                delattr(self._connection_cache, 'connection')
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


class FlowPersistenceSecurityManager:
    """Flow persistence security management system"""
    
    def __init__(self, config: PersistenceConfig = None):
        self.config = config or PersistenceConfig()
        self.logger = logging.getLogger(__name__)
        self.persistence = SecureSQLitePersistence(config)
        
        # Security monitoring
        self.operation_history: List[Dict[str, Any]] = []
        self.max_operation_history = 1000
    
    def create_flow(self, flow_data: Dict[str, Any], 
                   flow_id: Optional[str] = None,
                   metadata: Dict[str, Any] = None) -> Optional[FlowState]:
        """Create new flow with security validation"""
        try:
            # Generate secure flow ID if not provided
            if not flow_id:
                flow_id = str(uuid.uuid4())
            
            # Validate flow ID format
            try:
                uuid.UUID(flow_id)
            except ValueError:
                raise ValidationError(f"Invalid flow ID format: {flow_id}")
            
            # Create flow state
            current_time = datetime.now(timezone.utc)
            flow_state = FlowState(
                flow_id=flow_id,
                state_data=flow_data,
                created_at=current_time,
                updated_at=current_time,
                version=1,
                metadata=metadata or {}
            )
            
            # Save flow state
            if self.persistence.save_flow_state(flow_state):
                self._record_operation("create", flow_id, {"success": True})
                return flow_state
            else:
                self._record_operation("create", flow_id, {"success": False})
                return None
                
        except Exception as e:
            self._record_operation("create", flow_id or "unknown", {"success": False, "error": str(e)})
            error = SecurityError(f"Failed to create flow: {str(e)}", "FLOW_CREATE_001")
            security_error_handler.handle_error(error)
            return None
    
    def update_flow(self, flow_id: str, flow_data: Dict[str, Any],
                   metadata: Dict[str, Any] = None) -> Optional[FlowState]:
        """Update existing flow with version control"""
        try:
            # Load existing flow
            existing_flow = self.persistence.load_flow_state(flow_id)
            if not existing_flow:
                self._record_operation("update", flow_id, {"success": False, "error": "not_found"})
                return None
            
            # Update flow state
            current_time = datetime.now(timezone.utc)
            existing_flow.state_data = flow_data
            existing_flow.updated_at = current_time
            
            if metadata:
                existing_flow.metadata.update(metadata)
            
            # Save updated flow
            if self.persistence.save_flow_state(existing_flow):
                self._record_operation("update", flow_id, {"success": True, "version": existing_flow.version})
                return existing_flow
            else:
                self._record_operation("update", flow_id, {"success": False})
                return None
                
        except Exception as e:
            self._record_operation("update", flow_id, {"success": False, "error": str(e)})
            error = SecurityError(f"Failed to update flow: {str(e)}", "FLOW_UPDATE_001")
            security_error_handler.handle_error(error)
            return None
    
    def get_flow(self, flow_id: str, version: Optional[int] = None) -> Optional[FlowState]:
        """Get flow with security logging"""
        try:
            flow_state = self.persistence.load_flow_state(flow_id, version)
            
            if flow_state:
                self._record_operation("read", flow_id, {"success": True, "version": version})
            else:
                self._record_operation("read", flow_id, {"success": False, "not_found": True})
            
            return flow_state
            
        except Exception as e:
            self._record_operation("read", flow_id, {"success": False, "error": str(e)})
            error = SecurityError(f"Failed to get flow: {str(e)}", "FLOW_GET_001")
            security_error_handler.handle_error(error)
            return None
    
    def delete_flow(self, flow_id: str) -> bool:
        """Delete flow with security auditing"""
        try:
            success = self.persistence.delete_flow_state(flow_id)
            self._record_operation("delete", flow_id, {"success": success})
            return success
            
        except Exception as e:
            self._record_operation("delete", flow_id, {"success": False, "error": str(e)})
            error = SecurityError(f"Failed to delete flow: {str(e)}", "FLOW_DELETE_001")
            security_error_handler.handle_error(error)
            return False
    
    def list_flows(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List flows with access logging"""
        try:
            flows = self.persistence.list_flows(limit, offset)
            self._record_operation("list", "all", {"success": True, "count": len(flows)})
            return flows
            
        except Exception as e:
            self._record_operation("list", "all", {"success": False, "error": str(e)})
            return []
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            # Get persistence statistics
            persistence_stats = self.persistence.get_persistence_statistics()
            
            # Analyze operation history
            recent_operations = self.operation_history[-100:] if self.operation_history else []
            
            operation_counts = {}
            error_counts = {}
            
            for op in recent_operations:
                op_type = op.get('operation', 'unknown')
                operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
                
                if not op.get('context', {}).get('success', True):
                    error_counts[op_type] = error_counts.get(op_type, 0) + 1
            
            success_rates = {}
            for op_type, total in operation_counts.items():
                errors = error_counts.get(op_type, 0)
                success_rates[op_type] = ((total - errors) / total) * 100 if total > 0 else 100
            
            return {
                'persistence_statistics': persistence_stats,
                'operation_counts': operation_counts,
                'error_counts': error_counts,
                'success_rates_pct': success_rates,
                'total_operations': len(self.operation_history),
                'recent_operations': len(recent_operations),
                'database_healthy': self._check_database_health()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting security status: {e}")
            return {'error': str(e)}
    
    def _record_operation(self, operation: str, flow_id: str, context: Dict[str, Any]):
        """Record operation for security auditing"""
        operation_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation,
            'flow_id': flow_id,
            'context': context
        }
        
        self.operation_history.append(operation_record)
        
        # Limit operation history
        if len(self.operation_history) > self.max_operation_history:
            self.operation_history = self.operation_history[-self.max_operation_history // 2:]
    
    def _check_database_health(self) -> bool:
        """Check database health"""
        try:
            # Simple health check by querying database
            flows = self.persistence.list_flows(limit=1)
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close persistence manager and cleanup resources"""
        try:
            self.persistence.close()
            self.logger.info("Flow persistence security manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing persistence manager: {e}")


# Global flow persistence security manager
flow_persistence_security = FlowPersistenceSecurityManager()


# Convenience functions
def create_secure_flow(flow_data: Dict[str, Any], 
                      flow_id: Optional[str] = None,
                      metadata: Dict[str, Any] = None) -> Optional[FlowState]:
    """Convenience function to create secure flow"""
    return flow_persistence_security.create_flow(flow_data, flow_id, metadata)


def get_secure_flow(flow_id: str, version: Optional[int] = None) -> Optional[FlowState]:
    """Convenience function to get secure flow"""
    return flow_persistence_security.get_flow(flow_id, version)


def update_secure_flow(flow_id: str, flow_data: Dict[str, Any],
                      metadata: Dict[str, Any] = None) -> Optional[FlowState]:
    """Convenience function to update secure flow"""
    return flow_persistence_security.update_flow(flow_id, flow_data, metadata)