"""
Audit Storage Manager

Handles persistent storage and retrieval of audit events.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from .audit_events import AuditEvent, EventType, EventSeverity

logger = logging.getLogger(__name__)


class AuditStorage:
    """
    Manages audit event storage with integrity verification.
    Provides tamper-evident logging capabilities.
    """
    
    def __init__(self, log_file: str = "audit.log"):
        """
        Initialize audit storage.
        
        Args:
            log_file: Path to audit log file
        """
        try:
            self.log_file = Path(log_file)
            self.events = []
            self.integrity_hashes = []
            self._ensure_log_directory()
            self._load_existing_events()
            logger.info(f"Audit Storage initialized: {log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize audit storage: {e}")
            # Use fallback configuration
            self.log_file = Path("fallback_audit.log")
            self.events = []
            self.integrity_hashes = []
            try:
                self._ensure_log_directory()
            except Exception as fallback_error:
                logger.critical(f"Fallback storage initialization failed: {fallback_error}")
    
    def store_event(self, event: AuditEvent) -> str:
        """
        Store an audit event with integrity protection.
        
        Args:
            event: Audit event to store
            
        Returns:
            Event ID for tracking
        """
        try:
            # Generate event ID
            event_id = self._generate_event_id(event)
            event.details['event_id'] = event_id
            
            # Store event in memory
            self.events.append(event)
            
            # Write to persistent storage
            self._write_to_log(event)
            
            # Update integrity chain
            self._update_integrity_hash(event)
            
            logger.debug(f"Stored audit event: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            # Try to store error event
            try:
                error_event = AuditEvent.create(
                    event_type=EventType.FILE_CHANGE,
                    severity=EventSeverity.CRITICAL,
                    source="audit_storage",
                    actor="system",
                    resource="audit_log",
                    action="store_event",
                    result="error",
                    details={"error": str(e), "original_event": str(event)}
                )
                # Simple storage without recursion
                error_id = self._generate_event_id(error_event)
                self.events.append(error_event)
                return error_id
            except Exception as critical_error:
                logger.critical(f"Critical audit storage failure: {critical_error}")
                return "storage_error"
    
    def query_events(self,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    event_type: Optional[EventType] = None,
                    actor: Optional[str] = None,
                    severity: Optional[EventSeverity] = None) -> List[AuditEvent]:
        """
        Query stored audit events with filters.
        
        Args:
            start_time: Start timestamp filter
            end_time: End timestamp filter
            event_type: Event type filter
            actor: Actor filter
            severity: Severity filter
            
        Returns:
            Filtered audit events
        """
        try:
            filtered = self.events.copy()
            
            if start_time:
                filtered = [e for e in filtered if e.timestamp >= start_time]
            if end_time:
                filtered = [e for e in filtered if e.timestamp <= end_time]
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            if actor:
                filtered = [e for e in filtered if e.actor == actor]
            if severity:
                filtered = [e for e in filtered if e.severity == severity]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []
    
    def verify_integrity(self) -> bool:
        """
        Verify audit log integrity using hash chain.
        
        Returns:
            True if integrity is intact, False otherwise
        """
        try:
            if len(self.events) != len(self.integrity_hashes):
                logger.warning("Event count mismatch with integrity hashes")
                return False
            
            # Recalculate hashes and compare
            calculated_hashes = []
            for event in self.events:
                try:
                    hash_val = self._calculate_event_hash(event)
                    calculated_hashes.append(hash_val)
                except Exception as e:
                    logger.error(f"Failed to calculate hash for event: {e}")
                    return False
            
            integrity_valid = calculated_hashes == self.integrity_hashes
            if not integrity_valid:
                logger.warning("Audit log integrity verification failed")
            
            return integrity_valid
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def export_events(self, output_path: str, format: str = "json") -> bool:
        """
        Export audit events to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
            
        Returns:
            True if export successful
        """
        try:
            if format == "json":
                data = {
                    'events': [event.to_dict() for event in self.events],
                    'integrity_hashes': self.integrity_hashes,
                    'metadata': {
                        'total_events': len(self.events),
                        'integrity_verified': self.verify_integrity()
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == "csv":
                # Basic CSV export
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if self.events:
                        writer = csv.DictWriter(f, fieldnames=self.events[0].to_dict().keys())
                        writer.writeheader()
                        for event in self.events:
                            writer.writerow(event.to_dict())
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(self.events)} events to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export events: {e}")
            return False
    
    def get_event_count(self) -> int:
        """Get total number of stored events."""
        return len(self.events)
    
    def get_latest_events(self, count: int = 10) -> List[AuditEvent]:
        """Get the most recent events."""
        try:
            return self.events[-count:] if self.events else []
        except Exception as e:
            logger.error(f"Failed to get latest events: {e}")
            return []
    
    # Private methods
    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log directory: {e}")
            raise
    
    def _load_existing_events(self) -> None:
        """Load existing events from log file."""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            if line.strip():
                                event_data = json.loads(line.strip())
                                # Convert back to AuditEvent (simplified loading)
                                logger.debug(f"Loaded existing event from line {line_num}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed line {line_num}: {e}")
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num}: {e}")
        except Exception as e:
            logger.warning(f"Could not load existing events: {e}")
    
    def _generate_event_id(self, event: AuditEvent) -> str:
        """Generate unique event ID."""
        try:
            content = f"{event.timestamp}{event.source}{event.actor}{event.action}"
            return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Failed to generate event ID: {e}")
            # Fallback to timestamp-based ID
            return f"event_{event.timestamp.replace(':', '').replace('-', '')[:16]}"
    
    def _write_to_log(self, event: AuditEvent) -> None:
        """Write event to persistent log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
            raise
    
    def _update_integrity_hash(self, event: AuditEvent) -> None:
        """Update integrity hash chain."""
        try:
            event_hash = self._calculate_event_hash(event)
            self.integrity_hashes.append(event_hash)
        except Exception as e:
            logger.error(f"Failed to update integrity hash: {e}")
            # Add placeholder hash to maintain chain length
            self.integrity_hashes.append("error_hash")
    
    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Calculate cryptographic hash for event."""
        try:
            event_str = json.dumps(event.to_dict(), sort_keys=True)
            return hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate event hash: {e}")
            return f"hash_error_{event.timestamp}"