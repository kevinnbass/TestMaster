#!/usr/bin/env python3
"""
Quantum Retry Engine Database Operations
========================================

Database operations and state management for the quantum retry engine.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import sqlite3
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from retry_models import (
    QuantumRetryContext,
    QuantumRetryAttempt,
    RetryStrategy,
    FailurePattern,
    RetryPriority,
    RetryStatistics
)

logger = logging.getLogger(__name__)


class QuantumRetryDatabase:
    """Handles all database operations for quantum retry engine."""

    def __init__(self, db_path: str):
        """Initialize the database handler."""
        self.db_path = db_path
        self._init_quantum_database()

    def _init_quantum_database(self):
        """Initialize quantum retry database with advanced schema."""
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                # Quantum contexts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quantum_contexts (
                        operation_id TEXT PRIMARY KEY,
                        operation_data TEXT NOT NULL,
                        current_strategy TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        max_attempts INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        last_attempt TEXT,
                        failure_pattern TEXT,
                        success_probability REAL,
                        predicted_next_success TEXT,
                        quantum_entanglement TEXT
                    )
                ''')

                # Quantum attempts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quantum_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        attempt_number INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        delay_used REAL NOT NULL,
                        priority INTEGER NOT NULL,
                        success INTEGER NOT NULL,
                        error_info TEXT,
                        processing_time REAL,
                        quantum_state TEXT,
                        FOREIGN KEY (operation_id) REFERENCES quantum_contexts (operation_id)
                    )
                ''')

                # Quantum patterns table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quantum_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        failure_pattern TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        detected_at TEXT NOT NULL,
                        operation_count INTEGER NOT NULL,
                        success_rate REAL NOT NULL,
                        recommended_strategy TEXT,
                        quantum_signature TEXT
                    )
                ''')

                # Quantum entanglements table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quantum_entanglements (
                        entanglement_id TEXT PRIMARY KEY,
                        operation_a TEXT NOT NULL,
                        operation_b TEXT NOT NULL,
                        correlation_strength REAL NOT NULL,
                        discovered_at TEXT NOT NULL,
                        entanglement_type TEXT,
                        stability REAL DEFAULT 1.0
                    )
                ''')

                # Create advanced indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_quantum_timestamp ON quantum_attempts(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_quantum_operation ON quantum_attempts(operation_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_quantum_pattern ON quantum_patterns(detected_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_quantum_entanglement ON quantum_entanglements(correlation_strength)')

                conn.commit()

        except Exception as e:
            logger.error(f"Quantum database initialization failed: {e}")
            raise

    def save_context(self, context: QuantumRetryContext):
        """Save quantum context to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO quantum_contexts
                    (operation_id, operation_data, current_strategy, priority, max_attempts,
                     created_at, last_attempt, failure_pattern, success_probability,
                     predicted_next_success, quantum_entanglement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    context.operation_id,
                    json.dumps(context.operation_data),
                    context.current_strategy.value,
                    context.priority.value,
                    context.max_attempts,
                    context.created_at.isoformat(),
                    context.last_attempt.isoformat() if context.last_attempt else None,
                    context.failure_pattern.value,
                    context.success_probability,
                    context.predicted_next_success.isoformat() if context.predicted_next_success else None,
                    json.dumps(context.quantum_entanglement) if context.quantum_entanglement else '{}'
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save quantum context: {e}")

    def save_attempt(self, attempt: QuantumRetryAttempt):
        """Save quantum attempt to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO quantum_attempts
                    (attempt_id, operation_id, strategy, attempt_number, timestamp,
                     delay_used, priority, success, error_info, processing_time, quantum_state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    attempt.attempt_id,
                    attempt.operation_id,
                    attempt.strategy.value,
                    attempt.attempt_number,
                    attempt.timestamp.isoformat(),
                    attempt.delay_used,
                    attempt.priority.value,
                    1 if attempt.success else 0,
                    attempt.error_info,
                    attempt.processing_time,
                    json.dumps(attempt.quantum_state) if attempt.quantum_state else '{}'
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save quantum attempt: {e}")

    def load_context(self, operation_id: str) -> Optional[QuantumRetryContext]:
        """Load quantum context from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM quantum_contexts WHERE operation_id = ?
                ''', (operation_id,))

                row = cursor.fetchone()
                if row:
                    # Load failure history
                    failure_history = self._load_attempt_history(operation_id)

                    return QuantumRetryContext(
                        operation_id=row[0],
                        operation_data=json.loads(row[1]),
                        failure_history=failure_history,
                        current_strategy=RetryStrategy(row[2]),
                        priority=RetryPriority(row[3]),
                        max_attempts=row[4],
                        created_at=datetime.fromisoformat(row[5]),
                        last_attempt=datetime.fromisoformat(row[6]) if row[6] else None,
                        failure_pattern=FailurePattern(row[7]) if row[7] else FailurePattern.UNKNOWN,
                        success_probability=row[8] or 0.5,
                        predicted_next_success=datetime.fromisoformat(row[9]) if row[9] else None,
                        quantum_entanglement=json.loads(row[10]) if row[10] else {}
                    )

        except Exception as e:
            logger.error(f"Failed to load quantum context: {e}")

        return None

    def _load_attempt_history(self, operation_id: str) -> List[QuantumRetryAttempt]:
        """Load attempt history for an operation."""
        attempts = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM quantum_attempts
                    WHERE operation_id = ?
                    ORDER BY attempt_number
                ''', (operation_id,))

                for row in cursor.fetchall():
                    attempts.append(QuantumRetryAttempt(
                        attempt_id=row[0],
                        operation_id=row[1],
                        strategy=RetryStrategy(row[2]),
                        attempt_number=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        delay_used=row[5],
                        priority=RetryPriority(row[6]),
                        success=bool(row[7]),
                        error_info=row[8],
                        processing_time=row[9] or 0.0,
                        quantum_state=json.loads(row[10]) if row[10] else {}
                    ))

        except Exception as e:
            logger.error(f"Failed to load attempt history: {e}")

        return attempts

    def get_quantum_statistics(self) -> RetryStatistics:
        """Get comprehensive quantum retry statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic operation statistics
                cursor = conn.execute('''
                    SELECT
                        COUNT(DISTINCT operation_id) as total_operations,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_operations,
                        COUNT(CASE WHEN success = 0 THEN 1 END) as failed_operations,
                        AVG(attempt_number) as average_attempts,
                        AVG(processing_time) as average_processing_time
                    FROM quantum_attempts
                ''')

                row = cursor.fetchone()
                stats = RetryStatistics(
                    total_operations=row[0] or 0,
                    successful_operations=row[1] or 0,
                    failed_operations=row[2] or 0,
                    average_attempts=row[3] or 0.0,
                    average_processing_time=row[4] or 0.0
                )

                # Strategy success rates
                cursor = conn.execute('''
                    SELECT strategy,
                           AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM quantum_attempts
                    GROUP BY strategy
                ''')

                for row in cursor.fetchall():
                    strategy = RetryStrategy(row[0])
                    stats.strategy_success_rates[strategy] = row[1] or 0.0

                # Pattern distribution
                cursor = conn.execute('''
                    SELECT failure_pattern, COUNT(*) as count
                    FROM quantum_contexts
                    WHERE failure_pattern IS NOT NULL
                    GROUP BY failure_pattern
                ''')

                for row in cursor.fetchall():
                    pattern = FailurePattern(row[0])
                    stats.pattern_distribution[pattern] = row[1]

                # Priority distribution
                cursor = conn.execute('''
                    SELECT priority, COUNT(*) as count
                    FROM quantum_contexts
                    GROUP BY priority
                ''')

                for row in cursor.fetchall():
                    priority = RetryPriority(row[0])
                    stats.priority_distribution[priority] = row[1]

                return stats

        except Exception as e:
            logger.error(f"Failed to get quantum statistics: {e}")
            return RetryStatistics()

    def cleanup_old_data(self, days: int = 30):
        """Clean up old quantum retry data."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                # Delete old attempts
                conn.execute('''
                    DELETE FROM quantum_attempts
                    WHERE timestamp < ?
                ''', (cutoff_date,))

                # Delete old contexts that have no attempts
                conn.execute('''
                    DELETE FROM quantum_contexts
                    WHERE operation_id NOT IN (
                        SELECT DISTINCT operation_id FROM quantum_attempts
                    )
                ''')

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    def get_active_operations(self) -> List[str]:
        """Get list of active operation IDs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT operation_id FROM quantum_contexts
                    ORDER BY created_at DESC
                ''')

                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get active operations: {e}")
            return []
