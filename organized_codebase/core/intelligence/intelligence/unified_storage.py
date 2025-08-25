"""
Unified Analysis Result Storage System

Provides centralized storage, indexing, and retrieval of analysis results
from all C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.intelligence subsystems with versioning, caching, and compression.
"""

import json
import pickle
import gzip
import sqlite3
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import uuid


class StorageBackend(Enum):
    """Storage backend types."""
    SQLITE = "sqlite"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    HYBRID = "hybrid"


class CompressionType(Enum):
    """Compression types for storage."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
    JSON_GZIP = "json_gzip"


class ResultStatus(Enum):
    """Status of stored results."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    CORRUPTED = "corrupted"


@dataclass
class StorageConfig:
    """Configuration for unified storage system."""
    backend: StorageBackend = StorageBackend.HYBRID
    base_path: str = "./analysis_storage"
    compression: CompressionType = CompressionType.JSON_GZIP
    max_memory_cache_mb: int = 512
    auto_archive_days: int = 30
    auto_cleanup_days: int = 90
    enable_indexing: bool = True
    enable_versioning: bool = True
    backup_interval_hours: int = 24


@dataclass 
class ResultMetadata:
    """Metadata for stored analysis results."""
    result_id: str
    analysis_type: str
    project_path: str
    timestamp: datetime
    size_bytes: int
    compression: CompressionType
    version: int = 1
    status: ResultStatus = ResultStatus.ACTIVE
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)  # Other result IDs
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "analysis_type": self.analysis_type,
            "project_path": self.project_path,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "compression": self.compression.value,
            "version": self.version,
            "status": self.status.value,
            "tags": list(self.tags),
            "dependencies": self.dependencies,
            "checksum": self.checksum
        }


class UnifiedAnalysisStorage:
    """
    Unified storage system for all analysis results.
    
    Features:
    - Multiple storage backends (SQLite, file system, memory, hybrid)
    - Compression and versioning
    - Indexing and fast search
    - Automatic archiving and cleanup
    - Result dependencies and relationships
    - Data integrity validation
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage backends
        self.db_connection: Optional[sqlite3.Connection] = None
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.file_cache: Dict[str, str] = {}  # result_id -> file_path
        
        # Metadata and indexing
        self.metadata_index: Dict[str, ResultMetadata] = {}
        self.type_index: Dict[str, Set[str]] = {}  # analysis_type -> result_ids
        self.project_index: Dict[str, Set[str]] = {}  # project_path -> result_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> result_ids
        
        # Performance tracking
        self.performance_metrics = {
            "total_stored_results": 0,
            "total_size_bytes": 0,
            "cache_hit_rate": 0.0,
            "average_retrieval_time_ms": 0.0,
            "compression_ratio": 0.0
        }
        
        # System state
        self.is_initialized = False
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize storage
        asyncio.create_task(self._initialize_storage())
    
    async def _initialize_storage(self):
        """Initialize the storage system."""
        try:
            # Create base directory
            Path(self.config.base_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite database
            if self.config.backend in [StorageBackend.SQLITE, StorageBackend.HYBRID]:
                await self._initialize_database()
            
            # Load existing metadata
            await self._load_existing_metadata()
            
            # Start background tasks
            if self.config.auto_cleanup_days > 0:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.is_initialized = True
            self.logger.info("Unified storage system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize SQLite database."""
        db_path = Path(self.config.base_path) / "analysis_results.db"
        self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        cursor = self.db_connection.cursor()
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS result_metadata (
                result_id TEXT PRIMARY KEY,
                analysis_type TEXT NOT NULL,
                project_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                compression TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                tags TEXT DEFAULT '[]',
                dependencies TEXT DEFAULT '[]',
                checksum TEXT,
                file_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Results table for small results stored directly
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS result_data (
                result_id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                FOREIGN KEY (result_id) REFERENCES result_metadata (result_id)
            )
        """)
        
        # Indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_type ON result_metadata (analysis_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_path ON result_metadata (project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON result_metadata (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON result_metadata (status)")
        
        self.db_connection.commit()
        self.logger.info("Database initialized")
    
    async def _load_existing_metadata(self):
        """Load existing metadata into memory indexes."""
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT * FROM result_metadata WHERE status != 'deleted'")
            
            for row in cursor.fetchall():
                metadata = self._row_to_metadata(row)
                self._update_indexes(metadata)
            
            self.logger.info(f"Loaded {len(self.metadata_index)} existing results")
    
    def _row_to_metadata(self, row) -> ResultMetadata:
        """Convert database row to ResultMetadata."""
        return ResultMetadata(
            result_id=row[0],
            analysis_type=row[1],
            project_path=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            size_bytes=row[4],
            compression=CompressionType(row[5]),
            version=row[6],
            status=ResultStatus(row[7]),
            tags=set(json.loads(row[8])),
            dependencies=json.loads(row[9]),
            checksum=row[10]
        )
    
    async def store_result(self, 
                          analysis_type: str,
                          project_path: str,
                          result_data: Any,
                          tags: Optional[Set[str]] = None,
                          dependencies: Optional[List[str]] = None,
                          result_id: Optional[str] = None) -> str:
        """Store an analysis result."""
        if not self.is_initialized:
            await self._initialize_storage()
        
        # Generate result ID
        if not result_id:
            result_id = str(uuid.uuid4())
        
        # Serialize and compress data
        serialized_data, compression_used = await self._serialize_data(result_data)
        
        # Create metadata
        metadata = ResultMetadata(
            result_id=result_id,
            analysis_type=analysis_type,
            project_path=project_path,
            timestamp=datetime.now(),
            size_bytes=len(serialized_data),
            compression=compression_used,
            tags=tags or set(),
            dependencies=dependencies or [],
            checksum=hashlib.sha256(serialized_data).hexdigest()
        )
        
        # Store based on backend configuration
        await self._store_data(result_id, serialized_data, metadata)
        
        # Update indexes
        self._update_indexes(metadata)
        
        # Update performance metrics
        self.performance_metrics["total_stored_results"] += 1
        self.performance_metrics["total_size_bytes"] += metadata.size_bytes
        
        self.logger.info(f"Stored result {result_id} ({analysis_type})")
        return result_id
    
    async def _serialize_data(self, data: Any) -> Tuple[bytes, CompressionType]:
        """Serialize and compress data based on configuration."""
        if self.config.compression == CompressionType.NONE:
            if isinstance(data, bytes):
                return data, CompressionType.NONE
            else:
                return json.dumps(data).encode(), CompressionType.NONE
        
        elif self.config.compression == CompressionType.JSON_GZIP:
            json_data = json.dumps(data, default=str).encode()
            compressed_data = gzip.compress(json_data)
            return compressed_data, CompressionType.JSON_GZIP
        
        elif self.config.compression == CompressionType.PICKLE:
            pickled_data = pickle.dumps(data)
            return pickled_data, CompressionType.PICKLE
        
        elif self.config.compression == CompressionType.GZIP:
            if isinstance(data, str):
                data = data.encode()
            elif not isinstance(data, bytes):
                data = str(data).encode()
            compressed_data = gzip.compress(data)
            return compressed_data, CompressionType.GZIP
        
        else:
            raise ValueError(f"Unsupported compression type: {self.config.compression}")
    
    async def _store_data(self, result_id: str, data: bytes, metadata: ResultMetadata):
        """Store data based on backend configuration."""
        if self.config.backend == StorageBackend.MEMORY:
            # Store in memory cache
            self.memory_cache[result_id] = (data, datetime.now())
            
        elif self.config.backend == StorageBackend.FILE_SYSTEM:
            # Store as file
            file_path = await self._store_as_file(result_id, data)
            self.file_cache[result_id] = file_path
            
        elif self.config.backend == StorageBackend.SQLITE:
            # Store in database
            await self._store_in_database(result_id, data, metadata)
            
        elif self.config.backend == StorageBackend.HYBRID:
            # Use hybrid approach based on size
            if len(data) < 1024 * 1024:  # < 1MB store in database
                await self._store_in_database(result_id, data, metadata)
            else:  # >= 1MB store as file
                file_path = await self._store_as_file(result_id, data)
                self.file_cache[result_id] = file_path
                
                # Store metadata in database with file reference
                await self._store_metadata_in_database(metadata, file_path)
        
        # Also store in memory cache if within limits
        current_cache_size = sum(len(data) for data, _ in self.memory_cache.values())
        max_cache_size = self.config.max_memory_cache_mb * 1024 * 1024
        
        if current_cache_size + len(data) <= max_cache_size:
            self.memory_cache[result_id] = (data, datetime.now())
        elif len(data) < max_cache_size // 10:  # Store small items, evict others
            self._evict_from_memory_cache(len(data))
            self.memory_cache[result_id] = (data, datetime.now())
    
    async def _store_as_file(self, result_id: str, data: bytes) -> str:
        """Store data as a file."""
        file_path = Path(self.config.base_path) / "results" / f"{result_id}.data"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(data)
        
        return str(file_path)
    
    async def _store_in_database(self, result_id: str, data: bytes, metadata: ResultMetadata):
        """Store data and metadata in database."""
        cursor = self.db_connection.cursor()
        
        # Store metadata
        cursor.execute("""
            INSERT OR REPLACE INTO result_metadata 
            (result_id, analysis_type, project_path, timestamp, size_bytes, 
             compression, version, status, tags, dependencies, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.result_id,
            metadata.analysis_type,
            metadata.project_path,
            metadata.timestamp.isoformat(),
            metadata.size_bytes,
            metadata.compression.value,
            metadata.version,
            metadata.status.value,
            json.dumps(list(metadata.tags)),
            json.dumps(metadata.dependencies),
            metadata.checksum
        ))
        
        # Store data
        cursor.execute("""
            INSERT OR REPLACE INTO result_data (result_id, data)
            VALUES (?, ?)
        """, (result_id, data))
        
        self.db_connection.commit()
    
    async def _store_metadata_in_database(self, metadata: ResultMetadata, file_path: str):
        """Store only metadata in database with file reference."""
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO result_metadata 
            (result_id, analysis_type, project_path, timestamp, size_bytes, 
             compression, version, status, tags, dependencies, checksum, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.result_id,
            metadata.analysis_type,
            metadata.project_path,
            metadata.timestamp.isoformat(),
            metadata.size_bytes,
            metadata.compression.value,
            metadata.version,
            metadata.status.value,
            json.dumps(list(metadata.tags)),
            json.dumps(metadata.dependencies),
            metadata.checksum,
            file_path
        ))
        
        self.db_connection.commit()
    
    def _evict_from_memory_cache(self, needed_space: int):
        """Evict items from memory cache to make space."""
        # Simple LRU eviction
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1][1]  # Sort by timestamp
        )
        
        freed_space = 0
        for result_id, (data, _) in sorted_items:
            if freed_space >= needed_space:
                break
            
            freed_space += len(data)
            del self.memory_cache[result_id]
    
    def _update_indexes(self, metadata: ResultMetadata):
        """Update in-memory indexes."""
        self.metadata_index[metadata.result_id] = metadata
        
        # Type index
        if metadata.analysis_type not in self.type_index:
            self.type_index[metadata.analysis_type] = set()
        self.type_index[metadata.analysis_type].add(metadata.result_id)
        
        # Project index
        if metadata.project_path not in self.project_index:
            self.project_index[metadata.project_path] = set()
        self.project_index[metadata.project_path].add(metadata.result_id)
        
        # Tag index
        for tag in metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(metadata.result_id)
    
    async def retrieve_result(self, result_id: str) -> Optional[Any]:
        """Retrieve an analysis result by ID."""
        if result_id not in self.metadata_index:
            return None
        
        metadata = self.metadata_index[result_id]
        
        # Check memory cache first
        if result_id in self.memory_cache:
            data, _ = self.memory_cache[result_id]
            return await self._deserialize_data(data, metadata.compression)
        
        # Try other storage backends
        data = await self._retrieve_data(result_id, metadata)
        if data is None:
            return None
        
        # Add to memory cache if possible
        current_cache_size = sum(len(d) for d, _ in self.memory_cache.values())
        max_cache_size = self.config.max_memory_cache_mb * 1024 * 1024
        
        if current_cache_size + len(data) <= max_cache_size:
            self.memory_cache[result_id] = (data, datetime.now())
        
        return await self._deserialize_data(data, metadata.compression)
    
    async def _retrieve_data(self, result_id: str, metadata: ResultMetadata) -> Optional[bytes]:
        """Retrieve raw data from storage."""
        if self.config.backend == StorageBackend.FILE_SYSTEM or result_id in self.file_cache:
            # Load from file
            if result_id in self.file_cache:
                file_path = self.file_cache[result_id]
            else:
                file_path = Path(self.config.base_path) / "results" / f"{result_id}.data"
            
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    return f.read()
        
        if self.config.backend in [StorageBackend.SQLITE, StorageBackend.HYBRID]:
            # Load from database
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT data FROM result_data WHERE result_id = ?", (result_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            
            # Check if it's stored as file (hybrid mode)
            cursor.execute("SELECT file_path FROM result_metadata WHERE result_id = ?", (result_id,))
            row = cursor.fetchone()
            if row and row[0]:
                file_path = row[0]
                if Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        return f.read()
        
        return None
    
    async def _deserialize_data(self, data: bytes, compression: CompressionType) -> Any:
        """Deserialize and decompress data."""
        if compression == CompressionType.NONE:
            try:
                return json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return data
        
        elif compression == CompressionType.JSON_GZIP:
            decompressed = gzip.decompress(data)
            return json.loads(decompressed.decode())
        
        elif compression == CompressionType.PICKLE:
            return SafePickleHandler.safe_load(data)
        
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data).decode()
        
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def search_results(self, 
                      analysis_type: Optional[str] = None,
                      project_path: Optional[str] = None,
                      tags: Optional[Set[str]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      limit: int = 100) -> List[ResultMetadata]:
        """Search for results based on criteria."""
        candidate_ids = set(self.metadata_index.keys())
        
        # Filter by analysis type
        if analysis_type and analysis_type in self.type_index:
            candidate_ids &= self.type_index[analysis_type]
        elif analysis_type:
            return []  # No results for unknown type
        
        # Filter by project path
        if project_path and project_path in self.project_index:
            candidate_ids &= self.project_index[project_path]
        elif project_path:
            return []  # No results for unknown project
        
        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tag_index:
                    candidate_ids &= self.tag_index[tag]
                else:
                    return []  # No results if any tag is missing
        
        # Filter by time range
        results = []
        for result_id in candidate_ids:
            metadata = self.metadata_index[result_id]
            
            if start_time and metadata.timestamp < start_time:
                continue
            if end_time and metadata.timestamp > end_time:
                continue
            
            results.append(metadata)
        
        # Sort by timestamp (newest first) and limit
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]
    
    async def delete_result(self, result_id: str, permanent: bool = False):
        """Delete a result (soft delete by default)."""
        if result_id not in self.metadata_index:
            return False
        
        metadata = self.metadata_index[result_id]
        
        if permanent:
            # Permanent deletion
            await self._permanent_delete(result_id, metadata)
            self._remove_from_indexes(result_id, metadata)
        else:
            # Soft delete
            metadata.status = ResultStatus.DELETED
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "UPDATE result_metadata SET status = ? WHERE result_id = ?",
                    (ResultStatus.DELETED.value, result_id)
                )
                self.db_connection.commit()
        
        # Remove from memory cache
        if result_id in self.memory_cache:
            del self.memory_cache[result_id]
        
        self.logger.info(f"Deleted result {result_id} (permanent: {permanent})")
        return True
    
    async def _permanent_delete(self, result_id: str, metadata: ResultMetadata):
        """Permanently delete a result from all storage."""
        # Delete from database
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM result_data WHERE result_id = ?", (result_id,))
            cursor.execute("DELETE FROM result_metadata WHERE result_id = ?", (result_id,))
            self.db_connection.commit()
        
        # Delete file if exists
        if result_id in self.file_cache:
            file_path = Path(self.file_cache[result_id])
            if file_path.exists():
                file_path.unlink()
            del self.file_cache[result_id]
        
        # Remove from memory
        if result_id in self.memory_cache:
            del self.memory_cache[result_id]
    
    def _remove_from_indexes(self, result_id: str, metadata: ResultMetadata):
        """Remove result from all indexes."""
        if result_id in self.metadata_index:
            del self.metadata_index[result_id]
        
        # Remove from type index
        if metadata.analysis_type in self.type_index:
            self.type_index[metadata.analysis_type].discard(result_id)
            if not self.type_index[metadata.analysis_type]:
                del self.type_index[metadata.analysis_type]
        
        # Remove from project index
        if metadata.project_path in self.project_index:
            self.project_index[metadata.project_path].discard(result_id)
            if not self.project_index[metadata.project_path]:
                del self.project_index[metadata.project_path]
        
        # Remove from tag indexes
        for tag in metadata.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(result_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
    
    async def _cleanup_loop(self):
        """Background cleanup of old results."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config.auto_cleanup_days)
                archive_time = datetime.now() - timedelta(days=self.config.auto_archive_days)
                
                cleaned_count = 0
                archived_count = 0
                
                for result_id, metadata in list(self.metadata_index.items()):
                    if metadata.timestamp < cutoff_time and metadata.status == ResultStatus.DELETED:
                        await self._permanent_delete(result_id, metadata)
                        self._remove_from_indexes(result_id, metadata)
                        cleaned_count += 1
                    elif metadata.timestamp < archive_time and metadata.status == ResultStatus.ACTIVE:
                        metadata.status = ResultStatus.ARCHIVED
                        archived_count += 1
                
                if cleaned_count > 0 or archived_count > 0:
                    self.logger.info(f"Cleanup: removed {cleaned_count}, archived {archived_count}")
                
                # Sleep for an hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics."""
        active_results = sum(1 for m in self.metadata_index.values() if m.status == ResultStatus.ACTIVE)
        archived_results = sum(1 for m in self.metadata_index.values() if m.status == ResultStatus.ARCHIVED)
        deleted_results = sum(1 for m in self.metadata_index.values() if m.status == ResultStatus.DELETED)
        
        total_size = sum(m.size_bytes for m in self.metadata_index.values())
        cache_size = sum(len(data) for data, _ in self.memory_cache.values())
        
        return {
            "total_results": len(self.metadata_index),
            "active_results": active_results,
            "archived_results": archived_results,
            "deleted_results": deleted_results,
            "total_size_bytes": total_size,
            "memory_cache_size_bytes": cache_size,
            "memory_cache_entries": len(self.memory_cache),
            "file_cache_entries": len(self.file_cache),
            "analysis_types": len(self.type_index),
            "projects": len(self.project_index),
            "tags": len(self.tag_index),
            "performance_metrics": self.performance_metrics.copy()
        }
    
    async def backup_metadata(self, backup_path: str):
        """Backup metadata to a file."""
        backup_data = {
            "metadata": {rid: meta.to_dict() for rid, meta in self.metadata_index.items()},
            "backup_timestamp": datetime.now().isoformat(),
            "config": {
                "backend": self.config.backend.value,
                "compression": self.config.compression.value
            }
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        self.logger.info(f"Metadata backed up to {backup_path}")
    
    async def close(self):
        """Close storage system and cleanup resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        if self.db_connection:
            self.db_connection.close()
        
        self.memory_cache.clear()
        self.file_cache.clear()
        self.metadata_index.clear()
        
        self.logger.info("Storage system closed")


# Factory function
def create_unified_storage(backend: str = "hybrid", 
                         base_path: str = "./analysis_storage",
                         **kwargs) -> UnifiedAnalysisStorage:
    """Create a unified storage instance with configuration."""
    config = StorageConfig(
        backend=StorageBackend(backend),
        base_path=base_path,
        **kwargs
    )
    return UnifiedAnalysisStorage(config)