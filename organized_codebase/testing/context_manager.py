"""
Context Preservation System

Inspired by OpenAI Swarm's context preservation patterns.
Provides deep context management for test generation workflows.

Features:
- Deep context copying and preservation
- Context injection into generation pipelines
- Context compression for efficiency
- Context history tracking
- Multi-level context stacks
"""

import copy
import pickle
import gzip
import base64
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import json

from .feature_flags import FeatureFlags
from .shared_state import get_shared_state
from .monitoring_decorators import monitor_performance


@dataclass
class ContextSnapshot:
    """A snapshot of context at a specific point in time."""
    snapshot_id: str
    timestamp: datetime
    context_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    size_bytes: int = 0
    
    def __post_init__(self):
        if not self.size_bytes:
            self.size_bytes = len(str(self.context_data))


@dataclass
class ContextHistory:
    """History of context changes."""
    snapshots: List[ContextSnapshot] = field(default_factory=list)
    max_snapshots: int = 50
    total_changes: int = 0
    
    def add_snapshot(self, snapshot: ContextSnapshot):
        """Add a snapshot to history."""
        self.snapshots.append(snapshot)
        self.total_changes += 1
        
        # Maintain max snapshots
        if len(self.snapshots) > self.max_snapshots:
            # Remove oldest, keeping most recent
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def get_recent(self, count: int = 5) -> List[ContextSnapshot]:
        """Get recent snapshots."""
        return self.snapshots[-count:]


class ContextManager:
    """
    Advanced context preservation system for TestMaster.
    
    Provides OpenAI Swarm-style context preservation with:
    - Deep copying of context objects
    - Context compression for large datasets
    - History tracking and rollback
    - Multi-level context stacks
    - Context injection into pipelines
    """
    
    def __init__(self):
        """Initialize context manager."""
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._context_history: Dict[str, ContextHistory] = {}
        self._context_stack: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._compression_threshold = 10000  # bytes
        self._context_transformers: Dict[str, Callable] = {}
        
        # Get configuration
        if FeatureFlags.is_enabled('layer1_test_foundation', 'context_preservation'):
            config = FeatureFlags.get_config('layer1_test_foundation', 'context_preservation')
            self._deep_copy_enabled = config.get('deep_copy', True)
            self._compression_enabled = config.get('compression', True)
            self._history_enabled = config.get('history', True)
            self._max_context_size = config.get('max_size_mb', 50) * 1024 * 1024  # Convert to bytes
            
            print("Context preservation system initialized")
            print(f"   Deep copy: {'enabled' if self._deep_copy_enabled else 'disabled'}")
            print(f"   Compression: {'enabled' if self._compression_enabled else 'disabled'}")
            print(f"   History: {'enabled' if self._history_enabled else 'disabled'}")
        else:
            self._deep_copy_enabled = False
            self._compression_enabled = False
            self._history_enabled = False
            self._max_context_size = 1024 * 1024  # 1MB default
        
        # Initialize shared state if enabled
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
    
    @monitor_performance(name="context_preserve")
    def preserve(self, context: Dict[str, Any], context_id: str = None) -> Dict[str, Any]:
        """
        Preserve context with deep copying and optional compression.
        
        Args:
            context: Context dictionary to preserve
            context_id: Optional ID for the context
            
        Returns:
            Preserved context dictionary
        """
        with self._lock:
            # Generate context ID if not provided
            if not context_id:
                context_id = f"ctx_{int(datetime.now().timestamp() * 1000)}"
            
            # Deep copy if enabled
            if self._deep_copy_enabled:
                preserved_context = copy.deepcopy(context)
            else:
                preserved_context = context.copy()
            
            # Add preservation metadata
            preserved_context['_preservation'] = {
                'context_id': context_id,
                'preserved_at': datetime.now().isoformat(),
                'deep_copied': self._deep_copy_enabled,
                'original_size': len(str(context)),
                'version': '1.0'
            }
            
            # Check size and compress if needed
            context_str = str(preserved_context)
            if len(context_str) > self._compression_threshold and self._compression_enabled:
                preserved_context = self._compress_context(preserved_context)
            
            # Store context
            self._contexts[context_id] = preserved_context
            
            # Create history snapshot if enabled
            if self._history_enabled:
                snapshot = ContextSnapshot(
                    snapshot_id=f"{context_id}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    context_data=preserved_context.copy(),
                    metadata={'action': 'preserve', 'context_id': context_id},
                    compressed='_compressed' in preserved_context,
                    size_bytes=len(str(preserved_context))
                )
                
                if context_id not in self._context_history:
                    self._context_history[context_id] = ContextHistory()
                
                self._context_history[context_id].add_snapshot(snapshot)
            
            # Update shared state if enabled
            if self.shared_state:
                self.shared_state.set(f"context_{context_id}", {
                    'preserved_at': datetime.now().isoformat(),
                    'size': len(str(preserved_context)),
                    'compressed': '_compressed' in preserved_context
                }, ttl=3600)
                self.shared_state.increment("contexts_preserved")
            
            return preserved_context
    
    @monitor_performance(name="context_retrieve")
    def retrieve(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a preserved context.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            Retrieved context or None if not found
        """
        with self._lock:
            if context_id not in self._contexts:
                return None
            
            context = self._contexts[context_id]
            
            # Decompress if needed
            if '_compressed' in context:
                context = self._decompress_context(context)
            
            return context
    
    def inject_context(self, target_data: str, context: Dict[str, Any]) -> str:
        """
        Inject context into target data (e.g., source code).
        
        Args:
            target_data: Target data to inject context into
            context: Context to inject
            
        Returns:
            Target data with injected context
        """
        if not context:
            return target_data
        
        # Create context injection header
        context_header = self._create_context_header(context)
        
        # Inject context at the beginning
        if target_data.strip().startswith('"""') or target_data.strip().startswith("'''"):
            # Insert after existing docstring
            lines = target_data.split('\n')
            docstring_end = self._find_docstring_end(lines)
            lines.insert(docstring_end + 1, context_header)
            return '\n'.join(lines)
        else:
            # Insert at the beginning
            return f"{context_header}\n{target_data}"
    
    def _create_context_header(self, context: Dict[str, Any]) -> str:
        """Create a context injection header."""
        # Extract relevant context information
        relevant_context = {}
        
        # Add key context items
        for key in ['module_path', 'generation_phase', 'previous_attempts', 'error_history']:
            if key in context:
                relevant_context[key] = context[key]
        
        # Add preservation metadata
        if '_preservation' in context:
            relevant_context['preservation_info'] = context['_preservation']
        
        # Create header comment
        header = '"""\nContext Information:\n'
        for key, value in relevant_context.items():
            if isinstance(value, (str, int, float, bool)):
                header += f"  {key}: {value}\n"
            elif isinstance(value, (list, dict)):
                header += f"  {key}: {json.dumps(value, default=str)[:100]}...\n"
        
        header += '"""\n'
        return header
    
    def _find_docstring_end(self, lines: List[str]) -> int:
        """Find the end of a docstring."""
        in_docstring = False
        quote_type = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                quote_type = stripped[:3]
                in_docstring = True
                if stripped.count(quote_type) >= 2:  # Single line docstring
                    return i
            elif in_docstring and quote_type in line:
                return i
        
        return 0  # Fallback
    
    def push_context(self, context: Dict[str, Any]):
        """Push context onto the context stack."""
        with self._lock:
            if self._deep_copy_enabled:
                self._context_stack.append(copy.deepcopy(context))
            else:
                self._context_stack.append(context.copy())
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop context from the context stack."""
        with self._lock:
            if self._context_stack:
                return self._context_stack.pop()
            return None
    
    def peek_context(self) -> Optional[Dict[str, Any]]:
        """Peek at the top context without removing it."""
        with self._lock:
            if self._context_stack:
                return self._context_stack[-1]
            return None
    
    def merge_contexts(self, *contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple contexts into one.
        
        Args:
            *contexts: Variable number of context dictionaries
            
        Returns:
            Merged context dictionary
        """
        merged = {}
        
        for context in contexts:
            if context:
                # Merge with conflict resolution
                for key, value in context.items():
                    if key in merged:
                        # Handle conflicts
                        if isinstance(merged[key], dict) and isinstance(value, dict):
                            merged[key].update(value)
                        elif isinstance(merged[key], list) and isinstance(value, list):
                            merged[key].extend(value)
                        else:
                            # Keep most recent value
                            merged[key] = value
                    else:
                        merged[key] = value
        
        return merged
    
    def update(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a preserved context.
        
        Args:
            context_id: ID of the context to update
            updates: Updates to apply
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            if context_id not in self._contexts:
                return False
            
            # Decompress if needed
            context = self._contexts[context_id]
            if '_compressed' in context:
                context = self._decompress_context(context)
            
            # Apply updates
            context.update(updates)
            
            # Update preservation metadata
            if '_preservation' in context:
                context['_preservation']['last_updated'] = datetime.now().isoformat()
                context['_preservation']['update_count'] = context['_preservation'].get('update_count', 0) + 1
            
            # Re-compress if needed
            context_str = str(context)
            if len(context_str) > self._compression_threshold and self._compression_enabled:
                context = self._compress_context(context)
            
            # Store updated context
            self._contexts[context_id] = context
            
            # Create history snapshot
            if self._history_enabled:
                snapshot = ContextSnapshot(
                    snapshot_id=f"{context_id}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    context_data=context.copy(),
                    metadata={'action': 'update', 'updates': list(updates.keys())},
                    compressed='_compressed' in context,
                    size_bytes=len(str(context))
                )
                
                self._context_history[context_id].add_snapshot(snapshot)
            
            return True
    
    def _compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress large context data."""
        # Serialize the context
        context_bytes = pickle.dumps(context)
        
        # Compress using gzip
        compressed_bytes = gzip.compress(context_bytes)
        
        # Encode as base64
        encoded = base64.b64encode(compressed_bytes).decode('utf-8')
        
        # Create compressed context
        compressed_context = {
            '_compressed': True,
            'data': encoded,
            'original_size': len(context_bytes),
            'compressed_size': len(encoded),
            'compression_ratio': len(context_bytes) / len(encoded),
            'compressed_at': datetime.now().isoformat()
        }
        
        # Preserve metadata that shouldn't be compressed
        if '_preservation' in context:
            compressed_context['_preservation'] = context['_preservation']
        
        print(f"   Compressed context: {len(context_bytes)} -> {len(encoded)} bytes (ratio: {compressed_context['compression_ratio']:.2f})")
        
        return compressed_context
    
    def _decompress_context(self, compressed_context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress compressed context data."""
        if not compressed_context.get('_compressed'):
            return compressed_context
        
        try:
            # Decode from base64
            encoded = compressed_context['data']
            compressed_bytes = base64.b64decode(encoded)
            
            # Decompress using gzip
            context_bytes = gzip.decompress(compressed_bytes)
            
            # Deserialize the context
            context = SafePickleHandler.safe_load(context_bytes)
            
            return context
            
        except Exception as e:
            print(f"Error decompressing context: {e}")
            return compressed_context
    
    def register_transformer(self, name: str, transformer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a context transformer function.
        
        Args:
            name: Name of the transformer
            transformer: Function that transforms context
        """
        self._context_transformers[name] = transformer
    
    def apply_transformer(self, context: Dict[str, Any], transformer_name: str) -> Dict[str, Any]:
        """
        Apply a registered transformer to context.
        
        Args:
            context: Context to transform
            transformer_name: Name of the transformer to apply
            
        Returns:
            Transformed context
        """
        if transformer_name in self._context_transformers:
            return self._context_transformers[transformer_name](context)
        return context
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context usage."""
        with self._lock:
            total_contexts = len(self._contexts)
            compressed_contexts = sum(1 for ctx in self._contexts.values() if '_compressed' in ctx)
            total_size = sum(len(str(ctx)) for ctx in self._contexts.values())
            
            # Calculate history statistics
            total_snapshots = sum(len(hist.snapshots) for hist in self._context_history.values())
            total_changes = sum(hist.total_changes for hist in self._context_history.values())
            
            return {
                'total_contexts': total_contexts,
                'compressed_contexts': compressed_contexts,
                'compression_rate': (compressed_contexts / max(1, total_contexts)) * 100,
                'total_size_bytes': total_size,
                'average_size_bytes': total_size / max(1, total_contexts),
                'context_stack_depth': len(self._context_stack),
                'total_snapshots': total_snapshots,
                'total_changes': total_changes,
                'registered_transformers': len(self._context_transformers)
            }
    
    def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Clean up old contexts to free memory."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            contexts_to_remove = []
            for context_id, context in self._contexts.items():
                preservation_info = context.get('_preservation', {})
                preserved_at_str = preservation_info.get('preserved_at')
                
                if preserved_at_str:
                    try:
                        preserved_at = datetime.fromisoformat(preserved_at_str)
                        if preserved_at < cutoff_time:
                            contexts_to_remove.append(context_id)
                    except ValueError:
                        pass
            
            # Remove old contexts
            for context_id in contexts_to_remove:
                del self._contexts[context_id]
                if context_id in self._context_history:
                    del self._context_history[context_id]
            
            if contexts_to_remove:
                print(f"Cleaned up {len(contexts_to_remove)} old contexts")
    
    def export_context(self, context_id: str, output_path: str):
        """Export context to a file."""
        context = self.retrieve(context_id)
        if context:
            with open(output_path, 'w') as f:
                json.dump(context, f, indent=2, default=str)
            print(f"Context {context_id} exported to {output_path}")
    
    def import_context(self, input_path: str, context_id: str = None) -> str:
        """Import context from a file."""
        with open(input_path, 'r') as f:
            context = json.load(f)
        
        preserved_context = self.preserve(context, context_id)
        context_id = preserved_context['_preservation']['context_id']
        print(f"Context imported with ID: {context_id}")
        return context_id


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# Convenience functions
def preserve_context(context: Dict[str, Any], context_id: str = None) -> Dict[str, Any]:
    """Preserve context using the global context manager."""
    return get_context_manager().preserve(context, context_id)


def retrieve_context(context_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve context using the global context manager."""
    return get_context_manager().retrieve(context_id)


def inject_context(target_data: str, context: Dict[str, Any]) -> str:
    """Inject context using the global context manager."""
    return get_context_manager().inject_context(target_data, context)


def push_context(context: Dict[str, Any]):
    """Push context onto the global context stack."""
    get_context_manager().push_context(context)


def pop_context() -> Optional[Dict[str, Any]]:
    """Pop context from the global context stack."""
    return get_context_manager().pop_context()