"""
Context Manager Module
======================
Manages execution context for TestMaster components.
"""

import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages execution context and state."""
    
    def __init__(self):
        """Initialize context manager."""
        self.contexts = {}
        self.current_context = {}
        self.lock = threading.Lock()
        logger.info("Context Manager initialized")
    
    def create_context(self, context_id: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new context."""
        with self.lock:
            context = {
                'id': context_id,
                'created_at': datetime.now().isoformat(),
                'data': data or {},
                'status': 'active'
            }
            self.contexts[context_id] = context
            return context
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get a context by ID."""
        return self.contexts.get(context_id)
    
    def update_context(self, context_id: str, data: Dict[str, Any]) -> bool:
        """Update context data."""
        with self.lock:
            if context_id in self.contexts:
                self.contexts[context_id]['data'].update(data)
                self.contexts[context_id]['updated_at'] = datetime.now().isoformat()
                return True
            return False
    
    def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        with self.lock:
            if context_id in self.contexts:
                del self.contexts[context_id]
                return True
            return False
    
    def set_current(self, context_id: str) -> bool:
        """Set the current active context."""
        if context_id in self.contexts:
            self.current_context = self.contexts[context_id]
            return True
        return False
    
    def get_current(self) -> Dict[str, Any]:
        """Get the current active context."""
        return self.current_context


# Global instance
_context_manager = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# Convenience functions
def create_context(context_id: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new context."""
    return get_context_manager().create_context(context_id, data)


def get_context(context_id: str) -> Optional[Dict[str, Any]]:
    """Get a context by ID."""
    return get_context_manager().get_context(context_id)


def get_current_context() -> Dict[str, Any]:
    """Get the current active context."""
    return get_context_manager().get_current()
