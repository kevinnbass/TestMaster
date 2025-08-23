#!/usr/bin/env python3
"""
🏗️ API COMPONENTS MODULE - Intelligence API Components
==================================================================

📋 PURPOSE:
    Module initialization for API components extracted
    via STEELCLAD protocol from unified_intelligence_api.py

🎯 EXPORTS:
    • RestApiRoutes - REST API endpoint handlers
    • WebSocketEvents - WebSocket event management
    • AgentInitializer - Intelligence agent setup and coordination

🔄 STEELCLAD EXTRACTION:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 MODULAR ARCHITECTURE
   └─ Source: unified_intelligence_api.py (482 lines)
   └─ Target: 3 focused modules + streamlined main file
   └─ Status: EXTRACTION COMPLETE
"""

from .rest_api_routes import RestApiRoutes
from .websocket_events import WebSocketEvents
from .agent_initialization import AgentInitializer

__all__ = [
    'RestApiRoutes',
    'WebSocketEvents', 
    'AgentInitializer'
]