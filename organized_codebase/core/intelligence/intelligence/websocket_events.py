#!/usr/bin/env python3
"""
🏗️ MODULE: WebSocket Events - Real-time Communication Events
==================================================================

📋 PURPOSE:
    WebSocket event handlers and real-time communication functionality
    extracted from unified_intelligence_api.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Client connection handling
    • Real-time metrics subscription
    • Analysis request handling via WebSocket
    • Background task management
    • Event callbacks for task completion

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract WebSocket events from unified_intelligence_api.py
   └─ Source: Lines 393-442 (49 lines)
   └─ Purpose: Separate real-time communication into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: Flask-SocketIO, datetime, logging
📤 Provides: WebSocket event handling and real-time updates
"""

from flask_socketio import emit
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketEvents:
    """Handles WebSocket events for real-time communication."""
    
    def __init__(self, socketio, coordinator, real_time_metrics):
        self.socketio = socketio
        self.coordinator = coordinator
        self.real_time_metrics = real_time_metrics
        
        self.setup_websocket_events()
    
    def setup_websocket_events(self):
        """Setup WebSocket events for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to intelligence API")
            emit('connected', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from intelligence API")
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics():
            """Subscribe to real-time metrics updates"""
            def send_metrics():
                while True:
                    self.socketio.emit('metrics_update', {
                        'metrics': self.real_time_metrics,
                        'agents': self.coordinator.get_system_status(),
                        'timestamp': datetime.now().isoformat()
                    })
                    self.socketio.sleep(2)  # Update every 2 seconds
            
            self.socketio.start_background_task(send_metrics)
        
        @self.socketio.on('start_analysis')
        def handle_start_analysis(data):
            """Handle analysis request via WebSocket"""
            project_path = data.get('project_path', '.')
            analysis_type = data.get('analysis_type', 'comprehensive')
            
            # Submit task
            task_id = self.coordinator.submit_task(
                f'analyze_{analysis_type}',
                f'WebSocket analysis: {analysis_type}',
                {'project_path': project_path},
                'HIGH'  # TaskPriority.HIGH equivalent
            )
            
            emit('analysis_started', {
                'task_id': task_id,
                'analysis_type': analysis_type
            })
    
    def register_task_callbacks(self):
        """Register event callbacks for real-time updates"""
        def on_task_completed(task, agent):
            self.socketio.emit('task_completed', {
                'task_id': task.task_id,
                'agent_id': agent.agent_id,
                'result': task.result
            })
        
        def on_task_failed(task, agent):
            self.socketio.emit('task_failed', {
                'task_id': task.task_id,
                'agent_id': agent.agent_id,
                'error': task.error
            })
        
        self.coordinator.register_event_callback('task_completed', on_task_completed)
        self.coordinator.register_event_callback('task_failed', on_task_failed)