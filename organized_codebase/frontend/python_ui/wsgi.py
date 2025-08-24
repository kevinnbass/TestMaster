#!/usr/bin/env python3
"""
WSGI Entry Point for TestMaster Dashboard
==========================================

Production WSGI application entry point for Gunicorn deployment.

Usage:
    gunicorn --config gunicorn_config.py wsgi:application
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Add parent TestMaster directory for imports
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/testmaster/dashboard.log', 'a') if os.path.exists('/var/log/testmaster/') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_application():
    """
    Application factory for WSGI deployment.
    
    Returns:
        Flask application instance configured for production
    """
    try:
        # Import after path setup
        from server import create_app
        
        # Create Flask application
        app = create_app()
        
        # Production optimizations
        app.config.update(
            DEBUG=False,
            TESTING=False,
            ENV='production',
            SECRET_KEY=os.environ.get('SECRET_KEY', 'prod-secret-key-change-me'),
            # Disable Flask development server warnings
            TEMPLATES_AUTO_RELOAD=False,
            SEND_FILE_MAX_AGE_DEFAULT=31536000,  # 1 year cache for static files
        )
        
        logger.info("TestMaster Dashboard WSGI application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create WSGI application: {e}")
        raise

# Create the WSGI application
try:
    application = create_application()
    logger.info("WSGI application ready for deployment")
except Exception as e:
    logger.error(f"WSGI initialization failed: {e}")
    # Create a minimal error application
    from flask import Flask, jsonify
    application = Flask(__name__)
    
    @application.route('/')
    def error_page():
        return jsonify({
            'error': 'Dashboard initialization failed',
            'status': 'error',
            'message': str(e)
        }), 503

if __name__ == '__main__':
    # For development testing
    application.run(host='0.0.0.0', port=5000, debug=False)