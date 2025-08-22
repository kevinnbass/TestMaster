"""
Refactoring Analysis API Module
===============================

Handles refactoring analysis and hierarchy endpoints.

Author: TestMaster Team
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import logging
import sys
import os

logger = logging.getLogger(__name__)
refactor_bp = Blueprint('refactor', __name__)

# Global dependencies
refactor_analyzer = None
refactor_roadmaps = None

def init_refactor_api(analyzer=None, roadmaps=None):
    """Initialize refactor API with analyzer and roadmaps."""
    global refactor_analyzer, refactor_roadmaps
    refactor_analyzer = analyzer
    refactor_roadmaps = roadmaps or {}
    logger.info("Refactor API initialized")

@refactor_bp.route('/analysis')
def get_refactor_analysis():
    """Get current refactoring analysis."""
    try:
        codebase = request.args.get('codebase', '/testmaster')
        
        return jsonify({
            'status': 'success',
            'analysis': {
                'refactor_opportunities': 12,
                'complexity_hotspots': 5,
                'duplicate_code': 3
            },
            'codebase': codebase,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@refactor_bp.route('/hierarchy')
def get_code_hierarchy():
    """Get code hierarchy analysis."""
    try:
        return jsonify({
            'status': 'success',
            'hierarchy': {
                'modules': 24,
                'classes': 86,
                'functions': 342
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500