#!/usr/bin/env python3
"""
Quick test to verify template content
"""

from flask import Flask, render_template_string
import sys
from pathlib import Path

# Import the template from the main file
sys.path.insert(0, str(Path(__file__).parent))
from enhanced_linkage_dashboard import ENHANCED_DASHBOARD_HTML

app = Flask(__name__)

@app.route('/')
def test():
    print(f"Template length: {len(ENHANCED_DASHBOARD_HTML)}")
    print(f"Contains tab-navigation: {'tab-navigation' in ENHANCED_DASHBOARD_HTML}")
    print(f"Contains d3js: {'d3js.org' in ENHANCED_DASHBOARD_HTML}")
    return render_template_string(ENHANCED_DASHBOARD_HTML)

if __name__ == '__main__':
    app.run(debug=True, port=5001)