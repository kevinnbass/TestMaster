#!/usr/bin/env python3
"""
Debug the template issue by comparing what's served vs what's in the variable
"""

import enhanced_linkage_dashboard
from flask import Flask, render_template_string

# Check the actual template content
template = enhanced_linkage_dashboard.ENHANCED_DASHBOARD_HTML
print(f"Template length: {len(template)}")
print(f"Template ends with: {template[-200:]}")
print(f"Contains tab-navigation: {'tab-navigation' in template}")

# Test with regular Flask
app = Flask(__name__)

@app.route('/')
def test():
    return render_template_string(template)

if __name__ == '__main__':
    print("Starting debug server on port 5002...")
    app.run(port=5002, debug=False)