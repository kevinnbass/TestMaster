#!/usr/bin/env python3
"""Test new routes are registered"""

from web_monitor import WebMonitoringServer

# Create an instance
server = WebMonitoringServer(port=8082)

print("Testing route registration...")
print("=" * 60)

# Check all routes
routes = []
for rule in server.app.url_map.iter_rules():
    routes.append(str(rule))

# Check for new routes
new_routes = ['/api/llm/list-modules', '/api/llm/estimate-cost']
for route in new_routes:
    if route in routes:
        print(f"[OK] {route} is registered")
    else:
        print(f"[MISSING] {route} not found")

print("\nAll registered routes:")
for route in sorted(routes):
    if '/api/' in route:
        print(f"  {route}")

print("\n[SUCCESS] Routes test complete")