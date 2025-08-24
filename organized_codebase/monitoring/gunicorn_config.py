# Gunicorn Configuration for TestMaster Dashboard
# Production deployment configuration

import multiprocessing
import os

# Server socket configuration
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload application for improved performance
preload_app = True

# Logging configuration
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "testmaster_dashboard"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Performance optimizations
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

# Graceful shutdowns
graceful_timeout = 30

# Development vs Production detection
if os.getenv("ENVIRONMENT") == "development":
    workers = 1
    reload = True
    loglevel = "debug"