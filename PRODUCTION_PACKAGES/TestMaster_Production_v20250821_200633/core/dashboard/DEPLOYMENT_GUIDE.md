# TestMaster Dashboard Production Deployment Guide

Complete guide for deploying the TestMaster Dashboard v2.0 in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Security](#security)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows Server 2019+, or macOS 11+
- **Python**: 3.9+ 
- **RAM**: Minimum 2GB, Recommended 4GB+
- **CPU**: 2+ cores recommended for production
- **Disk**: 10GB+ free space
- **Network**: Port 5000 accessible (or custom port)

### Software Dependencies
- Python 3.9+
- pip package manager
- Git (for deployment)
- Nginx (recommended for reverse proxy)
- systemd (Linux) or Windows Service

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd testmaster/TestMaster/dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Test

```bash
# Test development server
python server.py

# Test WSGI application
python wsgi.py

# Verify health
curl http://localhost:5000/api/health
```

### 3. Production Start

```bash
# Linux/macOS
./start_production.sh

# Windows
start_production.bat
```

## Production Deployment

### Option 1: Direct Gunicorn Deployment

```bash
# Start with Gunicorn directly
gunicorn --config gunicorn_config.py wsgi:application

# Or with custom settings
gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 4 \
  --worker-class sync \
  --timeout 30 \
  --preload \
  --access-logfile - \
  --error-logfile - \
  wsgi:application
```

### Option 2: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app

# Start application
CMD ["gunicorn", "--config", "gunicorn_config.py", "wsgi:application"]
```

Build and run:
```bash
# Build image
docker build -t testmaster-dashboard .

# Run container
docker run -d \
  --name testmaster-dashboard \
  -p 5000:5000 \
  -e ENVIRONMENT=production \
  testmaster-dashboard

# Check status
docker logs testmaster-dashboard
```

### Option 3: Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ENVIRONMENT=production
      - PYTHONPATH=/app
    volumes:
      - ./logs:/app/logs
      - /var/log/testmaster:/var/log/testmaster
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dashboard
    restart: unless-stopped
```

Deploy:
```bash
docker-compose up -d
docker-compose logs -f dashboard
```

## Configuration

### Environment Variables

Create `.env` file:
```env
# Production settings
ENVIRONMENT=production
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/testmaster/dashboard.log

# Performance
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=30

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/testmaster

# External APIs
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

### Gunicorn Configuration

The included `gunicorn_config.py` provides production-ready settings:

```python
# Key settings for production
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
timeout = 30
keepalive = 2
max_requests = 1000
preload_app = True
```

Customize for your environment:
```python
# High-traffic settings
workers = 8
worker_connections = 2000
max_requests = 2000

# Low-resource settings  
workers = 2
worker_connections = 500
max_requests = 500
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/testmaster-dashboard`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Static files (if serving directly)
    location /static/ {
        alias /path/to/testmaster/dashboard/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API-specific settings
    location /api/performance/realtime {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Shorter timeouts for real-time endpoint
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 10s;
    }

    # Health check
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:5000/api/health;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/testmaster-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Security

### SSL/TLS Setup

#### Option 1: Let's Encrypt (Free)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### Option 2: Self-signed (Development)
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Firewall Configuration

```bash
# Ubuntu/Debian with ufw
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Block direct access to application port
sudo ufw deny 5000
```

### Application Security

1. **Secret Key**: Generate strong secret key
```python
import secrets
print(secrets.token_hex(32))
```

2. **API Rate Limiting**: Consider implementing rate limiting
3. **Input Validation**: Already implemented in error handling
4. **CORS**: Configured for production domains
5. **Headers**: Security headers added via Nginx

## Monitoring

### System Service (Linux)

Create `/etc/systemd/system/testmaster-dashboard.service`:
```ini
[Unit]
Description=TestMaster Dashboard
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/testmaster/dashboard
Environment=PATH=/path/to/testmaster/dashboard/venv/bin
Environment=PYTHONPATH=/path/to/testmaster/dashboard
EnvironmentFile=/path/to/testmaster/dashboard/.env
ExecStart=/path/to/testmaster/dashboard/venv/bin/gunicorn --config gunicorn_config.py wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable testmaster-dashboard
sudo systemctl start testmaster-dashboard
sudo systemctl status testmaster-dashboard
```

### Health Checks

```bash
# Simple health check script
#!/bin/bash
# health_check.sh

ENDPOINT="http://localhost:5000/api/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ $RESPONSE -eq 200 ]; then
    echo "Dashboard is healthy"
    exit 0
else
    echo "Dashboard is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

### Logging

Configure log rotation `/etc/logrotate.d/testmaster-dashboard`:
```
/var/log/testmaster/dashboard.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload testmaster-dashboard
    endscript
}
```

### Monitoring with Prometheus/Grafana

Add metrics endpoint to your application and configure Prometheus to scrape:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'testmaster-dashboard'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/api/metrics'
    scrape_interval: 15s
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000
sudo netstat -tulpn | grep :5000

# Kill process
sudo kill -9 <PID>
```

#### 2. Permission Errors
```bash
# Fix file permissions
sudo chown -R www-data:www-data /path/to/testmaster/dashboard
sudo chmod +x start_production.sh
```

#### 3. Import Errors
```bash
# Check Python path
export PYTHONPATH=/path/to/testmaster/dashboard:$PYTHONPATH

# Verify dependencies
pip list
pip check
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Reduce Gunicorn workers
# Edit gunicorn_config.py: workers = 2
```

### Debug Mode

Temporarily enable debug for troubleshooting:
```bash
# Set environment
export FLASK_DEBUG=1
export ENVIRONMENT=development

# Start with debug logging
gunicorn --config gunicorn_config.py --log-level debug wsgi:application
```

### Log Analysis

```bash
# Monitor logs in real-time
tail -f /var/log/testmaster/dashboard.log

# Search for errors
grep -i error /var/log/testmaster/dashboard.log

# Performance analysis
grep "Response-Time" /var/log/testmaster/dashboard.log | tail -100
```

## Maintenance

### Updates

```bash
# Backup current version
sudo systemctl stop testmaster-dashboard
cp -r /path/to/testmaster/dashboard /path/to/dashboard-backup-$(date +%Y%m%d)

# Pull updates
cd /path/to/testmaster/dashboard
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl start testmaster-dashboard
sudo systemctl status testmaster-dashboard
```

### Database Maintenance

If using a database:
```bash
# Backup database
pg_dump testmaster > backup-$(date +%Y%m%d).sql

# Vacuum and analyze
psql testmaster -c "VACUUM ANALYZE;"
```

### Performance Tuning

#### Monitor Performance
```bash
# Check Gunicorn worker status
curl http://localhost:5000/api/performance/status

# Monitor system resources
htop
iotop
```

#### Optimize Settings
1. **Workers**: Start with `(2 × CPU cores) + 1`
2. **Memory**: Monitor with `ps aux --sort=-%mem`
3. **Connections**: Tune based on concurrent users
4. **Timeouts**: Adjust based on response times

### Backup Strategy

```bash
# Create backup script
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/testmaster-dashboard"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application
tar -czf $BACKUP_DIR/dashboard-$DATE.tar.gz /path/to/testmaster/dashboard

# Backup logs
cp /var/log/testmaster/dashboard.log $BACKUP_DIR/dashboard-$DATE.log

# Keep only last 30 days
find $BACKUP_DIR -name "dashboard-*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "dashboard-*.log" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/dashboard-$DATE.tar.gz"
```

Schedule with cron:
```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

## Performance Benchmarks

Expected performance on recommended hardware:

- **Concurrent Users**: 100+ simultaneous users
- **Response Time**: < 100ms for API endpoints
- **Real-time Updates**: 100ms collection interval maintained
- **Memory Usage**: ~200MB per worker process
- **CPU Usage**: <50% under normal load

## Support

### Health Checks
1. **Application**: `curl http://localhost:5000/api/health`
2. **Performance**: `curl http://localhost:5000/api/performance/status`
3. **LLM**: `curl http://localhost:5000/api/llm/status`

### Contact
- Check application logs: `/var/log/testmaster/dashboard.log`
- Review system logs: `journalctl -u testmaster-dashboard`
- Monitor resource usage: `htop`, `free -h`, `df -h`

## Scaling

For high-traffic deployments:

1. **Load Balancer**: Use Nginx/HAProxy with multiple Gunicorn instances
2. **Database**: Separate database server with connection pooling
3. **Cache**: Redis for session/data caching
4. **CDN**: CloudFlare/AWS CloudFront for static assets
5. **Monitoring**: Comprehensive monitoring with Prometheus/Grafana

This completes the production deployment guide for TestMaster Dashboard v2.0.