# TestMaster Setup & Configuration Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)
- Optional: Virtual environment tool (venv, conda)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/testmaster.git
cd testmaster

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=sqlite:///testmaster.db
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=5000
DEBUG_MODE=False

# Testing Configuration
TEST_COVERAGE_THRESHOLD=80
MAX_PARALLEL_TESTS=4
TEST_TIMEOUT=300

# Monitoring Configuration
MONITORING_ENABLED=True
MONITORING_INTERVAL=60
ALERT_EMAIL=admin@example.com

# Performance Configuration
CACHE_ENABLED=True
CACHE_TTL=3600
MAX_WORKERS=8
```

## Configuration Files

### 1. Main Configuration (`config/default.json`)

The main configuration file contains system-wide settings:

```json
{
  "application": {
    "name": "TestMaster",
    "version": "1.0.0",
    "environment": "development"
  },
  "intelligence": {
    "enabled": true,
    "analytics_hub": true,
    "testing_hub": true,
    "integration_hub": true
  },
  "api": {
    "base_url": "http://localhost:5000",
    "timeout": 30,
    "rate_limit": 100
  }
}
```

### 2. Python Configuration (`config/testmaster_config.py`)

Python-based configuration for dynamic settings:

```python
import os
from pathlib import Path

class Config:
    # Base configuration
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # API Configuration
    API_KEY = os.getenv('GEMINI_API_KEY')
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    
    # Testing Configuration
    COVERAGE_THRESHOLD = float(os.getenv('TEST_COVERAGE_THRESHOLD', 80))
    PARALLEL_TESTS = int(os.getenv('MAX_PARALLEL_TESTS', 4))
    
    # Feature Flags
    ENABLE_ML_FEATURES = os.getenv('ENABLE_ML_FEATURES', 'true').lower() == 'true'
    ENABLE_MONITORING = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
```

### 3. YAML Templates (`config/templates/*.yaml`)

Configuration templates for different deployment scenarios:

#### Development Template (`config/templates/deployment_template.yaml`)
```yaml
environment: development
debug: true
database:
  type: sqlite
  path: ./dev.db
logging:
  level: DEBUG
  file: ./logs/dev.log
```

#### Production Template
```yaml
environment: production
debug: false
database:
  type: postgresql
  host: db.example.com
  port: 5432
  name: testmaster_prod
logging:
  level: INFO
  file: /var/log/testmaster/prod.log
```

## Advanced Configuration

### 1. Multi-Environment Setup

Support for multiple environments:

```bash
# Development
export TESTMASTER_ENV=development
python run.py

# Staging
export TESTMASTER_ENV=staging
python run.py

# Production
export TESTMASTER_ENV=production
python run.py
```

### 2. Configuration Hierarchy

Configuration loading order (later overrides earlier):
1. Default configuration (`config/default.json`)
2. Environment-specific config (`config/{env}.json`)
3. Environment variables
4. Command-line arguments

### 3. Dynamic Configuration

Runtime configuration updates:

```python
from config.testmaster_config import Config

# Update configuration at runtime
Config.set('api.timeout', 60)
Config.reload()

# Get configuration value
timeout = Config.get('api.timeout')
```

## Deployment Configuration

### 1. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV TESTMASTER_ENV=production
ENV SERVER_PORT=8080

CMD ["python", "run.py"]
```

### 2. Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: testmaster-config
data:
  default.json: |
    {
      "application": {
        "name": "TestMaster",
        "environment": "kubernetes"
      }
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: testmaster-secrets
type: Opaque
data:
  gemini-api-key: <base64-encoded-key>
  database-password: <base64-encoded-password>
```

### 3. CI/CD Configuration

GitHub Actions example:

```yaml
name: TestMaster CI/CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      TESTMASTER_ENV: ci
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Error: `KeyError: 'GEMINI_API_KEY'`
   - Solution: Ensure `.env` file exists and contains all required variables

2. **Configuration Not Loading**
   - Error: `FileNotFoundError: config/default.json`
   - Solution: Run from project root directory

3. **Permission Errors**
   - Error: `PermissionError: [Errno 13] Permission denied`
   - Solution: Check file permissions and user privileges

### Configuration Validation

Run configuration validation:

```bash
python -m testmaster.utils.validate_config
```

This will check:
- All required environment variables are set
- Configuration files are valid JSON/YAML
- File permissions are correct
- Dependencies are installed

## Security Considerations

### 1. Sensitive Data

Never commit sensitive data:
- Add `.env` to `.gitignore`
- Use environment variables for secrets
- Rotate API keys regularly

### 2. Configuration Encryption

For sensitive configuration:

```python
from cryptography.fernet import Fernet

# Encrypt configuration
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted_config = cipher.encrypt(config_json.encode())

# Decrypt configuration
decrypted_config = cipher.decrypt(encrypted_config)
```

### 3. Access Control

Limit configuration access:
- Use file permissions (chmod 600)
- Implement role-based access
- Audit configuration changes

---

*Configuration & Setup Documentation - TestMaster Intelligence Framework*
*Last Updated: {datetime.now().isoformat()}*
