"""
Test Configuration & Setup Documentation
Agent D - Hour 5: Configuration & Setup Documentation
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import configparser

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

def analyze_configuration_files():
    """Analyze all configuration files in TestMaster."""
    
    config_analysis = {
        'json_configs': [],
        'yaml_configs': [],
        'python_configs': [],
        'env_configs': [],
        'ini_configs': [],
        'templates': []
    }
    
    # Scan for configuration files
    testmaster_path = Path("TestMaster")
    
    # JSON configuration files
    for json_file in testmaster_path.rglob("*.json"):
        if "node_modules" not in str(json_file) and ".git" not in str(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                config_analysis['json_configs'].append({
                    'path': str(json_file),
                    'name': json_file.name,
                    'size': json_file.stat().st_size,
                    'keys': list(data.keys()) if isinstance(data, dict) else None,
                    'type': 'json'
                })
            except:
                pass
    
    # YAML configuration files  
    for yaml_file in testmaster_path.rglob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config_analysis['yaml_configs'].append({
                'path': str(yaml_file),
                'name': yaml_file.name,
                'size': yaml_file.stat().st_size,
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'type': 'yaml'
            })
        except:
            pass
    
    for yml_file in testmaster_path.rglob("*.yml"):
        try:
            with open(yml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config_analysis['yaml_configs'].append({
                'path': str(yml_file),
                'name': yml_file.name,
                'size': yml_file.stat().st_size,
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'type': 'yml'
            })
        except:
            pass
    
    # Python configuration files
    for py_file in testmaster_path.glob("**/config/**/*.py"):
        config_analysis['python_configs'].append({
            'path': str(py_file),
            'name': py_file.name,
            'size': py_file.stat().st_size,
            'type': 'python'
        })
    
    # Environment configuration files
    for env_file in testmaster_path.rglob("*.env"):
        config_analysis['env_configs'].append({
            'path': str(env_file),
            'name': env_file.name,
            'size': env_file.stat().st_size,
            'type': 'env'
        })
    
    # INI configuration files
    for ini_file in testmaster_path.rglob("*.ini"):
        config_analysis['ini_configs'].append({
            'path': str(ini_file),
            'name': ini_file.name,
            'size': ini_file.stat().st_size,
            'type': 'ini'
        })
    
    # Template files
    template_dir = testmaster_path / "config" / "templates"
    if template_dir.exists():
        for template_file in template_dir.iterdir():
            if template_file.is_file():
                config_analysis['templates'].append({
                    'path': str(template_file),
                    'name': template_file.name,
                    'size': template_file.stat().st_size,
                    'type': template_file.suffix[1:] if template_file.suffix else 'unknown'
                })
    
    return config_analysis

def analyze_environment_variables():
    """Analyze environment variables used in the project."""
    
    env_vars = set()
    env_usage = {}
    
    # Scan Python files for os.environ usage
    for py_file in Path("TestMaster").rglob("*.py"):
        if "node_modules" not in str(py_file) and ".git" not in str(py_file):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for os.environ patterns
                import re
                
                # Pattern for os.environ['VAR'] or os.environ.get('VAR')
                pattern1 = r"os\.environ\[[\'\"]([A-Z_]+)[\'\"]\]"
                pattern2 = r"os\.environ\.get\([\'\"]([A-Z_]+)[\'\"]"
                pattern3 = r"os\.getenv\([\'\"]([A-Z_]+)[\'\"]"
                
                for pattern in [pattern1, pattern2, pattern3]:
                    matches = re.findall(pattern, content)
                    for var in matches:
                        env_vars.add(var)
                        if var not in env_usage:
                            env_usage[var] = []
                        env_usage[var].append(str(py_file))
            except:
                pass
    
    # Check .env file for defined variables
    env_file = Path("TestMaster/.env")
    defined_vars = {}
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        defined_vars[key.strip()] = value.strip()
        except:
            pass
    
    return {
        'required_vars': list(env_vars),
        'defined_vars': defined_vars,
        'usage_map': env_usage
    }

def generate_setup_documentation():
    """Generate comprehensive setup documentation."""
    
    setup_doc = """# TestMaster Setup & Configuration Guide

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
"""
    
    return setup_doc

def test_configuration_documentation():
    """Test configuration and setup documentation."""
    
    print("=" * 80)
    print("Agent D - Hour 5: Configuration & Setup Documentation")
    print("Testing Configuration Documentation Systems")
    print("=" * 80)
    
    # Analyze configuration files
    print("\n1. Analyzing Configuration Files...")
    config_analysis = analyze_configuration_files()
    
    total_configs = sum(len(configs) for configs in config_analysis.values())
    print(f"   Found {total_configs} configuration files")
    
    for config_type, configs in config_analysis.items():
        if configs:
            print(f"\n   {config_type}:")
            for config in configs[:3]:  # Show first 3
                print(f"   - {config['name']} ({config['size']} bytes)")
                if config.get('keys'):
                    print(f"     Keys: {', '.join(config['keys'][:5])}")
    
    # Analyze environment variables
    print("\n2. Analyzing Environment Variables...")
    env_analysis = analyze_environment_variables()
    
    print(f"   Required Variables: {len(env_analysis['required_vars'])}")
    for var in env_analysis['required_vars'][:10]:  # Show first 10
        print(f"   - {var}")
        if var in env_analysis['defined_vars']:
            print(f"     [OK] Defined in .env")
        else:
            print(f"     [X] Not defined")
    
    # Generate documentation
    print("\n3. Generating Setup Documentation...")
    setup_doc = generate_setup_documentation()
    
    # Create output directory
    output_dir = Path("TestMaster/docs/configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save setup documentation
    setup_path = output_dir / "SETUP_GUIDE.md"
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(setup_doc)
    print(f"   Setup guide: {setup_path}")
    
    # Generate configuration inventory
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Configuration Documentation",
        "summary": {
            "total_config_files": total_configs,
            "json_configs": len(config_analysis['json_configs']),
            "yaml_configs": len(config_analysis['yaml_configs']),
            "python_configs": len(config_analysis['python_configs']),
            "env_configs": len(config_analysis['env_configs']),
            "templates": len(config_analysis['templates'])
        },
        "environment_variables": {
            "required": len(env_analysis['required_vars']),
            "defined": len(env_analysis['defined_vars']),
            "undefined": len(set(env_analysis['required_vars']) - set(env_analysis['defined_vars'].keys()))
        },
        "configuration_files": config_analysis
    }
    
    # Save inventory
    inventory_path = output_dir / "configuration_inventory.json"
    with open(inventory_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    print(f"   Configuration inventory: {inventory_path}")
    
    # Generate environment template
    env_template = "# TestMaster Environment Configuration\n\n"
    for var in sorted(env_analysis['required_vars']):
        value = env_analysis['defined_vars'].get(var, 'your_value_here')
        env_template += f"{var}={value}\n"
    
    env_template_path = output_dir / ".env.template"
    with open(env_template_path, 'w', encoding='utf-8') as f:
        f.write(env_template)
    print(f"   Environment template: {env_template_path}")
    
    print("\n" + "=" * 80)
    print("Configuration Documentation Test Complete!")
    print("=" * 80)
    
    return inventory

if __name__ == "__main__":
    inventory = test_configuration_documentation()