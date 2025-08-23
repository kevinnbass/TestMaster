# LLM Intelligence System - Configuration Guide

This guide provides comprehensive information about configuring the LLM Intelligence System for different use cases, environments, and requirements.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Core Configuration Options](#core-configuration-options)
3. [LLM Provider Configuration](#llm-provider-configuration)
4. [Analysis Configuration](#analysis-configuration)
5. [Integration Configuration](#integration-configuration)
6. [Execution Configuration](#execution-configuration)
7. [Performance Configuration](#performance-configuration)
8. [Sample Configurations](#sample-configurations)
9. [Environment Variables](#environment-variables)
10. [Configuration Validation](#configuration-validation)

## Configuration Overview

The system uses a hierarchical configuration system:

1. **Default Configuration**: Built-in sensible defaults
2. **JSON Configuration Files**: Override defaults with JSON files
3. **Environment Variables**: Override specific values
4. **Runtime Parameters**: Command-line overrides

### Configuration Loading Order

1. Load default configuration
2. Override with JSON configuration file (if specified)
3. Override with environment variables
4. Override with command-line arguments

### Configuration File Format

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "api_key": "sk-your-key",
  "min_confidence_threshold": 0.7,
  "max_concurrent": 3,
  "enable_static_analysis": true,
  "cache_enabled": true,
  "backup_enabled": true
}
```

## Core Configuration Options

### Basic Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `llm_provider` | string | `"mock"` | LLM provider: `"openai"`, `"ollama"`, `"mock"` |
| `llm_model` | string | `"gpt-4"` | Model name for the LLM provider |
| `api_key` | string | `null` | API key for LLM provider |
| `enable_static_analysis` | boolean | `true` | Enable static analysis integration |
| `preserve_directory_order` | boolean | `true` | Maintain exact directory structure |

### File Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_file_size` | integer | `50000` | Maximum file size in bytes |
| `max_lines_per_file` | integer | `1000` | Maximum lines per file |
| `file_encoding` | string | `"utf-8"` | File encoding for reading |
| `exclude_patterns` | array | `[]` | Additional file patterns to exclude |

### Output and Storage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_dir` | string | `"intelligence_output"` | Output directory relative to root |
| `cache_dir` | string | `"llm_cache"` | Cache directory relative to reorganizer |
| `log_dir` | string | `"logs"` | Log directory relative to reorganizer |
| `backup_dir` | string | `"backups"` | Backup directory relative to reorganizer |

## LLM Provider Configuration

### OpenAI Configuration

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "api_key": "sk-your-openai-key",
  "llm_temperature": 0.0,
  "llm_max_tokens": 2000,
  "openai_base_url": "https://api.openai.com/v1",
  "openai_timeout": 60
}
```

**OpenAI Models:**
- `gpt-4`: Most capable, highest cost
- `gpt-4-turbo`: Fast and capable, lower cost
- `gpt-3.5-turbo`: Fast, lower cost, good for bulk analysis

**Cost Optimization:**
```json
{
  "llm_model": "gpt-3.5-turbo",
  "llm_temperature": 0.0,
  "llm_max_tokens": 1500,
  "requests_per_minute": 60
}
```

### Ollama Configuration (Local Models)

```json
{
  "llm_provider": "ollama",
  "llm_model": "llama2:7b",
  "ollama_base_url": "http://localhost:11434",
  "ollama_timeout": 120,
  "ollama_stream": false,
  "ollama_options": {
    "temperature": 0.0,
    "num_predict": 1500
  }
}
```

**Recommended Ollama Models:**
- `codellama:7b`: Optimized for code understanding
- `llama2:13b`: Better reasoning, higher resource usage
- `mistral:7b`: Good balance of capability and speed
- `codellama:13b`: Best code understanding

### Mock Configuration (Testing)

```json
{
  "llm_provider": "mock",
  "mock_response_delay": 0.1,
  "mock_confidence_range": [0.7, 0.95]
}
```

### Anthropic Configuration (Future)

```json
{
  "llm_provider": "anthropic",
  "llm_model": "claude-3-sonnet-20240229",
  "api_key": "sk-ant-your-key",
  "anthropic_max_tokens": 2000,
  "anthropic_temperature": 0.0
}
```

## Analysis Configuration

### Confidence Thresholds

```json
{
  "min_confidence_threshold": 0.7,
  "high_confidence_threshold": 0.85,
  "consensus_threshold": 0.6,
  "low_confidence_threshold": 0.4
}
```

### Analysis Scope

```json
{
  "analyze_imports": true,
  "analyze_comments": true,
  "analyze_docstrings": true,
  "analyze_function_names": true,
  "analyze_class_names": true,
  "analyze_code_patterns": true,
  "analyze_dependencies": true,
  "analyze_complexity": true
}
```

### Static Analysis Integration

```json
{
  "enable_semantic_analysis": true,
  "enable_relationship_analysis": true,
  "enable_pattern_detection": true,
  "enable_quality_analysis": true,
  "static_analysis_timeout": 30
}
```

### LLM Analysis Settings

```json
{
  "llm_analysis_enabled": true,
  "llm_analysis_timeout": 60,
  "retry_attempts": 3,
  "retry_delay": 5,
  "chunk_large_files": true,
  "chunk_size": 4000,
  "chunk_overlap": 200,
  "max_chunks_per_file": 5
}
```

## Integration Configuration

### Integration Methods

```json
{
  "integration_method": "weighted_voting",
  "classification_weights": {
    "llm_confidence": 0.35,
    "semantic_confidence": 0.20,
    "pattern_confidence": 0.15,
    "quality_confidence": 0.15,
    "relationship_confidence": 0.10,
    "agreement_confidence": 0.05
  }
}
```

**Available Integration Methods:**
- `"weighted_voting"`: Default, combines all sources with weights
- `"consensus_with_fallback"`: Requires agreement, falls back to LLM
- `"llm_dominant"`: Uses LLM with static analysis for validation
- `"static_dominant"`: Uses static analysis with LLM enhancement
- `"adaptive_confidence"`: Chooses method based on confidence levels

### Classification Settings

```json
{
  "primary_categories": [
    "security", "intelligence", "frontend_dashboard",
    "documentation", "testing", "utility", "api",
    "database", "data_processing", "orchestration",
    "automation", "monitoring", "analytics", "devops"
  ],
  "enable_secondary_classifications": true,
  "max_secondary_classifications": 3,
  "classification_confidence_threshold": 0.6
}
```

### Validation Settings

```json
{
  "validate_llm_responses": true,
  "require_json_responses": true,
  "repair_invalid_json": true,
  "validate_classifications": true,
  "check_classification_consistency": true,
  "confidence_validation_enabled": true
}
```

## Execution Configuration

### Safety Settings

```json
{
  "dry_run_enabled": true,
  "backup_enabled": true,
  "import_validation_enabled": true,
  "dependency_check_enabled": true,
  "rollback_enabled": true,
  "safety_check_level": "high"
}
```

### Risk Management

```json
{
  "risk_thresholds": {
    "low": 0.8,
    "medium": 0.6,
    "high": 0.4
  },
  "auto_approve_risk_levels": ["low"],
  "require_review_risk_levels": ["high", "critical"],
  "max_auto_operations": 10,
  "require_approval_for_batches": true
}
```

### Execution Control

```json
{
  "execution_mode": "batch",
  "batch_size": 5,
  "batch_timeout": 300,
  "task_timeout": 60,
  "concurrent_tasks": 3,
  "stop_on_first_error": false,
  "continue_on_partial_failure": true
}
```

### Monitoring and Logging

```json
{
  "log_level": "INFO",
  "log_to_file": true,
  "log_to_console": true,
  "enable_metrics": true,
  "metrics_retention_days": 30,
  "alert_on_errors": true,
  "alert_on_low_confidence": true
}
```

## Performance Configuration

### Concurrency and Parallelism

```json
{
  "max_concurrent": 3,
  "max_concurrent_files": 10,
  "thread_pool_size": 5,
  "process_pool_size": 2,
  "use_async_processing": true
}
```

### Caching Configuration

```json
{
  "cache_enabled": true,
  "cache_backend": "file",
  "cache_compression": true,
  "cache_validation_mode": "hash",
  "cache_max_age_days": 30,
  "cache_max_size_gb": 1.0,
  "cache_cleanup_enabled": true
}
```

### Resource Management

```json
{
  "memory_limit_mb": 2048,
  "disk_space_warning_gb": 1.0,
  "cpu_usage_limit_percent": 80,
  "network_timeout_seconds": 30,
  "rate_limit_requests_per_minute": 60
}
```

### Performance Optimization

```json
{
  "enable_streaming": true,
  "enable_compression": true,
  "batch_requests": true,
  "prefetch_data": true,
  "optimize_memory_usage": true,
  "use_memory_mapped_files": true
}
```

## Sample Configurations

### Development Configuration

```json
{
  "llm_provider": "mock",
  "enable_static_analysis": true,
  "min_confidence_threshold": 0.5,
  "max_concurrent": 1,
  "cache_enabled": false,
  "dry_run_enabled": true,
  "log_level": "DEBUG"
}
```

### Production Configuration

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "min_confidence_threshold": 0.7,
  "max_concurrent": 5,
  "cache_enabled": true,
  "backup_enabled": true,
  "enable_static_analysis": true,
  "log_level": "INFO",
  "auto_approve_risk_levels": ["low"],
  "require_review_risk_levels": ["high", "critical"]
}
```

### High-Performance Configuration

```json
{
  "llm_provider": "ollama",
  "llm_model": "codellama:7b",
  "max_concurrent": 10,
  "enable_static_analysis": true,
  "cache_enabled": true,
  "batch_requests": true,
  "optimize_memory_usage": true,
  "log_level": "WARNING"
}
```

### Cost-Effective Configuration

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo",
  "llm_max_tokens": 1500,
  "max_concurrent": 2,
  "requests_per_minute": 30,
  "cache_enabled": true,
  "triage_enabled": true,
  "min_confidence_threshold": 0.75
}
```

### Security-Focused Configuration

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "min_confidence_threshold": 0.8,
  "enable_static_analysis": true,
  "auto_approve_risk_levels": [],
  "require_review_risk_levels": ["medium", "high", "critical"],
  "backup_enabled": true,
  "import_validation_enabled": true,
  "safety_check_level": "maximum",
  "audit_trail_enabled": true
}
```

### Large Codebase Configuration

```json
{
  "llm_provider": "ollama",
  "llm_model": "codellama:7b",
  "max_concurrent": 8,
  "chunk_large_files": true,
  "triage_enabled": true,
  "progressive_analysis": true,
  "batch_size": 20,
  "cache_enabled": true,
  "memory_limit_mb": 4096
}
```

## Environment Variables

The system supports configuration via environment variables:

### LLM Provider Variables

```bash
export LLM_INTELLIGENCE_PROVIDER="openai"
export LLM_INTELLIGENCE_MODEL="gpt-4"
export LLM_INTELLIGENCE_API_KEY="sk-your-key"
export LLM_INTELLIGENCE_TEMPERATURE="0.0"
export LLM_INTELLIGENCE_MAX_TOKENS="2000"
```

### Analysis Variables

```bash
export LLM_INTELLIGENCE_MIN_CONFIDENCE="0.7"
export LLM_INTELLIGENCE_MAX_CONCURRENT="3"
export LLM_INTELLIGENCE_ENABLE_STATIC="true"
export LLM_INTELLIGENCE_CACHE_ENABLED="true"
```

### Execution Variables

```bash
export LLM_INTELLIGENCE_DRY_RUN="true"
export LLM_INTELLIGENCE_BACKUP_ENABLED="true"
export LLM_INTELLIGENCE_AUTO_APPROVE="low"
```

### Path Variables

```bash
export LLM_INTELLIGENCE_ROOT_DIR="/path/to/codebase"
export LLM_INTELLIGENCE_OUTPUT_DIR="intelligence_output"
export LLM_INTELLIGENCE_CACHE_DIR="llm_cache"
export LLM_INTELLIGENCE_LOG_DIR="logs"
```

### Loading Environment Variables

The system automatically loads environment variables with the `LLM_INTELLIGENCE_` prefix:

```python
# Environment variables are automatically loaded
# LLM_INTELLIGENCE_API_KEY becomes config['api_key']
# LLM_INTELLIGENCE_MAX_CONCURRENT becomes config['max_concurrent']
```

## Configuration Validation

### Validation Rules

The system validates configuration for:

1. **Required Fields**: Ensures all required configuration is present
2. **Type Validation**: Checks that values have correct types
3. **Range Validation**: Validates numeric ranges and thresholds
4. **Provider Validation**: Ensures provider-specific settings are correct
5. **Path Validation**: Checks that directories exist and are writable

### Validation Example

```python
from llm_intelligence_system import LLMIntelligenceScanner

try:
    scanner = LLMIntelligenceScanner(Path("/path/to/codebase"), config)
    # Configuration is valid
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fix configuration based on error message
```

### Common Validation Errors

1. **Missing API Key**: `ConfigurationError: API key required for OpenAI provider`
2. **Invalid Model**: `ConfigurationError: Model 'gpt-5' not supported by OpenAI`
3. **Invalid Path**: `ConfigurationError: Output directory not writable`
4. **Invalid Threshold**: `ConfigurationError: Confidence threshold must be between 0.0 and 1.0`

### Configuration Testing

Test your configuration before running the full pipeline:

```bash
# Test configuration
python run_intelligence_system.py --test-config

# Test with mock provider (no API key needed)
python run_intelligence_system.py --provider mock --test-connection

# Validate specific configuration file
python run_intelligence_system.py --config-file my_config.json --validate-config
```

## Configuration Best Practices

### 1. Start with Defaults
Use the built-in defaults and only override what you need:

```json
{
  "llm_provider": "openai",
  "api_key": "sk-your-key"
}
```

### 2. Environment Variables for Secrets
Never put API keys in configuration files:

```bash
export LLM_INTELLIGENCE_API_KEY="sk-your-key"
```

### 3. Version Control Safe
Create configuration files that are safe for version control:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "min_confidence_threshold": 0.7,
  "max_concurrent": 3
}
```

### 4. Environment-Specific Configs

Create different configurations for different environments:

- `config.development.json`: Mock provider, relaxed thresholds
- `config.staging.json`: Ollama provider, moderate settings
- `config.production.json`: OpenAI provider, strict thresholds

### 5. Progressive Configuration

Start simple and add complexity as needed:

**Level 1 (Basic)**:
```json
{
  "llm_provider": "mock"
}
```

**Level 2 (Functional)**:
```json
{
  "llm_provider": "openai",
  "api_key": "sk-your-key",
  "enable_static_analysis": true
}
```

**Level 3 (Optimized)**:
```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "api_key": "sk-your-key",
  "min_confidence_threshold": 0.75,
  "max_concurrent": 5,
  "enable_static_analysis": true,
  "cache_enabled": true,
  "backup_enabled": true
}
```

### 6. Backup and Recovery

Always configure backup settings:

```json
{
  "backup_enabled": true,
  "backup_dir": "tools/codebase_reorganizer/backups",
  "backup_before_each_batch": true,
  "backup_retention_days": 7
}
```

### 7. Monitoring and Alerts

Configure monitoring for production use:

```json
{
  "enable_metrics": true,
  "metrics_retention_days": 30,
  "alert_on_errors": true,
  "alert_on_low_confidence": true,
  "alert_webhook_url": "https://hooks.slack.com/...",
  "health_check_enabled": true,
  "health_check_interval_minutes": 5
}
```

## Advanced Configuration Patterns

### Conditional Configuration

```python
# Load configuration based on environment
import os

env = os.getenv('ENVIRONMENT', 'development')

if env == 'development':
    config = {
        'llm_provider': 'mock',
        'min_confidence_threshold': 0.5
    }
elif env == 'staging':
    config = {
        'llm_provider': 'ollama',
        'llm_model': 'codellama:7b'
    }
else:  # production
    config = {
        'llm_provider': 'openai',
        'llm_model': 'gpt-4',
        'min_confidence_threshold': 0.8
    }
```

### Dynamic Configuration

```python
# Adjust configuration based on codebase size
codebase_size = count_python_files(root_dir)

if codebase_size < 50:
    config['max_concurrent'] = 2
    config['llm_model'] = 'gpt-3.5-turbo'
elif codebase_size < 200:
    config['max_concurrent'] = 5
    config['llm_model'] = 'gpt-4'
else:
    config['max_concurrent'] = 10
    config['llm_model'] = 'gpt-4'
    config['chunk_large_files'] = True
```

### Provider Fallback Configuration

```json
{
  "llm_providers": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "priority": 1,
      "fallback_on_error": true
    },
    {
      "provider": "ollama",
      "model": "codellama:7b",
      "priority": 2,
      "fallback_on_error": true
    },
    {
      "provider": "mock",
      "priority": 3,
      "fallback_on_error": false
    }
  ]
}
```

This comprehensive configuration system allows you to adapt the LLM Intelligence System to your specific needs, whether you're doing quick analysis for development or running large-scale reorganization in production.

