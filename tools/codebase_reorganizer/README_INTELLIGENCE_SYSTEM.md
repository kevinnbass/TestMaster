# LLM-Based Code Intelligence System

A comprehensive system for understanding, classifying, and reorganizing Python codebases using LLM analysis combined with traditional static analysis techniques.

## Overview

This system addresses the challenge of managing "spaghetti" codebases by providing:

1. **Deep Code Understanding**: LLM-powered semantic analysis that goes beyond simple pattern matching
2. **Intelligent Classification**: Automated categorization of code modules by functionality and purpose
3. **Confidence-Based Decision Making**: Risk-aware reorganization planning with human-in-the-loop controls
4. **Preserved Structure**: Maintains directory ordering while providing actionable reorganization insights

## Architecture

The system consists of four main components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLM Scanner    │───▶│ Integration     │───▶│ Reorganization  │───▶│   Execution    │
│                 │    │ Engine          │    │ Planner        │    │                 │
│ • Directory scan│    │ • LLM + Static  │    │ • Risk assess  │    │ • Batch exec    │
│ • LLM analysis  │    │ • Confidence calc│    │ • Phase plan   │    │ • Rollback      │
│ • JSON output   │    │ • Consensus      │    │ • Task creation│    │ • Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### 🔍 Deep Analysis
- **LLM Semantic Analysis**: Understands what code *does*, not just what it contains
- **Static Analysis Integration**: Combines with proven AST-based techniques
- **Multi-Perspective Classification**: Uses both LLM and traditional methods for robust categorization

### 🎯 Intelligent Classification
- **Security**: Authentication, encryption, access control
- **Intelligence**: ML models, data analysis, predictive systems
- **Frontend/Dashboard**: UI components, visualization, user interfaces
- **Documentation**: Code documentation, help systems
- **Testing**: Unit tests, integration tests, test frameworks
- **API**: REST endpoints, GraphQL, web services
- **Database**: Models, migrations, data persistence
- **Automation**: Scripts, workflows, scheduled tasks
- **Monitoring**: Logging, metrics, alerting systems
- **DevOps**: Deployment, infrastructure, CI/CD

### 🛡️ Risk-Aware Planning
- **Confidence Scoring**: Every classification has a confidence level (0-1)
- **Risk Assessment**: Automated risk evaluation for reorganization actions
- **Phased Execution**: Low-risk changes first, high-risk changes last
- **Rollback Support**: Backup and restore capabilities

### 📊 Preserved Structure
- **Directory Ordering**: Maintains exact on-disk file ordering in JSON output
- **Hierarchical Output**: Preserves folder structure in intelligence maps
- **Path Preservation**: All file paths are maintained throughout the process

## Installation & Setup

### Prerequisites
- Python 3.9+
- LLM API access (OpenAI, Ollama, etc.) or use mock mode for testing

### Installation
```bash
# Navigate to the reorganizer directory
cd tools/codebase_reorganizer

# Install required dependencies
pip install -r requirements.txt

# For LLM providers:
pip install openai          # For OpenAI API
# OR
pip install ollama          # For local Ollama
```

## Quick Start

### 1. System Status Check
```bash
python run_intelligence_system.py --status
```

### 2. Run Complete Pipeline (Mock Mode)
```bash
python run_intelligence_system.py --full-pipeline --max-files 5
```

### 3. Run with Real LLM (OpenAI)
```bash
python run_intelligence_system.py --full-pipeline \
    --provider openai \
    --api-key YOUR_OPENAI_API_KEY \
    --model gpt-4 \
    --max-files 10
```

### 4. Run with Local LLM (Ollama)
```bash
# Start Ollama server first
ollama serve

# Run with local model
python run_intelligence_system.py --full-pipeline \
    --provider ollama \
    --model llama2:7b \
    --max-files 10
```

## Step-by-Step Usage

### Step 1: LLM Intelligence Scanning
```bash
python run_intelligence_system.py --step scan \
    --provider openai \
    --api-key YOUR_KEY \
    --max-files 20
```

**Output**: `llm_intelligence_map_TIMESTAMP.json`
- Directory-ordered JSON with LLM analysis for each file
- Includes: summary, functionality, dependencies, security, testing, architecture
- Classifications with confidence scores
- Reorganization recommendations

### Step 2: Intelligence Integration
```bash
python run_intelligence_system.py --step integrate \
    --llm-map llm_intelligence_map_TIMESTAMP.json
```

**Output**: `integrated_intelligence_TIMESTAMP.json`
- Combines LLM analysis with static analysis
- Calculates integrated confidence scores
- Determines final classifications with reasoning
- Creates reorganization priorities

### Step 3: Reorganization Planning
```bash
python run_intelligence_system.py --step plan \
    --llm-map llm_intelligence_map_TIMESTAMP.json \
    --integrated integrated_intelligence_TIMESTAMP.json
```

**Output**: `reorganization_plan_TIMESTAMP.json`
- Detailed reorganization plan with executable tasks
- Phased approach (low-risk → high-risk)
- Risk assessment and mitigation strategies
- Success metrics and validation criteria

### Step 4: Batch Execution
```bash
# Review the plan first
python run_intelligence_system.py --generate-report \
    --results-file pipeline_results_TIMESTAMP.json

# Execute a specific batch (dry run first)
python run_intelligence_system.py --step execute \
    --plan reorganization_plan_TIMESTAMP.json \
    --batch-id batch_low_risk_moves \
    --dry-run

# Execute for real (if dry run looks good)
python run_intelligence_system.py --step execute \
    --plan reorganization_plan_TIMESTAMP.json \
    --batch-id batch_low_risk_moves
```

## Output Files Explained

### LLM Intelligence Map (`llm_intelligence_map.json`)
```json
{
  "scan_timestamp": "2025-01-15T10:30:00Z",
  "total_files_scanned": 150,
  "directory_structure": {
    "src": {
      "core": {
        "security": {
          "auth.py": {
            "type": "file",
            "size": 8432,
            "path": "src/core/security/auth.py"
          }
        }
      }
    }
  },
  "intelligence_entries": [
    {
      "full_path": "src/core/security/auth.py",
      "relative_path": "src/core/security/auth.py",
      "file_hash": "sha256:...",
      "module_summary": "Handles JWT token generation and validation...",
      "primary_classification": "security",
      "secondary_classifications": ["authentication", "encryption"],
      "confidence_score": 0.92,
      "complexity_assessment": "medium",
      "reorganization_recommendations": [
        "Move to src/core/security/",
        "Group with other security modules"
      ]
    }
  ]
}
```

### Integrated Intelligence (`integrated_intelligence.json`)
```json
{
  "integrated_intelligence": [
    {
      "file_path": "src/core/security/auth.py",
      "integrated_classification": "security",
      "reorganization_priority": 8,
      "integration_confidence": 0.89,
      "final_recommendations": [
        "Move to src/core/security/",
        "Consider adding rate limiting"
      ],
      "synthesis_reasoning": "LLM=security(0.92) | semantic=security(0.85) | integrated=0.89"
    }
  ]
}
```

### Reorganization Plan (`reorganization_plan.json`)
```json
{
  "plan_id": "reorg_20250115_103000",
  "total_tasks": 45,
  "total_batches": 3,
  "batches": [
    {
      "batch_id": "batch_low_risk_moves",
      "batch_name": "Low Risk File Moves",
      "description": "Automated moves for high-confidence, low-risk modules",
      "risk_level": "low",
      "tasks": [
        {
          "task_id": "move_src_utils_helper_py",
          "action": "move_file",
          "source_path": "src/utils/helper.py",
          "target_path": "src/core/utility/helper.py",
          "confidence": 0.87,
          "priority": 6,
          "risk_level": "low"
        }
      ]
    }
  ],
  "risk_mitigation": {
    "backup_strategy": {"enabled": true},
    "rollback_procedures": ["Restore from backup"],
    "testing_requirements": ["Run full test suite after each batch"]
  }
}
```

## Configuration Options

### LLM Providers
- **Mock**: For testing and development (no API required)
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Ollama**: Local models (llama2, codellama, etc.)
- **Anthropic**: Claude models
- **Grok**: xAI models

### Risk Thresholds
```python
config = {
    'min_confidence_threshold': 0.7,    # Minimum confidence for auto-actions
    'high_confidence_threshold': 0.85,  # High confidence threshold
    'risk_thresholds': {
        'low': 0.8,      # Risk score threshold for low risk
        'medium': 0.6,   # Risk score threshold for medium risk
        'high': 0.4      # Risk score threshold for high risk
    }
}
```

### Execution Controls
```python
config = {
    'auto_approve_risk_levels': ['low'],        # Auto-approve these risk levels
    'require_review_risk_levels': ['high', 'critical'],  # Require human review
    'backup_enabled': True,                     # Create backups before changes
    'dry_run_enabled': True,                    # Support dry-run mode
    'import_validation_enabled': True          # Validate imports after moves
}
```

## Understanding Confidence Scores

### Integration Confidence
The system calculates confidence using multiple factors:

```
Integration Confidence = weighted_average([
    llm_confidence: 0.35,           # LLM classification confidence
    semantic_confidence: 0.20,      # Static semantic analysis confidence
    pattern_confidence: 0.15,       # Design pattern recognition confidence
    quality_confidence: 0.15,       # Code quality assessment confidence
    relationship_confidence: 0.10,  # Dependency analysis confidence
    agreement_confidence: 0.05      # Agreement between analysis methods
])
```

### Reorganization Priority
Tasks are prioritized based on multiple factors:

```python
priority = base_score + modifiers
base_score = 5
modifiers:
    +2 if security-related
    +2 if confidence >= 0.85
    +1 if priority >= 8
    +1 if high complexity
    -1 if quality_score < 0.7
    -1 if agreement_confidence < 0.6
```

## Troubleshooting

### Common Issues

1. **LLM API Errors**
   ```bash
   # Test with mock mode first
   python run_intelligence_system.py --full-pipeline --provider mock --max-files 5
   ```

2. **Import Errors**
   ```bash
   # Check component availability
   python run_intelligence_system.py --status
   ```

3. **Low Confidence Scores**
   - Increase `min_confidence_threshold` in config
   - Use a more capable LLM model
   - Enable static analysis for better consensus

4. **Missing Output Files**
   - Check write permissions on output directory
   - Verify disk space availability
   - Check for file system errors

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=$PYTHONPATH:.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python run_intelligence_system.py --full-pipeline --max-files 3
```

## Advanced Usage

### Custom Integration Methods
```python
from intelligence_integration_engine import IntegrationMethod

config = {
    'integration_method': IntegrationMethod.ADAPTIVE_CONFIDENCE.value
}
```

### Custom Classification Taxonomy
```python
# Modify classification_taxonomy in intelligence_integration_engine.py
custom_taxonomy = {
    'primary_categories': ['custom_category1', 'custom_category2'],
    'category_mappings': {
        'my_keyword': 'custom_category1'
    }
}
```

### Batch Processing
```bash
# Process in batches for large codebases
python run_intelligence_system.py --step scan --max-files 50
python run_intelligence_system.py --step integrate --llm-map map1.json
python run_intelligence_system.py --step integrate --llm-map map2.json
# Merge results manually or create custom merge script
```

## Security Considerations

1. **API Key Management**: Never commit API keys to version control
2. **Data Privacy**: Be cautious about sending sensitive code to external APIs
3. **Local Models**: Use Ollama for local processing when privacy is required
4. **Access Controls**: Limit system access to authorized personnel only

## Performance Optimization

1. **Caching**: Enable caching to avoid re-analyzing unchanged files
2. **Concurrency**: Adjust `max_concurrent` based on API rate limits
3. **Batching**: Use smaller batches for better progress tracking
4. **Incremental**: Run on new/changed files only using git diff

## Contributing

1. Test with mock provider before implementing new LLM providers
2. Add comprehensive logging for debugging
3. Include confidence scoring for all new analysis methods
4. Document risk assessment procedures for new actions
5. Add unit tests for core functionality

## Support

For issues and questions:
1. Check the generated logs in `tools/codebase_reorganizer/logs/`
2. Run with `--status` to verify system health
3. Test with mock provider to isolate LLM-related issues
4. Review confidence scores and risk assessments in output files

---

**Remember**: This system is designed to augment human decision-making, not replace it. Always review high-risk changes and maintain backups before executing any reorganization tasks.

