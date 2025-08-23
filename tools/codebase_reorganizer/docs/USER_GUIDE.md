# LLM Intelligence System - User Guide

This comprehensive user guide will walk you through every aspect of using the LLM Intelligence System to analyze and reorganize your Python codebase.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Running the System](#running-the-system)
5. [Understanding Outputs](#understanding-outputs)
6. [Executing Changes](#executing-changes)
7. [Advanced Usage](#advanced-usage)
8. [Monitoring & Debugging](#monitoring--debugging)

## Quick Start

### Prerequisites
- Python 3.9 or higher
- LLM API access (OpenAI, Ollama) OR use mock mode for testing

### 5-Minute Setup
```bash
# 1. Navigate to the system directory
cd tools/codebase_reorganizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test the system
python test_intelligence_system.py

# 4. Run on your code (mock mode for testing)
python run_intelligence_system.py --full-pipeline --max-files 10

# 5. View results
# Check the generated JSON files in tools/codebase_reorganizer/intelligence_output/
```

### Expected Output
After running the test, you should see:
```
🧪 Testing LLM Intelligence System
=======================================
✅ Scan completed successfully!
   Files scanned: 4
   Lines analyzed: 287
📊 Integration completed - 4 entries
✅ Reorganization plan created - 2 batches
🎉 Test completed successfully!
```

## Installation

### Basic Installation
```bash
# Clone or navigate to your repository
cd tools/codebase_reorganizer

# Install Python dependencies
pip install -r requirements.txt
```

### LLM Provider Setup

#### OpenAI Setup
```bash
# Install OpenAI library
pip install openai

# Set your API key (replace with your actual key)
export OPENAI_API_KEY="your-api-key-here"
```

#### Ollama Setup (Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama2:7b
# OR for code-focused analysis
ollama pull codellama:7b
```

#### Mock Mode (No API Required)
```bash
# No additional setup needed - works out of the box
python run_intelligence_system.py --provider mock --full-pipeline
```

### Directory Structure After Installation
```
tools/codebase_reorganizer/
├── llm_intelligence_system.py      # Main scanner
├── intelligence_integration_engine.py # Integration engine
├── reorganization_planner.py        # Planning system
├── run_intelligence_system.py       # Main CLI
├── test_intelligence_system.py      # Test suite
├── README_INTELLIGENCE_SYSTEM.md    # Main README
├── docs/                           # Documentation
│   ├── OVERVIEW.md                # System overview
│   ├── USER_GUIDE.md              # This file
│   ├── API_REFERENCE.md           # API docs
│   ├── CONFIGURATION.md           # Configuration guide
│   ├── TROUBLESHOOTING.md        # Troubleshooting
│   ├── BEST_PRACTICES.md         # Best practices
│   ├── INTEGRATION_GUIDE.md      # Integration guide
│   └── DEVELOPMENT.md            # Development guide
├── intelligence_output/           # Generated outputs
└── llm_cache/                    # Caching directory
```

## Basic Concepts

### Intelligence Pipeline

The system processes your code through four main stages:

1. **Scanning**: LLM analyzes each Python file for semantic understanding
2. **Integration**: Combines LLM insights with static analysis
3. **Planning**: Creates safe reorganization plans with risk assessment
4. **Execution**: Applies changes in controlled batches

### Classification Categories

The system classifies code into these categories:

- **Security**: Authentication, encryption, access control
- **Intelligence**: ML models, data analysis, predictive systems
- **Frontend/Dashboard**: UI components, visualization, user interfaces
- **Documentation**: Code documentation, help systems
- **Testing**: Unit tests, integration tests, test frameworks
- **Utility**: Helper functions, common utilities, shared code
- **API**: REST endpoints, GraphQL, web services
- **Database**: Models, migrations, data persistence
- **Data Processing**: ETL, data cleaning, transformation
- **Automation**: Scripts, workflows, scheduled tasks
- **Orchestration**: Coordination, messaging, event handling
- **Monitoring**: Logging, metrics, alerting systems
- **Analytics**: Reporting, insights, statistical analysis
- **DevOps**: Deployment, infrastructure, CI/CD
- **Uncategorized**: Code that doesn't fit other categories

### Confidence Scoring

Every analysis includes confidence scores (0.0-1.0):

- **0.0-0.3**: Very low confidence - likely incorrect
- **0.3-0.5**: Low confidence - needs human review
- **0.5-0.7**: Medium confidence - acceptable with caution
- **0.7-0.85**: High confidence - reliable
- **0.85-1.0**: Very high confidence - highly reliable

### Risk Levels

Operations are assigned risk levels:

- **Low**: Safe operations like utility file moves
- **Medium**: Operations requiring some caution
- **High**: Critical operations needing review
- **Critical**: High-risk operations that require extensive review

## Running the System

### Command Line Interface

The main interface is `run_intelligence_system.py`:

```bash
python run_intelligence_system.py [OPTIONS] COMMAND
```

#### Available Commands

- `--full-pipeline`: Run complete analysis pipeline
- `--step scan`: Run only the scanning phase
- `--step integrate`: Run only the integration phase
- `--step plan`: Run only the planning phase
- `--step execute`: Execute a specific batch
- `--status`: Show system status
- `--generate-report`: Generate report from existing results

### Full Pipeline Examples

#### 1. Mock Mode (Testing)
```bash
python run_intelligence_system.py --full-pipeline --max-files 20
```

#### 2. OpenAI with GPT-4
```bash
python run_intelligence_system.py --full-pipeline \
    --provider openai \
    --api-key sk-your-key-here \
    --model gpt-4 \
    --max-files 50
```

#### 3. Local Ollama Model
```bash
python run_intelligence_system.py --full-pipeline \
    --provider ollama \
    --model codellama:7b \
    --max-files 30
```

#### 4. High-Confidence Mode
```bash
python run_intelligence_system.py --full-pipeline \
    --provider openai \
    --api-key sk-your-key-here \
    --max-files 100 \
    --config-file high_confidence_config.json
```

### Step-by-Step Execution

#### Step 1: Scanning
```bash
# Scan your codebase and generate intelligence map
python run_intelligence_system.py --step scan \
    --provider openai \
    --api-key sk-your-key-here \
    --max-files 100

# Output: llm_intelligence_map_TIMESTAMP.json
```

#### Step 2: Integration
```bash
# Integrate LLM analysis with static analysis
python run_intelligence_system.py --step integrate \
    --llm-map llm_intelligence_map_TIMESTAMP.json

# Output: integrated_intelligence_TIMESTAMP.json
```

#### Step 3: Planning
```bash
# Create reorganization plan
python run_intelligence_system.py --step plan \
    --llm-map llm_intelligence_map_TIMESTAMP.json \
    --integrated integrated_intelligence_TIMESTAMP.json

# Output: reorganization_plan_TIMESTAMP.json
```

#### Step 4: Execution
```bash
# Execute a batch (dry run first)
python run_intelligence_system.py --step execute \
    --plan reorganization_plan_TIMESTAMP.json \
    --batch-id batch_low_risk_moves \
    --dry-run

# Execute for real (if dry run looks good)
python run_intelligence_system.py --step execute \
    --plan reorganization_plan_TIMESTAMP.json \
    --batch-id batch_low_risk_moves
```

### Configuration Files

Create custom configuration files for different scenarios:

#### High Confidence Config (`high_confidence_config.json`)
```json
{
    "min_confidence_threshold": 0.8,
    "high_confidence_threshold": 0.9,
    "integration_method": "consensus_with_fallback",
    "risk_thresholds": {
        "low": 0.85,
        "medium": 0.7,
        "high": 0.5
    },
    "auto_approve_risk_levels": ["low"],
    "require_review_risk_levels": ["high", "critical"]
}
```

#### Conservative Config (`conservative_config.json`)
```json
{
    "min_confidence_threshold": 0.9,
    "integration_method": "consensus_with_fallback",
    "auto_approve_risk_levels": [],
    "require_review_risk_levels": ["medium", "high", "critical"],
    "backup_enabled": true,
    "dry_run_enabled": true
}
```

## Understanding Outputs

### LLM Intelligence Map Structure
```json
{
  "scan_timestamp": "2025-01-15T10:30:00Z",
  "scan_id": "abc123",
  "total_files_scanned": 150,
  "total_lines_analyzed": 15432,
  "directory_structure": {
    "src": {
      "core": {
        "security": {
          "auth.py": {
            "type": "file",
            "classification": "security",
            "confidence": 0.92,
            "size": 8432,
            "lines": 271
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
      "analysis_timestamp": "2025-01-15T10:30:15Z",
      "module_summary": "Handles JWT token generation and validation for user authentication...",
      "functionality_details": "Provides JWT token generation, user authentication, role-based permissions...",
      "dependencies_analysis": "Uses cryptography library for encryption, JWT for tokens...",
      "security_implications": "Handles sensitive authentication data, implements secure token storage...",
      "testing_requirements": "Requires comprehensive security testing, mock authentication...",
      "architectural_role": "Security service layer providing authentication services...",
      "primary_classification": "security",
      "secondary_classifications": ["authentication", "encryption"],
      "reorganization_recommendations": ["Move to src/core/security/", "Group with other security modules"],
      "confidence_score": 0.92,
      "key_features": ["JWT tokens", "role-based access", "password hashing"],
      "integration_points": ["User management", "API endpoints", "Database layer"],
      "complexity_assessment": "medium",
      "maintainability_notes": "Regular security updates required, follow OWASP guidelines",
      "file_size": 8432,
      "line_count": 271,
      "class_count": 2,
      "function_count": 12
    }
  ]
}
```

### Integrated Intelligence Structure
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
        "Add tests for token expiry edge-cases"
      ],
      "synthesis_reasoning": "LLM=security(0.92) | semantic=security(0.85) | integrated=0.89"
    }
  ]
}
```

### Reorganization Plan Structure
```json
{
  "plan_id": "reorg_20250115_103000",
  "total_tasks": 45,
  "total_batches": 3,
  "batches": [
    {
      "batch_id": "batch_low_risk_moves",
      "batch_name": "Low Risk File Moves",
      "risk_level": "low",
      "estimated_total_time": 45,
      "tasks": [
        {
          "task_id": "move_helper_py",
          "action": "move_file",
          "source_path": "utils/helper.py",
          "target_path": "src/core/utility/helper.py",
          "confidence": 0.87,
          "priority": 6,
          "risk_level": "low",
          "rationale": "Utility module with high confidence classification",
          "estimated_effort_minutes": 5,
          "success_criteria": ["File moved successfully", "Imports updated", "Tests pass"],
          "rollback_plan": "Restore from backup"
        }
      ]
    }
  ]
}
```

## Executing Changes

### Safety First - Always Dry Run
```bash
# Always test with dry run first
python run_intelligence_system.py --step execute \
    --plan reorganization_plan_TIMESTAMP.json \
    --batch-id batch_low_risk_moves \
    --dry-run

# Review the output to ensure it looks correct
# Check for any warnings or errors
```

### Batch Execution Strategy

1. **Start with Low-Risk Batches**
   ```bash
   # Execute low-risk batch
   python run_intelligence_system.py --step execute \
       --plan reorganization_plan_TIMESTAMP.json \
       --batch-id batch_low_risk_moves
   ```

2. **Run Tests After Each Batch**
   ```bash
   # Run your test suite
   python -m pytest tests/
   # OR
   python run_tests.py
   ```

3. **Review Medium-Risk Batches**
   ```bash
   # Generate detailed report for medium-risk batch
   python run_intelligence_system.py --generate-report \
       --results-file pipeline_results_TIMESTAMP.json
   ```

4. **Execute High-Risk Batches with Caution**
   ```bash
   # High-risk batches require explicit review
   python run_intelligence_system.py --step execute \
       --plan reorganization_plan_TIMESTAMP.json \
       --batch-id batch_security_critical \
       --dry-run
   ```

### Rollback Procedures

If something goes wrong:

1. **Immediate Rollback**
   ```bash
   # The system creates backups automatically
   # Check the backups directory for recovery options
   ls tools/codebase_reorganizer/backups/
   ```

2. **Git-Based Recovery**
   ```bash
   # If using git, you can revert changes
   git status  # See what changed
   git checkout -- file_path  # Revert specific file
   git reset --hard HEAD~1  # Revert all changes (if committed)
   ```

3. **Manual Recovery**
   ```bash
   # Check the execution logs for details
   cat tools/codebase_reorganizer/logs/reorganization_planner_TIMESTAMP.log
   ```

## Advanced Usage

### Custom Configuration
```python
# Create custom config file
config = {
    "llm_provider": "openai",
    "llm_model": "gpt-4-turbo",
    "min_confidence_threshold": 0.8,
    "integration_method": "adaptive_confidence",
    "max_concurrent": 5,
    "cache_enabled": True,
    "backup_enabled": True,
    "dry_run_enabled": True
}

# Save as JSON
import json
with open('my_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Use with command
python run_intelligence_system.py --full-pipeline --config-file my_config.json
```

### Incremental Analysis
```bash
# Only analyze changed files (requires git)
git diff --name-only | grep '\.py$' > changed_files.txt

# Create custom analysis scope
python run_intelligence_system.py --step scan \
    --file-list changed_files.txt \
    --provider openai \
    --api-key sk-your-key
```

### Batch Processing for Large Codebases
```bash
# Process in batches for very large codebases
python run_intelligence_system.py --step scan --max-files 100 --output-prefix batch1_
python run_intelligence_system.py --step scan --max-files 100 --offset 100 --output-prefix batch2_

# Merge results manually or create custom merge script
```

### API Integration
```python
from llm_intelligence_system import LLMIntelligenceScanner

# Initialize scanner
scanner = LLMIntelligenceScanner(
    root_dir="path/to/codebase",
    config={
        "llm_provider": "openai",
        "api_key": "your-key",
        "max_concurrent": 3
    }
)

# Run analysis
intelligence_map = scanner.scan_and_analyze(max_files=50)

# Access results
for entry in intelligence_map.intelligence_entries:
    print(f"File: {entry.relative_path}")
    print(f"Classification: {entry.primary_classification}")
    print(f"Confidence: {entry.confidence_score}")
```

## Monitoring & Debugging

### Log Files
The system generates comprehensive logs:

```
tools/codebase_reorganizer/logs/
├── llm_intelligence_TIMESTAMP.log      # Scanning logs
├── integration_engine_TIMESTAMP.log    # Integration logs
└── reorganization_planner_TIMESTAMP.log # Planning logs
```

### Common Log Messages

#### Success Messages
```
INFO - LLM analysis completed for src/core/auth.py
INFO - Integration confidence: 0.87 for src/core/auth.py
INFO - Task completed: move_file src/utils/helper.py -> src/core/utility/helper.py
```

#### Warning Messages
```
WARNING - Low confidence (0.45) for src/legacy/old_code.py
WARNING - Static analysis failed for src/broken/syntax_error.py
WARNING - Import validation failed for src/moved/new_location.py
```

#### Error Messages
```
ERROR - LLM API rate limit exceeded
ERROR - Failed to parse LLM response as JSON
ERROR - File not found: src/deleted/file.py
```

### Debugging Tips

1. **Check System Status**
   ```bash
   python run_intelligence_system.py --status
   ```

2. **Enable Debug Logging**
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
   python run_intelligence_system.py --full-pipeline --max-files 5
   ```

3. **Test with Mock Provider**
   ```bash
   # Isolate LLM-related issues
   python run_intelligence_system.py --provider mock --full-pipeline --max-files 5
   ```

4. **Validate JSON Outputs**
   ```bash
   # Check for JSON syntax errors
   python -m json.tool llm_intelligence_map_TIMESTAMP.json
   ```

5. **Check File Permissions**
   ```bash
   # Ensure the system can read/write files
   ls -la tools/codebase_reorganizer/intelligence_output/
   ls -la tools/codebase_reorganizer/llm_cache/
   ```

### Performance Tuning

1. **Adjust Concurrency**
   ```bash
   # Reduce for API rate limits
   python run_intelligence_system.py --max-concurrent 2 --full-pipeline
   ```

2. **Enable Caching**
   ```bash
   # Caching is enabled by default, but you can clear it
   rm -rf tools/codebase_reorganizer/llm_cache/
   ```

3. **Use Local Models**
   ```bash
   # Ollama is faster and has no API limits
   python run_intelligence_system.py --provider ollama --model llama2:7b
   ```

## Best Practices

### 1. Start Small
- Begin with 10-20 files to validate the system
- Review outputs carefully before scaling up
- Use dry-run mode for all executions initially

### 2. Review High-Confidence Operations
- Even high-confidence operations should be reviewed
- Check the reasoning provided by the system
- Verify that proposed moves make architectural sense

### 3. Backup Before Execution
- The system creates backups automatically, but verify they exist
- Consider creating additional backups before major operations
- Keep backups until you're confident in the changes

### 4. Monitor System Health
- Check logs regularly for warnings and errors
- Monitor confidence scores over time
- Update configurations based on observed performance

### 5. Security-First Approach
- Always review security-related classifications carefully
- High-risk security modules should have additional manual review
- Consider running security tools after reorganization

### 6. Continuous Improvement
- The system learns from your feedback
- Provide corrections for misclassifications
- Update configurations based on your preferences

## Troubleshooting

### Common Issues

1. **"LLM provider not available"**
   - Check API key for OpenAI
   - Ensure Ollama server is running
   - Use mock provider for testing

2. **"No Python files found"**
   - Check root directory path
   - Verify file permissions
   - Ensure Python files aren't excluded

3. **"Low confidence scores"**
   - Try a more capable LLM model
   - Increase `min_confidence_threshold`
   - Enable static analysis for better consensus

4. **"JSON decode error"**
   - Check LLM API response format
   - Verify temperature settings (use 0.0 for consistency)
   - Test with mock provider

5. **"Import validation failed"**
   - Check for circular dependencies
   - Verify import paths are correct
   - Use dry-run mode to test changes

### Getting Help

1. **Check the Documentation**
   ```bash
   # View all documentation
   ls tools/codebase_reorganizer/docs/
   ```

2. **Run the Test Suite**
   ```bash
   python test_intelligence_system.py
   ```

3. **Check System Status**
   ```bash
   python run_intelligence_system.py --status
   ```

4. **Generate Detailed Reports**
   ```bash
   python run_intelligence_system.py --generate-report --results-file results.json
   ```

---

**Remember: The LLM Intelligence System is designed to augment human decision-making, not replace it. Always review outputs and use your expertise to validate recommendations.**
