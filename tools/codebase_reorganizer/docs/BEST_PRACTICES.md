# LLM Intelligence System - Best Practices

This guide provides recommended practices for effectively using the LLM Intelligence System to analyze and reorganize your Python codebase.

## Table of Contents

1. [Planning and Preparation](#planning-and-preparation)
2. [Configuration Best Practices](#configuration-best-practices)
3. [Execution Strategies](#execution-strategies)
4. [Quality Assurance](#quality-assurance)
5. [Performance Optimization](#performance-optimization)
6. [Team Integration](#team-integration)
7. [Maintenance and Evolution](#maintenance-and-evolution)
8. [Common Pitfalls](#common-pitfalls)

## Planning and Preparation

### 1. Understand Your Goals

**Before running the system, clarify your objectives:**

- **Codebase Onboarding**: Helping new developers understand the codebase structure
- **Legacy Refactoring**: Untangling spaghetti code into organized modules
- **Security Audit**: Identifying security-related code and ensuring proper organization
- **Architecture Documentation**: Creating living documentation of system structure
- **Performance Optimization**: Identifying performance bottlenecks and optimization opportunities

### 2. Assess Your Codebase

**Run initial analysis to understand your codebase:**

```bash
# Get a quick overview
find . -name "*.py" | wc -l                    # Count Python files
find . -name "*.py" -exec wc -l {} + | tail -1  # Total lines of code
find . -name "*.py" -exec grep -l "class " {} \; | wc -l  # Classes
find . -name "*.py" -exec grep -l "def " {} \; | wc -l     # Functions
```

**Questions to answer:**
- How many Python files are there?
- What's the typical file size and complexity?
- Are there existing organizational patterns?
- What are the main functional areas?

### 3. Choose the Right LLM Provider

**Select based on your needs:**

| Provider | Best For | Cost | Speed | Privacy |
|----------|----------|------|-------|---------|
| **OpenAI GPT-4** | Production, complex analysis | High | Fast | External |
| **OpenAI GPT-3.5** | Development, cost-effective | Medium | Fast | External |
| **Ollama Local** | Privacy, unlimited usage | Free | Medium | Local |
| **Mock** | Testing, development | Free | Instant | Local |

### 4. Start with a Small Scope

**Begin with a representative subset:**

```bash
# Analyze a specific directory first
python run_intelligence_system.py --step scan \
    --root src/core \
    --provider mock \
    --max-files 10

# Or analyze specific file types
python run_intelligence_system.py --step scan \
    --file-pattern "*security*.py" \
    --provider mock
```

## Configuration Best Practices

### 1. Environment-Specific Configurations

**Create different configurations for different environments:**

```bash
# Development
cp config.development.json config.json
# Use mock provider, relaxed thresholds

# Production
cp config.production.json config.json
# Use real provider, strict thresholds
```

### 2. API Key Management

**Never commit API keys to version control:**

```bash
# Use environment variables
export LLM_INTELLIGENCE_API_KEY="sk-your-key-here"

# Or use .env files (add to .gitignore)
echo "LLM_INTELLIGENCE_API_KEY=sk-your-key-here" > .env
```

### 3. Confidence Threshold Tuning

**Adjust confidence thresholds based on your risk tolerance:**

```json
{
  "min_confidence_threshold": 0.7,    // Conservative
  "high_confidence_threshold": 0.85,  // High confidence actions
  "consensus_threshold": 0.6         // Agreement required
}
```

**Guidelines:**
- **Development**: 0.5-0.6 (more suggestions, review manually)
- **Staging**: 0.6-0.7 (balanced approach)
- **Production**: 0.7-0.8 (conservative, high confidence)

### 4. Resource Management

**Configure based on your system's capabilities:**

```json
{
  "max_concurrent": 3,               // CPU cores / 2
  "memory_limit_mb": 2048,          // Available RAM / 2
  "cache_enabled": true,            // Always enable for speed
  "backup_enabled": true            // Always enable for safety
}
```

## Execution Strategies

### 1. Phased Execution Approach

**Execute in phases to minimize risk:**

```bash
# Phase 1: Analyze and plan
python run_intelligence_system.py --full-pipeline --dry-run

# Phase 2: Execute low-risk changes
python run_intelligence_system.py --step execute \
    --plan reorganization_plan.json \
    --batch-id batch_low_risk_moves

# Phase 3: Review and execute medium-risk changes
python run_intelligence_system.py --step execute \
    --plan reorganization_plan.json \
    --batch-id batch_security_critical \
    --dry-run

# Phase 4: Execute remaining changes
```

### 2. Incremental Analysis

**For large codebases, analyze incrementally:**

```bash
# Analyze by directory
for dir in src/*/; do
    echo "Analyzing $dir"
    python run_intelligence_system.py --step scan \
        --root "$dir" \
        --output-prefix "$(basename "$dir")_"
done

# Merge results (future feature)
```

### 3. Triage-Based Analysis

**Prioritize files that need the most attention:**

```bash
# Focus on complex files first
python run_intelligence_system.py --step scan \
    --triage complex \
    --max-files 20

# Focus on security-related files
python run_intelligence_system.py --step scan \
    --triage security \
    --max-files 15
```

### 4. Continuous Integration

**Integrate into your development workflow:**

```bash
# Pre-commit hook
#!/bin/bash
python run_intelligence_system.py --step scan \
    --file-list $(git diff --cached --name-only | grep '\.py$') \
    --provider mock

# CI/CD pipeline
python run_intelligence_system.py --full-pipeline \
    --provider ollama \
    --fail-on-low-confidence
```

## Quality Assurance

### 1. Validate Results

**Always review the generated intelligence:**

```bash
# Generate comprehensive report
python run_intelligence_system.py --generate-report \
    --results-file pipeline_results.json

# Check confidence distribution
cat intelligence_output/llm_intelligence_map.json | \
    jq -r '.intelligence_entries[].confidence_score' | \
    sort -n | uniq -c
```

### 2. Test After Each Phase

**Ensure functionality is preserved:**

```bash
# Run your test suite after each batch
python -m pytest tests/

# Check for import errors
python -c "import sys; sys.path.append('.'); import your_main_module"

# Validate specific functionality
python -m pytest tests/test_specific_feature.py
```

### 3. Backup and Recovery

**Always have recovery options:**

```bash
# Check backup integrity
ls -la tools/codebase_reorganizer/backups/
du -sh tools/codebase_reorganizer/backups/

# Test restore procedure
# (The system creates backups automatically)
```

### 4. Performance Monitoring

**Monitor system performance:**

```bash
# Check analysis speed
grep "Analysis took" tools/codebase_reorganizer/logs/*.log

# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Performance Optimization

### 1. Caching Strategy

**Optimize cache usage:**

```bash
# Clear corrupted cache
rm -rf tools/codebase_reorganizer/llm_cache/

# Pre-populate cache for common files
python run_intelligence_system.py --step scan \
    --cache-only \
    --file-pattern "src/utils/*.py"
```

### 2. Concurrency Tuning

**Adjust based on your system and API limits:**

```json
{
  "max_concurrent": 2,              // API rate limit / 2
  "requests_per_minute": 30,       // API tier limit
  "batch_delay_seconds": 1         // Rate limiting
}
```

### 3. Model Selection

**Choose the right model for your use case:**

```json
{
  // Fast analysis, lower quality
  "llm_model": "gpt-3.5-turbo",
  "llm_max_tokens": 1000,

  // Balanced approach
  "llm_model": "gpt-4",
  "llm_max_tokens": 1500,

  // High quality, slower
  "llm_model": "gpt-4-turbo",
  "llm_max_tokens": 2000
}
```

### 4. File Size Optimization

**Handle different file sizes appropriately:**

```json
{
  "max_file_size": 50000,          // Skip very large files
  "chunk_large_files": true,       // Enable chunking
  "chunk_size": 4000,              // Optimal chunk size
  "chunk_overlap": 200             // Overlap for context
}
```

## Team Integration

### 1. Knowledge Sharing

**Share the intelligence with your team:**

```bash
# Generate team-friendly report
python run_intelligence_system.py --generate-report \
    --format markdown \
    --audience team

# Create architecture documentation
python run_intelligence_system.py --generate-docs \
    --output docs/architecture.md
```

### 2. Code Review Integration

**Use intelligence in code reviews:**

```bash
# Analyze pull request changes
python run_intelligence_system.py --step scan \
    --file-list <(git diff origin/main --name-only -- '*.py')

# Generate review comments
python run_intelligence_system.py --generate-review-comments \
    --pr-number 123
```

### 3. Documentation Generation

**Auto-generate living documentation:**

```bash
# Generate API documentation
python run_intelligence_system.py --generate-api-docs \
    --output docs/api/

# Generate architecture overview
python run_intelligence_system.py --generate-architecture-docs \
    --output docs/architecture/

# Update README with current structure
python run_intelligence_system.py --update-readme
```

### 4. Training and Onboarding

**Help new team members understand the codebase:**

```bash
# Generate onboarding guide
python run_intelligence_system.py --generate-onboarding-guide \
    --skill-level junior \
    --focus-areas security,api,database

# Create learning paths
python run_intelligence_system.py --generate-learning-path \
    --role backend-developer
```

## Maintenance and Evolution

### 1. Regular Analysis

**Keep intelligence up to date:**

```bash
# Weekly analysis
crontab -e
# Add: 0 2 * * 1 /path/to/python run_intelligence_system.py --full-pipeline --incremental

# On code changes
# Add to CI/CD pipeline
python run_intelligence_system.py --step scan --incremental
```

### 2. Configuration Evolution

**Update configuration as your needs change:**

```bash
# Version control configurations
git tag config-v1.0
cp config.json config.v1.0.json

# A/B test configurations
python run_intelligence_system.py --config config_a.json --output results_a.json
python run_intelligence_system.py --config config_b.json --output results_b.json
```

### 3. Model Updates

**Stay current with LLM improvements:**

```bash
# Test new model versions
python run_intelligence_system.py --step scan \
    --llm-model gpt-4-turbo \
    --max-files 10 \
    --output gpt4_turbo_results.json

# Compare results
python compare_results.py gpt4_results.json gpt4_turbo_results.json
```

### 4. Feedback Loop

**Improve analysis quality over time:**

```bash
# Review and correct classifications
python run_intelligence_system.py --review-mode \
    --confidence-threshold 0.6

# Update taxonomy based on corrections
python update_taxonomy.py corrections.json
```

## Common Pitfalls

### 1. Over-Reliance on Automation

**❌ Don't do this:**
```bash
# Blindly execute all recommendations
python run_intelligence_system.py --full-pipeline --auto-execute
```

**✅ Do this instead:**
```bash
# Review first, then execute in phases
python run_intelligence_system.py --full-pipeline --dry-run
# Review results
python run_intelligence_system.py --step execute --batch-id batch_1
```

### 2. Ignoring Low Confidence

**❌ Don't ignore low confidence warnings:**
```
WARNING: Low confidence (0.45) for src/utils/helper.py
```

**✅ Address low confidence appropriately:**
```bash
# Investigate why confidence is low
python run_intelligence_system.py --debug-file src/utils/helper.py

# Consider manual review or improvement
python run_intelligence_system.py --manual-review src/utils/helper.py
```

### 3. No Backup Strategy

**❌ Don't run without backups:**
```bash
python run_intelligence_system.py --no-backup --step execute
```

**✅ Always backup:**
```bash
# Backups are enabled by default
ls tools/codebase_reorganizer/backups/
# Verify backup integrity
```

### 4. Running on Unstable Code

**❌ Don't analyze broken code:**
```bash
# If tests are failing
python run_intelligence_system.py --full-pipeline
```

**✅ Fix issues first:**
```bash
# Ensure codebase is stable
python -m pytest tests/
python -m flake8 src/

# Then analyze
python run_intelligence_system.py --full-pipeline
```

### 5. Using Wrong Provider for Task

**❌ Don't use expensive API for simple tasks:**
```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "min_confidence_threshold": 0.5
}
```

**✅ Match provider to task:**
```json
{
  "llm_provider": "ollama",        // Free, local
  "llm_model": "codellama:7b",     // Good for code
  "min_confidence_threshold": 0.6   // Balanced
}
```

### 6. No Monitoring or Logging

**❌ Don't run without monitoring:**
```bash
python run_intelligence_system.py --no-logs --full-pipeline
```

**✅ Enable comprehensive monitoring:**
```json
{
  "log_level": "INFO",
  "enable_metrics": true,
  "alert_on_errors": true,
  "health_check_enabled": true
}
```

## Success Metrics

### 1. Quality Metrics

**Track these to ensure quality:**

- **Confidence Score Distribution**: Aim for >70% high confidence (>0.8)
- **Classification Accuracy**: Manual review should confirm >90% accuracy
- **Import Error Rate**: Should be 0% after reorganization
- **Test Pass Rate**: Should maintain or improve after reorganization

### 2. Productivity Metrics

**Measure impact on development:**

- **Code Navigation Time**: Time to find relevant code
- **Onboarding Time**: Time for new developers to become productive
- **Bug Fix Time**: Time to locate and fix bugs
- **Feature Development Time**: Time to implement new features

### 3. Architecture Metrics

**Track architectural improvements:**

- **Module Cohesion**: Higher cohesion scores
- **Coupling Reduction**: Lower coupling between modules
- **Separation of Concerns**: Clearer functional boundaries
- **Documentation Coverage**: More comprehensive documentation

## Advanced Patterns

### 1. Custom Integration Rules

**Create domain-specific rules:**

```python
# custom_integration.py
def custom_security_integration(llm_entry, static_analysis):
    """Custom integration for security modules"""
    if 'security' in llm_entry.primary_classification:
        # Boost confidence for security files
        return max(llm_entry.confidence_score, 0.8)
    return llm_entry.confidence_score

def custom_api_integration(llm_entry, static_analysis):
    """Custom integration for API modules"""
    if 'api' in llm_entry.primary_classification:
        # Check for common API patterns
        has_endpoints = 'route' in llm_entry.functionality_details.lower()
        has_validation = 'validate' in llm_entry.functionality_details.lower()

        if has_endpoints and has_validation:
            return min(llm_entry.confidence_score + 0.1, 1.0)

    return llm_entry.confidence_score
```

### 2. Automated Testing Integration

**Integrate with your testing framework:**

```bash
# Pre-analysis test run
python -m pytest tests/ --tb=short

# Analysis
python run_intelligence_system.py --full-pipeline

# Post-analysis test run
python -m pytest tests/ --tb=short

# Compare results
python compare_test_results.py pre_analysis.json post_analysis.json
```

### 3. Git Integration

**Integrate with version control:**

```bash
# Analyze only changed files
python run_intelligence_system.py --step scan \
    --file-list <(git diff --name-only origin/main -- '*.py')

# Create reorganization branch
git checkout -b reorganization-2025-01-15

# Commit reorganization in logical chunks
git add -A
git commit -m "feat: reorganize utility modules

- Moved helper functions to src/utils/
- Updated import statements
- Verified functionality preserved"

# Create pull request
gh pr create --title "Codebase Reorganization" --body "Automated reorganization based on LLM intelligence analysis"
```

### 4. CI/CD Integration

**Add to your pipeline:**

```yaml
# .github/workflows/reorganization.yml
name: Codebase Reorganization
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  reorganize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r tools/codebase_reorganizer/requirements.txt

      - name: Run analysis
        run: |
          python tools/codebase_reorganizer/run_intelligence_system.py \
            --full-pipeline \
            --provider ollama \
            --model codellama:7b \
            --dry-run

      - name: Generate report
        run: |
          python tools/codebase_reorganizer/run_intelligence_system.py \
            --generate-report \
            --results-file pipeline_results.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: reorganization-results
          path: tools/codebase_reorganizer/intelligence_output/
```

## Conclusion

The LLM Intelligence System is a powerful tool for understanding and reorganizing Python codebases. By following these best practices, you can maximize its effectiveness while minimizing risks:

1. **Start Small**: Begin with a subset of your codebase
2. **Review Results**: Always review before executing changes
3. **Backup First**: Never run without backups enabled
4. **Test Thoroughly**: Validate functionality after each change
5. **Iterate Gradually**: Execute in phases, not all at once
6. **Monitor Performance**: Track metrics and adjust configuration
7. **Team Integration**: Share knowledge and involve the team
8. **Continuous Improvement**: Refine configuration based on results

**Remember**: The system augments human intelligence, it doesn't replace it. Use the insights to inform your decisions, but always apply your expertise and understanding of your specific codebase and domain.
