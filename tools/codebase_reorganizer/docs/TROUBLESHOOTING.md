# LLM Intelligence System - Troubleshooting Guide

This guide provides solutions to common issues encountered when using the LLM Intelligence System. Follow the troubleshooting steps in order to resolve problems efficiently.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [LLM Provider Issues](#llm-provider-issues)
5. [Analysis Errors](#analysis-errors)
6. [Execution Problems](#execution-problems)
7. [Performance Issues](#performance-issues)
8. [File System Issues](#file-system-issues)
9. [Network and Connectivity](#network-and-connectivity)
10. [Advanced Debugging](#advanced-debugging)

## Quick Diagnosis

### System Health Check

Run this command to quickly diagnose system health:

```bash
python run_intelligence_system.py --status
```

**Expected Output:**
```
🤖 LLM Intelligence System Status
=======================================
Root Directory: /path/to/your/codebase
Output Directory: tools/codebase_reorganizer/intelligence_output
Components Available: ✅
Available Components:
  ✅ LLM Intelligence Scanner
  ✅ Intelligence Integration Engine
  ✅ Reorganization Planner
Scanner Config: mock / gpt-4
Output Directory Exists: ✅
```

### Quick Test

Run the included test to verify the system works:

```bash
python test_intelligence_system.py
```

**Expected Output:**
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

## Installation Issues

### Problem: Import Errors

**Symptoms:**
```
ImportError: No module named 'llm_intelligence_system'
```

**Solutions:**

1. **Check Python Path**
   ```bash
   # Make sure you're in the right directory
   cd tools/codebase_reorganizer

   # Check if files exist
   ls -la *.py
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python Version**
   ```bash
   python --version
   # Should be 3.9 or higher
   ```

4. **Verify Installation**
   ```bash
   python -c "import sys; print(sys.path)"
   python -c "from llm_intelligence_system import LLMIntelligenceScanner; print('Import successful')"
   ```

### Problem: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'tools/codebase_reorganizer/intelligence_output'
```

**Solutions:**

1. **Fix Directory Permissions**
   ```bash
   # Create directories with proper permissions
   mkdir -p tools/codebase_reorganizer/intelligence_output
   mkdir -p tools/codebase_reorganizer/llm_cache
   mkdir -p tools/codebase_reorganizer/logs
   mkdir -p tools/codebase_reorganizer/backups

   # Set permissions
   chmod -R 755 tools/codebase_reorganizer/
   ```

2. **Check User Permissions**
   ```bash
   # Check current user
   whoami
   id

   # Check directory ownership
   ls -la tools/codebase_reorganizer/
   ```

3. **Run as Administrator (Windows)**
   ```cmd
   # Right-click and "Run as administrator"
   python run_intelligence_system.py --status
   ```

## Configuration Problems

### Problem: Invalid Configuration

**Symptoms:**
```
ConfigurationError: Invalid configuration: 'llm_provider' must be one of ['openai', 'ollama', 'mock']
```

**Solutions:**

1. **Check Configuration File**
   ```bash
   # Validate JSON syntax
   python -m json.tool my_config.json
   ```

2. **Use Default Configuration**
   ```bash
   # Remove custom config and use defaults
   rm my_config.json
   python run_intelligence_system.py --status
   ```

3. **Fix Common Configuration Issues**
   ```json
   // Correct configuration
   {
     "llm_provider": "openai",
     "llm_model": "gpt-4",
     "api_key": "sk-your-key-here",
     "min_confidence_threshold": 0.7
   }

   // Incorrect configuration
   {
     "llm_provider": "invalid_provider",  // ❌ Wrong value
     "api_key": null,                     // ❌ Missing key
     "min_confidence_threshold": 1.5      // ❌ Out of range
   }
   ```

### Problem: Environment Variables Not Working

**Symptoms:**
```
ConfigurationError: API key required for OpenAI provider
```

**Solutions:**

1. **Set Environment Variables**
   ```bash
   # Set the API key
   export LLM_INTELLIGENCE_API_KEY="sk-your-key-here"

   # Verify it's set
   echo $LLM_INTELLIGENCE_API_KEY
   ```

2. **Check Variable Names**
   ```bash
   # Correct variable names
   export LLM_INTELLIGENCE_PROVIDER="openai"
   export LLM_INTELLIGENCE_MODEL="gpt-4"
   export LLM_INTELLIGENCE_API_KEY="sk-your-key"

   # List all LLM_INTELLIGENCE variables
   env | grep LLM_INTELLIGENCE
   ```

3. **Use .env File**
   ```bash
   # Create .env file
   echo "LLM_INTELLIGENCE_API_KEY=sk-your-key-here" > .env

   # Install python-dotenv
   pip install python-dotenv

   # The system will automatically load .env files
   ```

## LLM Provider Issues

### Problem: OpenAI API Key Invalid

**Symptoms:**
```
LLMProviderError: Authentication failed: Invalid API key
```

**Solutions:**

1. **Check API Key**
   ```bash
   # Verify API key format
   echo $LLM_INTELLIGENCE_API_KEY | head -c 20
   # Should start with "sk-"
   ```

2. **Test API Key**
   ```bash
   curl -H "Authorization: Bearer $LLM_INTELLIGENCE_API_KEY" \
        https://api.openai.com/v1/models
   ```

3. **Check Account Status**
   - Verify your OpenAI account has credits
   - Check if your API key is expired
   - Ensure the API key has the correct permissions

4. **Use Test Key**
   ```bash
   # Use a minimal test configuration
   python run_intelligence_system.py --provider mock --full-pipeline --max-files 3
   ```

### Problem: Ollama Not Running

**Symptoms:**
```
LLMProviderError: Connection failed: Connection refused
```

**Solutions:**

1. **Start Ollama Server**
   ```bash
   # Start Ollama in the background
   ollama serve &

   # Wait a few seconds
   sleep 5

   # Check if it's running
   curl http://localhost:11434/api/tags
   ```

2. **Install Model**
   ```bash
   # Pull a model
   ollama pull llama2:7b

   # List available models
   ollama list
   ```

3. **Check Configuration**
   ```json
   {
     "llm_provider": "ollama",
     "llm_model": "llama2:7b",
     "ollama_base_url": "http://localhost:11434"
   }
   ```

### Problem: Rate Limit Exceeded

**Symptoms:**
```
LLMProviderError: Rate limit exceeded
```

**Solutions:**

1. **Reduce Concurrency**
   ```bash
   python run_intelligence_system.py --max-concurrent 1 --full-pipeline
   ```

2. **Add Delays**
   ```bash
   # Modify configuration
   {
     "max_concurrent": 2,
     "requests_per_minute": 30
   }
   ```

3. **Use Caching**
   ```bash
   # Enable caching to avoid repeated requests
   python run_intelligence_system.py --cache-enabled --full-pipeline
   ```

4. **Switch to Local Model**
   ```bash
   python run_intelligence_system.py --provider ollama --full-pipeline
   ```

## Analysis Errors

### Problem: No Python Files Found

**Symptoms:**
```
AnalysisError: No Python files found to analyze
```

**Solutions:**

1. **Check Root Directory**
   ```bash
   # Verify you're in the right place
   pwd
   ls -la

   # Check for Python files
   find . -name "*.py" | head -10
   ```

2. **Check Exclusions**
   ```bash
   # The system excludes certain directories by default
   # Check if your files are in excluded directories:
   # - archive, archives
   # - PROD*, PRODUCTION_PACKAGES
   # - research repos (MetaGPT, PraisonAI, etc.)
   ```

3. **Specify Correct Root**
   ```bash
   # Make sure you're pointing to the right directory
   python run_intelligence_system.py --root /path/to/your/codebase --status
   ```

4. **Check File Permissions**
   ```bash
   # Ensure files are readable
   ls -la src/*.py
   ```

### Problem: Syntax Error in Files

**Symptoms:**
```
AnalysisError: Syntax error in file: src/broken.py
```

**Solutions:**

1. **Fix Syntax Errors**
   ```bash
   # Check the specific file
   python -m py_compile src/broken.py

   # Fix any syntax issues
   python src/broken.py
   ```

2. **Skip Problematic Files**
   ```bash
   # The system will automatically skip files with syntax errors
   # Check logs for which files were skipped
   cat tools/codebase_reorganizer/logs/*.log | grep "skipped"
   ```

3. **Validate Python Files**
   ```bash
   # Check all Python files for syntax errors
   find . -name "*.py" -exec python -m py_compile {} \;
   ```

### Problem: Low Confidence Scores

**Symptoms:**
```
WARNING: Low confidence (0.45) for src/utils/helper.py
```

**Solutions:**

1. **Use Better LLM Model**
   ```bash
   python run_intelligence_system.py --provider openai --model gpt-4 --full-pipeline
   ```

2. **Enable Static Analysis**
   ```bash
   python run_intelligence_system.py --enable-static-analysis --full-pipeline
   ```

3. **Adjust Confidence Threshold**
   ```bash
   # Lower the threshold for testing
   python run_intelligence_system.py --min-confidence 0.5 --full-pipeline
   ```

4. **Check File Quality**
   ```bash
   # Some files might be too short or have unclear purpose
   wc -l src/utils/helper.py
   head -20 src/utils/helper.py
   ```

### Problem: Analysis Timeout

**Symptoms:**
```
AnalysisError: Analysis timeout for src/large_file.py
```

**Solutions:**

1. **Increase Timeout**
   ```bash
   {
     "llm_analysis_timeout": 120,
     "static_analysis_timeout": 60
   }
   ```

2. **Chunk Large Files**
   ```bash
   {
     "chunk_large_files": true,
     "chunk_size": 3000,
     "max_chunks_per_file": 3
   }
   ```

3. **Skip Large Files**
   ```bash
   {
     "max_file_size": 100000,
     "max_lines_per_file": 2000
   }
   ```

## Execution Problems

### Problem: Execution Failed

**Symptoms:**
```
ExecutionError: Failed to move file: src/old.py -> src/new.py
```

**Solutions:**

1. **Check File Permissions**
   ```bash
   ls -la src/old.py
   ls -la src/
   ```

2. **Check Disk Space**
   ```bash
   df -h
   ```

3. **Check File Locks**
   ```bash
   # On Windows
   tasklist /FI "IMAGENAME eq python.exe"

   # On Linux/Mac
   ps aux | grep python
   ```

4. **Run in Dry Mode First**
   ```bash
   python run_intelligence_system.py --step execute --plan plan.json --batch-id batch_1 --dry-run
   ```

### Problem: Import Validation Failed

**Symptoms:**
```
ExecutionError: Import validation failed for src/moved.py
```

**Solutions:**

1. **Check Import Statements**
   ```bash
   grep -n "import\|from" src/moved.py
   ```

2. **Verify Module Paths**
   ```bash
   # Test the imports
   python -c "import sys; sys.path.append('.'); import src.moved"
   ```

3. **Update Import Statements**
   ```bash
   # The system should have updated imports automatically
   # Check if there were any import updates in the logs
   cat tools/codebase_reorganizer/logs/*.log | grep "import"
   ```

4. **Manual Import Fix**
   ```python
   # Before: from src.old_module import function
   # After:  from src.new_module import function
   ```

### Problem: Backup Creation Failed

**Symptoms:**
```
ExecutionError: Backup creation failed: Permission denied
```

**Solutions:**

1. **Check Backup Directory**
   ```bash
   ls -la tools/codebase_reorganizer/backups/
   ```

2. **Fix Permissions**
   ```bash
   chmod -R 755 tools/codebase_reorganizer/backups/
   ```

3. **Disable Backups (Not Recommended)**
   ```bash
   python run_intelligence_system.py --no-backup --step execute --plan plan.json --batch-id batch_1
   ```

## Performance Issues

### Problem: Analysis Too Slow

**Symptoms:**
```
INFO: Analysis took 45 minutes for 50 files
```

**Solutions:**

1. **Increase Concurrency**
   ```bash
   python run_intelligence_system.py --max-concurrent 5 --full-pipeline
   ```

2. **Use Faster Model**
   ```bash
   python run_intelligence_system.py --provider openai --model gpt-3.5-turbo --full-pipeline
   ```

3. **Enable Caching**
   ```bash
   # Caching is enabled by default
   # Clear cache if it's corrupted
   rm -rf tools/codebase_reorganizer/llm_cache/
   ```

4. **Use Local Model**
   ```bash
   python run_intelligence_system.py --provider ollama --model llama2:7b --full-pipeline
   ```

### Problem: Memory Usage High

**Symptoms:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**

1. **Reduce Concurrency**
   ```bash
   python run_intelligence_system.py --max-concurrent 1 --full-pipeline
   ```

2. **Process in Batches**
   ```bash
   # Process 20 files at a time
   python run_intelligence_system.py --max-files 20 --full-pipeline
   ```

3. **Optimize Memory Usage**
   ```json
   {
     "memory_limit_mb": 1024,
     "optimize_memory_usage": true,
     "use_memory_mapped_files": true
   }
   ```

4. **Monitor Memory Usage**
   ```bash
   # Monitor memory during execution
   while true; do ps aux | grep python | grep -v grep; sleep 5; done
   ```

### Problem: Disk Space Full

**Symptoms:**
```
OSError: No space left on device
```

**Solutions:**

1. **Check Disk Space**
   ```bash
   df -h
   du -sh tools/codebase_reorganizer/
   ```

2. **Clear Cache**
   ```bash
   rm -rf tools/codebase_reorganizer/llm_cache/
   rm -rf tools/codebase_reorganizer/intelligence_output/
   ```

3. **Reduce Output Retention**
   ```json
   {
     "cache_max_age_days": 7,
     "metrics_retention_days": 14
   }
   ```

4. **Use Compression**
   ```json
   {
     "cache_compression": true,
     "output_compression": true
   }
   ```

## File System Issues

### Problem: File Not Found Errors

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'src/missing.py'
```

**Solutions:**

1. **Check File Exists**
   ```bash
   ls -la src/missing.py
   find . -name "missing.py"
   ```

2. **Check Case Sensitivity**
   ```bash
   # On case-insensitive filesystems, this can cause issues
   ls -la | grep -i missing
   ```

3. **Verify Path Encoding**
   ```bash
   # Check for special characters in paths
   python -c "import os; print(repr('src/missing.py'))"
   ```

4. **Check Git Status**
   ```bash
   git status
   git ls-files | grep missing.py
   ```

### Problem: Path Too Long (Windows)

**Symptoms:**
```
OSError: [WinError 206] The filename or extension is too long
```

**Solutions:**

1. **Shorten Directory Structure**
   ```bash
   # Move to a shorter path
   cd /c/short/path/to/codebase
   ```

2. **Use SUBST (Windows)**
   ```cmd
   # Create a drive letter mapping
   subst X: C:\very\long\path\to\your\project
   cd X:
   ```

3. **Configure Shorter Paths**
   ```json
   {
     "output_dir": "output",
     "cache_dir": "cache",
     "log_dir": "logs"
   }
   ```

### Problem: Cross-Platform Path Issues

**Symptoms:**
```
Path issues on different operating systems
```

**Solutions:**

1. **Use Path Objects**
   ```python
   from pathlib import Path
   root_dir = Path("/path/to/codebase")  # Works on all platforms
   ```

2. **Normalize Paths**
   ```python
   import os
   normalized_path = os.path.normpath(some_path)
   ```

3. **Check Path Separators**
   ```python
   import os
   print(os.sep)  # '/' on Unix, '\\' on Windows
   ```

## Network and Connectivity

### Problem: Network Timeout

**Symptoms:**
```
LLMProviderError: Connection timeout
```

**Solutions:**

1. **Increase Timeout**
   ```json
   {
     "llm_analysis_timeout": 120,
     "network_timeout_seconds": 60,
     "retry_attempts": 3
   }
   ```

2. **Check Network Connection**
   ```bash
   ping google.com
   curl -I https://api.openai.com
   ```

3. **Use Local Model**
   ```bash
   python run_intelligence_system.py --provider ollama --full-pipeline
   ```

4. **Retry Logic**
   ```bash
   # The system has built-in retry logic
   # Check the retry configuration
   ```

### Problem: SSL Certificate Issues

**Symptoms:**
```
SSLError: Certificate verify failed
```

**Solutions:**

1. **Update Certificates**
   ```bash
   # Update CA certificates
   pip install --upgrade certifi
   ```

2. **Disable SSL Verification (Not Recommended)**
   ```bash
   export PYTHONHTTPSVERIFY=0  # Only for testing
   ```

3. **Check System Time**
   ```bash
   date
   # SSL certificates require correct system time
   ```

4. **Corporate Proxy**
   ```bash
   export HTTP_PROXY="http://proxy.company.com:8080"
   export HTTPS_PROXY="http://proxy.company.com:8080"
   ```

## Advanced Debugging

### Enable Debug Logging

```bash
# Set environment variable
export LLM_INTELLIGENCE_LOG_LEVEL="DEBUG"

# Or modify configuration
{
  "log_level": "DEBUG",
  "log_to_console": true,
  "log_to_file": true
}
```

### Debug LLM Responses

```bash
# Enable LLM response logging
export LLM_INTELLIGENCE_DEBUG_LLM="true"

# Check the logs for raw LLM responses
cat tools/codebase_reorganizer/logs/*.log | grep "LLM Response"
```

### Profile Performance

```python
# Add profiling to identify bottlenecks
import cProfile
import pstats

def profile_system():
    pr = cProfile.Profile()
    pr.enable()

    # Run your analysis here
    scanner = LLMIntelligenceScanner(Path("test_dir"))
    intelligence_map = scanner.scan_and_analyze(max_files=5)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions by cumulative time
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Add to your script
from memory_profiler import profile

@profile
def run_analysis():
    scanner = LLMIntelligenceScanner(Path("test_dir"))
    return scanner.scan_and_analyze(max_files=10)

run_analysis()
```

### Generate System Report

```bash
# Generate comprehensive system report
python run_intelligence_system.py --generate-system-report
```

This will create a detailed report including:
- System information
- Configuration details
- Recent errors
- Performance metrics
- Cache statistics

### Emergency Recovery

If the system gets into a bad state:

1. **Clear All Cache and Output**
   ```bash
   rm -rf tools/codebase_reorganizer/llm_cache/
   rm -rf tools/codebase_reorganizer/intelligence_output/
   rm -rf tools/codebase_reorganizer/backups/
   ```

2. **Reset to Defaults**
   ```bash
   python run_intelligence_system.py --provider mock --status
   ```

3. **Rebuild Step by Step**
   ```bash
   python run_intelligence_system.py --step scan --provider mock --max-files 3
   python run_intelligence_system.py --step integrate --llm-map intelligence_map.json
   python run_intelligence_system.py --step plan --llm-map intelligence_map.json --integrated integrated.json
   ```

### Getting Help

If you're still having issues:

1. **Check the Documentation**
   ```bash
   # All documentation is in the docs/ directory
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

4. **Generate Debug Report**
   ```bash
   python run_intelligence_system.py --debug-report > debug_report.txt
   ```

5. **Community Support**
   - Check GitHub issues for similar problems
   - Review the troubleshooting guide for your specific error
   - Create a minimal reproduction case

Remember: The LLM Intelligence System is designed to be robust and handle errors gracefully. Most issues can be resolved by checking the configuration, verifying file permissions, and ensuring the LLM provider is accessible.

