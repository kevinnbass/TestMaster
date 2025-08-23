# LLM Intelligence System - Integration Guide

This guide explains how to integrate the LLM Intelligence System with your existing development workflow, tools, and processes.

## Table of Contents

1. [Development Workflow Integration](#development-workflow-integration)
2. [IDE Integration](#ide-integration)
3. [Version Control Integration](#version-control-integration)
4. [CI/CD Pipeline Integration](#cicd-pipeline-integration)
5. [Testing Framework Integration](#testing-framework-integration)
6. [Documentation Integration](#documentation-integration)
7. [Team Collaboration Integration](#team-collaboration-integration)
8. [External Tool Integration](#external-tool-integration)

## Development Workflow Integration

### 1. Daily Development Workflow

**Integrate intelligence analysis into your daily development:**

```bash
# Morning: Analyze recent changes
python run_intelligence_system.py --step scan \
    --file-list <(git log --since="24 hours ago" --name-only --pretty=format: | grep '\.py$' | sort | uniq) \
    --provider ollama \
    --max-files 10

# After implementing features: Quick analysis
python run_intelligence_system.py --step scan \
    --file-list <(git diff --cached --name-only -- '*.py') \
    --provider mock

# Before commit: Validate organization
python run_intelligence_system.py --validate-organization
```

### 2. Pre-Commit Hooks

**Add intelligence checks to your pre-commit workflow:**

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running LLM Intelligence analysis..."

# Analyze staged Python files
python tools/codebase_reorganizer/run_intelligence_system.py \
    --step scan \
    --file-list <(git diff --cached --name-only -- '*.py') \
    --provider mock \
    --output /tmp/precommit_analysis.json

# Check for issues
if [ -f "/tmp/precommit_analysis.json" ]; then
    # Parse results and warn about potential issues
    python tools/codebase_reorganizer/check_analysis_results.py /tmp/precommit_analysis.json
fi

echo "LLM Intelligence analysis completed."
```

### 3. Feature Branch Workflow

**Integrate into your feature development process:**

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Implement feature
# ... development work ...

# 3. Run intelligence analysis on changes
python run_intelligence_system.py --step scan \
    --file-list <(git diff origin/main --name-only -- '*.py') \
    --provider openai \
    --api-key $OPENAI_API_KEY

# 4. Review analysis results
# Check for reorganization suggestions or architectural concerns

# 5. Address any issues identified
# Refactor based on intelligence insights

# 6. Run tests
python -m pytest tests/

# 7. Create pull request with analysis summary
```

### 4. Code Review Integration

**Enhance your code review process:**

```python
# code_review_helper.py
import json
from pathlib import Path

def generate_code_review_comments(pr_number: int, repo_path: Path):
    """Generate AI-powered code review comments"""

    # Get changed files
    import subprocess
    result = subprocess.run(
        ["git", "diff", f"origin/main...HEAD", "--name-only", "--", "*.py"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )

    changed_files = result.stdout.strip().split('\n')

    if not changed_files or changed_files == ['']:
        return []

    # Run intelligence analysis
    analysis_result = subprocess.run([
        "python", "tools/codebase_reorganizer/run_intelligence_system.py",
        "--step", "scan",
        "--file-list", ";".join(changed_files),
        "--provider", "ollama",
        "--model", "codellama:7b",
        "--output", "/tmp/code_review_analysis.json"
    ], cwd=repo_path)

    # Parse results and generate comments
    comments = []

    try:
        with open("/tmp/code_review_analysis.json", 'r') as f:
            analysis_data = json.load(f)

        for entry in analysis_data.get('intelligence_entries', []):
            file_path = entry['relative_path']
            confidence = entry['confidence_score']
            classification = entry['primary_classification']

            if confidence < 0.7:
                comments.append({
                    'file': file_path,
                    'line': 1,
                    'body': f"⚠️ Low confidence ({confidence:.2f}) in classification as '{classification}'. Consider manual review."
                })

            # Add reorganization suggestions
            suggestions = entry.get('reorganization_recommendations', [])
            if suggestions:
                comments.append({
                    'file': file_path,
                    'line': 1,
                    'body': f"💡 Reorganization suggestion: {suggestions[0]}"
                })

    except Exception as e:
        comments.append({
            'file': 'README.md',
            'line': 1,
            'body': f"Failed to generate AI review comments: {e}"
        })

    return comments
```

## IDE Integration

### 1. Visual Studio Code

**Create VS Code tasks and extensions:**

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Analyze Current File",
            "type": "shell",
            "command": "python",
            "args": [
                "tools/codebase_reorganizer/run_intelligence_system.py",
                "--step", "scan",
                "--file-list", "${file}",
                "--provider", "mock"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "silent",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Analyze Workspace",
            "type": "shell",
            "command": "python",
            "args": [
                "tools/codebase_reorganizer/run_intelligence_system.py",
                "--step", "scan",
                "--root", "${workspaceFolder}",
                "--provider", "ollama",
                "--max-files", "20"
            ],
            "group": "build"
        }
    ]
}
```

**VS Code Extension Integration:**

```python
# vs_code_extension.py
import vscode
from pathlib import Path
import json

def analyze_current_file():
    """Analyze the currently open file"""
    editor = vscode.window.activeTextEditor
    if not editor:
        return

    file_path = Path(editor.document.fileName)

    # Run analysis
    import subprocess
    result = subprocess.run([
        "python", "tools/codebase_reorganizer/run_intelligence_system.py",
        "--step", "scan",
        "--file-list", str(file_path),
        "--provider", "mock",
        "--output", "/tmp/vscode_analysis.json"
    ], capture_output=True, text=True)

    # Parse results and show in output panel
    try:
        with open("/tmp/vscode_analysis.json", 'r') as f:
            analysis = json.load(f)

        if analysis.get('intelligence_entries'):
            entry = analysis['intelligence_entries'][0]
            message = f"""
File: {entry['relative_path']}
Classification: {entry['primary_classification']} (confidence: {entry['confidence_score']:.2f})
Summary: {entry['module_summary'][:200]}...
            """

            vscode.window.showInformationMessage("Analysis Complete")
            vscode.window.createOutputChannel("LLM Intelligence").appendLine(message)

    except Exception as e:
        vscode.window.showErrorMessage(f"Analysis failed: {e}")
```

### 2. PyCharm/IntelliJ

**Create external tools configuration:**

```xml
<!-- .idea/externalTools/Analyze_Current_File.xml -->
<toolSet name="LLM Intelligence">
    <tool name="Analyze Current File" showInMainMenu="true" showInEditor="true" showInProject="false">
        <exec>
            <option name="COMMAND" value="python" />
            <option name="PARAMETERS" value="tools/codebase_reorganizer/run_intelligence_system.py --step scan --file-list $FilePath$ --provider mock" />
            <option name="WORKING_DIRECTORY" value="$ProjectFileDir$" />
        </exec>
    </tool>
    <tool name="Analyze Project" showInMainMenu="true" showInEditor="false" showInProject="true">
        <exec>
            <option name="COMMAND" value="python" />
            <option name="PARAMETERS" value="tools/codebase_reorganizer/run_intelligence_system.py --full-pipeline --max-files 20 --provider ollama" />
            <option name="WORKING_DIRECTORY" value="$ProjectFileDir$" />
        </exec>
    </tool>
</toolSet>
```

### 3. Vim/Neovim

**Add to your vimrc:**

```vim
" LLM Intelligence Analysis
function! AnalyzeCurrentFile()
    let current_file = expand('%:p')
    execute '!python tools/codebase_reorganizer/run_intelligence_system.py --step scan --file-list ' . shellescape(current_file) . ' --provider mock --output /tmp/vim_analysis.json'
    if filereadable('/tmp/vim_analysis.json')
        execute '!python -c "
import json
with open(\"/tmp/vim_analysis.json\") as f:
    data = json.load(f)
if data.get(\"intelligence_entries\"):
    entry = data[\"intelligence_entries\"][0]
    print(f\"File: {entry[\"relative_path\"]}\")
    print(f\"Classification: {entry[\"primary_classification\"]} (confidence: {entry[\"confidence_score\"]:.2f})")
    print(f\"Summary: {entry[\"module_summary\"][:100]}...\")
"'
    endif
endfunction

" Map to F5
nmap <F5> :call AnalyzeCurrentFile()<CR>
```

### 4. Jupyter Notebook Extension

**Create a notebook extension:**

```python
# jupyter_extension.py
from IPython.core.magic import register_cell_magic
from pathlib import Path
import json
import subprocess

@register_cell_magic
def analyze_code(line, cell):
    """Analyze the current cell or file with LLM Intelligence"""

    # Write cell content to temporary file
    temp_file = Path("/tmp/jupyter_analysis.py")
    temp_file.write_text(cell)

    # Run analysis
    result = subprocess.run([
        "python", "tools/codebase_reorganizer/run_intelligence_system.py",
        "--step", "scan",
        "--file-list", str(temp_file),
        "--provider", "mock",
        "--output", "/tmp/jupyter_result.json"
    ], capture_output=True, text=True)

    # Display results
    try:
        with open("/tmp/jupyter_result.json", 'r') as f:
            analysis = json.load(f)

        if analysis.get('intelligence_entries'):
            entry = analysis['intelligence_entries'][0]
            print("🤖 LLM Intelligence Analysis:"            print(f"   📁 File: {entry['relative_path']}")
            print(f"   🏷️  Classification: {entry['primary_classification']}")
            print(f"   📊 Confidence: {entry['confidence_score']:.2f}")
            print(f"   📝 Summary: {entry['module_summary']}")
            print(f"   💡 Suggestions: {entry['reorganization_recommendations'][:2]}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

# Usage in notebook:
# %%analyze_code
# def my_function():
#     return "Hello, World!"
```

## Version Control Integration

### 1. Git Integration

**Create git hooks for automatic analysis:**

```bash
# .git/hooks/post-commit
#!/bin/bash

echo "Running post-commit intelligence analysis..."

# Analyze committed changes
python tools/codebase_reorganizer/run_intelligence_system.py \
    --step scan \
    --file-list <(git show --name-only --pretty=format: | grep '\.py$') \
    --provider mock \
    --output /tmp/post_commit_analysis.json

# Log results
echo "Analysis completed. Check /tmp/post_commit_analysis.json for details."
```

**GitHub Actions Integration:**

```yaml
# .github/workflows/intelligence-analysis.yml
name: Intelligence Analysis
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run intelligence analysis
        run: |
          python tools/codebase_reorganizer/run_intelligence_system.py \
            --step scan \
            --file-list <(git diff origin/main...HEAD --name-only -- '*.py') \
            --provider ollama \
            --model codellama:7b \
            --output intelligence_results.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: intelligence-analysis
          path: intelligence_results.json

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        run: |
          python tools/codebase_reorganizer/generate_pr_comment.py \
            --results intelligence_results.json \
            --pr-number ${{ github.event.number }}
```

### 2. SVN Integration (Legacy)

**Create SVN hooks:**

```bash
# hooks/post-commit
#!/bin/bash

REPOS="$1"
TXN="$2"

# Get changed files
CHANGED_FILES=$(svnlook changed -t "$TXN" "$REPOS" | grep '\.py$' | awk '{print $2}')

if [ -n "$CHANGED_FILES" ]; then
    echo "$CHANGED_FILES" | python tools/codebase_reorganizer/run_intelligence_system.py \
        --step scan \
        --file-list /dev/stdin \
        --provider mock \
        --output /tmp/svn_analysis.json
fi
```

## CI/CD Pipeline Integration

### 1. Jenkins Integration

**Add to Jenkins pipeline:**

```groovy
pipeline {
    agent any

    stages {
        stage('Intelligence Analysis') {
            steps {
                script {
                    // Run intelligence analysis
                    sh '''
                        python tools/codebase_reorganizer/run_intelligence_system.py \
                            --full-pipeline \
                            --provider ollama \
                            --model codellama:7b \
                            --max-files 50 \
                            --dry-run
                    '''

                    // Archive results
                    archiveArtifacts artifacts: 'tools/codebase_reorganizer/intelligence_output/*.json', allowEmptyArchive: true

                    // Check for issues
                    sh '''
                        python tools/codebase_reorganizer/check_analysis_quality.py \
                            --results tools/codebase_reorganizer/intelligence_output/llm_intelligence_map.json \
                            --min-confidence 0.7 \
                            --fail-on-low-confidence
                    '''
                }
            }
        }

        stage('Reorganization') {
            steps {
                script {
                    // Execute low-risk changes automatically
                    sh '''
                        python tools/codebase_reorganizer/run_intelligence_system.py \
                            --step execute \
                            --plan tools/codebase_reorganizer/intelligence_output/reorganization_plan.json \
                            --batch-id batch_low_risk_moves
                    '''

                    // Run tests to ensure nothing broke
                    sh 'python -m pytest tests/'
                }
            }
        }
    }

    post {
        always {
            // Publish reports
            publishHTML(target: [
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'tools/codebase_reorganizer/intelligence_output',
                reportFiles: 'intelligence_report.html',
                reportName: 'LLM Intelligence Report'
            ])
        }
    }
}
```

### 2. GitLab CI Integration

**Add to `.gitlab-ci.yml`:**

```yaml
stages:
  - analyze
  - reorganize
  - test

intelligence_analysis:
  stage: analyze
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python tools/codebase_reorganizer/run_intelligence_system.py --full-pipeline --provider ollama --max-files 30 --dry-run
  artifacts:
    paths:
      - tools/codebase_reorganizer/intelligence_output/
    reports:
      junit: tools/codebase_reorganizer/intelligence_output/test-results.xml
  only:
    - merge_requests

reorganization_execution:
  stage: reorganize
  image: python:3.9
  script:
    - python tools/codebase_reorganizer/run_intelligence_system.py --step execute --plan tools/codebase_reorganizer/intelligence_output/reorganization_plan.json --batch-id batch_low_risk_moves
  dependencies:
    - intelligence_analysis
  only:
    - main

post_reorganization_tests:
  stage: test
  image: python:3.9
  script:
    - python -m pytest tests/
  dependencies:
    - reorganization_execution
```

### 3. Azure DevOps Integration

**Add to `azure-pipelines.yml`:**

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  PYTHONUNBUFFERED: '1'

stages:
- stage: IntelligenceAnalysis
  jobs:
  - job: Analyze
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.9'

    - script: |
        pip install -r requirements.txt
      displayName: 'Install dependencies'

    - script: |
        python tools/codebase_reorganizer/run_intelligence_system.py \
          --full-pipeline \
          --provider ollama \
          --max-files 40
      displayName: 'Run intelligence analysis'

    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: 'tools/codebase_reorganizer/intelligence_output'
        artifactName: 'intelligence-results'

- stage: Reorganization
  dependsOn: IntelligenceAnalysis
  jobs:
  - job: Reorganize
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.9'

    - task: DownloadBuildArtifacts@0
      inputs:
        downloadType: 'single'
        artifactName: 'intelligence-results'
        downloadPath: '$(System.ArtifactsDirectory)'

    - script: |
        python tools/codebase_reorganizer/run_intelligence_system.py \
          --step execute \
          --plan $(System.ArtifactsDirectory)/intelligence-results/reorganization_plan.json \
          --batch-id batch_low_risk_moves
      displayName: 'Execute reorganization'
```

## Testing Framework Integration

### 1. Pytest Integration

**Create test fixtures that use intelligence analysis:**

```python
# tests/conftest.py
import pytest
import json
from pathlib import Path

@pytest.fixture(scope="session")
def intelligence_analysis():
    """Load intelligence analysis results for testing"""
    analysis_file = Path("tools/codebase_reorganizer/intelligence_output/llm_intelligence_map.json")

    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            return json.load(f)
    else:
        # Generate mock analysis for testing
        return {
            "intelligence_entries": [
                {
                    "relative_path": "src/core/security/auth.py",
                    "primary_classification": "security",
                    "confidence_score": 0.9,
                    "module_summary": "Authentication module"
                }
            ]
        }

@pytest.fixture
def security_modules(intelligence_analysis):
    """Get all security-related modules"""
    return [
        entry for entry in intelligence_analysis.get('intelligence_entries', [])
        if entry.get('primary_classification') == 'security'
    ]
```

**Test cases that validate intelligence analysis:**

```python
# tests/test_intelligence_analysis.py
import pytest

def test_security_modules_have_high_confidence(security_modules):
    """Ensure security modules have high confidence scores"""
    for module in security_modules:
        assert module['confidence_score'] >= 0.8, \
            f"Security module {module['relative_path']} has low confidence: {module['confidence_score']}"

def test_no_orphaned_modules(intelligence_analysis):
    """Ensure all modules have valid classifications"""
    valid_classifications = {'security', 'intelligence', 'frontend_dashboard',
                           'documentation', 'testing', 'utility', 'api',
                           'database', 'data_processing', 'orchestration',
                           'automation', 'monitoring', 'analytics', 'devops'}

    for entry in intelligence_analysis.get('intelligence_entries', []):
        classification = entry.get('primary_classification', 'uncategorized')
        assert classification in valid_classifications or classification == 'uncategorized', \
            f"Invalid classification: {classification} for {entry['relative_path']}"

def test_confidence_distribution(intelligence_analysis):
    """Check that confidence scores are reasonable"""
    confidences = [
        entry['confidence_score']
        for entry in intelligence_analysis.get('intelligence_entries', [])
    ]

    assert len(confidences) > 0, "No confidence scores found"

    avg_confidence = sum(confidences) / len(confidences)
    assert 0.5 <= avg_confidence <= 0.95, f"Unusual average confidence: {avg_confidence}"

    # Should have some high-confidence results
    high_confidence = [c for c in confidences if c >= 0.8]
    assert len(high_confidence) >= len(confidences) * 0.3, "Too few high-confidence results"
```

### 2. Test Coverage Integration

**Use intelligence analysis to improve test coverage:**

```python
# test_coverage_helper.py
import json
from pathlib import Path

def identify_untested_modules():
    """Identify modules that need more testing based on intelligence analysis"""

    # Load intelligence analysis
    analysis_file = Path("tools/codebase_reorganizer/intelligence_output/llm_intelligence_map.json")

    if not analysis_file.exists():
        return []

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    untested_modules = []

    for entry in analysis.get('intelligence_entries', []):
        module_path = entry['relative_path']
        testing_requirements = entry.get('testing_requirements', '')

        if testing_requirements and ('none' not in testing_requirements.lower()):
            # Check if corresponding test file exists
            test_path = module_path.replace('src/', 'tests/').replace('.py', '_test.py')
            test_file = Path(test_path)

            if not test_file.exists():
                untested_modules.append({
                    'module': module_path,
                    'testing_requirements': testing_requirements,
                    'complexity': entry.get('complexity_assessment', 'unknown'),
                    'priority': 'high' if 'security' in entry.get('primary_classification', '') else 'medium'
                })

    return untested_modules

def generate_test_suggestions():
    """Generate test implementation suggestions"""

    untested = identify_untested_modules()

    suggestions = []

    for module in untested:
        if 'api' in module['testing_requirements'].lower():
            suggestions.append(f"""
# {module['module']} - API Testing
def test_{Path(module['module']).stem}_endpoints():
    # Test API endpoints
    # Mock dependencies
    # Verify responses
    pass
""")
        elif 'security' in module['testing_requirements'].lower():
            suggestions.append(f"""
# {module['module']} - Security Testing
def test_{Path(module['module']).stem}_security():
    # Test authentication
    # Test authorization
    # Test input validation
    # Test security edge cases
    pass
""")
        else:
            suggestions.append(f"""
# {module['module']} - General Testing
def test_{Path(module['module']).stem}_functionality():
    # Test main functionality
    # Test edge cases
    # Test error handling
    pass
""")

    return suggestions
```

## Documentation Integration

### 1. Auto-Generated Documentation

**Generate living documentation from intelligence analysis:**

```python
# docs_generator.py
import json
from pathlib import Path
from typing import Dict, List

def generate_module_documentation(analysis_file: Path, output_dir: Path):
    """Generate documentation for modules based on intelligence analysis"""

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Group by classification
    by_classification = {}
    for entry in analysis.get('intelligence_entries', []):
        classification = entry.get('primary_classification', 'uncategorized')
        if classification not in by_classification:
            by_classification[classification] = []
        by_classification[classification].append(entry)

    # Generate documentation for each classification
    for classification, modules in by_classification.items():
        doc_content = generate_classification_docs(classification, modules)

        output_file = output_dir / f"{classification.replace('_', '-')}.md"
        with open(output_file, 'w') as f:
            f.write(doc_content)

def generate_classification_docs(classification: str, modules: List[Dict]) -> str:
    """Generate documentation for a classification"""

    content = f"# {classification.replace('_', ' ').title()} Modules\n\n"
    content += f"Documentation automatically generated from intelligence analysis.\n\n"

    for module in sorted(modules, key=lambda x: x['relative_path']):
        content += f"## {module['relative_path']}\n\n"
        content += f"**Classification Confidence:** {module['confidence_score']:.2f}\n\n"
        content += f"**Summary:** {module['module_summary']}\n\n"
        content += f"**Functionality:** {module['functionality_details']}\n\n"

        if module.get('key_features'):
            content += "**Key Features:**\n"
            for feature in module['key_features']:
                content += f"- {feature}\n"
            content += "\n"

        if module.get('integration_points'):
            content += "**Integration Points:**\n"
            for point in module['integration_points']:
                content += f"- {point}\n"
            content += "\n"

        content += "---\n\n"

    return content

def update_architecture_documentation(analysis_file: Path, arch_doc: Path):
    """Update architecture documentation with current analysis"""

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Generate architecture overview
    stats = analysis.get('scan_statistics', {})

    arch_content = f"""# System Architecture

## Overview
This document describes the system architecture based on automated intelligence analysis.

## Module Statistics
- **Total Modules:** {stats.get('total_files', 0)}
- **Average Confidence:** {stats.get('confidence_stats', {}).get('mean', 0):.2f}
- **High Confidence Modules:** {stats.get('confidence_stats', {}).get('high_confidence_count', 0)}

## Classification Distribution
"""

    for category, count in stats.get('classification_distribution', {}).get('primary', {}).items():
        percentage = count / stats.get('total_files', 1) * 100
        arch_content += f"- **{category}:** {count} modules ({percentage:.1f}%)\n"

    arch_content += "\n## Architecture Insights\n"

    # Add insights from reorganization plan
    plan_file = analysis_file.parent / "reorganization_plan.json"
    if plan_file.exists():
        with open(plan_file, 'r') as f:
            plan = json.load(f)

        arch_content += "\n### Recommended Architecture Improvements\n"
        for phase in plan.get('reorganization_phases', []):
            arch_content += f"- **{phase['phase_name']}:** {phase['description']}\n"

    with open(arch_doc, 'w') as f:
        f.write(arch_content)
```

### 2. README Auto-Update

**Keep project README synchronized with analysis:**

```python
# update_readme.py
import json
from pathlib import Path

def update_project_readme(analysis_file: Path, readme_file: Path):
    """Update project README with current architecture information"""

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    stats = analysis.get('scan_statistics', {})

    # Read current README
    if readme_file.exists():
        with open(readme_file, 'r') as f:
            readme_content = f.read()
    else:
        readme_content = "# Project\n\n"

    # Find or create architecture section
    arch_start = readme_content.find("## Architecture")
    if arch_start == -1:
        # Add architecture section at end
        readme_content += "\n## Architecture\n\n"
        arch_start = len(readme_content)

    arch_end = readme_content.find("\n## ", arch_start + 1)
    if arch_end == -1:
        arch_end = len(readme_content)

    # Generate architecture content
    arch_content = f"""## Architecture

### System Overview
- **Total Modules:** {stats.get('total_files', 0)}
- **Primary Classifications:** {len(stats.get('classification_distribution', {}).get('primary', {}))}
- **Average Confidence:** {stats.get('confidence_stats', {}).get('mean', 0):.2f}

### Module Distribution
"""

    for category, count in sorted(stats.get('classification_distribution', {}).get('primary', {}).items()):
        percentage = count / stats.get('total_files', 1) * 100
        arch_content += f"- **{category}:** {count} modules ({percentage:.1f}%)\n"

    arch_content += "\n### Key Architecture Decisions\n"
    arch_content += "- This documentation is automatically maintained by LLM Intelligence System\n"
    arch_content += "- Module classifications are updated based on code analysis\n"
    arch_content += "- Architecture insights are derived from dependency analysis\n"

    # Update README
    new_readme = (
        readme_content[:arch_start] +
        arch_content +
        readme_content[arch_end:]
    )

    with open(readme_file, 'w') as f:
        f.write(new_readme)
```

## Team Collaboration Integration

### 1. Slack Integration

**Send analysis results to team Slack:**

```python
# slack_integration.py
import json
import requests
from pathlib import Path

def send_analysis_to_slack(analysis_file: Path, webhook_url: str):
    """Send analysis summary to Slack"""

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    stats = analysis.get('scan_statistics', {})
    confidence = stats.get('confidence_stats', {})

    message = {
        "channel": "#architecture",
        "username": "LLM Intelligence System",
        "icon_emoji": ":robot_face:",
        "attachments": [
            {
                "color": "good" if confidence.get('mean', 0) > 0.7 else "warning",
                "title": "Codebase Intelligence Analysis Complete",
                "fields": [
                    {
                        "title": "Total Modules",
                        "value": stats.get('total_files', 0),
                        "short": True
                    },
                    {
                        "title": "Average Confidence",
                        "value": ".2f",
                        "short": True
                    },
                    {
                        "title": "High Confidence",
                        "value": confidence.get('high_confidence_count', 0),
                        "short": True
                    }
                ],
                "footer": "LLM Intelligence System",
                "ts": analysis.get('scan_timestamp', '')
            }
        ]
    }

    response = requests.post(webhook_url, json=message)
    return response.status_code == 200

def send_reorganization_alert(plan_file: Path, webhook_url: str):
    """Send reorganization plan alert"""

    with open(plan_file, 'r') as f:
        plan = json.load(f)

    message = {
        "channel": "#engineering",
        "username": "LLM Intelligence System",
        "icon_emoji": ":construction:",
        "text": f"🤖 Reorganization Plan Generated: {plan.get('total_batches', 0)} batches, {plan.get('total_tasks', 0)} tasks",
        "attachments": [
            {
                "color": "warning",
                "title": "Reorganization Plan Ready",
                "text": ".1f",
                "fields": [
                    {
                        "title": "High Priority Tasks",
                        "value": plan.get('summary', {}).get('task_statistics', {}).get('high_priority', 0),
                        "short": True
                    },
                    {
                        "title": "Security Modules",
                        "value": plan.get('summary', {}).get('task_statistics', {}).get('security_modules', 0),
                        "short": True
                    }
                ]
            }
        ]
    }

    response = requests.post(webhook_url, json=message)
    return response.status_code == 200
```

### 2. Jira Integration

**Create Jira tickets for reorganization tasks:**

```python
# jira_integration.py
import json
from pathlib import Path
from jira import JIRA

def create_reorganization_tickets(plan_file: Path, jira_server: str, jira_user: str, jira_token: str):
    """Create Jira tickets for reorganization tasks"""

    # Connect to Jira
    jira = JIRA(server=jira_server, basic_auth=(jira_user, jira_token))

    with open(plan_file, 'r') as f:
        plan = json.load(f)

    # Create epic for reorganization
    epic = jira.create_issue(
        project='PROJ',
        summary=f"Codebase Reorganization - {plan.get('plan_id', 'Unknown')}",
        description=".1f",
        issuetype={'name': 'Epic'}
    )

    # Create tasks for each batch
    for batch in plan.get('reorganization_phases', []):
        issue = jira.create_issue(
            project='PROJ',
            summary=f"Reorganize: {batch['batch_name']}",
            description=f"""Batch: {batch['batch_id']}
Description: {batch['description']}
Risk Level: {batch['risk_level']}
Estimated Time: {batch['estimated_time_minutes']} minutes

Tasks: {len(batch['tasks'])}

Prerequisites:
{chr(10).join(f"- {prereq}" for prereq in batch['prerequisites'])}

Postconditions:
{chr(10).join(f"- {post}" for post in batch['postconditions'])}
""",
            issuetype={'name': 'Task'},
            parent={'key': epic.key},
            customfield_risk_level=batch['risk_level'],
            timeestimate=batch['estimated_time_minutes'] * 60  # Convert to seconds
        )

        # Link to epic
        jira.create_issue_link('Relates', epic, issue)

    return epic.key
```

### 3. Confluence Integration

**Update Confluence documentation:**

```python
# confluence_integration.py
from atlassian import Confluence
import json
from pathlib import Path

def update_confluence_architecture(analysis_file: Path, confluence_url: str,
                                 username: str, password: str, space: str, page_id: str):
    """Update Confluence architecture documentation"""

    confluence = Confluence(url=confluence_url, username=username, password=password)

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Generate content
    content = generate_architecture_content(analysis)

    # Update page
    confluence.update_page(
        page_id=page_id,
        title="System Architecture",
        body=content,
        space=space
    )

def generate_architecture_content(analysis: Dict) -> str:
    """Generate Confluence content from analysis"""

    stats = analysis.get('scan_statistics', {})

    content = """
<h2>System Architecture Overview</h2>

<p>This page is automatically maintained by the LLM Intelligence System.</p>

<h3>Module Statistics</h3>
<table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td>Total Modules</td>
        <td>{total_files}</td>
    </tr>
    <tr>
        <td>Average Confidence</td>
        <td>{avg_confidence:.2f}</td>
    </tr>
    <tr>
        <td>High Confidence Modules</td>
        <td>{high_confidence}</td>
    </tr>
</table>

<h3>Module Distribution by Classification</h3>
<ul>
""".format(
        total_files=stats.get('total_files', 0),
        avg_confidence=stats.get('confidence_stats', {}).get('mean', 0),
        high_confidence=stats.get('confidence_stats', {}).get('high_confidence_count', 0)
    )

    for category, count in stats.get('classification_distribution', {}).get('primary', {}).items():
        percentage = count / stats.get('total_files', 1) * 100
        content += f"<li><strong>{category}:</strong> {count} modules ({percentage:.1f}%)</li>\n"

    content += "</ul><p><em>Last updated: {timestamp}</em></p>".format(
        timestamp=analysis.get('scan_timestamp', 'Unknown')
    )

    return content
```

## External Tool Integration

### 1. SonarQube Integration

**Integrate with SonarQube for enhanced code quality:**

```python
# sonarqube_integration.py
import json
from pathlib import Path
import requests

def send_analysis_to_sonarqube(analysis_file: Path, sonarqube_url: str, project_key: str):
    """Send intelligence analysis to SonarQube as external analysis"""

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Convert to SonarQube generic issue format
    issues = []

    for entry in analysis.get('intelligence_entries', []):
        if entry['confidence_score'] < 0.7:
            issues.append({
                "engineId": "llm-intelligence",
                "ruleId": "low-confidence-classification",
                "severity": "MINOR" if entry['confidence_score'] > 0.5 else "MAJOR",
                "type": "CODE_SMELL",
                "primaryLocation": {
                    "message": f"Low confidence classification: {entry['primary_classification']} ({entry['confidence_score']:.2f})",
                    "filePath": entry['relative_path'],
                    "textRange": {
                        "startLine": 1,
                        "endLine": 1
                    }
                }
            })

        # Add reorganization recommendations as info issues
        for recommendation in entry.get('reorganization_recommendations', []):
            issues.append({
                "engineId": "llm-intelligence",
                "ruleId": "reorganization-suggestion",
                "severity": "INFO",
                "type": "CODE_SMELL",
                "primaryLocation": {
                    "message": f"Reorganization suggestion: {recommendation}",
                    "filePath": entry['relative_path'],
                    "textRange": {
                        "startLine": 1,
                        "endLine": 1
                    }
                }
            })

    # Send to SonarQube
    payload = {
        "issues": issues
    }

    response = requests.post(
        f"{sonarqube_url}/api/issues/import",
        json=payload,
        params={"projectKey": project_key}
    )

    return response.status_code == 200
```

### 2. GitHub Integration

**Enhanced GitHub integration with code review automation:**

```python
# github_integration.py
import json
from pathlib import Path
from github import Github

def create_intelligence_pr_comment(repo_name: str, pr_number: int, token: str):
    """Create PR comment with intelligence analysis"""

    g = Github(token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Get intelligence analysis
    analysis_file = Path("tools/codebase_reorganizer/intelligence_output/llm_intelligence_map.json")

    if not analysis_file.exists():
        return False

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Generate comment content
    comment_body = "## 🤖 LLM Intelligence Analysis\n\n"

    # Add summary statistics
    stats = analysis.get('scan_statistics', {})
    confidence = stats.get('confidence_stats', {})

    comment_body += f"**Analysis Summary:**\n"
    comment_body += f"- Total modules analyzed: {stats.get('total_files', 0)}\n"
    comment_body += f"- Average confidence: {confidence.get('mean', 0):.2f}\n"
    comment_body += f"- High confidence modules: {confidence.get('high_confidence_count', 0)}\n\n"

    # Add key findings
    comment_body += "**Key Findings:**\n"

    # Find security modules
    security_modules = [
        entry for entry in analysis.get('intelligence_entries', [])
        if entry.get('primary_classification') == 'security'
    ]

    if security_modules:
        comment_body += f"- 🔒 **Security modules identified:** {len(security_modules)}\n"

    # Find low confidence modules
    low_confidence = [
        entry for entry in analysis.get('intelligence_entries', [])
        if entry.get('confidence_score', 1.0) < 0.7
    ]

    if low_confidence:
        comment_body += f"- ⚠️ **Low confidence modules:** {len(low_confidence)} (manual review recommended)\n"

    # Add top reorganization suggestions
    suggestions = []
    for entry in analysis.get('intelligence_entries', []):
        if entry.get('reorganization_recommendations'):
            suggestions.extend(entry['reorganization_recommendations'][:1])

    if suggestions:
        comment_body += "\n**Top Reorganization Suggestions:**\n"
        for i, suggestion in enumerate(suggestions[:3], 1):
            comment_body += f"{i}. {suggestion}\n"

    # Add disclaimer
    comment_body += "\n---\n*This analysis was generated automatically by the LLM Intelligence System. Please review and validate the findings.*"

    # Post comment
    pr.create_issue_comment(comment_body)
    return True

def create_reorganization_issue(repo_name: str, token: str):
    """Create GitHub issue for reorganization plan"""

    g = Github(token)
    repo = g.get_repo(repo_name)

    # Load reorganization plan
    plan_file = Path("tools/codebase_reorganizer/intelligence_output/reorganization_plan.json")

    if not plan_file.exists():
        return None

    with open(plan_file, 'r') as f:
        plan = json.load(f)

    # Create issue body
    body = f"""## 🤖 Automated Reorganization Plan

**Plan ID:** {plan.get('plan_id', 'Unknown')}
**Generated:** {plan.get('created_timestamp', 'Unknown')}

### Overview
- Total tasks: {plan.get('total_tasks', 0)}
- Total batches: {plan.get('total_batches', 0)}
- Estimated time: {plan.get('estimated_total_time_hours', 0):.1f} hours

### Reorganization Phases

"""

    for phase in plan.get('reorganization_phases', []):
        body += f"""#### Phase {phase['phase_number']}: {phase['phase_name']}
- **Description:** {phase['description']}
- **Risk Level:** {phase['risk_level']}
- **Estimated Time:** {phase['estimated_time_minutes']} minutes
- **Tasks:** {len(phase['tasks'])}

"""

    body += """
### Implementation Guidelines
"""

    for guideline in plan.get('execution_guidelines', []):
        body += f"- {guideline}\n"

    # Create issue
    issue = repo.create_issue(
        title=f"🤖 Codebase Reorganization Plan - {plan.get('plan_id', 'Unknown')}",
        body=body,
        labels=["automation", "architecture", "reorganization"]
    )

    return issue.number
```

### 3. ELK Stack Integration

**Send analysis data to Elasticsearch for visualization:**

```python
# elk_integration.py
import json
from pathlib import Path
from elasticsearch import Elasticsearch

def send_analysis_to_elk(analysis_file: Path, elasticsearch_url: str):
    """Send analysis data to Elasticsearch"""

    es = Elasticsearch([elasticsearch_url])

    with open(analysis_file, 'r') as f:
        analysis = json.load(f)

    # Index analysis metadata
    es.index(
        index='llm-intelligence-metadata',
        document={
            'scan_timestamp': analysis.get('scan_timestamp'),
            'total_files': analysis.get('total_files_scanned'),
            'total_lines': analysis.get('total_lines_analyzed'),
            'scan_id': analysis.get('scan_id')
        }
    )

    # Index individual module data
    for entry in analysis.get('intelligence_entries', []):
        document = {
            'scan_id': analysis.get('scan_id'),
            'relative_path': entry['relative_path'],
            'primary_classification': entry['primary_classification'],
            'secondary_classifications': entry['secondary_classifications'],
            'confidence_score': entry['confidence_score'],
            'complexity_assessment': entry['complexity_assessment'],
            'file_size': entry['file_size'],
            'line_count': entry['line_count'],
            'class_count': entry['class_count'],
            'function_count': entry['function_count'],
            'analysis_timestamp': entry['analysis_timestamp']
        }

        es.index(
            index='llm-intelligence-modules',
            document=document
        )

    return True

def create_kibana_dashboard(elasticsearch_url: str):
    """Create Kibana dashboard for intelligence analysis"""

    dashboard_config = {
        "title": "LLM Intelligence Analysis Dashboard",
        "description": "Monitor codebase intelligence analysis results",
        "panels": [
            {
                "type": "pie",
                "title": "Module Distribution by Classification",
                "index": "llm-intelligence-modules",
                "field": "primary_classification"
            },
            {
                "type": "histogram",
                "title": "Confidence Score Distribution",
                "index": "llm-intelligence-modules",
                "field": "confidence_score",
                "interval": 0.1
            },
            {
                "type": "line",
                "title": "Analysis Trends Over Time",
                "index": "llm-intelligence-metadata",
                "time_field": "scan_timestamp",
                "metrics": ["total_files", "total_lines"]
            }
        ]
    }

    # This would integrate with Kibana API to create the actual dashboard
    # For now, just return the configuration
    return dashboard_config
```

---

This comprehensive integration guide shows how the LLM Intelligence System can be seamlessly integrated into your existing development workflow, tools, and processes. The system is designed to enhance rather than replace your current practices, providing intelligent insights that augment human decision-making.

