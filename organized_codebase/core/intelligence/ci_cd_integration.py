"""
CI/CD Integration for Documentation Generation

This module provides CI/CD pipeline integration for automated documentation
generation, validation, and deployment. Supports GitHub Actions, GitLab CI,
Jenkins, and other popular CI/CD platforms.
"""

import os
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..core.doc_generator import DocumentationEngine
from ..quality.doc_validator import DocumentationValidator
from ..quality.style_checker import DocumentationStyleChecker


class CIPlatform(Enum):
    """Supported CI/CD platforms."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"


@dataclass
class CIConfig:
    """Configuration for CI/CD integration."""
    platform: CIPlatform
    project_path: str
    output_path: str = "docs/"
    validate_on_pr: bool = True
    auto_deploy_docs: bool = False
    notify_on_failure: bool = True
    style_checks: bool = True
    coverage_threshold: float = 0.8
    fail_on_style_violations: bool = False
    
    # Platform-specific settings
    github_token_var: str = "GITHUB_TOKEN"
    gitlab_token_var: str = "GITLAB_TOKEN"
    deployment_branch: str = "main"
    pages_branch: str = "gh-pages"


class CICDDocumentationIntegration:
    """CI/CD integration for automated documentation workflows."""
    
    def __init__(self, config: CIConfig):
        self.config = config
        self.doc_engine = DocumentationEngine()
        self.validator = DocumentationValidator()
        self.style_checker = DocumentationStyleChecker()
    
    def generate_ci_config(self) -> str:
        """Generate CI/CD configuration file for the specified platform."""
        if self.config.platform == CIPlatform.GITHUB_ACTIONS:
            return self._generate_github_actions_config()
        elif self.config.platform == CIPlatform.GITLAB_CI:
            return self._generate_gitlab_ci_config()
        elif self.config.platform == CIPlatform.JENKINS:
            return self._generate_jenkins_config()
        elif self.config.platform == CIPlatform.AZURE_DEVOPS:
            return self._generate_azure_devops_config()
        elif self.config.platform == CIPlatform.CIRCLECI:
            return self._generate_circleci_config()
        else:
            raise ValueError(f"Unsupported CI platform: {self.config.platform}")
    
    def _generate_github_actions_config(self) -> str:
        """Generate GitHub Actions workflow configuration."""
        config = {
            "name": "Documentation CI/CD",
            "on": {
                "push": {"branches": [self.config.deployment_branch]},
                "pull_request": {"branches": [self.config.deployment_branch]}
            },
            "jobs": {
                "documentation": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Generate documentation",
                            "run": f"python -m testmaster.intelligence.documentation.cli generate --output {self.config.output_path}"
                        }
                    ]
                }
            }
        }
        
        # Add validation step if enabled
        if self.config.validate_on_pr:
            config["jobs"]["documentation"]["steps"].append({
                "name": "Validate documentation",
                "run": f"python -m testmaster.intelligence.documentation.cli validate --path {self.config.output_path}"
            })
        
        # Add style checking if enabled
        if self.config.style_checks:
            config["jobs"]["documentation"]["steps"].append({
                "name": "Check documentation style",
                "run": f"python -m testmaster.intelligence.documentation.cli style-check --path {self.config.project_path}"
            })
        
        # Add deployment step if enabled
        if self.config.auto_deploy_docs:
            config["jobs"]["documentation"]["steps"].extend([
                {
                    "name": "Deploy to GitHub Pages",
                    "uses": "peaceiris/actions-gh-pages@v3",
                    "if": f"github.ref == 'refs/heads/{self.config.deployment_branch}'",
                    "with": {
                        "github_token": f"${{{{ secrets.{self.config.github_token_var} }}}}",
                        "publish_dir": self.config.output_path
                    }
                }
            ])
        
        return yaml.dump(config, default_flow_style=False)
    
    def _generate_gitlab_ci_config(self) -> str:
        """Generate GitLab CI configuration."""
        config = {
            "stages": ["validate", "build", "deploy"],
            "variables": {
                "DOC_OUTPUT_PATH": self.config.output_path
            },
            "before_script": [
                "pip install -r requirements.txt"
            ],
            "validate_docs": {
                "stage": "validate",
                "script": [
                    f"python -m testmaster.intelligence.documentation.cli validate --path {self.config.project_path}"
                ],
                "only": ["merge_requests"]
            },
            "build_docs": {
                "stage": "build",
                "script": [
                    f"python -m testmaster.intelligence.documentation.cli generate --output {self.config.output_path}"
                ],
                "artifacts": {
                    "paths": [self.config.output_path],
                    "expire_in": "1 hour"
                }
            }
        }
        
        if self.config.auto_deploy_docs:
            config["pages"] = {
                "stage": "deploy",
                "script": [
                    f"mkdir public && cp -r {self.config.output_path}/* public/"
                ],
                "artifacts": {
                    "paths": ["public"]
                },
                "only": [self.config.deployment_branch]
            }
        
        return yaml.dump(config, default_flow_style=False)
    
    def _generate_jenkins_config(self) -> str:
        """Generate Jenkinsfile configuration."""
        jenkins_script = f"""
pipeline {{
    agent any
    
    stages {{
        stage('Setup') {{
            steps {{
                sh 'pip install -r requirements.txt'
            }}
        }}
        
        stage('Validate Documentation') {{
            when {{
                changeRequest()
            }}
            steps {{
                sh 'python -m testmaster.intelligence.documentation.cli validate --path {self.config.project_path}'
            }}
        }}
        
        stage('Generate Documentation') {{
            steps {{
                sh 'python -m testmaster.intelligence.documentation.cli generate --output {self.config.output_path}'
            }}
        }}
        
        stage('Style Check') {{
            when {{
                expression {{ {str(self.config.style_checks).lower()} }}
            }}
            steps {{
                sh 'python -m testmaster.intelligence.documentation.cli style-check --path {self.config.project_path}'
            }}
        }}
        
        stage('Deploy Documentation') {{
            when {{
                branch '{self.config.deployment_branch}'
                expression {{ {str(self.config.auto_deploy_docs).lower()} }}
            }}
            steps {{
                // Add your deployment commands here
                sh 'echo "Deploy documentation to your preferred hosting service"'
            }}
        }}
    }}
    
    post {{
        failure {{
            // Notification logic
            echo 'Documentation pipeline failed!'
        }}
        success {{
            archiveArtifacts artifacts: '{self.config.output_path}/**', fingerprint: true
        }}
    }}
}}
"""
        return jenkins_script.strip()
    
    def _generate_azure_devops_config(self) -> str:
        """Generate Azure DevOps pipeline configuration."""
        config = {
            "trigger": [self.config.deployment_branch],
            "pr": [self.config.deployment_branch],
            "pool": {"vmImage": "ubuntu-latest"},
            "steps": [
                {
                    "task": "UsePythonVersion@0",
                    "inputs": {"versionSpec": "3.9"}
                },
                {
                    "script": "pip install -r requirements.txt",
                    "displayName": "Install dependencies"
                },
                {
                    "script": f"python -m testmaster.intelligence.documentation.cli generate --output {self.config.output_path}",
                    "displayName": "Generate documentation"
                }
            ]
        }
        
        if self.config.validate_on_pr:
            config["steps"].append({
                "script": f"python -m testmaster.intelligence.documentation.cli validate --path {self.config.project_path}",
                "displayName": "Validate documentation",
                "condition": "eq(variables['Build.Reason'], 'PullRequest')"
            })
        
        return yaml.dump(config, default_flow_style=False)
    
    def _generate_circleci_config(self) -> str:
        """Generate CircleCI configuration."""
        config = {
            "version": 2.1,
            "jobs": {
                "documentation": {
                    "docker": [{"image": "python:3.9"}],
                    "steps": [
                        "checkout",
                        {"run": "pip install -r requirements.txt"},
                        {
                            "run": {
                                "name": "Generate documentation",
                                "command": f"python -m testmaster.intelligence.documentation.cli generate --output {self.config.output_path}"
                            }
                        }
                    ]
                }
            },
            "workflows": {
                "version": 2,
                "documentation_workflow": {
                    "jobs": ["documentation"]
                }
            }
        }
        
        if self.config.validate_on_pr:
            config["jobs"]["documentation"]["steps"].append({
                "run": {
                    "name": "Validate documentation",
                    "command": f"python -m testmaster.intelligence.documentation.cli validate --path {self.config.project_path}"
                }
            })
        
        return yaml.dump(config, default_flow_style=False)
    
    def setup_ci_environment(self) -> Dict[str, str]:
        """Set up CI environment with required files and configurations."""
        files_created = {}
        
        # Generate main CI config file
        config_content = self.generate_ci_config()
        config_path = self._get_config_file_path()
        
        # Ensure directory exists
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config file
        with open(config_path, 'w') as f:
            f.write(config_content)
        files_created[config_path] = "Main CI/CD configuration"
        
        # Create documentation CLI script if it doesn't exist
        cli_script = self._generate_cli_script()
        cli_path = os.path.join(self.config.project_path, "docs_cli.py")
        with open(cli_path, 'w') as f:
            f.write(cli_script)
        files_created[cli_path] = "Documentation CLI script"
        
        # Create requirements snippet for documentation dependencies
        requirements_snippet = self._generate_requirements_snippet()
        req_path = os.path.join(self.config.project_path, "docs_requirements.txt")
        with open(req_path, 'w') as f:
            f.write(requirements_snippet)
        files_created[req_path] = "Documentation requirements"
        
        return files_created
    
    def _get_config_file_path(self) -> str:
        """Get the appropriate config file path for the CI platform."""
        if self.config.platform == CIPlatform.GITHUB_ACTIONS:
            return os.path.join(self.config.project_path, ".github", "workflows", "docs.yml")
        elif self.config.platform == CIPlatform.GITLAB_CI:
            return os.path.join(self.config.project_path, ".gitlab-ci.yml")
        elif self.config.platform == CIPlatform.JENKINS:
            return os.path.join(self.config.project_path, "Jenkinsfile")
        elif self.config.platform == CIPlatform.AZURE_DEVOPS:
            return os.path.join(self.config.project_path, "azure-pipelines.yml")
        elif self.config.platform == CIPlatform.CIRCLECI:
            return os.path.join(self.config.project_path, ".circleci", "config.yml")
        else:
            raise ValueError(f"Unknown CI platform: {self.config.platform}")
    
    def _generate_cli_script(self) -> str:
        """Generate a CLI script for documentation operations."""
        return '''#!/usr/bin/env python3
"""
Documentation CLI for CI/CD operations.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from testmaster.C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.intelligence.documentation import DocumentationEngine
from testmaster.intelligence.documentation.quality.doc_validator import DocumentationValidator
from testmaster.intelligence.documentation.quality.style_checker import DocumentationStyleChecker


def generate_docs(output_path: str):
    """Generate documentation."""
    engine = DocumentationEngine()
    try:
        # Generate comprehensive documentation
        results = engine.generate_comprehensive_docs(output_path)
        print(f"Documentation generated successfully in {output_path}")
        return True
    except Exception as e:
        print(f"Error generating documentation: {e}")
        return False


def validate_docs(path: str):
    """Validate existing documentation."""
    validator = DocumentationValidator()
    try:
        results = validator.validate_project(path)
        if results.is_valid:
            print("Documentation validation passed")
            return True
        else:
            print(f"Documentation validation failed: {results.errors}")
            return False
    except Exception as e:
        print(f"Error validating documentation: {e}")
        return False


def check_style(path: str):
    """Check documentation style compliance."""
    checker = DocumentationStyleChecker()
    try:
        results = checker.check_project(path)
        if results.compliant:
            print("Style check passed")
            return True
        else:
            print(f"Style violations found: {len(results.violations)}")
            for violation in results.violations[:5]:  # Show first 5
                print(f"  - {violation.file_path}:{violation.line}: {violation.message}")
            return False
    except Exception as e:
        print(f"Error checking style: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Documentation CI/CD operations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate documentation")
    gen_parser.add_argument("--output", required=True, help="Output directory")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate documentation")
    val_parser.add_argument("--path", required=True, help="Path to validate")
    
    # Style check command
    style_parser = subparsers.add_parser("style-check", help="Check documentation style")
    style_parser.add_argument("--path", required=True, help="Path to check")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        success = generate_docs(args.output)
    elif args.command == "validate":
        success = validate_docs(args.path)
    elif args.command == "style-check":
        success = check_style(args.path)
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
'''
    
    def _generate_requirements_snippet(self) -> str:
        """Generate requirements for documentation dependencies."""
        return """# Documentation generation dependencies
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
jinja2>=3.1.0
pystache>=0.6.0
pyyaml>=6.0
beautifulsoup4>=4.12.0
markdown>=3.5.0
"""
    
    def run_documentation_pipeline(self) -> Dict[str, Any]:
        """Run the complete documentation pipeline locally for testing."""
        results = {
            "generation": False,
            "validation": False,
            "style_check": False,
            "errors": []
        }
        
        try:
            # Generate documentation
            print("Generating documentation...")
            self.doc_engine.generate_comprehensive_docs(self.config.output_path)
            results["generation"] = True
            print("✓ Documentation generation completed")
            
            # Validate documentation
            if self.config.validate_on_pr:
                print("Validating documentation...")
                validation_result = self.validator.validate_project(self.config.project_path)
                results["validation"] = validation_result.is_valid
                if not validation_result.is_valid:
                    results["errors"].extend(validation_result.errors)
                print("✓ Documentation validation completed")
            
            # Check style
            if self.config.style_checks:
                print("Checking documentation style...")
                style_result = self.style_checker.check_project(self.config.project_path)
                results["style_check"] = style_result.compliant
                if not style_result.compliant:
                    results["errors"].extend([v.message for v in style_result.violations[:10]])
                print("✓ Style check completed")
            
        except Exception as e:
            results["errors"].append(str(e))
            print(f"✗ Pipeline failed: {e}")
        
        return results


def create_ci_integration(platform: str, project_path: str, **kwargs) -> CICDDocumentationIntegration:
    """Factory function to create CI/CD integration."""
    platform_enum = CIPlatform(platform.lower())
    config = CIConfig(
        platform=platform_enum,
        project_path=project_path,
        **kwargs
    )
    return CICDDocumentationIntegration(config)