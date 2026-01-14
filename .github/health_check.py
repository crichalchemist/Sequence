#!/usr/bin/env python3
"""
Health Check Service Configuration
Validates that the CI/CD infrastructure is properly configured
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class HealthCheckResult:
    """Results from a single health check"""
    name: str
    status: str  # "pass", "warn", "fail"
    message: str
    severity: str  # "critical", "warning", "info"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class CICDHealthCheck:
    """Comprehensive CI/CD health check suite"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.results: List[HealthCheckResult] = []
        
    def check_workflow_files(self) -> bool:
        """Verify workflow files exist and are valid YAML"""
        workflows_dir = self.repo_root / ".github" / "workflows"
        
        required_workflows = [
            "comprehensive-ci.yml",
            "sentry-monitoring.yml",
            "claude-analysis.yml",
            "env-config.yml"
        ]
        
        if not workflows_dir.exists():
            self.results.append(HealthCheckResult(
                name="Workflow Directory",
                status="fail",
                message=f"Workflows directory not found at {workflows_dir}",
                severity="critical"
            ))
            return False
        
        missing = []
        for workflow in required_workflows:
            workflow_path = workflows_dir / workflow
            if not workflow_path.exists():
                missing.append(workflow)
            else:
                # Basic YAML validation
                try:
                    import yaml
                    with open(workflow_path) as f:
                        yaml.safe_load(f)
                except ImportError:
                    pass  # YAML not installed, skip validation
                except Exception as e:
                    self.results.append(HealthCheckResult(
                        name=f"Workflow Syntax: {workflow}",
                        status="fail",
                        message=f"Invalid YAML: {e}",
                        severity="critical"
                    ))
                    return False
        
        if missing:
            self.results.append(HealthCheckResult(
                name="Workflow Files",
                status="fail",
                message=f"Missing workflows: {', '.join(missing)}",
                severity="critical"
            ))
            return False
        
        self.results.append(HealthCheckResult(
            name="Workflow Files",
            status="pass",
            message=f"All {len(required_workflows)} workflow files present and valid",
            severity="info"
        ))
        return True
    
    def check_secrets_configuration(self) -> bool:
        """Verify secrets are documented and configured"""
        secrets_file = self.repo_root / ".github" / "SECRETS_SETUP.md"
        
        required_secrets = [
            "SENTRY_DSN",
            "SENTRY_AUTH_TOKEN",
            "SENTRY_ORG",
            "SENTRY_PROJECT",
            "OPENAI_API_KEY",
            "CLAUDE_API_KEY"
        ]
        
        if not secrets_file.exists():
            self.results.append(HealthCheckResult(
                name="Secrets Documentation",
                status="warn",
                message="SECRETS_SETUP.md not found",
                severity="warning"
            ))
            return False
        
        # Check for documentation of required secrets
        with open(secrets_file) as f:
            content = f.read()
        
        missing_docs = [s for s in required_secrets if s not in content]
        
        if missing_docs:
            self.results.append(HealthCheckResult(
                name="Secrets Documentation",
                status="warn",
                message=f"Missing documentation for: {', '.join(missing_docs)}",
                severity="warning"
            ))
        else:
            self.results.append(HealthCheckResult(
                name="Secrets Documentation",
                status="pass",
                message=f"All {len(required_secrets)} secrets documented",
                severity="info"
            ))
        
        return True
    
    def check_configuration_files(self) -> bool:
        """Verify CI/CD configuration files exist"""
        required_files = {
            ".github/SECRETS_SETUP.md": "Secrets setup guide",
            ".github/CI_CD_INTEGRATION_GUIDE.md": "CI/CD integration guide",
            ".github/workflows/README.md": "Workflows README",
            "pyproject.toml": "Python project configuration",
            "pytest.ini": "Pytest configuration",
        }
        
        all_found = True
        for file_path, description in required_files.items():
            full_path = self.repo_root / file_path
            if full_path.exists():
                self.results.append(HealthCheckResult(
                    name=f"Config: {file_path}",
                    status="pass",
                    message=f"{description} found",
                    severity="info"
                ))
            else:
                self.results.append(HealthCheckResult(
                    name=f"Config: {file_path}",
                    status="warn",
                    message=f"{description} not found",
                    severity="warning"
                ))
                all_found = False
        
        return all_found
    
    def check_test_structure(self) -> bool:
        """Verify test structure is in place"""
        test_dirs = ["tests/", "tests/train/", "tests/data/"]
        
        all_found = True
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if test_path.exists():
                test_files = list(test_path.glob("test_*.py"))
                self.results.append(HealthCheckResult(
                    name=f"Test Directory: {test_dir}",
                    status="pass",
                    message=f"Found {len(test_files)} test files",
                    severity="info"
                ))
            else:
                # test_data/ is optional
                if test_dir != "tests/data/":
                    self.results.append(HealthCheckResult(
                        name=f"Test Directory: {test_dir}",
                        status="warn",
                        message=f"Directory not found",
                        severity="warning"
                    ))
                    all_found = False
        
        return all_found
    
    def check_python_environment(self) -> bool:
        """Check Python version and dependencies"""
        try:
            import sys
            version = sys.version_info
            
            if version.major < 3 or (version.major == 3 and version.minor < 10):
                self.results.append(HealthCheckResult(
                    name="Python Version",
                    status="warn",
                    message=f"Python {version.major}.{version.minor} detected (3.10+ recommended)",
                    severity="warning"
                ))
                return False
            
            self.results.append(HealthCheckResult(
                name="Python Version",
                status="pass",
                message=f"Python {version.major}.{version.minor} configured",
                severity="info"
            ))
            return True
        
        except Exception as e:
            self.results.append(HealthCheckResult(
                name="Python Version Check",
                status="fail",
                message=str(e),
                severity="critical"
            ))
            return False
    
    def check_dependencies(self) -> bool:
        """Verify required dependencies are installable"""
        required_packages = {
            "pytest": "Testing framework",
            "ruff": "Linting tool",
            "black": "Code formatter",
            "isort": "Import sorter",
            "bandit": "Security scanner",
        }
        
        all_found = True
        for package, description in required_packages.items():
            try:
                __import__(package)
                self.results.append(HealthCheckResult(
                    name=f"Dependency: {package}",
                    status="pass",
                    message=f"{description} installed",
                    severity="info"
                ))
            except ImportError:
                self.results.append(HealthCheckResult(
                    name=f"Dependency: {package}",
                    status="warn",
                    message=f"{description} not installed",
                    severity="warning"
                ))
                all_found = False
        
        return all_found
    
    def run_all_checks(self) -> Dict:
        """Run all health checks and return summary"""
        print("🏥 Running CI/CD Health Checks...\n")
        
        checks = [
            ("Workflow Files", self.check_workflow_files),
            ("Configuration Files", self.check_configuration_files),
            ("Secrets Setup", self.check_secrets_configuration),
            ("Test Structure", self.check_test_structure),
            ("Python Environment", self.check_python_environment),
            ("Dependencies", self.check_dependencies),
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                print(f"{'✅' if result else '⚠️ '} {check_name}")
            except Exception as e:
                print(f"❌ {check_name}: {e}")
                self.results.append(HealthCheckResult(
                    name=check_name,
                    status="fail",
                    message=str(e),
                    severity="critical"
                ))
        
        # Generate summary
        passed = len([r for r in self.results if r.status == "pass"])
        warned = len([r for r in self.results if r.status == "warn"])
        failed = len([r for r in self.results if r.status == "fail"])
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": len(self.results),
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "status": "healthy" if failed == 0 else "degraded" if warned == 0 else "unhealthy",
            "results": [asdict(r) for r in self.results]
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print health check summary"""
        print("\n" + "="*60)
        print(f"Health Check Summary: {summary['status'].upper()}")
        print("="*60)
        print(f"Total Checks:  {summary['total_checks']}")
        print(f"✅ Passed:     {summary['passed']}")
        print(f"⚠️  Warned:     {summary['warned']}")
        print(f"❌ Failed:     {summary['failed']}")
        print("="*60)
        
        if summary['warned'] > 0:
            print("\nWarnings:")
            for result in summary['results']:
                if result['status'] == 'warn':
                    print(f"  ⚠️  {result['name']}: {result['message']}")
        
        if summary['failed'] > 0:
            print("\nFailures:")
            for result in summary['results']:
                if result['status'] == 'fail':
                    print(f"  ❌ {result['name']}: {result['message']}")
        
        print()
    
    def export_results(self, output_file: str = "health_check_results.json"):
        """Export results to JSON"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results exported to {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI/CD Health Check")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--export", type=str, help="Export results to file")
    
    args = parser.parse_args()
    
    checker = CICDHealthCheck(args.repo)
    summary = checker.run_all_checks()
    
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        checker.print_summary(summary)
    
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Exit with appropriate code
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
