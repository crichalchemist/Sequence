# Comprehensive CI/CD Workflow Integration Guide

## Architecture Overview

This document describes the complete CI/CD pipeline architecture that integrates with Sentry, CodeRabbit, Claude, and GitHub Actions.

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Event Trigger                        │
│              (Push, Pull Request, Schedule)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬─────────────────┐
        │                  │                  │                 │
        ▼                  ▼                  ▼                 ▼
   ┌─────────┐      ┌────────────┐    ┌─────────────┐    ┌──────────┐
   │ Quality │      │    Test    │    │ CodeRabbit  │    │ Claude   │
   │ Checks  │      │   Suite    │    │   Review    │    │ Analysis │
   │ (Ruff)  │      │  (pytest)  │    │  (AI PR)    │    │ (AI)     │
   └────┬────┘      └─────┬──────┘    └──────┬──────┘    └────┬─────┘
        │                 │                  │                │
        └─────────────────┼──────────────────┼────────────────┘
                          │
        ┌─────────────────┼──────────────────────────┐
        │                 │                          │
        ▼                 ▼                          ▼
   ┌──────────┐    ┌───────────┐          ┌──────────────────┐
   │  Build   │    │ Sentry    │          │ Performance      │
   │Artifacts │    │Integration│          │ Monitoring       │
   └────┬─────┘    └─────┬─────┘          └────────┬─────────┘
        │                │                         │
        └────────────────┼─────────────────────────┘
                         │
                         ▼
                   ┌─────────────┐
                   │   Report    │
                   │  & Status   │
                   └─────────────┘
```

## Workflow Files

### 1. **comprehensive-ci.yml** - Main CI Pipeline

**Triggers:**
- Push to main, master, develop branches
- Pull requests to main, master, develop
- Daily schedule (2 AM UTC)

**Jobs (Sequential):**
1. **quality-checks** - Ruff, Black, isort, Bandit, Safety
2. **test-suite** - Unit, integration, Colab tests
3. **coderabbit-review** - Automated code review (PRs only)
4. **claude-analysis** - AI code analysis (PRs only)
5. **build-artifacts** - Package building and validation
6. **sentry-release** - Release management (main only)
7. **report-status** - Final reporting

**Key Features:**
- Parallel test groups (unit, integration, colab)
- Coverage reporting with Codecov
- Artifact retention and uploads
- Concurrency control (cancels in-progress on new push)

**Outputs:**
- Coverage reports (htmlcov/)
- Quality reports (format.diff, import.diff, bandit, safety)
- Build distributions (dist/)
- GitHub Step Summary integration

### 2. **sentry-monitoring.yml** - Error Tracking & Monitoring

**Triggers:**
- Workflow completion (failures)
- Push to main
- Every 6 hours (health check)

**Jobs:**
1. **monitor-failures** - Reports CI failures to Sentry
2. **health-check** - System health verification
3. **sentry-release-artifacts** - Release & sourcemap management
4. **performance-monitoring** - Metrics collection
5. **auto-triage** - Issue auto-labeling

**Key Features:**
- Automatic error reporting
- Release tracking with commit history
- Performance metrics collection
- Issue auto-triage and staling

**Integration Points:**
- Sends workflow failures as Sentry events
- Creates releases with sourcemaps
- Reports health status to Sentry

### 3. **claude-analysis.yml** - AI Code Review

**Triggers:**
- Pull request (opened, synchronize, reopened)
- Issue comments containing `@claude`

**Jobs:**
1. **claude-review** - Comprehensive code analysis
   - Architecture review
   - Performance analysis
   - Security review
2. **claude-commands** - Interactive commands

**Key Features:**
- Multi-dimensional code analysis
- Automated PR comments with findings
- Interactive @claude commands
- Security vulnerability detection

**Command Examples:**
```
@claude analyze    # Run full analysis
@claude suggest    # Generate suggestions
@claude explain    # Explain code
```

## Configuration Files

### pyproject.toml - Project Configuration

Already configured with:
- Ruff rules: E, W, F, I, N, UP, B, C4, SIM
- Line length: 100
- Python target: 3.10+
- Per-file ignores for tests

### pytest.ini - Test Configuration

Already configured with:
- Test paths and patterns
- Markers for test categorization
- Timeout defaults
- Coverage settings

## Integration Details

### Sentry Integration

**Environment Setup:**
```python
# In your application code
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[DjangoIntegration()],
    traces_sample_rate=1.0,
    environment=os.getenv('ENVIRONMENT', 'development')
)
```

**Release Tracking:**
```python
# Sentry automatically tracks releases via CI/CD
# Commit history is automatically populated
# Source maps are uploaded for debugging
```

**Error Capture:**
- Automatic exception capture
- Custom event reporting
- Performance monitoring
- Release correlation

### CodeRabbit Integration

**PR Review Process:**
1. PR created/updated
2. CodeRabbit checks out code
3. Uses OpenAI to analyze changes
4. Posts detailed review comments
5. Suggests improvements

**Configuration in workflows/coderabbit:**
```yaml
- uses: coderabbitai/action@main
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Review Focus Areas:**
- Code quality and style
- Potential bugs
- Security issues
- Performance concerns
- Best practices

### Claude Integration

**Analysis Workflow:**
1. PR files downloaded
2. Python analysis scripts execute
3. Architecture, performance, security reviewed
4. Findings compiled to markdown
5. Comment posted to PR

**Custom Analysis Points:**
- sys.path manipulation
- Bare except clauses
- Global variables
- Function complexity (>50 functions)
- Nested loops and N+1 queries
- Security vulnerabilities (eval, exec, pickle)

### Codex Integration

While Codex is mentioned, modern implementations typically use:
- GitHub Copilot (built-in)
- Claude for generation
- OpenAI API for code completion

Can be integrated by:
1. Enabling GitHub Copilot on repository
2. Using Claude for code suggestions
3. Adding code generation workflows

## Running Workflows Locally

### Test Locally with Act

```bash
# Install act
brew install act

# Run a specific workflow
act -j quality-checks

# Run with specific event
act pull_request

# List all jobs
act -l
```

### Manual Testing

```bash
# Run quality checks
ruff check .
black --check .
isort --check .

# Run tests
pytest tests/ -v

# Check security
bandit -r . -ll

# Check dependencies
safety check
```

## Monitoring & Alerts

### GitHub Status Checks

All workflows create status checks visible in:
- Pull request UI
- Branch protection rules
- Commit history

### Sentry Dashboard

Monitor at: https://sentry.io/organizations/{ORG}/issues/

Track:
- Error rates
- Release stability
- Performance metrics
- User impact

### GitHub Actions Dashboard

Visible in repository:
- Actions tab
- Workflow run history
- Job duration and status
- Artifact downloads

## Customization Guide

### Adding New Quality Checks

In `comprehensive-ci.yml` quality-checks job:

```yaml
- name: Your New Check
  id: newcheck
  run: |
    # Your check command
    echo "Running custom check..."
```

### Extending Claude Analysis

In `claude-analysis.yml`:

```python
# Add to analysis script
if 'TODO' in content:
    analysis["todo_items"].append(f"Found TODO in {py_file}")
```

### Adding New Test Groups

In `comprehensive-ci.yml` test-suite matrix:

```yaml
matrix:
  test-group: [unit, integration, colab, performance]
```

## Troubleshooting

### Workflow Not Triggering

- Check branch name matches trigger
- Verify `.github/workflows/` permissions
- Ensure YAML syntax is valid
- Check repository settings → Actions

### Secrets Not Found

```bash
# Verify secrets are set
gh secret list

# Add missing secrets
gh secret set SECRET_NAME -b "value"
```

### Tests Timing Out

- Increase `timeout-minutes` in job
- Add `--timeout=N` to pytest
- Check for resource constraints
- Optimize test performance

### Sentry Not Receiving Events

- Verify DSN is correct
- Check SENTRY_AUTH_TOKEN permissions
- Ensure environment variables passed
- Check Sentry project settings

## Performance Optimization

### Caching Strategy

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### Parallel Execution

- Test groups run in parallel
- Reviews run independently
- Separate workflow files for different concerns

### Resource Management

- Linux runners (fastest)
- Artifact retention limited to 30 days
- Concurrency control prevents resource waste

## Security Considerations

1. **Secret Management**
   - Never log secrets
   - Use GitHub Secrets, not environment variables in YAML
   - Rotate tokens regularly

2. **Workflow Security**
   - Read-only checkouts by default
   - Specific permissions per job
   - No elevated privileges

3. **Dependency Security**
   - Bandit for code security
   - Safety for dependency vulnerabilities
   - Regular updates via dependency scanning

4. **Access Control**
   - Branch protection rules
   - Required status checks
   - Code review requirements

## Best Practices

1. **Keep workflows simple** - Use separate files for concerns
2. **Fail fast** - Run quick checks first
3. **Cache aggressively** - Minimize downloads
4. **Archive artifacts** - Useful for debugging
5. **Monitor costs** - CI/CD can be expensive at scale
6. **Document changes** - Update this guide when modifying
7. **Test locally** - Use `act` before pushing
8. **Review regularly** - Audit workflows quarterly

## Maintenance Schedule

- **Weekly**: Monitor Sentry issues, review PR trends
- **Monthly**: Check API usage/costs, update dependencies
- **Quarterly**: Rotate API tokens, audit access permissions
- **Yearly**: Major version updates, comprehensive review

---

**Document Version**: 1.0
**Last Updated**: January 2024
**Maintained By**: Your Team

For questions or updates, please create an issue with the `ci-cd` label.
