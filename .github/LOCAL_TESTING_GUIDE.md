# Local Testing Guide for CI/CD Workflows

This guide helps you test GitHub Actions workflows locally before committing.

## 🚀 Quick Start

### Install Act

Act is a tool to run GitHub Actions locally.

```bash
# macOS
brew install act

# Linux (Ubuntu/Debian)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

# Windows (using chocolatey)
choco install act-cli

# Or Docker approach (platform independent)
docker pull ghcr.io/nektos/act
```

### Verify Installation

```bash
act --version
act --list
```

## 📋 Running Workflows Locally

### List All Workflows

```bash
act --list
```

Output shows all jobs and their commands.

### Run Specific Job

```bash
# Run quality-checks job
act -j quality-checks

# Run test-suite job
act -j test-suite

# Run a specific workflow
act --workflows .github/workflows/comprehensive-ci.yml
```

### Run Full Workflow

```bash
# Run all jobs in comprehensive-ci.yml
act -W .github/workflows/comprehensive-ci.yml
```

### Run on Specific Event

```bash
# Simulate pull request event
act pull_request

# Simulate push event
act push

# Simulate schedule event
act schedule
```

## 🔧 Configuration

### Set Up Act Configuration

Create `.actrc` in repository root:

```bash
# Use Ubuntu runners instead of default
-P ubuntu-latest=ghcr.io/catthehacker/ubuntu:latest

# Set remote repo (for checkout)
--remote-name origin

# Set actor name
--actor github-user

# Use Docker socket (mount host Docker)
--use-docker-host
```

### Create Local Secret File

Create `.secrets` or use GitHub CLI:

```bash
# Method 1: Create .secrets file (git ignored)
cat > .secrets << 'EOF'
SENTRY_DSN=https://example@example.ingest.sentry.io/123456
SENTRY_AUTH_TOKEN=sntrys_xxxxx
SENTRY_ORG=your-org
SENTRY_PROJECT=your-project
OPENAI_API_KEY=sk-xxxxx
CLAUDE_API_KEY=sk-ant-xxxxx
GITHUB_TOKEN=ghp_xxxxx
EOF

# Method 2: Use GitHub CLI
gh secret list  # See secrets in GitHub

# Method 3: Set in environment
export SENTRY_DSN="https://example@example.ingest.sentry.io/123456"
```

### Pass Secrets to Act

```bash
# Load from file
act -j quality-checks --secret-file .secrets

# Pass individual secret
act -j quality-checks --secret SENTRY_DSN="https://example@example.ingest.sentry.io/123456"

# Load from environment
export SENTRY_DSN="..."
act -j quality-checks
```

## 🧪 Testing Scenarios

### Test 1: Quality Checks

```bash
# Run code quality checks locally
act -j quality-checks --secret-file .secrets

# What it does:
# - Ruff linting
# - Black formatting check
# - isort import checking
# - Bandit security scan
# - Safety dependency check
```

### Test 2: Unit Tests

```bash
# Run unit tests
act -j test-suite -l  # List the job details first

# Run with custom Python
act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:latest \
    -j test-suite \
    --secret-file .secrets

# What it does:
# - Runs pytest for unit tests
# - Generates coverage reports
# - Uploads artifacts
```

### Test 3: Build Artifacts

```bash
# Test package building
act -j build-artifacts --secret-file .secrets

# What it does:
# - Creates Python packages
# - Validates distributions
# - Tests twine check
```

### Test 4: Full Pipeline

```bash
# Run entire comprehensive-ci workflow
act -W .github/workflows/comprehensive-ci.yml \
    --secret-file .secrets \
    -v  # verbose output

# Dry run (no actual execution)
act -W .github/workflows/comprehensive-ci.yml \
    --dry  # Show what would execute
```

### Test 5: Sentry Integration

```bash
# Test Sentry monitoring workflow
act -W .github/workflows/sentry-monitoring.yml \
    --secret-file .secrets \
    -e push  # Simulate push event
```

### Test 6: Claude Analysis

```bash
# Test Claude code analysis workflow
act -W .github/workflows/claude-analysis.yml \
    --secret-file .secrets \
    -e pull_request

# This will:
# - Analyze Python files
# - Run architecture review
# - Check performance
# - Scan for security issues
```

## 🔍 Debugging

### Verbose Output

```bash
act -j quality-checks -v  # Very verbose
act -j quality-checks --debug  # Debug mode
```

### Keep Docker Container

```bash
# Keep container for inspection
act -j quality-checks --keep-containers
docker ps  # View running containers
docker exec -it CONTAINER_ID /bin/bash
```

### View Container Logs

```bash
# Detailed logs
act -j quality-checks --log-level debug 2>&1 | tee act.log

# Save to file for analysis
act -j quality-checks > test-results.log 2>&1
```

### Test Single Step

Edit workflow to run only step you want:

```yaml
steps:
  - name: Check Python
    run: python --version
  
  # Comment out other steps for faster iteration
  # - name: Run tests
  #   run: pytest tests/
```

## 📊 Common Troubleshooting

### Issue: "Docker not running"

```bash
# Start Docker
sudo systemctl start docker  # Linux
open -a Docker  # macOS
```

### Issue: "Permission denied"

```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "Secrets not found"

```bash
# Verify secrets file exists and is readable
cat .secrets | head -5

# Make sure act can read it
act -j quality-checks --secret-file .secrets -v
```

### Issue: "Container not found"

```bash
# Pull required image
docker pull ghcr.io/catthehacker/ubuntu:latest

# Try with specific image
act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:latest \
    -j quality-checks
```

### Issue: "Python packages not found"

```bash
# Cache might be invalid, force refresh
act -j quality-checks --cache-retention-days 0
```

## 🔄 Workflow for Development

### Typical Development Cycle

```bash
# 1. Make code changes
vim my_code.py

# 2. Run local checks
act -j quality-checks --secret-file .secrets

# 3. Fix any issues
# ... edit code ...

# 4. Run tests
act -j test-suite --secret-file .secrets

# 5. Test full pipeline
act -W .github/workflows/comprehensive-ci.yml --secret-file .secrets

# 6. Commit and push
git add .
git commit -m "Fix: ..."
git push

# 7. Monitor GitHub Actions (if needed)
gh run list
```

### Pre-Commit Testing

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running local quality checks..."
act -j quality-checks --secret-file .secrets || exit 1
echo "✅ Checks passed!"
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

## 📈 Advanced Usage

### Run with Different Python Version

```bash
# Use specific Python image
act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:full-latest \
    -e python --version
```

### Custom Environment Variables

```bash
# Set environment
act -j quality-checks \
    -e PYTHON_VERSION=3.11 \
    -e MIN_TEST_COVERAGE=85
```

### Multiple Jobs in Sequence

```bash
# Run specific jobs in order
act -j quality-checks && act -j test-suite && act -j build-artifacts
```

### Parallel Execution

```bash
# Run jobs in parallel (use with caution)
act --parallel 2 -W .github/workflows/comprehensive-ci.yml
```

## 🎯 Best Practices

1. **Always test locally before pushing**
   ```bash
   act -j quality-checks
   act -j test-suite
   ```

2. **Keep secrets in `.secrets` file (git ignored)**
   ```bash
   echo ".secrets" >> .gitignore
   ```

3. **Use verbose output to debug**
   ```bash
   act -j quality-checks -v
   ```

4. **Save logs for review**
   ```bash
   act -j quality-checks > test.log 2>&1
   ```

5. **Update act regularly**
   ```bash
   act --version
   brew upgrade act  # or your package manager
   ```

6. **Keep .actrc in sync with team**
   ```bash
   # Share .actrc configuration
   git add .actrc
   git commit -m "docs: Update act configuration"
   ```

## 📚 Additional Resources

- [Act GitHub Repository](https://github.com/nektos/act)
- [Act Documentation](https://github.com/nektos/act/blob/master/README.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub - catthehacker/ubuntu](https://hub.docker.com/r/catthehacker/ubuntu)

## 🆘 Getting Help

If act doesn't work:

```bash
# Check GitHub issues
act --help

# Create issue with verbose output
act -j quality-checks -v 2>&1 > debug.log

# Share debug.log in issue
```

---

**Last Updated**: January 2024
**Tested With**: Act v0.2.50+, Docker 20.10+, Python 3.11
