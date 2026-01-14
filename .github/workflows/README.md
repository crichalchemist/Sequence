# CI/CD Pipeline Configuration

This directory contains the comprehensive GitHub Actions CI/CD pipeline configuration for the Sequence project.

## 📊 Status Badges

Add these to your main README.md to display pipeline status:

```markdown
[![Comprehensive CI/CD Pipeline](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/comprehensive-ci.yml/badge.svg?branch=main)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/comprehensive-ci.yml)
[![Sentry Monitoring](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/sentry-monitoring.yml/badge.svg?branch=main)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/sentry-monitoring.yml)
[![Claude Analysis](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/claude-analysis.yml/badge.svg?branch=main)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/claude-analysis.yml)

[![codecov](https://codecov.io/gh/YOUR_ORG/YOUR_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_ORG/YOUR_REPO)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## 📁 Workflow Files

### Core Workflows

#### 1. **comprehensive-ci.yml**
Main CI/CD pipeline with 7 sequential stages:
- Quality Checks (Ruff, Black, isort)
- Test Suite (Unit, Integration, Colab tests)
- CodeRabbit Review (PR code review)
- Claude Analysis (AI code analysis)
- Build Artifacts (Package distribution)
- Sentry Release (Error tracking setup)
- Report Status (Final reporting)

**Triggers:** Push (main/develop), PR, Daily schedule
**Duration:** ~15-20 minutes

#### 2. **sentry-monitoring.yml**
Error tracking and monitoring integration:
- Monitor workflow failures
- Health checks
- Release management
- Performance metrics
- Auto-triage issues

**Triggers:** Workflow completion, Push (main), Every 6 hours
**Duration:** ~5-10 minutes

#### 3. **claude-analysis.yml**
AI-powered code analysis:
- Architecture review
- Performance analysis
- Security review
- Interactive commands (@claude)

**Triggers:** PR opened/synchronized, Issue comments
**Duration:** ~10-15 minutes

#### 4. **env-config.yml**
Reusable workflow for environment configuration:
- Centralized environment variables
- Configuration validation
- Consistency enforcement

**Usage:** Called by other workflows
**Duration:** ~2 minutes

## 🔐 Required Secrets

Add these secrets to GitHub (Settings → Secrets and variables → Actions):

### Sentry
- `SENTRY_DSN` - Your Sentry DSN endpoint
- `SENTRY_AUTH_TOKEN` - Sentry API token
- `SENTRY_ORG` - Organization slug
- `SENTRY_PROJECT` - Project slug

### AI Services
- `OPENAI_API_KEY` - OpenAI API key (for CodeRabbit)
- `CLAUDE_API_KEY` - Anthropic Claude API key

### GitHub
- `GITHUB_TOKEN` - Automatically provided by GitHub

**Setup Guide:** See [SECRETS_SETUP.md](./SECRETS_SETUP.md)

## 🔧 Quick Start

### 1. Enable GitHub Actions
```bash
# Ensure Actions are enabled in repository settings
Settings → Actions → Allow all actions and reusable workflows
```

### 2. Configure Required Secrets
```bash
# Using GitHub CLI
gh secret set SENTRY_DSN -b "https://xxxxx@xxxxx.ingest.sentry.io/xxxxx"
gh secret set SENTRY_AUTH_TOKEN -b "sntrys_xxxxx"
gh secret set SENTRY_ORG -b "your-org"
gh secret set SENTRY_PROJECT -b "your-project"
gh secret set OPENAI_API_KEY -b "sk-xxxxx"
gh secret set CLAUDE_API_KEY -b "sk-ant-xxxxx"
```

### 3. Verify Configuration
```bash
# List configured secrets
gh secret list

# Verify workflow syntax
gh workflow list
```

### 4. Make a Test Commit
```bash
git add .github/workflows/
git commit -m "Add CI/CD pipeline configuration"
git push origin main
```

## 📊 Workflow Performance

### Expected Runtimes

| Workflow | Stage | Duration |
|----------|-------|----------|
| comprehensive-ci | Quality checks | 2-3 min |
| comprehensive-ci | Test suite | 8-10 min |
| comprehensive-ci | Code reviews | 3-5 min |
| comprehensive-ci | Build | 2-3 min |
| comprehensive-ci | Total | ~15-20 min |
| sentry-monitoring | Monitor/Report | 5-10 min |
| claude-analysis | Analysis | 10-15 min |

### Optimization Tips

1. **Caching** - pip cache reduces install time by 50%
2. **Parallel tests** - Run unit/integration tests in parallel
3. **Fail fast** - Quality checks run before expensive tests
4. **Artifact cleanup** - 30-day retention prevents storage bloat
5. **Concurrency control** - Cancels old runs on new push

## 📈 Monitoring & Dashboards

### GitHub Actions Dashboard
- **Location:** Repository → Actions tab
- **Metrics:** Run duration, success rate, artifact storage
- **Features:** Retry failed jobs, download artifacts

### Sentry Dashboard
- **Location:** https://sentry.io/organizations/{ORG}/
- **Metrics:** Error rate, release stability, performance
- **Features:** Issue tracking, release correlation, alerts

### Codecov Dashboard
- **Location:** https://codecov.io/gh/{ORG}/{REPO}
- **Metrics:** Coverage trend, file coverage, PR coverage delta
- **Features:** Coverage gates, report comments

## 🎯 Branch Protection Rules

Recommend adding to `main` branch:

```
Settings → Branches → Add rule
├─ Require status checks to pass
│  ├─ quality-checks
│  ├─ test-suite
│  └─ build-artifacts
├─ Require code reviews (1 approved)
├─ Require branches to be up to date
└─ Dismiss stale reviews
```

## 📝 Customization

### Add Custom Quality Check

In `.github/workflows/comprehensive-ci.yml`:

```yaml
- name: Custom linter
  run: |
    pip install my-linter
    my-linter check .
```

### Extend Test Matrix

```yaml
matrix:
  test-group: [unit, integration, colab, performance, security]
```

### Configure Notification

Add to any job:

```yaml
- name: Notify on failure
  if: failure()
  run: |
    # Send to Slack, email, etc.
```

## 🚀 Deployment Integration

To add production deployment:

```yaml
deploy-production:
  needs: [comprehensive-ci, sentry-release]
  if: github.ref == 'refs/heads/main' && success()
  runs-on: ubuntu-latest
  environment: production
  steps:
    - uses: actions/checkout@v4
    - name: Deploy
      run: |
        # Your deployment script
```

## 🐛 Troubleshooting

### Issue: "Workflow not triggering"
- **Solution:** Check branch name and trigger conditions
- **Debug:** View webhook deliveries in Settings → Webhooks

### Issue: "Secret not found"
- **Solution:** Run `gh secret list` to verify
- **Debug:** Secrets need 'Actions' scope

### Issue: "Tests timeout"
- **Solution:** Increase `timeout-minutes` or optimize tests
- **Debug:** Check test logs for slow tests

### Issue: "Sentry not receiving events"
- **Solution:** Verify DSN and token permissions
- **Debug:** Check Sentry API status page

## 📚 Additional Resources

- **[CI/CD Integration Guide](./CI_CD_INTEGRATION_GUIDE.md)** - Detailed architecture and implementation
- **[Secrets Setup Guide](./SECRETS_SETUP.md)** - Step-by-step secret configuration
- **[GitHub Actions Docs](https://docs.github.com/en/actions)** - Official documentation
- **[Sentry Docs](https://docs.sentry.io/)** - Error tracking setup
- **[CodeRabbit Docs](https://coderabbit.ai/docs)** - Code review automation
- **[Anthropic Claude API](https://docs.anthropic.com)** - Claude integration

## 🔄 Maintenance

### Weekly
- Review workflow runs for failures
- Check Sentry issues dashboard
- Monitor API usage and costs

### Monthly
- Update dependencies in workflows
- Verify all secrets are still valid
- Check GitHub Actions usage quota

### Quarterly
- Rotate API tokens and secrets
- Review and update workflow files
- Audit access permissions

## 💡 Best Practices

1. **Keep workflows simple** - Use multiple files for different concerns
2. **Cache aggressively** - Reduce install times significantly
3. **Fail fast** - Run quick checks before expensive operations
4. **Monitor costs** - GitHub Actions has usage limits
5. **Document changes** - Update this README when modifying
6. **Test locally** - Use `act` to test workflows locally
7. **Review regularly** - Audit workflows and secrets monthly

## 🤝 Contributing

To contribute to CI/CD improvements:

1. Create a feature branch: `git checkout -b feature/improve-ci`
2. Make changes to workflow files
3. Test locally: `act -j quality-checks`
4. Create a pull request
5. Get review from team lead
6. Merge and monitor results

## 📞 Support

- **Questions?** Create an issue with `ci-cd` label
- **Bug reports?** Include workflow logs and error messages
- **Suggestions?** Open a discussion in the repository

---

**Last Updated:** January 2024
**Version:** 1.0
**Maintainer:** Your Team

For a detailed integration guide, see [CI_CD_INTEGRATION_GUIDE.md](./CI_CD_INTEGRATION_GUIDE.md).
