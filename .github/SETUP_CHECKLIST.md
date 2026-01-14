# CI/CD Setup Checklist - Print This! 📋

## Phase 1: Prerequisites (5 minutes)

```
□ Have GitHub account access
□ Have Sentry account created
□ Have OpenAI account with API key
□ Have Anthropic Claude account with API key
□ Have git command installed
□ Have GitHub CLI (gh) installed
□ Authenticated with GitHub: gh auth status
```

## Phase 2: Documentation (15 minutes)

```
□ Read: .github/INDEX.md (navigation guide)
□ Read: .github/IMPLEMENTATION_SUMMARY.md (overview)
□ Understand: 4 workflows being deployed
□ Understand: 6 secrets that need configuration
□ Know where: .github/SECRETS_SETUP.md (secret guide)
```

## Phase 3: Secret Gathering (15 minutes)

### Sentry Secrets
```
□ Go to https://sentry.io
□ Create/select project
□ Get DSN from: Settings → Projects → Client Keys
   Value: _______________________________________________

□ Go to: Settings → Account → API Tokens
□ Create new token with scope: project:write, releases:write
   Value: _______________________________________________

□ Note your organization slug
   Value: _______________________________________________

□ Note your project name/slug
   Value: _______________________________________________
```

### AI Service Secrets
```
□ Go to https://platform.openai.com/api-keys
□ Create new API key (keep this secret!)
   Value: _______________________________________________

□ Go to https://console.anthropic.com/api-keys
□ Create new API key
   Value: _______________________________________________
```

## Phase 4: Secret Configuration (10 minutes)

### Option A: Automated (RECOMMENDED)
```
□ Run: bash .github/QUICK_START.sh
□ Follow all prompts
□ Done!
```

### Option B: Manual via CLI
```
□ gh secret set SENTRY_DSN -b "https://xxxxx@xxxxx.ingest.sentry.io/xxxxx"
□ gh secret set SENTRY_AUTH_TOKEN -b "sntrys_xxxxx"
□ gh secret set SENTRY_ORG -b "your-org"
□ gh secret set SENTRY_PROJECT -b "your-project"
□ gh secret set OPENAI_API_KEY -b "sk-xxxxx"
□ gh secret set CLAUDE_API_KEY -b "sk-ant-xxxxx"
```

### Option C: Manual via GitHub Web UI
```
□ Go to: Settings → Secrets and variables → Actions
□ Click: "New repository secret"
□ Add SENTRY_DSN with value: _______________
□ Add SENTRY_AUTH_TOKEN with value: _______________
□ Add SENTRY_ORG with value: _______________
□ Add SENTRY_PROJECT with value: _______________
□ Add OPENAI_API_KEY with value: _______________
□ Add CLAUDE_API_KEY with value: _______________
```

## Phase 5: Verification (5 minutes)

```
□ Run: python3 .github/health_check.py
□ All checks pass with ✅
□ Run: gh secret list
□ All 6 secrets appear in list
□ Check: .github/workflows/ directory exists
□ Check: .github/workflows/comprehensive-ci.yml exists
□ Check: .github/workflows/sentry-monitoring.yml exists
□ Check: .github/workflows/claude-analysis.yml exists
```

## Phase 6: Deployment (10 minutes)

```
□ Commit changes: git add .github/
□ Commit with message: git commit -m "Add comprehensive CI/CD pipeline"
□ Push to main: git push origin main
□ Monitor at: https://github.com/YOUR_ORG/YOUR_REPO/actions
□ Wait for first workflow run to complete (~20 min)
□ Check ✅ for: quality-checks
□ Check ✅ for: test-suite
□ Check ✅ for: build-artifacts
```

## Phase 7: First Workflow Verification (20 minutes)

Wait for workflow to complete, then check:

```
□ GitHub Actions tab shows success
□ Coverage report generated
□ Sentry received the release
   Visit: https://sentry.io/organizations/YOUR_ORG/releases/
□ No errors in workflow logs
```

## Phase 8: Repository Configuration (10 minutes)

### Optional: Branch Protection
```
□ Go to: Settings → Branches → Add rule
□ Branch name: main
□ Check: "Require status checks to pass before merging"
□ Select: quality-checks
□ Select: test-suite
□ Select: build-artifacts
□ Check: "Require branches to be up to date before merging"
```

### Optional: Status Badges
```
□ Add to README.md from: .github/workflows/README.md
□ Add: Comprehensive CI/CD badge
□ Add: Sentry badge (if needed)
□ Add: Coverage badge (if needed)
```

## Phase 9: Team Communication (Today)

```
□ Share: .github/INDEX.md with team
□ Explain: New review process (CodeRabbit, Claude)
□ Document: How to use @claude commands in PRs
□ Example: "@claude analyze" (for analysis)
□ Example: "@claude suggest" (for suggestions)
□ Example: "@claude explain" (for explanations)
```

## Phase 10: Local Development Setup (Optional)

For developers who want to test locally:

```
□ Install act: brew install act
□ Create .secrets file in repo root
□ Add secrets to .secrets file (from Phase 3)
□ Run: act -j quality-checks --secret-file .secrets
□ Test runs locally before pushing
```

## Ongoing Maintenance

### Weekly
```
□ Check GitHub Actions dashboard
□ Review Sentry issues
□ Monitor API usage
```

### Monthly
```
□ Update dependencies: pip list --outdated
□ Verify all secrets still valid
□ Check for workflow errors
```

### Quarterly
```
□ Rotate API tokens and secrets
□ Review and audit workflow files
□ Check GitHub Actions usage
□ Update documentation as needed
```

---

## 📞 Support & Troubleshooting

### If Something Goes Wrong

1. **Run Health Check**
   ```
   python3 .github/health_check.py
   ```

2. **Check Logs**
   - Go to: GitHub Actions tab
   - Click on failed workflow run
   - Review error messages

3. **Verify Secrets**
   ```
   gh secret list
   ```

4. **Review Documentation**
   - SECRETS_SETUP.md - Secret configuration issues
   - CI_CD_INTEGRATION_GUIDE.md - Architecture issues
   - LOCAL_TESTING_GUIDE.md - Local testing issues

5. **Check Services**
   - Sentry: https://sentry.io
   - OpenAI: https://platform.openai.com
   - Anthropic: https://console.anthropic.com

### Common Issues

| Issue | Solution |
|-------|----------|
| Secrets not found | Run `gh secret list`, re-add if needed |
| Workflow not running | Check branch name, push to correct branch |
| Tests failing | Review test logs in Actions tab |
| Sentry not receiving | Check DSN and token validity |
| CodeRabbit not working | Check OpenAI API key validity |

---

## ✅ Final Checklist

Before declaring success:

```
□ All 6 secrets configured
□ Health check passes
□ First workflow run successful
□ Sentry received release
□ Team notified
□ Documentation reviewed
□ Local testing setup (optional)
□ Status badges added (optional)
□ Branch protection configured (optional)
```

---

## 🎉 You're Done!

**Congratulations!** Your CI/CD pipeline is now:
- ✅ Deployed
- ✅ Configured
- ✅ Running
- ✅ Monitoring
- ✅ Reviewing code

**Next time:**
- Make a PR to see CodeRabbit & Claude analysis
- Comment with @claude commands
- Check Sentry for errors
- Review coverage reports

**Questions?** See `.github/INDEX.md` for documentation index.

---

**Last Updated**: January 2024
**Print This**: Yes! Keep it nearby during setup.
**Estimated Setup Time**: 90 minutes total
