# GitHub Secrets Configuration Guide

## Overview
This guide helps you configure all required GitHub Secrets for the comprehensive CI/CD pipeline with Sentry, CodeRabbit, Claude, and Codex integrations.

## Required Secrets

### 1. Sentry Configuration

```yaml
SENTRY_DSN:
  Description: Your Sentry Data Source Name (DSN) endpoint
  Value: https://xxxxx@xxxxx.ingest.sentry.io/xxxxx
  How to Get:
    1. Go to https://sentry.io
    2. Create or select a project
    3. Go to Settings → Projects → Your Project → Client Keys (DSN)
    4. Copy the full DSN URL

SENTRY_AUTH_TOKEN:
  Description: Sentry authentication token for API access
  Value: sntrys_xxxxx
  How to Get:
    1. In Sentry, go to Settings → Account → API tokens
    2. Click "Create New Token"
    3. Grant: project:write, releases:write, org:read
    4. Copy the generated token

SENTRY_ORG:
  Description: Your Sentry organization slug
  Value: your-org-name
  How to Get:
    1. In Sentry, go to Settings
    2. Look at the URL: sentry.io/settings/{ORG}/ ← this is your ORG

SENTRY_PROJECT:
  Description: Your Sentry project name/slug
  Value: your-project-name
  How to Get:
    1. In Sentry project, go to Settings
    2. The project name appears in the sidebar
```

### 2. CodeRabbit Configuration

```yaml
OPENAI_API_KEY:
  Description: OpenAI API key for CodeRabbit integration
  Value: sk-xxxxx
  How to Get:
    1. Go to https://platform.openai.com/api-keys
    2. Click "Create new secret key"
    3. Copy and save securely
  Note: CodeRabbit uses this for enhanced code reviews

CODERABBIT_API_KEY:
  Description: CodeRabbit API key (if using advanced features)
  Value: crb_xxxxx
  How to Get:
    1. Go to https://coderabbit.ai
    2. Install GitHub app from dashboard
    3. Access API keys from settings
    4. Copy your organization API key
```

### 3. Claude Configuration

```yaml
CLAUDE_API_KEY:
  Description: Anthropic Claude API key for code analysis
  Value: sk-ant-xxxxx
  How to Get:
    1. Go to https://console.anthropic.com
    2. Sign in or create account
    3. Click "API keys" in the sidebar
    4. Click "Create key"
    5. Copy the generated key

CLAUDE_MODEL:
  Description: Claude model version to use
  Value: claude-3-sonnet-20240229
  Default: claude-3-sonnet-20240229
```

### 4. GitHub Configuration

```yaml
GITHUB_TOKEN:
  Description: GitHub Personal Access Token (usually provided by GitHub)
  Scope: repo, workflow, write:packages, read:packages
  Note: GitHub Actions automatically provides this in most cases
```

### 5. Optional: Deployment Configuration

```yaml
DEPLOY_TOKEN:
  Description: Token for deploying to your hosting provider
  Value: Depends on your deployment target
  Optional: Only needed if deploying from CI/CD

DOCKER_REGISTRY_TOKEN:
  Description: Docker Hub or registry authentication
  Value: Depends on your registry
  Optional: Only if using Docker builds
```

## How to Add Secrets to GitHub

### Method 1: Via GitHub Web Interface

1. Navigate to your repository
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Enter secret name (e.g., `SENTRY_DSN`)
5. Paste the secret value
6. Click **Add secret**

### Method 2: Via GitHub CLI

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com

# Add a secret
gh secret set SENTRY_DSN -b "https://xxxxx@xxxxx.ingest.sentry.io/xxxxx"
gh secret set SENTRY_AUTH_TOKEN -b "sntrys_xxxxx"
gh secret set SENTRY_ORG -b "your-org"
gh secret set SENTRY_PROJECT -b "your-project"
gh secret set OPENAI_API_KEY -b "sk-xxxxx"
gh secret set CLAUDE_API_KEY -b "sk-ant-xxxxx"

# List all secrets
gh secret list

# Delete a secret
gh secret delete SENTRY_DSN
```

### Method 3: Using Environment Variables

Create `.env` file (never commit this):
```bash
export SENTRY_DSN="https://xxxxx@xxxxx.ingest.sentry.io/xxxxx"
export SENTRY_AUTH_TOKEN="sntrys_xxxxx"
export SENTRY_ORG="your-org"
export SENTRY_PROJECT="your-project"
export OPENAI_API_KEY="sk-xxxxx"
export CLAUDE_API_KEY="sk-ant-xxxxx"
```

Load before deployment:
```bash
source .env
gh secret set SENTRY_DSN -b "$SENTRY_DSN"
# ... repeat for others
```

## Verification Checklist

- [ ] SENTRY_DSN configured and tested
- [ ] SENTRY_AUTH_TOKEN has project:write permission
- [ ] SENTRY_ORG matches your organization
- [ ] SENTRY_PROJECT matches your project
- [ ] OPENAI_API_KEY valid and has credits
- [ ] CLAUDE_API_KEY valid and active
- [ ] GITHUB_TOKEN has appropriate scopes
- [ ] All secrets are URL-safe and properly encoded
- [ ] Secrets are rotated regularly (quarterly recommended)

## Troubleshooting

### "Invalid DSN" Error
- Verify DSN format: `https://key@host/project-id`
- Ensure no extra spaces in the value
- Check in Sentry: Settings → Projects → Client Keys

### "Authentication Failed" Error
- Verify auth token is correctly copied
- Check token hasn't expired in Sentry settings
- Verify token has required permissions
- Re-generate token if needed

### "API rate limit exceeded"
- Check OpenAI/Claude API usage in console
- Consider API rate limiting in workflows
- Add delays between API calls

## Security Best Practices

1. **Never commit secrets** - Use GitHub Secrets only
2. **Rotate regularly** - Update tokens every 90 days
3. **Limit scope** - Grant minimum necessary permissions
4. **Use organization secrets** - For shared across repos
5. **Audit usage** - Monitor API calls and spending
6. **Use separate accounts** - Dev vs Production tokens
7. **Enable MFA** - On API key management accounts
8. **Review permissions** - Check token scopes monthly

## Service-Specific Setup

### Sentry Setup Example

```bash
# 1. Create Sentry account
# 2. Create project (e.g., "ml-trading-platform")
# 3. Get DSN
curl -X GET "https://sentry.io/api/0/projects/{org}/{project}/" \
  -H "Authorization: Bearer {SENTRY_AUTH_TOKEN}"

# 4. Test DSN
python -c "
import sentry_sdk
sentry_sdk.init('YOUR_DSN')
sentry_sdk.capture_exception(Exception('Test'))
"
```

### CodeRabbit Setup Example

```bash
# 1. Install GitHub app from https://coderabbit.ai
# 2. Authorize repository
# 3. Get API key from dashboard
# 4. Add OPENAI_API_KEY for enhanced reviews
# 5. Workflow will auto-run on PRs
```

### Claude Setup Example

```bash
# 1. Sign up at https://console.anthropic.com
# 2. Add payment method
# 3. Create API key
# 4. Test in workflow:
python -c "
import anthropic
client = anthropic.Anthropic(api_key='YOUR_KEY')
msg = client.messages.create(model='claude-3-sonnet-20240229', max_tokens=100, messages=[{'role': 'user', 'content': 'test'}])
print(msg.content)
"
```

## Monitoring & Alerts

Set up alerts for:
- Failed Sentry reports
- API rate limit warnings
- Expired API keys
- Unusual authentication failures
- CloudWatch/similar monitoring

## Additional Resources

- [Sentry Documentation](https://docs.sentry.io/)
- [CodeRabbit Documentation](https://coderabbit.ai/docs)
- [Anthropic Claude API](https://docs.anthropic.com)
- [GitHub Secrets Management](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [OpenAI API Keys](https://platform.openai.com/docs/guides/authentication)

---

**Last Updated**: January 2024
**Maintainer**: Your Team
