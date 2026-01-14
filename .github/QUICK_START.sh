#!/bin/bash
# Quick Start Script for CI/CD Setup
# Run this to get your CI/CD pipeline up and running in minutes

set -e

echo "================================================"
echo "  Sequence Project CI/CD Quick Start Setup"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check git
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed"
    exit 1
fi
echo "✅ Git found"

# Check gh CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI is not installed"
    echo "   Install from: https://cli.github.com"
    exit 1
fi
echo "✅ GitHub CLI found"

# Check gh auth
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub"
    echo "   Run: gh auth login"
    exit 1
fi
echo "✅ GitHub authentication OK"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi
echo "✅ Python 3 found"

echo ""
echo -e "${BLUE}Verifying workflow files...${NC}"

# Check if workflow files exist
WORKFLOWS=(
    ".github/workflows/comprehensive-ci.yml"
    ".github/workflows/sentry-monitoring.yml"
    ".github/workflows/claude-analysis.yml"
    ".github/workflows/env-config.yml"
)

for workflow in "${WORKFLOWS[@]}"; do
    if [ -f "$workflow" ]; then
        echo "✅ $workflow"
    else
        echo "❌ Missing $workflow"
        exit 1
    fi
done

echo ""
echo -e "${YELLOW}Step 1: Health Check${NC}"
echo "Running health check..."
python3 .github/health_check.py --json > /tmp/health_check.json 2>&1

if python3 -c "import json; data = json.load(open('/tmp/health_check.json')); exit(0 if data.get('failed', 0) == 0 else 1)" 2>/dev/null; then
    echo "✅ Health check passed"
else
    echo "⚠️  Health check completed with warnings (this is OK)"
fi

echo ""
echo -e "${YELLOW}Step 2: Secret Configuration${NC}"
echo ""
echo "The following secrets are required:"
echo "  1. SENTRY_DSN"
echo "  2. SENTRY_AUTH_TOKEN"
echo "  3. SENTRY_ORG"
echo "  4. SENTRY_PROJECT"
echo "  5. OPENAI_API_KEY"
echo "  6. CLAUDE_API_KEY"
echo ""
echo "Quick setup options:"
echo ""
echo "Option A: Interactive Setup (Recommended)"
echo "  Run: ./scripts/setup-secrets.sh"
echo ""
echo "Option B: Manual Setup via GitHub CLI"
echo "  Commands:"
echo "    gh secret set SENTRY_DSN -b '...'"
echo "    gh secret set SENTRY_AUTH_TOKEN -b '...'"
echo "    gh secret set SENTRY_ORG -b '...'"
echo "    gh secret set SENTRY_PROJECT -b '...'"
echo "    gh secret set OPENAI_API_KEY -b '...'"
echo "    gh secret set CLAUDE_API_KEY -b '...'"
echo ""
echo "Option C: Manual Setup via GitHub Web UI"
echo "  Go to: Settings → Secrets and variables → Actions"
echo "  Add each secret individually"
echo ""

read -p "Have you configured all required secrets? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please configure secrets first using one of the options above."
    echo "Then run this script again."
    exit 0
fi

echo ""
echo -e "${YELLOW}Step 3: Verify Secrets${NC}"
echo "Checking configured secrets..."

SECRETS=$(gh secret list)

REQUIRED_SECRETS=(
    "SENTRY_DSN"
    "SENTRY_AUTH_TOKEN"
    "SENTRY_ORG"
    "SENTRY_PROJECT"
    "OPENAI_API_KEY"
    "CLAUDE_API_KEY"
)

MISSING_SECRETS=()
for secret in "${REQUIRED_SECRETS[@]}"; do
    if echo "$SECRETS" | grep -q "^$secret"; then
        echo "✅ $secret"
    else
        echo "❌ $secret"
        MISSING_SECRETS+=("$secret")
    fi
done

if [ ${#MISSING_SECRETS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing secrets: ${MISSING_SECRETS[@]}"
    echo "Please add them using GitHub CLI or Web UI"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 4: Create Test Commit${NC}"
echo ""
echo "Creating a test commit to trigger the CI/CD pipeline..."
echo ""

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo "ℹ️  No changes to commit. Creating a test file..."
    echo "# CI/CD Setup Test
This file was created to test the CI/CD pipeline." > .github/.ci-test
    git add .github/.ci-test
    git commit -m "test: Trigger CI/CD pipeline (setup verification)"
else
    echo "ℹ️  Found uncommitted changes"
    git status
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -am "test: Trigger CI/CD pipeline (setup verification)"
    else
        echo "Skipping commit. You can commit manually later."
        exit 0
    fi
fi

echo ""
echo -e "${YELLOW}Step 5: Push Changes${NC}"
echo "Pushing to GitHub..."
git push origin HEAD

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  ✅ CI/CD Pipeline Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "📊 Next Steps:"
echo ""
echo "1. Monitor the workflow:"
echo "   → Visit: https://github.com/$(gh repo view --json nameWithOwner --jq .nameWithOwner)/actions"
echo "   → Or run: gh run list"
echo ""
echo "2. View the workflow run:"
echo "   → Run: gh run list --limit 1"
echo "   → Then: gh run view <run-id>"
echo ""
echo "3. Add status badges to README.md:"
echo "   [![CI/CD](https://github.com/...)/badge.svg)](https://github.com/...)"
echo ""
echo "4. Configure branch protection (optional):"
echo "   → Settings → Branches → Add rule for 'main'"
echo "   → Require status checks: quality-checks, test-suite, build-artifacts"
echo ""
echo "📚 Documentation:"
echo "   → SECRETS_SETUP.md - Detailed secret configuration"
echo "   → CI_CD_INTEGRATION_GUIDE.md - Complete architecture guide"
echo "   → LOCAL_TESTING_GUIDE.md - Local workflow testing"
echo "   → IMPLEMENTATION_SUMMARY.md - Full implementation details"
echo ""
echo "❓ Need Help?"
echo "   → Check .github/health_check.py for diagnostics"
echo "   → Review workflow logs in GitHub Actions"
echo "   → See LOCAL_TESTING_GUIDE.md for local testing"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
