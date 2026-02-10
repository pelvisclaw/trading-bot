# Rotation Trading - GitHub Actions Setup

## Quick Start

### 1. Add API Credentials to GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions:

| Secret Name | Value |
|-------------|-------|
| `KRAKEN_API_KEY` | Your Kraken API key |
| `KRAKEN_API_SECRET` | Your Kraken API secret |
| `MODE` | `dry_run` (or `live` for real trading) |

### 2. Enable the Workflow

The workflow runs automatically every 5 minutes once secrets are set.

### 3. Manual Trigger

You can also trigger manually:
1. Go to Actions → Rotation Trading
2. Click "Run workflow"

## Modes

| Mode | Behavior |
|------|----------|
| `dry_run` | Analyzes but doesn't trade (safe) |
| `live` | Executes real trades |

## Workflow Behavior

```
Every 5 minutes:
1. Checkout code
2. Setup Python
3. Install dependencies
4. Run rotation_trader.py
5. Upload logs to artifacts (7 day retention)
```

## Monitoring

- **Logs**: Check Actions tab for run history
- **Artifacts**: Download rotation_trading.log from each run
- **Status**: Green checkmark = success, Red X = error

## Troubleshooting

### Workflow not running?
- Check repository has Actions enabled
- Verify secrets are set correctly

### Errors in logs?
- Check rotation_trading.log artifact
- Common issues: API rate limits, network timeouts

## Safety

- **Dry run by default**: Set `MODE: live` to enable trading
- **5 minute frequency**: Catches most regime changes
- **7 day log retention**: Enough to debug issues

## Example Log Output

```
==================================================
REGIME-TRADER v3 ROTATION - DRY RUN
==================================================
[2026-02-10 15:58:54] SCORES: XBT: -0.7, ETH: -30.7, SOL: -31.4, LINK: -30.6
[2026-02-10 15:58:56] ROTATING: SOL → XBT
[2026-02-10 15:58:57] DRY RUN: Would swap 0.034 SOL → XBT via SOLXBT
```

## For Manual Runs

```bash
# Dry run (default)
python3 scripts/rotation_trader.py --api-key "..." --api-secret "..."

# Live trading
python3 scripts/rotation_trader.py --api-key "..." --api-secret "..." --live
```
