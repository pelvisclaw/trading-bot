# Regime-Aware Ensemble Trading System

A multi-timeframe, regime-adaptive trading system for SOLUSD on Kraken.

## Architecture

### Core Components

1. **Regime Detector** - Identifies market regime (bull, bear, high_vol, sideways, mean_reversion)
2. **Strategy Ensemble** - Three specialized strategies voting on signals:
   - **Trend Strategy** (EMA crossover)
   - **Volatility Strategy** (Bollinger Bands)
   - **Mean Reversion Strategy** (RSI)
3. **Consensus Voting** - Requires 2+ strategies to agree before trading

### Timeframes Analyzed
- 1h (primary signals)
- 4h (regime confirmation)
- 24h (trend context)

## Strategy Details

### Trend Strategy
- Entry: EMA 9 > EMA 21 > EMA 50 + regime confirms
- Exit: Stop loss 5% or take profit 10%

### Volatility Strategy
- Entry: Price breaks above upper Bollinger Band
- Exit: Stop loss 5% or take profit 10%

### Mean Reversion Strategy
- Entry: RSI < 35 (oversold) or > 65 (overbought)
- Exit: RSI returns to 45-55 range

### Consensus Rules
- LONG: 2+ strategies agree on long signal
- SHORT: 2+ strategies agree on short signal
- NEUTRAL: No consensus or conflicting signals

## Performance (Backtest)

| Metric | Value |
|--------|-------|
| Period | ~30 days (721 candles) |
| Trades | 13 |
| Win Rate | 69.2% |
| Avg PnL | +0.18% per trade |
| Max Drawdown | 3.1% |
| Sharpe Ratio | 2.10 |
| Total Return | +2.3% |

## Usage

### Training (GitHub Actions)
```bash
gh workflow run train-regime-v2.yml
```

### Paper Trading
```bash
gh workflow run paper-trading.yml
```

### Local Backtest
```bash
python scripts/train_regime_v2.py --api-key KEY --api-secret SECRET
```

### Live Trading
```bash
python scripts/paper_trader_regime.py --api-key KEY --api-secret SECRET --live
```

## Deployment

### Required Secrets (GitHub)
- `KRAKEN_API_KEY` - Kraken API key
- `KRAKEN_PRIVATE_KEY` - Kraken API secret
- `PAPER_CAPITAL` - Initial capital (default: 10000)

### Production Checklist
- [ ] 90+ trades in paper mode
- [ ] Sharpe > 1.5 sustained
- [ ] Max drawdown < 15%
- [ ] Win rate > 55%
- [ ] 20+ trading days without manual intervention

## Risk Management

- Position size: 10% of capital per trade
- Stop loss: 5% per trade
- Take profit: 10% per trade
- Maximum concurrent positions: 1

## Files

- `src/strategies/regime_ensemble_v2.py` - Core strategy
- `scripts/train_regime_v2.py` - Backtest script
- `scripts/paper_trader_regime.py` - Paper/live trader
- `.github/workflows/` - CI/CD pipelines

## License

MIT
