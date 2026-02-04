# Live Trading System

⚠️ **USE WITH EXTREME CAUTION** ⚠️

This system executes REAL TRADES with REAL MONEY.

## Safety Features

1. **Disabled by default** - Must be explicitly enabled
2. **Emergency stop** - One command to halt all trading
3. **Position limits** - Maximum 10% of capital per trade
4. **Daily trade limit** - Maximum 10 trades per day
5. **Cooldown** - 5 minutes between trades
6. **Dry run mode** - Test without real money

## Commands

### Test (Dry Run)
```bash
python scripts/live_trader.py --api-key KEY --api-secret SECRET --analyze
```

### Enable Live Trading
```bash
python scripts/live_trader.py --api-key KEY --api-secret SECRET --enable
```

### Check Status
```bash
python scripts/live_trader.py --api-key KEY --api-secret SECRET --status
```

### Emergency Stop
```bash
python scripts/live_trader.py --api-key KEY --api-secret SECRET --emergency-stop
```

### Disable Trading
```bash
python scripts/live_trader.py --api-key KEY --api-secret SECRET --disable
```

## Workflow

1. **Always dry run first** - Verify signals match expectations
2. **Enable with --enable** - Creates trading_config.json
3. **Monitor with --status** - Check balance and positions
4. **Use --emergency-stop** if anything goes wrong

## Files

- `scripts/live_trader.py` - Main trading script
- `trading_config.json` - Configuration (auto-created)
- `live_trades.log` - Trade history

## Warning

- Start with small position sizes
- Monitor first few trades closely
- Keep emergency stop ready
- Never trade more than you can afford to lose
