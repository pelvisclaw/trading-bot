#!/usr/bin/env python3
"""
Regime-Aware Multi-Timeframe Training Script v2
- Fetches 1h, 4h, daily data
- Detects market regime
- Trains ensemble with walk-forward validation
- Outputs best performing configuration
"""
import argparse
import json
import sys
from datetime import datetime
sys.path.insert(0, 'src')

from strategies.regime_ensemble_v2 import Backtester, RegimeDetector, EnsembleTrader

def main():
    parser = argparse.ArgumentParser(description='Train Regime-Aware Trading System v2')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--api-secret', required=True)
    parser.add_argument('--pair', default='SOLUSD')
    parser.add_argument('--output', default='regime_strategy_v2.json')
    
    args = parser.parse_args()
    
    from strategies.regime_ensemble_v2 import KrakenAPI
    api = KrakenAPI(args.api_key, args.api_secret)
    backtester = Backtester(api)
    
    print("=== Regime-Aware Multi-Timeframe Training v2 ===")
    
    # Fetch data
    print(f"\nFetching multi-timeframe data for {args.pair}...")
    ohlc_data = backtester.fetch_multi_timeframe(args.pair)
    
    if not ohlc_data:
        print("ERROR: Failed to fetch data from Kraken")
        sys.exit(1)
    
    for tf, candles in ohlc_data.items():
        print(f"  {tf}min: {len(candles)} candles")
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run_backtest(args.pair)
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        sys.exit(1)
    
    print(f"\n=== Results ===")
    print(f"Trades: {results['trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Avg PnL: {results['avg_pnl']:.2%}")
    print(f"Max Drawdown: {results['max_dd']:.1%}")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"Total Return: {results['total_return']:.1%}")
    
    # Regime analysis
    print(f"\n=== Regime Detection Sample ===")
    for tf in [60, 240, 1440]:
        if tf in ohlc_data:
            closes = [float(c[4]) for c in ohlc_data[tf]]
            regime = RegimeDetector.detect(closes)
            print(f"  {tf}min: {regime['regime']} (trend={regime['trend']:.3f}, vol={regime['volatility']:.2f})")
    
    # Save results
    output = {
        'strategy': 'regime_ensemble_v2',
        'config': {
            'timeframes': [60, 240, 1440],
            'strategies': ['trend_ema', 'volatility_bb', 'mean_reversion_rsi']
        },
        'backtest': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

if __name__ == '__main__':
    main()
