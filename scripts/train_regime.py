#!/usr/bin/env python3
"""
Regime-Aware Multi-Timeframe Training Script
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

from strategies.regime_ensemble import RegimeTrainer, RegimeDetector, EnsembleTrader

def main():
    parser = argparse.ArgumentParser(description='Train Regime-Aware Trading System')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--private-key', required=True)
    parser.add_argument('--pair', default='SOLUSD')
    parser.add_argument('--output', default='regime_strategy.json')
    
    args = parser.parse_args()
    
    from strategies.regime_ensemble import KrakenAPI
    api = KrakenAPI(args.api_key, args.private_key)
    trainer = RegimeTrainer(api)
    
    print("=== Regime-Aware Multi-Timeframe Training ===")
    
    # Fetch data
    print(f"\nFetching multi-timeframe data for {args.pair}...")
    ohlc_data = trainer.fetch_multi_timeframe(args.pair)
    
    if not ohlc_data:
        print("ERROR: Failed to fetch data from Kraken")
        sys.exit(1)
    
    for tf, candles in ohlc_data.items():
        print(f"  {tf}min: {len(candles)} candles")
    
    # Walk-forward training
    print("\nRunning walk-forward optimization...")
    wf_results = trainer.walk_forward_train(args.pair)
    
    if 'error' in wf_results:
        print(f"ERROR: {wf_results['error']}")
        sys.exit(1)
    
    # Backtest ensemble
    print("\nBacktesting ensemble...")
    bt_results = trainer.backtest_ensemble(args.pair)
    
    print(f"\n=== Results ===")
    print(f"Trades: {bt_results['trades']}")
    print(f"Win Rate: {bt_results['win_rate']:.1%}")
    print(f"Avg PnL: {bt_results['avg_pnl']:.2%}")
    print(f"Max Drawdown: {bt_results['max_dd']:.1%}")
    print(f"Sharpe: {bt_results['sharpe']:.2f}")
    
    # Save results
    output = {
        'strategy': 'regime_ensemble',
        'config': {
            'timeframes': [60, 240, 1440],
            'strategies': ['trend', 'volatility', 'mean_reversion']
        },
        'walk_forward': wf_results,
        'backtest': bt_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

if __name__ == '__main__':
    main()
