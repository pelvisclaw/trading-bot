#!/usr/bin/env python3
"""
Paper Trading Script - Validates strategy on historical data
"""
import argparse
import json
import random
import sys
sys.path.insert(0, 'src')

from strategies.genetic_trader_enhanced import GeneticTrader, StrategyGenes, KrakenAPI

def generate_synthetic_ohlc(count: int = 500) -> list:
    """Generate synthetic OHLC data for testing when API fails"""
    base_price = 100.0
    ohlc = []
    for i in range(count):
        timestamp = 1700000000 + (i * 3600)  # Hourly
        change = random.gauss(0, 0.02)  # 2% volatility
        close = base_price * (1 + change)
        high = close * (1 + random.uniform(0, 0.01))
        low = close * (1 - random.uniform(0, 0.01))
        volume = random.uniform(1000, 10000)
        ohlc.append([str(timestamp), str(high), str(low), str(close), str(close), str(volume), "0"])
        base_price = close
    return ohlc

def main():
    parser = argparse.ArgumentParser(description='Paper Trading Validation')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--private-key', required=True)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--candles', type=int, default=500, help='Number of candles to fetch')
    parser.add_argument('--output', default='paper_results.json')
    
    args = parser.parse_args()
    
    # Load strategy
    with open(args.strategy, 'r') as f:
        data = json.load(f)
        genes = StrategyGenes.from_dict(data['genes'])
    
    print(f"Loaded strategy with {genes.complexity()} complexity")
    print(f"Base position size: {genes.base_position_size}")
    print(f"Stop loss: {genes.base_stop_loss}, Take profit: {genes.base_take_profit}")
    
    # Fetch candles - with fallback to synthetic data
    api = KrakenAPI()
    ohlc_data = {}
    use_synthetic = False
    
    try:
        result = api.get_ohlc('SOLUSD', 60, args.candles)
        if 'result' in result and len(result['result']) > 1:
            data = list(result['result'].values())[0]
            ohlc_data['SOLUSD'] = data
            print(f"Fetched {len(data)} candles: {data[0][4]} -> {data[-1][4]}")
        else:
            use_synthetic = True
    except Exception as e:
        print(f"API error: {e}")
        use_synthetic = True
    
    if use_synthetic:
        ohlc_data['SOLUSD'] = generate_synthetic_ohlc(args.candles)
        print(f"Using synthetic data: {args.candles} candles")
    
    if not ohlc_data:
        print("Error: No data available")
        sys.exit(1)
    
    # Evaluate strategy
    from strategies.genetic_trader_enhanced import StrategyEvaluator
    evaluator = StrategyEvaluator(genes)
    
    results = {}
    for pair, data in ohlc_data.items():
        result = evaluator.evaluate(data, pair)
        results[pair] = {
            'fitness': result['fitness'],
            'trades': result['trades'],
            'pnl': result.get('pnl', 0),
            'sharpe': result.get('sharpe', 0),
            'win_rate': result.get('win_rate', 0),
            'max_drawdown': result.get('max_drawdown', 0)
        }
        print(f"{pair}: fitness={result['fitness']:.2f}, trades={result['trades']}, pnl={result.get('pnl', 0):.2f}%")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPaper trading results saved to {args.output}")

if __name__ == '__main__':
    main()
