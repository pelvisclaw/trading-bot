#!/usr/bin/env python3
"""
Paper Trading Script - Validates strategy on historical data
"""
import argparse
import json
import sys
sys.path.insert(0, 'src')

from strategies.genetic_trader_enhanced import GeneticTrader, StrategyGenes

def main():
    parser = argparse.ArgumentParser(description='Paper Trading Validation')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--private-key', required=True)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--paper', action='store_true', help='Run in paper mode')
    parser.add_argument('--output', default='paper_results.json')
    
    args = parser.parse_args()
    
    # Load strategy
    with open(args.strategy, 'r') as f:
        data = json.load(f)
        genes = StrategyGenes.from_dict(data['genes'])
    
    print(f"Loaded strategy with {genes.complexity()} complexity")
    print(f"Base position size: {genes.base_position_size}")
    print(f"Stop loss: {genes.base_stop_loss}, Take profit: {genes.base_take_profit}")
    
    # For paper trading, we just validate on recent data
    trader = GeneticTrader(args.api_key, args.private_key)
    
    # Fetch recent data
    ohlc_data = trader.fetch_ohlc_multi()
    
    if not ohlc_data:
        print("Error: Failed to fetch OHLC data")
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
