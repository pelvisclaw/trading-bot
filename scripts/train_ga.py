#!/usr/bin/env python3
"""
GA Training Script for GitHub Actions
"""
import argparse
import json
import sys
sys.path.insert(0, 'src')

from strategies.genetic_trader_enhanced import GeneticTrader, StrategyGenes

def main():
    parser = argparse.ArgumentParser(description='Train Genetic Algorithm Trading System')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--private-key', required=True)
    parser.add_argument('--pair', default='SOLUSD')
    parser.add_argument('--candles', type=int, default=2000)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--output', default='best_strategy.json')
    
    args = parser.parse_args()
    
    trader = GeneticTrader(args.api_key, args.private_key)
    genes = trader.train(args.generations)
    
    # Get fitness from GA history
    best_fitness = trader.ga.history[-1]['best_fitness'] if trader.ga.history else 0
    
    # Save strategy
    strategy_data = {
        'strategy': 'genetic_enhanced',
        'genes': genes.to_dict(),
        'fitness': best_fitness,
        'pair': args.pair,
        'candles': args.candles
    }
    
    with open(args.output, 'w') as f:
        json.dump(strategy_data, f, indent=2)
    
    print(f"✅ Saved best strategy to {args.output}")
    print(f"📈 Best fitness: {best_fitness:.2f}")
    print(f"🔧 Strategy genes: {json.dumps(genes.to_dict(), indent=2)}")

if __name__ == '__main__':
    main()
