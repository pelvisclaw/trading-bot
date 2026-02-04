#!/usr/bin/env python3
"""
Walk-Forward Validation & Paper Trading for Kraken Genetic Trader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import argparse

from scripts.genetic_trader_enhanced import StrategyEvaluator, StrategyGenes, GeneticTrader


class WalkForwardValidator:
    """Validate strategy on out-of-sample data"""
    
    def __init__(self, strategy_file: str = "best_strategy_enhanced.json"):
        self.strategy_file = strategy_file
    
    def load_strategy(self):
        """Load saved strategy"""
        if Path(self.strategy_file).exists():
            with open(self.strategy_file, 'r') as f:
                data = json.load(f)
                return data.get('genes', {})
        return None
    
    def split_data(self, ohlc_data: List, train_pct: float = 0.7):
        """Split data into train/test sets"""
        split = int(len(ohlc_data) * train_pct)
        return ohlc_data[:split], ohlc_data[split:]
    
    def validate(self, ohlc_data: Dict, genes: Dict) -> Dict:
        """Run walk-forward validation"""
        strategy = StrategyGenes.from_dict(genes)
        evaluator = StrategyEvaluator(strategy)
        
        results = {}
        
        for pair, data in ohlc_data.items():
            # Use 80/20 split and ensure enough data
            train_size = int(len(data) * 0.8)
            if train_size < 60:
                train_size = len(data) - 30  # Leave 30 for test
            
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Ensure minimum bars
            if len(train_data) < 50:
                train_data = data[:50]
                test_data = data[50:]
            
            # Evaluate on train
            train_result = evaluator.evaluate(train_data, pair)
            
            # Evaluate on test
            test_result = evaluator.evaluate(test_data, pair)
            
            # Calculate overfit ratio
            overfit_ratio = test_result['fitness'] / (train_result['fitness'] + 0.001)
            
            results[pair] = {
                'train_fitness': train_result['fitness'],
                'test_fitness': test_result['fitness'],
                'train_pnl': train_result['pnl'],
                'test_pnl': test_result['pnl'],
                'train_trades': train_result['trades'],
                'test_trades': test_result['trades'],
                'overfit_ratio': overfit_ratio,
                'valid': 0.7 < overfit_ratio < 1.3  # Allow ±30% difference
            }
        
        return results


class PaperTrader:
    """Paper trading simulator - evaluates strategy on historical data"""
    
    def __init__(self, initial_capital: float = 200.0, position_pct: float = 0.5):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = []
        self.trades = []
        self.log_file = Path("/home/nueki/.openclaw/workspace/paper_trading.log")
        self.position_pct = position_pct
    
    def evaluate_strategy(self, ohlc_data: List, strategy: StrategyGenes):
        """Run strategy on historical data and track trades"""
        evaluator = StrategyEvaluator(strategy)
        
        for pair, data in ohlc_data.items():
            result = evaluator.evaluate(data, pair)
            
            # Log trades from evaluation
            trades = result.get('trades', 0)
            pnl = result.get('pnl', 0)
            
            # Simulate paper trading P&L
            self.trades.append({
                'pair': pair,
                'count': trades,
                'pnl': pnl,
                'sharpe': result.get('sharpe', 0)
            })
            
            # Approximate capital change
            self.capital *= (1 + pnl / 100)
    
    def run_backtest(self, strategy: StrategyGenes, ohlc_data: Dict):
        """Run full backtest and show results"""
        self.trades = []
        self.capital = self.initial_capital
        
        for pair, data in ohlc_data.items():
            evaluator = StrategyEvaluator(strategy)
            result = evaluator.evaluate(data, pair)
            
            trades = result.get('trades', 0)
            pnl = result.get('pnl', 0)
            sharpe = result.get('sharpe', 0)
            
            self.trades.append({
                'pair': pair,
                'trades': trades,
                'pnl': pnl,
                'sharpe': sharpe
            })
            
            self.capital = self.initial_capital * (1 + pnl / 100)
        
        return self.status()
    
    def status(self) -> Dict:
        total_pnl = sum(t['pnl'] for t in self.trades) if self.trades else 0
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0) if self.trades else 0
        losses = sum(1 for t in self.trades if t.get('pnl', 0) < 0) if self.trades else 0
        
        return {
            'capital': self.capital,
            'positions': len(self.positions),
            'trades': sum(t.get('trades', 0) for t in self.trades),
            'pnl_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Validation & Paper Trading')
    parser.add_argument('--validate', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--paper', action='store_true', help='Start paper trading')
    parser.add_argument('--status', action='store_true', help='Show paper trading status')
    parser.add_argument('--capital', type=float, default=200.0, help='Initial capital')
    
    args = parser.parse_args()
    
    if args.validate:
        print("Running walk-forward validation...")
        trader = GeneticTrader()
        ohlc_data = trader.fetch_ohlc_multi(300)  # More data for validation
        
        validator = WalkForwardValidator()
        genes = validator.load_strategy()
        
        if not genes:
            print("No strategy found. Train first.")
            return
        
        results = validator.validate(ohlc_data, genes)
        
        print("\n=== Walk-Forward Validation Results ===")
        for pair, res in results.items():
            status = "VALID" if res['valid'] else "OVERFIT"
            print(f"\n{pair}:")
            print(f"  Train fitness: {res['train_fitness']:.2f}")
            print(f"  Test fitness:  {res['test_fitness']:.2f}")
            print(f"  Overfit ratio: {res['overfit_ratio']:.3f} [{status}]")
            print(f"  Train trades: {res['train_trades']}, Test trades: {res['test_trades']}")
    
    elif args.paper:
        print(f"Starting paper trading with ${args.capital}...")
        trader = PaperTrader(args.capital)
        
        # Load strategy and evaluate on recent OHLC data
        trader_backend = GeneticTrader()
        ohlc_data = trader_backend.fetch_ohlc_multi(200)  # More candles
        
        validator = WalkForwardValidator()
        genes = validator.load_strategy()
        
        if not genes:
            print("No strategy found. Train first.")
            return
        
        strategy = StrategyGenes.from_dict(genes)
        status = trader.run_backtest(strategy, ohlc_data)
        
        print(f"\n=== Paper Trading Results ===")
        print(f"Capital: ${status['capital']:.2f}")
        print(f"Positions: {status['positions']}")
        print(f"Trades: {status['trades']}")
        print(f"PnL: {status['pnl_pct']:.2f}%")
        print(f"Win Rate: {status['win_rate']:.1f}%")
        
        # Individual pair results
        for t in trader.trades:
            print(f"  {t['pair']}: {t['trades']} trades, PnL: {t['pnl']:.2f}%, Sharpe: {t['sharpe']:.2f}")
    
    elif args.status:
        trader = PaperTrader()
        status = trader.status()
        print(f"\n=== Paper Trading Status ===")
        print(f"Capital: ${status['capital']:.2f}")
        print(f"Positions: {status['positions']}")
        print(f"Trades: {status['trades']}")
        print(f"PnL: {status['pnl_pct']:.2f}%")
    
    else:
        print("Options:")
        print("  --validate   Run walk-forward validation")
        print("  --paper      Start paper trading")
        print("  --status     Show paper trading status")
        print("  --capital 200  Set initial capital")


if __name__ == "__main__":
    main()
