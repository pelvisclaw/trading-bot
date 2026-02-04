#!/usr/bin/env python3
"""
Backtest Integration for Trading Agent Evolution
Connects evolution framework to actual trading performance
"""
import json
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from strategies.regime_ensemble_v2 import KrakenAPI


class EvolutionBacktester:
    """Backtest trading profiles and update fitness"""
    
    def __init__(self):
        self.api = KrakenAPI()
    
    def backtest_profile(self, profile_name: str, rules: dict) -> dict:
        """Backtest a trading profile and return metrics"""
        ohlc_data = self._fetch_data()
        if not ohlc_data:
            return {'error': 'Failed to fetch data'}
        
        closes = [float(c[4]) for c in ohlc_data[60]]
        trades = self._simulate_trades(closes, rules)
        
        if not trades:
            return {'trades': 0, 'win_rate': 0, 'sharpe': 0, 'max_dd': 0}
        
        wins = sum(1 for t in trades if t['pnl'] > 0)
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        
        if len(trades) > 1:
            std_pnl = (sum((t['pnl'] - avg_pnl)**2 for t in trades) / len(trades)) ** 0.5
            sharpe = (avg_pnl / (std_pnl + 0.001)) * 16 if std_pnl > 0 else 0
        else:
            sharpe = 0
        
        equity = [10000]
        for t in trades:
            equity.append(equity[-1] * (1 + t['pnl']))
        max_dd = self._calc_max_dd(equity)
        
        return {
            'trades': len(trades),
            'win_rate': wins / len(trades),
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': (equity[-1] / equity[0] - 1)
        }
    
    def _fetch_data(self) -> dict:
        data = {}
        for interval in [60, 240, 1440]:
            result = self.api.get_ohlc('SOLUSD', interval, 500)
            if 'result' in result and len(result['result']) > 1:
                data[interval] = list(result['result'].values())[0]
        return data
    
    def _simulate_trades(self, closes: list, rules: dict) -> list:
        trades = []
        position = None
        
        ema_fast = rules.get('parameters', {}).get('ema_fast', 9)
        ema_slow = rules.get('parameters', {}).get('ema_slow', 21)
        rsi_low = rules.get('parameters', {}).get('rsi_low', 30)
        
        for i in range(50, len(closes)):
            price = closes[i]
            ema_f = self._ema(closes[:i+1], ema_fast)
            ema_s = self._ema(closes[:i+1], ema_slow)
            
            signal = 'neutral'
            if ema_f > ema_s:
                signal = 'long'
            elif ema_f < ema_s:
                signal = 'short'
            
            if signal == 'long' and position is None:
                position = {'entry': price, 'direction': 'long'}
            elif signal == 'short' and position is None:
                position = {'entry': price, 'direction': 'short'}
            elif signal == 'neutral' and position:
                pnl = (price - position['entry']) / position['entry']
                if position['direction'] == 'short':
                    pnl = -pnl
                trades.append({'pnl': pnl})
                position = None
            
            if position:
                pnl = (price - position['entry']) / position['entry']
                if position['direction'] == 'short':
                    pnl = -pnl
                
                sl = rules.get('stop_loss', 0.05)
                tp = rules.get('take_profit', 0.10)
                
                if pnl <= -sl or pnl >= tp:
                    trades.append({'pnl': pnl})
                    position = None
        
        return trades
    
    def _ema(self, closes: list, period: int) -> float:
        if len(closes) < period:
            return sum(closes) / len(closes)
        alpha = 2 / (period + 1)
        ema = closes[0]
        for close in closes[1:]:
            ema = alpha * close + (1 - alpha) * ema
        return ema
    
    def _calc_max_dd(self, equity: list) -> float:
        max_eq = equity[0]
        max_dd = 0
        for eq in equity:
            dd = (max_eq - eq) / max_eq
            max_dd = max(max_dd, dd)
            max_eq = max(max_eq, eq)
        return max_dd


PROFILE_RULES = {
    "momentum_v1": {"entry": "ema_crossover", "parameters": {"ema_fast": 9, "ema_slow": 21}, "stop_loss": 0.05, "take_profit": 0.10},
    "mean_reversion_v1": {"entry": "rsi_oversold", "parameters": {"rsi_low": 30, "rsi_high": 70}, "stop_loss": 0.05, "take_profit": 0.05},
    "volatility_v1": {"entry": "bb_breakout", "parameters": {"bb_period": 20, "bb_std": 2.0}, "stop_loss": 0.05, "take_profit": 0.15},
    "breakout_v1": {"entry": "resistance_break", "parameters": {"lookback": 20}, "stop_loss": 0.03, "take_profit": 0.10},
    "regime_aware_v1": {"entry": "ema_crossover", "parameters": {"ema_fast": 9, "ema_slow": 21}, "stop_loss": 0.05, "take_profit": 0.10},
    "trend_follower_v1": {"entry": "ema_crossover", "parameters": {"ema_fast": 50, "ema_slow": 200}, "stop_loss": 0.05, "take_profit": 0.20},
}


def main():
    print("=== Profile Backtest for Evolution ===\n")
    
    backtester = EvolutionBacktester()
    results = {}
    
    for profile_name, rules in PROFILE_RULES.items():
        print(f"Backtesting {profile_name}...")
        metrics = backtester.backtest_profile(profile_name, rules)
        results[profile_name] = metrics
        
        if 'error' in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            print(f"  Trades: {metrics['trades']}, Win: {metrics['win_rate']:.1%}, Sharpe: {metrics['sharpe']:.2f}, DD: {metrics['max_dd']:.1%}")
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to backtest_results.json")


if __name__ == "__main__":
    main()
