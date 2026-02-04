#!/usr/bin/env python3
"""
Enhanced Regime-Aware Trading System v3
Includes:
- Original 3 strategies (Trend EMA, Bollinger, RSI)
- NEW: 3 Advanced Volatility Strategies
- Multi-timeframe regime detection
- Consensus voting
"""
import json
import time
import hashlib
import hmac
import base64
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev


# ============ REGIME DETECTION ============

class RegimeDetector:
    """Detect market regime from price action"""
    
    @staticmethod
    def detect(closes: List[float]) -> Dict:
        """Return regime analysis"""
        if len(closes) < 50:
            return {'regime': 'unknown', 'confidence': 0, 'trend': 0, 'volatility': 0}
        
        # Trend detection
        sma_fast = mean(closes[-20:]) if len(closes) >= 20 else mean(closes)
        sma_slow = mean(closes[-50:]) if len(closes) >= 50 else mean(closes)
        trend = (sma_fast - sma_slow) / sma_slow if sma_slow != 0 else 0
        
        # Volatility
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        vol = stdev(returns) * 100 if len(returns) > 1 else 0
        
        # Mean reversion
        ma = mean(closes[-20:]) if len(closes) >= 20 else mean(closes)
        std = stdev(closes[-20:]) if len(closes) >= 20 else 1
        mr_signal = abs(closes[-1] - ma) / std if std != 0 else 0
        
        # Classify
        if vol > 4:
            regime = 'high_vol'
        elif trend > 0.015:
            regime = 'bull'
        elif trend < -0.015:
            regime = 'bear'
        elif mr_signal > 1.5:
            regime = 'mean_reversion'
        else:
            regime = 'sideways'
        
        return {
            'regime': regime,
            'confidence': abs(trend) * 10 + vol + mr_signal,
            'trend': trend,
            'volatility': vol,
            'mr_signal': mr_signal
        }


# ============ INDICATORS ============

class Indicators:
    @staticmethod
    def sma(closes: List[float], period: int) -> float:
        return mean(closes[-period:]) if len(closes) >= period else mean(closes)
    
    @staticmethod
    def ema(closes: List[float], period: int) -> float:
        if len(closes) < period:
            return mean(closes)
        alpha = 2 / (period + 1)
        ema_val = closes[0]
        for close in closes[1:]:
            ema_val = alpha * close + (1 - alpha) * ema_val
        return ema_val
    
    @staticmethod
    def rsi(closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50
        gains, losses = [], []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = mean(gains[-period:]) if gains else 0.01
        avg_loss = mean(losses[-period:]) if losses else 0.01
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(closes: List[float], period: int = 20) -> Tuple[float, float, float]:
        ma = mean(closes[-period:]) if len(closes) >= period else mean(closes)
        std = stdev(closes[-period:]) if len(closes) >= period else 1
        return ma + 2 * std, ma, ma - 2 * std
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.01 * mean(closes)
        trs = []
        for i in range(1, min(len(highs), len(lows), len(closes))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return mean(trs[-period:]) if trs else 0.01 * mean(closes)


# ============ ORIGINAL 3 STRATEGIES ============

@dataclass
class StrategyConfig:
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.10


class TrendStrategy:
    """Trend-following using EMA"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        ema9 = Indicators.ema(closes, 9)
        ema21 = Indicators.ema(closes, 21)
        ema50 = Indicators.ema(closes, 50)
        trend = regime['trend']
        regime_type = regime['regime']
        
        if ema9 > ema21 > ema50 and trend > 0.005 and regime_type in ['bull', 'sideways']:
            return 'long'
        if ema9 < ema21 < ema50 and trend < -0.005 and regime_type in ['bear', 'sideways']:
            return 'short'
        return 'neutral'


class VolatilityStrategyBB:
    """Bollinger Bands breakout"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        upper, _, lower = Indicators.bollinger_bands(closes)
        price = closes[-1]
        regime_type = regime['regime']
        
        if price > upper and regime_type in ['high_vol', 'bull', 'sideways']:
            return 'long'
        if price < lower and regime_type in ['high_vol', 'bear', 'sideways']:
            return 'short'
        return 'neutral'


class MeanReversionRSI:
    """RSI mean reversion"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        rsi = Indicators.rsi(closes, 14)
        regime_type = regime['regime']
        
        if rsi < 35 and regime_type in ['sideways', 'mean_reversion', 'bull']:
            return 'long'
        if rsi > 65 and regime_type in ['sideways', 'mean_reversion', 'bear']:
            return 'short'
        return 'neutral'


# ============ ADVANCED VOLATILITY STRATEGIES ============

class VolatilityBreakoutStrategy:
    """Advanced volatility breakout - buys compression, sells expansion"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        atr = Indicators.atr(highs, lows, closes)
        upper, _, lower = Indicators.bollinger_bands(closes)
        
        # Calculate compression
        period_high = max(closes[-20:])
        period_low = min(closes[-20:])
        range_pct = (period_high - period_low) / period_low if period_low > 0 else 0
        atr_pct = atr / mean(closes[-20:]) if closes[-20:] else 0
        
        compression = 1 - min(range_pct * 10 + atr_pct * 10, 1)
        price = closes[-1]
        
        # Buy breakout from compression
        if compression > 0.7 and price > upper:
            return 'long'
        if compression > 0.7 and price < lower:
            return 'short'
        
        return 'neutral'


class VolatilityMRStrategy:
    """Volatility mean reversion - buy low vol, sell high vol"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        atr = Indicators.atr(highs, lows, closes)
        atr_pct = atr / mean(closes[-20:]) if closes[-20:] else 0
        
        # Buy when vol compresses (expecting expansion)
        if atr_pct < 0.015:
            recent_low = min(closes[-20:])
            if closes[-1] > recent_low * 1.01:
                return 'long'
        
        # Sell when vol expands too much
        if atr_pct > 0.04:
            recent_high = max(closes[-20:])
            if closes[-1] < recent_high * 0.99:
                return 'short'
        
        return 'neutral'


class VolatilityTrendStrategy:
    """Volatility-adjusted trend following"""
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        if len(closes) < 50:
            return 'neutral'
        
        atr = Indicators.atr(highs, lows, closes)
        atr_pct = atr / mean(closes[-20:]) if closes[-20:] else 0
        
        # Adjust signal strength based on volatility
        vol_regime = 'high' if atr_pct > 0.03 else 'normal'
        
        sma20 = Indicators.sma(closes, 20)
        sma50 = Indicators.sma(closes, 50)
        trend = (sma20 - sma50) / sma50 if sma50 > 0 else 0
        
        # Stronger trend requirement in high vol
        threshold = 0.02 if vol_regime == 'high' else 0.01
        
        if trend > threshold and closes[-1] > sma20:
            return 'long'
        if trend < -threshold and closes[-1] < sma20:
            return 'short'
        
        return 'neutral'


# ============ ENHANCED ENSEMBLE ============

class EnhancedEnsembleTrader:
    """6-strategy ensemble: 3 original + 3 advanced volatility"""
    
    def __init__(self):
        self.config = StrategyConfig()
        self.strategies = [
            ('trend', TrendStrategy(self.config)),
            ('bb', VolatilityStrategyBB(self.config)),
            ('rsi', MeanReversionRSI(self.config)),
            ('vol_breakout', VolatilityBreakoutStrategy(self.config)),
            ('vol_mr', VolatilityMRStrategy(self.config)),
            ('vol_trend', VolatilityTrendStrategy(self.config)),
        ]
    
    def generate_signal(self, prices: Dict[int, List[float]], regimes: Dict[int, Dict]) -> Dict:
        """Generate signal from 6-strategy ensemble"""
        # Use 1h for primary signals
        closes = prices.get(60, prices.get(240, list(prices.values())[0] if prices else []))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]
        regime = regimes.get(60, regimes.get(240, {'regime': 'unknown', 'trend': 0}))
        
        votes = {'long': 0, 'short': 0, 'neutral': 0}
        signals = {}
        
        # Original 3 strategies
        for name, strategy in self.strategies[:3]:
            if name == 'trend':
                sig = strategy.generate_signal(closes, regime)
            elif name == 'bb':
                sig = strategy.generate_signal(closes, regime)
            else:  # rsi
                sig = strategy.generate_signal(closes, regime)
            signals[name] = sig
            votes[sig] += 1
        
        # Advanced volatility strategies
        for name, strategy in self.strategies[3:]:
            sig = strategy.generate_signal(closes, highs, lows, regime)
            signals[name] = sig
            votes[sig] += 1
        
        # Final signal
        if votes['long'] >= 3:
            signal = 'long'
        elif votes['short'] >= 3:
            signal = 'short'
        elif votes['long'] >= 2 and votes['neutral'] == 1:
            signal = 'long'
        elif votes['short'] >= 2 and votes['neutral'] == 1:
            signal = 'short'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'votes': votes,
            'signals': signals,
            'regime': regime['regime'],
            'trend': regime['trend']
        }


# ============ KRAKEN API ============

class KrakenAPI:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.kraken.com"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EnhancedTrader/3.0'})
    
    def _request(self, path: str, data: Dict = None, public: bool = False) -> Dict:
        url = f"{self.base_url}/0{path}"
        headers = {}
        
        try:
            if public and data:
                response = self.session.get(url, params=data, timeout=5)
            else:
                response = self.session.get(url, headers=headers, timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_ohlc(self, pair: str = "SOLUSD", interval: int = 60, count: int = 200) -> Dict:
        return self._request(f"/public/OHLC", {"pair": pair, "interval": interval}, public=True)


# ============ BACKTEST ============

class EnhancedBacktester:
    def __init__(self, api: KrakenAPI):
        self.api = api
        self.ensemble = EnhancedEnsembleTrader()
        self.config = StrategyConfig()
    
    def fetch_data(self, pair: str = "SOLUSD") -> Dict[int, List]:
        data = {}
        for interval in [60, 240, 1440]:
            result = self.api.get_ohlc(pair, interval, 500)
            if 'result' in result and len(result['result']) > 1:
                data[interval] = list(result['result'].values())[0]
        return data
    
    def run_backtest(self, pair: str = "SOLUSD") -> Dict:
        ohlc_data = self.fetch_data(pair)
        
        if not ohlc_data or 60 not in ohlc_data:
            return {'error': 'Failed to fetch data'}
        
        closes_1h = [float(c[4]) for c in ohlc_data[60]]
        highs_1h = [float(c[2]) for c in ohlc_data[60]]
        lows_1h = [float(c[3]) for c in ohlc_data[60]]
        
        trades = []
        position = None
        equity = [10000]
        
        for i in range(50, len(closes_1h)):
            prices = {tf: [float(c[4]) for c in ohlc_data[tf][:i+1]] for tf in ohlc_data}
            regimes = {tf: RegimeDetector.detect([float(c[4]) for c in ohlc_data[tf][:i+1]]) for tf in ohlc_data}
            
            result = self.ensemble.generate_signal(prices, regimes)
            signal = result['signal']
            price = closes_1h[i]
            
            # Execute trades
            if signal == 'long' and position is None:
                position = {'entry_price': price, 'direction': 'long', 'entry_bar': i}
            elif signal == 'short' and position is None:
                position = {'entry_price': price, 'direction': 'short', 'entry_bar': i}
            elif signal == 'neutral' and position:
                pnl = (price - position['entry_price']) / position['entry_price']
                if position['direction'] == 'short':
                    pnl = -pnl
                trades.append({'pnl': pnl})
                equity.append(equity[-1] * (1 + pnl))
                position = None
        
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            returns = [(pnls[i+1] - pnls[i]) for i in range(len(pnls)-1)] if len(pnls) > 1 else [0]
            
            return {
                'trades': len(trades),
                'win_rate': wins / len(trades),
                'avg_pnl': mean(pnls),
                'max_dd': self._calc_max_drawdown(equity),
                'sharpe': (mean(pnls) / (stdev(pnls) + 0.001)) * 16 if len(pnls) > 1 else 0,
                'total_return': (equity[-1] / equity[0] - 1),
                'regime_analysis': result.get('regime', 'unknown'),
                'vote_breakdown': result.get('votes', {})
            }
        
        return {'trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0, 'sharpe': 0}
    
    def _calc_max_drawdown(self, equity: List[float]) -> float:
        max_eq = equity[0]
        max_dd = 0
        for eq in equity:
            dd = (max_eq - eq) / max_eq
            max_dd = max(max_dd, dd)
            max_eq = max(max_eq, eq)
        return max_dd


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Enhanced Regime-Aware Trading System v3')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--api-key', help='Kraken API key')
    parser.add_argument('--api-secret', help='Kraken API secret')
    parser.add_argument('--pair', default='SOLUSD')
    
    args = parser.parse_args()
    
    api = KrakenAPI(args.api_key, args.api_secret)
    backtester = EnhancedBacktester(api)
    
    if args.backtest:
        print("=== Enhanced Ensemble Backtest (v3) ===")
        results = backtester.run_backtest(args.pair)
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
        else:
            print(f"\n=== RESULTS ===")
            print(f"Trades: {results['trades']}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Avg PnL: {results['avg_pnl']:.2%}")
            print(f"Max Drawdown: {results['max_dd']:.1%}")
            print(f"Sharpe: {results['sharpe']:.2f}")
            print(f"Total Return: {results['total_return']:.1%}")
            print(f"Regime: {results['regime_analysis']}")
            print(f"Votes: {results['vote_breakdown']}")
    else:
        print("Enhanced Regime-Aware Trading System v3")
        print("Options: --backtest")


if __name__ == "__main__":
    main()
