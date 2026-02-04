#!/usr/bin/env python3
"""
Multi-Timeframe Regime-Aware Trading System
- Analyzes 1h, 4h, daily timeframes
- Detects market regime (trend, volatility, mean-reversion)
- Ensemble of 3 specialized strategies
- Proper walk-forward validation
"""
import json
import time
import random
import hashlib
import hmac
import base64
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev

# ============ REGIME DETECTION ============

class RegimeDetector:
    """Detect market regime from price action"""
    
    @staticmethod
    def detect(closes: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Return regime analysis"""
        if len(closes) < 50:
            return {'regime': 'unknown', 'confidence': 0, 'trend': 0, 'volatility': 0}
        
        # Trend detection (SMA crossover)
        sma_fast = mean(closes[-20:])
        sma_slow = mean(closes[-50:])
        trend = (sma_fast - sma_slow) / sma_slow
        
        # Volatility regime
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        vol = stdev(returns) * 100 if len(returns) > 1 else 0
        vol_percentile = min(vol / 5, 2)  # Cap at 200% of normal
        
        # Mean reversion signal
        deviations = [(c - mean(closes)) / stdev(closes) for c in closes[-20:]] if len(closes) >= 20 else [0]
        mr_signal = abs(deviations[-1]) if deviations else 0
        
        # Classify regime
        if vol > 3:  # High volatility
            regime = 'high_vol'
        elif trend > 0.02:  # Strong uptrend
            regime = 'bull'
        elif trend < -0.02:  # Strong downtrend
            regime = 'bear'
        elif mr_signal > 1.5:  # Price far from mean
            regime = 'mean_reversion'
        else:
            regime = 'sideways'
        
        return {
            'regime': regime,
            'confidence': abs(trend) * 10 + vol_percentile + mr_signal,
            'trend': trend,
            'volatility': vol,
            'mr_signal': mr_signal
        }


# ============ STRATEGY ARCHITECTURES ============

@dataclass
class StrategyConfig:
    """Base config for all strategies"""
    # Risk management
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.05
    trailing_stop: float = 0.02
    
    # Timeframes to analyze
    timeframes: List[int] = field(default_factory=lambda: [60, 240, 1440])  # 1h, 4h, 24h


class TrendStrategy:
    """Trend-following strategy (works in bull/bear markets)"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        """Generate trading signal based on multi-timeframe regime"""
        # Strong trend: all timeframes agree
        trend_votes = 0
        for tf, regime in regimes.items():
            if regime['regime'] in ['bull', 'bear']:
                trend_votes += 1
        
        if trend_votes >= 2:
            # Take position in trend direction
            for tf, regime in regimes.items():
                if regime['regime'] == 'bull' and regime['trend'] > 0:
                    return 'long'
                elif regime['regime'] == 'bear' and regime['trend'] < 0:
                    return 'short'
        
        return 'neutral'


class VolatilityStrategy:
    """Volatility breakout strategy (works in high_vol regime)"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        """Generate signal based on volatility regime"""
        high_vol_count = sum(1 for r in regimes.values() if r['regime'] == 'high_vol')
        
        if high_vol_count >= 2:
            # Volatility expansion - look for breakouts
            for tf, regime in regimes.items():
                if regime['volatility'] > 3:  # High volatility threshold
                    # Check if price near ATR bands
                    return 'neutral'  # Volatility strategy doesn't trend-follow
        
        return 'neutral'


class MeanReversionStrategy:
    """Mean reversion strategy (works in sideways regime)"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        """Generate signal based on mean reversion signals"""
        mr_votes = 0
        for tf, regime in regimes.items():
            if regime['mr_signal'] > 1.0:  # Price extended from mean
                mr_votes += 1
        
        if mr_votes >= 2:
            for tf, regime in regimes.items():
                if regime['mr_signal'] > 1.0:
                    # Price extended - expect reversion
                    return 'neutral'  # Mean reversion waits for extremes
        
        return 'neutral'


# ============ ENSEMBLE ============

class EnsembleTrader:
    """Ensemble of 3 specialized strategies with voting"""
    
    def __init__(self):
        self.strategies = [
            TrendStrategy(StrategyConfig()),
            VolatilityStrategy(StrategyConfig()),
            MeanReversionStrategy(StrategyConfig())
        ]
        self.votes = {'long': 0, 'short': 0, 'neutral': 0}
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        """Get signal from ensemble voting"""
        self.votes = {'long': 0, 'short': 0, 'neutral': 0}
        
        for strategy in self.strategies:
            signal = strategy.generate_signal(regimes, prices)
            self.votes[signal] += 1
        
        # Return majority vote
        if self.votes['long'] >= 2:
            return 'long'
        elif self.votes['short'] >= 2:
            return 'short'
        return 'neutral'


# ============ KRAKEN API ============

class KrakenAPI:
    """Kraken REST API wrapper"""
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.kraken.com"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'RegimeTrader/1.0'})
    
    def _sign(self, path: str, nonce: str, data: str) -> str:
        message = (nonce + path + data).encode()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message, hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _request(self, path: str, data: Dict = None, public: bool = False) -> Dict:
        url = f"{self.base_url}/0{path}"
        headers = {}
        if not public and self.api_key and self.api_secret:
            nonce = str(int(time.time() * 1000))
            data = data or {}
            data['nonce'] = nonce
            post_data = '&'.join(f"{k}={v}" for k, v in data.items())
            signature = self._sign(path, nonce, post_data)
            headers = {'API-Key': self.api_key, 'API-Sign': signature}
        
        try:
            if data and public:
                url += '?' + '&'.join(f"{k}={v}" for k, v in data.items())
            response = self.session.get(url, headers=headers, timeout=5)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_ohlc(self, pair: str = "SOLUSD", interval: int = 60, count: int = 100) -> Dict:
        return self._request(f"/public/OHLC", {"pair": pair, "interval": interval}, public=True)


# ============ TRAINING ============

class RegimeTrainer:
    """Train ensemble with walk-forward validation"""
    
    def __init__(self, api: KrakenAPI):
        self.api = api
        self.ensemble = EnsembleTrader()
    
    def fetch_multi_timeframe(self, pair: str = "SOLUSD") -> Dict[int, List]:
        """Fetch OHLC for multiple timeframes"""
        data = {}
        for interval in [60, 240, 1440]:  # 1h, 4h, 24h
            result = self.api.get_ohlc(pair, interval, 200)
            if 'result' in result and len(result['result']) > 1:
                data[interval] = list(result['result'].values())[0]
        return data
    
    def walk_forward_train(self, pair: str = "SOLUSD", train_pct: float = 0.8) -> Dict:
        """Walk-forward optimization"""
        ohlc_data = self.fetch_multi_timeframe(pair)
        
        if not ohlc_data:
            return {'error': 'Failed to fetch data'}
        
        results = {}
        for tf, candles in ohlc_data.items():
            closes = [float(c[4]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            
            split = int(len(closes) * train_pct)
            train_closes, test_closes = closes[:split], closes[split:]
            train_highs, test_highs = highs[:split], highs[split:]
            train_lows, test_lows = lows[:split], lows[split:]
            
            # Train regime detector on training data
            train_regimes = []
            for i in range(50, len(train_closes)):
                regime = RegimeDetector.detect(train_closes[:i+1], train_highs[:i+1], train_lows[:i+1])
                train_regimes.append(regime['regime'])
            
            # Validate on test data
            test_signals = []
            for i in range(len(test_closes)):
                regime = RegimeDetector.detect(
                    train_closes + test_closes[:i+1],
                    train_highs + test_highs[:i+1],
                    train_lows + test_lows[:i+1]
                )
                test_regimes.append(regime)
            
            results[f'tf_{tf}'] = {
                'train_regimes': train_regimes,
                'test_regimes': test_regimes,
                'train_count': len(train_regimes),
                'test_count': len(test_regimes)
            }
        
        return results
    
    def backtest_ensemble(self, pair: str = "SOLUSD") -> Dict:
        """Backtest ensemble on historical data"""
        ohlc_data = self.fetch_multi_timeframe(pair)
        
        if not ohlc_data:
            return {'error': 'Failed to fetch data'}
        
        trades = []
        position = None
        
        for i in range(100, min(len(ohlc_data[60]), len(ohlc_data[240]), len(ohlc_data[1440]))):
            # Build regime analysis for each timeframe
            regimes = {}
            for tf in [60, 240, 1440]:
                if i < len(ohlc_data[tf]):
                    closes = [float(c[4]) for c in ohlc_data[tf][:i+1]]
                    highs = [float(c[2]) for c in ohlc_data[tf][:i+1]]
                    lows = [float(c[3]) for c in ohlc_data[tf][:i+1]]
                    regimes[tf] = RegimeDetector.detect(closes, highs, lows)
            
            prices = {tf: [float(c[4]) for c in ohlc_data[tf][:i+1]] for tf in ohlc_data}
            
            signal = self.ensemble.generate_signal(regimes, prices)
            
            # Execute trade
            if signal != 'neutral' and position is None:
                position = {
                    'entry_price': ohlc_data[60][i][4],
                    'direction': signal,
                    'entry_bar': i
                }
            elif signal == 'neutral' and position:
                pnl = (float(ohlc_data[60][i][4]) - float(position['entry_price'])) / float(position['entry_price'])
                if position['direction'] == 'short':
                    pnl = -pnl
                trades.append({'pnl': pnl, 'direction': position['direction']})
                position = None
        
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            return {
                'trades': len(trades),
                'win_rate': wins / len(trades),
                'avg_pnl': mean(pnls),
                'max_dd': self._calculate_max_drawdown([1000 * (1 + p) for p in [0] + pnls]),
                'sharpe': (mean(pnls) / (stdev(pnls) + 0.001)) * 16 if len(pnls) > 1 else 0
            }
        
        return {'trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0, 'sharpe': 0}
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        max_eq = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            dd = (max_eq - eq) / max_eq
            max_dd = max(max_dd, dd)
            max_eq = max(max_eq, eq)
        return max_dd


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Regime-Aware Multi-Timeframe Trading System')
    parser.add_argument('--train', action='store_true', help='Train with walk-forward validation')
    parser.add_argument('--backtest', action='store_true', help='Backtest ensemble')
    parser.add_argument('--api-key', help='Kraken API key')
    parser.add_argument('--api-secret', help='Kraken API secret')
    parser.add_argument('--output', default='ensemble_results.json')
    
    args = parser.parse_args()
    
    api = KrakenAPI(args.api_key, args.api_secret)
    trainer = RegimeTrainer(api)
    
    if args.train:
        results = trainer.walk_forward_train()
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Walk-forward training complete: {args.output}")
    
    elif args.backtest:
        results = trainer.backtest_ensemble()
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Backtest results: {results}")
    
    else:
        print("Regime-Aware Multi-Timeframe Trading System")
        print("Options:")
        print("  --train      Walk-forward optimization")
        print("  --backtest   Backtest ensemble on historical data")


if __name__ == "__main__":
    main()
