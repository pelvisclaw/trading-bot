#!/usr/bin/env python3
"""
Multi-Timeframe Regime-Aware Trading System v3 - AGGRESSIVE MODE
- Trades ALL market conditions including bear
- Shorts aggressively when regime confirms
- No hiding from volatility
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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev


# ============ REGIME DETECTION ============

class RegimeDetector:
    """Detect market regime from price action"""
    
    @staticmethod
    def detect(closes: List[float]) -> Dict:
        if len(closes) < 20:
            return {'regime': 'unknown', 'confidence': 0, 'trend': 0, 'volatility': 0}
        
        # Trend detection (EMA vs SMA)
        sma_fast = mean(closes[-10:]) if len(closes) >= 10 else mean(closes)
        sma_slow = mean(closes[-30:]) if len(closes) >= 30 else mean(closes)
        trend = (sma_fast - sma_slow) / sma_slow if sma_slow != 0 else 0
        
        # Recent momentum (last 5 candles)
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5]
        else:
            momentum = 0
        
        # Volatility
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        vol = stdev(returns[-20:]) * 100 if len(returns) >= 2 else 2
        
        # Classify regime
        if vol > 5:
            regime = 'high_vol'
        elif trend > 0.01 and momentum > 0:
            regime = 'bull'
        elif trend < -0.01 or momentum < -0.02:
            regime = 'bear'
        else:
            regime = 'sideways'
        
        return {
            'regime': regime,
            'confidence': abs(trend) * 20 + abs(momentum) * 50 + vol,
            'trend': trend,
            'momentum': momentum,
            'volatility': vol
        }


# ============ TECHNICAL INDICATORS ============

class Indicators:
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
    def sma(closes: List[float], period: int) -> float:
        if len(closes) < period:
            return mean(closes)
        return mean(closes[-period:])
    
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


# ============ BEAR MARKET STRATEGY (NEW - AGGRESSIVE) ============

class BearMarketStrategy:
    """Aggressive shorting strategy for bear markets"""
    
    def __init__(self):
        pass
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        if 60 not in prices or len(prices[60]) < 20:
            return 'neutral'
        
        closes = prices[60]
        regime_1h = regimes.get(60, {}).get('regime', 'unknown')
        regime_4h = regimes.get(240, {}).get('regime', 'unknown')
        regime_24h = regimes.get(1440, {}).get('regime', 'unknown')
        
        # Current price vs EMAs
        ema9 = Indicators.ema(closes, 9)
        ema21 = Indicators.ema(closes, 21)
        ema50 = Indicators.ema(closes, 50)
        
        # RSI
        rsi_val = Indicators.rsi(closes, 14)
        
        # Price position
        upper, middle, lower = Indicators.bollinger_bands(closes, 20)
        price = closes[-1]
        bb_position = (price - lower) / (upper - lower) if upper != lower else 0.5
        
        # BEAR MARKET = AGGRESSIVE SHORTS (no hiding!)
        if regime_4h == 'bear' or regime_1h == 'bear':
            # 1. EMA waterfall - strongest signal
            if ema9 < ema21 < ema50:
                return 'short'
            
            # 2. Price below 50 EMA
            if price < ema50:
                return 'short'
            
            # 3. RSI declining but not oversold
            if rsi_val > 45 and rsi_val < 75:
                return 'short'
            
            # 4. Bollinger Band position (lower half = selling)
            if bb_position < 0.4:
                return 'short'
        
        # HIGH VOLATILITY = TREND FOLLOWING
        if regime_24h == 'high_vol':
            if ema9 > ema21 > ema50:
                return 'long'
            if ema9 < ema21 < ema50:
                return 'short'
        
        # SIDEWAYS = MEAN REVERSION
        if regime_1h == 'sideways' and regime_4h == 'sideways':
            if rsi_val < 35:
                return 'long'
            if rsi_val > 65:
                return 'short'
        
        # DEFAULT: Trade with EMAs
        if ema9 > ema21 > ema50:
            return 'long'
        if ema9 < ema21 < ema50:
            return 'short'
        
        return 'neutral'


# ============ TREND STRATEGY ============

class TrendStrategy:
    """Trend following - trades with momentum"""
    
    def __init__(self):
        pass
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        if 60 not in prices or len(prices[60]) < 50:
            return 'neutral'
        
        closes = prices[60]
        ema9 = Indicators.ema(closes, 9)
        ema21 = Indicators.ema(closes, 21)
        ema50 = Indicators.ema(closes, 50)
        momentum = regimes.get(60, {}).get('momentum', 0)
        
        # Strong bullish momentum
        if ema9 > ema21 > ema50 and momentum > 0.005:
            return 'long'
        
        # Strong bearish momentum
        if ema9 < ema21 < ema50 and momentum < -0.005:
            return 'short'
        
        return 'neutral'


# ============ VOLATILITY STRATEGY ============

class VolatilityStrategy:
    """Volatility breakout - trades volatility expansions"""
    
    def __init__(self):
        pass
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        if 60 not in prices or len(prices[60]) < 20:
            return 'neutral'
        
        closes = prices[60]
        upper, middle, lower = Indicators.bollinger_bands(closes, 20)
        
        price = closes[-1]
        rsi_val = Indicators.rsi(closes, 14)
        
        # Volatility squeeze breakout
        band_width = (upper - lower) / middle
        atr_pct = stdev(closes[-20:]) / mean(closes[-20:]) if len(closes) >= 20 else 0.02
        
        # Strong breakout above
        if price > upper and rsi_val < 70:
            return 'long'
        
        # Strong breakout below
        if price < lower and rsi_val > 30:
            return 'short'
        
        return 'neutral'


# ============ MEAN REVERSION STRATEGY ============

class MeanReversionStrategy:
    """Mean reversion - buys dips, sells rips"""
    
    def __init__(self):
        pass
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        if 60 not in prices or len(prices[60]) < 20:
            return 'neutral'
        
        closes = prices[60]
        rsi_val = Indicators.rsi(closes, 14)
        regime = regimes.get(60, {}).get('regime', 'unknown')
        
        # Oversold - buy
        if rsi_val < 30:
            return 'long'
        
        # Overbought - sell
        if rsi_val > 70:
            return 'short'
        
        return 'neutral'


# ============ ENSEMBLE (SIMPLE VOTING) ============

class EnsembleTrader:
    """4-strategy ensemble - any 2 agree = signal"""
    
    def __init__(self):
        self.strategies = [
            BearMarketStrategy(),    # NEW - aggressive bear shorts
            TrendStrategy(),         # Momentum following
            VolatilityStrategy(),    # Breakouts
            MeanReversionStrategy()  # Oversold/overbought
        ]
    
    def generate_signal(self, regimes: Dict[int, Dict], prices: Dict[int, List[float]]) -> str:
        votes = {'long': 0, 'short': 0, 'neutral': 0}
        
        for strategy in self.strategies:
            signal = strategy.generate_signal(regimes, prices)
            votes[signal] += 1
        
        # Any 1+ agree = signal (aggressive mode)
        if votes['long'] >= 1:
            return 'long'
        if votes['short'] >= 1:
            return 'short'
        return 'neutral'


# ============ KRAKEN API ============

class KrakenAPI:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.kraken.com"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'RegimeTrader/3.0'})
    
    def get_ohlc(self, pair: str = "SOLUSD", interval: int = 60, count: int = 200) -> Dict:
        params = {"pair": pair, "interval": interval}
        try:
            resp = self.session.get(f"{self.base_url}/0/public/OHLC", params=params, timeout=5)
            data = resp.json()
            if 'result' in data and len(data['result']) > 1:
                return {'result': list(data['result'].values())[0]}
        except Exception as e:
            pass
        return {}


# ============ BACKTEST ============

class Backtester:
    def __init__(self, api: KrakenAPI):
        self.api = api
        self.ensemble = EnsembleTrader()
    
    def fetch_data(self) -> Dict[int, List]:
        data = {}
        for interval in [60, 240, 1440]:
            result = self.api.get_ohlc('SOLUSD', interval, 1000)
            if 'result' in result:
                data[interval] = result['result']
        return data
    
    def run_backtest(self) -> Dict:
        ohlc_data = self.fetch_data()
        
        if not ohlc_data or 60 not in ohlc_data:
            return {'error': 'No data'}
        
        closes_1h = [float(c[4]) for c in ohlc_data[60]]
        
        trades = []
        position = None
        
        for i in range(30, len(closes_1h)):
            # Build price history
            prices = {}
            regimes = {}
            
            for tf, candles in ohlc_data.items():
                tf_closes = [float(c[4]) for c in candles[:i+1]]
                prices[tf] = tf_closes
                regimes[tf] = RegimeDetector.detect(tf_closes)
            
            signal = self.ensemble.generate_signal(regimes, prices)
            price = closes_1h[i]
            
            # Execute
            if signal == 'long' and position is None:
                position = {'entry': price, 'dir': 'long'}
            elif signal == 'short' and position is None:
                position = {'entry': price, 'dir': 'short'}
            elif signal == 'neutral' and position:
                pnl = (price - position['entry']) / position['entry']
                if position['dir'] == 'short':
                    pnl = -pnl
                trades.append(pnl)
                position = None
        
        # Close open
        if position:
            pnl = (closes_1h[-1] - position['entry']) / position['entry']
            if position['dir'] == 'short':
                pnl = -pnl
            trades.append(pnl)
        
        if not trades:
            return {'trades': 0, 'win_rate': 0, 'sharpe': 0, 'return': 0}
        
        wins = sum(1 for t in trades if t > 0)
        avg = mean(trades) if trades else 0
        std = stdev(trades) if len(trades) > 1 else 0.01
        
        return {
            'trades': len(trades),
            'win_rate': wins / len(trades),
            'sharpe': (avg / std) * 16 if std > 0 else 0,
            'return': mean(trades) if trades else 0
        }


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='RegimeTrader v3 - AGGRESSIVE')
    parser.add_argument('--backtest', action='store_true')
    args = parser.parse_args()
    
    api = KrakenAPI()
    bt = Backtester(api)
    
    if args.backtest:
        print("=== RegimeTrader v3 BACKTEST ===")
        results = bt.run_backtest()
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
        else:
            print(f"\nResults:")
            print(f"  Trades: {results['trades']}")
            print(f"  Win Rate: {results['win_rate']:.1%}")
            print(f"  Sharpe: {results['sharpe']:.2f}")
            print(f"  Avg Return: {results['return']:.2%}")
    else:
        # Live signal
        print("=== RegimeTrader v3 SIGNAL ===")
        ohlc_data = bt.fetch_data()
        
        if not ohlc_data:
            print("ERROR: No data")
            return
        
        prices = {}
        regimes = {}
        for tf, candles in ohlc_data.items():
            closes = [float(c[4]) for c in candles]
            prices[tf] = closes
            regimes[tf] = RegimeDetector.detect(closes)
        
        signal = bt.ensemble.generate_signal(regimes, prices)
        
        print(f"\nRegimes:")
        for tf in sorted(regimes.keys()):
            r = regimes[tf]
            print(f"  {tf}m: {r['regime']} (mom={r.get('momentum', 0):.3f})")
        
        print(f"\n SIGNAL: {signal.upper()}")
        print(f"  Price: ${prices[60][-1]:.2f}")


if __name__ == "__main__":
    main()
