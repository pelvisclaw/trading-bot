#!/usr/bin/env python3
"""
Advanced Volatility Strategies for Regime Ensemble
- Volatility Mean Reversion
- Volatility Regime Detection (compression/expansion)
- ATR-Based Position Sizing
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
from dataclasses import dataclass
from statistics import mean, stdev


class AdvancedVolatilityStrategies:
    """Advanced volatility trading strategies"""
    
    def __init__(self, config: 'VolatilityConfig'):
        self.config = config
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return 0.01 * mean(closes)
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return mean(trs[-period:]) if trs else 0.01 * mean(closes)
    
    @staticmethod
    def calculate_volatility_regime(closes: List[float], atr: float, period: int = 20) -> Dict:
        """Detect volatility regime (compression vs expansion)"""
        if len(closes) < period:
            return {'regime': 'unknown', 'compression_score': 0, 'expansion_score': 0}
        
        # Price range over period
        period_high = max(closes[-period:])
        period_low = min(closes[-period:])
        range_pct = (period_high - period_low) / period_low if period_low > 0 else 0
        
        # ATR as percentage
        avg_price = mean(closes[-period:])
        atr_pct = (atr / avg_price) if avg_price > 0 else 0
        
        # Bollinger Band width
        std = stdev(closes[-period:]) if len(closes) >= period else 1
        bb_width = (4 * std) / avg_price if avg_price > 0 else 0
        
        # Compression detection (low volatility → likely expansion)
        compression_score = 1 - min(range_pct * 10 + atr_pct * 10 + bb_width * 5, 1)
        
        # Expansion detection (high volatility → likely mean reversion)
        expansion_score = min(range_pct * 10 + atr_pct * 10 + bb_width * 5, 1)
        
        # Regime classification
        if compression_score > 0.7:
            regime = 'vol_compression'  # Low vol, preparing for breakout
        elif expansion_score > 0.7:
            regime = 'vol_expansion'  # High vol, likely mean reversion
        elif range_pct < 0.02:
            regime = 'vol_squeeze'  # Tight range, imminent move
        else:
            regime = 'vol_normal'
        
        return {
            'regime': regime,
            'compression_score': compression_score,
            'expansion_score': expansion_score,
            'range_pct': range_pct,
            'atr_pct': atr_pct,
            'bb_width': bb_width
        }
    
    def volatility_breakout_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        """
        Volatility Breakout Strategy
        Buy when price breaks out of compression
        """
        if len(closes) < 50:
            return 'neutral'
        
        atr = self.calculate_atr(highs, lows, closes)
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, 20, 2)
        
        price = closes[-1]
        regime_analysis = self.calculate_volatility_regime(closes, atr)
        
        # Long signal: Price breaks above upper band in compression
        if regime_analysis['regime'] in ['vol_compression', 'vol_squeeze']:
            if price > bb_upper:
                return 'long'
        
        # Short signal: Price breaks below lower band in compression
        if regime_analysis['regime'] in ['vol_compression', 'vol_squeeze']:
            if price < bb_lower:
                return 'short'
        
        return 'neutral'
    
    def volatility_mean_reversion_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        """
        Volatility Mean Reversion Strategy
        Sell volatility when high, buy when low
        """
        if len(closes) < 50:
            return 'neutral'
        
        atr = self.calculate_atr(highs, lows, closes)
        regime_analysis = self.calculate_volatility_regime(closes, atr)
        
        # Buy when volatility compresses (expecting expansion)
        if regime_analysis['regime'] == 'vol_compression':
            # Look for support level
            recent_low = min(closes[-20:])
            if closes[-1] > recent_low * 1.02:
                return 'long'
        
        # Sell/reduce when volatility expands
        if regime_analysis['regime'] == 'vol_expansion':
            # Look for resistance
            recent_high = max(closes[-20:])
            if closes[-1] < recent_high * 0.98:
                return 'short'
        
        return 'neutral'
    
    def volatility_trend_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> str:
        """
        Volatility Trend Strategy
        Trend-following adapted to volatility regime
        """
        if len(closes) < 50:
            return 'neutral'
        
        atr = self.calculate_atr(highs, lows, closes)
        regime_analysis = self.calculate_volatility_regime(closes, atr)
        
        # Calculate trend
        sma20 = mean(closes[-20:])
        sma50 = mean(closes[-50:]) if len(closes) >= 50 else sma20
        trend = (sma20 - sma50) / sma50 if sma50 > 0 else 0
        
        # In compression, wait for clear trend
        if regime_analysis['regime'] in ['vol_compression', 'vol_squeeze']:
            if trend > 0.02 and closes[-1] > sma20:
                return 'long'
            elif trend < -0.02 and closes[-1] < sma20:
                return 'short'
        
        # In expansion, shorter trend-following
        if regime_analysis['regime'] == 'vol_expansion':
            if trend > 0.01 and closes[-1] > sma20:
                return 'long'
            elif trend < -0.01 and closes[-1] < sma20:
                return 'short'
        
        return 'neutral'
    
    def _bollinger_bands(self, closes: List[float], period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        ma = mean(closes[-period:]) if len(closes) >= period else mean(closes)
        std = stdev(closes[-period:]) if len(closes) >= period else 1
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        return upper, ma, lower
    
    def get_volatility_signal(self, closes: List[float], highs: List[float], lows: List[float], regime: Dict) -> Dict:
        """
        Get combined volatility signal from all strategies
        """
        breakout = self.volatility_breakout_signal(closes, highs, lows, regime)
        mean_reversion = self.volatility_mean_reversion_signal(closes, highs, lows, regime)
        trend = self.volatility_trend_signal(closes, highs, lows, regime)
        
        atr = self.calculate_atr(highs, lows, closes)
        vol_regime = self.calculate_volatility_regime(closes, atr)
        
        # Voting
        votes = {'long': 0, 'short': 0, 'neutral': 0}
        if breakout == 'long':
            votes['long'] += 1
        elif breakout == 'short':
            votes['short'] += 1
        else:
            votes['neutral'] += 1
            
        if mean_reversion == 'long':
            votes['long'] += 1
        elif mean_reversion == 'short':
            votes['short'] += 1
        else:
            votes['neutral'] += 1
            
        if trend == 'long':
            votes['long'] += 1
        elif trend == 'short':
            votes['short'] += 1
        else:
            votes['neutral'] += 1
        
        # Final signal
        if votes['long'] >= 2:
            signal = 'long'
        elif votes['short'] >= 2:
            signal = 'short'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'votes': votes,
            'breakout': breakout,
            'mean_reversion': mean_reversion,
            'trend': trend,
            'volatility_regime': vol_regime,
            'atr_pct': vol_regime['atr_pct'],
            'compression_score': vol_regime['compression_score']
        }


def test_strategies():
    """Test the volatility strategies"""
    # Generate synthetic data
    import random
    random.seed(42)
    
    closes = [100]
    for i in range(200):
        change = random.gauss(0, 1)
        closes.append(closes[-1] * (1 + change / 100))
    
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    
    config = None  # Not used
    strategies = AdvancedVolatilityStrategies(config)
    
    result = strategies.get_volatility_signal(closes, highs, lows, {'regime': 'unknown'})
    
    print("=== Volatility Strategy Test ===")
    print(f"Signal: {result['signal']}")
    print(f"Votes: {result['votes']}")
    print(f"Breakout: {result['breakout']}")
    print(f"Mean Reversion: {result['mean_reversion']}")
    print(f"Trend: {result['trend']}")
    print(f"Volatility Regime: {result['volatility_regime']['regime']}")
    print(f"ATR %: {result['atr_pct']:.2%}")
    print(f"Compression Score: {result['compression_score']:.2f}")


if __name__ == "__main__":
    test_strategies()
