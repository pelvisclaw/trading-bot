#!/usr/bin/env python3
"""
RegimeTrader v3 Rotation Strategy - CROSS-COIN TRADING
- Trades between coins directly (no USD conversion)
- Rotates to strongest performer
- Minimizes fees
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, '/home/nueki/.openclaw/workspace/trading-bot/src')

from kraken_client import KrakenClient
from strategies.regime_trader_v3 import RegimeDetector


# v3 Aggressive signal generation
def generate_v3_signal(regimes: dict, prices: dict) -> str:
    """v3 aggressive signal generation"""
    regime_60 = regimes.get(60, 'sideways')
    regime_240 = regimes.get(240, 'sideways')
    regime_1440 = regimes.get(1440, 'sideways')
    
    if regime_60 == 'bear':
        return 'short'
    elif regime_60 == 'bull':
        return 'long'
    elif regime_60 == 'sideways':
        if regime_240 == 'bull':
            return 'long'
        elif regime_240 == 'bear':
            return 'short'
    
    return 'neutral'


class RotationTrader:
    """
    Cross-coin rotation strategy.
    Holds strongest coin, rotates when another outperforms.
    Avoids USD conversions.
    """
    
    STATE_FILE = "rotation_trader_state.json"
    LOG_FILE = "rotation_trading.log"
    
    # Supported coins and their base pair codes
    COINS = {
        'XBT': {'ticker_pair': 'XXBTZUSD', 'crypto_pair': None, 'asset': 'XXBT'},
        'ETH': {'ticker_pair': 'XETHZUSD', 'crypto_pair': 'XBTETH', 'asset': 'XETH'},
        'SOL': {'ticker_pair': 'SOLUSD', 'crypto_pair': 'SOLETH', 'asset': 'SOL'},
        'LINK': {'ticker_pair': 'LINKUSD', 'crypto_pair': 'XBTCH', 'asset': 'LINK'},
    }
    
    # Trading pairs (from coin to coin)
    CRYPTO_PAIRS = {
        ('ETH', 'XBT'): 'XBTETH',  # Buy XBT with ETH
        ('XBT', 'ETH'): 'XBTETH',  # Buy ETH with XBT
        ('SOL', 'ETH'): 'SOLETH',  # Buy SOL with ETH
        ('ETH', 'SOL'): 'SOLETH',  # Buy ETH with SOL
        ('XBT', 'SOL'): 'SOLXBT',  # Buy SOL with XBT
        ('SOL', 'XBT'): 'SOLXBT',  # Buy XBT with SOL
    }
    
    def __init__(self, api: KrakenClient, initial_capital: float = 10000, dry_run: bool = True):
        self.api = api
        self.dry_run = dry_run
        self.capital = initial_capital  # In USD equivalent
        self.holding = None  # Coin we're holding: 'XBT', 'ETH', 'SOL', 'LINK'
        self.holding_amount = 0  # Amount of coin held
        self.trades = []
        self._load_state()
    
    def _log(self, msg: str):
        """Log to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")
        print(f"[{timestamp}] {msg}")
    
    def _save_state(self):
        """Persist state"""
        state = {
            'capital': self.capital,
            'holding': self.holding,
            'holding_amount': self.holding_amount,
            'trades': self.trades
        }
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load persisted state"""
        try:
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)
                self.capital = state.get('capital', self.capital)
                self.holding = state.get('holding')
                self.holding_amount = state.get('holding_amount', 0)
                self.trades = state.get('trades', [])
        except FileNotFoundError:
            pass
    
    def get_balance(self) -> dict:
        """Get account balance"""
        try:
            balance = self.api.get_balance()
            return {
                'XBT': float(balance.get('XXBT', 0)),
                'ETH': float(balance.get('XETH', 0)),
                'SOL': float(balance.get('SOL', 0)),
                'LINK': float(balance.get('LINK', 0)),
                'USD': float(balance.get('ZUSD', 0)),
            }
        except Exception as e:
            self._log(f"ERROR getting balance: {e}")
            return {}
    
    def get_prices(self) -> dict:
        """Get current prices for all coins"""
        prices = {}
        for coin, config in self.COINS.items():
            try:
                if config['crypto_pair']:
                    # Try crypto-crypto price first
                    try:
                        ticker = self.api.get_ticker(config['crypto_pair'])
                        if ticker:
                            prices[f'{coin}_price'] = float(ticker.get('c', [0])[0])
                        else:
                            raise Exception("No ticker data")
                    except:
                        # Fall back to USD price
                        ticker = self.api.get_ticker(config['ticker_pair'])
                        if ticker:
                            prices[f'{coin}_price'] = float(ticker.get('c', [0])[0])
                else:
                    # Get USD price
                    ticker = self.api.get_ticker(config['ticker_pair'])
                    if ticker:
                        prices[f'{coin}_price'] = float(ticker.get('c', [0])[0])
            except Exception as e:
                self._log(f"ERROR getting {coin} price: {e}")
        return prices
    
    def get_regimes(self) -> dict:
        """Get regime for each coin"""
        regimes = {}
        for coin, config in self.COINS.items():
            try:
                # Get OHLC data
                candles = self.api.get_ohlc(config['ticker_pair'], 60, 100)
                if candles:
                    closes = [float(c.close) for c in candles]
                    regimes[coin] = RegimeDetector.detect(closes)
            except Exception as e:
                self._log(f"ERROR getting {coin} regime: {e}")
        return regimes
    
    def score_coin(self, coin: str, regime: dict, price: float) -> float:
        """Score a coin based on regime and momentum"""
        score = 0
        
        # Regime score
        regime_score = {'bull': 1, 'sideways': 0, 'bear': -1, 'high_vol': 0.5}
        score += regime_score.get(regime['regime'], 0) * 30
        
        # Momentum score
        score += regime.get('momentum', 0) * 50
        
        # Trend score
        score += regime.get('trend', 0) * 20
        
        return score
    
    def analyze(self) -> dict:
        """Analyze all coins and determine rotation"""
        prices = self.get_prices()
        regimes = self.get_regimes()
        
        # Calculate scores
        scores = {}
        for coin in self.COINS.keys():
            if coin in regimes and f'{coin}_price' in prices:
                score = self.score_coin(coin, regimes[coin], prices[f'{coin}_price'])
                scores[coin] = {
                    'score': score,
                    'regime': regimes[coin]['regime'],
                    'momentum': regimes[coin].get('momentum', 0),
                    'price': prices[f'{coin}_price']
                }
        
        return {
            'scores': scores,
            'prices': prices,
            'current_holding': self.holding
        }
    
    def execute_rotation(self, analysis: dict):
        """Execute rotation if needed"""
        scores = analysis['scores']
        if not scores:
            self._log("ERROR: No scores available")
            return
        
        # Find best coin
        best_coin = max(scores.keys(), key=lambda c: scores[c]['score'])
        best_score = scores[best_coin]['score']
        
        current_score = scores.get(self.holding, {'score': 0})['score'] if self.holding else 0
        
        # Rotate if best coin is significantly better
        threshold = 10  # Score difference threshold
        
        if self.holding is None:
            # No holding - buy best coin
            self._log(f"NO HOLDING - Buying {best_coin}")
            self._rotate_to(best_coin, analysis['prices'])
        
        elif best_score - current_score > threshold:
            # Best coin is significantly better - rotate
            self._log(f"ROTATING: {self.holding} ({current_score:.1f}) → {best_coin} ({best_score:.1f})")
            self._rotate_to(best_coin, analysis['prices'])
        
        else:
            # Keep current holding
            self._log(f"HOLDING: {self.holding} (score: {current_score:.1f}, best: {best_score:.1f})")
    
    def _rotate_to(self, target_coin: str, prices: dict):
        """Rotate holdings to target coin"""
        if self.holding == target_coin:
            return
        
        balance = self.get_balance()
        
        # If we're holding a coin, sell it and buy target
        if self.holding and balance.get(self.holding, 0) > 0.0001:
            from_coin = self.holding
            to_coin = target_coin
            
            # Find crypto-crypto trading pair
            # For SOL→XBT, we need SOLXBT pair (buy SOL with XBT = sell SOL for XBT)
            pair = self.CRYPTO_PAIRS.get((from_coin, to_coin))
            
            if pair:
                amount = balance[from_coin]
                
                if self.dry_run:
                    self._log(f"DRY RUN: Would swap {amount:.6f} {from_coin} → {to_coin}")
                    self.holding = to_coin
                    self._save_state()
                    return
                
                # Execute cross-coin trade: sell from_coin, buy to_coin
                try:
                    result = self.api.place_order(pair, 'sell', 'market', amount)
                    if 'error' in str(result).lower():
                        self._log(f"ERROR swapping {from_coin} → {to_coin}: {result}")
                        return
                    
                    self._log(f"ROTATED: {from_coin} → {to_coin} via {pair}")
                    self.holding = to_coin
                    self._save_state()
                    
                except Exception as e:
                    self._log(f"ERROR: {e}")
            else:
                # No direct pair - need to go through USD
                self._log(f"No direct pair {from_coin} → {to_coin}, trying USD path...")
                
                # Sell from_coin for USD
                usd_pair = self.COINS[from_coin]['ticker_pair']
                try:
                    result = self.api.place_order(usd_pair, 'sell', 'market', amount)
                    if 'error' in str(result).lower():
                        self._log(f"ERROR selling {from_coin}: {result}")
                        return
                    
                    # Buy to_coin with USD
                    new_balance = self.get_balance()
                    usd = new_balance.get('USD', 0)
                    
                    to_pair = self.COINS[to_coin]['ticker_pair']
                    result = self.api.place_order(to_pair, 'buy', 'market', usd)
                    if 'error' in str(result).lower():
                        self._log(f"ERROR buying {to_coin}: {result}")
                        return
                    
                    self._log(f"ROTATED VIA USD: {from_coin} → USD → {to_coin}")
                    self.holding = to_coin
                    self._save_state()
                except Exception as e:
                    self._log(f"ERROR: {e}")
        
        # If no holding, buy target coin with USD
        elif target_coin:
            balance = self.get_balance()
            usd_amount = balance.get('USD', 0)
            
            if usd_amount < 1:
                self._log(f"ERROR: Insufficient USD ({usd_amount}) to buy {target_coin}")
                return
            
            # Find USD pair for target coin
            usd_pair = self.COINS[target_coin]['ticker_pair']
            
            if self.dry_run:
                self._log(f"DRY RUN: Would buy {target_coin} with ${usd_amount}")
                self.holding = target_coin
                self.holding_amount = usd_amount / prices.get(f'{target_coin}_price', 1)
                self._save_state()
                return
            
            # Execute USD buy
            try:
                result = self.api.place_order(usd_pair, 'buy', 'market', usd_amount)
                if 'error' in str(result).lower():
                    self._log(f"ERROR buying {target_coin}: {result}")
                    return
                
                self._log(f"BOUGHT: {target_coin} with ${usd_amount}")
                self.holding = target_coin
                self.holding_amount = usd_amount / prices.get(f'{target_coin}_price', 1)
                self._save_state()
            except Exception as e:
                self._log(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='RegimeTrader v3 Rotation Strategy')
    parser.add_argument('--api-key', 
                       default=os.environ.get('KRAKEN_API_KEY', ''),
                       help='Kraken API key (or set KRAKEN_API_KEY env var)')
    parser.add_argument('--api-secret',
                       default=os.environ.get('KRAKEN_API_SECRET', ''),
                       help='Kraken API secret (or set KRAKEN_API_SECRET env var)')
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (default if --live not specified)')
    parser.add_argument('--live', action='store_true',
                       help='Execute real trades (default is dry run)')
    args = parser.parse_args()
    
    # dry_run is True unless --live is explicitly specified
    dry_run = not args.live
    
    # Create credentials wrapper
    class C:
        def __init__(self, k, s): self._k, self._s = k, s
        @property
        def api_key(self): return self._k
        @property
        def api_secret(self): return self._s
        @property
        def is_available(self): return bool(self._k and self._s)
    
    creds = C(args.api_key, args.api_secret)
    
    if not creds.is_available:
        print("ERROR: API credentials required")
        print("Set KRAKEN_API_KEY and KRAKEN_API_SECRET env vars, or pass --api-key/--api-secret")
        sys.exit(1)
    
    api = KrakenClient(credentials=creds)
    
    trader = RotationTrader(api, args.capital, dry_run=dry_run)
    
    mode = "DRY RUN" if dry_run else "LIVE TRADING"
    print("=" * 50)
    print(f"REGIME-TRADER v3 ROTATION - {mode}")
    print("=" * 50)
    
    if not dry_run:
        print("\n⚠️  WARNING: REAL MONEY AT RISK ⚠️")
        print("Ctrl+C to cancel within 5 seconds...")
        time.sleep(5)
    
    # Analyze
    analysis = trader.analyze()
    
    print(f"\nScores:")
    for coin, data in analysis['scores'].items():
        holding_marker = " 👈 HOLDING" if coin == analysis['current_holding'] else ""
        print(f"  {coin}: {data['score']:.1f} ({data['regime']}){holding_marker}")
    
    # Execute rotation
    trader.execute_rotation(analysis)
    
    # Status
    balance = trader.get_balance()
    print(f"\nStatus:")
    print(f"  Holding: {trader.holding} ({trader.holding_amount})")
    print(f"  Balance: {balance}")
    print(f"  Trades: {len(trader.trades)}")


if __name__ == "__main__":
    main()
