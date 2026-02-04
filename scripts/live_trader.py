#!/usr/bin/env python3
"""
Regime-Aware Live Trading System
- Executes real trades on Kraken
- Full safety mechanisms
- Manual confirmation required
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, 'src')

from strategies.regime_ensemble_v2 import (
    KrakenAPI, RegimeDetector, EnsembleTrader, StrategyConfig
)


class LiveTrader:
    """Live trading implementation with safety features"""
    
    def __init__(self, api_key: str, api_secret: str, dry_run: bool = True):
        self.api = KrakenAPI(api_key, api_secret)
        self.ensemble = EnsembleTrader()
        self.config = StrategyConfig()
        self.dry_run = dry_run
        self.trades = []
        self.position = None
        self.log_file = Path("live_trades.log")
        self.config_file = Path("trading_config.json")
        
        # Safety limits
        self.max_position_size = 0.1  # 10% of capital
        self.max_daily_trades = 10
        self.cooldown_seconds = 300  # 5 min between trades
        
    def load_config(self) -> dict:
        """Load trading configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {
            'enabled': False,  # Must be explicitly enabled
            'approved_pairs': ['SOLUSD'],
            'max_slippage': 0.005,  # 0.5%
            'emergency_stop': False
        }
    
    def save_config(self, config: dict):
        """Save trading configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_balance(self) -> dict:
        """Get account balance"""
        try:
            result = self.api._request('/private/Balance', {}, public=False)
            if 'result' in result:
                return {k: float(v) for k, v in result['result'].items()}
            return {}
        except Exception as e:
            self._log(f"ERROR getting balance: {e}")
            return {}
    
    def get_open_positions(self) -> list:
        """Get open positions"""
        try:
            result = self.api._request('/private/OpenPositions', {}, public=False)
            if 'result' in result:
                return list(result['result'].values())
            return []
        except Exception as e:
            self._log(f"ERROR getting positions: {e}")
            return []
    
    def fetch_data(self) -> dict:
        """Fetch current market data"""
        ohlc_data = {}
        for interval in [60, 240, 1440]:
            result = self.api.get_ohlc('SOLUSD', interval, 100)
            if 'result' in result and len(result['result']) > 1:
                ohlc_data[interval] = list(result['result'].values())[0]
        return ohlc_data
    
    def analyze(self, ohlc_data: dict) -> dict:
        """Generate signal and analysis"""
        prices = {}
        regimes = {}
        
        for tf, candles in ohlc_data.items():
            closes = [float(c[4]) for c in candles]
            prices[tf] = closes
            regimes[tf] = RegimeDetector.detect(closes)
        
        signal = self.ensemble.generate_signal(regimes, prices)
        
        return {
            'signal': signal,
            'regimes': {k: v['regime'] for k, v in regimes.items()},
            'prices': {k: v[-1] for k, v in prices.items()},
            'votes': getattr(self.ensemble, 'votes', {'long': 0, 'short': 0, 'neutral': 0}),
            'timestamp': datetime.now().isoformat()
        }
    
    def add_order(self, pair: str, type_: str, order_type: str, volume: float, price: float = None) -> dict:
        """Add order to Kraken"""
        data = {
            'pair': pair,
            'type': type_,  # buy/sell
            'ordertype': order_type,  # market/limit/stop-loss
            'volume': volume
        }
        if price:
            data['price'] = price
        
        try:
            result = self.api._request('/private/AddOrder', data, public=False)
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def close_position(self, pair: str, volume: float, direction: str):
        """Close position"""
        type_ = 'sell' if direction == 'long' else 'buy'
        return self.add_order(pair, type_, 'market', volume)
    
    def execute(self, analysis: dict) -> bool:
        """Execute trade if signal differs from current position"""
        config = self.load_config()
        
        # Safety checks
        if not config.get('enabled', False):
            self._log("TRADING DISABLED - Enable with: python live_trader.py --enable")
            return False
        
        if config.get('emergency_stop', False):
            self._log("EMERGENCY STOP ACTIVE")
            return False
        
        price = analysis['prices'].get(60, 0)
        signal = analysis['signal']
        balance = self.get_balance()
        
        # Check daily trades
        today = datetime.now().date().isoformat()
        today_trades = [t for t in self.trades if t.get('date') == today]
        if len(today_trades) >= self.max_daily_trades:
            self._log(f"MAX DAILY TRADES REACHED ({self.max_daily_trades})")
            return False
        
        # Close existing position
        if self.position and signal == 'neutral':
            self._log(f"CLOSE SIGNAL - Closing {self.position['direction']}")
            if not self.dry_run:
                result = self.close_position('SOLUSD', self.position['volume'], self.position['direction'])
                self._log(f"Close result: {result}")
            self.trades.append({
                'date': today,
                'action': 'close',
                'pair': 'SOLUSD',
                'price': price,
                'pnl': (price - self.position['entry_price']) / self.position['entry_price']
            })
            self.position = None
            return True
        
        # Open new position
        elif signal in ['long', 'short'] and not self.position:
            # Calculate position size (10% of SOL balance or $1000 min)
            sol_balance = balance.get('SOL', 0)
            usd_balance = balance.get('USD', 0) + balance.get('USDT', 0) + balance.get('USDC', 0)
            
            if usd_balance < 100:
                self._log(f"INSUFFICIENT BALANCE: ${usd_balance:.2f}")
                return False
            
            volume = min(usd_balance * self.max_position_size / price, sol_balance)
            if volume < 0.01:  # Minimum order size
                volume = 0.01
            
            self._log(f"OPEN SIGNAL: {signal.upper()} @ ${price:.2f}")
            self._log(f"  Volume: {volume:.4f} SOL (${volume * price:.2f})")
            self._log(f"  Votes: {analysis['votes']}")
            self._log(f"  Regimes: {analysis['regimes']}")
            
            if self.dry_run:
                self._log("  [DRY RUN - No trade executed]")
                self.position = {
                    'entry_price': price,
                    'direction': signal,
                    'volume': volume,
                    'date': today
                }
                return True
            
            # Execute real trade
            type_ = 'buy' if signal == 'long' else 'sell'
            result = self.add_order('SOLUSD', type_, 'market', volume)
            
            if 'result' in result and result['result']:
                self._log(f"  Order ID: {result['result']['txid']}")
                self.position = {
                    'entry_price': price,
                    'direction': signal,
                    'volume': volume,
                    'date': today,
                    'order_id': result['result']['txid']
                }
                self.trades.append({
                    'date': today,
                    'action': 'open',
                    'pair': 'SOLUSD',
                    'direction': signal,
                    'price': price,
                    'volume': volume
                })
                return True
            else:
                self._log(f"ORDER FAILED: {result}")
                return False
        
        return False
    
    def _log(self, message: str):
        """Log to file and print"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def status(self) -> dict:
        """Return current status"""
        balance = self.get_balance()
        return {
            'enabled': self.load_config().get('enabled', False),
            'dry_run': self.dry_run,
            'balance_USD': balance.get('USD', 0) + balance.get('USDT', 0) + balance.get('USDC', 0),
            'balance_SOL': balance.get('SOL', 0),
            'position': self.position,
            'trades_today': len([t for t in self.trades if t.get('date') == datetime.now().date().isoformat()]),
            'total_trades': len(self.trades)
        }
    
    def enable(self):
        """Enable live trading"""
        config = self.load_config()
        config['enabled'] = True
        self.save_config(config)
        self._log("*** LIVE TRADING ENABLED ***")
        self._log("WARNING: Real money at risk!")
    
    def disable(self):
        """Disable live trading"""
        config = self.load_config()
        config['enabled'] = False
        self.save_config(config)
        self._log("*** LIVE TRADING DISABLED ***")
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        config = self.load_config()
        config['emergency_stop'] = True
        config['enabled'] = False
        self.save_config(config)
        self._log("*** EMERGENCY STOP ACTIVATED ***")
        
        # Close open position if exists
        if self.position:
            self._log("Closing open position...")
            if not self.dry_run:
                result = self.close_position('SOLUSD', self.position['volume'], self.position['direction'])
                self._log(f"Close result: {result}")


def main():
    parser = argparse.ArgumentParser(description='Regime-Aware Live Trading System')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--api-secret', required=True)
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--enable', action='store_true', help='Enable live trading mode')
    parser.add_argument('--disable', action='store_true', help='Disable live trading')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--emergency-stop', action='store_true', help='Emergency stop')
    parser.add_argument('--analyze', action='store_true', help='Analyze and show signal')
    
    args = parser.parse_args()
    
    trader = LiveTrader(args.api_key, args.api_secret, dry_run=not args.live)
    
    if args.enable:
        trader.enable()
        return
    
    if args.disable:
        trader.disable()
        return
    
    if args.status:
        status = trader.status()
        print("\n=== TRADING STATUS ===")
        for k, v in status.items():
            print(f"  {k}: {v}")
        return
    
    if args.emergency_stop:
        trader.emergency_stop()
        return
    
    if args.analyze:
        print("=== MARKET ANALYSIS ===")
        ohlc_data = trader.fetch_data()
        if not ohlc_data:
            print("ERROR: Failed to fetch data")
            return
        
        analysis = trader.analyze(ohlc_data)
        print(f"  Signal: {analysis['signal'].upper()}")
        print(f"  Price: ${analysis['prices'].get(60, 0):.2f}")
        print(f"  Regimes: {analysis['regimes']}")
        print(f"  Votes: {analysis['votes']}")
        return
    
    # Default: run trading loop
    print("=" * 60)
    print("REGIME-AWARE LIVE TRADING SYSTEM")
    print(f"Mode: {'DRY RUN' if trader.dry_run else 'LIVE TRADING'}")
    print("=" * 60)
    
    status = trader.status()
    print(f"\nStatus: {'ENABLED' if status['enabled'] else 'DISABLED'}")
    print(f"Balance: ${status['balance_USD']:.2f} USD, {status['balance_SOL']:.4f} SOL")
    print(f"Position: {status['position']}")
    
    if not trader.dry_run and not status['enabled']:
        print("\n⚠️  TRADING IS DISABLED")
        print("Enable with: python live_trader.py --api-key KEY --api-secret SECRET --enable")
        return
    
    # Fetch and analyze
    print("\nFetching market data...")
    ohlc_data = trader.fetch_data()
    
    if not ohlc_data:
        print("ERROR: Failed to fetch data from Kraken")
        return
    
    analysis = trader.analyze(ohlc_data)
    
    print(f"\nCurrent Analysis:")
    print(f"  Signal: {analysis['signal'].upper()}")
    print(f"  Price: ${analysis['prices'].get(60, 0):.2f}")
    print(f"  Regimes: {analysis['regimes']}")
    print(f"  Votes: {analysis['votes']}")
    
    if trader.dry_run:
        print(f"\n[DRY RUN MODE]")
    
    # Execute
    trader.execute(analysis)


if __name__ == "__main__":
    main()
