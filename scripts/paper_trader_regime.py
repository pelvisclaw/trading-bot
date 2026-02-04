#!/usr/bin/env python3
"""
Regime-Aware Paper Trader
- Runs the ensemble strategy in paper mode
- Simulates trades without real money
- Logs all signals and decisions
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, 'src')

from strategies.regime_ensemble_v2 import (
    Backtester, KrakenAPI, RegimeDetector, EnsembleTrader, StrategyConfig
)


class PaperTrader:
    """Paper trading implementation"""
    
    def __init__(self, api: KrakenAPI, initial_capital: float = 10000):
        self.api = api
        self.ensemble = EnsembleTrader()
        self.config = StrategyConfig()
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.signals = []
        self.log_file = Path("paper_trading.log")
        
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
            'timestamp': datetime.now().isoformat()
        }
    
    def execute(self, analysis: dict):
        """Execute paper trade if signal differs from current position"""
        price = analysis['prices'].get(60, 0)
        signal = analysis['signal']
        
        # Close existing position
        if self.position and signal == 'neutral':
            pnl = (price - self.position['entry_price']) / self.position['entry_price']
            if self.position['direction'] == 'short':
                pnl = -pnl
            
            self.trades.append({
                'entry_price': self.position['entry_price'],
                'exit_price': price,
                'direction': self.position['direction'],
                'pnl': pnl,
                'timestamp': datetime.now().isoformat()
            })
            
            self.capital *= (1 + pnl)
            self.position = None
            
            self._log(f"CLOSED {self.position['direction'].upper()} @ {price:.2f} | PnL: {pnl:.2%}")
        
        # Open new position
        elif signal in ['long', 'short'] and not self.position:
            self.position = {
                'entry_price': price,
                'direction': signal,
                'timestamp': datetime.now().isoformat()
            }
            self._log(f"OPENED {signal.upper()} @ {price:.2f}")
        
        # Log signal even if no action
        self.signals.append({
            'signal': signal,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log(self, message: str):
        """Log to file and print"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def status(self) -> dict:
        """Return current status"""
        return {
            'capital': self.capital,
            'position': self.position,
            'trades_count': len(self.trades),
            'open_pnl': (
                (self.capital - 10000) / 10000 
                if not self.position else None
            )
        }


def main():
    parser = argparse.ArgumentParser(description='Regime-Aware Paper Trader')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--api-secret', required=True)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--interval', type=int, default=300)  # Check every 5 min
    
    args = parser.parse_args()
    
    api = KrakenAPI(args.api_key, args.api_secret)
    trader = PaperTrader(api, args.capital)
    
    print("=" * 50)
    print("REGIME-AWARE PAPER TRADER STARTED")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Check Interval: {args.interval} seconds")
    print("=" * 50)
    
    # Initial fetch and analyze
    ohlc_data = trader.fetch_data()
    analysis = trader.analyze(ohlc_data)
    
    print(f"\\nCurrent Analysis:")
    print(f"  Signal: {analysis['signal'].upper()}")
    print(f"  Regimes: {analysis['regimes']}")
    print(f"  Price: ${analysis['prices'].get(60, 'N/A'):.2f}")
    
    # Execute
    trader.execute(analysis)
    
    # Print status
    status = trader.status()
    print(f"\\nStatus:")
    print(f"  Capital: ${status['capital']:,.2f}")
    print(f"  Position: {status['position']}")
    print(f"  Trades: {status['trades_count']}")


if __name__ == "__main__":
    main()
