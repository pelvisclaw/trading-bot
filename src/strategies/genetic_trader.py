#!/usr/bin/env python3
"""
Genetic Algorithm Trading System for Kraken
- Generates trading strategies
- Tests on historical data
- Evolves best performers
- Eliminates poor strategies
"""

import json
import time
import random
import hashlib
import hmac
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import argparse

# Strategy gene representation
@dataclass
class StrategyGenes:
    """Genes that define a trading strategy"""
    # Entry rules
    entry_ma_short: int = 5      # Short MA period
    entry_ma_long: int = 20      # Long MA period  
    entry_rsi_oversold: float = 30.0  # RSI oversold threshold
    entry_rsi_overbought: float = 70.0  # RSI overbought threshold
    entry_vol_multiplier: float = 1.5   # Volume spike multiplier
    
    # Exit rules
    exit_ma_short: int = 5
    exit_ma_long: int = 20
    exit_rsi_threshold: float = 50.0
    exit_atr_multiplier: float = 2.0   # ATR-based stop
    
    # Risk management
    position_size: float = 0.1   # 10% of capital max
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    # Time filters
    trade_start_hour: int = 0
    trade_end_hour: int = 23
    
    def mutate(self, mutation_rate=0.1):
        """Mutate genes with given probability"""
        for field in self.__dataclass_fields__:
            if random.random() < mutation_rate:
                current = getattr(self, field)
                if isinstance(current, int):
                    change = random.randint(-2, 2)
                    setattr(self, field, max(1, current + change))
                elif isinstance(current, float):
                    change = random.uniform(-0.1, 0.1)
                    setattr(self, field, max(0.01, current + change))
    
    def crossover(self, other) -> 'StrategyGenes':
        """Crossover with another strategy"""
        child = StrategyGenes()
        for field in self.__dataclass_fields__:
            if random.random() < 0.5:
                setattr(child, field, getattr(self, field))
            else:
                setattr(child, field, getattr(other, field))
        return child
    
    def to_dict(self) -> Dict:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyGenes':
        genes = cls()
        for field, value in data.items():
            if hasattr(genes, field):
                setattr(genes, field, value)
        return genes

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    direction: TradeDirection
    pnl_pct: Optional[float]
    genes: StrategyGenes
    status: str = "open"  # open, closed

class KrakenAPI:
    """Kraken REST API wrapper"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.kraken.com"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GeneticTrader/1.0'})
    
    def _get_nonce(self):
        return str(int(time.time() * 1000))
    
    def _sign(self, path: str, nonce: str, data: str) -> str:
        """Create API signature"""
        nonce_path = f"/0/private/{path}"
        message = (nonce + nonce_path + data).encode()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _request(self, path: str, data: Dict = None, public: bool = False) -> Dict:
        """Make API request"""
        url = f"{self.base_url}/0{path}"
        
        headers = {}
        if not public and self.api_key and self.api_secret:
            nonce = self._get_nonce()
            if data is None:
                data = {}
            data['nonce'] = nonce
            
            post_data = '&'.join(f"{k}={v}" for k, v in data.items())
            signature = self._sign(path.split('/')[-1], nonce, post_data)
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        
        if data and public:
            url += '?' + '&'.join(f"{k}={v}" for k, v in data.items())
        
        response = self.session.get(url, headers=headers, timeout=10)
        return response.json()
    
    # Public endpoints
    def get_ticker(self, pair: str = "XXBTZUSD") -> Dict:
        return self._request(f"/public/Ticker", {"pair": pair}, public=True)
    
    def get_ohlc(self, pair: str = "XXBTZUSD", interval: int = 60) -> Dict:
        """Get OHLC data"""
        return self._request(f"/public/OHLC", {"pair": pair, "interval": interval}, public=True)
    
    def get_trades(self, pair: str = "XXBTZUSD") -> Dict:
        return self._request(f"/public/Trades", {"pair": pair}, public=True)
    
    def get_system_status(self) -> Dict:
        return self._request("/public/SystemStatus", public=True)
    
    # Private endpoints (require auth)
    def get_balance(self) -> Dict:
        return self._request("/private/Balance")
    
    def get_trade_balance(self) -> Dict:
        return self._request("/private/TradeBalance")
    
    def get_open_positions(self) -> Dict:
        return self._request("/private/OpenPositions")
    
    def add_order(self, pair: str, type: str, ordertype: str, volume: float, 
                  price: float = None, leverage: str = "none") -> Dict:
        data = {
            "pair": pair,
            "type": type,
            "ordertype": ordertype,
            "volume": str(volume),
            "leverage": leverage
        }
        if price:
            data["price"] = str(price)
        return self._request("/private/AddOrder", data)


class TradingStrategy:
    """Trading strategy with genetic genes"""
    
    def __init__(self, genes: StrategyGenes):
        self.genes = genes
        self.trades: List[Trade] = []
        self.fitness = 0.0
    
    def evaluate(self, ohlc_data: List[Dict]) -> float:
        """Evaluate strategy on OHLC data, return fitness score"""
        balance = 1000.0  # Start with $1000
        position = None
        trades = []
        
        for i, candle in enumerate(ohlc_data):
            close = float(candle[4])  # Close price
            high = float(candle[2])
            low = float(candle[3])
            volume = float(candle[6])
            
            # Calculate indicators
            if i < 50:
                continue
            
            # Moving averages
            short_ma = self._ma(ohlc_data, i, self.genes.entry_ma_short)
            long_ma = self._ma(ohlc_data, i, self.genes.entry_ma_long)
            rsi = self._rsi(ohlc_data, i, 14)
            
            # Entry signal
            if position is None:
                if short_ma > long_ma and rsi < self.genes.entry_rsi_oversold:
                    # Buy signal
                    position = {
                        'entry_price': close,
                        'size': balance * self.genes.position_size / close,
                        'stop_loss': close * (1 - self.genes.stop_loss_pct),
                        'take_profit': close * (1 + self.genes.take_profit_pct)
                    }
            
            # Exit signal
            elif position:
                # Stop loss
                if close <= position['stop_loss']:
                    pnl = (close - position['entry_price']) / position['entry_price']
                    balance *= (1 + pnl)
                    trades.append({'pnl': pnl, 'reason': 'stop_loss'})
                    position = None
                # Take profit
                elif close >= position['take_profit']:
                    pnl = (close - position['entry_price']) / position['entry_price']
                    balance *= (1 + pnl)
                    trades.append({'pnl': pnl, 'reason': 'take_profit'})
                    position = None
                # MA crossover exit
                else:
                    short_ma_now = self._ma(ohlc_data, i, self.genes.exit_ma_short)
                    long_ma_now = self._ma(ohlc_data, i, self.genes.exit_ma_long)
                    if short_ma_now < long_ma_now and position:
                        pnl = (close - position['entry_price']) / position['entry_price']
                        balance *= (1 + pnl)
                        trades.append({'pnl': pnl, 'reason': 'ma_crossover'})
                        position = None
        
        # Calculate fitness: risk-adjusted return
        if trades:
            returns = [t['pnl'] for t in trades]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            sharpe = avg_return / (variance ** 0.5 + 0.0001) if variance > 0 else 0
            self.fitness = balance + sharpe * 100
        else:
            self.fitness = 1000.0  # No trades, no profit/loss
        
        return self.fitness
    
    def _ma(self, ohlc_data: List, i: int, period: int) -> float:
        """Calculate moving average"""
        if i < period:
            return 0
        prices = [float(ohlc_data[j][4]) for j in range(i - period, i)]
        return sum(prices) / period
    
    def _rsi(self, ohlc_data: List, i: int, period: int) -> float:
        """Calculate RSI"""
        if i < period + 1:
            return 50
        gains = 0
        losses = 0
        for j in range(i - period, i):
            change = float(ohlc_data[j+1][4]) - float(ohlc_data[j][4])
            if change > 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100
        rs = gains / losses
        return 100 - (100 / (1 + rs))


class GeneticAlgorithm:
    """Genetic algorithm for evolving trading strategies"""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[TradingStrategy] = []
        self.generation = 0
        self.best_strategy: Optional[TradingStrategy] = None
    
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            genes = StrategyGenes()
            # Randomize genes
            genes.entry_ma_short = random.randint(2, 20)
            genes.entry_ma_long = random.randint(20, 50)
            genes.entry_rsi_oversold = random.uniform(20, 40)
            genes.entry_rsi_overbought = random.uniform(60, 80)
            genes.exit_ma_short = random.randint(2, 15)
            genes.exit_ma_long = random.randint(15, 40)
            genes.position_size = random.uniform(0.05, 0.3)
            genes.stop_loss_pct = random.uniform(0.01, 0.05)
            genes.take_profit_pct = random.uniform(0.02, 0.1)
            
            self.population.append(TradingStrategy(genes))
    
    def evolve(self, ohlc_data: List[Dict], generations: int = 10):
        """Run genetic algorithm"""
        self.initialize_population()
        
        for gen in range(generations):
            # Evaluate all strategies
            for strategy in self.population:
                strategy.evaluate(ohlc_data)
            
            # Sort by fitness
            self.population.sort(key=lambda s: s.fitness, reverse=True)
            
            # Track best
            if not self.best_strategy or self.population[0].fitness > self.best_strategy.fitness:
                self.best_strategy = self.population[0]
            
            print(f"Gen {gen+1}: Best fitness={self.population[0].fitness:.2f}, "
                  f"Avg={sum(s.fitness for s in self.population)/len(self.population):.2f}")
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep top 2
            next_gen.append(self.population[0])
            next_gen.append(self.population[1])
            
            # Tournament selection + crossover
            while len(next_gen) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child_genes = parent1.genes.crossover(parent2.genes)
                child_genes.mutate(self.mutation_rate)
                next_gen.append(TradingStrategy(child_genes))
            
            self.population = next_gen
            self.generation += 1
        
        # Final evaluation
        for strategy in self.population:
            strategy.evaluate(ohlc_data)
        
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        self.best_strategy = self.population[0]
        
        return self.best_strategy
    
    def _tournament_select(self, k: int = 3) -> TradingStrategy:
        """Tournament selection"""
        selected = random.sample(self.population, min(k, len(self.population)))
        return max(selected, key=lambda s: s.fitness)


class GeneticTrader:
    """Main trading system combining Kraken API and genetic algorithms"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.kraken = KrakenAPI(api_key, api_secret)
        self.ga = GeneticAlgorithm(population_size=20, mutation_rate=0.15)
        self.best_strategy = None
    
    def fetch_ohlc(self, pair: str = "XXBTZUSD", interval: int = 60, count: int = 500) -> List[Dict]:
        """Fetch OHLC data from Kraken"""
        data = self.kraken.get_ohlc(pair, interval)
        if 'result' in data and len(data['result']) > 1:
            ohlc = list(data['result'].values())[0]
            return ohlc[-count:]  # Return last 'count' candles
        return []
    
    def train(self, pair: str = "XXBTZUSD", generations: int = 20) -> StrategyGenes:
        """Train genetic algorithm on historical data"""
        print(f"Fetching OHLC data for {pair}...")
        ohlc_data = self.fetch_ohlc(pair)
        
        if not ohlc_data:
            raise ValueError("Failed to fetch OHLC data")
        
        print(f"Training on {len(ohlc_data)} candles...")
        self.best_strategy = self.ga.evolve(ohlc_data, generations)
        
        print(f"\nBest strategy found:")
        print(json.dumps(self.best_strategy.genes.to_dict(), indent=2))
        
        return self.best_strategy.genes
    
    def save_strategy(self, path: str = "best_strategy.json"):
        """Save best strategy to file"""
        if self.best_strategy:
            with open(path, 'w') as f:
                json.dump({
                    'genes': self.best_strategy.genes.to_dict(),
                    'fitness': self.best_strategy.fitness,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"Strategy saved to {path}")
    
    def load_strategy(self, path: str = "best_strategy.json") -> Optional[StrategyGenes]:
        """Load strategy from file"""
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                genes = StrategyGenes.from_dict(data['genes'])
                print(f"Loaded strategy with fitness: {data.get('fitness', 'unknown')}")
                return genes
        return None


def main():
    parser = argparse.ArgumentParser(description='Genetic Trading System')
    parser.add_argument('--train', action='store_true', help='Train new strategy')
    parser.add_argument('--pair', default='XXBTZUSD', help='Trading pair')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--load', action='store_true', help='Load existing strategy')
    parser.add_argument('--api-key', help='Kraken API key')
    parser.add_argument('--api-secret', help='Kraken API secret')
    
    args = parser.parse_args()
    
    trader = GeneticTrader(args.api_key, args.api_secret)
    
    if args.train:
        genes = trader.train(args.pair, args.generations)
        trader.save_strategy()
    elif args.load:
        genes = trader.load_strategy()
        if genes:
            print("Ready to trade with loaded strategy")
    else:
        print("Genetic Trading System for Kraken")
        print("Options:")
        print("  --train       Train new strategy")
        print("  --load        Load existing strategy")
        print("  --pair XXBTZUSD  Trading pair (default)")
        print("  --generations 20   Number of generations")
        print("Example: python3 genetic_trader.py --train --generations 50")


if __name__ == "__main__":
    main()
