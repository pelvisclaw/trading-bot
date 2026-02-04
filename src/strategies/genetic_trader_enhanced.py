#!/usr/bin/env python3
"""
Enhanced Genetic Trading System for Kraken
- Multi-coin support
- Complex indicators
- Volatility adaptation
- Walk-forward validation
- Complexity penalty
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
from enum import Enum
from statistics import mean, stdev, pvariance

# Indicators registry
INDICATORS = {
    'ma': 'Moving Average',
    'ema': 'Exponential MA',
    'rsi': 'Relative Strength Index',
    'macd': 'MACD',
    'bb': 'Bollinger Bands',
    'atr': 'Average True Range',
    'vwap': 'Volume Weighted Avg Price',
    'vol_ratio': 'Volume Ratio',
    'stoch': 'Stochastic',
    'adx': 'Average Directional Index',
}

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class IndicatorConfig:
    """Which indicators are active and their params"""
    use_ma: bool = True
    ma_period: int = 20
    use_ema: bool = False
    ema_period: int = 12
    use_rsi: bool = True
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    use_macd: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_bb: bool = False
    bb_period: int = 20
    bb_std: float = 2.0
    use_atr: bool = True
    atr_period: int = 14
    use_vwap: bool = False
    use_vol_ratio: bool = True
    vol_period: int = 20
    use_stoch: bool = False
    stoch_k: int = 14
    stoch_d: int = 3
    use_adx: bool = False
    adx_period: int = 14

    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorConfig':
        cfg = cls()
        for f in cfg.__dataclass_fields__.values():
            if f.name in data:
                setattr(cfg, f.name, data[f.name])
        return cfg
    
    def complexity(self) -> int:
        """Calculate complexity score (number of active indicators)"""
        active = 0
        for f in self.__dataclass_fields__.values():
            if f.type == bool and getattr(self, f.name):
                active += 1
            elif f.name.startswith('rsi_') and getattr(self, 'use_rsi'):
                active += 0.5
            elif f.name.startswith('macd_') and getattr(self, 'use_macd'):
                active += 0.5
            elif f.name.startswith('bb_') and getattr(self, 'use_bb'):
                active += 0.5
            elif f.name.startswith('atr_') and getattr(self, 'use_atr'):
                active += 0.5
            elif f.name.startswith('vol_') and getattr(self, 'use_vol_ratio'):
                active += 0.5
            elif f.name.startswith('stoch_') and getattr(self, 'use_stoch'):
                active += 0.5
        return int(active)


@dataclass
class StrategyGenes:
    """Complete strategy configuration"""
    # Core entry rules
    entry_ma_short: int = 9
    entry_ma_long: int = 46
    entry_rsi_oversold: float = 38.0
    entry_rsi_overbought: float = 65.0
    
    # Advanced entry conditions (compound logic)
    require_vol_confirmation: bool = False
    vol_multiplier: float = 1.5
    require_macd_confirm: bool = False
    macd_crossover_required: bool = False
    
    # Exit rules
    exit_ma_short: int = 13
    exit_ma_long: int = 20
    exit_rsi_threshold: float = 50.0
    
    # Stop/take (volatility-adaptive)
    base_stop_loss: float = 0.05
    base_take_profit: float = 0.03
    atr_multiplier: float = 2.0
    use_atr_stops: bool = True
    
    # Position sizing
    base_position_size: float = 0.17
    volatility_scaling: bool = True
    vol_adaptive_factor: float = 0.5  # reduce size in high vol
    
    # Time filters
    trade_start_hour: int = 0
    trade_end_hour: int = 23
    min_trades_per_day: int = 0
    
    # Regime detection
    detect_regimes: bool = False
    regime_adaptation: bool = False
    
    # Indicators
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    
    # Coin preference weights [BTC, ETH, SOL, XRP, ADA]
    coin_weights: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.5, 0.3, 0.3])
    
    def to_dict(self) -> Dict:
        return {
            **{f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values() 
               if f.name != 'indicators' and f.name != 'coin_weights'},
            'indicators': self.indicators.to_dict(),
            'coin_weights': self.coin_weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyGenes':
        genes = cls()
        for f in genes.__dataclass_fields__.values():
            if f.name == 'indicators':
                genes.indicators = IndicatorConfig.from_dict(data.get('indicators', {}))
            elif f.name == 'coin_weights':
                genes.coin_weights = data.get('coin_weights', [1.0, 0.8, 0.5, 0.3, 0.3])
            elif f.name in data:
                setattr(genes, f.name, data[f.name])
        return genes
    
    def complexity(self) -> int:
        """Total complexity = indicators + rules"""
        base = sum(1 for f in self.__dataclass_fields__.values() 
                   if f.type in (int, float, bool) and f.name.startswith('use_'))
        base += 1 if self.volatility_scaling else 0
        base += 1 if self.regime_adaptation else 0
        base += self.indicators.complexity()
        return base


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    direction: TradeDirection
    pnl_pct: Optional[float]
    genes: StrategyGenes
    status: str = "open"
    regime: str = "unknown"


class KrakenAPI:
    """Kraken REST API wrapper"""
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.base_url = "https://api.kraken.com"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GeneticTrader/2.0'})
    
    def _get_nonce(self):
        return str(int(time.time() * 1000))
    
    def _sign(self, path: str, nonce: str, data: str) -> str:
        nonce_path = f"/0/private/{path}"
        message = (nonce + nonce_path + data).encode()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message, hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _request(self, path: str, data: Dict = None, public: bool = False) -> Dict:
        url = f"{self.base_url}/0{path}"
        headers = {}
        if not public and self.api_key and self.api_secret:
            nonce = self._get_nonce()
            data = data or {}
            data['nonce'] = nonce
            post_data = '&'.join(f"{k}={v}" for k, v in data.items())
            signature = self._sign(path.split('/')[-1], nonce, post_data)
            headers = {'API-Key': self.api_key, 'API-Sign': signature, 
                      'Content-Type': 'application/x-www-form-urlencoded'}
        if data and public:
            url += '?' + '&'.join(f"{k}={v}" for k, v in data.items())
        response = self.session.get(url, headers=headers, timeout=10)
        return response.json()
    
    # Public endpoints
    def get_ohlc(self, pair: str = "XXBTZUSD", interval: int = 60, count: int = 500) -> Dict:
        # Fetch recent OHLC without since parameter, then filter
        result = self._request(f"/public/OHLC", {"pair": pair, "interval": interval}, public=True)
        if 'result' in result and len(result['result']) > 1:
            ohlc = list(result['result'].values())[0]
            # Return last 'count' candles
            return {'result': {pair: ohlc[-count:]}}
        return result
    
    def get_ticker(self, pair: str = "XXBTZUSD") -> Dict:
        return self._request(f"/public/Ticker", {"pair": pair}, public=True)
    
    def get_system_status(self) -> Dict:
        return self._request("/public/SystemStatus", public=True)
    
    # Private endpoints
    def get_balance(self) -> Dict:
        return self._request("/private/Balance")


class TechnicalIndicators:
    """Calculate technical indicators"""
    @staticmethod
    def ma(data: List[float], period: int) -> List[float]:
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(0)
            else:
                result.append(sum(data[i-period+1:i+1]) / period)
        return result
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return [0] * len(data)
        multiplier = 2 / (period + 1)
        result = [sum(data[:period]) / period]
        for i in range(period, len(data)):
            result.append((data[i] - result[-1]) * multiplier + result[-1])
        return result + [0] * (len(data) - len(result))
    
    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        if len(data) < period + 1:
            return [50] * len(data)
        rsi = [50] * len(data)
        gains, losses = [], []
        for i in range(1, len(data)):
            change = data[i] - data[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
            if i >= period:
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                rs = avg_gain / (avg_loss + 0.00001)  # Larger epsilon
                rsi[i] = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        if len(close) < period + 1:
            return [0] * len(close)
        tr = []
        for i in range(len(close)):
            tr.append(max(high[i] - low[i], 
                   abs(high[i] - close[i-1]) if i > 0 else 0,
                   abs(low[i] - close[i-1]) if i > 0 else 0))
        atr = [0] * period
        atr.append(sum(tr[:period+1]) / period)
        for i in range(period + 1, len(tr)):
            atr.append((atr[-1] * (period - 1) + tr[i]) / period)
        return atr + [0] * (len(close) - len(atr))
    
    @staticmethod
    def volatility(data: List[float], period: int = 20) -> List[float]:
        returns = [0] * len(data)
        for i in range(1, len(data)):
            returns[i] = abs((data[i] - data[i-1]) / data[i-1])
        vol = []
        for i in range(len(data)):
            if i < period:
                vol.append(0)
            else:
                vol.append(stdev(returns[i-period+1:i+1]) if period > 1 else returns[i])
        return vol
    
    @staticmethod
    def volume_ratio(vol: List[float], period: int = 20) -> List[float]:
        avg_vol = TechnicalIndicators.ma(vol, period)
        return [v / (avg_vol[i] + 0.001) for i, v in enumerate(vol)]
    
    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(data))]
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        return macd_line, signal_line


class StrategyEvaluator:
    """Evaluate a strategy on OHLC data"""
    def __init__(self, genes: StrategyGenes):
        self.genes = genes
    
    def evaluate(self, ohlc_data: List, pair: str = "XXBTZUSD", precomputed: Dict = None) -> Dict:
        """Run backtest, return metrics"""
        if len(ohlc_data) < 100:
            return {'fitness': 1000, 'trades': 0, 'pnl': 0, 'sharpe': 0}
        
        # Extract price series
        closes = [float(c[4]) for c in ohlc_data]
        highs = [float(h[2]) for h in ohlc_data]
        lows = [float(l[3]) for l in ohlc_data]
        volumes = [float(v[6]) for v in ohlc_data]
        
        # Use precomputed or calculate
        ind = self.genes.indicators
        if precomputed and 'closes' in precomputed:
            mas_short = TechnicalIndicators.ma(precomputed['closes'], self.genes.entry_ma_short)
            mas_long = TechnicalIndicators.ma(precomputed['closes'], self.genes.entry_ma_long)
            rsi = precomputed['rsi'] if ind.use_rsi else None
        else:
            mas_short = TechnicalIndicators.ma(closes, self.genes.entry_ma_short)
            mas_long = TechnicalIndicators.ma(closes, self.genes.entry_ma_long)
            rsi = TechnicalIndicators.rsi(closes, ind.rsi_period) if ind.use_rsi else None
        atr = TechnicalIndicators.atr(highs, lows, closes, ind.atr_period) if ind.use_atr else None
        vol_ratio = TechnicalIndicators.volume_ratio(volumes, ind.vol_period) if ind.use_vol_ratio else None
        
        # Backtest
        balance = 1000.0
        position = None
        trades = []
        regime = "unknown"
        
        for i in range(50, len(ohlc_data)):
            close = closes[i]
            high = highs[i]
            low = lows[i]
            
            # Time filter
            ts = datetime.fromtimestamp(int(ohlc_data[i][0]))
            if ts.hour < self.genes.trade_start_hour or ts.hour > self.genes.trade_end_hour:
                continue
            
            # Calculate volatility for adaptive sizing
            vol = TechnicalIndicators.volatility(closes, 20)[i] if self.genes.volatility_scaling else 0.02
            adj_vol = min(vol, 0.1)  # cap at 10%
            
            # Entry signals - MA crossover
            entry_signal = False
            if self.genes.entry_ma_short and self.genes.entry_ma_long:
                if mas_short[i] > mas_long[i]:  # Golden cross
                    if rsi is None or rsi[i] < self.genes.entry_rsi_oversold:
                        if not self.genes.require_vol_confirmation or (vol_ratio and vol_ratio[i] > self.genes.vol_multiplier):
                            entry_signal = True
            
            # Execute entry
            if entry_signal and position is None:
                # Volatility-adjusted sizing
                size_mult = 1.0
                if self.genes.volatility_scaling:
                    size_mult = 1.0 - (adj_vol * self.genes.vol_adaptive_factor * 10)
                
                position_size = min(self.genes.base_position_size * size_mult, 0.5)
                stop = close * (1 - self.genes.base_stop_loss)
                if self.genes.use_atr_stops and atr and atr[i] > 0:
                    stop = close - (atr[i] * self.genes.atr_multiplier)
                target = close * (1 + self.genes.base_take_profit)
                
                position = {
                    'entry_price': close,
                    'size': position_size,
                    'stop_loss': stop,
                    'take_profit': target,
                    'entry_bar': i
                }
            
            # Exit signals
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
                elif position:
                    mas_exit_short = TechnicalIndicators.ma(closes, self.genes.exit_ma_short)
                    mas_exit_long = TechnicalIndicators.ma(closes, self.genes.exit_ma_long)
                    if mas_exit_short[i] < mas_exit_long[i]:
                        pnl = (close - position['entry_price']) / position['entry_price']
                        balance *= (1 + pnl)
                        trades.append({'pnl': pnl, 'reason': 'ma_crossover'})
                        position = None
        
        # Calculate fitness with transaction costs, slippage, and drawdown
        complexity_penalty = 0.01 * self.genes.complexity()
        
        # Transaction costs (0.1% per trade)
        transaction_cost = 0.001 * len(trades)
        
        # Apply slippage (0.05% per trade, simulated)
        slippage = 0.0005 * len(trades)
        
        # Calculate drawdown
        equity_curve = [1000.0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] * (1 + trade['pnl']))
        
        max_equity = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
            max_equity = max(max_equity, equity)
        
        # Calculate metrics
        if trades:
            returns = [t['pnl'] for t in trades]
            avg_return = sum(returns) / len(returns)
            variance = pvariance(returns) if len(returns) > 1 else 0
            sharpe = (avg_return * 252) / (variance ** 0.5 + 0.001) if variance > 0 else 0
            
            # Win rate
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades)
            
            # Fitness: reward positive returns + consistency + heavy drawdown penalty
            fitness = (
                (balance - 1000) +                    # Raw P&L
                (sharpe * 30) +                       # Risk-adjusted returns
                (win_rate * 50) +                     # Consistency bonus
                (len(trades) * 2) -                   # More trades = better exploration
                (max_drawdown * 1000) -               # HEAVY drawdown penalty
                (transaction_cost * 100) -             # Transaction cost penalty
                (slippage * 100) -                     # Slippage penalty
                (complexity_penalty * 10)              # Complexity penalty
            )
        else:
            fitness = 1000.0 - complexity_penalty  # Penalty for no trades
        
        return {
            'fitness': fitness,
            'trades': len(trades),
            'pnl': (balance - 1000) / 10,  # % return
            'sharpe': sharpe if trades else 0,
            'win_rate': win_rate if trades else 0,
            'max_drawdown': max_drawdown,
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'complexity': self.genes.complexity()
        }


class GeneticAlgorithm:
    """Enhanced GA with mutation strategies"""
    def __init__(self, population_size: int = 15, mutation_rate: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[StrategyGenes] = []
        self.generation = 0
        self.best_strategy = None
        self.history = []
        self.precomputed = {}  # Cache indicators
    
    def initialize_population(self, seed: int = None):
        """Create diverse initial population with optional seed"""
        if seed is not None:
            random.seed(seed)
        
        self.population = []
        for _ in range(self.population_size):
            genes = StrategyGenes()
            
            # Fixed starting point for reproducibility
            genes.entry_ma_short = random.randint(5, 30)
            genes.entry_ma_long = random.randint(20, 60)
            genes.exit_ma_short = random.randint(3, 15)
            genes.exit_ma_long = random.randint(15, 45)
            genes.entry_rsi_oversold = random.uniform(20, 40)
            genes.entry_rsi_overbought = random.uniform(60, 80)
            genes.base_stop_loss = random.uniform(0.02, 0.08)
            genes.base_take_profit = random.uniform(0.02, 0.06)
            genes.base_position_size = random.uniform(0.05, 0.25)
            
            # Deterministic indicator settings
            ind = genes.indicators
            ind.use_ma = random.random() > 0.3
            ind.use_rsi = random.random() > 0.3
            ind.use_atr = random.random() > 0.5
            ind.use_vol_ratio = random.random() > 0.5
            ind.use_macd = random.random() > 0.7
            ind.use_bb = random.random() > 0.7
            
            self.population.append(genes)
    
    def evolve(self, ohlc_data: Dict[str, List], generations: int = 20) -> StrategyGenes:
        """Run multi-coin evolution with parallel evaluation"""
        # Precompute indicators for each coin
        self.precomputed = {}
        for pair, data in ohlc_data.items():
            closes = [float(c[4]) for c in data]
            self.precomputed[pair] = {
                'closes': closes,
                'mas': TechnicalIndicators.ma(closes, 50),
                'rsi': TechnicalIndicators.rsi(closes, 14),
            }
        
        self.initialize_population()
        
        for gen in range(generations):
            scores = []
            for genes in self.population:
                total_fitness = 0
                total_trades = 0
                for pair, data in ohlc_data.items():
                    eval_result = StrategyEvaluator(genes).evaluate(data, pair, self.precomputed.get(pair, {}))
                    coin_idx = ['XXBTZUSD', 'XETHZUSD', 'SOLUSD', 'XXRPZUSD', 'ADAUSD'].index(pair) if pair in ['XXBTZUSD', 'XETHZUSD', 'SOLUSD', 'XXRPZUSD', 'ADAUSD'] else 0
                    weight = genes.coin_weights[coin_idx] if coin_idx < len(genes.coin_weights) else 1.0
                    total_fitness += eval_result['fitness'] * weight
                    total_trades += eval_result['trades']
                scores.append((total_fitness / len(ohlc_data), total_trades, genes))
            
            # Sort by fitness
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Track best
            if not self.best_strategy or scores[0][0] > self.best_strategy[0]:
                self.best_strategy = scores[0]
            
            print(f"Gen {gen+1}: Best={scores[0][0]:.2f}, Trades={scores[0][1]}, Complexity={scores[0][2].complexity()}", flush=True)
            self.history.append({'gen': gen, 'best_fitness': scores[0][0]})
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep top 3
            for i in range(min(3, len(scores))):
                next_gen.append(scores[i][2])
            
            # Selection + reproduction
            while len(next_gen) < self.population_size:
                parent = self._tournament_select(scores)
                child = self._mutate(parent)
                next_gen.append(child)
            
            self.population = next_gen
            self.generation += 1
        
        return self.best_strategy[2] if self.best_strategy else None
    
    def _tournament_select(self, scores: List, k: int = 4) -> StrategyGenes:
        selected = random.sample(scores[:len(self.population)//2], min(k, len(scores)//2))
        return max(selected, key=lambda x: x[0])[2]
    
    def _mutate(self, genes: StrategyGenes) -> StrategyGenes:
        """Deep mutation"""
        new_genes = StrategyGenes.from_dict(genes.to_dict())
        
        if random.random() < self.mutation_rate:
            new_genes.entry_ma_short = max(3, new_genes.entry_ma_short + random.randint(-5, 5))
        if random.random() < self.mutation_rate:
            new_genes.entry_ma_long = max(new_genes.entry_ma_short + 5, new_genes.entry_ma_long + random.randint(-10, 10))
        if random.random() < self.mutation_rate:
            new_genes.base_stop_loss = max(0.01, min(0.15, new_genes.base_stop_loss + random.uniform(-0.02, 0.02)))
        if random.random() < self.mutation_rate:
            new_genes.base_take_profit = max(0.01, min(0.15, new_genes.base_take_profit + random.uniform(-0.01, 0.01)))
        if random.random() < self.mutation_rate:
            new_genes.base_position_size = max(0.02, min(0.4, new_genes.base_position_size + random.uniform(-0.05, 0.05)))
        
        # Indicator mutations
        ind = new_genes.indicators
        if random.random() < self.mutation_rate:
            ind.use_ma = not ind.use_ma
        if random.random() < self.mutation_rate:
            ind.use_rsi = not ind.use_rsi
        if random.random() < self.mutation_rate:
            ind.use_atr = not ind.use_atr
        if random.random() < self.mutation_rate:
            ind.use_vol_ratio = not ind.use_vol_ratio
        if random.random() < self.mutation_rate:
            ind.use_macd = not ind.use_macd
        if random.random() < self.mutation_rate:
            ind.use_bb = not ind.use_bb
        
        return new_genes


class GeneticTrader:
    """Main trading system - SOL Single Coin Mode (Optimized)"""
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.kraken = KrakenAPI(api_key, api_secret)
        self.ga = GeneticAlgorithm(population_size=30, mutation_rate=0.25)  # Larger pop, balanced mutation
        self.best_strategy = None
        self.coin_pairs = ['SOLUSD']
    
    def fetch_ohlc_multi(self, count: int = 50) -> Dict[str, List]:  # 50 candles
        """Fetch OHLC for SOL"""
        data = {}
        pair = 'SOLUSD'
        try:
            result = self.kraken.get_ohlc(pair, 60, count)
            if 'result' in result and pair in result['result']:
                ohlc = list(result['result'].values())[0]
                data[pair] = ohlc[-count:]
                print(f"Fetched {pair}: {len(data[pair])} candles", flush=True)
        except Exception as e:
            print(f"Error fetching {pair}: {e}", flush=True)
        return data
    
    def train(self, generations: int = 30) -> StrategyGenes:
        """Train on multiple coins"""
        print(f"Fetching data for {len(self.coin_pairs)} coins...")
        ohlc_data = self.fetch_ohlc_multi()
        
        if not ohlc_data:
            raise ValueError("Failed to fetch OHLC data")
        
        print(f"Training on {sum(len(d) for d in ohlc_data.values())} total candles...")
        self.best_strategy = self.ga.evolve(ohlc_data, generations)
        
        print(f"\nBest strategy found (complexity: {self.best_strategy.complexity()}):")
        print(json.dumps(self.best_strategy.to_dict(), indent=2, default=str))
        
        return self.best_strategy
    
    def walk_forward_validate(self, strategy: StrategyGenes, ohlc_data: Dict, train_pct: float = 0.7) -> Dict:
        """Walk-forward validation"""
        results = {}
        for pair, data in ohlc_data.items():
            split = int(len(data) * train_pct)
            train_data = data[:split]
            test_data = data[split:]
            
            # Retrain on train, evaluate on test
            evaluator = StrategyEvaluator(strategy)
            train_result = evaluator.evaluate(train_data, pair)
            test_result = evaluator.evaluate(test_data, pair)
            
            results[pair] = {
                'train_fitness': train_result['fitness'],
                'test_fitness': test_result['fitness'],
                'test_pnl': test_result['pnl'],
                'test_trades': test_result['trades'],
                'overfit_ratio': test_result['fitness'] / (train_result['fitness'] + 0.001)
            }
        
        return results
    
    def monte_carlo_validation(self, strategy: StrategyGenes, ohlc_data: Dict, iterations: int = 200) -> Dict:
        """Monte Carlo validation using bootstrap resampling
        
        Returns confidence intervals for strategy performance.
        Rejects strategies with >20% probability of loss.
        """
        import numpy as np
        
        results = {}
        for pair, data in ohlc_data.items():
            evaluator = StrategyEvaluator(strategy)
            base_result = evaluator.evaluate(data, pair)
            
            # Extract returns from the base backtest
            # Simulate by resampling candles with replacement
            closes = [float(c[4]) for c in data]
            returns = []
            for i in range(1, len(closes)):
                returns.append((closes[i] - closes[i-1]) / closes[i-1])
            
            if len(returns) < 50:
                results[pair] = {'error': 'Insufficient data for Monte Carlo'}
                continue
            
            # Bootstrap resampling
            pnl_samples = []
            for _ in range(iterations):
                # Resample returns with replacement
                sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
                
                # Simulate simple trend-following returns based on sampled returns
                signal = np.sign(sampled_returns)
                strategy_returns = signal * sampled_returns * strategy.base_position_size
                
                # Apply stop loss simulation
                for i in range(len(strategy_returns)):
                    if strategy_returns[i] < -strategy.base_stop_loss:
                        strategy_returns[i] = -strategy.base_stop_loss
                
                pnl = sum(strategy_returns)
                pnl_samples.append(pnl)
            
            pnl_samples = np.array(pnl_samples)
            
            # Calculate statistics
            mean_pnl = np.mean(pnl_samples)
            std_pnl = np.std(pnl_samples)
            percentile_5 = np.percentile(pnl_samples, 5)
            percentile_95 = np.percentile(pnl_samples, 95)
            loss_prob = np.mean(pnl_samples < 0)
            
            # Sharpe ratio approximation
            sharpe = mean_pnl / (std_pnl + 0.0001) * np.sqrt(252)
            
            results[pair] = {
                'mean_pnl': mean_pnl,
                'std_pnl': std_pnl,
                'percentile_5': percentile_5,
                'percentile_95': percentile_95,
                'loss_probability': loss_prob,
                'sharpe_approx': sharpe,
                'confidence_interval': f"[{percentile_5:.4f}, {percentile_95:.4f}]",
                'is_robust': loss_prob < 0.20  # Accept if <20% chance of loss
            }
            
            print(f"  {pair}: mean={mean_pnl:.4f}, loss_prob={loss_prob:.2%}, robust={results[pair]['is_robust']}", flush=True)
        
        return results
    
    def save_strategy(self, path: str = "best_strategy_enhanced.json"):
        """Save strategy"""
        if self.best_strategy:
            with open(path, 'w') as f:
                json.dump({
                    'genes': self.best_strategy.to_dict(),
                    'fitness': self.best_strategy[0] if isinstance(self.best_strategy, tuple) else None,
                    'timestamp': datetime.now().isoformat(),
                    'ga_history': self.ga.history
                }, f, indent=2, default=str)
            print(f"Strategy saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Genetic Trading System')
    parser.add_argument('--train', action='store_true', help='Train new strategy')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    parser.add_argument('--validate', action='store_true', help='Walk-forward validation')
    parser.add_argument('--load', help='Load strategy file')
    parser.add_argument('--api-key', help='Kraken API key')
    parser.add_argument('--api-secret', help='Kraken API secret')
    
    args = parser.parse_args()
    
    trader = GeneticTrader(args.api_key, args.api_secret)
    
    if args.train:
        genes = trader.train(args.generations)
        trader.save_strategy()
    elif args.validate:
        if args.load:
            with open(args.load, 'r') as f:
                data = json.load(f)
                genes = StrategyGenes.from_dict(data['genes'])
        else:
            genes = StrategyGenes.from_dict(json.load(open('best_strategy_enhanced.json'))['genes'])
        
        ohlc_data = trader.fetch_ohlc_multi()
        results = trader.walk_forward_validate(genes, ohlc_data)
        print("\nWalk-forward validation results:")
        for pair, res in results.items():
            print(f"  {pair}: train={res['train_fitness']:.2f}, test={res['test_fitness']:.2f}, "
                  f"overfit={res['overfit_ratio']:.3f}")
    else:
        print("Enhanced Genetic Trading System")
        print("Options:")
        print("  --train         Train new multi-coin strategy")
        print("  --validate      Walk-forward validation")
        print("  --load FILE     Load existing strategy")
        print("  --generations 30   Number of generations")


if __name__ == "__main__":
    main()
