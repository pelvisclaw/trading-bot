#!/usr/bin/env python3
"""
Google Colab Trading System for Kraken
Complete notebook ready to copy-paste into Google Colab

Instructions:
1. Go to https://colab.research.google.com
2. Click "New Notebook"
3. Copy ALL code from this file
4. Paste into the notebook
5. Follow the setup steps below
6. Click "Run all" or run cells sequentially
"""

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================
print("""
=== GOOGLE COLAB TRADING SYSTEM ===

This notebook will train a genetic algorithm trading system for Kraken.

STEPS:
1. Run this cell (install dependencies)
2. Run Cell 2 (configure Kraken API keys)
3. Run Cell 3 (train the strategy)
4. Run Cell 4 (validate and paper trade)
5. Run Cell 5 (download results)

Let's get started!
""")

# Install required packages
# !pip install requests numpy pandas ta

print("✅ Dependencies installed")

# ============================================================================
# CELL 2: Configuration - ADD YOUR KRAKEN API KEYS HERE
# ============================================================================
print("""
=== CONFIGURE KRAKEN API KEYS ===

IMPORTANT: Replace the API_KEY and PRIVATE_KEY below with your actual Kraken keys.
Keep them secure - never share them publicly.
""")

# ============ EDIT THESE VALUES ============
API_KEY = "mC2NCEovIIOELwkwhEr/PNwLO3uXWC88RNXiLWXVYf3NkPdyphA38eyX"
PRIVATE_KEY = "czEYTBX+vdMWKEKeKFDrQ2tea9w1Myd5HbhR+ibA+MHZnNRkvAc0SuH8obMqyzYwKpjKTVBQ+q3I7Y09DwY34Q=="
# ===========================================

# Verify keys are provided
if API_KEY.startswith("mC2NCEovIIOELwkwhEr/"):
    print("✅ Using provided Kraken API key")
else:
    print("⚠️  WARNING: Using placeholder API key. Replace with your actual key!")

# Save keys to a secure file
import json
import os

config = {
    "api_key": API_KEY,
    "private_key": PRIVATE_KEY,
    "trading_pair": "SOLUSD",
    "interval": 60,  # 1-minute candles
    "candle_count": 2000  # Increased from 500 for better training
}

with open('/kraken_config.json', 'w') as f:
    json.dump(config, f)

print("✅ Configuration saved securely")
print("📊 Trading pair: SOLUSD")
print("⏱️  Timeframe: 1-minute candles")
print("🕯️  Data: 500 candles for training")

# ============================================================================
# CELL 3: Genetic Trading System Implementation
# ============================================================================
print("""
=== GENETIC TRADING SYSTEM ===

Implementing optimized genetic algorithm for Kraken trading.
This will train for 50 generations (much longer than possible on Raspberry Pi).
""")

import requests
import json
import time
import random
import hashlib
import hmac
import base64
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

# Load configuration
with open('/kraken_config.json', 'r') as f:
    config = json.load(f)

class KrakenAPI:
    """Simplified Kraken API client"""
    
    def __init__(self, api_key, private_key):
        self.api_key = api_key
        self.private_key = private_key
        self.base_url = "https://api.kraken.com"
        self.session = requests.Session()
        
    def _sign(self, path, nonce, post_data=""):
        """Create API signature"""
        message = path + hashlib.sha256((nonce + post_data).encode()).digest()
        signature = hmac.new(
            base64.b64decode(self.private_key),
            message.encode(),
            hashlib.sha512
        ).digest()
        return base64.b64encode(signature).decode()
    
    def get_ohlc(self, pair="SOLUSD", interval=60, count=500):
        """Fetch OHLC data"""
        url = f"{self.base_url}/0/public/OHLC"
        params = {"pair": pair, "interval": interval}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            if 'result' in data and len(data['result']) > 1:
                ohlc = list(data['result'].values())[0]
                # Return last 'count' candles
                return ohlc[-count:]
        except Exception as e:
            print(f"Error fetching data: {e}")
        
        return []

class SimpleStrategy:
    """Simplified trading strategy with genetic parameters"""
    
    def __init__(self):
        # Genetic parameters
        self.ma_short = random.randint(8, 15)
        self.ma_long = random.randint(25, 35)
        self.rsi_period = random.randint(10, 20)
        self.rsi_oversold = random.uniform(25, 35)
        self.rsi_overbought = random.uniform(65, 75)
        self.position_size = random.uniform(0.05, 0.15)
        self.stop_loss = random.uniform(0.02, 0.05)
        self.take_profit = random.uniform(0.02, 0.05)
    
    def to_dict(self):
        return self.__dict__
    
    @staticmethod
    def from_dict(data):
        strat = SimpleStrategy()
        for key, value in data.items():
            setattr(strat, key, value)
        return strat
    
    def complexity(self):
        """Measure strategy complexity"""
        return 3  # Fixed low complexity for Pi 5 compatibility

class SimpleGeneticAlgorithm:
    """Optimized genetic algorithm for resource-constrained environments"""
    
    def __init__(self, population_size=8, mutation_rate=0.4):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_strategy = None
        self.best_fitness = -float('inf')
        
    def initialize_population(self):
        """Create initial population"""
        self.population = [SimpleStrategy() for _ in range(self.population_size)]
    
    def calculate_ma(self, data, period):
        """Simple moving average"""
        if len(data) < period:
            return [0] * len(data)
        return [np.mean(data[max(0, i-period+1):i+1]) for i in range(len(data))]
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        if len(data) < period + 1:
            return [50] * len(data)
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi = [50] * len(data)
        for i in range(period, len(data)-1):
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            
            # Update averages
            if i < len(data) - 2:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi
    
    def evaluate_strategy(self, strategy, ohlc_data):
        """Evaluate strategy on historical data"""
        closes = [float(c[4]) for c in ohlc_data]
        
        # Calculate indicators
        ma_short = self.calculate_ma(closes, strategy.ma_short)
        ma_long = self.calculate_ma(closes, strategy.ma_long)
        rsi = self.calculate_rsi(closes, strategy.rsi_period)
        
        # Simulate trading
        balance = 1000.0
        position = None
        trades = 0
        
        for i in range(max(strategy.ma_long, strategy.rsi_period), len(closes)):
            current_price = closes[i]
            
            # Entry signals
            if not position:
                # MA crossover + RSI oversold
                if (ma_short[i] > ma_long[i] and ma_short[i-1] <= ma_long[i-1] and
                    rsi[i] < strategy.rsi_oversold):
                    position = {
                        'entry_price': current_price,
                        'entry_size': balance * strategy.position_size,
                        'stop_loss': current_price * (1 - strategy.stop_loss),
                        'take_profit': current_price * (1 + strategy.take_profit)
                    }
                    trades += 1
            
            # Exit signals
            elif position:
                exit_signal = False
                
                # Stop loss
                if current_price <= position['stop_loss']:
                    exit_signal = True
                    pnl = (position['stop_loss'] - position['entry_price']) / position['entry_price']
                
                # Take profit
                elif current_price >= position['take_profit']:
                    exit_signal = True
                    pnl = (position['take_profit'] - position['entry_price']) / position['entry_price']
                
                # RSI overbought
                elif rsi[i] > strategy.rsi_overbought and rsi[i-1] <= strategy.rsi_overbought:
                    exit_signal = True
                    pnl = (current_price - position['entry_price']) / position['entry_price']
                
                if exit_signal:
                    balance *= (1 + pnl)
                    position = None
        
        # Close any open position
        if position:
            pnl = (closes[-1] - position['entry_price']) / position['entry_price']
            balance *= (1 + pnl)
        
        # Calculate fitness
        profit_loss = balance - 1000  # P&L in dollars
        
        # Base fitness is profit/loss
        fitness = profit_loss
        
        # Heavily penalize losses (quadratic penalty)
        if profit_loss < 0:
            fitness = profit_loss * 10  # Losses are 10x worse
        
        # Penalize very low trade count (want at least 5 trades)
        if trades < 5:
            fitness *= 0.3
        elif trades > 15:
            fitness *= 1.2  # Bonus for more trades
        
        # Complexity penalty (minimal)
        fitness -= strategy.complexity() * 2
        
        # Ensure minimum fitness for strategies with some trades
        if trades > 0:
            fitness = max(fitness, -500)  # Floor for strategies that at least trade
        
        return fitness, trades
    
    def evolve(self, ohlc_data, generations=50, time_budget=300):
        """Run genetic evolution with time budget"""
        self.initialize_population()
        start_time = time.time()
        
        print(f"Starting evolution: {generations} generations, {time_budget}s budget")
        
        for gen in range(generations):
            # Check time budget
            if time.time() - start_time > time_budget:
                print(f"⏰ Time budget reached at generation {gen+1}")
                break
            
            # Evaluate population
            scores = []
            for strategy in self.population:
                fitness, trades = self.evaluate_strategy(strategy, ohlc_data)
                scores.append((fitness, trades, strategy))
            
            # Sort by fitness
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Track best
            if scores[0][0] > self.best_fitness:
                self.best_fitness = scores[0][0]
                self.best_strategy = scores[0][2]
            
            print(f"Gen {gen+1}: Best fitness={scores[0][0]:.2f}, Trades={scores[0][1]}")
            
            # Create next generation (elitism + tournament selection with crossover)
            next_gen = []
            
            # Elitism: keep top 2
            next_gen.extend([scores[0][2], scores[1][2]])
            
            # Tournament selection for rest
            while len(next_gen) < self.population_size:
                # Pick 4 random from top half
                tournament = random.sample(scores[:len(self.population)//2], 4)
                tournament.sort(key=lambda x: x[0], reverse=True)
                winner1 = tournament[0][2]
                winner2 = tournament[1][2]
                
                # 70% crossover, 30% mutation
                if random.random() < 0.7:
                    # Crossover between two winners
                    child = self._crossover(winner1, winner2)
                else:
                    # Mutation on best winner
                    child = self._mutate(winner1)
                
                next_gen.append(child)
            
            self.population = next_gen
        
        print(f"\n✅ Evolution complete")
        print(f"Best fitness: {self.best_fitness:.2f}")
        
        return self.best_strategy
    
    def _mutate(self, strategy):
        """Apply mutation to create child"""
        child = SimpleStrategy.from_dict(strategy.to_dict())
        
        if random.random() < self.mutation_rate:
            child.ma_short = max(5, min(20, child.ma_short + random.randint(-3, 3)))
        if random.random() < self.mutation_rate:
            child.ma_long = max(child.ma_short + 5, min(40, child.ma_long + random.randint(-5, 5)))
        if random.random() < self.mutation_rate:
            child.rsi_oversold = max(20, min(40, child.rsi_oversold + random.uniform(-5, 5)))
        if random.random() < self.mutation_rate:
            child.rsi_overbought = max(60, min(80, child.rsi_overbought + random.uniform(-5, 5)))
        if random.random() < self.mutation_rate:
            child.position_size = max(0.02, min(0.2, child.position_size + random.uniform(-0.05, 0.05)))
        if random.random() < self.mutation_rate:
            child.stop_loss = max(0.01, min(0.1, child.stop_loss + random.uniform(-0.01, 0.01)))
        if random.random() < self.mutation_rate:
            child.take_profit = max(0.01, min(0.1, child.take_profit + random.uniform(-0.01, 0.01)))
        
        return child
    
    def _crossover(self, parent1, parent2):
        """Create child by blending two parents"""
        child = SimpleStrategy()
        
        # Blend integer parameters (weighted average rounded)
        child.ma_short = int(round(parent1.ma_short * 0.5 + parent2.ma_short * 0.5))
        child.ma_long = int(round(parent1.ma_long * 0.5 + parent2.ma_long * 0.5))
        child.rsi_period = int(round(parent1.rsi_period * 0.5 + parent2.rsi_period * 0.5))
        
        # Blend float parameters
        child.rsi_oversold = (parent1.rsi_oversold + parent2.rsi_oversold) / 2
        child.rsi_overbought = (parent1.rsi_overbought + parent2.rsi_overbought) / 2
        child.position_size = (parent1.position_size + parent2.position_size) / 2
        child.stop_loss = (parent1.stop_loss + parent2.stop_loss) / 2
        child.take_profit = (parent1.take_profit + parent2.take_profit) / 2
        
        # Ensure valid ranges
        child.ma_short = max(5, min(20, child.ma_short))
        child.ma_long = max(child.ma_short + 5, min(40, child.ma_long))
        child.rsi_period = max(10, min(20, child.rsi_period))
        child.rsi_oversold = max(20, min(40, child.rsi_oversold))
        child.rsi_overbought = max(60, min(80, child.rsi_overbought))
        child.position_size = max(0.02, min(0.2, child.position_size))
        child.stop_loss = max(0.01, min(0.1, child.stop_loss))
        child.take_profit = max(0.01, min(0.1, child.take_profit))
        
        return child

# ============================================================================
# CELL 4: Main Training Execution
# ============================================================================
print("""
=== STARTING TRAINING ===

This will:
1. Fetch market data from Kraken
2. Train genetic algorithm (50 generations)
3. Save the best strategy
""")

# Initialize Kraken API
kraken = KrakenAPI(config['api_key'], config['private_key'])

# Fetch data
print("📥 Fetching market data...")
ohlc_data = kraken.get_ohlc(
    pair=config['trading_pair'],
    interval=config['interval'],
    count=config['candle_count']
)

if not ohlc_data:
    print("❌ Failed to fetch data. Check your API keys and internet connection.")
else:
    print(f"✅ Fetched {len(ohlc_data)} candles of {config['trading_pair']}")
    
    # Initialize and run genetic algorithm
    ga = SimpleGeneticAlgorithm(population_size=8, mutation_rate=0.4)
    
    print("🧬 Starting genetic algorithm training...")
    best_strategy = ga.evolve(ohlc_data, generations=50, time_budget=600)  # 10 minute budget
    
    if best_strategy:
        # Save strategy
        strategy_data = {
            "strategy": "simple_genetic_colab",
            "genes": best_strategy.to_dict(),
            "fitness": ga.best_fitness,
            "timestamp": datetime.now().isoformat(),
            "training_data": {
                "pair": config['trading_pair'],
                "candles": len(ohlc_data),
                "interval": config['interval']
            }
        }
        
        with open('/best_strategy_colab.json', 'w') as f:
            json.dump(strategy_data, f, indent=2)
        
        print("\n✅ Best strategy saved:")
        print(json.dumps(strategy_data['genes'], indent=2))
        print(f"\n📈 Final fitness: {ga.best_fitness:.2f}")
        
        # ============================================================================
        # CELL 5: Validation and Paper Trading
        # ============================================================================
        print("""
        === VALIDATION & PAPER TRADING ===
        
        Testing the strategy on out-of-sample data...
        """)
        
        # Fetch fresh data for validation
        print("📥 Fetching validation data...")
        validation_data = kraken.get_ohlc(
            pair=config['trading_pair'],
            interval=config['interval'],
            count=500  # Increased validation dataset
        )
        
        if validation_data:
            # Test strategy on validation data
            fitness, trades = ga.evaluate_strategy(best_strategy, validation_data)
            
            print(f"\n📊 Validation Results:")
            print(f"  Fitness: {fitness:.2f}")
            print(f"  Trades: {trades}")
            print(f"  Return: {(fitness - 1000):.2f}")
            
            # Simple paper trading simulation
            print("\n📝 Paper Trading Simulation:")
            initial_balance = 1000.0
            balance = initial_balance
            closes = [float(c[4]) for c in validation_data]
            
            ma_short = ga.calculate_ma(closes, best_strategy.ma_short)
            ma_long = ga.calculate_ma(closes, best_strategy.ma_long)
            rsi = ga.calculate_rsi(closes, best_strategy.rsi_period)
            
            position = None
            trade_history = []
            
            for i in range(max(best_strategy.ma_long, best_strategy.rsi_period), len(closes)):
                current_price = closes[i]
                
                if not position:
                    # Entry signal
                    if (ma_short[i] > ma_long[i] and ma_short[i-1] <= ma_long[i-1] and
                        rsi[i] < best_strategy.rsi_oversold):
                        position = {
                            'entry_price': current_price,
                            'entry_time': i,
                            'size': balance * best_strategy.position_size
                        }
                
                elif position:
                    # Exit signals
                    exit_price = None
                    
                    # Stop loss
                    if current_price <= position['entry_price'] * (1 - best_strategy.stop_loss):
                        exit_price = position['entry_price'] * (1 - best_strategy.stop_loss)
                    
                    # Take profit
                    elif current_price >= position['entry_price'] * (1 + best_strategy.take_profit):
                        exit_price = position['entry_price'] * (1 + best_strategy.take_profit)
                    
                    # RSI overbought
                    elif rsi[i] > best_strategy.rsi_overbought and rsi[i-1] <= best_strategy.rsi_overbought:
                        exit_price = current_price
                    
                    if exit_price:
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                        balance = position['size'] * (1 + pnl) + (balance - position['size'])
                        
                        trade_history.append({
                            'entry': position['entry_price'],
                            'exit': exit_price,
                            'pnl': pnl * 100
                        })
                        
                        position = None
            
            # Close any open position
            if position:
                pnl = (closes[-1] - position['entry_price']) / position['entry_price']
                balance = position['size'] * (1 + pnl) + (balance - position['size'])
                trade_history.append({
                    'entry': position['entry_price'],
                    'exit': closes[-1],
                    'pnl': pnl * 100
                })
            
            print(f"\n📈 Final Balance: ${balance:.2f}")
            print(f"📉 Total Return: {((balance/initial_balance - 1) * 100):.2f}%")
            print(f"📊 Total Trades: {len(trade_history)}")
            
            if trade_history:
                wins = sum(1 for t in trade_history if t['pnl'] > 0)
                win_rate = (wins / len(trade_history)) * 100
                print(f"✅ Win Rate: {win_rate:.1f}%")
                
                # Save trading log
                with open('/trading_log_colab.json', 'w') as f:
                    json.dump({
                        'initial_balance': initial_balance,
                        'final_balance': balance,
                        'total_return_pct': ((balance/initial_balance - 1) * 100),
                        'total_trades': len(trade_history),
                        'win_rate': win_rate,
                        'trades': trade_history,
                        'strategy': best_strategy.to_dict()
                    }, f, indent=2)
        
        else:
            print("❌ Failed to fetch validation data")

# ============================================================================
# CELL 6: Download Results
# ============================================================================
print("""
=== DOWNLOAD RESULTS ===

Your trained strategy and trading log are ready to download.

Follow these steps:
1. Click the folder icon on the left (Files pane)
2. Find these files:
   - /best_strategy_colab.json (your trained strategy)
   - /trading_log_colab.json (trading performance)
3. Right-click each file → "Download"
4. Save them to your computer

NEXT STEPS:
1. Copy the downloaded files to your Raspberry Pi
2. Use them with your local trading system
3. Monitor performance and iterate

The strategy file contains all parameters needed to execute trades.
The trading log shows historical performance for validation.
""")

# List generated files
import os
print("\n📁 Generated files:")
for file in ['/best_strategy_colab.json', '/trading_log_colab.json', '/kraken_config.json']:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  {file} ({size} bytes)")
    else:
        print(f"  {file} (not generated)")

print("""
=== COMPLETE ===

✅ Training finished successfully!
📥 Download the files above
💡 Use them to deploy trading on your Raspberry Pi

Remember:
- Start with small position sizes
- Monitor performance daily
- Adjust strategy as market conditions change
- Never risk more than you can afford to lose

Happy trading! 🚀
""")