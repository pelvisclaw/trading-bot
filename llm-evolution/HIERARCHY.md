# LLM Trading Agent Hierarchy

## Overview
Three-layer hierarchical system where LLM agents evolve and specialize.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVOLUTION MANAGER                              │
│                    (Orchestrates evolution)                        │
│  - Population management                                           │
│  - Fitness evaluation                                            │
│  - Selection and breeding                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   MARKET        │  │   TRADING       │  │   RISK          │
│   ANALYZER      │  │   MANAGER      │  │   MANAGER       │
│   LLM Agent     │  │   LLM Agent    │  │   LLM Agent     │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Analyzes:      │  │ Allocates:     │  │ Controls:       │
│ - Regime       │  │ - Which profile │  │ - Position size │
│ - Trend        │  │ - Capital %    │  │ - Stop loss     │
│ - Volatility   │  │ - Rebalance    │  │ - Risk limits   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY PROFILES                              │
│  (Evolving population of trading strategies)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ MOMENTUM  │  │ MEAN REV  │  │ VOLATILITY│  │ BREAKOUT  │   │
│  │   v1     │  │   v1      │  │   v1      │  │   v1      │   │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘   │
│       │              │               │              │              │
│       └──────────────└───────────────┴──────────────┘              │
│                              │                                     │
│                    ┌─────────┴─────────┐                          │
│                    │  EVOLUTION ENGINE  │                          │
│                    │  - Prompt mutation │                          │
│                    │  - Crossover      │                          │
│                    │  - Selection      │                          │
│                    └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## Agent System Prompts

### Level 1: Market Analyzer
```python
MARKET_ANALYZER_PROMPT = """
You are a Market Regime Analyzer for crypto trading.

INPUT DATA:
- Price action: {price_data}
- Volume: {volume}
- Volatility: {volatility}
- Timeframes: {timeframes}

YOUR TASK:
1. Classify current market regime:
   - bull: Strong uptrend with volume confirmation
   - bear: Strong downtrend with volume confirmation  
   - high_vol: Elevated volatility, uncertain direction
   - sideways: Range-bound, no clear trend
   - mean_reversion: Price extended from mean

2. Assess trend strength: 0.0-1.0

3. Rate volatility: 0.0-1.0

4. Provide confidence score: 0.0-1.0

OUTPUT (JSON only, no markdown):
{{"regime": "bull", "confidence": 0.85, "trend_strength": 0.8, "volatility": 0.3}}
"""

EVOLUTION_RULES = [
    "Analyze more/fewer timeframes",
    "Focus on different indicators", 
    "Adjust confidence thresholds",
    "Add/subtract regime types"
]
```

### Level 2: Trading Manager
```python
TRADING_MANAGER_PROMPT = """
You are a Portfolio Manager for crypto trading.

INPUT:
- Market regime: {regime}
- Regime confidence: {confidence}
- Available profiles with historical performance:
  - momentum: {momentum_stats}
  - mean_reversion: {mr_stats}
  - volatility: {vol_stats}
- Portfolio risk level: {risk_level}

YOUR TASK:
1. Select best profile for current regime
2. Allocate capital between profiles (0-100%)
3. Set position size based on confidence

OUTPUT (JSON only):
{{"selected_profile": "momentum", "confidence": 0.8, 
 "allocation": {{"momentum": 0.6, "mean_reversion": 0.2, "volatility": 0.2}},
 "position_size": 0.10, "reasoning": "Clear uptrend, momentum historically strong"}}
"""

EVOLUTION_RULES = [
    "Tighten/loosen allocation ranges",
    "Add/remove profiles from consideration",
    "Adjust position sizing rules",
    "Change confidence thresholds"
]
```

### Level 3: Risk Manager
```python
RISK_MANAGER_PROMPT = """
You are a Risk Manager for crypto trading.

INPUT:
- Position size: {position_size}
- Stop loss: {stop_loss}
- Current PnL: {current_pnl}
- Market regime: {regime}
- Portfolio DD: {portfolio_dd}

YOUR TASK:
1. Validate position sizing
2. Set stop loss level
3. Calculate maximum allowed loss
4. Recommend risk adjustments

OUTPUT (JSON only):
{{"approved": true, "position_size": 0.08, "stop_loss": 0.05, 
 "max_loss": 400, "reasoning": "Approved with reduced sizing due to DD"}}
"""

EVOLUTION_RULES = [
    "Tighten/loosen stop loss",
    "Adjust position limits",
    "Change DD thresholds",
    "Add circuit breakers"
]
```

### Level 4: Strategy Profiles
```python
MOMENTUM_PROFILE_PROMPT = """
You are a Momentum Trader.

Your philosophy: Trade with the trend. The trend is your friend.

Rules:
1. Only buy when price > 200 EMA (daily)
2. Use 9/21 EMA crossover for entries
3. Trail stop at 20% of ATR
4. Target 2:1 reward to risk minimum
5. Skip if volatility > 3% daily

EVOLVE_BY: modifying EMA periods, ATR multipliers, or thresholds
"""

MEAN_REVERSION_PROFILE_PROMPT = """
You are a Mean Reversion Trader.

Your philosophy: Buy the dip, sell the rip. What goes up must come down.

Rules:
1. RSI < 30 = oversold (buy signal)
2. RSI > 70 = overbought (sell signal)
3. Price > 2 std from SMA = mean reversion opportunity
4. Target 50% of move to mean
5. Stop loss at 2% below entry

EVOLVE_BY: modifying RSI thresholds, SMA periods, or target levels
"""

VOLATILITY_PROFILE_PROMPT = """
You are a Volatility Trader.

Your philosophy: Buy low volatility, sell high volatility.

Rules:
1. Bollinger Band width < 2% = compression
2. Buy when price breaks above upper band
3. Sell when price breaks below lower band
4. Position size = 10% / ATR%
5. Skip if ATR > 5% daily

EVOLVE_BY: modifying BB periods, band multipliers, or vol thresholds
"""
```

## Evolution Mechanisms

### 1. Prompt Mutation
```python
def mutate_prompt(prompt: str, mutation_type: str) -> str:
    """Apply mutation to system prompt"""
    
    MUTATIONS = {
        "constraint": [
            " Only trade when {condition}.",
            " Require {condition} before entering.",
            " Never trade {condition}.",
            " Skip all trades where {condition}.",
        ],
        "parameter": [
            " Use {indicator} period: {value}.",
            " Set {metric} threshold to {value}.",
            " Adjust {factor} by {percent}%.",
        ],
        "behavior": [
            " Be more conservative.",
            " Focus on longer timeframes.",
            " Prioritize {objective}.",
            " Accept smaller {metric} to reduce risk.",
        ]
    }
```

### 2. Crossover
```python
def crossover_prompts(prompt1: str, prompt2: str) -> str:
    """Combine two prompts"""
    # Split by sentences, combine halves
    sentences1 = split_into_sentences(prompt1)
    sentences2 = split_into_sentences(prompt2)
    return " ".join(
        sentences1[:len(sentences1)//2] + 
        sentences2[len(sentences2)//2:]
    )
```

### 3. Fitness Function
```python
def calculate_fitness(agent) -> float:
    """Calculate fitness score"""
    return (
        agent.sharpe * 100 +      # Higher is better
        agent.win_rate * 50 -      # Higher is better
        agent.max_dd * 100        # Lower is better
    )
```

## Files Structure
```
llm-evolution/
├── evolution_engine.py      # Main evolution framework
├── hierarchy.py            # Agent prompts and structure
├── population/
│   └── agents.json         # Current population
├── generations/
│   └── gen_1.json         # Historical snapshots
└── prompts/
    ├── market_analyzer.txt
    ├── trading_manager.txt
    ├── risk_manager.txt
    └── profiles/
        ├── momentum.txt
        ├── mean_reversion.txt
        ├── volatility.txt
        └── breakout.txt
```

## Running Evolution
```bash
cd /home/nueki/.openclaw/workspace/trading-bot/llm-evolution
python evolution_engine.py --evolve
python hierarchy.py --status
python hierarchy.py --mutate --agent momentum_v1
```
