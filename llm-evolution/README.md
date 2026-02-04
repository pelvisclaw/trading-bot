# LLM Trading Agent Evolution System

Evolving multi-agent trading system using LLM-generated strategies.

## Quick Start

```bash
cd /home/nueki/.openclaw/workspace/trading-bot/llm-evolution

# Run evolution
python3 evolution_engine.py --evolve

# Check status
python3 evolution_engine.py --status

# Mutate specific agent
python3 evolution_engine.py --mutate --agent momentum_v1
```

## Architecture

```
Market Analyzer LLM → Trading Manager LLM → Risk Manager LLM → Strategy Profiles
                        (selects best profile)   (validates risk)
                                      │
                                      ▼
                         Evolution Engine (improves prompts)
```

## Agents

| Agent | Role | Status |
|-------|------|--------|
| momentum_v1 | Momentum trader | ✅ Active |
| mean_reversion_v1 | Contrarian trader | ✅ Active |
| volatility_v1 | Volatility trader | ✅ Active |
| breakout_v1 | Breakout trader | ✅ Active |
| regime_aware_v1 | Regime-adaptive | ✅ Active |
| trend_follower_v1 | Pure trend follower | ✅ Active |

## Evolution

Each generation:
1. Backtest all profiles
2. Score by fitness (Sharpe + win rate - DD)
3. Remove bottom 50%
4. Breed top 50% with mutations
5. Repeat

## Files

- `evolution_engine.py` - Main framework
- `HIERARCHY.md` - Agent prompts and structure
- `population.json` - Current state
- `generations/` - Historical snapshots

## Using with Codex

```bash
# Generate new strategy from profile
codex "Create a momentum trading strategy using the momentum_v1 profile. 
       Generate Python code with entry/exit rules based on EMA crossovers.
       Output only the strategy class."

# Evolve prompt
codex "Mutate this trading prompt to be more conservative: {prompt}"
```
