#!/usr/bin/env python3
"""
LLM Trading Agent Evolution Framework
Hierarchical multi-agent system with evolving prompts
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random


# ============ HIERARCHY STRUCTURE ============

@dataclass
class AgentConfig:
    """Configuration for an LLM agent"""
    name: str
    role: str
    system_prompt: str
    temperature: float = 0.7
    model: str = "gpt-4"
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    generation: int = 0
    fitness: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMEvolutionEngine:
    """Manages hierarchy of evolving LLM trading agents"""
    
    def __init__(self, base_dir: str = "/home/nueki/.openclaw/workspace/trading-bot/llm-evolution"):
        self.base_dir = Path(base_dir)
        self.agents_dir = self.base_dir / "agents"
        self.prompts_dir = self.base_dir / "prompts"
        self.generations_dir = self.base_dir / "generations"
        self.population_file = self.base_dir / "population.json"
        
        # Create directories
        self.agents_dir.mkdir(exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)
        self.generations_dir.mkdir(exist_ok=True)
        
        self.population: Dict[str, AgentConfig] = {}
        self.generation_count = 0
    
    # ============ PROMPT MUTATIONS ============
    
    def mutate_prompt(self, prompt: str, mutation_type: str = "random") -> str:
        """Apply mutation to a system prompt"""
        mutations = {
            "add_constraint": [
                " Only trade when volatility is below {threshold}%.",
                " Require confirmation from two timeframes.",
                " Never trade against the daily trend.",
                " Reduce position size by 50% in uncertain regimes.",
                " Only enter trades with >2:1 reward to risk.",
                " Wait for price to retest support before buying.",
            ],
            "modify_behavior": [
                " Be more conservative with entries.",
                " Focus on longer timeframes.",
                " Prioritize win rate over total return.",
                " Accept smaller profits to reduce risk.",
                " Be more patient with trade entries.",
            ],
            "add_filter": [
                " Skip all trades during high volatility periods.",
                " Avoid trading during low volume hours.",
                " Only trade major pairs (BTC, ETH, SOL).",
                " Require volume confirmation > 1.5x average.",
            ]
        }
        
        if mutation_type == "random":
            category = random.choice(list(mutations.keys()))
            mutation = random.choice(mutations[category])
            return prompt + mutation
        
        elif mutation_type == "parameter":
            # Mutate numeric thresholds
            import re
            numbers = re.findall(r'\d+', prompt)
            if numbers:
                num = random.choice(numbers)
                new_num = str(int(num) + random.randint(-10, 10))
                return prompt.replace(num, new_num, 1)
        
        return prompt
    
    def crossover_prompts(self, prompt1: str, prompt2: str) -> str:
        """Combine two prompts"""
        # Split by sentences and combine
        import re
        sentences1 = re.split(r'(?<=[.!?])\s+', prompt1)
        sentences2 = re.split(r'(?<=[.!?])\s+', prompt2)
        
        # Take first half from one, second from other
        if random.random() < 0.5:
            return " ".join(sentences1[:len(sentences1)//2] + sentences2[len(sentences2)//2:])
        return " ".join(sentences2[:len(sentences2)//2] + sentences1[len(sentences1)//2:])
    
    # ============ POPULATION MANAGEMENT ============
    
    def create_agent(self, config: AgentConfig):
        """Add agent to population"""
        self.population[config.name] = config
        self._save_agent(config)
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        return self.population.get(name)
    
    def evolve_population(self, survival_rate: float = 0.5):
        """Run one generation of evolution"""
        self.generation_count += 1
        
        # Sort by fitness
        ranked = sorted(
            self.population.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        
        # Kill bottom performers
        survivors_count = int(len(ranked) * survival_rate)
        survivors = ranked[:survivors_count]
        
        # Create next generation
        next_gen = {}
        for agent in survivors:
            agent.generation = self.generation_count
            next_gen[agent.name] = agent
        
        # Breed top performers
        for i in range(len(ranked) - survivors_count):
            parent1, parent2 = random.sample(survivors, 2)
            
            # Create child
            child_name = f"{parent1.name}_gen{self.generation_count}_{i}"
            child_prompt = self.crossover_prompts(parent1.system_prompt, parent2.system_prompt)
            
            # Occasionally mutate
            if random.random() < 0.3:
                child_prompt = self.mutate_prompt(child_prompt)
            
            child = AgentConfig(
                name=child_name,
                role=parent1.role,
                system_prompt=child_prompt,
                parent=parent1.name,
                generation=self.generation_count,
                fitness=0.0  # Reset fitness
            )
            next_gen[child_name] = child
        
        self.population = next_gen
        self._save_population()
        self._save_generation_snapshot()
        
        return self.generation_count
    
    # ============ FITNESS UPDATE ============
    
    def update_fitness(self, name: str, metrics: Dict):
        """Update agent fitness from backtest results"""
        if name in self.population:
            agent = self.population[name]
            agent.trade_count = metrics.get('trades', 0)
            agent.win_rate = metrics.get('win_rate', 0)
            agent.sharpe = metrics.get('sharpe', 0)
            agent.max_dd = metrics.get('max_dd', 0)
            
            # Fitness formula: reward high Sharpe, penalize high DD
            agent.fitness = (
                agent.sharpe * 100 + 
                agent.win_rate * 50 - 
                agent.max_dd * 100
            )
    
    # ============ PERSISTENCE ============
    
    def _save_agent(self, agent: AgentConfig):
        """Save agent to disk"""
        agent_data = {
            'name': agent.name,
            'role': agent.role,
            'system_prompt': agent.system_prompt,
            'temperature': agent.temperature,
            'model': agent.model,
            'parent': agent.parent,
            'children': agent.children,
            'generation': agent.generation,
            'fitness': agent.fitness,
            'trade_count': agent.trade_count,
            'win_rate': agent.win_rate,
            'sharpe': agent.sharpe,
            'max_dd': agent.max_dd,
            'created_at': agent.created_at
        }
        
        with open(self.agents_dir / f"{agent.name}.json", 'w') as f:
            json.dump(agent_data, f, indent=2)
    
    def _save_population(self):
        """Save full population state"""
        with open(self.population_file, 'w') as f:
            json.dump({
                'generation': self.generation_count,
                'population': {
                    name: {
                        'name': a.name,
                        'role': a.role,
                        'system_prompt': a.system_prompt,
                        'fitness': a.fitness,
                        'generation': a.generation
                    }
                    for name, a in self.population.items()
                }
            }, f, indent=2)
    
    def _save_generation_snapshot(self):
        """Save snapshot of current generation"""
        snapshot = {
            'generation': self.generation_count,
            'timestamp': datetime.now().isoformat(),
            'agents': [
                {
                    'name': a.name,
                    'role': a.role,
                    'fitness': a.fitness,
                    'win_rate': a.win_rate,
                    'sharpe': a.sharpe,
                    'max_dd': a.max_dd,
                    'trades': a.trade_count
                }
                for a in self.population.values()
            ]
        }
        
        with open(self.generations_dir / f"gen_{self.generation_count}.json", 'w') as f:
            json.dump(snapshot, f, indent=2)
    
    def load_population(self):
        """Load existing population"""
        if self.population_file.exists():
            with open(self.population_file) as f:
                data = json.load(f)
                self.generation_count = data.get('generation', 0)


# ============ INITIAL POPULATION ============

def create_initial_population(engine: LLMEvolutionEngine):
    """Create starting population of trading agents"""
    
    agents = [
        AgentConfig(
            name="momentum_v1",
            role="momentum_trader",
            system_prompt="""You are a momentum trader. 
Your philosophy: "The trend is your friend."
Trade with the trend. Buy breakouts, sell breakdowns.
Use moving averages to identify trend direction.
Enter on pullbacks to key moving averages.
Trail stops to lock in profits.
Never fight the trend."""
        ),
        AgentConfig(
            name="mean_reversion_v1",
            role="mean_reversion_trader",
            system_prompt="""You are a contrarian trader.
Your philosophy: "What goes up must come down."
Buy when others are fearful. Sell when others are greedy.
Use RSI to identify oversold/overbought conditions.
Trade reversals at support/resistance levels.
Small profits, high win rate focus.
Patience is your greatest weapon."""
        ),
        AgentConfig(
            name="volatility_v1",
            role="volatility_trader",
            system_prompt="""You are a volatility trader.
Your philosophy: "Buy low volatility, sell high volatility."
Trade Bollinger Band breakouts and squeezes.
Use ATR to measure volatility expansion.
Enter during compression, exit during expansion.
Position size inversely proportional to volatility.
High volatility = reduced exposure."""
        ),
        AgentConfig(
            name="breakout_v1",
            role="breakout_trader",
            system_prompt="""You are a breakout trader.
Your philosophy: "When in doubt, break out."
Identify consolidation patterns and ranges.
Enter when price breaks above resistance or below support.
Tight stops, let winners run.
Volume confirmation required.
Target 2:1 minimum reward to risk."""
        ),
        AgentConfig(
            name="regime_aware_v1",
            role="regime_adaptive_trader",
            system_prompt="""You are a regime-adaptive trader.
Your philosophy: "Adapt or die."
First identify market regime (bull/bear/sideways).
Bull = momentum strategy.
Bear = mean reversion or sit out.
Sideways = range trading.
High volatility = reduce position size.
Always respect the current regime."""
        ),
        AgentConfig(
            name="trend_follower_v1",
            role="trend_follower",
            system_prompt="""You are a pure trend follower.
Your philosophy: "The trend is your only friend."
Only trade in direction of 200-day moving average.
Use multiple timeframes: trend on daily, entries on 4h.
Enter when trend aligns across timeframes.
Stop loss below recent swing low/high.
Never short in uptrend, never long in downtrend."""
        )
    ]
    
    for agent in agents:
        engine.create_agent(agent)
    
    return agents


def main():
    """Initialize the evolution framework"""
    engine = LLMEvolutionEngine()
    
    print("=== LLM Trading Agent Evolution Framework ===")
    print(f"Base directory: {engine.base_dir}")
    print(f"Agents directory: {engine.agents_dir}")
    
    # Check for existing population
    if engine.population_file.exists():
        engine.load_population()
        print(f"Loaded existing population: {len(engine.population)} agents")
    else:
        print("Creating initial population...")
        create_initial_population(engine)
        print(f"Created {len(engine.population)} agents")
    
    print(f"\nCurrent generation: {engine.generation_count}")
    print("\nPopulation:")
    for name, agent in engine.population.items():
        print(f"  - {name} (gen {agent.generation}, fitness: {agent.fitness:.2f})")


if __name__ == "__main__":
    main()
