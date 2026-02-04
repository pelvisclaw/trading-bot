#!/usr/bin/env python3
"""
Evolution Engine v2 - Integrates backtest results
Loads agents from individual JSON files
"""
import json
import random
from datetime import datetime
from pathlib import Path


class EvolutionEngine:
    def __init__(self, base_dir: str = "/home/nueki/.openclaw/workspace/trading-bot/llm-evolution"):
        self.base_dir = Path(base_dir)
        self.agents_dir = self.base_dir / "agents"
        self.generations_dir = self.base_dir / "generations"
        self.population_file = self.base_dir / "population.json"
        self.backtest_file = self.base_dir / "backtest_results.json"
        
        self.population = {}
        self.generation_count = 0
    
    def load_population(self):
        """Load population from individual agent files"""
        self.population = {}
        max_gen = 0
        
        for agent_file in self.agents_dir.glob("*.json"):
            with open(agent_file) as f:
                data = json.load(f)
                name = data.get('name', agent_file.stem)
                self.population[name] = data
                max_gen = max(max_gen, data.get('generation', 0))
        
        self.generation_count = max_gen
        return self.population
    
    def load_backtest_results(self) -> dict:
        if self.backtest_file.exists():
            with open(self.backtest_file) as f:
                return json.load(f)
        return {}
    
    def calculate_fitness(self, results: dict) -> dict:
        """Calculate fitness from backtest results"""
        fitness = {}
        for name, metrics in results.items():
            if 'error' not in metrics and metrics['trades'] > 0:
                fitness[name] = (
                    metrics.get('sharpe', 0) * 100 +
                    metrics.get('win_rate', 0) * 50 -
                    metrics.get('max_dd', 0) * 100
                )
            else:
                fitness[name] = -999
        return fitness
    
    def evolve(self, survival_rate: float = 0.5):
        """Run one generation of evolution"""
        self.load_population()
        results = self.load_backtest_results()
        fitness = self.calculate_fitness(results)
        
        # Update fitness
        for name in self.population:
            self.population[name]['fitness'] = fitness.get(name, -999)
            self.population[name]['backtest'] = results.get(name, {})
        
        # Rank by fitness
        ranked = sorted(
            self.population.items(),
            key=lambda x: x[1].get('fitness', -999),
            reverse=True
        )
        
        print(f"\n=== Generation {self.generation_count + 1} ===")
        print("\nRanked Population:")
        for i, (name, agent) in enumerate(ranked):
            fit = agent.get('fitness', -999)
            bt = agent.get('backtest', {})
            trades = bt.get('trades', 0)
            win_rate = bt.get('win_rate', 0)
            sharpe = bt.get('sharpe', 0)
            print(f"  {i+1}. {name}: fitness={fit:.1f}, trades={trades}, win={win_rate:.1%}, sharpe={sharpe:.2f}")
        
        # Survival selection
        survivors_count = max(2, int(len(ranked) * survival_rate))
        survivors = ranked[:survivors_count]
        
        self.generation_count += 1
        next_gen = {}
        
        for name, agent in survivors:
            agent['generation'] = self.generation_count
            next_gen[name] = agent
        
        # Breed new agents
        needed = len(ranked) - survivors_count
        for i in range(needed):
            parent1_name, parent1 = random.choice(survivors)
            parent2_name, parent2 = random.choice(survivors)
            
            child_name = f"{parent1_name[:8]}_g{self.generation_count}_{i}"
            
            child_prompt = self.crossover_prompts(
                parent1.get('system_prompt', ''),
                parent2.get('system_prompt', '')
            )
            
            if random.random() < 0.3:
                child_prompt = self.mutate_prompt(child_prompt)
            
            child = {
                'name': child_name,
                'role': parent1.get('role', 'trader'),
                'system_prompt': child_prompt,
                'parent': parent1_name,
                'generation': self.generation_count,
                'fitness': 0,
                'backtest': {},
                'created_at': datetime.now().isoformat()
            }
            next_gen[child_name] = child
            
            # Save child to file
            with open(self.agents_dir / f"{child_name}.json", 'w') as f:
                json.dump(child, f, indent=2)
        
        self.population = next_gen
        
        # Update survivors files
        for name, agent in next_gen.items():
            with open(self.agents_dir / f"{name}.json", 'w') as f:
                json.dump(agent, f, indent=2)
        
        # Save generation snapshot
        snapshot = {
            'generation': self.generation_count,
            'timestamp': datetime.now().isoformat(),
            'agents': [
                {'name': name, 'fitness': a.get('fitness', 0), 'backtest': a.get('backtest', {})}
                for name, a in next_gen.items()
            ]
        }
        with open(self.generations_dir / f"gen_{self.generation_count}.json", 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"\n✓ Evolution complete: {len(next_gen)} agents")
        print(f"  Survivors: {len(survivors)}")
        print(f"  New breeds: {needed}")
        
        return self.generation_count
    
    def crossover_prompts(self, p1: str, p2: str) -> str:
        import re
        s1 = re.split(r'(?<=[.!?])\s+', p1)
        s2 = re.split(r'(?<=[.!?])\s+', p2)
        if len(s1) >= 2 and len(s2) >= 2:
            return " ".join(s1[:len(s1)//2] + s2[len(s2)//2:])
        return p1
    
    def mutate_prompt(self, prompt: str) -> str:
        mutations = [
            " Be more conservative.",
            " Focus on longer timeframes.",
            " Prioritize win rate.",
            " Require stronger signals.",
            " Reduce position size.",
            " Skip volatile periods.",
        ]
        return prompt + random.choice(mutations)


def main():
    print("=== Evolution Engine v2 ===\n")
    
    engine = EvolutionEngine()
    engine.load_population()
    print(f"Loaded {len(engine.population)} agents")
    
    engine.evolve(survival_rate=0.5)


if __name__ == "__main__":
    main()
