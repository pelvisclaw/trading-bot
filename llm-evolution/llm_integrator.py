#!/usr/bin/env python3
"""
LLM Integration for Trading Agent Evolution
Supports: OpenAI, Anthropic, or local models
"""
import json
import os
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str  # "openai", "anthropic", "local"
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000


class LLMEvolver:
    """Evolve trading prompts using LLM"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with system + user prompt"""
        
        if self.config.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.config.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.config.provider == "local":
            return self._call_local(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _call_openai(self, system: str, user: str) -> str:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI error: {response.text}")
    
    def _call_anthropic(self, system: str, user: str) -> str:
        """Call Anthropic API"""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": f"\n\nSystem: {system}\n\nHuman: {user}"}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Anthropic error: {response.text}")
    
    def _call_local(self, system: str, user: str) -> str:
        """Call local model (Ollama, LM Studio, etc.)"""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": self.config.temperature
        }
        
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Local model error: {response.text}")


# ============ TRADING PROMPTS ============

MUTATION_PROMPT = """
You are a Prompt Engineer for a crypto trading system.

CURRENT PROMPT:
{prompt}

YOUR TASK:
1. Mutate this prompt to improve trading performance
2. Add one new constraint or rule
3. Modify one parameter or threshold
4. Keep the core philosophy intact

OUTPUT (only valid JSON):
{{"mutated_prompt": "...", "changes": ["..."]}}
"""

ANALYSIS_PROMPT = """
You are a Market Regime Analyst.

CURRENT MARKET DATA:
- Regime: {regime}
- Confidence: {confidence}%
- Volatility: {volatility}
- Trend: {trend}

HISTORICAL PROFILE PERFORMANCE:
{performance}

YOUR TASK:
1. Analyze which profile works best for this regime
2. Recommend profile allocation
3. Set confidence level

OUTPUT (only valid JSON):
{{"selected_profile": "name", "allocation": {{"profile1": 0.6, "profile2": 0.4}}, "confidence": 0.8}}
"""

STRATEGY_PROMPT = """
You are a Quant Developer.

PROFILE: {profile_name}
PHILOSOPHY: {philosophy}
RULES:
{rules}

YOUR TASK:
Generate Python strategy code with:
1. generate_signal(prices: List[float], regime: Dict) -> str
2. Entry/exit rules based on profile
3. Risk management rules

OUTPUT (only code, no markdown):
```python
class {class_name}:
    def __init__(self):
        self.stop_loss = 0.05
        self.take_profit = 0.10
        self.position_size = 0.10
    
    def generate_signal(self, prices, regime):
        # Your logic here
        return 'long' | 'short' | 'neutral'
```
"""


# ============ EVOLUTION CYCLE ============

def evolve_prompt(engine, agent_name: str, mutation_type: str = "random") -> str:
    """Evolve a trading agent's prompt"""
    agent = engine.get_agent(agent_name)
    if not agent:
        raise ValueError(f"Agent not found: {agent_name}")
    
    user_prompt = MUTATION_PROMPT.replace("{prompt}", agent.system_prompt)
    result = engine.llm.generate(
        system_prompt="You are a prompt engineering expert.",
        user_prompt=user_prompt
    )
    
    # Parse JSON response
    try:
        data = json.loads(result)
        return data.get("mutated_prompt", agent.system_prompt)
    except:
        return agent.system_prompt


def generate_strategy_code(engine, profile_name: str) -> str:
    """Generate strategy code from profile"""
    profiles = {
        "momentum": {
            "class_name": "MomentumStrategy",
            "philosophy": "Trade with the trend",
            "rules": "1. EMA 9 > EMA 21 > EMA 50\n2. Trail stops\n3. Buy breakouts"
        },
        "mean_reversion": {
            "class_name": "MeanReversionStrategy", 
            "philosophy": "Buy low, sell high",
            "rules": "1. RSI < 30 = buy\n2. RSI > 70 = sell\n3. Target mean"
        },
        "volatility": {
            "class_name": "VolatilityStrategy",
            "philosophy": "Buy low vol, sell high vol",
            "rules": "1. Bollinger Band breakout\n2. ATR-based sizing\n3. Skip high vol"
        }
    }
    
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")
    
    profile = profiles[profile_name]
    user_prompt = STRATEGY_PROMPT.format(
        profile_name=profile_name,
        class_name=profile["class_name"],
        philosophy=profile["philosophy"],
        rules=profile["rules"]
    )
    
    result = engine.llm.generate(
        system_prompt="You are an expert Python developer.",
        user_prompt=user_prompt
    )
    
    # Extract code from response
    if "```python" in result:
        code = result.split("```python")[1].split("```")[0]
    elif "```" in result:
        code = result.split("```")[1].split("```")[0]
    else:
        code = result
    
    return code.strip()


# ============ MAIN ============

def main():
    """Example usage"""
    import os
    
    # Configure LLM (use environment variable)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return
    
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key=api_key,
        temperature=0.7
    )
    
    llm = LLMEvolver(config)
    
    # Test mutation
    result = llm.generate(
        system_prompt="You are a trading expert.",
        user_prompt="Improve this prompt: 'Buy when RSI < 30'"
    )
    
    print("LLM Response:")
    print(result)


if __name__ == "__main__":
    main()
