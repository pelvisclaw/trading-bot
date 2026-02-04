#!/usr/bin/env python3
"""
Market Analysis for Trading Agent Selection
Analyzes current market and recommends best profile
"""
import sys
import json
import os
sys.path.insert(0, 'llm-evolution')

from evolution_engine import LLMEvolutionEngine


def analyze_market():
    """Analyze current market conditions"""
    # Fetch current data would go here
    # For now, simulate analysis
    
    market_data = {
        "regime": "bear",
        "confidence": 0.75,
        "volatility": "high",
        "trend": "down",
        "volume": "normal"
    }
    
    print("=== Market Analysis ===")
    print(f"Regime: {market_data['regime']}")
    print(f"Confidence: {market_data['confidence']*100:.0f}%")
    print(f"Volatility: {market_data['volatility']}")
    print(f"Trend: {market_data['trend']}")
    print(f"Volume: {market_data['volume']}")
    
    return market_data


def load_population():
    """Load current population performance"""
    engine = LLMEvolutionEngine()
    
    if os.path.exists("llm-evolution/population.json"):
        engine.load_population()
        return engine.population
    
    return {}


def recommend_profile(market_data, population):
    """Recommend best profile for current market"""
    
    # Simple rule-based recommendation
    # In production, this would use the Trading Manager LLM
    
    regime = market_data["regime"]
    
    recommendations = {
        "bull": {
            "profile": "momentum_v1",
            "allocation": {"momentum_v1": 0.6, "trend_follower_v1": 0.4},
            "position_size": 0.10,
            "reasoning": "Bull market favors momentum strategies"
        },
        "bear": {
            "profile": "mean_reversion_v1",
            "allocation": {"mean_reversion_v1": 0.5, "regime_aware_v1": 0.5},
            "position_size": 0.05,
            "reasoning": "Bear market: mean reversion or stay defensive"
        },
        "high_vol": {
            "profile": "volatility_v1",
            "allocation": {"volatility_v1": 0.4, "regime_aware_v1": 0.6},
            "position_size": 0.05,
            "reasoning": "High volatility: reduce exposure, use volatility strategies"
        },
        "sideways": {
            "profile": "breakout_v1",
            "allocation": {"breakout_v1": 0.5, "mean_reversion_v1": 0.5},
            "position_size": 0.08,
            "reasoning": "Sideways market: range trading or mean reversion"
        },
        "mean_reversion": {
            "profile": "mean_reversion_v1",
            "allocation": {"mean_reversion_v1": 0.7, "volatility_v1": 0.3},
            "position_size": 0.08,
            "reasoning": "Extended from mean: mean reversion opportunity"
        }
    }
    
    if regime in recommendations:
        rec = recommendations[regime]
    else:
        rec = {
            "profile": "regime_aware_v1",
            "allocation": {"regime_aware_v1": 1.0},
            "position_size": 0.05,
            "reasoning": "Uncertain regime: use adaptive strategy"
        }
    
    print("\n=== Profile Recommendation ===")
    print(f"Selected Profile: {rec['profile']}")
    print(f"Allocation: {rec['allocation']}")
    print(f"Position Size: {rec['position_size']*100:.0f}%")
    print(f"Reasoning: {rec['reasoning']}")
    
    # Save recommendation
    with open("llm-evolution/recommendation.json", "w") as f:
        json.dump({
            "market": market_data,
            "recommendation": rec,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }, f, indent=2)
    
    return rec


def main():
    print("=== Trading Agent Market Analysis ===\n")
    
    # Analyze market
    market_data = analyze_market()
    
    # Load population
    population = load_population()
    
    # Recommend profile
    recommend_profile(market_data, population)
    
    print("\n✓ Analysis complete. Recommendation saved to llm-evolution/recommendation.json")


if __name__ == "__main__":
    main()
