"""
RL Agents Module

Reinforcement learning agents for cognitive strategy selection training.

This module integrates with:
- src.cognitive_models.memory: Unified memory system (ACT-R, Exemplar)
- src.cognitive_models: Forward reasoning strategies (LR Calc, LR Heur, DT)

Environments:
- DTForwardEnvironment: Decision Tree strategy selection
- LRCalculationEnvironment: LR Calculation with feature selection
- LRHeuristicEnvironment: LR Heuristic with parameter adaptation
- MetaRouterEnv: Multi-strategy orchestrator for meta-level reasoning

Agents:
- DTAgent: Trained on Decision Tree environment
- LRCalculationAgent: Trained on LR Calculation environment
- LRHeuristicAgent: Trained on LR Heuristic environment
- LRAgent: Unified agent supporting both LR strategies
- MetaRouterAgent: Meta-level agent for strategy selection

Utilities:
- InferenceManager: High-level prediction interface with caching
- TrainingManager: Training orchestration and logging

Example:
    ```python
    from src.rl_agents import DTAgent, AgentConfig, EnvironmentConfig
    
    # Configure agent
    agent_config = AgentConfig(
        agent_name="dt_agent_v1",
        learning_rate=3e-4,
        total_timesteps=100_000,
    )
    
    # Create and train agent
    agent = DTAgent(agent_config)
    results = agent.train(n_envs=4)
    
    # Save for later use
    agent.save("./models/dt_agent.zip")
    
    # Load and inference
    agent.load("./models/dt_agent.zip")
    action, _ = agent.predict(observation)
    ```
"""

from .environments import (
    BaseRLEnvironment,
    EnvironmentConfig,
    DTForwardEnvironment,
    LRCalculationEnvironment,
    LRHeuristicEnvironment,
    MetaRouterEnv,
    # Constants for MetaRouterEnv
    STRAT_DT,
    STRAT_LR_CALC,
    STRAT_LR_HEUR,
    COND_DT,
    COND_LR,
    COND_DTLR,
)

from .agents import (
    BaseRLAgent,
    AgentConfig,
    DTAgent,
    LRAgent,
    MetaRouterAgent,
)

from .utils import (
    InferenceManager,
    PredictionCache,
    TrainingManager,
)

from .api import (
    ForwardSimulationRunner,
    CounterfactualSimulationRunner,
    create_forward_runner,
    create_counterfactual_runner,
    TYPE_DT,
    TYPE_LR,
)

__version__ = "0.2.0"

__all__ = [
    # Environments
    "BaseRLEnvironment",
    "EnvironmentConfig",
    "DTForwardEnvironment",
    "LRCalculationEnvironment",
    "LRHeuristicEnvironment",
    "MetaRouterEnv",
    # Environment Constants
    "STRAT_DT",
    "STRAT_LR_CALC",
    "STRAT_LR_HEUR",
    "COND_DT",
    "COND_LR",
    "COND_DTLR",
    "TYPE_DT",
    "TYPE_LR",
    # Agents
    "BaseRLAgent",
    "AgentConfig",
    "DTAgent",
    "LRAgent",
    "MetaRouterAgent",
    # Utilities
    "InferenceManager",
    "PredictionCache",
    "TrainingManager",
    # API
    "ForwardSimulationRunner",
    "CounterfactualSimulationRunner",
    "create_forward_runner",
    "create_counterfactual_runner",
]
