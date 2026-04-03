"""
CoXAM RL Agents API Layer

Consolidated and unified RL agent implementations for cognitive modeling.
Supports Decision Tree (DT) and Logistic Regression (LR) strategies.

Module structure:
- environments/: Gym-compatible training environments
- agents/: RL agent implementations with PPO
- utils/: Training, inference, and weight management utilities
- model_weights/: Organized pre-trained model checkpoints

Usage:
    from src.coxam.RL_agents import DTAgent, LRAgent, DTForwardEnvironment
    
    # Create environment
    from src.coxam.RL_agents.environments import EnvironmentConfig
    config = EnvironmentConfig(
        instances_per_episode=40,
        ai_dataset_loaders={...},
        explainers={...}
    )
    env = DTForwardEnvironment(config)
    
    # Create and train agent
    from src.coxam.RL_agents.agents import AgentConfig
    agent_config = AgentConfig(
        agent_id="dt_agent_v1",
        agent_type="dt",
        model_weights_dir="./weights/dt"
    )
    agent = DTAgent(agent_config)
    agent.train(env, total_timesteps=100000)
    
    # Inference
    obs, _ = env.reset()
    action, _ = agent.predict(obs)
"""

from .environments import (
    BaseRLEnvironment,
    EnvironmentConfig,
    DTForwardEnvironment,
    LRForwardEnvironment,
)

from .agents import (
    RLAgent,
    AgentConfig,
    DTAgent,
    LRAgent,
)

from .utils import (
    TrainingManager,
    WeightOrganizer,
    InferenceManager,
)

__all__ = [
    # Environments
    "BaseRLEnvironment",
    "EnvironmentConfig",
    "DTForwardEnvironment",
    "LRForwardEnvironment",
    # Agents
    "RLAgent",
    "AgentConfig",
    "DTAgent",
    "LRAgent",
    # Utilities
    "TrainingManager",
    "WeightOrganizer",
    "InferenceManager",
]
