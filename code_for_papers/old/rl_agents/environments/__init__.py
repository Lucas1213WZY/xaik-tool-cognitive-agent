"""
RL Environments Module

Gymnasium-compatible environments for training RL agents to select
cognitive reasoning strategies (LR calculation, LR heuristic, Decision Tree).

Environments:
- BaseRLEnvironment: Abstract base with memory and strategy loading
- DTForwardEnvironment: Decision Tree strategy selection with cognitive parameters
- LRCalculationEnvironment: LR Calculation strategy with feature selection
- LRHeuristicEnvironment: LR Heuristic strategy with minimal parameters
- MetaRouterEnv: Meta-level environment for orchestrating multiple strategies
"""

from .base_env import BaseRLEnvironment, EnvironmentConfig
from .dt_forward_env import DTForwardEnvironment
from .lr_forward_env import LRCalculationEnvironment, LRHeuristicEnvironment
from .counterfactual_env import CounterfactualEnv
from .meta_router_env import (
    MetaRouterEnv,
    STRAT_DT, STRAT_LR_CALC, STRAT_LR_HEUR, LR_FAMILY,
    COND_DT, COND_LR, COND_DTLR,
    TYPE_DT, TYPE_LR,
    _build_with_xai_schedule,
    _build_trial_type_schedule,
    _onehot_condition,
    _onehot_trial_type,
    _strategy_allowed_under_condition,
)

__all__ = [
    "BaseRLEnvironment",
    "EnvironmentConfig",
    "DTForwardEnvironment",
    "LRCalculationEnvironment",
    "LRHeuristicEnvironment",
    "CounterfactualEnv",
    "MetaRouterEnv",
    # Constants
    "STRAT_DT",
    "STRAT_LR_CALC",
    "STRAT_LR_HEUR",
    "LR_FAMILY",
    "COND_DT",
    "COND_LR",
    "COND_DTLR",
    "TYPE_DT",
    "TYPE_LR",
    # Helper Functions
    "_build_with_xai_schedule",
    "_build_trial_type_schedule",
    "_onehot_condition",
    "_onehot_trial_type",
    "_strategy_allowed_under_condition",
]

