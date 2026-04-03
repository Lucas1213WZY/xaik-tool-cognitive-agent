"""
Meta Router RL Agent

Specialized PPO agent for multi-strategy selection training.
Uses MetaRouterEnv to orchestrate multiple reasoning strategies.

Supports both:
- Training: via create_env() -> Gym environment
- Inference: via run_episode() -> Direct inference on data
"""

from typing import Optional, Dict, Any, Sequence
import numpy as np
import logging
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from .base_agent import BaseRLAgent, AgentConfig
from src.rl_agents.environments import MetaRouterEnv
from src.rl_agents.environments.meta_router_env import (
    STRAT_DT, STRAT_LR_CALC, STRAT_LR_HEUR, LR_FAMILY,
    COND_DT, COND_LR, COND_DTLR,
    TYPE_DT, TYPE_LR,
    _build_with_xai_schedule,
    _build_trial_type_schedule,
    _onehot_condition,
    _onehot_trial_type,
    _strategy_allowed_under_condition,
)

logger = logging.getLogger(__name__)


class MetaRouterAgent(BaseRLAgent):
    """
    RL Agent for meta-strategy selection.
    
    Trains on MetaRouterEnv to learn which reasoning strategy (DT, LR Calc, LR Heur)
    to dispatch for each trial under different conditions.
    
    Features:
    - Multi-strategy orchestration
    - Episode-level conditions (DT-only, LR-only, mixed)
    - Per-strategy performance tracking
    - XAI trial scheduling
    - Mismatch handling
    
    Supports both training (create_env) and inference (run_episode).
    """
    
    def __init__(self, config: AgentConfig, 
                 strategies: Dict[str, Any],
                 dataset_loaders: Dict[str, Any],
                 training_cog_params: Dict[str, Any],
                 instances_per_episode: int = 40,
                 max_features: int = 6,
                 xai_trial_ratio: float = 0.5,
                 shuffle_features: bool = True,
                 condition: Optional[str] = None,
                 invalid_action_penalty: float = -1.0):
        """
        Initialize Meta Router Agent.
        
        Args:
            config: AgentConfig with PPO hyperparameters
            strategies: Dict mapping strategy names to instances
            dataset_loaders: Dict of data loaders by dataset ID
            training_cog_params: Cognitive parameter ranges/values
            instances_per_episode: Trials per episode
            max_features: Feature dimension
            xai_trial_ratio: Fraction with XAI
            shuffle_features: Whether to permute features
            condition: Fixed condition or None for random
            invalid_action_penalty: Penalty for invalid actions
        """
        super().__init__(config, MetaRouterEnv)
        
        self.strategies = strategies
        self.dataset_loaders = dataset_loaders
        self.training_cog_params = training_cog_params
        self.instances_per_episode = instances_per_episode
        self.max_features = max_features
        self.xai_trial_ratio = xai_trial_ratio
        self.shuffle_features = shuffle_features
        self.condition = condition
        self.invalid_action_penalty = invalid_action_penalty
    
    def create_env(self, n_envs: int = 1, is_eval: bool = False) -> VecEnv:
        """
        Create vectorized MetaRouterEnv.
        
        Args:
            n_envs: Number of parallel environments
            is_eval: Whether for evaluation
        
        Returns:
            VecEnv instance
        """
        def _make_env(rank: int):
            def _init():
                return MetaRouterEnv(
                    strategies=self.strategies,
                    dataset_loaders=self.dataset_loaders,
                    training_cog_params=self.training_cog_params,
                    instances_per_episode=self.instances_per_episode,
                    max_features=self.max_features,
                    xai_trial_ratio=self.xai_trial_ratio,
                    shuffle_features=self.shuffle_features,
                    condition=self.condition,
                    invalid_action_penalty=self.invalid_action_penalty,
                )
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([_make_env(0)])
        else:
            return SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    
    def run_episode(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        X_norm: Optional[np.ndarray] = None,
        with_xai_schedule: Optional[np.ndarray] = None,
        with_xai_ratio: Optional[float] = None,
        trial_type_schedule: Optional[np.ndarray] = None,
        condition: str = COND_DTLR,
        strategy_order: Optional[Sequence[str]] = None,
        perm: Optional[np.ndarray] = None,
        dataset_id: Optional[int] = None,
        episode_cogs: Optional[Dict[str, Any]] = None,
        chi_value: float = 0.01,
        deterministic: bool = False,
        invalid_action_penalty: float = -1.0,
        rng_seed: int = 123,
    ) -> Dict[str, Any]:
        """
        Run one inference episode with the trained meta model.
        
        Mirrors run_meta_on_batch functionality for direct inference on provided data.
        
        Args:
            X_raw: Raw features (N, F_raw)
            y_raw: True labels (N,)
            X_norm: Normalized features (N, F_norm), or None to use X_raw
            with_xai_schedule: Boolean array indicating with/without XAI per trial
            with_xai_ratio: Probability of with_xai if schedule not provided
            trial_type_schedule: Trial type (DT/LR) per trial
            condition: Overall condition (DT, LR, DT+LR)
            strategy_order: Order of strategies in observation
            perm: Feature permutation (for analysis)
            dataset_id: Dataset identifier
            episode_cogs: Cognitive parameters for this episode
            chi_value: Time cost parameter
            deterministic: Use deterministic policy
            invalid_action_penalty: Penalty for invalid strategy selection
            rng_seed: Random seed
        
        Returns:
            Dict with episode results and logs:
            {
                "total_reward": float,
                "mean_reward": float,
                "logs": dict (per-trial data),
                "meta": dict (episode metadata)
            }
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call .load() first.")
        
        rng = np.random.default_rng(rng_seed)
        
        assert X_raw.ndim == 2
        N = X_raw.shape[0]
        
        if X_norm is None:
            X_norm = X_raw
        assert X_norm.shape[0] == N
        assert y_raw.shape[0] == N
        
        # Strategy ordering
        if strategy_order is None:
            strategy_order = list(self.strategies.keys())
        S = len(strategy_order)
        name_from_idx = {i: strategy_order[i] for i in range(S)}
        
        # Build schedules
        if with_xai_schedule is None:
            ratio = float(with_xai_ratio if with_xai_ratio is not None else 0.5)
            with_xai_schedule = _build_with_xai_schedule(N, ratio, rng)
        else:
            with_xai_schedule = np.asarray(with_xai_schedule, dtype=bool)
        
        if trial_type_schedule is None:
            trial_type_schedule = _build_trial_type_schedule(N, condition, rng)
        else:
            trial_type_schedule = np.asarray(trial_type_schedule, dtype=object)
        
        # Feature permutation
        F = X_raw.shape[1]
        if perm is None:
            perm = np.arange(F, dtype=np.int64)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(F, dtype=np.int64)
        
        # Chi normalization
        chi_spec = self.training_cog_params.get("chi", [0.0, 0.03])
        if isinstance(chi_spec, (list, tuple)) and len(chi_spec) == 2:
            chi_high = float(chi_spec[1])
        else:
            chi_high = 0.03
        chi_high = max(chi_high, 1e-9)
        chi_norm = float(chi_value / chi_high)
        
        # Reset strategies
        shared_reset = dict(
            rng=rng,
            with_xai_schedule=with_xai_schedule,
            perm=perm,
            inv_perm=inv_perm,
            episode_cogs=dict(episode_cogs or {}),
            dataset_id=(dataset_id if dataset_id is not None else 1),
        )
        for sname in strategy_order:
            if sname in self.strategies and hasattr(self.strategies[sname], 'reset'):
                self.strategies[sname].reset(**shared_reset)
        
        # Per-episode stats
        stats = {
            name: {
                "with": {"count": 0, "sum_pr": 0.0},
                "without": {"count": 0, "sum_pr": 0.0},
            }
            for name in strategy_order
        }
        denom_N = float(max(1, N))
        
        def _stats_vector() -> np.ndarray:
            out = []
            for name in strategy_order:
                w = stats[name]["with"]
                wo = stats[name]["without"]
                count_w = float(w["count"])
                count_wo = float(wo["count"])
                mean_w = (w["sum_pr"] / count_w) if count_w > 0 else 0.0
                mean_wo = (wo["sum_pr"] / count_wo) if count_wo > 0 else 0.0
                out.extend([
                    count_w / denom_N,
                    float(mean_w),
                    count_wo / denom_N,
                    float(mean_wo),
                ])
            return np.asarray(out, dtype=np.float32)
        
        # Run episode
        total_reward = 0.0
        logs = {
            "strategy_name": [], "action_idx": [],
            "with_xai_requested": [], "with_xai_used": [],
            "trial_type": [], "condition": [], "mismatch_applied": [],
            "invalid_under_condition": [],
            "prob_correct": [], "probs": [], "pred_time": [], "reward": [], "info": [],
        }
        
        cond_oh = _onehot_condition(condition)
        
        for t in range(N):
            with_xai_req = bool(with_xai_schedule[t])
            trial_type = str(trial_type_schedule[t])
            
            # Build observation
            obs = np.concatenate([
                np.array([chi_norm, float(t / N), float(with_xai_req)], dtype=np.float32),
                cond_oh,
                _onehot_trial_type(trial_type),
                _stats_vector(),
            ]).astype(np.float32)
            
            # Get action from meta model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            a = int(action) if (0 <= int(action) < S) else 0
            sname = name_from_idx[a]
            
            # Check condition gating
            if not _strategy_allowed_under_condition(condition, sname):
                reward = float(invalid_action_penalty)
                total_reward += reward
                logs["strategy_name"].append(sname)
                logs["action_idx"].append(a)
                logs["with_xai_requested"].append(with_xai_req)
                logs["with_xai_used"].append(False)
                logs["trial_type"].append(trial_type)
                logs["condition"].append(condition)
                logs["mismatch_applied"].append(False)
                logs["invalid_under_condition"].append(True)
                logs["prob_correct"].append(0.0)
                logs["pred_time"].append(0.0)
                logs["probs"].append([0.0, 0.0])
                logs["reward"].append(reward)
                logs["info"].append({"invalid_under_condition": True})
                continue
            
            # Get strategy
            strat = self.strategies.get(sname)
            if strat is None:
                logger.warning(f"Strategy {sname} not found")
                continue
            
            # Handle with-XAI mismatch
            with_xai_used = with_xai_req
            mismatch = False
            if with_xai_req:
                if trial_type == TYPE_DT and sname in LR_FAMILY:
                    with_xai_used = False
                    mismatch = True
                elif trial_type == TYPE_LR and sname == STRAT_DT:
                    with_xai_used = False
                    mismatch = True
            
            # Execute strategy
            try:
                if hasattr(strat, 'step'):
                    probs, pred_time, info = strat.step(
                        x_raw=X_raw[t],
                        x_norm=X_norm[t],
                        y_true=int(y_raw[t]),
                        with_xai=with_xai_used,
                        chi_value=float(chi_value),
                    )
                else:
                    # Fallback: random prediction
                    probs = np.array([0.5, 0.5])
                    pred_time = 0.1
                    info = {}
            except Exception as e:
                logger.warning(f"Strategy step error: {e}")
                probs = np.array([0.5, 0.5])
                pred_time = 0.1
                info = {"error": str(e)}
            
            # Compute reward
            pr = float(probs[int(y_raw[t])])
            reward = pr - float(chi_value) * float(pred_time)
            total_reward += reward
            
            # Update stats
            mode_key = "with" if with_xai_used else "without"
            entry = stats[sname][mode_key]
            entry["count"] += 1
            entry["sum_pr"] += pr
            
            # Log
            logs["strategy_name"].append(sname)
            logs["action_idx"].append(a)
            logs["with_xai_requested"].append(with_xai_req)
            logs["with_xai_used"].append(with_xai_used)
            logs["trial_type"].append(trial_type)
            logs["condition"].append(condition)
            logs["mismatch_applied"].append(mismatch)
            logs["invalid_under_condition"].append(False)
            logs["prob_correct"].append(pr)
            logs["pred_time"].append(float(pred_time))
            logs["probs"].append(probs.tolist())
            logs["reward"].append(float(reward))
            logs["info"].append(info or {})
        
        return {
            "total_reward": float(total_reward),
            "mean_reward": float(np.mean(logs["reward"])) if N > 0 else 0.0,
            "logs": logs,
            "meta": {
                "N": N,
                "chi_value": float(chi_value),
                "chi_high": float(chi_high),
                "strategy_order": list(strategy_order),
                "dataset_id": dataset_id,
                "episode_cogs": dict(episode_cogs or {}),
                "condition": condition,
            },
        }
