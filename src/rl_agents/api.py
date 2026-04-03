"""
RL Agents API for Forward and Counterfactual Simulations

Provides high-level interfaces for:
- Running forward trial simulations with meta RL agent and strategies
- Counterfactual trial generation with pre-trained agents
- Flexible trial scheduling based on experimental design parameters
"""

from typing import Dict, Any, List, Optional, Sequence, Tuple, Callable
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import logging

logger = logging.getLogger(__name__)


# === Strategy and Condition Constants ===
STRAT_DT = "dt"
STRAT_LR_CALC = "lr_calc"
STRAT_LR_HEUR = "lr_heur"
LR_FAMILY = {STRAT_LR_CALC, STRAT_LR_HEUR}

COND_DT, COND_LR, COND_DTLR = "DT", "LR", "DT+LR"
TYPE_DT, TYPE_LR = "DT", "LR"


# === Helper Functions ===

def _build_with_xai_schedule(N: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Build random with_xai schedule with given ratio."""
    k = int(round(N * ratio))
    flags = np.array([1] * k + [0] * (N - k), dtype=np.int32)
    rng.shuffle(flags)
    return flags.astype(bool)


def _build_trial_type_schedule(N: int, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Build trial type schedule based on condition."""
    if condition == COND_DT:
        return np.array([TYPE_DT] * N, dtype=object)
    if condition == COND_LR:
        return np.array([TYPE_LR] * N, dtype=object)
    # DT+LR: half/half shuffled
    m = N // 2
    arr = np.array([TYPE_DT] * m + [TYPE_LR] * (N - m), dtype=object)
    rng.shuffle(arr)
    return arr


def _onehot_condition(condition: str) -> np.ndarray:
    """One-hot encode condition."""
    return np.array([
        1.0 if condition == COND_DT else 0.0,
        1.0 if condition == COND_LR else 0.0,
        1.0 if condition == COND_DTLR else 0.0,
    ], dtype=np.float32)


def _onehot_trial_type(tt: str) -> np.ndarray:
    """One-hot encode trial type."""
    return np.array([
        1.0 if tt == TYPE_DT else 0.0,
        1.0 if tt == TYPE_LR else 0.0,
    ], dtype=np.float32)


def _strategy_allowed_under_condition(condition: str, strat_name: str) -> bool:
    """Check if strategy is allowed under given condition."""
    if condition == COND_DT:
        return strat_name == STRAT_DT
    if condition == COND_LR:
        return strat_name in LR_FAMILY
    return True  # DT+LR allows all


class ForwardSimulationRunner:
    """
    High-level runner for forward simulation using meta RL agent.
    
    Implements the full meta episode loop matching the training environment.
    """
    
    def __init__(
        self,
        meta_model: PPO,
        strategies: Dict[str, Any],
        training_cog_params: Dict[str, Any],
    ):
        """
        Initialize forward simulation runner.
        
        Args:
            meta_model: Trained meta PPO agent from stable_baselines3
            strategies: Dict of strategies {name -> strategy_obj}
            training_cog_params: Base cognitive parameters from training
        """
        self.meta_model = meta_model
        self.strategies = strategies
        self.training_cog_params = training_cog_params
    
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
        Run one meta episode.
        
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
            Dict with episode results and logs
        """
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
            action, _ = self.meta_model.predict(obs, deterministic=deterministic)
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


class CounterfactualSimulationRunner:
    """
    High-level runner for counterfactual trial generation using trained agents.
    
    Uses pre-trained counterfactual RL agent to suggest feature changes
    and generate counterfactual predictions.
    """
    
    def __init__(
        self,
        counterfactual_agent: PPO,
        ai_dataset_loader: Any,
    ):
        """
        Initialize counterfactual runner.
        
        Args:
            counterfactual_agent: Trained counterfactual PPO agent
            ai_dataset_loader: Dataset loader
        """
        self.counterfactual_agent = counterfactual_agent
        self.ai_dataset_loader = ai_dataset_loader
    
    def run_counterfactual_episode(
        self,
        instance_ids: Sequence[int],
        app_id: str,
        model_name: str,
        condition: str,
        participant_id: Optional[str] = None,
        phase: str = "counterfactual",
        with_xai_schedule: Optional[np.ndarray] = None,
        deterministic: bool = True,
        rng_seed: int = 123,
    ) -> Dict[str, Any]:
        """
        Run counterfactual episode.
        
        Args:
            instance_ids: Instance IDs for counterfactual trials
            app_id: Application/dataset ID
            model_name: Model name
            condition: XAI condition (DT, LR, DT+LR)
            participant_id: Optional tracking ID
            phase: Trial phase (typically "counterfactual")
            with_xai_schedule: Whether with/without XAI for each trial
            deterministic: Use deterministic policy
            rng_seed: Random seed
        
        Returns:
            Dict with counterfactual predictions
        """
        logger.warning("CounterfactualSimulationRunner not fully implemented yet")
        return {
            "participant_id": participant_id,
            "phase": phase,
            "instance_ids": list(instance_ids),
            "cf_predictions": [0.5] * len(instance_ids),
        }


def create_forward_runner(
    meta_model: PPO,
    strategies: Dict[str, Any],
    training_cog_params: Dict[str, Any],
) -> ForwardSimulationRunner:
    """
    Factory function to create ForwardSimulationRunner.
    
    Args:
        meta_model: Trained PPO model from stable_baselines3
        strategies: Dict of strategy objects
        training_cog_params: Training cognitive parameters
    
    Returns:
        ForwardSimulationRunner instance
    """
    return ForwardSimulationRunner(
        meta_model=meta_model,
        strategies=strategies,
        training_cog_params=training_cog_params,
    )


def create_counterfactual_runner(
    counterfactual_agent: PPO,
    ai_dataset_loader: Any,
) -> CounterfactualSimulationRunner:
    """
    Factory function to create CounterfactualSimulationRunner.
    
    Args:
        counterfactual_agent: Trained PPO agent from stable_baselines3
        ai_dataset_loader: Dataset loader
    
    Returns:
        CounterfactualSimulationRunner instance
    """
    return CounterfactualSimulationRunner(
        counterfactual_agent=counterfactual_agent,
        ai_dataset_loader=ai_dataset_loader,
    )
