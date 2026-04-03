"""
Meta Router Environment for Multi-Strategy Selection

A meta-level environment that orchestrates multiple reasoning strategies
and trains an RL agent to select the best strategy for each trial.

Features:
- Strategy selection policy (RL agent chooses which strategy to use)
- Episode-level conditions (DT-only, LR-only, or mixed DT+LR)
- Per-strategy performance tracking (with/without XAI)
- XAI trial scheduling and mismatch handling
- Multi-strategy coordination with shared memory and context
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


# Strategy type constants
STRAT_DT = "dt"
STRAT_LR_CALC = "lr_calc"
STRAT_LR_HEUR = "lr_heur"
LR_FAMILY = {STRAT_LR_CALC, STRAT_LR_HEUR}

# Condition constants
COND_DT = "DT"       # Decision Tree only
COND_LR = "LR"       # LR strategies only
COND_DTLR = "DT+LR"  # Mixed condition

TYPE_DT = "DT"       # DT trial type
TYPE_LR = "LR"       # LR trial type


def split_cog_cfg(cog_cfg: Optional[Dict[str, Any]]) -> Tuple[Dict, Dict]:
    """
    Split cognitive config into ranges (sampled) and fixed values.
    
    Args:
        cog_cfg: Dict with keys mapping to values or [low, high] tuples
    
    Returns:
        (episodic_ranges, episodic_fixed) dicts
    """
    episodic_ranges = {}
    episodic_fixed = {}
    
    for k, v in (cog_cfg or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            low, high = float(v[0]), float(v[1])
            episodic_ranges[k] = (low, high)
        elif isinstance(v, (int, float)):
            episodic_fixed[k] = float(v)
        else:
            episodic_fixed[k] = v
    
    return episodic_ranges, episodic_fixed


def sample_episode_cogs(rng: np.random.Generator, 
                       training_cog_params: Dict[str, Any]) -> Dict[str, float]:
    """
    Sample cognitive parameters for this episode.
    
    Args:
        rng: NumPy random generator
        training_cog_params: Config with ranges or fixed values
    
    Returns:
        Dict with sampled parameter values
    """
    r, f = split_cog_cfg(training_cog_params)
    out = dict(f)
    for k, (lo, hi) in r.items():
        out[k] = float(rng.uniform(lo, hi))
    return out


# === Module-level helper functions for inference ===

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


class MetaRouterEnv:
    """
    Meta environment orchestrating multiple reasoning strategies.
    
    The agent learns to select which strategy to dispatch for each trial
    under different conditions (DT-only, LR-only, or mixed).
    
    Architecture:
    - Agent selects strategy via Discrete action space
    - Multiple strategies share memory and context
    - Episodes have fixed conditions + XAI schedule
    - Observations include per-strategy statistics
    - Reward = correctness - chi * prediction time
    
    Attributes:
        strategies: Dict mapping strategy names to instances
        action_space: Discrete(num_strategies)
        observation_space: Includes base features + per-strategy stats
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        *,
        strategies: Dict[str, Any],
        dataset_loaders: Dict[str, Any],
        training_cog_params: Dict[str, Any],
        instances_per_episode: int = 40,
        max_features: int = 6,
        xai_trial_ratio: float = 0.5,
        shuffle_features: bool = True,
        condition: Optional[str] = None,
        invalid_action_penalty: float = -1.0,
    ):
        """
        Initialize Meta Router Environment.
        
        Args:
            strategies: Dict like {"dt": dt_policy, "lr_calc": lr_calc_policy, ...}
            dataset_loaders: Dict of data loaders by dataset ID
            training_cog_params: Cognitive parameter ranges/values
            instances_per_episode: Trials per episode
            max_features: Max feature dimension
            xai_trial_ratio: Fraction of trials with XAI
            shuffle_features: Whether to permute features each episode
            condition: Fixed condition or None for random (DT, LR, or DT+LR)
            invalid_action_penalty: Reward for invalid action under condition
        """
        super().__init__()
        
        self.strategy_names = list(strategies.keys())
        self.strategies = strategies
        self.num_strategies = len(self.strategy_names)
        
        self.loaders = dataset_loaders
        self.instances_per_episode = int(instances_per_episode)
        self.max_features = int(max_features)
        self.xai_trial_ratio = float(xai_trial_ratio)
        self.shuffle_features = bool(shuffle_features)
        self.cog_cfg = dict(training_cog_params)
        self.fixed_condition = condition if condition in {COND_DT, COND_LR, COND_DTLR} else None
        self.invalid_action_penalty = float(invalid_action_penalty)
        
        # Define action space: which strategy to select
        self.action_space = spaces.Discrete(self.num_strategies)
        
        # Define observation space
        # Base: [chi_norm, trial_progress, with_xai_flag]
        # Condition: one-hot [3]
        # Trial type: one-hot [2]
        # Per-strategy stats: for each strategy [count_with, mean_with, count_without, mean_without] = 4 * S
        base_dim = 3
        condition_dim = 3
        trial_type_dim = 2
        stats_dim = 4 * self.num_strategies
        
        self._obs_dim = base_dim + condition_dim + trial_type_dim + stats_dim
        
        # Observation bounds
        low = np.zeros(self._obs_dim, dtype=np.float32)
        high = np.zeros(self._obs_dim, dtype=np.float32)
        
        # Base features normalized in [0, 1]
        high[:base_dim] = 1.0
        
        # Condition one-hot [0, 1]
        high[base_dim:base_dim+condition_dim] = 1.0
        
        # Trial type one-hot [0, 1]
        high[base_dim+condition_dim:base_dim+condition_dim+trial_type_dim] = 1.0
        
        # Per-strategy stats
        # counts in [0, instances_per_episode], means in [0, 1]
        for i in range(self.num_strategies):
            idx = base_dim + condition_dim + trial_type_dim + i * 4
            high[idx] = float(self.instances_per_episode)       # count_with
            high[idx+1] = 1.0                                    # mean_with
            high[idx+2] = float(self.instances_per_episode)     # count_without
            high[idx+3] = 1.0                                    # mean_without
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Episode state
        self.np_random = None
        self.condition = None
        self.with_xai = None
        self.trial_types = None
        self.episode_cogs = None
        self.chi = None
        self.chi_low = None
        self.chi_high = None
        
        # Current step
        self.t = 0
        self.X_raw = None
        self.X_norm = None
        self.y = None
        self.perm = None
        self.inv_perm = None
        
        # Per-episode statistics
        self._stats = None
    
    @staticmethod
    def _build_with_xai_schedule(n: int, ratio: float, 
                                 rng: np.random.Generator) -> np.ndarray:
        """Build schedule of which trials have XAI."""
        k = int(round(n * ratio))
        flags = np.array([1] * k + [0] * (n - k), dtype=np.int32)
        rng.shuffle(flags)
        return flags.astype(bool)
    
    @staticmethod
    def _build_trial_type_schedule(n: int, condition: str,
                                   rng: np.random.Generator) -> np.ndarray:
        """
        Build schedule of trial types (DT vs LR).
        
        Args:
            n: Number of trials
            condition: "DT", "LR", or "DT+LR"
            rng: Random generator
        
        Returns:
            Array of trial type strings
        """
        if condition == COND_DT:
            return np.array([TYPE_DT] * n, dtype=object)
        elif condition == COND_LR:
            return np.array([TYPE_LR] * n, dtype=object)
        else:  # DT+LR mixed
            m = n // 2
            types = np.array([TYPE_DT] * m + [TYPE_LR] * (n - m), dtype=object)
            rng.shuffle(types)
            return types
    
    def _onehot_condition(self) -> np.ndarray:
        """One-hot encode current condition."""
        return np.array([
            1.0 if self.condition == COND_DT else 0.0,
            1.0 if self.condition == COND_LR else 0.0,
            1.0 if self.condition == COND_DTLR else 0.0,
        ], dtype=np.float32)
    
    def _onehot_trial_type(self, trial_idx: int) -> np.ndarray:
        """One-hot encode trial type at given index."""
        tt = self.trial_types[trial_idx] if (0 <= trial_idx < self.instances_per_episode) else TYPE_DT
        return np.array([
            1.0 if tt == TYPE_DT else 0.0,
            1.0 if tt == TYPE_LR else 0.0,
        ], dtype=np.float32)
    
    def _stats_vector(self) -> np.ndarray:
        """
        Flatten per-strategy statistics into vector.
        
        For each strategy in order:
        [count_with, mean_with, count_without, mean_without]
        """
        out = []
        for name in self.strategy_names:
            w = self._stats[name]["with"]
            wo = self._stats[name]["without"]
            count_w = float(w["count"])
            count_wo = float(wo["count"])
            mean_w = (w["sum_pr"] / count_w) if count_w > 0 else 0.0
            mean_wo = (wo["sum_pr"] / count_wo) if count_wo > 0 else 0.0
            out.extend([count_w, float(mean_w), count_wo, float(mean_wo)])
        return np.asarray(out, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        with_xai_flag = float(self.with_xai[self.t]) if self.t < self.instances_per_episode else 0.0
        
        base = np.array([
            float(self.chi / self.chi_high) if self.chi_high > 0 else 0.0,
            float(self.t / self.instances_per_episode),
            with_xai_flag,
        ], dtype=np.float32)
        
        obs = np.concatenate([
            base,
            self._onehot_condition(),
            self._onehot_trial_type(self.t),
            self._stats_vector(),
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Optional options (can override defaults)
        
        Returns:
            (observation, info)
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        rng = self.np_random if self.np_random is not None else np.random.default_rng()
        
        # Feature permutation
        if self.shuffle_features:
            self.perm = rng.permutation(self.max_features).astype(np.int64)
            inv = np.empty_like(self.perm)
            inv[self.perm] = np.arange(self.max_features, dtype=np.int64)
            self.inv_perm = inv
        else:
            self.perm = np.arange(self.max_features, dtype=np.int64)
            self.inv_perm = np.arange(self.max_features, dtype=np.int64)
        
        # Sample dataset
        dataset_key = rng.choice(list(self.loaders.keys()))
        loader = self.loaders[dataset_key]
        
        # Load instances
        indices = rng.choice(list(range(1, 400)), 
                           size=self.instances_per_episode, 
                           replace=False).tolist()
        X_raw, y = loader.load_instances(indices, normalize=False)
        X_norm = loader.load_instances(indices, normalize=True)[0]
        
        self.X_raw = np.asarray(X_raw, dtype=np.float32)
        self.X_norm = np.asarray(X_norm, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        
        # Episode cognitive parameters
        self.episode_cogs = sample_episode_cogs(rng, self.cog_cfg)
        
        # Chi and XAI schedule
        chi_spec = self.cog_cfg.get("chi", [0.0, 0.03])
        if isinstance(chi_spec, (list, tuple)) and len(chi_spec) == 2:
            self.chi_low, self.chi_high = float(chi_spec[0]), float(chi_spec[1])
            self.chi = float(rng.uniform(self.chi_low, self.chi_high))
        else:
            self.chi_low = self.chi_high = float(chi_spec) if isinstance(chi_spec, (int, float)) else 0.015
            self.chi = self.chi_low
        
        self.with_xai = self._build_with_xai_schedule(self.instances_per_episode, 
                                                      self.xai_trial_ratio, rng)
        
        # Episode condition and trial types
        if self.fixed_condition is None:
            self.condition = rng.choice([COND_DT, COND_LR, COND_DTLR])
        else:
            self.condition = self.fixed_condition
        
        self.trial_types = self._build_trial_type_schedule(self.instances_per_episode,
                                                           self.condition, rng)
        
        # Reset step counter
        self.t = 0
        
        # Reset all strategies with shared context
        shared_context = {
            "rng": rng,
            "with_xai_schedule": self.with_xai,
            "perm": self.perm,
            "inv_perm": self.inv_perm,
            "episode_cogs": self.episode_cogs,
            "dataset_id": dataset_key,
        }
        
        for strategy in self.strategies.values():
            if hasattr(strategy, 'reset'):
                strategy.reset(**shared_context)
        
        # Initialize per-episode statistics
        self._stats = {
            name: {
                "with": {"count": 0, "sum_pr": 0.0},
                "without": {"count": 0, "sum_pr": 0.0},
            }
            for name in self.strategy_names
        }
        
        return self._get_observation(), {}
    
    def _strategy_allowed(self, strategy_name: str) -> bool:
        """Check if strategy is allowed under current condition."""
        if self.condition == COND_DT:
            return strategy_name == STRAT_DT
        elif self.condition == COND_LR:
            return strategy_name in LR_FAMILY
        else:  # COND_DTLR
            return True
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[
        np.ndarray, float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one step: dispatch chosen strategy, compute reward.
        
        Args:
            action: Strategy index
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action_idx = int(action) if isinstance(action, (int, np.integer)) else int(action)
        
        # Validate action index
        if not (0 <= action_idx < self.num_strategies):
            self.t += 1
            terminated = self.t >= self.instances_per_episode
            return self._get_observation(), float(self.invalid_action_penalty), False, terminated, {
                "error": "invalid strategy index",
                "condition": self.condition,
                "invalid_under_condition": True,
            }
        
        strategy_name = self.strategy_names[action_idx]
        
        # Check if strategy allowed under condition
        if not self._strategy_allowed(strategy_name):
            self.t += 1
            terminated = self.t >= self.instances_per_episode
            return self._get_observation(), float(self.invalid_action_penalty), False, terminated, {
                "condition": self.condition,
                "strategy": strategy_name,
                "invalid_under_condition": True,
            }
        
        # Get trial data
        strategy = self.strategies[strategy_name]
        x_raw = self.X_raw[self.t]
        x_norm = self.X_norm[self.t]
        y_true = int(self.y[self.t])
        with_xai_req = bool(self.with_xai[self.t])
        trial_type = str(self.trial_types[self.t])
        
        # Mismatch logic: WITH-XAI trial but wrong type → run WITHOUT-XAI
        with_xai_used = with_xai_req
        mismatch = False
        
        if with_xai_req:
            if trial_type == TYPE_DT and strategy_name in LR_FAMILY:
                with_xai_used = False
                mismatch = True
            elif trial_type == TYPE_LR and strategy_name == STRAT_DT:
                with_xai_used = False
                mismatch = True
        
        # Dispatch to strategy
        try:
            if hasattr(strategy, 'step'):
                result = strategy.step(
                    x_raw=x_raw,
                    x_norm=x_norm,
                    y_true=y_true,
                    with_xai=with_xai_used,
                    chi_value=float(self.chi),
                )
                probs, pred_time, sinfo = result if len(result) == 3 else (result[0], result[1], {})
            else:
                # Fallback inference
                probs, pred_time, sinfo = {0: 0.5, 1: 0.5}, 0.5, {}
        except Exception as e:
            logger.warning(f"Strategy step failed: {e}")
            probs, pred_time, sinfo = {0: 0.5, 1: 0.5}, 0.5, {}
        
        # Compute reward
        pr = float(probs.get(y_true, 0.5))
        reward = pr - float(self.chi) * float(pred_time)
        
        # Update statistics
        mode_key = "with" if with_xai_used else "without"
        entry = self._stats[strategy_name][mode_key]
        entry["count"] += 1
        entry["sum_pr"] += pr
        
        self.t += 1
        terminated = self.t >= self.instances_per_episode
        truncated = False
        
        info = {
            "strategy": strategy_name,
            "prob_correct": pr,
            "pred_time": float(pred_time),
            **(sinfo or {}),
            "with_xai_requested": with_xai_req,
            "with_xai_used": with_xai_used,
            "mismatch_applied": mismatch,
            "condition": self.condition,
            "trial_type": trial_type,
            "invalid_under_condition": False,
        }
        
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode."""
        if self._stats is None:
            return {}
        
        stats = {}
        for name in self.strategy_names:
            w = self._stats[name]["with"]
            wo = self._stats[name]["without"]
            total = w["count"] + wo["count"]
            
            stats[name] = {
                "with_count": w["count"],
                "with_mean": (w["sum_pr"] / w["count"]) if w["count"] > 0 else 0.0,
                "without_count": wo["count"],
                "without_mean": (wo["sum_pr"] / wo["count"]) if wo["count"] > 0 else 0.0,
                "total": total,
                "overall_mean": ((w["sum_pr"] + wo["sum_pr"]) / total) if total > 0 else 0.0,
            }
        
        return stats
