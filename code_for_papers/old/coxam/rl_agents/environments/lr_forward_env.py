"""
Logistic Regression Forward Strategy RL Environment.

Supports LR heuristic and LR calculation strategies with feature selection masks.
Extracted from RL_feature_selection_agents_v0.3 and v0.4 notebooks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from gymnasium import spaces

from .base_env import BaseRLEnvironment, EnvironmentConfig
from src.memory.memory import DeclarativeMemory, CombinedMemory


# Action mapping for 5 strategies
STRATEGY_MAP = {
    1: {"strategy": "lr_calc", "with_xai": True},
    2: {"strategy": "lr_calc", "with_xai": False},
    3: {"strategy": "dt", "with_xai": True},
    4: {"strategy": "dt", "with_xai": False},
    5: {"strategy": "lr_heur", "with_xai": None},  # XAI follows trial flag
}


class LRForwardEnvironment(BaseRLEnvironment):
    """
    RL environment for LR forward strategies (heuristic and calculation).
    
    Supports feature selection via binary masks and choice among
    5 strategies: LR calc (with/wo XAI), DT (with/wo XAI), LR heuristic.
    
    Action space: MultiDiscrete([6, ...binary mask...])
      - action[0]: strategy ID (0=invalid, 1-5=strategies)
      - action[1:]: feature selection mask (binary per feature)
      
    Observation space:
      [chi_norm, trial_norm, with_xai_flag,
       strategy_counts[5], strategy_success[5],
       feature_contribution_stds[max_features]]
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        condition: str = "Hybrid",  # "LR", "DT", or "Hybrid"
        complexity: str = "low",     # "low" or "high"
    ):
        """
        Initialize LR Forward environment.
        
        Args:
            config: EnvironmentConfig with setup parameters
            condition: Which model type to use in trials
            complexity: Complexity level for trials
        """
        super().__init__(config)
        
        self.condition = str(condition)
        self.complexity = str(complexity)
        
        # Action space: [strategy_id (0-5), feature_mask (binary)]
        self.action_space = spaces.MultiDiscrete(
            [6] + [2] * config.max_features
        )
        
        # Observation space
        # [chi_norm, trial_norm, with_xai,
        #  5*strategy_counts, 5*strategy_success,
        #  max_features*contribution_stds]
        n_obs = 1 + 1 + 1 + 5 + 5 + config.max_features
        
        low = np.zeros(n_obs, dtype=np.float32)
        high = np.ones(n_obs, dtype=np.float32)
        
        # Adjust bounds for counts
        counts_start = 3
        counts_end = 3 + 5
        high[counts_start:counts_end] = float(config.instances_per_episode)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Strategy statistics (5 strategies)
        self.strategy_counts = np.zeros(5, dtype=np.int32)
        self.strategy_success = np.zeros(5, dtype=np.float32)
        
        # Per-trial condition specification
        self.trials: List[Dict[str, Any]] = []
        
        # Feature contribution tracking
        self.contributions = {i: [] for i in range(config.max_features)}
        
        # Current explainers (LR and DT)
        self.lr_exp = None
        self.dt_exp = None
    
    # ========== Helper Methods ==========
    
    def _generate_episode_conditions(self) -> List[Dict[str, Any]]:
        """Generate per-trial condition specifications."""
        rng = self._rng
        n = self.config.instances_per_episode
        n_xai = int(round(n * self.config.xai_trial_ratio))
        
        flags = np.array(
            [True] * n_xai + [False] * (n - n_xai),
            dtype=np.bool_
        )
        rng.shuffle(flags)
        
        if self.condition in ("LR", "DT"):
            trials = [
                {
                    "with_xai": bool(f),
                    "model_type": self.condition,
                    "complexity": self.complexity,
                }
                for f in flags
            ]
        else:  # Hybrid
            n_lr = n // 2
            trials = (
                [
                    {
                        "with_xai": bool(f),
                        "model_type": "LR",
                        "complexity": self.complexity,
                    }
                    for f in flags[:n_lr]
                ]
                + [
                    {
                        "with_xai": bool(f),
                        "model_type": "DT",
                        "complexity": self.complexity,
                    }
                    for f in flags[n_lr:]
                ]
            )
            rng.shuffle(trials)
        
        return trials
    
    def _get_strategy_params(self) -> Dict[str, float]:
        """Get strategy parameters from cognitive params."""
        params = dict(
            T_enc=2.0,
            T_op=0.2,
            ddm_a=1.0,
            ddm_s=1.0,
            ddm_Tnd=0.30,
            ddm_norm="l2",
            compute_sf=2,
            lapse=0.05,
            num_samples=40,
            K_top=3,
        )
        
        for k, v in (self.current_cog_params or {}).items():
            if k in params and isinstance(v, (int, float)):
                params[k] = float(v)
        
        return params
    
    def _update_feature_contributions(self):
        """Update feature contribution tracking."""
        x_raw = self.X_raw[self.step_idx]
        
        for feat in range(min(self.config.max_features, x_raw.shape[-1])):
            # Get factor value (coefficient for this feature)
            factor_value = 0.0
            if (
                self.lr_exp is not None
                and hasattr(self.lr_exp, "coefficients")
            ):
                factor_value = self.lr_exp.coefficients.get(f"a{feat}", 0.0)
            
            self.contributions[feat].append(
                float(x_raw[feat]) * float(factor_value)
            )
    
    def _compute_feature_sums(self) -> np.ndarray:
        """Compute normalized feature contribution standard deviations."""
        feature_stds = np.array(
            [
                np.std(self.contributions[i])
                if len(self.contributions[i]) > 1
                else 0.0
                for i in range(self.config.max_features)
            ],
            dtype=np.float32,
        )
        
        if feature_stds.sum() > 0:
            return feature_stds / feature_stds.sum()
        else:
            return feature_stds
    
    # ========== Abstract Method Implementations ==========
    
    def _initialize_memory(self):
        """Initialize memory for this episode."""
        rt = float(self.current_cog_params.get("retrieval_threshold", 0.5))
        lf = float(self.current_cog_params.get("latency_factor", 3.0))
        
        dm = DeclarativeMemory(
            retrieval_threshold=rt,
            latency_factor=lf,
            latency_exponent=0.5,
            max_assoc_strength=2.0,
            mismatch_penalty=-1.0,
            activation_noise=0.1,
            decay=0.5,
        )
        
        self.memory = CombinedMemory(dm, wm_capacity=10)
        
        # Optionally seed with LR/DT knowledge
        try:
            from src.lr_memory import add_lr_calculation_to_memory
            from src.heuristic_lr_model import add_lr_heuristic_to_memory
            from src.dt_memory import add_dt_to_memory
            
            if self.lr_exp is not None:
                add_lr_calculation_to_memory(self.lr_exp, self.memory)
                add_lr_heuristic_to_memory(
                    self.lr_exp, self.memory, initial_var=1.0
                )
            
            if self.dt_exp is not None:
                add_dt_to_memory(self.memory, self.dt_exp)
        except ImportError:
            pass
        
        self.memory.tick(90)
    
    def _build_obs(self) -> np.ndarray:
        """Build observation vector."""
        if self.step_idx >= self.config.instances_per_episode:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        self._update_feature_contributions()
        with_xai_flag = float(self.with_xai_schedule[self.step_idx])
        
        feature_stds = self._compute_feature_sums()
        
        obs = np.array(
            [
                float(self.curr_chi / max(self.config.chi_high, 1e-9)),
                float(self.step_idx / max(self.config.instances_per_episode, 1)),
                with_xai_flag,
                *self.strategy_counts.tolist(),
                *self.strategy_success.tolist(),
                *feature_stds.tolist(),
            ],
            dtype=np.float32,
        )
        
        return obs
    
    def _run_decision_strategy(
        self, action: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Run selected strategy for current step.
        
        Returns:
            (reward, pred_time, info_dict)
        """
        a = np.asarray(action, dtype=np.int64)
        action_id = int(a[0]) if a.ndim > 0 else int(a)
        mask_bits = a[1:].tolist() if a.ndim > 0 and a.shape[0] > 1 else []
        
        active_indices = [
            i for i, b in enumerate(mask_bits[:self.config.max_features])
            if b == 1
        ]
        
        # Check valid action
        if action_id not in STRATEGY_MAP:
            raise ValueError(f"Invalid action id {action_id}")
        
        # Get trial and strategy info
        trial = self.trials[self.step_idx]
        with_xai_trial = bool(trial["with_xai"])
        
        spec = STRATEGY_MAP[action_id]
        strategy = spec["strategy"]
        with_xai = spec["with_xai"]
        if with_xai is None:  # For lr_heur
            with_xai = with_xai_trial
        
        # Get instances
        x_raw = self.X_raw[self.step_idx]
        x_norm = self.X_norm[self.step_idx]
        y_true = int(self.y[self.step_idx])
        
        # Get strategy params
        SP = self._get_strategy_params()
        
        # Choose explainer
        explainer = (
            self.lr_exp
            if strategy in ("lr_calc", "lr_heur")
            else self.dt_exp
        )
        
        # Run strategy (delegate to actual implementation)
        try:
            from src.lr_memory import (
                lr_calculation,
                refresh_lr_calculation_in_memory,
            )
            from src.heuristic_lr_model import (
                lr_heuristic,
                refresh_lr_heuristic_in_memory,
            )
            from src.dt_memory import (
                dt_traverse,
                refresh_dt_path_in_memory,
            )
            
            if strategy == "lr_calc":
                probs, pred_time, aux = lr_calculation(
                    x_raw,
                    self.memory,
                    lr_exp=explainer,
                    T_enc=SP["T_enc"],
                    T_op=SP["T_op"],
                    ddm_a=SP["ddm_a"],
                    ddm_s=SP["ddm_s"],
                    compute_sf=SP["compute_sf"],
                    mode="read" if with_xai else "retrieve",
                )
            elif strategy == "lr_heur":
                probs, pred_time, aux = lr_heuristic(
                    x_norm,
                    self.memory,
                    explainer,
                    num_samples=int(SP["num_samples"]),
                    K_top=int(SP["K_top"]),
                    T_READ_NUM=SP["T_enc"],
                    T_INTUITIVE_OP=SP["T_op"],
                    ddm_a=SP["ddm_a"],
                    ddm_s=SP["ddm_s"],
                    ddm_Tnd=0.30,
                    ddm_norm="l2",
                    active_indices=None,
                    verbose=False,
                )
            elif strategy == "dt":
                probs, pred_time, aux = dt_traverse(
                    x_raw,
                    self.memory,
                    explainer,
                    mode="read" if with_xai else "retrieve",
                    compute_sf=int(SP["compute_sf"]),
                    T_enc=SP["T_enc"],
                    ddm_a=SP["ddm_a"],
                    ddm_s=SP["ddm_s"],
                    ddm_Tnd=0.30,
                    ddm_norm="l2",
                    n_mc=64,
                    topk_k=3,
                    refresh_prob_cap=1.0,
                    verbose=False,
                )
            
            # Compute reward
            prob_correct = float(probs[y_true])
            if SP["lapse"] > 0.0:
                prob_correct = (
                    (1.0 - SP["lapse"]) * prob_correct + 0.5 * SP["lapse"]
                )
            
            reward = prob_correct - float(self.curr_chi) * float(pred_time)
            
            # Update stats
            idx = action_id - 1  # Map action 1-5 to index 0-4
            self.strategy_counts[idx] += 1
            n = int(self.strategy_counts[idx])
            old = float(self.strategy_success[idx])
            self.strategy_success[idx] = (
                (old * (n - 1) + (1.0 if prob_correct > 0.5 else 0.0))
                / max(n, 1)
            )
            
            # Post-read refresh
            if strategy == "lr_calc" and with_xai:
                refresh_lr_calculation_in_memory(
                    self.memory,
                    explainer,
                    intercept_display_sf=int(SP["compute_sf"]),
                    factor_display_sf=int(SP["compute_sf"]),
                )
            elif strategy == "lr_heur":
                refresh_lr_heuristic_in_memory(
                    self.memory,
                    explainer,
                    aux,
                    actual=int(y_true),
                    active_indices=active_indices,
                    w_min=1e-4,
                    verbose=False,
                )
            elif strategy == "dt" and with_xai:
                refresh_dt_path_in_memory(
                    self.memory,
                    explainer,
                    x_raw,
                    thresh_sf=int(SP["compute_sf"]),
                )
            
            info = {
                "action_id": action_id,
                "strategy": strategy,
                "with_xai_trial": with_xai_trial,
                "active_indices": active_indices,
                "prob_correct": prob_correct,
                "pred_time": float(pred_time),
            }
            
            return reward, float(pred_time), info
            
        except ImportError as e:
            raise RuntimeError(
                f"Required strategy module not available: {e}"
            )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode."""
        # Seed
        if seed is not None:
            self._seed(seed)
        
        # Generate trial conditions
        self.trials = self._generate_episode_conditions()
        
        # Select dataset and explainers
        self.current_ai_dataset_loader, _ = self._select_dataset()
        
        # Get both LR and DT explainers if available
        dataset_key = list(self.config.ai_dataset_loaders.keys())[0]
        explainers = self.config.explainers or {}
        self.lr_exp = explainers.get(f"{dataset_key}_lr", None)
        self.dt_exp = explainers.get(f"{dataset_key}_dt", None)
        # Fallback: try single explainer if available
        if self.current_explainer is not None and self.lr_exp is None:
            self.lr_exp = self.current_explainer
        if self.current_explainer is not None and self.dt_exp is None:
            self.dt_exp = self.current_explainer
        
        # Build with-XAI schedule
        self.with_xai_schedule = self._build_with_xai_schedule(
            self.config.instances_per_episode,
            self.config.xai_trial_ratio,
        )
        
        # Sample cognitive parameters
        self.current_cog_params = self._sample_cog_params()
        
        # Initialize memory
        self._initialize_memory()
        
        # Sample chi
        self.curr_chi = float(
            self._rng.uniform(self.config.chi_low, self.config.chi_high)
        )
        
        # Load instances
        indices = self._rng.choice(
            self.config.instance_id_pool,
            size=self.config.instances_per_episode,
            replace=False,
        ).tolist()
        
        self.X_raw, self.y = self._load_instances(indices, normalize=False)
        self.X_norm, _ = self._load_instances(indices, normalize=True)
        
        # Reset stats
        self.step_idx = 0
        self.strategy_counts[:] = 0
        self.strategy_success[:] = 0
        self.contributions = {i: [] for i in range(self.config.max_features)}
        
        obs = self._build_obs()
        info = {
            "cog_params": self.current_cog_params.copy(),
            "chi": self.curr_chi,
            "condition": self.condition,
            "complexity": self.complexity,
        }
        
        return obs, info
