"""
Decision Tree Forward Strategy RL Environment.

Extracted and consolidated from RL_feature_selection_agents_v0.5 notebook.
Agent chooses between read/retrieve modes and DDM-a parameter bins.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from gymnasium import spaces

from .base_env import BaseRLEnvironment, EnvironmentConfig
from code_for_papers.old.coxam.src.memory import DeclarativeMemory, CombinedMemory


# Mode mapping for DT strategy
DT_MODES = {1: "read", 2: "retrieve"}


class DTForwardEnvironment(BaseRLEnvironment):
    """
    RL environment for Decision Tree forward strategy.
    
    Action space: MultiDiscrete([3, ddm_a_bins])
      - action[0]: strategy mode (0=invalid, 1=read, 2=retrieve)
      - action[1]: DDM-a bin index
      
    Observation space (compact):
      [chi_norm, trial_norm, with_xai_flag,
       count_read, count_retrieve,
       succ_read, succ_retrieve]
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        ddm_a_bins: int = 3,
        ddm_a_min: float = 0.6,
        ddm_a_max: float = 1.7,
    ):
        """
        Initialize DT Forward environment.
        
        Args:
            config: EnvironmentConfig with setup parameters
            ddm_a_bins: Number of bins for quantizing DDM-a parameter
            ddm_a_min: Lower bound for DDM-a range
            ddm_a_max: Upper bound for DDM-a range
        """
        super().__init__(config)
        
        # DDM-a binning
        self.ddm_a_bins = int(ddm_a_bins)
        self.ddm_a_min = float(ddm_a_min)
        self.ddm_a_max = float(ddm_a_max)
        
        # Action space: [strategy_id (0-2), ddm_a_bin (0 to ddm_a_bins-1)]
        self.action_space = spaces.MultiDiscrete([3, self.ddm_a_bins])
        
        # Observation space (compact)
        # [chi_norm, trial_norm, with_xai, count_read, count_retrieve, succ_read, succ_retrieve]
        low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )
        high = np.array(
            [
                1.0,  # chi_norm
                1.0,  # trial_norm
                1.0,  # with_xai_flag
                float(config.instances_per_episode),  # count_read
                float(config.instances_per_episode),  # count_retrieve
                1.0,  # succ_read (rate)
                1.0,  # succ_retrieve (rate)
            ],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Strategy statistics (read=0, retrieve=1)
        self.strategy_counts = np.zeros(2, dtype=np.int32)
        self.strategy_success = np.zeros(2, dtype=np.float32)
        
        # Current DT explainer (will be set during reset)
        self.dt_exp = None
    
    # ========== Helper Methods ==========
    
    def _ddm_a_from_bin(self, bin_idx: int) -> float:
        """Convert DDM-a bin index to continuous value."""
        b = int(np.clip(bin_idx, 0, self.ddm_a_bins - 1))
        
        if self.ddm_a_bins == 1 or self.ddm_a_min == self.ddm_a_max:
            return float(self.ddm_a_min)
        
        frac = (b + 0.5) / self.ddm_a_bins
        return float(
            self.ddm_a_min + frac * (self.ddm_a_max - self.ddm_a_min)
        )
    
    def _get_strategy_params(self) -> Dict[str, float]:
        """Get strategy parameters from cognitive params."""
        params = dict(
            T_enc=2.0,
            ddm_a=1.0,
            ddm_s=1.0,
            ddm_Tnd=0.30,
            ddm_norm="l2",
            compute_sf=2,
            lapse=0.05,
        )
        
        # Override with current cognitive parameters
        for k, v in (self.current_cog_params or {}).items():
            if k in params and isinstance(v, (int, float)) and k != "ddm_a":
                params[k] = float(v)
        
        return params
    
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
            mismatch_penalty=-2.0,
            activation_noise=0.3,
            decay=0.5,
        )
        
        self.memory = CombinedMemory(dm, wm_capacity=7)
        
        # Import DT memory initialization if available
        try:
            from src.dt_memory import add_dt_to_memory
            if self.dt_exp is not None:
                add_dt_to_memory(self.memory, self.dt_exp)
        except ImportError:
            pass
        
        self.memory.tick(90)
    
    def _build_obs(self) -> np.ndarray:
        """Build compact observation vector."""
        if self.step_idx >= self.config.instances_per_episode:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        with_xai_flag = float(self.with_xai_schedule[self.step_idx])
        
        obs = np.array(
            [
                float(self.curr_chi / max(self.config.chi_high, 1e-9)),
                float(self.step_idx / max(self.config.instances_per_episode, 1)),
                with_xai_flag,
                float(self.strategy_counts[0]),
                float(self.strategy_counts[1]),
                float(self.strategy_success[0]),
                float(self.strategy_success[1]),
            ],
            dtype=np.float32,
        )
        
        return obs
    
    def _run_decision_strategy(
        self, action: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Run DT forward strategy for current step.
        
        Returns:
            (reward, pred_time, info_dict)
        """
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        
        if a.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {a.shape}")
        
        action_id = int(a[0])  # 0..2
        ddm_a_bin = int(a[1])  # 0..B-1
        
        with_xai_trial = bool(self.with_xai_schedule[self.step_idx])
        
        # Check valid strategy
        if action_id not in DT_MODES:
            self.step_idx += 1
            raise ValueError(f"Invalid action id {action_id} (not in {list(DT_MODES.keys())})")
        
        chosen_mode = DT_MODES[action_id]
        
        # Guard: no "read" without XAI
        if (not with_xai_trial) and (chosen_mode == "read"):
            raise ValueError("Cannot read without XAI trial availability")
        
        # Get current instance
        x_raw = self.X_raw[self.step_idx]
        y_true = int(self.y[self.step_idx])
        
        # Get strategy params
        SP = self._get_strategy_params()
        SP["ddm_a"] = float(self._ddm_a_from_bin(ddm_a_bin))
        
        # Run DT forward strategy
        # NOTE: This calls the actual dt_traverse from src.dt_memory
        try:
            from src.dt_memory import dt_traverse, refresh_dt_path_in_memory
            
            probs, pred_time, aux = dt_traverse(
                x_raw,
                self.memory,
                self.dt_exp,
                mode=chosen_mode,
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
            
            # Update success stats
            idx = 0 if (chosen_mode == "read") else 1
            self.strategy_counts[idx] += 1
            n = int(self.strategy_counts[idx])
            old = float(self.strategy_success[idx])
            self.strategy_success[idx] = (
                (old * (n - 1) + (1.0 if prob_correct > 0.5 else 0.0))
                / max(n, 1)
            )
            
            # Post-read refresh
            if with_xai_trial:
                refresh_dt_path_in_memory(
                    self.memory, self.dt_exp, x_raw, thresh_sf=int(SP["compute_sf"])
                )
            
            info = {
                "chosen_mode": chosen_mode,
                "with_xai_trial": with_xai_trial,
                "prob_correct": prob_correct,
                "pred_time": float(pred_time),
                "ddm_a": float(SP["ddm_a"]),
                "ddm_a_bin": int(ddm_a_bin),
            }
            
            return reward, float(pred_time), info
            
        except ImportError:
            raise RuntimeError(
                "dt_traverse function not available from src.dt_memory. "
                "Ensure dt_memory module is properly set up."
            )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode."""
        # Call parent reset first
        obs, info = super().reset(seed=seed, options=options)
        
        # Additionally set DT explainer
        self.dt_exp = self.current_explainer
        
        # Reset strategy statistics
        self.strategy_counts[:] = 0
        self.strategy_success[:] = 0.0
        
        return obs, info
