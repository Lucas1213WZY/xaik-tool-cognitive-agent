"""
Unified Counterfactual Reasoning Environment

Single environment for counterfactual explanation generation supporting both DT and LR strategies.

This environment learns to select feature changes that flip model predictions (generate counterfactuals).

Action Space:
- MultiDiscrete([5, 3]): 
  - a[0] ∈ {0..4}: Strategy selection (5 strategies)
    - 0: change_path_dt
    - 1: zero_out_lr_heuristic
    - 2: zero_out_lr_displayed
    - 3: recall_change_dt
    - 4: recall_change_lr
  - a[1] ∈ {0..2}: Depth parameter for DT strategy

Observation Space:
- Box containing: [chi, step_idx, with_xai, xai_type, xai_type_shown, 
                   counts_per_strategy, success_rates, mean_times, varied_cog_params]

Reward:
- success - chi * time_cost where success=1 if prediction flips, else 0
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from gymnasium import spaces
import logging
from src.cognitive_models.counterfactual.coxam_counterfactual_rs import (
    ZeroOutLRHeuristic,
    ZeroOutLRDisplayed,
    ChangeDTPath,
    RecallChanges,
    MemoryBasedCF
)
from src.cognitive_models.interface import StrategyConfig

logger = logging.getLogger(__name__)

# Strategy constants
STRATEGIES = {
    0: "change_path_dt",
    1: "zero_out_lr_heuristic",
    2: "zero_out_lr_displayed",
    3: "recall_change_dt",
    4: "recall_change_lr",
}

XAI_TYPES = {
    'DT': 'DT',
    'LR': 'LR', 
    'DT+LR': 'DT+LR',
}


class CounterfactualEnv:
    """
    Unified counterfactual explanation environment supporting multiple strategies.
    
    This is a placeholder that documents the correct interface. The actual implementation
    should import strategy functions and memory management from the codebase.
    
    Key differences from DTForwardEnvironment/LRForwardEnvironment:
    - Single unified class (not separate per-strategy)
    - Supports 5 strategies in action space
    - Manages forward_trials and counterfactual_trials separately
    - Success metric: did prediction flip?
    """
    
    def __init__(
        self,
        ai_dataset_loaders: Dict[str, Any],
        ais: Dict[str, Any],
        transforms: Dict[str, Any],
        lr_exps: Dict[str, Any],
        dt_exps: Dict[str, Any],
        cog_params: Dict[str, Any],
        instances_per_episode: int = 40,
        max_features: int = 6,
        eval_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize unified counterfactual environment.
        
        Args:
            ai_dataset_loaders: Dict mapping dataset_id -> loader
            ais: Dict mapping dataset_id -> AI model
            transforms: Dict mapping dataset_id -> feature transformer
            lr_exps: Dict mapping dataset_id -> LR explainer
            dt_exps: Dict mapping dataset_id -> DT explainer
            cog_params: Cognitive parameters config with (lo, hi) ranges
            instances_per_episode: Trials per episode
            max_features: Max features in dataset
            eval_overrides: Override settings for evaluation
        """
        self.ai_dataset_loaders = ai_dataset_loaders
        self.ais = ais
        self.transforms = transforms
        self.lr_exps = lr_exps
        self.dt_exps = dt_exps
        self.cog_params = cog_params
        self.instances_per_episode = instances_per_episode
        self.max_features = max_features
        self.eval_overrides = eval_overrides or {}
        
        # Action space: 5 strategies + depth parameter
        self.action_space = spaces.MultiDiscrete([5, 3])
        
        # Extract chi range
        self.chi_low, self.chi_high = cog_params.get('chi', (0.0, 1.0))
        
        # Extract other cognitive parameters
        self.varied_cogparams_low = []
        self.varied_cogparams_high = []
        self.varied_param_names = []
        for k, v in cog_params.items():
            if k == "chi":
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2:
                self.varied_param_names.append(k)
                self.varied_cogparams_low.append(float(v[0]))
                self.varied_cogparams_high.append(float(v[1]))
        
        # Observation space dimensions
        n_strategies = len(STRATEGIES)
        base_obs_dim = 5 + 3 * n_strategies  # chi, step, with_xai, xai_type, xai_type_shown + per-strategy stats
        total_obs_dim = base_obs_dim + len(self.varied_param_names)
        
        low_vec = [self.chi_low, 0.0, 0.0, 0.0, 0.0] + [0.0] * 3 * n_strategies + [0.0] * len(self.varied_param_names)
        high_vec = [
            self.chi_high, 
            float(instances_per_episode - 1), 
            1.0, 
            float(len(XAI_TYPES) - 1), 
            float(len(XAI_TYPES) - 1)
        ] + [float(instances_per_episode), 1.0, 30.0] * n_strategies + [1.0] * len(self.varied_param_names)
        
        self.observation_space = spaces.Box(
            low=np.array(low_vec, dtype=np.float32),
            high=np.array(high_vec, dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode state
        self.step_idx = 0
        self.curr_chi = 0.0
        self.current_cog_params = {}
        self.forward_trials = []
        self.counterfactual_trials = []
        self.with_xai_schedule = []
        self.xai_schedule = []
        self.xai_type = None
        self.xai_type_shown = None
        
        # Per-strategy tracking
        self.counts = {k: 0 for k in STRATEGIES.keys()}
        self.success_rates = {k: 0.0 for k in STRATEGIES.keys()}
        self.mean_times = {k: 0.0 for k in STRATEGIES.keys()}
        
        # Initialize reasoning strategies
        self._initialize_strategies()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset episode.
        
        Returns:
            (observation, info)
        """
        rng = np.random.default_rng(seed)
        
        # Sample dataset
        self.key = rng.choice(list(self.ai_dataset_loaders.keys()))
        self.ai_dataset_loader = self.ai_dataset_loaders[self.key]
        self.ai = self.ais[self.key]
        self.transform = self.transforms[self.key]
        self.lr_exp = self.lr_exps[self.key]
        self.dt_exp = self.dt_exps[self.key]
        self.app_id = self.lr_exp.app_id
        
        # Sample XAI condition
        force_xai_type = self.eval_overrides.get("xai_type", None)
        if force_xai_type:
            self.xai_type = force_xai_type
        else:
            self.xai_type = rng.choice(list(XAI_TYPES.keys()))
        
        # Set XAI schedule
        if XAI_TYPES[self.xai_type] == 'DT':
            self.xai_schedule = ['DT'] * self.instances_per_episode
        elif XAI_TYPES[self.xai_type] == 'LR':
            self.xai_schedule = ['LR'] * self.instances_per_episode
        else:
            self.xai_schedule = rng.choice(['DT', 'LR'], size=self.instances_per_episode, p=[0.5, 0.5]).tolist()
        
        if "xai_schedule" in self.eval_overrides:
            self.xai_schedule = list(self.eval_overrides["xai_schedule"])
        
        # With-XAI schedule
        with_xai_prob = float(self.eval_overrides.get("with_xai_prob", 0.5))
        self.with_xai_schedule = rng.choice([0, 1], size=self.instances_per_episode, p=[1.0 - with_xai_prob, with_xai_prob])
        
        if "with_xai_schedule" in self.eval_overrides:
            self.with_xai_schedule = np.asarray(self.eval_overrides["with_xai_schedule"], dtype=int)
        
        # Initialize forward and counterfactual trials
        forward_ids = rng.choice(range(400), size=self.instances_per_episode, replace=False)
        counterfactual_ids = rng.choice(range(400), size=self.instances_per_episode, replace=False)
        
        self.forward_trials = [
            {'Tested w/ XAI': int(self.with_xai_schedule[i]), 'Instance Id': int(forward_ids[i])}
            for i in range(self.instances_per_episode)
        ]
        self.counterfactual_trials = [
            {'Tested w/ XAI': int(self.with_xai_schedule[i]), 'Instance Id': int(counterfactual_ids[i])}
            for i in range(self.instances_per_episode)
        ]
        
        # Sample cognitive parameters
        force_cog = self.eval_overrides.get("cog_params_fixed", None)
        self.current_cog_params = {}
        if force_cog:
            for k, v in force_cog.items():
                self.current_cog_params[k] = float(v)
            if "chi" in force_cog:
                self.curr_chi = float(force_cog["chi"])
        else:
            for k, v in (self.cog_params or {}).items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    self.current_cog_params[k] = float(rng.uniform(v[0], v[1]))
                elif isinstance(v, (int, float)):
                    self.current_cog_params[k] = float(v)
        
        self.step_idx = 0
        if force_cog is None or "chi" not in force_cog:
            self.curr_chi = float(rng.uniform(self.chi_low, self.chi_high))
        
        # Reset tracking
        self.counts = {k: 0 for k in STRATEGIES.keys()}
        self.success_rates = {k: 0.0 for k in STRATEGIES.keys()}
        self.mean_times = {k: 0.0 for k in STRATEGIES.keys()}
        self.xai_type_shown = self.xai_schedule[0]
        
        obs = self._build_obs()
        return obs, {}
    
    def step(self, action: Union[int, np.ndarray, List[int]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: [strategy_id, depth]
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action).flatten().astype(int)
        strategy_id = action[0] % len(STRATEGIES)
        depth = action[1] % 3 if len(action) > 1 else 0
        
        strategy_name = STRATEGIES[strategy_id]
        
        # Get trial and check validity
        trial = self.counterfactual_trials[int(self.step_idx)]
        with_xai = trial['Tested w/ XAI']
        self.xai_type_shown = self.xai_schedule[int(self.step_idx)]
        
        # Check if strategy is valid for condition
        invalid_strategy = self._is_invalid_strategy(strategy_name, with_xai)
        
        if invalid_strategy:
            self.step_idx += 1
            truncated = self.step_idx >= self.instances_per_episode
            obs = self._build_obs()
            return obs, -1.0, False, truncated, {
                "error": "invalid_strategy",
                "strategy": strategy_name,
                "success": 0,
                "time": 0.0,
            }
        
        # Execute strategy (simplified - real implementation calls actual strategy functions)
        success, time_cost = self._execute_strategy(strategy_name, depth, trial, with_xai)
        
        # Update tracking
        self.counts[strategy_id] += 1
        n = self.counts[strategy_id]
        self.success_rates[strategy_id] = ((self.success_rates[strategy_id] * (n - 1)) + success) / n
        self.mean_times[strategy_id] = ((self.mean_times[strategy_id] * (n - 1)) + time_cost) / n
        
        # Reward
        reward = float(success) - time_cost * self.curr_chi
        
        self.step_idx += 1
        truncated = self.step_idx >= self.instances_per_episode
        
        obs = self._build_obs()
        info = {
            "strategy": strategy_name,
            "success": success,
            "time": time_cost,
            "reward": reward,
        }
        
        return obs, reward, False, truncated, info
    
    def _build_obs(self) -> np.ndarray:
        """Build observation vector."""
        obs = [
            self.curr_chi,
            float(self.step_idx),
            float(self.with_xai_schedule[min(int(self.step_idx), len(self.with_xai_schedule) - 1)]),
            float(list(XAI_TYPES.values()).index(XAI_TYPES[self.xai_type]) if self.xai_type in XAI_TYPES else 0),
            float(list(XAI_TYPES.values()).index(self.xai_type_shown) if self.xai_type_shown in XAI_TYPES.values() else 0),
        ]
        
        for k in STRATEGIES.keys():
            obs += [
                float(self.counts[k]),
                float(self.success_rates[k]),
                float(self.mean_times[k]),
            ]
        
        for name in self.varied_param_names:
            obs.append(float(self.current_cog_params.get(name, 0.0)))
        
        return np.array(obs, dtype=np.float32)
    
    def _is_invalid_strategy(self, strategy_name: str, with_xai: int) -> bool:
        """Check if strategy is valid for current condition."""
        if XAI_TYPES[self.xai_type] == 'DT':
            if strategy_name in ['zero_out_lr_displayed', 'zero_out_lr_heuristic', 'recall_change_lr']:
                return True
        elif XAI_TYPES[self.xai_type] == 'LR':
            if strategy_name in ['change_path_dt', 'recall_change_dt']:
                return True
            if strategy_name == 'zero_out_lr_displayed' and with_xai == 0:
                return True
        return False
    
    def _initialize_strategies(self):
        """Initialize all reasoning strategy instances."""
        # Create strategy config template
        def make_config(strategy_name: str, **extra_params):
            return StrategyConfig(
                name=strategy_name,
                extra_params=extra_params or {},
                decay_param=0.5,
                time_manager=None,  # Will use default timing
            )
        
        # Initialize strategy instances
        self.strategies = {
            "change_path_dt": ChangeDTPath(make_config("change_path_dt", n_splits=2)),
            "zero_out_lr_heuristic": ZeroOutLRHeuristic(make_config("zero_out_lr_heuristic", k=2)),
            "zero_out_lr_displayed": ZeroOutLRDisplayed(make_config("zero_out_lr_displayed", k=2)),
            "recall_change_dt": RecallChanges(make_config("recall_change_dt")),
            "recall_change_lr": MemoryBasedCF(make_config("recall_change_lr", k=3)),
        }
    
    def _execute_strategy(self, strategy_name: str, depth: int, trial: Dict[str, Any], with_xai: int) -> Tuple[int, float]:
        """
        Execute strategy using reasoning strategy and return (success, time_cost).
        
        Calls the appropriate reasoning strategy, applies suggested changes to features,
        and checks if the model prediction flips.
        """
        try:
            # Load instances
            instance_id = trial['Instance Id']
            instances, preds = self.ai_dataset_loader.load_instances([instance_id], normalize=False)
            if not instances or len(instances) == 0:
                return 0, 0.5
            
            instance = instances[0]
            old_pred = preds[0]
            
            # Get strategy
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                logger.warning(f"Strategy {strategy_name} not found")
                return 0, 0.5
            
            # Build explanation (using None for placeholder)
            explanation = None
            
            # Call strategy's suggest_change
            suggested = strategy.suggest_change(
                features=instance,
                explanation=explanation,
                current_prediction=old_pred,
                target_label=1 - old_pred,  # Flip prediction
            )
            
            # Extract suggested features
            new_instance = suggested.get('suggested_features', instance.copy())
            
            # Run prediction on new instance
            try:
                prepared = self.transform.prepare_instances_for_model(new_instance, one_hot_encode=True)
                preds_cf = self.ai.predict(prepared)
                new_pred = int(np.argmax(preds_cf[0]))
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
                new_pred = old_pred
            
            # Check success (did prediction flip?)
            success = 1 if new_pred != old_pred else 0
            
            # Estimate time based on strategy complexity
            time_cost = self._estimate_strategy_time(strategy_name, depth)
            
            # Update strategy memory with feedback
            try:
                strategy.feedback(
                    features=new_instance,
                    true_label=new_pred,
                )
            except Exception as e:
                logger.debug(f"Strategy feedback error: {e}")
            
            return success, time_cost
        
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            return 0, 0.5
    
    def _estimate_strategy_time(self, strategy_name: str, depth: int) -> float:
        """
        Estimate time cost for strategy execution.
        
        Args:
            strategy_name: Name of the strategy
            depth: Depth parameter (for DT strategies)
        
        Returns:
            Estimated time in seconds
        """
        # Base times for each strategy
        base_times = {
            "change_path_dt": 0.2 + depth * 0.1,  # Deeper trees take longer
            "zero_out_lr_heuristic": 0.3,
            "zero_out_lr_displayed": 0.25,
            "recall_change_dt": 0.4,
            "recall_change_lr": 0.35,
        }
        
        base_time = base_times.get(strategy_name, 0.3)
        
        # Add lapse/variability
        lapse = self.current_cog_params.get('lapse', 0.1)
        variability = np.random.normal(0, lapse * 0.1)
        
        return max(0.1, base_time + variability)
