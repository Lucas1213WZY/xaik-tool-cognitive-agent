"""
RL Agent Utilization API

Provides convenient interfaces for loading and using trained RL agents for:
- Forward reasoning prediction (strategy selection)
- Counterfactual explanation generation

Classes:
- ParticipantParameterLoader: Load per-participant cognitive parameters
- RLAgentPredictor: Base class for agent inference
- RLAgentForwardPredictor: Inference with forward agents
- RLAgentCounterfactualPredictor: Inference with counterfactual agents
"""

import csv
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from src.rl_agents.environments import EnvironmentConfig

logger = logging.getLogger(__name__)


class ParticipantParameterLoader:
    """Load participant cognitive parameters from CSV."""
    
    def __init__(self, param_csv_path: str):
        """
        Initialize loader.
        
        Args:
            param_csv_path: Path to CoXAM_counterfactual_simulation_cog_param.csv
        """
        self.param_csv_path = Path(param_csv_path)
        self.participants = {}
        self._load_params()
    
    def _load_params(self):
        """Load all participant parameters from CSV."""
        if not self.param_csv_path.exists():
            logger.warning(f"Parameter CSV not found: {self.param_csv_path}")
            return
        
        with open(self.param_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['Participant Id']
                self.participants[pid] = {
                    'Best NLL': float(row['Best NLL']),
                    'Best MAE': float(row['Best MAE']),
                    'Best time': float(row['Best time']),
                    'Best retrieval_threshold': float(row['Best retrieval_threshold']),
                    'Best over_margin': float(row['Best over_margin']),
                    'Best chi': float(row['Best chi']),
                    'app_id': row['app_id'],
                    'model': row['model'],
                    'complexity': row['complexity'],
                    'condition': row['condition'],
                }
        
        logger.info(f"Loaded parameters for {len(self.participants)} participants")
    
    def get_params(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get parameters for a specific participant.
        
        Args:
            participant_id: Participant ID
        
        Returns:
            Dict with cognitive parameters, or None if not found
        """
        return self.participants.get(participant_id)
    
    def list_participants(self) -> List[str]:
        """Get list of all loaded participant IDs."""
        return list(self.participants.keys())


class RLAgentPredictor:
    """Base class for RL agent prediction."""
    
    def __init__(self, weights_path: str, env_config: Optional[EnvironmentConfig] = None):
        """
        Initialize predictor.
        
        Args:
            weights_path: Path to best_model.zip weights file
            env_config: Environment configuration (optional)
        """
        self.weights_path = Path(weights_path)
        self.env_config = env_config or EnvironmentConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained PPO model."""
        if not self.weights_path.exists():
            logger.warning(f"Weights file not found: {self.weights_path}")
            return
        
        try:
            self.model = PPO.load(str(self.weights_path))
            logger.info(f"Loaded model from {self.weights_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make prediction using the loaded model.
        
        Args:
            observation: Input observation array
            deterministic: Whether to use deterministic policy
        
        Returns:
            (action, state) tuple from model prediction
        """
        if self.model is None:
            logger.warning("Model not loaded, returning None")
            return None, None
        
        try:
            action, state = self.model.predict(observation, deterministic=deterministic)
            return action, state
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None


class RLAgentForwardPredictor(RLAgentPredictor):
    """
    Predictor for forward reasoning strategy selection.
    
    Uses trained DT/LR forward agents to predict:
    - Strategy choice (read/retrieve/...)
    - Parameter settings (e.g., ddm_a bin)
    """
    
    def predict_strategy(self, observation: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict strategy selection and parameters.
        
        Args:
            observation: Environment observation
        
        Returns:
            Dict with strategy_id, strategy_name, parameters, or None on error
        """
        action, _ = self.predict(observation, deterministic=True)
        
        if action is None:
            return None
        
        # Parse action based on forward environment format
        # Typically: [strategy_id, parameter_bin]
        action = np.asarray(action).flatten()
        
        return {
            'action': action,
            'strategy_id': int(action[0]) if len(action) > 0 else 1,
            'param_bin': int(action[1]) if len(action) > 1 else 0,
        }


class RLAgentCounterfactualPredictor(RLAgentPredictor):
    """
    Predictor for counterfactual explanation generation.
    
    Uses trained counterfactual agents to predict:
    - Feature to change
    - Change magnitude
    """
    
    def predict_counterfactual(self, observation: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict counterfactual feature change.
        
        Args:
            observation: Environment observation (instance features + metadata)
        
        Returns:
            Dict with feature_idx, magnitude_bin, details, or None on error
        """
        action, _ = self.predict(observation, deterministic=True)
        
        if action is None:
            return None
        
        # Parse action: [feature_idx, magnitude_bin]
        action = np.asarray(action).flatten()
        
        return {
            'action': action,
            'feature_idx': int(action[0]) if len(action) > 0 else 0,
            'magnitude_bin': int(action[1]) if len(action) > 1 else 0,
            'description': f"Change feature {int(action[0]) if len(action) > 0 else 0} with magnitude bin {int(action[1]) if len(action) > 1 else 0}"
        }


class TrialCounterfactualGenerator:
    """Generate counterfactual predictions for trials."""
    
    def __init__(
        self,
        dt_weights_path: str,
        lr_weights_path: Optional[str] = None,
        param_csv_path: Optional[str] = None
    ):
        """
        Initialize generator.
        
        Args:
            dt_weights_path: Path to DT counterfactual weights
            lr_weights_path: Path to LR counterfactual weights (optional)
            param_csv_path: Path to participant parameters CSV
        """
        self.dt_predictor = RLAgentCounterfactualPredictor(dt_weights_path)
        self.lr_predictor = RLAgentCounterfactualPredictor(lr_weights_path) if lr_weights_path else None
        self.param_loader = ParticipantParameterLoader(param_csv_path) if param_csv_path else None
    
    def generate_for_trial(
        self,
        trial_data: Dict[str, Any],
        participant_id: str,
        strategy: str = "dt"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate counterfactual prediction for a single trial.
        
        Args:
            trial_data: Trial metadata dict (usually from CSV row)
            participant_id: ID of participant
            strategy: "dt" or "lr"
        
        Returns:
            Dict with prediction results, or None on error
        """
        # Load participant parameters
        if self.param_loader:
            params = self.param_loader.get_params(participant_id)
            if params is None:
                logger.warning(f"No parameters found for participant {participant_id}")
                return None
        else:
            params = {}
        
        # Construct observation from trial data
        # This is simplified - in practice you'd extract features from trial_data
        n_features = 6  # Default
        observation = np.zeros(n_features + 3, dtype=np.float32)  # features + metadata
        observation[-1] = 1.0  # with_xai flag
        
        # Get predictor based on strategy
        predictor = self.dt_predictor if strategy == "dt" else self.lr_predictor
        if predictor is None:
            logger.warning(f"No predictor available for strategy {strategy}")
            return None
        
        # Predict counterfactual
        cf_pred = predictor.predict_counterfactual(observation)
        
        if cf_pred is None:
            return None
        
        return {
            'participant_id': participant_id,
            'strategy': strategy,
            'feature_idx': cf_pred['feature_idx'],
            'magnitude_bin': cf_pred['magnitude_bin'],
            'params': params,
            'prediction': cf_pred,
        }
