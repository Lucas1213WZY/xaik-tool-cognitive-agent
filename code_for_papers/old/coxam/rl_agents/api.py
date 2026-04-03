"""
COXAM RL Agent API - Unified interface for trial generation

Provides high-level functions for generating forward and counterfactual trials
with configurable experimental design parameters.
"""

from typing import Dict, Any, List, Optional, Sequence
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExperimentalDesign:
    """
    Encapsulates experimental design configuration.
    
    Specifies how to generate trials across conditions, XAI types, and phases.
    """
    
    def __init__(
        self,
        n_trials_per_condition: int = 40,
        conditions: List[str] = None,
        xai_types: List[str] = None,
        phases: List[str] = None,
        with_xai_prob: float = 0.5,
    ):
        """
        Initialize experimental design.
        
        Args:
            n_trials_per_condition: Trials per condition combination
            conditions: XAI conditions to test ['DT', 'LR', 'DT+LR']
            xai_types: XAI types ['DT', 'LR', 'DT+LR']
            phases: Phases ['forward', 'counterfactual'] or ['train', 'test']
            with_xai_prob: Probability of showing XAI explanations
        """
        self.n_trials = n_trials_per_condition
        self.conditions = conditions or ['DT', 'LR', 'DT+LR']
        self.xai_types = xai_types or ['DT', 'LR']
        self.phases = phases or ['forward']
        self.with_xai_prob = with_xai_prob
    
    def generate_configs(
        self,
        app_id: str,
        model_name: str,
        complexity: str,
        cog_params_template: Dict[str, float],
        deterministic: bool = True,
        rng_seed: int = 123,
    ) -> List[Dict[str, Any]]:
        """
        Generate trial configurations for all condition combinations.
        
        Args:
            app_id: Application ID
            model_name: Model name
            complexity: Complexity level
            cog_params_template: Base cognitive parameters (will use chi from this)
            deterministic: Deterministic agent policy
            rng_seed: Random seed
        
        Returns:
            List of config dicts for simulator.generate_forward_trials()
        """
        from .coxam_forward_simulator import TrialScheduleGenerator
        
        configs = []
        rng = np.random.default_rng(rng_seed)
        
        for condition in self.conditions:
            for phase in self.phases:
                # Generate trial schedule
                schedule = TrialScheduleGenerator.balanced_xai_schedule(
                    n_trials=self.n_trials,
                    condition=condition,
                    rng_seed=int(rng.integers(0, 2**31)),
                )
                
                configs.append({
                    "app_id": app_id,
                    "model_name": model_name,
                    "condition": condition,
                    "complexity": complexity,
                    "trial_schedule": schedule,
                    "cog_params": cog_params_template,
                    "deterministic": deterministic,
                    "rng_seed": int(rng.integers(0, 2**31)),
                })
        
        logger.info(f"Generated {len(configs)} configurations")
        return configs


class RLAgentTrialGenerator:
    """
    High-level API for generating trials using RL agents.
    
    Orchestrates trial generation across experimental designs.
    """
    
    def __init__(
        self,
        simulator: "COXAMForwardSimulator",
        ai_dataset_loader: Any,
    ):
        """
        Initialize generator.
        
        Args:
            simulator: COXAMForwardSimulator instance
            ai_dataset_loader: AIDatasetLoader for loading instances
        """
        self.simulator = simulator
        self.ai_dataset_loader = ai_dataset_loader
    
    def generate_trials_from_design(
        self,
        design: ExperimentalDesign,
        app_id: str,
        model_name: str,
        complexity: str,
        cog_params: Dict[str, float],
        output_path: Optional[str] = None,
        deterministic: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all trials for an experimental design.
        
        Args:
            design: ExperimentalDesign instance
            app_id: Application ID
            model_name: Model name
            complexity: Complexity level
            cog_params: Cognitive parameters
            output_path: Optional path to save results
            deterministic: Deterministic policy
        
        Returns:
            Combined DataFrame with all generated trials
        """
        configs = design.generate_configs(
            app_id=app_id,
            model_name=model_name,
            complexity=complexity,
            cog_params_template=cog_params,
            deterministic=deterministic,
        )
        
        # Generate all trials
        df = self.simulator.generate_batch_trials(
            configs=configs,
            output_path=output_path,
        )
        
        return df
    
    def generate_participant_trials(
        self,
        n_participants: int,
        n_trials_per_participant: int,
        app_id: str,
        model_name: str,
        condition: str,
        complexity: str,
        cog_params: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate trials for multiple synthetic participants.
        
        Args:
            n_participants: Number of synthetic participants
            n_trials_per_participant: Trials per participant
            app_id: Application ID
            model_name: Model name
            condition: Condition
            complexity: Complexity
            cog_params: Cognitive parameters (same for all)
            output_path: Optional output path
        
        Returns:
            DataFrame with all participant trials
        """
        from .coxam_forward_simulator import TrialScheduleGenerator
        
        all_dfs = []
        rng = np.random.default_rng()
        
        for p in range(n_participants):
            logger.info(f"Generating trials for participant {p+1}/{n_participants}")
            
            schedule = TrialScheduleGenerator.balanced_xai_schedule(
                n_trials=n_trials_per_participant,
                condition=condition,
                rng_seed=int(rng.integers(0, 2**31)),
            )
            
            df = self.simulator.generate_forward_trials(
                app_id=app_id,
                model_name=model_name,
                condition=condition,
                complexity=complexity,
                trial_schedule=schedule,
                cog_params=cog_params,
                deterministic=True,
                rng_seed=int(rng.integers(0, 2**31)),
            )
            
            all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)
            logger.info(f"✓ Saved {len(combined)} trials to {output_path}")
        
        return combined


# Example configuration templates
STANDARD_COG_PARAMS = {
    "T_enc": 1.5,
    "T_op": 0.3,
    "latency_factor": 0.2,
    "ddm_a": 1.2,
    "ddm_s": 0.9,
    "lapse": 0.04,
    "retrieval_threshold": 0.0,
    "chi": 0.01,  # Time cost sensitivity
}

EXPERIMENTAL_DESIGNS = {
    "basic": ExperimentalDesign(
        n_trials_per_condition=40,
        conditions=['DT', 'LR', 'DT+LR'],
        phases=['forward'],
    ),
    "balanced": ExperimentalDesign(
        n_trials_per_condition=50,
        conditions=['DT', 'LR'],
        xai_types=['DT', 'LR'],
        phases=['forward'],
        with_xai_prob=0.5,
    ),
    "full": ExperimentalDesign(
        n_trials_per_condition=100,
        conditions=['DT', 'LR', 'DT+LR'],
        xai_types=['DT', 'LR'],
        phases=['forward', 'counterfactual'],
        with_xai_prob=0.5,
    ),
}
