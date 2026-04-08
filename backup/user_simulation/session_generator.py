"""
Session Generator - Generate full sessions of synthetic participant data.

High-level API for generating complete synthetic datasets with:
- Multiple participants with customizable strategy distribution
- Configurable XAI explanations and explainer models
- Sensible defaults from parameter distributions
- Batch simulation and export
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from .distribution_loader import DistributionLoader
from .parameter_sampler import ParameterSampler
from .trial_simulator import TrialSimulator, TrialConfig, TrialResult
# Sequential IDs are generated in this module; utility ID generator remains
# available for standalone use in other workflows.

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a specific strategy in session generation."""
    
    strategy_name: str
    percentage: float  # Percentage of participants using this strategy (0-100)
    xai_type: str = "importance"  # "importance", "attribution", or "none"
    tested_with_xai: bool = True  # Whether participants see explanations
    cognitive_params: Optional[Dict[str, float]] = None  # Override default params
    explainer: Optional[Any] = None  # Explanation model (e.g., DecisionTreeInterpreter)
    
    def __post_init__(self):
        if not 0 <= self.percentage <= 100:
            raise ValueError(f"percentage must be 0-100, got {self.percentage}")


@dataclass
class SessionConfig:
    """Configuration for generating a full session of synthetic participants."""
    
    # Dataset and basic setup
    dataset_name: str  # "adult", "wine_quality", "forest_cover", etc.
    n_participants: int = 10
    n_trials_per_participant: int = 40
    
    # Strategy distribution
    strategy_configs: List[StrategyConfig] = field(default_factory=list)
    
    # Data sources
    ai_dataset_loader: Optional[Any] = None  # AIDatasetLoader instance
    distribution_file: Optional[str] = None  # Path to distributions JSON
    
    # Generation settings
    sampling_method: str = "truncated_normal"  # "normal", "uniform", "truncated_normal"
    random_seed: Optional[int] = None
    # By default, participant IDs are sequential: "0", "1", ..., "n-1".
    # Set a prefix (e.g., "p") to produce "p0", "p1", ...
    participant_id_prefix: Optional[str] = None
    
    # XAI global settings
    xai_explainer: Optional[Any] = None  # Single explainer for all strategies (can override per-strategy)
    
    def __post_init__(self):
        if not self.strategy_configs:
            raise ValueError("At least one strategy_config must be provided")
        
        total_percentage = sum(sc.percentage for sc in self.strategy_configs)
        if not (99.9 <= total_percentage <= 100.1):  # Allow small floating point error
            logger.warning(f"Strategy percentages sum to {total_percentage}% (not 100%)")


class SessionGenerator:
    """
    Generate complete synthetic participant sessions.
    
    Usage:
        generator = SessionGenerator()
        
        config = SessionConfig(
            dataset_name="wine_quality",
            n_participants=25,
            n_trials_per_participant=40,
            distribution_file="distributions.json",
            strategy_configs=[
                StrategyConfig(
                    strategy_name="sensitive_features",
                    percentage=50.0,
                    xai_type="importance",
                    tested_with_xai=True
                ),
                StrategyConfig(
                    strategy_name="salient_features",
                    percentage=50.0,
                    xai_type="importance",
                    tested_with_xai=False
                ),
            ]
        )
        
        results = generator.generate(config)
        df = generator.results_to_dataframe(results)
        df.to_csv("session_data.csv", index=False)
    """
    
    def __init__(self):
        self.loader = DistributionLoader()
        self.sampler: Optional[ParameterSampler] = None
        self.simulator = TrialSimulator()
        self.results: List[TrialResult] = []
    
    def generate(self, config: SessionConfig) -> List[TrialResult]:
        """
        Generate a full session of synthetic participants.
        
        Args:
            config: SessionConfig with generation settings
            
        Returns:
            List of all TrialResult objects from all participants
        """
        # Validate and setup
        if config.distribution_file:
            self.loader.load_from_json(config.distribution_file)
            self.sampler = ParameterSampler(seed=config.random_seed)
            self.sampler.load_distributions(config.distribution_file)
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Show configuration
        self._log_configuration(config)
        
        # Normalize strategy percentages
        total_pct = sum(sc.percentage for sc in config.strategy_configs)
        normalized_configs = [
            {
                **vars(sc),
                "percentage": sc.percentage / total_pct * 100.0
            }
            for sc in config.strategy_configs
        ]
        
        # Allocate participants to strategies
        allocations = self._allocate_participants(config.n_participants, normalized_configs)
        
        # Generate participants
        self.results = []
        participant_idx = 0
        
        for strategy_config_dict, n_for_strategy in allocations:
            for i in range(n_for_strategy):
                participant_id = self._build_participant_id(participant_idx, config.participant_id_prefix)
                
                logger.info(f"Generating participant {participant_idx + 1}/{config.n_participants}: "
                           f"{participant_id} (strategy: {strategy_config_dict['strategy_name']})")
                
                try:
                    results = self._generate_participant(
                        config=config,
                        participant_id=participant_id,
                        strategy_config_dict=strategy_config_dict
                    )
                    self.results.extend(results)
                    participant_idx += 1
                except Exception as e:
                    logger.error(f"Failed to generate participant {participant_id}: {e}")
                    continue
        
        logger.info(f"✓ Generated {len(self.results)} trials for {participant_idx} participants")
        return self.results
    
    def _generate_participant(self, 
                             config: SessionConfig,
                             participant_id: str,
                             strategy_config_dict: Dict[str, Any]) -> List[TrialResult]:
        """
        Generate trials for a single participant.
        
        Args:
            config: SessionConfig
            participant_id: Unique participant ID
            strategy_config_dict: Strategy configuration dict
            
        Returns:
            List of TrialResult objects for this participant
        """
        strategy_name = strategy_config_dict["strategy_name"]
        xai_type = strategy_config_dict.get("xai_type", "importance")
        tested_with_xai = strategy_config_dict.get("tested_with_xai", True)
        
        # Get cognitive parameters
        if strategy_config_dict.get("cognitive_params"):
            # Use provided parameters
            cognitive_params = strategy_config_dict["cognitive_params"]
        elif self.sampler and self.loader:
            # Sample from distribution
            cognitive_params = self.sampler.sample(
                dataset=config.dataset_name,
                strategy=strategy_name,
                xai_type=xai_type,
                tested_with_xai="w/ XAI" if tested_with_xai else "w/o XAI",
                method=config.sampling_method
            )
        else:
            # Use defaults from loader
            cognitive_params = self.loader.get_default_params(
                dataset=config.dataset_name,
                strategy=strategy_name,
                xai_type=xai_type,
                tested_with_xai=tested_with_xai
            )
        
        # Get explainer
        explainer = strategy_config_dict.get("explainer") or config.xai_explainer
        
        # Create trial config
        trial_config = TrialConfig(
            participant_id=participant_id,
            dataset_name=config.dataset_name,
            strategy_name=strategy_name,
            xai_type=xai_type,
            tested_with_xai=tested_with_xai,
            cognitive_params=cognitive_params,
            explainer=explainer,
            ai_dataset_loader=config.ai_dataset_loader,
            n_trials=config.n_trials_per_participant,
            random_seed=config.random_seed,
        )
        
        # Simulate trials
        results = self.simulator.simulate(trial_config)
        return results
    
    def results_to_dataframe(self, results: Optional[List[TrialResult]] = None) -> pd.DataFrame:
        """
        Convert trial results to DataFrame.
        
        Args:
            results: List of TrialResult (uses self.results if None)
            
        Returns:
            DataFrame with one row per trial
        """
        if results is None:
            results = self.results
        
        data = [r.to_dict() for r in results]
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_path: str, results: Optional[List[TrialResult]] = None) -> None:
        """
        Export results to CSV.
        
        Args:
            output_path: Path to output CSV
            results: List of TrialResult (uses self.results if None)
        """
        df = self.results_to_dataframe(results)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Exported {len(df)} trials to {output_path}")
    
    def export_summary(self, output_path: str) -> None:
        """
        Export summary statistics to JSON.
        
        Args:
            output_path: Path to output JSON summary
        """
        import json
        
        results_df = self.results_to_dataframe()
        
        summary = {
            "total_trials": len(results_df),
            "total_participants": results_df["Participant ID"].nunique(),
            "mean_accuracy_ai": float(results_df["Response==AI"].mean()),
            "mean_accuracy_explainer": float(results_df["Response==Explainer"].mean()),
            "mean_response_time": float(results_df["Response Time (s)"].mean()),
            "strategies": {}
        }
        
        # Per-strategy summary
        for strategy in results_df["Strategy"].unique():
            strategy_data = results_df[results_df["Strategy"] == strategy]
            summary["strategies"][strategy] = {
                "n_trials": len(strategy_data),
                "n_participants": strategy_data["Participant ID"].nunique(),
                "mean_accuracy_ai": float(strategy_data["Response==AI"].mean()),
                "mean_accuracy_explainer": float(strategy_data["Response==Explainer"].mean()),
            }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Exported summary to {output_path}")
    
    @staticmethod
    def _allocate_participants(n_total: int, 
                              strategy_configs: List[Dict[str, Any]]) -> List[Tuple[Dict, int]]:
        """
        Allocate participants to strategies based on percentages.
        
        Args:
            n_total: Total number of participants
            strategy_configs: List of strategy config dicts with 'percentage'
            
        Returns:
            List of (strategy_config_dict, count) tuples
        """
        allocations = []
        remaining = n_total
        
        for i, config in enumerate(strategy_configs):
            if i == len(strategy_configs) - 1:
                # Last strategy gets all remaining participants
                count = remaining
            else:
                # Allocate based on percentage
                count = max(1, int(n_total * config["percentage"] / 100.0))
                remaining -= count
            
            allocations.append((config, count))
        
        return allocations

    @staticmethod
    def _build_participant_id(index: int, prefix: Optional[str] = None) -> str:
        """Build deterministic participant IDs as 0..n-1 (or prefixed)."""
        if prefix:
            return f"{prefix}{index}"
        return str(index)
    
    @staticmethod
    def _log_configuration(config: SessionConfig) -> None:
        """Log session configuration."""
        logger.info("=" * 70)
        logger.info("SESSION GENERATION CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Participants: {config.n_participants}")
        logger.info(f"Trials per participant: {config.n_trials_per_participant}")
        logger.info(f"Total trials: {config.n_participants * config.n_trials_per_participant}")
        logger.info("\nStrategy Distribution:")
        for sc in config.strategy_configs:
            logger.info(f"  - {sc.strategy_name}: {sc.percentage}% (XAI: {sc.xai_type}, with_xai: {sc.tested_with_xai})")
        logger.info("=" * 70)
