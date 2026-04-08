"""
Forward Trial Generator for CoXAM/CoAX using RL Agents

Generates forward trial predictions based on experimental design parameters.
Does not require participant-specific fitting - uses pre-trained models.

Key parameters:
- instance_id: Which instance to use for each trial
- xai_type: Condition (DT, LR, DT+LR)
- with_xai: Whether to show XAI (0/1 per trial or ratio)
- phase: Trial phase (forward/counterfactual for CoXAM, train/test for CoAX)
"""

from typing import Dict, Any, List, Optional, Sequence, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrialSchedule:
    """
    Experimental design specification for trial generation.
    
    Specifies the conditions and parameters for each trial.
    """
    
    def __init__(
        self,
        instance_ids: Sequence[int],
        xai_types: Optional[Sequence[str]] = None,
        with_xai_schedule: Optional[Sequence[int]] = None,
        with_xai_ratio: float = 0.5,
        phase: str = "forward",
    ):
        """
        Initialize trial schedule.
        
        Args:
            instance_ids: Instance ID for each trial
            xai_types: XAI condition per trial (DT, LR, DT+LR), or None for uniform
            with_xai_schedule: Binary array (1=with XAI, 0=without), or None for random
            with_xai_ratio: Probability of with_xai if with_xai_schedule is None
            phase: Trial phase (forward, counterfactual, train, test)
        """
        self.n_trials = len(instance_ids)
        self.instance_ids = np.asarray(instance_ids)
        self.phase = phase
        
        # XAI type schedule
        if xai_types is None:
            self.xai_types = np.array(["DT"] * self.n_trials)
        else:
            self.xai_types = np.asarray(xai_types)
        
        # With-XAI schedule
        if with_xai_schedule is None:
            rng = np.random.default_rng()
            self.with_xai_schedule = rng.choice(
                [0, 1],
                size=self.n_trials,
                p=[1.0 - with_xai_ratio, with_xai_ratio]
            )
        else:
            self.with_xai_schedule = np.asarray(with_xai_schedule, dtype=int)
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "TrialSchedule":
        """Create TrialSchedule from configuration dict."""
        return TrialSchedule(**config)


class ExperimentalDesign:
    """
    Specification for CoXAM/CoAX experimental generation.
    
    Describes the structure of generated trials across:
    - Multiple datasets (app_ids)
    - Multiple models  
    - Multiple complexity levels
    - Multiple XAI conditions
    - Forward and counterfactual phases
    """
    
    def __init__(
        self,
        app_ids: Sequence[str],
        model_names: Sequence[str],
        complexities: Sequence[str],
        n_trials_per_condition: int = 100,
        with_xai_ratio: float = 0.5,
        xai_type_distribution: Optional[Dict[str, float]] = None,
        phases: Sequence[str] = ("forward",),
        rng_seed: int = 123,
    ):
        """
        Initialize experimental design.
        
        Args:
            app_ids: Dataset/application IDs
            model_names: Model names (mlp, xgb, etc.)
            complexities: Complexity levels (low, medium, high)
            n_trials_per_condition: Number of trials per condition
            with_xai_ratio: Probability of with_xai trials
            xai_type_distribution: Distribution over XAI types (DT, LR, DT+LR)
            phases: Trial phases (forward, counterfactual, train, test)
            rng_seed: Random seed
        """
        self.app_ids = list(app_ids)
        self.model_names = list(model_names)
        self.complexities = list(complexities)
        self.n_trials_per_condition = n_trials_per_condition
        self.with_xai_ratio = with_xai_ratio
        self.phases = list(phases)
        self.rng = np.random.default_rng(rng_seed)
        
        # XAI type distribution
        if xai_type_distribution is None:
            xai_types = ["DT", "LR", "DT+LR"]
            probs = [1.0 / len(xai_types)] * len(xai_types)
            self.xai_type_distribution = dict(zip(xai_types, probs))
        else:
            self.xai_type_distribution = xai_type_distribution
    
    def generate_trial_schedule(
        self,
        ai_dataset_loader: Any,
        app_id: str,
        model_name: str,
        n_trials: Optional[int] = None,
    ) -> TrialSchedule:
        """
        Generate trial schedule for given app and model.
        
        Args:
            ai_dataset_loader: Dataset loader with instance info
            app_id: Selected app ID
            model_name: Selected model
            n_trials: Number of trials (default: n_trials_per_condition)
        
        Returns:
            TrialSchedule with trial parameters
        """
        if n_trials is None:
            n_trials = self.n_trials_per_condition
        
        # Get available instances
        try:
            from src.data_loaders.filters import filter_by_app_and_model
            fl = filter_by_app_and_model(ai_dataset_loader, app_id, model_name)
            available_instances = fl.get_instance_ids()
            if not available_instances:
                logger.warning(f"No instances for {app_id}/{model_name}")
                available_instances = list(range(1, 401))  # Fallback
        except Exception as e:
            logger.warning(f"Could not load instances: {e}")
            available_instances = list(range(1, 401))
        
        # Sample instance IDs
        instance_ids = self.rng.choice(
            available_instances,
            size=min(n_trials, len(available_instances)),
            replace=False
        )
        
        # Sample XAI types
        xai_types = self.rng.choice(
            list(self.xai_type_distribution.keys()),
            size=n_trials,
            p=list(self.xai_type_distribution.values())
        )
        
        # With-XAI schedule
        with_xai_schedule = self.rng.choice(
            [0, 1],
            size=n_trials,
            p=[1.0 - self.with_xai_ratio, self.with_xai_ratio]
        )
        
        return TrialSchedule(
            instance_ids=instance_ids,
            xai_types=xai_types,
            with_xai_schedule=with_xai_schedule,
            phase="forward",
        )


class ForwardTrialDatasetGenerator:
    """
    Generates complete forward trial datasets using RL agents.
    
    Combines:
    - Experimental design specification
    - RL agent runners
    - CSV output formatting
    """
    
    def __init__(
        self,
        forward_runner: Any,  # ForwardSimulationRunner
        experimental_design: ExperimentalDesign,
        ai_dataset_loader: Any,
    ):
        """
        Initialize generator.
        
        Args:
            forward_runner: ForwardSimulationRunner instance
            experimental_design: ExperimentalDesign specification
            ai_dataset_loader: Dataset loader
        """
        self.forward_runner = forward_runner
        self.experimental_design = experimental_design
        self.ai_dataset_loader = ai_dataset_loader
    
    def generate(
        self,
        output_path: Optional[str] = None,
        episode_cogs: Optional[Dict[str, Any]] = None,
        chi_value: float = 0.01,
        deterministic: bool = True,
        rng_seed: int = 123,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Generate complete forward trial dataset.
        
        Args:
            output_path: Path to save CSV (default: ./forward_trials.csv)
            episode_cogs: Shared cognitive parameters across all trials
            chi_value: Time cost parameter (shared)
            deterministic: Use deterministic policy
            rng_seed: Random seed
        
        Returns:
            (output_path, dataframe)
        """
        if output_path is None:
            output_path = "./forward_trials.csv"
        
        rows = []
        trial_index = 0
        
        # Iterate over all design conditions
        for phase in self.experimental_design.phases:
            for app_id in self.experimental_design.app_ids:
                for model_name in self.experimental_design.model_names:
                    for complexity in self.experimental_design.complexities:
                        logger.info(
                            f"Generating {phase}/{app_id}/{model_name}/{complexity}..."
                        )
                        
                        # Generate trial schedule
                        trial_schedule = self.experimental_design.generate_trial_schedule(
                            self.ai_dataset_loader,
                            app_id,
                            model_name,
                        )
                        
                        # Determine condition from xai_types
                        condition = self._infer_condition(trial_schedule.xai_types)
                        
                        # Run episode
                        episode_output = self.forward_runner.run_episode(
                            instance_ids=trial_schedule.instance_ids,
                            app_id=app_id,
                            model_name=model_name,
                            condition=condition,
                            complexity=complexity,
                            phase=phase,
                            with_xai_schedule=trial_schedule.with_xai_schedule,
                            trial_type_schedule=trial_schedule.xai_types,
                            episode_cogs=episode_cogs,
                            chi_value=chi_value,
                            deterministic=deterministic,
                            rng_seed=rng_seed + trial_index,
                        )
                        
                        # Format rows
                        for t, iid in enumerate(trial_schedule.instance_ids):
                            rows.append({
                                "Phase": phase,
                                "AppId": app_id,
                                "Model": model_name,
                                "Complexity": complexity,
                                "Condition": condition,
                                "Trial Index": t,
                                "Instance Id": iid,
                                "XAIType": trial_schedule.xai_types[t] if t < len(trial_schedule.xai_types) else "DT",
                                "Tested w/ XAI": "w/ XAI" if trial_schedule.with_xai_schedule[t] else "w/o XAI",
                                "Model Prediction": episode_output["responses"][t],
                                "Response Prob (p1)": episode_output["response_probs"][t][1] if t < len(episode_output["response_probs"]) else 0.5,
                                "Pred Time": episode_output["pred_times"][t],
                                "AI Prediction": episode_output["y_raw"][t],
                                "Objective": episode_output["objective"],
                            })
                            trial_index += 1
        
        # Create and save dataframe
        df = pd.DataFrame(rows)
        
        # Organize columns
        col_order = [
            "Phase", "AppId", "Model", "Complexity", "Condition",
            "Trial Index", "Instance Id", "XAIType", "Tested w/ XAI",
            "Model Prediction", "Response Prob (p1)", "Pred Time", "AI Prediction",
            "Objective"
        ]
        df = df[[c for c in col_order if c in df.columns]]
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} trials to {output_path}")
        
        return output_path, df
    
    @staticmethod
    def _infer_condition(xai_types: np.ndarray) -> str:
        """Infer dominant condition from xai type distribution."""
        unique, counts = np.unique(xai_types, return_counts=True)
        dominant = unique[np.argmax(counts)]
        return str(dominant)


# Convenience functions

def generate_forward_trials(
    forward_runner: Any,
    ai_dataset_loader: Any,
    app_ids: Sequence[str],
    model_names: Sequence[str],
    complexities: Sequence[str],
    n_trials_per_condition: int = 100,
    output_path: str = "./forward_trials.csv",
    episode_cogs: Optional[Dict[str, Any]] = None,
    chi_value: float = 0.01,
    rng_seed: int = 123,
) -> Tuple[str, pd.DataFrame]:
    """
    Convenience function to generate forward trials.
    
    Args:
        forward_runner: ForwardSimulationRunner instance
        ai_dataset_loader: Dataset loader
        app_ids: Application IDs (datasets)
        model_names: Model names
        complexities: Complexity levels
        n_trials_per_condition: Trials per condition
        output_path: Output CSV path
        episode_cogs: Cognitive parameters
        chi_value: Time cost
        rng_seed: Random seed
    
    Returns:
        (output_path, dataframe)
    """
    design = ExperimentalDesign(
        app_ids=app_ids,
        model_names=model_names,
        complexities=complexities,
        n_trials_per_condition=n_trials_per_condition,
        phases=["forward"],
        rng_seed=rng_seed,
    )
    
    generator = ForwardTrialDatasetGenerator(
        forward_runner=forward_runner,
        experimental_design=design,
        ai_dataset_loader=ai_dataset_loader,
    )
    
    return generator.generate(
        output_path=output_path,
        episode_cogs=episode_cogs,
        chi_value=chi_value,
        rng_seed=rng_seed,
    )
