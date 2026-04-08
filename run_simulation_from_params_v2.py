"""
================================================================================
UPDATED SIMULATION RUNNER WITH NEW REASONING STRATEGIES API
================================================================================

This updated script loads cognitive model parameters from the fitted CSV file
and runs simulations using the new unified reasoning strategies API from
src/cognitive_models/ instead of the old consolidated_human_models.

Key improvements:
- Uses the unified ReasoningStrategy interface from src/
- Supports multiple reasoning strategy backends (CoAX, CoXAM)
- Improved memory management with UnifiedMemory
- Better parameter mapping to new strategy format
- Cleaner API and extensibility

Usage:
    python run_simulation_from_params_v2.py
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add paths
current_dir = Path(__file__).parent
coax_dir = current_dir / "code_for_papers" / "old" / "coax"
src_path = current_dir / "src"

sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(coax_dir))
sys.path.insert(0, str(src_path.parent))

# Import from new API
from src.cognitive_models.forward import (
    SensitiveFeatures,
    SalientFeatures,
    ImportanceCategorization,
    AttributionSum
)
from src.cognitive_models.interface import StrategyConfig, ReasoningMode, StrategyType

# Import data loader
from data_loader import AIDatasetLoader

# Create a simple mock UI for display
class UI:
    """Simple mock UI for display purposes."""
    def display(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        pass


# Simple data loader for numpy format datasets
class SimpleDataLoader:
    """Simple loader for numpy-based datasets."""
    def __init__(self, X, y):
        """
        Initialize with feature matrix and labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
        """
        self.X = X
        self.y = y
        self.n_samples = len(X)
    
    def get_instance(self, instance_id):
        """
        Get features and label for an instance.
        
        Args:
            instance_id: Index of the instance
            
        Returns:
            tuple: (features, label)
        """
        if instance_id >= len(self.X):
            raise IndexError(f"Instance {instance_id} out of range (max: {len(self.X)-1})")
        
        features = self.X[instance_id]
        label = int(self.y[instance_id]) if isinstance(self.y[instance_id], (int, np.integer)) else int(round(self.y[instance_id]))
        
        return features, label


# =============================================================================
# PART 1: TIME MANAGER FOR REASONING STRATEGIES
# =============================================================================

class SimpleTimeManager:
    """Minimal time manager for reasoning strategies."""
    
    def __init__(self):
        self.current_time = 0.0
    
    def tick(self, dt: float = 1.0):
        """Advance time by dt seconds."""
        self.current_time += dt
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.current_time
    
    def add_time(self, dt: float) -> float:
        """Add time and return new current time."""
        self.tick(dt)
        return self.current_time


# =============================================================================
# PART 2: PARAMETER LOADING AND CONVERSION
# =============================================================================

class CSVParameterLoaderV2:
    """
    Updated parameter loader that maps CSV parameters to new API format.
    """
    
    def __init__(self, csv_path):
        """Initialize the loader with a CSV file path."""
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(self.df)} rows")
        print(f"  Columns: {', '.join(self.df.columns.tolist())}")
    
    def filter_parameters(self, 
                         strategy=None, 
                         xai_type=None, 
                         tested_with_xai=None,
                         dataset=None):
        """
        Filter the DataFrame based on conditions.
        
        Args:
            strategy (str, optional): Strategy name
            xai_type (str, optional): XAI type (importance, attribution, etc.)
            tested_with_xai (str, optional): 'w/ XAI' or 'w/o XAI'
            dataset (str, optional): Dataset name
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        df = self.df.copy()
        
        if strategy:
            df = df[df['Strategy'] == strategy]
        if xai_type:
            df = df[df['XAIType'] == xai_type]
        if tested_with_xai:
            df = df[df['Tested w/ XAI'] == tested_with_xai]
        if dataset:
            df = df[df['appId'] == dataset]
        
        print(f"\n✓ Filtered to {len(df)} rows")
        if len(df) == 0:
            print("  ⚠ WARNING: No rows match the filter criteria!")
        
        return df
    
    def select_random_params(self, filtered_df, seed=None):
        """Randomly select one row from the filtered dataframe."""
        if len(filtered_df) == 0:
            print("\n⚠ Cannot select parameters from empty dataframe!")
            return None
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        selected_row = filtered_df.sample(n=1).iloc[0]
        print(f"\n✓ Randomly selected parameters for participant: {selected_row.get('Participant Id', 'N/A')}")
        
        return selected_row

    @staticmethod
    def create_strategy_config(param_row: pd.Series, strategy_name: str) -> StrategyConfig:
        """
        Convert CSV row to StrategyConfig for new API.
        
        Args:
            param_row (pd.Series): A row from the DataFrame
            strategy_name (str): The strategy name
        
        Returns:
            StrategyConfig: Configuration object
        """
        # Extract common parameters
        decay_param = float(param_row.get('decay_param', 0.5))
        retrieval_threshold = float(param_row.get('retrieval_threshold', -2.0))
        sensitivity = float(param_row.get('sensitivity', 10.0))
        
        # Build strategy-specific extra params
        extra_params = {
            'sensitivity': sensitivity,
            'k': int(param_row.get('k', 1)),
        }
        
        # Add strategy-specific parameters
        if strategy_name == "Attribution Sum":
            extra_params['scaling_factor'] = float(param_row.get('scaling_factor', 1.0))
            extra_params['explanation_type'] = param_row.get('explanation_type', 'importance')
        
        # Create config
        config = StrategyConfig(
            strategy_name=strategy_name,
            strategy_type=StrategyType.COAX_FORWARD,
            mode=ReasoningMode.RETRIEVE,
            decay_param=decay_param,
            retrieval_threshold=retrieval_threshold,
            sensitivity=sensitivity,
            extra_params=extra_params
        )
        
        return config


# =============================================================================
# PART 3: STRATEGY FACTORY
# =============================================================================

STRATEGY_CLASSES_NEW_API = {
    "Attribution Sum": AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
}


def instantiate_strategy_new_api(strategy_name: str, 
                                 config: StrategyConfig,
                                 time_manager: Optional[SimpleTimeManager] = None):
    """
    Instantiate a strategy using the new API.
    
    Args:
        strategy_name (str): Name of the strategy
        config (StrategyConfig): Configuration object
        time_manager (SimpleTimeManager, optional): Time manager for strategy
    
    Returns:
        ReasoningStrategy: Instantiated strategy object
    """
    if strategy_name not in STRATEGY_CLASSES_NEW_API:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Add time manager to config if provided
    if time_manager:
        config.time_manager = time_manager
    
    StrategyClass = STRATEGY_CLASSES_NEW_API[strategy_name]
    strategy = StrategyClass(config)
    
    print(f"✓ Instantiated {strategy_name} (New API)")
    print(f"  decay_param: {config.decay_param}")
    print(f"  retrieval_threshold: {config.retrieval_threshold}")
    print(f"  sensitivity: {config.sensitivity}")
    if config.extra_params:
        print(f"  extra_params: {config.extra_params}")
    
    return strategy


# =============================================================================
# PART 4: EXPERIMENT RUNNER
# =============================================================================

class SimulationRunnerV2:
    """
    Runs simulation using new reasoning strategy API.
    """
    
    def __init__(self, strategy, data_loader, time_manager=None):
        """
        Initialize the runner.
        
        Args:
            strategy: ReasoningStrategy instance
            data_loader: SimpleDataLoader or similar
            time_manager (optional): Time manager for tracking simulation time
        """
        self.strategy = strategy
        self.data_loader = data_loader
        self.time_manager = time_manager or SimpleTimeManager()
        self.logs = []
    
    def run_trial_sequence(self, trial_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run a sequence of trials.
        
        Args:
            trial_sequence: List of trial dicts with keys:
                - instance_id (int or str)
                - is_training (bool)
                - with_explanation (bool)
        
        Returns:
            List[Dict]: Experiment logs
        """
        print("\n" + "=" * 80)
        print("RUNNING TRIAL SEQUENCE")
        print("=" * 80)
        
        for trial_idx, trial in enumerate(trial_sequence):
            instance_id = trial["instance_id"]
            is_training = trial["is_training"]
            with_explanation = trial["with_explanation"]
            
            print(f"\nTrial {trial_idx + 1}/{len(trial_sequence)}: Instance {instance_id}")
            
            # Signal new instance
            self.strategy.new_instance()
            
            # Get data
            try:
                features, true_label = self.data_loader.get_instance(instance_id)
            except Exception as e:
                print(f"  ⚠ Error retrieving data: {e}")
                continue
            
            # Prepare features dict
            features_dict = self._prepare_features(features)
            
            # Prepare explanation (dummy for now since we don't have real explanations)
            explanation_data = None
            if with_explanation:
                explanation_data = self._prepare_explanation(features)
            
            # Run inference
            try:
                probs, time_cost, info = self.strategy.infer(
                    features=features_dict,
                    explanation=explanation_data,
                    ai_prediction=true_label
                )
                
                # Log inference
                self._log_event(
                    instance_id=instance_id,
                    step="infer",
                    is_training=is_training,
                    with_explanation=with_explanation,
                    features=features,
                    explanation=explanation_data,
                    probabilities=probs,
                    true_label=true_label,
                    time_cost=time_cost,
                    info=info
                )
                
                # Run feedback if training
                if is_training:
                    try:
                        fb_time = self.strategy.feedback(
                            features=features_dict,
                            true_label=true_label,
                            explanation=explanation_data
                        )
                        print(f"  ↻ Feedback processed (time: {fb_time:.3f}s)")
                    except Exception as e:
                        print(f"  ⚠ Feedback error: {e}")
                
            except Exception as e:
                print(f"  ✗ Inference error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✓ Trial sequence completed")
        print(f"  Total trials run: {len(self.logs)}")
        
        return self.logs
    
    def _prepare_features(self, feature_values) -> Dict[str, Any]:
        """Convert feature values to dict format."""
        if isinstance(feature_values, dict):
            return feature_values
        elif isinstance(feature_values, (list, np.ndarray)):
            return {f"f{i}": float(v) for i, v in enumerate(feature_values)}
        else:
            return {"features": feature_values}
    
    def _prepare_explanation(self, features) -> Optional[Dict[str, Any]]:
        """Create dummy explanation data."""
        if isinstance(features, (list, np.ndarray)):
            # Use features themselves as explanation (or random)
            return {f"e{i}": float(v) for i, v in enumerate(features)}
        else:
            return {"explanation": features}
    
    def _log_event(self, **kwargs):
        """Log an event."""
        event_log = {
            'trial': len(self.logs) + 1,
            **kwargs
        }
        self.logs.append(event_log)
        
        instance_id = kwargs.get('instance_id')
        step = kwargs.get('step')
        probs = kwargs.get('probabilities', {})
        true_label = kwargs.get('true_label')
        
        # Print summary
        if step == "infer":
            best_prob = max(probs.values()) if probs else 0
            best_label = max(probs, key=probs.get) if probs else '?'
            is_correct = "✓" if best_label == true_label else "✗"
            print(f"    {is_correct} Inference: pred={best_label} (p={best_prob:.4f}), true_label={true_label}")


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================

def run_simulation_with_csv_params_v2(
    csv_path: str,
    dataset_name: str,
    strategy_filter: Optional[str] = None,
    xai_type_filter: Optional[str] = None,
    tested_with_xai_filter: Optional[str] = None,
    trial_sequence: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
) -> Tuple[Any, SimulationRunnerV2, List[Dict[str, Any]]]:
    """
    Main function to run a simulation with parameters loaded from CSV.
    
    Args:
        csv_path: Path to CSV with fitted parameters
        dataset_name: Name of the dataset (e.g., 'adult', 'wine_quality')
        strategy_filter: Filter by strategy name
        xai_type_filter: Filter by XAI type
        tested_with_xai_filter: Filter by 'w/ XAI' or 'w/o XAI'
        trial_sequence: List of trials to run
        seed: Random seed
    
    Returns:
        tuple: (strategy, runner, logs)
    """
    
    # Step 1: Load and filter parameters
    print("=" * 80)
    print("STEP 1: Loading and filtering parameters")
    print("=" * 80)
    
    loader = CSVParameterLoaderV2(csv_path)
    filtered_df = loader.filter_parameters(
        strategy=strategy_filter,
        xai_type=xai_type_filter,
        tested_with_xai=tested_with_xai_filter,
        dataset=dataset_name
    )
    
    if len(filtered_df) == 0:
        raise ValueError("No rows match filter criteria!")
    
    # Step 2: Select random parameters
    print("\n" + "=" * 80)
    print("STEP 2: Selecting random participant parameters")
    print("=" * 80)
    
    param_row = loader.select_random_params(filtered_df, seed=seed)
    print(f"\n  Strategy: {param_row['Strategy']}")
    print(f"  XAI Type: {param_row['XAIType']}")
    print(f"  Tested w/ XAI: {param_row['Tested w/ XAI']}")
    
    # Step 3: Create strategy config and instantiate
    print("\n" + "=" * 80)
    print("STEP 3: Creating strategy config and instantiation")
    print("=" * 80)
    
    strategy_name = param_row['Strategy']
    config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
    
    time_manager = SimpleTimeManager()
    strategy = instantiate_strategy_new_api(strategy_name, config, time_manager)
    
    # Step 4: Load data
    print("\n" + "=" * 80)
    print("STEP 4: Loading dataset")
    print("=" * 80)
    
    # Load numpy-based dataset
    import numpy as np
    current_dir = Path(__file__).parent
    coax_dir = current_dir / "code_for_papers" / "old" / "coax"
    dataset_dir = coax_dir / "datasets" / dataset_name
    
    try:
        X = np.load(dataset_dir / "X.npy")
        y = np.load(dataset_dir / "y.npy")
        print(f"✓ Dataset loaded (shape: {X.shape})")
    except Exception as e:
        raise FileNotFoundError(f"Could not load dataset from {dataset_dir}: {e}")
    
    data_loader = SimpleDataLoader(X, y)
    
    # Step 5: Run experiment
    print("\n" + "=" * 80)
    print("STEP 5: Running experiment")
    print("=" * 80)
    
    runner = SimulationRunnerV2(strategy, data_loader, time_manager)
    
    if trial_sequence is None:
        raise ValueError("trial_sequence must be provided!")
    
    logs = runner.run_trial_sequence(trial_sequence)
    
    print(f"\n✓ Experiment completed!")
    print(f"  Total logs: {len(logs)}")
    
    return strategy, runner, logs


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    current_dir = Path(__file__).parent
    coax_dir = current_dir / "code_for_papers" / "old" / "coax"
    
    csv_path = coax_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    # Example trial sequence
    trial_sequence = [
        {"instance_id": 0, "is_training": True, "with_explanation": True},
        {"instance_id": 1, "is_training": True, "with_explanation": True},
        {"instance_id": 2, "is_training": False, "with_explanation": False},
        {"instance_id": 3, "is_training": False, "with_explanation": True},
    ]
    
    try:
        strategy, runner, logs = run_simulation_with_csv_params_v2(
            csv_path=str(csv_path),
            dataset_name="adult",
            strategy_filter="Sensitive-features categorization",
            xai_type_filter="importance",
            tested_with_xai_filter="w/ XAI",
            trial_sequence=trial_sequence,
            seed=42
        )
        
        # Process results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        infer_logs = [log for log in logs if log.get('step') == 'infer']
        print(f"Total inferences: {len(infer_logs)}")
        
        # Count correct predictions
        if infer_logs:
            correct = sum(1 for log in infer_logs if log.get('probabilities') and max(log['probabilities'], key=log['probabilities'].get) == log.get('true_label'))
            print(f"Correct predictions: {correct}/{len(infer_logs)}")
        
        # Save results
        results_df = pd.DataFrame(logs)
        results_path = coax_dir / "results" / "simulation_results_v2.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
        
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
