"""
================================================================================
SIMULATION RUNNER WITH CSV-BASED PARAMETER LOADING
================================================================================

This script loads cognitive model parameters from the fitted CSV file
and runs simulations with randomly selected participant parameters.

Key features:
- Load parameters from 'three datasets strategies.csv'
- Filter by strategy, XAI type, test condition (with/without XAI)
- Randomly select one participant's parameter set
- Instantiate the appropriate strategy
- Run the experiment with those parameters

Usage:
    python run_simulation_from_params.py
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path

# Set up paths
parent_dir = os.getcwd()
sys.path.insert(0, parent_dir)

from consolidated_human_models import (
    AttributionSum, 
    SalientFeatures, 
    SensitiveFeatures, 
    ImportanceCategorization
)
from data_loader import AIDatasetLoader
from experiment_runner import StrategyComparisonRunner
from ui import UI


# =============================================================================
# PART 1: PARAMETER LOADING UTILITIES
# =============================================================================

class CSVParameterLoader:
    """
    Loads and filters cognitive model parameters from the fitted CSV file.
    """
    
    def __init__(self, csv_path):
        """
        Initialize the loader with a CSV file path.
        
        Args:
            csv_path (str): Path to the CSV file with fitted parameters
        """
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
            strategy (str, optional): Strategy name (e.g., 'Sensitive-features categorization')
            xai_type (str, optional): XAI type (e.g., 'importance', 'attribution')
            tested_with_xai (str, optional): 'w/ XAI' or 'w/o XAI'
            dataset (str, optional): Dataset name (e.g., 'adult')
        
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
        """
        Randomly select one row from the filtered dataframe.
        
        Args:
            filtered_df (pd.DataFrame): The filtered dataframe
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            dict: A single row as a dictionary, or None if dataframe is empty
        """
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
    def extract_config(param_row, strategy_name):
        """
        Extract configuration parameters from a CSV row.
        
        Args:
            param_row (pd.Series): A row from the DataFrame
            strategy_name (str): The strategy name
        
        Returns:
            dict: Configuration dictionary for the strategy
        """
        config = {
            "k": int(param_row.get('k', 1)),
            "retrieval_threshold": param_row.get('retrieval_threshold', -2.0),
            "decay_param": param_row.get('decay_param', 0.5),
        }
        
        if strategy_name == "Attribution Sum":
            config["scaling_factor"] = param_row.get('scaling_factor', 1.0)
            config["explanation_type"] = param_row.get('explanation_type', 'importance')
        else:
            config["sensitivity"] = param_row.get('sensitivity', 1.0)
        
        # Remove NaN values
        config = {k: v for k, v in config.items() if pd.notna(v)}
        
        return config


# =============================================================================
# PART 2: STRATEGY INSTANTIATION
# =============================================================================

STRATEGY_CLASSES = {
    "Attribution Sum": AttributionSum,
    "Sensitive-features categorization": SensitiveFeatures,
    "Salient-features categorization": SalientFeatures,
    "Importance categorization": ImportanceCategorization,
}


def instantiate_strategy(strategy_name, config):
    """
    Instantiate a strategy object with the given configuration.
    
    Args:
        strategy_name (str): Name of the strategy
        config (dict): Configuration parameters
    
    Returns:
        object: Instantiated strategy object
    """
    if strategy_name not in STRATEGY_CLASSES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    StrategyClass = STRATEGY_CLASSES[strategy_name]
    strategy = StrategyClass(**config)
    
    print(f"✓ Instantiated {strategy_name}")
    print(f"  Config: {config}")
    
    return strategy


# =============================================================================
# PART 3: MAIN SIMULATION RUNNER
# =============================================================================

def run_simulation_with_csv_params(
    csv_path,
    dataset_config,
    strategy_filter=None,
    xai_type_filter=None,
    tested_with_xai_filter=None,
    dataset_filter=None,
    trial_sequence=None,
    seed=None,
):
    """
    Main function to run a simulation with parameters loaded from CSV.
    
    Args:
        csv_path (str): Path to the fitted parameters CSV
        dataset_config (dict): Dataset configuration with:
            - 'values_csv': path to feature values
            - 'metadata_csv': path to metadata
            - 'explanation_csv': path to explanations
            - 'explanation_columns': list of explanation column names
        strategy_filter (str, optional): Filter by strategy name
        xai_type_filter (str, optional): Filter by XAI type
        tested_with_xai_filter (str, optional): Filter by 'w/ XAI' or 'w/o XAI'
        dataset_filter (str, optional): Filter by dataset (e.g., 'adult')
        trial_sequence (list, optional): Trial sequence for experiment
            Each trial: {"instance_id": int, "is_training": bool, "with_explanation": bool}
        seed (int, optional): Random seed
    
    Returns:
        tuple: (strategy, runner, logs)
    """
    
    # Step 1: Load and filter parameters
    print("=" * 80)
    print("STEP 1: Loading and filtering parameters")
    print("=" * 80)
    
    loader = CSVParameterLoader(csv_path)
    filtered_df = loader.filter_parameters(
        strategy=strategy_filter,
        xai_type=xai_type_filter,
        tested_with_xai=tested_with_xai_filter,
        dataset=dataset_filter
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
    print(f"  Parameters:")
    for col in ['k', 'sensitivity', 'retrieval_threshold', 'decay_param', 'scaling_factor']:
        val = param_row.get(col)
        if pd.notna(val):
            print(f"    - {col}: {val}")
    
    # Step 3: Extract config and instantiate strategy
    print("\n" + "=" * 80)
    print("STEP 3: Instantiating strategy")
    print("=" * 80)
    
    strategy_name = param_row['Strategy']
    config = CSVParameterLoader.extract_config(param_row, strategy_name)
    strategy = instantiate_strategy(strategy_name, config)
    
    # Step 4: Load data
    print("\n" + "=" * 80)
    print("STEP 4: Loading dataset")
    print("=" * 80)
    
    df_values = pd.read_csv(dataset_config['values_csv'])
    df_metadata = pd.read_csv(dataset_config['metadata_csv'])
    df_explanation = pd.read_csv(dataset_config['explanation_csv'])
    
    ai_loader = (
        AIDatasetLoader(
            feature_values_df=df_values,
            metadata_df=df_metadata,
            explanation_values_df=df_explanation,
            explanation_columns=dataset_config['explanation_columns']
        )
    )
    print(f"✓ Dataset loaded")
    
    # Step 5: Create runner and execute
    print("\n" + "=" * 80)
    print("STEP 5: Running experiment")
    print("=" * 80)
    
    ui = UI()
    runner = StrategyComparisonRunner(strategy, ai_loader, ui)
    
    if trial_sequence is None:
        raise ValueError("trial_sequence must be provided!")
    
    logs = runner.generalized_run_experiment(trial_sequence)
    
    print(f"\n✓ Experiment completed!")
    print(f"  Total logs: {len(logs)}")
    
    return strategy, runner, logs


# =============================================================================
# PART 4: EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # Configure paths
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    data_dir = current_dir / "data" / "datasets" / "standard set"
    
    dataset_config = {
        'values_csv': str(data_dir / 'values.csv'),
        'metadata_csv': str(data_dir / 'metadata.csv'),
        'explanation_csv': str(data_dir / 'importance.csv'),  # or 'attribution.csv'
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    }
    
    # Example trial sequence (create your own)
    trial_sequence = [
        {"instance_id": 0, "is_training": True, "with_explanation": True},
        {"instance_id": 1, "is_training": True, "with_explanation": True},
        {"instance_id": 2, "is_training": False, "with_explanation": False},
        {"instance_id": 3, "is_training": False, "with_explanation": True},
    ]
    
    try:
        strategy, runner, logs = run_simulation_with_csv_params(
            csv_path=str(csv_path),
            dataset_config=dataset_config,
            strategy_filter="Sensitive-features categorization",
            xai_type_filter="importance",
            tested_with_xai_filter="w/ XAI",
            dataset_filter="adult",
            trial_sequence=trial_sequence,
            seed=42
        )
        
        # Process results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        # Example: count correct predictions
        correct_count = sum(1 for log in logs if log.get('Step') == 'infer' 
                           and log.get('response') and max(log['response'].values()) 
                           == log['response'].get(log.get('ai_prediction')))
        
        print(f"Total inferences: {sum(1 for log in logs if log.get('Step') == 'infer')}")
        print(f"Correct predictions: {correct_count}")
        
        # Save results if desired
        results_df = pd.DataFrame(logs)
        results_path = current_dir / "results" / "simulation_results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
        
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
