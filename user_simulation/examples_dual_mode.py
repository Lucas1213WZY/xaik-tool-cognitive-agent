"""
Minimal working examples for both data input modes.
Copy and paste these templates for quick start.
"""

import numpy as np
import pandas as pd
from generate_trials_full import generate_trials_from_params_csv

# ===== TEMPLATE 1: MODE 1 (Load via ai_dataset_loader) =====
def example_mode1_with_loaders():
    """Standard mode - requires working loaders."""
    
    # Your existing loaders (must be working)
    user_loader = ...  # Your UserDataLoader instance
    ai_dataset_loader = ...  # Your AIDatasetLoader instance
    
    # Your model and data
    ppo_model = ...  # Trained PPO model
    lr_data = ...  # Logistic regression data
    dt_data = ...  # Decision tree data
    metadata = ...  # Metadata
    
    # Define mappings
    strategies = {
        0: 'change_path_dt',
        1: 'zero_out_lr_heuristic',
        2: 'zero_out_lr_displayed',
        3: 'recall_change_dt',
        4: 'recall_change_lr'
    }
    
    XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
    
    training_cog_params = {
        'retrieval_threshold': [-2.0, 0.5],
        'over_margin': [0.0, 0.5],
        'chi': [0.0, 0.02],
        'lapse': [0.0, 0.1]
    }
    
    # Generate trials using MODE 1
    result_df = generate_trials_from_params_csv(
        model=ppo_model,
        user_loader=user_loader,
        ai_dataset_loader=ai_dataset_loader,
        lr_df=lr_data,
        dt_df=dt_data,
        metadata_df=metadata,
        strategies=strategies,
        XAI_types=XAI_types,
        training_cog_params=training_cog_params,
        param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
        mode='experiment',
        output_csv='outputs/trials_mode1.csv',
        n_participants=10,
        n_trials_per_participant=40,
        random_seed=42
        # data_instances=None ← Omit (defaults to None → MODE 1)
    )
    
    print(f"Generated {len(result_df)} trials")
    return result_df


# ===== TEMPLATE 2: MODE 2A (Synthetic Data - Single Participant) =====
def example_mode2a_synthetic_single():
    """Use synthetic random data for one participant."""
    
    # Generate random data (no loaders needed!)
    n_trials = 40
    n_features = 6
    synthetic_data = [
        np.random.rand(n_features)
        for _ in range(n_trials)
    ]
    
    # Model and mappings
    ppo_model = ...  # Trained PPO model
    
    strategies = {
        0: 'change_path_dt',
        1: 'zero_out_lr_heuristic',
        2: 'zero_out_lr_displayed',
        3: 'recall_change_dt',
        4: 'recall_change_lr'
    }
    
    XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
    
    training_cog_params = {
        'retrieval_threshold': [-2.0, 0.5],
        'over_margin': [0.0, 0.5],
        'chi': [0.0, 0.02],
        'lapse': [0.0, 0.1]
    }
    
    # Generate trials using MODE 2A
    result_df = generate_trials_from_params_csv(
        model=ppo_model,
        user_loader=None,  # ← Not needed for MODE 2
        ai_dataset_loader=None,  # ← Not needed for MODE 2
        lr_df=None,
        dt_df=None,
        metadata_df=None,
        strategies=strategies,
        XAI_types=XAI_types,
        training_cog_params=training_cog_params,
        param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
        mode='participant',
        output_csv='outputs/trials_synthetic_single.csv',
        n_trials_per_participant=40,
        data_instances=synthetic_data  # ← MODE 2: Use synthetic data
    )
    
    print(f"Generated {len(result_df)} trials (MODE 2A)")
    return result_df


# ===== TEMPLATE 2B: MODE 2B (External Data - Multiple Participants) =====
def example_mode2b_external_data():
    """Load data from external source, generate for multiple participants."""
    
    # Load parameter CSV to get participant IDs
    param_df = pd.read_csv(
        'user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv'
    )
    participant_ids = param_df['Participant Id'].unique()
    
    # OPTION A: Load from CSV files
    def load_participant_data(participant_id):
        """Load pre-computed data for this participant."""
        # Example: Read from 'data/{participant_id}.csv'
        csv_path = f'path/to/data/{participant_id}.csv'
        try:
            df = pd.read_csv(csv_path)
            # Assume columns 'feat_1', 'feat_2', ..., 'feat_6'
            feature_cols = [f'feat_{i}' for i in range(1, 7)]
            data = df[feature_cols].values  # Convert to numpy array (N, 6)
            return [row for row in data]  # List of arrays
        except FileNotFoundError:
            print(f"Warning: Data not found for {participant_id}")
            return None
    
    # Create data dict for all participants
    data_instances_dict = {}
    for pid in participant_ids[:10]:  # First 10 participants
        data = load_participant_data(pid)
        if data is not None:
            data_instances_dict[pid] = data
    
    print(f"Loaded data for {len(data_instances_dict)} participants")
    
    # Model and mappings
    ppo_model = ...  # Trained PPO model
    
    strategies = {
        0: 'change_path_dt',
        1: 'zero_out_lr_heuristic',
        2: 'zero_out_lr_displayed',
        3: 'recall_change_dt',
        4: 'recall_change_lr'
    }
    
    XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
    
    training_cog_params = {
        'retrieval_threshold': [-2.0, 0.5],
        'over_margin': [0.0, 0.5],
        'chi': [0.0, 0.02],
        'lapse': [0.0, 0.1]
    }
    
    # Generate trials using MODE 2B
    result_df = generate_trials_from_params_csv(
        model=ppo_model,
        user_loader=None,  # ← Not needed
        ai_dataset_loader=None,  # ← Not needed
        lr_df=None,
        dt_df=None,
        metadata_df=None,
        strategies=strategies,
        XAI_types=XAI_types,
        training_cog_params=training_cog_params,
        param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
        mode='experiment',
        output_csv='outputs/trials_external_data.csv',
        n_participants=10,
        n_trials_per_participant=40,
        data_instances_dict=data_instances_dict  # ← MODE 2B: Use loaded data
    )
    
    print(f"Generated {len(result_df)} trials (MODE 2B)")
    return result_df


# ===== TEMPLATE 2C: MODE 2C (Synthetic - Full Experiment) =====
def example_mode2c_synthetic_full_experiment():
    """Generate synthetic data for full experiment (multiple participants)."""
    
    # Load parameter CSV
    param_df = pd.read_csv(
        'user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv'
    )
    participant_ids = param_df['Participant Id'].unique()
    
    # Generate synthetic data for each participant
    n_trials_per_participant = 40
    n_features = 6
    
    data_instances_dict = {}
    for pid in participant_ids:  # All participants
        # Generate random data for this participant
        participant_data = [
            np.random.rand(n_features)
            for _ in range(n_trials_per_participant)
        ]
        data_instances_dict[pid] = participant_data
    
    print(f"Generated synthetic data for {len(data_instances_dict)} participants")
    
    # Model and mappings
    ppo_model = ...  # Trained PPO model
    
    strategies = {
        0: 'change_path_dt',
        1: 'zero_out_lr_heuristic',
        2: 'zero_out_lr_displayed',
        3: 'recall_change_dt',
        4: 'recall_change_lr'
    }
    
    XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
    
    training_cog_params = {
        'retrieval_threshold': [-2.0, 0.5],
        'over_margin': [0.0, 0.5],
        'chi': [0.0, 0.02],
        'lapse': [0.0, 0.1]
    }
    
    # Generate trials
    result_df = generate_trials_from_params_csv(
        model=ppo_model,
        user_loader=None,
        ai_dataset_loader=None,
        lr_df=None,
        dt_df=None,
        metadata_df=None,
        strategies=strategies,
        XAI_types=XAI_types,
        training_cog_params=training_cog_params,
        param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
        mode='experiment',
        output_csv='outputs/trials_synthetic_full.csv',
        random_seed=42,
        data_instances_dict=data_instances_dict  # ← MODE 2C: Full synthetic
    )
    
    print(f"Generated {len(result_df)} trials (MODE 2C)")
    print(f"Participants: {result_df['participant_id'].nunique()}")
    return result_df


# ===== TEMPLATE 3: Comparison Script =====
def compare_modes():
    """Run examples from both modes and compare outputs."""
    
    print("\n" + "="*60)
    print("MODE 1: Using ai_dataset_loader")
    print("="*60)
    try:
        df1 = example_mode1_with_loaders()
        print(f"✓ MODE 1 generated {len(df1)} trials")
    except Exception as e:
        print(f"✗ MODE 1 failed: {e}")
        df1 = None
    
    print("\n" + "="*60)
    print("MODE 2B: Using external data")
    print("="*60)
    try:
        df2 = example_mode2b_external_data()
        print(f"✓ MODE 2B generated {len(df2)} trials")
    except Exception as e:
        print(f"✗ MODE 2B failed: {e}")
        df2 = None
    
    print("\n" + "="*60)
    print("MODE 2C: Using synthetic data (full experiment)")
    print("="*60)
    try:
        df3 = example_mode2c_synthetic_full_experiment()
        print(f"✓ MODE 2C generated {len(df3)} trials")
    except Exception as e:
        print(f"✗ MODE 2C failed: {e}")
        df3 = None
    
    # Compare if both succeeded
    if df1 is not None and df3 is not None:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"MODE 1 shape: {df1.shape}")
        print(f"MODE 2C shape: {df3.shape}")
        print(f"MODE 1 columns: {list(df1.columns)}")
        print(f"MODE 2C 'data_instance' col present: {'data_instance' in df3.columns}")


if __name__ == "__main__":
    # Pick one to run:
    
    # Mode 1 (requires working loaders)
    # example_mode1_with_loaders()
    
    # Mode 2A (single participant, synthetic data)
    # example_mode2a_synthetic_single()
    
    # Mode 2B (multiple participants, external data)
    # example_mode2b_external_data()
    
    # Mode 2C (full experiment, synthetic data)
    # example_mode2c_synthetic_full_experiment()
    
    # Compare all modes
    # compare_modes()
    
    print("See code above for examples. Uncomment one to run.")
