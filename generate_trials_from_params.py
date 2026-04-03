"""
Generate trial-by-trial data from CoXAM best parameters using RL agent.
Supports three execution modes: trial, participant, experiment.
"""

import numpy as np
import pandas as pd
import random
import math
import os
from pathlib import Path

# ===== 1. LOAD PARAMETERS FROM CSV =====
def load_best_params(csv_path):
    """Load best parameters from CoXAM counterfactual simulation CSV."""
    df = pd.read_csv(csv_path)
    return df


def sample_param_row(df, random_seed=None):
    """Randomly sample one parameter row from the CSV."""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    return df.sample(n=1).iloc[0].to_dict()


# ===== 2. GENERATE TRIALS (NO SCORING/FITTING) =====
def generate_trial(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    param_row, strategies, XAI_types, training_cog_params,
    lapse=0.1
):
    """
    Generate a SINGLE trial using RL agent with given parameters.
    
    Args:
        model: Trained RL agent (PPO)
        user_loader: Data loader for participant info
        ai_dataset_loader: AI predictions loader
        lr_df, dt_df, metadata_df: Explainer data
        param_row: Dict with best params (from CSV row)
        strategies: Dict mapping strategy IDs to strategy names
        XAI_types: Dict mapping XAI condition keys to names
        training_cog_params: Original training params (for obs builder)
        lapse: Lapse rate for probability smoothing
        
    Returns:
        Dict with trial result containing model predictions, observations, etc.
    """
    
    # Extract params
    rt = param_row['Best retrieval_threshold']
    over_margin = param_row['Best over_margin']
    chi = param_row['Best chi']
    app_id = param_row['app_id']
    complexity = param_row['complexity']
    condition = param_row['condition']
    
    # Setup (would be called from actual environment)
    # This is a placeholder - in real execution, you'd have actual instances
    trial_result = {
        'participant_id': param_row.get('Participant Id', 'synthetic'),
        'app_id': app_id,
        'condition': condition,
        'complexity': complexity,
        'retrieval_threshold': rt,
        'over_margin': over_margin,
        'chi': chi,
        # Model outputs would go here when actually executed
    }
    
    return trial_result


def generate_participant_trials(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    param_row, strategies, XAI_types, training_cog_params,
    lapse=0.1, n_trials=None, data_instances=None
):
    """
    Generate ALL trials for one participant-equivalent session using RL agent.
    
    DUAL MODE:
    - Mode 1 (data_instances=None): Load instances from user_loader via ai_dataset_loader
    - Mode 2 (data_instances provided): Use pre-computed data, skip ai_dataset_loader
    
    Args:
        model: Trained RL agent (PPO)
        user_loader: Data loader 
        ai_dataset_loader: AI predictions loader
        lr_df, dt_df, metadata_df: Explainer data
        param_row: Dict with best params (from CSV row)
        strategies: Dict mapping strategy IDs to strategy names
        XAI_types: Dict mapping XAI condition keys to names
        training_cog_params: Original training params
        lapse: Lapse rate
        n_trials: Number of counterfactual trials to generate (default: all trials from user_loader)
        data_instances: Optional list/array of pre-computed feature values.
                       Each element should be a dict or array-like with feature values.
                       If provided, skips ai_dataset_loader. If None, uses loader.
        
    Returns:
        List of trial dicts, one per trial
    """
    
    rt = param_row['Best retrieval_threshold']
    over_margin = param_row['Best over_margin']
    chi = param_row['Best chi']
    app_id = param_row['app_id']
    complexity = param_row['complexity']
    condition = param_row['condition']
    participant_id = param_row.get('Participant Id', 'synthetic')
    
    # MODE SELECTION
    if data_instances is not None:
        # MODE 2: Use pre-computed data instances (skip ai_dataset_loader)
        print(f"  MODE 2: Using {len(data_instances)} pre-computed data instances")
        if n_trials is not None:
            data_instances = data_instances[:n_trials]
        cf = pd.DataFrame([
            {'trial_idx': i, 'data_instance': inst}
            for i, inst in enumerate(data_instances)
        ])
    else:
        # MODE 1: Load via ai_dataset_loader (original behavior)
        print(f"  MODE 1: Loading instances via ai_dataset_loader")
        try:
            cf = user_loader.get_counterfactual_trials(participant_id)
            if n_trials is not None:
                cf = cf.head(n_trials)
        except Exception as e:
            print(f"Warning: Could not load trials for participant {participant_id}: {e}")
            # Fall back to generating dummy trials
            if n_trials is None:
                n_trials = 40
            cf = pd.DataFrame([{'trial_idx': i} for i in range(n_trials)])
    
    trials_log = []
    
    # Setup memory, interpretability models, etc. (similar to original function)
    # This would contain the full trial-by-trial loop from score_participant_with_theta
    # but WITHOUT the scoring/fitting components
    
    # Placeholder loop structure:
    for ti, tr in cf.iterrows():
        # Extract data_instance if present (from MODE 2)
        data_instance = tr.get('data_instance', None)
        
        trial_dict = {
            'participant_id': participant_id,
            'trial_index': int(ti),
            'app_id': app_id,
            'condition': condition,
            'complexity': complexity,
            'retrieval_threshold': rt,
            'over_margin': over_margin,
            'chi': chi,
            'data_instance': data_instance,  # Include data_instance if provided
            # Model outputs would populate: strategy, depth, changed_feature, changed_amount, etc.
        }
        trials_log.append(trial_dict)
    
    return trials_log


def generate_experiment_trials(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    param_df, strategies, XAI_types, training_cog_params,
    lapse=0.1, n_participants=None, n_trials_per_participant=None, random_seed=None,
    data_instances_dict=None
):
    """
    Generate trials for the ENTIRE EXPERIMENT using all (or sampled) parameter rows.
    
    Args:
        model: Trained RL agent
        user_loader: Data loader
        ai_dataset_loader: AI predictions loader
        lr_df, dt_df, metadata_df: Explainer data
        param_df: Full parameters DataFrame (from CSV)
        strategies: Strategy ID mapping
        XAI_types: XAI condition mapping
        training_cog_params: Original training params
        lapse: Lapse rate
        n_participants: Number of participants to sample (default: all)
        n_trials_per_participant: Number of trials per participant (default: all)
        random_seed: Seed for reproducibility
        data_instances_dict: Optional dict mapping participant_id → list of data instances.
                            If provided, uses MODE 2 (pre-computed data).
                            If None, uses MODE 1 (ai_dataset_loader).
        
    Returns:
        DataFrame with all trials from all participants
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Sample participant parameter rows
    if n_participants is not None and n_participants < len(param_df):
        param_rows = param_df.sample(n=n_participants, random_state=random_seed)
    else:
        param_rows = param_df
    
    all_trials = []
    total_participants = len(param_rows)
    
    for idx, (_, param_row) in enumerate(param_rows.iterrows(), 1):
        print(f"Generating trials for participant {idx}/{total_participants}: {param_row.get('Participant Id', 'synthetic')}")
        
        # Get data_instances for this participant if available (MODE 2)
        participant_data_instances = None
        if data_instances_dict is not None:
            pid = param_row.get('Participant Id', 'synthetic')
            if pid in data_instances_dict:
                participant_data_instances = data_instances_dict[pid]
        
        participant_trials = generate_participant_trials(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row.to_dict(),
            strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_trials=n_trials_per_participant,
            data_instances=participant_data_instances  # Pass data if available
        )
        
        all_trials.extend(participant_trials)
    
    return pd.DataFrame(all_trials)


# ===== 3. MAIN EXECUTION INTERFACE =====
def generate_trials_from_csv(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    strategies, XAI_types, training_cog_params,
    param_csv_path,
    mode='participant',  # 'trial', 'participant', or 'experiment'
    output_csv=None,
    lapse=0.1,
    random_seed=None,
    data_instances=None,
    data_instances_dict=None,
    **kwargs  # Additional args: n_participants, n_trials_per_participant, etc.
):
    """
    Main entry point: Generate trials from CSV in one of three execution modes.
    Supports two data input modes:
    
    DATA MODE 1: Load via ai_dataset_loader (default)
        - data_instances=None, data_instances_dict=None
        - Uses user_loader.get_counterfactual_trials() to load instances
    
    DATA MODE 2: Pre-computed data instances (optional)
        - Provide data_instances (single participant) or data_instances_dict (multiple)
        - Skips ai_dataset_loader entirely
    
    Args:
        model: Trained RL agent (PPO)
        user_loader: Participant data loader
        ai_dataset_loader: AI model predictions loader
        lr_df, dt_df, metadata_df: Explainer/interpretability data
        strategies: Dict[int, str] strategy ID to name mapping
        XAI_types: Dict[int, str] XAI type mapping
        training_cog_params: Dict of original training cognitive params
        param_csv_path: Path to CoXAM best params CSV
        mode: One of 'trial', 'participant', 'experiment'
        output_csv: Optional output CSV path
        lapse: Lapse rate for response noise
        random_seed: Reproducibility seed
        data_instances: For 'trial'/'participant' mode - list/array of pre-computed feature values.
                       Each element = one data instance (dict/array with features).
        data_instances_dict: For 'experiment' mode - dict mapping participant_id → list of instances.
        **kwargs: Additional arguments:
            - n_participants: For 'experiment' mode
            - n_trials_per_participant: For 'participant' or 'experiment' modes
            
    Returns:
        For 'trial': Dict with one trial result
        For 'participant': DataFrame with all trials for one participant
        For 'experiment': DataFrame with all trials for all participants
    """
    
    # Load parameters CSV
    param_df = load_best_params(param_csv_path)
    print(f"Loaded {len(param_df)} parameter rows from {param_csv_path}")
    
    if mode == 'trial':
        # Generate single trial
        print("Mode: SINGLE TRIAL")
        param_row = sample_param_row(param_df, random_seed=random_seed)
        result = generate_trial(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            data_instances=data_instances  # Pass data if provided
        )
        return result
    
    elif mode == 'participant':
        # Generate all trials for one participant
        print("Mode: PARTICIPANT (all trials for one participant)")
        param_row = sample_param_row(param_df, random_seed=random_seed)
        n_trials = kwargs.get('n_trials_per_participant', None)
        results_df = pd.DataFrame(generate_participant_trials(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_trials=n_trials,
            data_instances=data_instances  # Pass data if provided
        ))
        
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            results_df.to_csv(output_csv, index=False)
            print(f"Saved participant trials to: {output_csv}")
        
        return results_df
    
    elif mode == 'experiment':
        # Generate trials for all (or sampled) participants
        print("Mode: EXPERIMENT (all participants)")
        n_participants = kwargs.get('n_participants', None)
        n_trials = kwargs.get('n_trials_per_participant', None)
        
        results_df = generate_experiment_trials(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_df, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_participants=n_participants,
            n_trials_per_participant=n_trials,
            random_seed=random_seed,
            data_instances_dict=data_instances_dict  # Pass data dict if provided
        )
        
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            results_df.to_csv(output_csv, index=False)
            print(f"Saved experiment trials to: {output_csv}")
        
        return results_df
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trial', 'participant', or 'experiment'")


# ===== 4. EXAMPLE USAGE =====
if __name__ == "__main__":
    """
    Example of how to use this with actual data:
    
    # Load your data and model
    model = PPO.load("path/to/model.zip")
    user_loader = UserDataLoader(...)
    ai_dataset_loader = AIDatasetLoader(...)
    
    # Define strategy and XAI mappings
    strategies = {0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...}
    XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
    training_cog_params = {...}
    
    # Option 1: Generate single trial
    trial = generate_trials_from_csv(
        model, user_loader, ai_dataset_loader,
        lr_df, dt_df, metadata_df,
        strategies, XAI_types, training_cog_params,
        "user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv",
        mode='trial'
    )
    
    # Option 2: Generate all trials for one participant
    participant_df = generate_trials_from_csv(
        model, user_loader, ai_dataset_loader,
        lr_df, dt_df, metadata_df,
        strategies, XAI_types, training_cog_params,
        "user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv",
        mode='participant',
        output_csv='generated_trials_participant.csv',
        n_trials_per_participant=40  # optional limit
    )
    
    # Option 3: Generate all trials for entire experiment
    experiment_df = generate_trials_from_csv(
        model, user_loader, ai_dataset_loader,
        lr_df, dt_df, metadata_df,
        strategies, XAI_types, training_cog_params,
        "user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv",
        mode='experiment',
        output_csv='generated_trials_experiment.csv',
        n_participants=10,  # sample 10 participants from CSV
        n_trials_per_participant=20,  # 20 trials each
        random_seed=42
    )
    """
    print("See example usage in docstring above")
