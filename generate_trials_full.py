"""
Full trial generation from best CoXAM parameters using RL agent.
Adapted from counterfactual_simulations.py but WITHOUT scoring/fitting.
Supports: single trial, participant session, full experiment.
"""

import numpy as np
import pandas as pd
import random
import math
import os
from pathlib import Path

FEAT_COL, DELTA_COL = 'Changed feature index', 'Changed amount'


# ===== 1. PARAMETER LOADING =====
def load_best_params_csv(csv_path):
    """Load best parameters from CoXAM counterfactual simulation CSV."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} participant parameter sets from CSV")
    return df


def sample_param_row(param_df, random_seed=None):
    """Sample one random parameter row from CSV."""
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        param_row = param_df.sample(n=1, random_state=rng).iloc[0]
    else:
        param_row = param_df.sample(n=1).iloc[0]
    return param_row.to_dict()


# ===== 2. OBSERVATION BUILDER (from original) =====
def make_obs_builder(training_cog_params, strategies, XAI_types):
    """Build observation encoder (same as original)."""
    varied_param_names = []
    for k, v in training_cog_params.items():
        if k == "chi":
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            varied_param_names.append(k)

    xai_key_from_name = {v: k for k, v in XAI_types.items()}

    def build_obs(curr_chi, step_idx, with_xai, condition_name, shown_name,
                  counts, success_rates, mean_times, current_cog_params):
        cond_key = float(xai_key_from_name[condition_name])
        shown_key = float(xai_key_from_name.get(shown_name, 0))

        obs = [float(curr_chi), float(step_idx), float(with_xai), cond_key, shown_key]

        for sid in sorted(strategies.keys()):
            obs += [
                float(counts.get(sid, 0)),
                float(success_rates.get(sid, 0.0)),
                float(mean_times.get(sid, 0.0)),
            ]

        for name in varied_param_names:
            obs.append(float(current_cog_params.get(name, 0.0)))

        return np.array(obs, dtype=np.float32), varied_param_names

    return build_obs, varied_param_names


# ===== 3. SINGLE TRIAL GENERATION =====
def generate_single_trial(
    model, ai_loader, lr_exp, dt_exp, mem_lr, mem_dt,
    param_theta, bounds, transform, ai_model,
    strategies, XAI_types, training_cog_params,
    trial_data, step_idx=0, observation_builder=None,
    lapse=0.1, xai_idx=None, data_instance=None
):
    """
    Generate predictions for a SINGLE counterfactual trial.
    
    Args:
        data_instance: Optional pre-computed feature values (array/list).
                       If provided, skips ai_loader feature extraction.
                       If None, features are extracted via ai_loader.
    
    Returns dict with model predictions and trial metadata for this one trial.
    """
    if observation_builder is None:
        observation_builder, _ = make_obs_builder(training_cog_params, strategies, XAI_types)
    if xai_idx is None:
        xai_idx = {v: k for k, v in XAI_types.items()}
    
    rt = param_theta['retrieval_threshold']
    over_margin = param_theta['over_margin']
    chi = param_theta['chi']
    lapse_param = param_theta.get('lapse', lapse)
    
    # Extract trial info
    iid = trial_data.get('Instance Id', f'inst_{step_idx}')
    ai_pred = trial_data.get('AI prediction', None)
    feat_chosen = trial_data.get(FEAT_COL, 'a0')
    delta_chosen = trial_data.get(DELTA_COL, 0.0)
    with_xai = trial_data.get('Tested w/ XAI', 0)
    if isinstance(with_xai, str):
        with_xai = 1 if with_xai.lower().startswith('w') else 0
    
    condition = trial_data.get('condition', 'DT')
    shown = trial_data.get('XAIType', condition)
    
    # Build observation for RL model
    counts = {k: 0 for k in strategies.keys()}
    succ = {k: 0.0 for k in strategies.keys()}
    mtime = {k: 0.0 for k in strategies.keys()}
    
    current_cog_params = {
        'retrieval_threshold': rt,
        'lapse': lapse_param,
        'over_margin': over_margin,
    }
    
    obs, _ = observation_builder(
        curr_chi=chi,
        step_idx=step_idx,
        with_xai=with_xai,
        condition_name=condition,
        shown_name=shown,
        counts=counts,
        success_rates=succ,
        mean_times=mtime,
        current_cog_params=current_cog_params
    )
    
    # Get RL agent action
    action, _ = model.predict(obs, deterministic=True)
    strat_id = int(action[0])
    depth = int(action[1])
    strat = strategies.get(strat_id, 'unknown')
    mode = 'read' if with_xai else 'retrieve'
    
    # Fallback strategy selection (from original code)
    if condition == 'DT':
        if strat in ('zero_out_lr_displayed', 'zero_out_lr_heuristic', 'recall_change_lr'):
            strat = 'change_path_dt'
    elif condition == 'LR':
        if strat in ('change_path_dt', 'recall_change_dt'):
            strat = 'zero_out_lr_heuristic'
    elif shown == 'DT':
        if strat in ('zero_out_lr_displayed',):
            strat = 'change_path_dt'
    
    # Return trial record (without executing full counterfactual logic)
    trial_record = {
        'instance_id': iid,
        'ai_prediction': ai_pred,
        'participant_choice_feature': feat_chosen,
        'participant_choice_delta': delta_chosen,
        'with_xai': with_xai,
        'xai_shown': shown,
        'condition': condition,
        'step': step_idx,
        
        # Model outputs
        'model_strategy_id': strat_id,
        'model_strategy': strat,
        'model_depth': depth if strat == 'change_path_dt' else None,
        
        # Hyperparams used
        'retrieval_threshold': rt,
        'over_margin': over_margin,
        'chi': chi,
        'lapse': lapse_param,
        
        # Data instance (if provided)
        'data_instance': data_instance,
    }
    
    return trial_record


# ===== 4. PARTICIPANT SESSION GENERATION =====
def generate_participant_session(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    param_row, strategies, XAI_types, training_cog_params,
    lapse=0.1, n_trials=None, data_instances=None
):
    """
    Generate entire session for one participant using best params from CSV.
    
    DUAL MODE:
    - Mode 1 (data_instances=None): Load instances from user_loader via ai_dataset_loader
    - Mode 2 (data_instances provided): Use pre-computed data, skip ai_dataset_loader
    
    Args:
        model: Trained RL agent
        user_loader: User/participant data loader
        ai_dataset_loader: AI model loader
        lr_df, dt_df, metadata_df: Explainer data
        param_row: Dict with 'Participant Id', 'app_id', 'Best retrieval_threshold', etc.
        strategies: Dict[int, str] strategy mapping
        XAI_types: Dict[int, str] XAI type mapping
        training_cog_params: Original training parameters
        lapse: Lapse rate
        n_trials: Limit number of trials (None = all)
        data_instances: Optional list/array of pre-computed feature values.
                       Each element should be a dict or array-like with feature values.
                       If provided, skips ai_dataset_loader. If None, uses loader.
        
    Returns:
        List of trial records for this participant
    """
    
    participant_id = param_row.get('Participant Id', 'synthetic')
    app_id = param_row['app_id']
    condition = param_row['condition']
    complexity = param_row['complexity']
    
    # MODE SELECTION
    if data_instances is not None:
        # MODE 2: Use pre-computed data instances (skip ai_dataset_loader)
        print(f"  MODE 2: Using {len(data_instances)} pre-computed data instances")
        info = {'model': 'mlp', 'app_id': app_id, 'condition': condition, 'complexity': complexity}
        bounds = {f'a{i}': (0, 1) for i in range(len(data_instances[0]) if data_instances else 6)}
        # Convert data_instances to trial format
        cf_trials = pd.DataFrame([
            {'Instance Id': f'inst_{i}', 'data_instance': inst}
            for i, inst in enumerate(data_instances)
        ])
    else:
        # MODE 1: Load via ai_dataset_loader (original behavior)
        print(f"  MODE 1: Loading instances via ai_dataset_loader")
        try:
            info = user_loader.get_participant_info(participant_id)
            ai_loader = ai_dataset_loader.get_loader_for_app_model(app_id, info['model'])
            bounds = ai_loader.get_bounds_for_app(app_id)
            transform, ai_model = ai_dataset_loader.load_transform_and_ai(app_id)
            
            cf_trials = user_loader.get_counterfactual_trials(participant_id)
            fwd_trials = user_loader.get_forward_trials(participant_id)
        except Exception as e:
            print(f"  Warning: Could not fully load data for {participant_id}: {e}")
            # Generate dummy metadata
            info = {'model': 'mlp', 'app_id': app_id, 'condition': condition, 'complexity': complexity}
            bounds = {f'a{i}': (0, 1) for i in range(6)}
            cf_trials = pd.DataFrame([{'Instance Id': f'inst_{i}'} for i in range(n_trials or 40)])
    
    # Setup interpretability models
    lr_exp = create_lr_interpreter(lr_df, metadata_df, app_id, complexity) if lr_df is not None else None
    dt_exp = create_dt_interpreter(dt_df, metadata_df, app_id, complexity) if dt_df is not None else None
    
    # Setup memory (would be actual memory initialization)
    mem_lr = None  # _make_memory(retrieval_threshold=param_row['Best retrieval_threshold'], ...)
    mem_dt = None  # _make_memory(retrieval_threshold=param_row['Best retrieval_threshold'], ...)
    
    # Generate observation builder
    obs_builder, _ = make_obs_builder(training_cog_params, strategies, XAI_types)
    
    # Limit trials if requested
    if n_trials is not None:
        cf_trials = cf_trials.head(n_trials)
    
    # Generate trial-by-trial predictions
    trials_log = []
    xai_idx = {v: k for k, v in XAI_types.items()}
    
    for step_idx, (_, cf_row) in enumerate(cf_trials.iterrows()):
        trial_data = {**cf_row.to_dict(), 'condition': condition}
        # Extract data_instance if present (from MODE 2)
        data_instance = trial_data.pop('data_instance', None)
        
        trial_record = generate_single_trial(
            model=model,
            ai_loader=None,  # Would be used for feature extraction
            lr_exp=lr_exp,
            dt_exp=dt_exp,
            mem_lr=mem_lr,
            mem_dt=mem_dt,
            param_theta=param_row,
            bounds=bounds,
            transform=None,  # Would be used for AI prediction
            ai_model=None,  # Would be used for AI prediction
            strategies=strategies,
            XAI_types=XAI_types,
            training_cog_params=training_cog_params,
            trial_data=trial_data,
            step_idx=step_idx,
            observation_builder=obs_builder,
            lapse=lapse,
            xai_idx=xai_idx,
            data_instance=data_instance  # Pass data_instance to single trial generator
        )
        
        # Add participant metadata
        trial_record.update({
            'participant_id': participant_id,
            'app_id': app_id,
            'complexity': complexity,
            'condition': condition,
        })
        
        trials_log.append(trial_record)
    
    print(f"  ✓ Generated {len(trials_log)} trials for participant {participant_id}")
    return trials_log


# ===== 5. FULL EXPERIMENT GENERATION =====
def generate_full_experiment(
    model, user_loader, ai_dataset_loader,
    lr_df, dt_df, metadata_df,
    param_df, strategies, XAI_types, training_cog_params,
    lapse=0.1, n_participants=None, n_trials_per_participant=None, random_seed=None,
    data_instances_dict=None
):
    """
    Generate trials for entire experiment across multiple participants.
    
    Args:
        param_df: Full DataFrame of parameters from CSV
        n_participants: Number of participants to sample (None = all)
        n_trials_per_participant: Limit trials per participant (None = all)
        random_seed: Reproducibility
        data_instances_dict: Optional dict mapping participant_id → list of data instances.
                            If provided, uses MODE 2 (pre-computed data).
                            If None, uses MODE 1 (ai_dataset_loader).
        
    Returns:
        DataFrame with all generated trials
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Sample participants
    if n_participants is not None and n_participants < len(param_df):
        param_rows = param_df.sample(n=n_participants, random_state=random_seed)
        print(f"Sampling {n_participants} participants from {len(param_df)} total")
    else:
        param_rows = param_df
        n_participants = len(param_df)
    
    all_trials = []
    
    for idx, (_, param_row) in enumerate(param_rows.iterrows(), 1):
        pid = param_row['Participant Id']
        print(f"[{idx}/{n_participants}] Generating session for participant {pid}")
        
        # Get data_instances for this participant if available (MODE 2)
        participant_data_instances = None
        if data_instances_dict is not None and pid in data_instances_dict:
            participant_data_instances = data_instances_dict[pid]
        
        session_trials = generate_participant_session(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row.to_dict(),
            strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_trials=n_trials_per_participant,
            data_instances=participant_data_instances
        )
        
        all_trials.extend(session_trials)
    
    return pd.DataFrame(all_trials)


# ===== 6. MAIN INTERFACE =====
def generate_trials_from_params_csv(
    model,
    user_loader,
    ai_dataset_loader,
    lr_df,
    dt_df,
    metadata_df,
    strategies,
    XAI_types,
    training_cog_params,
    param_csv_path,
    mode='participant',  # 'trial', 'participant', or 'experiment'
    output_csv=None,
    lapse=0.1,
    random_seed=None,
    data_instances=None,
    data_instances_dict=None,
    **kwargs
):
    """
    Main entry point: Generate trials from best params CSV in three execution modes.
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
        ai_dataset_loader: AI predictions loader
        lr_df, dt_df, metadata_df: Explainer data
        strategies: Dict[int, str] strategy ID → name
        XAI_types: Dict[int, str] XAI type ID → name
        training_cog_params: Original training cognitive params
        param_csv_path: Path to CoXAM best params CSV
        mode: 'trial' | 'participant' | 'experiment'
        output_csv: Optional output path
        lapse: Lapse parameter
        random_seed: For reproducibility
        data_instances: For 'trial'/'participant' mode - list/array of pre-computed feature values.
                       Each element = one data instance (dict/array with features).
        data_instances_dict: For 'experiment' mode - dict mapping participant_id → list of instances.
        **kwargs:
            n_participants: For 'experiment' mode
            n_trials_per_participant: For 'participant' or 'experiment' mode
    
    Returns:
        For 'trial': Dict with one trial result
        For 'participant': DataFrame with one participant's trials
        For 'experiment': DataFrame with all trials
    """
    
    # Load CSV
    param_df = load_best_params_csv(param_csv_path)
    
    print(f"\n{'='*60}")
    print(f"TRIAL GENERATION MODE: {mode.upper()}")
    print(f"{'='*60}\n")
    
    if mode == 'trial':
        # Single trial
        print("Generating single trial...")
        param_row = sample_param_row(param_df, random_seed=random_seed)
        print(f"  Sampled participant: {param_row.get('Participant Id', 'synthetic')}")
        
        # Generate one trial
        trials = generate_participant_session(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_trials=1,
            data_instances=data_instances  # Pass data if provided
        )
        result = trials[0] if trials else {}
        return result
    
    elif mode == 'participant':
        # One participant, all trials
        print("Generating participant session (all trials)...")
        param_row = sample_param_row(param_df, random_seed=random_seed)
        print(f"  Sampled participant: {param_row.get('Participant Id', 'synthetic')}")
        
        n_trials = kwargs.get('n_trials_per_participant', None)
        trials = generate_participant_session(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_row, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_trials=n_trials,
            data_instances=data_instances  # Pass data if provided
        )
        
        result_df = pd.DataFrame(trials)
        print(f"\n✓ Generated {len(result_df)} trials for 1 participant")
        
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            result_df.to_csv(output_csv, index=False)
            print(f"✓ Saved to: {output_csv}")
        
        return result_df
    
    elif mode == 'experiment':
        # All participants, all trials
        print("Generating full experiment...")
        n_participants = kwargs.get('n_participants', None)
        n_trials = kwargs.get('n_trials_per_participant', None)
        
        result_df = generate_full_experiment(
            model, user_loader, ai_dataset_loader,
            lr_df, dt_df, metadata_df,
            param_df, strategies, XAI_types, training_cog_params,
            lapse=lapse,
            n_participants=n_participants,
            n_trials_per_participant=n_trials,
            random_seed=random_seed,
            data_instances_dict=data_instances_dict  # Pass data dict if provided
        )
        
        print(f"\n✓ Generated {len(result_df)} total trials across {result_df['participant_id'].nunique()} participants")
        
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            result_df.to_csv(output_csv, index=False)
            print(f"✓ Saved to: {output_csv}")
        
        return result_df
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose: 'trial', 'participant', 'experiment'")


# ===== 7. HELPER FUNCTIONS (stubs) =====
def create_lr_interpreter(lr_df, metadata_df, app_id, complexity):
    """Stub: Create logistic regression interpreter."""
    return None


def create_dt_interpreter(dt_df, metadata_df, app_id, complexity):
    """Stub: Create decision tree interpreter."""
    return None


# ===== 8. EXAMPLE USAGE =====
if __name__ == "__main__":
    """
    Example usage:
    
    result = generate_trials_from_params_csv(
        model=your_ppo_model,
        user_loader=user_loader,
        ai_dataset_loader=ai_loader,
        lr_df=lr_data,
        dt_df=dt_data,
        metadata_df=meta_data,
        strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
        XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
        training_cog_params={...},
        param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
        mode='experiment',  # or 'participant' or 'trial'
        output_csv='generated_trials.csv',
        random_seed=42,
        n_participants=5,
        n_trials_per_participant=20
    )
    """
    print(__doc__)
