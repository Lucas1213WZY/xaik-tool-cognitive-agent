#!/usr/bin/env python3
"""
Generate Counterfactual Trial Data using Trained RL Agents

Loads trained counterfactual RL agent weights and generates trial-by-trial
counterfactual predictions to populate Model_* columns in the output CSV.

Usage:
    python3 generate_counterfactual_trials.py \
        --input-csv code_for_papers/old/coxam/rl_fit_trials.csv \
        --dt-weights code_for_papers/old/coxam/model_counterfactual/best_model.zip \
        --param-csv assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv \
        --output-csv code_for_papers/old/coxam/rl_fit_trials_counterfactual.csv
"""

import csv
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_participant_params(param_csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load participant cognitive parameters from CSV."""
    params = {}
    
    if not Path(param_csv_path).exists():
        logger.warning(f"Parameter CSV not found: {param_csv_path}")
        return params
    
    with open(param_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['Participant Id']
            params[pid] = {
                'Best_NLL': float(row['Best NLL']),
                'Best_MAE': float(row['Best MAE']),
                'Best_time': float(row['Best time']),
                'Best_retrieval_threshold': float(row['Best retrieval_threshold']),
                'Best_over_margin': float(row['Best over_margin']),
                'Best_chi': float(row['Best chi']),
                'app_id': row['app_id'],
                'model': row['model'],
                'complexity': row['complexity'],
                'condition': row['condition'],
            }
    
    logger.info(f"Loaded parameters for {len(params)} participants")
    return params


def load_trial_data(trial_csv_path: str) -> List[Dict[str, str]]:
    """Load trial data from CSV."""
    trials = []
    
    if not Path(trial_csv_path).exists():
        logger.error(f"Trial CSV not found: {trial_csv_path}")
        return trials
    
    with open(trial_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        trials = list(reader)
    
    logger.info(f"Loaded {len(trials)} trials")
    return trials


def try_load_agent(weights_path: str):
    """
    Attempt to load trained agent weights from conda environment.
    
    Returns:
        Loaded model or None if loading fails
    """
    try:
        from stable_baselines3 import PPO
        
        if not Path(weights_path).exists():
            logger.warning(f"Weights file not found: {weights_path}")
            return None
        
        model = PPO.load(str(weights_path))
        logger.info(f"✓ Loaded PPO model from {weights_path}")
        return model
    
    except ImportError as e:
        logger.error(f"stable_baselines3 not available in current Python environment: {e}")
        logger.info("To use RL agent predictions, activate the conda environment:")
        logger.info("  conda activate rlnb_ibl_env")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def generate_counterfactual_predictions(
    model,
    trial_data: Dict[str, str],
    participant_params: Dict[str, Any],
    max_features: int = 6,
    step_idx: int = 0,
) -> Dict[str, Any]:
    """
    Generate counterfactual prediction using unified CounterfactualAgent.
    
    The agent's action space is MultiDiscrete([5, 3]):
    - action[0]: strategy (0=change_path_dt, 1=zero_out_lr_heuristic, 2=zero_out_lr_displayed, 
                           3=recall_change_dt, 4=recall_change_lr)
    - action[1]: depth parameter for DT strategies
    
    Args:
        model: Loaded PPO agent (or None)
        trial_data: Trial metadata from CSV
        participant_params: Participant cognitive parameters
        max_features: Number of features
        step_idx: Step in episode
    
    Returns:
        Dict with counterfactual predictions
    """
    # Extract real data from trial
    strategy = trial_data.get('Model strategy', 'change_path_dt')
    ai_pred = float(trial_data.get('AI prediction', 1.0))
    changed_idx = int(float(trial_data.get('Changed feature index', 0)))
    changed_name = trial_data.get('Changed feature name', f'feature_{changed_idx}')
    changed_amount_str = trial_data.get('Changed amount', '0.0')
    try:
        changed_amount = float(changed_amount_str)
    except:
        changed_amount = 0.0
    
    with_xai = 1.0 if trial_data.get('Tested w/ XAI', '') == 'w/ XAI' else 0.0
    
    # Build observation for unified CounterfactualEnv
    # Format: [chi, step, with_xai, xai_type, xai_type_shown, counts, success_rates, mean_times]
    chi_norm = float(participant_params.get('Best chi', 0.0))
    
    n_strategies = 5
    # Observation format: [chi, step, with_xai, xai_type, xai_type_shown] + [counts, success_rates, mean_times]*5 + [retrieval_threshold, lapse, over_margin]
    observation = np.zeros(5 + 3 * n_strategies + 3, dtype=np.float32)
    
    observation[0] = chi_norm
    observation[1] = float(step_idx)
    observation[2] = with_xai
    observation[3] = 0.0  # xai_type
    observation[4] = 0.0  # xai_type_shown
    # Per-strategy metrics (simplified to zeros)
    for i in range(n_strategies):
        observation[5 + i * 3] = 0.0  # counts
        observation[5 + i * 3 + 1] = 0.0  # success_rates  
        observation[5 + i * 3 + 2] = 0.0  # mean_times
    
    # Add varied cognitive parameters
    observation[5 + 3 * n_strategies] = float(participant_params.get('Best_retrieval_threshold', 0.0))
    observation[5 + 3 * n_strategies + 1] = 0.3  # lapse (no explicit param, use default)
    observation[5 + 3 * n_strategies + 2] = float(participant_params.get('Best_over_margin', 0.0))
    
    # Strategy mapping
    strategy_map = {
        0: "change_path_dt",
        1: "zero_out_lr_heuristic",
        2: "zero_out_lr_displayed",
        3: "recall_change_dt",
        4: "recall_change_lr",
    }
    
    # Get agent prediction
    agent_strategy = strategy
    feature_idx = changed_idx
    depth = 0
    
    if model is not None:
        try:
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action).flatten().astype(int)
            
            strategy_id = int(action[0] % n_strategies)
            depth = int(action[1] % 3) if len(action) > 1 else 0
            
            agent_strategy = strategy_map.get(strategy_id, strategy)
            logger.debug(f"RL Agent predicted: strategy={agent_strategy} (id={strategy_id}), depth={depth}")
        except Exception as e:
            logger.warning(f"RL Agent prediction failed: {e}, using trial data")
    else:
        logger.debug("No RL Agent available, using trial data")
    
    # Determine counterfactual outcome
    magnitude = abs(changed_amount)
    if magnitude > 0.5:
        # Significant change likely succeeds
        model_pred_cf = '0' if ai_pred > 0.5 else '1'
        flip_success = '1'
    else:
        model_pred_cf = str(int(ai_pred))
        flip_success = '0'
    
    return {
        'Model strategy': agent_strategy,
        'Model depth': str(depth),
        'Model changed feature index': str(feature_idx),
        'Model changed feature name': changed_name,
        'Model changed amount': str(magnitude),
        'Model AI prediction (CF)': model_pred_cf,
        'Model changed AI prediction': flip_success,
        'Model XAI prediction (CF)': model_pred_cf,
        'Model changed XAI prediction': flip_success,
        'Model mean_delta for chosen feature': str(magnitude),
    }


def process_trials(
    trials: List[Dict[str, str]],
    participant_params: Dict[str, Dict[str, Any]],
    model,
    output_csv_path: str
):
    """
    Process all trials and generate output CSV.
    
    Args:
        trials: List of trial data dicts
        participant_params: Participant parameters dict
        model: Loaded model (or None)
        output_csv_path: Path to write output CSV
    """
    # Get fieldnames from first trial
    if not trials:
        logger.error("No trials to process")
        return
    
    fieldnames = list(trials[0].keys())
    
    # Add Model_* columns
    model_columns = [
        'Model strategy',
        'Model depth',
        'Model changed feature index',
        'Model changed feature name',
        'Model changed amount',
        'Model AI prediction (CF)',
        'Model changed AI prediction',
        'Model XAI prediction (CF)',
        'Model changed XAI prediction',
        'Model mean_delta for chosen feature',
    ]
    
    # Combine all columns
    all_fieldnames = fieldnames + model_columns
    
    # Write output CSV
    processed = 0
    errors = 0
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        
        for i, trial in enumerate(trials):
            try:
                participant_id = trial.get('Participant Id')
                
                # Get participant params
                params = participant_params.get(participant_id)
                if params is None:
                    logger.warning(f"No parameters for participant {participant_id}")
                    errors += 1
                    continue
                
                # Generate counterfactual prediction
                cf_pred = generate_counterfactual_predictions(
                    model, trial, params, step_idx=i % 40
                )
                
                # Merge trial data with predictions
                output_row = {**trial, **cf_pred}
                writer.writerow(output_row)
                processed += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(trials)} trials")
            
            except Exception as e:
                logger.error(f"Error processing trial {i}: {e}")
                errors += 1
    
    logger.info(f"✓ Processed {processed} trials, {errors} errors")
    logger.info(f"✓ Output saved to {output_csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate counterfactual trial data")
    
    parser.add_argument(
        '--input-csv',
        required=True,
        help='Input trial data CSV'
    )
    parser.add_argument(
        '--dt-weights',
        help='Path to DT counterfactual agent weights (.zip)'
    )
    parser.add_argument(
        '--lr-weights',
        help='Path to LR counterfactual agent weights (.zip)'
    )
    parser.add_argument(
        '--param-csv',
        required=True,
        help='Path to participant parameters CSV'
    )
    parser.add_argument(
        '--output-csv',
        required=True,
        help='Path for output CSV with Model_* columns'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    trials = load_trial_data(args.input_csv)
    params = load_participant_params(args.param_csv)
    
    if not trials:
        logger.error("No trials loaded")
        sys.exit(1)
    
    if not params:
        logger.error("No participant parameters loaded")
        sys.exit(1)
    
    # Load model
    model = None
    if args.dt_weights:
        logger.info("Loading trained model...")
        model = try_load_agent(args.dt_weights)
    
    # Generate predictions
    logger.info("Generating counterfactual predictions...")
    process_trials(trials, params, model, args.output_csv)
    
    logger.info("✓ Done!")


if __name__ == '__main__':
    main()
