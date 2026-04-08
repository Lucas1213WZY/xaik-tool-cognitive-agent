"""
Example: Generate Human-Like Synthetic Participant Data

This script demonstrates the complete workflow for:
1. Extracting parameter distributions from fitted CoAX data
2. Sampling parameters for synthetic participants
3. Simulating realistic trial responses
4. Exporting results to CSV

Usage:
    python example_generate_synthetic_data.py \\
        --fitted-data fitted_data.csv \\
        --output-dir synthetic_output/ \\
        --n-participants 10 \\
        --n-trials 40
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import numpy as np

from src.user_simulation import (
    ParameterEstimator,
    ParameterSampler,
    TrialSimulator,
    TrialConfig,
    generate_participant_id,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_parameter_distributions(fitted_data_path: str, 
                                   output_dir: str) -> str:
    """
    Extract and save parameter distributions from fitted data.
    
    Args:
        fitted_data_path: Path to CSV with fitted parameters
        output_dir: Directory to save distributions JSON
        
    Returns:
        Path to saved distributions JSON
    """
    logger.info(f"Loading fitted data from {fitted_data_path}")
    estimator = ParameterEstimator()
    estimator.load_fitted_data(fitted_data_path)
    
    logger.info("Estimating parameter distributions...")
    estimator.estimate_distributions()
    
    # Save distributions
    output_path = Path(output_dir) / "parameter_distributions.json"
    estimator.save_distributions(str(output_path))
    logger.info(f"Saved distributions to {output_path}")
    
    # Save summary
    summary_path = Path(output_dir) / "distributions_summary.json"
    estimator.export_summary_stats(str(summary_path))
    logger.info(f"Saved summary to {summary_path}")
    
    return str(output_path)


def generate_synthetic_participants(distributions_path: str,
                                   output_dir: str,
                                   n_participants: int = 10,
                                   n_trials: int = 40,
                                   dataset: str = "wine_quality",
                                   strategy: str = "sensitive_features",
                                   xai_type: str = "Importance",
                                   tested_with_xai: bool = True,
                                   ai_dataset_loader: Optional[object] = None,
                                   seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic participant data.
    
    Args:
        distributions_path: Path to parameter distributions JSON
        output_dir: Directory to save results
        n_participants: Number of synthetic participants
        n_trials: Number of trials per participant
        dataset: Dataset name
        strategy: Strategy name
        xai_type: XAI type (Importance, Attribution, None)
        tested_with_xai: Whether to show explanations
        ai_dataset_loader: AIDatasetLoader instance (optional)
        seed: Random seed
        
    Returns:
        DataFrame with all trial results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading parameter distributions from {distributions_path}")
    sampler = ParameterSampler(seed=seed)
    sampler.load_distributions(distributions_path)
    
    logger.info(f"Generating {n_participants} synthetic participants...")
    simulator = TrialSimulator()
    if ai_dataset_loader is not None:
        simulator.setup_dependencies(strategy_registry=None)
    
    all_results = []
    
    for i in range(n_participants):
        logger.info(f"  Participant {i+1}/{n_participants}")
        
        # Sample parameters
        try:
            params = sampler.sample(
                dataset=dataset,
                strategy=strategy,
                xai_type=xai_type,
                tested_with_xai="w/ XAI" if tested_with_xai else "w/o XAI",
                method="truncated_normal"  # Realistic distribution
            )
        except KeyError as e:
            logger.warning(f"Could not sample parameters: {e}")
            logger.info(f"Available distributions: {sampler.list_strategies_for_dataset(dataset)}")
            continue
        
        # Create trial config
        config = TrialConfig(
            participant_id=generate_participant_id(prefix=f"syn_{i:03d}_"),
            dataset_name=dataset,
            strategy_name=strategy,
            xai_type=xai_type,
            tested_with_xai=tested_with_xai,
            cognitive_params=params,
            n_trials=n_trials,
            ai_dataset_loader=ai_dataset_loader,
            random_seed=seed,
        )
        
        # Simulate trials
        try:
            results = simulator.simulate(config)
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Failed to simulate participant {i}: {e}")
            continue
    
    # Convert to DataFrame
    logger.info(f"Converting {len(all_results)} trial results to DataFrame...")
    df = simulator.results_to_dataframe(all_results)
    
    # Save to CSV
    output_csv = output_dir / "synthetic_participant_data.csv"
    simulator.export_to_csv(str(output_csv), all_results)
    logger.info(f"Exported trial data to {output_csv}")
    
    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary_stats = {
        "n_participants": n_participants,
        "n_trials_per_participant": n_trials,
        "total_trials": len(df),
        "dataset": dataset,
        "strategy": strategy,
        "xai_type": xai_type,
        "tested_with_xai": tested_with_xai,
        "mean_accuracy_ai": float(df["Response==AI"].mean()),
        "mean_accuracy_explainer": float(df["Response==Explainer"].mean()),
        "mean_response_time": float(df["Response Time (s)"].mean()),
        "std_response_time": float(df["Response Time (s)"].std()),
    }
    
    summary_json = output_dir / "synthetic_data_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_json}")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate human-like synthetic participant data"
    )
    parser.add_argument(
        "--fitted-data",
        type=str,
        required=True,
        help="Path to CSV file with fitted parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synthetic_output/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=10,
        help="Number of synthetic participants to generate"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of trials per participant"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wine_quality",
        help="Dataset name (adult, wine_quality, forest_cover)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sensitive_features",
        help="Strategy name"
    )
    parser.add_argument(
        "--xai-type",
        type=str,
        default="Importance",
        help="XAI type (Importance, Attribution, None)"
    )
    parser.add_argument(
        "--with-xai",
        action="store_true",
        default=True,
        help="Whether to show explanations in trials"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract parameter distributions
    logger.info("=" * 60)
    logger.info("STEP 1: Extract Parameter Distributions")
    logger.info("=" * 60)
    distributions_path = extract_parameter_distributions(
        args.fitted_data,
        args.output_dir
    )
    
    # Step 2: Generate synthetic participants
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Generate Synthetic Participants")
    logger.info("=" * 60)
    df = generate_synthetic_participants(
        distributions_path,
        args.output_dir,
        n_participants=args.n_participants,
        n_trials=args.n_trials,
        dataset=args.dataset,
        strategy=args.strategy,
        xai_type=args.xai_type,
        tested_with_xai=args.with_xai,
        seed=args.seed
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generated {len(df)} trial results")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Main output: {Path(args.output_dir) / 'synthetic_participant_data.csv'}")


if __name__ == "__main__":
    main()
