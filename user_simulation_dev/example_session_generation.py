"""
Example: Generate Complete Session of Synthetic Participant Data

Demonstrates the high-level SessionGenerator API for creating realistic
synthetic participant data with:
- Multiple strategies with customizable distribution
- XAI explanations and explainer models
- Sensible defaults from parameter distributions
- Batch generation and export

Usage (run from project root):
    python -m user_simulation.example_session_generation \\
        --distribution-file distributions.json \\
        --output-dir session_output/ \\
        --n-participants 50 \\
        --n-trials 40
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List
import logging

from . import (
    SessionGenerator,
    SessionConfig,
    StrategyConfig,
    DistributionLoader,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_wine_quality_session(
    distribution_file: str,
    n_participants: int = 50,
    n_trials: int = 40,
    random_seed: Optional[int] = None
) -> SessionConfig:
    """
    Create a session configuration for wine_quality dataset.
    
    Strategy distribution:
    - 50% sensitive_features (Importance with XAI)
    - 30% salient_features (Importance with and without XAI split)
    - 20% importance_categorization (Importance without XAI)
    """
    return SessionConfig(
        dataset_name="wine_quality",
        n_participants=n_participants,
        n_trials_per_participant=n_trials,
        distribution_file=distribution_file,
        sampling_method="truncated_normal",
        random_seed=random_seed,
        strategy_configs=[
            StrategyConfig(
                strategy_name="sensitive_features",
                percentage=50.0,
                xai_type="importance",
                tested_with_xai=True,
            ),
            StrategyConfig(
                strategy_name="salient_features",
                percentage=30.0,
                xai_type="importance",
                tested_with_xai=True,
            ),
            StrategyConfig(
                strategy_name="importance_categorization",
                percentage=20.0,
                xai_type="importance",
                tested_with_xai=False,
            ),
        ]
    )


def create_multi_dataset_session(
    distribution_file: str,
    dataset: str = "adult",
    n_participants: int = 30,
    n_trials: int = 40
) -> SessionConfig:
    """
    Create a balanced session for a given dataset with multiple strategies.
    """
    strategies = [
        StrategyConfig(
            strategy_name="sensitive_features",
            percentage=50.0,
            xai_type="importance",
            tested_with_xai=True,
        ),
        StrategyConfig(
            strategy_name="salient_features",
            percentage=50.0,
            xai_type="importance",
            tested_with_xai=True,
        ),
    ]
    
    return SessionConfig(
        dataset_name=dataset,
        n_participants=n_participants,
        n_trials_per_participant=n_trials,
        distribution_file=distribution_file,
        sampling_method="truncated_normal",
        strategy_configs=strategies,
    )


def create_xai_comparison_session(
    distribution_file: str,
    dataset: str = "wine_quality",
    n_participants_per_group: int = 25,
    n_trials: int = 40
) -> SessionConfig:
    """
    Create a session to compare with/without XAI effects.
    
    Two equal groups:
    - 50% sensitive_features WITH XAI (Importance)
    - 50% sensitive_features WITHOUT XAI (Importance)
    """
    return SessionConfig(
        dataset_name=dataset,
        n_participants=n_participants_per_group * 2,
        n_trials_per_participant=n_trials,
        distribution_file=distribution_file,
        sampling_method="truncated_normal",
        strategy_configs=[
            StrategyConfig(
                strategy_name="sensitive_features",
                percentage=50.0,
                xai_type="importance",
                tested_with_xai=True,
                # Will use default parameters from distribution
            ),
            StrategyConfig(
                strategy_name="sensitive_features",
                percentage=50.0,
                xai_type="importance",
                tested_with_xai=False,
                # Will use distribution params for without-XAI condition
            ),
        ]
    )


def create_custom_params_session(
    distribution_file: str,
    dataset: str = "wine_quality",
    n_participants: int = 20,
    n_trials: int = 40,
    custom_params: Optional[dict] = None
) -> SessionConfig:
    """
    Create a session with custom (override) cognitive parameters.
    
    Allows experimenting with specific parameter values instead of
    sampling from the distribution.
    """
    if custom_params is None:
        custom_params = {
            "sensitivity": 75.0,
            "k": 2,
            "retrieval_threshold": -2.5
        }
    
    return SessionConfig(
        dataset_name=dataset,
        n_participants=n_participants,
        n_trials_per_participant=n_trials,
        distribution_file=distribution_file,
        sampling_method="truncated_normal",
        strategy_configs=[
            StrategyConfig(
                strategy_name="sensitive_features",
                percentage=100.0,
                xai_type="importance",
                tested_with_xai=True,
                cognitive_params=custom_params,  # Override distribution
            ),
        ]
    )


def print_distribution_summary(distribution_file: str) -> None:
    """Print summary of available distributions."""
    logger.info("Loading distribution summary...")
    loader = DistributionLoader()
    loader.load_from_json(distribution_file)
    
    summary = loader.get_summary()
    logger.info(f"\nTotal distributions: {summary['total_distributions']}")
    
    logger.info("\nAvailable datasets and strategies:")
    for dataset, dataset_info in summary["datasets"].items():
        logger.info(f"\n  {dataset.upper()}:")
        logger.info(f"    Total distributions: {dataset_info['total']}")
        
        for strategy, strategy_info in dataset_info["strategies"].items():
            logger.info(f"    - {strategy}:")
            for xai_type, conditions in strategy_info["xai_types"].items():
                for cond in conditions:
                    logger.info(f"        {cond['tested_with_xai']}: {cond['n_samples']} samples")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate complete session of synthetic participant data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Generate wine_quality session with default strategy distribution
  python example_session_generation.py \\
      --distribution-file distributions.json \\
      --n-participants 50 \\
      --n-trials 40 \\
      --output-dir wine_output/

  # Generate adult dataset session
  python example_session_generation.py \\
      --distribution-file distributions.json \\
      --dataset adult \\
      --n-participants 30 \\
      --output-dir adult_output/

  # Compare with/without XAI effects
  python example_session_generation.py \\
      --distribution-file distributions.json \\
      --session-type xai_comparison \\
      --output-dir xai_comparison_output/

  # Print available distributions
  python example_session_generation.py \\
      --distribution-file distributions.json \\
      --list-distributions
        """
    )
    
    parser.add_argument(
        "--distribution-file",
        type=str,
        required=True,
        help="Path to distributions JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wine_quality",
        help="Dataset name (adult, wine_quality, forest_cover)"
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=50,
        help="Number of participants"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Trials per participant"
    )
    parser.add_argument(
        "--session-type",
        type=str,
        default="default",
        choices=["default", "xai_comparison", "custom_params"],
        help="Type of session to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="session_output/",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--list-distributions",
        action="store_true",
        help="Print available distributions and exit"
    )
    
    args = parser.parse_args()
    
    # Check if just listing distributions
    if args.list_distributions:
        logger.info("AVAILABLE DISTRIBUTIONS")
        logger.info("=" * 70)
        print_distribution_summary(args.distribution_file)
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create session config based on type
    logger.info("Creating session configuration...")
    if args.session_type == "default":
        if args.dataset == "wine_quality":
            config = create_wine_quality_session(
                distribution_file=args.distribution_file,
                n_participants=args.n_participants,
                n_trials=args.n_trials,
                random_seed=args.seed
            )
        else:
            config = create_multi_dataset_session(
                distribution_file=args.distribution_file,
                dataset=args.dataset,
                n_participants=args.n_participants,
                n_trials=args.n_trials
            )
    
    elif args.session_type == "xai_comparison":
        config = create_xai_comparison_session(
            distribution_file=args.distribution_file,
            dataset=args.dataset,
            n_participants_per_group=args.n_participants // 2,
            n_trials=args.n_trials
        )
    
    elif args.session_type == "custom_params":
        config = create_custom_params_session(
            distribution_file=args.distribution_file,
            dataset=args.dataset,
            n_participants=args.n_participants,
            n_trials=args.n_trials
        )
    
    else:
        raise ValueError(f"Unknown session type: {args.session_type}")
    
    # Generate session
    logger.info("\nGenerating session...")
    logger.info("=" * 70)
    
    generator = SessionGenerator()
    results = generator.generate(config)
    
    # Export results
    logger.info("\nExporting results...")
    output_csv = output_dir / "session_trials.csv"
    generator.export_to_csv(str(output_csv))
    
    output_summary = output_dir / "session_summary.json"
    generator.export_summary(str(output_summary))
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total trials generated: {len(results)}")
    logger.info(f"Total participants: {len(set(r.participant_id for r in results))}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - Trial data: {output_csv}")
    logger.info(f"  - Summary: {output_summary}")
    
    # Show strategy distribution in results
    df = generator.results_to_dataframe(results)
    logger.info(f"\nStrategy distribution in results:")
    for strategy, count in df["Strategy"].value_counts().items():
        percentage = count / len(df) * 100
        logger.info(f"  - {strategy}: {count} trials ({percentage:.1f}%)")
    
    logger.info(f"\nXAI condition distribution:")
    for xai_cond, count in df["Tested w/ XAI"].value_counts().items():
        percentage = count / len(df) * 100
        logger.info(f"  - {xai_cond}: {count} trials ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
