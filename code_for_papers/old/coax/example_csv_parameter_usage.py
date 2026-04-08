"""
================================================================================
EXAMPLES: Running Simulations with CSV Parameters
================================================================================

This file shows different ways to use run_simulation_from_params.py
for different experimental scenarios.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import random

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from run_simulation_from_params import (
    CSVParameterLoader,
    run_simulation_with_csv_params,
    instantiate_strategy
)


# =============================================================================
# EXAMPLE 1: Simple scenario - one participant, one strategy
# =============================================================================

def example_1_simple_single_participant():
    """
    Run a simple experiment with a randomly selected participant
    from a specific strategy and condition.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Single Participant")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    # Load and explore parameters
    loader = CSVParameterLoader(str(csv_path))
    
    # Filter: only "Sensitive-features categorization" tested WITH XAI on importance
    filtered_df = loader.filter_parameters(
        strategy="Sensitive-features categorization",
        xai_type="importance",
        tested_with_xai="w/ XAI",
        dataset="adult"
    )
    
    print(f"\nFound {len(filtered_df)} participants matching criteria")
    
    # See what participants are available
    print("\nAvailable participant IDs:")
    for pid in filtered_df['Participant Id'].unique()[:5]:  # Show first 5
        print(f"  - {pid}")
    
    # Randomly select one
    selected = loader.select_random_params(filtered_df, seed=42)
    print(f"\nSelected participant: {selected['Participant Id']}")
    print(f"Parameters: k={selected['k']}, sensitivity={selected['sensitivity']}, "
          f"retrieval_threshold={selected['retrieval_threshold']}")


# =============================================================================
# EXAMPLE 2: Compare different conditions for same strategy
# =============================================================================

def example_2_compare_with_without_xai():
    """
    Load parameters for the same strategy, but compare
    "with XAI" vs "without XAI" conditions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare With/Without XAI")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    loader = CSVParameterLoader(str(csv_path))
    
    for condition in ["w/ XAI", "w/o XAI"]:
        print(f"\n--- Condition: {condition} ---")
        
        filtered_df = loader.filter_parameters(
            strategy="Salient-features categorization",
            xai_type="importance",
            tested_with_xai=condition,
            dataset="adult"
        )
        
        # Show statistics on parameters
        print(f"Number of participants: {len(filtered_df)}")
        print(f"Mean k: {filtered_df['k'].mean():.2f}")
        print(f"Mean sensitivity: {filtered_df['sensitivity'].mean():.2f}")
        print(f"Mean retrieval_threshold: {filtered_df['retrieval_threshold'].mean():.2f}")


# =============================================================================
# EXAMPLE 3: Test different strategies
# =============================================================================

def example_3_compare_strategies():
    """
    Compare parameters across different strategies.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Different Strategies")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    loader = CSVParameterLoader(str(csv_path))
    
    strategies = [
        "Sensitive-features categorization",
        "Salient-features categorization",
        "Importance categorization",
        "Attribution Sum"
    ]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        filtered_df = loader.filter_parameters(
            strategy=strategy,
            xai_type="importance",
            tested_with_xai="w/ XAI",
            dataset="adult"
        )
        
        if len(filtered_df) > 0:
            print(f"Number of participants: {len(filtered_df)}")
            print(f"NLL Range: {filtered_df['NLL'].min():.4f} to {filtered_df['NLL'].max():.4f}")
            print(f"Best NLL: {filtered_df['NLL'].min():.4f}")
        else:
            print("No participants found")


# =============================================================================
# EXAMPLE 4: Batch simulation - run multiple random participants
# =============================================================================

def example_4_batch_simulations():
    """
    Run multiple simulations with different randomly selected participants.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Simulations - Multiple Participants")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    loader = CSVParameterLoader(str(csv_path))
    
    # Filter to get a pool of participants
    filtered_df = loader.filter_parameters(
        strategy="Sensitive-features categorization",
        xai_type="importance",
        tested_with_xai="w/ XAI",
        dataset="adult"
    )
    
    # Run simulations with different participants
    results = []
    for i in range(3):  # Run 3 times
        print(f"\n--- Simulation {i+1} ---")
        
        selected = loader.select_random_params(filtered_df, seed=42+i)
        print(f"Participant: {selected['Participant Id']}")
        print(f"LLH: {selected['NLL']}")
        
        results.append({
            'participant_id': selected['Participant Id'],
            'k': selected['k'],
            'sensitivity': selected['sensitivity'],
            'nll': selected['NLL']
        })
    
    print("\n--- Summary ---")
    for res in results:
        print(f"Participant {res['participant_id']}: k={res['k']}, sensitivity={res['sensitivity']:.2f}, NLL={res['nll']:.4f}")


# =============================================================================
# EXAMPLE 5: View all unique values in CSV
# =============================================================================

def example_5_explore_csv_structure():
    """
    Explore the structure and unique values in the CSV.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Explore CSV Structure")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    df = pd.read_csv(csv_path)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n--- Unique Values ---")
    print(f"Strategies: {df['Strategy'].unique().tolist()}")
    print(f"XAI Types: {df['XAIType'].unique().tolist()}")
    print(f"Test Conditions: {df['Tested w/ XAI'].unique().tolist()}")
    print(f"Datasets: {df['appId'].unique().tolist()}")
    
    print("\n--- Parameter Ranges ---")
    print(f"k: {df['k'].min():.0f} to {df['k'].max():.0f}")
    print(f"sensitivity: {df['sensitivity'].min():.2f} to {df['sensitivity'].max():.2f}")
    print(f"retrieval_threshold: {df['retrieval_threshold'].min():.2f} to {df['retrieval_threshold'].max():.2f}")
    print(f"decay_param: {df['decay_param'].min():.2f} to {df['decay_param'].max():.2f}")
    if 'scaling_factor' in df.columns:
        print(f"scaling_factor: {df['scaling_factor'].min():.2f} to {df['scaling_factor'].max():.2f}")


# =============================================================================
# EXAMPLE 6: Select by participant ID
# =============================================================================

def example_6_specific_participant():
    """
    Load parameters for a specific participant only.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Load Specific Participant")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    df = pd.read_csv(csv_path)
    
    # Pick a specific participant ID
    specific_pid = df['Participant Id'].iloc[0]
    print(f"\nLooking for participant: {specific_pid}")
    
    # Filter for this participant
    participant_rows = df[df['Participant Id'] == specific_pid]
    print(f"Found {len(participant_rows)} rows for this participant")
    
    print("\nStrategies tested by this participant:")
    for idx, row in participant_rows.iterrows():
        print(f"  - {row['Strategy']} (XAI: {row['XAIType']}, Tested: {row['Tested w/ XAI']}, NLL: {row['NLL']:.4f})")


# =============================================================================
# EXAMPLE 7: Find best parameters by metric
# =============================================================================

def example_7_best_parameters():
    """
    Find the best performing parameters (lowest NLL).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Best Parameters by NLL")
    print("=" * 80)
    
    csv_path = parent_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    df = pd.read_csv(csv_path)
    
    # Filter to a specific condition
    filtered = df[
        (df['Strategy'] == "Salient-features categorization") &
        (df['XAIType'] == "importance") &
        (df['Tested w/ XAI'] == "w/ XAI") &
        (df['appId'] == "adult")
    ]
    
    # Find rows with best (lowest) NLL
    best_nll = filtered['NLL'].min()
    best_rows = filtered[filtered['NLL'] == best_nll]
    
    print(f"\nBest NLL: {best_nll:.6f}")
    print(f"Found {len(best_rows)} participant(s) with this NLL")
    
    for idx, row in best_rows.iterrows():
        print(f"\nParticipant: {row['Participant Id']}")
        print(f"  k: {row['k']}")
        print(f"  sensitivity: {row['sensitivity']}")
        print(f"  retrieval_threshold: {row['retrieval_threshold']}")


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLES")
    print("=" * 80)
    
    example_1_simple_single_participant()
    example_5_explore_csv_structure()
    example_2_compare_with_without_xai()
    example_3_compare_strategies()
    example_4_batch_simulations()
    example_6_specific_participant()
    example_7_best_parameters()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
