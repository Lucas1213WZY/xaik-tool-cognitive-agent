"""
================================================================================
QUICK REFERENCE: New Reasoning Strategies API with CSV Parameters
================================================================================

This file provides quick copy-paste code examples for common tasks.
"""

# =============================================================================
# EXAMPLE 1: Quick Start - Load CSV and Run One Simulation
# =============================================================================

def example_quick_start():
    """Minimal code to load CSV and run a simulation."""
    from run_simulation_from_params_v2 import run_simulation_with_csv_params_v2
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    dataset_config = {
        'values_csv': str(current_dir / 'data/datasets/standard set/values.csv'),
        'metadata_csv': str(current_dir / 'data/datasets/standard set/metadata.csv'),
        'explanation_csv': str(current_dir / 'data/datasets/standard set/importance.csv'),
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    }
    
    trial_sequence = [
        {"instance_id": 0, "is_training": True, "with_explanation": True},
        {"instance_id": 1, "is_training": False, "with_explanation": False},
    ]
    
    strategy, runner, logs = run_simulation_with_csv_params_v2(
        csv_path=str(csv_path),
        dataset_config=dataset_config,
        strategy_filter="Sensitive-features categorization",
        xai_type_filter="importance",
        tested_with_xai_filter="w/ XAI",
        dataset_filter="adult",
        trial_sequence=trial_sequence,
        seed=42
    )
    
    print(f"✓ Simulation complete with {len(logs)} logs")


# =============================================================================
# EXAMPLE 2: Manual Step-by-Step (More Control)
# =============================================================================

def example_manual_steps():
    """Detailed step-by-step execution."""
    import pandas as pd
    from pathlib import Path
    from run_simulation_from_params_v2 import (
        CSVParameterLoaderV2, 
        instantiate_strategy_new_api,
        SimpleTimeManager,
        SimulationRunnerV2
    )
    from data_loader import AIDatasetLoader
    
    current_dir = Path(__file__).parent
    
    # 1. Load and filter CSV
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    loader = CSVParameterLoaderV2(str(csv_path))
    
    filtered_df = loader.filter_parameters(
        strategy="Salient-features categorization",
        xai_type="importance",
        tested_with_xai="w/ XAI",
        dataset="adult"
    )
    
    # 2. Select random participant
    param_row = loader.select_random_params(filtered_df, seed=42)
    print(f"Selected participant: {param_row['Participant Id']}")
    print(f"Parameters: k={param_row['k']}, sensitivity={param_row['sensitivity']}")
    
    # 3. Create strategy config
    strategy_name = param_row['Strategy']
    config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
    
    # 4. Instantiate strategy
    time_manager = SimpleTimeManager()
    strategy = instantiate_strategy_new_api(strategy_name, config, time_manager)
    
    # 5. Load data
    data_dir = current_dir / 'data/datasets/standard set'
    df_values = pd.read_csv(data_dir / 'values.csv')
    df_meta = pd.read_csv(data_dir / 'metadata.csv')
    df_expl = pd.read_csv(data_dir / 'importance.csv')
    
    ai_loader = AIDatasetLoader(
        feature_values_df=df_values,
        metadata_df=df_meta,
        explanation_values_df=df_expl,
        explanation_columns=['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    )
    
    # 6. Create runner and run trials
    runner = SimulationRunnerV2(strategy, ai_loader, time_manager)
    
    trial_sequence = [
        {"instance_id": i, "is_training": i < 2, "with_explanation": True}
        for i in range(5)
    ]
    
    logs = runner.run_trial_sequence(trial_sequence)
    print(f"✓ Completed {len(logs)} trials")
    
    return strategy, runner, logs


# =============================================================================
# EXAMPLE 3: Filter by Specific Participant ID
# =============================================================================

def example_specific_participant():
    """Load and run a specific participant's parameters."""
    import pandas as pd
    from pathlib import Path
    from run_simulation_from_params_v2 import CSVParameterLoaderV2
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    df = pd.read_csv(csv_path)
    
    # Get a specific participant
    specific_pid = "67d72e56bfef3906846145ae"
    
    # Find all rows for this participant
    participant_rows = df[df['Participant Id'] == specific_pid]
    
    print(f"Found {len(participant_rows)} strategies for participant {specific_pid}")
    
    # Look at a specific strategy
    row = participant_rows[
        (participant_rows['Strategy'] == 'Sensitive-features categorization') &
        (participant_rows['XAIType'] == 'importance') &
        (participant_rows['Tested w/ XAI'] == 'w/ XAI')
    ].iloc[0]
    
    print(f"  Strategy: {row['Strategy']}")
    print(f"  k: {row['k']}")
    print(f"  sensitivity: {row['sensitivity']}")
    print(f"  NLL: {row['NLL']}")


# =============================================================================
# EXAMPLE 4: Compare Different Conditions
# =============================================================================

def example_compare_conditions():
    """Compare with/without XAI conditions."""
    from run_simulation_from_params_v2 import CSVParameterLoaderV2
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    loader = CSVParameterLoaderV2(str(csv_path))
    
    results = {}
    
    for condition in ["w/ XAI", "w/o XAI"]:
        print(f"\n=== {condition} ===")
        
        filtered_df = loader.filter_parameters(
            strategy="Sensitive-features categorization",
            xai_type="importance",
            tested_with_xai=condition,
            dataset="adult"
        )
        
        # Statistics
        stats = {
            'count': len(filtered_df),
            'mean_k': filtered_df['k'].mean(),
            'mean_sensitivity': filtered_df['sensitivity'].mean(),
            'mean_threshold': filtered_df['retrieval_threshold'].mean(),
            'mean_nll': filtered_df['NLL'].mean(),
            'best_nll': filtered_df['NLL'].min(),
        }
        
        for key, val in stats.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        
        results[condition] = stats
    
    return results


# =============================================================================
# EXAMPLE 5: Batch Run Multiple Simulations
# =============================================================================

def example_batch_simulations():
    """Run multiple simulations with different random participants."""
    from run_simulation_from_params_v2 import run_simulation_with_csv_params_v2
    from pathlib import Path
    import pandas as pd
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    dataset_config = {
        'values_csv': str(current_dir / 'data/datasets/standard set/values.csv'),
        'metadata_csv': str(current_dir / 'data/datasets/standard set/metadata.csv'),
        'explanation_csv': str(current_dir / 'data/datasets/standard set/importance.csv'),
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    }
    
    trial_sequence = [
        {"instance_id": i, "is_training": i < 3, "with_explanation": True}
        for i in range(6)
    ]
    
    batch_results = []
    
    for run_num in range(3):
        print(f"\n--- Batch Run {run_num + 1}/3 ---")
        
        strategy, runner, logs = run_simulation_with_csv_params_v2(
            csv_path=str(csv_path),
            dataset_config=dataset_config,
            strategy_filter="Salient-features categorization",
            xai_type_filter="importance",
            tested_with_xai_filter="w/ XAI",
            dataset_filter="adult",
            trial_sequence=trial_sequence,
            seed=42 + run_num  # Different seed each time
        )
        
        # Collect statistics
        infer_logs = [log for log in logs if log.get('step') == 'infer']
        
        batch_results.append({
            'run': run_num + 1,
            'num_inferences': len(infer_logs),
            'total_logs': len(logs),
        })
    
    results_df = pd.DataFrame(batch_results)
    print(f"\n✓ Batch results:\n{results_df}")
    
    return batch_results


# =============================================================================
# EXAMPLE 6: Use with Custom Trial Sequence
# =============================================================================

def example_custom_trial_sequence():
    """Create and use a custom trial sequence."""
    from run_simulation_from_params_v2 import run_simulation_with_csv_params_v2
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    # Define a custom trial sequence
    trial_sequence = [
        # Training phase: with explanation
        {"instance_id": 0, "is_training": True, "with_explanation": True},
        {"instance_id": 1, "is_training": True, "with_explanation": True},
        {"instance_id": 2, "is_training": True, "with_explanation": True},
        
        # Testing phase 1: without explanation
        {"instance_id": 3, "is_training": False, "with_explanation": False},
        {"instance_id": 4, "is_training": False, "with_explanation": False},
        
        # Testing phase 2: with explanation
        {"instance_id": 5, "is_training": False, "with_explanation": True},
        {"instance_id": 6, "is_training": False, "with_explanation": True},
    ]
    
    dataset_config = {
        'values_csv': str(current_dir / 'data/datasets/standard set/values.csv'),
        'metadata_csv': str(current_dir / 'data/datasets/standard set/metadata.csv'),
        'explanation_csv': str(current_dir / 'data/datasets/standard set/importance.csv'),
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    }
    
    strategy, runner, logs = run_simulation_with_csv_params_v2(
        csv_path=str(csv_path),
        dataset_config=dataset_config,
        strategy_filter="Importance categorization",
        xai_type_filter="importance",
        tested_with_xai_filter="w/ XAI",
        dataset_filter="adult",
        trial_sequence=trial_sequence,
        seed=123
    )
    
    # Analyze by phase
    training_logs = [log for log in logs if log.get('is_training')]
    test_no_xai = [log for log in logs if not log.get('is_training') and not log.get('with_explanation')]
    test_with_xai = [log for log in logs if not log.get('is_training') and log.get('with_explanation')]
    
    print(f"\nPhase breakdown:")
    print(f"  Training: {len(training_logs)} logs")
    print(f"  Test (no XAI): {len(test_no_xai)} logs")
    print(f"  Test (with XAI): {len(test_with_xai)} logs")
    
    return strategy, runner, logs


# =============================================================================
# EXAMPLE 7: Inspect Strategy State and Metadata
# =============================================================================

def example_strategy_inspection():
    """Inspect strategy properties and state."""
    from run_simulation_from_params_v2 import (
        CSVParameterLoaderV2,
        instantiate_strategy_new_api,
        SimpleTimeManager
    )
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    loader = CSVParameterLoaderV2(str(csv_path))
    
    filtered_df = loader.filter_parameters(
        strategy="Sensitive-features categorization",
        xai_type="importance",
        tested_with_xai="w/ XAI",
        dataset="adult"
    )
    
    param_row = loader.select_random_params(filtered_df, seed=42)
    strategy_name = param_row['Strategy']
    
    config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
    time_manager = SimpleTimeManager()
    strategy = instantiate_strategy_new_api(strategy_name, config, time_manager)
    
    # Inspect metadata
    metadata = strategy.metadata
    print(f"\nMetadata:")
    print(f"  Name: {metadata.display_name}")
    print(f"  Category: {metadata.category}")
    print(f"  Description: {metadata.description}")
    print(f"  Supported Modes: {metadata.supported_modes}")
    print(f"  Parameters: {metadata.parameters}")
    
    # Inspect config
    print(f"\nConfig:")
    print(f"  decay_param: {config.decay_param}")
    print(f"  retrieval_threshold: {config.retrieval_threshold}")
    print(f"  sensitivity: {config.sensitivity}")
    print(f"  extra_params: {config.extra_params}")
    
    # Inspect state (initially empty)
    state = strategy.get_state()
    print(f"\nState (initial):")
    print(f"  memory_size: {state.get('memory_size', 'N/A')}")
    print(f"  exemplars_count: {state.get('exemplars_count', 0)}")


# =============================================================================
# EXAMPLE 8: Export Results to CSV
# =============================================================================

def example_export_results():
    """Run simulation and export results to CSV."""
    import pandas as pd
    from pathlib import Path
    from run_simulation_from_params_v2 import run_simulation_with_csv_params_v2
    
    current_dir = Path(__file__).parent
    csv_path = current_dir / "02-01-2026-fitted-data-params" / "three datasets strategies.csv"
    
    dataset_config = {
        'values_csv': str(current_dir / 'data/datasets/standard set/values.csv'),
        'metadata_csv': str(current_dir / 'data/datasets/standard set/metadata.csv'),
        'explanation_csv': str(current_dir / 'data/datasets/standard set/importance.csv'),
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    }
    
    trial_sequence = [
        {"instance_id": i, "is_training": i < 3, "with_explanation": True}
        for i in range(6)
    ]
    
    strategy, runner, logs = run_simulation_with_csv_params_v2(
        csv_path=str(csv_path),
        dataset_config=dataset_config,
        strategy_filter="Sensitive-features categorization",
        xai_type_filter="importance",
        tested_with_xai_filter="w/ XAI",
        dataset_filter="adult",
        trial_sequence=trial_sequence,
        seed=42
    )
    
    # Convert to DataFrame
    results_df = pd.DataFrame(logs)
    
    # Save to CSV
    output_path = current_dir / "results" / "simulation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results exported to {output_path}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {results_df.columns.tolist()}")
    
    return results_df


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Quick Start', example_quick_start),
        '2': ('Manual Steps', example_manual_steps),
        '3': ('Specific Participant', example_specific_participant),
        '4': ('Compare Conditions', example_compare_conditions),
        '5': ('Batch Simulations', example_batch_simulations),
        '6': ('Custom Trial Sequence', example_custom_trial_sequence),
        '7': ('Strategy Inspection', example_strategy_inspection),
        '8': ('Export Results', example_export_results),
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\n{'=' * 80}")
            print(f"Running: {name}")
            print('=' * 80)
            func()
        else:
            print(f"Unknown example: {example_num}")
    else:
        print("\nUsage: python example_csv_parameter_usage_v2.py <example_num>")
        print("Or import individual functions for your own scripts")
