# New Reasoning Strategies API Integration - Summary

## Overview

I've created a set of tools and guides to help you use the **new unified reasoning strategies API** (from `src/cognitive_models/`) with parameters loaded from your CSV file (`three datasets strategies.csv`).

## What I Created

### 1. **`run_simulation_from_params_v2.py`** (Main Script)
The updated runner that:
- Loads parameters from your CSV file
- Maps CSV columns to new `StrategyConfig` format
- Instantiates strategies using the new API
- Runs simulations with trial sequences
- Handles data loading and inference/feedback loops

**Key Classes:**
- `CSVParameterLoaderV2`: Loads, filters, and converts CSV parameters
- `SimpleTimeManager`: Minimal time tracking for strategies
- `SimulationRunnerV2`: Runs simulation with new API strategies

**Key Functions:**
- `run_simulation_with_csv_params_v2()`: Main entry point
- `instantiate_strategy_new_api()`: Creates strategy from config

### 2. **`example_csv_parameter_usage_v2.py`** (Examples)
8 practical examples showing:
1. Quick start (minimal code)
2. Manual step-by-step execution
3. Load specific participant
4. Compare with/without XAI
5. Batch run multiple simulations
6. Custom trial sequences
7. Inspect strategy properties
8. Export results to CSV

Run examples:
```bash
python example_csv_parameter_usage_v2.py 1    # Quick start
python example_csv_parameter_usage_v2.py 2    # Manual steps
# etc...
```

### 3. **`GUIDE_NEW_API_WITH_CSV.md`** (Comprehensive Guide)
Complete documentation covering:
- Mapping between CSV columns and strategy parameters
- Step-by-step usage workflow
- Complete workflow example
- Available strategies and their parameters
- `StrategyConfig` dataclass reference
- Debugging & inspection techniques
- Common issues & solutions
- Migration guide from old API

### 4. **`example_csv_parameter_usage.py`** (Old API Examples - Kept for Reference)
Original examples using old `consolidated_human_models.py`

## Quick Start

### Basic Usage (3 lines of core code):

```python
from run_simulation_from_params_v2 import run_simulation_with_csv_params_v2

strategy, runner, logs = run_simulation_with_csv_params_v2(
    csv_path="path/to/three datasets strategies.csv",
    dataset_config={
        'values_csv': 'data/datasets/standard set/values.csv',
        'metadata_csv': 'data/datasets/standard set/metadata.csv',
        'explanation_csv': 'data/datasets/standard set/importance.csv',
        'explanation_columns': ['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
    },
    strategy_filter="Sensitive-features categorization",
    xai_type_filter="importance",
    tested_with_xai_filter="w/ XAI",
    dataset_filter="adult",
    trial_sequence=[
        {"instance_id": 0, "is_training": True, "with_explanation": True},
        {"instance_id": 1, "is_training": False, "with_explanation": False},
    ],
    seed=42
)
```

## How It Works

### Step-by-Step Flow:

1. **Load CSV** 
   ```python
   loader = CSVParameterLoaderV2("three datasets strategies.csv")
   ```

2. **Filter by conditions** (strategy, XAI type, with/without XAI, dataset)
   ```python
   filtered_df = loader.filter_parameters(
       strategy="Sensitive-features categorization",
       xai_type="importance",
       tested_with_xai="w/ XAI",
       dataset="adult"
   )
   ```

3. **Randomly select one participant**
   ```python
   param_row = loader.select_random_params(filtered_df, seed=42)
   ```

4. **Convert to StrategyConfig**
   ```python
   config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
   ```

5. **Instantiate strategy**
   ```python
   strategy = SensitiveFeatures(config)
   ```

6. **Run inference/feedback loop**
   ```python
   for trial in trial_sequence:
       strategy.new_instance()
       probs, time, info = strategy.infer(features, explanation, ai_prediction)
       strategy.feedback(features, true_label, explanation)
   ```

## CSV Column Mapping

Your CSV columns map to strategy parameters like this:

| CSV Column | StrategyConfig Field | Purpose |
|---|---|---|
| `k` | `extra_params['k']` | Number of features to focus on |
| `sensitivity` | `extra_params['sensitivity']` | Feature discrimination (higher = more sensitive) |
| `retrieval_threshold` | `retrieval_threshold` | Memory activation threshold |
| `decay_param` | `decay_param` | Temporal decay rate (0.1-1.0) |
| `scaling_factor` | `extra_params['scaling_factor']` | Attribution sum scaling |
| `explanation_type` | `extra_params['explanation_type']` | 'importance' or 'attribution' |

## Available Strategies

All these strategies are now available via the new API:

**CoAX Forward Reasoning:**
- `SensitiveFeatures` - Focus on discriminative features
- `SalientFeatures` - Focus on high-magnitude explanations
- `ImportanceCategorization` - Use explanations for categorization
- `AttributionSum` - Sum attribution for decisions

**CoXAM Forward Reasoning:**
- `LRCalculation` - Logistic regression with learning
- `LRHeuristic` - Simplified heuristic
- `DTTraversal` - Decision tree reasoning

## Key Differences: Old vs New API

### Old API (consolidated_human_models.py)
```python
strategy = SensitiveFeatures(k=3, sensitivity=10.0, decay_param=0.5)
response, time = strategy.infer_no_explanation(features, ai_pred)
```

### New API (src/cognitive_models/)
```python
config = StrategyConfig(..., extra_params={'k': 3, 'sensitivity': 10.0})
strategy = SensitiveFeatures(config)
probs, time, info = strategy.infer(features=features, ai_prediction=ai_pred)
```

**Benefits of new API:**
- Config-driven (easier to test and compose)
- Returns debug info (activated exemplars, focus features, etc.)
- Unified memory interface
- Registry-based strategy discovery
- Better extensibility

## File Locations

```
code_for_papers/old/coax/
├── run_simulation_from_params_v2.py      ← Main runner (NEW)
├── example_csv_parameter_usage_v2.py     ← 8 examples (NEW)
├── GUIDE_NEW_API_WITH_CSV.md             ← Comprehensive guide (NEW)
├── run_simulation_from_params.py         ← Old runner v1
├── example_csv_parameter_usage.py        ← Old examples v1
├── consolidated_human_models.py          ← Old strategy classes
├── 02-01-2026-fitted-data-params/
│   └── three datasets strategies.csv     ← Your parameter CSV
└── data/datasets/standard set/
    ├── values.csv
    ├── metadata.csv
    ├── importance.csv (or attribution.csv)
    └── ...
```

## Next Steps

1. **Try the quick start example:**
   ```python
   python example_csv_parameter_usage_v2.py 1
   ```

2. **Read the comprehensive guide:**
   ```
   open GUIDE_NEW_API_WITH_CSV.md
   ```

3. **Explore individual examples:**
   ```python
   python example_csv_parameter_usage_v2.py 2    # Manual steps
   python example_csv_parameter_usage_v2.py 4    # Compare conditions
   python example_csv_parameter_usage_v2.py 5    # Batch runs
   ```

4. **Integrate into your workflow:**
   - Copy the parameter loading logic from `run_simulation_from_params_v2.py`
   - Adapt the trial sequence to your needs
   - Use results for analysis

## Debugging Tips

If you encounter issues:

1. **Check imports:**
   ```python
   # Make sure src/ is in path
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```

2. **Verify CSV structure:**
   ```python
   from run_simulation_from_params_v2 import CSVParameterLoaderV2
   loader = CSVParameterLoaderV2("three datasets strategies.csv")
   # Check columns and sample data
   ```

3. **Inspect strategy config:**
   ```python
   config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
   print(f"decay_param: {config.decay_param}")
   print(f"extra_params: {config.extra_params}")
   ```

4. **Get strategy metadata:**
   ```python
   metadata = strategy.metadata
   print(f"Name: {metadata.display_name}")
   print(f"Parameters: {metadata.parameters}")
   ```

## For More Information

See these documentation files in `src/`:
- `src/cognitive_models/interface.py` - Abstract base class
- `src/cognitive_models/registry.py` - Strategy registry
- `src/cognitive_models/forward/coax_forward_rs.py` - Strategy implementations
- `src/cognitive_models/memory/__init__.py` - Memory module

## Example Output

Running a simulation produces logs like:
```
Trial 1/4: Instance 0
  ↻ Feedback processed (time: 0.045s)
  Inference: pred=1 (p=0.7234), ai_truth=1

Trial 2/4: Instance 1
  Inference: pred=0 (p=0.6123), ai_truth=0

Total inferences: 3
```

Results are saved as CSV with columns:
- `trial`, `instance_id`, `step`, `is_training`, `with_explanation`
- `probabilities`, `time_cost`, `ai_prediction`, `info`

---

**Need help?** Check the examples in `example_csv_parameter_usage_v2.py` or read `GUIDE_NEW_API_WITH_CSV.md` for detailed explanations!
