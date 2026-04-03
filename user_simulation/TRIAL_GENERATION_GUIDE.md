# Trial Generation from Best CoXAM Parameters

## Overview

Generate trial-by-trial counterfactual simulation data using the best parameters from your CoXAM optimization CSV, without the scoring/fitting loop. Three execution modes are supported:

1. **`trial`** — Generate a single trial
2. **`participant`** — Generate all trials for one participant session  
3. **`experiment`** — Generate trials for entire experiment (multiple participants)

## Key Features

- **Parameter Loading**: Automatically loads best params from CSV (`CoXAM_counterfactual_simulation_cog_param.csv`)
- **Random Sampling**: Sample parameter rows randomly from the CSV
- **RL Agent Integration**: Uses trained RL agent to select strategies trial-by-trial
- **Three Execution Modes**: Flexible granularity (single trial → full experiment)
- **CSV Output**: Save results to CSV files for analysis

## Architecture

### Files Created

| File | Purpose |
|------|---------|
| `generate_trials_from_params.py` | Skeleton implementation with structure |
| `generate_trials_full.py` | Full implementation with all trial logic |

### Function Hierarchy

```
generate_trials_from_params_csv()  [Main entry point]
├── load_best_params_csv()
├── sample_param_row()
└── Mode-specific generators:
    ├── generate_single_trial()         [mode='trial']
    ├── generate_participant_session()  [mode='participant']
    └── generate_full_experiment()      [mode='experiment']
```

## Usage Examples

### Example 1: Generate Single Trial
```python
from generate_trials_full import generate_trials_from_params_csv

result = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=user_loader,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_data,
    dt_df=dt_data,
    metadata_df=metadata,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='trial',
    random_seed=42
)

# result is a dict with one trial's predictions
print(result)
# {'participant_id': '6722f0dfa5df03f57bfa6c41', 'model_strategy': 'change_path_dt', ...}
```

### Example 2: Generate Participant Session (All Trials)
```python
participant_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=user_loader,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_data,
    dt_df=dt_data,
    metadata_df=metadata,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='participant',
    output_csv='outputs/participant_trials.csv',
    n_trials_per_participant=40,  # Optional limit
    random_seed=42
)

# participant_df is a DataFrame with 40 rows (one per trial)
print(participant_df.shape)  # (40, n_columns)
participant_df.head()
```

### Example 3: Generate Full Experiment
```python
experiment_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=user_loader,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_data,
    dt_df=dt_data,
    metadata_df=metadata,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='user_simulation/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='experiment',
    output_csv='outputs/experiment_all_trials.csv',
    n_participants=10,                  # Sample 10 participants from CSV
    n_trials_per_participant=30,        # 30 trials each
    random_seed=42
)

# experiment_df is a DataFrame with 10 × 30 = 300 rows
print(experiment_df.shape)  # (300, n_columns)
experiment_df.groupby('participant_id').size()  # All should be 30
```

## Parameter CSV Format

Your CSV must have these columns:
```
Participant Id,Best NLL,Best MAE,Best time,Best retrieval_threshold,
Best over_margin,Best chi,app_id,model,complexity,condition
```

**Key columns used:**
- `Participant Id` — Maps to participant in user_loader
- `Best retrieval_threshold` — ACT-R memory retrieval parameter
- `Best over_margin` — Margin for feature changes
- `Best chi` — Model temperature/uncertainty parameter
- `app_id` — Dataset (wine_quality, mushrooms, etc.)
- `complexity` — Task difficulty (low/high)
- `condition` — XAI condition (DT, LR, DT+LR)

## Output CSV Columns

Each generated row contains:
- **Participant Info**: participant_id, app_id, condition, complexity
- **Trial Data**: instance_id, ai_prediction, with_xai, xai_shown
- **Participant Choice**: participant_choice_feature, participant_choice_delta
- **Model Strategy**: model_strategy_id, model_strategy, model_depth
- **Hyperparameters Used**: retrieval_threshold, over_margin, chi, lapse

## Integration Notes

### What's Already Implemented
- ✅ CSV loading and random sampling
- ✅ Observation builder (from original training)
- ✅ RL agent prediction logic
- ✅ Three execution modes with proper formatting
- ✅ Output CSV generation

### What Needs Your Data
The following require integration with your actual loaders:
- **`user_loader.get_participant_info()`** — Get participant metadata
- **`user_loader.get_counterfactual_trials()`** — Load CF trials
- **`user_loader.get_forward_trials()`** — Load forward trials (if needed)
- **`ai_dataset_loader`** — Load AI model predictions and bounds
- **Interpreter models** — lr_exp, dt_exp for feature importance

These are called in `generate_participant_session()` with try/except fallbacks.

### Adaptation Steps

1. **Import your loaders**:
   ```python
   from your_module import UserDataLoader, AIDatasetLoader
   user_loader = UserDataLoader(...)
   ai_dataset_loader = AIDatasetLoader(...)
   ```

2. **Load your RL model**:
   ```python
   from stable_baselines3 import PPO
   model = PPO.load("path/to/best_model.zip")
   ```

3. **Define strategy and XAI mappings**:
   ```python
   strategies = {0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...}
   XAI_types = {0: 'DT', 1: 'LR', 2: 'DT+LR'}
   training_cog_params = {'retrieval_threshold': [-2.0, 0.5], ...}
   ```

4. **Call the main function** with your data

## Comparison: Before vs After

### Before (Optimization)
```python
# Fit each participant's parameters using GPBO
def fit_participant_with_gpbo(...):
    best_params = optimize_for_nll_mae_time(...)  # ← Searching
    return best_params

# Loop over many random initializations and iterations
```

### After (Forward Simulation)
```python
# Use pre-optimized parameters directly
def generate_trials_from_params_csv(..., mode='participant'):
    param_row = load_from_csv(...)  # ← Pre-optimized, no search
    trials = generate_participant_session(param_row)  # ← Direct generation
    return trials
```

## Performance Considerations

- **Single Trial**: ~1 sec (RL agent inference only)
- **Participant Session** (40 trials): ~30-40 sec
- **Full Experiment** (50 participants × 40 trials): ~30-40 min

To speed up:
1. Set `n_participants` to subset (don't always run all)
2. Set `n_trials_per_participant` to reduce trials per person
3. Run multiple experiments in parallel (different random seeds)

## Troubleshooting

**Error: "Could not load trials for participant"**
- Your `user_loader` might not have trials for that synthetic participant ID
- The code falls back to generating dummy trials with `n_trials=40`

**Error: "KeyError" in observation builder**
- Ensure `XAI_types` dict mapping matches the condition names in your data (DT, LR, DT+LR)

**Empty CSV output**
- Check that CounterfactualSimulation data exists in your user_loader
- Ensure parameter row has valid app_id and condition

## Future Extensions

- Add parallelization across participants (multiprocessing)
- Support custom strategy fallback rules per condition
- Add memory warmup from forward trials
- Generate feature importance rankings alongside predictions
- Export model explanations (which features the model considered)
