# Dual-Mode Data Input for Trial Generation

## Overview

The trial generation system now supports **two data input modes**:

### Mode 1: Load via `ai_dataset_loader` (Default)
- **Default behavior** (backward compatible)
- Loads instances from `user_loader.get_counterfactual_trials()`
- Uses `ai_dataset_loader` to extract features and predictions
- Requires: Working `user_loader` and `ai_dataset_loader` implementations

### Mode 2: Pre-Computed Data Instances (Optional)
- **New optional feature**
- Pass pre-computed feature arrays/dicts directly
- Skips `ai_dataset_loader` entirely
- Useful for: Testing, synthetic data, external data sources

---

## Usage Examples

### Mode 1: Standard Behavior (Load via ai_dataset_loader)

```python
from generate_trials_full import generate_trials_from_params_csv

# No data_instances provided → uses MODE 1
result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=user_loader,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_data,
    dt_df=dt_data,
    metadata_df=metadata,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='experiment',
    output_csv='trials_mode1.csv'
    # data_instances=None (default)
    # data_instances_dict=None (default)
)
```

---

### Mode 2a: Single Trial with Data Instances

```python
import numpy as np

# Create sample data instances (3 features, 5 instances)
data = [
    np.array([0.1, 0.5, 0.9, 0.2, 0.3, 0.4]),  # Instance 1
    np.array([0.3, 0.2, 0.7, 0.5, 0.1, 0.6]),  # Instance 2
    np.array([0.5, 0.8, 0.3, 0.9, 0.4, 0.2]),  # Instance 3
    np.array([0.2, 0.4, 0.6, 0.1, 0.5, 0.3]),  # Instance 4
    np.array([0.7, 0.1, 0.9, 0.3, 0.2, 0.8]),  # Instance 5
]

result = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,  # Can be None for MODE 2
    ai_dataset_loader=None,  # Can be None for MODE 2
    lr_df=None,
    dt_df=None,
    metadata_df=None,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='trial',
    data_instances=data  # ← MODE 2: Use pre-computed data
)

print(result)
# {'model_strategy': 'change_path_dt', 'data_instance': array([0.1, 0.5, ...]), ...}
```

---

### Mode 2b: One Participant with Data Instances

```python
# Create sample data for one participant (40 trials, 6 features each)
participant_data = [np.random.rand(6) for _ in range(40)]

result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,  # Can be None for MODE 2
    ai_dataset_loader=None,  # Can be None for MODE 2
    lr_df=None,
    dt_df=None,
    metadata_df=None,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='participant',
    output_csv='trials_participant_mode2.csv',
    data_instances=participant_data,  # ← MODE 2: 40 pre-computed instances
    n_trials_per_participant=40
)

print(result_df.shape)  # (40, n_columns)
print(result_df['data_instance'].head())
# Shows first 5 data instances
```

---

### Mode 2c: Full Experiment with Data Instances (Multiple Participants)

```python
import pandas as pd

# Load the parameter CSV to get participant IDs
param_df = pd.read_csv('assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv')
participant_ids = param_df['Participant Id'].unique()

# Create synthetic data for each participant
# data_instances_dict: {participant_id → list of 40 data instances}
data_instances_dict = {}
for pid in participant_ids[:10]:  # For first 10 participants
    # 40 trials, 6 features each
    data_instances_dict[pid] = [np.random.rand(6) for _ in range(40)]

result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,  # Can be None for MODE 2
    ai_dataset_loader=None,  # Can be None for MODE 2
    lr_df=None,
    dt_df=None,
    metadata_df=None,
    strategies={0: 'change_path_dt', 1: 'zero_out_lr_heuristic', ...},
    XAI_types={0: 'DT', 1: 'LR', 2: 'DT+LR'},
    training_cog_params=original_train_params,
    param_csv_path='assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='experiment',
    output_csv='trials_experiment_mode2.csv',
    n_participants=10,
    n_trials_per_participant=40,
    data_instances_dict=data_instances_dict  # ← MODE 2: Dict of data per participant
)

print(result_df.shape)  # (400, n_columns) - 10 participants × 40 trials
print(result_df.groupby('participant_id').size())
# All should be 40
```

---

### Mode 2d: Load Data from External Source

```python
import csv
import numpy as np

# Load data instances from CSV file
def load_data_from_csv(csv_path, feature_cols):
    """Load features from CSV into numpy arrays."""
    data_instances = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = np.array([float(row[col]) for col in feature_cols])
            data_instances.append(features)
    return data_instances

# Columns that contain feature values
feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6']

# Load instances for one participant from external CSV
external_data = load_data_from_csv('path/to/external_data.csv', feature_columns)

result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,
    ai_dataset_loader=None,
    lr_df=None,
    dt_df=None,
    metadata_df=None,
    strategies={...},
    XAI_types={...},
    training_cog_params={...},
    param_csv_path='...',
    mode='participant',
    data_instances=external_data,  # ← Use loaded data
    output_csv='trials_external_data.csv'
)
```

---

## Data Format

### Single Data Instance
- **Type**: numpy array or list
- **Shape**: 1D array with N features (typically 6 for wine_quality/mushrooms datasets)
- **Example**: `[0.1, 0.5, 0.9, 0.2, 0.3, 0.4]`

### List of Data Instances (for one participant)
- **Type**: list of arrays/lists
- **Example**: 
  ```python
  [
      [0.1, 0.5, 0.9, 0.2, 0.3, 0.4],
      [0.3, 0.2, 0.7, 0.5, 0.1, 0.6],
      [0.5, 0.8, 0.3, 0.9, 0.4, 0.2],
      ...
  ]
  ```

### Dictionary of Data Instances (for multiple participants)
- **Type**: dict mapping `str(participant_id)` → list of arrays
- **Example**:
  ```python
  {
      '6722f0dfa5df03f57bfa6c41': [
          [0.1, 0.5, 0.9, 0.2, 0.3, 0.4],
          [0.3, 0.2, 0.7, 0.5, 0.1, 0.6],
          ...  # 40 instances
      ],
      '662444ee097e69a6198e2b61': [
          [0.2, 0.4, 0.6, 0.1, 0.5, 0.3],
          [0.7, 0.1, 0.9, 0.3, 0.2, 0.8],
          ...  # 40 instances
      ]
  }
  ```

---

## Mode Selection Summary

| Question | Mode 1 (Default) | Mode 2 (Optional) |
|----------|------------------|-------------------|
| **Have ai_dataset_loader?** | Yes (required) | No (can be None) |
| **Have user_loader?** | Yes (required) | No (can be None) |
| **Provide data_instances?** | No (Pass None) | Yes (Pass array/dict) |
| **Use Case** | Production with real data | Testing / Synthetic / External data |
| **Speed** | Slower (loads from disk) | Faster (data already in memory) |
| **Flexibility** | Fixed to what loaders provide | Can customize feature values |

---

## Key Advantages

### Mode 1 Benefits
✅ Uses actual data from loaders  
✅ Maintains real experiment workflow  
✅ All loaders tightly integrated  

### Mode 2 Benefits
✅ No dependency on working loaders  
✅ Fast iteration for testing  
✅ Support for synthetic/external data  
✅ Easy to control feature distributions  
✅ Reproducible with fixed data  
✅ Parallel experiments with different data  

---

## Implementation Details

### In `generate_participant_session()`:
```python
if data_instances is not None:
    # MODE 2: Use pre-computed data
    print(f"  MODE 2: Using {len(data_instances)} pre-computed data instances")
    # Skip ai_dataset_loader, use data directly
else:
    # MODE 1: Load via ai_dataset_loader
    print(f"  MODE 1: Loading instances via ai_dataset_loader")
    # Use user_loader.get_counterfactual_trials(), etc.
```

### In Output CSV:
Both modes populate a `data_instance` column:
- MODE 1: `None` or actual feature array
- MODE 2: The pre-computed feature array

---

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code continues to work
- Simply don't pass `data_instances` or `data_instances_dict`
- Defaults to MODE 1 (original behavior)

---

## Troubleshooting

**Error: "KeyError" in MODE 2**
- Ensure data_instances_dict has correct participant IDs
- IDs must match those in CSV Participant Id column

**Error: "Index out of range"**
- Ensure data instances have correct number of features (usually 6)
- Check shape: `len(data_instances[0])`

**Data not appearing in output**
- Check that `data_instance` column shows arrays (not None)
- For MODE 1, `data_instance` might be None by design
- For MODE 2, should contain the feature array

**Performance issues with MODE 2**
- Check data isn't being copied excessively
- Verify numpy arrays vs lists (arrays faster)
- Consider chunking large datasets
