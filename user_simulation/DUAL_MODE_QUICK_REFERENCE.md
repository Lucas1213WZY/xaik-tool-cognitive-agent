# Data Input Modes: Quick Reference

## Main Function Signature

```python
generate_trials_from_params_csv(
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
    mode='participant',              # 'trial' | 'participant' | 'experiment'
    output_csv=None,
    lapse=0.1,
    random_seed=None,
    data_instances=None,             # NEW: Pre-computed instances (single participant)
    data_instances_dict=None,        # NEW: Pre-computed instances dict (multiple participants)
    **kwargs  # n_participants, n_trials_per_participant
)
```

---

## Mode Selection Matrix

### **MODE 1: Load via ai_dataset_loader** (Default)
```python
# Parameters to pass:
param = {
    'user_loader': your_loader,           # ✅ Required
    'ai_dataset_loader': your_loader,     # ✅ Required
    'data_instances': None,               # ✅ Set to None (default)
    'data_instances_dict': None,          # ✅ Set to None (default)
}

# Use when:
# - You have working user_loader and ai_dataset_loader
# - You want to use actual data from loaders
# - You need full experiment workflow
```

### **MODE 2: Use Pre-Computed Data Instances** (New)
```python
# For single trial or participant:
param = {
    'user_loader': None,                  # ✅ Can be None
    'ai_dataset_loader': None,            # ✅ Can be None
    'data_instances': your_data_array,    # ✅ Provide array/list
    'data_instances_dict': None,          # Set to None (for trial/participant mode)
}

# For full experiment:
param = {
    'user_loader': None,                  # ✅ Can be None
    'ai_dataset_loader': None,            # ✅ Can be None
    'data_instances': None,               # Set to None (for experiment mode)
    'data_instances_dict': your_dict,     # ✅ Provide dict {pid → list}
}

# Use when:
# - Loaders not available or not working
# - You have pre-computed feature values
# - Testing with synthetic/external data
# - Reproducible experiments with fixed data
```

---

## Execution Modes (Independent of Data Modes)

### **Execution: Single Trial**
```python
result = generate_trials_from_params_csv(
    ...,
    mode='trial',              # Generate 1 trial
    data_instances=data_array  # Optional: [feature_array]
)
# Returns: dict with one trial's predictions
```

### **Execution: One Participant**
```python
result_df = generate_trials_from_params_csv(
    ...,
    mode='participant',                 # Generate all trials for 1 participant
    n_trials_per_participant=40,        # Optional limit
    data_instances=data_array,          # Optional: [40 feature arrays]
    output_csv='out.csv'
)
# Returns: DataFrame(40 rows)
```

### **Execution: Full Experiment**
```python
result_df = generate_trials_from_params_csv(
    ...,
    mode='experiment',                   # Generate all trials for all participants
    n_participants=10,                   # Sample 10 participants
    n_trials_per_participant=40,         # 40 trials each
    data_instances_dict=data_dict,       # Optional: {pid → [40 arrays]}
    output_csv='out.csv'
)
# Returns: DataFrame(400 rows)
```

---

## Examples by Combination

### **Combo 1: Standard + Single Trial**
```python
result = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=user_loader,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_data, dt_df=dt_data, metadata_df=metadata,
    strategies={...}, XAI_types={...}, training_cog_params={...},
    param_csv_path='params.csv',
    mode='trial'
    # Neither data_instances nor data_instances_dict provided
)
# MODE: 1 (Load via loader)
# EXECUTION: Single trial
# Result: dict
```

### **Combo 2: Synthetic Data + One Participant**
```python
import numpy as np

synthetic_data = [np.random.rand(6) for _ in range(40)]

result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,
    ai_dataset_loader=None,
    lr_df=None, dt_df=None, metadata_df=None,
    strategies={...}, XAI_types={...}, training_cog_params={...},
    param_csv_path='params.csv',
    mode='participant',
    data_instances=synthetic_data,
    n_trials_per_participant=40
)
# MODE: 2 (Pre-computed data)
# EXECUTION: One participant (40 trials)
# Result: DataFrame(40 rows)
```

### **Combo 3: External Data + Full Experiment**
```python
# Load external data as dict
external_data_dict = {
    'pid_001': load_data_for_participant('pid_001'),
    'pid_002': load_data_for_participant('pid_002'),
    ...
}

result_df = generate_trials_from_params_csv(
    model=ppo_model,
    user_loader=None,
    ai_dataset_loader=None,
    lr_df=None, dt_df=None, metadata_df=None,
    strategies={...}, XAI_types={...}, training_cog_params={...},
    param_csv_path='params.csv',
    mode='experiment',
    n_participants=50,
    n_trials_per_participant=40,
    data_instances_dict=external_data_dict
)
# MODE: 2 (Pre-computed data)
# EXECUTION: Full experiment (50 participants × 40 trials)
# Result: DataFrame(2000 rows)
```

### **Combo 4: Mixed (Some From Loader, Some From Data)**
```python
# Not directly supported (would require custom logic)
# Workaround: Either MODE 1 (all from loader) or MODE 2 (all from data)
```

---

## Parameter Validation

The system **automatically detects** which mode to use:

```
IF data_instances is not None:
    USE MODE 2 (Pre-computed data)
    data_instances_dict is ignored
ELIF data_instances_dict is not None:
    USE MODE 2 (Pre-computed data)
ELSE:
    USE MODE 1 (Load from ai_dataset_loader)
    Requires: user_loader and ai_dataset_loader
```

---

## Common Mistakes & Fixes

### ❌ **Mistake 1**: Providing both `data_instances` and trying to use `ai_dataset_loader`
```python
result = generate_trials_from_params_csv(
    user_loader=loader,
    ai_dataset_loader=loader,
    data_instances=my_data,  # ← This takes priority!
    ...
)
# Will use MODE 2, ai_dataset_loader ignored
```
**Fix**: Choose ONE mode. Either provide `data_instances` OR ensure they're None.

### ❌ **Mistake 2**: Wrong data structure
```python
data_instances = np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])  # 2D

# Should be list of arrays:
data_instances = [np.array([0.1, 0.2, ...]), np.array([0.3, 0.4, ...])]
```

### ❌ **Mistake 3**: Using `data_instances_dict` in 'trial' mode
```python
result = generate_trials_from_params_csv(
    mode='trial',
    data_instances_dict={...}  # ← Wrong! Use data_instances for 'trial' mode
)
```
**Fix**: 
- For 'trial' or 'participant': use `data_instances`
- For 'experiment': use `data_instances_dict`

### ❌ **Mistake 4**: Not providing loaders in MODE 1
```python
result = generate_trials_from_params_csv(
    user_loader=None,  # ← Error! Required for MODE 1
    ai_dataset_loader=None,  # ← Error! Required for MODE 1
    data_instances=None,  # MODE 1 is selected by default
    ...
)
# Will crash trying to call methods on None
```
**Fix**: Provide loaders OR provide `data_instances`.

---

## Decision Tree

```
Do you have pre-computed feature arrays?
├─ YES (data_instances or data_instances_dict)
│  └─ Use MODE 2
│     ├─ Single trial/participant? Use data_instances
│     └─ Full experiment? Use data_instances_dict
├─ NO (data_instances = None)
│  └─ Use MODE 1
│     └─ Need working user_loader + ai_dataset_loader
```

---

## Testing Your Choice

### Check MODE 1 Setup
```python
# Test loaders work
try:
    info = user_loader.get_participant_info('sample_id')
    trials = user_loader.get_counterfactual_trials('sample_id')
    print("✓ Loaders work - MODE 1 ready")
except Exception as e:
    print(f"✗ Loader error - use MODE 2: {e}")
```

### Check MODE 2 Setup
```python
# Test data format
assert len(data_instances) > 0, "data_instances empty"
assert len(data_instances[0]) == 6, "Expected 6 features"
print("✓ Data ready - MODE 2 ready")
```

---

## Output Verification

After running, check the output CSV:

### MODE 1 (Loader-based)
```
participant_id,trial_index,model_strategy,data_instance,...
p1,0,change_path_dt,None,...        # data_instance is None (loaded separately)
p1,1,zero_out_lr,None,...
```

### MODE 2 (Pre-computed)
```
participant_id,trial_index,model_strategy,data_instance,...
p1,0,change_path_dt,"[0.1, 0.2, ...]",...  # data_instance is the feature array
p1,1,zero_out_lr,"[0.3, 0.4, ...]",...
```

If `data_instance` column looks wrong, check:
- MODE 1: Should be mostly None (data loaded via separate path)
- MODE 2: Should show feature arrays
