# Architectural Restructuring: Public API Design

## Overview

The XAIK project has been restructured to follow a clear **public API vs. implementation** separation, inspired by frameworks like Quantus and SHAP.

## New Structure

```
xaik-tool-cognitive-agent/
├── experiment_planner_interface/ ← Experiment planner and participant-facing XAI tester (UI + Experiment)
│   ├── pyproject.toml            (installable xai-tester package)
│   ├── README.md                 (planner-specific usage docs)
│   ├── xai_tester/               (human-subject XAI experiment API)
│   │   ├── control/              (experiment lifecycle: initialise/start/pause/end)
│   │   ├── design/               (Experiment → Session → Trial hierarchy)
│   │   ├── io/                   (terminal presenter and data recorder)
│   │   └── misc/                 (clock and defaults)
│   ├── Examples/                 (loan approval and wine quality studies)
│   └── tests/                    (xai_tester tests)
│
├── UI_components_
│
├── src/                          ← IMPLEMENTATION (internal)
│   ├── cognitive_models/     (cognitive models - reasoning strategies & strategy selector)
│   │   ├── forward/              (CoAX/CoXAM forward strategies)
│   │   ├── counterfactual/       (CoXAM counterfactual strategies)
│   │   ├── memory/               (cognitive memory backends)
│   │   └── cr_agent/             (CoXAM CR agent orchestration)
│   ├── models/                   (AI model implementations)
│   ├── data_loaders/             (data processing, XAI dataset CSV parsing)
│   ├── xai_adapter/              (XAI methods: attribution, rules/weights)
│   └── virtual_experiment_executor/ (API-driven virtual experiment simulation)
│
├── generate_trials_full.py       ← TRIAL GENERATION API
├── generate_trials_from_params.py
└── README.md                      (this file)
```

## Key Changes

### 1. **Public API Layer** (`user_simulation/`)

**Load User Simulation Functions/Classes:**
```python
from user_simulation import TrialSimulator, SessionGenerator
```

**Benefits:**
- Clear user-facing API at project root
- Matches conventions of Quantus, SHAP, Captum
- Easier to discover for new users
- Can evolve independently from `src/` internals

### 2. **Implementation Layer** (`src/`)

**Current internal structure:**
- `src.cognitive_models` → Cognitive reasoning API and strategy registry
- `src.cognitive_models.forward` → CoAX/CoXAM forward reasoning strategies
- `src.cognitive_models.counterfactual` → CoXAM counterfactual strategies
- `src.cognitive_models.memory` → Cognitive memory backends
- `src.cognitive_models.cr_agent` → CoXAM CR agent orchestration
- `src.models` → AI models (CoAX & CoXAM)
- `src.data_loaders` → Data processing and XAI dataset CSV parsing
- `src.xai_adapter` → XAI methods (SHAP, LIME, Captum, rules/weights, precomputed wrappers)
- `src.virtual_experiment_executor` → API-driven virtual experiment simulation

### 3. **Experiment Planner Interface** (`experiment_planner_interface/`)

**Purpose:** Plan and run human-subject XAI evaluation sessions.

Install from the interface folder:

```bash
cd experiment_planner_interface
pip install -e .
```

```python
from xai_tester import control, design, io

exp = design.Experiment(name="LIME Study", labels=["Approved", "Rejected"])
exp.load_csv(
    "data.csv",
    ai_label_col="ai_label",
    xai_cols=["xai_age", "xai_income", "xai_score"],
)

control.initialise(exp)
control.start(participant_id="P01")

for i, trial in enumerate(exp.session.trials, start=1):
    io.present_trial(trial, trial_number=i, total_trials=exp.session.n_trials, labels=exp.labels)
    response, rt = io.get_response(trial)
    exp.data.record(trial, response, rt)

control.end()
```

**Components:**
- `xai_tester.design`: Defines `Experiment`, `Session`, and `Trial`
- `xai_tester.control`: Manages lifecycle calls such as `initialise()`, `start()`, `pause()`, and `end()`
- `xai_tester.io`: Presents trials and records participant responses
- `xai_tester.misc`: Provides timing and configurable defaults
- `Examples/`: Runnable loan approval and wine quality studies

### 4. **Trial Generation API** (NEW)

**Purpose:** Generate trial-by-trial data from best-optimized parameters

```python
from generate_trials_full import generate_trials_from_params_csv

result_df = generate_trials_from_params_csv(
    mode='experiment',
    model=ppo_model,
    data_instances_dict=data_dict,  # Pre-computed or loaded
    param_csv_path='params.csv',
    output_csv='trials.csv'
)
```

**Supports two data modes:**
- **MODE 1**: Load via `ai_dataset_loader` (production)
- **MODE 2**: Pre-supplied data instances (testing/external data)

See [README_API.md](README_API.md) for complete trial generation documentation.

## Import Patterns

### User-Facing (Public API)

```python
# Generate synthetic responses
from user_simulation import (
    TrialSimulator, 
    SessionGenerator,
    ParameterSampler,
    TrialConfig
)

simulator = TrialSimulator()
responses = simulator.simulate(config)
```

### Experiment Planner Interface

```python
# Run participant-facing XAI evaluation sessions
from xai_tester import control, design, io

exp = design.Experiment(name="Loan Approval Study", labels=["Approved", "Rejected"])
exp.load_csv(
    "experiment_planner_interface/Examples/loan_approval/loan_data.csv",
    ai_label_col="ai_label",
    ground_truth_col="ground_truth",
    xai_cols=[
        "xai_age",
        "xai_income_k",
        "xai_credit_score",
        "xai_employment_years",
        "xai_loan_amount_k",
        "xai_debt_ratio",
    ],
)

control.initialise(exp)
control.start(participant_id="P01")
```

### Trial Generation API

```python
# Generate trials from optimized parameters
from generate_trials_full import generate_trials_from_params_csv

df = generate_trials_from_params_csv(
    model=ppo_model,
    mode='experiment',
    data_instances_dict=my_data,
    param_csv_path='params.csv'
)
```

### Internal (Implementation)

```python
# Within src modules, import from src
from src.cognitive_models import StrategyRegistry
from src.cognitive_models.memory import UnifiedMemory
from src.models import ModelFactory
from src.data_loaders import UnifiedDataLoader
from src.xai_adapter import create_xai_method
```

## Benefits of This Architecture

| Aspect | Benefit |
|--------|---------|
| **Discoverability** | Users find main API at project root |
| **Separation of Concerns** | Implementation details in `src/` |
| **Extensibility** | Can add `notebooks/`, `cli/`, `web/` without cluttering |
| **Backward Compatibility** | Can deprecate `src/user_simulation` gradually |
| **Framework Pattern** | Matches Quantus, SHAP, scikit-learn conventions |
| **Documentation** | Clear distinction between public vs internal |
| **Trial Generation** | Standalone API for data generation workflows |

## Quick Start

### Generate Trial Data

```python
import numpy as np
from generate_trials_full import generate_trials_from_params_csv

# Create sample data
data = [np.random.rand(6) for _ in range(40)]

# Generate trials
result_df = generate_trials_from_params_csv(
    model=your_ppo_model,
    user_loader=None,
    ai_dataset_loader=None,
    strategies={0: 'strategy_0', 1: 'strategy_1'},
    XAI_types={0: 'DT', 1: 'LR'},
    training_cog_params={'rt': [-2, 0.5]},
    param_csv_path='assets/param_config/CoXAM_counterfactual_simulation_cog_param.csv',
    mode='participant',
    data_instances=data,
    output_csv='output.csv'
)

print(f"✓ Generated {len(result_df)} trials")
```

### Run A Participant-Facing XAI Study

```python
from xai_tester import control, design, io

exp = design.Experiment(name="Loan Approval Study", labels=["Approved", "Rejected"])
exp.load_csv(
    "experiment_planner_interface/Examples/loan_approval/loan_data.csv",
    ai_label_col="ai_label",
    ground_truth_col="ground_truth",
    xai_cols=[
        "xai_age",
        "xai_income_k",
        "xai_credit_score",
        "xai_employment_years",
        "xai_loan_amount_k",
        "xai_debt_ratio",
    ],
)

control.initialise(exp)
control.start(participant_id="P01")

for i, trial in enumerate(exp.session.trials, start=1):
    io.present_trial(trial, trial_number=i, total_trials=exp.session.n_trials, labels=exp.labels)
    response, rt = io.get_response(trial)
    exp.data.record(trial, response, rt)

control.end()
```

## Project Structure by Layer

### Layer 1: User-Facing APIs

```
user_simulation/          ← Use this
├── TrialSimulator
├── SessionGenerator
├── ParameterSampler
├── ParameterEstimator
└── ...
```

### Layer 2: Experiment Planner Interface

```
experiment_planner_interface/ ← Use this for participant-facing XAI studies
├── xai_tester/
│   ├── control/
│   ├── design/
│   ├── io/
│   └── misc/
├── Examples/
└── tests/
```

### Layer 3: Trial Generation API

```
generate_trials_*         ← Use this for data generation
├── generate_trials_from_params_csv()
├── generate_participant_session()
└── generate_single_trial()
```

### Layer 4: Implementation (Internal)

```
src/                      ← Don't use directly
├── cognitive_models/
│   ├── forward/
│   ├── counterfactual/
│   ├── memory/
│   └── cr_agent/
├── models/
├── data_loaders/
├── xai_adapter/
└── virtual_experiment_executor/
```

## Contributing

To extend this project:

1. **Public API changes**: Update `user_simulation/`
2. **Trial generation**: Extend `generate_trials_full.py`
3. **Experiment planner**: Update `experiment_planner_interface/xai_tester/`
4. **Implementation**: Modify `src/` modules

Always maintain the public/internal separation!

## License

See [LICENSE](LICENSE) file.
