# Architectural Restructuring: Public API Design

## Overview

The XAIK project has been restructured to follow a clear **public API vs. implementation** separation, inspired by frameworks like Quantus and SHAP.

## New Structure

```
xaik-tool-cognitive-agent/
├── user_simulation/              ← PUBLIC API (moved from src/)
│   ├── __init__.py              (main exports)
│   ├── trial_simulator.py        (per-trial simulation)
│   ├── session_generator.py      (multi-trial sessions)
│   ├── parameter_estimator.py    (extract distributions)
│   ├── parameter_sampler.py      (sample parameters)
│   ├── forward_trial_generator.py
│   ├── distribution_loader.py
│   ├── utils.py
│   ├── param_config/             (default parameter files)
│
├── experiments/                  ← USAGE EXAMPLES & RUNNERS (NEW)
│   ├── __init__.py
│   ├── experiment_runner.py      (orchestrate experiments)
│   ├── evaluation.py             (compute metrics)
│   ├── coax_evaluation.py        (CoAX-specific experiments)
│   ├── coxam_evaluation.py       (CoXAM-specific experiments)
│   └── examples/                 (runnable examples)
│
├── src/                          ← IMPLEMENTATION (internal)
│   ├── reasoning_strategies/     (cognitive reasoning layer)
│   │   ├── forward/              (CoAX/CoXAM forward strategies)
│   │   ├── counterfactual/       (CoXAM counterfactual strategies)
│   │   ├── memory/               (cognitive memory backends)
│   │   └── cr_agent/             (CoXAM CR agent orchestration)
│   ├── models/                   (AI model implementations)
│   ├── data_loaders/             (data processing, XAI dataset CSV parsing)
│   ├── xai_adapter/              (XAI methods: attribution, rules/weights)
│   ├── virtual_experiment_executor/ (API-driven virtual experiment simulation)
│   └── rl_agents/                (legacy RL agents)
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
- `src.reasoning_strategies` → Cognitive reasoning API and strategy registry
- `src.reasoning_strategies.forward` → CoAX/CoXAM forward reasoning strategies
- `src.reasoning_strategies.counterfactual` → CoXAM counterfactual strategies
- `src.reasoning_strategies.memory` → Cognitive memory backends
- `src.reasoning_strategies.cr_agent` → CoXAM CR agent orchestration
- `src.models` → AI models (CoAX & CoXAM)
- `src.data_loaders` → Data processing and XAI dataset CSV parsing
- `src.xai_adapter` → XAI methods (SHAP, LIME, Captum, rules/weights, precomputed wrappers)
- `src.virtual_experiment_executor` → API-driven virtual experiment simulation

### 3. **Experiments Layer** (`experiments/`)

**Purpose:** Example workflows and experiment runners

```python
from experiments import ExperimentRunner, EvaluationMetrics

config = ExperimentRunner.Config(
    dataset="wine_quality",
    reasoning_model="coxam",
    n_participants=50
)
runner = ExperimentRunner(config)
results = runner.run()
metrics = EvaluationMetrics(results)
```

**Components:**
- `experiment_runner.py`: Orchestrate full experiments
- `evaluation.py`: Compute metrics (accuracy, response time, XAI impact)
- `coax_evaluation.py`: CoAX-specific evaluation
- `coxam_evaluation.py`: CoXAM-specific evaluation with CR agent

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

### Experiment Runners

```python
# Run evaluation workflows
from experiments import ExperimentRunner, EvaluationMetrics
from user_simulation import TrialSimulator

runner = ExperimentRunner(config)
results = runner.run()
metrics = EvaluationMetrics(results).summary()
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
from src.reasoning_strategies import StrategyRegistry
from src.reasoning_strategies.memory import UnifiedMemory
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

### Run Experiments

```python
from experiments import ExperimentRunner

runner = ExperimentRunner()
results = runner.run(
    n_participants=50,
    n_trials_per_participant=40,
    dataset='wine_quality'
)
print(f"✓ Completed {len(results)} trials")
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

### Layer 2: Experiment Runners

```
experiments/              ← Use this for workflows
├── ExperimentRunner
├── EvaluationMetrics
└── examples/
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
├── reasoning_strategies/
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
3. **Experiments**: Add to `experiments/` folder
4. **Implementation**: Modify `src/` modules

Always maintain the public/internal separation!

## License

See [LICENSE](LICENSE) file.
