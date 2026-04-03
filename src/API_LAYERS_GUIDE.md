# XAIK API Layers - Simplified Guide

## High-Level Architecture

```
🎯 user_simulation/         ← Start here for human-like response generation
    ├─ CoAX path:
    │  └── 📚 reasoning_strategies/    (Forward: SensitiveFeatures, SalientFeatures, ImportanceCategorization, AttributionSum)
    │      ├── uses: 🧠 memory/
    │      └── uses: 🤖 models/
    │
    └─ CoXAM path:
       └──🟦 cr_agent/                (strategy selection + memory)
           └── 📚 reasoning_strategies/    (Forward: DTTraversal, LRCalculation, LRHeuristic; CF: ZeroOutLRHeuristic, ChangeDTPath, MemoryBasedCF)
               ├── uses: 🧠 memory/
               └── uses: 🤖 models/
    
💾 data_loaders/            ← Data processing (shared: Explainers, Normalizers, Filters)
```

---

## Directory Structure

```
src/
├── user_simulation/                    ← Layer 5: High-level user response generation
│   ├── trial_simulator.py
│   ├── session_generator.py
│   ├── parameter_sampler.py
│   ├── parameter_estimator.py
│   ├── forward_trial_generator.py
│   ├── example_session_generation.py
│   └── param_config/
│       └── CoXAM_forward_simulation_cog_param.csv
│
├── cr_agent/                           ← Layer 4: CoXAM-only orchestration
│   ├── interface.py
│   ├── forward_meta_router.py
│   ├── counterfactual_meta_router.py
│   ├── headless_policies.py
│   ├── registry.py
│   ├── tests/
│   └── weights/                        (Pre-trained PPO meta models)
│
├── reasoning_strategies/               ← Layer 3b: Reasoning strategy implementations
│   ├── interface.py
│   ├── registry.py
│   ├── forward/
│   │   ├── coax_forward_rs.py         (SensitiveFeatures, SalientFeatures, etc.)
│   │   └── coxam_forward_rs.py        (DTTraversal, LRCalculation, LRHeuristic)
│   ├── counterfactual/
│   │   └── coxam_counterfactual_rs.py (ZeroOutLRHeuristic, ChangeDTPath, etc.)
│   └── memory/                         ← Layer 3a: Integrated memory infrastructure
│       ├── actr_memory.py          (ACT-R activation, decay, retrieval)
│       ├── exemplar_memory.py      (Similarity-based exemplar storage)
│       ├── unified_memory.py       (Shared interface)
│       └── utils.py
│
├── models/                             ← Layer 2: AI model implementations
│   ├── models.py                       (Model factory)
│   ├── registry.py                     (Model registry)
│   ├── coax/                           (CoAX: Exemplar-based)
│   │   ├── base_engine.py
│   │   ├── mlp/
│   │   └── xgboost/
│   └── coxam/                          (CoXAM: Memory-based)
│       ├── base_engine.py
│       ├── mlp/
│       └── xgboost/
│
└── data_loaders/                       ← Layer 1: Data processing (stateless)
    ├── unified_loader.py               (Main data loading API)
    ├── explainers/
    │   ├── decision_tree.py
    │   ├── logistic_regression.py
    │   ├── shap_explainer.py
    │   ├── lime_explainer.py
    │   └── ...
    ├── normalizers/
    │   ├── minmax.py
    │   └── zscore.py
    ├── filters/
    │   └── filter_builder.py
    └── sources/
        ├── coax_adapter.py
        └── coxam_adapter.py
```

---

## API Layers (Bottom-Up)

### Layer 1: Data Processing - `data_loaders/`
**Purpose:** Prepare and process raw data without memory.

**What it does:**
- Load datasets from multiple sources
- Explain features (DT, LR, SHAP, LIME, LOFO)
- Normalize features (MinMax, Z-Score)
- Filter and select features

**When to use:**
- Preparing training/test data
- Computing feature importance
- Feature normalization for models

**Quick Example:**
```python
from src.data_loaders import UnifiedDataLoader, ExplainerRegistry

loader = UnifiedDataLoader()
X_train, y_train = loader.load_dataset('adult')

# Get feature importance
explainer = ExplainerRegistry.get('logistic_regression')
importances = explainer.explain(X_train, y_train)
```

---

### Layer 2: AI Models - `models/`
**Purpose:** Make predictions and store cognitive data.

**What it does:**
- Unified model factory (LR, DT, MLP, XGBoost)
- Two implementations: CoAX (exemplar-based) & CoXAM (memory-based)
- Model registry and presets

**When to use:**
- Getting AI predictions
- Running counterfactual analysis
- Training/loading saved models

**Quick Example:**
```python
from src.models import ModelFactory

model = ModelFactory.create('logistic_regression', model_type='coxam')
predictions = model.predict(X_test)
explanations = model.explain(X_test)
```

---

### Layer 3a: Core Memory - `memory/`
**Purpose:** Integrated memory infrastructure used by reasoning strategies.

**What it does:**
- ACT-R memory (activation, decay, timing, retrieval)
- Exemplar memory (similarity-based storage/retrieval)
- Unified memory interface shared across strategies

**When to use:**
- Implementing memory effects in strategies
- Modeling human memory decay and interference
- Cross-trial learning and adaptation

**Quick Example:**
```python
from src.reasoning_strategies.memory import UnifiedMemory, ACTRMemory

memory = ACTRMemory()
# Memory is used internally by reasoning strategies
```

---

### Layer 3b: Reasoning Strategies - `reasoning_strategies/`
**Purpose:** Define how humans reason (CoAX vs CoXAM with integrated memory).

**What it does:**
- **Forward strategies:** DTTraversal, LRCalculation, LRHeuristic, SensitiveFeatures, SalientFeatures
- **Counterfactual strategies:** ZeroOutLRHeuristic, ChangeDTPath, MemoryBasedCF
- **Uses:** Integrated memory (`memory/`) + AI models (`models/`)

**When to use:**
- Implementing human-like reasoning
- Applying memory effects (retrieval, interference)
- Combining multiple reasoning approaches

**Quick Example:**
```python
from src.reasoning_strategies import StrategyRegistry, DTTraversal
from src.reasoning_strategies.core.memory import UnifiedMemory

# Load strategy with integrated memory
strategy = DTTraversal(config)
memory = UnifiedMemory()

# Infer with memory effects
probs, time, info = strategy.infer(
    features=X_sample,
    explanation=explanation,
    memory=memory  # Memory integrated into strategy
)
```

---

### Layer 4: CoXAM Orchestration - `cr_agent/` (CoXAM-Only)
**Purpose:** Select optimal reasoning strategy with memory for each decision **in the CoXAM path only**.

**What it does:**
- Strategy selection using trained PPO meta-model (CoXAM strategies: DT, LR Calculation, LR Heuristic)
- Forward episode execution
- Counterfactual Gym environment
- Bridge between strategies and user simulation for CoXAM reasoning

**Note:** This layer is **NOT used** for CoAX reasoning. CoAX strategies go directly through `reasoning_strategies/`.

**When to use:**
- Need dynamic strategy selection in CoXAM path
- Running meta-learning experiments with memory
- Multi-trial reasoning with adaptation (CoXAM)

**Quick Example:**
```python
from src.cr_agent import CRAgentRunner

runner = CRAgentRunner(
    meta_model_path="path/to/meta_model.zip",
    feature_names=feature_cols,
    model_path="path/to/prediction_model.pkl",
)

# Forward episode with strategy selection + memory (CoXAM only)
results = runner.run_forward_episode(
    X_raw=test_data,
    y_true=test_labels,
)

# results include selected strategy, times, memory states
```

---

### Layer 5: User Simulation - `user_simulation/`
**Purpose:** Generate realistic human-like responses at scale.

**What it does:**
- Trial generation (forward, counterfactual)
- Session generation (multi-trial with adaptation)
- Parameter sampling and estimation
- Distribution management

**When to use:**
- Creating synthetic user datasets
- Simulating experimental sessions
- Evaluating XAI effectiveness

**Quick Example:**
```python
from src.user_simulation import (
    TrialSimulator,
    ParameterSampler,
    SessionGenerator,
)

# Simple: Generate single trials
simulator = TrialSimulator()
responses = simulator.generate_forward_trials(
    X_data=X_train,
    y_data=y_train,
    reasoning_model='coxam',  # Uses cr_agent + memory
    parameters=params,
)

# Advanced: Generate full session
session_gen = SessionGenerator()
session = session_gen.generate_session(
    n_trials=50,
    reasoning_model='coxam',
    adaptation=True,  # Learn from feedback
)
```

---

## Usage Patterns

### Pattern 1: Simple Forward Trial (One Decision)
```python
from src.user_simulation import TrialSimulator
from src.data_loaders import UnifiedDataLoader

# 1. Load data
loader = UnifiedDataLoader()
X_train, y_train = loader.load_dataset('adult')

# 2. Generate human-like response for single instance
simulator = TrialSimulator()
response = simulator.generate_forward_trials(
    X_data=X_train[:100],
    y_data=y_train[:100],
    reasoning_model='coxam',
    parameters={'T_enc': 1.5, 'T_op': 0.5, ...}
)

# response = [user decisions with times, explanations, memory effects]
```

### Pattern 2: Custom Strategy + Memory
```python
from src.reasoning_strategies import DTTraversal, StrategyConfig
from src.reasoning_strategies.core.memory import ACTRMemory

# 1. Create strategy
config = StrategyConfig(strategy_name='dt_traversal')
strategy = DTTraversal(config)

# 2. Initialize memory
memory = ACTRMemory()

# 3. Execute reasoning
for sample in X_batch:
    probs, time, info = strategy.infer(
        features=sample,
        explanation=explanation,
        memory=memory  # Integrated memory
    )
    
    # Memory is updated after each decision
    memory.encode(info['decision'])
```

### Pattern 3: Meta-Learning with Strategy Selection
```python
from src.cr_agent import run_meta_on_batch, load_forward_strategies

# 1. Load all forward strategies
strategies = load_forward_strategies()  # {dt, lr_calc, lr_heur}

# 2. Load trained meta-model
meta_model = load_meta_model('path/to/model.zip')

# 3. Run batch with strategy selection
results = run_meta_on_batch(
    meta_model=meta_model,
    strategies=strategies,
    X_batch=X_test,
    y_batch=y_test,
    depths=[1, 2, 3],  # DT depths to test
)

# results[i] = {
#   'selected_strategy': 'dt' | 'lr_calc' | 'lr_heur',
#   'time': float,
#   'confidence': float,
#   'memory_state': {...}
# }
```

### Pattern 4: Full Session Simulation
```python
from src.user_simulation import SessionGenerator, ParameterEstimator

# 1. Learn parameters from data (if available)
estimator = ParameterEstimator()
params = estimator.fit(historical_responses, X_historical)

# 2. Generate realistic session
session_gen = SessionGenerator(estimated_params=params)
session = session_gen.generate_session(
    n_trials=30,
    reasoning_model='coxam',
    adaptation=True,  # Adapt to task feedback
)

# session = [
#   {trial_idx: 0, decision: 1, time: 2.5, memory: {...}, ...},
#   {trial_idx: 1, decision: 0, time: 1.8, memory: {...}, ...},
#   ...
# ]
```

---

## Key Design Principles

| Layer | Key Principle | Example |
|-------|---------------|---------|
| `data_loaders/` | **Stateless** — no memory, just processing | Feature normalization, importance |
| `models/` | **Unified factory** — CoAX or CoXAM | Model factory with registry |
| `memory/` | **Integrated infrastructure** — shared by all strategies | ACT-R activation & decay |
| `reasoning_strategies/` | **Plugins + Memory** — used by both CoAX & CoXAM | DTTraversal (CoXAM), SensitiveFeatures (CoAX) |
| `cr_agent/` | **CoXAM-only orchestration** — selects CoXAM strategy | PPO model picks DT vs LR (CoXAM only) |
| `user_simulation/` | **High-level generation** — both CoAX & CoXAM paths | SessionGenerator with reasoning_model='coxam' or 'coax' |

---

## API Entry Points

| Task | Import | Method | Path |
|------|--------|--------|------|
| **Generate responses** | `user_simulation.TrialSimulator` | `.generate_forward_trials()` | CoAX or CoXAM |
| **Full session** | `user_simulation.SessionGenerator` | `.generate_session()` | CoAX or CoXAM |
| **Strategy selection** | `cr_agent.run_meta_on_batch()` | Pass meta_model + strategies | **CoXAM only** |
| **CoAX reasoning** | `reasoning_strategies.SensitiveFeatures` | `.infer()` with memory | CoAX only |
| **Custom reasoning** | `reasoning_strategies.DTTraversal` | `.infer()` with memory | **CoXAM only** |
| **Load data** | `data_loaders.UnifiedDataLoader` | `.load_dataset()` | Shared |
| **Get model** | `models.ModelFactory` | `.create()` | Shared |

---

## Memory Integration

Memory is **integrated into reasoning strategies** at Layer 3:

**For CoXAM path:** Memory is used through `cr_agent/` → `reasoning_strategies/` → `memory/`

```python
# CoXAM: Memory integrated via cr_agent
from src.cr_agent import CRAgentRunner

runner = CRAgentRunner(...)  # Uses cr_agent for strategy selection
results = runner.run_forward_episode(...)  # Memory handled internally
```

**For CoAX path:** Memory is used directly through `reasoning_strategies/` → `memory/`

```python
# CoAX: Memory integrated directly in strategies
from src.reasoning_strategies import SensitiveFeatures
from src.reasoning_strategies.memory import UnifiedMemory

strategy = SensitiveFeatures(config)
memory = UnifiedMemory()

probs, time, info = strategy.infer(
    features=X,
    explanation=expl,
    memory=memory  # ← Memory integrated here
)
```

**No memory in:**
- `data_loaders/` — stateless processing
- `models/` — single predictions only

**Memory located in:**
- `reasoning_strategies/memory/` — shared infrastructure (ACT-R, Exemplar)
- `cr_agent/` — preserves & manages memory across strategy selections (CoXAM only)
- `user_simulation/` — tracks memory across trials for both paths

---

## Recommended Starting Points

### For Generating Synthetic Data
```python
from src.user_simulation import SessionGenerator, ParameterSampler

session_gen = SessionGenerator()
session = session_gen.generate_session(n_trials=100)
```

### For Understanding Human Reasoning
```python
from src.reasoning_strategies import StrategyRegistry

strategies = StrategyRegistry.get_all_forward()
for name, strategy in strategies.items():
    print(f"{name}: {strategy.describe()}")
```

### For Evaluating XAI Impact (CoXAM)
```python
from src.cr_agent import CRAgentRunner

runner = CRAgentRunner(...)  # CoXAM-only orchestration
with_xai = runner.run_forward_episode(..., condition='DT')
without_xai = runner.run_forward_episode(..., condition='Control')
```

### For CoAX Exemplar-Based Reasoning
```python
from src.reasoning_strategies import SensitiveFeatures, SalientFeatures

# CoAX strategies used directly (no cr_agent)
sensitive = SensitiveFeatures(config)
salient = SalientFeatures(config)

response = sensitive.infer(features=X, explanation=expl)
```
```python
from src.user_simulation import ParameterEstimator

estimator = ParameterEstimator()
params = estimator.fit(user_responses, X_features)
# params = {T_enc, T_op, chi_value, ddm_a, ...}
```

---

## See Also

- [API_STRUCTURE.md](API_STRUCTURE.md) — Detailed file-level documentation
- [cr_agent/README.md](cr_agent/README.md) — CoXAM orchestration details
- [reasoning_strategies/REASONING_STRATEGIES_GUIDE.md](reasoning_strategies/REASONING_STRATEGIES_GUIDE.md) — Strategy specifics
- [data_loaders/README.md](data_loaders/README.md) — Data processing examples
