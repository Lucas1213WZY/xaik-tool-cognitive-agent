# XAIK Tool Cognitive Agent - Complete API Structure

## Directory Tree

```
src/
├── cr_agent/                               # 🟦 COXAM ORCHESTRATION - Cognitive Router Agent
│   ├── __init__.py                         # Public API exports
│   ├── interface.py                        # High-level API (CRAgentRunner, MetaRunner)
│   ├── registry.py                         # Agent/environment registry & presets
│   ├── headless_policies.py               # Forward reasoning policies
│   ├── forward_meta_router.py             # Forward strategy selection & execution
│   ├── counterfactual_meta_router.py      # Counterfactual Gym environment
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── run_standalone_tests.py        # 5/5 tests passing ✅
│   │   └── README.md
│   ├── weights/                            # Pre-trained PPO meta models
│   │   ├── models_meta/best/
│   │   ├── model_calculation/
│   │   ├── model_counterfactual/
│   │   ├── model_dt/
│   │   ├── model_heuristic/
│   │   └── (evaluation results)
│   ├── README.md
│   ├── COMPARISON_ANALYSIS.md
│   └── CONSOLIDATION_COMPLETE.md
│
├── reasoning_strategies/                   # 📚 STRATEGY API LAYER + CORE MEMORY
│   ├── __init__.py                         # Public strategy exports
│   ├── interface.py                        # Strategy contracts (ReasoningStrategy, CounterfactualStrategy)
│   ├── registry.py                         # Strategy plugin system
│   ├── forward/                            # Forward reasoning strategies
│   │   ├── __init__.py
│   │   ├── coax_forward_rs.py             # CoAX exemplar-based strategies
│   │   │   ├── SensitiveFeatures
│   │   │   ├── SalientFeatures
│   │   │   ├── ImportanceCategorization
│   │   │   └── AttributionSum
│   │   └── coxam_forward_rs.py            # CoXAM memory-based strategies
│   │       ├── DTTraversal
│   │       ├── LRCalculation
│   │       └── LRHeuristic
│   ├── counterfactual/                     # Counterfactual strategies
│   │   ├── __init__.py
│   │   └── coxam_counterfactual_rs.py     # CoXAM counterfactual strategies
│   │       ├── ZeroOutLRHeuristic
│   │       ├── ZeroOutLRDisplayed
│   │       ├── ChangeDTPath
│   │       ├── RecallChanges
│   │       └── MemoryBasedCF
│   └── core/                               # 💾 INTEGRATED CORE MEMORY INFRASTRUCTURE
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── interface.py                # Memory interface contract
│       │   ├── actr_memory.py              # ACT-R memory model
│       │   ├── exemplar_memory.py          # Exemplar-based memory
│       │   ├── unified_memory.py           # Unified memory system
│       │   └── utils.py                    # Memory utilities
│       └── __init__.py
│
├── models/                                 # AI model implementations
│   ├── __init__.py
│   ├── models.py                          # AI model factory & API
│   ├── registry.py                        # AI model registry
│   ├── api_examples.py
│   ├── README.md
│   ├── REFACTORING_COMPLETE.md
│   ├── coax/                              # CoAX AI model implementations
│   │   ├── base_model.py
│   │   ├── mlp/
│   │   └── xgboost/
│   └── coxam/                             # CoXAM AI model implementations
│       ├── base_model.py
│       ├── mlp/                          # Neural network backends
│       └── xgboost/                      # Gradient boosting backends
│
├── data_loaders/                          # Data loading layer
│   ├── __init__.py
│   ├── unified_loader.py                 # Main data loader API
│   ├── examples.py
│   ├── tutorial.py
│   ├── README.md
│   ├── base/                             # Base classes
│   │   ├── __init__.py
│   │   ├── data_source.py               # Data source interface
│   │   ├── explainer.py                 # Explainer interface
│   │   └── normalizer.py
│   ├── explainers/                       # Feature explanation methods
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── decision_tree.py
│   │   ├── logistic_regression.py
│   │   ├── attribution_explainer.py
│   │   ├── gradient_based_explainers.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── lofo_explainer.py
│   │   ├── dependencies.py
│   │   ├── examples_extended.py
│   │   └── CONSOLIDATION_SUMMARY.md
│   ├── filters/                          # Feature filtering
│   │   ├── __init__.py
│   │   └── filter_builder.py
│   ├── normalizers/                      # Feature normalization
│   │   ├── __init__.py
│   │   ├── minmax.py
│   │   └── zscore.py
│   └── sources/                          # Data source adapters
│       ├── __init__.py
│       ├── coax_adapter.py
│       └── coxam_adapter.py
│
└── user_simulation/                       # User simulation tools
    ├── __init__.py
    ├── utils.py
    ├── rl_agent_utils.py
    ├── distribution_loader.py
    ├── parameter_sampler.py
    ├── parameter_estimator.py
    ├── forward_trial_generator.py
    ├── trial_simulator.py
    ├── session_generator.py
    ├── example_generate_synthetic_data.py
    ├── example_session_generation.py
    ├── README.md
    ├── param_config/
    │   └── CoXAM_counterfactual_simulation_cog_param.csv
    └── (strategies distributions JSON)
```

## API Layers (Bottom-Up to Top)

### Layer 1: AI Model Implementations  
```
models/ (CoAX + CoXAM)
├── Model interfaces and registries
├── CoAX: Exemplar-based AI models
├── CoXAM: Memory-based AI models (LR, DT, MLP, XGBoost)
└── Use: Predict, explain, store cognitive data
```

### Layer 2: Data Processing
```
data_loaders/  (NO MEMORY)
├── Base interfaces: DataSource, Explainer, Normalizer
├── Explainers: DT, LR, SHAP, LIME, LOFO attribution methods
├── Normalizers: MinMax, Z-Score
├── Filters: Feature selection and filtering
└── Adapters: CoAX/CoXAM-specific data preparation
```

### Layer 3: Reasoning Strategies API (with Integrated Core Infrastructure)
```
reasoning_strategies/  (STRATEGY INTERFACE + CORE MEMORY)
├── Strategy layer:
│   ├── Interface: ReasoningStrategy, CounterfactualStrategy contracts
│   ├── Registry: Plugin-based strategy management
│   ├── Forward: DTTraversal, LRCalculation, LRHeuristic, CoAX methods
│   └── Counterfactual: ZeroOut*, ChangeDTPath, RecallChanges, MemoryBased*
└── Integrated core/memory infrastructure:
    ├── ACT-R Memory: Activation-based retrieval, decay, timing
    ├── Exemplar Memory: Similarity-based exemplar storage/retrieval
    ├── Unified Memory: Coordinated memory backend for all strategies
    └── Use: Cognitive memory modeling embedded in strategy execution
```

### Layer 4: Cognitive Router Agent (CoXAM Orchestration)
```
cr_agent/  🟦 (COXAM STRATEGY ORCHESTRATOR)
├── Public exports via __init__.py:
│   ├── CRAgentRunner (high-level orchestration)
│   ├── MetaRunner (meta episode management)
│   ├── load_forward_strategies()
│   ├── load_counterfactual_strategies()
│   └── Strategy policy classes
├── Internal modules:
│   ├── forward_meta_router.py (forward episode execution)
│   ├── counterfactual_meta_router.py (Gym environment)
│   ├── headless_policies.py (PPO policy classes)
│   └── interface.py (CRAgentRunner, MetaRunner)
└── Use: CoXAM-based reasoning with strategy selection & memory from reasoning_strategies
```

### Layer 5: User Simulation (Top-Level - Human-Like Response Generation)
```
user_simulation/  👥 (PRIMARY USER-FACING API)
├── Integration layer:
│   ├── CoXAM path: Uses cr_agent for memory-based reasoning
│   ├── CoAX path: Uses reasoning_strategies for exemplar-based reasoning
│   └── Can select/combine both approaches
├── Trial generators (forward, counterfactual)
├── Session generators
├── Parameter sampling and estimation
├── Distribution loaders
└── Use: Generate realistic human-like responses by:
    - Selecting reasoning strategy (CoAX vs CoXAM)
    - Executing via cr_agent or reasoning_strategies
    - Processing with data_loaders
    - Leveraging integrated core/memory in reasoning_strategies
```

## Key Design Patterns

### 🔌 Plugin Architecture
- Strategies registered dynamically via `StrategyRegistry`
- Models registered dynamically via `ModelRegistry`
- Explainers registered via `ExplainerRegistry`

### 📦 Single Source of Truth
- Strategies loaded from `reasoning_strategies` API
- Models accessed via `models` API
- Data processing via `data_loaders` API

### 🎯 Layered Dependencies (Bottom-Up)
```
user_simulation (top - human-like responses) 👥
  ├── uses (CoXAM path)
  │   └── cr_agent (memory-based reasoning)
  │       └── reasoning_strategies (loads from)
  │           ├── core/memory (integrated & leveraged)
  │           └── models (executes)
  ├── uses (CoAX path) 
  │   └── reasoning_strategies (exemplar-based)
  │       ├── core/memory (integrated & leveraged)
  │       └── models (executes)
  └── leverages
      ├── models (LR, DT, MLP, XGBoost)
      └── data_loaders (explainers, normalizers - NO MEMORY)
```

**What This Means:**
- `user_simulation` is the primary entry point (generates human-like responses)
- `cr_agent` handles CoXAM-based reasoning with strategy selection
- `reasoning_strategies` provides both CoAX and CoXAM strategy implementations
- **`core/memory` is ONLY in `reasoning_strategies`** — integrated into strategy execution itself
- `data_loaders` is stateless — explainers, normalizers, filters with no memory
- Both reasoning paths (`cr_agent`, `reasoning_strategies + CoAX`) use shared `models` and `data_loaders`

## Import Patterns

### Quick Start - User Simulation (Top-Level API)
```python
from src.user_simulation import (
    TrialSimulator,
    SessionGenerator,
    ParameterSampler,
    ForwardTrialGenerator,
)

# Generate human-like responses
simulator = TrialSimulator()
responses = simulator.generate_responses(
    strategy='coxam',  # or 'coax'
    parameters=params,
    data=features,
)
```

### CoXAM Reasoning (via cr_agent)
```python
from src.cr_agent import (
    CRAgentRunner,
    run_meta_on_batch,
    CounterfactualMetaRouter,
    load_forward_strategies,
)

# Memory-based reasoning with strategy selection
strategies = load_forward_strategies()
results = run_meta_on_batch(meta_model=model, strategies=strategies, ...)
```

### CoAX Reasoning (via reasoning_strategies)
```python
from src.reasoning_strategies import (
    StrategyRegistry,
    SensitiveFeatures,
    SalientFeatures,
)

# Exemplar-based reasoning
coax_strategies = {
    'sensitive': SensitiveFeatures(config),
    'salient': SalientFeatures(config),
}
```

### Strategy API (Advanced)
```python
from src.reasoning_strategies import (
    StrategyRegistry,
    DTTraversal,
    LRCalculation,
    ZeroOutLRHeuristic,
)
```

### Model API (Integration)
```python
from src.models import ModelFactory, ModelRegistry
```

### Data Loading
```python
from src.data_loaders import (
    UnifiedDataLoader,
    ExplainerRegistry,
    NormalizerRegistry,
)
```

## API Usage Examples

### Example 1: Generate Human-Like Responses (User Simulation)
```python
from src.user_simulation import TrialSimulator, ParameterSampler
from src.data_loaders import UnifiedDataLoader

# Load data
loader = UnifiedDataLoader(...)
X_train, y_train = loader.load_dataset('adult')

# Sample cognitive parameters
params = ParameterSampler().sample_forward_params()

# Generate responses using CoXAM reasoning
simulator = TrialSimulator()
responses = simulator.generate_forward_trials(
    X_data=X_train,
    y_data=y_train,
    reasoning_model='coxam',  # or 'coax'
    parameters=params,
)

# responses = list of human-like decisions with explanations
```

### Example 2: Forward Episode with Strategy Selection (CoXAM via cr_agent)
```python
from src.cr_agent import run_meta_on_batch, load_forward_strategies

strategies = load_forward_strategies()  # {dt, lr_calc, lr_heur}
results = run_meta_on_batch(
    meta_model=trained_ppo_model,
    strategies=strategies,
    X_batch=features,
    y_batch=labels,
    depths=[1, 2, 3],
)

# results include per-strategy performance, meta model decisions
```

### Example 3: Counterfactual Gym Environment (CoXAM)
```python
from src.cr_agent import CounterfactualMetaRouter

env = CounterfactualMetaRouter(
    X_data=features,
    y_data=labels,
    meta_model_path="path/to/model.zip",
)

obs, info = env.reset()
for _ in range(100):
    action = agent.predict(obs)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Example 4: CoAX Exemplar-Based Reasoning
```python
from src.reasoning_strategies import SensitiveFeatures, StrategyConfig

# Define strategy
config = StrategyConfig(
    strategy_name='sensitive_features',
    strategy_type=StrategyType.COAX_FORWARD,
)

strategy = SensitiveFeatures(config)

# Generate response
probabilities, time_cost, info = strategy.infer(
    features={'age': 35, 'income': 50000},
    explanation=salience_vector,
)
```

### Example 5: High-Level CoXAM Runner
```python
from src.cr_agent import CRAgentRunner

runner = CRAgentRunner(
    meta_model_path="path/to/meta_model.zip",
    feature_names=feature_cols,
    model_path="path/to/prediction_model.pkl",
)

# Forward reasoning with strategy selection + memory
fwd_results = runner.run_forward_episode(
    X_raw=test_data,
    y_true=test_labels,
    condition='DT',
)

# Counterfactual explanations with memory
cf_results = runner.run_counterfactual_episode(
    X_raw=test_sample,
    target_class=1,
)
```

### Example 6: Session-Level Human-Like Simulation
```python
from src.user_simulation import SessionGenerator, ParameterEstimator

# Estimate cognitive parameters from data
estimator = ParameterEstimator()
params = estimator.fit(X_data=historical_responses)

# Generate full session with multiple trials
session_gen = SessionGenerator(estimated_params=params)
session = session_gen.generate_session(
    n_trials=50,
    reasoning_model='coxam',  # Uses cr_agent internally
    adaptation=True,  # Adapt to feedback
)

# session = realistic user session with memory effects, learning, etc.
```

## Test Coverage

All modules validated:
- ✅ Module imports (1/1)
- ✅ Counterfactual strategy loading (5 strategies)
- ✅ Forward strategy loading (3 strategies)
- ✅ Registry presets
- ✅ Interface classes

**See:** [src/cr_agent/tests/run_standalone_tests.py](src/cr_agent/tests/run_standalone_tests.py)

## Version History

| Version | Status | Changes |
|---------|--------|---------|
| v1.0 | ✅ Current | Consolidated architecture, API-driven design, 5/5 tests passing |
| v0.9 | Legacy | Redundant agents/ and environments/ subdirectories |

## Documentation

- [cr_agent README](src/cr_agent/README.md) - High-level API overview
- [reasoning_strategies interface](src/reasoning_strategies/interface.py) - Strategy contracts
- [models README](src/models/README.md) - Model factory documentation
- [data_loaders README](src/data_loaders/README.md) - Data processing examples
- [Consolidation notes](src/cr_agent/CONSOLIDATION_COMPLETE.md) - Architecture decisions
