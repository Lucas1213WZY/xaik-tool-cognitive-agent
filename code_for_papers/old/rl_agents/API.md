# RL Agents API Reference

## Overview

The `src/rl_agents` module provides a complete reinforcement learning framework for training cognitive agents to select optimal reasoning strategies. It integrates seamlessly with the unified memory system (`src.core.memory`) and reasoning strategies registry (`src.reasoning_strategies`).

**Latest Version:** 0.1.0

### Key Features

- ✅ Gymnasium-compatible environments for PPO training
- ✅ Integration with unified memory system (ACT-R + Exemplar backends)
- ✅ Integration with reasoning strategies (LR Calculation, LR Heuristic, Decision Tree)
- ✅ Stable-Baselines3 PPO agents with full training/eval harness
- ✅ High-level inference API with caching and uncertainty estimation
- ✅ Training orchestration with logging and checkpoint management

---

## Architecture

```
src/rl_agents/
├── environments/           # Gymnasium environments
│   ├── base_env.py        # BaseRLEnvironment (abstract)
│   ├── dt_forward_env.py  # Decision Tree environment
│   └── lr_forward_env.py  # LR Calculation & Heuristic environments
├── agents/                # PPO agents
│   ├── base_agent.py      # BaseRLAgent (abstract)
│   ├── dt_agent.py        # DTAgent
│   └── lr_agent.py        # LRAgent, LRCalculationAgent, LRHeuristicAgent
├── utils/                 # Utilities
│   ├── inference.py       # InferenceManager, PredictionCache
│   └── training.py        # TrainingManager
└── __init__.py            # Public API
```

### Integration Points

```
┌─────────────────────┐
│   RL Agents Layer   │
│  (src/rl_agents/)   │
└──────────┬──────────┘
           │
      ┌────┴─────┐
      │           │
      ▼           ▼
┌──────────────┐ ┌──────────────────────┐
│ Core Memory  │ │ Reasoning Strategies │
│(ACT-R/Exem) │ │  (LR, DT, Routes)    │
└──────────────┘ └──────────────────────┘
```

---

## Quick Start

### 1. Train a Decision Tree Agent

```python
from src.rl_agents import DTAgent, AgentConfig, EnvironmentConfig

# Configure training
agent_config = AgentConfig(
    agent_name="dt_agent_v1",
    learning_rate=3e-4,
    total_timesteps=50_000,
    n_steps=2048,
    batch_size=64,
    verbose=1,
)

# Create agent
agent = DTAgent(agent_config)

# Train on 4 parallel environments
results = agent.train(n_envs=4)
print(f"Training complete: {results}")

# Save model
agent.save("./models/dt_agent.zip")
```

### 2. Train an LR Agent

```python
from src.rl_agents import LRAgent, AgentConfig

# Configure (same as DT)
config = AgentConfig(
    agent_name="lr_agent_v1",
    total_timesteps=100_000,
)

# Create and train LR Calculation agent
agent = LRAgent(config, strategy_type="calculation")
agent.train(n_envs=4)

# Or LR Heuristic agent
agent = LRAgent(config, strategy_type="heuristic")
agent.train(n_envs=4)
```

### 3. Use Trained Agent for Inference

```python
from src.rl_agents import InferenceManager
import numpy as np

# Load trained agent
agent = DTAgent(config)
agent.load("./models/dt_agent.zip")

# Single prediction
obs = np.random.randn(7)  # Observation (7 dims for DT env)
action, _value = agent.predict(obs, deterministic=True)

# Batch inference with caching
inference_mgr = InferenceManager(agent, use_cache=True)
observations = [np.random.randn(7) for _ in range(100)]
results = inference_mgr.predict_batch(observations)

# Get cache statistics
stats = inference_mgr.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
```

### 4. Training with Orchestration

```python
from src.rl_agents import TrainingManager

manager = TrainingManager(agent, log_dir="./training_logs")

# Train and log
results = manager.train(
    n_envs=4,
    total_timesteps=100_000,
    eval_freq=5_000,
)

# Export training history
manager.export_history(format="json")
summary = manager.get_summary()
```

---

## API Reference

### Environments

#### EnvironmentConfig

```python
@dataclass
class EnvironmentConfig:
    # Dataset and task
    dataset_name: str = "wine_quality"
    model_type: str = "mlp"
    model_path: Optional[str] = None
    
    # Memory system
    memory_backend: MemoryBackend = MemoryBackend.ACTR
    decay_param: float = 0.5
    retrieval_threshold: float = -2.5
    latency_factor: float = 0.5
    activation_noise: float = 0.1
    max_assoc_strength: float = 2.0
    mismatch_penalty: float = -2.0
    
    # Cognitive parameters (ranges for sampling)
    chi_range: Tuple[float, float] = (0.5, 2.0)
    ddm_a_range: Tuple[float, float] = (0.3, 1.0)
    ddm_s_range: Tuple[float, float] = (0.8, 1.0)
    
    # Training
    max_features: int = 6
    instances_per_episode: int = 40
    xai_trial_ratio: float = 0.5  # Fraction with XAI
    
    seed: Optional[int] = None
    verbose: bool = False
```

#### BaseRLEnvironment

Abstract base class implementing Gymnasium API with memory and strategy integration.

**Methods:**
- `reset(seed, options)` → (observation, info)
- `step(action)` → (observation, reward, terminated, truncated, info)
- `seed(seed)` → [seed]
- `get_episode_stats()` → Dict with episode metrics

**Properties:**
- `action_space`: Gymnasium action space
- `observation_space`: Gymnasium observation space
- `memory`: Unified memory instance

#### DTForwardEnvironment

Environment for Decision Tree strategy selection.

**Action Space:** `MultiDiscrete([3, 5])`
- `a[0]` ∈ {1, 2}: 1="read", 2="retrieve"
- `a[1]` ∈ {0..4}: ddm_a bin discretization

**Observation Space:** `Box(low=0, high=1, shape=(7,))`
- trial progress, strategy counts, success rates, XAI flag

**Example:**
```python
from src.rl_agents import DTForwardEnvironment, EnvironmentConfig

config = EnvironmentConfig(dataset_name="wine_quality")
env = DTForwardEnvironment(config)

obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

#### LRCalculationEnvironment

Environment for LR Calculation with feature selection.

**Action Space:** `MultiBinary(max_features)`
- Binary mask selecting which features to use

**Observation Space:** `Box(low=-5, high=5, shape=(2*max_features + 3,))`
- Instance features, feature history, success statistics

#### LRHeuristicEnvironment

Environment for LR Heuristic with parameter adaptation.

**Action Space:** `Discrete(3)`
- 0: decrease chi, 1: keep, 2: increase chi

**Observation Space:** `Box(low=-5, high=5, shape=(max_features + 4,))`

---

### Agents

#### AgentConfig

```python
@dataclass
class AgentConfig:
    # Identity
    agent_name: str = "default_agent"
    agent_type: str = "ppo"
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Network architecture
    net_arch: List[int] = [64, 64]
    activation_fn: Type = nn.Tanh
    
    # Training
    total_timesteps: int = 100_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10
    save_freq: int = 10_000
    save_path: str = "./model_checkpoints"
    
    seed: Optional[int] = None
    verbose: int = 1
```

#### BaseRLAgent

Abstract base class for all RL agents.

**Methods:**
- `train(n_envs, total_timesteps)` → Dict with results
- `evaluate(n_episodes)` → Dict with metrics
- `predict(observation, deterministic)` → (action, value)
- `predict_batch(observations, deterministic)` → np.ndarray
- `save(path, include_metadata)` → bool
- `load(path)` → bool
- `get_metadata()` → Dict

**Properties:**
- `model`: Underlying PPO model
- `config`: AgentConfig instance
- `training_metadata`: Training history

**Example:**
```python
from src.rl_agents import DTAgent, AgentConfig

config = AgentConfig(agent_name="dt_v1", total_timesteps=50_000)
agent = DTAgent(config)

# Train
results = agent.train(n_envs=4)

# Evaluate
metrics = agent.evaluate(n_episodes=20)
print(f"Mean reward: {metrics['mean_reward']:.3f}")

# Save/Load
agent.save("./models/dt_v1.zip")
agent.load("./models/dt_v1.zip")

# Inference
action, _ = agent.predict(obs)
```

#### DTAgent

Specialized agent for Decision Tree environment.

```python
from src.rl_agents import DTAgent, AgentConfig, EnvironmentConfig

env_config = EnvironmentConfig(
    dataset_name="wine_quality",
    instances_per_episode=40,
    xai_trial_ratio=0.5,
)

agent_config = AgentConfig(
    agent_name="dt_agent",
    learning_rate=1e-3,
    total_timesteps=100_000,
)

agent = DTAgent(agent_config, env_config)
agent.train(n_envs=4)
```

#### LRAgent & Variants

```python
from src.rl_agents import LRAgent, LRCalculationAgent, LRHeuristicAgent

# Option 1: Unified LRAgent
agent = LRAgent(config, strategy_type="calculation")

# Option 2: Specific agent
calc_agent = LRCalculationAgent(config)
heur_agent = LRHeuristicAgent(config)

# All expose same interface
agent.train(n_envs=4)
metrics = agent.evaluate(n_episodes=10)
```

---

### Utilities

#### InferenceManager

High-level API for agent inference with caching and uncertainty estimation.

```python
from src.rl_agents import InferenceManager
import numpy as np

# Initialize
agent = DTAgent(config)
agent.load("./models/dt.zip")
inf_mgr = InferenceManager(agent, use_cache=True, cache_size=1000)

# Single prediction
obs = np.random.randn(7)
result = inf_mgr.predict(obs, deterministic=True)
print(f"Action: {result.action}, Confidence: {result.confidence}")

# Batch predictions
observations = [np.random.randn(7) for _ in range(50)]
results = inf_mgr.predict_batch(observations, batch_size=16)

# With uncertainty estimation
result = inf_mgr.estimate_uncertainty(obs, n_samples=10)
print(f"Mean: {result['mean']}, Std: {result['std']}")

# Get statistics
stats = inf_mgr.get_stats()
print(f"Inferences: {stats['inference_count']}")
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
```

**Methods:**
- `predict(obs, deterministic, use_cache)` → PredictionResult
- `predict_batch(observations, deterministic, batch_size)` → List[PredictionResult]
- `estimate_uncertainty(obs, n_samples)` → Dict
- `predict_with_uncertainty(obs, estimate_uncertainty, n_samples)` → Dict
- `get_stats()` → Dict

#### PredictionCache

Simple LRU cache for prediction results.

```python
from src.rl_agents.utils import PredictionCache

cache = PredictionCache(max_size=1000)

# Get/put prediction
result = cache.get(obs)
if result is None:
    result = agent.predict(obs)
    cache.put(obs, result)

# Statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

#### TrainingManager

Orchestrates training runs with logging and checkpointing.

```python
from src.rl_agents import TrainingManager

manager = TrainingManager(agent, log_dir="./logs")

# Train with orchestration
results = manager.train(
    n_envs=4,
    total_timesteps=100_000,
    eval_freq=5_000,
    n_eval_episodes=10,
)

# Save checkpoints
manager.save_checkpoint("./models/ckpt_50k.zip")

# Export history
manager.export_history(format="json")
manager.export_history(format="csv")

# Get summary
summary = manager.get_summary()
```

**Methods:**
- `train(n_envs, total_timesteps, eval_freq, n_eval_episodes)` → Dict
- `save_checkpoint(path, metadata)` → bool
- `export_history(format)` → str (file path)
- `get_summary()` → Dict

---

## Integration with Core Layers

### With Unified Memory

Environments automatically initialize memory with configured backend:

```python
env_config = EnvironmentConfig(
    memory_backend=MemoryBackend.ACTR,  # Use ACT-R backend
    decay_param=0.5,
    retrieval_threshold=-2.5,
)

env = DTForwardEnvironment(env_config)
# Memory is initialized in env.memory
```

### With Reasoning Strategies

Environments automatically load strategies from registry:

```python
# In DTForwardEnvironment.__init__:
self.dt_strategy = StrategyRegistry.get("dt_traversal")

# In step():
probs, time_cost, info = self.dt_strategy.infer(
    features=self.current_instance,
    explanation=None if not self.with_xai else {},
    **cognitive_params
)
```

---

## Training Workflow

### Basic Training

```python
from src.rl_agents import DTAgent, AgentConfig

config = AgentConfig(
    agent_name="dt_v1",
    total_timesteps=50_000,
    eval_freq=5_000,
    n_eval_episodes=10,
)

agent = DTAgent(config)
results = agent.train(n_envs=4)

if results["success"]:
    print(f"Training successful!")
    print(f"Eval results: {results['eval_results']}")
    
    agent.save("./models/dt_v1.zip")
```

### Advanced Training with Monitoring

```python
from src.rl_agents import TrainingManager, DTAgent

agent = DTAgent(config)
manager = TrainingManager(agent, log_dir="./training_logs")

# Train
results = manager.train(n_envs=8, total_timesteps=200_000)

# Save checkpoints during training (can be integrated into callback)
manager.save_checkpoint(
    "./models/checkpoint_100k.zip",
    metadata={"timesteps": 100_000}
)

# Export
manager.export_history(format="json")
summary = manager.get_summary()
print(summary)
```

---

## Evaluation Strategies

### Deterministic Evaluation

```python
# Load model
agent.load("./models/dt.zip")

# Deterministic predictions (best action)
metrics = agent.evaluate(n_episodes=50)
print(f"Reward: {metrics['mean_reward']:.3f}")
```

### Stochastic Evaluation with Uncertainty

```python
from src.rl_agents import InferenceManager

inf_mgr = InferenceManager(agent)

# Each observation has mean + uncertainty
for obs in test_obs:
    result = inf_mgr.predict_with_uncertainty(
        obs, 
        estimate_uncertainty=True,
        n_samples=20
    )
    
    print(f"Action: {result['action']}")
    print(f"Uncertainty: {result['uncertainty']}")
```

---

## Troubleshooting

### Model Not Training

```python
# Check environment creation
env = DTForwardEnvironment(config)
obs, info = env.reset()
assert obs.shape == env.observation_space.shape

# Check agent initialization
agent = DTAgent(config)
results = agent.train(n_envs=1, total_timesteps=100)  # Quick test
```

### Memory Issues with Parallel Envs

```python
# Use DummyVecEnv instead of SubprocVecEnv
# (Done automatically in agent.create_env for small n_envs)

# Reduce n_steps or batch_size in AgentConfig
config.n_steps = 1024
config.batch_size = 32
```

### Cache Not Helping

```python
# Check cache stats
stats = inf_mgr.get_stats()
if stats['cache']['hit_rate'] < 0.1:
    # Cache not effective; disable it
    inf_mgr = InferenceManager(agent, use_cache=False)
```

---

## Performance Characteristics

### Training

| Configuration | Time (100K steps) | Memory |
|---------------|------------------|---------|
| 1 env, DT | ~2-3 min | ~500MB |
| 4 envs, DT | ~1-2 min | ~1.5GB |
| 8 envs, LR | ~1 min | ~2GB |

### Inference

| Scenario | Latency | Throughput |
|----------|---------|-----------|
| Single prediction | ~5-10ms | 100 pred/sec |
| Batch (32) | ~50-100ms | 300-600 pred/sec |
| With uncertainty (10 samples) | ~50-100ms | 100-200 pred/sec |

---

## Migration Guide

### From Notebook-based Development

If using the notebooks in `RL_training/`:

```python
# Old notebook way:
# - env = DTForward(config)
# - model.learn()

# New API way:
from src.rl_agents import DTAgent, AgentConfig

agent = DTAgent(AgentConfig())
agent.train(n_envs=4)
```

### Using with Existing Memory/Strategies

```python
# Core memory, reasoning strategies already available
# Just provide config with proper parameters

env_config = EnvironmentConfig(
    # Memory system will use these parameters
    memory_backend=MemoryBackend.ACTR,
    decay_param=0.5,
    # Strategy will be loaded from registry
)

env = DTForwardEnvironment(env_config)
```

---

## MetaRouterAgent: Multi-Strategy Inference

### Overview

The `MetaRouterAgent` orchestrates multiple reasoning strategies (Decision Tree, LR Calculation, LR Heuristic) using a trained meta-level PPO policy. It supports both training and direct inference on experimental data.

**Key Features:**
- Learns which strategy to dispatch for each trial
- Handles with/without XAI scheduling
- Manages condition-based strategy gating (DT-only, LR-only, mixed)
- Per-strategy performance tracking
- Direct inference on provided data (no environment needed)

### Training MetaRouter

```python
from src.rl_agents import MetaRouterAgent, AgentConfig

# Configure agent
config = AgentConfig(
    agent_name="meta_router_v1",
    learning_rate=3e-4,
    total_timesteps=100_000,
)

# Create agent with strategies
agent = MetaRouterAgent(
    config=config,
    strategies={
        "dt": dt_strategy,
        "lr_calc": lr_calc_strategy,
        "lr_heur": lr_heur_strategy,
    },
    dataset_loaders={
        1: dataset_loader_1,
        2: dataset_loader_2,
    },
    training_cog_params={
        "chi": [0.0, 0.03],  # Time cost range
        "T_enc": 1.5,
        "T_op": 0.3,
    },
    instances_per_episode=40,
    xai_trial_ratio=0.5,
)

# Train
results = agent.train(n_envs=4)

# Save
agent.save("./models_meta/best_model.zip")
```

### Direct Inference with run_episode()

```python
from src.rl_agents import MetaRouterAgent, COND_DTLR
import numpy as np

# Load trained meta agent
agent = MetaRouterAgent(config, strategies=strategies, ...)
agent.load("./models_meta/best_model.zip")

# Prepare data
X_raw = np.random.randn(100, 23).astype(np.float32)  # 100 trials, 23 features
y_raw = np.random.randint(0, 2, 100)
X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-9)

# Run inference episode
result = agent.run_episode(
    X_raw=X_raw,
    y_raw=y_raw,
    X_norm=X_norm,
    condition=COND_DTLR,  # Mixed DT+LR condition
    with_xai_ratio=0.5,  # 50% of trials with XAI
    episode_cogs={
        "T_enc": 1.5,
        "T_op": 0.3,
        "latency_factor": 0.2,
        "ddm_a": 1.2,
        "ddm_s": 0.9,
        "lapse": 0.04,
        "retrieval_threshold": 0.0,
    },
    chi_value=0.01,
    deterministic=False,  # Use stochastic policy
)

# Access results
print(f"Total Reward: {result['total_reward']:.4f}")
print(f"Mean Reward: {result['mean_reward']:.4f}")

# Per-trial logs
logs = result['logs']
print(f"Strategy selections: {logs['strategy_name']}")
print(f"Correctness: {logs['prob_correct']}")
print(f"Prediction times: {logs['pred_time']}")
print(f"Rewards: {logs['reward']}")
```

### Output Format

`run_episode()` returns a dict with:

```python
{
    "total_reward": float,           # Sum of all trial rewards
    "mean_reward": float,            # Average reward
    "logs": {
        "strategy_name": List[str],           # Selected strategy per trial
        "action_idx": List[int],              # Action index (strategy id)
        "prob_correct": List[float],          # P(correct) per trial
        "pred_time": List[float],             # Prediction time per trial
        "reward": List[float],                # Reward per trial
        "with_xai_requested": List[bool],     # XAI scheduling
        "with_xai_used": List[bool],          # XAI actually used
        "trial_type": List[str],              # DT or LR trial type
        "condition": List[str],               # Episode condition
        "mismatch_applied": List[bool],       # Strategy/trial type mismatch
        "invalid_under_condition": List[bool],# Condition gating violation
        "probs": List[List[float]],           # Full prob distributions
        "info": List[Dict],                   # Strategy-specific info
    },
    "meta": {
        "N": int,                  # Number of trials
        "chi_value": float,        # Time cost parameter (actual)
        "chi_high": float,         # Time cost normalization (reference)
        "strategy_order": List[str],# Strategy names in action space order
        "episode_cogs": Dict,      # Cognitive parameters used
        "condition": str,          # Episode condition
    }
}
```

### Condition Gating

MetaRouter respects strategy-condition boundaries:

| Condition | Allowed Strategies |
|-----------|-------------------|
| `"DT"` | Only `"dt"` |
| `"LR"` | Only `"lr_calc"`, `"lr_heur"` |
| `"DT+LR"` | All strategies |

Disallowed actions receive a penalty:

```python
# This will be penalized if condition is DT
# because LR strategies aren't allowed
agent.run_episode(
    X_raw=X_raw,
    y_raw=y_raw,
    condition="DT",  # Only DT allowed
    invalid_action_penalty=-1.0,  # Penalty for invalid actions
)
```

### With-XAI Mismatch Handling

If a trial requests XAI but doesn't match the strategy:

- **Trial type is DT, strategy is LR** → Run LR WITHOUT XAI
- **Trial type is LR, strategy is DT** → Run DT WITHOUT XAI
- **Otherwise** → Use requested XAI setting

```python
# Example: condition is DT, but trial_type_schedule varies
# Meta agent might select LR strategy on some DT trials
# Those trials will automatically drop XAI to avoid mismatch
result = agent.run_episode(
    X_raw=X_raw,
    y_raw=y_raw,
    condition=COND_DTLR,
    trial_type_schedule=trial_types,  # [DT, LR, DT, DT, LR, ...]
)
# Check logs for mismatch_applied to see where it happened
mismatches = [m for m in result['logs']['mismatch_applied'] if m]
print(f"Mismatches handled: {len(mismatches)}")
```

### Integration with ForwardTrialDatasetGenerator

Use with the forward trial generator for full experimental workflows:

```python
from src.rl_agents import create_forward_runner
from src.user_simulation import generate_forward_trials, ExperimentalDesign

# Create runner
forward_runner = create_forward_runner(
    meta_model=agent.model,
    strategies=agent.strategies,
    training_cog_params=agent.training_cog_params,
)

# Define experimental design
design = ExperimentalDesign(
    app_ids=["wine_quality", "mushrooms"],
    model_names=["mlp", "xgb"],
    complexities=["low", "high"],
    n_trials_per_condition=100,
    xai_type_distribution={"DT": 0.3, "LR": 0.3, "DT+LR": 0.4},
)

# Generate all forward trials
output_path, df = generate_forward_trials(
    forward_runner=forward_runner,
    ai_dataset_loader=dataset_loader,
    design=design,
    episode_cogs=episode_cogs,
    chi_value=0.01,
)

print(f"Generated {len(df)} trials in {output_path}")
```

---

## Best Practices

✅ **DO:**
- Use `TrainingManager` for orchestrated training
- Cache predictions in production inference
- Evaluate on separate episode batches
- Save checkpoints during long training runs
- Vary cognitive parameters during environment episodes

❌ **DON'T:**
- Reuse environments across different agents without reset()
- Skip normalization/scaling for neural network inputs
- Train with very small batch sizes (< 32)
- Use deterministic=False for production decision-making
- Forget to close environments after use

---

## API Stability

**Current Status:** Beta (v0.1.0)

Interfaces are stable but may change based on user feedback.

**Stable:**
- EnvironmentConfig, BaseRLEnvironment contracts
- AgentConfig, BaseRLAgent public methods
- InferenceManager API

**May Change:**
- Specific environment observation/action spaces
- PPO hyperparameter defaults
- Training callback system

---

## See Also

- [Memory Layer Docs](../memory/)
- [Reasoning Strategies Docs](../reasoning_strategies/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
