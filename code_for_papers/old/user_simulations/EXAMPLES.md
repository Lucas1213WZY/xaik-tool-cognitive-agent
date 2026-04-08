# User Simulation - Running Examples

## Quick Start

The `user_simulation/` package provides tools for generating human-like synthetic participant responses.

### Running Example Scripts

**Important:** Run example scripts as **modules** from the **project root**, not as direct scripts.

#### Example 1: Generate Synthetic Data

```bash
cd /path/to/xaik-tool-cognitive-agent

# Generate synthetic participant data from fitted parameters
python -m user_simulation.example_generate_synthetic_data \
    --fitted-data assets/data/my_fitted_params.csv \
    --output-dir synthetic_output/ \
    --n-participants 50 \
    --n-trials 40 \
    --dataset wine_quality \
    --strategy sensitive_features
```

#### Example 2: Generate Session Data

```bash
cd /path/to/xaik-tool-cognitive-agent

# Generate multi-trial sessions with parameter distributions
python -m user_simulation.example_session_generation \
    --distribution-file assets/param_config/distributions.json \
    --output-dir session_output/ \
    --n-participants 100
```

## API Usage (in your code)

```python
from user_simulation import (
    TrialSimulator,
    SessionGenerator,
    ParameterSampler,
    TrialConfig
)

# Generate trials
simulator = TrialSimulator()
config = TrialConfig(
    participant_id="p001",
    dataset_name="wine_quality",
    strategy_name="sensitive_features",
    cognitive_params={"sensitivity": 76.5, "k": 1},
    n_trials=40
)
results = simulator.simulate(config)
df = simulator.results_to_dataframe(results)
```

## Troubleshooting

### "No module named 'user_simulation'"

**Problem:** Running the script directly from within the folder:
```bash
cd user_simulation
python example_generate_synthetic_data.py  # ❌ Won't work
```

**Solution:** Run as module from project root:
```bash
cd ..
python -m user_simulation.example_generate_synthetic_data  # ✅ Works
```

### "ModuleNotFoundError: No module named 'src'"

**Problem:** The scripts import from internal `src/` modules but they're not installed.

**Solution:** Install the package in development mode:
```bash
cd /path/to/xaik-tool-cognitive-agent
pip install -e .
```

Or ensure the project root is in your Python path.

## Key Components

- **ParameterEstimator** - Extract parameter distributions from fitted data
- **ParameterSampler** - Sample cognitive parameters for new participants  
- **TrialSimulator** - Simulate individual trials with CoAX strategies
- **SessionGenerator** - Generate multi-trial sessions with adaptation
- **ForwardTrialGenerator** - Generate experimental designs with RL

See [user_simulation/__init__.py](./user_simulation/__init__.py) for complete API documentation.
