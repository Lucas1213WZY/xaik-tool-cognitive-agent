"""
================================================================================
GUIDE: Using New Reasoning Strategies API with CSV Parameters
================================================================================

This guide explains how to use the new unified reasoning strategies API
from src/reasoning_strategies/ with parameters loaded from your CSV file.

KEY DIFFERENCES: Old API vs. New API
================================================================================

OLD API (consolidated_human_models.py):
  - Monolithic module with all strategies defined in one place
  - Direct instantiation: strategy = SensitiveFeatures(k=3, sensitivity=10)
  - Parameters embedded in strategy __init__()
  - Memory managed internally per strategy
  
NEW API (src/reasoning_strategies/):
  - Modular, registry-based architecture
  - Config-driven instantiation: strategy = SensitiveFeatures(config)
  - StrategyConfig dataclass encapsulates all parameters
  - Unified memory interface with pluggable backends
  - Better for testing, composition, and extension


MAPPING: CSV COLUMNS TO STRATEGY PARAMETERS
================================================================================

Your CSV file has these parameter columns:
  - k: number of features to focus on
  - sensitivity: feature discrimination sensitivity (t-test)
  - retrieval_threshold: memory activation threshold
  - decay_param: temporal decay rate
  - scaling_factor: for Attribution Sum only
  - explanation_type: 'importance' or 'attribution'

These map to the new StrategyConfig:

  CSV Column              →  StrategyConfig Field
  ─────────────────────────────────────────────────
  k                       →  extra_params['k']
  sensitivity             →  extra_params['sensitivity']
  retrieval_threshold     →  retrieval_threshold
  decay_param             →  decay_param
  scaling_factor          →  extra_params['scaling_factor']
  explanation_type        →  extra_params['explanation_type']


STEP-BY-STEP USAGE
================================================================================

1. LOAD PARAMETERS FROM CSV
──────────────────────────

  from run_simulation_from_params_v2 import CSVParameterLoaderV2
  
  loader = CSVParameterLoaderV2("path/to/three datasets strategies.csv")
  
  # Filter by criteria
  filtered_df = loader.filter_parameters(
      strategy="Sensitive-features categorization",
      xai_type="importance",
      tested_with_xai="w/ XAI",
      dataset="adult"
  )
  
  # Randomly select one participant
  param_row = loader.select_random_params(filtered_df, seed=42)


2. CONVERT TO STRATEGY CONFIG
──────────────────────────────

  from run_simulation_from_params_v2 import CSVParameterLoaderV2
  from src.reasoning_strategies.interface import StrategyConfig
  
  strategy_name = param_row['Strategy']
  config = CSVParameterLoaderV2.create_strategy_config(param_row, strategy_name)
  
  # config is now a StrategyConfig object ready to instantiate


3. INSTANTIATE STRATEGY
──────────────────────

  from src.reasoning_strategies.forward import SensitiveFeatures
  from run_simulation_from_params_v2 import SimpleTimeManager
  
  time_manager = SimpleTimeManager()
  config.time_manager = time_manager  # Optional: add time tracking
  
  strategy = SensitiveFeatures(config)
  # or use instantiate_strategy_new_api helper:
  strategy = instantiate_strategy_new_api(strategy_name, config, time_manager)


4. RUN INFERENCE
────────────────

  # Prepare features (can be dict, list, or np.ndarray)
  features = {"age": 35, "income": 50000, "credit": 750}
  # or
  features = [35, 50000, 750]
  
  # Prepare explanation (optional)
  explanation = {"age": 0.2, "income": 0.5, "credit": 0.3}
  
  # Make inference (returns probabilities, time cost, debug info)
  probs, time_cost, info = strategy.infer(
      features=features,
      explanation=explanation,
      ai_prediction=1  # optional: the true AI prediction
  )
  
  # probs = {0: 0.3, 1: 0.7}
  # time_cost = 0.045  # seconds (cognitive time)
  # info = {'activated_exemplars': 3, 'focus_features': [0, 1], ...}


5. PROVIDE FEEDBACK (LEARNING)
───────────────────────────────

  true_label = 1  # Ground truth for this trial
  
  fb_time = strategy.feedback(
      features=features,
      true_label=true_label,
      explanation=explanation
  )
  
  # strategy's memory is now updated with this new exemplar


6. SIGNAL NEW INSTANCE
───────────────────────

  # This should be called at the start of each new decision
  # It finalizes learning from the previous trial
  strategy.new_instance()


COMPLETE WORKFLOW EXAMPLE
================================================================================

1. Load parameters from CSV and filter:

   from run_simulation_from_params_v2 import CSVParameterLoaderV2, instantiate_strategy_new_api, SimpleTimeManager
   from src.reasoning_strategies.forward import SensitiveFeatures
   import pandas as pd

   loader = CSVParameterLoaderV2("three datasets strategies.csv")
   filtered_df = loader.filter_parameters(
       strategy="Sensitive-features categorization",
       xai_type="importance",
       tested_with_xai="w/ XAI",
       dataset="adult"
   )
   param_row = loader.select_random_params(filtered_df, seed=42)

2. Create config and instantiate:

   config = CSVParameterLoaderV2.create_strategy_config(
       param_row, 
       "Sensitive-features categorization"
   )
   time_manager = SimpleTimeManager()
   strategy = instantiate_strategy_new_api(
       "Sensitive-features categorization",
       config,
       time_manager
   )

3. Run a trial sequence:

   trials = [
       {"instance_id": 0, "is_training": True, "with_explanation": True},
       {"instance_id": 1, "is_training": True, "with_explanation": True},
       {"instance_id": 2, "is_training": False, "with_explanation": False},
   ]
   
   from data_loader import AIDatasetLoader
   import pandas as pd
   
   # Load data
   df_values = pd.read_csv("data/datasets/standard set/values.csv")
   df_meta = pd.read_csv("data/datasets/standard set/metadata.csv")
   df_expl = pd.read_csv("data/datasets/standard set/importance.csv")
   
   ai_loader = AIDatasetLoader(
       feature_values_df=df_values,
       metadata_df=df_meta,
       explanation_values_df=df_expl,
       explanation_columns=['a0_i', 'a1_i', 'a2_i', 'a3_i', 'a4_i']
   )
   
4. Run through each trial:

   results = []
   
   for trial in trials:
       instance_id = trial["instance_id"]
       is_training = trial["is_training"]
       with_explanation = trial["with_explanation"]
       
       # Signal new instance
       strategy.new_instance()
       
       # Get AI data
       features, ai_pred, explanation = ai_loader._get_ai_data(instance_id)
       
       # Run inference
       probs, time_cost, info = strategy.infer(
           features=features,
           explanation=explanation if with_explanation else None,
           ai_prediction=ai_pred
       )
       
       # Store result
       results.append({
           'instance': instance_id,
           'prediction': max(probs, key=probs.get),
           'probabilities': probs,
           'time_cost': time_cost,
       })
       
       # Provide feedback if training
       if is_training:
           strategy.feedback(
               features=features,
               true_label=ai_pred,
               explanation=explanation if with_explanation else None
           )
   
   # Analyze results
   correct = sum(1 for r in results if r['prediction'] == ai_pred)
   print(f"Accuracy: {correct}/{len(results)}")


AVAILABLE STRATEGIES
================================================================================

All strategies inherit from ReasoningStrategy and implement:
  - new_instance(): Signal trial boundary
  - infer(...): Make prediction
  - feedback(...): Learn from outcome
  - get_state(): Export strategy memory state
  - set_state(state): Restore strategy memory state

NEW API STRATEGIES (in src/reasoning_strategies/forward/):

  CoAX Forward Reasoning:
  ──────────────────────
    • SensitiveFeatures
      - Focus on discriminative features (via t-test)
      - Parameters: k, sensitivity, decay_param, retrieval_threshold
    
    • SalientFeatures
      - Focus on high-magnitude explanation components
      - Parameters: k, decay_param, retrieval_threshold
    
    • ImportanceCategorization
      - Use explanation vectors for categorization
      - Parameters: k, sensitivity, decay_param, retrieval_threshold
    
    • AttributionSum
      - Sum attribution/importance for binary decisions
      - Parameters: k, scaling_factor, decay_param, retrieval_threshold
      
  CoXAM Forward Reasoning (also available):
  ─────────────────────────────────────────
    • LRCalculation
      - Logistic regression with learning
    
    • LRHeuristic
      - Simplified heuristic approximation
    
    • DTTraversal
      - Decision tree-based reasoning


STRATEGY CONFIG DATACLASS
================================================================================

StrategyConfig fields:

  strategy_name (str)
      Human-readable name of the strategy
  
  strategy_type (StrategyType: enum)
      Type: COAX_FORWARD, COXAM_FORWARD, COXAM_COUNTERFACTUAL
  
  mode (ReasoningMode: enum)
      Operation mode: RETRIEVE (memory-based), READ (deterministic), HEURISTIC
  
  decay_param (float, default=0.5)
      Temporal decay rate for memory (0.1-1.0)
      0.1 = slow decay (long memory)
      1.0 = fast decay (short memory)
  
  retrieval_threshold (float, default=-2.5)
      Minimum memory activation for retrieval
      Higher threshold = fewer exemplars retrieved
      Lower threshold (more negative) = more exemplars
  
  sensitivity (float, default=10.0)
      Feature sensitivity/discrimination (strategy-specific)
      Higher = more sensitive to feature differences
  
  time_manager (optional)
      Time tracking object with tick(), get_time(), add_time()
      If None, strategies don't track cognitive time
  
  extra_params (dict, default={})
      Strategy-specific parameters:
      - 'k': number of features to attend
      - 'scaling_factor': for attribution strategies
      - 'explanation_type': 'importance' or 'attribution'


DEBUGGING & INSPECTION
================================================================================

1. Inspect strategy metadata:

   strategy = SensitiveFeatures(config)
   metadata = strategy.metadata
   
   print(f"Name: {metadata.display_name}")
   print(f"Category: {metadata.category}")
   print(f"Parameters: {metadata.parameters}")

2. Get state for debugging:

   state = strategy.get_state()
   print(f"Memory size: {state.get('memory_size')}")
   print(f"Exemplars: {state.get('exemplars_count')}")

3. Inspect inference details:

   probs, time_cost, info = strategy.infer(features, explanation)
   
   print(f"Activated exemplars: {info.get('activated_exemplars')}")
   print(f"Focus features: {info.get('focus_features')}")
   print(f"Basis: {info.get('basis')}")  # e.g., "top-3 exemplars"

4. Check config values:

   print(f"decay_param: {config.decay_param}")
   print(f"retrieval_threshold: {config.retrieval_threshold}")
   print(f"Extra params: {config.extra_params}")


COMMON ISSUES & SOLUTIONS
================================================================================

Q: "ImportError: No module named 'src'"
A: Make sure src/ is in your Python path:
   sys.path.insert(0, str(Path(__file__).parent.parent))

Q: "Strategy infer returns None"
A: Check that strategy has exemplars in memory:
   state = strategy.get_state()
   if state['exemplars_count'] == 0:
       print("No exemplars stored yet - run feedback() first")

Q: "Features format error"
A: Features should be dict, list, or np.ndarray:
   # Valid:
   features = {"age": 35, "income": 50000}
   features = [35, 50000]
   features = np.array([35, 50000])
   
   # Invalid:
   features = pd.Series([35, 50000])  # Convert to dict first

Q: "Time manager not working"
A: Check that config.time_manager is set before instantiation:
   time_manager = SimpleTimeManager()
   config.time_manager = time_manager
   strategy = SensitiveFeatures(config)


MIGRATION FROM OLD API
================================================================================

Old code:
  from consolidated_human_models import SensitiveFeatures
  strategy = SensitiveFeatures(k=3, sensitivity=10.0, decay_param=0.5)
  response = strategy.infer_no_explanation(instance_id, features, ai_pred)

New code:
  from src.reasoning_strategies.forward import SensitiveFeatures
  from src.reasoning_strategies.interface import StrategyConfig, ReasoningMode
  
  config = StrategyConfig(
      strategy_name="Sensitive Features",
      mode=ReasoningMode.RETRIEVE,
      decay_param=0.5,
      extra_params={'k': 3, 'sensitivity': 10.0}
  )
  strategy = SensitiveFeatures(config)
  probs, time, info = strategy.infer(features=features, ai_prediction=ai_pred)

Key differences:
  - Config object instead of direct parameters
  - infer() returns (probs, time, info) tuple instead of just response
  - strategy.new_instance() called explicitly instead of implicitly
  - Parameters passed via extra_params dict


FOR MORE INFORMATION
================================================================================

See these files in src/reasoning_strategies/:
  - interface.py: Abstract base class and data structures
  - registry.py: Strategy discovery and registration
  - forward/coax_forward_rs.py: CoAX strategy implementations
  - forward/coxam_forward_rs.py: CoXAM strategy implementations

See these files in src/memory/:
  - unified_memory.py: Unified memory interface
  - exemplar_memory.py: CoAX exemplar backend
  - actr_memory.py: CoXAM ACT-R inspired backend

"""

# This is a documentation file - no executable code


if __name__ == "__main__":
    print(__doc__)
