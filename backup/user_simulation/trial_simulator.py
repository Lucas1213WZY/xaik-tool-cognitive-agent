"""
Trial Simulator - Generate per-trial human-like responses using CoAX strategies.

Coordinates CoAX reasoning strategies, memory, and explanations to simulate
realistic participant responses on a per-trial basis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TrialConfig:
    """Configuration for trial simulation."""
    
    # Participant info
    participant_id: str
    
    # Dataset and data loader
    dataset_name: str  # "adult", "wine_quality", "forest_cover"
    ai_dataset_loader: Optional[Any] = None  # AIDatasetLoader instance
    
    # Strategy and XAI settings
    strategy_name: str = "sensitive_features"
    xai_type: str = "Importance"  # "Importance", "Attribution", or "None"
    tested_with_xai: bool = True
    
    # Cognitive parameters
    cognitive_params: Dict[str, float] = field(default_factory=dict)
    
    # Explanation and model setup
    explainer: Optional[Any] = None  # Explanation model (e.g., DecisionTreeInterpreter)
    ai_model: Optional[Any] = None  # Fitted AI model for predictions
    
    # Trial setup
    n_trials: int = 40
    random_seed: Optional[int] = None
    
    # Timing parameters
    t_read_num: float = 2.0  # Time for reading explanations
    w0_ans: float = 0.1  # Base activation
    lapse: float = 0.05  # Lapse rate


@dataclass
class TrialResult:
    """Result of a single trial simulation."""
    
    # Trial identifiers
    participant_id: str
    trial_idx: int
    instance_id: int
    
    # Trial conditions
    tested_with_xai: bool
    strategy: str
    xai_type: str
    
    # Instance data
    ai_prediction: int
    explainer_prediction: int
    true_label: Optional[int] = None
    
    # Participant response
    participant_response: int = 0
    response_prob: Dict[int, float] = field(default_factory=dict)
    response_time: float = 0.0
    
    # Performance metrics
    response_matches_ai: bool = False
    response_matches_explainer: bool = False
    correct: bool = False
    
    # Meta information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame conversion."""
        return {
            "Participant ID": self.participant_id,
            "Trial Index": self.trial_idx,
            "Instance Id": self.instance_id,
            "Tested w/ XAI": "w/ XAI" if self.tested_with_xai else "w/o XAI",
            "Strategy": self.strategy,
            "XAI Type": self.xai_type,
            "AI Prediction": self.ai_prediction,
            "Explainer Prediction": self.explainer_prediction,
            "Response": self.participant_response,
            "Response Prob 0": self.response_prob.get(0, 0.0),
            "Response Prob 1": self.response_prob.get(1, 0.0),
            "Response Time (s)": self.response_time,
            "Response==AI": int(self.response_matches_ai),
            "Response==Explainer": int(self.response_matches_explainer),
            "Correct": int(self.correct) if self.true_label is not None else None,
            **self.metadata,
        }


class TrialSimulator:
    """
    Simulate per-trial participant responses using CoAX reasoning strategies.
    
    Usage:
        simulator = TrialSimulator()
        config = TrialConfig(
            participant_id="p001",
            dataset_name="wine_quality",
            strategy_name="sensitive_features",
            cognitive_params={"sensitivity": 76.7, "k": 1, "retrieval_threshold": -2.97}
        )
        results = simulator.simulate(config)
        df = simulator.results_to_dataframe(results)
    """
    
    def __init__(self):
        self.strategy_registry = None
        self.memory_system = None
        self.results: List[TrialResult] = []
    
    def setup_dependencies(self, 
                          strategy_registry: Optional[Any] = None,
                          memory_system: Optional[Any] = None) -> None:
        """
        Setup external dependencies (strategy registry, memory system).
        
        Args:
            strategy_registry: StrategyRegistry instance for loading strategies
            memory_system: Memory system for cognitive processing
        """
        self.strategy_registry = strategy_registry
        self.memory_system = memory_system
    
    def simulate(self, config: TrialConfig) -> List[TrialResult]:
        """
        Simulate a full session of trials for a participant.
        
        Args:
            config: TrialConfig with participant and trial setup
            
        Returns:
            List of TrialResult objects for each trial
        """
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        results = []
        
        # Sample instance IDs for trials from the loader when available.
        all_instance_ids = self._get_available_instance_ids(config)
        selected_instances = np.random.choice(
            all_instance_ids,
            size=min(config.n_trials, len(all_instance_ids)),
            replace=False
        )
        
        # Build XAI trial schedule
        xai_schedule = self._build_xai_schedule(
            config.n_trials,
            with_xai_prob=0.5 if config.tested_with_xai else 0.0
        )
        
        logger.info(f"Starting simulation for participant {config.participant_id} "
                   f"({config.n_trials} trials, strategy: {config.strategy_name})")
        
        for trial_idx, instance_id in enumerate(selected_instances):
            trial_result = self._simulate_trial(
                config=config,
                trial_idx=trial_idx,
                instance_id=instance_id,
                with_xai=bool(xai_schedule[trial_idx])
            )
            results.append(trial_result)
        
        self.results.extend(results)
        logger.info(f"Completed {len(results)} trials")
        
        return results
    
    def _build_xai_schedule(self, n_trials: int, with_xai_prob: float = 0.5) -> np.ndarray:
        """
        Build a random schedule of which trials have explanations.
        
        Args:
            n_trials: Total number of trials
            with_xai_prob: Probability of each trial having explanations
            
        Returns:
            Boolean array indicating which trials have XAI
        """
        return np.random.binomial(1, with_xai_prob, size=n_trials).astype(bool)
    
    def _simulate_trial(self, 
                       config: TrialConfig,
                       trial_idx: int,
                       instance_id: int,
                       with_xai: bool) -> TrialResult:
        """
        Simulate a single trial.
        
        Args:
            config: Trial configuration
            trial_idx: Trial index (0-based)
            instance_id: Instance ID to present
            with_xai: Whether to show explanations
            
        Returns:
            TrialResult with response and metrics
        """
        start_time = time.time()
        
        # Load instance data
        if config.ai_dataset_loader is None:
            raise ValueError("ai_dataset_loader required")
        
        instances, ai_preds = self._load_instances(config.ai_dataset_loader, [instance_id])
        instance = instances[0]
        ai_prediction = int(ai_preds[0])
        
        # Get explainer prediction
        explainer_prediction = self._get_explainer_prediction(config, instance)
        
        # Get explanation if available and XAI is enabled
        explanation = None
        if with_xai and config.explainer is not None:
            explanation = self._apply_explainer(config.explainer, instance)
        
        # Use strategy to make prediction
        if self.strategy_registry is not None:
            # Via registry (full integration)
            prob_dist, response_time, info = self._infer_via_strategy(
                config, instance, explanation, ai_prediction, with_xai
            )
        else:
            # Fallback: simple heuristic
            prob_dist, response_time, info = self._simple_inference(
                config, ai_prediction, explainer_prediction
            )
        
        # Sample participant response from probability distribution
        try:
            participant_response = int(np.random.choice(
                [0, 1],
                p=[prob_dist.get(0, 0.5), prob_dist.get(1, 0.5)]
            ))
        except (ValueError, IndexError):
            # Fallback if probabilities invalid
            participant_response = int(np.random.choice([0, 1]))
        
        elapsed_time = time.time() - start_time + response_time
        
        # Build result
        result = TrialResult(
            participant_id=config.participant_id,
            trial_idx=trial_idx,
            instance_id=instance_id,
            tested_with_xai=with_xai,
            strategy=config.strategy_name,
            xai_type=config.xai_type,
            ai_prediction=ai_prediction,
            explainer_prediction=explainer_prediction,
            participant_response=participant_response,
            response_prob=prob_dist,
            response_time=elapsed_time,
            response_matches_ai=(participant_response == ai_prediction),
            response_matches_explainer=(participant_response == explainer_prediction),
            metadata=info
        )
        
        return result
    
    def _get_explainer_prediction(self, config: TrialConfig, instance: Any) -> int:
        """Get prediction from explanation model."""
        if config.explainer is None:
            return 0
        
        try:
            result = self._apply_explainer(config.explainer, instance)
            if isinstance(result, dict) and "class_index" in result:
                return int(result["class_index"])
            else:
                # Logistic regression style (probability -> binary)
                prob = float(result)
                return 1 if prob > 0.5 else 0
        except Exception as e:
            logger.warning(f"Failed to get explainer prediction: {e}")
            return 0
    
    def _infer_via_strategy(self, config: TrialConfig, instance: Any, 
                           explanation: Optional[Any], ai_prediction: int, 
                           with_xai: bool) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """
        Get inference from CoAX strategy via registry.
        
        Args:
            config: Trial configuration
            instance: Instance data
            explanation: Explanation object (if with_xai)
            ai_prediction: AI model prediction
            with_xai: Whether explanation is available
            
        Returns:
            (prob_dist, response_time, metadata)
        """
        if self.strategy_registry is None:
            return {0: 0.5, 1: 0.5}, 0.0, {}
        
        try:
            from src.cognitive_models import StrategyConfig, ReasoningMode, StrategyType, StrategyRegistry

            registry = self.strategy_registry or StrategyRegistry

            # Keep known config keys in StrategyConfig and pass everything else via extra_params.
            raw_params = dict(config.cognitive_params or {})
            decay_param = float(raw_params.pop('decay_param', 0.5))
            retrieval_threshold = float(raw_params.pop('retrieval_threshold', -2.5))
            sensitivity = float(raw_params.pop('sensitivity', 10.0))

            # Distribution sampling may return integer-like params as floats.
            if 'k' in raw_params:
                try:
                    raw_params['k'] = int(round(float(raw_params['k'])))
                except Exception:
                    pass
            
            # Create strategy config
            strategy_config = StrategyConfig(
                strategy_name=config.strategy_name,
                strategy_type=StrategyType.COAX_FORWARD,
                mode=ReasoningMode.READ if with_xai else ReasoningMode.RETRIEVE,
                decay_param=decay_param,
                retrieval_threshold=retrieval_threshold,
                sensitivity=sensitivity,
                extra_params=raw_params,
            )
            
            # Get or create strategy
            strategy = registry.get(config.strategy_name, strategy_config)
            
            # Infer
            prob_dist, rt, info = strategy.infer(
                features=instance,
                explanation=explanation,
                ai_prediction=ai_prediction
            )
            
            return prob_dist, rt, info or {}
        
        except Exception as e:
            logger.warning(f"Strategy inference failed: {e}, falling back to heuristic")
            return self._simple_inference(config, ai_prediction, 0)
    
    def _simple_inference(self, config: TrialConfig, ai_prediction: int, 
                         explainer_prediction: int) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """
        Simple fallback inference (no explainer dependency).
        
        Args:
            config: Trial configuration
            ai_prediction: AI model's prediction
            explainer_prediction: Explainer's prediction
            
        Returns:
            (prob_dist, response_time, metadata)
        """
        # Simple heuristic: tend to agree with AI prediction
        if ai_prediction == 1:
            prob_dist = {0: 0.3, 1: 0.7}
        else:
            prob_dist = {0: 0.7, 1: 0.3}
        
        return prob_dist, 0.5, {"inference_method": "simple_heuristic"}

    def _load_instances(self, loader: Any, instance_ids: List[int]) -> Tuple[List[Any], List[Any]]:
        """Load instances from either legacy or unified data loader APIs."""
        if hasattr(loader, 'load_instances') and callable(loader.load_instances):
            return loader.load_instances(instance_ids)
        if hasattr(loader, 'get_instances') and callable(loader.get_instances):
            return loader.get_instances(instance_ids)
        raise ValueError("ai_dataset_loader must implement load_instances() or get_instances()")

    def _apply_explainer(self, explainer: Any, instance: Any) -> Any:
        """Apply explainer using either legacy or unified explainer APIs."""
        if hasattr(explainer, 'apply_to_instance') and callable(explainer.apply_to_instance):
            return explainer.apply_to_instance(instance)
        if hasattr(explainer, 'apply') and callable(explainer.apply):
            return explainer.apply(instance)
        raise ValueError("explainer must implement apply_to_instance() or apply()")

    def _get_available_instance_ids(self, config: TrialConfig) -> List[int]:
        """Get available instance IDs from loader; fallback to legacy fixed range."""
        loader = config.ai_dataset_loader
        try:
            if loader is not None and hasattr(loader, 'get_feature_values'):
                df = loader.get_feature_values()
                if df is not None and 'instanceId' in df.columns:
                    ids = df['instanceId'].dropna().astype(int).unique().tolist()
                    if ids:
                        return ids

            if loader is not None and hasattr(loader, 'feature_values_df'):
                df = getattr(loader, 'feature_values_df')
                if df is not None and 'instanceId' in df.columns:
                    ids = df['instanceId'].dropna().astype(int).unique().tolist()
                    if ids:
                        return ids
        except Exception:
            pass

        return list(range(400))
    
    def results_to_dataframe(self, results: Optional[List[TrialResult]] = None) -> pd.DataFrame:
        """
        Convert trial results to DataFrame.
        
        Args:
            results: List of TrialResult objects (uses self.results if None)
            
        Returns:
            DataFrame with one row per trial
        """
        if results is None:
            results = self.results
        
        data = [r.to_dict() for r in results]
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_path: str, results: Optional[List[TrialResult]] = None) -> None:
        """
        Export trial results to CSV.
        
        Args:
            output_path: Path to output CSV
            results: List of TrialResult objects (uses self.results if None)
        """
        df = self.results_to_dataframe(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} trial results to {output_path}")
