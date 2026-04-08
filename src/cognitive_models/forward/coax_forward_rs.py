"""
CoAX Forward Reasoning Strategies

Implements exemplar-based reasoning strategies inspired by the GCM/EBRW model.
These strategies use the unified memory module with exemplar backend for
probabilistic inference based on feature similarity and memory activation.

Strategies:
- SensitiveFeatures: Focus on discriminative features via t-test
- SalientFeatures: Focus on high-magnitude explanation components
- ImportanceCategorization: Use explanation vectors for categorization
- AttributionSum: Sum attribution/importance for binary decisions
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math
from datetime import datetime

from src.cognitive_models.memory import (
    UnifiedMemory,
    MemoryConfig,
    Exemplar,
    euclidean_distance,
    temporal_decay,
    normalize_probabilities,
)
from ..interface import ReasoningStrategy, StrategyConfig, StrategyMetadata, StrategyType


class SensitiveFeatures(ReasoningStrategy):
    """
    Reasoning strategy: Focus on features that best discriminate between learned classes.
    
    Algorithm:
    1. Dynamically selects k most discriminative features using t-test metric
    2. Masks non-focus features as None during similarity computation
    3. Retrieves similar exemplars from memory
    4. Aggregates label probabilities via similarity weighting
    
    The t-test metric compares feature means between the two label groups,
    helping the model focus on features that distinguish different decisions.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="sensitive_features",
            display_name="Sensitive Features (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Focus on discriminative features identified via t-test",
            category="CoAX",
            parameters={
                'sensitivity': {'default': 10.0, 'range': (1.0, 100.0)},
                'k': {'default': 3, 'range': (1, 10)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize SensitiveFeatures strategy."""
        self.config = config
        
        # Extract strategy-specific parameters
        extra = config.extra_params or {}
        self.sensitivity = extra.get('sensitivity', 10.0)
        self.k = extra.get('k', 3)
        
        # Initialize memory with exemplar backend
        mem_config = MemoryConfig.coax_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        # Time tracking
        self.time = config.time_manager
        
        # Trial state
        self.last_inference_probs = None
        self.last_features = None
        self.focus_indices = []
    
    def new_instance(self):
        """Finalize previous trial by storing last inference as exemplar."""
        if self.last_inference_probs is not None and self.last_features is not None:
            exemplar = Exemplar(
                label=np.argmax(list(self.last_inference_probs.values())),
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def _select_focus_indices(self, features: np.ndarray) -> List[int]:
        """
        Select k most discriminative features using t-test metric.
        
        Returns list of feature indices to focus on.
        """
        n_features = len(features)
        n_focus = max(1, min(self.k, n_features))
        
        # Get exemplars from memory backend
        exemplar_backend = self.memory.get_exemplar_memory()
        if not exemplar_backend:
            return list(range(n_focus))  # Default: first k features
        
        exemplars_dict = exemplar_backend.get_exemplars()
        if not exemplars_dict:
            return list(range(n_focus))  # Default: first k features
        
        exemplars_list = list(exemplars_dict.values())
        
        # Group exemplars by label
        groups = {}
        for ex in exemplars_list:
            label = ex.label
            if label not in groups:
                groups[label] = []
            groups[label].append(ex.features)
        
        # Need exactly 2 groups for t-test
        if len(groups) != 2:
            return list(range(n_focus))  # Default fallback
        
        # Get features for each group
        group_keys = list(groups.keys())
        group1 = np.array(groups[group_keys[0]])  # shape: (n1, n_features)
        group2 = np.array(groups[group_keys[1]])  # shape: (n2, n_features)
        
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return list(range(n_focus))
        
        # Compute t-statistics for each feature
        means1 = np.mean(group1, axis=0)
        means2 = np.mean(group2, axis=0)
        vars1 = np.var(group1, axis=0, ddof=1)
        vars2 = np.var(group2, axis=0, ddof=1)
        
        # Standard error
        se = np.sqrt(vars1 / n1 + vars2 / n2 + 1e-10)
        t_stats = np.abs(means1 - means2) / se
        
        # Select top-k by t-statistic
        sorted_indices = np.argsort(t_stats)[::-1]
        return sorted_indices[:n_focus].tolist()
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """
        Make inference by focusing on discriminative features.
        
        Returns (probabilities, time_cost, info)
        """
        # Convert features if needed
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        start_time = self.time.get_time() if self.time else 0.0
        
        # Select focus indices
        self.focus_indices = self._select_focus_indices(feature_array)
        
        # Create masked query vector
        query_features = np.array([
            feature_array[i] if i in self.focus_indices else None
            for i in range(len(feature_array))
        ])
        
        self.last_features = feature_array
        
        # Retrieve similar exemplars
        exemplar_query = Exemplar(
            label=0,
            features=query_features,
            label_probs={},
            explanation_vector=np.array([])
        )
        
        results = self.memory.retrieve(exemplar_query, k=5)
        
        if not results:
            # Default: uniform
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            # Aggregate probabilities from retrieved exemplars
            # results is List[Tuple[key, activation_score, exemplar]]
            label_strengths = {}
            for key, activation_score, exemplar in results:
                # Compute similarity based on distance
                dist = euclidean_distance(query_features, exemplar.features)
                similarity = np.exp(-self.sensitivity * dist) * (1.0 + activation_score)
                
                # Add to label strengths
                for label, prob in exemplar.label_probs.items():
                    label_strengths[label] = label_strengths.get(label, 0.0) + similarity * prob
            
            # Normalize to probabilities
            if label_strengths:
                self.last_inference_probs = normalize_probabilities(label_strengths)
            else:
                self.last_inference_probs = {0: 0.5, 1: 0.5}
        
        end_time = self.time.get_time() if self.time else 0.0
        time_cost = max(0.1, end_time - start_time)  # At least 0.1s
        
        info = {
            'focus_features': self.focus_indices,
            'exemplars_retrieved': len(results),
            'memory_size': self.memory.get_size()
        }
        
        return self.last_inference_probs, time_cost, info
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing hard-labeled exemplar."""
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        exemplar = Exemplar(
            label=true_label,
            features=feature_array,
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
        
        return 0.5  # Time cost
    
    def get_state(self) -> Dict[str, Any]:
        """Export state for debugging."""
        return {
            'memory_size': self.memory.get_size(),
            'focus_indices': self.focus_indices,
            'exemplars_count': len(self.memory.get_exemplars())
        }


class SalientFeatures(ReasoningStrategy):
    """
    Reasoning strategy: Attend only to features with top-k explanation magnitude.
    
    Assumes explanations are available. Uses explanation magnitude to guide attention.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="salient_features",
            display_name="Salient Features (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Focus on features with highest explanation magnitude",
            category="CoAX",
            parameters={
                'sensitivity': {'default': 10.0, 'range': (1.0, 100.0)},
                'k': {'default': 3, 'range': (1, 10)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize SalientFeatures strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity = extra.get('sensitivity', 10.0)
        self.k = extra.get('k', 3)
        
        mem_config = MemoryConfig.coax_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_features = None
    
    def new_instance(self):
        """Finalize previous trial."""
        if self.last_inference_probs is not None and self.last_features is not None:
            exemplar = Exemplar(
                label=np.argmax(list(self.last_inference_probs.values())),
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference focusing on salient (high-magnitude) features."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        # Select top-k features by explanation magnitude
        if explanation is not None:
            exp_array = np.array(explanation)
            top_indices = np.argsort(np.abs(exp_array))[-self.k:].tolist()
        else:
            top_indices = list(range(min(self.k, len(feature_array))))
        
        # Mask non-top features
        query_features = np.array([
            feature_array[i] if i in top_indices else None
            for i in range(len(feature_array))
        ])
        
        self.last_features = feature_array
        
        # Retrieve exemplars
        exemplar_query = Exemplar(
            label=0,
            features=query_features,
            label_probs={},
            explanation_vector=np.array([])
        )
        
        results = self.memory.retrieve(exemplar_query, k=5)
        
        if not results:
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            label_strengths = {}
            for key, activation_score, exemplar in results:
                dist = euclidean_distance(query_features, exemplar.features)
                similarity = np.exp(-self.sensitivity * dist) * (1.0 + activation_score)
                for label, prob in exemplar.label_probs.items():
                    label_strengths[label] = label_strengths.get(label, 0.0) + similarity * prob
            
            if label_strengths:
                self.last_inference_probs = normalize_probabilities(label_strengths)
            else:
                self.last_inference_probs = {0: 0.5, 1: 0.5}
        
        return self.last_inference_probs, 0.1, {
            'salient_features': top_indices,
            'exemplars_retrieved': len(results),
            'memory_size': self.memory.get_size()
        }
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing exemplar."""
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        exemplar = Exemplar(
            label=true_label,
            features=feature_array,
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
        return 0.5
    
    def get_state(self) -> Dict[str, Any]:
        return {'memory_size': self.memory.get_size(), 'exemplars_count': len(self.memory.get_exemplars())}


class ImportanceCategorization(ReasoningStrategy):
    """
    Reasoning strategy: Use explanation vectors (not raw features) for categorization.
    
    Selects top-k most discriminative explanation dimensions via t-test.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="importance_categorization",
            display_name="Importance Categorization (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Categorize based on explanation dimensions using t-test",
            category="CoAX",
            parameters={
                'sensitivity': {'default': 10.0, 'range': (1.0, 100.0)},
                'k': {'default': 3, 'range': (1, 10)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize ImportanceCategorization strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.sensitivity = extra.get('sensitivity', 10.0)
        self.k = extra.get('k', 3)
        
        mem_config = MemoryConfig.coax_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_explanation = None
    
    def new_instance(self):
        """Finalize previous trial."""
        if self.last_inference_probs is not None and self.last_explanation is not None:
            exemplar = Exemplar(
                label=np.argmax(list(self.last_inference_probs.values())),
                features=self.last_explanation,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_explanation = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference using explanation vectors."""
        
        if explanation is None:
            explanation = np.zeros(len(features) if isinstance(features, (list, dict)) else len(features))
        
        explanation = np.array(explanation)
        self.last_explanation = explanation
        
        # Query on explanation
        query = Exemplar(
            label=0,
            features=explanation,
            label_probs={},
            explanation_vector=np.array([])
        )
        
        results = self.memory.retrieve(query, k=5)
        
        if not results:
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            label_strengths = {}
            for key, activation_score, exemplar in results:
                dist = euclidean_distance(explanation, exemplar.features)
                similarity = np.exp(-self.sensitivity * dist) * (1.0 + activation_score)
                for label, prob in exemplar.label_probs.items():
                    label_strengths[label] = label_strengths.get(label, 0.0) + similarity * prob
            
            if label_strengths:
                self.last_inference_probs = normalize_probabilities(label_strengths)
            else:
                self.last_inference_probs = {0: 0.5, 1: 0.5}
        
        return self.last_inference_probs, 0.1, {'exemplars_retrieved': len(results), 'memory_size': self.memory.get_size()}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing exemplar."""
        if explanation is None:
            explanation = np.zeros(len(features) if isinstance(features, (list, dict)) else 10)
        
        exemplar = Exemplar(
            label=true_label,
            features=np.array(explanation),
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_explanation = None
        return 0.5
    
    def get_state(self) -> Dict[str, Any]:
        return {'memory_size': self.memory.get_size()}


class AttributionSum(ReasoningStrategy):
    """
    Reasoning strategy: Sum top-k attribution values for decision-making.
    
    Uses explanation magnitude to select top features, sums them,
    and passes through logistic function.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="attribution_sum",
            display_name="Attribution Sum (CoAX)",
            strategy_type=StrategyType.COAX_FORWARD,
            description="Sum top-k attribution values for decision",
            category="CoAX",
            parameters={
                'scaling_factor': {'default': 1.0, 'range': (0.1, 10.0)},
                'k': {'default': 2, 'range': (1, 10)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize AttributionSum strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.scaling_factor = extra.get('scaling_factor', 1.0)
        self.k = extra.get('k', 2)
        
        mem_config = MemoryConfig.coax_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
    
    def new_instance(self):
        """Finalize previous trial."""
        self.last_inference_probs = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference by summing top-k attributions."""
        
        if explanation is None:
            explanation = np.zeros(10)
        
        explanation = np.array(explanation)
        
        # Sum top-k attributions
        top_k_sum = np.sum(np.sort(np.abs(explanation))[-self.k:])
        
        # Apply logistic
        evidence = self.scaling_factor * top_k_sum
        p_pos = 1.0 / (1.0 + np.exp(-evidence))
        
        self.last_inference_probs = {0: 1.0 - p_pos, 1: p_pos}
        
        return self.last_inference_probs, 0.05, {
            'top_k_sum': float(top_k_sum),
            'evidence': float(evidence)
        }
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """No learning in attribution sum strategy."""
        return 0.1
    
    def get_state(self) -> Dict[str, Any]:
        return {'no_memory': True}
