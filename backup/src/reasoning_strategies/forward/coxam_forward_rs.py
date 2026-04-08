"""
CoXAM Forward Reasoning Strategies

Implements memory-based reasoning strategies using the ACTR/Cognitive model.
These strategies use the unified memory module with ACT-R backend for
probabilistic inference based on memory activation, spreading activation, and
dynamic feature processing.

Strategies:
- LRCalculation: Monte Carlo sampling from memory with evidence model
- LRHeuristic: Probabilistic LR with simplified activation
- DTTraversal: Stochastic decision tree path following
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math

from src.memory import (
    UnifiedMemory,
    MemoryConfig,
    Exemplar,
    euclidean_distance,
    normalize_probabilities,
)
from ..interface import ReasoningStrategy, StrategyConfig, StrategyMetadata, StrategyType, ReasoningMode


class LRCalculation(ReasoningStrategy):
    """
    Reasoning strategy: Likelihood Ratio calculation via memory sampling.
    
    Algorithm:
    1. Sample exemplars from memory weighted by activation
    2. Compute likelihood ratios for each sampled exemplar
    3. Average LRs to get final probability via evidence model
    
    Supports RETRIEVE mode (memory-based) and READ mode (calculation-only).
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="lr_calculation",
            display_name="LR Calculation (CoXAM)",
            strategy_type=StrategyType.COXAM_FORWARD,
            description="Memory-based likelihood ratio via Monte Carlo sampling",
            category="CoXAM",
            parameters={
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)},
                'activation_noise': {'default': 0.1, 'range': (0.0, 0.5)},
                'noise_variance': {'default': 0.02, 'range': (0.0, 0.1)},
                'ddm_v': {'default': 1.0, 'range': (0.5, 3.0)},
                'ddm_a': {'default': 0.5, 'range': (0.3, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize LRCalculation strategy."""
        self.config = config
        extra = config.extra_params or {}
        
        # Memory parameters
        self.activation_noise = extra.get('activation_noise', 0.1)
        self.noise_variance = extra.get('noise_variance', 0.02)
        
        # Evidence model parameters (DDM)
        self.ddm_v = extra.get('ddm_v', 1.0)  # drift rate
        self.ddm_a = extra.get('ddm_a', 0.5)  # boundary
        
        # Initialize memory with ACT-R backend
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_features = None
        self.reasoning_mode = extra.get('reasoning_mode', ReasoningMode.RETRIEVE)
    
    def new_instance(self):
        """Finalize previous trial by storing as exemplar."""
        if self.last_inference_probs is not None and self.last_features is not None:
            label = 1 if self.last_inference_probs.get(1, 0) > 0.5 else 0
            exemplar = Exemplar(
                label=label,
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def _feature_distance(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute similarity-based distance with partial matching."""
        mismatch_penalty = 1.0
        penalties = 0.0
        matches = 0
        
        for i in range(min(len(f1), len(f2))):
            if f1[i] is None or f2[i] is None:
                continue  # Skip None values
            
            if abs(f1[i] - f2[i]) > 0.5:  # Mismatch threshold
                penalties += mismatch_penalty
            else:
                matches += 1
        
        return penalties / (matches + penalties + 1e-10)
    
    def _lr_for_exemplar(self, features: np.ndarray, exemplar: Exemplar) -> float:
        """Compute likelihood ratio for an exemplar."""
        distance = self._feature_distance(features, exemplar.features)
        
        # Similarity is inverse of distance
        similarity = np.exp(-distance + np.random.normal(0, self.noise_variance))
        
        # LR depends on predicted label
        if exemplar.label == 1:
            return similarity
        else:
            return 1.0 / (similarity + 1e-10)
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference via MC sampling and LR averaging."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        self.last_features = feature_array
        
        # Sample exemplars from memory
        exemplar_backend = self.memory.get_exemplar_memory()
        if exemplar_backend:
            exemplars_dict = exemplar_backend.get_exemplars()
        else:
            exemplars_dict = {}
        
        if not exemplars_dict:
            # No exemplars: return uniform
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            exemplars_list = list(exemplars_dict.values())
            
            # MC sampling: sample with replacement
            n_samples = min(10, len(exemplars_list))
            sampled_indices = np.random.choice(len(exemplars_list), n_samples, replace=True)
            
            # Compute LRs
            lrs = []
            for idx in sampled_indices:
                exemplar = exemplars_list[idx]
                lr = self._lr_for_exemplar(feature_array, exemplar)
                lrs.append(lr)
            
            # Average LR
            mean_lr = np.mean(lrs)
            
            # Convert to probability via evidence model (DDM)
            evidence = self.ddm_v * mean_lr
            try:
                p_pos = 1.0 / (1.0 + np.exp(-evidence))
            except:
                p_pos = 0.5
            
            self.last_inference_probs = {0: 1.0 - p_pos, 1: p_pos}
        
        return self.last_inference_probs, 0.2, {
            'exemplars_sampled': len(exemplars_dict),
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
        return 0.3
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }


class LRHeuristic(ReasoningStrategy):
    """
    Reasoning strategy: Simplified LR heuristic using direct memory matching.
    
    Algorithm:
    1. Find closest exemplar to current features
    2. Use logistic model: p = 1 / (1 + exp(-k * distance))
    3. Fast, requires fewer memory accesses
    
    Supports HEURISTIC mode with reduced computational cost.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="lr_heuristic",
            display_name="LR Heuristic (CoXAM)",
            strategy_type=StrategyType.COXAM_FORWARD,
            description="Simplified LR heuristic using nearest exemplar",
            category="CoXAM",
            parameters={
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)},
                'scaling_factor': {'default': 2.0, 'range': (0.5, 5.0)},
                'use_label_bias': {'default': False, 'type': 'boolean'}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize LRHeuristic strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.scaling_factor = extra.get('scaling_factor', 2.0)
        self.use_label_bias = extra.get('use_label_bias', False)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_features = None
    
    def new_instance(self):
        """Finalize previous trial."""
        if self.last_inference_probs is not None and self.last_features is not None:
            label = 1 if self.last_inference_probs.get(1, 0) > 0.5 else 0
            exemplar = Exemplar(
                label=label,
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference via nearest exemplar."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        self.last_features = feature_array
        
        exemplar_backend = self.memory.get_exemplar_memory()
        exemplars_dict = exemplar_backend.get_exemplars() if exemplar_backend else {}
        
        if not exemplars_dict:
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            exemplars_list = list(exemplars_dict.values())
            
            # Find nearest exemplar
            min_distance = float('inf')
            nearest_label = 0
            
            for ex in exemplars_list:
                dist = euclidean_distance(feature_array, ex.features)
                if dist < min_distance:
                    min_distance = dist
                    nearest_label = ex.label
            
            # Apply logistic
            decision_value = -self.scaling_factor * min_distance
            if nearest_label == 0:
                decision_value = -decision_value
            
            try:
                p_pos = 1.0 / (1.0 + np.exp(-decision_value))
            except:
                p_pos = 0.5
            
            self.last_inference_probs = {0: 1.0 - p_pos, 1: p_pos}
        
        return self.last_inference_probs, 0.05, {
            'exemplars_count': len(exemplars_dict),
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
        return 0.1
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }


class DTTraversal(ReasoningStrategy):
    """
    Reasoning strategy: Stochastic decision tree traversal using memory features.
    
    Algorithm:
    1. Retrieve exemplars that match current features
    2. Use exemplar labels to build implicit decision tree
    3. Traverse tree based on feature thresholds
    4. Aggregate probabilities from leaf predictions
    
    Supports both RETRIEVE and HEURISTIC modes.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="dt_traversal",
            display_name="DT Traversal (CoXAM)",
            strategy_type=StrategyType.COXAM_FORWARD,
            description="Stochastic decision tree path following via memory",
            category="CoXAM",
            parameters={
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)},
                'threshold_percentile': {'default': 50, 'range': (25, 75)},
                'max_depth': {'default': 5, 'range': (2, 10)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize DTTraversal strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.threshold_percentile = extra.get('threshold_percentile', 50)
        self.max_depth = extra.get('max_depth', 5)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_features = None
    
    def new_instance(self):
        """Finalize previous trial."""
        if self.last_inference_probs is not None and self.last_features is not None:
            label = 1 if self.last_inference_probs.get(1, 0) > 0.5 else 0
            exemplar = Exemplar(
                label=label,
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def _traverse_tree(self, features: np.ndarray, exemplars: List[Exemplar], depth: int = 0) -> Dict[int, float]:
        """Recursively traverse implicit decision tree."""
        
        if depth >= self.max_depth or not exemplars:
            # Leaf: aggregate labels from exemplars
            label_counts = {0: 0.0, 1: 0.0}
            for ex in exemplars:
                label_counts[ex.label] += 1.0
            
            if sum(label_counts.values()) == 0:
                return {0: 0.5, 1: 0.5}
            
            total = sum(label_counts.values())
            return {lbl: cnt/total for lbl, cnt in label_counts.items()}
        
        # Find best split feature
        n_features = len(features)
        best_feature = np.random.randint(0, n_features)  # Random feature for now
        
        # Compute threshold
        feature_values = np.array([ex.features[best_feature] for ex in exemplars if best_feature < len(ex.features)])
        if len(feature_values) == 0:
            return {0: 0.5, 1: 0.5}
        
        threshold = np.percentile(feature_values, self.threshold_percentile)
        
        # Split exemplars
        left = []
        right = []
        for ex in exemplars:
            if best_feature < len(ex.features):
                if ex.features[best_feature] <= threshold:
                    left.append(ex)
                else:
                    right.append(ex)
            else:
                left.append(ex)
        
        if not left:
            left = exemplars
        if not right:
            right = exemplars
        
        # Recursive split based on current feature
        if best_feature < len(features):
            if features[best_feature] <= threshold:
                return self._traverse_tree(features, left, depth + 1)
            else:
                return self._traverse_tree(features, right, depth + 1)
        else:
            return {0: 0.5, 1: 0.5}
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference via stochastic DT traversal."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        self.last_features = feature_array
        
        exemplar_backend = self.memory.get_exemplar_memory()
        exemplars_dict = exemplar_backend.get_exemplars() if exemplar_backend else {}
        
        if not exemplars_dict:
            self.last_inference_probs = {0: 0.5, 1: 0.5}
        else:
            exemplars_list = list(exemplars_dict.values())
            self.last_inference_probs = self._traverse_tree(feature_array, exemplars_list)
        
        return self.last_inference_probs, 0.15, {
            'exemplars_used': len(exemplars_dict),
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
        return 0.2
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }
