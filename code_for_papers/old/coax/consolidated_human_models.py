"""
================================================================================
CONSOLIDATED HUMAN MODELS & REASONING STRATEGIES
================================================================================

This module consolidates all human model classes, memory modules, and reasoning
strategies used in the xAIK cognitive agent framework.

ORGANIZATION:
1. Memory Module (MemoryBase) - Handles exemplar storage and retrieval
2. Utility Functions - Domain-specific helper functions
3. Base Classes - Foundation classes for model inheritance
4. Reasoning Strategies - Specific human behavioral models
5. Baseline Model Handlers - Generic model wrappers and orchestrators

Key features:
- Memory-based exemplar retrieval with decay (EBRW-inspired)
- Multiple reasoning strategies (sensitive features, salient features, etc.)
- Time tracking for human decision-making processes
- Support for explanation-aware and explanation-agnostic modes
- Flexible architecture for adding new strategies

Author: xAIK Team
Date: 2024-2026
"""

import math
import numpy as np
import copy
import random
import itertools


# =============================================================================
# PART 1: MEMORY MODULE
# =============================================================================

class MemoryBase:
    """
    Base memory module for storing and retrieving exemplars with temporal decay.
    
    Each exemplar is stored as:
    {
      'label_probs': {label -> probability},  # Soft-labeled distribution over classes
      'features': [...],                       # Feature vector (may contain None)
      'explanation': [... or None],            # Explanation vector (if provided)
      'time_stored': float                     # Timestamp when exemplar was stored
    }
    
    Retrieval uses an activation function inspired by EBRW (Exemplar-Based Random Walk):
        activation = -decay_param * ln(time_elapsed)
    Only exemplars with activation >= retrieval_threshold (if set) are returned.
    
    Attributes:
        _exemplars (list): Private list of stored exemplars
        decay_param (float): Temporal decay parameter (larger = faster forgetting)
        retrieval_threshold (float): Minimum activation for exemplar retrieval
    """
    
    def __init__(self, decay_param=0.5, retrieval_threshold=None):
        """
        Initialize the memory module.
        
        Args:
            decay_param (float): Controls how quickly exemplars "age" out.
                                 Higher values = faster decay.
            retrieval_threshold (float, optional): Only return exemplars with 
                                                   activation >= this threshold.
        """
        self._exemplars = []  # Made private to encapsulate internal state
        self.decay_param = decay_param
        self.retrieval_threshold = retrieval_threshold

    def store_exemplar(self, label_probs, features, explanation, time_stored):
        """
        Store a new exemplar in memory.
        
        Args:
            label_probs (dict): Label probability distribution {label -> prob}
            features (list): Feature vector for this exemplar
            explanation (list or None): Explanation vector (if available)
            time_stored (float): Timestamp of storage
        """
        ex = {
            'label_probs': copy.deepcopy(label_probs),
            'features': copy.deepcopy(features),
            'explanation': copy.deepcopy(explanation),
            'time_stored': time_stored
        }
        self._exemplars.append(ex)

    def retrieve_exemplars(self, current_time):
        """
        Retrieve all exemplars with their activations, applying temporal decay.
        
        Computation:
            time_elapsed = current_time - exemplar.time_stored
            activation = -decay_param * ln(time_elapsed)
        
        Args:
            current_time (float): Current timestamp
            
        Returns:
            list: Pairs of (exemplar_dict, activation_value) that pass the threshold.
                  If no threshold is set, all exemplars are returned.
        """
        out = []
        for ex in self._exemplars:
            # Compute time since storage, with epsilon to avoid log(0)
            time_elapsed = max(1e-12, current_time - ex['time_stored'])
            
            # EBRW activation: older exemplars have lower (more negative) activation
            activation = -self.decay_param * math.log(time_elapsed)
            
            # Apply retrieval threshold if configured
            if self.retrieval_threshold is not None and activation < self.retrieval_threshold:
                continue
                
            out.append((ex, activation))
        return out


# =============================================================================
# PART 2: UTILITY FUNCTIONS
# =============================================================================

def euclidean_distance(vec1, vec2):
    """
    Compute Euclidean distance between two vectors, robustly handling None values.
    
    Algorithm:
    1. Convert to numpy arrays, replacing None with NaN
    2. Identify valid (non-NaN) dimensions
    3. Clip differences to [-1, 1] to bound the metric
    4. Normalize by sqrt(# valid dimensions)
    
    This is useful for bounded feature spaces where we want to limit the impact
    of any single dimension.
    
    Args:
        vec1 (list): First vector (may contain None values)
        vec2 (list): Second vector (may contain None values)
        
    Returns:
        float or None: Normalized Euclidean distance, or None if no valid dims
    """
    arr1 = np.array([x if x is not None else np.nan for x in vec1])
    arr2 = np.array([x if x is not None else np.nan for x in vec2])

    # Find dimensions where both vectors have valid values
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    num_valid = np.sum(valid)
    
    if num_valid == 0:
        return None  # No valid comparison dimensions

    # Compute differences and clip to [-1, 1]
    diffs = arr1[valid] - arr2[valid]
    clipped_diffs = np.clip(diffs, -1, 1)
    
    # Normalize by sqrt(num_valid) to scale with dimensionality
    dist = np.linalg.norm(clipped_diffs)
    normalized_dist = dist / math.sqrt(num_valid)
    return normalized_dist


def compute_similarity(dist, activation, sensitivity=1.0, temperature=1.0):
    """
    Compute GCM/EBRW-style similarity between a test case and stored exemplar.
    
    Formula:
        similarity = exp((-sensitivity * dist + activation) / temperature)
    
    This combines:
    - Distance-based component (Gaussian-like falloff with sensitivity parameter)
    - Activation-based component (temporal decay bonus)
    - Temperature parameter (controls sharpness of similarity distribution)
    
    Args:
        dist (float): Euclidean distance between test and exemplar
        activation (float): Memory activation (from MemoryBase.retrieve_exemplars)
        sensitivity (float): Controls how quickly similarity decays with distance
        temperature (float): Inverse-temperature parameter for softmax-like behavior
        
    Returns:
        float: Similarity value in (0, 1] range
    """
    return math.exp((-sensitivity * dist + activation) / temperature)


def normalize_label_strengths(label_strengths):
    """
    Convert raw label strengths to a probability distribution.
    
    Ensures output is a valid probability distribution (sums to 1).
    If all strengths are near-zero, falls back to uniform distribution.
    
    Args:
        label_strengths (dict): {label -> raw_strength} mapping
        
    Returns:
        dict: {label -> probability} where probabilities sum to 1.0
    """
    total = sum(label_strengths.values())
    
    if total < 1e-12:
        # Fallback to uniform if strengths are negligible
        if len(label_strengths) == 0:
            return {}
        n = len(label_strengths)
        return {lbl: 1.0/n for lbl in label_strengths}
    else:
        # Normalize to probability distribution
        return {lbl: val / total for lbl, val in label_strengths.items()}


# =============================================================================
# PART 3: BASE CLASSES
# =============================================================================

class BaseModel:
    """
    Generic base class representing a placeholder "human model" that simulates
    how a user might respond to stimuli.
    
    This is the most minimal interface. Subclasses should override all methods
    to provide specific modeling behavior.
    
    Methods:
        new_instance() - Reset state for a new trial/instance
        infer() - Make a prediction/decision based on UI input
        feedback() - Process feedback/learning signal
    """
    
    def __init__(self):
        """Initialize the base model."""
        pass

    def new_instance(self):
        """
        Called to signal the start of a new trial/instance.
        
        Subclasses should use this to reset internal state if needed.
        """
        pass

    def infer(self, ui):
        """
        Perform inference/decision-making based on current UI state.
        
        Args:
            ui: UI object with get_value() method to retrieve:
                - 'features': Feature vector
                - 'explanation': Explanation (if provided)
                - 'ai_prediction': AI's prediction
        
        Returns:
            tuple: (response, time_used)
                - response: dict mapping labels to probabilities {0: p0, 1: p1}
                - time_used: float, time cost of this operation
        """
        # Default: uniform prediction with 0.5s cost
        return {0: 0.5, 1: 0.5}, 0.5

    def feedback(self, ui):
        """
        Process feedback/learning signal (e.g., correct label revealed).
        
        Args:
            ui: UI object with get_value() method to retrieve feedback info
        
        Returns:
            float: Time cost of processing feedback
        """
        # Default: 0.5s cost
        return 0.5


class HumanModelBase(BaseModel):
    """
    Enhanced base class with memory, reasoning, and temporal management.
    
    This class extends BaseModel with:
    - Exemplar-based memory for storing inference/feedback history
    - Time tracking for cognitive cost modeling
    - Label set tracking across all seen instances
    - Placeholder storage for last inference data
    
    Subclasses (like SensitiveFeatures, ImportanceCategorization) implement
    specific reasoning strategies by overriding infer() and feedback().
    
    Attributes:
        time: Time manager object (tracks cumulative time)
        decay_param (float): Memory decay parameter
        all_labels (set): All labels encountered during learning
        last_inference_probs (dict): Last prediction before feedback
        last_features (list): Last feature vector seen
        last_explanation (list): Last explanation vector seen
        constant_inference_time (float): Base time cost for inference
        constant_feedback_time (float): Base time cost for feedback
    """
    
    def __init__(self, time=None, decay_param=0.5):
        """
        Initialize the human model base.
        
        Args:
            time: Time manager object (assumed to have get_time() and add_time())
            decay_param (float): Memory decay parameter passed to MemoryBase
        """
        super().__init__()
        self.time = time
        self.decay_param = decay_param
        self.all_labels = set()  # Track all unique labels seen

        # Placeholders for the last inference/feedback cycle
        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        # Typical time costs (can be overridden by subclasses)
        self.constant_inference_time = 0.1
        self.constant_feedback_time = 1.0

    def new_instance(self):
        """Subclasses must implement this."""
        raise NotImplementedError

    def infer(self, ui):
        """Subclasses must implement this."""
        raise NotImplementedError

    def feedback(self, ui):
        """Subclasses must implement this."""
        raise NotImplementedError

    def _default_response(self):
        """
        Return a default response when the model has no better information.
        
        If labels have been seen, return uniform over seen labels.
        Otherwise, return uniform binary [0.5, 0.5].
        """
        if not self.all_labels:
            return {0: 0.5, 1: 0.5}
        n = len(self.all_labels)
        return {lbl: 1.0/n for lbl in self.all_labels}


# =============================================================================
# PART 4: REASONING STRATEGIES
# =============================================================================

class SensitiveFeatures(HumanModelBase):
    """
    Reasoning strategy: Focus on features that best discriminate between learned classes.
    
    This strategy:
    1. Dynamically selects k most discriminative features using a t-test metric
    2. Masks all other features as None during similarity computation
    3. Uses memory-based exemplar matching to make inferences
    4. Learns by storing hard-labeled exemplars (AI prediction as label)
    
    The t-test metric compares feature means between the two label groups,
    helping the model focus on features that distinguish different decisions.
    
    Attributes:
        memory (MemoryBase): Stores exemplars
        sensitivity (float): Parameter controlling similarity decay with distance
        k (int): Number of features to focus on
        focus_indices (list): Current subset of feature indices to attend to
        constant_time (float): Time cost constant
    """
    
    def __init__(self, time=None, decay_param=0.5,
                 retrieval_threshold=-2.5,
                 sensitivity=10.0,
                 k=3,
                 **kwargs):
        """
        Initialize SensitiveFeatures strategy.
        
        Args:
            time: Time manager
            decay_param (float): Memory decay
            retrieval_threshold (float): Memory activation threshold
            sensitivity (float): Similarity decay parameter
            k (int): Number of focus features
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(time=time, decay_param=decay_param)
        
        self.memory = MemoryBase(decay_param=decay_param, 
                                 retrieval_threshold=retrieval_threshold)
        self.sensitivity = sensitivity
        self.k = k
        self.focus_indices = []  # Will be set during infer()
        self.constant_time = 5

    def select_focus_indices(self, features):
        """
        Automatically select k most discriminative features using t-test.
        
        Algorithm:
        1. Retrieve active exemplars from memory
        2. Group exemplars by their most-probable label
        3. For each feature, compute a t-statistic comparing means between groups
        4. Select top-k features by |t-statistic|
        
        Fallback: If not exactly 2 groups or groups too small, select random k features
        
        Args:
            features (list): Current feature vector
        """
        n_features = len(features)
        n_focus = max(1, int(self.k))

        def _random_focus_indices():
            """Fallback: randomly select focus indices."""
            indices = list(range(n_features))
            random.shuffle(indices)
            self.focus_indices = indices

        # Retrieve only currently active exemplars (with memory decay)
        current_time = self.time.get_time() if self.time else 0.0
        active_exemplars = [
            ex for (ex, activation) in self.memory.retrieve_exemplars(current_time=current_time)
        ]

        # Fallback 1: no active exemplars yet
        if not active_exemplars:
            return _random_focus_indices()

        # Group exemplars by their most-probable label
        groups = {}
        for exemplar in active_exemplars:
            if exemplar['label_probs'] == {}:
                print(f"Warning: Empty label_probs in exemplar")
                print(exemplar)
                print(active_exemplars)
            label = max(exemplar['label_probs'], key=exemplar['label_probs'].get)
            groups.setdefault(label, []).append(exemplar['features'])

        # Fallback 2: not exactly 2 groups (can't do t-test)
        if len(groups) != 2:
            return _random_focus_indices()

        # Extract the two groups
        group_keys = list(groups.keys())
        group1_features = np.array(groups[group_keys[0]])  # shape: (n1, n_features)
        group2_features = np.array(groups[group_keys[1]])  # shape: (n2, n_features)
        n1 = group1_features.shape[0]
        n2 = group2_features.shape[0]

        # Fallback 3: either group too small for reliable variance estimate
        if n1 <= 1 or n2 <= 1:
            return _random_focus_indices()

        # Compute sample means for each feature in each group
        means1 = np.nanmean(group1_features, axis=0)
        means2 = np.nanmean(group2_features, axis=0)

        # Compute sample variances (unbiased estimator with ddof=1)
        vars1 = np.nanvar(group1_features, axis=0, ddof=1)
        vars2 = np.nanvar(group2_features, axis=0, ddof=1)

        # Apply lower bound to avoid division by zero
        var_eps = 1e-8
        vars1 = np.maximum(vars1, var_eps)
        vars2 = np.maximum(vars2, var_eps)

        # Compute standard error for each feature: sqrt(var1/n1 + var2/n2)
        se = np.sqrt(vars1 / n1 + vars2 / n2)
        # Extra safeguard
        se[se == 0] = 1e-12

        # Compute absolute t-statistic: |mean_diff| / SE
        t_stats = np.abs(means1 - means2) / se

        # Sort by descending t-statistic and select top-k
        sorted_indices = np.argsort(t_stats)[::-1]
        self.focus_indices = sorted_indices[:n_focus].tolist()

    def new_instance(self):
        """
        Finalize the previous trial by storing last inference as soft-labeled exemplar.
        
        This exemplar will influence future inferences through memory matching.
        """
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            if self.last_inference_probs == {}:
                print(f"Warning: Empty inference probs in SensitiveFeatures.new_instance()")
            
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                features=self.last_features,
                explanation=self.last_explanation,
                time_stored=current_time
            )
            if self.time:
                self.time.add_time(self.constant_time)
        
        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, ui):
        """
        Make inference by focusing on discriminative features.
        
        Algorithm:
        1. Select focus features via t-test
        2. Mask test vector (set non-focus features to None)
        3. Retrieve similar exemplars from memory
        4. Aggregate label probabilities via similarity weighting
        
        Args:
            ui: UI object
            
        Returns:
            tuple: (probabilities_dict, time_cost)
        """
        start_time = self.time.get_time()

        features = ui.get_value('features')
        explanation = ui.get_value('explanation')

        # Select focus indices based on learned exemplars
        self.select_focus_indices(features)
        focus_indices = self.focus_indices

        # Retrieve stored exemplars from memory
        current_time = self.time.get_time() if self.time else 0.0
        stored = self.memory.retrieve_exemplars(current_time=current_time)

        if self.time:
            self.time.add_time(self.constant_inference_time)

        # No stored exemplars → default response
        if not stored:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = self.time.get_time() - start_time
            return probs, total_time
        
        # Create masked test vector (only focus features are non-None)
        test_vec = np.array(
            [features[i] if i in focus_indices else None for i in range(len(features))],
            dtype=float
        )
        self.last_features = features

        # Accumulate weighted label probabilities
        label_strengths = {}

        for (ex, activation) in stored:
            dist = euclidean_distance(test_vec, ex["features"])
            if dist is None:
                continue

            # Similarity integrates distance and temporal activation
            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            # Weight exemplar's label distribution by similarity
            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        # Convert strengths to probabilities
        probs = normalize_label_strengths(label_strengths)
        if probs == {}:
            print("Warning: Empty probs in SensitiveFeatures.infer()")
        self.last_inference_probs = probs
        
        return probs, self.time.get_time() - start_time

    def feedback(self, ui):
        """
        Learn from feedback by storing hard-labeled exemplar.
        
        The AI's prediction becomes the label for this exemplar, creating
        a direct signal that influences future inferences.
        
        Args:
            ui: UI object with 'ai_prediction' and other values
            
        Returns:
            float: Time cost of feedback processing
        """
        features = ui.get_value('features')
        explanation = ui.get_value('explanation')
        ai_pred = ui.get_value('ai_prediction')

        # Create hard label from AI prediction
        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        # Store exemplar in memory
        self.memory.store_exemplar(
            label_probs=label_probs,
            features=features,
            explanation=explanation,
            time_stored=current_time
        )

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time


class SalientFeatures(HumanModelBase):
    """
    Reasoning strategy: Attend only to features with top-k explanation magnitude.
    
    This strategy assumes explanations are available and uses them to guide attention:
    1. Zero out all but top-k features by explanation magnitude
    2. Use masked features for memory-based similarity matching
    3. Store both masked features and explanation with exemplars
    
    This models a human who focuses on the explanation's most salient components.
    
    Attributes:
        memory (MemoryBase): Exemplar storage
        k (int): Number of salient features to attend to
        sensitivity (float): Similarity decay parameter
        constant_time (float): Time cost constant
    """

    def __init__(
        self, time=None, decay_param=0.5,
        k=3,
        sensitivity=10.0,
        retrieval_threshold=-2.5,
        **kwargs
    ):
        """
        Initialize SalientFeatures strategy.
        
        Args:
            time: Time manager
            decay_param (float): Memory decay
            k (int): Number of top salient features
            sensitivity (float): Similarity decay parameter
            retrieval_threshold (float): Memory activation threshold
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param, 
                                 retrieval_threshold=retrieval_threshold)
        self.k = k
        self.sensitivity = sensitivity
        self.constant_time = 5.0

    def _mask_top_k_features(self, features, explanation):
        """
        Zero out all but top-k features by explanation magnitude.
        
        Args:
            features (list): Original feature vector
            explanation (list or None): Explanation vector
            
        Returns:
            list: Masked feature vector (non-top-k turned to None)
        """
        if explanation is None:
            return features
        top_k_indices = np.argsort(explanation)[-self.k:]
        return [
            features[i] if i in top_k_indices else None
            for i in range(len(features))
        ]

    def new_instance(self):
        """
        Finalize previous trial by storing masked features as soft-labeled exemplar.
        """
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            top_k_features = self._mask_top_k_features(self.last_features, 
                                                        self.last_explanation)
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                features=top_k_features,
                explanation=self.last_explanation,
                time_stored=current_time
            )
            if self.time:
                self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, ui):
        """
        Infer by attending only to salient (top-k) features.
        
        Args:
            ui: UI object
            
        Returns:
            tuple: (probabilities_dict, time_cost)
        """
        start_time = self.time.get_time() if self.time else 0.0

        features = ui.get_value('features')
        explanation = ui.get_value('explanation')

        self.last_explanation = explanation

        # Mask all but top-k features by explanation magnitude
        if explanation is not None:
            top_k_indices = np.argsort(explanation)[-self.k:]
            masked_test_vec = [
                features[i] if i in top_k_indices else None
                for i in range(len(features))
            ]
        else:
            masked_test_vec = features

        self.last_features = masked_test_vec

        # Retrieve stored exemplars
        current_time = self.time.get_time() if self.time else 0.0
        stored_exemplars = self.memory.retrieve_exemplars(current_time=current_time)

        if self.time:
            self.time.add_time(self.constant_time)

        # No stored exemplars → default
        if not stored_exemplars:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = (self.time.get_time() - start_time)
            return probs, total_time

        test_vec = np.array(masked_test_vec, dtype=float)
        label_strengths = {}
        valid_exemplar_found = False

        # Compare test case to stored exemplars
        for (ex, activation) in stored_exemplars:
            # Also mask exemplar features if it has explanation
            if ex.get('explanation') is not None:
                ex_top_k_indices = np.argsort(ex['explanation'])[-self.k:]
                masked_ex_features = [
                    ex['features'][i] if i in ex_top_k_indices else None
                    for i in range(len(ex['features']))
                ]
            else:
                masked_ex_features = ex['features']

            ex_vec = np.array(masked_ex_features, dtype=float)
            dist = euclidean_distance(test_vec, ex_vec)
            if dist is None:
                continue
            valid_exemplar_found = True

            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            # Accumulate label strengths
            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        if not valid_exemplar_found:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = (self.time.get_time() - start_time)
            return probs, total_time

        probs = normalize_label_strengths(label_strengths)
        self.last_inference_probs = probs

        return probs, (self.time.get_time() - start_time)

    def feedback(self, ui):
        """
        Learn by storing masked exemplar with hard label.
        
        Args:
            ui: UI object
            
        Returns:
            float: Time cost of feedback
        """
        features = ui.get_value('features')
        explanation = ui.get_value('explanation')
        ai_pred = ui.get_value('ai_prediction')

        top_k_features = self._mask_top_k_features(features, explanation)
        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        self.memory.store_exemplar(
            label_probs=label_probs,
            features=top_k_features,
            explanation=explanation,
            time_stored=current_time
        )
        self.all_labels.add(ai_pred)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time


class ImportanceCategorization(HumanModelBase):
    """
    Reasoning strategy: Use explanation vectors (not raw features) as the basis for reasoning.
    
    This strategy:
    1. Ignores raw features; uses only explanation values for similarity
    2. Selects top-k most discriminative explanation dimensions via t-test
    3. Stores full exemplars but compares on selected dimensions only
    4. Learns by storing hard-labeled exemplars
    
    This models a human who categorizes instances based on the AI's explanation
    values rather than the underlying feature space.
    
    Attributes:
        memory (MemoryBase): Exemplar storage
        sensitivity (float): Similarity decay parameter
        k (int): Number of explanation dimensions to focus on
        constant_time (float): Time cost constant
    """

    def __init__(
        self, time=None, decay_param=0.5,
        sensitivity=10.0,
        k=3,
        retrieval_threshold=-2.5,
        **kwargs
    ):
        """
        Initialize ImportanceCategorization strategy.
        
        Args:
            time: Time manager
            decay_param (float): Memory decay
            sensitivity (float): Similarity decay parameter
            k (int): Number of explanation dimensions to focus on
            retrieval_threshold (float): Memory activation threshold
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param,
                                 retrieval_threshold=retrieval_threshold)
        self.sensitivity = sensitivity
        self.k = k
        self.constant_time = 5.0

    def select_focus_indices(self, explanation):
        """
        Select top-k explanation dimensions using t-test, with fallback to all.
        
        Algorithm:
        1. Retrieve exemplars that have explanations
        2. Group by label
        3. Compute t-statistic for each explanation dimension
        4. Select top-k by |t-statistic|
        
        If any prerequisite fails (not 2 groups, too few exemplars), 
        return all indices.
        
        Args:
            explanation (list): Current explanation vector
            
        Returns:
            list: Indices of focus dimensions
        """
        n_dims = len(explanation)
        n_focus = max(1, int(self.k))

        def _all_indices():
            """Fallback: use all dimensions."""
            return list(range(n_dims))

        current_time = self.time.get_time() if self.time else 0.0

        # Retrieve only exemplars with explanations
        retrieved = self.memory.retrieve_exemplars(current_time=current_time)
        active_exemplars = [
            ex for (ex, activation) in retrieved
            if ex.get('explanation') is not None
        ]

        # Fallback 1: no exemplars with explanation
        if not active_exemplars:
            return _all_indices()

        # Group exemplars with explanations by label
        groups = {}
        for exemplar in active_exemplars:
            label = max(exemplar['label_probs'], key=exemplar['label_probs'].get)
            groups.setdefault(label, []).append(exemplar['explanation'])

        # Fallback 2: not exactly two label groups
        if len(groups) != 2:
            return _all_indices()

        # Extract groups as numpy arrays
        labels = list(groups.keys())
        group1 = np.array(groups[labels[0]])
        group2 = np.array(groups[labels[1]])

        n1 = group1.shape[0]
        n2 = group2.shape[0]

        # Fallback 3: insufficient samples
        if n1 <= 1 or n2 <= 1:
            return _all_indices()

        # Compute means and variances
        means1 = np.mean(group1, axis=0)
        means2 = np.mean(group2, axis=0)

        vars1 = np.var(group1, axis=0, ddof=1)
        vars2 = np.var(group2, axis=0, ddof=1)

        # Compute standard error
        se = np.sqrt(vars1/n1 + vars2/n2)
        se[se == 0] = 1e-12

        # t-statistic
        t_stats = np.abs(means1 - means2) / se

        # Sort by decreasing t and select top-k
        sorted_indices = np.argsort(t_stats)[::-1]
        return sorted_indices[:n_focus].tolist()

    def new_instance(self):
        """
        Store last inference as soft-labeled exemplar.
        """
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                features=self.last_features,        # raw features
                explanation=self.last_explanation,  # explanation vector
                time_stored=current_time
            )
            if self.time:
                self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, ui):
        """
        Infer using explanation-based categorization.
        
        Args:
            ui: UI object
            
        Returns:
            tuple: (probabilities_dict, time_cost)
        """
        start_time = self.time.get_time() if self.time else 0.0

        features = ui.get_value('features')
        explanation = ui.get_value('explanation')

        self.last_features = features
        self.last_explanation = explanation

        if self.time:
            self.time.add_time(self.constant_time)

        # No explanation → default response
        if explanation is None:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)
        
        # Select focus dimensions
        focus_indices = self.select_focus_indices(explanation)
        if not focus_indices:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        # Retrieve exemplars
        current_time = self.time.get_time() if self.time else 0.0
        stored = self.memory.retrieve_exemplars(current_time=current_time)

        if not stored:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        # Compare on focus dimensions only
        test_vec = np.array([explanation[i] for i in focus_indices], dtype=float)
        label_strengths = {}
        valid_exemplar_found = False

        for ex, activation in stored:
            if ex.get('explanation') is None:
                continue

            ex_vec = np.array([ex['explanation'][i] for i in focus_indices], dtype=float)
            dist = euclidean_distance(test_vec, ex_vec)
            if dist is None:
                continue
            valid_exemplar_found = True

            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            # Accumulate label strengths
            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        if not valid_exemplar_found or not label_strengths:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        probs = normalize_label_strengths(label_strengths)
        self.last_inference_probs = probs

        return probs, (self.time.get_time() - start_time)

    def feedback(self, ui):
        """
        Learn by storing hard-labeled exemplar.
        
        Args:
            ui: UI object
            
        Returns:
            float: Time cost of feedback
        """
        features = ui.get_value('features')
        explanation = ui.get_value('explanation')
        ai_pred = ui.get_value('ai_prediction')

        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        self.memory.store_exemplar(
            label_probs=label_probs,
            features=features,
            explanation=explanation,
            time_stored=current_time
        )
        self.all_labels.add(ai_pred)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time


class AttributionSum(HumanModelBase):
    """
    Reasoning strategy: Sum top-k attribution/importance values for decision-making.
    
    This strategy has two modes:
    
    1. 'attribution' mode:
       - Ignores memory; directly sums top-k explanation values
       - Passes sum through logistic function to get binary probabilities
       - Fast, model-agnostic, explanation-dependent
    
    2. 'importance' mode:
       - Uses memory to compute "importance" for each feature
       - Derives importance from memory matches (similarity-weighted averaging)
       - Sums top-k importance values to make decisions
       - More sophisticated, learns from feedback
    
    Both modes can impute feature importance when explanation is unavailable,
    using stored exemplar information.
    
    Attributes:
        memory (MemoryBase): Exemplar storage
        sensitivity (float): Similarity decay parameter
        scaling_factor (float): Logistic scaling parameter
        k (int): Number of top features/explanations to consider
        explanation_type (str): 'importance' or 'attribution'
        constant_time (float): Time cost constant
        last_importances (list): Per-feature importance values (tracked for learning)
    """

    def __init__(
        self,
        time=None,
        decay_param=0.5,
        retrieval_threshold=-0.3,
        sensitivity=15.0,
        scaling_factor=1.0,
        k=2,
        explanation_type='importance',
        **kwargs
    ):
        """
        Initialize AttributionSum strategy.
        
        Args:
            time: Time manager
            decay_param (float): Memory decay
            retrieval_threshold (float): Memory activation threshold
            sensitivity (float): Similarity decay parameter
            scaling_factor (float): Logistic scaling parameter
            k (int): Number of top features to sum
            explanation_type (str): 'importance' or 'attribution'
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param,
                                 retrieval_threshold=retrieval_threshold)

        self.sensitivity = sensitivity
        self.scaling_factor = scaling_factor
        self.k = k
        self.explanation_type = explanation_type

        self.constant_time = 5.0

        # NEW: store per-feature importance/attribution for current instance
        self.last_importances = None

    def new_instance(self):
        """
        Store previous trial as single-feature exemplars.
        
        Each stored exemplar contains only one non-None feature dimension,
        selected from top-k by importance magnitude. This focused storage
        helps the model learn feature-label associations efficiently.
        """
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time()

            if self.last_importances is not None:
                # Select top-k by importance magnitude
                idx_val_imp = [
                    (i, self.last_features[i], self.last_importances[i])
                    for i in range(len(self.last_features))
                ]
                top_k = sorted(
                    idx_val_imp,
                    key=lambda x: abs(x[2]),
                    reverse=True
                )[:self.k]
            else:
                # Fallback: store all features if no importance info
                top_k = [
                    (i, self.last_features[i], None)
                    for i in range(len(self.last_features))
                ]

            # Store each feature as a separate single-feature exemplar
            for (i, val, imp_val) in top_k:
                feat_arr = [None] * len(self.last_features)
                feat_arr[i] = val

                # Store the full importance vector as explanation
                explanation_vector = self.last_importances

                self.memory.store_exemplar(
                    label_probs=self.last_inference_probs,
                    features=feat_arr,
                    explanation=explanation_vector,
                    time_stored=current_time
                )

            self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None
        self.last_importances = None

    def infer(self, ui):
        """
        Perform inference using attribution/importance summation.
        
        Algorithm:
        Case 1 (attribution mode with explanation):
            - Sum top-k explanation values directly
            - Apply logistic function to get probability
            
        Case 2 (importance mode with explanation):
            - For each feature, retrieve similar exemplars
            - Compute per-feature importance from label agreement
            - Sum top-k importances and apply logistic
            
        Case 3 (importance mode, no explanation):
            - Impute importance by similarity-weighted averaging
            - Over all stored exemplar features and their explanations
            - Sum importances and apply logistic
        
        Args:
            ui: UI object
            
        Returns:
            tuple: (probabilities_dict, time_cost)
        """
        start_time = self.time.get_time()

        features = ui.get_value('features')
        explanation = ui.get_value('explanation')

        self.last_features = features
        self.last_explanation = explanation
        self.last_importances = None

        if features is not None:
            # Initialize importance vector for current instance
            self.last_importances = [0.0] * len(features)

        # =====================================================================
        # CASE 1: Pure attribution mode with explanation provided
        # =====================================================================
        if self.explanation_type == 'attribution' and explanation is not None:
            # Directly sum top-k explanation values
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            total_attr = sum(explanation[i] for i in top_k_indices)

            # Store per-feature attribution values
            if self.last_importances is not None:
                for i in range(len(explanation)):
                    self.last_importances[i] = explanation[i]

            # Apply logistic to get probabilities
            prob_label_1 = 1.0 / (1.0 + math.exp(-self.scaling_factor * total_attr))
            prob_label_0 = 1.0 - prob_label_1
            probs = {0: prob_label_0, 1: prob_label_1}

            self.last_inference_probs = probs
            self.time.add_time(self.constant_time)
            total_infer_time = self.time.get_time() - start_time
            return probs, total_infer_time

        # =====================================================================
        # CASE 2 & 3: Importance mode (with or without explanation)
        # =====================================================================
        total_attribution = 0.0
        current_time = self.time.get_time()

        # CASE 2a: Explanation provided, importance mode
        if explanation is not None and self.explanation_type == 'importance':
            # Select top-k most important features by explanation magnitude
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            top_k = [(i, features[i], explanation[i]) for i in top_k_indices]

            # For each top feature, compute its importance via exemplar matching
            for (i, feat_val, exp_val) in top_k:
                stored_exemplars = self.memory.retrieve_exemplars(
                    current_time=current_time
                )

                # Accumulate label strengths for this feature
                local_strengths = {}
                for (ex, activation) in stored_exemplars:
                    if i < len(ex['features']) and ex['features'][i] is not None:
                        # Distance in feature space for dimension i
                        dist = abs(feat_val - ex['features'][i])
                        sim = math.exp(-self.sensitivity * dist + activation)
                        
                        # Accumulate exemplar's label distribution
                        for lbl, p_lbl in ex['label_probs'].items():
                            local_strengths[lbl] = local_strengths.get(lbl, 0.0) + sim * p_lbl

                if not local_strengths:
                    continue

                # Normalize to get label distribution for this feature
                local_dist = normalize_label_strengths(local_strengths)
                
                # Compute expected sign: +1 if label 1 dominant, -1 if label 0 dominant
                raw_sign = sum(
                    (+1.0 if lbl == 1 else -1.0) * p_lbl
                    for lbl, p_lbl in local_dist.items()
                )

                if abs(raw_sign) < 0.01:
                    continue  # Neutral feature (no clear label preference)

                # Scale explanation value by label agreement sign
                expected_sign = 1.0 if raw_sign > 0 else -1.0
                partial_attr = expected_sign * exp_val
                total_attribution += partial_attr

                # Store per-feature partial attribution
                if self.last_importances is not None:
                    self.last_importances[i] = partial_attr

        # CASE 2b: No explanation, impute importance from memory
        elif explanation is None and features is not None:
            # For each feature, impute its importance
            for i, feat_val in enumerate(features):
                stored_exemplars = self.memory.retrieve_exemplars(
                    current_time=current_time
                )

                sim_weighted_sum = 0.0
                total_sim = 0.0

                for ex, activation in stored_exemplars:
                    if i < len(ex['features']) and ex['features'][i] is not None:
                        ex_val = ex['features'][i]
                        
                        # Distance for this feature
                        dist = abs(feat_val - ex_val)
                        sim = math.exp(-self.sensitivity * dist + activation)

                        # Extract exemplar's label
                        label_probs = ex['label_probs']
                        label = max(label_probs, key=label_probs.get)
                        label_sign = 1.0 if label == 1 else -1.0

                        # Extract exemplar's explanation for this feature
                        explanation_vector = ex.get("explanation")
                        if explanation_vector is None or i >= len(explanation_vector):
                            continue

                        raw_exp_val = explanation_vector[i]

                        # Align explanation with label: if they disagree, flip
                        exp_val = raw_exp_val
                        if math.copysign(1, raw_exp_val) != label_sign:
                            exp_val *= -1

                        sim_weighted_sum += sim * exp_val
                        total_sim += sim

                # Compute weighted average importance
                if total_sim > 0:
                    imputed_importance = sim_weighted_sum / total_sim
                else:
                    imputed_importance = 0.0

                total_attribution += imputed_importance

                # Store imputed importance
                if self.last_importances is not None:
                    self.last_importances[i] = imputed_importance

        # Clip attribution to avoid numerical issues
        total_attribution = max(-1e3, min(1e3, total_attribution))

        # Apply logistic transformation
        prob_label_1 = 1.0 / (1.0 + math.exp(-total_attribution * self.scaling_factor))
        prob_label_0 = 1.0 - prob_label_1
        probs = {0: prob_label_0, 1: prob_label_1}
        self.last_inference_probs = probs

        self.time.add_time(self.constant_time)
        total_infer_time = self.time.get_time() - start_time
        return probs, total_infer_time

    def feedback(self, ui):
        """
        Learn from feedback by storing single-feature exemplars.
        
        Process:
        - If importance mode: all exemplars are labeled with ai_prediction
        - If attribution mode: each exemplar is labeled by sign of its explanation
        - Stores only top-k features (by importance/explanation magnitude)
        
        Args:
            ui: UI object with 'ai_prediction' and other values
            
        Returns:
            float: Time cost of feedback processing
        """
        start_time = self.time.get_time()

        features = ui.get_value('features')
        explanation = ui.get_value('explanation')
        ai_pred = ui.get_value('ai_prediction')

        # Determine global label (importance mode) or per-feature (attribution mode)
        if self.explanation_type == 'importance':
            label_probs_global = {ai_pred: 1.0}
        else:
            label_probs_global = None  # Will be per-feature

        current_time = self.time.get_time()

        # Select which features to store exemplars for
        if explanation is not None:
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            top_k = [(i, features[i], explanation[i]) for i in top_k_indices]
        else:
            top_k = [(i, features[i], None) for i in range(len(features))]

        # Store single-feature exemplar for each selected feature
        for (i, val, exp_val) in top_k:
            feat_arr = [None] * len(features)
            feat_arr[i] = val

            # Attribution mode: label by explanation sign
            if self.explanation_type == 'attribution':
                if exp_val is not None and exp_val >= 0:
                    local_label_probs = {1: 1.0}
                else:
                    local_label_probs = {0: 1.0}
            else:
                # Importance mode: use global label
                local_label_probs = label_probs_global

            self.memory.store_exemplar(
                label_probs=local_label_probs,
                features=feat_arr,
                explanation=explanation,
                time_stored=current_time
            )
            self.all_labels.update(local_label_probs.keys())

        # Reset for next instance
        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None
        self.last_importances = None

        self.time.add_time(self.constant_time)
        total_time = self.time.get_time() - start_time
        return total_time


# =============================================================================
# PART 5: BASELINE MODEL HANDLERS & ORCHESTRATORS
# =============================================================================

class BaselineModelHandler:
    """
    Generic wrapper for baseline ML models (Decision Trees, NaiveBayes, etc.).
    
    Responsibilities:
    1. Dynamically instantiate the correct model type
    2. Track training state (untrained models return uniform probabilities)
    3. Provide add_exemplar() and train() interfaces
    4. Proxy attribute access to underlying model
    5. Handle special cases (e.g., MLP periodic retraining)
    
    This allows any model type to be used with the experimental framework
    without requiring specific changes to experiment runners.
    
    Attributes:
        _model: The underlying model instance
        is_trained (bool): Whether model has been trained
        train_calls (int): Counter for MLP retraining schedule
    """

    def __init__(self, model_type='DecisionTree', **kwargs):
        """
        Initialize baseline model wrapper.
        
        Args:
            model_type (str): One of: 'DecisionTree', 'KNN', 'NaiveBayes', 'Dummy',
                                      'GCM', 'LogisticRegression', 'MLP', 'Random'
            **kwargs: Hyperparameters passed to the model constructor
        """
        # Dynamically instantiate the appropriate model class
        if model_type == 'DecisionTree':
            self._model = DecisionTreeModel(**kwargs)
        elif model_type == "KNN":
            self._model = KNNModel(**kwargs)
        elif model_type == 'NaiveBayes':
            self._model = NaiveBayesModel(**kwargs)
        elif model_type == 'Dummy':
            self._model = DummyModel(**kwargs)
        elif model_type == 'GCM':
            self._model = GeneralizedContextModel(**kwargs)
        elif model_type == 'LogisticRegression':
            self._model = LogisticRegressionModel(**kwargs)
        elif model_type == 'MLP':
            self._model = MLPModel(**kwargs)
            self.train_calls = 0  # Track calls for periodic retraining
        elif model_type == 'Random':
            self._model = RandomModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: '{model_type}'.")

        self.is_trained = False  # Track training state

    def add_exemplar(self, features, label):
        """
        Add training example to the model.
        
        Args:
            features (list or array): Feature vector
            label (int): Class label (0 or 1)
        """
        self._model.add_exemplar(features, label)

    def train(self):
        """
        Train the model if it has a train() method.
        
        Special handling:
        - MLP: Retrain only after every 3rd call (to avoid excessive retraining)
        - Other models: Train immediately
        """
        if hasattr(self._model, 'train'):
            if isinstance(self._model, MLPModel):
                if not self.is_trained:
                    self._model.train()
                    self.is_trained = True
                else:
                    self.train_calls += 1
                    if self.train_calls % 3 == 0:
                        self._model.train()
            else:
                self._model.train()
                self.is_trained = True
        else:
            # Model without train() is always "trained"
            self.is_trained = True

    def infer(self, features, actual_ai_prediction=None):
        """
        Predict probabilities for given features.
        
        Returns uniform [0.5, 0.5] if model hasn't been trained yet.
        
        Args:
            features (list or array): Feature vector to predict
            actual_ai_prediction (optional): AI prediction (for certain model types)
            
        Returns:
            dict: {label -> probability} for each class
        """
        if not self.is_trained:
            # Untrained model: uninformed guess
            return {0: 0.5, 1: 0.5}

        # Delegate to underlying model
        if isinstance(self._model, DummyModel):
            return self._model.infer(features, actual_ai_prediction)
        return self._model.infer(features)

    def __getattr__(self, name):
        """
        Proxy attribute/method access to underlying model.
        
        This allows direct access to model-specific attributes without
        explicitly forwarding every possible method.
        
        Args:
            name (str): Attribute or method name
            
        Returns:
            The requested attribute from the underlying model
        """
        if hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ExplanationAwareHumanModel:
    """
    Orchestrator for two parallel baseline models: one with, one without explanation.
    
    This wrapper allows study of how explanation affects human decision-making
    by maintaining two separate models that learn independently:
    1. model_no_exp: Used when no explanation is shown
    2. model_with_exp: Used when explanation is shown (features + explanation concatenated)
    
    During feedback, only the model(s) actually used during inference are trained,
    allowing for differential learning rates and representations.
    
    Attributes:
        model_type (str): Type of baseline models to use
        model_kwargs (dict): Hyperparameters for both models
        model_no_exp (BaselineModelHandler): Model trained without explanations
        model_with_exp (BaselineModelHandler): Model trained with explanations
        last_inference_no_exp_features (list or None): Last input to model_no_exp
        last_inference_with_exp_features (list or None): Last input to model_with_exp
    """

    def __init__(self, model_type='DecisionTree', **kwargs):
        """
        Initialize explanation-aware model wrapper.
        
        Creates two identical models of the specified type with identical hyperparameters.
        
        Args:
            model_type (str): Model type (same as BaselineModelHandler)
            **kwargs: Hyperparameters passed to both models
        """
        self.model_type = model_type
        self.model_kwargs = kwargs or {}

        # Create two independent models with same type/hyperparameters
        self.model_no_exp = BaselineModelHandler(model_type=self.model_type, 
                                                **self.model_kwargs)
        self.model_with_exp = BaselineModelHandler(model_type=self.model_type, 
                                                  **self.model_kwargs)

        # Track which models were used in last inference (for selective training)
        self.last_inference_no_exp_features = None
        self.last_inference_with_exp_features = None

    def new_instance(self):
        """
        Reset model state for new trial.
        
        Called between trials to reset tracking of which models were used.
        """
        self.last_inference_no_exp_features = None
        self.last_inference_with_exp_features = None

    def infer(self, ui):
        """
        Perform inference, choosing model based on explanation availability.
        
        Logic:
        - If explanation is shown: concatenate features + explanation, use model_with_exp
        - If no explanation: use raw features, use model_no_exp
        - Track which model was used for selective training during feedback
        
        Args:
            ui: UI object with get_value() for 'features', 'explanation', 'ai_prediction'
            
        Returns:
            tuple: (probabilities_dict, time_cost)
        """
        # Retrieve inputs from UI
        feature_values = ui.get_value('features') or []
        explanation = ui.get_value('explanation')
        has_explanation = (explanation is not None)

        # Choose model and input based on explanation
        if has_explanation:
            combined_input = list(feature_values) + list(explanation)
            prediction_dict = self.model_with_exp.infer(combined_input, 
                                                       actual_ai_prediction=ui.get_value('ai_prediction'))
            self.last_inference_with_exp_features = combined_input
        else:
            prediction_dict = self.model_no_exp.infer(feature_values, 
                                                     actual_ai_prediction=ui.get_value('ai_prediction'))
            self.last_inference_no_exp_features = feature_values

        # Time cost
        time_used = 0.5
        return prediction_dict, time_used

    def feedback(self, ui):
        """
        Train only the model(s) that were actually used during inference.
        
        This selective training allows comparison of how explanation affects learning.
        
        Args:
            ui: UI object with get_value() for 'ai_prediction' (correct label)
            
        Returns:
            float: Time cost of feedback processing
        """
        # Extract correct label from UI
        label = ui.get_value('ai_prediction')
        if label is None:
            return 0.1  # Can't train without label

        # Train model_no_exp if it was used in inference
        if self.last_inference_no_exp_features is not None:
            self.model_no_exp.add_exemplar(self.last_inference_no_exp_features, label)
            self.model_no_exp.train()

        # Train model_with_exp if it was used in inference
        if self.last_inference_with_exp_features is not None:
            self.model_with_exp.add_exemplar(self.last_inference_with_exp_features, label)
            self.model_with_exp.train()

        # Time cost
        return 0.2


class RandomModel:
    """
    Baseline model that always returns uniform probabilities.
    
    This model ignores all features and explanations, serving as a control
    baseline to measure the effect of the actual reasoning strategies.
    
    Does not learn from exemplars (all methods are no-ops).
    """
    
    def __init__(self, **kwargs):
        """Initialize the random model (no-op)."""
        pass

    def add_exemplar(self, features, label):
        """
        No-op: Random model does not learn.
        """
        pass

    def train(self):
        """
        No-op: Random model does not have a training phase.
        """
        pass

    def infer(self, features):
        """
        Return uniform probability distribution.
        
        Args:
            features (list or array): Ignored
            
        Returns:
            dict: Always {0: 0.5, 1: 0.5}
        """
        return {0: 0.5, 1: 0.5}
