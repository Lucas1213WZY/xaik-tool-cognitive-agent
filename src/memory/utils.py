"""
Utility functions for memory operations shared by both backends.
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1, higher is more similar)."""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom < 1e-10:
        return 0.0
    return float(1.0 - (np.dot(vec1, vec2) / denom) / 2.0)  # Convert to distance


def temporal_decay(time_since_encoding: float, decay_rate: float = 0.5) -> float:
    """
    Compute temporal decay weight.
    
    CoAX uses exponential decay: activation = 1 / (1 + decay_rate * time)
    
    Args:
        time_since_encoding: Time elapsed since memory encoding
        decay_rate: Decay coefficient
        
    Returns:
        Activation weight (0-1)
    """
    if time_since_encoding < 0:
        return 1.0
    return 1.0 / (1.0 + decay_rate * time_since_encoding)


def base_level_learning(reference_count: int, decay_param: float = 0.5) -> float:
    """
    Compute Base-Level Learning (BLL) activation for ACT-R.
    
    ACT-R BLL: B = ln(sum(t_i ^ -d))
    where t_i is time since i-th retrieval, d is decay parameter
    
    Args:
        reference_count: Number of times this chunk has been retrieved
        decay_param: Decay exponent (typically 0.5)
        
    Returns:
        Base-level learning activation
    """
    if reference_count <= 0:
        return 0.0
    # Simplified: Assume uniform spacing
    # Full implementation would track actual retrieval times
    return np.log(sum((i + 1) ** (-decay_param) for i in range(reference_count)))


def compute_similarity_activation(mismatch_penalty: float = 1.5) -> float:
    """
    Compute similarity-based activation component for ACT-R.
    
    S_i_c = sum(w_k * S_{i_k,c_k})
    where w_k is slot weight, S is similarity
    
    Args:
        mismatch_penalty: Penalty factor for mismatches (typically 1.5)
        
    Returns:
        Similarity activation component (can be used in full activation)
    """
    # Placeholder: actual implementation depends on slot-by-slot comparison
    return 0.0  # This is integrated into retrieve() for actual items


def add_activation_noise(base_activation: float, noise_sd: float = 0.0) -> float:
    """
    Add stochastic noise to activation (ACT-R retrieval variability).
    
    Args:
        base_activation: Base activation score
        noise_sd: Standard deviation of noise (typically 0.0 for deterministic)
        
    Returns:
        Activation with noise applied
    """
    if noise_sd > 0:
        noise = np.random.normal(0, noise_sd)
    else:
        noise = 0.0
    return base_activation + noise


def compute_retrieval_latency(activation: float, latency_factor: float = 0.0) -> float:
    """
    Compute retrieval latency from activation (ACT-R).
    
    RT = F * exp(-activation)
    where F is latency factor, typical range: 50-100ms
    
    Args:
        activation: Chunk activation
        latency_factor: Latency scaling factor (50-100ms)
        
    Returns:
        Retrieval latency in milliseconds
    """
    if latency_factor <= 0:
        return 0.0
    return latency_factor * np.exp(-activation)


def normalize_probabilities(probs: dict) -> dict:
    """Normalize a probability dictionary to sum to 1.0."""
    total = sum(probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}


def compute_chunk_similarity(chunk1_slots: dict, chunk2_slots: dict, 
                             slot_weights: Optional[dict] = None) -> float:
    """
    Compute similarity between two chunks based on slot comparison.
    
    Args:
        chunk1_slots: Slots from first chunk
        chunk2_slots: Slots from second chunk
        slot_weights: Optional weight for each slot
        
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if not chunk1_slots or not chunk2_slots:
        return 0.0
    
    common_slots = set(chunk1_slots.keys()) & set(chunk2_slots.keys())
    if not common_slots:
        return 0.0
    
    if slot_weights is None:
        slot_weights = {slot: 1.0 for slot in common_slots}
    
    total_weight = 0.0
    match_score = 0.0
    
    for slot in common_slots:
        weight = slot_weights.get(slot, 1.0)
        total_weight += weight
        
        val1, val2 = chunk1_slots[slot], chunk2_slots[slot]
        
        # Handle numeric values
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Similarity decreases with difference
            similarity = 1.0 / (1.0 + abs(val1 - val2))
        # Handle array/vector values
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            similarity = 1.0 - cosine_similarity(val1, val2)
        # Handle categorical/string values
        else:
            similarity = 1.0 if val1 == val2 else 0.0
        
        match_score += weight * similarity
    
    return match_score / total_weight if total_weight > 0 else 0.0


def get_timestamp_diff(ts1: datetime, ts2: Optional[datetime] = None) -> float:
    """Get time difference in seconds between two timestamps."""
    if ts2 is None:
        ts2 = datetime.now()
    delta = ts2 - ts1
    return delta.total_seconds()
