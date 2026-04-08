"""
User Simulation Utilities - Helper functions for human-like data generation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_probabilities(probs: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize probability distribution to sum to 1.
    
    Args:
        probs: Dict mapping class (0, 1) to probability
        
    Returns:
        Normalized probability dict
    """
    total = sum(probs.values())
    if total <= 0:
        return {0: 0.5, 1: 0.5}  # Default uniform
    return {k: v / total for k, v in probs.items()}


def apply_lapse_rate(prob_dist: Dict[int, float], lapse: float) -> Dict[int, float]:
    """
    Apply lapse rate (random guessing) to probability distribution.
    
    Args:
        prob_dist: Original probability dict
        lapse: Lapse rate (probability of random response)
        
    Returns:
        Modified probability dict with lapse applied
    """
    if lapse <= 0 or lapse >= 1:
        return prob_dist
    
    # Blend with uniform distribution
    uniform = {0: 0.5, 1: 0.5}
    blended = {
        k: (1 - lapse) * prob_dist.get(k, 0) + lapse * uniform[k]
        for k in [0, 1]
    }
    return normalize_probabilities(blended)


def add_response_noise(response: int, noise_rate: float) -> int:
    """
    Add noise to participant response (flip with probability).
    
    Args:
        response: Original response (0 or 1)
        noise_rate: Probability of flipping response
        
    Returns:
        Potentially flipped response
    """
    if np.random.random() < noise_rate:
        return 1 - response
    return response


def add_response_time_jitter(response_time: float, jitter_std: float) -> float:
    """
    Add jitter to response time (realistic variation).
    
    Args:
        response_time: Base response time (seconds)
        jitter_std: Standard deviation of jitter
        
    Returns:
        Jittered response time (minimum 0.1s)
    """
    jitter = np.random.normal(0, jitter_std)
    return max(0.1, response_time + jitter)


def create_trial_schedule(n_trials: int, 
                         n_xai_trials: int,
                         randomize: bool = True) -> np.ndarray:
    """
    Create a schedule of which trials show explanations.
    
    Args:
        n_trials: Total number of trials
        n_xai_trials: Number of trials with XAI
        randomize: Whether to randomize position of XAI trials
        
    Returns:
        Boolean array (True = with XAI, False = without XAI)
    """
    schedule = np.array([True] * n_xai_trials + [False] * (n_trials - n_xai_trials))
    
    if randomize:
        np.random.shuffle(schedule)
    
    return schedule


def stratify_instances(n_total: int, n_select: int, 
                      stratification_key: Optional[List[int]] = None,
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Select instances with stratification (balanced sampling).
    
    Args:
        n_total: Total number of available instances
        n_select: Number of instances to select
        stratification_key: Stratum assignments (e.g., class labels for balance)
        seed: Random seed
        
    Returns:
        Indices of selected instances
    """
    if seed is not None:
        np.random.seed(seed)
    
    if stratification_key is None:
        # No stratification: simple random sampling
        return np.random.choice(n_total, size=n_select, replace=False)
    
    # Stratified sampling
    stratification_key = np.asarray(stratification_key)
    strata = np.unique(stratification_key)
    
    selected = []
    for stratum in strata:
        stratum_indices = np.where(stratification_key == stratum)[0]
        n_stratum = max(1, int(n_select * len(stratum_indices) / len(stratification_key)))
        selected_stratum = np.random.choice(stratum_indices, size=min(n_stratum, len(stratum_indices)), replace=False)
        selected.extend(selected_stratum)
    
    # Ensure we have exactly n_select
    if len(selected) < n_select:
        remaining_indices = list(set(range(n_total)) - set(selected))
        additional = np.random.choice(remaining_indices, size=n_select - len(selected), replace=False)
        selected.extend(additional)
    elif len(selected) > n_select:
        selected = np.random.choice(selected, size=n_select, replace=False)
    
    return np.array(selected)


def generate_participant_id(prefix: str = "p", format_str: str = "{prefix}{idx:04d}") -> str:
    """
    Generate a unique participant ID.
    
    Args:
        prefix: Prefix for ID (e.g., "p" for "p0001")
        format_str: Format string with {prefix} and {idx} placeholders
        
    Returns:
        Formatted participant ID
    """
    import uuid
    idx = int(uuid.uuid4().int % 10000)
    return format_str.format(prefix=prefix, idx=idx)


def compute_accuracy(responses: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Compute accuracy of responses.
    
    Args:
        responses: Array of participant responses
        true_labels: Array of true labels
        
    Returns:
        Accuracy (0-1)
    """
    if len(responses) == 0:
        return 0.0
    return float(np.mean(responses == true_labels))


def compute_agreement(responses1: np.ndarray, responses2: np.ndarray) -> float:
    """
    Compute agreement between two response arrays.
    
    Args:
        responses1: First response array
        responses2: Second response array
        
    Returns:
        Agreement rate (0-1)
    """
    if len(responses1) != len(responses2):
        raise ValueError("Response arrays must have same length")
    if len(responses1) == 0:
        return 0.0
    return float(np.mean(responses1 == responses2))


def compute_response_time_stats(response_times: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of response times.
    
    Args:
        response_times: Array of response times (seconds)
        
    Returns:
        Dict with mean, median, std, min, max
    """
    return {
        "mean": float(np.mean(response_times)),
        "median": float(np.median(response_times)),
        "std": float(np.std(response_times)),
        "min": float(np.min(response_times)),
        "max": float(np.max(response_times)),
    }
