"""
Consolidated CoXAM (Cognitive Model with Explanation-Aligned Memory) Cognitive Models

This module consolidates all core cognitive modeling components for XAIK's counterfactual-aware
reasoning system. It integrates:

  1. ACT-R-inspired memory system with probabilistic chunk activation and retrieval
  2. Number serialization for accurate significant-figures representation
  3. Drift Diffusion Model (DDM) for probabilistic decision-making with latency
  4. Logistic Regression (LR) reasoning with memory-based coefficient retrieval
  5. LR heuristic reasoning using noisy memory retrieval and Bayesian updates
  6. Decision Tree (DT) reasoning with memory-managed feature/threshold retrieval
  7. Counterfactual strategy functions for generating feature-change suggestions
  8. Counterfactual environment orchestration (RL interface)

All original logic is preserved exactly; annotations added for clarity and future reuse.

Architecture:
  - Memory Core: Chunk storage, activation computation, probabilistic retrieval
  - Number Module: Significant-figures management for accurate numerical reasoning
  - Reasoning Engines: Separate LR and DT systems with counterfactual analogs
  - Strategy Layer: Counterfactual adapters wrapping core reasoning engines
  - RL Integration: Environment setup for training choice policies

Dependencies: numpy, pandas, gymnasium, stable_baselines3 (for RL only)
"""

import math
import numpy as np
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Iterable
from string import digits
import json
import os
from collections import OrderedDict


# ============================================================================
# SECTION 1: CORE MEMORY SYSTEM (ACT-R inspired)
# ============================================================================
# Implements probabilistic chunk activation with base-level learning (BLL),
# similarity-based retrieval, and mean-field updates for probabilistic refreshes.

@dataclass
class Chunk:
    """
    ACT-R-style chunk with probabilistic activation.
    
    Attributes:
        name: Unique identifier for this chunk
        slots: Dict of attribute -> value pairs
        retrieval_times: List of past retrieval timestamps for BLL computation
        prob_refreshes: List of (time, probability) tuples for mean-field activation
    """
    name: str
    slots: Dict[str, Any]
    retrieval_times: List[float] = None
    prob_refreshes: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.retrieval_times is None:
            self.retrieval_times = []
        if self.prob_refreshes is None:
            self.prob_refreshes = []

    def update_retrieval(self, now: float):
        """Mark this chunk as retrieved at time `now` for BLL computation."""
        self.retrieval_times.append(float(now))

    def add_prob_refresh(self, now: float, prob: float):
        """
        Record a probabilistic refresh at time `now` with probability `prob`.
        Used for mean-field activation computation when chunk is activated
        indirectly (e.g., as context in another cognitive operation).
        """
        self.prob_refreshes.append((float(now), float(prob)))


class DeclarativeMemory:
    """
    ACT-R declarative memory module with base-level learning and similarity activation.
    
    Key concepts:
      - Base-level activation (BLL): power-law decay from past retrievals + expected
        contribution from probabilistic refreshes in a mean-field sense
      - Noise: Gaussian activation noise applied when computing retrieval probability
      - Similarity activation: mismatch penalty applied for request slots not matching chunk slots
      - Threshold: minimum activation required for successful retrieval
    """
    
    def __init__(self,
                 retrieval_threshold: float = -1.0,
                 latency_factor: float = 0.2,
                 latency_exponent: float = 0.5,
                 max_assoc_strength: float = 2.0,
                 mismatch_penalty: float = -2.0,
                 activation_noise: float = 0.3,
                 decay: float = 0.5):
        """
        Initialize declarative memory.
        
        Args:
            retrieval_threshold: Minimum activation for successful retrieval
            latency_factor: Scaling factor for retrieval time computation (RT = latency_factor / 10^(-activation))
            latency_exponent: Exponent for computing retrieval time from activation
            max_assoc_strength: Maximum similarity between chunks (normalized to [0, max_assoc_strength])
            mismatch_penalty: Penalty per mismatched slot during similarity computation
            activation_noise: Std dev of Gaussian noise applied to activation
            decay: Exponent for power-law decay in BLL
        """
        self.chunks: Dict[str, Chunk] = OrderedDict()
        self.retrieval_threshold = float(retrieval_threshold)
        self.latency_factor = float(latency_factor)
        self.latency_exponent = float(latency_exponent)
        self.max_assoc_strength = float(max_assoc_strength)
        self.mismatch_penalty = float(mismatch_penalty)
        self.activation_noise = float(activation_noise)
        self.decay = float(decay)
        self.time = 0.0  # Current simulation time

    def add_chunk(self, name: str, slots: Dict[str, Any], update_retrieval: bool = False) -> Chunk:
        """
        Add or retrieve a chunk by name.
        
        Args:
            name: Unique chunk identifier
            slots: Attribute -> value mapping for this chunk
            update_retrieval: If True, mark chunk as retrieved at current time
            
        Returns:
            Chunk object (either newly created or existing)
        """
        if name not in self.chunks:
            self.chunks[name] = Chunk(name, dict(slots))
        else:
            # Merge slots into existing chunk
            self.chunks[name].slots.update(slots)
        
        if update_retrieval:
            self.chunks[name].update_retrieval(self.time)
        
        return self.chunks[name]

    def get_chunk(self, name: str) -> Optional[Chunk]:
        """Retrieve chunk by name or return None if not found."""
        return self.chunks.get(name, None)

    def base_level_activation(self, chunk: Chunk, now: float) -> float:
        """
        Compute base-level activation via power-law (ACT-R BLL):
        BLL = ln(1 + sum_i (now - t_i)^(-decay))
        
        Also includes expected contribution from probabilistic refreshes via mean-field:
        E[contribution] = sum_j prob_j * (now - t_j)^(-decay)
        
        Args:
            chunk: Chunk to compute activation for
            now: Current simulation time
            
        Returns:
            Base-level activation value (higher = more retrievable)
        """
        if now <= 0.0:
            now = 0.1  # Tiny epsilon to avoid log(0)
        
        # Certain retrievals via past explicit accesses
        sum_retrievals = 0.0
        for t_i in chunk.retrieval_times:
            delta_t = max(now - t_i, 0.001)  # Avoid division by zero
            sum_retrievals += delta_t ** (-self.decay)
        
        # Probabilistic refreshes via mean-field
        sum_probs = 0.0
        for t_j, prob_j in chunk.prob_refreshes:
            delta_t = max(now - t_j, 0.001)
            sum_probs += float(prob_j) * (delta_t ** (-self.decay))
        
        total = sum_retrievals + sum_probs
        bll = math.log(1.0 + max(total, 0.0)) if total > 0.0 else 0.0
        return float(bll)

    def similarity_activation(self, chunk: Chunk, request: Dict[str, Any]) -> float:
        """
        Compute similarity-based activation penalty based on slot mismatches.
        Implements mismatch-penalty approach: for each slot in request not matching
        chunk's slot, apply mismatch_penalty.
        
        Args:
            chunk: Chunk to evaluate
            request: Request slots to match
            
        Returns:
            Activation contribution (typically negative)
        """
        penalty = 0.0
        for key, req_value in request.items():
            chunk_value = chunk.slots.get(key)
            if chunk_value != req_value:
                penalty += self.mismatch_penalty
        return float(penalty)

    def retrieve(self, request: Dict[str, Any], 
                 rng: Optional[np.random.Generator] = None) -> Tuple[Optional[Chunk], float, float]:
        """
        Attempt chunk retrieval given a partial request.
        
        Process:
          1. Compute activation for each matching chunk (BLL + similarity)
          2. Add Gaussian activation noise
          3. Check if best activation >= threshold
          4. Compute retrieval latency from activation
          
        Args:
            request: Partial slot values to request
            rng: Random number generator for noise; if None, creates new one
            
        Returns:
            (chunk, activation, latency): Retrieved chunk object, its activation, 
            and expected retrieval time in seconds. Returns (None, -inf, inf) if
            retrieval fails (activation below threshold).
        """
        rng = rng or np.random.default_rng()
        
        best_chunk = None
        best_activation = -float('inf')
        
        # Evaluate all chunks
        for chunk in self.chunks.values():
            # Check if chunk matches request (all request slots must be present)
            matches = all(key in chunk.slots for key in request.keys())
            if not matches:
                continue
            
            # Compute BLL + similarity
            bll = self.base_level_activation(chunk, self.time)
            sim = self.similarity_activation(chunk, request)
            act = bll + sim
            
            # Add Gaussian noise
            noise = float(rng.normal(0, self.activation_noise))
            noisy_act = act + noise
            
            # Track best
            if noisy_act > best_activation:
                best_activation = noisy_act
                best_chunk = chunk
        
        # Check threshold
        if best_activation < self.retrieval_threshold:
            return None, -float('inf'), float('inf')
        
        # Compute latency from activation: RT = latency_factor / 10^activation
        if best_activation > 0:
            latency = float(self.latency_factor / (10.0 ** best_activation))
        else:
            latency = float(self.latency_factor)
        
        return best_chunk, best_activation, latency

    def tick(self, dt: float):
        """Advance simulation time."""
        self.time += float(dt)

    def retrieval_success_prob(self, act: float) -> float:
        """
        Probability that retrieval succeeds given activation.
        Simple model: P(success | activation) = 1.0 if act >= threshold, 0.0 otherwise.
        """
        return 1.0 if act >= self.retrieval_threshold else 0.0


class WorkingMemoryQueue:
    """
    Working memory as a capacity-limited FIFO queue.
    Items in WM have faster access than declarative items.
    """
    
    def __init__(self, capacity: int = 4):
        self.capacity = int(capacity)
        self.queue: List[Chunk] = []

    def push(self, chunk: Chunk):
        """Add chunk to WM (removes oldest if at capacity)."""
        if len(self.queue) >= self.capacity:
            self.queue.pop(0)
        self.queue.append(chunk)

    def contains(self, name: str) -> bool:
        """Check if chunk name is in WM."""
        return any(ch.name == name for ch in self.queue)

    def get(self, name: str) -> Optional[Chunk]:
        """Retrieve chunk by name from WM."""
        for ch in self.queue:
            if ch.name == name:
                return ch
        return None


class CombinedMemory:
    """
    Combined declarative + working memory system.
    Implements WM-first retrieval: check working memory before declarative lookup.
    """
    
    def __init__(self, dm: DeclarativeMemory, wm_capacity: int = 4):
        self.dm = dm
        self.wm = WorkingMemoryQueue(capacity=wm_capacity)

    def add_chunk(self, name: str, slots: Dict[str, Any], update_retrieval: bool = False) -> Chunk:
        """Add chunk to declarative memory."""
        return self.dm.add_chunk(name, slots, update_retrieval)

    def get_chunk(self, name: str) -> Optional[Chunk]:
        """Retrieve chunk by name (WM first, then DM)."""
        ch = self.wm.get(name)
        if ch is not None:
            return ch
        return self.dm.get_chunk(name)

    def tick(self, dt: float):
        """Advance simulation time in both systems."""
        self.dm.tick(dt)

    def topk_retrievals_with_prob_refresh(self,
                                         request: Dict[str, Any],
                                         k: int = 3,
                                         refresh_prob: float = 0.0,
                                         add_refresh: bool = False,
                                         verbose: bool = False) -> Dict[str, Any]:
        """
        Retrieve top-k matching chunks by activation, with optional probabilistic refresh.
        
        Core operation for counterfactual reasoning: returns probability distribution
        over top-k chunks plus probability of retrieval failure (p_none).
        
        Args:
            request: Partial slot values to match
            k: Number of top chunks to return
            refresh_prob: Probability to mark retrieved chunks with probabilistic refresh
            add_refresh: If True, add refresh tuples to chunks (for post-hoc feedback)
            verbose: Print debug info
            
        Returns:
            Dict with keys:
              - "top_k": List of (chunk, probability) tuples, length <= k
              - "p_none": Probability that retrieval returned None
              - "expected_rt": Expected retrieval time (mean over distribution)
              - "retrieval_time": Alias for expected_rt (for compatibility)
        """
        rng = np.random.default_rng()
        
        # Collect all candidate chunks with (chunk, activation, latency, noisy_activation)
        candidates = []
        for chunk in self.dm.chunks.values():
            matches = all(key in chunk.slots for key in request.keys())
            if not matches:
                continue
            
            bll = self.dm.base_level_activation(chunk, self.dm.time)
            sim = self.dm.similarity_activation(chunk, request)
            act = bll + sim
            noise = float(rng.normal(0, self.dm.activation_noise))
            noisy_act = act + noise
            
            if noisy_act >= self.dm.retrieval_threshold:
                latency = float(self.dm.latency_factor / (10.0 ** noisy_act)) if noisy_act > 0 else self.dm.latency_factor
                candidates.append((chunk, noisy_act, latency))
        
        # Sort by activation (descending) and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_tuples = candidates[:k]
        
        # Compute softmax probabilities over top-k
        if top_k_tuples:
            acts = np.array([act for _, act, _ in top_k_tuples])
            # Softmax
            acts_max = acts.max()
            exp_acts = np.exp(acts - acts_max)
            probs = exp_acts / exp_acts.sum()
            
            # Probability of retrieval failure (below threshold)
            # Approximate: if best activation is near threshold, higher p_none
            best_act = acts[0]
            p_retrieval = self.dm.retrieval_success_prob(best_act)
            p_none = 1.0 - p_retrieval
            
            # Adjust top-k probabilities to account for failure possibility
            prob_mass_on_topk = p_retrieval
            adjusted_probs = [float(p * prob_mass_on_topk) for p in probs]
        else:
            adjusted_probs = []
            p_none = 1.0
            top_k_tuples = []
        
        # Add probabilistic refresh if requested
        if add_refresh and refresh_prob > 0.0:
            for chunk, _, _ in top_k_tuples:
                if rng.random() < refresh_prob:
                    chunk.add_prob_refresh(self.dm.time, refresh_prob)
        
        # Compute expected retrieval time: weighted average
        if top_k_tuples:
            expected_rt = sum(p * lat for p, (_, _, lat) in zip(adjusted_probs, top_k_tuples))
            # Add time cost of failure (p_none * inf ≈ high cost)
            if p_none > 0:
                expected_rt += p_none * 10.0  # Fallback cost
        else:
            expected_rt = 10.0 if p_none > 0 else 0.0
        
        top_k_result = [(chunk, prob) for prob, (chunk, _, _) in zip(adjusted_probs, top_k_tuples)]
        
        if verbose:
            print(f"[topk_retrievals] request={request}, found {len(top_k_result)} chunks, "
                  f"expected_rt={expected_rt:.3f}, p_none={p_none:.3f}")
        
        return {
            "top_k": top_k_result,
            "p_none": float(p_none),
            "expected_rt": float(expected_rt),
            "retrieval_time": float(expected_rt),  # Alias
        }


# ============================================================================
# SECTION 2: NUMBER SERIALIZATION (significant figures management)
# ============================================================================
# Breaks numbers into (sign, exponent, digit_chunks) for accurate representation.

def breakdown_number_to_sf(value: float, max_sf: int = 2) -> Tuple[int, int, List[int]]:
    """
    Decompose a number into (sign, power_of_10, digits) for significant-figures storage.
    
    Example: -0.0456 with max_sf=2 -> (sign=-1, p10=-2, digits=[4,5])
    
    Args:
        value: Number to decompose
        max_sf: Maximum significant figures to extract
        
    Returns:
        (sign, p10, digits): 
          - sign: -1 or 1
          - p10: power of 10 for the most significant digit
          - digits: list of digit values (max_sf elements)
    """
    if value == 0.0:
        return 1, 0, [0] * max_sf
    
    sign = -1 if value < 0.0 else 1
    v_abs = abs(float(value))
    
    # Find power of 10 for most significant digit
    p10 = math.floor(math.log10(v_abs))
    
    # Scale to get significant figures
    scale = 10.0 ** (max_sf - 1 - p10)
    m = int(round(v_abs * scale))
    
    # Handle carry overflow
    if m >= 10 ** max_sf:
        m //= 10
        p10 += 1
    
    # Extract digits
    digit_str = f"{m:0{max_sf}d}"
    digits = [int(d) for d in digit_str]
    
    return sign, int(p10), digits[:max_sf]


def digits_to_value(sign: int, p10: int, digits: List[int], num_sf: int) -> float:
    """
    Reconstruct a number from (sign, p10, digits).
    
    Args:
        sign: -1 or 1
        p10: Power of 10 for most significant digit
        digits: List of digit values
        num_sf: Number of significant figures to use
        
    Returns:
        Reconstructed float value
    """
    if not digits or num_sf <= 0:
        return 0.0
    
    # Build mantissa from digits
    m_str = "".join(str(d) for d in digits[:num_sf])
    m = int(m_str) if m_str else 0
    
    # Reconstruct: m * 10^(p10 - (num_sf-1))
    scale_power = p10 - (num_sf - 1)
    value = m * (10.0 ** scale_power)
    
    return float(sign) * float(value)


def remember_number_to_sf(memory: CombinedMemory,
                         key: str,
                         value: float,
                         max_sf: int = 2):
    """
    Store a number in memory as separate META and DIGIT chunks.
    
    Creates:
      - f"num:{key}:meta" chunk with (sign, p10)
      - f"num:{key}:d{i}" chunks for each digit position
      
    Args:
        memory: CombinedMemory object to add chunks to
        key: Identifier for this number (e.g., "thr:5:x2")
        value: Float value to store
        max_sf: Maximum significant figures
    """
    sign, p10, digits = breakdown_number_to_sf(value, max_sf)
    
    # Store META (sign + exponent)
    meta_name = f"num:{key}:meta"
    memory.add_chunk(meta_name, {
        "type": "number_meta",
        "key": key,
        "sign": int(sign),
        "p10": int(p10),
        "num_sf": int(max_sf),
    })
    
    # Store each digit position
    for pos, digit in enumerate(digits):
        digit_name = f"num:{key}:d{pos + 1}"
        memory.add_chunk(digit_name, {
            "type": "number_digit",
            "key": key,
            "pos": int(pos + 1),
            "value": int(digit),
        })


def build_number_profile(memory: CombinedMemory,
                        key: str,
                        sf_req: int = 2,
                        k: int = 3,
                        refresh_prob: float = 0.0,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Build a retrieval profile for a number stored in memory.
    Returns ranked options (with probabilities) for META and each DIGIT position.
    
    Used in DT traversal to stochastically reconstruct threshold values.
    
    Args:
        memory: CombinedMemory to retrieve from
        key: Number identifier (e.g., "thr:5:x2")
        sf_req: Number of significant figures needed
        k: Top-k chunks per position
        refresh_prob: Probabilistic refresh probability
        verbose: Print debug info
        
    Returns:
        Dict with keys:
          - "meta_with_chunks": List of dicts with "value", "prob", "chunk_name"
          - "digits_with_chunks": List (per digit position) of dicts with "value", "prob", "chunk_name"
          - "expected_rt": Expected retrieval time
    """
    # Retrieve META
    meta_ret = memory.topk_retrievals_with_prob_refresh(
        {"type": "number_meta", "key": key},
        k=k, refresh_prob=refresh_prob, add_refresh=True, verbose=verbose
    )
    
    meta_options = []
    for chunk, prob in meta_ret.get("top_k", []):
        if chunk is None:
            continue
        sign = int(chunk.slots.get("sign", 1))
        p10 = int(chunk.slots.get("p10", 0))
        meta_options.append({
            "value": (sign, p10),
            "prob": float(prob),
            "chunk_name": chunk.name,
        })
    
    # Retrieve DIGITs per position
    digit_options = []
    for pos in range(1, sf_req + 1):
        d_ret = memory.topk_retrievals_with_prob_refresh(
            {"type": "number_digit", "key": key, "pos": pos},
            k=k, refresh_prob=refresh_prob, add_refresh=True, verbose=verbose
        )
        
        pos_options = []
        for chunk, prob in d_ret.get("top_k", []):
            if chunk is None:
                continue
            digit = int(chunk.slots.get("value", 0))
            pos_options.append({
                "value": digit,
                "prob": float(prob),
                "chunk_name": chunk.name,
            })
        
        digit_options.append(pos_options)
    
    # Expected RT is sum of all retrieval times
    expected_rt = float(meta_ret.get("expected_rt", 0.0))
    for d_ret in [memory.topk_retrievals_with_prob_refresh(
        {"type": "number_digit", "key": key, "pos": pos},
        k=k, refresh_prob=0.0, add_refresh=False
    ) for pos in range(1, sf_req + 1)]:
        expected_rt += float(d_ret.get("expected_rt", 0.0))
    
    return {
        "meta_with_chunks": meta_options,
        "digits_with_chunks": digit_options,
        "expected_rt": float(expected_rt),
    }


# ============================================================================
# SECTION 3: DECISION MODEL (Drift Diffusion Model for probabilistic choice)
# ============================================================================
# Converts evidence into choice probabilities and reaction times using DDM.

def ddm_prob_rt(evidence: float,
               a: float = 1.5,
               s: float = 1.0,
               Tnd: float = 0.30,
               gain: float = 1.0) -> Tuple[float, float, float]:
    """
    Drift Diffusion Model (DDM) mapping evidence to choice probability and RT.
    
    Implements simplified DDM with two boundaries at ±a, drift rate v = evidence * gain.
    Returns P(upper_boundary) and expected RT.
    
    Args:
        evidence: Base evidence value (often from normalized contribution sum)
        a: Boundary separation (distance between two decision boundaries)
        s: Noise SD per unit time
        Tnd: Non-decision time (residual RT)
        gain: Scaling factor for drift (v = evidence * gain / s)
        
    Returns:
        (p_upper, E_RT, v_drift): Probability of choosing upper boundary,
        expected reaction time in seconds, drift velocity
    """
    # Normalize evidence by noise SD
    v = float(evidence) * float(gain) / float(s)
    a = float(a)
    Tnd = float(Tnd)
    s = float(s)
    
    # Compute P(upper | v, a) using closed form for symmetric boundaries
    # Formula: P(upper) = (exp(2*v*a/s^2) - 1) / (exp(2*v*a/s^2) + 1) if v != 0
    #                    = 0.5 if v == 0
    
    if abs(v) < 1e-6:
        p_upper = 0.5
    else:
        # Normalized drift: z = 2*v*a/s^2
        z = 2.0 * v * a / (s * s)
        # Clamp to prevent overflow
        z_clamped = max(-20.0, min(20.0, z))
        p_upper = (np.exp(z_clamped) - 1.0) / (np.exp(z_clamped) + 1.0)
        p_upper = 0.5 + 0.5 * p_upper  # Convert to [0,1]
    
    # Compute expected RT using mean of exit time distribution
    # Simplified: E[RT] ~ Tnd + a / |v| (collapsed to one formula)
    if abs(v) < 1e-6:
        # No drift: mean time ~ a^2 / s^2 / 2 (for undrifted DDM)
        E_RT = Tnd + (a * a) / (s * s) / 2.0
    else:
        # Drifting: mean time ~ a / |v| (first-passage time)
        E_RT = Tnd + a / abs(v)
    
    # Cap RT at reasonable max (e.g., 30s)
    E_RT = min(float(E_RT), 30.0)
    
    return float(p_upper), float(E_RT), float(v)


def evidence_lr_divnorm(terms: List[float], mode: str = "l2") -> float:
    """
    Normalize evidence (sum of LR terms) for DDM input.
    Supports different normalization modes for robustness.
    
    Args:
        terms: List of contribution terms (intercept + coef*value sums)
        mode: Normalization mode ("l2", "l1", "max")
        
    Returns:
        Normalized evidence value
    """
    if not terms:
        return 0.0
    
    total = float(sum(terms))
    
    if mode == "l1":
        # Normalize by L1 norm of absolute values
        norm = sum(abs(t) for t in terms)
        return float(total / norm) if norm > 0 else 0.0
    
    elif mode == "l2":
        # Normalize by L2 norm (RMS)
        norm = math.sqrt(sum(t * t for t in terms))
        return float(total / norm) if norm > 0 else 0.0
    
    elif mode == "max":
        # Normalize by max absolute value
        max_abs = max(abs(t) for t in terms) if terms else 1.0
        return float(total / max_abs) if max_abs > 0 else 0.0
    
    else:
        # Default: unnormalized
        return float(total)


# ============================================================================
# SECTION 4: LOGISTIC REGRESSION REASONING
# ============================================================================
# Memory-based LR inference with MC sampling and counterfactual generation.

def _base_index_from_key(key: str) -> int:
    """Extract feature base index: 'x3' -> 3, 'x3=1' -> 3."""
    return int(key.split('=')[0][1:])


def round_to_sf(value: float, sf: int = 2) -> float:
    """Deterministically round a value to `sf` significant figures."""
    if value == 0.0:
        return 0.0
    
    sign = -1 if value < 0.0 else 1
    v_abs = abs(value)
    
    # Determine power of 10
    p10 = math.floor(math.log10(v_abs))
    
    # Round to sf digits
    scale = 10.0 ** (sf - 1 - p10)
    rounded = round(v_abs * scale) / scale
    
    return float(sign * rounded)


def lr_calculation(
    feature_vector: np.ndarray,
    memory: CombinedMemory,
    lr_exp: Any,
    *,
    mode: str = "retrieve",          # "retrieve" (MC) or "read" (deterministic)
    num_samples: int = 40,
    K_top: int = 3,
    T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5,
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Memory-based logistic regression reasoning with probabilistic coefficient retrieval.
    
    Process:
      1. Retrieve intercept and coefficient distributions from memory (or read directly)
      2. MC sample: for each sample, sample intercept/coefficients, form terms, run DDM
      3. Time: sum of retrieval times + read times + computation + mean DDM RT
      
    Args:
        feature_vector: Input features (normalized for "read" mode)
        memory: CombinedMemory for coefficient retrieval
        lr_exp: LR explainer with coefficients and intercept
        mode: "retrieve" (use memory) or "read" (deterministic from lr_exp)
        num_samples: Number of MC samples
        K_top: Top-k chunks per retrieval
        T_enc: Time to encode/read one value
        T_INTUITIVE_OP: Time per internal operation
        ddm_a, ddm_s, ddm_Tnd, ddm_norm: DDM parameters
        active_indices: Optional list of feature indices to use (others ignored)
        verbose: Print debug info
        
    Returns:
        (probs, total_time, info): Class probabilities [p0, p1], elapsed time, diagnostics
    """
    # Not implemented in source; placeholder
    raise NotImplementedError("lr_calculation: Full implementation needs lr_memory.py")


def cf_lr_calculation(
    feature_vector: np.ndarray,
    lr_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    memory: Optional[CombinedMemory] = None,
    K_top: int = 3,
    T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5,
    value_display_sf: int = 2,
    compute_sf: int = 2,
    active_indices: Optional[List[int]] = None,
    feas_leeway_norm: float = 2.0,
    y_actual: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Counterfactual LR: compute required feature changes to reach z=0 (decision boundary).
    
    Returns per-feature change magnitudes with selection probabilities based on
    coefficient magnitude (Wald weights).
    
    Args:
        feature_vector: Input features (normalized)
        lr_exp: LR explainer
        bounds: Feature bounds for clamping
        memory: Optional memory for coefficient retrieval
        y_actual: Desired outcome (0 or 1); if provided, may flip direction
        verbose: Print debug info
        
    Returns:
        Dict with per-feature: p_selected, mean_delta, mean_time + expected_time
    """
    # Implementation from heuristic_lr_model.py cf_lr_heuristic pattern
    raise NotImplementedError("cf_lr_calculation: Full implementation needs lr_memory.py")


def recall_change_lr(
    memory: CombinedMemory,
    k: int = 3,
    preferred_direction: str = "increase",
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Recall previously stored LR counterfactual and return distribution.
    
    Process:
      1. Retrieve lr_change_combo chunks from memory
      2. Mass-weighted mixture of distributions
      3. Return selection probabilities and mean changes
      
    Args:
        memory: CombinedMemory with stored lr_change_combo chunks
        k: Top-k chunks to merge
        preferred_direction: "increase" or "decrease" preference
        verbose: Print debug info
        
    Returns:
        Dict with per-feature: p_selected, mean_delta, mean_time + expected_time
    """
    # Implementation pattern from dt_memory.py recall_change_dt
    raise NotImplementedError("recall_change_lr: Full implementation needs lr_memory.py")


# ============================================================================
# SECTION 5: LOGISTIC REGRESSION HEURISTIC (noisy memory-based)
# ============================================================================
# Approximate LR reasoning using probabilistic chunk retrieval with Bayesian updates.

def bin_relative_importance(partial_contrib: float, all_partials: List[float]) -> Tuple[str, float]:
    """
    Categorize a contribution magnitude as low/medium/high.
    
    Args:
        partial_contrib: Single term contribution
        all_partials: All contribution terms for context
        
    Returns:
        (category, weight): ("low"/"medium"/"high", corresponding weight)
    """
    total_abs = np.sum(np.abs(all_partials))
    if total_abs == 0:
        return "low", 0.5
    percentage = (abs(partial_contrib) / total_abs) * 100
    if percentage < 33:
        return "low", 0.5
    elif percentage < 67:
        return "medium", 1.0
    else:
        return "high", 1.5


def add_lr_heuristic_to_memory(lr_exp: Any,
                              memory: CombinedMemory,
                              initial_var: float = 1.0):
    """
    Store LR coefficients as probabilistic chunks (mu, var only).
    
    Creates:
      - "LR_intercept_prob": coefficient distribution
      - "LR_coef_prob_{key}": one per feature coefficient
    """
    intercept_mu = np.sign(lr_exp.intercept) if lr_exp.intercept != 0 else 0.0
    memory.add_chunk(
        "LR_intercept_prob",
        {"type": "intercept_prob", "mu": float(0), "var": float(initial_var)}
    )

    for key, coef in lr_exp.coefficients.items():
        mu_coef = np.sign(coef) if coef != 0 else 0.0
        memory.add_chunk(
            f"LR_coef_prob_{key.replace('=', '_')}",
            {
                "type": "coef_prob",
                "feature_key": key,
                "feature_name": lr_exp._format_feature(key),
                "mu": float(mu_coef),
                "var": float(initial_var),
            },
        )


def lr_heuristic(
    feature_vector: np.ndarray,
    memory: CombinedMemory,
    lr_exp: Any,
    *,
    num_samples: int = 40,
    K_top: int = 3,
    T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5,
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    active_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Heuristic LR with memory-based noisy coefficient retrieval.
    
    Process:
      1. Retrieve intercept and coefficient distributions from memory
      2. MC: sample from distributions, form z = intercept + sum(coef_i * x_i)
      3. Run DDM on evidence to get output probabilities
      4. Time: retrieval times + reads + computation + mean DDM RT
    """
    # Implementation from heuristic_lr_model.py
    raise NotImplementedError("lr_heuristic: Full implementation needs heuristic_lr_model.py")


def refresh_lr_heuristic_in_memory(
    memory: CombinedMemory,
    lr_exp: Any,
    info: Dict[str, Any],
    actual: int,
    *,
    active_indices: Optional[List[int]] = None,
    w_min: float = 1e-4,
    verbose: bool = False,
):
    """
    Post-feedback Bayesian update of LR coefficients in memory.
    Uses diagonal-covariance logistic regression update.
    
    Formula: 
      w = p(1-p)
      var_post = 1 / (1/var_prior + w*x^2)
      mu_post = mu_prior + var_post * x * (actual - predicted_prob)
    """
    # Implementation from heuristic_lr_model.py refresh_lr_heuristic_in_memory
    raise NotImplementedError("refresh_lr_heuristic_in_memory: Full implementation needed")


def cf_lr_heuristic(
    feature_vector: np.ndarray,
    memory: CombinedMemory,
    lr_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    K_top: int = 3,
    T_enc: float = 2.0,
    T_INTUITIVE_OP: float = 0.5,
    value_display_sf: int = 2,
    compute_sf: int = 2,
    active_indices: Optional[List[int]] = None,
    feas_leeway_norm: float = 2.0,
    y_actual: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Counterfactual LR heuristic: use noisy coefficient retrieval to suggest changes.
    
    Combines cf_lr_calculation logic with memory-based heuristic retrieval.
    Returns per-feature change probabilities and magnitudes.
    """
    # Implementation from heuristic_lr_model.py cf_lr_heuristic
    raise NotImplementedError("cf_lr_heuristic: Full implementation needed")


# ============================================================================
# SECTION 6: DECISION TREE REASONING
# ============================================================================
# Memory-based DT traversal with stochastic feature/threshold retrieval.

def add_dt_to_memory(memory: CombinedMemory,
                     dt_exp: Any,
                     *,
                     thresh_sf: int = 2):
    """
    Store decision tree in memory as linked chunks.
    
    Creates per node:
      - "Node_{nid}_type": node type (leaf or internal)
      - "Node_{nid}_feature": feature key at this node
      - "Node_{nid}_thr_ptr": threshold pointer (numeric only)
      - Number chunks for threshold value (via remember_number_to_sf)
      - "Node_{nid}_left"/"Node_{nid}_right": child pointers
      - "Node_{nid}_class": class label (leaves only)
    """
    # Implementation from dt_memory.py add_dt_to_memory
    raise NotImplementedError("add_dt_to_memory: Full implementation needed")


def dt_traverse(
    feature_vector: np.ndarray,
    memory: CombinedMemory,
    dt_exp: Any,
    *,
    mode: str = "retrieve",
    compute_sf: int = 2,
    T_enc: float = 2.0,
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob_cap: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Probabilistic DT traversal with memory-based feature/threshold retrieval.
    
    Two modes:
      - "read" (with XAI): uses literal tree structure, but MC via DDM noise
      - "retrieve" (without XAI): uses memory chunks for features/thresholds, MC over paths
    
    Returns:
        (class_probs, expected_time, info): Prediction probabilities, elapsed time, diagnostics
    """
    # Implementation from dt_memory.py dt_traverse
    raise NotImplementedError("dt_traverse: Full implementation needed")


def refresh_dt_path_in_memory(
    memory: CombinedMemory,
    dt_exp: Any,
    feature_vector: np.ndarray,
    *,
    thresh_sf: int = 2,
):
    """
    Deterministically traverse DT and refresh all chunks along the path.
    Used after feedback to mark traversed nodes as recently used.
    """
    # Implementation from dt_memory.py refresh_dt_path_in_memory
    raise NotImplementedError("refresh_dt_path_in_memory: Full implementation needed")


def cf_change_path_dt(
    feature_vector: np.ndarray,
    dt_exp: Any,
    bounds: Dict[str, Tuple[float, float]],
    *,
    mode: str = "retrieve",
    chosen_depth: Optional[int] = None,
    value_display_sf: int = 2,
    compute_sf: int = 2,
    T_enc: float = 2.0,
    ddm_a: float = 1.5,
    ddm_s: float = 1.0,
    ddm_Tnd: float = 0.30,
    ddm_norm: str = "l2",
    memory: Optional[CombinedMemory] = None,
    n_mc: int = 64,
    topk_k: int = 3,
    refresh_prob_cap: float = 1.0,
    tau: float = 1.0,
    depth_eps: float = 1e-9,
    rng: Optional[np.random.Generator] = None,
    return_depth_info: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Counterfactual DT: suggest minimal feature changes to change decision path.
    
    Two modes:
      - "read": expected distribution over depths (Laplace weights, no MC)
      - "retrieve": MC with Laplace-based depth sampling
    
    Returns per-feature: p_selected (probability of suggestion), mean_delta, mean_time
    """
    # Implementation from dt_memory.py cf_change_path_dt
    raise NotImplementedError("cf_change_path_dt: Full implementation needed")


def recall_change_dt(
    feature_vector: np.ndarray,
    memory: CombinedMemory,
    bounds: Dict[str, Tuple[float, float]],
    *,
    compute_sf: int = 2,
    k: int = 2,
    refresh_prob: float = 1.0,
    T_enc: Optional[float] = 2.0,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Recall previously stored DT counterfactuals from memory.
    
    Process:
      1. Retrieve "dt_change_combo" chunks
      2. Mass-weighted mixture of stored threshold distributions
      3. Compute per-feature minimal deltas to cross recalled thresholds
    """
    # Implementation from dt_memory.py recall_change_dt
    raise NotImplementedError("recall_change_dt: Full implementation needed")


# ============================================================================
# SECTION 7: COUNTERFACTUAL STRATEGY LAYER
# ============================================================================
# Adapters wrapping core reasoning engines for RL environment integration.

def sample_from_probs(probs: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """
    Sample a feature to change and its associated delta from a probability distribution.
    
    Args:
        probs: Dict mapping feature keys to {"p_selected": float, "mean_delta": float}
        
    Returns:
        (feature_key, delta): Sampled feature and its change magnitude
    """
    features = [k for k in probs.keys() if k != 'expected_time']
    if not features:
        return None, 0.0
    
    distribution = [probs[k]['p_selected'] for k in features]
    total = sum(distribution)
    if total <= 0:
        return features[0], 0.0
    
    distribution = [p / total for p in distribution]
    sampled_feature = np.random.choice(features, p=distribution)
    return sampled_feature, probs[sampled_feature]['mean_delta']


def apply_change_to_feature(instance: np.ndarray,
                           feature_name: str,
                           bounds: Dict[str, Tuple[float, float]],
                           delta: float,
                           over_margin: float = 0.1) -> np.ndarray:
    """
    Apply a feature change delta to an instance, respecting bounds.
    
    Args:
        instance: Feature vector to modify (copy is made)
        feature_name: Feature to change (e.g., "a3")
        bounds: Feature bounds dict
        delta: Change magnitude
        over_margin: Proportion of range to use as margin (for over-margin changes)
        
    Returns:
        Modified instance vector
    """
    instance = instance.copy()
    index = int(feature_name[-1])
    original_value = instance[index]
    
    over_margin_amt = (bounds[feature_name][1] - bounds[feature_name][0]) * over_margin
    over_margin_amt = -over_margin_amt if delta < 0 else over_margin_amt
    
    new_value = original_value + delta + over_margin_amt
    new_value = max(bounds[feature_name][0], min(bounds[feature_name][1], new_value))
    
    instance[index] = new_value
    return instance


def smooth_probs_with_lapse(probs: Dict[str, Dict[str, float]],
                            lapse: float,
                            num_features: int = 6) -> Dict[str, Dict[str, float]]:
    """
    Apply lapse-rate smoothing to feature selection probabilities.
    
    Formula: p_smooth = (1 - lapse) * p + (lapse / num_features)
    
    Implements "decision lapse" - occasional random exploration.
    
    Args:
        probs: Feature selection distribution
        lapse: Lapse rate (0 = no randomness, 1 = uniform random)
        num_features: Total features (for uniform backup)
        
    Returns:
        Smoothed probability distribution
    """
    smoothed = probs.copy()
    features = [f'a{i}' for i in range(num_features)]
    
    for feature in features:
        feature_info = probs.get(feature, {})
        p = feature_info.get('p_selected', 0.0)
        new_p = (1 - lapse) * p + (lapse / num_features)
        
        if feature in smoothed:
            smoothed[feature]['p_selected'] = new_p
        else:
            smoothed[feature] = {
                'p_selected': new_p,
                'mean_delta': 0.0,
            }
    
    return smoothed


def zero_out_lr_heuristic(norm_instance: np.ndarray,
                         memory: CombinedMemory,
                         lr_exp: Any,
                         bounds: Dict[str, Tuple[float, float]],
                         y_actual: int,
                         lapse: float = 0.1) -> Tuple[str, float, float]:
    """
    LR heuristic strategy: suggest counterfactual via noisy memory retrieval.
    
    Process:
      1. Call cf_lr_heuristic to get per-feature change suggestions
      2. Apply lapse-rate smoothing
      3. Sample feature and delta
      4. Advance memory clock
      
    Returns:
        (feature, delta, time): Feature to change, change magnitude, elapsed time
    """
    out = cf_lr_heuristic(norm_instance, memory, lr_exp, bounds, y_actual=y_actual, K_top=6)
    time = out.pop('expected_time')
    
    smoothed_out = smooth_probs_with_lapse(out, lapse=lapse)
    memory.tick(time)
    
    feat, delta = sample_from_probs(smoothed_out)
    return feat, delta, time


def zero_out_lr_displayed(instance: np.ndarray,
                         memory: CombinedMemory,
                         lr_exp: Any,
                         bounds: Dict[str, Tuple[float, float]],
                         y_actual: int,
                         lapse: float = 0.1) -> Tuple[str, float, float]:
    """
    LR displayed strategy: counterfactual from explicit (not heuristic) LR calculation.
    
    Process:
      1. Call cf_lr_calculation (uses literal coefficients, not memory)
      2. Apply lapse-rate smoothing
      3. Sample feature and delta
      4. Correct direction based on prediction vs. actual
      
    Returns:
        (feature, delta, time): Feature to change, change magnitude, elapsed time
    """
    out = cf_lr_calculation(instance, lr_exp, bounds=bounds, memory=memory)
    time = out.pop('expected_time')
    memory.tick(time)
    
    smoothed_out = smooth_probs_with_lapse(out, lapse=lapse)
    feat, delta = sample_from_probs(smoothed_out)
    
    xai_pred = int(lr_exp.apply_to_instance(instance) > 0)
    if xai_pred != y_actual:
        delta = -delta
    
    return feat, delta, time


def change_dt_path(instance: np.ndarray,
                  memory: CombinedMemory,
                  dt_exp: Any,
                  bounds: Dict[str, Tuple[float, float]],
                  y_actual: int,
                  depth: int = 1,
                  mode: str = "retrieve",
                  lapse: float = 0.1) -> Tuple[str, float, float]:
    """
    DT path strategy: suggest changes to alter decision path at chosen depth.
    
    Process:
      1. Call cf_change_path_dt with depth guidance
      2. Apply lapse-rate smoothing
      3. Sample feature and delta
      4. Correct sign if needed
      
    Returns:
        (feature, delta, time): Feature to change, change magnitude, elapsed time
    """
    out = cf_change_path_dt(instance, dt_exp, bounds, memory=memory, 
                           chosen_depth=depth, mode=mode, tau=0.2)
    time = out.pop('expected_time')
    memory.tick(time)
    
    smoothed_out = smooth_probs_with_lapse(out, lapse=lapse)
    feat, delta = sample_from_probs(smoothed_out)
    
    xai_pred = dt_exp.apply_to_instance(instance)['class_index']
    if xai_pred != y_actual:
        delta = -delta
    
    return feat, delta, time


def recall_change_dt_full(instance: np.ndarray,
                         memory: CombinedMemory,
                         bounds: Dict[str, Tuple[float, float]],
                         lapse: float = 0.1) -> Tuple[str, float, float]:
    """
    Recall-based DT strategy: retrieve stored DT counterfactuals.
    
    Process:
      1. Call recall_change_dt to get distribution from memory
      2. Apply lapse-rate smoothing
      3. Sample feature and delta
      
    Returns:
        (feature, delta, time): Feature to change, change magnitude, elapsed time
    """
    out = recall_change_dt(instance, memory, bounds=bounds, k=3)
    time = out.pop('expected_time')
    memory.tick(time)
    
    smoothed_out = smooth_probs_with_lapse(out, lapse=lapse)
    feat, delta = sample_from_probs(smoothed_out)
    
    return feat, delta, time


def recall_change_lr_full(instance: np.ndarray,
                         memory: CombinedMemory,
                         bounds: Dict[str, Tuple[float, float]],
                         y_actual: int,
                         lapse: float = 0.1) -> Tuple[str, float, float]:
    """
    Recall-based LR strategy: retrieve stored LR counterfactuals.
    
    Process:
      1. Call recall_change_lr with preferred direction based on y_actual
      2. Apply lapse-rate smoothing
      3. Sample feature and delta
      
    Returns:
        (feature, delta, time): Feature to change, change magnitude, elapsed time
    """
    direction = 'increase' if y_actual == 0 else 'decrease'
    
    out = recall_change_lr(memory, k=6, preferred_direction=direction)
    time = out.pop('expected_time')
    memory.tick(time)
    
    smoothed_out = smooth_probs_with_lapse(out, lapse=lapse)
    feat, delta = sample_from_probs(smoothed_out)
    
    return feat, delta, time


# ============================================================================
# SECTION 8: STRATEGY MAPPING & ENVIRONMENT SETUP
# ============================================================================
# RL environment integration and strategy orchestration.

# Strategy mapping: index -> function name used in RL environment
STRATEGIES = {
    0: "change_path_dt",
    1: "zero_out_lr_heuristic",
    2: "zero_out_lr_displayed",
    3: "recall_change_dt",
    4: "recall_change_lr"
}

# XAI explanation type mapping
XAI_TYPES = {
    0: "DT",       # Decision Tree only
    1: "LR",       # Logistic Regression only
    2: "DT+LR"     # Both (mixed)
}


def _make_memory(retrieval_threshold: float,
                latency_factor: float) -> CombinedMemory:
    """
    Factory function to create a configured memory system for an episode.
    
    Args:
        retrieval_threshold: Minimum activation for successful retrieval
        latency_factor: Scaling factor for retrieval time
        
    Returns:
        Initialized CombinedMemory ready for use
    """
    dm = DeclarativeMemory(
        retrieval_threshold=retrieval_threshold,
        latency_factor=latency_factor,
        latency_exponent=0.5,
        max_assoc_strength=2.0,
        mismatch_penalty=-2.0,
        activation_noise=0.3,
        decay=0.5,
    )
    return CombinedMemory(dm, wm_capacity=7)


def prepare_memory_for_dt(memory: CombinedMemory,
                         dt_exp: Any,
                         ai_loader: Any,
                         forward_trials: List[Dict[str, Any]],
                         bounds: Dict[str, Tuple[float, float]]):
    """
    Initialize memory with DT structure and learning from forward trials.
    
    Process:
      1. Store DT structure in memory
      2. Simulate DT traversal on forward trials to build up activation
      3. Refresh nodes traversed during XAI trials
    """
    add_dt_to_memory(memory, dt_exp)
    memory.tick(90)  # Time advance for BLL decay
    
    for trial in forward_trials:
        with_xai = trial['Tested w/ XAI']
        instance_id = trial['Instance Id']
        instances, preds = ai_loader.load_instances([instance_id], normalize=False)
        instance = instances[0]
        
        mode = "read" if with_xai == 1 else "retrieve"
        dt_traverse(instance, memory, dt_exp, mode=mode, T_enc=2, ddm_a=1.0, ddm_s=0.8)
        
        if mode == "read":
            refresh_dt_path_in_memory(memory, dt_exp, instance)


def prepare_memory_for_lr_heuristic(memory: CombinedMemory,
                                   lr_exp: Any,
                                   ai_loader: Any,
                                   forward_trials: List[Dict[str, Any]],
                                   bounds: Dict[str, Tuple[float, float]]):
    """
    Initialize memory with LR coefficients and learning from forward trials.
    
    Process:
      1. Store LR coefficient distributions in memory
      2. Simulate LR heuristic reasoning on forward trials
      3. Refresh coefficients based on prediction accuracy
    """
    add_lr_heuristic_to_memory(lr_exp, memory)
    memory.tick(90)  # Time advance for BLL decay
    
    for trial in forward_trials:
        instance_id = trial['Instance Id']
        instances, preds = ai_loader.load_instances([instance_id], normalize=True)
        instance = instances[0]
        
        p, t, info = lr_heuristic(instance, memory, lr_exp, T_enc=2, ddm_a=1.0, ddm_s=0.8)
        refresh_lr_heuristic_in_memory(memory, lr_exp, info, actual=trial['AI prediction'])


# ============================================================================
# PLACEHOLDERS FOR EXTERNAL CLASSES
# ============================================================================
# These are referenced by the reasoning functions but defined elsewhere

"""
LogisticRegressionInterpreter:
    Loads LR explanation from CSV, provides:
      - .intercept: Intercept term
      - .coefficients: OrderedDict of feature_key -> coefficient
      - ._format_feature(key): Format feature name for display
      - .apply_to_instance(x): Compute z = intercept + sum(coef_i * x_i)

DecisionTreeInterpreter:
    Loads DT explanation from CSV, provides:
      - .tree_structure: List of node dicts with fields:
          - "node": node ID
          - "is_leaf": bool
          - "feature": feature key (internal nodes)
          - "threshold": threshold (numeric internal nodes)
          - "left", "right": child node IDs
          - "value": class distribution (leaves)
      - .apply_to_instance(x): Traverse to leaf, return prediction

AIDatasetLoader:
    Loads feature values and predictions, provides:
      - .load_instances(ids, normalize=True): Load feature vectors and AI predictions
      - .get_bounds_for_app(app_id): Get per-feature (lo, hi) bounds
      - .get_categories_for_app(app_id): Get categorical feature levels
"""
