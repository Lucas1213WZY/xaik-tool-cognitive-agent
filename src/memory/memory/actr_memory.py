"""
ACT-R-based memory backend for CoXAM.

Implements probabilistic activation using Base-Level Learning (BLL),
associative strength, and partial matching for chunk retrieval.
"""

from typing import List, Optional, Tuple, Any, Dict, Union, Set
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from .interface import MemoryInterface, Chunk, ReasoningContext, MemoryBackend
from .utils import (
    base_level_learning, compute_similarity_activation, add_activation_noise,
    compute_retrieval_latency, compute_chunk_similarity, get_timestamp_diff
)


class ACTRMemory(MemoryInterface):
    """
    ACT-R-based probabilistic memory backend for CoXAM.
    
    Storage: Hierarchical chunks with BLL tracking
    Retrieval: Probabilistic based on activation
    Activation: BLL + Associative Strength + Partial Matching
    
    Key characteristics:
    - Chunks store structured knowledge (slots)
    - Base-Level Learning (BLL): ln(sum(t_i^-d))
    - Associative strength: w_j * s_ji (spreading activation)
    - Partial matching: mismatch penalties
    - Stochastic retrieval with latency variability
    """
    
    def __init__(self, context: ReasoningContext):
        super().__init__(context)
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_retrievals: Dict[str, List[float]] = defaultdict(list)
        self.activation = dict()
        self.time = context.current_time or 0.0
        
        # Associative links: source_chunk -> {target_chunk -> strength}
        self.associations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Working memory queue for active chunks
        self.working_memory: deque = deque(maxlen=context.wm_capacity)
        
        # Validate backend
        if self.context.backend != MemoryBackend.ACTR:
            raise ValueError(f"ACTRMemory requires backend=ACTR, got {self.backend}")
    
    def store(self, key: str, value: Union[Chunk, None]) -> None:
        """
        Store a chunk in declarative memory.
        
        Args:
            key: Unique chunk identifier
            value: Chunk object
        """
        if not isinstance(value, Chunk):
            raise TypeError(f"ACTRMemory.store() expects Chunk, got {type(value)}")
        
        self.chunks[key] = value
        self.chunk_retrievals[key] = [self.time]  # Initial encoding
        
        # Add to working memory
        self.working_memory.append(key)
    
    def retrieve(self, query: Any, k: int = 1,
                 similarity_threshold: Optional[float] = None) -> List[Tuple[str, float, Chunk]]:
        """
        Retrieve top-k chunks matching query using probabilistic activation.
        
        Activation = A_i = B_i + sum_j(S_ji) + sum_k(W_k * S_ik)
        where:
          B_i = Base-Level Learning
          S_ji = Associative strength from source
          W_k * S_ik = Partial matching with mismatch penalty
        
        Args:
            query: Query chunk or slots dict for matching
            k: Number of chunks to retrieve
            similarity_threshold: Minimum activation (default: retrieval_threshold)
            
        Returns:
            List of (key, activation_score, chunk) tuples, sorted by activation desc
        """
        if not self.chunks:
            return []
        
        if similarity_threshold is None:
            similarity_threshold = self.context.retrieval_threshold
        
        # Extract query slots
        if isinstance(query, Chunk):
            query_slots = query.slots
        elif isinstance(query, dict):
            query_slots = query
        else:
            raise TypeError(f"Query must be Chunk or dict, got {type(query)}")
        
        activations = []
        
        for key, chunk in self.chunks.items():
            # 1. Base-Level Learning
            retrieval_times = self.chunk_retrievals.get(key, [])
            ref_count = len(retrieval_times) - 1  # Exclude initial encoding for some ACT-R implementations
            if ref_count < 0:
                ref_count = 1
            bll = base_level_learning(ref_count, self.context.decay_param)
            
            # 2. Associative Strength (from working memory sources)
            assoc_strength = 0.0
            for source_key in self.working_memory:
                if source_key in self.associations and key in self.associations[source_key]:
                    strength = self.associations[source_key][key]
                    assoc_strength += strength
            
            # 3. Partial Matching (similarity-based activation bonus/penalty)
            partial_match = self._compute_partial_match(query_slots, chunk.slots)
            
            # Combined activation
            activation = bll + assoc_strength + partial_match
            
            # Add retrieval noise if specified
            activation = add_activation_noise(activation, self.context.activation_noise)
            
            # Apply retrieval threshold
            if activation < similarity_threshold:
                continue
            
            activations.append((key, activation, chunk))
        
        # Sort by activation descending
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # Update retrievals for selected chunks
        for key, _, _ in activations[:k]:
            self.chunk_retrievals[key].append(self.time)
        
        return activations[:k]
    
    def retrieve_with_latency(self, query: Any, k: int = 1) -> Tuple[List[Tuple[str, Chunk]], float]:
        """
        Retrieve chunks and compute access latency.
        
        ACT-R latency: RT = F * exp(-activation)
        
        Returns:
            Tuple of ([(key, chunk), ...], latency_ms)
        """
        retrieved = self.retrieve(query, k)
        
        if not retrieved:
            return [], 0.0
        
        # Use highest activation for latency
        best_activation = retrieved[0][1]
        latency = compute_retrieval_latency(best_activation, self.context.latency_factor)
        
        result = [(key, chunk) for key, _, chunk in retrieved]
        return result, latency
    
    def get(self, key: str) -> Optional[Chunk]:
        """Get a specific chunk by key."""
        return self.chunks.get(key)
    
    def update_activation(self, key: str, increase: float) -> None:
        """
        Update chunk activation on retrieval.
        
        In ACT-R, this updates BLL by recording a new retrieval time.
        """
        if key in self.chunks:
            # Record retrieval at current time
            self.chunk_retrievals[key].append(self.time)
            
            # Update chunk's activation tracking
            chunk = self.chunks[key]
            chunk.reference_count += 1
            new_activation = base_level_learning(chunk.reference_count, self.context.decay_param)
            chunk.activations.append(new_activation)
    
    def clear(self) -> None:
        """Clear all chunks and associations."""
        self.chunks.clear()
        self.chunk_retrievals.clear()
        self.associations.clear()
        self.working_memory.clear()
        self.activation.clear()
    
    def get_size(self) -> int:
        """Get number of stored chunks."""
        return len(self.chunks)
    
    def export_state(self) -> Dict[str, Any]:
        """Export memory state for inspection/debugging."""
        return {
            "backend": self.backend.value,
            "current_time": self.time,
            "chunks_count": len(self.chunks),
            "chunks": {
                key: {
                    "chunk_type": chunk.chunk_type,
                    "creation_time": chunk.creation_time,
                    "reference_count": chunk.reference_count,
                    "activations_count": len(chunk.activations),
                    "slots_keys": list(chunk.slots.keys())
                }
                for key, chunk in self.chunks.items()
            },
            "associations_count": sum(len(v) for v in self.associations.values()),
            "working_memory_size": len(self.working_memory),
            "context": {
                "decay_param": self.context.decay_param,
                "retrieval_threshold": self.context.retrieval_threshold,
                "latency_factor": self.context.latency_factor,
                "wm_capacity": self.context.wm_capacity,
                "mismatch_penalty": self.context.mismatch_penalty
            }
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state (restoration)."""
        self.time = state.get("current_time", 0.0)
        # Note: Restoring chunks requires access to original slot data
    
    def add_association(self, source_key: str, target_key: str, strength: float) -> None:
        """
        Add associative link between chunks.
        
        Args:
            source_key: Chunk that activates the association
            target_key: Target chunk that benefits from association
            strength: Associative strength (typically 0-2)
        """
        strength = min(strength, self.context.max_assoc_strength)
        self.associations[source_key][target_key] = strength
    
    def update_time(self, new_time: float) -> None:
        """Update internal time (used for BLL decay calculations)."""
        self.time = new_time
    
    def _compute_partial_match(self, query_slots: Dict[str, Any], 
                                chunk_slots: Dict[str, Any]) -> float:
        """
        Compute partial matching bonus/penalty.
        
        Compares query slots to chunk slots, applying mismatch penalties
        for non-matches and similarity bonuses for matches.
        
        Returns:
            Matching score (can be negative if many mismatches)
        """
        if not query_slots:
            return 0.0
        
        slot_weights = {}  # Equal weights by default
        similarity = compute_chunk_similarity(query_slots, chunk_slots, slot_weights)
        
        # Convert similarity [0,1] to activation impact
        # Full match (1.0) -> bonus, partial/no match -> penalty
        matching_bonus = similarity * self.context.max_assoc_strength
        
        # Penalty for query slots not in chunk
        query_only = set(query_slots.keys()) - set(chunk_slots.keys())
        penalty = len(query_only) * self.context.mismatch_penalty
        
        return matching_bonus - penalty
    
    def get_working_memory(self) -> List[str]:
        """Get IDs of chunks currently in working memory."""
        return list(self.working_memory)
    
    def get_chunk_history(self, key: str) -> Dict[str, Any]:
        """Get detailed history for a chunk."""
        if key not in self.chunks:
            return {}
        
        chunk = self.chunks[key]
        retrievals = self.chunk_retrievals.get(key, [])
        
        return {
            "chunk_id": key,
            "chunk_type": chunk.chunk_type,
            "creation_time": chunk.creation_time,
            "retrieval_times": retrievals,
            "retrieval_count": len(retrievals),
            "activation_history": chunk.activations
        }
