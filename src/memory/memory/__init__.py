"""
Unified Memory Module - CoAX and CoXAM Memory Backend Consolidation

This module provides a single, configuration-driven abstraction for managing
two different cognitive memory systems:
  - CoAX: Exemplar-based, temporal decay (simple, fast)
  - CoXAM: ACT-R inspired, probabilistic (complex, realistic)

Public API:
  - MemoryConfig: Configuration dataclass with presets for CoAX/CoXAM
  - UnifiedMemory: Main factory class for backend-agnostic access
  - MemoryBackend: Enum for backend selection
  - Exemplar, Chunk: Data structures for CoAX and CoXAM
  - ReasoningContext: Full configuration context

Usage Examples:

  # CoAX (exemplar) backend
  from src.memory.memory import UnifiedMemory
  memory = UnifiedMemory.create_for_coax(decay_param=0.3)
  memory.store("ex1", Exemplar(...))
  retrieved = memory.retrieve(query, k=5)
  
  # CoXAM (ACT-R) backend
  memory = UnifiedMemory.create_for_coxam(wm_capacity=6)
  memory.store("chunk1", Chunk(...))
  retrieved, latency = memory.retrieve_with_latency(query, k=3)
"""

from .interface import (
    MemoryInterface,
    MemoryBackend,
    Exemplar,
    Chunk,
    ReasoningContext,
    ActivationFunction,
    SimilarityFunction,
)

from .unified_memory import UnifiedMemory, MemoryConfig

from .exemplar_memory import ExemplarMemory

from .actr_memory import ACTRMemory

from .utils import (
    euclidean_distance,
    cosine_similarity,
    temporal_decay,
    base_level_learning,
    compute_retrieval_latency,
    normalize_probabilities,
    compute_chunk_similarity,
)

__all__ = [
    # Main classes
    "UnifiedMemory",
    "MemoryConfig",
    "ExemplarMemory",
    "ACTRMemory",
    
    # Interfaces
    "MemoryInterface",
    "ActivationFunction",
    "SimilarityFunction",
    
    # Data structures
    "Exemplar",
    "Chunk",
    "ReasoningContext",
    
    # Enums
    "MemoryBackend",
    
    # Utilities
    "euclidean_distance",
    "cosine_similarity",
    "temporal_decay",
    "base_level_learning",
    "compute_retrieval_latency",
    "normalize_probabilities",
    "compute_chunk_similarity",
]
