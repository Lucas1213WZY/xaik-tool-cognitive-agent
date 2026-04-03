"""
Core module - Central abstractions for cognitive agent reasoning.

Provides unified interfaces for:
  - Memory management (exemplar and ACT-R backends)
  - Reasoning strategy orchestration
  - Configuration management
"""

from .memory import (
    UnifiedMemory,
    MemoryConfig,
    MemoryBackend,
    Exemplar,
    Chunk,
    ReasoningContext,
    ExemplarMemory,
    ACTRMemory,
)

__all__ = [
    "UnifiedMemory",
    "MemoryConfig",
    "MemoryBackend",
    "Exemplar",
    "Chunk",
    "ReasoningContext",
    "ExemplarMemory",
    "ACTRMemory",
]
