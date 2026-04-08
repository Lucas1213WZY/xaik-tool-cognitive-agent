"""Plugin registry for explainers."""

from typing import Dict, Any, Type, Callable, Optional
from ..base import BaseExplainer


class ExplainerRegistry:
    """
    Registry for managing explainer implementations.
    Supports registration and retrieval of explainer types.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._registry: Dict[str, Type[BaseExplainer]] = {}
        self._instances: Dict[str, BaseExplainer] = {}

    def register(self, name: str, explainer_class: Type[BaseExplainer], 
                 aliases: list = None) -> None:
        """
        Register an explainer type.
        
        Args:
            name: Primary name for the explainer
            explainer_class: Explainer class (must inherit from BaseExplainer)
            aliases: Optional list of alternative names
        """
        if not issubclass(explainer_class, BaseExplainer):
            raise TypeError(f"{explainer_class} must inherit from BaseExplainer")
        
        self._registry[name] = explainer_class
        if aliases:
            for alias in aliases:
                self._registry[alias] = explainer_class

    def get_class(self, name: str) -> Type[BaseExplainer]:
        """
        Get explainer class by name.
        
        Args:
            name: Explainer name (case-insensitive)
            
        Returns:
            Explainer class
            
        Raises:
            ValueError if not found
        """
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Explainer '{name}' not found. Available: {available}"
            )
        return self._registry[name_lower]

    def create(self, name: str, **kwargs) -> BaseExplainer:
        """
        Create an explainer instance.
        
        Args:
            name: Explainer name
            **kwargs: Arguments passed to explainer constructor
            
        Returns:
            Instantiated explainer
        """
        explainer_class = self.get_class(name)
        return explainer_class(**kwargs)

    def list_available(self) -> list:
        """Get list of registered explainer names (without duplicates)."""
        return list(set(self._registry.keys()))

    def is_registered(self, name: str) -> bool:
        """Check if an explainer is registered."""
        return name.lower() in self._registry

    def __repr__(self) -> str:
        available = list(set(self._registry.keys()))
        return f"ExplainerRegistry(registered={len(available)}, types={available})"
