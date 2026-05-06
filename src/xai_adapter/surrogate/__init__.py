"""Surrogate XAI methods for rules-vs-weights comparisons."""

from .base import SurrogateMethod
from .decision_tree import DecisionTreeSurrogateMethod
from .generator import (
    GeneratedSurrogateMethods,
    generate_decision_tree_table,
    generate_logistic_regression_table,
    generate_surrogate_tables,
)
from .logistic_regression import LogisticRegressionSurrogateMethod

__all__ = [
    "SurrogateMethod",
    "DecisionTreeSurrogateMethod",
    "LogisticRegressionSurrogateMethod",
    "GeneratedSurrogateMethods",
    "generate_decision_tree_table",
    "generate_logistic_regression_table",
    "generate_surrogate_tables",
]
