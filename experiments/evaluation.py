"""
Evaluation Metrics - Compute metrics for assessing XAI and simulation quality.

Provides tools for:
1. Accuracy and response metrics
2. XAI effectiveness measures
3. Memory and cognitive modeling quality
4. Comparison across reasoning strategies
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics from simulation results.
    
    Supports:
    - Prediction accuracy
    - Response time analysis
    - XAI impact assessment
    - Memory effects evaluation
    - CoAX vs CoXAM comparison
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize evaluation with results.
        
        Args:
            results: List of trial results from user simulation
        """
        self.results = results
        self.metrics = {}
        self._compute_all()
    
    def _compute_all(self) -> None:
        """Compute all available metrics."""
        if not self.results:
            logger.warning("No results provided - metrics will be empty")
            return
        
        self.metrics = {
            "n_trials": len(self.results),
            "n_participants": len(set(r.get("participant_id") for r in self.results)),
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all computed metrics."""
        return self.metrics
    
    def accuracy_metrics(self) -> Dict[str, float]:
        """
        Compute accuracy-based metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall
        """
        # Placeholder implementation
        return {"accuracy": 0.0}
    
    def response_time_metrics(self) -> Dict[str, float]:
        """
        Compute response time statistics.
        
        Returns:
            Dictionary with mean, std, min, max response times
        """
        response_times = [
            r.get("response_time", 0) 
            for r in self.results 
            if "response_time" in r
        ]
        
        if not response_times:
            return {"mean_rt": 0, "std_rt": 0}
        
        return {
            "mean_rt": float(np.mean(response_times)),
            "std_rt": float(np.std(response_times)),
            "min_rt": float(np.min(response_times)),
            "max_rt": float(np.max(response_times)),
        }
    
    def xai_effectiveness(self) -> Dict[str, float]:
        """
        Assess XAI effectiveness (e.g., improved accuracy with explanations).
        
        Returns:
            Dictionary with XAI impact metrics
        """
        # Placeholder: would compare with/without XAI conditions
        return {"xai_impact": 0.0}
    
    def memory_quality(self) -> Dict[str, float]:
        """
        Evaluate quality of memory effects in simulation.
        
        Returns:
            Dictionary with memory modeling metrics
        """
        # Placeholder: would evaluate ACT-R or exemplar memory quality
        return {"memory_fit": 0.0}
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EvaluationMetrics({len(self.results)} trials, {self.metrics})"


class ComparisonMetrics:
    """
    Compare metrics across different conditions (e.g., CoAX vs CoXAM).
    """
    
    def __init__(self):
        """Initialize comparison metrics."""
        self.groups = {}
    
    def add_group(self, name: str, metrics: EvaluationMetrics) -> None:
        """Add a metrics group for comparison."""
        self.groups[name] = metrics
    
    def compare(self) -> Dict[str, Any]:
        """
        Compare metrics across all groups.
        
        Returns:
            Dictionary with comparisons
        """
        if not self.groups:
            return {}
        
        comparisons = {}
        for name, metrics in self.groups.items():
            comparisons[name] = metrics.summary()
        
        return comparisons


if __name__ == "__main__":
    # Example usage
    sample_results = [
        {"participant_id": "p001", "trial": 1, "response_time": 2.5, "accuracy": 1.0},
        {"participant_id": "p001", "trial": 2, "response_time": 2.3, "accuracy": 1.0},
        {"participant_id": "p002", "trial": 1, "response_time": 3.1, "accuracy": 0.8},
    ]
    
    metrics = EvaluationMetrics(sample_results)
    print(metrics.summary())
    print(metrics.response_time_metrics())
