"""
Experiment Runner - Base class for conducting simulation experiments.

Provides orchestration for:
1. Data loading and preprocessing
2. Parameter sampling
3. User simulation (CoAX or CoXAM paths)
4. Results collection and export
5. Evaluation metrics
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    
    # Dataset and data setup
    dataset_name: str = "wine_quality"
    n_samples: int = 100
    
    # Simulation parameters
    n_participants: int = 50
    n_trials: int = 40
    reasoning_model: str = "coxam"  # "coax" or "coxam"
    
    # XAI settings
    xai_type: str = "Importance"
    with_explanations: bool = True
    
    # Output
    output_dir: str = "experiment_results"
    save_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for export."""
        return {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_participants": self.n_participants,
            "n_trials": self.n_trials,
            "reasoning_model": self.reasoning_model,
            "xai_type": self.xai_type,
            "with_explanations": self.with_explanations,
        }


class ExperimentRunner:
    """
    Base class for running user simulation experiments.
    
    Orchestrates the complete workflow:
    1. Load and prepare data
    2. Sample/estimate cognitive parameters
    3. Run user simulation (CoAX or CoXAM)
    4. Collect and export results
    5. Compute evaluation metrics
    
    Usage:
        config = ExperimentConfig(
            dataset_name='wine_quality',
            reasoning_model='coxam',
            n_participants=50
        )
        runner = ExperimentRunner(config)
        results = runner.run()
        metrics = runner.evaluate()
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: ExperimentConfig instance
        """
        self.config = config
        self.results = []
        self.metrics = {}
        
        # Setup output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ExperimentRunner with config: {config}")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the complete experiment.
        
        Returns:
            List of result dictionaries (one per trial)
        """
        logger.info(f"Starting experiment: {self.config.dataset_name} with {self.config.reasoning_model}")
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        # data = self._load_data()
        
        # Step 2: Estimate or sample parameters
        logger.info("Step 2: Estimating cognitive parameters...")
        # parameters = self._estimate_parameters()
        
        # Step 3: Run simulation
        logger.info("Step 3: Running user simulation...")
        # self.results = self._run_simulation(data, parameters)
        
        # Step 4: Save results
        if self.config.save_results:
            logger.info("Step 4: Saving results...")
            self._save_results()
        
        logger.info("Experiment completed successfully")
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics on results.
        
        Returns:
            Dictionary of metric name -> value
        """
        if not self.results:
            logger.warning("No results to evaluate - run experiment first")
            return {}
        
        logger.info("Computing evaluation metrics...")
        
        # Placeholder metrics
        self.metrics = {
            "n_trials": len(self.results),
            "n_participants": self.config.n_participants,
            "dataset": self.config.dataset_name,
            "reasoning_model": self.config.reasoning_model,
        }
        
        return self.metrics
    
    def _load_data(self) -> Dict[str, Any]:
        """Load and prepare dataset."""
        # Implementation will depend on data_loaders API
        pass
    
    def _estimate_parameters(self) -> Dict[str, Any]:
        """Estimate cognitive parameters from data."""
        # Implementation will use ParameterSampler
        pass
    
    def _run_simulation(self, data: Dict, parameters: Dict) -> List[Dict]:
        """Run user simulation with given data and parameters."""
        # Implementation will use TrialSimulator or SessionGenerator
        pass
    
    def _save_results(self) -> None:
        """Save experiment results to disk."""
        results_file = self.output_path / "results.json"
        config_file = self.output_path / "config.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Config saved to {config_file}")


def run_experiment(config: ExperimentConfig) -> ExperimentRunner:
    """
    Convenience function to run an experiment.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        Completed ExperimentRunner instance
    """
    runner = ExperimentRunner(config)
    runner.run()
    runner.evaluate()
    return runner


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(
        dataset_name="wine_quality",
        reasoning_model="coxam",
        n_participants=10,
        n_trials=20
    )
    
    runner = run_experiment(config)
    print(runner.metrics)
