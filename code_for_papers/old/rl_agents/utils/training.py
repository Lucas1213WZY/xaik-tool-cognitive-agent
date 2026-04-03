"""
Training Utilities for RL Agents

Training orchestration, logging, and monitoring.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class TrainingLog:
    """Log entry for training session."""
    
    timestamp: str
    episode: int
    timestep: int
    mean_reward: float
    std_reward: float
    episode_length: float
    learning_rate: Optional[float] = None
    loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "timestamp": self.timestamp,
            "episode": self.episode,
            "timestep": self.timestep,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "episode_length": self.episode_length,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
        }


class TrainingManager:
    """
    Manages training runs with logging and monitoring.
    
    Features:
    - Training run orchestration
    - Metrics logging
    - Checkpoint management
    - Training history export
    """
    
    def __init__(self, agent: Any, log_dir: str = "./training_logs"):
        """
        Initialize TrainingManager.
        
        Args:
            agent: RL agent instance
            log_dir: Directory for logging
        """
        self.agent = agent
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.logs: List[TrainingLog] = []
        self.checkpoints: List[str] = []
        
        # Setup logger
        log_file = os.path.join(self.run_dir, "training.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    
    def train(self, 
              n_envs: int = 4,
              total_timesteps: int = 100_000,
              eval_freq: int = 5_000,
              n_eval_episodes: int = 10) -> Dict[str, Any]:
        """
        Execute training run.
        
        Args:
            n_envs: Number of parallel environments
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
        
        Returns:
            Training results
        """
        logger.info(f"Starting training run {self.run_id}")
        logger.info(f"Total timesteps: {total_timesteps}")
        
        start_time = datetime.now()
        
        try:
            results = self.agent.train(
                n_envs=n_envs,
                total_timesteps=total_timesteps
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log results
            training_result = {
                "run_id": self.run_id,
                "success": results.get("success", False),
                "total_timesteps": total_timesteps,
                "duration_sec": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "eval_results": results.get("eval_results", {}),
            }
            
            # Save results
            self._save_results(training_result)
            
            logger.info(f"Training complete. Duration: {duration}s")
            return training_result
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "run_id": self.run_id,
                "success": False,
                "error": str(e),
            }
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional metadata
        
        Returns:
            Success flag
        """
        try:
            self.agent.save(path, include_metadata=True)
            self.checkpoints.append(path)
            
            if metadata:
                with open(path + ".metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved: {path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save training results to JSON."""
        results_file = os.path.join(self.run_dir, "results.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def export_history(self, format: str = "json") -> str:
        """
        Export training history.
        
        Args:
            format: Export format ("json" or "csv")
        
        Returns:
            Path to exported file
        """
        if format == "json":
            output_file = os.path.join(self.run_dir, "history.json")
            with open(output_file, "w") as f:
                history = [log.to_dict() for log in self.logs]
                json.dump(history, f, indent=2)
        
        elif format == "csv":
            output_file = os.path.join(self.run_dir, "history.csv")
            import csv
            
            if self.logs:
                with open(output_file, "w", newline="") as f:
                    fieldnames = list(self.logs[0].to_dict().keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for log in self.logs:
                        writer.writerow(log.to_dict())
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"History exported to {output_file}")
        return output_file
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training run summary."""
        return {
            "run_id": self.run_id,
            "num_logs": len(self.logs),
            "num_checkpoints": len(self.checkpoints),
            "run_dir": self.run_dir,
            "agent_config": {
                "name": self.agent.config.agent_name,
                "learning_rate": self.agent.config.learning_rate,
            },
        }
