"""
Training utilities for RL agents.

Provides training managers and utilities for organizing model weights.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class WeightManifest:
    """Manifest for tracking model weights and metadata."""
    
    agent_id: str
    agent_type: str
    model_path: str
    training_steps: int
    mean_reward: Optional[float] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightManifest":
        """Create from dictionary."""
        return cls(**data)


class WeightOrganizer:
    """
    Organizes pre-trained model weights from old structure to unified API.
    
    Maps:
    - src/coxam/model_calculation/ → RL_agents/model_weights/lr_calculation/
    - src/coxam/model_dt/ → RL_agents/model_weights/dt/
    - src/coxam/model_heuristic/ → RL_agents/model_weights/lr_heuristic/
    - src/coxam/model_counterfactual/ → RL_agents/model_weights/counterfactual/
    """
    
    def __init__(self, workspace_root: str):
        """
        Initialize weight organizer.
        
        Args:
            workspace_root: Root of the xaik-tool-cognitive-agent workspace
        """
        self.workspace_root = Path(workspace_root)
        self.coxam_dir = self.workspace_root / "src" / "coxam"
        self.rl_agents_dir = self.coxam_dir / "RL_agents"
        self.weights_dir = self.rl_agents_dir / "model_weights"
    
    def organize_weights(self, copy: bool = True):
        """
        Organize weights from old structure to unified API.
        
        Args:
            copy: If True, copy files. If False, create symlinks.
        """
        # Mapping of old → new directories
        mappings = {
            "model_calculation": "lr_calculation",
            "model_dt": "dt",
            "model_heuristic": "lr_heuristic",
            "model_counterfactual": "counterfactual",
        }
        
        results = {}
        
        for old_name, new_name in mappings.items():
            old_path = self.coxam_dir / old_name
            new_path = self.weights_dir / new_name
            
            if old_path.exists():
                os.makedirs(new_path, exist_ok=True)
                
                # Copy or link all files
                for file in old_path.iterdir():
                    if file.is_file():
                        new_file = new_path / file.name
                        if copy:
                            shutil.copy2(file, new_file)
                        else:
                            os.symlink(file, new_file)
                
                results[old_name] = {
                    "status": "organized",
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                }
            else:
                results[old_name] = {
                    "status": "not_found",
                    "old_path": str(old_path),
                }
        
        return results
    
    def create_weight_manifest(
        self,
        manifests: List[WeightManifest]
    ):
        """
        Create and save manifest of all weights.
        
        Args:
            manifests: List of WeightManifest objects
        """
        manifest_path = self.weights_dir / "manifest.json"
        
        manifest_data = {
            "weights": [m.to_dict() for m in manifests],
            "organized_from_old_structure": True,
        }
        
        os.makedirs(self.weights_dir, exist_ok=True)
        
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        
        return manifest_path


class TrainingManager:
    """
    Manages training of RL agents for different strategies.
    
    Handles:
    - Environment setup
    - Training execution
    - Checkpoint management
    - Metrics logging
    """
    
    def __init__(
        self,
        agent_type: str,
        model_weights_dir: str,
        verbose: bool = True,
    ):
        """
        Initialize training manager.
        
        Args:
            agent_type: Type of agent to train ("dt", "lr_heuristic", "lr_calculation")
            model_weights_dir: Directory to save model weights
            verbose: Whether to print status messages
        """
        self.agent_type = agent_type
        self.model_weights_dir = Path(model_weights_dir)
        self.verbose = verbose
        
        os.makedirs(self.model_weights_dir, exist_ok=True)
        
        self.training_log = {
            "agent_type": agent_type,
            "training_runs": [],
        }
    
    def log_training_run(
        self,
        run_id: str,
        total_timesteps: int,
        mean_reward: float,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a training run.
        
        Args:
            run_id: Unique identifier for this training run
            total_timesteps: Total timesteps trained
            mean_reward: Mean reward achieved
            metrics: Optional additional metrics
        """
        run_entry = {
            "run_id": run_id,
            "total_timesteps": total_timesteps,
            "mean_reward": mean_reward,
            "metrics": metrics or {},
        }
        
        self.training_log["training_runs"].append(run_entry)
        
        log_path = self.model_weights_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        
        if self.verbose:
            print(
                f"✓ Logged training run {run_id}: "
                f"{total_timesteps} steps, mean_reward={mean_reward:.3f}"
            )
    
    def get_best_checkpoint(
        self, metric: str = "mean_reward"
    ) -> Optional[str]:
        """
        Get path to best checkpoint based on metric.
        
        Args:
            metric: Metric to optimize ("mean_reward", etc.)
            
        Returns:
            Path to best model checkpoint or None
        """
        if not self.training_log["training_runs"]:
            return None
        
        best_run = max(
            self.training_log["training_runs"],
            key=lambda r: r.get(metric, float("-inf")),
        )
        
        checkpoint_path = self.model_weights_dir / f"{best_run['run_id']}.zip"
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        return None
