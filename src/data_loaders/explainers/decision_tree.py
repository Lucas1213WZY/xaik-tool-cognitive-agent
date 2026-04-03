"""Decision Tree explainer implementation from CoXAM."""

from typing import Dict, List, Any, Union
import json
import numpy as np
import pandas as pd
from ..base.explainer import BaseExplainer


class DecisionTreeExplainer(BaseExplainer):
    """
    Decision Tree explainer for surrogate models.
    Loads tree structure and applies it to predict class labels.
    
    From: src/coxam/utils.py DecisionTreeInterpreter
    """

    def __init__(self, explanation_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                 app_id: str, model_name: str, depth: int = 3):
        """
        Initialize Decision Tree explainer.
        
        Args:
            explanation_df: DataFrame with tree structures (has 'tree_structure' column)
            metadata_df: DataFrame with feature metadata
            app_id: Dataset identifier
            model_name: Model name
            depth: Tree depth (primarily for filtering/documentation)
        """
        super().__init__(
            explainer_type='decision_tree',
            metadata={
                'app_id': app_id,
                'model_name': model_name,
                'depth': depth
            }
        )
        
        # Find the correct row
        row = explanation_df[explanation_df['appId'] == app_id]
        if row.empty:
            raise ValueError(f"No decision tree explanation found for appId: {app_id}")
        self.explanation_row = row.iloc[0]

        # Get metadata
        meta_row = metadata_df[metadata_df['appId'] == app_id]
        if meta_row.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        self.metadata_row = meta_row.iloc[0]

        self.app_id = app_id
        self.model = model_name
        self.fidelity = float(self.explanation_row.get('fidelity', 0.0))
        self.tree_structure = json.loads(self.explanation_row['tree_structure'])
        
        # Load class labels if available
        if "class_labels" in self.explanation_row:
            try:
                self.class_labels = json.loads(self.explanation_row["class_labels"])
            except Exception:
                self.class_labels = None
        else:
            self.class_labels = None

    def _format_feature(self, feature_key: str) -> str:
        """Format feature key with human-readable names."""
        if '=' in feature_key:
            base, cat_index = feature_key.split('=')
            v_index = base.replace('a', 'v')
            cat_col = f"{v_index}_{cat_index}"
            feat_name = self.metadata_row.get(base, base)
            cat_label = self.metadata_row.get(cat_col, f"Category {cat_index}")
            return f"{feat_name} = {cat_label}"
        else:
            return self.metadata_row.get(feature_key, feature_key)

    def print_tree(self, as_name: bool = False) -> None:
        """
        Print the tree structure (for debugging).
        
        Args:
            as_name: If True, use feature names instead of raw keys
        """
        print(f"Decision Tree (appId={self.app_id}, model={self.model})")
        print(f"Fidelity: {self.fidelity:.4f}")
        self._print_node(0, 0, as_name)

    def _print_node(self, node_id: int, depth: int, as_name: bool) -> None:
        """Recursively print tree nodes."""
        node = next(n for n in self.tree_structure if n["node"] == node_id)
        prefix = "  " * depth
        if node["is_leaf"]:
            class_id = int(np.argmax(node["value"]))
            class_label = (
                self.class_labels[class_id]
                if self.class_labels and class_id < len(self.class_labels)
                else f"class {class_id}"
            )
            probs = np.round(node["value"], 4)
            print(f"{prefix}→ Predict {class_label} (probs: {probs})")
        else:
            feature = self._format_feature(node['feature']) if as_name else node['feature']
            print(f"{prefix}if {feature} <= {node['threshold']}:")
            self._print_node(node["left"], depth + 1, as_name)
            print(f"{prefix}else:")
            self._print_node(node["right"], depth + 1, as_name)

    def apply(self, raw_input: Union[List, np.ndarray]) -> Dict[str, Any]:
        """
        Apply the tree to a single instance.
        
        Args:
            raw_input: Feature vector (raw, unnormalized)
            
        Returns:
            Dict with 'probs', 'class_index', 'class_label'
        """
        if isinstance(raw_input, list):
            raw_input = np.array(raw_input)
            
        node = next(n for n in self.tree_structure if n["node"] == 0)
        while not node["is_leaf"]:
            feature_key = node["feature"]
            if '=' in feature_key:
                base, cat_idx = feature_key.split('=')
                col_idx = int(base[1:])
                val = 1.0 if int(raw_input[col_idx]) == int(cat_idx) else 0.0
            else:
                col_idx = int(feature_key[1:])
                val = raw_input[col_idx]

            if val <= node["threshold"]:
                node = next(n for n in self.tree_structure if n["node"] == node["left"])
            else:
                node = next(n for n in self.tree_structure if n["node"] == node["right"])

        class_index = int(np.argmax(node["value"]))
        return {
            "probs": node["value"],
            "class_index": class_index,
            "class_label": self.class_labels[class_index] if self.class_labels else None
        }

    def apply_batch(self, instances: List[Union[List, np.ndarray]]) -> List[Dict[str, Any]]:
        """Apply the tree to multiple instances."""
        return [self.apply(inst) for inst in instances]

    def get_tree_structure(self) -> Dict[str, Any]:
        """Get the underlying tree structure."""
        return self.tree_structure
