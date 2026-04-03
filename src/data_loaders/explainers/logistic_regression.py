"""Logistic Regression explainer implementation from CoXAM."""

from typing import Dict, List, Any, Union
from collections import OrderedDict
import numpy as np
import pandas as pd
from ..base.explainer import BaseExplainer


class LogisticRegressionExplainer(BaseExplainer):
    """
    Logistic Regression explainer for surrogate models.
    Loads and applies normalized/unnormalized coefficients.
    
    From: src/coxam/utils.py LogisticRegressionInterpreter
    Handles complex unnormalization logic for continuous and categorical features.
    """

    def __init__(self, explanation_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                 app_id: str, model_name: str, variant: str = "dense"):
        """
        Initialize Logistic Regression explainer.
        
        Args:
            explanation_df: DataFrame with LR coefficients
            metadata_df: DataFrame with feature metadata and normalization bounds
            app_id: Dataset identifier
            model_name: Model name
            variant: 'sparse' or 'dense' - determines feature subset
        """
        super().__init__(
            explainer_type='logistic_regression',
            metadata={
                'app_id': app_id,
                'model_name': model_name,
                'variant': variant
            }
        )
        
        # Find the correct row
        row = explanation_df[(explanation_df['appId'] == app_id) & 
                             (explanation_df.get('variant', 'dense') == variant)]
        if row.empty:
            # Fallback: if variant doesn't match, try first row with app_id
            row = explanation_df[explanation_df['appId'] == app_id]
            if row.empty:
                raise ValueError(
                    f"No logistic regression explanation found for appId: {app_id}, "
                    f"variant: {variant}"
                )
        self.explanation_row = row.iloc[0]

        # Get metadata
        meta_row = metadata_df[metadata_df['appId'] == app_id]
        if meta_row.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        self.metadata_row = meta_row.iloc[0]

        self.app_id = app_id
        self.model = model_name
        self.variant = variant
        self.fidelity = float(self.explanation_row.get('fidelity', 0.0))
        self._intercept_norm = float(self.explanation_row.get('intercept', 0.0))

        # Parse and unnormalize coefficients
        self._parse_coefficients()

    def _parse_coefficients(self) -> None:
        """Parse and unnormalize coefficients from normalized space to raw space."""
        # Collect normalized-space coefficients in buckets by base index
        coef_keys = [
            k for k in self.explanation_row.index
            if k.startswith("coef_") and pd.notna(self.explanation_row[k])
        ]

        # coef_a{idx} (continuous) or coef_a{idx}={cat} (categorical)
        buckets = {}
        for k in coef_keys:
            val = float(self.explanation_row[k])
            name = k.replace("coef_", "")
            if "=" in name:
                base, cat = name.split("=")
                idx = int(base[1:])
                buckets.setdefault(idx, []).append(("cat", int(cat), val, name))
            else:
                idx = int(name[1:])
                buckets.setdefault(idx, []).append(("cont", None, val, name))

        # Collapse to RAW space (unnormalized inputs)
        icpt = float(self._intercept_norm)
        raw_coef_map = {}

        for idx, items in buckets.items():
            conts = [it for it in items if it[0] == "cont"]
            cats = [it for it in items if it[0] == "cat"]

            if len(conts) == 1 and len(cats) == 0:
                # Continuous feature
                c_norm = conts[0][2]
                vmin_key = f"v{idx}_min"
                vmax_key = f"v{idx}_max"
                vmin = self.metadata_row.get(vmin_key, None)
                vmax = self.metadata_row.get(vmax_key, None)

                if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                    raw_coef_map[f"a{idx}"] = c_norm
                else:
                    scale = (vmax - vmin)
                    c_raw = c_norm / scale
                    icpt -= (vmin * c_norm / scale)
                    raw_coef_map[f"a{idx}"] = c_raw

            elif len(cats) == 2 and all(c[1] in (0, 1) for c in cats):
                # Binary categorical
                c0 = next((v for (_t, cat, v, _n) in cats if cat == 0), None)
                c1 = next((v for (_t, cat, v, _n) in cats if cat == 1), None)
                if c0 is not None and c1 is not None and np.isfinite(c0) and np.isfinite(c1):
                    icpt += c0
                    raw_coef_map[f"a{idx}"] = (c1 - c0)
                else:
                    for (_t, cat, v, _n) in cats:
                        raw_coef_map[f"a{idx}={cat}"] = v
            else:
                # Multi-category or unusual
                for (_t, cat, v, _n) in items:
                    if cat is None:
                        raw_coef_map[f"a{idx}"] = v
                    else:
                        raw_coef_map[f"a{idx}={cat}"] = v

        # Store in ordered form
        def sort_key(k):
            if "=" in k:
                base, cat = k.split("=")
                return (int(base[1:]), 1, int(cat), k)
            return (int(k[1:]), 0, -1, k)

        ordered_keys = sorted(raw_coef_map.keys(), key=sort_key)
        self.intercept = float(icpt)
        self.coefficients = OrderedDict((k, raw_coef_map[k]) for k in ordered_keys)

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

    def print_model(self, as_name: bool = False) -> None:
        """
        Print the model coefficients (for debugging).
        
        Args:
            as_name: If True, use feature names instead of raw keys
        """
        print(f"Logistic Regression (appId={self.app_id}, model={self.model})")
        print(f"Fidelity: {self.fidelity:.4f}")
        print(f"Intercept (raw): {self.intercept:.6g}")
        print("Coefficients (raw inputs):")
        for key, val in self.coefficients.items():
            name = self._format_feature(key) if as_name else key
            print(f"  {name:40} → {val:.6g}")

    def apply(self, raw_input: Union[List, np.ndarray]) -> float:
        """
        Apply the logistic regression model to a single instance.
        
        Args:
            raw_input: Feature vector (raw, unnormalized)
            
        Returns:
            Predicted probability (sigmoid output)
        """
        if isinstance(raw_input, list):
            raw_input = np.array(raw_input)

        z = float(self.intercept)
        for key, coef in self.coefficients.items():
            if '=' in key:
                base, cat_idx = key.split('=')
                col_idx = int(base[1:])
                val = 1.0 if int(raw_input[col_idx]) == int(cat_idx) else 0.0
            else:
                col_idx = int(key[1:])
                val = float(raw_input[col_idx])
            z += float(coef) * val

        # Sigmoid
        return float(1.0 / (1.0 + np.exp(-z)))

    def apply_batch(self, instances: List[Union[List, np.ndarray]]) -> List[float]:
        """Apply the model to multiple instances."""
        return [self.apply(inst) for inst in instances]

    def get_coefficients(self) -> Dict[str, float]:
        """Get the model coefficients."""
        return dict(self.coefficients)

    def get_intercept(self) -> float:
        """Get the intercept term."""
        return self.intercept
