"""Convenience functions for creating and applying XAI methods."""

from typing import Any, List

from .base import BaseExplainer
from .explainers import get_registry


def create_explainer(explainer_type: str, **kwargs) -> BaseExplainer:
    """Create an explainer instance from the global XAI method registry."""
    return get_registry().create(explainer_type, **kwargs)


def create_coxam_explainer(loader,
                           explainer_type: str,
                           app_id: str,
                           model_name: str,
                           **kwargs) -> BaseExplainer:
    """
    Create a CoXAM explainer from explanation tables loaded by a data loader.

    Args:
        loader: Data loader with metadata and explanation tables.
        explainer_type: 'decision_tree' or 'logistic_regression'.
        app_id: Dataset appId.
        model_name: Model name, e.g. 'mlp' or 'xgboost'.
        **kwargs: Extra filters, e.g. depth=3 or variant='sparse'.
    """
    if getattr(loader.data_source, "source_type", None) != "coxam":
        raise AttributeError("create_coxam_explainer requires a CoXAM data loader")

    key = explainer_type.lower().strip()
    if key not in {"decision_tree", "logistic_regression"}:
        raise ValueError("explainer_type must be 'decision_tree' or 'logistic_regression'")

    explanation_df = loader.get_explanation_table(key)
    if "model" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["model"] == model_name]

    if key == "decision_tree" and "depth" in kwargs and "depth" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["depth"] == kwargs["depth"]]
    if key == "logistic_regression" and "variant" in kwargs and "variant" in explanation_df.columns:
        explanation_df = explanation_df[explanation_df["variant"] == kwargs["variant"]]

    if explanation_df.empty:
        raise ValueError(
            f"No rows found for explainer={key}, app_id={app_id}, model_name={model_name}, filters={kwargs}"
        )

    return create_explainer(
        key,
        explanation_df=explanation_df,
        metadata_df=loader.get_metadata(),
        app_id=app_id,
        model_name=model_name,
        **kwargs
    )


def get_coxam_xai_predictions(loader,
                              instance_ids: List[int],
                              explainer_type: str,
                              app_id: str,
                              model_name: str,
                              **kwargs) -> List[Any]:
    """Apply a CoXAM XAI method to raw features from a data loader."""
    explainer = create_coxam_explainer(
        loader,
        explainer_type=explainer_type,
        app_id=app_id,
        model_name=model_name,
        **kwargs
    )
    raw_features = loader.get_features(instance_ids, normalize=False)
    return explainer.apply_batch(raw_features)
