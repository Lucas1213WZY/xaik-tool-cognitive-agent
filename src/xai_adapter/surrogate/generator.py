"""Generate CoXAM-style surrogate explanation tables from parsed datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class GeneratedSurrogateMethods:
    """Generated surrogate tables and fitted XAI method objects."""

    decision_tree_df: Optional[pd.DataFrame]
    logistic_regression_df: Optional[pd.DataFrame]
    metadata_df: pd.DataFrame
    methods: Dict[str, Any]


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError(
            "Surrogate generation currently expects exactly two prediction labels; "
            f"got {len(labels)} labels: {labels.tolist()}"
        )
    return labels


def _feature_names(n_features: int, feature_names: Optional[Sequence[str]] = None) -> List[str]:
    if feature_names is None:
        return [f"a{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(feature_names)}")
    return list(feature_names)


def _metadata_from_data(
    X: np.ndarray,
    *,
    app_id: str,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    names = _feature_names(X.shape[1], feature_names)
    row: Dict[str, Any] = {"appId": app_id}
    for idx, name in enumerate(names):
        row[f"a{idx}"] = name
        col = X[:, idx]
        numeric_col = col[np.isfinite(col)]
        if numeric_col.size:
            row[f"v{idx}_min"] = float(np.min(numeric_col))
            row[f"v{idx}_max"] = float(np.max(numeric_col))
        else:
            row[f"v{idx}_min"] = np.nan
            row[f"v{idx}_max"] = np.nan
    return pd.DataFrame([row])


def _minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize numeric feature columns to match CoXAM LR table semantics."""
    X_array = np.asarray(X, dtype=float)
    mins = np.min(X_array, axis=0)
    maxs = np.max(X_array, axis=0)
    scales = maxs - mins
    normalized = X_array.copy()
    valid = scales != 0.0
    normalized[:, valid] = (normalized[:, valid] - mins[valid]) / scales[valid]
    return normalized


def _tree_to_coxam_structure(tree, class_count: int) -> List[Dict[str, Any]]:
    from sklearn.tree import _tree

    nodes = []
    for node_id in range(tree.node_count):
        is_leaf = tree.children_left[node_id] == _tree.TREE_LEAF
        counts = tree.value[node_id][0].astype(float)
        total = float(counts.sum())
        probs = (counts / total).tolist() if total > 0.0 else (np.ones(class_count) / class_count).tolist()
        node = {
            "node": int(node_id),
            "feature": None if is_leaf else f"a{int(tree.feature[node_id])}",
            "threshold": None if is_leaf else float(tree.threshold[node_id]),
            "left": None if is_leaf else int(tree.children_left[node_id]),
            "right": None if is_leaf else int(tree.children_right[node_id]),
            "value": probs,
            "is_leaf": bool(is_leaf),
        }
        nodes.append(node)
    return nodes


def generate_decision_tree_table(
    X: Any,
    y: Any,
    *,
    app_id: str,
    model_name: str,
    depths: Iterable[int] = (3,),
    random_state: int = 0,
    class_labels: Optional[Sequence[Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Train decision-tree surrogates and return CoXAM-style explanation rows."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    labels = _ensure_binary_labels(y_array)
    serialized_labels = list(class_labels) if class_labels is not None else labels.tolist()

    rows = []
    for depth in depths:
        classifier = DecisionTreeClassifier(max_depth=depth, random_state=random_state, **kwargs)
        classifier.fit(X_array, y_array)
        fidelity = float(accuracy_score(y_array, classifier.predict(X_array)))
        rows.append(
            {
                "appId": app_id,
                "model": model_name,
                "depth": int(depth),
                "fidelity": fidelity,
                "tree_structure": json.dumps(
                    _tree_to_coxam_structure(classifier.tree_, class_count=len(classifier.classes_))
                ),
                "class_labels": json.dumps(serialized_labels),
            }
        )
    return pd.DataFrame(rows)


def _top_k_feature_indices(coefs: np.ndarray, top_k: Optional[int]) -> List[int]:
    if top_k is None or top_k >= len(coefs):
        return list(range(len(coefs)))
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    return np.argsort(np.abs(coefs))[-top_k:][::-1].tolist()


def generate_logistic_regression_table(
    X: Any,
    y: Any,
    *,
    app_id: str,
    model_name: str,
    variants: Iterable[str] = ("dense", "sparse"),
    top_k: int = 3,
    C: float = 1.0,
    random_state: int = 0,
    max_iter: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """Train logistic-regression surrogates and return CoXAM-style explanation rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    _ensure_binary_labels(y_array)

    X_model = _minmax_normalize(X_array)

    classifier = LogisticRegression(C=C, random_state=random_state, max_iter=max_iter, **kwargs)
    classifier.fit(X_model, y_array)

    coefs = classifier.coef_[0].astype(float)
    intercept = float(classifier.intercept_[0])
    fidelity = float(accuracy_score(y_array, classifier.predict(X_model)))

    rows = []
    for variant in variants:
        if variant == "dense":
            keep_indices = list(range(X_array.shape[1]))
        elif variant == "sparse":
            keep_indices = _top_k_feature_indices(coefs, top_k)
        else:
            raise ValueError("variants must contain only 'dense' and/or 'sparse'")

        row: Dict[str, Any] = {
            "appId": app_id,
            "model": model_name,
            "variant": variant,
            "fidelity": fidelity,
            "intercept": intercept,
            "C": C,
            "nnz": len(keep_indices),
            "k": top_k if variant == "sparse" else np.nan,
        }
        for idx in keep_indices:
            row[f"coef_a{idx}"] = float(coefs[idx])
        if variant == "sparse":
            row["kept_groups"] = json.dumps([f"a{idx}" for idx in keep_indices])
        rows.append(row)

    return pd.DataFrame(rows)


def generate_surrogate_tables(
    X: Any,
    y: Any,
    *,
    app_id: str = "custom_dataset",
    model_name: str = "external_model",
    feature_names: Optional[Sequence[str]] = None,
    methods: Sequence[str] = ("decision_tree", "logistic_regression"),
    depths: Iterable[int] = (3,),
    variants: Iterable[str] = ("dense", "sparse"),
    top_k: int = 3,
    random_state: int = 0,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """Generate surrogate explanation tables from feature rows and AI predictions."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    metadata_df = _metadata_from_data(X_array, app_id=app_id, feature_names=feature_names)
    requested = {method.lower() for method in methods}

    decision_tree_df = None
    if requested & {"decision_tree", "dt", "rules"}:
        decision_tree_df = generate_decision_tree_table(
            X_array,
            y_array,
            app_id=app_id,
            model_name=model_name,
            depths=depths,
            random_state=random_state,
        )

    logistic_regression_df = None
    if requested & {"logistic_regression", "lr", "weights"}:
        logistic_regression_df = generate_logistic_regression_table(
            X_array,
            y_array,
            app_id=app_id,
            model_name=model_name,
            variants=variants,
            top_k=top_k,
            random_state=random_state,
        )

    return decision_tree_df, logistic_regression_df, metadata_df
