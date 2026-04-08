# XAI Adapter

`src.xai_adapter` wraps external XAI libraries behind a consistent adapter API.
It owns the sklearn-like XAI method interface for feature attribution, CoXAM
rules-vs-weights surrogates, and thin wrappers around precomputed CSV explanations.

Supported adapters:

- `lofo`: local leave-one-feature-out, numpy only
- `shap` / `shap_kernel`: SHAP `KernelExplainer`
- `lime` / `lime_tabular`: LIME tabular explainer
- `gradient_input`: gradient times input for torch models
- `deeplift`: Captum DeepLift
- `integrated_gradients` / `ig`: Captum Integrated Gradients
- `sklearn_global`: sklearn-style `feature_importances_` or `coef_`
- `precomputed_csv` / `csv`: expose precomputed explanation vectors from a `data_loaders.XAIDatasetParser`
- `decision_tree` / `rules`: CoXAM decision-tree surrogate from explanation tables
- `logistic_regression` / `weights`: CoXAM logistic-regression surrogate from explanation tables
- `generate_surrogate_xai_methods`: train fresh rules/weights surrogates when a new CSV has instances and AI predictions but no precomputed CoXAM tables

Example:

```python
from src.xai_adapter import create_xai_method

method = create_xai_method(
    "shap",
    predict_fn=model.predict_proba,
)
method.fit(X_train)

result = method.explain(X_test[:5])
values = result.values
base_values = result.base_values
```

Use `preprocessing_fn` and `postprocessing_fn` when the model consumes
preprocessed/OHE features but you want to explain raw feature rows.

For CoAX-style engine objects:

```python
from src.xai_adapter import create_xai_method_from_engine

method = create_xai_method_from_engine(
    "lofo",
    engine=engine,
    train_data=train_data,
    preprocessing_fn=preprocessing_fn,
)

result = method.explain(instances)
```

Rules-vs-weights methods use the same pattern:

```python
from src.xai_adapter import create_xai_method

rules = create_xai_method(
    "rules",
    app_id="wine_quality",
    model_name="mlp",
    depth=3,
)
rules.fit(explanation_df=decision_tree_df, metadata_df=metadata_df)

weights = create_xai_method(
    "weights",
    app_id="wine_quality",
    model_name="mlp",
    variant="sparse",
)
weights.fit(explanation_df=logistic_regression_df, metadata_df=metadata_df)
```

If a user provides a new dataset CSV instead of precomputed surrogate tables,
the same methods can generate fresh surrogates from feature rows and AI
predictions:

```python
from src.xai_adapter import create_xai_method

X = dataset_predictions_df[["v0", "v1", "v2"]].to_numpy()
y = dataset_predictions_df["pred"].to_numpy()

rules = create_xai_method("rules", app_id="my_dataset", model_name="external_model", depth=3)
rules.fit(X, y)

weights = create_xai_method(
    "weights",
    app_id="my_dataset",
    model_name="external_model",
    variant="sparse",
    top_k=3,
)
weights.fit(X, y)

rules_result = rules.explain(X)
weights_result = weights.explain(X)

# Optional: persist generated CoXAM-style tables.
decision_tree_df = rules.to_explanation_table()
logistic_regression_df = weights.to_explanation_table()
metadata_df = rules.to_metadata_table()
```

The dataset helper wraps that same flow when you want to start from a CSV-like
object:

```python
from src.data_loaders import XAIDatasetParser
from src.xai_adapter import generate_surrogate_xai_methods

dataset = XAIDatasetParser.from_dataframe(
    dataset_predictions_df,
    missing_explanation_strategy="zeros",
)

generated = generate_surrogate_xai_methods(
    dataset=dataset,
    app_id="my_dataset",
    model_name="external_model",
    methods=("decision_tree", "logistic_regression"),
    depths=(3,),
    variants=("dense", "sparse"),
    variant="sparse",
    top_k=3,
)

rules = generated.methods["rules"]
weights = generated.methods["weights"]

feature_rows = dataset.get_features(dataset.df["instanceId"].tolist())
rules_result = rules.explain(feature_rows)
weights_result = weights.explain(feature_rows)
```

CSV-backed predictions and explanations:

```text
instanceId,pred,v0,v1,v2,a0_i,a1_i,a2_i
0,1,0.2,0.7,0.1,0.4,0.5,0.1
```

```python
from src.data_loaders import XAIDatasetParser
from src.xai_adapter import create_xai_method

dataset = XAIDatasetParser.from_csv("assets/data/my_dataset_predictions.csv")
method = create_xai_method("csv", dataset=dataset)

records = dataset.get_records([0, 1, 2])
explanations = [record.explanation for record in records]
ai_predictions = [record.ai_prediction for record in records]
```

If the CSV uses the common `v0`, `v1`, ... and `a0_i`, `a1_i`, ... names, the
feature and explanation columns are inferred automatically. `intercept` is
optional; when it is absent, adapter result `base_values` default to `0.0`.

For a prediction-only CSV, set `missing_explanation_strategy="zeros"` on
`XAIDatasetParser` to produce control/no-XAI explanation vectors with the same
length as the feature vector.
