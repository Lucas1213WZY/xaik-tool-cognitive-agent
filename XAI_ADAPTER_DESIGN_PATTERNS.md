# xai_adapter — Design Pattern Reference

## Pattern Summary

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Template Method** | XAIAdapter base class | Define algorithm skeleton (`fit()`, `explain()`) — subclasses fill in `explain()` specifics |
| **Strategy** | Attribution vs SurrogateMethod branches | Switch between local feature scores and rule-based explanations without changing calling code |
| **Registry** | `create_xai_method(name)` | Instantiate methods by name; `register_xai_method(name, fn)` for custom algorithms |
| **Adapter** | CustomAttribution / CustomSurrogate / custom callables | Wrap sklearn-style or callable algorithms into the XAIAdapter interface |
| **Factory** | Implied in Registry | `create_xai_method()` builds instances dynamically |

---

## Class Hierarchy

```
XAIAdapter (abstract)
├── Attribution
│   ├── Built-in Methods
│   │   ├── KernelShap
│   │   ├── LimeTabular
│   │   ├── GradientInput
│   │   ├── DeepLift
│   │   ├── IntegratedGradients
│   │   └── LeaveOneFeatureOut
│   └── CustomAttribution (wraps user algorithm)
│
└── SurrogateMethod
    ├── CustomSurrogate (wraps fit/explain callables)
    ├── DecisionTreeSurrogateMethod
    │   ├── RuleListSurrogateMethod
    │   └── RuleSetSurrogateMethod
    ├── LogisticRegressionSurrogateMethod
    └── (extensible for new surrogates)
```

---

## Extension Points

### 1. **Add New Local Attribution**
```python
class MyAttribution(Attribution):
    def explain(self, instances):
        # Your logic here
        return XAIAdapterResult(values=..., metadata=...)
```
**When:** Custom perturbation strategy, SHAP variant, or proprietary scoring.

---

### 2. **Add New Surrogate Method**
```python
class MyRuleSurrogate(SurrogateMethod):
    def fit(self, X, y):
        # Train rule engine
        pass
    
    def explain(self, instances):
        return XAIAdapterResult(values=..., metadata={'rules': ...})
```
**When:** New rule-learning algorithm, decision list, or approximation model.

---

### 3. **Wrap Custom Algorithm (No Subclass)**
```python
my_fn = MyCustomEstimator().fit(X, y).explain
xai = make_attribution(my_fn, method_name='my_custom')
# or
register_xai_method('my_custom', my_fn)
xai = create_xai_method('my_custom')
```
**When:** Quick integration of attribution-style black-box or sklearn-compatible estimators.

### 4. **Wrap Custom Surrogate (No Subclass)**
```python
def fit_fn(X, y, **kwargs):
    surrogate_model.fit(X, y, **kwargs)

def explain_fn(instances):
    return surrogate_model.explain(instances)

xai = make_surrogate(fit_fn, explain_fn, name='my_surrogate')
# or
xai = create_custom_surrogate_method(fit_fn, explain_fn, method_name='my_surrogate')
```
**When:** Quick integration of global surrogate or rule-learning code that already has separate fit/explain operations.

---

## Standardized Output

All methods return:
```python
@dataclass
class XAIAdapterResult:
    values: ndarray              # shape: (n_instances, n_features)
    base_values: ndarray         # background/intercept
    method: str                  # name of the method used
    metadata: dict               # algorithm-specific info
```

**Benefits:**
- Unified interface across all methods
- Easy swapping between Attribution ↔ SurrogateMethod
- Consistent post-processing and reporting

---

## Data Input Paths

### Attribution Methods
- **Input:** Raw `X, y`
- **Fit:** Each method trains independently (no shared preprocessing)
- **Output:** Feature attributions + metadata

### Surrogate Methods
Two paths:

| Path | Input | Use Case |
|------|-------|----------|
| **Raw** | `X, y` (train your own model) | Quick prototyping; no precomputation required |
| **Precomputed** | `explanation_df`, `metadata_df` | Large-scale batch processing; explanations precomputed offline |

---

## Why This Design?

- **Decoupling:** Attribution and Surrogate are independent branches; changes to one don't affect the other.
- **Extensibility:** Three clear ways to extend without modifying core code.
- **Composability:** Registry allows dynamic method selection; preprocessing/postprocessing functions are pluggable.
- **Consistency:** All outputs follow the same `XAIAdapterResult` schema.
