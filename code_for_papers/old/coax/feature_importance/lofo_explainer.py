import numpy as np

class LOFOExplainer:
    """
    Local Leave-One-Feature-Out (model-agnostic):
    For each instance and feature j:
      - Replace x_j with baseline_j (e.g., train mean after preprocessing)
      - Importance_j = f(x)_1 - f(x_{-j})_1  (delta class-1 prob)
    Positive => feature j increases class-1 probability for that instance.
    """
    def __init__(self, engine, train_data, preprocessing_fn, baseline="mean", **kwargs):
        self.engine = engine
        self.preprocessing_fn = preprocessing_fn

        train_pp = self.preprocessing_fn(train_data.X)  # (N, d)
        if baseline == "mean":
            self.baseline_vec = np.mean(train_pp, axis=0)
        elif baseline == "median":
            self.baseline_vec = np.median(train_pp, axis=0)
        else:
            self.baseline_vec = np.mean(train_pp, axis=0)

        # Intercept: model prob at global baseline
        self.intercept_value = float(self.engine.predict(self.baseline_vec[None, :])[0, 1])

    def explain(self, instances, postprocessing_fn):
        X_pp = self.preprocessing_fn(instances)                      # (n, d)
        n, d = X_pp.shape
        base_probs = self.engine.predict(X_pp)[:, 1]                 # (n,)

        attributions = np.zeros((n, d), dtype=float)
        for j in range(d):
            X_masked = X_pp.copy()
            X_masked[:, j] = self.baseline_vec[j]                    # replace feature j
            probs_masked = self.engine.predict(X_masked)[:, 1]
            attributions[:, j] = base_probs - probs_masked           # delta p1

        imps = postprocessing_fn(instances, attributions)            # aggregate if OHE
        imps = np.asarray(imps).T                                     # (d_eff, n)

        intercepts = np.full((imps.shape[1],), self.intercept_value, dtype=float)
        return imps, intercepts
