import torch
import numpy as np

class GradientInputExplainer:
    """
    Gradient × Input: for differentiable models (e.g., MLP).
    Attribution_j = (∂y/∂x_j) * x_j  (on preprocessed features)
    """
    def __init__(self, engine, train_data, preprocessing_fn, **kwargs):
        self.engine = engine
        self.preprocessing_fn = preprocessing_fn
        self.engine.model.eval()

        # Optional: a global "baseline intercept" as model prob at mean feature vector
        with torch.no_grad():
            train_X = torch.tensor(self.preprocessing_fn(train_data.X), dtype=torch.float32)
            self.baseline = train_X.mean(dim=0)
            base_prob = self.engine.predict(self.baseline.numpy()[None, :])[0, 1]
        self.intercept_value = float(base_prob)

    def explain(self, instances, postprocessing_fn):
        X = torch.tensor(self.preprocessing_fn(instances), dtype=torch.float32, requires_grad=True)
        # Forward
        probs = self.engine.forward_logits_or_probs(X)  # Prefer a method; else use engine.predict
        if probs.ndim == 2:
            y1 = probs[:, 1]  # class-1 probability/logit
        else:
            y1 = probs

        # Backward
        self.engine.model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(outputs=y1, inputs=X,
                                    grad_outputs=torch.ones_like(y1),
                                    retain_graph=False, create_graph=False)[0]

        attributions = (grads * X).detach().cpu().numpy()          # (n_instances, n_features)
        imps = postprocessing_fn(instances, attributions)           # aggregate OHE if needed
        imps = np.asarray(imps).T                                    # (n_features, n_instances)

        # One intercept per instance (constant)
        intercepts = np.full((imps.shape[1],), self.intercept_value, dtype=float)
        return imps, intercepts
