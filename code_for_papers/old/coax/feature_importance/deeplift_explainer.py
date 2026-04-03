import torch
import numpy as np
from captum.attr import DeepLift

class DeepLIFTExplainer:
    """
    DeepLIFT (reference-based). Often yields stronger, less noisy signals than raw gradients.
    """
    def __init__(self, engine, train_data, preprocessing_fn, **kwargs):
        self.engine = engine
        self.preprocessing_fn = preprocessing_fn
        self.engine.model.eval()

        self.attr = DeepLift(self.engine.model)

        # Reference/baseline = mean of preprocessed train set
        train_pp = self.preprocessing_fn(train_data.X)
        self.baseline_vec = torch.tensor(train_pp, dtype=torch.float32).mean(dim=0, keepdim=True)

        # Intercept as model prob at baseline
        with torch.no_grad():
            base_prob = self.engine.predict(self.baseline_vec.numpy())[0, 1]
        self.intercept_value = float(base_prob)

    def explain(self, instances, postprocessing_fn):
        X = torch.tensor(self.preprocessing_fn(instances), dtype=torch.float32, requires_grad=False)
        baselines = self.baseline_vec.repeat(X.shape[0], 1)

        attributions = self.attr.attribute(X, baselines=baselines, target=1)  # class index 1
        attributions = attributions.detach().cpu().numpy()                    # (n_instances, n_features)

        imps = postprocessing_fn(instances, attributions)
        imps = np.asarray(imps).T

        intercepts = np.full((imps.shape[1],), self.intercept_value, dtype=float)
        return imps, intercepts
