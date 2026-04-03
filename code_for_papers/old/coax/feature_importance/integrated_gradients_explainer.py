import torch
from captum.attr import IntegratedGradients
import numpy as np

class IntegratedGradientsExplainer:
    def __init__(self, engine, train_data, preprocessing_fn):
        self.engine = engine
        self.train_data = train_data
        self.preprocessing_fn = preprocessing_fn

        self.integrated_gradients = IntegratedGradients(self.engine.model)
        
        # Calculate the mean of each feature across the training data
        self.baseline = torch.tensor(preprocessing_fn(train_data.X), dtype=torch.float32).mean(dim=0)

        self.intercept = self.engine.predict(self.baseline)[0, 1]

    def explain(self, instances, postprocessing_fn):
        self.engine.model.eval()
        inputs = torch.tensor(self.preprocessing_fn(instances), dtype=torch.float32)
        inputs.requires_grad = True

        # Ensure baseline is repeated to match the number of instances
        num_instances = inputs.shape[0]
        baselines = self.baseline.repeat(num_instances, 1)  # Repeat the baseline for each instance

        importances, _ = self.integrated_gradients.attribute(inputs, baselines=baselines, target=1, return_convergence_delta=True, n_steps=50)
        importances = importances.detach().numpy()

        pred = self.engine.predict(self.preprocessing_fn(instances))[:, 1]
        # sum_ = np.sum(importances, -1)#+np.full((num_instances,), self.intercept)
        # print(importances.shape)
        # print(sum_-pred)
        # print(sum_)
        
        print(importances.shape)
        return postprocessing_fn(instances, importances).transpose(), np.full((num_instances,), self.intercept)


