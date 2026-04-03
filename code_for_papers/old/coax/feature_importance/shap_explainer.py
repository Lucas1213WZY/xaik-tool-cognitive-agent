import shap
import numpy as np

class SHAPExplainer:
    def __init__(self, engine, train_data, preprocessing_fn, **kwargs):
        
        background_data = shap.kmeans(train_data.X, 45)

        self.explainer = shap.KernelExplainer(lambda x : engine.predict(preprocessing_fn(x)), background_data)

        self.preprocessing_fn = preprocessing_fn

    def explain(self, instances, postprocessing_fn):
        # Ensure instances is a 2D array
        if len(instances.shape) == 1:
            instances = instances.reshape(1, -1)

        # Compute SHAP values
        shap_values = self.explainer(instances)

        # Extract values and base values
        values = shap_values.values[:, :, 1].transpose()
        base_values = shap_values.base_values[:, 1]

        return np.array(values), np.array(base_values)


# import shap
# import numpy as np

# class SHAPExplainer:
#     def __init__(self, engine, train_data, preprocessing_fn, **kwargs):
#         background_data = shap.kmeans(train_data.X, 50)
#         self.explainer = shap.KernelExplainer(lambda x: engine.predict(preprocessing_fn(x)), background_data)
#         self.preprocessing_fn = preprocessing_fn

#     def explain(self, instances, postprocessing_fn, num_features=5):
#         # Ensure instances is a 2D array
#         if len(instances.shape) == 1:
#             instances = instances.reshape(1, -1)

#         # Compute SHAP values with regularization to enforce sparsity
#         shap_values = self.explainer.shap_values(instances, l1_reg=f"num_features({num_features})")

#         # Extract values and base values for the positive class (assuming binary classification)
#         values = shap_values[1]
#         base_values = self.explainer.expected_value[1]

#         return np.array(values), np.array(base_values)

# # Example usage
# # Assuming you have an engine, train_data, preprocessing_fn, and instances defined
# explainer = SHAPExplainer(engine, train_data, preprocessing_fn)
# values, base_values = explainer.explain(instances, postprocessing_fn, num_features=5)

# # The `values` array now contains SHAP values with at most 5 non-zero attributions per instance
