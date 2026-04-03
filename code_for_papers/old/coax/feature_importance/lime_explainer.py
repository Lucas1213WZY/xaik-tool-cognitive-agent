import lime.lime_tabular
import numpy as np
from lime.discretize import BaseDiscretizer
import matplotlib.pyplot as plt

class CustomDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None, bins=4):
        self.num_bins = bins
        super().__init__(data, categorical_features, feature_names, labels=labels, random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Use 10 bins instead of quartiles
            qts = np.percentile(data[:, feature], np.linspace(0, 100, self.num_bins + 1))
            bins.append(qts)
        return bins

class LimeExplainer:
    def __init__(self, engine, train_data, preprocessing_fn, kernel_width=1.5):
        '''engine is the model object'''
        self.categorical_features = train_data.categorical_feature_indices
        self.feature_names = train_data.feature_names

        # Define custom discretizer with 10 bins
        custom_discretizer = CustomDiscretizer(
            train_data.X,
            categorical_features=self.categorical_features,
            feature_names=self.feature_names,
            bins=4
        )

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            train_data.X,
            mode='classification',
            training_labels=train_data.y,
            categorical_features=self.categorical_features,
            feature_selection='auto',
            kernel_width=kernel_width,
            discretizer=custom_discretizer,
            discretize_continuous=True,
            sample_around_instance=True
        )
        self.engine = engine
        self.preprocessing_fn = preprocessing_fn

        print(f"kernel width: {kernel_width}")

    def explain(self, instances, postprocessing_fn):
        if len(instances.shape) == 1:
            instances = instances.reshape(1, -1)

        explanations = [self.explainer.explain_instance(
                            instance,
                            lambda x: self.engine.predict(self.preprocessing_fn(x)),
                            num_features=50,
                            num_samples=5000,
                        ) for instance in instances]

        # for i in range(5):
        #     explanations[i].as_pyplot_figure(label=1)
        #     plt.show()

        importances = []
        for idx, explanation in enumerate(explanations):
            # Extract the importance values directly from the LIME explanation
            importance_values = np.zeros(len(self.feature_names))
            explanation_map = dict(explanation.as_map()[1])
            for feature_index, importance in explanation_map.items():
                importance_values[feature_index] = importance

            importances.append(importance_values)

        intercepts = [explanation.intercept[1] for explanation in explanations]

        return np.array(importances).T, np.array(intercepts)