import datasets
import models
import xai
import csv_saver
import random
import numpy as np
import pickle
import os
import pandas as pd
from datasets.tabular_dataset import TabularDataset

def save_data(data, file_path):
    """ Save data and metadata to files, ensuring directory exists. """
    print(f"saving to {file_path}")
    base_path = os.path.dirname(file_path)
    print(base_path)
    os.makedirs(base_path, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(base_path, 'metadata.csv')
    metadata = {
        'feature_names': str(data.feature_names),
        'target_name': data.target_name,
        'target_options': str(data.target_options),
        'ordinal_feature_indices': str(data.ordinal_feature_indices),
        'categorical_feature_options': str(data.categorical_feature_options),
        'feature_boundaries': str(data.feature_boundaries),
        'dataset_name': data.dataset_name
    }
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
    
    # Save X and y data
    np.save(os.path.join(base_path, 'X.npy'), data.X)
    np.save(os.path.join(base_path, 'y.npy'), data.y)
    print(os.path.join(base_path, 'X.npy'))
    print("saving")

def load_data(file_path):
    """ Load data from files including all metadata and array data. """
    base_path = os.path.dirname(file_path)
    metadata_path = os.path.join(base_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path).iloc[0].to_dict()
    
    # Convert stringified lists and dictionaries back to their original types
    metadata['feature_names'] = eval(metadata['feature_names'])
    metadata['target_options'] = eval(metadata['target_options'])
    metadata['ordinal_feature_indices'] = eval(metadata['ordinal_feature_indices'])
    metadata['categorical_feature_options'] = eval(metadata['categorical_feature_options'])
    metadata['feature_boundaries'] = eval(metadata['feature_boundaries'])
    
    X = np.load(os.path.join(base_path, 'X.npy'))
    y = np.load(os.path.join(base_path, 'y.npy'))
    
    return TabularDataset(X, y,
                          feature_names=metadata['feature_names'],
                          categorical_feature_options=metadata['categorical_feature_options'],
                          ordinal_feature_indices=metadata['ordinal_feature_indices'],
                          target_name=metadata['target_name'],
                          target_options=metadata['target_options'],
                          feature_boundaries=metadata['feature_boundaries'],
                          dataset_name=metadata['dataset_name'])

def load_or_create_data(dataset_name):
    """ Load or create dataset splits and save/load metadata appropriately. """
    paths = {split: f"datasets/{dataset_name}/{split}/" for split in ["train", "dev", "test"]}
    
    if all(os.path.exists(paths[split] + '/X.npy') for split in paths):
        print("Loading existing datasets...")
        return (load_data(paths[split]) for split in paths)

    print("Creating new datasets...")
    # Assume datasets.get_dataset() retrieves and preprocesses your data
    dataset = datasets.get_dataset(dataset_name, load_previous=False)
    
    # Example split function - replace with actual data splitting logic
    train_data, dev_data, test_data = dataset.dropna().balance().split()
    
    for split, data in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
        save_data(data, paths[split])
    
    return train_data, dev_data, test_data

def main(dataset_name, model_name, xai_method, features_to_use=None, save_absolute=True, requires_one_hot_encoding=True, use_pre_existing_dataset=True, use_pre_existing_model=True):
    # Load or create dataset
    train_data, dev_data, test_data = load_or_create_data(dataset_name)

    # Prepare data for model training using all features
    X_train, y_train = train_data.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
    X_dev, y_dev = dev_data.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
    test_data = test_data.create_smaller_dataset(1, 150)

    input_dim = X_train.shape[-1]
    num_classes = len(train_data.target_options)

    print("Input Dimension: {}".format(input_dim), f"Num Classes : {num_classes}")

    # Load and train the model
    model = models.get_model(model_name, input_dim=input_dim, num_classes=num_classes)
    save_model_path = f"{dataset_name}_model_weights.pth"
    try:
        model.load(save_model_path)
    except:
        model.train(X_train, y_train, X_dev=X_dev, y_dev=y_dev, epochs=500)
    model.save(save_model_path)

    accuracy = model.evaluate(*test_data.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding))
    print(f"\n\nTest Time Accuracy: {accuracy:.4f}")

    # Generate explanations for all features
    explainer = xai.get_explainer(xai_method, model, train_data, preprocessing_fn=lambda x: train_data.prepare_instances_for_model(x, one_hot_encode=requires_one_hot_encoding))
    importances, intercepts = explainer.explain(test_data.X,
        postprocessing_fn=test_data.aggregate_importances if requires_one_hot_encoding else lambda instances, importances: importances)

    # # Sum up all the importances of the other feature values to the intercept
    # importances_sum = np.sum(importances, axis=0)
    # intercepts += importances_sum

    # intercepts = intercepts - 0.5
    # preds = np.argmax(model.predict(test_data.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)[0]), axis=1)

    # # Filter importances based on the specified features
    # if features_to_use is not None:
    #     feature_indices = [train_data.feature_names.index(feature) for feature in features_to_use]
    #     importances = importances[feature_indices, :]
    #     filtered_feature_names = [train_data.feature_names[i] for i in feature_indices]
    # else:
    #     filtered_feature_names = train_data.feature_names

    # if save_absolute:
    #     importances = abs(importances)
    #     intercepts = abs(intercepts)
    #     explanation_file_name = "importance.csv"
    # else:
    #     explanation_file_name = "attribution.csv"

    # # Create a temporary TabularDataset for saving purposes
    # filtered_test_data = test_data.use_specific_features(filtered_feature_names)

    # csv_saver.save_to_csv_with_importances(filtered_test_data, preds, importances, intercepts,
    #                                        model_name=model_name, explanation_name=xai_method, explanation_file_name=explanation_file_name)

if __name__=="__main__":
    xai_methods = ["shap", "lime", "integrated_gradients"]
    model_requires_one_hot_encoding = {"mlp":True, "xgboost":False}

    dataset_features = {
        # "cardiotocography": ['ASTV', 'AC', 'ALTV', 'Mode', 'Median'],
        # "mushrooms": ['Ring', 'Height', 'Width', 'Bruises', 'Cap Diameter'],
        # "wine_quality":['SO2', 'pH', 'Sulphates', 'Alcohol', 'Vinegar'],
        "breast_cancer": ['texture1', 'radius3', 'perimeter3', 'smoothness3', 'concave_points3'],
    }

    save_absolutes = [True, False]

    for xai_method in xai_methods:
        for model_name, requires_one_hot_encoding in model_requires_one_hot_encoding.items():
            if xai_method == "integrated_gradients" and model_name != "mlp":
                continue
            for dataset_name, features_to_use in dataset_features.items():
                for save_absolute in save_absolutes:
                    print(f"Running for {xai_method} on {model_name} for {dataset_name} with features {features_to_use} and save_absolute={save_absolute}")
                    main(dataset_name=dataset_name, model_name=model_name, xai_method=xai_method, features_to_use=features_to_use, save_absolute=save_absolute,
                        requires_one_hot_encoding=requires_one_hot_encoding)
