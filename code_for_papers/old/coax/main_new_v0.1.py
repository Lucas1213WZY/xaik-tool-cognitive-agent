# main_new_v0.1.py

import datasets
import models
import xai
import csv_saver
import numpy as np
import os
import pandas as pd
from datasets.tabular_dataset import TabularDataset

# Dictionary with dataset IDs mapping to LOFO ranking CSV paths
lofo_rankings = {
    "wine_quality": "lofo_rankings/wine_quality_lofo_ranking.csv",
    "forest_cover": "lofo_rankings/forest_cover_lofo_ranking.csv",
    "mushrooms": "lofo_rankings/mushrooms_lofo_ranking.csv",
    "heart_disease": "lofo_rankings/heart_disease_lofo_ranking.csv",
    "king_county_housing": "lofo_rankings/king_county_housing_lofo_ranking.csv",
    "prima_diabetes": "lofo_rankings/prima_diabetes_lofo_ranking.csv",
    "breast_cancer": "lofo_rankings/breast_cancer_lofo_ranking.csv",
    "cardiotocography": "lofo_rankings/cardiotocography_lofo_ranking.csv",
}

# Dictionary with preferred top 5 features per dataset (if provided)
dataset_preferred_features = {
    "wine_quality": ['Alcohol', 'Sulphates', 'SO2', 'Vinegar Taint', 'pH'],
    "forest_cover": ['Elevation', 'Aspect', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am'],
}


def load_save_data(data, file_path):
    """ Save data and metadata to files, ensuring directory exists. """
    base_path = os.path.dirname(file_path)
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


def load_saved_data(file_path):
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


def load_or_create_data(dataset_name, use_pre_existing_dataset=True):
    """ Load or create dataset splits and save/load metadata appropriately. """
    paths = {split: f"datasets/{dataset_name}/{split}/" for split in ["train", "dev", "test"]}
    
    # Load datasets if pre-existing datasets are requested and available
    if use_pre_existing_dataset and all(os.path.exists(paths[split] + '/X.npy') for split in paths):
        print("Loading existing datasets...")
        return tuple(load_saved_data(paths[split]) for split in ["train", "dev", "test"])

    print("Creating new datasets...")
    dataset = datasets.get_dataset(dataset_name, load_previous=True)
    
    # Split dataset
    train_data, dev_data, test_data = dataset.dropna().balance().split()
    print(train_data.feature_names)
    for split, data in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
        load_save_data(data, paths[split])
    
    return train_data, dev_data, test_data


def get_top_10_features(dataset_name, all_features):
    """
    Get top 10 features based on LOFO rankings.
    
    If preferred features are provided for the dataset, keep them at the top
    (ranked within themselves), then extend to top 10 with remaining features.
    
    Returns:
        List of top 10 feature names
    """
    if dataset_name not in lofo_rankings:
        # No LOFO ranking available, use all features as-is
        return all_features[:10] if len(all_features) >= 10 else all_features
    
    # Load LOFO rankings
    lofo_df = pd.read_csv(lofo_rankings[dataset_name])
    lofo_ranked_features = lofo_df['feature_name'].tolist()
    
    # Get preferred features if available
    preferred = dataset_preferred_features.get(dataset_name, [])
    
    if preferred:
        # Re-rank preferred features according to LOFO
        preferred_ranked = [f for f in lofo_ranked_features if f in preferred]
        
        # Get remaining features from LOFO (excluding preferred)
        remaining = [f for f in lofo_ranked_features if f not in preferred]
        
        # Combine: preferred (re-ranked) + remaining
        top_10 = (preferred_ranked + remaining)[:10]
    else:
        # No preferred features, just use top 10 from LOFO
        top_10 = lofo_ranked_features[:10]
    
    return top_10


def main_incremental(
    dataset_name,
    model_name="mlp",
    xai_method="shap",
    requires_one_hot_encoding=True,
    use_pre_existing_dataset=True,
    use_pre_existing_model=False
):
    """
    Main pipeline that:
    1) Loads dataset
    2) Gets top 10 features ranked by LOFO
    3) For iterations 1 to 10:
       - Train model on first i features
       - Generate XAI explanations (none, values, attribution, importance)
       - Append to shared _v0.1 CSV files (csv_saver handles merging)
    """
    
    print(f"\n{'='*70}")
    print(f"Starting incremental analysis for {dataset_name}")
    print(f"{'='*70}\n")
    
    # 1) Load data
    train_data, dev_data, test_data = load_or_create_data(dataset_name, use_pre_existing_dataset)
    
    # Limit test data for faster processing
    filtered = min(300, len(test_data.y))
    test_data = test_data.create_smaller_dataset(list(range(filtered)))
    
    # 2) Get top 10 features
    top_10_features = get_top_10_features(dataset_name, train_data.feature_names)
    print(f"Top 10 features: {top_10_features}\n")
    
    # Use a single output directory for all iterations
    output_dir = f"outputs_v0.1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3) Iterate from 1 to 10 features
    for iteration in range(1, 11):
        print(f"\n{'-'*70}")
        print(f"Iteration {iteration}: Training with {iteration} feature(s)")
        print(f"Features: {top_10_features[:iteration]}")
        print(f"{'-'*70}")
        
        try:
            # Select features for this iteration
            current_features = top_10_features[:iteration]
            
            # Create dataset subsets with only these features
            train_data_subset = train_data.use_specific_features(current_features)
            dev_data_subset = dev_data.use_specific_features(current_features)
            test_data_subset = test_data.use_specific_features(current_features)
            
            # Prepare data
            X_train, y_train = train_data_subset.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
            X_dev, y_dev = dev_data_subset.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
            X_test, y_test = test_data_subset.prepare_data_for_model(one_hot_encode=requires_one_hot_encoding)
            
            input_dim = X_train.shape[-1]
            num_classes = len(train_data_subset.target_options)
            
            print(f"Input Dimension: {input_dim}, Num Classes: {num_classes}")
            
            # Train model
            model = models.get_model(model_name, input_dim=input_dim, num_classes=num_classes)
            save_model_path = f"{dataset_name}_iter{iteration}_model_weights.pth"
            
            if use_pre_existing_model and os.path.exists(save_model_path):
                model.load(save_model_path)
                print(f"Model loaded from {save_model_path}")
            else:
                print(f"Training model for iteration {iteration}...")
                model.train(X_train, y_train, X_dev=X_dev, y_dev=y_dev, epochs=30)
                model.save(save_model_path)
            
            # Evaluate
            accuracy = model.evaluate(X_test, y_test)
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Get predictions
            preds = np.argmax(model.predict(X_test), axis=1)
            
            # Create a copy of test_data_subset with iteration-specific appId
            test_data_iter = test_data_subset
            test_data_iter.dataset_name = f"{dataset_name}_iter{iteration}"
            
            # Save "none" explanation (predictions only)
            # csv_saver will automatically append to none_v0.1.csv
            csv_saver.save_to_csv_with_importances(
                dataset=test_data_iter,
                preds=preds,
                importances=None,
                intercepts=None,
                file_location=output_dir,
                model_name=model_name,
                explanation_name="none",
                explanation_file_name="none_v0.1.csv"
            )
            
            # Generate XAI explanations
            explainer = xai.get_explainer(
                xai_method,
                model,
                train_data_subset,
                preprocessing_fn=lambda x: train_data_subset.prepare_instances_for_model(
                    x,
                    one_hot_encode=requires_one_hot_encoding
                )
            )
            
            # Get importances
            post_fn = test_data_subset.aggregate_importances if requires_one_hot_encoding else lambda instances, imps: imps
            importances, intercepts = explainer.explain(
                test_data_subset.X,
                postprocessing_fn=post_fn
            )
            intercepts = intercepts - 0.5
            
            # Modify importances based on predicted class
            modified_importances = np.copy(importances)
            for i in range(len(preds)):
                if preds[i] == 0:
                    modified_importances[:, i] = np.where(importances[:, i] < 0, -importances[:, i], 0)
                else:
                    modified_importances[:, i] = np.where(importances[:, i] > 0, importances[:, i], 0)
            
            # Save raw attributions
            # csv_saver will automatically append to attribution_v0.1.csv
            csv_saver.save_to_csv_with_importances(
                dataset=test_data_iter,
                preds=preds,
                importances=importances,
                intercepts=intercepts,
                file_location=output_dir,
                model_name=model_name,
                explanation_name=xai_method,
                explanation_file_name="attribution_v0.1.csv"
            )
            
            # Save modified importances
            # csv_saver will automatically append to importance_v0.1.csv
            csv_saver.save_to_csv_with_importances(
                dataset=test_data_iter,
                preds=preds,
                importances=modified_importances,
                intercepts=intercepts,
                file_location=output_dir,
                model_name=model_name,
                explanation_name=xai_method,
                explanation_file_name="importance_v0.1.csv"
            )
            
            print(f"✓ Iteration {iteration} appended to shared CSV files in {output_dir}")
            
        except Exception as e:
            print(f"✗ Error in iteration {iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"All iterations completed for {dataset_name}")
    print(f"Files saved in {output_dir}:")
    print(f"  - metadata_new.csv")
    print(f"  - values_new.csv")
    print(f"  - none_v0.1.csv")
    print(f"  - attribution_v0.1.csv")
    print(f"  - importance_v0.1.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Configure datasets and models to process
    datasets_to_analyze = [
        "wine_quality",
        "forest_cover",
        # "mushrooms",
        # "heart_disease",
        # "king_county_housing",
        # "prima_diabetes",
        # "breast_cancer",
        # "cardiotocography"
    ]
    
    model_name = "mlp"
    xai_method = "shap"
    
    for dataset_name in datasets_to_analyze:
        try:
            main_incremental(
                dataset_name=dataset_name,
                model_name=model_name,
                xai_method=xai_method,
                requires_one_hot_encoding=True,
                use_pre_existing_dataset=False,
                use_pre_existing_model=False
            )
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue