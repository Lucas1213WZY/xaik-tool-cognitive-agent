import os
import pandas as pd
import numpy as np

def save_to_csv_with_importances(
    dataset,
    preds,
    importances,
    intercepts,
    file_location=os.path.dirname(__file__),
    model_name=None,
    explanation_name=None,
    explanation_file_name=None,
    return_dataframe=False,
    **kwargs
):
    """
    Save 3 CSVs:
      1) metadata.csv : one row describing each feature's name, range/categories, target info.
      2) values.csv   : row-per-instance with each v{i} = the feature's value.
      3) explanation_file_name (e.g., "attribution.csv", "importance.csv", "none.csv")
                       row-per-instance with i_{x} = the importance for each feature x, plus intercept.
    If 'importances' is None, we produce no columns for i_{x}.
    
    Parameters
    ----------
    return_dataframe : bool
        If True, return (values_df, explanation_df, metadata_df) without saving to disk.
        If False (default), save to disk and return None.
    """

    # Initialize directories and file paths (only if saving to disk)
    if not return_dataframe:
        os.makedirs(file_location, exist_ok=True)
    metadata_path = os.path.join(file_location, "metadata_new.csv") if not return_dataframe else None
    values_path = os.path.join(file_location, "values_new.csv") if not return_dataframe else None
    explanation_path = os.path.join(file_location, explanation_file_name) if (not return_dataframe and explanation_file_name) else None

    # -----------------------------
    # 1) Prepare METADATA
    # -----------------------------
    metadata_dict = {"appId": dataset.dataset_name}

    for i, feature_name in enumerate(dataset.feature_names):
        # Name of the feature
        metadata_dict[f"a{i}"] = feature_name

        # For categorical features
        if i in dataset.categorical_feature_options:
            categories = dataset.categorical_feature_options[i]
            metadata_dict[f"v{i}_options"] = len(categories)
            for j, category in enumerate(categories):
                metadata_dict[f"v{i}_{j}"] = category
        else:
            # For continuous, store boundaries
            metadata_dict[f"v{i}_min"] = dataset.feature_boundaries[i][0]
            metadata_dict[f"v{i}_max"] = dataset.feature_boundaries[i][1]

    # Process target data
    metadata_dict["y"] = dataset.target_name
    for i, option in enumerate(dataset.target_options):
        metadata_dict[f"y{i}"] = option

    metadata_df_new = pd.DataFrame([metadata_dict])

    # -----------------------------
    # 2) Prepare VALUES
    # -----------------------------
    value_dict = {
        "appId": [dataset.dataset_name] * dataset.X.shape[0],
        "instanceId": list(range(dataset.X.shape[0]))
    }

    # Add the feature values
    for i in range(dataset.X.shape[1]):
        value_dict[f"v{i}"] = dataset.X[:, i]

    # Add the actual labels (y)
    value_dict["y"] = dataset.y

    value_df_new = pd.DataFrame(value_dict)

    # -----------------------------
    # 3) Prepare EXPLANATIONS
    # -----------------------------
    # If we have no importances (e.g. "none.csv"), we skip their creation
    explanation_dict = {
        "appId": [dataset.dataset_name] * dataset.X.shape[0],
        "modelName": [model_name] * dataset.X.shape[0],
        "expMethod": [explanation_name] * dataset.X.shape[0] if importances is not None else None,
        "instanceId": list(range(dataset.X.shape[0])),
        "pred": preds
    }

    if importances is None:
        explanation_dict.pop("expMethod", None)

    if importances is not None and intercepts is not None:
        # The 90th percentile (absolute) magnitude, for consistent scaling
        i_max_val = max(
            np.percentile(np.abs(importances), 90) if importances.size > 0 else 0,
            np.percentile(np.abs(intercepts), 90) if intercepts.size > 0 else 0
        )
        explanation_dict["i_max"] = [i_max_val] * dataset.X.shape[0]

        # Add each feature's importance
        # importances shape = (num_features, num_instances)
        for i, importance_array in enumerate(importances):
            explanation_dict[f"a{i}_i"] = importance_array

        # Add intercept
        explanation_dict["intercept"] = intercepts
    else:
        # If no importances, place dummy i_max = 0
        explanation_dict["i_max"] = [0] * dataset.X.shape[0]

    # Include any extra columns passed in
    for key, value in kwargs.items():
        explanation_dict[key] = value

    explanation_df_new = pd.DataFrame(explanation_dict)

    # ----------------------------------------------------
    # Merge with existing CSVs and keep consistent columns
    # ----------------------------------------------------
    def align_and_combine(existing_df, new_df, filter_condition, order_func):
        # print(existing_df, new_df, filter_condition, order_func)
        
        # Remove any existing rows that match condition (same appId, modelName, etc.)
        filtered_existing_df = existing_df[~filter_condition(existing_df)]

        # Combine
        combined_df = pd.concat([filtered_existing_df, new_df], ignore_index=True)

        # Re-order columns in final
        combined_df = order_func(combined_df)
        return combined_df

    def filter_metadata(df):
        # For metadata, remove existing rows that share the same appId
        return df['appId'] == dataset.dataset_name

    def filter_values(df):
        # For values, remove existing rows that share the same appId
        return df['appId'] == dataset.dataset_name

    def filter_explanations(df):
        cond = (df['appId'] == dataset.dataset_name) & (df['modelName'] == model_name)
        if "expMethod" in df.columns:
            cond &= (df["expMethod"] == explanation_name)

        return cond


    # -----------------------------
    # Column ordering functions
    # -----------------------------
    def order_metadata_columns(df):
        """
        We want columns in the order:
          [ 'appId',
            a0, v0_options / v0_min,v0_max, v0_0,...,
            a1, v1_options / v1_min,v1_max, v1_0,...,
            ...
            'y', 'y0', 'y1', ...
            plus leftover columns (alphabetical)
          ]
        """
        base_cols = ['appId']

        # We look for how many a{i} exist
        i = 0
        while f"a{i}" in df.columns:
            base_cols.append(f"a{i}")
            if f"v{i}_options" in df.columns:
                base_cols.append(f"v{i}_options")
                # Include v{i}_j if they exist
                j = 0
                while f"v{i}_{j}" in df.columns:
                    base_cols.append(f"v{i}_{j}")
                    j += 1
            else:
                # Possibly continuous: v{i}_min, v{i}_max
                if f"v{i}_min" in df.columns:
                    base_cols.append(f"v{i}_min")
                if f"v{i}_max" in df.columns:
                    base_cols.append(f"v{i}_max")
            i += 1

        # Add target columns
        if 'y' in df.columns:
            base_cols.append('y')
        # Then y0, y1, ...
        i = 0
        while f"y{i}" in df.columns:
            base_cols.append(f"y{i}")
            i += 1

        # Any leftover columns go to the end in sorted order
        leftover = [c for c in df.columns if c not in base_cols]
        leftover_sorted = sorted(leftover)

        final_cols = base_cols + leftover_sorted
        final_cols = [c for c in final_cols if c in df.columns]  # ensure existence
        return df[final_cols]

    def order_values_columns(df):
        """
        We want columns in the order:
          ['appId', 'instanceId', 'v0', 'v1', ..., 'y', leftover...]
        """
        base_cols = ['appId', 'instanceId']

        # gather v0, v1, ...
        i = 0
        while f"v{i}" in df.columns:
            base_cols.append(f"v{i}")
            i += 1

        if 'y' in df.columns:
            base_cols.append('y')

        leftover = [c for c in df.columns if c not in base_cols]
        leftover_sorted = sorted(leftover)

        final_cols = base_cols + leftover_sorted
        final_cols = [c for c in final_cols if c in df.columns]
        return df[final_cols]

    def order_explanations_columns(df):
        """
        We want columns in the order:
          ['appId', 'modelName', 'expMethod', 'instanceId', 'pred', 'i_max',
           'a0_i', 'a1_i', ..., 'intercept', leftover...]
        """
        # base_cols = ['appId', 'modelName', 'expMethod', 'instanceId', 'pred', 'i_max']
        base_cols = ['appId', 'modelName', 'instanceId', 'pred', 'i_max']
        if 'expMethod' in df.columns:
            base_cols.insert(2, 'expMethod')  # Maintain order if it exists


        # gather a{i}_i
        i = 0
        while f"a{i}_i" in df.columns:
            base_cols.append(f"a{i}_i")
            i += 1

        if 'intercept' in df.columns:
            base_cols.append('intercept')

        leftover = [c for c in df.columns if c not in base_cols]
        leftover_sorted = sorted(leftover)

        final_cols = base_cols + leftover_sorted
        final_cols = [c for c in final_cols if c in df.columns]
        return df[final_cols]

    # -----------------------------
    # Merge METADATA
    # -----------------------------
    if not return_dataframe:
        if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
            try:
                existing_metadata = pd.read_csv(metadata_path)
                metadata_df = align_and_combine(
                    existing_metadata,
                    metadata_df_new,
                    filter_condition=filter_metadata,
                    order_func=order_metadata_columns
                )
            except pd.errors.EmptyDataError:
                # File is empty, just use the new metadata
                metadata_df = order_metadata_columns(metadata_df_new)
        else:
            metadata_df = order_metadata_columns(metadata_df_new)

        # Save
        metadata_df.to_csv(metadata_path, index=False)
    else:
        metadata_df = order_metadata_columns(metadata_df_new)

    # -----------------------------
    # Merge VALUES
    # -----------------------------
    if not return_dataframe:
        if os.path.exists(values_path) and os.path.getsize(values_path) > 0:
            try:
                existing_values = pd.read_csv(values_path)
                value_df = align_and_combine(
                    existing_values,
                    value_df_new,
                    filter_condition=filter_values,
                    order_func=order_values_columns
                )
            except pd.errors.EmptyDataError:
                # File is empty, just use the new values
                value_df = order_values_columns(value_df_new)
        else:
            value_df = order_values_columns(value_df_new)

        # Save
        value_df.to_csv(values_path, index=False)
    else:
        value_df = order_values_columns(value_df_new)

    # -----------------------------
    # Merge EXPLANATIONS
    # -----------------------------
    if not return_dataframe:
        if os.path.exists(explanation_path) and os.path.getsize(explanation_path) > 0:
            try:
                existing_explanations = pd.read_csv(explanation_path)
                explanation_df = align_and_combine(
                    existing_explanations,
                    explanation_df_new,
                    filter_condition=filter_explanations,
                    order_func=order_explanations_columns
                )
            except pd.errors.EmptyDataError:
                # File is empty, just use the new explanations
                explanation_df = order_explanations_columns(explanation_df_new)
        else:
            explanation_df = order_explanations_columns(explanation_df_new)

        explanation_df.to_csv(explanation_path, index=False)
    else:
        explanation_df = order_explanations_columns(explanation_df_new)
    
    # Return dataframes if requested, otherwise return None
    if return_dataframe:
        return value_df, explanation_df, metadata_df
    else:
        return None
