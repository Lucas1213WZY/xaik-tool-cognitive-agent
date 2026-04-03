from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import numpy as np
import pandas as pd
import os

def load_data(**kwargs):
    # ---------------------------
    # 1. Load CSV
    # ---------------------------
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data.csv")
    df = pd.read_csv(file_path)
    print(f"Loaded {os.path.basename(file_path)}")

    # ---------------------------
    # 2. Rename columns
    # ---------------------------
    rename_map = {
        'Diabetes_binary': 'Diabetes',
        'HighBP': 'High BP',
        'HighChol': 'High Cholesterol',
        'CholCheck': 'Cholesterol Check',
        'BMI': 'BMI',
        'Smoker': 'Smoker',
        'Stroke': 'Stroke History',
        'HeartDiseaseorAttack': 'Heart Disease',
        'PhysActivity': 'Physical Activity',
        'Fruits': 'Eats Fruits',
        'Veggies': 'Eats Vegetables',
        'HvyAlcoholConsump': 'Drinks Alcohol',
        'AnyHealthcare': 'Has Healthcare Access',
        'NoDocbcCost': 'Avoided Doctor (Cost)',
        'GenHlth': 'General Health',
        'MentHlth': 'Mental Health Bad Days',
        'PhysHlth': 'Physical Health Bad Days',
        'DiffWalk': 'Difficulty Walking',
        'Sex': 'Sex',
        'Age': 'Age Category',
        'Education': 'Education Level',
        'Income': 'Income Level'
    }
    df.rename(columns=rename_map, inplace=True)

    # ---------------------------
    # 3. Target
    # ---------------------------
    y = df['Diabetes'].values
    y = 1 - y

    # ---------------------------
    # 4. Raw-code → label maps
    # (collapse where desired)
    # ---------------------------
    general_health_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}

    # collapse 13 raw age codes into 5 bins
    age_category_map = {
        1: "18-29", 2: "18-29",
        3: "30-39", 4: "30-39",
        5: "40-49", 6: "40-49",
        7: "50-59", 8: "50-59",
        9: "60+", 10: "60+", 11: "60+", 12: "60+", 13: "60+"
    }

    education_level_map = {
        1: "No School",
        2: "Grades 1-8",
        3: "Grades 9-11",
        4: "High School Graduate",
        5: "College",
        6: "College Graduate"
    }

    income_level_map = {
        1: "<$10k", 2: "$10k-$15k", 3: "$15k-$20k", 4: "$20k-$25k",
        5: "$25k-$35k", 6: "$35k-$50k", 7: "$50k-$75k", 8: "$75k+"
    }

    sex_map = {1: "Male", 0: "Female"}

    # Apply raw→label maps
    label_maps = {
        'General Health': general_health_map,
        'Age Category': age_category_map,
        'Education Level': education_level_map,
        'Income Level': income_level_map,
        'Sex': sex_map,
    }
    for col, mapping in label_maps.items():
        df[col] = df[col].map(mapping)

    # ---------------------------
    # 5. Binary columns → 0/1 ints
    # ---------------------------
    # Keeping numeric 0/1 here is simpler for modeling *and* for UI threshold sliders.
    # If you want Yes/No strings in the UI, change `binary_encode="string"` below.
    binary_cols = [
        'High BP', 'High Cholesterol', 'Cholesterol Check', 'Smoker', 'Stroke History',
        'Heart Disease', 'Physical Activity', 'Eats Fruits', 'Eats Vegetables',
        'Drinks Alcohol', 'Has Healthcare Access', 'Avoided Doctor (Cost)', 'Difficulty Walking'
    ]
    # The raw data already uses 0/1, so just ensure dtype:
    df[binary_cols] = df[binary_cols].astype(int)

    # ---------------------------
    # 6. Null guard (catch mapping mistakes early)
    # ---------------------------
    null_counts = df.isna().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0]
        raise ValueError(
            "Unexpected nulls after mapping. Columns:\n"
            + bad.to_string()
            + "\nCheck that mapping dicts cover all raw codes."
        )

    # ---------------------------
    # 7. Build feature matrix
    # ---------------------------
    X = df.drop(columns=['Diabetes'])
    feature_names = X.columns.tolist()

    # ---------------------------
    # 8. Which features are ordinal?
    # Provide ordered label lists (low→high)
    # ---------------------------
    ordinal_orders = {
        'General Health': ["Poor", "Fair", "Good", "Very Good", "Excellent"],
        'Age Category': ["18-29", "30-39", "40-49", "50-59", "60+"],
        'Education Level': [
            "No School/Kindergarten", "Grades 1-8", "Grades 9-11",
            "High School Graduate", "Some College/Tech School", "College Graduate"
        ],
        'Income Level': [
            "<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
            "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"
        ],
    }
    # Nominal categorical features (string/object) that are NOT ordinal:
    nominal_cols = ['Sex']  # extend if needed later

    # ---------------------------
    # 9. Encode
    # ---------------------------
    # We'll allocate an object array copy so we can write in-place by column index.
    X_numpy = X.values.astype(object)
    categorical_feature_options = {}

    # Ordinal encodings (manual; preserves intended order)
    for col, ordered_labels in ordinal_orders.items():
        i = X.columns.get_loc(col)
        mapping = {lab: idx for idx, lab in enumerate(ordered_labels)}
        X_numpy[:, i] = X[col].map(mapping).values
        categorical_feature_options[i] = ordered_labels  # preserve label order for UI

    # Nominal encodings (LabelEncoder)
    for col in nominal_cols:
        i = X.columns.get_loc(col)
        le = preprocessing.LabelEncoder()
        X_numpy[:, i] = le.fit_transform(X[col])
        categorical_feature_options[i] = list(le.classes_)

    # Binary columns: already numeric 0/1. We *can* pass options to make UI show Yes/No.
    # Up to you; doing so keeps them "categorical" to your UI, but model already sees numeric.
    binary_options = ['No', 'Yes']
    for col in binary_cols:
        i = X.columns.get_loc(col)
        categorical_feature_options[i] = binary_options

    # Continuous columns: BMI, Mental Health Bad Days, Physical Health Bad Days
    # (No entry needed in categorical_feature_options.)

    # Cast to float for modeling
    X_numpy = X_numpy.astype(float)

    # ---------------------------
    # 10. Build dataset
    # ---------------------------
    dataset = TabularDataset(
        X_numpy,
        y,
        feature_names=feature_names,
        target_name="Diabetes",
        target_options=["Diabetes", "No Diabetes"],
        categorical_feature_options=categorical_feature_options,
        dataset_name="diabetes"
    )

    return dataset