from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import numpy as np
import pandas as pd
import os

def load_data(**kwargs):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "data.csv")  # Adjust filename
    df = pd.read_csv(file_path)

    # Optional: Rename for display friendliness
    rename_map = {
        "Loan_ID": "Loan ID",
        "Gender": "Gender",
        "Married": "Married",
        "Dependents": "Dependents",
        "Education": "Graduated",
        "Self_Employed": "Self-Employed",
        "ApplicantIncome": "Applicant Income",
        "CoapplicantIncome": "Coapplicant Income",
        "LoanAmount": "Loan Amount",
        "Loan_Amount_Term": "Loan Term",
        "Credit_History": "Credit History",
        "Property_Area": "Property Area",
        "Loan_Status": "Loan Status"
    }
    df.rename(columns=rename_map, inplace=True)

    # Strip and clean strings (especially target)
    df["Loan Status"] = df["Loan Status"].astype(str).str.strip().str.upper()
    y = np.where(df["Loan Status"] == "Y", 1, 0)


    X = df.drop(columns=["Loan Status", "Loan ID"])
    feature_names = X.columns.tolist()
    X_numpy = X.values

    # Identify categorical features
    categorical_cols = ["Gender", "Married", "Graduated", "Self-Employed", "Property Area"]
    categorical_features = [X.columns.get_loc(col) for col in categorical_cols]

    categorical_feature_options = {}
    for i in categorical_features:
        le = preprocessing.LabelEncoder()
        non_null = X.iloc[:, i].dropna()
        le.fit(non_null)

        # Transform full column while keeping NaNs
        full_column = X.iloc[:, i]
        transformed_data = full_column.map(lambda val: le.transform([val])[0] if pd.notna(val) else np.nan)

        # try:
        #     missing_label_index = le.transform(['Missing'])[0]
        #     transformed_data = np.where(transformed_data == missing_label_index, np.nan, transformed_data)
        # except:
        #     pass
        X_numpy[:, i] = transformed_data
        categorical_feature_options[i] = list(le.classes_)

    # Convert to float
    X_numpy = X_numpy.astype(float)

    # Return TabularDataset
    dataset = TabularDataset(
        X_numpy, y,
        feature_names=feature_names,
        target_name="Loan Status",
        target_options=["Rejected", "Approved"],
        categorical_feature_options=categorical_feature_options,
        dataset_name="loan_approval"
    )

    return dataset
