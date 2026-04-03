# from sklearn import preprocessing
# from datasets.tabular_dataset import TabularDataset
# import numpy as np
# import pandas as pd
# import os

# def load_data(**kwargs):
#     # 1. Load raw data
#     script_dir = os.path.dirname(__file__)
#     file_path = os.path.join(script_dir, "german.data")
#     df = pd.read_csv(file_path, sep=' ', header=None)
#     print("Loaded german.data")

#     # 2. Define column names (20 features + 1 target)
#     column_names = [
#         "Checking Account", "Duration", "Credit History", "Purpose", "Credit Amount",
#         "Savings", "Employment", "Installment Rate", "Personal Status", "Guarantors",
#         "Residence Duration", "Property", "Age", "Installment Plans", "Housing",
#         "Existing Credits", "Job", "Dependents", "Telephone", "Foreign Worker",
#         "Credit Risk"
#     ]
#     df.columns = column_names

#     # 3. Map symbolic values to readable strings
#     mapping_dicts = {
#         "Checking Account": {
#             "A11": "< 0 DM", "A12": "0-200 DM", "A13": "≥ 200 DM", "A14": "none"
#         },
#         "Credit History": {
#             "A30": "good", "A31": "good", "A32": "good",
#             "A33": "delayed",
#             "A34": "risky"
#         },
#         "Purpose": {
#             "A40": "car", "A41": "car",
#             "A42": "consumer goods", "A43": "consumer goods", "A44": "consumer goods", "A45": "consumer goods",
#             "A46": "education", "A48": "education",
#             "A47": "other", "A49": "other", "A410": "other"
#         },
#         "Savings": {
#             "A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM", 
#             "A64": "≥ 1000 DM", "A65": "unknown"
#         },
#         "Employment": {
#             "A71": "unemployed", "A72": "< 1 year", "A73": "1–4 years", 
#             "A74": "4–7 years", "A75": "≥ 7 years"
#         },
#         "Personal Status": {
#             "A91": "male-div/sep", "A92": "female-div/mar", "A93": "male-single",
#             "A94": "male-mar/wid", "A95": "female-single"
#         },
#         "Guarantors": {
#             "A101": "none", "A102": "co-applicant", "A103": "guarantor"
#         },
#         "Property": {
#             "A121": "real estate", "A122": "savings/insurance", 
#             "A123": "car/other", "A124": "unknown"
#         },
#         "Installment Plans": {
#             "A141": "bank", "A142": "stores", "A143": "none"
#         },
#         "Housing": {
#             "A151": "rent", "A152": "own", "A153": "free"
#         },
#         "Job": {
#             "A171": "unskilled-nonresident", "A172": "unskilled-resident", 
#             "A173": "skilled", "A174": "high skill/self-employed"
#         },
#         "Telephone": {
#             "A191": "none", "A192": "yes"
#         },
#         "Foreign Worker": {
#             "A201": "yes", "A202": "no"
#         }
#     }

#     ordered_categories = {
#         "Checking Account": ["none", "< 0 DM", "0-200 DM", "≥ 200 DM"],
#         "Savings": ["unknown", "< 100 DM", "100–500 DM", "500–1000 DM", "≥ 1000 DM"],
#         "Employment": ["unemployed", "< 1 year", "1–4 years", "4–7 years", "≥ 7 years"],
#     }


#     for col, mapping in mapping_dicts.items():
#         df[col] = df[col].map(mapping)

#     # 4. Target: map 1 → 0 (Good), 2 → 1 (Bad)
#     y = df["Credit Risk"].astype(int)
#     print(y)
#     y = np.where(y == 1, 1, 0)  # 1: Good, 0: Bad

#     X = df.drop(columns=["Credit Risk"])
#     feature_names = X.columns.tolist()
#     X_numpy = X.values

#     # 5. Encode categorical features
#     categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
#     categorical_features = [X.columns.get_loc(col) for col in categorical_cols]
#     categorical_feature_options = {}
#     X_numpy = X.values.astype(object)  # make mutable

#     for col in categorical_cols:                    # iterate over names
#         i = X.columns.get_loc(col)                  # column index

#         if col in ordered_categories:               # ordinal → manual order
#             cat_order = ordered_categories[col]
#             cat_map = {label: idx for idx, label in enumerate(cat_order)}
#             X_numpy[:, i] = X[col].map(cat_map).astype(float)
#             categorical_feature_options[i] = cat_order
#         else:                                       # nominal → LabelEncoder
#             le = preprocessing.LabelEncoder()
#             le.fit(X[col].dropna())
#             X_numpy[:, i] = X[col].map(
#                 lambda v: le.transform([v])[0] if pd.notna(v) else np.nan
#             )
#             categorical_feature_options[i] = list(le.classes_)
            
#     # finally cast the whole matrix to float
#     X_numpy = X_numpy.astype(float)

#     # 6. Create TabularDataset
#     dataset = TabularDataset(
#         X_numpy, y,
#         feature_names=feature_names,
#         target_name="Credit Risk",
#         target_options=["Rejected","Accepted"],
#         categorical_feature_options=categorical_feature_options,
#         dataset_name="german_credit"
#     )

#     return dataset


from sklearn import preprocessing
from datasets.tabular_dataset import TabularDataset
import numpy as np
import pandas as pd
import os

def load_data(**kwargs):
    # 1. Load CSV
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "german_credit.csv")  # Adjust filename as needed
    df = pd.read_csv(file_path)
    print(f"Loaded {os.path.basename(file_path)}")

    # 2. Rename to match original German dataset format
    rename_map = {
        "status": "Checking Account",
        "duration": "Duration",
        "credit_history": "Credit History",
        "purpose": "Purpose",
        "amount": "Credit Amount",
        "savings": "Savings",
        "employment_duration": "Employment",
        "installment_rate": "Installment Rate",
        "personal_status_sex": "Personal Status",
        "other_debtors": "Guarantors",
        "present_residence": "Residence Duration",
        "property": "Property",
        "age": "Age",
        "other_installment_plans": "Installment Plans",
        "housing": "Housing",
        "number_credits": "Existing Credits",
        "job": "Job",
        "people_liable": "Dependents",
        "telephone": "Telephone",
        "foreign_worker": "Foreign Worker",
        "credit_risk": "Credit Risk"
    }
    df.rename(columns=rename_map, inplace=True)

    simplified_category_mappings = {
        "Checking Account": {
            "no checking account": "none",
            "... < 0 DM": "< 0",
            "0<= ... < 200 DM": "0–200",
            "... >= 200 DM / salary for at least 1 year": "≥ 200"
        },
        "Credit History": {
            "no credits taken/all credits paid back duly": "good",
            "all credits at this bank paid back duly": "good",
            "existing credits paid back duly till now": "good",
            "delay in paying off in the past": "delayed",
            "critical account/other credits elsewhere": "risky"
        },
        "Purpose": {
            "car (new)": "car",
            "car (used)": "car",
            "furniture/equipment": "household",
            "domestic appliances": "household",
            "radio/television": "household",
            "repairs": "household",
            "retraining": "education",
            "business": "business",
            "vacation": "other",
            "others": "other"
        },
        "Savings": {
            "unknown/no savings account": "unknown",
            "... <  100 DM": "0-100",
            "100 <= ... <  500 DM": "100–500",
            "500 <= ... < 1000 DM": "500–1000",
            "... >= 1000 DM": "≥ 1000"
        },
        "Employment": {
            "unemployed": "unemployed",
            "< 1 yr": "0-1 yr",
            "1 <= ... < 4 yrs": "1–4 yrs",
            "4 <= ... < 7 yrs": "4–7 yrs",
            ">= 7 yrs": "≥ 7 yrs"
        },
        "Installment Rate": {
            "< 20": "low",
            "20 <= ... < 25": "mid-low",
            "25 <= ... < 35": "mid-high",
            ">= 35": "high"
        },
        "Personal Status": {
            "female : non-single or male : single": "single",
            "female : single": "single",
            "male : divorced/separated": "not married",
            "male : married/widowed": "married"
        },
        "Guarantors": {
            "none": "none",
            "co-applicant": "coapplicant",
            "guarantor": "guarantor"
        },
        "Residence Duration": {
            "< 1 yr": "0-1",
            "1 <= ... < 4 yrs": "1–3",
            "4 <= ... < 7 yrs": "4–6",
            ">= 7 yrs": "7+"
        },
        "Property": {
            "real estate": "real estate",
            "building soc. savings agr./life insurance": "savings/insurance",
            "car or other": "other",
            "unknown / no property": "unknown"
        },
        "Installment Plans": {
            "none": "none",
            "bank": "bank",
            "stores": "stores"
        },
        "Housing": {
            "rent": "rent",
            "own": "own",
            "for free": "free"
        },
        "Existing Credits": {
            "1": "1",
            "3-Feb": "2-3",
            "5-Apr": "4-5",
            ">= 6": "6+"
        },
        "Job": {
            "unemployed/unskilled - non-resident": "unskilled",
            "unskilled - resident": "unskilled",
            "skilled employee/official": "skilled",
            "manager/self-empl./highly qualif. employee": "high skill"
        },
        "Dependents": {
            "0 to 2": "0-2",
            "3 or more": "3+"
        },
        "Telephone": {
            "no": "no",
            "yes (under customer name)": "yes"
        },
        "Foreign Worker": {
            "yes": "yes",
            "no": "no"
        }
    }


    for col, mapping in simplified_category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)



    # 3. Target: convert "good"/"bad" to 0/1
    df["Credit Risk"] = df["Credit Risk"].str.strip().str.lower().map({"good": 1, "bad": 0})
    y = df["Credit Risk"].values

    # 4. Ordinal encodings based on official codebook
    ordered_categories = {
        "Checking Account": [
            "none", "> 0", "0–200", "≥ 200 DM",
        ],
        "Savings": [
            "unknown", "0-100", "100–500", "500–1000", "≥ 1000"
        ],
        "Employment": [
            "unemployed", "0-1 yr", "1–4 yrs", "4–7 yrs", "≥ 7 yrs"
        ],
        "Credit History": [
            "risky", "delayed", "good"
        ],
        "Job": [
            "unskilled", "skilled", "high skill"]
    }

    # 5. Process features
    X = df.drop(columns=["Credit Risk"])
    feature_names = X.columns.tolist()
    X_numpy = X.values.astype(object)

    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    categorical_feature_options = {}

    for col in categorical_cols:
        i = X.columns.get_loc(col)
        if col in ordered_categories:
            cat_order = ordered_categories[col]
            cat_map = {label: idx for idx, label in enumerate(cat_order)}
            X_numpy[:, i] = X[col].map(cat_map).astype(float)
            categorical_feature_options[i] = cat_order
        else:
            le = preprocessing.LabelEncoder()
            le.fit(X[col].dropna())
            X_numpy[:, i] = X[col].map(lambda v: le.transform([v])[0] if pd.notna(v) else np.nan)
            categorical_feature_options[i] = list(le.classes_)

    # 6. Numeric conversion
    X_numpy = X_numpy.astype(float)

    # 7. Wrap in TabularDataset
    dataset = TabularDataset(
        X_numpy, y,
        feature_names=feature_names,
        target_name="Credit Risk",
        target_options=["Rejected", "Accepted"],
        categorical_feature_options=categorical_feature_options,
        dataset_name="german_credit"
    )

    return dataset
