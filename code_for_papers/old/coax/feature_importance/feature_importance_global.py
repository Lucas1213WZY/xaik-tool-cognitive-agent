from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Load Adult Income dataset
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame.copy()

# Clean missing values represented as '?'
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Binary target: 1 if >50K, else 0
df["income"] = (df["class"] == ">50K").astype(int)
y = df["income"]
X = df.drop(columns=["income", "class"])

# Separate features by type
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
])

# Create a pipeline: preprocess then train
clf = Pipeline(steps=[
    ("pre", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Extract feature names post-preprocessing
cat_feature_names = clf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

# Extract feature importances
importances = clf.named_steps["rf"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Show top 10
print(importance_df.head(10))
