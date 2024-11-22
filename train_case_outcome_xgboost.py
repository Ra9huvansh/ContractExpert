from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Dataset
file_path = "justice.csv"  # Adjust the path if needed
data = pd.read_csv(file_path)

# Step 2: Handle Missing Values
data = data.dropna(subset=["first_party_winner"])  # Drop rows where target is missing
data["issue_area"] = data["issue_area"].fillna("unknown")
data["decision_type"] = data["decision_type"].fillna("unknown")
data["facts_len"] = data["facts_len"].fillna(data["facts_len"].median())
data["facts"] = data["facts"].fillna("")  # Fill missing 'facts' with an empty string

# Convert 'term' to numeric (extract the first year if it's a range)
if data["term"].dtype == "object":  # Only process if it's not numeric
    data["term"] = data["term"].str.split("-").str[0].astype(int)

# Step 3: Encode Categorical Features
categorical_features = ["issue_area", "decision_type"]
for feature in categorical_features:
    encoder = LabelEncoder()
    data[feature] = encoder.fit_transform(data[feature])

# Step 4: Process the 'facts' Column with SBERT
print("Encoding 'facts' with SBERT embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight SBERT model
facts_embeddings = sbert_model.encode(data["facts"].tolist())  # Generate embeddings for all rows

# Step 5: Combine SBERT Embeddings with Structured Data
X_structured = data[[
    "facts_len",
    "majority_vote",
    "minority_vote",
    "issue_area",
    "decision_type",
    "term"
]].values  # Convert structured features to NumPy array

X = np.hstack((X_structured, facts_embeddings))  # Combine structured and text features

# Define the target variable
y = data["first_party_winner"].map({True: 1, False: 0})  # Convert to binary (1/0)

# Step 6: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Address Class Imbalance with SMOTE
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 8: Define XGBoost Model and Parameter Grid
print("Starting Hyperparameter Tuning for XGBoost...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, scale_pos_weight=len(y_train_balanced) / sum(y_train_balanced))  # Handle class imbalance
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Step 9: Train the Best Model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
best_model.fit(X_train_balanced, y_train_balanced)

# Step 10: Evaluate the Tuned Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tuned Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Save the Best Model and SBERT
with open("best_case_outcome_model_xgboost.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("sbert_model.pkl", "wb") as sbert_file:
    pickle.dump(sbert_model, sbert_file)

print("Tuned model and SBERT saved as 'best_case_outcome_model_xgboost.pkl' and 'sbert_model.pkl'.")
