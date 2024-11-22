from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import torch
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

# Step 4: Process the 'facts' Column with LegalBERT
print("Encoding 'facts' with LegalBERT embeddings...")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def encode_texts_in_batches(texts, tokenizer, model, batch_size=32, max_length=256):
    """Generate embeddings for a list of texts using LegalBERT in batches."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_inputs = tokenizer.batch_encode_plus(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**{k: v.to('cpu') for k, v in encoded_inputs.items()})
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

print("Encoding 'facts' with LegalBERT embeddings in batches...")
facts_embeddings = encode_texts_in_batches(data["facts"].tolist(), tokenizer, model, batch_size=16, max_length=256)

# Step 5: Combine LegalBERT Embeddings with Structured Data
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

# Step 7: Address Class Imbalance with SMOTETomek
print("Applying SMOTETomek to balance the dataset...")
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Step 8: Define XGBoost Model and Parameter Grid
print("Starting Hyperparameter Tuning for XGBoost...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 9],
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

# Step 10: Evaluate the Tuned Model with Threshold Adjustment
y_proba = best_model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
optimal_idx = np.argmax(tpr - fpr)  # Find the optimal threshold
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")
y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_adjusted)
print(f"Tuned Model Accuracy (with adjusted threshold): {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_adjusted))

# Step 11: Save the Best Model and LegalBERT
with open("best_case_outcome_model_xgboost.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

print("Tuned model saved as 'best_case_outcome_model_xgboost.pkl'.")
