from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Step 4: Process the 'facts' Column with TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")  # Limit to top 500 features
facts_tfidf = vectorizer.fit_transform(data["facts"]).toarray()

# Step 5: Combine TF-IDF Features with Structured Data
X_structured = data[[
    "facts_len",
    "majority_vote",
    "minority_vote",
    "issue_area",
    "decision_type",
    "term"
]].values  # Convert structured features to NumPy array

X = np.hstack((X_structured, facts_tfidf))  # Combine structured and text features

# Define the target variable
y = data["first_party_winner"].map({True: 1, False: 0})  # Convert to binary (1/0)

# Step 6: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Address Class Imbalance with SMOTE
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 8: Define Parameter Grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Step 9: Grid Search with Cross Validation
print("Starting Grid Search for Hyperparameter Tuning...")
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Step 10: Train the Best Model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
best_model.fit(X_train_balanced, y_train_balanced)

# Step 11: Evaluate the Tuned Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tuned Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 12: Save the Best Model and Vectorizer
with open("best_case_outcome_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Tuned model and vectorizer saved as 'best_case_outcome_model.pkl' and 'tfidf_vectorizer.pkl'.")
