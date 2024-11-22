import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the Dataset
file_path = "justice.csv"  # Adjust the path if needed
data = pd.read_csv(file_path)

print(data.columns)
print(data["first_party_winner"].value_counts())


# Convert 'term' to numeric (extract the first year if it's a range)
if data["term"].dtype == "object":  # Only process if it's not numeric
    data["term"] = data["term"].str.split("-").str[0].astype(int)