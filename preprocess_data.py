import pandas as pd

# Load dataset
df = pd.read_csv('corporate_cases.csv')  # Replace with the correct path to your dataset

# Select relevant columns
# Adjust the column names to match those in your dataset
df = df[['case_text', 'judgment', 'tags']]

# Save the preprocessed data
df.to_csv('preprocessed_cases.csv', index=False)

print("Preprocessed data saved as 'preprocessed_cases.csv'.")
