import sqlite3
import pandas as pd

# Load the preprocessed data
df = pd.read_csv('preprocessed_cases.csv')

# Connect to SQLite database
conn = sqlite3.connect('cases.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS legal_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_text TEXT,
        judgment_summary TEXT,
        tags TEXT
    )
''')

# Insert data into the table
for _, row in df.iterrows():
    c.execute('''
        INSERT INTO legal_cases (case_text, judgment_summary, tags)
        VALUES (?, ?, ?)
    ''', (row['case_text'], row['judgment'], row['tags']))

conn.commit()
conn.close()

print("Data successfully inserted into database.")
