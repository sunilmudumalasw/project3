"""
Clean Data
Author: Sunil Mudumala
Date: 26 Sep 2025
"""
import pandas as pd

df = pd.read_csv('/mnt/e/nd0821/project3/data/census.csv')
print("Original columns:", df.columns.tolist())

# Remove spaces and replace hyphens with underscores
df.columns = df.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
print("Cleaned columns:", df.columns.tolist())

# Save the cleaned data
df.to_csv('/mnt/e/nd0821/project3/data/census_clean.csv', index=False)