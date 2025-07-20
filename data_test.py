"""Tests cleanliness of processed data
    Robert Hoang
    2025-07-19"""

import pandas as pd
import numpy as np

# load the encoded dataset
ENCODED_PATH = r"D:\tennis_processed\dataset_encoded.csv"
df = pd.read_csv(ENCODED_PATH)
print(f"Loaded {len(df)} rows from {ENCODED_PATH}\n")

# check for missing or non‐finite values
print("NaN counts per column:")
print(df.isna().sum(), "\n")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Non‐finite counts in numeric columns:")
print((~np.isfinite(df[num_cols])).sum(), "\n")

# check categorical code ranges
print("surface_code value counts:")
print(df['surface_code'].value_counts(dropna=False), "\n")

print("p0_hand_code value counts:")
print(df['p0_hand_code'].value_counts(dropna=False), "\n")

print("p1_hand_code value counts:")
print(df['p1_hand_code'].value_counts(dropna=False), "\n")

print("Output value counts:")
print(df['Output'].value_counts(dropna=False), "\n")

# check continuous feature statistics
cont_cols = [
    'player_0_age', 'player_1_age',
    'player_0_rank', 'player_0_rank_points',
    'player_1_rank', 'player_1_rank_points'
]
print("Continuous feature summary:")
print(df[cont_cols].describe().T)