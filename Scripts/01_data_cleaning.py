#!/usr/bin/env python3
"""
01_data_cleaning.py

Loads the raw EV dataset, cleans missing and inconsistent values,
and writes out cleaned_ev_data.csv for downstream use.
"""

import os
import sys
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing key fields
    df = df.dropna(subset=['Model Year', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type'])

    # Ensure numeric types
    df['Model Year'] = df['Model Year'].astype(int)
    df['Electric Range'] = df['Electric Range'].astype(float)
    df['Base MSRP'] = df['Base MSRP'].astype(float)

    # Standardize EV type labels
    df['Electric Vehicle Type'] = df['Electric Vehicle Type'].str.upper().str.replace(r'\s+', ' ').str.strip()

    return df

def main():
    raw_path = os.path.join('data', 'ev_data.csv')
    clean_path = os.path.join('data', 'cleaned_ev_data.csv')

    if not os.path.exists(raw_path):
        sys.exit(f"ERROR: Raw data not found at {raw_path}")

    df = pd.read_csv(raw_path)
    df_clean = clean_dataframe(df)

    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned data written to {clean_path}")
    print(f"Shape: {df_clean.shape}")

if __name__ == '__main__':
    main()
