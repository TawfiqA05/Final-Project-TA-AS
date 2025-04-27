#!/usr/bin/env python3
"""
03_feature_engineering.py

One-hot encodes categorical features and splits into train/test sets.
Pickles the splits for model training.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

def main():
    cleaned = os.path.join('data', 'cleaned_ev_data.csv')
    if not os.path.exists(cleaned):
        sys.exit(f"ERROR: cleaned data not found at {cleaned}")

    df = pd.read_csv(cleaned)

    # One-hot encode EV Type
    X_type = pd.get_dummies(df['Electric Vehicle Type'], prefix='EVType')
    # Numeric features
    X_num = df[['Electric Range', 'Base MSRP']]
    X = pd.concat([X_type, X_num], axis=1)
    y = df['Model Year']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs('models', exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), 'models/data_splits.joblib')
    print("Saved train/test splits to models/data_splits.joblib")
    print("Shapes:",
          "X_train", X_train.shape,
          "X_test",  X_test.shape,
          "y_train", y_train.shape,
          "y_test",  y_test.shape)

if __name__ == '__main__':
    main()
