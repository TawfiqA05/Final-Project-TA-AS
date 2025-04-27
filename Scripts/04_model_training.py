#!/usr/bin/env python3
"""
04_model_training.py

Loads the pre-split data, trains a LinearRegression to predict Model Year,
evaluates on test set, and saves performance plot.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    splits_path = os.path.join('models', 'data_splits.joblib')
    if not os.path.exists(splits_path):
        sys.exit(f"ERROR: data splits not found at {splits_path}")

    X_train, X_test, y_train, y_test = joblib.load(splits_path)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, R²: {r2:.3f}")

    # Save performance plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2)
    plt.xlabel('Actual Model Year')
    plt.ylabel('Predicted Model Year')
    plt.title('Linear Regression Performance')
    plt.tight_layout()
    plt.savefig('images/model_performance.png')
    plt.close()
    print("Saved images/model_performance.png")

if __name__ == '__main__':
    main()
