#!/usr/bin/env python3
"""
02_eda.py

Performs exploratory data analysis on cleaned_ev_data.csv,
prints summary stats, and generates two bar charts:
 - vehicle_type_counts.png
 - vehicle_type_distribution_seaborn.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    cleaned = os.path.join('data', 'cleaned_ev_data.csv')
    if not os.path.exists(cleaned):
        sys.exit(f"ERROR: cleaned data not found at {cleaned}")

    df = pd.read_csv(cleaned)
    print("=== EDA SUMMARY ===")
    print(df[['Electric Vehicle Type', 'Model Year', 'Electric Range', 'Base MSRP']].describe(include='all'))
    print("\nUnique EV types:", df['Electric Vehicle Type'].nunique())
    print("Year range:", df['Model Year'].min(), "–", df['Model Year'].max())

    os.makedirs('images', exist_ok=True)

    # 1) simple countplot with Matplotlib
    counts = df['Electric Vehicle Type'].value_counts()
    plt.figure(figsize=(8,4))
    counts.plot(kind='bar')
    plt.title('Count of EV Vehicle Types')
    plt.ylabel('Number of Registrations')
    plt.tight_layout()
    plt.savefig('images/vehicle_type_counts.png')
    plt.close()

    # 2) Seaborn distribution
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, y='Electric Vehicle Type', order=counts.index)
    plt.title('EV Type Distribution')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('images/vehicle_type_distribution_seaborn.png')
    plt.close()

    print("Saved images/vehicle_type_counts.png and vehicle_type_distribution_seaborn.png")

if __name__ == '__main__':
    main()