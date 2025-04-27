#!/usr/bin/env python3
"""
05_visualizations.py

Additional final charts: electric range vs price scatter with regression line.
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

    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.regplot(data=df,
                x='Electric Range',
                y='Base MSRP',
                scatter_kws={'alpha':0.3, 's':10},
                line_kws={'color':'red'})
    plt.title('Electric Range vs. Base MSRP')
    plt.xlabel('Electric Range (miles)')
    plt.ylabel('Base MSRP ($)')
    plt.tight_layout()
    plt.savefig('images/range_vs_price_scatter.png')
    plt.close()
    print("Saved images/range_vs_price_scatter.png")

if __name__ == '__main__':
    main()