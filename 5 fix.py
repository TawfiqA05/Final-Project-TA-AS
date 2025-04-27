#!/usr/bin/env python3
"""
05_visualizations.py

Creates a final scatter + regression‐line chart of Electric Range vs. Base MSRP.
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
    out_path = os.path.join('images', 'range_vs_price_scatter.png')

    plt.figure(figsize=(8,6))
    sns.regplot(data=df,
                x='Electric Range',
                y='Base MSRP',
                scatter_kws={'alpha':0.3, 's':10},
                line_kws={'color':'red', 'lw':2},
                label='Best‐fit line')
    plt.title('Electric Range vs. Base MSRP')
    plt.xlabel('Electric Range (miles)')
    plt.ylabel('Base MSRP ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved scatter plot to {out_path}")

if __name__ == '__main__':
    main()