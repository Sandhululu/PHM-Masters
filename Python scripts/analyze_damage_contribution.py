#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the damage contribution by different strain ranges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    """Analyze damage contribution by strain range"""
    print("\n=== Damage Contribution Analysis ===\n")
    
    # Load the enhanced cycle details CSV file with damage
    csv_path = os.path.join(os.getcwd(), "enhanced_cycle_details_with_damage.csv")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} cycles from {csv_path}")
    
    # Sort by damage in descending order
    df_sorted = df.sort_values(by='Damage', ascending=False)
    
    # Calculate cumulative damage
    df_sorted['Cumulative Damage'] = df_sorted['Damage'].cumsum()
    df_sorted['Damage Percentage'] = df_sorted['Damage'] / df_sorted['Damage'].sum() * 100
    df_sorted['Cumulative Percentage'] = df_sorted['Cumulative Damage'] / df_sorted['Damage'].sum() * 100
    
    # Print the top 10 cycles by damage
    print("\nTop 10 cycles by damage contribution:")
    print(df_sorted[['Cycle #', 'Range', 'Mean', 'Count', 'N_f', 'Damage', 'Damage Percentage', 'Cumulative Percentage']].head(10))
    
    # Find how many cycles contribute to 90% of total damage
    cycles_for_90pct = sum(df_sorted['Cumulative Percentage'] <= 90)
    print(f"\nNumber of cycles contributing to 90% of total damage: {cycles_for_90pct} out of {len(df_sorted)}")
    
    # Group cycles by strain range bins
    # Create logarithmically spaced bins for strain range
    min_range = df['Range'].min()
    max_range = df['Range'].max()
    num_bins = 10
    
    # Create log-spaced bins
    log_bins = np.logspace(np.log10(min_range), np.log10(max_range), num_bins+1)
    
    # Assign each cycle to a bin
    df['Range Bin'] = pd.cut(df['Range'], bins=log_bins, right=False)
    
    # Calculate damage contribution by range bin
    damage_by_bin = df.groupby('Range Bin')['Damage'].sum().reset_index()
    damage_by_bin['Percentage'] = damage_by_bin['Damage'] / df['Damage'].sum() * 100
    
    # Sort by damage percentage
    damage_by_bin = damage_by_bin.sort_values(by='Percentage', ascending=False)
    
    print("\nDamage contribution by strain range bins:")
    for i, row in damage_by_bin.iterrows():
        bin_range = row['Range Bin']
        percentage = row['Percentage']
        print(f"  Bin {bin_range}: {percentage:.2f}%")
    
    # Create a pareto chart of damage by cycle
    plt.figure(figsize=(12, 8))
    
    # Bar chart of individual damage percentage
    bars = plt.bar(range(len(df_sorted)), df_sorted['Damage Percentage'])
    
    # Line chart of cumulative percentage
    plt.plot(range(len(df_sorted)), df_sorted['Cumulative Percentage'], 'r-', linewidth=2)
    
    # Highlight the top 10 cycles
    for i in range(min(10, len(bars))):
        bars[i].set_color('orange')
    
    # Add horizontal line at 90%
    plt.axhline(y=90, color='blue', linestyle='--', alpha=0.7)
    plt.text(len(df_sorted)*0.02, 91, '90% of damage', color='blue')
    
    # Add annotations
    top_n = sum(df_sorted['Cumulative Percentage'] <= 90)
    plt.annotate(f'Top {top_n} cycles\n({top_n/len(df_sorted)*100:.1f}% of all cycles)',
                 xy=(top_n, 90),
                 xytext=(top_n + len(df_sorted)*0.05, 80),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 fontsize=10)
    
    plt.title('Pareto Chart of Damage Contribution by Cycle')
    plt.xlabel('Cycle (sorted by damage contribution)')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('damage_pareto.png')
    
    # Create a pie chart of damage by strain range bin
    plt.figure(figsize=(10, 8))
    
    # Only show top bins that contribute at least 1% of damage
    significant_bins = damage_by_bin[damage_by_bin['Percentage'] >= 1]
    
    # If there are too few significant bins, just use all bins
    if len(significant_bins) < 3:
        significant_bins = damage_by_bin
    
    # Format bin labels for readability
    def format_bin_label(bin_range):
        return f"{bin_range.left:.2e} - {bin_range.right:.2e}"
    
    bin_labels = [format_bin_label(bin_range) for bin_range in significant_bins['Range Bin']]
    
    plt.pie(significant_bins['Percentage'], labels=bin_labels, autopct='%1.1f%%',
            startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Damage Contribution by Strain Range')
    plt.tight_layout()
    plt.savefig('damage_by_range_pie.png')
    
    print("\nCreated damage pareto chart: damage_pareto.png")
    print("Created damage by range pie chart: damage_by_range_pie.png")
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 