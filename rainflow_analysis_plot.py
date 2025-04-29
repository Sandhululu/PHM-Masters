#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize rainflow analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_data
from fatigue_analysis import identify_cycles

def plot_rainflow_results(strain_signal):
    """
    Create visualizations of rainflow analysis results
    
    Args:
        strain_signal: Array of strain values
    """
    # Clean signal by removing NaNs
    mask = ~np.isnan(strain_signal)
    if not np.any(mask):
        print("Warning: All NaN values in signal")
        return
    
    # Interpolate missing data
    indices = np.arange(len(strain_signal))
    clean_signal = np.interp(indices, indices[mask], strain_signal[mask])
    
    # Get rainflow cycles
    cycles = identify_cycles(clean_signal)
    
    if len(cycles) == 0:
        print("No cycles identified in signal")
        return
        
    # Extract cycle data
    ranges = np.array([c[0] for c in cycles])
    means = np.array([c[1] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Histogram of cycle ranges
    ax1 = plt.subplot(131)
    ax1.hist(ranges, bins=20, weights=counts, edgecolor='black')
    ax1.set_title('Cycle Range Distribution')
    ax1.set_xlabel('Strain Range')
    ax1.set_ylabel('Count')
    ax1.grid(True)
    
    # 2. Range vs Mean scatter plot
    ax2 = plt.subplot(132)
    scatter = ax2.scatter(means, ranges, c=counts, 
                         cmap='viridis', s=50*counts/max(counts))
    plt.colorbar(scatter, label='Count')
    ax2.set_title('Cycle Range vs Mean')
    ax2.set_xlabel('Mean Strain')
    ax2.set_ylabel('Strain Range')
    ax2.grid(True)
    
    # 3. Cycle count summary
    ax3 = plt.subplot(133)
    ax3.axis('off')
    summary_text = [
        'Rainflow Analysis Summary',
        '----------------------',
        f'Total Cycles: {len(cycles)}',
        f'Total Count: {sum(counts):.0f}',
        '',
        'Range Statistics:',
        f'Max: {max(ranges):.2e}',
        f'Min: {min(ranges):.2e}',
        f'Mean: {np.mean(ranges):.2e}',
        '',
        'Mean Statistics:',
        f'Max: {max(means):.2e}',
        f'Min: {min(means):.2e}',
        f'Mean: {np.mean(means):.2e}'
    ]
    ax3.text(0.1, 0.5, '\n'.join(summary_text), 
             transform=ax3.transAxes, 
             verticalalignment='center',
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('rainflow_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Load data and create rainflow analysis plots"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract major principal strain data
    if isinstance(data, dict):
        major_principal_strain = data.get('major_principal_strain')
        time_points = data.get('time_points')
    else:
        print("Error: Could not load strain data properly")
        return
    
    if major_principal_strain is None:
        print("Error: Missing required strain data")
        return
    
    # Find point with maximum strain for visualization
    try:
        max_strain = np.nanmax(np.abs(major_principal_strain), axis=0)
        if np.all(np.isnan(max_strain)):
            print("Error: All strain values are NaN")
            return
            
        max_point = np.unravel_index(np.nanargmax(max_strain), max_strain.shape)
        row, col = max_point
        
        print(f"\nCreating rainflow analysis plots for point ({row}, {col})...")
        strain_signal = major_principal_strain[:, row, col]
        
        # Create visualization
        plot_rainflow_results(strain_signal)
        print("\nPlots saved as 'rainflow_analysis.png'")
        
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == '__main__':
    main() 