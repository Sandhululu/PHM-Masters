#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize rainflow cycle analysis by showing original signal and extracted cycles
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rainflow
import pandas as pd
from data_loader import load_all_data
from fatigue_analysis import identify_cycles

def plot_rainflow_visualization(strain_signal, time_points):
    """
    Create a visualization showing original signal and identified rainflow cycles
    
    Args:
        strain_signal: Array of strain values
        time_points: Array of corresponding time points
    """
    # Clean signal by removing NaNs
    mask = ~np.isnan(strain_signal)
    if not np.any(mask):
        print("Warning: All NaN values in signal")
        return
    
    # Interpolate missing data
    indices = np.arange(len(strain_signal))
    clean_signal = np.interp(indices, indices[mask], strain_signal[mask])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot original signal
    ax1.plot(time_points, clean_signal, 'b-', label='Original Signal')
    ax1.set_title('Original Strain Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Major Principal Strain')
    ax1.grid(True)
    ax1.legend()
    
    # Get rainflow cycles
    cycles = identify_cycles(clean_signal)
    
    if len(cycles) == 0:
        print("No cycles identified in signal")
        return
    
    # Plot original signal in bottom plot
    ax2.plot(time_points, clean_signal, 'b-', alpha=0.3, label='Original Signal')
    
    # Plot identified cycles
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cycles)))
    
    for i, cycle in enumerate(cycles):
        rng, mean, count = cycle[0], cycle[1], cycle[2]
        
        # Calculate cycle start and end points
        cycle_duration = len(time_points) // len(cycles)  # Approximate duration
        t_start = time_points[i * cycle_duration]
        t_end = time_points[min((i + 1) * cycle_duration, len(time_points) - 1)]
        
        # Plot cycle as rectangle
        rect = Rectangle((t_start, mean - rng/2), t_end - t_start, rng,
                        facecolor=colors[i], alpha=0.3)
        ax2.add_patch(rect)
        
        # Add cycle information
        ax2.text(t_start, mean + rng/2, f'Cycle {i+1}\nRange: {rng:.2e}\nMean: {mean:.2e}',
                fontsize=8, verticalalignment='bottom')
    
    ax2.set_title('Rainflow Cycle Identification')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Major Principal Strain')
    ax2.grid(True)
    ax2.legend()
    
    # Set y-limits to show full range
    max_strain = np.max(clean_signal)
    min_strain = np.min(clean_signal)
    strain_range = max_strain - min_strain
    ax2.set_ylim(min_strain - 0.1*strain_range, max_strain + 0.2*strain_range)
    
    # Add summary statistics
    stats_text = (f'Total Cycles: {len(cycles)}\n'
                 f'Max Range: {np.max([c[0] for c in cycles]):.2e}\n'
                 f'Mean Range: {np.mean([c[0] for c in cycles]):.2e}\n'
                 f'Total Count: {np.sum([c[2] for c in cycles]):.1f}')
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('rainflow_visualization.png', bbox_inches='tight', dpi=300)
    print(f"Created visualization with {len(cycles)} identified cycles")
    plt.close()

def main():
    """Load data and create rainflow visualization"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract major principal strain data
    if isinstance(data, dict):
        major_principal_strain = data.get('major_principal_strain')
        time_points = data.get('time_points')
    else:
        print("Error: Could not load strain data properly")
        return
    
    if major_principal_strain is None or time_points is None:
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
        
        print(f"\nCreating rainflow visualization for point ({row}, {col})...")
        strain_signal = major_principal_strain[:, row, col]
        
        # Create visualization
        plot_rainflow_visualization(strain_signal, time_points)
        print("\nVisualization saved as 'rainflow_visualization.png'")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == '__main__':
    main() 