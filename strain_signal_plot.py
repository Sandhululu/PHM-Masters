#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot original strain signal
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_data

def plot_strain_signal(strain_signal, time_points):
    """
    Create a visualization showing just the original strain signal
    
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
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot signal
    plt.plot(time_points, clean_signal, 'b-', linewidth=1)
    plt.title('Strain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Major Principal Strain')
    plt.grid(True)
    
    # Save plot
    plt.savefig('strain_signal.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Load data and create strain signal plot"""
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
        
        print(f"\nCreating strain signal plot for point ({row}, {col})...")
        strain_signal = major_principal_strain[:, row, col]
        
        # Create visualization
        plot_strain_signal(strain_signal, time_points)
        print("\nPlot saved as 'strain_signal.png'")
        
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == '__main__':
    main() 