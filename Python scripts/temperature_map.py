#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a temperature map from thermal strain data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the project
from data_loader import load_all_data

def create_temperature_map(thermal_strain, time_points):
    """Create a temperature map from thermal strain data
    
    Args:
        thermal_strain: 3D array of thermal strain data (time, rows, cols)
        time_points: Array of time points
        
    Returns:
        None (saves map as file)
    """
    # Calculate average thermal strain across all time points
    avg_thermal_strain = np.nanmean(thermal_strain, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap of thermal strain (which corresponds to temperature)
    im = ax.imshow(avg_thermal_strain, cmap='inferno')
    ax.set_title('Average Thermal Strain Map (Temperature Distribution)', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Thermal Strain (ε)', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'thermal_strain_map.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), 'thermal_strain_map.png'), bbox_inches='tight', dpi=300)
    
    print("Thermal strain map saved as 'thermal_strain_map.svg/png'")
    
    # Also create a time-based visualization showing thermal strain evolution
    # Pick a representative point in the middle of the map
    rows, cols = avg_thermal_strain.shape
    mid_row, mid_col = rows // 2, cols // 2
    
    # Create a second figure for time series
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot thermal strain over time at the middle point
    thermal_at_mid = thermal_strain[:, mid_row, mid_col]
    ax1.plot(time_points, thermal_at_mid, 'r-', linewidth=2)
    ax1.set_title(f'Thermal Strain Over Time at Point ({mid_row},{mid_col})', fontsize=14)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Thermal Strain', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # For the second plot, show the average thermal strain across the entire map over time
    avg_thermal_over_time = np.nanmean(thermal_strain, axis=(1, 2))
    ax2.plot(time_points, avg_thermal_over_time, 'b-', linewidth=2)
    ax2.set_title('Average Thermal Strain Over Time (Entire Component)', fontsize=14)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Average Thermal Strain', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Save time series figure
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'thermal_strain_time_series.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), 'thermal_strain_time_series.png'), bbox_inches='tight', dpi=300)
    
    print("Thermal strain time series saved as 'thermal_strain_time_series.svg/png'")
    plt.show()

def main():
    """Main function to generate temperature map"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract thermal strain and time points
    thermal_strain = data['thermal_strain']
    time_points = data['time_points']
    
    print("\nCreating temperature map from thermal strain...")
    create_temperature_map(thermal_strain, time_points)
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 