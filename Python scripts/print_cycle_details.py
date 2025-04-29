#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to print out rainflow cycles in full format (range, mean, count, i_start, i_end)

@author: Jayron Sandhu
"""

import numpy as np
import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def main():
    """Load strain data, identify cycles, and print them in full format"""
    print("\n=== Rainflow Cycle Analysis - Full Format ===\n")
    
    # Load strain data
    print("Loading strain data...")
    data = load_all_data()
    
    # Extract data from dictionary or tuple
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
        time_points = data['time_points']
    else:
        # Fallback to tuple format for backward compatibility
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, _ = data
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Find the point with maximum principal strain
    max_principal_loc = np.unravel_index(
        np.nanargmax(np.nanmax(major_principal_strain, axis=0)), 
        major_principal_strain[0].shape
    )
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    
    # Extract strain signal at this location
    strain_signal = major_principal_strain[:, max_principal_loc[0], max_principal_loc[1]]
    
    # Identify cycles in strain signal
    print("\nIdentifying cycles in strain signal...")
    cycles = identify_cycles(strain_signal)
    
    # Check if cycles were identified
    if len(cycles) == 0:
        print("No cycles identified. Exiting.")
        return 1
    
    # Print cycles in full format
    print(f"\nTotal cycles identified: {len(cycles)}")
    print(f"Format: (range, mean, count, i_start, i_end)")
    print("\nDetailed cycle information:")
    
    # Create a pandas DataFrame for better formatting
    columns = ["Range", "Mean", "Count", "Start Index", "End Index"]
    df = pd.DataFrame(cycles, columns=columns)
    
    # Add a cycle number column at the beginning
    df.insert(0, "Cycle #", range(1, len(cycles) + 1))
    
    # Print the DataFrame
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}" if isinstance(x, (float, np.floating)) else str(x)))
    
    # Save to CSV
    csv_path = os.path.join(os.getcwd(), "cycle_details.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCycle details saved to: {csv_path}")
    
    # Print some additional statistics
    print("\nCycle Statistics:")
    print(f"  Minimum range: {df['Range'].min():.6e}")
    print(f"  Maximum range: {df['Range'].max():.6e}")
    print(f"  Average range: {df['Range'].mean():.6e}")
    print(f"  Total cycle count: {df['Count'].sum()}")
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 