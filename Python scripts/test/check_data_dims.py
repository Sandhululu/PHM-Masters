#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check the dimensions of the DIC data and time points
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_loader import load_all_data

def main():
    """Check data dimensions"""
    print("\n=== Checking DIC Data Dimensions ===\n")
    
    # Load strain data
    print("Loading strain data...")
    data = load_all_data()
    
    # Extract data
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
        time_points = data['time_points']
    else:
        # Fallback to tuple format for backward compatibility
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, _ = data
    
    # Print dimensions
    print("\nData Dimensions:")
    print(f"  DIC Exx: {strain_exx.shape}")
    print(f"  DIC Eyy: {strain_eyy.shape}")
    print(f"  Thermal Strain: {thermal_strain.shape}")
    print(f"  Time Points: {time_points.shape}")
    
    # Print time points info
    print("\nTime Points Information:")
    print(f"  Number of time points: {len(time_points)}")
    print(f"  Time range: {time_points[0]} to {time_points[-1]} seconds")
    print(f"  Time step: {time_points[1] - time_points[0]} seconds")
    print(f"  Total time span: {time_points[-1] - time_points[0]} seconds")
    
    # Check if number of time points matches thermal strain first dimension
    if len(time_points) == thermal_strain.shape[0]:
        print(f"\nTime dimension matches: {len(time_points)} time points = {thermal_strain.shape[0]} strain points")
    else:
        print(f"\nWarning: Time dimension mismatch: {len(time_points)} time points ≠ {thermal_strain.shape[0]} strain points")
    
    # Check number of CSV files in original data
    print("\nChecking for CSV files in DIC data directory...")
    import glob
    dirDIC = f'/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/exx'
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    print(f"  Number of CSV files: {len(csv_files)}")
    
    # Compare with value in original.py
    original_time_value = 70.8
    original_time_points = 354
    time_per_point = original_time_value / (original_time_points - 1)
    
    print(f"\nOriginal time information from original.py:")
    print(f"  Total time span: {original_time_value} seconds")
    print(f"  Number of points: {original_time_points}")
    print(f"  Time per point: {time_per_point} seconds")
    
    # Calculate time per file if each file represents one timestep
    if len(csv_files) > 0:
        time_per_file = original_time_value / len(csv_files)
        print(f"\nCalculated time per CSV file: {time_per_file:.4f} seconds")
    
    return 0

if __name__ == "__main__":
    main() 