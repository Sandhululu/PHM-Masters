#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to determine the dimensions of the strain data 
and the 1D strain signal passed to rainflow analysis
"""

import os
import sys
import numpy as np

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def main():
    """Show dimensions of strain data"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract data
    if isinstance(data, dict):
        ThermalStrain = data['thermal_strain']
        DICExx = data['strain_exx']
        DICEyy = data['strain_eyy']
        time_points = data['time_points']
    else:
        _, DICExx, _, DICEyy, ThermalStrain, time_points, _, _, _, _ = data
    
    # Print raw data dimensions
    print("\n--- Raw Data Dimensions ---")
    print(f"DICExx shape: {DICExx.shape}")
    print(f"DICEyy shape: {DICEyy.shape}")
    print(f"ThermalStrain shape: {ThermalStrain.shape}")
    print(f"Time points shape: {time_points.shape}")
    print(f"Time points range: {time_points[0]} to {time_points[-1]} (step: {time_points[1]-time_points[0]})")
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Print principal strain dimensions
    print("\n--- Principal Strain Dimensions ---")
    print(f"Major principal strain shape: {major_principal_strain.shape}")
    print(f"Minor principal strain shape: {minor_principal_strain.shape}")
    print(f"Max shear strain shape: {max_shear_strain.shape}")
    
    # Find a valid point (not all NaN)
    print("\nFinding valid spatial points for rainflow analysis...")
    time_dim, rows, cols = major_principal_strain.shape
    valid_points = []
    
    # Sample every 10th point in each dimension to speed up search
    for r in range(0, rows, 10):
        for c in range(0, cols, 10):
            strain_signal = major_principal_strain[:, r, c]
            if not np.all(np.isnan(strain_signal)):
                valid_points.append((r, c))
                if len(valid_points) >= 3:  # Only need a few examples
                    break
        if len(valid_points) >= 3:
            break
    
    # Print info about valid points and their strain signals
    print(f"Found {len(valid_points)} valid spatial points for demonstration")
    
    for i, (r, c) in enumerate(valid_points):
        print(f"\n--- Example Point {i+1}: Location ({r}, {c}) ---")
        strain_signal = major_principal_strain[:, r, c]
        
        print(f"1D strain signal shape: {strain_signal.shape}")
        print(f"First 5 values: {strain_signal[:5]}")
        print(f"Last 5 values: {strain_signal[-5:]}")
        print(f"Min strain: {np.nanmin(strain_signal)}")
        print(f"Max strain: {np.nanmax(strain_signal)}")
        
        # Try to identify cycles
        try:
            cycles = identify_cycles(strain_signal)
            print(f"Number of cycles identified: {len(cycles)}")
            if len(cycles) > 0:
                print(f"Example cycle: {cycles[0]}")
        except Exception as e:
            print(f"Error in cycle identification: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 