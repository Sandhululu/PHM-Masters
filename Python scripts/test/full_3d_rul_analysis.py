#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full 3D RUL Analysis

This script demonstrates how we're loading and processing the 3D strain data,
showing the dimensions and structure of the data from all CSV files.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data, read_strain_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def examine_3d_strain_data():
    """Load the strain data and examine its structure to understand the 3D nature"""
    print("\n=== EXAMINING 3D STRAIN DATA STRUCTURE ===\n")
    
    # Method 1: Using load_all_data()
    print("Method 1: Using load_all_data()")
    data = load_all_data()
    
    # Check format of returned data
    if isinstance(data, dict):
        # Dictionary format
        print("  Data returned as dictionary")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape = {value.shape}, type = {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        # Tuple format (older version)
        print("  Data returned as tuple")
        avg_exx, dic_exx, avg_eyy, dic_eyy, thermal_strain, time_points, *_ = data
        print(f"  DICExx shape: {dic_exx.shape}")
        print(f"  DICEyy shape: {dic_eyy.shape}")
        print(f"  Thermal strain shape: {thermal_strain.shape}")
        print(f"  Time points: {len(time_points)} points from {time_points[0]} to {time_points[-1]} seconds")
    
    # Method 2: Direct loading with read_strain_data()
    print("\nMethod 2: Direct loading with read_strain_data()")
    avg_exx, dic_exx = read_strain_data('exx')
    avg_eyy, dic_eyy = read_strain_data('eyy')
    
    print(f"  DICExx shape: {dic_exx.shape}")
    print(f"  DICEyy shape: {dic_eyy.shape}")
    
    # Calculate principal strains
    major, minor, shear = calculate_principal_strains((dic_exx + dic_eyy) / 2, dic_exx, dic_eyy)
    
    print(f"\nPrincipal strain shapes:")
    print(f"  Major principal strain: {major.shape}")
    print(f"  Minor principal strain: {minor.shape}")
    print(f"  Max shear strain: {shear.shape}")
    
    # Examine signal extraction for a specific point
    print("\nSignal extraction for a specific point:")
    rows, cols = major.shape[1], major.shape[2]
    r, c = rows//2, cols//2  # Center point
    
    signal = major[:, r, c]
    print(f"  Signal shape for point ({r},{c}): {signal.shape}")
    print(f"  Signal has {len(signal)} time points")
    
    # Examine cycles identified
    cycles = identify_cycles(signal, is_shear_strain=False)
    if cycles is not None and len(cycles) > 0:
        print(f"  Identified {len(cycles)} cycles from signal")
    else:
        print("  No cycles identified from signal")

def main():
    """Run the 3D strain data examination"""
    # Examine 3D strain data structure
    examine_3d_strain_data()

if __name__ == "__main__":
    main() 