#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to print the cycles_array from fatigue_analysis
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue

def main():
    """Load data, identify cycles, and print the cycles_array"""
    print("\n=== Printing cycles_array from fatigue_analysis ===\n")
    
    # Load data
    print("Loading strain data...")
    data = load_all_data()
    
    # Extract and process data
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
    else:
        _, strain_exx, _, strain_eyy, thermal_strain, _, _, _, _, _ = data
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, _, _ = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Find the point with max strain
    max_principal_loc = np.unravel_index(
        np.nanargmax(np.nanmax(major_principal_strain, axis=0)), 
        major_principal_strain[0].shape
    )
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    
    # Extract strain signal at this location
    strain_signal = major_principal_strain[:, max_principal_loc[0], max_principal_loc[1]]
    
    # Identify cycles from strain data
    print("\nIdentifying cycles in strain data...")
    cycles = identify_cycles(strain_signal)
    
    print(f"Number of cycles identified: {len(cycles)}")
    print(f"Cycles shape: {cycles.shape}")
    print(f"Sample of cycles data (first 5 rows):")
    for i in range(min(5, len(cycles))):
        print(f"  {i}: {cycles[i]}")
    
    # Analyze fatigue
    print("\nAnalyzing fatigue to get cycles_array...")
    fatigue_results = analyze_fatigue(cycles)
    
    # Now print the cycles_array in a more readable format
    cycles_array = fatigue_results.get('cycles_array')
    print("\nCycles Array from fatigue_analysis.py:")
    print(f"Type: {type(cycles_array)}")
    print(f"Shape: {cycles_array.shape}")
    print(f"Content:")
    print(cycles_array)
    
    # Show relationship between cycles_array and the other data
    print("\nRelationship between cycles_array and other data:")
    print("cycles_array is just an index array (0 to n-1) used to reference strain ranges")
    print("For example, for the first 5 entries:")
    
    cycles = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    damages = fatigue_results.get('damages', np.array([]))
    
    for i in range(min(5, len(cycles_array))):
        idx = cycles_array[i]
        print(f"  Index {idx}: Strain range = {cycles[i]:.8e}, Count = {counts[i]}, Damage = {damages[i]:.8e}")
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 