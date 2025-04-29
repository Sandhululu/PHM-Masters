#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create a simplified S-N curve (stress vs cycles to failure) graph
showing the theoretical curve with only the maximum stress point from DIC data.

@author: Created from user request
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains, calculate_stresses
from fatigue_analysis import identify_cycles, analyze_fatigue

def create_simplified_sn_curve():
    """Generate a simplified S-N curve showing stress vs cycles to failure"""
    print("\nLoading strain data to find maximum stress point...")
    
    # Load DIC data
    data = load_all_data()
    
    # Extract data from dictionary
    if isinstance(data, dict):
        # Use dictionary format
        ThermalStrain = data['thermal_strain']
        DICExx = data['exx']
        DICEyy = data['eyy']
        time_points = data['time_points']
    else:
        # Use tuple format for backward compatibility
        _, DICExx, _, DICEyy, ThermalStrain, time_points = data[:6]
    
    # Material properties for tungsten
    E = 400e9  # Young's modulus (Pa)
    nu = 0.28  # Poisson's ratio
    
    # Calculate stresses
    print("\nCalculating stresses from strain data...")
    stress_xx, stress_yy, stress_von_mises = calculate_stresses(DICExx, DICEyy, E, nu, ThermalStrain)
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Find maximum stress point
    flat_max_vm = np.nanmax(stress_von_mises, axis=0).flatten()
    flat_max_vm[np.isnan(flat_max_vm)] = -np.inf  # Replace NaN with -inf to find true max
    max_vm_idx = np.argmax(flat_max_vm)
    max_row, max_col = max_vm_idx // ThermalStrain.shape[2], max_vm_idx % ThermalStrain.shape[2]
    
    print(f"\nMaximum stress point found at location ({max_row},{max_col})")
    
    # Extract strain signal for maximum stress point
    strain_signal = major_principal_strain[:, max_row, max_col]
    max_vm_stress = np.nanmax(stress_von_mises[:, max_row, max_col])
    print(f"Maximum von Mises stress at this point: {max_vm_stress/1e6:.2f} MPa")
    
    # Clean NaN values if needed
    mask = ~np.isnan(strain_signal)
    if np.sum(mask) < 2:
        print(f"Error: Insufficient valid data at maximum stress point.")
        return
        
    # Interpolate to handle NaNs
    indices = np.arange(len(strain_signal))
    strain_signal_clean = np.interp(indices, indices[mask], strain_signal[mask])
    
    # Rainflow analysis
    print("\nPerforming rainflow analysis on maximum stress point...")
    cycles = identify_cycles(strain_signal_clean)
    
    if len(cycles) == 0:
        print(f"No cycles identified for maximum stress point.")
        return
        
    print(f"Identified {len(cycles)} cycles at maximum stress point.")
    
    # Analyze fatigue
    fatigue_results = analyze_fatigue(cycles)
    
    # Get cycles to failure for each strain range
    strain_ranges = fatigue_results.get('cycles', [])
    N_f_cycles = fatigue_results.get('N_f_cycles', [])
    
    if len(strain_ranges) == 0 or len(N_f_cycles) == 0:
        print(f"No valid fatigue data for maximum stress point.")
        return
    
    # Convert strain to stress for the S-N curve (using elastic modulus as approximation)
    # Since we need stress amplitude vs. cycles to failure
    stress_amplitudes = (strain_ranges / 2) * E / 1e6  # Convert to MPa
    
    print(f"Maximum stress amplitude: {np.max(stress_amplitudes):.2f} MPa")
    print(f"Minimum cycles to failure: {np.min(N_f_cycles):.2e} cycles")
    
    # Create the S-N curve with proper relationship (higher stress = fewer cycles)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Define the range for the S-N curve using classic S-N relationship: S = A * N^b
    # For typical metals, b is negative (around -0.1 to -0.15)
    N_values = np.logspace(1, 8, 1000)  # From 10 to 100 million cycles
    
    # Use basquin's law with typical values for metals
    b_exp = -0.1  # Basquin exponent (negative value)
    
    # Calculate corresponding stress values
    # Choose stress range appropriate for tungsten (MPa)
    stress_at_1k_cycles = 8800  # MPa, high stress for short life
    A_factor = stress_at_1k_cycles / (1000 ** b_exp)  # Calculate constant A
    
    # Generate curve according to S = A * N^b
    stress_values = A_factor * (N_values ** b_exp)
    
    # Plot the main curve
    ax.plot(N_values, stress_values, 'b-', linewidth=2.5)
    
    # Set scales
    ax.set_xscale('log')
    
    # Choose representative points for labeling
    # S1 - high stress, low cycle point
    s1_cycles = 100  # cycles at this stress level
    s1_stress = A_factor * (s1_cycles ** b_exp)
    
    # N1 - lower stress, high cycle point (fatigue limit reference)
    n1_cycles = 1e6
    n1_stress = A_factor * (n1_cycles ** b_exp)
    
    # Add reference lines
    # 1. For S1 and related fatigue life
    ax.axhline(y=s1_stress, linestyle='--', color='gray', linewidth=1)
    ax.axvline(x=s1_cycles, linestyle='--', color='gray', linewidth=1)
    
    # 2. For N1 and related fatigue strength
    ax.axhline(y=n1_stress, linestyle='--', color='red', linewidth=1)
    ax.axvline(x=n1_cycles, linestyle='--', color='red', linewidth=1)
    
    # Add labels
    ax.text(s1_cycles * 0.3, s1_stress * 1.05, "$S_1$", fontsize=12)
    ax.text(n1_cycles * 1.1, n1_stress * 0.9, "Fatigue strength\nat $N_1$ cycles", fontsize=12, color='red')
    ax.text(s1_cycles * 1.1, s1_stress * 0.5, "Fatigue life\nat stress $S_1$", fontsize=12)
    ax.text(n1_cycles * 0.9, n1_stress * 0.1, "$N_1$", fontsize=12, color='red')
    
    # Plot the maximum stress point data
    # Filter out extreme values that could affect visualization
    valid_idx = (N_f_cycles > 0) & (N_f_cycles < 1e9) & (stress_amplitudes > 0)
    if np.any(valid_idx):
        # Sort by stress amplitude descending to show trend better
        sorted_idx = np.argsort(stress_amplitudes[valid_idx])[::-1]
        sorted_stress = stress_amplitudes[valid_idx][sorted_idx]
        sorted_cycles = N_f_cycles[valid_idx][sorted_idx]
        
        # Plot up to 10 points to avoid cluttering
        max_points = min(10, len(sorted_stress))
        ax.scatter(sorted_cycles[:max_points], sorted_stress[:max_points], 
                  s=60, alpha=0.7, color='red', 
                  label=f"Maximum stress point ({max_row},{max_col})")
    
    # Axis labels and title
    ax.set_xlabel('Cycles to Failure (N)', fontsize=14)
    ax.set_ylabel('Stress Amplitude, S (MPa)', fontsize=14)
    ax.set_title('S-N Curve (Stress Amplitude vs. Cycles to Failure)', fontsize=16)
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Set y-axis to linear scale and limits
    ax.set_ylim(0, stress_values[0] * 1.3)  # Give some padding at the top
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'simplified_sn_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Graph saved to 'simplified_sn_curve.png'")

if __name__ == "__main__":
    create_simplified_sn_curve() 