#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate why maximum stress point shows
low cycles to failure despite low stress amplitude values.

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

def check_stress_cycle_relationship():
    """Diagnostic function to check stress-cycle relationship"""
    print("\nDIAGNOSTIC: Investigating stress-cycle relationship...")
    
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
    
    # Original strain values diagnostics
    print("\nDIAGNOSTIC: Strain signal statistics")
    print(f"Original strain range: {np.min(strain_signal_clean):.2e} to {np.max(strain_signal_clean):.2e}")
    print(f"Strain range (max-min): {np.max(strain_signal_clean) - np.min(strain_signal_clean):.2e}")
    
    # Plot the strain signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, strain_signal_clean, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Major Principal Strain')
    plt.title(f'Strain Signal at Maximum Stress Point ({max_row},{max_col})')
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(), 'strain_signal_at_max_point.png'), dpi=300)
    plt.close()
    
    # Rainflow analysis
    print("\nPerforming rainflow analysis on maximum stress point...")
    cycles = identify_cycles(strain_signal_clean)
    
    if len(cycles) == 0:
        print(f"No cycles identified for maximum stress point.")
        return
        
    print(f"\nDIAGNOSTIC: Rainflow analysis")
    print(f"Identified {len(cycles)} cycles at maximum stress point.")
    
    # Examine the cycles in detail
    print("\nDIAGNOSTIC: Top 5 cycles by strain range:")
    
    # Extract cycle data
    strain_ranges = np.array([c[0] for c in cycles])  # First column is range
    means = np.array([c[1] for c in cycles]) if cycles[0][1] is not None else np.zeros_like(strain_ranges)
    counts = np.array([c[2] for c in cycles]) if len(cycles[0]) > 2 else np.ones_like(strain_ranges)
    
    # Sort cycles by strain range (descending)
    sorted_idx = np.argsort(strain_ranges)[::-1]
    sorted_ranges = strain_ranges[sorted_idx]
    sorted_means = means[sorted_idx]
    sorted_counts = counts[sorted_idx]
    
    # Print top 5 cycles
    for i in range(min(5, len(sorted_ranges))):
        print(f"Cycle {i+1}: Range = {sorted_ranges[i]:.2e}, Mean = {sorted_means[i]:.2e}, Count = {sorted_counts[i]}")
    
    # Analyze fatigue using the original analyze_fatigue function
    print("\nAnalyzing fatigue with default parameters...")
    fatigue_results = analyze_fatigue(cycles)
    
    # Get cycles to failure for each strain range
    N_f_cycles = fatigue_results.get('N_f_cycles', [])
    
    # Convert strain to stress for the S-N curve (using elastic modulus as approximation)
    stress_amplitudes = (strain_ranges / 2) * E / 1e6  # Convert to MPa
    
    print(f"\nDIAGNOSTIC: Top 5 cycles by stress amplitude:")
    sorted_stress_idx = np.argsort(stress_amplitudes)[::-1]
    sorted_stress_amps = stress_amplitudes[sorted_stress_idx]
    sorted_stress_cycles = N_f_cycles[sorted_stress_idx] if len(N_f_cycles) > 0 else []
    
    for i in range(min(5, len(sorted_stress_amps))):
        if len(sorted_stress_cycles) > i:
            print(f"Cycle {i+1}: Stress Amplitude = {sorted_stress_amps[i]:.2f} MPa, "
                  f"Cycles to Failure = {sorted_stress_cycles[i]:.2e}")
        else:
            print(f"Cycle {i+1}: Stress Amplitude = {sorted_stress_amps[i]:.2f} MPa")
    
    # Print original fatigue parameters used
    print("\nDIAGNOSTIC: Fatigue parameters used in original analysis:")
    # Extract from analyze_fatigue in fatigue_analysis.py
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    print(f"Young's modulus (E): {E_mod/1e9:.0f} GPa")
    print(f"Fatigue strength coefficient (sigma_f'): {sigma_f_prime/1e6:.0f} MPa")
    print(f"Fatigue ductility coefficient (epsilon_f'): {epsilon_f_prime:.2f}")
    print(f"Fatigue strength exponent (b): {b}")
    print(f"Fatigue ductility exponent (c): {c}")
    print(f"Safety factor: {safety_factor}")
    
    # Now recalculate fatigue life WITHOUT the safety factor and WITHOUT the 1e6 cycles cap
    stress_amps = (strain_ranges / 2) * E / 1e6  # Convert to MPa
    N_values = np.logspace(1, 15, 1000)  # Extended range for accurate calculation
    
    uncapped_N_f_cycles = []
    for strain_range in strain_ranges:
        strain_amp = strain_range / 2
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life WITHOUT safety factor
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))]
        uncapped_N_f_cycles.append(N_f)  # No cap, no safety factor
    
    uncapped_N_f_cycles = np.array(uncapped_N_f_cycles)
    
    # Print comparison of results
    print("\nDIAGNOSTIC: Comparison of cycles to failure with and without safety factor/capping:")
    for i in range(min(5, len(sorted_ranges))):
        idx = sorted_idx[i]
        if idx < len(N_f_cycles) and idx < len(uncapped_N_f_cycles):
            print(f"Cycle {i+1}: Strain Range = {strain_ranges[idx]:.2e}")
            print(f"  - With safety factor ({safety_factor}x) and 1e6 cap: {N_f_cycles[idx]:.2e} cycles")
            print(f"  - Without safety factor and cap: {uncapped_N_f_cycles[idx]:.2e} cycles")
            print(f"  - Difference factor: {uncapped_N_f_cycles[idx]/N_f_cycles[idx]:.1f}x")
    
    # Create a comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Define theoretical S-N curve parameters
    N_plot_values = np.logspace(1, 8, 1000)
    b_exp = -0.1  # Typical basquin exponent
    stress_at_1k_cycles = 8800  # MPa
    A_factor = stress_at_1k_cycles / (1000 ** b_exp)
    theoretical_stress = A_factor * (N_plot_values ** b_exp)
    
    # Plot theoretical curve
    plt.loglog(N_plot_values, theoretical_stress, 'b-', linewidth=2.5, label='Theoretical S-N Curve')
    
    # Plot actual data points WITH safety factor
    valid_idx = (N_f_cycles > 0) & (stress_amplitudes > 0)
    if np.any(valid_idx):
        plt.scatter(N_f_cycles[valid_idx], stress_amplitudes[valid_idx], 
                   s=60, alpha=0.7, color='red', 
                   label=f'With safety factor ({safety_factor}x)')
    
    # Plot actual data points WITHOUT safety factor
    valid_idx = (uncapped_N_f_cycles > 0) & (stress_amplitudes > 0)
    if np.any(valid_idx):
        plt.scatter(uncapped_N_f_cycles[valid_idx], stress_amplitudes[valid_idx], 
                   s=60, alpha=0.7, color='green', 
                   label='Without safety factor')
    
    plt.xlabel('Cycles to Failure (N)', fontsize=14)
    plt.ylabel('Stress Amplitude, S (MPa)', fontsize=14)
    plt.title('S-N Curve Comparison: Effect of Safety Factor', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'sn_curve_safety_factor_comparison.png'), dpi=300)
    
    print("\nDiagnostic complete! Check strain_signal_at_max_point.png and sn_curve_safety_factor_comparison.png")

if __name__ == "__main__":
    check_stress_cycle_relationship() 