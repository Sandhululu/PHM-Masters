#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create a graph showing how top 5 stresses from DIC data
impact cycles to failure through principal strain calculation,
rainflow analysis, and Coffin-Manson equation application.

@author: Created from user request
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_loader import load_all_data, print_statistical_summary
from strain_calculator import calculate_principal_strains, calculate_stresses
from fatigue_analysis import identify_cycles, analyze_fatigue

def create_stress_to_failure_graph():
    """Generate a graph showing how stresses impact cycles to failure"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract data from dictionary
    if isinstance(data, dict):
        # Use dictionary format
        ThermalStrain = data['thermal_strain']
        DICExx = data['exx']
        DICEyy = data['eyy']
        time_points = data['time_points']
        high_strain_points = data['high_strain_points']
        max_strain = data['max_strain']
    else:
        # Use tuple format for backward compatibility
        _, DICExx, _, DICEyy, ThermalStrain, time_points, _, _, max_strain, high_strain_points = data
    
    # Material properties for tungsten
    E = 400e9  # Young's modulus (Pa)
    nu = 0.28  # Poisson's ratio
    
    # 1. Calculate stresses
    print("\nCalculating stresses from strain data...")
    stress_xx, stress_yy, stress_von_mises = calculate_stresses(DICExx, DICEyy, E, nu, ThermalStrain)
    
    # 2. Find top 5 stress points based on maximum von Mises stress
    flat_max_vm = np.nanmax(stress_von_mises, axis=0).flatten()
    # Replace NaN with -inf to find true max values
    flat_max_vm[np.isnan(flat_max_vm)] = -np.inf
    # Get indices of top 5 stress points
    top_5_flat_indices = np.argsort(flat_max_vm)[-5:]
    
    # Convert flat indices back to 2D
    rows = ThermalStrain.shape[1]
    cols = ThermalStrain.shape[2]
    top_5_points = [(idx // cols, idx % cols) for idx in top_5_flat_indices]
    
    # 3. Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # 4. Process each point - rainflow analysis and Coffin-Manson
    results = []
    
    # Material properties for Coffin-Manson
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    print("\nAnalyzing top 5 stress points...")
    
    for i, (row, col) in enumerate(top_5_points):
        point_id = f"Point {i+1} ({row},{col})"
        max_vm_stress = np.nanmax(stress_von_mises[:, row, col])
        
        # Extract strain signal
        strain_signal = major_principal_strain[:, row, col]
        
        # Clean NaN values if needed
        mask = ~np.isnan(strain_signal)
        if np.sum(mask) < 2:
            print(f"Skipping {point_id} - insufficient valid data")
            continue
            
        # Interpolate to handle NaNs
        indices = np.arange(len(strain_signal))
        strain_signal_clean = np.interp(indices, indices[mask], strain_signal[mask])
        
        # Rainflow analysis
        cycles = identify_cycles(strain_signal_clean)
        
        if len(cycles) == 0:
            print(f"No cycles identified for {point_id}")
            continue
            
        # Analyze fatigue
        fatigue_results = analyze_fatigue(cycles)
        
        # Get cycles to failure for each strain range
        strain_ranges = fatigue_results.get('cycles', [])
        N_f_cycles = fatigue_results.get('N_f_cycles', [])
        
        if len(strain_ranges) == 0 or len(N_f_cycles) == 0:
            print(f"No valid fatigue data for {point_id}")
            continue
            
        # Store results for plotting
        results.append({
            'point_id': point_id,
            'max_stress': max_vm_stress / 1e6,  # Convert to MPa
            'strain_ranges': strain_ranges,
            'N_f_cycles': N_f_cycles,
            'cycles_count': len(cycles)
        })
        
        print(f"Analyzed {point_id}: Max Stress = {max_vm_stress/1e6:.2f} MPa, "
              f"Cycles = {len(cycles)}, Min N_f = {np.min(N_f_cycles):.2e}")
    
    # Create the visualization
    if not results:
        print("No valid results to plot")
        return
    
    # Prepare for the comprehensive visualization
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1.5, 1.5])
    
    # 1. Top left: Bar chart of max stresses
    ax1 = fig.add_subplot(gs[0, 0])
    point_ids = [r['point_id'] for r in results]
    max_stresses = [r['max_stress'] for r in results]
    
    bars = ax1.bar(point_ids, max_stresses, color='steelblue')
    ax1.set_ylabel('Max von Mises Stress (MPa)')
    ax1.set_title('Maximum Stress at Each Point')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 2. Top right: Bar chart of cycle counts
    ax2 = fig.add_subplot(gs[0, 1])
    cycle_counts = [r['cycles_count'] for r in results]
    
    bars = ax2.bar(point_ids, cycle_counts, color='indianred')
    ax2.set_ylabel('Number of Cycles')
    ax2.set_title('Cycles Identified by Rainflow Analysis')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom')
    
    # 3. Middle: Strain Range vs Cycles to Failure (S-N curve style)
    ax3 = fig.add_subplot(gs[1, :])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        strain_ranges = result['strain_ranges']
        N_f_cycles = result['N_f_cycles']
        point_id = result['point_id']
        
        # Sort by strain range for better visualization
        sorted_idx = np.argsort(strain_ranges)
        strain_ranges = strain_ranges[sorted_idx]
        N_f_cycles = N_f_cycles[sorted_idx]
        
        ax3.loglog(strain_ranges, N_f_cycles, 'o-', color=colors[i], label=point_id)
    
    ax3.set_xlabel('Strain Range (ε)')
    ax3.set_ylabel('Cycles to Failure (N_f)')
    ax3.set_title('Strain Range vs Cycles to Failure (S-N Curve)')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend(loc='best')
    
    # 4. Bottom: Coffin-Manson equation visualization
    ax4 = fig.add_subplot(gs[2, :])
    
    # Generate strain range values for visualization
    strain_range_values = np.logspace(-6, -2, 1000)
    N_values = np.logspace(1, 10, 1000)
    
    # Generate a surface for the Coffin-Manson equation
    X, Y = np.meshgrid(N_values, strain_range_values)
    
    # Calculate strain components from Coffin-Manson equation
    Z1 = (sigma_f_prime/E_mod) * (2*X)**b  # Elastic component
    Z2 = epsilon_f_prime * (2*X)**c        # Plastic component
    Z = Z1 + Z2                           # Total strain amplitude
    
    # Plot the line for different cycle counts
    cycle_counts = [10, 100, 1000, 10000, 100000, 1000000]
    for cycle_count in cycle_counts:
        elastic_strain = (sigma_f_prime/E_mod) * (2*cycle_count)**b
        plastic_strain = epsilon_f_prime * (2*cycle_count)**c
        total_strain = elastic_strain + plastic_strain
        
        ax4.loglog([cycle_count], [total_strain], 'ko', markersize=8)
        ax4.text(cycle_count*1.1, total_strain*1.1, f"{cycle_count}", 
                 fontsize=9, ha='left', va='bottom')
    
    # Plot the elastic, plastic and total components
    N_range = np.logspace(1, 10, 100)
    elastic_strains = (sigma_f_prime/E_mod) * (2*N_range)**b
    plastic_strains = epsilon_f_prime * (2*N_range)**c
    total_strains = elastic_strains + plastic_strains
    
    ax4.loglog(N_range, elastic_strains, 'b--', linewidth=1.5, label='Elastic Component')
    ax4.loglog(N_range, plastic_strains, 'r--', linewidth=1.5, label='Plastic Component')
    ax4.loglog(N_range, total_strains, 'k-', linewidth=2, label='Total Strain')
    
    # Add the actual points from our analysis
    for i, result in enumerate(results):
        N_f_cycles = result['N_f_cycles']
        strain_ranges = result['strain_ranges']
        point_id = result['point_id']
        
        # Convert strain ranges to amplitudes
        strain_amps = strain_ranges / 2
        
        # Plot the minimum life point prominently
        min_idx = np.argmin(N_f_cycles)
        ax4.loglog([N_f_cycles[min_idx]], [strain_amps[min_idx]], 'o', 
                  color=colors[i], markersize=10, label=f"{point_id} (Min Life)")
    
    ax4.set_xlabel('Cycles to Failure (N_f)')
    ax4.set_ylabel('Strain Amplitude (ε_a)')
    ax4.set_title('Coffin-Manson Equation: Strain Amplitude vs Cycles to Failure')
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend(loc='best')
    
    # Add overall title and information
    plt.suptitle('Stress to Fatigue Life Analysis for Tungsten', fontsize=16, fontweight='bold')
    
    # Add information about the analysis process
    process_text = (
        "Analysis Process:\n"
        "1. DIC strain data → Stress calculation using material properties\n"
        "2. Identify top 5 stress points from von Mises stress\n"
        "3. Calculate principal strains at these points\n"
        "4. Apply rainflow analysis to identify cycles\n"
        "5. Use Coffin-Manson equation to determine cycles to failure"
    )
    
    fig.text(0.5, 0.02, process_text, ha='center', fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(os.getcwd(), 'stress_to_failure_cycles.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Graph saved to 'stress_to_failure_cycles.png'")

if __name__ == "__main__":
    create_stress_to_failure_graph() 