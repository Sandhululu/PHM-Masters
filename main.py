#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for RUL estimation from strain DIC measurement

This script orchestrates the full analysis process:
1. Data loading and preparation
2. Strain and stress calculations
3. Fatigue analysis
4. RUL estimation
5. Visualization

@author: Jayron Sandhu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import load_all_data, print_statistical_summary
from strain_calculator import (calculate_principal_strains, calculate_stresses,
                              find_extreme_locations, print_analysis_summary)
from fatigue_analysis import analyze_fatigue, estimate_rul
from plotter import (plot_initial_strain_analysis, plot_stress_analysis, 
                    plot_strain_analysis, plot_fatigue_analysis_signals,
                    plot_rul_estimation)

def run_analysis(cycle_multiplier=10, material='tungsten'):
    """
    Run the full analysis pipeline for RUL estimation
    
    Args:
        cycle_multiplier: Multiplier for number of cycles (default: 10)
        material: Material type for fatigue properties (default: 'tungsten')
    """
    print("=" * 80)
    print("Starting RUL estimation from strain DIC measurement")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Check data type and handle appropriately
    print("DEBUG - In main.py - data type:", type(data))
    
    if isinstance(data, dict):
        # Handle dictionary return
        ThermalStrain = data['thermal_strain']
        DICExx = data['exx']
        DICEyy = data['eyy']
        time_points = data['time_points']
        high_strain_points = data['high_strain_points']
        max_strain = data['max_strain']
    elif isinstance(data, tuple):
        # Handle tuple return (original implementation)
        print("DEBUG - In main.py - data is a tuple of length:", len(data))
        AverageExx, DICExx, AverageEyy, DICEyy, ThermalStrain, time_points, mean_strain, std_strain, max_strain, high_strain_points = data
    else:
        print("ERROR: Unexpected data type from load_all_data:", type(data))
        return
    
    # Print statistical summary
    print_statistical_summary(ThermalStrain, high_strain_points)
    
    # Step 2: Plot initial strain analysis
    print("\nCreating initial strain analysis plots...")
    top_5_indices = plot_initial_strain_analysis(ThermalStrain, DICExx, DICEyy, 
                                               time_points, high_strain_points, max_strain)
    
    # Step 3: Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Step 4: Find locations of extreme strain values
    print("\nFinding locations of extreme strain values...")
    max_principal_loc, min_principal_loc, max_shear_loc = find_extreme_locations(
        major_principal_strain, minor_principal_strain, max_shear_strain)
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    print(f"Minimum principal strain location: {min_principal_loc}")
    print(f"Maximum shear strain location: {max_shear_loc}")
    
    # Step 5: Calculate stresses
    print("\nCalculating stresses...")
    # Material properties for tungsten
    E = 400e9  # Young's modulus (Pa)
    poisson = 0.28  # Poisson's ratio
    
    stress_xx, stress_yy, stress_von_mises = calculate_stresses(
        DICExx, DICEyy, E, poisson, ThermalStrain)
    
    # Print analysis summary
    print_analysis_summary(stress_xx, stress_yy, stress_von_mises,
                          major_principal_strain, minor_principal_strain, max_shear_strain,
                          max_principal_loc, min_principal_loc, max_shear_loc)
    
    # Step 6: Plot stress analysis
    print("\nCreating stress analysis plots...")
    plot_stress_analysis(time_points, stress_xx, stress_yy, stress_von_mises,
                        high_strain_points, top_5_indices)
    
    # Step 7: Plot strain analysis
    print("\nCreating strain analysis plots...")
    plot_strain_analysis(time_points, major_principal_strain, minor_principal_strain, max_shear_strain,
                        max_principal_loc, min_principal_loc, max_shear_loc)
    
    # Step 8: Fatigue Analysis
    print("\nPerforming fatigue analysis...")
    fatigue_results = analyze_fatigue(time_points, major_principal_strain, max_shear_strain,
                                     max_principal_loc, max_shear_loc)
    
    # Step 9: Plot fatigue signals
    print("\nCreating fatigue analysis plots...")
    fig_fatigue, axes_fatigue = plot_fatigue_analysis_signals(
        time_points, major_principal_strain, max_shear_strain,
        max_principal_loc, max_shear_loc)
    
    # Add legends to fatigue plots with proper positioning
    axes_fatigue[0].legend(frameon=True, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    axes_fatigue[1].legend(frameon=True, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add text about fatigue analysis
    ps_cycles = fatigue_results['principal_strain']['cycles_array']
    if ps_cycles is not None and len(ps_cycles) > 0:
        safety_factor = estimate_fatigue_life(ps_cycles[0, 2]) / (len(ps_cycles) * cycle_multiplier)
        row, col = max_principal_loc
        axes_fatigue[0].text(0.97, 0.97, 
                           f"Cycles detected: {len(ps_cycles)}\nCycle multiplier: {cycle_multiplier}\n"
                           f"Est. life: {len(ps_cycles) * cycle_multiplier * safety_factor:.0f} cycles\n"
                           f"Safety factor: {safety_factor:.1f}",
                           transform=axes_fatigue[0].transAxes, fontsize=11,
                           horizontalalignment='right', verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
    
    ss_cycles = fatigue_results['shear_strain']['cycles_array']
    if ss_cycles is not None and len(ss_cycles) > 0:
        safety_factor = estimate_fatigue_life(ss_cycles[0, 2]) / (len(ss_cycles) * cycle_multiplier)
        row, col = max_shear_loc
        axes_fatigue[1].text(0.97, 0.97, 
                           f"Cycles detected: {len(ss_cycles)}\nCycle multiplier: {cycle_multiplier}\n"
                           f"Est. life: {len(ss_cycles) * cycle_multiplier * safety_factor:.0f} cycles\n"
                           f"Safety factor: {safety_factor:.1f}",
                           transform=axes_fatigue[1].transAxes, fontsize=11,
                           horizontalalignment='right', verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
    
    fig_fatigue.savefig(os.path.join(os.getcwd(), 'fatigue_analysis.svg'), bbox_inches='tight', dpi=300)
    plt.show()
    
    # Step 10: RUL Estimation
    print("\nPerforming RUL estimation...")
    
    # Estimate RUL for maximum principal strain location
    rul_max_strain, cycles_max_strain_plot = estimate_rul(
        fatigue_results['principal_strain']['cycles_array'], cycle_multiplier)
    
    # Estimate RUL for maximum shear strain location
    rul_max_shear, cycles_max_shear_plot = estimate_rul(
        fatigue_results['shear_strain']['cycles_array'], cycle_multiplier)
    
    # Step 11: Plot RUL estimation
    print("\nCreating RUL estimation plots...")
    plot_rul_estimation(rul_max_strain, cycles_max_strain_plot,
                      rul_max_shear, cycles_max_shear_plot,
                      max_principal_loc, max_shear_loc)
    
    # Print analysis results
    print("\nRUL Analysis Summary:")
    print("-" * 50)
    
    if rul_max_strain is not None:
        initial_rul_ps = rul_max_strain[0]
        final_rul_ps = rul_max_strain[-1]
        total_cycles_ps = len(cycles_max_strain_plot) * cycle_multiplier
        life_used_ps = (1 - (final_rul_ps / initial_rul_ps)) * 100
        
        print("For Maximum Principal Strain Location:")
        print(f"- Total cycles simulated: {total_cycles_ps}")
        print(f"- Initial RUL: {initial_rul_ps:.1f} cycles")
        print(f"- Final RUL: {final_rul_ps:.1f} cycles")
        print(f"- Life used: {life_used_ps:.2f}%")
    
    if rul_max_shear is not None:
        initial_rul_ss = rul_max_shear[0]
        final_rul_ss = rul_max_shear[-1]
        total_cycles_ss = len(cycles_max_shear_plot) * cycle_multiplier
        life_used_ss = (1 - (final_rul_ss / initial_rul_ss)) * 100
        
        print("\nFor Maximum Shear Strain Location:")
        print(f"- Total cycles simulated: {total_cycles_ss}")
        print(f"- Initial RUL: {initial_rul_ss:.1f} cycles")
        print(f"- Final RUL: {final_rul_ss:.1f} cycles")
        print(f"- Life used: {life_used_ss:.2f}%")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Run the analysis with the specified cycle multiplier
    run_analysis(cycle_multiplier=10) 