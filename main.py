#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script to run RUL estimation from strain DIC measurement

@author: Jayron Sandhu
"""

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import os

# Import modules from our project
from data_loader import load_all_data, print_statistical_summary
from plotter import (plot_initial_strain_analysis, plot_stress_analysis, 
                    plot_strain_analysis, plot_fatigue_analysis_signals,
                    plot_rul_estimation)
from fatigue_analysis import (identify_cycles, analyze_fatigue,
                             estimate_fatigue_life)
from strain_calculator import (calculate_principal_strains, calculate_stresses)

def main():
    """Main function to run the strain analysis and RUL estimation workflow"""
    # -------------------
    # 1. Load strain data
    # -------------------
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Check if data is dictionary or tuple (for backward compatibility)
    if isinstance(data, dict):
        # Use dictionary format
        ThermalStrain = data['thermal_strain']
        DICExx = data['strain_exx']
        DICEyy = data['strain_eyy']
        time_points = data['time_points']
        high_strain_points = data['high_strain_points']
        mean_strain = data['mean_strain']
        max_strain = data['max_strain'] 
        std_strain = data['std_strain']
    else:
        # Use tuple format for backward compatibility
        # data_loader.py returns: (AverageExx, DICExx, AverageEyy, DICEyy, ThermalStrain, 
        #                          time_points, mean_strain, std_strain, max_strain, high_strain_points)
        _, DICExx, _, DICEyy, ThermalStrain, time_points, mean_strain, std_strain, max_strain, high_strain_points = data
    
    # Print statistical summary
    print_statistical_summary(ThermalStrain, high_strain_points)
    
    # ---------------------------
    # 2. Plot initial strain data
    # ---------------------------
    print("\nPlotting initial strain analysis...")
    top_5_indices = plot_initial_strain_analysis(ThermalStrain, DICExx, DICEyy, 
                                              time_points, high_strain_points, 
                                              max_strain)
    
    # -----------------------
    # 3. Calculate stress data
    # -----------------------
    print("\nCalculating stress based on strain data...")
    # Material properties for tungsten
    E = 400e9  # Young's modulus (Pa)
    nu = 0.28  # Poisson's ratio
    
    stress_xx, stress_yy, stress_von_mises = calculate_stresses(DICExx, DICEyy, E, nu, ThermalStrain)
    
    # Calculate yield strength and safety factor
    yield_strength = 1000e6  # Approximate yield strength for tungsten (~1000 MPa)
    max_vm_stress = np.nanmax(stress_von_mises)
    safety_factor = yield_strength / max_vm_stress
    
    print(f"Maximum σxx: {np.nanmax(stress_xx)/1e6:.2f} MPa")
    print(f"Maximum σyy: {np.nanmax(stress_yy)/1e6:.2f} MPa")
    print(f"Maximum von Mises stress: {max_vm_stress/1e6:.2f} MPa")
    print(f"Yield strength of tungsten: ~{yield_strength/1e6} MPa")
    print(f"Safety factor: {safety_factor:.2f}")
    
    # -----------------------
    # 4. Plot stress analysis
    # -----------------------
    print("\nPlotting stress analysis...")
    plot_stress_analysis(time_points, stress_xx, stress_yy, stress_von_mises,
                         high_strain_points, top_5_indices)
    
    # ---------------------------------
    # 5. Calculate principal strains
    # ---------------------------------
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Find locations of extreme strain values
    max_principal_loc = np.unravel_index(np.nanargmax(major_principal_strain), major_principal_strain.shape)[1:]
    min_principal_loc = np.unravel_index(np.nanargmin(minor_principal_strain), minor_principal_strain.shape)[1:]
    max_shear_loc = np.unravel_index(np.nanargmax(max_shear_strain), max_shear_strain.shape)[1:]
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    print(f"Minimum principal strain location: {min_principal_loc}")
    print(f"Maximum shear strain location: {max_shear_loc}")
    
    # -----------------------
    # 6. Plot strain analysis
    # -----------------------
    print("\nPlotting principal strain analysis...")
    plot_strain_analysis(time_points, major_principal_strain, minor_principal_strain, max_shear_strain,
                        max_principal_loc, min_principal_loc, max_shear_loc)
    
    # ---------------------------------
    # 7. Perform fatigue analysis
    # ---------------------------------
    print("\nPerforming fatigue analysis...")
    
    # Initialize fatigue analysis visualization
    fig_fatigue, _ = plot_fatigue_analysis_signals(time_points,
                                                major_principal_strain, max_shear_strain, 
                                                max_principal_loc, max_shear_loc)
    
    # Identify cycles from strain data
    print("\nIdentifying cycles in strain data...")
    cycles_max_strain = identify_cycles(major_principal_strain[:, max_principal_loc[0], max_principal_loc[1]])
    cycles_max_shear = identify_cycles(max_shear_strain[:, max_shear_loc[0], max_shear_loc[1]], is_shear_strain=True)
    
    print(f"Number of cycles identified for principal strain: {len(cycles_max_strain)}")
    print(f"Number of cycles identified for shear strain: {len(cycles_max_shear)}")
    
    # Analyze fatigue for these cycles
    print("\nAnalyzing fatigue for identified cycles...")
    fatigue_max_strain = analyze_fatigue(cycles_max_strain)
    fatigue_max_shear = analyze_fatigue(cycles_max_shear)
    
    # Estimate fatigue life
    print("\nEstimating fatigue life...")
    cycle_multiplier = 10  # Use a multiplier of 10 as requested
    
    # Calculate RUL for max principal strain
    rul_max_strain, cycles_max_strain_plot = estimate_fatigue_life(
        fatigue_max_strain, cycle_multiplier
    )
    
    # Calculate RUL for max shear strain
    rul_max_shear, cycles_max_shear_plot = estimate_fatigue_life(
        fatigue_max_shear, cycle_multiplier, force_shear=True
    )
    
    # Plot RUL estimation
    print("\nPlotting RUL estimation...")
    plot_rul_estimation(
        rul_max_strain, cycles_max_strain_plot,
        rul_max_shear, cycles_max_shear_plot,
        max_principal_loc, max_shear_loc,
        time_per_cycle=70.6  # Time per cycle in seconds (updated from 0.2 to 70.6)
    )
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main()