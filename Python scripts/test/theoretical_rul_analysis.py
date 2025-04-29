#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Theoretical RUL Analysis Script

This script calculates RUL (Remaining Useful Life) using the theoretical approach 
to determine initial RUL based on actual damage rates rather than using a fixed 
predefined value.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue

# Implementation of the theoretical estimate_fatigue_life
def theoretical_estimate_fatigue_life(fatigue_results, cycle_multiplier=1, force_shear=False):
    """Estimate fatigue life using the theoretical approach, where initial RUL is calculated
    from the actual damage rate, not a predetermined fixed value.
    
    Args:
        fatigue_results: Fatigue analysis results from analyze_fatigue
        cycle_multiplier: Multiplier for number of cycles (default: 1)
        force_shear: Flag to indicate if this is shear strain data
    
    Returns:
        tuple: (rul_values, cycles_experienced)
    """
    # Extract cycle data from fatigue_results
    cycles = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
    damages = fatigue_results.get('damages', np.array([]))
    
    # Check if we have valid cycle data
    if len(cycles) == 0 or len(N_f_cycles) == 0:
        print("No valid cycle data available for RUL estimation")
        return np.array([0]), np.array([0])
    
    # Apply cycle multiplier to counts (if not already applied)
    scaled_counts = counts * cycle_multiplier
    
    # Calculate damage metrics
    damage_per_cycle = scaled_counts / N_f_cycles
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Calculate average damage rate for initial slope
    total_cycles = np.sum(scaled_counts)
    avg_damage_rate = np.sum(damage_per_cycle) / total_cycles if total_cycles > 0 else 0
    
    # Calculate cycles experienced
    cycles_experienced = np.cumsum(scaled_counts)
    
    # Determine if this is shear strain data
    is_shear_strain = force_shear
    
    # Calculate initial RUL (before any cycles) using theoretical approach
    # Initial RUL represents the cycles to failure at the current damage rate
    if avg_damage_rate > 0:
        initial_rul = 1 / avg_damage_rate
    else:
        initial_rul = float('inf')
        # No cap - use true theoretical value
        print("Very low damage rate detected, theoretical initial RUL is effectively infinite")
    
    print(f"Theoretical initial RUL: {initial_rul:.1f} cycles" if initial_rul != float('inf') else "Theoretical initial RUL: infinite cycles")
    
    # Calculate RUL for each point using remaining damage capacity
    # The theoretical RUL is (1-D)/Ḋ, where D is cumulative damage and Ḋ is avg damage rate
    if avg_damage_rate > 0:
        rul_values = np.maximum((1 - cumulative_damage) / avg_damage_rate, 0)
    else:
        rul_values = np.ones_like(cumulative_damage) * initial_rul
    
    # Create a smooth RUL curve with interpolation
    if len(cycles_experienced) > 1:
        # Generate 100 points for a smooth curve
        interp_cycles = np.linspace(0, cycles_experienced[-1], 100)
        interp_rul = np.interp(interp_cycles, cycles_experienced, rul_values)
        
        # Add point at cycle 0 for initial RUL
        cycles_plot = np.insert(interp_cycles, 0, 0)
        rul_plot = np.insert(interp_rul, 0, initial_rul)
    else:
        # If only one cycle point, create a simple two-point curve
        cycles_plot = np.array([0, cycle_multiplier])
        rul_plot = np.array([initial_rul, initial_rul * (1 - avg_damage_rate * cycle_multiplier)])
    
    # Calculate and print RUL metrics
    final_percentage_used = (1 - (rul_plot[-1] / rul_plot[0])) * 100
    print(f"\nTheoretical RUL Analysis:")
    print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
    print(f"  Initial RUL: {rul_plot[0]:.1f} cycles")
    print(f"  Final RUL (at {cycles_plot[-1]:.1f} cycles): {rul_plot[-1]:.1f} cycles")
    print(f"  Life used: {final_percentage_used:.2f}%")
    print(f"  Damage rate: {avg_damage_rate:.8f} per cycle")
    
    return np.array(rul_plot), np.array(cycles_plot)

def load_strain_data():
    """Load and prepare strain data in the correct format for RUL analysis
    
    Returns:
        dict: Dictionary containing strain data and calculated principal strains
    """
    # Load raw data
    data = load_all_data()
    
    # Extract data arrays
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['strain_exx']
        strain_eyy = data['strain_eyy']
        time_points = data['time_points']
    else:
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, _ = data
    
    # Calculate principal strains
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Create dictionary with all data
    strain_data = {
        'thermal_strain': thermal_strain,
        'DICExx': strain_exx,
        'DICEyy': strain_eyy,
        'time': time_points,
        'major_principal_strain': major_principal_strain,
        'minor_principal_strain': minor_principal_strain,
        'max_shear_strain': max_shear_strain
    }
    
    return strain_data

def calculate_rul_for_sample_points(strain_data, cycle_multiplier=50, time_per_cycle=70.6):
    """Calculate RUL for a few representative spatial points using the theoretical approach
    
    Args:
        strain_data: Dictionary containing strain data arrays
        cycle_multiplier: Multiplier for projecting cycles into future (default: 50)
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    
    Returns:
        dict: Analysis results with RUL curves for different points
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Time points: {time_dim} points from {strain_data['time'][0]} to {strain_data['time'][-1]} seconds")
    
    print("\nCalculating principal strains for theoretical RUL analysis...")
    
    # We'll select a few representative points to analyze in detail
    # Choose points from different areas of the strain field
    sample_points = [
        (rows//4, cols//4),      # Upper left quadrant
        (rows//4, 3*cols//4),    # Upper right quadrant
        (3*rows//4, cols//4),    # Lower left quadrant
        (3*rows//4, 3*cols//4),  # Lower right quadrant
        (rows//2, cols//2)       # Center point
    ]
    
    # Store results for each point
    results = {}
    rul_curves = {}
    
    # Process each sample point
    for i, (r, c) in enumerate(sample_points):
        point_id = f"Point_{i+1}_at_{r}_{c}"
        results[point_id] = {"location": (r, c)}
        
        try:
            # Extract 1D strain signal at this spatial point
            signal = major_principal_strain[:, r, c]
            
            # Skip if all values are NaN
            if np.all(np.isnan(signal)):
                print(f"Skipping {point_id} - all NaN values")
                continue
            
            # Identify cycles using rainflow counting (explicitly specify NOT shear strain)
            cycles = identify_cycles(signal, is_shear_strain=False)
            
            if len(cycles) == 0:
                print(f"Skipping {point_id} - no cycles identified")
                continue
            
            # Analyze fatigue
            fatigue_results = analyze_fatigue(cycles)
            
            # Calculate RUL using theoretical approach
            rul_values, cycles_plot = theoretical_estimate_fatigue_life(
                fatigue_results, 
                cycle_multiplier=cycle_multiplier,
                force_shear=False
            )
            
            # Store RUL curve result
            if len(rul_values) > 0:
                initial_rul = rul_values[0]
                final_rul = rul_values[-1]
                damage_rate = (initial_rul - final_rul) / cycles_plot[-1] if cycles_plot[-1] > 0 else 0
                
                results[point_id].update({
                    "rul": rul_values,
                    "cycles": cycles_plot,
                    "initial_rul": initial_rul,
                    "final_rul": final_rul,
                    "damage_rate": damage_rate
                })
            
                # Store for RUL curve visualization
                key = f"Point {i+1}"
                rul_curves[key] = {
                    'rul': rul_values,
                    'cycles': cycles_plot,
                    'location': (r, c),
                    'initial_rul': initial_rul,
                    'damage_rate': damage_rate
                }
            
            print(f"Successfully analyzed {point_id} with {len(cycles)} cycles")
            
        except Exception as e:
            print(f"Error processing {point_id}: {e}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    return rul_curves

def create_rul_comparison(rul_curves, time_per_cycle=70.6):
    """Create comparison of RUL curves for different points using theoretical approach
    
    Args:
        rul_curves: Dictionary of RUL curves for different points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Define colors for different points
    point_colors = {
        'Point 1': '#d62728',    # Red
        'Point 2': '#2ca02c',    # Green
        'Point 3': '#1f77b4',    # Blue
        'Point 4': '#9467bd',    # Purple
        'Point 5': '#ff7f0e'     # Orange
    }
    
    # Calculate the max lifetime across all curves for proper scaling
    max_lifetime_cycles = 0
    max_lifetime_hours = 0
    
    # Process all curves to find maximum projected lifetime
    for key, curve_data in rul_curves.items():
        rul_values = curve_data['rul']
        cycles = curve_data['cycles']
        initial_rul = rul_values[0]
        
        # Get damage rate and total lifetime
        damage_rate = (rul_values[0] - rul_values[-1]) / (cycles[-1] - cycles[0]) if cycles[-1] > cycles[0] else 0
        if damage_rate > 0:
            # Calculate total cycles to failure
            total_cycles_to_failure = initial_rul / damage_rate
            max_lifetime_cycles = max(max_lifetime_cycles, total_cycles_to_failure)
            
            # Calculate total hours to failure
            total_hours_to_failure = (total_cycles_to_failure * time_per_cycle) / 3600
            max_lifetime_hours = max(max_lifetime_hours, total_hours_to_failure)
    
    # Add 10% margin to max lifetime for better visualization
    max_lifetime_cycles *= 1.1
    max_lifetime_hours *= 1.1
    
    # Plot RUL curves in cycles
    for key, curve_data in rul_curves.items():
        color = point_colors.get(key, '#000000')
        location = curve_data['location']
        label = f"{key} at {location}"
        
        rul_values = curve_data['rul']
        cycles = curve_data['cycles']
        
        # Get initial RUL
        initial_rul = rul_values[0]
        
        # Calculate damage rate for this curve
        damage_rate = (rul_values[0] - rul_values[-1]) / (cycles[-1] - cycles[0]) if cycles[-1] > cycles[0] else 0
        
        # Calculate total expected lifetime (time to reach RUL=0)
        if damage_rate > 0:
            total_cycles_to_failure = initial_rul / damage_rate
            
            # Create extended x-axis to show full lifetime
            extended_cycles = np.linspace(0, total_cycles_to_failure, 1000)
            
            # Create extended y-axis values that go to zero
            extended_rul = np.maximum(initial_rul - damage_rate * extended_cycles, 0)
            
            # Plot projected failure curve with improved visibility
            ax1.plot(extended_cycles, extended_rul, '--', color=color, linewidth=1.5, 
                     label=f"{key} - Projected to failure ({total_cycles_to_failure:.0f} cycles)")
        
        # Plot original data points
        ax1.plot(cycles, rul_values, '-', color=color, linewidth=2.0, marker='o', 
                 markersize=5, label=f"{key} at {location}")
        
        # For the first point, add annotation with initial RUL value
        if key == 'Point 1':
            ax1.annotate(f"Initial RUL: {initial_rul:.0f} cycles\nDamage rate: {damage_rate:.8f} per cycle",
                        xy=(cycles[0], rul_values[0]), xytext=(cycles[0]+cycles[-1]*0.1, rul_values[0]*0.9),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))
    
    # Set consistent x-axis limit to show full lifetime for all curves
    if max_lifetime_cycles > 0:
        ax1.set_xlim(0, max_lifetime_cycles)
        
        # Set y-axis to show full range from 0 to max initial RUL
        max_initial_rul = max([curve_data['initial_rul'] for curve_data in rul_curves.values()])
        ax1.set_ylim(0, max_initial_rul * 1.05)
    
    ax1.set_xlabel('Cycles Experienced', fontsize=14)
    ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=14)
    ax1.set_title('Theoretical RUL Comparison Across Different Locations', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Plot RUL curves in time
    if time_per_cycle > 0:
        for key, curve_data in rul_curves.items():
            color = point_colors.get(key, '#000000')
            location = curve_data['location']
            
            # Convert cycle data to hours
            rul_hours = np.array(curve_data['rul']) * time_per_cycle / 3600  # Convert to hours
            cycles_hours = np.array(curve_data['cycles']) * time_per_cycle / 3600  # Convert to hours
            
            # Get initial RUL in hours
            initial_rul_hours = rul_hours[0]
            
            # Calculate damage rate in hours
            damage_per_hour = (rul_hours[0] - rul_hours[-1]) / (cycles_hours[-1] - cycles_hours[0]) if cycles_hours[-1] > cycles_hours[0] else 0
            
            # Calculate total expected lifetime in hours
            if damage_per_hour > 0:
                total_hours_to_failure = initial_rul_hours / damage_per_hour
                
                # Create extended x-axis to show full lifetime
                extended_hours = np.linspace(0, total_hours_to_failure, 1000)
                
                # Create extended y-axis values
                extended_rul_hours = np.maximum(initial_rul_hours - damage_per_hour * extended_hours, 0)
                
                # Plot projected failure curve
                ax2.plot(extended_hours, extended_rul_hours, '--', color=color, linewidth=1.5,
                         label=f"{key} - Projected to failure ({total_hours_to_failure:.1f} hours)")
            
            # Plot original data
            ax2.plot(cycles_hours, rul_hours, '-', color=color, linewidth=2.0, 
                     marker='o', markersize=5, label=f"{key} at {location}")
        
        # Set expanded x-axis limit
        if max_lifetime_hours > 0:
            ax2.set_xlim(0, max_lifetime_hours)
            
            # Set y-axis to show full range
            max_initial_rul_hours = max([curve_data['initial_rul'] * time_per_cycle / 3600 for curve_data in rul_curves.values()])
            ax2.set_ylim(0, max_initial_rul_hours * 1.05)
        
        ax2.set_xlabel('Time Experienced (hours)', fontsize=14)
        ax2.set_ylabel('Remaining Useful Life (hours)', fontsize=14)
        ax2.set_title('Theoretical RUL Comparison in Real Time', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Make the entire figure look better
    plt.tight_layout()
    
    # Save the visualizations in both formats
    plt.savefig('theoretical_rul_comparison.svg', bbox_inches='tight', dpi=300)
    plt.savefig('theoretical_rul_comparison.png', bbox_inches='tight', dpi=300)
    print("Theoretical RUL curve comparison saved as 'theoretical_rul_comparison.svg/png'")

def main():
    """Run the theoretical RUL analysis"""
    print("\n========== THEORETICAL RUL ANALYSIS ==========\n")
    
    # Load strain data
    print("\nLoading strain data...")
    strain_data = load_strain_data()
    
    # Calculate RUL for sample points
    results = calculate_rul_for_sample_points(
        strain_data, 
        cycle_multiplier=50,
        time_per_cycle=70.6
    )
    
    # Create RUL comparison visualization
    create_rul_comparison(results, time_per_cycle=70.6)
    
    print("\nTheoretical RUL analysis complete!")

if __name__ == "__main__":
    main() 