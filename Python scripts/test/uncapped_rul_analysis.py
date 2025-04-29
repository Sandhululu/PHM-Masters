#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncapped Theoretical RUL Analysis Script

This script calculates RUL (Remaining Useful Life) using a fully theoretical approach 
with NO capping at any level - showing the true theoretical values based on actual 
damage rates, without artificial limits.
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
from fatigue_analysis import identify_cycles  # Import original function

# Create uncapped versions of fatigue analysis functions to see true theoretical values
def identify_cycles_uncapped(signal, is_shear_strain=False):
    """Identify cycles in strain signal using rainflow counting
    
    Args:
        signal: 1D strain signal array
        is_shear_strain: Flag to identify if this is shear strain data
    
    Returns:
        tuple: (ranges, means, counts, cycle_indices)
    """
    try:
        import numpy as np
        from rainflow import count_cycles
    except ImportError:
        print("Error: rainflow package not found. Install using: pip install rainflow")
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Clean signal by removing NaN values
    clean_signal = np.array(signal)
    nan_mask = np.isnan(clean_signal)
    
    if np.all(nan_mask):
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Handle missing data by interpolation if needed
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        all_indices = np.arange(len(clean_signal))
        clean_signal = np.interp(all_indices, valid_indices, clean_signal[valid_indices])
    
    # Special handling for shear strain data
    if is_shear_strain:
        # Adjust shear strain signal for better cycle detection
        # Typically, this might involve scaling or filtering
        pass
    
    # Extract cycles using rainflow algorithm
    # This will return tuples of (range, mean, count, cycle_indices)
    extracted_cycles = []
    
    try:
        # Use rainflow algorithm with standard parameters for fatigue analysis
        for rng, mean, count, i_start, i_end in count_cycles(clean_signal, nbins=100):
            # Store cycle information
            extracted_cycles.append((float(rng), float(mean), float(count), (i_start, i_end)))
    except Exception as e:
        print(f"Error in rainflow cycle counting: {e}")
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Extract and format cycle data from the results
    if extracted_cycles:
        ranges, means, counts, cycle_indices = zip(*extracted_cycles)
        return (np.array(ranges), np.array(means), np.array(counts), np.array(cycle_indices))
    else:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

def analyze_fatigue_uncapped(cycles_data):
    """Analyze identified fatigue cycles to estimate damage - NO CAPPING
    
    Args:
        cycles_data: Tuple of (ranges, means, counts, cycle_indices) from identify_cycles
    
    Returns:
        dict: Fatigue analysis results
    """
    # Extract cycle data - Check format returned by fatigue_analysis.identify_cycles
    if isinstance(cycles_data, tuple) and len(cycles_data) >= 3:
        ranges, means, counts = cycles_data[:3]
    elif isinstance(cycles_data, list) and len(cycles_data) > 0:
        # Original function returns a list of cycle tuples
        ranges = np.array([c[0] for c in cycles_data])
        means = np.array([c[1] for c in cycles_data])
        counts = np.array([c[2] for c in cycles_data])
    else:
        print("Invalid cycle data format")
        return {'cycles': np.array([]), 'counts': np.array([]), 'N_f_cycles': np.array([]), 
                'damages': np.array([]), 'cumulative_damage': np.array([]), 'cycles_array': np.array([])}
    
    # Check if we have valid cycle data
    if len(ranges) == 0:
        return {'cycles': np.array([]), 'counts': np.array([]), 'N_f_cycles': np.array([]), 
                'damages': np.array([]), 'cumulative_damage': np.array([]), 'cycles_array': np.array([])}
    
    # Sort cycles by range values
    sorted_indices = np.argsort(ranges)
    sorted_ranges = ranges[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Set material parameters for fatigue calculations (tungsten)
    E_mod = 400000  # Young's modulus (MPa)
    sigma_f_prime = 1000  # Fatigue strength coefficient (MPa)
    epsilon_f_prime = 0.25  # Fatigue ductility coefficient
    b = -0.12  # Fatigue strength exponent
    c = -0.6  # Fatigue ductility exponent
    safety_factor = 1.0  # Safety factor for cycle life calculation
    
    # Calculate cycles to failure for each strain range - NO CAPPING
    N_f_cycles = []
    for strain_range in sorted_ranges:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 15, 1000)  # Extended range for higher values
        
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life with safety factor - NO CAPPING
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))] / safety_factor
        N_f_cycles.append(N_f)  # No cap! Use true theoretical value
    
    # Calculate damage per cycle using Miner's rule
    damage_per_cycle = [count/N_f for count, N_f in zip(sorted_counts, N_f_cycles)]
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Return comprehensive fatigue analysis results
    return {
        'cycles': sorted_ranges,
        'counts': sorted_counts,
        'N_f_cycles': np.array(N_f_cycles),
        'damages': np.array(damage_per_cycle),
        'cumulative_damage': cumulative_damage,
        'cycles_array': np.arange(len(sorted_ranges))
    }

def uncapped_estimate_fatigue_life(fatigue_results, cycle_multiplier=1, force_shear=False):
    """Estimate fatigue life using fully theoretical approach with NO CAPPING whatsoever
    
    Args:
        fatigue_results: Fatigue analysis results from analyze_fatigue_uncapped
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
    
    # Calculate initial RUL (before any cycles) using pure theoretical approach - NO CAPPING
    # Initial RUL represents the cycles to failure at the current damage rate
    if avg_damage_rate > 0:
        initial_rul = 1 / avg_damage_rate
    else:
        initial_rul = float('inf')
        print("Very low damage rate detected, theoretical initial RUL is effectively infinite")
    
    if initial_rul != float('inf'):
        print(f"Uncapped theoretical initial RUL: {initial_rul:.1f} cycles")
    else:
        print("Uncapped theoretical initial RUL: infinite cycles")
    
    # Calculate RUL for each point using remaining damage capacity - NO CAPPING
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
        if avg_damage_rate > 0:
            rul_plot = np.array([initial_rul, initial_rul * (1 - avg_damage_rate * cycle_multiplier)])
        else:
            rul_plot = np.array([initial_rul, initial_rul])
    
    # Calculate and print RUL metrics
    if rul_plot[0] != float('inf') and rul_plot[-1] != float('inf'):
        final_percentage_used = (1 - (rul_plot[-1] / rul_plot[0])) * 100
        print(f"\nUncapped Theoretical RUL Analysis:")
        print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
        print(f"  Initial RUL: {rul_plot[0]:.1f} cycles")
        print(f"  Final RUL (at {cycles_plot[-1]:.1f} cycles): {rul_plot[-1]:.1f} cycles")
        print(f"  Life used: {final_percentage_used:.2f}%")
        print(f"  Damage rate: {avg_damage_rate:.8f} per cycle")
    else:
        print(f"\nUncapped Theoretical RUL Analysis:")
        print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
        print(f"  Initial RUL: infinite cycles")
        print(f"  Final RUL (at {cycles_plot[-1]:.1f} cycles): infinite cycles")
        print(f"  Life used: 0.00%")
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
    """Calculate uncapped RUL for representative spatial points
    
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
    
    print("\nCalculating principal strains for uncapped theoretical RUL analysis...")
    
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
            
            # Identify cycles using the original identify_cycles function
            cycles = identify_cycles(signal, is_shear_strain=False)
            
            # Better way to check if cycles is empty
            if cycles is None or len(cycles) == 0:
                print(f"Skipping {point_id} - no cycles identified")
                continue
            
            # Analyze fatigue with uncapped approach
            fatigue_results = analyze_fatigue_uncapped(cycles)
            
            # Calculate RUL using uncapped theoretical approach
            rul_values, cycles_plot = uncapped_estimate_fatigue_life(
                fatigue_results, 
                cycle_multiplier=cycle_multiplier,
                force_shear=False
            )
            
            # Store RUL curve result
            if len(rul_values) > 0:
                # Handle infinite values for display
                if rul_values[0] == float('inf'):
                    initial_rul = float('inf')
                    final_rul = float('inf')
                    damage_rate = 0
                else:
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
            
            print(f"Successfully analyzed {point_id} with {len(cycles[0])} cycles")
            
        except Exception as e:
            print(f"Error processing {point_id}: {e}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    return rul_curves

def format_large_number(number):
    """Format large numbers in scientific notation for better readability"""
    if number == float('inf'):
        return "Infinite"
    elif number > 1e9:
        return f"{number:.2e} cycles"
    elif number > 1e6:
        return f"{number/1e6:.2f} million cycles"
    else:
        return f"{number:.1f} cycles"

def create_rul_comparison(rul_curves, time_per_cycle=70.6):
    """Create comparison of uncapped RUL curves for different points
    
    Args:
        rul_curves: Dictionary of RUL curves for different points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Define colors for different points
    point_colors = {
        'Point 1': '#d62728',    # Red
        'Point 2': '#2ca02c',    # Green
        'Point 3': '#1f77b4',    # Blue
        'Point 4': '#9467bd',    # Purple
        'Point 5': '#ff7f0e'     # Orange
    }
    
    # Calculate statistics to display
    has_infinite = any(curve_data.get('initial_rul', 0) == float('inf') for curve_data in rul_curves.values())
    finite_initial_ruls = [curve_data.get('initial_rul', 0) for curve_data in rul_curves.values() 
                         if curve_data.get('initial_rul', 0) != float('inf')]
    
    # Gather information for plot title and annotation
    min_rul = min(finite_initial_ruls) if finite_initial_ruls else "N/A"
    max_rul = max(finite_initial_ruls) if finite_initial_ruls else "N/A"
    
    # Plot RUL curves in cycles
    for key, curve_data in rul_curves.items():
        color = point_colors.get(key, '#000000')
        location = curve_data['location']
        initial_rul = curve_data.get('initial_rul', 0)
        damage_rate = curve_data.get('damage_rate', 0)
        
        is_infinite = initial_rul == float('inf')
        label_suffix = " (Infinite RUL)" if is_infinite else f" ({format_large_number(initial_rul)})"
        
        rul_values = curve_data['rul']
        cycles = curve_data['cycles']
        
        if not is_infinite:
            # Plot projection line for finite values
            # Calculate total expected lifetime (time to reach RUL=0)
            if damage_rate > 0:
                total_cycles_to_failure = initial_rul / damage_rate if damage_rate > 0 else float('inf')
                
                # Create extended x-axis to show full lifetime (up to reasonable limit for visualization)
                if total_cycles_to_failure < 1e9:  # Cap display at billion cycles
                    extended_cycles = np.linspace(0, total_cycles_to_failure, 1000)
                    
                    # Create extended y-axis values that go to zero
                    extended_rul = np.maximum(initial_rul - damage_rate * extended_cycles, 0)
                    
                    # Plot projected failure curve with improved visibility
                    ax1.plot(extended_cycles, extended_rul, '--', color=color, linewidth=1.5, 
                            label=f"{key} - Projected to failure ({format_large_number(total_cycles_to_failure)})")
        
        # Plot original data points (using log scale for infinite values)
        ax1.plot(cycles, rul_values, '-', color=color, linewidth=2.0, marker='o', 
                markersize=5, label=f"{key} at {location}{label_suffix}")
    
    # Set appropriate scales based on data
    if has_infinite:
        ax1.set_title('Uncapped Theoretical RUL Comparison (Includes Infinite Values)', fontsize=14)
    else:
        if finite_initial_ruls:
            max_initial_rul = max(finite_initial_ruls)
            ax1.set_ylim(0, max_initial_rul * 1.05)
            
        ax1.set_title('Uncapped Theoretical RUL Comparison', fontsize=14)
    
    # Find reasonable x-axis limit
    max_cycles_to_show = 0
    for curve_data in rul_curves.values():
        initial_rul = curve_data.get('initial_rul', 0)
        damage_rate = curve_data.get('damage_rate', 0)
        
        if initial_rul != float('inf') and damage_rate > 0:
            total_lifetime = initial_rul / damage_rate
            if total_lifetime < 1e9:  # Cap display at billion cycles 
                max_cycles_to_show = max(max_cycles_to_show, total_lifetime * 1.1)
    
    if max_cycles_to_show > 0:
        ax1.set_xlim(0, max_cycles_to_show)
    
    ax1.set_xlabel('Cycles Experienced', fontsize=14)
    ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add annotation with statistics
    stats_text = f"Statistics:\n"
    if finite_initial_ruls:
        stats_text += f"Minimum RUL: {format_large_number(min_rul)}\n"
        stats_text += f"Maximum RUL: {format_large_number(max_rul)}\n"
    if has_infinite:
        stats_text += "Some points have infinite theoretical RUL"
    
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             fontsize=10, verticalalignment='bottom')
    
    # Plot RUL curves in time (hours) for second subplot
    if time_per_cycle > 0:
        for key, curve_data in rul_curves.items():
            color = point_colors.get(key, '#000000')
            location = curve_data['location']
            initial_rul = curve_data.get('initial_rul', 0)
            damage_rate = curve_data.get('damage_rate', 0)
            
            is_infinite = initial_rul == float('inf')
            
            # Convert cycle data to hours
            if not is_infinite:
                rul_hours = np.array(curve_data['rul']) * time_per_cycle / 3600  # Convert to hours
                cycles_hours = np.array(curve_data['cycles']) * time_per_cycle / 3600  # Convert to hours
                
                # Plot original data
                ax2.plot(cycles_hours, rul_hours, '-', color=color, linewidth=2.0, 
                        marker='o', markersize=5, label=f"{key} at {location}")
                
                # Calculate damage rate in hours
                damage_per_hour = (rul_hours[0] - rul_hours[-1]) / (cycles_hours[-1] - cycles_hours[0]) if cycles_hours[-1] > cycles_hours[0] else 0
                
                # Calculate total expected lifetime in hours
                if damage_per_hour > 0:
                    total_hours_to_failure = initial_rul * time_per_cycle / 3600 / damage_rate if damage_rate > 0 else float('inf')
                    
                    if total_hours_to_failure < 1e8:  # Reasonable cap for display
                        # Create extended x-axis to show full lifetime
                        extended_hours = np.linspace(0, total_hours_to_failure, 1000)
                        
                        # Create extended y-axis values
                        extended_rul_hours = np.maximum(rul_hours[0] - damage_per_hour * extended_hours, 0)
                        
                        # Plot projected failure curve
                        ax2.plot(extended_hours, extended_rul_hours, '--', color=color, linewidth=1.5,
                                label=f"{key} - Projected failure ({total_hours_to_failure:.1f} hours)")
        
        # Calculate reasonable axis limits for hours
        max_hours_to_show = 0
        max_rul_hours = 0
        
        for curve_data in rul_curves.values():
            initial_rul = curve_data.get('initial_rul', 0)
            damage_rate = curve_data.get('damage_rate', 0)
            
            if initial_rul != float('inf') and damage_rate > 0:
                total_lifetime_hours = initial_rul * time_per_cycle / 3600 / damage_rate
                if total_lifetime_hours < 1e8:  # Reasonable cap
                    max_hours_to_show = max(max_hours_to_show, total_lifetime_hours * 1.1)
                
                initial_rul_hours = initial_rul * time_per_cycle / 3600
                max_rul_hours = max(max_rul_hours, initial_rul_hours)
        
        if max_hours_to_show > 0:
            ax2.set_xlim(0, max_hours_to_show)
        
        if max_rul_hours > 0:
            ax2.set_ylim(0, max_rul_hours * 1.05)
        
        ax2.set_xlabel('Time Experienced (hours)', fontsize=14)
        ax2.set_ylabel('Remaining Useful Life (hours)', fontsize=14)
        ax2.set_title('Uncapped Theoretical RUL Comparison in Real Time', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Make the entire figure look better
    plt.tight_layout()
    
    # Save the visualizations in both formats
    plt.savefig('uncapped_rul_comparison.svg', bbox_inches='tight', dpi=300)
    plt.savefig('uncapped_rul_comparison.png', bbox_inches='tight', dpi=300)
    print("Uncapped theoretical RUL curve comparison saved as 'uncapped_rul_comparison.svg/png'")

def main():
    """Run the uncapped theoretical RUL analysis"""
    print("\n========== UNCAPPED THEORETICAL RUL ANALYSIS ==========\n")
    
    # Load strain data
    print("\nLoading strain data...")
    strain_data = load_strain_data()
    
    # Calculate RUL for sample points with uncapped approach
    results = calculate_rul_for_sample_points(
        strain_data, 
        cycle_multiplier=50,
        time_per_cycle=70.6
    )
    
    # Create RUL comparison visualization
    create_rul_comparison(results, time_per_cycle=70.6)
    
    print("\nUncapped theoretical RUL analysis complete!")

if __name__ == "__main__":
    main() 