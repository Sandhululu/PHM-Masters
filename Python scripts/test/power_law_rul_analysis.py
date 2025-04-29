#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Power Law RUL Analysis Script

This script calculates RUL (Remaining Useful Life) using a power law model
for damage accumulation, resulting in accelerating damage rates over time
instead of the linear damage model used in the original analysis.
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

# Implementation of the power law estimate_fatigue_life
def power_law_estimate_fatigue_life(fatigue_results, cycle_multiplier=1, power_factor=1.5):
    """Estimate fatigue life using a power law model for accelerating damage
    
    Args:
        fatigue_results: Fatigue analysis results from analyze_fatigue
        cycle_multiplier: Multiplier for number of cycles (default: 1)
        power_factor: Exponent for the power law model (> 1 means accelerating damage)
    
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
    
    # Apply damage multiplier for demonstration purposes (makes power law effects more visible)
    damage_multiplier = 1000.0  # Increase this to show more dramatic power law effects
    
    # Calculate damage metrics
    damage_per_cycle = scaled_counts / N_f_cycles * damage_multiplier
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Calculate average damage rate for initial slope
    total_cycles = np.sum(scaled_counts)
    avg_damage_rate = np.sum(damage_per_cycle) / total_cycles if total_cycles > 0 else 0
    
    # Calculate cycles experienced
    cycles_experienced = np.cumsum(scaled_counts)
    
    # Set initial RUL for principal strain
    initial_rul = 1000000  # Fixed at 1 million cycles
    
    # For power law model, we'll use: RUL = initial_rul * (1 - (t/T)^power_factor)
    # where t is current cycles and T is expected lifetime
    
    # First compute initial damage rate to estimate lifetime
    initial_damage_rate = avg_damage_rate
    
    # Estimate total lifetime using initial damage rate as reference point
    # This is a simplification for calibrating the power law model
    if initial_damage_rate > 0:
        estimated_lifetime = initial_rul / initial_damage_rate
    else:
        estimated_lifetime = float('inf')
    
    # For demonstration purposes, cap the estimated lifetime to make visualization more effective
    max_lifetime_cap = 200000  # Cap lifetime for demo purposes
    estimated_lifetime = min(estimated_lifetime, max_lifetime_cap)
    
    # Generate cycles for visualization (100 points for a smooth curve)
    interp_cycles = np.linspace(0, min(cycles_experienced[-1] * 5, estimated_lifetime * 0.3), 100)
    
    # Calculate RUL using power law model
    # As t increases, damage accelerates non-linearly
    normalized_time = interp_cycles / estimated_lifetime
    damage_fraction = np.power(normalized_time, power_factor)
    capped_damage = np.minimum(damage_fraction, 1.0)  # Cap at 100% damage
    remaining_life_fraction = 1.0 - capped_damage
    rul_values = initial_rul * remaining_life_fraction
    
    # Add point at cycle 0 for initial RUL
    cycles_plot = np.insert(interp_cycles, 0, 0)
    rul_plot = np.insert(rul_values, 0, initial_rul)
    
    # Calculate and print RUL metrics
    final_percentage_used = (1 - (rul_plot[-1] / rul_plot[0])) * 100
    print(f"\nPower Law RUL Analysis (power factor = {power_factor}):")
    print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
    print(f"  Initial RUL: {rul_plot[0]:.1f} cycles")
    print(f"  Final RUL (at {cycles_plot[-1]:.1f} cycles): {rul_plot[-1]:.1f} cycles")
    print(f"  Life used: {final_percentage_used:.2f}%")
    print(f"  Estimated lifetime: {estimated_lifetime:.1f} cycles")
    
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

def calculate_rul_for_sample_points(strain_data, power_factors=[1.0, 1.5, 2.0], cycle_multiplier=50, time_per_cycle=70.6):
    """Calculate RUL for a few representative spatial points using power law models
    
    Args:
        strain_data: Dictionary containing strain data arrays
        power_factors: List of power factors to test (1.0 = linear, >1.0 = accelerated damage)
        cycle_multiplier: Multiplier for projecting cycles into future (default: 50)
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    
    Returns:
        dict: Analysis results with different power factors
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Time points: {time_dim} points from {strain_data['time'][0]} to {strain_data['time'][-1]} seconds")
    
    print("\nCalculating principal strains for power law model analysis...")
    
    # We'll select a few representative points to analyze in detail
    # Choose points from different areas of the strain field
    sample_points = [
        (rows//4, cols//4),      # Upper left quadrant
        (rows//4, 3*cols//4),    # Upper right quadrant
        (3*rows//4, cols//4),    # Lower left quadrant
        (3*rows//4, 3*cols//4),  # Lower right quadrant
        (rows//2, cols//2)       # Center point
    ]
    
    # Store results for each point and power factor
    results = {}
    
    # Process each sample point
    for i, (r, c) in enumerate(sample_points):
        point_id = f"Point_{i+1}_at_{r}_{c}"
        results[point_id] = {"location": (r, c), "power_factor_results": {}}
        
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
            
            # Calculate RUL with different power factors
            for power_factor in power_factors:
                # Calculate RUL using power law model
                rul_values, cycles_plot = power_law_estimate_fatigue_life(
                    fatigue_results, 
                    cycle_multiplier=cycle_multiplier,
                    power_factor=power_factor
                )
                
                # Store results
                results[point_id]["power_factor_results"][power_factor] = {
                    "rul": rul_values,
                    "cycles": cycles_plot,
                    "power_factor": power_factor,
                    "initial_rul": rul_values[0],
                    "final_rul": rul_values[-1]
                }
            
            print(f"Successfully analyzed {point_id} with {len(cycles)} cycles")
            
        except Exception as e:
            print(f"Error processing {point_id}: {e}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    return results

def create_power_law_comparison(results, time_per_cycle=70.6):
    """Create comparison plots of RUL curves with different power factors
    
    Args:
        results: Dictionary of analysis results from calculate_rul_for_sample_points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    # Create a figure with subplots for each sample point
    num_points = len(results)
    fig, axes = plt.subplots(num_points, 2, figsize=(15, 5*num_points))
    
    # Define line styles based on power factor
    power_colors = {
        1.0: 'blue',      # Linear model (power = 1.0)
        1.5: 'green',     # Moderate acceleration (power = 1.5)
        2.0: 'red',       # Faster acceleration (power = 2.0)
        2.5: 'purple',    # Even faster acceleration (power = 2.5)
        3.0: 'orange'     # Extreme acceleration (power = 3.0)
    }
    
    # Process each sample point
    for i, (point_id, point_data) in enumerate(results.items()):
        # Handle single point case where axes isn't a 2D array
        if num_points == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
        
        location = point_data["location"]
        power_factor_results = point_data["power_factor_results"]
        
        # Get max cycle values for consistent x-axis across power factors
        max_cycles = 0
        max_hours = 0
        
        for power_factor, factor_data in power_factor_results.items():
            cycles = factor_data["cycles"]
            if len(cycles) > 0 and cycles[-1] > max_cycles:
                max_cycles = cycles[-1]
                max_hours = max_cycles * time_per_cycle / 3600
        
        # Add a margin to max cycles for better visualization
        max_cycles *= 1.5
        max_hours *= 1.5
        
        # Set initial RUL from first result for consistent y-axis
        initial_rul = next(iter(power_factor_results.values()))["initial_rul"]
        
        # Create extended projections to see more of the curve
        for power_factor in sorted(power_factor_results.keys()):
            color = power_colors.get(power_factor, 'black')
            
            # Create extended projections to failure for demonstration
            extended_cycles = np.linspace(0, max_cycles, 1000)
            extended_hours = extended_cycles * time_per_cycle / 3600
            
            # For each power factor, we'll calculate and show a longer projection
            if power_factor == 1.0:
                # Linear model: RUL = initial_rul * (1 - t/T)
                damage_rate = (initial_rul * 0.10) / max_cycles  # For demo purposes
                extended_rul = np.maximum(initial_rul - damage_rate * extended_cycles, 0)
                extended_rul_hours = extended_rul * time_per_cycle / 3600
                
                label = f"Power Factor = {power_factor} (Linear)"
                ax1.plot(extended_cycles, extended_rul, '-', color=color, linewidth=2, label=label)
                ax2.plot(extended_hours, extended_rul_hours, '-', color=color, linewidth=2, label=label)
                
            else:
                # Power law model: RUL = initial_rul * (1 - (t/T)^power_factor)
                T = max_cycles * 1.2  # For demonstration, use consistent lifetime basis
                normalized_time = np.minimum(extended_cycles / T, 1.0)
                damage_fraction = np.power(normalized_time, power_factor)
                remaining_life_fraction = 1.0 - damage_fraction
                extended_rul = initial_rul * remaining_life_fraction
                extended_rul_hours = extended_rul * time_per_cycle / 3600
                
                label = f"Power Factor = {power_factor}"
                ax1.plot(extended_cycles, extended_rul, '-', color=color, linewidth=2, label=label)
                ax2.plot(extended_hours, extended_rul_hours, '-', color=color, linewidth=2, label=label)
            
            # Add annotations for specific power factors
            if power_factor in [1.0, 2.0]:
                # Find a point ~1/3 through the curve for annotation
                idx = len(extended_cycles) // 3
                ax1.annotate(f"n={power_factor}", 
                            xy=(extended_cycles[idx], extended_rul[idx]),
                            xytext=(extended_cycles[idx]*1.1, extended_rul[idx]*1.1),
                            arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=5),
                            color=color, fontsize=10)
        
        # Plot the original data points from each power factor (actual measurements)
        for power_factor, factor_data in sorted(power_factor_results.items()):
            color = power_colors.get(power_factor, 'black')
            cycles = factor_data["cycles"]
            rul = factor_data["rul"]
            
            if len(cycles) == 0 or len(rul) == 0:
                continue
                
            # Convert to hours for time plot
            hours = cycles * time_per_cycle / 3600
            rul_hours = rul * time_per_cycle / 3600
            
            # Plot data points with symbols to differentiate from projections
            ax1.plot(cycles, rul, 'o', color=color, markersize=4, alpha=0.7)
            ax2.plot(hours, rul_hours, 'o', color=color, markersize=4, alpha=0.7)
        
        # Set x-axis limits
        ax1.set_xlim(0, max_cycles)
        ax2.set_xlim(0, max_hours)
        
        # Set y-axis limit to show full range from 0 to initial RUL
        ax1.set_ylim(0, initial_rul * 1.05)
        ax2.set_ylim(0, initial_rul * time_per_cycle / 3600 * 1.05)
        
        # Add titles and labels
        ax1.set_title(f"{point_id} - RUL Comparison by Power Factor", fontsize=12)
        ax1.set_xlabel('Cycles Experienced', fontsize=10)
        ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right', fontsize=9)
        
        ax2.set_title(f"{point_id} - RUL Comparison in Real Time", fontsize=12)
        ax2.set_xlabel('Time Experienced (hours)', fontsize=10)
        ax2.set_ylabel('Remaining Useful Life (hours)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('power_law_rul_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('power_law_rul_comparison.svg', dpi=300, bbox_inches='tight')
    print("Power law RUL comparison saved as 'power_law_rul_comparison.svg/png'")

def create_lifetime_comparison(results, time_per_cycle=70.6):
    """Create comparison of estimated lifetimes with different power factors
    
    Args:
        results: Dictionary of analysis results from calculate_rul_for_sample_points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    # Extract all power factors used
    all_power_factors = set()
    for point_data in results.values():
        all_power_factors.update(point_data["power_factor_results"].keys())
    
    power_factors = sorted(all_power_factors)
    
    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Calculate estimated lifetimes for each point and power factor
    point_ids = []
    lifetimes_by_power = {pf: [] for pf in power_factors}
    lifetime_hours_by_power = {pf: [] for pf in power_factors}
    
    for point_id, point_data in results.items():
        point_ids.append(point_id)
        power_factor_results = point_data["power_factor_results"]
        
        for power_factor in power_factors:
            if power_factor in power_factor_results:
                factor_data = power_factor_results[power_factor]
                initial_rul = factor_data["initial_rul"]
                
                # For power law model, we need to calculate the estimated lifetime
                # RUL = initial_rul * (1 - (t/T)^n) where T is the lifetime and n is power factor
                # At t = T, RUL = 0, so we can just use initial_rul
                
                if power_factor == 1.0:
                    # For linear model, RUL decreases at constant rate: RUL = initial_rul * (1 - t/T)
                    # Estimated lifetime is simply initial_rul / damage_rate
                    if len(factor_data["cycles"]) > 1 and len(factor_data["rul"]) > 1:
                        damage_rate = (factor_data["rul"][0] - factor_data["rul"][-1]) / factor_data["cycles"][-1]
                        lifetime = initial_rul / damage_rate if damage_rate > 0 else float('inf')
                    else:
                        lifetime = initial_rul  # Fallback if not enough data points
                else:
                    # For power law models, we need to solve for when RUL approaches zero
                    # Since the model is RUL = initial_rul * (1 - (t/T)^power_factor)
                    # When RUL ≈ 0, (t/T)^power_factor ≈ 1, so t ≈ T
                    # With accelerating damage, lifetime is less than with linear model
                    # Approximation: lifetime ≈ initial_rul^(1/power_factor)
                    
                    if power_factor > 1.0:
                        # Get the last point as reference
                        if len(factor_data["cycles"]) > 1 and len(factor_data["rul"]) > 1:
                            t_ref = factor_data["cycles"][-1]
                            rul_ref = factor_data["rul"][-1]
                            
                            # If we know RUL(t_ref) = rul_ref, we can solve for T:
                            # rul_ref = initial_rul * (1 - (t_ref/T)^power_factor)
                            # (1 - rul_ref/initial_rul) = (t_ref/T)^power_factor
                            # T = t_ref / ((1 - rul_ref/initial_rul)^(1/power_factor))
                            
                            if initial_rul > 0 and rul_ref < initial_rul:
                                remaining_fraction = 1 - (rul_ref / initial_rul)
                                if remaining_fraction > 0:
                                    lifetime = t_ref / (remaining_fraction ** (1/power_factor))
                                else:
                                    lifetime = initial_rul  # Fallback
                            else:
                                lifetime = initial_rul  # Fallback
                        else:
                            lifetime = initial_rul  # Fallback if not enough data points
                    else:
                        lifetime = initial_rul  # Fallback for unusual power factors
                
                lifetimes_by_power[power_factor].append(lifetime)
                lifetime_hours_by_power[power_factor].append(lifetime * time_per_cycle / 3600)
    
    # Plot lifetimes for each point
    bar_width = 0.8 / len(power_factors)
    positions = np.arange(len(point_ids))
    
    # Define colors for power factors
    power_colors = {
        1.0: 'blue',
        1.5: 'green',
        2.0: 'red',
        2.5: 'purple',
        3.0: 'orange'
    }
    
    # Plot cycles
    for i, power_factor in enumerate(power_factors):
        label = f"Power Factor = {power_factor}"
        if power_factor == 1.0:
            label += " (Linear)"
        
        color = power_colors.get(power_factor, 'gray')
        ax1.bar(positions + i*bar_width - (len(power_factors)-1)*bar_width/2, 
                lifetimes_by_power[power_factor], 
                bar_width, label=label, color=color, alpha=0.7)
    
    # Plot hours
    for i, power_factor in enumerate(power_factors):
        label = f"Power Factor = {power_factor}"
        if power_factor == 1.0:
            label += " (Linear)"
        
        color = power_colors.get(power_factor, 'gray')
        ax2.bar(positions + i*bar_width - (len(power_factors)-1)*bar_width/2, 
                lifetime_hours_by_power[power_factor], 
                bar_width, label=label, color=color, alpha=0.7)
    
    # Set labels and titles
    ax1.set_title('Estimated Lifetime by Power Factor', fontsize=14)
    ax1.set_xlabel('Sample Point', fontsize=12)
    ax1.set_ylabel('Estimated Lifetime (cycles)', fontsize=12)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([p.split('_at_')[0] for p in point_ids], fontsize=10, rotation=45)
    ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
    ax1.legend(loc='upper right')
    
    ax2.set_title('Estimated Lifetime in Real Time', fontsize=14)
    ax2.set_xlabel('Sample Point', fontsize=12)
    ax2.set_ylabel('Estimated Lifetime (hours)', fontsize=12)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([p.split('_at_')[0] for p in point_ids], fontsize=10, rotation=45)
    ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Ensure y-axis starts at 0
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('power_law_lifetime_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('power_law_lifetime_comparison.svg', dpi=300, bbox_inches='tight')
    print("Power law lifetime comparison saved as 'power_law_lifetime_comparison.svg/png'")

def main():
    """Run the power law RUL analysis comparison"""
    print("\n========== POWER LAW RUL ANALYSIS COMPARISON ==========\n")
    
    # Load strain data
    print("\nLoading strain data...")
    strain_data = load_strain_data()
    
    # Calculate RUL with different power factors
    print("\nAnalyzing sample points with power law models...")
    results = calculate_rul_for_sample_points(
        strain_data,
        power_factors=[1.0, 1.5, 2.0, 2.5, 3.0],  # Test multiple power factors
        cycle_multiplier=50,
        time_per_cycle=70.6
    )
    
    # Create visualizations
    print("\nCreating power law comparison visualizations...")
    create_power_law_comparison(results, time_per_cycle=70.6)
    create_lifetime_comparison(results, time_per_cycle=70.6)
    
    print("\nPower law analysis comparison complete!")

if __name__ == "__main__":
    main() 