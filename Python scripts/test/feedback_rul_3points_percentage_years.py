#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feedback RUL Model - 3 Points Analysis with Percentage and Years

This script implements a focused analysis of just 3 key points using
the feedback RUL model, where damage rate increases as cumulative damage accumulates.
The RUL is expressed as a percentage of initial RUL and time is converted to years
using 70.8 seconds per cycle.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles
from totally_uncapped_rul import analyze_fatigue_uncapped, format_large_number

# Modified RUL estimation function with feedback mechanism
def feedback_rul_model(fatigue_results, cycle_multiplier=1, feedback_factor=1.5):
    """Estimate fatigue life with a damage feedback mechanism.
    
    As cumulative damage increases, the damage rate also increases according to
    a feedback factor, resulting in accelerating damage and non-linear RUL decay.
    
    Args:
        fatigue_results: Fatigue analysis results from analyze_fatigue_uncapped
        cycle_multiplier: Multiplier for number of cycles (default: 1)
        feedback_factor: Controls how strongly damage rate increases with cumulative damage
                        1.0 = linear model (no feedback)
                        >1.0 = accelerating damage (higher values = faster acceleration)
    
    Returns:
        tuple: (rul_values, cycles_experienced, cumulative_damage)
    """
    # Extract cycle data from fatigue_results
    cycles = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
    
    # Check if we have valid cycle data
    if len(cycles) == 0 or len(N_f_cycles) == 0:
        print("No valid cycle data available for RUL estimation")
        return np.array([0]), np.array([0]), np.array([0])
    
    # Apply cycle multiplier to counts
    scaled_counts = counts * cycle_multiplier
    
    # Calculate initial damage metrics (without feedback)
    damage_per_cycle = scaled_counts / N_f_cycles
    base_damage_rate = np.sum(damage_per_cycle) / np.sum(scaled_counts) if np.sum(scaled_counts) > 0 else 0
    
    # Calculate initial RUL (before any cycles) 
    if base_damage_rate > 0:
        initial_rul = 1 / base_damage_rate
    else:
        initial_rul = float('inf')
        print("Very low damage rate detected, theoretical initial RUL is effectively infinite")
    
    # Format the initial RUL display with years conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    time_per_cycle = 70.8  # seconds per cycle
    
    if initial_rul != float('inf'):
        initial_rul_years = initial_rul * time_per_cycle / seconds_per_year
        
        if initial_rul > 1e9:
            print(f"Initial RUL: {initial_rul:.2e} cycles (that's {initial_rul/1e9:.2f} billion cycles)")
            print(f"             {initial_rul_years:.2f} years")
        elif initial_rul > 1e6:
            print(f"Initial RUL: {initial_rul:.1f} cycles (that's {initial_rul/1e6:.2f} million cycles)")
            print(f"             {initial_rul_years:.2f} years")
        else:
            print(f"Initial RUL: {initial_rul:.1f} cycles ({initial_rul_years:.2f} years)")
    else:
        print("Initial RUL: infinite cycles (zero damage rate)")
    
    # Generate a more detailed projection with feedback effect
    if base_damage_rate > 0:
        # Estimate a reasonable maximum lifetime
        max_cycles = min(initial_rul * 5, 1e8)  # Cap for visualization
        
        # Create a sequence of cycle points for projection
        cycle_points = np.linspace(0, max_cycles, 1000)
        
        # Initialize arrays for damage and RUL
        cumulative_damage = np.zeros_like(cycle_points)
        rul_values = np.zeros_like(cycle_points)
        
        # Set initial values
        rul_values[0] = initial_rul
        current_damage_rate = base_damage_rate
        
        # Calculate damage and RUL for each cycle point with feedback
        for i in range(1, len(cycle_points)):
            # Calculate cycle increment
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # Current damage rate increases with cumulative damage 
            # The feedback factor controls how strongly damage accelerates
            # D' = D0 * (1 + feedback_factor * D)
            current_damage_rate = base_damage_rate * (1 + feedback_factor * cumulative_damage[i-1])
            
            # Calculate damage increment for this step
            damage_increment = current_damage_rate * cycle_increment
            
            # Update cumulative damage
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            
            # Calculate RUL based on damage rate and remaining damage capacity
            remaining_capacity = 1 - cumulative_damage[i]
            
            # If we've reached failure, set RUL to zero and stop
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cycle_points = cycle_points[:i+1]
                rul_values = rul_values[:i+1]
                cumulative_damage = cumulative_damage[:i+1]
                break
                
            # Calculate RUL with current damage rate
            rul_values[i] = remaining_capacity / current_damage_rate
    else:
        # For zero damage rate, just return flat infinite RUL
        cycle_points = np.array([0, cycle_multiplier])
        rul_values = np.array([initial_rul, initial_rul])
        cumulative_damage = np.array([0, 0])
    
    # Calculate final metrics for display
    if rul_values[0] != float('inf') and len(rul_values) > 1:
        final_damage = cumulative_damage[-1]
        final_rul = rul_values[-1]
        total_cycles = cycle_points[-1]
        life_used_percentage = (1 - (final_rul / initial_rul)) * 100
        
        # Find where RUL reaches zero (failure point)
        failure_index = np.argmin(rul_values) if 0 in rul_values else -1
        failure_cycles = cycle_points[failure_index] if failure_index >= 0 else "N/A"
        
        # Calculate time in years
        if failure_cycles != "N/A":
            failure_years = failure_cycles * time_per_cycle / seconds_per_year
            failure_str = f"{format_large_number(failure_cycles)} cycles ({failure_years:.2f} years)"
        else:
            failure_str = "N/A"
            
        total_cycles_years = total_cycles * time_per_cycle / seconds_per_year
        
        print(f"\nFeedback RUL Model Analysis (feedback factor = {feedback_factor}):")
        print(f"  Initial RUL: {format_large_number(initial_rul)} cycles ({initial_rul * time_per_cycle / seconds_per_year:.2f} years)")
        if failure_cycles != "N/A":
            print(f"  Projected failure at: {failure_str}")
        print(f"  Final RUL (at {total_cycles:.1f} cycles, {total_cycles_years:.2f} years): {format_large_number(final_rul)}")
        print(f"  Final damage: {final_damage:.5f}")
        print(f"  Life used: {life_used_percentage:.5f}%")
        print(f"  Initial damage rate: {base_damage_rate:.10f} per cycle")
        print(f"  Final damage rate: {base_damage_rate * (1 + feedback_factor * final_damage):.10f} per cycle")
    else:
        print(f"\nFeedback RUL Model Analysis (feedback factor = {feedback_factor}):")
        print(f"  Initial RUL: infinite cycles")
        print(f"  No significant damage accumulation detected")
    
    return rul_values, cycle_points, cumulative_damage

def load_strain_data():
    """Load and prepare strain data in the correct format for RUL analysis"""
    # Load raw data
    data = load_all_data()
    
    # Extract data arrays
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
        time_points = data['time_points']
    else:
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, _ = data
    
    # Print data shape to confirm 3D nature
    print(f"\nStrain data shapes:")
    print(f"  DICExx shape: {strain_exx.shape}")
    print(f"  DICEyy shape: {strain_eyy.shape}")
    print(f"  Time points: {len(time_points)} points from {time_points[0]} to {time_points[-1]} seconds")
    
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

def analyze_three_points(strain_data, cycle_multiplier=50, feedback_factors=[1.0, 1.5, 3.0, 5.0]):
    """Analyze exactly 3 specific points with the feedback RUL model 
    
    Args:
        strain_data: Dictionary containing strain data arrays
        cycle_multiplier: Multiplier for projecting cycles
        feedback_factors: List of feedback factors to evaluate
        
    Returns:
        dict: Results containing RUL curves with different feedback factors
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Spatial dimensions: {rows}x{cols} points")
    print(f"Time points: {time_dim} points from {strain_data['time'][0]} to {strain_data['time'][-1]} seconds")
    
    print("\nAnalyzing 3 specific points with feedback RUL model...")
    
    # Define exactly 3 specific points of interest
    # These are the same points used in the original feedback_rul_model.py
    sample_points = [
        (rows//4, cols//4),      # Upper left quadrant - typically higher strain
        (rows//4, 3*cols//4),    # Upper right quadrant - typically higher strain
        (rows//2, cols//2)       # Center point - typically moderate strain
    ]
    
    # Store results
    results = {}
    
    # Process each sample point
    for i, (r, c) in enumerate(sample_points):
        point_id = f"Point_{i+1}_at_{r}_{c}"
        print(f"\n--- Analyzing {point_id} ---")
        
        try:
            # Extract strain signal for this point from the full 3D array
            signal = major_principal_strain[:, r, c]
            
            print(f"  Signal shape: {signal.shape} ({len(signal)} time points)")
            
            # Skip if all NaN
            if np.all(np.isnan(signal)):
                print(f"  Skipping {point_id} - all NaN values")
                continue
            
            # Identify cycles using rainflow analysis on the complete time series
            print(f"  Performing rainflow analysis on complete time series...")
            cycles = identify_cycles(signal, is_shear_strain=False)
            
            if cycles is None or len(cycles) == 0:
                print(f"  Skipping {point_id} - no cycles identified")
                continue
            
            print(f"  Identified {len(cycles)} cycles in the strain signal")
            
            # Analyze fatigue with uncapped approach
            fatigue_results = analyze_fatigue_uncapped(cycles)
            
            # Store results for this point with different feedback factors
            results[point_id] = {
                "location": (r, c),
                "feedback_results": {}
            }
            
            # Analyze with different feedback factors
            for feedback_factor in feedback_factors:
                print(f"\n  Analyzing with feedback factor = {feedback_factor}:")
                
                # Run feedback RUL model
                rul_values, cycle_points, cumulative_damage = feedback_rul_model(
                    fatigue_results,
                    cycle_multiplier=cycle_multiplier,
                    feedback_factor=feedback_factor
                )
                
                # Store results
                results[point_id]["feedback_results"][feedback_factor] = {
                    "rul": rul_values,
                    "cycles": cycle_points,
                    "damage": cumulative_damage,
                    "initial_rul": rul_values[0] if len(rul_values) > 0 else 0
                }
            
            print(f"\nSuccessfully analyzed {point_id} with {len(cycles)} cycles")
            
        except Exception as e:
            print(f"Error processing {point_id}: {e}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nProcessing time: {elapsed_time:.1f} seconds")
    
    return results

def format_percentage(value, pos):
    """Format function for percentage display on matplotlib axes"""
    return f'{value:.0f}%'

def create_feedback_visualizations(results, time_per_cycle=70.8):
    """Create visualizations comparing different feedback factors
    
    Args:
        results: Dictionary of RUL results with different feedback factors
        time_per_cycle: Time per cycle in seconds
    """
    # Get feedback factors from first point
    first_point = next(iter(results.values()))
    feedback_factors = list(first_point["feedback_results"].keys())
    
    # Define colors for different points
    point_colors = {
        'Point_1_at_22_8': '#d62728',    # Red
        'Point_2_at_22_26': '#2ca02c',   # Green 
        'Point_3_at_44_17': '#1f77b4',   # Blue
    }
    
    # Define line styles for different feedback factors
    factor_styles = {
        1.0: '-',      # Solid line (linear)
        1.5: '--',     # Dashed line
        3.0: '-.',     # Dash-dot line
        5.0: ':',      # Dotted line
    }
    
    # Create visualization 1: Feedback comparison for each point (as percentage vs years)
    create_feedback_comparison_per_point(results, feedback_factors, point_colors, factor_styles, time_per_cycle)
    
    # Create visualization 2: Point comparison for each feedback factor (as percentage vs years)
    create_point_comparison_per_factor(results, feedback_factors, point_colors, factor_styles, time_per_cycle)
    
    # Create visualization 3: Damage accumulation curves (vs years)
    create_damage_visualization(results, feedback_factors, point_colors, factor_styles, time_per_cycle)

def create_feedback_comparison_per_point(results, feedback_factors, point_colors, factor_styles, time_per_cycle=70.8):
    """Create comparison of different feedback factors for each point
    
    Args:
        results: Dictionary of RUL results
        feedback_factors: List of feedback factors
        point_colors: Dictionary of colors for each point
        factor_styles: Dictionary of line styles for each factor
        time_per_cycle: Time per cycle in seconds
    """
    # Calculate seconds per year for conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    
    num_points = len(results)
    fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))
    
    # Handle single point case
    if num_points == 1:
        axs = [axs]
    
    # Process each point
    for i, (point_id, point_data) in enumerate(results.items()):
        ax = axs[i]
        location = point_data["location"]
        color = point_colors.get(point_id, 'black')
        
        # Get initial RUL for proper scaling
        max_rul_years = 0
        
        # Process each feedback factor
        for factor, factor_data in point_data["feedback_results"].items():
            # Get RUL values
            rul_values = factor_data["rul"]
            cycle_points = factor_data["cycles"]
            
            if len(rul_values) > 0 and rul_values[0] != float('inf'):
                initial_rul = rul_values[0]
                
                # Convert RUL to percentage
                rul_percentage = (rul_values / initial_rul) * 100
                
                # Convert cycles to years
                years = cycle_points * time_per_cycle / seconds_per_year
                
                # Keep track of max years for axis limit
                if len(years) > 0:
                    max_rul_years = max(max_rul_years, years[-1])
                
                # Get line style
                line_style = factor_styles.get(factor, '-')
                
                # Plot the RUL curve as percentage vs years
                ax.plot(years, rul_percentage, line_style, 
                        color=color, linewidth=2, 
                        label=f"Feedback Factor = {factor}")
        
        # Set axis limits
        ax.set_ylim(0, 105)  # Percentage 0-100% with small margin
            
        if max_rul_years > 0:
            ax.set_xlim(0, max_rul_years * 1.05)
        
        # Set title and labels
        ax.set_title(f"{point_id} at {location} - Feedback Factor Comparison", fontsize=14)
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("Remaining Useful Life (%)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, loc='best')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    
    plt.tight_layout()
    plt.savefig('feedback_3points_percentage_per_point.png', bbox_inches='tight', dpi=300)
    plt.savefig('feedback_3points_percentage_per_point.svg', bbox_inches='tight', dpi=300)
    print("Saved feedback comparison per point visualization (percentage vs years)")

def create_point_comparison_per_factor(results, feedback_factors, point_colors, factor_styles, time_per_cycle=70.8):
    """Create comparison of different points for each feedback factor
    
    Args:
        results: Dictionary of RUL results
        feedback_factors: List of feedback factors
        point_colors: Dictionary of colors for each point
        factor_styles: Dictionary of line styles for each factor
        time_per_cycle: Time per cycle in seconds
    """
    # Calculate seconds per year for conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    
    num_factors = len(feedback_factors)
    fig, axs = plt.subplots(num_factors, 1, figsize=(12, 4 * num_factors))
    
    # Handle single factor case
    if num_factors == 1:
        axs = [axs]
    
    # Process each feedback factor
    for i, factor in enumerate(feedback_factors):
        ax = axs[i]
        
        # Get relevant data for proper scaling
        max_rul_years = 0
        
        # Process each point
        for point_id, point_data in results.items():
            location = point_data["location"]
            color = point_colors.get(point_id, 'black')
            
            # Get RUL data for this factor
            if factor in point_data["feedback_results"]:
                factor_data = point_data["feedback_results"][factor]
                rul_values = factor_data["rul"]
                cycle_points = factor_data["cycles"]
                
                if len(rul_values) > 0 and rul_values[0] != float('inf'):
                    initial_rul = rul_values[0]
                    
                    # Convert RUL to percentage
                    rul_percentage = (rul_values / initial_rul) * 100
                    
                    # Convert cycles to years
                    years = cycle_points * time_per_cycle / seconds_per_year
                    
                    # Keep track of max years for axis limit
                    if len(years) > 0:
                        max_rul_years = max(max_rul_years, years[-1])
                    
                    # Plot the RUL curve as percentage vs years
                    ax.plot(years, rul_percentage, '-', 
                            color=color, linewidth=2, 
                            label=f"{point_id} at {location}")
        
        # Set axis limits
        ax.set_ylim(0, 105)  # Percentage 0-100% with small margin
            
        if max_rul_years > 0:
            ax.set_xlim(0, max_rul_years * 1.05)
        
        # Set title and labels
        ax.set_title(f"Feedback Factor = {factor} - Point Comparison", fontsize=14)
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("Remaining Useful Life (%)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, loc='best')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(format_percentage))
    
    plt.tight_layout()
    plt.savefig('feedback_3points_percentage_per_factor.png', bbox_inches='tight', dpi=300)
    plt.savefig('feedback_3points_percentage_per_factor.svg', bbox_inches='tight', dpi=300)
    print("Saved point comparison per factor visualization (percentage vs years)")

def create_damage_visualization(results, feedback_factors, point_colors, factor_styles, time_per_cycle=70.8):
    """Create visualization of damage accumulation vs years
    
    Args:
        results: Dictionary of RUL results
        feedback_factors: List of feedback factors
        point_colors: Dictionary of colors for each point
        factor_styles: Dictionary of line styles for each factor
        time_per_cycle: Time per cycle in seconds
    """
    # Calculate seconds per year for conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    
    num_points = len(results)
    fig, axs = plt.subplots(num_points, 1, figsize=(12, 4 * num_points))
    
    # Handle single point case
    if num_points == 1:
        axs = [axs]
    
    # Process each point
    for i, (point_id, point_data) in enumerate(results.items()):
        ax = axs[i]
        location = point_data["location"]
        color = point_colors.get(point_id, 'black')
        
        # Get data for proper scaling
        max_years = 0
        
        # Process each feedback factor
        for factor, factor_data in point_data["feedback_results"].items():
            # Get damage values
            damage_values = factor_data["damage"]
            cycle_points = factor_data["cycles"]
            
            if len(damage_values) > 0:
                # Convert cycles to years
                years = cycle_points * time_per_cycle / seconds_per_year
                
                # Keep track of max years for axis limit
                if len(years) > 0:
                    max_years = max(max_years, years[-1])
                
                # Get line style
                line_style = factor_styles.get(factor, '-')
                
                # Find failure point if available
                if 1.0 in damage_values:
                    failure_idx = np.argmax(damage_values >= 1.0)
                    failure_years = years[failure_idx]
                    label = f"Factor = {factor}, Fails at {failure_years:.2f} yrs"
                else:
                    label = f"Factor = {factor}, No failure"
                
                # Plot the damage curve vs years
                ax.plot(years, damage_values, line_style, 
                        color=color, linewidth=2, 
                        label=label)
        
        # Add failure threshold line (D=1)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Failure Threshold (D=1)")
        
        # Set axis limits
        ax.set_ylim(0, 1.05)
            
        if max_years > 0:
            ax.set_xlim(0, max_years * 1.05)
        
        # Set title and labels
        ax.set_title(f"{point_id} at {location} - Damage Accumulation vs Time", fontsize=14)
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("Cumulative Damage", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig('feedback_3points_damage_vs_years.png', bbox_inches='tight', dpi=300)
    plt.savefig('feedback_3points_damage_vs_years.svg', bbox_inches='tight', dpi=300)
    print("Saved damage accumulation vs years visualization")

def main():
    """Run the feedback RUL model analysis on 3 specific points with percentage and years conversion"""
    print("\n========== FEEDBACK RUL MODEL (3 POINTS) ANALYSIS WITH PERCENTAGE AND YEARS ==========\n")
    
    # Load strain data from all CSV files
    print("Loading strain data from all CSV files...")
    strain_data = load_strain_data()
    
    # Define feedback factors to test
    feedback_factors = [1.0, 1.5, 3.0, 5.0]
    
    # Analyze 3 specific points with feedback model
    results = analyze_three_points(
        strain_data, 
        cycle_multiplier=50, 
        feedback_factors=feedback_factors
    )
    
    # Create visualizations with time_per_cycle=70.8 seconds
    print("\nCreating visualizations...")
    create_feedback_visualizations(results, time_per_cycle=70.8)
    
    print("\nFeedback RUL model (3 points) analysis with percentage and years complete!")

if __name__ == "__main__":
    main() 