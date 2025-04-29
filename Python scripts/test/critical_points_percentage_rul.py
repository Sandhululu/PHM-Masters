#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Critical Points Percentage RUL Analysis

This script enhances the critical points analysis by converting RUL to percentage
and cycles to years, using 70.8 seconds per cycle conversion. This provides a more
intuitive visualization of the RUL curves.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles
from totally_uncapped_rul import analyze_fatigue_uncapped, format_large_number

def format_large_number(num):
    """Format large numbers for display"""
    if num >= 1e9:
        return f"{num/1e9:.2f} billion"
    elif num >= 1e6:
        return f"{num/1e6:.2f} million"
    elif num >= 1e3:
        return f"{num/1e3:.2f}k"
    else:
        return f"{num:.1f}"

def format_percentage(value, pos):
    """Format function for percentage display on matplotlib axes"""
    return f'{value:.0f}%'

def feedback_rul_model(fatigue_results, cycle_multiplier=1, feedback_factor=1.5):
    """Estimate fatigue life with a damage feedback mechanism.
    
    Args:
        fatigue_results: Results from fatigue analysis
        cycle_multiplier: Multiplier for number of cycles
        feedback_factor: Factor that controls damage acceleration
    
    Returns:
        tuple: (rul_values, cycle_points, cumulative_damage)
    """
    # Extract cycle data from fatigue_results
    cycles = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
    
    # Check if we have valid cycle data
    if len(cycles) == 0 or len(N_f_cycles) == 0:
        print("No valid cycle data for RUL estimation")
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
    
    # Format the initial RUL display
    if initial_rul != float('inf'):
        if initial_rul > 1e9:
            print(f"Initial RUL: {initial_rul:.2e} cycles (that's {initial_rul/1e9:.2f} billion cycles)")
        elif initial_rul > 1e6:
            print(f"Initial RUL: {initial_rul:.1f} cycles (that's {initial_rul/1e6:.2f} million cycles)")
        else:
            print(f"Initial RUL: {initial_rul:.1f} cycles")
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
        
        print(f"\nFeedback RUL Model Analysis (feedback factor = {feedback_factor}):")
        print(f"  Initial RUL: {format_large_number(initial_rul)}")
        if failure_cycles != "N/A":
            print(f"  Projected failure at: {format_large_number(failure_cycles)}")
        print(f"  Final RUL (at {cycle_points[-1]:.1f} cycles): {format_large_number(final_rul)}")
        print(f"  Final damage: {final_damage:.5f}")
        print(f"  Life used: {life_used_percentage:.5f}%")
        print(f"  Initial damage rate: {base_damage_rate:.10f} per cycle")
        print(f"  Final damage rate: {base_damage_rate * (1 + feedback_factor * final_damage):.10f} per cycle")
    else:
        print(f"\nFeedback RUL Model Analysis (feedback factor = {feedback_factor}):")
        print(f"  Initial RUL: infinite cycles")
        print(f"  No significant damage accumulation detected")
    
    return rul_values, cycle_points, cumulative_damage

def find_critical_points(strain_data, num_points=5):
    """Find the most critical points based on average principal strain magnitude
    
    Args:
        strain_data: Dictionary containing strain data arrays
        num_points: Number of critical points to find
        
    Returns:
        list: List of (row, col, strain_value) tuples for critical points
    """
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print(f"\n--- Finding {num_points} Most Critical Points ---")
    print(f"Analyzing strain field with dimensions: {rows}x{cols} points")
    print(f"Time dimension: {time_dim} points")
    
    # Calculate average strain magnitude across all time points
    average_strain = np.nanmean(np.abs(major_principal_strain), axis=0)
    
    # Create a mask for valid data points (non-NaN)
    valid_mask = ~np.isnan(average_strain)
    
    # Find the points with highest average strain
    critical_points = []
    
    # Flatten the array for easier processing
    flat_indices = np.argsort(average_strain.flatten())[::-1]
    
    # Convert flat indices to 2D coordinates and collect points
    count = 0
    for flat_idx in flat_indices:
        r, c = np.unravel_index(flat_idx, (rows, cols))
        
        # Check if the point is valid (not NaN)
        if valid_mask[r, c]:
            strain_value = average_strain[r, c]
            critical_points.append((r, c, strain_value))
            count += 1
            print(f"Critical point #{count} at ({r},{c}): Average strain = {strain_value:.6f}")
            
            if count >= num_points:
                break
    
    return critical_points

def analyze_critical_points(strain_data, critical_points, cycle_multiplier=50, feedback_factor=1.5):
    """Analyze critical points with the feedback RUL model
    
    Args:
        strain_data: Dictionary containing strain data arrays
        critical_points: List of (row, col, strain_value) tuples for critical points
        cycle_multiplier: Multiplier for projecting cycles
        feedback_factor: Feedback factor for damage acceleration
        
    Returns:
        dict: Results containing RUL curves
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Spatial dimensions: {rows}x{cols} points")
    print(f"Time points: {time_dim} points from {strain_data['time'][0]} to {strain_data['time'][-1]} seconds")
    
    print(f"\nAnalyzing {len(critical_points)} critical points with feedback RUL model...")
    
    # Store results
    results = {}
    
    # Analyze each critical point
    for i, (r, c, avg_strain) in enumerate(critical_points):
        try:
            point_id = f"Point_{r}_{c}"
            print(f"\nProcessing {point_id} (#{i+1}/{len(critical_points)})...")
            
            # Extract strain signal for this point
            strain_signal = major_principal_strain[:, r, c]
            
            # Skip if all NaN
            if np.all(np.isnan(strain_signal)):
                print(f"Skipping {point_id} - all values are NaN")
                continue
            
            # Clean signal by removing NaNs through interpolation
            valid_indices = ~np.isnan(strain_signal)
            if np.sum(valid_indices) < 2:
                print(f"Skipping {point_id} - not enough valid data points")
                continue
                
            all_indices = np.arange(len(strain_signal))
            strain_signal_clean = np.interp(all_indices, all_indices[valid_indices], strain_signal[valid_indices])
            
            # Identify cycles using rainflow analysis
            cycles = identify_cycles(strain_signal_clean)
            
            if cycles is None or len(cycles) == 0:
                print(f"No cycles identified for {point_id}")
                continue
                
            print(f"Identified {len(cycles)} cycles in the strain signal")
            
            # Analyze fatigue using uncapped approach
            fatigue_results = analyze_fatigue_uncapped(cycles)
            
            # Store results for this point
            results[point_id] = {
                "location": (r, c),
                "avg_strain": avg_strain,
                "cycles": len(cycles)
            }
            
            # Calculate RUL with feedback model
            rul_values, cycle_points, cumulative_damage = feedback_rul_model(
                fatigue_results, cycle_multiplier, feedback_factor
            )
            
            # Store results
            results[point_id]["feedback_results"] = {
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

def create_percentage_rul_visualization(results, time_per_cycle=70.8, feedback_factor=1.5):
    """Create visualization of RUL curves with percentage and years
    
    Args:
        results: Dictionary of analysis results
        time_per_cycle: Time per cycle in seconds (default: 70.8)
        feedback_factor: Feedback factor used (for display)
    """
    # Skip if no results
    if not results:
        print("No results to visualize")
        return
    
    # Calculate seconds per year for conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    
    # Create figure for RUL percentage comparison
    plt.figure(figsize=(12, 8))
    
    # Define colors for different points
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#17becf', '#7f7f7f', '#bcbd22']
    
    # Extract and sort points by initial RUL (ascending)
    points_data = []
    for point_id, point_data in results.items():
        if "feedback_results" in point_data:
            feedback_data = point_data["feedback_results"]
            initial_rul = feedback_data.get("initial_rul", float('inf'))
            if initial_rul != float('inf') and initial_rul > 0:
                points_data.append((point_id, point_data, initial_rul))
    
    # Sort by initial RUL (ascending)
    points_data.sort(key=lambda x: x[2])
    
    # Plot RUL percentage curves
    for i, (point_id, point_data, initial_rul) in enumerate(points_data):
        location = point_data["location"]
        avg_strain = point_data["avg_strain"]
        feedback_data = point_data["feedback_results"]
        
        # Get RUL curve
        rul_values = feedback_data["rul"]
        cycle_points = feedback_data["cycles"]
        
        # Convert cycles to years
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Convert RUL to percentage
        rul_percentage = (rul_values / initial_rul) * 100
        
        # Use color from predefined list
        color = colors[i % len(colors)]
        
        # Plot the RUL percentage curve with years on x-axis
        plt.plot(years, rul_percentage, '-', color=color, linewidth=2,
                label=f"{point_id} (Avg ε: {avg_strain:.6f})")
    
    # Add horizontal line at 0% RUL (failure)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label="Failure (0% RUL)")
    
    # Configure the plot
    plt.title(f"Remaining Useful Life (%) vs Time (Feedback Factor = {feedback_factor})", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Remaining Useful Life (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    # Set y-axis limits for percentage (0-100%)
    plt.ylim([0, 100])
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percentage))
    
    plt.tight_layout()
    plt.savefig('critical_points_rul_percentage.png', bbox_inches='tight', dpi=300)
    plt.savefig('critical_points_rul_percentage.svg', bbox_inches='tight', dpi=300)
    print("Saved critical points RUL percentage visualization")
    
    # Create figure for damage accumulation vs years
    plt.figure(figsize=(12, 8))
    
    # Plot damage curves with years on x-axis
    for i, (point_id, point_data, initial_rul) in enumerate(points_data):
        location = point_data["location"]
        feedback_data = point_data["feedback_results"]
        
        # Get damage curve
        damage_values = feedback_data["damage"]
        cycle_points = feedback_data["cycles"]
        
        # Convert cycles to years
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Calculate years to failure
        if 1.0 in damage_values:
            failure_idx = np.argmax(damage_values >= 1.0)
            years_to_failure = years[failure_idx]
            failure_annotation = f", Fails at {years_to_failure:.2f} years"
        else:
            failure_annotation = ""
        
        # Use color from predefined list
        color = colors[i % len(colors)]
        
        # Plot the damage curve
        plt.plot(years, damage_values, '-', color=color, linewidth=2,
                label=f"{point_id} (Initial RUL: {format_large_number(initial_rul)}{failure_annotation})")
    
    # Add failure threshold line
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Failure Threshold (D=1)")
    
    # Configure the plot
    plt.title(f"Damage Accumulation vs Time (Feedback Factor = {feedback_factor})", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Cumulative Damage", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig('critical_points_damage_vs_years.png', bbox_inches='tight', dpi=300)
    plt.savefig('critical_points_damage_vs_years.svg', bbox_inches='tight', dpi=300)
    print("Saved critical points damage vs years visualization")
    
    # Create summary table with time conversion
    print("\n--- Critical Points Summary with Time Conversion ---")
    print(f"{'Point ID':<20} {'Location':<12} {'Avg Strain':<12} {'Initial RUL (cycles)':<18} {'Initial RUL (years)':<18} {'Failure at (years)':<18}")
    print("-" * 98)
    
    for point_id, point_data, initial_rul in points_data:
        location = point_data["location"]
        avg_strain = point_data["avg_strain"]
        feedback_data = point_data["feedback_results"]
        
        # Get projected failure point (where damage reaches 1.0)
        damage_values = feedback_data["damage"]
        cycle_points = feedback_data["cycles"]
        
        # Convert initial RUL to years
        initial_rul_years = initial_rul * time_per_cycle / seconds_per_year
        
        # Find failure point (if exists)
        if 1.0 in damage_values:
            failure_idx = np.argmax(damage_values >= 1.0)
            failure_cycles = cycle_points[failure_idx]
            failure_years = failure_cycles * time_per_cycle / seconds_per_year
            failure_str = f"{failure_years:.2f}"
        else:
            failure_str = "N/A"
        
        print(f"{point_id:<20} {str(location):<12} {avg_strain:.6f}    {format_large_number(initial_rul):<18} {initial_rul_years:.2f} years{' ':<10} {failure_str:<18}")

def compare_feedback_factors(strain_data, critical_points, cycle_multiplier=50, time_per_cycle=70.8):
    """Create visualization comparing different feedback factors for a single critical point
    
    Args:
        strain_data: Dictionary containing strain data arrays
        critical_points: List of critical points (only the first point will be used)
        cycle_multiplier: Multiplier for projecting cycles
        time_per_cycle: Time per cycle in seconds
    """
    print("\n--- Creating Feedback Factor Comparison ---")
    
    # Use only the first critical point for comparison
    if not critical_points:
        print("No critical points to analyze")
        return
    
    # Extract the first critical point
    first_point = critical_points[0]
    row, col, avg_strain = first_point
    point_id = f"Point_{row}_{col}"
    
    print(f"Using critical point at {(row, col)} for feedback factor comparison")
    
    # Extract strain signal
    major_principal_strain = strain_data['major_principal_strain']
    strain_signal = major_principal_strain[:, row, col]
    
    # Identify cycles
    cycles = identify_cycles(strain_signal)
    
    if len(cycles) == 0:
        print(f"No cycles identified for {point_id}")
        return
    
    # Analyze fatigue using uncapped approach
    fatigue_results = analyze_fatigue_uncapped(cycles)
    
    # Feedback factors to compare - added 0 (no feedback) and 0.5
    feedback_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Calculate seconds per year for conversion
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    
    # Store results for each feedback factor
    factor_results = {}
    max_years = 0
    
    # Process each feedback factor
    for factor in feedback_factors:
        print(f"Processing feedback factor {factor}...")
        
        # Run feedback model
        rul_values, cycle_points, damage_values = feedback_rul_model(
            fatigue_results, cycle_multiplier=cycle_multiplier, feedback_factor=factor)
        
        # Skip if infinite values
        if rul_values[0] == float('inf'):
            print(f"Skipping factor {factor} (infinite RUL)")
            continue
        
        # Convert cycles to years
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Update max years if this curve extends further
        if years[-1] > max_years:
            max_years = years[-1]
        
        # Convert RUL to percentage
        initial_rul = rul_values[0]
        rul_percentage = (rul_values / initial_rul) * 100
        
        # Store results
        factor_results[factor] = {
            "rul_percentage": rul_percentage,
            "years": years,
            "initial_rul": initial_rul,
            "damage": damage_values
        }
        
        # Find where RUL reaches zero (failure point) if it exists
        failure_index = np.argmin(rul_percentage) if 0 in rul_percentage else -1
        if failure_index >= 0:
            failure_years = years[failure_index]
            print(f"  Factor {factor}: Failure at {failure_years:.2f} years")
        else:
            # If curve doesn't reach zero, determine how low it gets
            min_rul_percentage = rul_percentage[-1]
            print(f"  Factor {factor}: Reaches {min_rul_percentage:.2f}% at {years[-1]:.2f} years")
    
    # If no valid results, exit
    if not factor_results:
        print("No valid factor results to plot")
        return
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Define colors for different factors - use a colormap for better differentiation
    import matplotlib.cm as cm
    colormap = cm.viridis
    colors = [colormap(i/len(feedback_factors)) for i in range(len(feedback_factors))]
    
    # Plot RUL percentage curves for each factor
    for i, factor in enumerate(feedback_factors):
        if factor not in factor_results:
            continue
            
        result = factor_results[factor]
        
        # Find where RUL reaches zero (failure point)
        rul_percentage = result["rul_percentage"]
        years = result["years"]
        failure_index = np.argmin(rul_percentage) if 0 in rul_percentage else -1
        
        # Special labeling for 0.0 and 0.5 factors
        if factor == 0.0:
            if failure_index >= 0:
                failure_years = years[failure_index]
                label = f"No Feedback (Linear Model)"
            else:
                label = f"No Feedback (Linear Model)"
            # Use a special line style and color for the linear model
            plt.plot(years, rul_percentage, '--', color='black', linewidth=3, label=label)
        elif factor == 0.5:
            if failure_index >= 0:
                failure_years = years[failure_index]
                label = f"Low Feedback (0.5)"
            else:
                label = f"Low Feedback (0.5)"
            # Use a distinct color for the 0.5 factor
            plt.plot(years, rul_percentage, '-.', color='#ff7f0e', linewidth=2.5, label=label)
        else:
            if failure_index >= 0:
                failure_years = years[failure_index]
                label = f"Factor {factor}"
            else:
                label = f"Factor {factor}"
            # Plot the curve with regular styling
            plt.plot(years, rul_percentage, '-', color=colors[i], linewidth=2, label=label)
    
    # Add horizontal line at 0% RUL (failure)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label="Failure (0% RUL)")
    
    # Configure the plot
    plt.title(f"RUL Comparison with Different Feedback Factors at {point_id}", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Remaining Useful Life (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    # Set y-axis limits for percentage (0-100%)
    plt.ylim([-5, 100])  # Extend slightly below 0 for better visibility
    
    # Ensure x-axis extends far enough to show all curves reaching 0%
    # Calculate required x range - either max_years or extend to reach zero
    for factor, result in factor_results.items():
        rul_percentage = result["rul_percentage"]
        years = result["years"]
        if 0 not in rul_percentage and rul_percentage[-1] > 0:
            # Estimate how much further to reach zero
            # Use the last two points to extrapolate
            if len(rul_percentage) >= 2:
                rate = (rul_percentage[-1] - rul_percentage[-2]) / (years[-1] - years[-2])
                if rate < 0:  # Only if decreasing
                    additional_years = rul_percentage[-1] / abs(rate)
                    if years[-1] + additional_years > max_years:
                        max_years = years[-1] + additional_years * 1.1  # Add 10% margin
    
    plt.xlim([0, max_years * 1.05])  # Add 5% margin
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percentage))
    
    plt.tight_layout()
    plt.savefig('feedback_factors_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('feedback_factors_comparison.svg', bbox_inches='tight', dpi=300)
    print("Saved feedback factors comparison visualization")
    
    # Create a table summarizing the results
    print("\n--- Feedback Factor Comparison Summary ---")
    print(f"Critical Point: {point_id} (Average Strain: {avg_strain:.6f})")
    print(f"{'Feedback Factor':<15} {'Initial RUL (cycles)':<20} {'Years to Failure':<18}")
    print("-" * 55)
    
    for factor in sorted(factor_results.keys()):
        result = factor_results[factor]
        initial_rul = result["initial_rul"]
        rul_percentage = result["rul_percentage"]
        years = result["years"]
        
        # Find failure point if it exists
        failure_index = np.argmin(rul_percentage) if 0 in rul_percentage else -1
        if failure_index >= 0:
            failure_years = years[failure_index]
            failure_str = f"{failure_years:.2f} years"
        else:
            # Calculate projected failure if trend continues
            if len(rul_percentage) >= 2 and rul_percentage[-1] > 0:
                rate = (rul_percentage[-1] - rul_percentage[-2]) / (years[-1] - years[-2])
                if rate < 0:  # Only if decreasing
                    additional_years = rul_percentage[-1] / abs(rate)
                    projected_years = years[-1] + additional_years
                    failure_str = f"Projected: {projected_years:.2f} years"
                else:
                    failure_str = "No failure projected"
            else:
                failure_str = "N/A"
        
        print(f"{factor:<15} {format_large_number(initial_rul):<20} {failure_str:<18}")

def load_data_and_calculate_strain():
    """Load data and calculate principal strains
    
    Returns:
        dict: Strain data dictionary
    """
    print("\n--- Loading Data and Calculating Principal Strains ---")
    
    # Load data
    data = load_all_data()
    
    # Check if data is dictionary or tuple (for backward compatibility)
    if isinstance(data, dict):
        # Use dictionary format
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
        time_points = data['time_points']
        high_strain_points = data['high_strain_points']
    else:
        # Use tuple format for backward compatibility
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, high_strain_points = data
    
    # Calculate principal strains
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Create data dictionary
    strain_data = {
        'major_principal_strain': major_principal_strain,
        'minor_principal_strain': minor_principal_strain,
        'max_shear_strain': max_shear_strain,
        'time': time_points,
        'thermal_strain': thermal_strain,
        'strain_exx': strain_exx, 
        'strain_eyy': strain_eyy,
        'high_strain_points': high_strain_points
    }
    
    return strain_data

def main():
    """Run the critical points percentage RUL analysis"""
    print("\n========== CRITICAL POINTS PERCENTAGE RUL ANALYSIS ==========\n")
    
    # Set analysis parameters
    num_critical_points = 5  # Number of critical points to analyze
    cycle_multiplier = 1     # Multiplier for projecting cycles - Changed from 50 to 1 to match single_point_individual_csv.py
    feedback_factor = 1.5    # Feedback factor for damage acceleration
    time_per_cycle = 70.8    # Time per cycle in seconds
    
    # Load data and calculate strain
    strain_data = load_data_and_calculate_strain()
    
    # Find critical points
    critical_points = find_critical_points(strain_data, num_critical_points)
    
    # Run the feedback factor comparison (using multiple factors)
    compare_feedback_factors(strain_data, critical_points, cycle_multiplier, time_per_cycle)
    
    # Also run the standard analysis for all critical points with the default factor
    results = analyze_critical_points(strain_data, critical_points, cycle_multiplier, feedback_factor)
    
    # Create percentage RUL visualization
    create_percentage_rul_visualization(results, time_per_cycle, feedback_factor)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 