#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Feedback RUL Model

This script implements a comprehensive analysis of all spatial points using
the feedback RUL model, where damage rate increases as cumulative damage accumulates.
It processes the entire strain field with a configurable sampling rate.
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
        
        # Calculate damage and RUL for each cycle point with feedback
        for i in range(1, len(cycle_points)):
            # Calculate cycle increment
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # Current damage rate increases with cumulative damage 
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
    
    return rul_values, cycle_points, cumulative_damage

def load_strain_data():
    """Load and prepare strain data in the correct format for RUL analysis"""
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

def calculate_rul_for_all_points(strain_data, cycle_multiplier=50, sample_rate=4, feedback_factor=1.5, time_per_cycle=70.6):
    """Calculate RUL for all spatial points in the strain field using the feedback model
    
    Args:
        strain_data: Dictionary containing strain data arrays
        cycle_multiplier: Multiplier for projecting cycles into future (default: 50)
        sample_rate: Rate at which to sample spatial points (default: 4, higher = fewer points)
        feedback_factor: Controls how strongly damage accelerates with damage (default: 1.5)
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    
    Returns:
        dict: Analysis results with RUL maps and curves
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Time points: {time_dim} points from {strain_data['time'][0]} to {strain_data['time'][-1]} seconds")
    print(f"Spatial dimensions: {rows}x{cols} points")
    print(f"Using sample rate: {sample_rate} (processing 1/{sample_rate**2} of all points)")
    
    print("\nCalculating feedback RUL analysis for all spatial points...")
    
    # Initialize arrays for results
    initial_rul_map = np.full((rows, cols), np.nan)
    final_rul_map = np.full((rows, cols), np.nan)
    damage_rate_map = np.full((rows, cols), np.nan)
    
    # Store representative RUL curves
    rul_curves = {}
    
    # Track statistics
    total_signals = 0
    valid_signals = 0
    signals_with_cycles = 0
    signals_with_rul = 0
    total_cycles = 0
    cycle_counts = []
    initial_ruls = []
    
    # Save a few representative locations
    representative_points = [
        (rows//4, cols//4),      # Upper left quadrant
        (rows//4, 3*cols//4),    # Upper right quadrant
        (3*rows//4, cols//4),    # Lower left quadrant
        (3*rows//4, 3*cols//4),  # Lower right quadrant
        (rows//2, cols//2)       # Center point
    ]
    
    # Loop through spatial points
    for r in range(0, rows, sample_rate):
        for c in range(0, cols, sample_rate):
            total_signals += 1
            if total_signals % 50 == 1:  # Update progress periodically
                progress = 100 * total_signals / (rows * cols / sample_rate**2)
                if total_signals > 1:
                    elapsed = time.time() - start_time
                    points_per_sec = total_signals / elapsed
                    remaining_points = (rows * cols / sample_rate**2) - total_signals
                    eta = remaining_points / points_per_sec
                    print(f"Processing point {total_signals}/{int(rows * cols / sample_rate**2)} ({progress:.1f}%) - ETA: {eta:.1f}s")
                else:
                    print(f"Processing point {total_signals}/{int(rows * cols / sample_rate**2)} ({progress:.1f}%)")
            
            try:
                # Extract 1D strain signal at this spatial point
                signal = major_principal_strain[:, r, c]
                
                # Skip if all values are NaN
                if np.all(np.isnan(signal)):
                    continue
                
                valid_signals += 1
                
                # Identify cycles using rainflow counting
                cycles = identify_cycles(signal, is_shear_strain=False)
                
                if cycles is None or len(cycles) == 0:
                    continue
                
                signals_with_cycles += 1
                cycle_counts.append(len(cycles))
                total_cycles += len(cycles)
                
                # Analyze fatigue with uncapped approach
                fatigue_results = analyze_fatigue_uncapped(cycles)
                
                # Calculate RUL with feedback model
                rul_values, cycle_points, cumulative_damage = feedback_rul_model(
                    fatigue_results, 
                    cycle_multiplier=cycle_multiplier,
                    feedback_factor=feedback_factor
                )
                
                # Store results in maps
                if len(rul_values) > 0:
                    signals_with_rul += 1
                    
                    # Handle infinite values
                    if rul_values[0] == float('inf'):
                        initial_rul = 1e12  # Use a very large number to represent infinity
                        final_rul = 1e12
                        damage_rate = 0
                    else:
                        initial_rul = rul_values[0]
                        final_rul = rul_values[-1]
                        damage_rate = (initial_rul - final_rul) / cycle_points[-1] if cycle_points[-1] > 0 else 0
                        initial_ruls.append(initial_rul)
                    
                    initial_rul_map[r, c] = initial_rul
                    final_rul_map[r, c] = final_rul
                    damage_rate_map[r, c] = damage_rate
                    
                    # Store RUL curves for representative points or if this is on our list
                    is_representative = False
                    for rep_r, rep_c in representative_points:
                        # Check if this point is close to a representative point
                        # (since we're sampling, we might not hit exact coordinates)
                        if abs(r - rep_r) <= sample_rate and abs(c - rep_c) <= sample_rate:
                            is_representative = True
                            break
                            
                    if is_representative:
                        point_id = f"Point_at_{r}_{c}"
                        rul_curves[point_id] = {
                            'rul': rul_values,
                            'cycles': cycle_points,
                            'damage': cumulative_damage,
                            'location': (r, c),
                            'initial_rul': initial_rul,
                            'damage_rate': damage_rate
                        }
            
            except Exception as e:
                # Just skip problematic points
                print(f"Error processing point at ({r}, {c}): {e}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n--- Analysis Summary ---")
    print(f"Total spatial points in field: {rows * cols}")
    print(f"Points processed: {total_signals} (every {sample_rate}th point)")
    print(f"Valid points (not all NaN): {valid_signals} ({valid_signals/total_signals*100:.1f}%)")
    print(f"Points with identified cycles: {signals_with_cycles} ({signals_with_cycles/valid_signals*100:.1f}% of valid)")
    print(f"Points with calculated RUL: {signals_with_rul} ({signals_with_rul/valid_signals*100:.1f}% of valid)")
    print(f"Total cycles identified: {total_cycles}")
    
    if cycle_counts:
        print(f"Cycles per point: Min={min(cycle_counts)}, Max={max(cycle_counts)}, " 
              f"Mean={sum(cycle_counts)/len(cycle_counts):.1f}")
    
    if initial_ruls:
        min_rul = min(initial_ruls)
        max_rul = max(initial_ruls)
        mean_rul = sum(initial_ruls)/len(initial_ruls)
        median_rul = np.median(initial_ruls)
        
        print(f"Initial RUL (cycles): Min={min_rul:.1f}, Max={max_rul:.1f}, Mean={mean_rul:.1f}, Median={median_rul:.1f}")
        
        # Convert to real time if time_per_cycle is provided
        if time_per_cycle > 0:
            print(f"Initial RUL (time): Min={format_time(min_rul, time_per_cycle)}, "
                  f"Max={format_time(max_rul, time_per_cycle)}, "
                  f"Mean={format_time(mean_rul, time_per_cycle)}")
    
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    # Return both maps and curves
    results = {
        'initial_rul_map': initial_rul_map,
        'final_rul_map': final_rul_map,
        'damage_rate_map': damage_rate_map,
        'rul_curves': rul_curves
    }
    
    return results

def format_time(cycles, time_per_cycle):
    """Format time from cycles and time_per_cycle in a human-readable way"""
    seconds = cycles * time_per_cycle
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 31536000:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/31536000:.1f} years"

def create_heatmap(data, title, filename, log_scale=False, vmin=None, vmax=None):
    """Create a heatmap visualization from 2D spatial data
    
    Args:
        data: 2D array of data to visualize
        title: Title for the plot
        filename: Filename to save the plot
        log_scale: Whether to use logarithmic scale for colormap (default: False)
        vmin: Minimum value for colormap (default: None, auto)
        vmax: Maximum value for colormap (default: None, auto)
    """
    # Skip if data is all NaN
    if np.all(np.isnan(data)):
        print(f"Warning: All NaN values in data for {title}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormap based on scale type
    if log_scale:
        # For log scale, handle data range and set up normalization
        data_no_nan = data[~np.isnan(data)]
        if len(data_no_nan) > 0:
            if vmin is None:
                vmin = max(np.min(data_no_nan), 1e-10)  # Avoid zero/negative
            if vmax is None:
                vmax = np.max(data_no_nan)
            
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            im = ax.imshow(data, norm=norm, cmap=cmap, interpolation='nearest')
    else:
        # For linear scale, use standard normalization
        data_no_nan = data[~np.isnan(data)]
        if len(data_no_nan) > 0:
            if vmin is None:
                vmin = np.min(data_no_nan)
            if vmax is None:
                vmax = np.max(data_no_nan)
            
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            im = ax.imshow(data, norm=norm, cmap=cmap, interpolation='nearest')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    
    # Add title and labels
    ax.set_title(title, fontsize=14)
    if log_scale:
        cbar.set_label('Log scale', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.svg", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap visualization: {filename}.png")

def create_spatial_visualizations(results, time_per_cycle=70.6):
    """Create comprehensive spatial visualizations for the RUL analysis
    
    Args:
        results: Dictionary of analysis results from calculate_rul_for_all_points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    print("\nCreating spatial visualizations...")
    
    # Extract maps
    initial_rul_map = results['initial_rul_map']
    final_rul_map = results['final_rul_map']
    damage_rate_map = results['damage_rate_map']
    
    # Create visualizations for each map
    create_heatmap(
        initial_rul_map, 
        "Initial RUL Map (Cycles) - Feedback Model", 
        "comprehensive_feedback_initial_rul_map",
        log_scale=True
    )
    
    create_heatmap(
        final_rul_map, 
        "Final RUL Map (Cycles) - Feedback Model", 
        "comprehensive_feedback_final_rul_map",
        log_scale=True
    )
    
    # Calculate and visualize life consumed percentage
    life_consumed_map = np.zeros_like(initial_rul_map)
    mask = ~np.isnan(initial_rul_map) & ~np.isnan(final_rul_map) & (initial_rul_map > 0)
    life_consumed_map[mask] = 100 * (1 - final_rul_map[mask] / initial_rul_map[mask])
    
    create_heatmap(
        life_consumed_map, 
        "Life Consumed Percentage - Feedback Model", 
        "comprehensive_feedback_life_consumed_map",
        log_scale=False,
        vmin=0,
        vmax=10  # Typically small percentages, cap at 10%
    )
    
    create_heatmap(
        damage_rate_map, 
        "Damage Rate Map (per cycle) - Feedback Model", 
        "comprehensive_feedback_damage_rate_map",
        log_scale=True
    )
    
    # Convert RUL to time
    if time_per_cycle > 0:
        time_rul_map = initial_rul_map * time_per_cycle / 3600  # Convert to hours
        
        create_heatmap(
            time_rul_map, 
            "Initial RUL Map (Hours) - Feedback Model", 
            "comprehensive_feedback_initial_time_rul_map",
            log_scale=True
        )

def create_rul_curves_visualization(results, time_per_cycle=70.6, feedback_factor=1.5):
    """Create visualization of RUL curves for representative points
    
    Args:
        results: Dictionary of analysis results from calculate_rul_for_all_points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
        feedback_factor: Feedback factor used in the model (for labels)
    """
    # Extract RUL curves
    rul_curves = results['rul_curves']
    
    if not rul_curves:
        print("No RUL curves available for visualization")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Define colors for different points
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#17becf']
    
    # Plot RUL curves (cycles)
    for i, (point_id, curve_data) in enumerate(rul_curves.items()):
        # Get curve data
        rul_values = curve_data['rul']
        cycle_points = curve_data['cycles']
        location = curve_data['location']
        initial_rul = curve_data['initial_rul']
        
        # Plot with consistent color
        color = colors[i % len(colors)]
        
        # Plot the RUL curve
        ax1.plot(cycle_points, rul_values, '-', color=color, linewidth=2, 
                label=f"{point_id} (Initial: {format_large_number(initial_rul)})")
    
    # Plot damage accumulation
    for i, (point_id, curve_data) in enumerate(rul_curves.items()):
        # Get damage data
        damage_values = curve_data['damage']
        cycle_points = curve_data['cycles']
        location = curve_data['location']
        
        # Plot with consistent color
        color = colors[i % len(colors)]
        
        # Plot the damage curve
        ax2.plot(cycle_points, damage_values, '-', color=color, linewidth=2, 
                label=f"{point_id}")
    
    # Add failure threshold line
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Failure Threshold (D=1)")
    
    # Format axes
    ax1.set_title(f"RUL Curves with Feedback Factor = {feedback_factor}", fontsize=14)
    ax1.set_xlabel("Cycles", fontsize=12)
    ax1.set_ylabel("Remaining Useful Life (cycles)", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='best')
    
    ax2.set_title(f"Damage Accumulation with Feedback Factor = {feedback_factor}", fontsize=14)
    ax2.set_xlabel("Cycles", fontsize=12)
    ax2.set_ylabel("Cumulative Damage", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=10, loc='best')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"comprehensive_feedback_rul_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"comprehensive_feedback_rul_curves.svg", dpi=300, bbox_inches='tight')
    
    print("Saved RUL curves visualization: comprehensive_feedback_rul_curves.png")

def main():
    """Run the comprehensive feedback RUL model analysis"""
    print("\n========== COMPREHENSIVE FEEDBACK RUL ANALYSIS ==========\n")
    
    # Load strain data
    print("Loading strain data...")
    strain_data = load_strain_data()
    
    # Define parameters
    cycle_multiplier = 50
    sample_rate = 2  # Process every 2nd point in each direction (1/4 of all points) for better resolution
    feedback_factor = 1.5
    time_per_cycle = 70.6
    
    # Calculate RUL for all spatial points
    results = calculate_rul_for_all_points(
        strain_data, 
        cycle_multiplier=cycle_multiplier,
        sample_rate=sample_rate,
        feedback_factor=feedback_factor,
        time_per_cycle=time_per_cycle
    )
    
    # Check if we have RUL curves and force-add representative points if needed
    if not results['rul_curves']:
        print("\nNo RUL curves captured in sampling. Adding representative points explicitly...")
        major_principal_strain = strain_data['major_principal_strain']
        time_dim, rows, cols = major_principal_strain.shape
        
        # Define key representative points
        representative_points = [
            (rows//4, cols//4),      # Upper left quadrant
            (rows//4, 3*cols//4),    # Upper right quadrant
            (3*rows//4, cols//4),    # Lower left quadrant
            (3*rows//4, 3*cols//4),  # Lower right quadrant
            (rows//2, cols//2)       # Center point
        ]
        
        # Process each representative point individually
        rul_curves = {}
        for r, c in representative_points:
            point_id = f"Point_at_{r}_{c}"
            print(f"Processing representative point {point_id}...")
            
            try:
                # Extract strain signal
                signal = major_principal_strain[:, r, c]
                
                # Skip if all NaN
                if np.all(np.isnan(signal)):
                    print(f"Skipping {point_id} - all NaN values")
                    continue
                
                # Identify cycles
                cycles = identify_cycles(signal, is_shear_strain=False)
                
                if cycles is None or len(cycles) == 0:
                    print(f"Skipping {point_id} - no cycles identified")
                    continue
                
                # Analyze fatigue with uncapped approach
                fatigue_results = analyze_fatigue_uncapped(cycles)
                
                # Calculate RUL with feedback model
                rul_values, cycle_points, cumulative_damage = feedback_rul_model(
                    fatigue_results, 
                    cycle_multiplier=cycle_multiplier,
                    feedback_factor=feedback_factor
                )
                
                # Store result
                if len(rul_values) > 0:
                    # Handle infinite values
                    if rul_values[0] == float('inf'):
                        initial_rul = 1e12  # Use a very large number for infinity
                        damage_rate = 0
                    else:
                        initial_rul = rul_values[0]
                        damage_rate = (initial_rul - rul_values[-1]) / cycle_points[-1] if cycle_points[-1] > 0 else 0
                    
                    rul_curves[point_id] = {
                        'rul': rul_values,
                        'cycles': cycle_points,
                        'damage': cumulative_damage,
                        'location': (r, c),
                        'initial_rul': initial_rul,
                        'damage_rate': damage_rate
                    }
                    
                    print(f"Successfully processed {point_id}")
                
            except Exception as e:
                print(f"Error processing {point_id}: {e}")
        
        # Update results with new RUL curves
        results['rul_curves'] = rul_curves
    
    # Create visualizations
    create_spatial_visualizations(results, time_per_cycle=time_per_cycle)
    create_rul_curves_visualization(results, time_per_cycle=time_per_cycle, feedback_factor=feedback_factor)
    
    print("\nComprehensive feedback RUL analysis complete!")

if __name__ == "__main__":
    main() 