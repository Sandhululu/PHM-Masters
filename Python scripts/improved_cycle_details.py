#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved script to print out rainflow cycles with proper mean values
and position tracking in the original signal.

@author: Jayron Sandhu
"""

import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def main():
    """Load strain data, identify cycles with proper mean values and positions"""
    print("\n=== Improved Rainflow Cycle Analysis ===\n")
    
    # Load strain data
    print("Loading strain data...")
    data = load_all_data()
    
    # Extract data from dictionary or tuple
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['exx']
        strain_eyy = data['eyy']
        time_points = data['time_points']
    else:
        # Fallback to tuple format for backward compatibility
        _, strain_exx, _, strain_eyy, thermal_strain, time_points, _, _, _, _ = data
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Find the point with maximum principal strain
    max_principal_loc = np.unravel_index(
        np.nanargmax(np.nanmax(major_principal_strain, axis=0)), 
        major_principal_strain[0].shape
    )
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    
    # Extract strain signal at this location
    strain_signal = major_principal_strain[:, max_principal_loc[0], max_principal_loc[1]]
    
    # Clean nan values from the signal
    clean_signal = np.copy(strain_signal)
    mask = ~np.isnan(clean_signal)
    if not np.any(mask):
        print("Warning: All NaN values in signal. Exiting.")
        return 1
    
    # Interpolate missing data
    indices = np.arange(len(clean_signal))
    clean_signal = np.interp(indices, indices[mask], clean_signal[mask])
    
    # Plot the strain signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, clean_signal)
    plt.title(f'Principal Strain Signal at Point {max_principal_loc}')
    plt.xlabel('Time (s)')
    plt.ylabel('Principal Strain')
    plt.grid(True)
    plt.savefig('strain_signal.png')
    
    print(f"\nStrain signal properties:")
    print(f"  Mean: {np.mean(clean_signal):.6e}")
    print(f"  Min: {np.min(clean_signal):.6e}")
    print(f"  Max: {np.max(clean_signal):.6e}")
    
    # Find turning points in the signal
    turning_points, turning_indices = find_turning_points(clean_signal)
    
    print(f"\nFound {len(turning_points)} turning points in the signal")
    
    # Identify cycles using the project's rainflow function
    print("\nIdentifying cycles with the original rainflow implementation...")
    cycles_original = identify_cycles(strain_signal)
    
    # Now create enhanced cycles with proper mean values
    print("\nEnhancing cycles with proper mean values...")
    enhanced_cycles = enhance_cycles_with_means(cycles_original, clean_signal, turning_points, turning_indices)
    
    # Check if cycles were identified
    if len(enhanced_cycles) == 0:
        print("No enhanced cycles could be created. Exiting.")
        return 1
    
    # Print enhanced cycles
    print(f"\nTotal enhanced cycles: {len(enhanced_cycles)}")
    print(f"Format: (range, mean, count, i_start, i_end)")
    print("\nDetailed cycle information:")
    
    # Create a pandas DataFrame for better formatting
    columns = ["Range", "Mean", "Count", "Start Index", "End Index"]
    df = pd.DataFrame(enhanced_cycles, columns=columns)
    
    # Add a cycle number column at the beginning
    df.insert(0, "Cycle #", range(1, len(enhanced_cycles) + 1))
    
    # Print the DataFrame
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6e}" if isinstance(x, (float, np.floating)) else str(x)))
    
    # Save to CSV
    csv_path = os.path.join(os.getcwd(), "enhanced_cycle_details.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEnhanced cycle details saved to: {csv_path}")
    
    # Print some additional statistics
    print("\nCycle Statistics:")
    print(f"  Minimum range: {df['Range'].min():.6e}")
    print(f"  Maximum range: {df['Range'].max():.6e}")
    print(f"  Average range: {df['Range'].mean():.6e}")
    print(f"  Minimum mean: {df['Mean'].min():.6e}")
    print(f"  Maximum mean: {df['Mean'].max():.6e}")
    print(f"  Average mean: {df['Mean'].mean():.6e}")
    print(f"  Total cycle count: {df['Count'].sum()}")
    
    # Create a histogram of cycle means
    plt.figure(figsize=(10, 6))
    plt.hist(df['Mean'], bins=20, alpha=0.7, color='blue')
    plt.title('Histogram of Cycle Mean Values')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('cycle_means_histogram.png')
    
    # Create a scatter plot of range vs. mean
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Range'], df['Mean'], alpha=0.7)
    plt.title('Cycle Range vs. Mean')
    plt.xlabel('Range')
    plt.ylabel('Mean')
    plt.grid(True, alpha=0.3)
    plt.savefig('cycle_range_vs_mean.png')
    
    print("\nAnalysis complete! Created histogram and scatter plot visualizations.")
    return 0

def find_turning_points(signal):
    """
    Find turning points (peaks and valleys) in a signal
    
    Args:
        signal: 1D array of signal values
        
    Returns:
        tuple: (turning_points, turning_indices)
    """
    turning_points = []
    turning_indices = []
    
    if len(signal) < 3:
        return np.array(turning_points), np.array(turning_indices)
    
    # Add first point as a turning point
    turning_points.append(signal[0])
    turning_indices.append(0)
    
    # Find turning points by checking slope changes
    for i in range(1, len(signal)-1):
        # Check if this is a turning point (peak or valley)
        if (signal[i-1] < signal[i] and signal[i] > signal[i+1]) or \
           (signal[i-1] > signal[i] and signal[i] < signal[i+1]):
            turning_points.append(signal[i])
            turning_indices.append(i)
    
    # Add last point as a turning point
    turning_points.append(signal[-1])
    turning_indices.append(len(signal)-1)
    
    return np.array(turning_points), np.array(turning_indices)

def enhance_cycles_with_means(original_cycles, signal, turning_points, turning_indices):
    """
    Enhance rainflow cycles with proper means and attempt to track positions
    
    Args:
        original_cycles: Original cycles from identify_cycles function
        signal: The original signal array
        turning_points: Array of turning point values
        turning_indices: Array of turning point indices
        
    Returns:
        list: Enhanced cycles with format (range, mean, count, i_start, i_end)
    """
    enhanced_cycles = []
    
    # Check if we have valid inputs
    if len(original_cycles) == 0 or len(turning_points) == 0:
        return enhanced_cycles
    
    # Check if original_cycles have means
    # If not, calculate means from signal
    for i, cycle in enumerate(original_cycles):
        # Extract cycle data
        if len(cycle) >= 3:  # Assuming (range, mean, count, ...)
            rng, mean, count = cycle[0], cycle[1], cycle[2]
            # Check if mean is already calculated
            if mean == 0:
                # Mean is zero, try to calculate a proper mean
                # For each cycle range, find potential matching turning points
                potential_matches = []
                
                for j in range(len(turning_points) - 1):
                    # Calculate local range between adjacent turning points
                    local_range = abs(turning_points[j] - turning_points[j+1])
                    
                    # If local range is close to the cycle range, consider it a match
                    if 0.8 * local_range <= rng <= 1.2 * local_range:
                        # Calculate mean for this potential match
                        local_mean = (turning_points[j] + turning_points[j+1]) / 2
                        start_idx = turning_indices[j]
                        end_idx = turning_indices[j+1]
                        potential_matches.append((local_mean, start_idx, end_idx))
                
                if potential_matches:
                    # Use the first potential match 
                    # (could be improved by choosing best match)
                    mean, i_start, i_end = potential_matches[0]
                    enhanced_cycles.append((rng, mean, count, i_start, i_end))
                else:
                    # No match found, use global signal mean
                    global_mean = np.mean(signal)
                    enhanced_cycles.append((rng, global_mean, count, 0, 0))
            else:
                # Mean is already calculated, keep it
                i_start = cycle[3] if len(cycle) > 3 else 0
                i_end = cycle[4] if len(cycle) > 4 else 0
                enhanced_cycles.append((rng, mean, count, i_start, i_end))
        elif len(cycle) == 2:  # Assuming (range, count)
            rng, count = cycle
            # Calculate a reasonable mean from the signal
            global_mean = np.mean(signal)
            enhanced_cycles.append((rng, global_mean, count, 0, 0))
    
    return enhanced_cycles

if __name__ == "__main__":
    main() 