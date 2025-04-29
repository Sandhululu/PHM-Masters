#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract time series data from rainflow cycles

This script demonstrates how to:
1. Load strain data and identify rainflow cycles
2. Find the time indices where each cycle occurs
3. Extract the corresponding time series for each cycle
4. Visualize cycles in the time domain

@author: Based on work by Jayron Sandhu
"""

import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def main():
    """Extract time series data from rainflow cycles"""
    print("\n=== Rainflow Time Series Extraction ===\n")
    
    # Step 1: Load strain data
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
    
    # Step 2: Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Step 3: Find the point with maximum principal strain
    max_principal_loc = np.unravel_index(
        np.nanargmax(np.nanmax(major_principal_strain, axis=0)), 
        major_principal_strain[0].shape
    )
    
    print(f"Maximum principal strain location: {max_principal_loc}")
    
    # Step 4: Extract strain signal at this location
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
    
    # Plot the full strain signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, clean_signal)
    plt.title(f'Principal Strain Signal at Point {max_principal_loc}')
    plt.xlabel('Time (s)')
    plt.ylabel('Principal Strain')
    plt.grid(True)
    plt.savefig('full_strain_signal.png')
    
    # Step 5: Find turning points (peaks and valleys)
    turning_points, turning_indices = find_turning_points(clean_signal)
    
    print(f"\nFound {len(turning_points)} turning points in the signal")
    
    # Plot with turning points highlighted
    plt.figure(figsize=(14, 7))
    plt.plot(time_points, clean_signal, 'b-', label='Signal')
    plt.plot(time_points[turning_indices], turning_points, 'ro', markersize=5, label='Turning Points')
    plt.title(f'Strain Signal with Turning Points Highlighted')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.grid(True)
    plt.legend()
    plt.savefig('strain_with_turning_points.png')
    
    # Step 6: Identify rainflow cycles
    print("\nIdentifying rainflow cycles...")
    cycles_original = identify_cycles(strain_signal)
    
    # Step 7: Match cycles to time series positions
    print("\nMatching cycles to time series positions...")
    cycle_time_series = extract_cycle_time_series(
        cycles_original, clean_signal, turning_points, turning_indices, time_points)
    
    # Create a pandas DataFrame for easier handling
    if cycle_time_series:
        cycles_df = pd.DataFrame(cycle_time_series)
        cycles_df.columns = ["Range", "Mean", "Count", "Start Index", "End Index", 
                           "Start Time", "End Time", "Duration", "Time Series", "Strain Series"]
        
        # Print some cycle statistics
        print(f"\nExtracted time series for {len(cycles_df)} cycles")
        print(f"  Min duration: {cycles_df['Duration'].min():.2f} seconds")
        print(f"  Max duration: {cycles_df['Duration'].max():.2f} seconds")
        print(f"  Avg duration: {cycles_df['Duration'].mean():.2f} seconds")
        
        # Step 8: Plot the largest cycles with their time series
        plot_top_cycles(cycles_df, 5)
        
        # Save cycle time series data to CSV
        cycles_df_to_save = cycles_df.drop(columns=['Time Series', 'Strain Series'])
        csv_path = os.path.join(os.getcwd(), "cycle_time_series.csv")
        cycles_df_to_save.to_csv(csv_path, index=False)
        print(f"\nSaved cycle time data to: {csv_path}")
    else:
        print("\nNo cycle time series could be extracted")
    
    print("\nAnalysis complete!")
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

def extract_cycle_time_series(original_cycles, signal, turning_points, turning_indices, time_points):
    """
    Extract time series data for each rainflow cycle
    
    Args:
        original_cycles: Original cycles from identify_cycles function
        signal: The original strain signal array
        turning_points: Array of turning point values
        turning_indices: Array of turning point indices
        time_points: Array of time points
        
    Returns:
        list: Cycles with time series data [(range, mean, count, i_start, i_end, 
                                           t_start, t_end, duration, time_series, strain_series), ...]
    """
    cycle_time_series = []
    
    # Check if we have valid inputs
    if len(original_cycles) == 0 or len(turning_points) == 0:
        return cycle_time_series
    
    # For each cycle, find the time points where it likely occurs
    for i, cycle in enumerate(original_cycles):
        # Extract cycle data
        if len(cycle) >= 3:  # Assuming (range, mean, count, ...)
            rng, mean, count = cycle[0], cycle[1], cycle[2]
            i_start, i_end = cycle[3] if len(cycle) > 3 else 0, cycle[4] if len(cycle) > 4 else 0
            
            # If the cycle doesn't have start/end indices, try to match with turning points
            if i_start == 0 and i_end == 0:
                # For each cycle range, find potential matching turning points
                potential_matches = []
                
                for j in range(len(turning_points) - 1):
                    # Calculate local range between adjacent turning points
                    local_range = abs(turning_points[j] - turning_points[j+1])
                    
                    # If local range is close to the cycle range, consider it a match
                    if 0.8 * local_range <= rng <= 1.2 * local_range:
                        start_idx = turning_indices[j]
                        end_idx = turning_indices[j+1]
                        potential_matches.append((start_idx, end_idx))
                
                if potential_matches:
                    # Use the first potential match (could be improved by choosing best match)
                    i_start, i_end = potential_matches[0]
            
            # If we have valid start/end indices, extract the time series
            if i_start > 0 and i_end > 0 and i_start < len(time_points) and i_end < len(time_points):
                # Get time values
                t_start = time_points[i_start]
                t_end = time_points[i_end]
                duration = t_end - t_start
                
                # Extract the time series data
                time_series = time_points[i_start:i_end+1]
                strain_series = signal[i_start:i_end+1]
                
                # Add to result
                cycle_time_series.append((rng, mean, count, i_start, i_end, 
                                         t_start, t_end, duration, time_series, strain_series))
        
        elif len(cycle) == 2:  # Simple format (range, count)
            # For the simple format, we can't reliably match to the time series
            # Just report the basic info
            rng, count = cycle
            cycle_time_series.append((rng, np.mean(signal), count, 0, 0, 0, 0, 0, [], []))
    
    return cycle_time_series

def plot_top_cycles(cycles_df, num_cycles=5):
    """
    Plot the top cycles by strain range
    
    Args:
        cycles_df: DataFrame with cycle time series data
        num_cycles: Number of top cycles to plot
    """
    # Sort by range in descending order
    top_cycles = cycles_df.sort_values('Range', ascending=False).head(num_cycles)
    
    # Create a figure with subplots
    n_rows = min(num_cycles, len(top_cycles))
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows))
    
    # Handle case where there's only one cycle
    if n_rows == 1:
        axes = [axes]
    
    # Plot each cycle
    for i, (_, cycle) in enumerate(top_cycles.iterrows()):
        ax = axes[i]
        
        # Check if we have time series data
        if len(cycle['Time Series']) > 0:
            # Plot the time series
            ax.plot(cycle['Time Series'], cycle['Strain Series'], 'b-o', markersize=4)
            
            # Add cycle information
            ax.set_title(f"Cycle #{i+1}: Range={cycle['Range']:.6f}, Mean={cycle['Mean']:.6f}, "
                        f"Duration={cycle['Duration']:.2f}s")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Strain')
            ax.grid(True)
            
            # Highlight the start and end points
            ax.plot(cycle['Time Series'][0], cycle['Strain Series'][0], 'go', markersize=8, label='Start')
            ax.plot(cycle['Time Series'][-1], cycle['Strain Series'][-1], 'ro', markersize=8, label='End')
            
            # Add a horizontal line at the mean value
            ax.axhline(y=cycle['Mean'], color='r', linestyle='--', alpha=0.7, label=f"Mean={cycle['Mean']:.6f}")
            
            # Add range annotation with arrows
            strain_min = min(cycle['Strain Series'])
            strain_max = max(cycle['Strain Series'])
            t_mid = cycle['Time Series'][len(cycle['Time Series'])//2]
            ax.annotate('', xy=(t_mid, strain_min), xytext=(t_mid, strain_max),
                      arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text(t_mid, (strain_min + strain_max)/2, f"Range\n{cycle['Range']:.6f}", 
                   ha='center', va='center', backgroundcolor='white')
            
            ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('top_cycles_time_series.png')
    print(f"\nSaved visualization of top {n_rows} cycles to: top_cycles_time_series.png")

def format_large_number(num):
    """Format large numbers with k/M/B suffixes for readability"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}k"
    else:
        return f"{num:.1f}"

if __name__ == "__main__":
    main() 