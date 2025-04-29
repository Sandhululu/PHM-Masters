#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for comprehensive RUL analysis using all spatial points

This script tests the functionality of analyzing all spatial points
without modifying the original code files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data, print_statistical_summary
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue, estimate_fatigue_life

def analyze_all_spatial_points(major_principal_strain, time_points, cycle_multiplier=50, sample_rate=1):
    """Perform rainflow analysis on all spatial points
    
    Args:
        major_principal_strain: 3D array of major principal strain (time, rows, cols)
        time_points: Array of time points
        cycle_multiplier: Multiplier for cycles (default: 50)
        sample_rate: Sample every Nth point to reduce computation (default: 1 = all points)
        
    Returns:
        tuple: (minimum_rul, average_rul, rul_field, location_of_minimum, representative_curves)
    """
    print("Analyzing spatial points for comprehensive RUL estimation...")
    
    # Get dimensions
    _, rows, cols = major_principal_strain.shape
    
    # Initialize arrays for results
    initial_rul_field = np.full((rows, cols), np.nan)
    final_rul_field = np.full((rows, cols), np.nan)
    all_rul_curves = []
    all_cycle_plots = []
    valid_locations = []
    
    # Process each spatial point (sampling if needed)
    total_points = rows * cols
    valid_points = 0
    sampled_points = 0
    
    # Simple progress tracking
    start_time = time.time()
    total_rows = len(range(0, rows, sample_rate))
    
    # Replace tqdm with simple progress tracking
    for r_idx, r in enumerate(range(0, rows, sample_rate)):
        # Print progress update
        if r_idx % max(1, total_rows // 10) == 0:
            elapsed = time.time() - start_time
            progress = (r_idx / total_rows) * 100
            if r_idx > 0:  # Avoid division by zero
                eta = (elapsed / r_idx) * (total_rows - r_idx)
                print(f"Processing row {r_idx}/{total_rows} ({progress:.1f}%) - ETA: {eta:.1f}s")
            else:
                print(f"Processing row {r_idx}/{total_rows} ({progress:.1f}%)")
                
        for c in range(0, cols, sample_rate):
            sampled_points += 1
            
            # Skip points with NaN values
            strain_signal = major_principal_strain[:, r, c]
            if np.all(np.isnan(strain_signal)):
                continue
            
            # Print the shape of the strain signal (only for the first valid point)
            if valid_points == 0:
                print(f"\nDEBUG INFO:")
                print(f"Major principal strain 3D array shape: {major_principal_strain.shape}")
                print(f"1D strain signal shape at point ({r},{c}): {strain_signal.shape}")
                print(f"First 5 values of strain signal: {strain_signal[:5]}")
                print(f"Last 5 values of strain signal: {strain_signal[-5:]}")
                
            try:
                # Extract strain signal at this location
                cycles = identify_cycles(strain_signal)
                
                # Print cycle info for the first valid point
                if valid_points == 0:
                    print(f"Number of cycles identified: {len(cycles)}")
                    if len(cycles) > 0:
                        print(f"Sample cycle: {cycles[0]}")
                    print("")  # Empty line to separate from other output
                
                # Skip if no cycles identified
                if len(cycles) == 0:
                    continue
                    
                # Analyze fatigue and estimate RUL
                fatigue_results = analyze_fatigue(cycles)
                rul, cycles_plot = estimate_fatigue_life(fatigue_results, cycle_multiplier)
                
                # Store results if valid
                if len(rul) > 0:
                    initial_rul_field[r, c] = rul[0]
                    final_rul_field[r, c] = rul[-1]
                    all_rul_curves.append(rul)
                    all_cycle_plots.append(cycles_plot)
                    valid_locations.append((r, c))
                    valid_points += 1
                    
            except Exception as e:
                print(f"Error processing point ({r},{c}): {e}")
    
    # Find minimum RUL (most critical)
    min_rul = np.nanmin(initial_rul_field)
    min_rul_loc = np.unravel_index(np.nanargmin(initial_rul_field), initial_rul_field.shape)
    
    # Calculate average RUL
    avg_rul = np.nanmean(initial_rul_field)
    
    # Get different representative curves - min, 25th percentile, median, 75th percentile, max
    representative_curves = {}
    
    # Get min RUL curve
    if min_rul_loc in valid_locations:
        min_idx = valid_locations.index(min_rul_loc)
    else:
        # Find the closest valid location to the min_rul_loc
        distances = [((loc[0] - min_rul_loc[0])**2 + (loc[1] - min_rul_loc[1])**2) for loc in valid_locations]
        min_idx = np.argmin(distances)
    
    representative_curves['min'] = {
        'label': 'Minimum RUL',
        'location': valid_locations[min_idx],
        'rul_curve': all_rul_curves[min_idx],
        'cycles_plot': all_cycle_plots[min_idx],
        'initial_rul': all_rul_curves[min_idx][0] if len(all_rul_curves[min_idx]) > 0 else None
    }
    
    # Find percentile-based curves
    if len(valid_locations) > 0:
        # Get all initial RUL values in a flat array
        valid_ruls = initial_rul_field[~np.isnan(initial_rul_field)]
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 100]  # 50 is median, 100 is max
        percentile_values = np.percentile(valid_ruls, percentiles)
        
        # For each percentile, find the location with closest RUL value
        for i, p in enumerate(percentiles):
            target_rul = percentile_values[i]
            rul_diff = np.abs(np.array([c[0] for c in all_rul_curves if len(c) > 0]) - target_rul)
            if len(rul_diff) > 0:
                idx = np.argmin(rul_diff)
                label = 'Median RUL' if p == 50 else f'{p}th Percentile RUL'
                
                representative_curves[f'p{p}'] = {
                    'label': label,
                    'location': valid_locations[idx],
                    'rul_curve': all_rul_curves[idx],
                    'cycles_plot': all_cycle_plots[idx],
                    'initial_rul': all_rul_curves[idx][0] if len(all_rul_curves[idx]) > 0 else None
                }
    
    # Calculate "average representative curve" using the top 20% around median
    if len(valid_locations) >= 5:  # Need enough points to make an average meaningful
        # Find points with RUL close to median (40th-60th percentile)
        median_rul = np.median(valid_ruls)
        threshold = median_rul * 0.2  # 20% tolerance around median
        
        # Find indices of locations with RUL near median
        median_indices = []
        for i, curve in enumerate(all_rul_curves):
            if len(curve) > 0 and abs(curve[0] - median_rul) < threshold:
                median_indices.append(i)
        
        # If we have enough locations around median, calculate average curve
        if len(median_indices) >= 3:
            # Find the longest cycle_plot among median curves
            max_cycles_len = max([len(all_cycle_plots[i]) for i in median_indices])
            
            # Create a common x-axis for all curves
            common_cycles = np.linspace(0, max_cycles_len-1, max_cycles_len)
            
            # Interpolate all curves to this common x-axis
            interpolated_curves = []
            for i in median_indices:
                cycle_plot = all_cycle_plots[i]
                rul_curve = all_rul_curves[i]
                
                if len(cycle_plot) >= 2:  # Need at least 2 points for interpolation
                    # Extend the cycle_plot and rul_curve to max_cycles_len
                    extended_rul = np.interp(
                        common_cycles, 
                        np.linspace(0, len(cycle_plot)-1, len(cycle_plot)), 
                        rul_curve
                    )
                    interpolated_curves.append(extended_rul)
            
            # Calculate average curve
            if interpolated_curves:
                avg_curve = np.mean(interpolated_curves, axis=0)
                
                representative_curves['avg'] = {
                    'label': 'Representative Average RUL',
                    'location': 'Multiple',
                    'rul_curve': avg_curve,
                    'cycles_plot': common_cycles,
                    'initial_rul': avg_curve[0] if len(avg_curve) > 0 else None
                }
    
    total_time = time.time() - start_time
    print(f"Analysis completed in {total_time:.1f} seconds")
    print(f"Analyzed {valid_points}/{sampled_points} valid points (sampled from {total_points} total)")
    print(f"Minimum RUL: {min_rul:.1f} cycles at location {min_rul_loc}")
    print(f"Average RUL: {avg_rul:.1f} cycles")
    
    for key, curve_data in representative_curves.items():
        print(f"{curve_data['label']}: {curve_data['initial_rul']:.1f} cycles at location {curve_data['location']}")
    
    return (min_rul, avg_rul, initial_rul_field, min_rul_loc, representative_curves)

def plot_comprehensive_rul(initial_rul_field, min_rul_loc, avg_rul, 
                          representative_curves, time_per_cycle=70.6):
    """Create a comprehensive RUL visualization using all spatial data
    
    Args:
        initial_rul_field: 2D array of initial RUL values across spatial points
        min_rul_loc: Location of minimum RUL
        avg_rul: Average RUL across all points
        representative_curves: Dictionary of representative curves
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    # Create a two-panel figure
    fig = plt.figure(figsize=(20, 10))
    
    # First panel: Spatial RUL map
    ax1 = fig.add_subplot(121)
    
    # Handle NaN values and extreme values
    masked_data = np.ma.masked_invalid(initial_rul_field)
    # Use log scale for better visualization
    norm = LogNorm(vmin=np.nanmin(initial_rul_field), 
                   vmax=np.nanmax(initial_rul_field))
    
    im = ax1.imshow(masked_data, cmap='viridis_r', norm=norm)
    ax1.set_title('Spatial Distribution of Initial RUL (cycles)', fontsize=14)
    ax1.set_xlabel('X coordinate', fontsize=12)
    ax1.set_ylabel('Y coordinate', fontsize=12)
    
    # Mark locations on the map
    colors = {'min': 'red', 'p25': 'orange', 'p50': 'yellow', 'p75': 'green', 'p100': 'blue'}
    
    for key, curve_data in representative_curves.items():
        if key != 'avg' and isinstance(curve_data['location'], tuple):  # Skip 'avg' which has 'Multiple' location
            loc = curve_data['location']
            color = colors.get(key, 'white')
            ax1.plot(loc[1], loc[0], 'x', color=color, markersize=10, markeredgewidth=2, label=curve_data['label'])
    
    ax1.legend(loc='upper right', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('RUL (cycles) - Log Scale', fontsize=12)
    
    # Second panel: RUL curves for all representative points
    ax2 = fig.add_subplot(122)
    
    # Define colors and markers for the curves
    curve_colors = {
        'min': '#d62728',    # Red
        'p25': '#ff7f0e',    # Orange
        'p50': '#2ca02c',    # Green
        'p75': '#1f77b4',    # Blue
        'p100': '#9467bd',   # Purple
        'avg': '#8c564b'     # Brown
    }
    
    # Plot each representative curve
    max_cycle_len = 0
    for key, curve_data in representative_curves.items():
        color = curve_colors.get(key, '#000000')
        rul_curve = curve_data['rul_curve']
        cycles_plot = curve_data['cycles_plot']
        
        if len(rul_curve) > 0:
            # Update max cycle length for consistent x-axis
            max_cycle_len = max(max_cycle_len, len(cycles_plot))
            
            # Plot the curve
            ax2.plot(cycles_plot, rul_curve, '-', color=color, linewidth=2.5, label=curve_data['label'])
            
            # Add markers at regular intervals
            marker_indices = np.linspace(0, len(cycles_plot)-1, min(10, len(cycles_plot))).astype(int)
            ax2.plot(cycles_plot[marker_indices], rul_curve[marker_indices], 'o', color=color, markersize=7)
    
    # Format axes
    ax2.set_xlabel('Cycles Experienced', fontsize=14)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=14)
    ax2.set_title('RUL Comparison Across Representative Locations', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Enforce consistent scale on x-axis
    ax2.set_xlim([0, max_cycle_len])
    
    # Adjust legend placement
    ax2.legend(loc='best', fontsize=10)
    
    # Add text information boxes
    def format_large_number(num):
        if num > 1_000_000: return f"{num/1_000_000:.1f}M"
        elif num > 1_000: return f"{num/1_000:.1f}k"
        else: return f"{num:.1f}"
    
    def format_time(cycles):
        seconds = cycles * time_per_cycle
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        elif seconds < 2592000:  # 30 days
            return f"{seconds/86400:.1f} days"
        elif seconds < 31536000:  # 365 days
            return f"{seconds/2592000:.1f} months"
        else:
            return f"{seconds/31536000:.1f} years"
    
    # Get min RUL data for info box
    min_data = representative_curves.get('min', {})
    min_rul_curve = min_data.get('rul_curve', [0])
    
    # Info box with statistics
    info_text = (
        f"Overall Statistics:\n"
        f"Minimum RUL: {format_large_number(min_rul_curve[0])} cycles\n"
        f"Final RUL: {format_large_number(min_rul_curve[-1] if len(min_rul_curve) > 0 else 0)} cycles\n"
        f"Average RUL: {format_large_number(avg_rul)} cycles\n"
    )
    
    if time_per_cycle > 0:
        info_text += f"Min lifetime: {format_time(min_rul_curve[0])}\n"
        info_text += f"Avg lifetime: {format_time(avg_rul)}"
    
    ax2.text(0.97, 0.97, info_text,
            transform=ax2.transAxes, fontsize=12,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
    
    # Add histogram
    ax_hist = fig.add_axes([0.13, 0.15, 0.32, 0.2])  # [left, bottom, width, height]
    valid_data = initial_rul_field[~np.isnan(initial_rul_field)]
    
    if len(valid_data) > 0:
        ax_hist.hist(valid_data.flatten(), bins=50, color='teal', alpha=0.7)
        
        # Add vertical lines for each representative curve
        for key, curve_data in representative_curves.items():
            if curve_data['initial_rul'] is not None:
                color = curve_colors.get(key, '#000000')
                ax_hist.axvline(x=curve_data['initial_rul'], color=color, linestyle='--', 
                               linewidth=2, label=curve_data['label'])
        
        ax_hist.axvline(x=avg_rul, color='black', linestyle='-', linewidth=2, label='Overall Average')
        ax_hist.set_xlabel('Initial RUL (cycles)', fontsize=10)
        ax_hist.set_ylabel('Count', fontsize=10)
        ax_hist.set_title('RUL Distribution', fontsize=12)
        # Skip legend here as it would be too crowded
    
    # Format and save
    plt.tight_layout()
    plt.savefig('comprehensive_rul_analysis.svg', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    """Main function to test the comprehensive RUL analysis"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Extract data
    if isinstance(data, dict):
        ThermalStrain = data['thermal_strain']
        DICExx = data['strain_exx']
        DICEyy = data['strain_eyy']
        time_points = data['time_points']
    else:
        _, DICExx, _, DICEyy, ThermalStrain, time_points, _, _, _, _ = data
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Test comprehensive analysis with sampling to reduce computation
    # Sample every 5th point (adjust as needed for your dataset size)
    sample_rate = 5
    print(f"\nTesting comprehensive RUL analysis (sampling every {sample_rate}th point)...")
    
    try:
        # Try with sampling first for quick test
        min_rul, avg_rul, rul_field, min_rul_loc, representative_curves = analyze_all_spatial_points(
            major_principal_strain, time_points, cycle_multiplier=50, sample_rate=sample_rate
        )
        
        # Plot the comprehensive RUL visualization
        print("\nPlotting comprehensive RUL visualization...")
        plot_comprehensive_rul(
            rul_field, min_rul_loc, avg_rul, representative_curves, time_per_cycle=70.6
        )
        
        print("\nTest successful! The comprehensive RUL analysis works with sampling.")
        
        # Ask if user wants to run with full dataset
        run_full = input("\nDo you want to run the analysis with the full dataset? (y/n): ")
        if run_full.lower() == 'y':
            print("\nRunning comprehensive RUL analysis with full dataset (this may take a while)...")
            min_rul, avg_rul, rul_field, min_rul_loc, representative_curves = analyze_all_spatial_points(
                major_principal_strain, time_points, cycle_multiplier=50, sample_rate=1
            )
            
            print("\nPlotting comprehensive RUL visualization (full dataset)...")
            plot_comprehensive_rul(
                rul_field, min_rul_loc, avg_rul, representative_curves, time_per_cycle=70.6
            )
    
    except Exception as e:
        print(f"Error during comprehensive RUL analysis: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTest complete!")

if __name__ == "__main__":
    main() 