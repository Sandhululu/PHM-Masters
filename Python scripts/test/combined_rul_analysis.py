#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined RUL Analysis

This script processes all points in the strain field to generate a combined
RUL estimate that's more accurate than using a single point. The approach:
1. Analyze each point in the strain field 
2. Perform rainflow analysis on each point's strain history
3. Combine the results to generate a comprehensive RUL estimate
4. Compare with single-point approaches
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles
from totally_uncapped_rul import analyze_fatigue_uncapped, format_large_number

def load_strain_data():
    """Load and prepare strain data in the correct format for RUL analysis"""
    # Load raw data
    print("\nLoading strain data...")
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
    print("\nCalculating principal strains...")
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

def analyze_all_points(strain_data, cycle_multiplier=10, sample_rate=4, threshold_percentile=95):
    """Analyze all points in the strain field and combine results
    
    Args:
        strain_data: Dictionary containing strain data arrays
        cycle_multiplier: Multiplier for projecting cycles
        sample_rate: Sample every Nth point (to reduce computational load)
        threshold_percentile: Only consider points above this strain percentile
        
    Returns:
        dict: Combined RUL results
    """
    start_time = time.time()
    
    # Extract major principal strain
    major_principal_strain = strain_data['major_principal_strain']
    time_points = strain_data['time']
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print("\n--- Data Dimensions ---")
    print(f"Spatial dimensions: {rows}x{cols} points")
    print(f"Time points: {time_dim} points from {time_points[0]} to {time_points[-1]} seconds")
    
    # Calculate strain statistics to determine critical areas
    max_strain_per_point = np.nanmax(np.abs(major_principal_strain), axis=0)
    strain_threshold = np.nanpercentile(max_strain_per_point, threshold_percentile)
    
    print(f"\nAnalyzing points with max strain > {strain_threshold:.6f} (top {100-threshold_percentile}%)")
    
    # Store results
    all_results = {}
    critical_points = []
    damage_rates = []
    initial_ruls = []
    
    # Process each point in the strain field
    for r in range(0, rows, sample_rate):
        for c in range(0, cols, sample_rate):
            # Check if this point exceeds the strain threshold
            max_strain = max_strain_per_point[r, c]
            if np.isnan(max_strain) or max_strain < strain_threshold:
                continue
            
            point_id = f"Point_r{r}_c{c}"
            
            try:
                # Extract strain signal for this point
                signal = major_principal_strain[:, r, c]
                
                # Skip if too many NaN values
                nan_percentage = np.isnan(signal).sum() / len(signal) * 100
                if nan_percentage > 20:  # Skip if more than 20% NaN
                    continue
                
                # Identify cycles using rainflow analysis
                cycles = identify_cycles(signal)
                
                if cycles is None or len(cycles) == 0:
                    continue
                
                # Analyze fatigue
                fatigue_results = analyze_fatigue_uncapped(cycles)
                
                # Calculate damage metrics
                cycles_array = fatigue_results.get('cycles', np.array([]))
                counts = fatigue_results.get('counts', np.array([]))
                N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
                
                # Skip if no valid cycles
                if len(cycles_array) == 0 or len(N_f_cycles) == 0:
                    continue
                
                # Calculate damage rate
                damage_per_cycle = counts / N_f_cycles
                total_damage = np.sum(damage_per_cycle)
                total_cycles = np.sum(counts)
                
                if total_cycles == 0 or total_damage == 0:
                    continue
                
                damage_rate = total_damage / total_cycles
                initial_rul = 1 / damage_rate if damage_rate > 0 else float('inf')
                
                # Store results if damage rate is significant
                if damage_rate > 0 and initial_rul != float('inf') and initial_rul < 1e9:
                    all_results[point_id] = {
                        'location': (r, c),
                        'damage_rate': damage_rate,
                        'initial_rul': initial_rul,
                        'max_strain': max_strain,
                        'cycles': len(cycles)
                    }
                    
                    critical_points.append((r, c))
                    damage_rates.append(damage_rate)
                    initial_ruls.append(initial_rul)
                    
                    # Print progress periodically
                    if len(all_results) % 10 == 0:
                        print(f"Processed {len(all_results)} critical points...")
                
            except Exception as e:
                print(f"Error processing point ({r},{c}): {e}")
    
    # Process combined results
    combined_results = {}
    if damage_rates:
        # Convert to numpy arrays
        damage_rates = np.array(damage_rates)
        initial_ruls = np.array(initial_ruls)
        
        # Filter out extreme outliers using Z-score
        z_scores = stats.zscore(damage_rates)
        mask = np.abs(z_scores) < 3
        
        filtered_damage_rates = damage_rates[mask]
        filtered_initial_ruls = initial_ruls[mask]
        
        # Calculate statistics
        mean_damage_rate = np.mean(filtered_damage_rates)
        median_damage_rate = np.median(filtered_damage_rates)
        min_initial_rul = np.min(filtered_initial_ruls)
        
        # Create combined RUL curve with different weightings
        # Linear weighting - standard approach
        cycle_points = np.linspace(0, min(min_initial_rul * 2, 1e8), 1000)
        linear_rul = np.maximum(min_initial_rul - cycle_points, 0)
        
        # Conservative approach - use mean damage rate
        mean_rul = np.maximum(1/mean_damage_rate - cycle_points * mean_damage_rate, 0)
        
        # Most conservative - use 90th percentile of damage rates
        percentile_90_damage = np.percentile(filtered_damage_rates, 90)
        conservative_rul = np.maximum(1/percentile_90_damage - cycle_points * percentile_90_damage, 0)
        
        # Store results
        combined_results = {
            'cycle_points': cycle_points,
            'linear_rul': linear_rul,
            'mean_rul': mean_rul,
            'conservative_rul': conservative_rul,
            'min_initial_rul': min_initial_rul,
            'mean_damage_rate': mean_damage_rate,
            'median_damage_rate': median_damage_rate,
            'critical_points': critical_points,
            'num_points_analyzed': len(all_results),
            'point_results': all_results
        }
        
        print(f"\nAnalysis complete!")
        print(f"Total critical points analyzed: {len(all_results)}")
        print(f"Minimum initial RUL: {min_initial_rul:.1f} cycles")
        print(f"Mean damage rate: {mean_damage_rate:.10f} per cycle")
    else:
        print("\nNo critical points found with significant damage rates")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    return combined_results

def create_combined_rul_visualization(results, time_per_cycle=70.6):
    """Create visualization of combined RUL curves
    
    Args:
        results: Dictionary of combined RUL results
        time_per_cycle: Time per cycle in seconds
    """
    if not results:
        print("No results to visualize")
        return
    
    # Extract data
    cycle_points = results['cycle_points']
    linear_rul = results['linear_rul']
    mean_rul = results['mean_rul']
    conservative_rul = results['conservative_rul']
    min_initial_rul = results['min_initial_rul']
    mean_damage_rate = results['mean_damage_rate']
    
    # Convert cycles to years for second x-axis
    seconds_per_year = 365.25 * 24 * 60 * 60
    years = cycle_points * time_per_cycle / seconds_per_year
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: RUL in cycles
    ax1.plot(cycle_points, linear_rul, 'b-', linewidth=2.5, label="Linear (Min RUL)")
    ax1.plot(cycle_points, mean_rul, 'g--', linewidth=2.5, label="Mean Damage Rate")
    ax1.plot(cycle_points, conservative_rul, 'r-.', linewidth=2.5, label="Conservative (90th percentile)")
    
    ax1.set_xlabel("Cycles Experienced", fontsize=12)
    ax1.set_ylabel("Remaining Useful Life (cycles)", fontsize=12)
    ax1.set_title("Combined RUL Analysis - Cycles", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='best')
    
    # Subplot 2: RUL in years
    ax2.plot(years, linear_rul, 'b-', linewidth=2.5, label="Linear (Min RUL)")
    ax2.plot(years, mean_rul, 'g--', linewidth=2.5, label="Mean Damage Rate")
    ax2.plot(years, conservative_rul, 'r-.', linewidth=2.5, label="Conservative (90th percentile)")
    
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Remaining Useful Life (cycles)", fontsize=12)
    ax2.set_title("Combined RUL Analysis - Years", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='best')
    
    # Add summary stats as text box
    stats_text = (
        f"Analysis based on {results['num_points_analyzed']} critical points\n"
        f"Minimum initial RUL: {format_large_number(min_initial_rul)}\n"
        f"Mean damage rate: {mean_damage_rate:.2e} per cycle\n"
        f"Time to failure (conservative): {years[np.argmin(conservative_rul) if 0 in conservative_rul else -1]:.2f} years"
    )
    
    ax1.text(0.95, 0.05, stats_text, transform=ax1.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('combined_rul_analysis.png', bbox_inches='tight', dpi=300)
    plt.savefig('combined_rul_analysis.svg', bbox_inches='tight', dpi=300)
    print("Saved combined RUL visualization")

def create_spatial_damage_map(results, strain_data):
    """Create a spatial visualization of damage rates across the strain field
    
    Args:
        results: Dictionary of combined RUL results
        strain_data: Original strain data dictionary
    """
    if not results or 'point_results' not in results or not results['point_results']:
        print("No results to visualize")
        return
    
    # Get dimensions
    major_principal_strain = strain_data['major_principal_strain']
    time_dim, rows, cols = major_principal_strain.shape
    
    # Create damage rate and initial RUL maps
    damage_rate_map = np.zeros((rows, cols)) + np.nan
    initial_rul_map = np.zeros((rows, cols)) + np.nan
    
    # Fill in values from analyzed points
    for point_id, point_data in results['point_results'].items():
        r, c = point_data['location']
        damage_rate_map[r, c] = point_data['damage_rate']
        initial_rul_map[r, c] = point_data['initial_rul']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot damage rate map
    im1 = ax1.imshow(damage_rate_map, cmap='viridis', interpolation='nearest')
    ax1.set_title('Spatial Distribution of Damage Rates', fontsize=14)
    ax1.set_xlabel('Column', fontsize=12)
    ax1.set_ylabel('Row', fontsize=12)
    fig.colorbar(im1, ax=ax1, label='Damage Rate (per cycle)')
    
    # Plot initial RUL map (log scale)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rul_map = np.log10(initial_rul_map)
    
    im2 = ax2.imshow(log_rul_map, cmap='plasma_r', interpolation='nearest')
    ax2.set_title('Spatial Distribution of Initial RUL (log10 scale)', fontsize=14)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)
    cbar = fig.colorbar(im2, ax=ax2, label='log10(Initial RUL) (cycles)')
    
    plt.tight_layout()
    plt.savefig('spatial_damage_map.png', bbox_inches='tight', dpi=300)
    plt.savefig('spatial_damage_map.svg', bbox_inches='tight', dpi=300)
    print("Saved spatial damage map visualization")

def main():
    """Run the combined RUL analysis"""
    print("\n========== COMBINED RUL ANALYSIS ==========\n")
    
    # Step 1: Load strain data
    strain_data = load_strain_data()
    
    # Step 2: Analyze points across the strain field (with sampling to reduce computation)
    combined_results = analyze_all_points(
        strain_data,
        cycle_multiplier=10,
        sample_rate=4,        # Sample every 4th point to reduce computation
        threshold_percentile=90  # Only consider points in the top 10% of strain magnitude
    )
    
    # Step 3: Create visualizations
    if combined_results:
        create_combined_rul_visualization(combined_results)
        create_spatial_damage_map(combined_results, strain_data)
    
    print("\nCombined RUL analysis complete!")

if __name__ == "__main__":
    main() 