#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to process every possible 1D strain signal from the 3D strain field
through rainflow analysis and collect statistics on the results
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles

def process_all_signals():
    """Process every 1D strain signal from the 3D strain field"""
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
    
    # Print raw data dimensions
    print("\n--- Data Dimensions ---")
    print(f"DICExx shape: {DICExx.shape}")
    print(f"Time points: {len(time_points)} points from {time_points[0]} to {time_points[-1]} seconds")
    
    # Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    total_points = rows * cols
    
    print(f"\nProcessing all spatial points: {rows} rows x {cols} columns = {total_points} total points")
    print(f"Each point has a 1D strain signal with {time_dim} time steps")
    
    # Statistics counters
    total_signals = 0
    valid_signals = 0
    signals_with_cycles = 0
    total_cycles = 0
    cycle_counts = []
    strain_ranges = []
    
    # 2D arrays to visualize results
    cycle_count_map = np.full((rows, cols), np.nan)
    max_strain_range_map = np.full((rows, cols), np.nan)
    
    # Track progress
    start_time = time.time()
    progress_interval = max(1, total_points // 20)  # Update progress every 5%
    
    # Process each spatial point
    for r in range(rows):
        for c in range(cols):
            point_idx = r * cols + c
            
            # Show progress
            if point_idx % progress_interval == 0:
                elapsed = time.time() - start_time
                progress = (point_idx / total_points) * 100
                if point_idx > 0:
                    eta = (elapsed / point_idx) * (total_points - point_idx)
                    print(f"Processing point {point_idx}/{total_points} ({progress:.1f}%) - ETA: {eta:.1f}s")
                else:
                    print(f"Processing point {point_idx}/{total_points} ({progress:.1f}%)")
            
            # Get 1D strain signal for this point
            strain_signal = major_principal_strain[:, r, c]
            total_signals += 1
            
            # Skip points with all NaN values
            if np.all(np.isnan(strain_signal)):
                continue
                
            valid_signals += 1
            
            # Process with rainflow algorithm
            try:
                cycles = identify_cycles(strain_signal)
                
                if len(cycles) > 0:
                    signals_with_cycles += 1
                    total_cycles += len(cycles)
                    cycle_counts.append(len(cycles))
                    
                    # Store in visualization maps
                    cycle_count_map[r, c] = len(cycles)
                    
                    # Get max strain range for this point
                    max_range = 0
                    for cycle in cycles:
                        if cycle[0] > max_range:  # First column is strain range/amplitude
                            max_range = cycle[0]
                    
                    strain_ranges.append(max_range)
                    max_strain_range_map[r, c] = max_range
                    
            except Exception as e:
                print(f"Error processing point ({r},{c}): {e}")
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    print("\n--- Rainflow Analysis Results ---")
    print(f"Total spatial points: {total_points}")
    print(f"Valid points (not all NaN): {valid_signals} ({valid_signals/total_points*100:.1f}%)")
    print(f"Points with identified cycles: {signals_with_cycles} ({signals_with_cycles/valid_signals*100:.1f}% of valid)")
    print(f"Total cycles identified: {total_cycles}")
    
    if cycle_counts:
        print(f"Cycles per point: Min={min(cycle_counts)}, Max={max(cycle_counts)}, " 
              f"Mean={sum(cycle_counts)/len(cycle_counts):.1f}")
    
    if strain_ranges:
        print(f"Max strain range: Min={min(strain_ranges):.6f}, Max={max(strain_ranges):.6f}, "
              f"Mean={sum(strain_ranges)/len(strain_ranges):.6f}")
    
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    # Create visualization of results
    print("\nCreating visualizations...")
    
    # Plot 1: Cycle Count Map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    im1 = ax1.imshow(cycle_count_map, cmap='viridis')
    ax1.set_title('Rainflow Cycle Count per Spatial Point')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    fig.colorbar(im1, ax=ax1, label='Number of Cycles')
    
    # Plot 2: Max Strain Range Map
    im2 = ax2.imshow(max_strain_range_map, cmap='plasma')
    ax2.set_title('Maximum Strain Range per Spatial Point')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    fig.colorbar(im2, ax=ax2, label='Strain Range')
    
    plt.tight_layout()
    plt.savefig('rainflow_analysis_maps.svg', bbox_inches='tight')
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    if cycle_counts:
        ax1.hist(cycle_counts, bins=30, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Cycle Counts')
        ax1.set_xlabel('Number of Cycles')
        ax1.set_ylabel('Count')
        ax1.axvline(sum(cycle_counts)/len(cycle_counts), color='red', linestyle='--', 
                    label=f'Mean: {sum(cycle_counts)/len(cycle_counts):.1f}')
        ax1.legend()
    
    if strain_ranges:
        ax2.hist(strain_ranges, bins=30, color='lightgreen', edgecolor='black')
        ax2.set_title('Distribution of Maximum Strain Ranges')
        ax2.set_xlabel('Strain Range')
        ax2.set_ylabel('Count')
        ax2.axvline(sum(strain_ranges)/len(strain_ranges), color='red', linestyle='--',
                   label=f'Mean: {sum(strain_ranges)/len(strain_ranges):.6f}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rainflow_analysis_histograms.svg', bbox_inches='tight')
    
    print("\nVisualization saved to 'rainflow_analysis_maps.svg' and 'rainflow_analysis_histograms.svg'")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    process_all_signals() 