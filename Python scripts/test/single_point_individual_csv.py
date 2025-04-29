#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Point Individual CSV Analysis

This script takes a single critical point based on average principal strain,
then analyzes this same point from each individual CSV file separately using 
rainflow analysis and determines RUL using the standard method.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data, read_strain_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles
from totally_uncapped_rul import analyze_fatigue_uncapped, format_large_number

def find_critical_point_from_average(data=None):
    """Find the most critical point based on average principal strain magnitude
    
    Args:
        data: Optional pre-loaded data dictionary
        
    Returns:
        tuple: (row, col, avg_strain) for the most critical point
    """
    # Load data if not provided
    if data is None:
        data = load_all_data()
    
    # Extract strain data
    if isinstance(data, dict):
        thermal_strain = data['thermal_strain']
        strain_exx = data['strain_exx']
        strain_eyy = data['strain_eyy']
    else:
        _, strain_exx, _, strain_eyy, thermal_strain, _, _, _, _, _ = data
    
    # Calculate principal strains
    major_principal_strain, minor_principal_strain, _ = calculate_principal_strains(
        thermal_strain, strain_exx, strain_eyy)
    
    # Get dimensions
    time_dim, rows, cols = major_principal_strain.shape
    
    print(f"\n--- Finding Most Critical Point ---")
    print(f"Analyzing strain field with dimensions: {rows}x{cols} points")
    print(f"Time dimension: {time_dim} points")
    
    # Calculate average strain magnitude across all time points
    average_strain = np.nanmean(np.abs(major_principal_strain), axis=0)
    
    # Create a mask for valid data points (non-NaN)
    valid_mask = ~np.isnan(average_strain)
    
    # Find the point with highest average strain
    flat_index = np.nanargmax(average_strain)
    r, c = np.unravel_index(flat_index, (rows, cols))
    strain_value = average_strain[r, c]
    
    print(f"Critical point identified at ({r},{c}): Average strain = {strain_value:.6f}")
    
    # Return the critical point
    return r, c, strain_value

def extract_point_from_individual_csvs(row, col):
    """Extract principal strain values for a specific point from each individual CSV file
    
    Args:
        row: Row index of the critical point
        col: Column index of the critical point
        
    Returns:
        list: List of principal strain values from each CSV file
    """
    print(f"\n--- Extracting Point ({row},{col}) from Individual CSVs ---")
    
    # Get directory paths
    exx_dir = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/exx'
    eyy_dir = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/eyy'
    
    # Get all CSV files sorted by name
    exx_files = sorted(glob.glob(os.path.join(exx_dir, '*.csv')))
    eyy_files = sorted(glob.glob(os.path.join(eyy_dir, '*.csv')))
    
    print(f"Found {len(exx_files)} exx CSV files and {len(eyy_files)} eyy CSV files")
    
    # Check if we have the same number of files
    if len(exx_files) != len(eyy_files):
        print(f"Warning: Different number of exx and eyy files!")
    
    # Take the minimum number to ensure we have matching pairs
    num_files = min(len(exx_files), len(eyy_files))
    
    # Prepare to store principal strain values
    principal_strains = []
    valid_files = []
    
    # Process each CSV file individually
    for i in range(num_files):
        try:
            # Read the CSV files
            exx_data = pd.read_csv(exx_files[i], header=None).values
            eyy_data = pd.read_csv(eyy_files[i], header=None).values
            
            # Check if the point is within bounds
            if (row < exx_data.shape[0] and col < exx_data.shape[1] and 
                row < eyy_data.shape[0] and col < eyy_data.shape[1]):
                
                # Extract the values at the critical point
                exx_value = exx_data[row, col]
                eyy_value = eyy_data[row, col]
                
                # Check if the values are valid
                if not (np.isnan(exx_value) or np.isnan(eyy_value)):
                    # Calculate principal strain
                    avg = (exx_value + eyy_value) / 2
                    diff = (exx_value - eyy_value) / 2
                    radius = np.sqrt(diff**2)  # Assuming exy = 0
                    major_principal = avg + radius
                    
                    # Store the major principal strain value
                    principal_strains.append(major_principal)
                    valid_files.append(i)
                else:
                    print(f"  Skipping file {i+1}: NaN values at point ({row},{col})")
            else:
                print(f"  Skipping file {i+1}: Point ({row},{col}) is out of bounds")
        except Exception as e:
            print(f"  Error processing file {i+1}: {e}")
    
    print(f"Successfully extracted {len(principal_strains)} valid strain values out of {num_files} files")
    
    # Create time points for plotting
    time_points = np.arange(0, len(valid_files) * 0.2, 0.2)
    
    # Plot the extracted strain history for verification
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, principal_strains, 'b-')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Major Principal Strain')
    plt.title(f'Major Principal Strain at Point ({row},{col}) - Individual CSVs')
    plt.grid(True)
    plt.savefig('single_point_strain_history.png', bbox_inches='tight', dpi=300)
    plt.savefig('single_point_strain_history.svg', bbox_inches='tight', dpi=300)
    
    print("Saved strain history plot")
    
    return principal_strains, valid_files

def single_point_individual_rul_analysis(principal_strains):
    """Analyze individual strain cycles and determine RUL
    
    Args:
        principal_strains: List of principal strain values from each CSV
        
    Returns:
        dict: Results containing RUL curves and damage metrics
    """
    print("\n--- Performing RUL Analysis on Individual Cycles ---")
    
    # Identify cycles using rainflow analysis
    cycles = identify_cycles(np.array(principal_strains))
    
    if cycles is None or len(cycles) == 0:
        print("No cycles identified in the strain signal")
        return None
    
    print(f"Identified {len(cycles)} cycles in the strain signal")
    
    # Analyze fatigue using uncapped approach
    fatigue_results = analyze_fatigue_uncapped(cycles)
    
    # Extract data from fatigue results
    strain_ranges = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
    
    # Calculate damage metrics
    damage_per_cycle = counts / N_f_cycles
    total_damage = np.sum(damage_per_cycle)
    
    # Calculate RUL (without feedback mechanism)
    if total_damage > 0:
        initial_rul = 1 / (total_damage / np.sum(counts)) if np.sum(counts) > 0 else float('inf')
    else:
        initial_rul = float('inf')
    
    # Format the initial RUL for display
    if initial_rul != float('inf'):
        if initial_rul > 1e9:
            print(f"Initial RUL: {initial_rul:.2e} cycles (that's {initial_rul/1e9:.2f} billion cycles)")
        elif initial_rul > 1e6:
            print(f"Initial RUL: {initial_rul:.1f} cycles (that's {initial_rul/1e6:.2f} million cycles)")
        else:
            print(f"Initial RUL: {initial_rul:.1f} cycles")
    else:
        print("Initial RUL: infinite cycles (zero damage rate)")
    
    # Generate RUL projection using standard method
    if initial_rul != float('inf'):
        # Create a sequence of cycle points for projection
        cycle_points = np.linspace(0, min(initial_rul * 2, 1e7), 500)
        
        # Calculate damage rate (per cycle)
        damage_rate = total_damage / np.sum(counts)
        
        # Calculate damage accumulation
        cumulative_damage = cycle_points * damage_rate
        
        # Calculate RUL at each point (linear model)
        rul_values = np.maximum(initial_rul - cycle_points, 0)
        
        # Find where RUL reaches zero (failure point)
        failure_index = np.argmin(rul_values) if 0 in rul_values else -1
        failure_cycles = cycle_points[failure_index] if failure_index >= 0 else "N/A"
        
        print("\nStandard RUL Model Analysis (linear model):")
        print(f"  Initial RUL: {format_large_number(initial_rul)}")
        if failure_cycles != "N/A":
            print(f"  Projected failure at: {format_large_number(failure_cycles)} cycles")
        print(f"  Damage rate: {damage_rate:.10f} per cycle")
        
        # Store results
        results = {
            "cycles": strain_ranges,
            "counts": counts,
            "N_f_cycles": N_f_cycles,
            "initial_rul": initial_rul,
            "cycle_points": cycle_points,
            "rul_values": rul_values,
            "cumulative_damage": cumulative_damage,
            "damage_rate": damage_rate
        }
        
        # Create RUL curve visualization
        create_rul_visualization(results)
        
        return results
    else:
        print("Cannot project RUL with zero damage rate")
        return None

def create_rul_visualization(results, time_per_cycle=70.8):
    """Create visualization of RUL curve
    
    Args:
        results: Dictionary of analysis results
        time_per_cycle: Time duration for each cycle in seconds (default: 70.8)
    """
    print("\n--- Creating RUL Visualization ---")
    
    cycle_points = results["cycle_points"]
    rul_values = results["rul_values"]
    cumulative_damage = results["cumulative_damage"]
    initial_rul = results["initial_rul"]
    
    # Convert cycles to years
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    years = cycle_points * time_per_cycle / seconds_per_year
    
    # Convert RUL to percentage of initial life
    rul_percentage = (rul_values / initial_rul) * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot RUL curve (percentage vs years)
    ax1.plot(years, rul_percentage, 'b-', linewidth=2.5)
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Remaining Useful Life (%)')
    ax1.set_title('RUL vs Time (Linear Model)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits for percentage (0-100%)
    ax1.set_ylim([0, 100])
    ax1.yaxis.set_major_locator(plt.MultipleLocator(10))  # 10% intervals
    
    # Format y-axis labels as percentages
    def format_func(value, pos):
        return f'{value:.0f}%'
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Add RUL information text
    ax1.text(0.95, 0.95, 
             f"Initial RUL: {format_large_number(initial_rul)} cycles\n" +
             f"Time to failure: {years[-1]:.2f} years",
             transform=ax1.transAxes, fontsize=12, horizontalalignment='right',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot damage accumulation (vs years)
    ax2.plot(years, cumulative_damage, 'r-', linewidth=2.5)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Failure Threshold')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Cumulative Damage')
    ax2.set_title('Damage Accumulation')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add damage rate information
    ax2.text(0.95, 0.95, 
             f"Damage Rate: {results['damage_rate']:.2e} per cycle\n" +
             f"({results['damage_rate'] * seconds_per_year / time_per_cycle:.2e} per year)",
             transform=ax2.transAxes, fontsize=12, horizontalalignment='right',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    plt.suptitle('Single Critical Point RUL Analysis (Individual CSV Method)', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig('single_point_individual_rul.png', bbox_inches='tight', dpi=300)
    plt.savefig('single_point_individual_rul.svg', bbox_inches='tight', dpi=300)
    
    print("Saved RUL visualization")

def main():
    """Run the single critical point analysis"""
    print("\n========== SINGLE CRITICAL POINT INDIVIDUAL CSV ANALYSIS ==========\n")
    
    # Step 1: Find the most critical point based on average principal strain
    critical_row, critical_col, avg_strain = find_critical_point_from_average()
    
    # Step 2: Extract this point from each individual CSV file
    principal_strains, valid_files = extract_point_from_individual_csvs(critical_row, critical_col)
    
    # Step 3: Perform RUL analysis on the individual cycles
    rul_results = single_point_individual_rul_analysis(principal_strains)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 