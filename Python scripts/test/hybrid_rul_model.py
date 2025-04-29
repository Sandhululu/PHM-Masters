# -*- coding: utf-8 -*-
"""
Single critical point RUL analysis using a hybrid damage model
Based on average principal strain

@author: Jayron Sandhu
Modified to implement a hybrid linear/feedback model for more realistic damage progression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
import glob

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue

def format_large_number(num):
    """Format large numbers for display"""
    if num > 1_000_000_000:
        return f"{num/1_000_000_000:.2f} billion"
    elif num > 1_000_000:
        return f"{num/1_000_000:.2f} million"
    elif num > 1_000:
        return f"{num/1_000:.2f}k"
    else:
        return f"{num:.1f}"

def find_critical_point_from_average():
    """Find the most critical point based on average principal strain"""
    print("\n--- Finding Most Critical Point ---")
    
    # Load all data
    data = load_all_data()
    
    # Extract data components
    if isinstance(data, dict):
        DICExx = data['exx']
        DICEyy = data['eyy']
    else:
        # Fallback for tuple return type
        _, DICExx, _, DICEyy, _, _, _, _, _, _ = data
    
    # Calculate principal strains
    major_principal_strain, _, _ = calculate_principal_strains(
        np.zeros_like(DICExx), DICExx, DICEyy)
    
    # Print dimensions for debugging
    print(f"Analyzing strain field with dimensions: {major_principal_strain.shape[1]}x{major_principal_strain.shape[2]} points")
    print(f"Time dimension: {major_principal_strain.shape[0]} points")
    
    # Calculate average absolute principal strain over time
    average_strain = np.nanmean(np.abs(major_principal_strain), axis=0)
    
    # Find the location with maximum average strain
    max_loc = np.unravel_index(np.nanargmax(average_strain), average_strain.shape)
    row, col = max_loc
    max_avg_strain = average_strain[row, col]
    
    print(f"Critical point identified at ({row},{col}): Average strain = {max_avg_strain:.6f}")
    
    return row, col, max_avg_strain

def extract_point_from_individual_csvs(row, col):
    """Extract strain values at a specific point from individual CSV files"""
    print(f"\n--- Extracting Point ({row},{col}) from Individual CSVs ---")
    
    # Get lists of file paths
    exx_files = sorted(glob.glob(os.path.join('/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/exx', '*.csv')))
    eyy_files = sorted(glob.glob(os.path.join('/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/eyy', '*.csv')))
    
    print(f"Found {len(exx_files)} exx CSV files and {len(eyy_files)} eyy CSV files")
    
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
    plt.savefig('strain_history_hybrid.png', bbox_inches='tight', dpi=300)
    
    print("Saved strain history plot")
    
    return principal_strains, valid_files

def analyze_fatigue_uncapped(strain_cycles):
    """Analyze fatigue cycles to estimate damage without capping N_f"""
    # Check if we have valid cycle data
    if len(strain_cycles) == 0:
        print("No cycle data available for fatigue analysis")
        return {
            'cycles': np.array([]),
            'damages': np.array([]),
            'cumulative_damage': np.array([]),
            'cycles_array': np.array([])
        }
    
    # Material properties for fatigue analysis
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    # Extract ranges and counts
    if strain_cycles.shape[1] >= 3:  # Make sure we have at least 3 columns
        ranges = strain_cycles[:, 0]  # First column is range
        counts = strain_cycles[:, 2]  # Third column is count
    else:
        # Fallback in case we have simple format
        ranges = strain_cycles[:, 0]
        counts = np.ones_like(ranges)
    
    # Sort cycles by range (amplitude) in descending order
    sorted_indices = np.argsort(ranges)[::-1]
    sorted_ranges = ranges[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Calculate cycles to failure for each strain range
    N_f_cycles = []
    for strain_range in sorted_ranges:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 10, 1000)
        
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life with safety factor
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))] / safety_factor
        N_f_cycles.append(N_f)  # No cap applied
    
    # Calculate damage per cycle using Miner's rule
    damage_per_cycle = [count/N_f for count, N_f in zip(sorted_counts, N_f_cycles)]
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Return comprehensive fatigue analysis results
    return {
        'cycles': sorted_ranges,
        'counts': sorted_counts,
        'N_f_cycles': np.array(N_f_cycles),
        'damages': np.array(damage_per_cycle),
        'cumulative_damage': cumulative_damage,
        'cycles_array': np.arange(len(sorted_ranges))
    }

def hybrid_rul_model(principal_strains, adjustment_models=None):
    """Analyze RUL using a hybrid model with different adjustment methods
    
    Args:
        principal_strains: List of principal strain values from each CSV
        adjustment_models: Dictionary with adjustment model parameters
        
    Returns:
        dict: Results containing RUL curves for different models
    """
    print("\n--- Performing Hybrid RUL Analysis ---")
    
    # Use default adjustment models if none provided
    if adjustment_models is None:
        adjustment_models = {
            'linear': {'name': 'Linear Model', 'function': lambda D: 1.0},  # No adjustment
            'linear_adjustment': {'name': 'Linear Adjustment', 'function': lambda D: 1.0 + 2.0 * D},  # Linear increase with damage
            'quadratic': {'name': 'Quadratic', 'function': lambda D: 1.0 + 5.0 * D**2},  # Quadratic increase with damage
            'exponential': {'name': 'Exponential', 'function': lambda D: 1.0 + 0.2 * np.exp(5.0 * D)}  # Exponential increase with damage
        }
    
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
    
    # Calculate base damage rate and initial RUL
    if total_damage > 0 and np.sum(counts) > 0:
        base_damage_rate = total_damage / np.sum(counts)
        initial_rul = 1 / base_damage_rate
    else:
        base_damage_rate = 0.0
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
    
    # Set up cycle points for projections
    if initial_rul != float('inf'):
        max_cycles = min(initial_rul * 3, 1e8)  # Cap for visualization
        cycle_points = np.linspace(0, max_cycles, 1000)
    else:
        cycle_points = np.linspace(0, 1e5, 1000)  # Default range for infinite RUL
    
    # Initialize dictionaries to store results for each model
    model_results = {}
    
    # Process each adjustment model
    for model_key, model_info in adjustment_models.items():
        model_name = model_info['name']
        adjustment_function = model_info['function']
        
        # Skip if damage rate is zero
        if base_damage_rate <= 0:
            print(f"Skipping {model_name} due to zero damage rate")
            continue
        
        # Initialize arrays for this model
        cumulative_damage = np.zeros_like(cycle_points)
        rul_values = np.zeros_like(cycle_points)
        damage_rates = np.zeros_like(cycle_points)
        
        # Set initial values
        rul_values[0] = initial_rul
        damage_rates[0] = base_damage_rate
        
        # Calculate damage progression with this adjustment model
        for i in range(1, len(cycle_points)):
            # Calculate cycle increment
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # Calculate adjustment factor based on current damage
            adjustment_factor = adjustment_function(cumulative_damage[i-1])
            
            # Calculate current damage rate
            current_damage_rate = base_damage_rate * adjustment_factor
            damage_rates[i] = current_damage_rate
            
            # Calculate damage increment for this step
            damage_increment = current_damage_rate * cycle_increment
            
            # Update cumulative damage
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            
            # Calculate RUL at this point
            remaining_capacity = 1.0 - cumulative_damage[i]
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cumulative_damage[i:] = min(1.0, cumulative_damage[i])
                damage_rates[i:] = damage_rates[i]
                break
            else:
                # Use the current damage rate to project forward
                rul_values[i] = remaining_capacity / current_damage_rate
        
        # Find failure point (where damage reaches 1.0)
        failure_index = np.argmax(cumulative_damage >= 1.0) if np.any(cumulative_damage >= 1.0) else -1
        
        if failure_index > 0:
            failure_cycles = cycle_points[failure_index]
            print(f"\n{model_name} Analysis:")
            print(f"  Initial RUL: {format_large_number(initial_rul)} cycles")
            print(f"  Projected failure at: {format_large_number(failure_cycles)} cycles")
            print(f"  Initial damage rate: {base_damage_rate:.10f} per cycle")
            print(f"  Final damage rate: {damage_rates[failure_index]:.10f} per cycle")
            print(f"  Rate increase: {damage_rates[failure_index]/base_damage_rate:.2f}x")
        else:
            print(f"\n{model_name} Analysis:")
            print(f"  Initial RUL: {format_large_number(initial_rul)} cycles")
            print(f"  No failure projected within analyzed range")
        
        # Store results for this model
        model_results[model_key] = {
            'name': model_name,
            'cycle_points': cycle_points,
            'cumulative_damage': cumulative_damage,
            'rul_values': rul_values,
            'damage_rates': damage_rates,
            'failure_index': failure_index
        }
    
    # Create visualizations
    create_hybrid_model_visualizations(model_results, initial_rul)
    
    # Return all model results
    return {
        'initial_rul': initial_rul,
        'base_damage_rate': base_damage_rate,
        'model_results': model_results
    }

def create_hybrid_model_visualizations(model_results, initial_rul):
    """Create visualizations comparing different RUL models"""
    print("\n--- Creating Hybrid Model Visualizations ---")
    
    # Create color cycle for plots
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Create three subplots: RUL, Damage, and Damage Rate
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Assuming one complete load cycle = 70.8 seconds
    time_per_cycle = 70.8
    seconds_per_year = 365.25 * 24 * 60 * 60
    
    # Plot each model
    for i, (model_key, results) in enumerate(model_results.items()):
        color = colors[i % len(colors)]
        model_name = results['name']
        cycle_points = results['cycle_points']
        cumulative_damage = results['cumulative_damage']
        rul_values = results['rul_values']
        damage_rates = results['damage_rates']
        
        # Convert cycles to years for plotting
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Trim data to failure or max 30 years
        max_year_index = min(np.searchsorted(years, 30), len(years)-1)
        if results['failure_index'] > 0:
            trim_index = min(results['failure_index'], max_year_index)
        else:
            trim_index = max_year_index
        
        years_trimmed = years[:trim_index+1]
        
        # Plot 1: RUL Percentage vs Years
        if initial_rul != float('inf'):
            rul_percentage = (rul_values / initial_rul) * 100
            ax1.plot(years_trimmed, rul_percentage[:trim_index+1], color=color, 
                    linewidth=2.5, label=model_name)
            
        # Plot 2: Damage vs Years
        ax2.plot(years_trimmed, cumulative_damage[:trim_index+1], color=color, 
                linewidth=2.5, label=model_name)
        
        # Plot 3: Damage Rate vs Years
        # Normalize damage rate to base rate for better comparison
        if model_key != 'linear' and initial_rul != float('inf'):
            normalized_rates = damage_rates[:trim_index+1] / damage_rates[0]
            ax3.plot(years_trimmed, normalized_rates, color=color, 
                    linewidth=2.5, label=model_name)
    
    # Configure Plot 1 (RUL Percentage)
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Remaining Useful Life (%)')
    ax1.set_title('RUL vs Time - Model Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    
    # Set y-axis limits for percentage (0-100%)
    ax1.set_ylim([0, 100])
    ax1.yaxis.set_major_locator(plt.MultipleLocator(10))  # 10% intervals
    
    # Format y-axis labels as percentages
    def format_func(value, pos):
        return f'{value:.0f}%'
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Configure Plot 2 (Damage)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Cumulative Damage')
    ax2.set_title('Damage Accumulation - Model Comparison')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Failure Threshold')
    
    # Configure Plot 3 (Damage Rate)
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Damage Rate (normalized)')
    ax3.set_title('Damage Rate Evolution - Model Comparison')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='best')
    
    # Overall title
    if initial_rul != float('inf'):
        plt.suptitle(f'Hybrid RUL Model Comparison (Initial RUL: {format_large_number(initial_rul)} cycles)', 
                    fontsize=16, y=0.98)
    else:
        plt.suptitle('Hybrid RUL Model Comparison (Initial RUL: infinite)', 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig('hybrid_rul_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('hybrid_rul_model_comparison.svg', bbox_inches='tight', dpi=300)
    
    print("Saved hybrid model comparison visualization")

def main():
    """Run the hybrid RUL model analysis"""
    print("\n========== HYBRID RUL MODEL ANALYSIS ==========\n")
    
    # Step 1: Find the most critical point based on average principal strain
    critical_row, critical_col, avg_strain = find_critical_point_from_average()
    
    # Step 2: Extract this point from each individual CSV file
    principal_strains, valid_files = extract_point_from_individual_csvs(critical_row, critical_col)
    
    # Step 3: Perform RUL analysis with hybrid models
    # Define adjustment models with different functions
    adjustment_models = {
        'linear': {
            'name': 'Linear (No Adjustment)', 
            'function': lambda D: 1.0
        },
        'linear_adjustment': {
            'name': 'Linear Adjustment', 
            'function': lambda D: 1.0 + 1.5 * D
        },
        'quadratic': {
            'name': 'Quadratic', 
            'function': lambda D: 1.0 + 3.0 * D**2
        },
        'logistic': {
            'name': 'Logistic', 
            'function': lambda D: 1.0 + 3.0 / (1 + np.exp(-12*(D-0.5)))
        }
    }
    
    hybrid_results = hybrid_rul_model(principal_strains, adjustment_models)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 