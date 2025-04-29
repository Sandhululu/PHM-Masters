#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial Hybrid RUL Model

This script implements a hybrid approach for RUL estimation that:
1. Analyzes multiple critical points across the entire strain field
2. Applies different damage progression models for each point
3. Combines the results to create a comprehensive spatial RUL map
4. Compares linear, feedback, and hybrid damage accumulation models

This provides more accurate RUL predictions by accounting for both
spatial variations in strain and non-linear damage progression.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

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

def apply_hybrid_damage_model(cycles, base_damage_rate, cycle_points, model_params):
    """Apply a hybrid damage model with specified parameters
    
    Args:
        cycles: Identified strain cycles
        base_damage_rate: Initial damage rate without adjustment
        cycle_points: Array of cycle points for projection
        model_params: Dictionary of model parameters
        
    Returns:
        dict: Damage progression data for this model
    """
    model_type = model_params['type']
    
    # Initialize arrays
    cumulative_damage = np.zeros_like(cycle_points)
    rul_values = np.zeros_like(cycle_points)
    damage_rates = np.zeros_like(cycle_points)
    
    # Set initial values
    initial_rul = 1 / base_damage_rate if base_damage_rate > 0 else float('inf')
    rul_values[0] = initial_rul
    damage_rates[0] = base_damage_rate
    
    # Apply different damage progression models
    if model_type == 'linear':
        # Linear model (constant damage rate)
        for i in range(1, len(cycle_points)):
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            damage_increment = base_damage_rate * cycle_increment
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            damage_rates[i] = base_damage_rate
            
            # Calculate RUL
            remaining_capacity = 1.0 - cumulative_damage[i]
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cumulative_damage[i:] = 1.0
                damage_rates[i:] = damage_rates[i]
                break
            else:
                rul_values[i] = remaining_capacity / base_damage_rate
    
    elif model_type == 'feedback':
        # Feedback model (damage rate increases with cumulative damage)
        feedback_factor = model_params.get('feedback_factor', 1.5)
        for i in range(1, len(cycle_points)):
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # Current damage rate increases with accumulation
            current_damage_rate = base_damage_rate * (1 + feedback_factor * cumulative_damage[i-1])
            damage_rates[i] = current_damage_rate
            
            # Calculate damage increment and update cumulative damage
            damage_increment = current_damage_rate * cycle_increment
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            
            # Calculate RUL
            remaining_capacity = 1.0 - cumulative_damage[i]
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cumulative_damage[i:] = 1.0
                damage_rates[i:] = damage_rates[i]
                break
            else:
                rul_values[i] = remaining_capacity / current_damage_rate
    
    elif model_type == 'hybrid_quadratic':
        # Hybrid quadratic model
        scale_factor = model_params.get('scale_factor', 2.0)
        for i in range(1, len(cycle_points)):
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # Quadratic damage rate progression
            current_damage_rate = base_damage_rate * (1 + scale_factor * cumulative_damage[i-1]**2)
            damage_rates[i] = current_damage_rate
            
            # Calculate damage increment and update cumulative damage
            damage_increment = current_damage_rate * cycle_increment
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            
            # Calculate RUL
            remaining_capacity = 1.0 - cumulative_damage[i]
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cumulative_damage[i:] = 1.0
                damage_rates[i:] = damage_rates[i]
                break
            else:
                rul_values[i] = remaining_capacity / current_damage_rate
    
    elif model_type == 'hybrid_logistic':
        # Hybrid logistic model (s-curve)
        scale_factor = model_params.get('scale_factor', 3.0)
        midpoint = model_params.get('midpoint', 0.5)
        steepness = model_params.get('steepness', 10.0)
        
        for i in range(1, len(cycle_points)):
            cycle_increment = cycle_points[i] - cycle_points[i-1]
            
            # S-curve damage rate progression using logistic function
            # 1 + scale_factor / (1 + exp(-steepness * (D - midpoint)))
            logistic_factor = scale_factor / (1 + np.exp(-steepness * (cumulative_damage[i-1] - midpoint)))
            current_damage_rate = base_damage_rate * (1 + logistic_factor)
            damage_rates[i] = current_damage_rate
            
            # Calculate damage increment and update cumulative damage
            damage_increment = current_damage_rate * cycle_increment
            cumulative_damage[i] = cumulative_damage[i-1] + damage_increment
            
            # Calculate RUL
            remaining_capacity = 1.0 - cumulative_damage[i]
            if remaining_capacity <= 0:
                rul_values[i:] = 0
                cumulative_damage[i:] = 1.0
                damage_rates[i:] = damage_rates[i]
                break
            else:
                rul_values[i] = remaining_capacity / current_damage_rate
    
    # Find failure point (where cumulative damage reaches 1.0)
    failure_index = np.argmax(cumulative_damage >= 1.0) if np.any(cumulative_damage >= 1.0) else -1
    
    if failure_index > 0:
        failure_cycles = cycle_points[failure_index]
    else:
        failure_cycles = None
    
    return {
        'cumulative_damage': cumulative_damage,
        'rul_values': rul_values,
        'damage_rates': damage_rates,
        'failure_index': failure_index,
        'failure_cycles': failure_cycles,
        'initial_rul': initial_rul
    }

def analyze_critical_points(strain_data, cycle_multiplier=10, sample_rate=4, threshold_percentile=90):
    """Analyze critical points with multiple damage models
    
    Args:
        strain_data: Dictionary containing strain data
        cycle_multiplier: Multiplier for projecting cycles
        sample_rate: Sample every Nth point to reduce computational load
        threshold_percentile: Only consider points above this strain percentile
        
    Returns:
        dict: Results for all analyzed points and models
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
    
    # Define damage models to apply
    damage_models = {
        'linear': {
            'name': 'Linear Model',
            'type': 'linear'
        },
        'feedback': {
            'name': 'Feedback Model',
            'type': 'feedback',
            'feedback_factor': 1.5
        },
        'hybrid_quadratic': {
            'name': 'Hybrid Quadratic',
            'type': 'hybrid_quadratic',
            'scale_factor': 3.0
        },
        'hybrid_logistic': {
            'name': 'Hybrid Logistic',
            'type': 'hybrid_logistic',
            'scale_factor': 3.0,
            'midpoint': 0.5,
            'steepness': 10.0
        }
    }
    
    # Store results
    all_results = {}
    critical_points = []
    damage_rates = []
    initial_ruls = []
    
    # Track summary metrics for each model
    model_summary = {model_key: {'failure_cycles': []} for model_key in damage_models.keys()}
    
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
                
                # Calculate base damage rate
                damage_per_cycle = counts / N_f_cycles
                total_damage = np.sum(damage_per_cycle)
                total_cycles = np.sum(counts)
                
                if total_cycles == 0 or total_damage == 0:
                    continue
                
                base_damage_rate = total_damage / total_cycles
                initial_rul = 1 / base_damage_rate if base_damage_rate > 0 else float('inf')
                
                # Skip points with unrealistically high initial RUL
                if initial_rul == float('inf') or initial_rul > 1e9:
                    continue
                
                # Create cycle points for projection based on initial RUL
                max_cycles = min(initial_rul * 3, 1e8)  # Cap for visualization
                cycle_points = np.linspace(0, max_cycles, 1000)
                
                # Store results for this point
                all_results[point_id] = {
                    'location': (r, c),
                    'max_strain': max_strain,
                    'cycles': len(cycles),
                    'base_damage_rate': base_damage_rate,
                    'initial_rul': initial_rul,
                    'model_results': {}
                }
                
                # Apply each damage model
                for model_key, model_params in damage_models.items():
                    model_result = apply_hybrid_damage_model(
                        cycles, base_damage_rate, cycle_points, model_params
                    )
                    
                    all_results[point_id]['model_results'][model_key] = model_result
                    
                    # Track failure cycles for summary
                    if model_result['failure_cycles'] is not None:
                        model_summary[model_key]['failure_cycles'].append(model_result['failure_cycles'])
                
                # Store metrics for combined analysis
                critical_points.append((r, c))
                damage_rates.append(base_damage_rate)
                initial_ruls.append(initial_rul)
                
                # Print progress periodically
                if len(all_results) % 10 == 0:
                    print(f"Processed {len(all_results)} critical points...")
                
            except Exception as e:
                print(f"Error processing point ({r},{c}): {e}")
    
    # Calculate summary statistics for each model
    for model_key in damage_models.keys():
        failure_cycles = model_summary[model_key]['failure_cycles']
        if failure_cycles:
            model_summary[model_key]['min_failure'] = np.min(failure_cycles)
            model_summary[model_key]['mean_failure'] = np.mean(failure_cycles)
            model_summary[model_key]['median_failure'] = np.median(failure_cycles)
            model_summary[model_key]['num_failures'] = len(failure_cycles)
    
    # Create output dictionary
    combined_results = {
        'critical_points': critical_points,
        'damage_rates': np.array(damage_rates),
        'initial_ruls': np.array(initial_ruls),
        'point_results': all_results,
        'num_points_analyzed': len(all_results),
        'model_summary': model_summary,
        'damage_models': damage_models
    }
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis complete!")
    print(f"Total critical points analyzed: {len(all_results)}")
    
    if initial_ruls:
        print(f"Minimum initial RUL: {np.min(initial_ruls):.1f} cycles")
        print(f"Median initial RUL: {np.median(initial_ruls):.1f} cycles")
        
        # Print failure statistics for each model
        print("\nModel Failure Statistics:")
        for model_key, summary in model_summary.items():
            if 'min_failure' in summary:
                print(f"  {damage_models[model_key]['name']}:")
                print(f"    Failures: {summary['num_failures']} points")
                print(f"    Min failure: {format_large_number(summary['min_failure'])} cycles")
                print(f"    Median failure: {format_large_number(summary['median_failure'])} cycles")
    
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    return combined_results

def create_model_comparison_visualization(results, time_per_cycle=70.6):
    """Create visualizations comparing different damage models
    
    Args:
        results: Combined results from critical point analysis
        time_per_cycle: Time per cycle in seconds
    """
    print("\nCreating model comparison visualization...")
    
    damage_models = results['damage_models']
    model_summary = results['model_summary']
    
    if results['num_points_analyzed'] == 0:
        print("No points to visualize")
        return
    
    # Set up 2x2 grid of plots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: RUL Comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot 2: Damage Accumulation Comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot 3: Damage Rate Comparison (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot 4: Failure Distribution (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Use a representative critical point (one with the median initial RUL)
    initial_ruls = results['initial_ruls']
    median_rul_index = np.argsort(initial_ruls)[len(initial_ruls)//2]
    median_rul = initial_ruls[median_rul_index]
    
    # Find the point with closest RUL to median
    representative_point = None
    min_diff = float('inf')
    
    for point_id, point_data in results['point_results'].items():
        diff = abs(point_data['initial_rul'] - median_rul)
        if diff < min_diff:
            min_diff = diff
            representative_point = point_id
    
    if representative_point is None:
        print("No representative point found")
        return
    
    # Get the representative point data
    rep_point_data = results['point_results'][representative_point]
    rep_location = rep_point_data['location']
    rep_initial_rul = rep_point_data['initial_rul']
    
    # Colors for different models
    model_colors = {
        'linear': '#1f77b4',       # Blue
        'feedback': '#ff7f0e',     # Orange
        'hybrid_quadratic': '#2ca02c',  # Green
        'hybrid_logistic': '#d62728'    # Red
    }
    
    # Plot each model for the representative point
    seconds_per_year = 365.25 * 24 * 60 * 60
    
    for model_key, model_params in damage_models.items():
        model_name = model_params['name']
        color = model_colors.get(model_key, 'black')
        
        model_result = rep_point_data['model_results'][model_key]
        cycle_points = np.linspace(0, rep_initial_rul * 3, 1000)
        
        # Calculate time in years
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Plot RUL percentage vs time (Plot 1)
        rul_percentage = (model_result['rul_values'] / rep_initial_rul) * 100
        ax1.plot(years, rul_percentage, color=color, linewidth=2.5, label=model_name)
        
        # Plot damage accumulation vs time (Plot 2)
        ax2.plot(years, model_result['cumulative_damage'], color=color, linewidth=2.5, label=model_name)
        
        # Plot damage rate (normalized) vs time (Plot 3)
        normalized_rates = model_result['damage_rates'] / model_result['damage_rates'][0]
        ax3.plot(years, normalized_rates, color=color, linewidth=2.5, label=model_name)
    
    # Configure RUL plot
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Remaining Useful Life (%)', fontsize=12)
    ax1.set_title(f'RUL vs Time at Point {rep_location}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='best')
    ax1.set_ylim([0, 100])
    
    # Add text with initial RUL
    ax1.text(0.05, 0.05, f"Initial RUL: {format_large_number(rep_initial_rul)} cycles",
            transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Configure damage accumulation plot
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Cumulative Damage', fontsize=12)
    ax2.set_title('Damage Accumulation Comparison', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='best')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
    
    # Configure damage rate plot
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Normalized Damage Rate', fontsize=12)
    ax3.set_title('Damage Rate Evolution', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=10, loc='best')
    
    # Plot failure distribution (Plot 4)
    failure_data = []
    model_labels = []
    
    for model_key, model_params in damage_models.items():
        if 'failure_cycles' in model_summary[model_key] and model_summary[model_key]['failure_cycles']:
            # Convert to years
            failure_years = np.array(model_summary[model_key]['failure_cycles']) * time_per_cycle / seconds_per_year
            failure_data.append(failure_years)
            model_labels.append(model_params['name'])
    
    if failure_data:
        # Create violin plot for failure distributions
        violin_parts = ax4.violinplot(failure_data, showmeans=True, showmedians=True)
        
        # Customize violins with model colors
        for i, model_key in enumerate(damage_models.keys()):
            if i < len(violin_parts['bodies']):
                violin_parts['bodies'][i].set_facecolor(model_colors.get(model_key, 'black'))
                violin_parts['bodies'][i].set_alpha(0.7)
        
        # Configure violin plot
        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('Time to Failure (years)', fontsize=12)
        ax4.set_title('Failure Time Distribution by Model', fontsize=14)
        ax4.set_xticks(range(1, len(model_labels) + 1))
        ax4.set_xticklabels(model_labels, rotation=15)
        ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
    else:
        ax4.text(0.5, 0.5, "No failure data available", 
                 transform=ax4.transAxes, fontsize=12, 
                 horizontalalignment='center', verticalalignment='center')
    
    # Add overall title
    plt.suptitle('Hybrid Damage Model Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('spatial_hybrid_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('spatial_hybrid_model_comparison.svg', bbox_inches='tight', dpi=300)
    
    print("Saved model comparison visualization")

def create_spatial_comparison_visualization(results, strain_data):
    """Create spatial visualization of different damage models
    
    Args:
        results: Combined results from critical point analysis
        strain_data: Original strain data dictionary
    """
    print("\nCreating spatial comparison visualization...")
    
    major_principal_strain = strain_data['major_principal_strain']
    time_dim, rows, cols = major_principal_strain.shape
    
    # Only proceed if we have results
    if results['num_points_analyzed'] == 0:
        print("No points to visualize")
        return
    
    # Define the models to visualize
    models_to_visualize = ['linear', 'feedback', 'hybrid_quadratic']
    
    # Set up figure with multiple rows
    fig, axes = plt.subplots(len(models_to_visualize), 1, figsize=(16, 5 * len(models_to_visualize)))
    
    # For single model case, convert to array
    if len(models_to_visualize) == 1:
        axes = [axes]
    
    # For each model, create a spatial visualization of failure times
    for i, model_key in enumerate(models_to_visualize):
        ax = axes[i]
        model_name = results['damage_models'][model_key]['name']
        
        # Create a failure time map
        failure_map = np.zeros((rows, cols)) + np.nan
        
        # Fill in values from analyzed points
        for point_id, point_data in results['point_results'].items():
            r, c = point_data['location']
            
            # Get failure time for this model and point
            if model_key in point_data['model_results']:
                model_result = point_data['model_results'][model_key]
                
                if model_result['failure_cycles'] is not None:
                    failure_map[r, c] = model_result['failure_cycles']
        
        # Plot failure time map
        cmap = plt.cm.plasma_r
        norm = mcolors.LogNorm()
        
        im = ax.imshow(failure_map, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'Failure Time Map - {model_name}', fontsize=14)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Cycles to Failure (log scale)')
    
    plt.tight_layout()
    plt.savefig('spatial_failure_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('spatial_failure_comparison.svg', bbox_inches='tight', dpi=300)
    
    print("Saved spatial failure comparison visualization")

def main():
    """Run the spatial hybrid RUL model analysis"""
    print("\n========== SPATIAL HYBRID RUL MODEL ANALYSIS ==========\n")
    
    # Step 1: Load strain data
    strain_data = load_strain_data()
    
    # Step 2: Analyze critical points with multiple damage models
    combined_results = analyze_critical_points(
        strain_data,
        cycle_multiplier=10,
        sample_rate=4,
        threshold_percentile=90
    )
    
    # Step 3: Create visualizations
    if combined_results['num_points_analyzed'] > 0:
        create_model_comparison_visualization(combined_results)
        create_spatial_comparison_visualization(combined_results, strain_data)
    
    print("\nSpatial hybrid RUL analysis complete!")

if __name__ == "__main__":
    main() 