#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feedback RUL Calculator

This script implements a Remaining Useful Life (RUL) estimation using
a feedback damage model where damage rate increases as damage accumulates.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original modules
from data_loader import load_all_data, print_statistical_summary
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue, estimate_fatigue_life_feedback
from plotter import plot_feedback_rul_estimation

def feedback_rul_calculator(feedback_factor=3.0, cycle_multiplier=50, time_per_cycle=70.8):
    """Run the feedback RUL model on the maximum principal strain location
    
    Args:
        feedback_factor: Factor controlling damage acceleration (default: 3.0)
        cycle_multiplier: Multiplier for cycles (default: 50)
        time_per_cycle: Time per cycle in seconds (default: 70.8)
    """
    print(f"\n===== FEEDBACK RUL MODEL (Factor: {feedback_factor}) =====\n")
    
    # 1. Load data
    print("Loading strain data...")
    data = load_all_data()
    
    # Extract strain data
    if isinstance(data, dict):
        ThermalStrain = data['thermal_strain']
        DICExx = data['exx']
        DICEyy = data['eyy']
        time_points = data['time_points']
    else:
        _, DICExx, _, DICEyy, ThermalStrain, time_points, _, _, _, _ = data
    
    # Print dimensions
    print(f"Strain data dimensions: {ThermalStrain.shape}")
    print(f"Time points: {len(time_points)} (from {time_points[0]} to {time_points[-1]} seconds)")
    
    # 2. Calculate principal strains
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # 3. Find maximum principal strain location
    max_principal_loc = np.unravel_index(np.nanargmax(major_principal_strain), major_principal_strain.shape)[1:]
    print(f"Maximum principal strain location: {max_principal_loc}")
    
    # 4. Identify cycles from strain data
    print("\nIdentifying cycles in strain data...")
    strain_signal = major_principal_strain[:, max_principal_loc[0], max_principal_loc[1]]
    cycles_max_strain = identify_cycles(strain_signal)
    
    if len(cycles_max_strain) == 0:
        print("No cycles identified. Aborting analysis.")
        return
        
    print(f"Number of cycles identified: {len(cycles_max_strain)}")
    
    # 5. Analyze fatigue for these cycles
    print("\nAnalyzing fatigue for identified cycles...")
    fatigue_max_strain = analyze_fatigue(cycles_max_strain)
    
    # Extract key data
    ranges = fatigue_max_strain.get('cycles', np.array([]))
    counts = fatigue_max_strain.get('counts', np.array([]))
    N_f_cycles = fatigue_max_strain.get('N_f_cycles', np.array([]))
    
    print(f"Damage analysis complete.")
    print(f"Max strain amplitude: {np.max(ranges/2) if len(ranges) > 0 else 'N/A'}")
    print(f"Min cycles to failure: {np.min(N_f_cycles) if len(N_f_cycles) > 0 else 'N/A'}")
    
    # 6. Run feedback RUL model
    print(f"\nRunning feedback RUL model with feedback factor {feedback_factor}...")
    rul_values, cycle_points, damage_values = estimate_fatigue_life_feedback(
        fatigue_max_strain, 
        cycle_multiplier=cycle_multiplier,
        feedback_factor=feedback_factor,
        projection_cycles=None  # Auto-determine projection cycles
    )
    
    # 7. Plot the results
    print("\nCreating visualization...")
    plot_feedback_rul_estimation(
        rul_values, 
        cycle_points, 
        damage_values,
        max_principal_loc, 
        time_per_cycle=time_per_cycle,
        feedback_factor=feedback_factor
    )
    
    print("\nFeedback RUL analysis complete!")
    return rul_values, cycle_points, damage_values, max_principal_loc

def compare_feedback_factors():
    """Compare different feedback factors on the same data"""
    print("\n===== COMPARING DIFFERENT FEEDBACK FACTORS =====\n")
    
    # Define feedback factors to compare
    feedback_factors = [0.0, 1.0, 3.0, 5.0]
    cycle_multiplier = 50
    
    # Store results for plotting
    results = {}
    
    # Run analysis for each feedback factor
    for factor in feedback_factors:
        print(f"\nRunning analysis with feedback factor {factor}...")
        rul_values, cycle_points, damage_values, point_loc = feedback_rul_calculator(
            feedback_factor=factor,
            cycle_multiplier=cycle_multiplier
        )
        
        # Store results
        results[factor] = {
            'rul_values': rul_values,
            'cycle_points': cycle_points,
            'damage_values': damage_values
        }
    
    # Create comparison plots
    create_factor_comparison(results, feedback_factors, point_loc)
    
    print("\nComparison complete!")

def create_factor_comparison(results, feedback_factors, point_loc, time_per_cycle=70.8):
    """Create side-by-side comparison of different feedback factors
    
    Args:
        results: Dictionary of results for each feedback factor
        feedback_factors: List of feedback factors
        point_loc: Location of the analyzed point
        time_per_cycle: Time per cycle in seconds
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for different factors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Seconds in a year for time conversion
    seconds_per_year = 365.25 * 24 * 60 * 60
    
    # 1. Left: RUL Percentage vs Years
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Remaining Useful Life (%)", fontsize=12)
    ax1.set_title("RUL Percentage vs Time", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 105)
    
    # 2. Right: Damage Accumulation vs Cycles
    ax2.set_xlabel("Cycles", fontsize=12)
    ax2.set_ylabel("Cumulative Damage", fontsize=12)
    ax2.set_title("Damage Accumulation", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 1.1)
    
    # Add failure threshold line to damage plot
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label="Failure Threshold")
    
    # Common x-axis limit for cycles
    max_cycles = 0
    
    # Plot data for each feedback factor
    for i, factor in enumerate(feedback_factors):
        if factor not in results:
            continue
            
        # Get data
        rul_values = results[factor]['rul_values']
        cycle_points = results[factor]['cycle_points']
        damage_values = results[factor]['damage_values']
        
        # Update max cycles
        max_cycles = max(max_cycles, cycle_points[-1])
        
        # Convert to years for RUL plot
        years = cycle_points * time_per_cycle / seconds_per_year
        
        # Convert RUL to percentage
        initial_rul = rul_values[0]
        rul_percentage = (rul_values / initial_rul) * 100
        
        # Color for this factor
        color = colors[i % len(colors)]
        
        # Determine label based on factor
        if factor == 0.0:
            label = "Linear (No Feedback)"
        else:
            # Find failure point if it exists
            failure_index = np.argmin(rul_values) if 0 in rul_values else -1
            if failure_index >= 0:
                failure_cycles = cycle_points[failure_index]
                label = f"Factor {factor} (Failure: {failure_cycles:.0f} cycles)"
            else:
                label = f"Feedback Factor {factor}"
        
        # Plot RUL percentage
        ax1.plot(years, rul_percentage, '-', color=color, linewidth=2, label=label)
        
        # Plot damage accumulation
        ax2.plot(cycle_points, damage_values, '-', color=color, linewidth=2, label=label)
        
        # Mark failure points if they exist
        if 0 in rul_values:
            failure_index = np.argmin(rul_values)
            failure_cycles = cycle_points[failure_index]
            failure_years = years[failure_index]
            
            ax1.scatter([failure_years], [0], color=color, s=80, zorder=10)
            ax2.scatter([failure_cycles], [1.0], color=color, s=80, zorder=10)
    
    # Set x-axis limits
    ax2.set_xlim(0, max_cycles * 1.05)
    
    # Find maximum years for RUL plot
    max_years = max_cycles * time_per_cycle / seconds_per_year
    ax1.set_xlim(0, max_years * 1.05)
    
    # Add legends
    ax1.legend(loc='upper right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    
    # Add percentage formatter for RUL plot
    def percentage_formatter(x, pos):
        return f"{x:.0f}%"
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(percentage_formatter))
    
    # Add title
    row, col = point_loc
    title = f"Feedback RUL Model Comparison at Point ({row},{col})"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(os.getcwd(), 'feedback_factors_comparison.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), 'feedback_factors_comparison.png'), bbox_inches='tight', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    print("\n===== FEEDBACK RUL CALCULATOR =====\n")
    
    # Run with default parameters (single feedback factor)
    feedback_rul_calculator(feedback_factor=3.0, cycle_multiplier=50)
    
    # Comment out the comparison to avoid segmentation fault
    # compare_feedback_factors()
    
    print("\nAnalysis complete!") 