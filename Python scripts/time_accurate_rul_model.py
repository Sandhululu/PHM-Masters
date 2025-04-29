#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-Accurate RUL Model

This script creates a more accurate RUL model and visualization by:
1. Using the extracted time series data from rainflow cycles
2. Calculating the actual time elapsed for each cycle
3. Projecting RUL based on real cycle times instead of a fixed time_per_cycle value

@author: Based on work by Jayron Sandhu
"""

import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains
from fatigue_analysis import identify_cycles, analyze_fatigue

def main():
    """Run time-accurate RUL model"""
    print("\n=== Time-Accurate RUL Model ===\n")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Time-Accurate RUL Model for Fatigue Life Estimation')
    parser.add_argument('--cycles', type=int, default=5000, 
                        help='Cycle multiplier for extending the simulation (default: 5000)')
    parser.add_argument('--feedback', type=float, default=3.0, 
                        help='Feedback factor for damage acceleration (default: 3.0)')
    parser.add_argument('--max-cycles', type=int, default=10000, 
                        help='Maximum number of cycles to simulate (default: 10000)')
    parser.add_argument('--quiet', action='store_true', 
                        help='Suppress debug and progress output')
    args = parser.parse_args()
    
    # Step 1: First check if we already have extracted cycle data
    cycle_time_series_path = os.path.join(os.getcwd(), "cycle_time_series.csv")
    
    if os.path.exists(cycle_time_series_path):
        print(f"Loading existing cycle time series data from: {cycle_time_series_path}")
        cycles_df = pd.read_csv(cycle_time_series_path)
        
        # Calculate fatigue properties for these cycles
        cycles_with_damage = calculate_fatigue_properties(cycles_df)
        
        # Run time-accurate RUL analysis
        run_time_accurate_rul_analysis(
            cycles_with_damage,
            cycle_multiplier=args.cycles,
            feedback_factor=args.feedback,
            max_simulation_cycles=args.max_cycles,
            quiet=args.quiet
        )
    else:
        print("No cycle time series data found. You need to run extract_rainflow_time_series.py first.")
        print("Running that script now...")
        
        # Extract the time series data from rainflow cycles
        extract_cycle_time_series()
        
        if os.path.exists(cycle_time_series_path):
            print(f"Loading newly created cycle time series data")
            cycles_df = pd.read_csv(cycle_time_series_path)
            
            # Calculate fatigue properties for these cycles
            cycles_with_damage = calculate_fatigue_properties(cycles_df)
            
            # Run time-accurate RUL analysis
            run_time_accurate_rul_analysis(
                cycles_with_damage,
                cycle_multiplier=args.cycles,
                feedback_factor=args.feedback,
                max_simulation_cycles=args.max_cycles,
                quiet=args.quiet
            )
        else:
            print("Failed to create cycle time series data. Please run extract_rainflow_time_series.py manually.")
            return 1
    
    print("\nAnalysis complete!")
    return 0

def extract_cycle_time_series():
    """Run the extract_rainflow_time_series.py script"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_rainflow_time_series.py")
    
    try:
        # Run the script using the system interpreter
        import subprocess
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Successfully extracted cycle time series data.")
            print(result.stdout)
        else:
            print(f"Error running extract_rainflow_time_series.py: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")

def calculate_fatigue_properties(cycles_df):
    """Calculate fatigue properties for the cycles
    
    Args:
        cycles_df: DataFrame with cycle data from extract_rainflow_time_series.py
        
    Returns:
        DataFrame: Cycles with added fatigue properties (cycles to failure, damage, etc.)
    """
    print("\nCalculating fatigue properties for cycles...")
    
    # Material properties for fatigue analysis (from fatigue_analysis.py)
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c = -0.12, -0.7  # Basquin and Coffin-Manson exponents
    
    # Calculate cycles to failure for each strain range
    N_f_cycles = []
    
    for strain_range in cycles_df['Range'].values:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 10, 1000)
        
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))]
        N_f_cycles.append(min(N_f, 1e6))  # Cap at 1 million cycles
    
    # Add cycles to failure and damage to the dataframe
    cycles_df['N_f'] = N_f_cycles
    cycles_df['Damage'] = cycles_df['Count'] / cycles_df['N_f']
    
    # Calculate damage rate (damage per second)
    # For cycles with zero duration, use the average duration
    avg_duration = cycles_df.loc[cycles_df['Duration'] > 0, 'Duration'].mean()
    cycles_df['Damage Rate'] = cycles_df['Damage'] / cycles_df['Duration'].replace(0, avg_duration)
    
    # Calculate cumulative damage
    cycles_df = cycles_df.sort_values('Start Time')
    cycles_df['Cumulative Damage'] = cycles_df['Damage'].cumsum()
    
    # Calculate total elapsed time for each cycle
    cycles_df['Elapsed Time'] = cycles_df['End Time']
    
    print(f"Calculated fatigue properties for {len(cycles_df)} cycles")
    print(f"Average cycles to failure (N_f): {cycles_df['N_f'].mean():.1f}")
    print(f"Total damage accumulated: {cycles_df['Damage'].sum():.6f}")
    print(f"Average damage rate: {cycles_df['Damage Rate'].mean():.6e} damage/second")
    
    return cycles_df

def run_time_accurate_rul_analysis(cycles_df, cycle_multiplier=5000, feedback_factor=3.0, max_simulation_cycles=10000, quiet=False):
    """Run time-accurate RUL analysis with the extracted cycle times
    
    Args:
        cycles_df: DataFrame with cycle data and fatigue properties
        cycle_multiplier: Multiplier for projecting future cycles (default: 5000)
        feedback_factor: Feedback factor for damage acceleration model (default: 3.0)
        max_simulation_cycles: Maximum number of cycles to simulate (default: 10000)
        quiet: Suppress debug and progress output (default: False)
    """
    print(f"\nRunning time-accurate RUL analysis (multiplier: {cycle_multiplier}, feedback: {feedback_factor}, max cycles: {max_simulation_cycles})...")
    
    # Step 1: Sort cycles by start time
    cycles_df = cycles_df.sort_values('Start Time')
    
    # Step 2: Calculate total damage and average damage rate
    total_damage = cycles_df['Damage'].sum()
    if total_damage == 0:
        print("No damage detected in cycles. Cannot calculate RUL.")
        return
    
    # Calculate proper time-weighted damage rate (damage per second)
    # Filter out zero-duration cycles for damage rate calculation
    valid_cycles = cycles_df[cycles_df['Duration'] > 0]
    if len(valid_cycles) == 0:
        print("No valid cycles with duration > 0. Cannot calculate time-accurate damage rate.")
        return
    
    # Calculate average damage rate (damage per second)
    total_duration = valid_cycles['Duration'].sum()
    time_weighted_damage_rate = valid_cycles['Damage'].sum() / total_duration
    
    # Calculate initial RUL (in seconds)
    initial_rul_seconds = (1.0 - 0) / time_weighted_damage_rate if time_weighted_damage_rate > 0 else float('inf')
    
    print(f"Initial analysis:")
    print(f"  Total cycles analyzed: {len(cycles_df)}")
    print(f"  Total elapsed time: {cycles_df['End Time'].max():.2f} seconds")
    print(f"  Total damage accumulated: {total_damage:.6f}")
    print(f"  Time-weighted damage rate: {time_weighted_damage_rate:.6e} damage/second")
    print(f"  Initial RUL (linear): {initial_rul_seconds:.2f} seconds")
    print(f"  Initial RUL (years): {initial_rul_seconds/(365.25*24*60*60):.2f} years")
    
    # Step 3: Project RUL using actual cycle times
    # Project damage and RUL into the future
    
    # Create a model of all analyzed cycles
    model_cycles = cycles_df.copy()
    
    # Extract cycle times, durations, and damages
    cycle_times = model_cycles['Start Time'].values
    cycle_durations = model_cycles['Duration'].values
    cycle_damages = model_cycles['Damage'].values
    
    # Create a DataFrame for the time-accurate RUL model
    # We'll project into the future by repeating observed cycles with the multiplier
    future_df = []
    
    # Repeat the observed time pattern multiple times
    max_time = cycle_times[-1] + cycle_durations[-1]
    
    # Track total damage and time
    cumulative_damage = total_damage
    current_time = max_time
    
    # Add initial point at time zero
    future_df.append({
        'Time': 0,
        'Cumulative Damage': 0,
        'RUL (Linear)': initial_rul_seconds,
        'RUL (Feedback)': initial_rul_seconds
    })
    
    # Add point at current maximum time
    future_df.append({
        'Time': max_time,
        'Cumulative Damage': cumulative_damage,
        'RUL (Linear)': (1.0 - cumulative_damage) / time_weighted_damage_rate,
        'RUL (Feedback)': (1.0 - cumulative_damage) / (time_weighted_damage_rate * (1 + feedback_factor * cumulative_damage))
    })
    
    failure_point_added = False  # Track if we explicitly added a failure point
    
    # Project into the future by repeating the observed pattern
    failure_time = None  # Store the exact time when failure occurs
    cycle_count = 0  # Track total number of cycles simulated
    max_cycles_reached = False  # Flag to break out of both loops
    current_cycle_time = max_time  # Initialize to avoid reference errors
    progress_interval = max(1, max_simulation_cycles // 10)  # Report progress every 10%
    
    for j in range(cycle_multiplier):
        for i in range(len(cycle_times)):
            # Track cycle count and check if we've reached the maximum
            cycle_count += 1
            
            # Report progress periodically
            if not quiet and cycle_count % progress_interval == 0:
                progress_pct = min(100, int(cycle_count / max_simulation_cycles * 100))
                print(f"  Progress: {progress_pct}% ({cycle_count}/{max_simulation_cycles} cycles, time: {current_cycle_time:.2f}s, damage: {cumulative_damage:.4f})")
            
            # Calculate projected time for this cycle
            cycle_time = current_time + (cycle_times[i] if i > 0 else 0)
            current_cycle_time = cycle_time + cycle_durations[i]
            
            # Check if we've reached the maximum cycles
            if cycle_count >= max_simulation_cycles:
                print(f"  *** Maximum simulation cycles ({max_simulation_cycles}) reached")
                print(f"  *** Simulated time: {current_cycle_time:.2f} seconds, cumulative damage: {cumulative_damage:.6f}")
                max_cycles_reached = True
                break
                
            # Calculate damage for this cycle (potentially accelerated due to feedback)
            if feedback_factor > 0:
                # Feedback model: damage increases as cumulative damage increases
                damage = cycle_damages[i] * (1 + feedback_factor * cumulative_damage)
            else:
                # Linear model: damage stays constant
                damage = cycle_damages[i]
            
            # Update cumulative damage
            cumulative_damage += damage
            
            # Calculate updated RUL
            linear_rul = (1.0 - cumulative_damage) / time_weighted_damage_rate
            feedback_rul = (1.0 - cumulative_damage) / (time_weighted_damage_rate * (1 + feedback_factor * cumulative_damage))
            
            # Add data point (after the cycle completes)
            future_df.append({
                'Time': current_cycle_time,
                'Cumulative Damage': cumulative_damage,
                'RUL (Linear)': max(0, linear_rul),
                'RUL (Feedback)': max(0, feedback_rul)
            })
            
            # If damage exceeds failure threshold, stop
            if cumulative_damage >= 1.0:
                failure_time = current_cycle_time
                print(f"  *** Failure threshold reached at cycle {j*len(cycle_times) + i}, time {failure_time:.2f} seconds")
                print(f"  *** Final cumulative damage: {cumulative_damage:.6f}")
                
                # Replace the last point with an exact failure point
                future_df[-1] = {
                    'Time': failure_time,
                    'Cumulative Damage': 1.0,
                    'RUL (Linear)': 0.0,
                    'RUL (Feedback)': 0.0
                }
                failure_point_added = True
                break
                
        # Update current time for the next repetition
        if cumulative_damage < 1.0 and not max_cycles_reached:
            current_time += max_time
        else:
            if cumulative_damage >= 1.0:
                print(f"  *** Breaking outer loop - cumulative damage: {cumulative_damage:.6f}")
            break
    
    # If no failure was detected but we're close, add a failure point
    if not failure_point_added and cumulative_damage > 0.9:
        # Estimate time to failure based on current trend
        if failure_time is None:  # Shouldn't happen, but just in case
            last_time = future_df[-1]['Time']
            max_damage = cumulative_damage
            failure_time = last_time + (1.0 - max_damage) * (last_time / max_damage)
        
        # Add failure point
        future_df.append({
            'Time': failure_time,
            'Cumulative Damage': 1.0,
            'RUL (Linear)': 0.0,
            'RUL (Feedback)': 0.0
        })
        print(f"  Added explicit failure point at time {failure_time:.2f} seconds")
        failure_point_added = True
    
    # If we completed all cycles without reaching failure, explain why
    if cumulative_damage < 1.0:
        print(f"\n  The simulation completed {cycle_multiplier} cycles without reaching failure.")
        print(f"  Final cumulative damage: {cumulative_damage:.6f}")
        print(f"  At this rate, failure would occur after approximately {1.0/cumulative_damage:.1f}x more cycles")
    
    # Convert to DataFrame
    rul_df = pd.DataFrame(future_df)
    
    # Debug: Check DataFrame for failure points
    if not quiet:
        print("\n  DEBUG: Checking DataFrame for failure points...")
        failure_points = rul_df[rul_df['Cumulative Damage'] >= 0.999]
        if not failure_points.empty:
            print(f"  Found {len(failure_points)} failure point(s) in DataFrame:")
            for idx, row in failure_points.iterrows():
                print(f"    Time: {row['Time']:.2f}, Damage: {row['Cumulative Damage']:.6f}, RUL: {row['RUL (Feedback)']:.6f}")
        else:
            print("  No failure points found in DataFrame!")
    
    # Filter out duplicate time points
    rul_df = rul_df.drop_duplicates(subset=['Time'])
    
    # Sort by time
    rul_df = rul_df.sort_values('Time')
    
    # Ensure the failure point has the exact desired values
    failure_mask = rul_df['Cumulative Damage'] >= 0.999
    if failure_mask.any():
        failure_index = failure_mask.idxmax()
        # Set precise values for clarity
        rul_df.at[failure_index, 'Cumulative Damage'] = 1.0
        rul_df.at[failure_index, 'RUL (Linear)'] = 0.0
        rul_df.at[failure_index, 'RUL (Feedback)'] = 0.0
        print(f"  Ensured failure point has damage=1.0 and RUL=0 at time {rul_df.at[failure_index, 'Time']:.2f} seconds")
        failure_time = rul_df.at[failure_index, 'Time']
    elif failure_time is not None:
        # If we had a failure time but it's not in the DataFrame, add it
        failure_point = {
            'Time': failure_time,
            'Cumulative Damage': 1.0,
            'RUL (Linear)': 0.0,
            'RUL (Feedback)': 0.0
        }
        rul_df = pd.concat([rul_df, pd.DataFrame([failure_point])], ignore_index=True)
        rul_df = rul_df.sort_values('Time')
        print(f"  Re-added failure point at time {failure_time:.2f} seconds")
    else:
        print(f"  No failure projected within {rul_df['Time'].max():.2f} seconds")
    
    # Step 4: Create visualizations
    plot_time_accurate_rul(rul_df, cycles_df, initial_rul_seconds, feedback_factor)
    
    # Step 5: Save the RUL data for future reference
    rul_df.to_csv('time_accurate_rul.csv', index=False)
    print(f"Saved time-accurate RUL data to time_accurate_rul.csv")
    
    return rul_df

def plot_time_accurate_rul(rul_df, cycles_df, initial_rul_seconds, feedback_factor):
    """Create visualization of time-accurate RUL model
    
    Args:
        rul_df: DataFrame with RUL model data
        cycles_df: DataFrame with original cycle data
        initial_rul_seconds: Initial RUL in seconds
        feedback_factor: Feedback factor used in the model
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert times to more readable units (days, years)
    seconds_per_day = 24 * 60 * 60
    seconds_per_year = 365.25 * seconds_per_day
    
    # Time units for x-axis
    x_unit = 'seconds'
    x_scale = 1.0
    x_max = rul_df['Time'].max()
    
    if x_max > 3 * seconds_per_year:
        x_unit = 'years'
        x_scale = seconds_per_year
    elif x_max > 30 * seconds_per_day:
        x_unit = 'days'
        x_scale = seconds_per_day
    elif x_max > 3600:
        x_unit = 'hours'
        x_scale = 3600
    
    # Convert times to chosen unit
    time_in_units = rul_df['Time'] / x_scale
    rul_linear_in_units = rul_df['RUL (Linear)'] / x_scale
    rul_feedback_in_units = rul_df['RUL (Feedback)'] / x_scale
    
    # Define colors
    linear_color = '#1f77b4'  # Blue
    feedback_color = '#ff7f0e'  # Orange
    damage_color = '#2ca02c'  # Green
    
    # Plot 1: Cumulative Damage vs Time
    ax1 = axes[0, 0]
    ax1.plot(time_in_units, rul_df['Cumulative Damage'], '-', color=damage_color, linewidth=2)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Failure Threshold')
    ax1.set_xlabel(f'Time ({x_unit})', fontsize=12)
    ax1.set_ylabel('Cumulative Damage', fontsize=12)
    ax1.set_title('Damage Accumulation vs Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight observed time period
    max_observed_time = cycles_df['End Time'].max() / x_scale
    ax1.axvline(x=max_observed_time, color='r', linestyle='--', alpha=0.5, label='End of Observed Data')
    
    # Plot 2: RUL (seconds) vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_in_units, rul_linear_in_units, '-', color=linear_color, linewidth=2, label='Linear Model')
    ax2.plot(time_in_units, rul_feedback_in_units, '-', color=feedback_color, linewidth=2, label=f'Feedback Model (Factor: {feedback_factor})')
    ax2.set_xlabel(f'Time ({x_unit})', fontsize=12)
    ax2.set_ylabel(f'RUL ({x_unit})', fontsize=12)
    ax2.set_title('Remaining Useful Life vs Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Highlight observed time period
    ax2.axvline(x=max_observed_time, color='r', linestyle='--', alpha=0.5, label='End of Observed Data')
    
    # Plot 3: RUL Percentage vs Time
    ax3 = axes[1, 0]
    rul_percentage_linear = rul_df['RUL (Linear)'] / initial_rul_seconds * 100
    rul_percentage_feedback = rul_df['RUL (Feedback)'] / initial_rul_seconds * 100
    
    ax3.plot(time_in_units, rul_percentage_linear, '-', color=linear_color, linewidth=2, label='Linear Model')
    ax3.plot(time_in_units, rul_percentage_feedback, '-', color=feedback_color, linewidth=2, label=f'Feedback Model (Factor: {feedback_factor})')
    ax3.set_xlabel(f'Time ({x_unit})', fontsize=12)
    ax3.set_ylabel('RUL (%)', fontsize=12)
    ax3.set_title('Remaining Useful Life Percentage vs Time', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    ax3.legend()
    
    # Highlight observed time period
    ax3.axvline(x=max_observed_time, color='r', linestyle='--', alpha=0.5, label='End of Observed Data')
    
    # Format y-axis as percentage
    def percentage_formatter(x, pos):
        return f'{x:.0f}%'
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(percentage_formatter))
    
    # Empty plot for the fourth panel
    ax4 = axes[1, 1]
    ax4.set_visible(False)
    
    # Add overall title
    plt.suptitle(f"Time-Accurate RUL Model (Feedback Factor: {feedback_factor})",
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig('time_accurate_rul_model.png', dpi=300, bbox_inches='tight')
    plt.savefig('time_accurate_rul_model.svg', bbox_inches='tight')
    print("Saved visualization to time_accurate_rul_model.png and time_accurate_rul_model.svg")

def plot_observed_cycles_detail(cycles_df):
    """Create detailed visualization of just the observed cycles
    
    Args:
        cycles_df: DataFrame with cycle data
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Sort cycles by time
    cycles_df = cycles_df.sort_values('Start Time')
    
    # Plot 1: Cycle properties over time
    ax1.plot(cycles_df['Start Time'], cycles_df['Range'], 'o-', markersize=4, alpha=0.7, label='Range')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Cycle Range', fontsize=12)
    ax1.set_title('Cycle Ranges vs Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for duration
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cycles_df['Start Time'], cycles_df['Duration'], 'r.-', markersize=4, alpha=0.5, label='Duration')
    ax1_twin.set_ylabel('Duration (seconds)', fontsize=12, color='r')
    ax1_twin.tick_params(axis='y', colors='r')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: Cumulative damage over time
    ax2.plot(cycles_df['End Time'], cycles_df['Cumulative Damage'], 'g-', linewidth=2)
    ax2.scatter(cycles_df['End Time'], cycles_df['Cumulative Damage'], c=cycles_df['Damage'], 
               cmap='viridis', s=30, alpha=0.7)
    
    # Format plot
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Cumulative Damage', fontsize=12)
    ax2.set_title('Observed Damage Accumulation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for damage
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(cycles_df['Damage'])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Damage per Cycle', fontsize=10)
    
    # Add damage rate trend line (if we have enough points)
    if len(cycles_df) > 5:
        # Add linear trend line to show damage rate
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            cycles_df['End Time'], cycles_df['Cumulative Damage'])
        
        x_trend = np.array([cycles_df['End Time'].min(), cycles_df['End Time'].max()])
        y_trend = slope * x_trend + intercept
        
        ax2.plot(x_trend, y_trend, 'r--', linewidth=1.5, 
                label=f'Trend: {slope:.2e} damage/second, R²={r_value**2:.2f}')
        ax2.legend()
    
    # Overall title
    plt.suptitle("Detailed Analysis of Observed Strain Cycles", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig('observed_cycles_detail.png', dpi=300, bbox_inches='tight')
    print("Saved detailed cycle visualization to observed_cycles_detail.png")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nScript completed in {time.time() - start_time:.2f} seconds") 