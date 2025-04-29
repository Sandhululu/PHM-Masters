#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved RUL Visualization Script

This script provides better visualization of the uncapped RUL analysis results,
using techniques like log scaling and multiple views to handle the extreme range of values.
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing functions but we'll only use the visualization part
from totally_uncapped_rul import load_strain_data, analyze_fatigue_uncapped, totally_uncapped_estimate_fatigue_life
from totally_uncapped_rul import calculate_rul_for_sample_points, format_large_number

def create_improved_rul_visualization(rul_curves, time_per_cycle=70.6):
    """Create improved visualizations of RUL curves that handle extreme value ranges
    
    Args:
        rul_curves: Dictionary of RUL curves for different points
        time_per_cycle: Time per cycle in seconds (default: 70.6)
    """
    # Create a figure with multiple subplots to handle different visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # Create a 2x2 grid of subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0))  # Linear scale - all points
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # Log scale - all points
    ax3 = plt.subplot2grid((2, 2), (1, 0))  # Linear scale - only million-range points
    ax4 = plt.subplot2grid((2, 2), (1, 1))  # Time-based view
    
    # Define colors for different points
    point_colors = {
        'Point 1': '#d62728',    # Red
        'Point 2': '#2ca02c',    # Green
        'Point 3': '#1f77b4',    # Blue
        'Point 4': '#9467bd',    # Purple
        'Point 5': '#ff7f0e'     # Orange
    }
    
    # Calculate statistics to display
    has_infinite = any(curve_data.get('initial_rul', 0) == float('inf') for curve_data in rul_curves.values())
    finite_initial_ruls = [curve_data.get('initial_rul', 0) for curve_data in rul_curves.values() 
                          if curve_data.get('initial_rul', 0) != float('inf')]
    
    # Separate points by magnitude for better visualization
    million_range_points = {}
    billion_range_points = {}
    trillion_range_points = {}
    
    for key, curve_data in rul_curves.items():
        initial_rul = curve_data.get('initial_rul', 0)
        if initial_rul != float('inf'):
            if initial_rul < 1e9:  # Less than a billion
                million_range_points[key] = curve_data
            elif initial_rul < 1e12:  # Less than a trillion
                billion_range_points[key] = curve_data
            else:  # Trillion or more
                trillion_range_points[key] = curve_data
    
    # Plot 1: Linear scale with all points (traditional view)
    plot_all_curves(ax1, rul_curves, point_colors, 'Linear Scale (All Points)')
    
    # Plot 2: Log scale with all points
    plot_all_curves(ax2, rul_curves, point_colors, 'Log Scale (All Points)', use_log_scale=True)
    
    # Plot 3: Linear scale with only million-range points
    if million_range_points:
        plot_all_curves(ax3, million_range_points, point_colors, 'Million-Range Points (Linear Scale)')
        # Also plot projected failure for these points
        plot_projected_failures(ax3, million_range_points, point_colors)
    else:
        ax3.text(0.5, 0.5, "No points in million-range", 
                ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Million-Range Points (Linear Scale)')
    
    # Plot 4: Time-based view (in hours)
    plot_time_based_view(ax4, rul_curves, point_colors, time_per_cycle)
    
    # Add an overall title
    plt.suptitle('Improved RUL Visualization with Multiple Views', fontsize=16, y=0.98)
    
    # Make the layout look better
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add an explanation text box at the bottom
    explanation = """Explanation of Views:
- Linear Scale (All Points): Shows all points on a linear scale - note the extreme range makes million-cycle points appear flat
- Log Scale (All Points): Log scale better shows the relative differences across orders of magnitude
- Million-Range Points: Zoomed in view showing only points with RUL < 1 billion cycles (with projections to failure)
- Time-Based View: Shows RUL in hours instead of cycles, focusing on more relevant timescales for engineering use"""
    
    fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the improved visualizations
    plt.savefig('improved_rul_visualization.svg', bbox_inches='tight', dpi=300)
    plt.savefig('improved_rul_visualization.png', bbox_inches='tight', dpi=300)
    print("Improved RUL visualization saved as 'improved_rul_visualization.svg/png'")

def plot_all_curves(ax, curves, colors, title, use_log_scale=False):
    """Plot all RUL curves on a single axis
    
    Args:
        ax: Matplotlib axis to plot on
        curves: Dictionary of curve data
        colors: Dictionary of colors for each curve
        title: Title for the plot
        use_log_scale: Whether to use log scale for y-axis
    """
    # Plot each curve
    for key, curve_data in curves.items():
        color = colors.get(key, '#000000')
        location = curve_data['location']
        initial_rul = curve_data.get('initial_rul', 0)
        
        # Format label suffix based on initial RUL
        if initial_rul == float('inf'):
            label_suffix = " (Infinite RUL)"
        elif initial_rul > 1e12:
            label_suffix = f" ({initial_rul/1e12:.2f} trillion cycles)"
        elif initial_rul > 1e9:
            label_suffix = f" ({initial_rul/1e9:.2f} billion cycles)"
        elif initial_rul > 1e6:
            label_suffix = f" ({initial_rul/1e6:.2f} million cycles)"
        else:
            label_suffix = f" ({initial_rul:.1f} cycles)"
        
        # Plot the RUL curve
        ax.plot(curve_data['cycles'], curve_data['rul'], '-', 
                color=color, linewidth=2.0, marker='o', markersize=5,
                label=f"{key} at {location}{label_suffix}")
    
    # Set axis labels and title
    ax.set_xlabel('Cycles Experienced', fontsize=12)
    ax.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Apply log scale if requested
    if use_log_scale:
        ax.set_yscale('log')
        
        # Add horizontal grid lines at decade intervals
        ax.yaxis.grid(True, which='major', linestyle='-', color='gray', alpha=0.3)
        ax.yaxis.grid(True, which='minor', linestyle=':', color='gray', alpha=0.2)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

def plot_projected_failures(ax, curves, colors):
    """Plot projected failure lines for curves
    
    Args:
        ax: Matplotlib axis to plot on
        curves: Dictionary of curve data
        colors: Dictionary of colors for each curve
    """
    for key, curve_data in curves.items():
        color = colors.get(key, '#000000')
        initial_rul = curve_data.get('initial_rul', 0)
        damage_rate = curve_data.get('damage_rate', 0)
        
        if initial_rul != float('inf') and damage_rate > 0:
            # Calculate total expected lifetime
            total_cycles_to_failure = initial_rul / damage_rate
            
            # Create extended x-axis to show full lifetime
            extended_cycles = np.linspace(0, total_cycles_to_failure, 1000)
            
            # Create extended y-axis values that go to zero
            extended_rul = np.maximum(initial_rul - damage_rate * extended_cycles, 0)
            
            # Plot projected failure curve
            ax.plot(extended_cycles, extended_rul, '--', color=color, linewidth=1.5, 
                    label=f"{key} - Projected failure ({total_cycles_to_failure/1e6:.2f} million cycles)")
    
    # Ensure the x-axis shows the full range
    max_x = max([curve_data.get('initial_rul', 0) / curve_data.get('damage_rate', 1) 
                for curve_data in curves.values() 
                if curve_data.get('initial_rul', 0) != float('inf') and curve_data.get('damage_rate', 0) > 0])
    
    ax.set_xlim(0, max_x * 1.1)
    
    # Update legend with new projected failure curves
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

def plot_time_based_view(ax, curves, colors, time_per_cycle=70.6):
    """Plot RUL curves in real time (hours)
    
    Args:
        ax: Matplotlib axis to plot on
        curves: Dictionary of curve data
        colors: Dictionary of colors for each curve
        time_per_cycle: Time per cycle in seconds
    """
    # Filter to only finite points
    finite_curves = {k: v for k, v in curves.items() if v.get('initial_rul', 0) != float('inf')}
    
    if not finite_curves:
        ax.text(0.5, 0.5, "No finite RUL points to display", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Time-Based View (Hours)', fontsize=14)
        return
    
    # Convert to hours based on cycle duration
    for key, curve_data in finite_curves.items():
        color = colors.get(key, '#000000')
        location = curve_data['location']
        initial_rul = curve_data.get('initial_rul', 0)
        damage_rate = curve_data.get('damage_rate', 0)
        
        # Convert cycles to hours
        cycles_hours = np.array(curve_data['cycles']) * time_per_cycle / 3600
        rul_hours = np.array(curve_data['rul']) * time_per_cycle / 3600
        
        # Format time for label
        if initial_rul * time_per_cycle / 3600 > 8760:  # More than a year
            years = initial_rul * time_per_cycle / 3600 / 8760
            time_suffix = f" ({years:.1f} years)"
        else:
            time_suffix = f" ({initial_rul * time_per_cycle / 3600:.1f} hours)"
        
        # Plot the time-based RUL curve
        ax.plot(cycles_hours, rul_hours, '-', 
                color=color, linewidth=2.0, marker='o', markersize=5,
                label=f"{key} at {location}{time_suffix}")
        
        # Add projection if damage rate is significant
        if damage_rate > 0:
            # Get total life in hours
            total_hours = initial_rul * time_per_cycle / 3600
            
            # Only show for reasonable timescales (less than 10 years)
            if total_hours < 87600:
                # Create extended hours
                extended_hours = np.linspace(0, total_hours, 1000)
                
                # Create extended y-axis values
                damage_per_hour = damage_rate * 3600 / time_per_cycle
                extended_rul_hours = np.maximum(total_hours - extended_hours, 0)
                
                # Plot projected failure
                ax.plot(extended_hours, extended_rul_hours, '--', color=color, linewidth=1.5,
                        label=f"{key} - Projected failure ({format_time(total_hours)})")
    
    # Set axis labels and title
    ax.set_xlabel('Time Experienced (hours)', fontsize=12)
    ax.set_ylabel('Remaining Useful Life (hours)', fontsize=12)
    ax.set_title('Time-Based View (Hours)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate reasonable axis limits
    max_initial_hours = max([v.get('initial_rul', 0) * time_per_cycle / 3600 
                           for v in finite_curves.values()])
    
    # For very large values, limit to something reasonable (1 year)
    if max_initial_hours > 8760:
        ax.set_ylim(0, 8760)
        ax.axhline(y=8760, color='black', linestyle='-.', alpha=0.5)
        ax.text(0, 8760, "1 year", va='bottom', ha='left', color='black', alpha=0.8)
    else:
        ax.set_ylim(0, max_initial_hours * 1.1)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

def format_time(hours):
    """Format time in hours to a more readable format"""
    if hours < 24:
        return f"{hours:.1f} hours"
    elif hours < 168:  # Less than a week
        return f"{hours/24:.1f} days"
    elif hours < 720:  # Less than a month
        return f"{hours/168:.1f} weeks"
    elif hours < 8760:  # Less than a year
        return f"{hours/720:.1f} months"
    else:
        return f"{hours/8760:.1f} years"

def main():
    """Create improved RUL visualizations using the existing analysis results"""
    print("\n========== CREATING IMPROVED RUL VISUALIZATIONS ==========\n")
    
    # Load strain data
    print("\nLoading strain data...")
    strain_data = load_strain_data()
    
    # Calculate RUL for sample points (reusing existing analysis)
    print("Calculating RUL for sample points...")
    rul_results = calculate_rul_for_sample_points(
        strain_data, 
        cycle_multiplier=50,
        time_per_cycle=70.6
    )
    
    # Create improved visualizations
    print("Creating improved visualizations...")
    create_improved_rul_visualization(rul_results, time_per_cycle=70.6)
    
    print("\nImproved RUL visualization complete!")

if __name__ == "__main__":
    main() 