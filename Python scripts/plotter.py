# -*- coding: utf-8 -*-
"""
Visualization module for RUL estimation from strain DIC measurement

@author: Jayron Sandhu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_initial_strain_analysis(ThermalStrain, DICExx, DICEyy, time_points, high_strain_points, max_strain):
    """Create and display the initial strain analysis plots
    
    Args:
        ThermalStrain, DICExx, DICEyy: Strain data arrays
        time_points: Array of time points
        high_strain_points: Locations of high strain
        max_strain: Maximum strain values
        
    Returns:
        tuple: Top 5 indices of high strain points for later use
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Original strain at (0,0)
    ax1.plot(time_points, ThermalStrain[:,0,0], 'b-', label='Thermal Strain (0,0)')
    ax1.plot(time_points, DICExx[:,0,0], 'r--', label='Exx Strain (0,0)')
    ax1.plot(time_points, DICEyy[:,0,0], 'g--', label='Eyy Strain (0,0)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Strain (ε)')  # Added strain symbol
    ax1.set_title('Strain vs Time at Point (0,0)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Points with highest strain
    top_5_indices = None
    if len(high_strain_points[0]) > 0:
        max_strain_values = max_strain[high_strain_points]
        top_5_indices = np.argsort(max_strain_values)[-5:]
        for idx in top_5_indices:
            row, col = high_strain_points[0][idx], high_strain_points[1][idx]
            ax2.plot(time_points, ThermalStrain[:,row,col], 
                    label=f'Point ({row},{col}), Max={max_strain[row,col]:.2e}')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Strain (ε)')  # Added strain symbol
    ax2.set_title('Strain vs Time at High Strain Points')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'strain_analysis.svg'), bbox_inches='tight')
    plt.show()
    
    return top_5_indices

def plot_stress_analysis(time_points, stress_xx, stress_yy, stress_von_mises, 
                         high_strain_points, top_5_indices):
    """Create and display stress analysis plots
    
    Args:
        time_points: Array of time points
        stress_xx, stress_yy, stress_von_mises: Stress data arrays
        high_strain_points: Locations of high strain
        top_5_indices: Indices of top 5 high strain points
    """
    fig_stress, axes_stress = plt.subplots(1, 2, figsize=(20, 10))  # One row, two columns for stress plots

    # Plot stress at point (0,0) over time
    axes_stress[0].plot(time_points, stress_xx[:, 0, 0]/1e6, 'r-', label='σxx')
    axes_stress[0].plot(time_points, stress_yy[:, 0, 0]/1e6, 'g-', label='σyy')
    axes_stress[0].plot(time_points, stress_von_mises[:, 0, 0]/1e6, 'b-', label='von Mises')
    axes_stress[0].set_xlabel('Time (seconds)', fontsize=12)
    axes_stress[0].set_ylabel('Stress (MPa)', fontsize=12, labelpad=0)
    axes_stress[0].set_title('Stress vs Time at Point (0,0)', fontsize=14)
    axes_stress[0].grid(True)
    axes_stress[0].legend(fontsize=10, loc='upper right')
    axes_stress[0].tick_params(axis='both', which='major', labelsize=10)

    # Plot von Mises stress for highest strain points
    if len(high_strain_points[0]) > 0 and top_5_indices is not None:
        for idx in top_5_indices:
            row, col = high_strain_points[0][idx], high_strain_points[1][idx]
            axes_stress[1].plot(time_points, stress_von_mises[:, row, col]/1e6, 
                          label=f'Pt({row},{col}), Max={np.nanmax(stress_von_mises[:, row, col])/1e6:.1f}MPa')
        axes_stress[1].set_xlabel('Time (seconds)', fontsize=12)
        axes_stress[1].set_ylabel('von Mises Stress (MPa)', fontsize=12, labelpad=0)
        axes_stress[1].set_title('von Mises Stress vs Time at High Strain Points', fontsize=14)
        axes_stress[1].grid(True)
        axes_stress[1].legend(fontsize=9, loc='upper right', framealpha=0.7)
        axes_stress[1].tick_params(axis='both', which='major', labelsize=10)

    # Adjust spacing between plots
    fig_stress.subplots_adjust(wspace=0.5)
    fig_stress.suptitle('Stress Analysis for Tungsten Component', fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(os.getcwd(), 'stress_analysis.svg'), bbox_inches='tight', dpi=300)
    plt.show()

def plot_strain_analysis(time_points, major_principal_strain, minor_principal_strain, max_shear_strain,
                          max_principal_loc, min_principal_loc, max_shear_loc):
    """Create and display principal strain analysis plots
    
    Args:
        time_points: Array of time points
        major_principal_strain, minor_principal_strain, max_shear_strain: Strain data arrays
        max_principal_loc, min_principal_loc, max_shear_loc: Locations of extreme strain values
    """
    fig_strain, axes_strain = plt.subplots(1, 2, figsize=(20, 10))  # One row, two columns for strain plots

    # Plot principal strains at point (0,0) over time
    axes_strain[0].plot(time_points, major_principal_strain[:, 0, 0], 'r-', label='Major Principal')
    axes_strain[0].plot(time_points, minor_principal_strain[:, 0, 0], 'g-', label='Minor Principal')
    axes_strain[0].set_xlabel('Time (seconds)', fontsize=12)
    axes_strain[0].set_ylabel('Strain (ε)', fontsize=12, labelpad=10)  # Added strain symbol
    axes_strain[0].set_title('Principal Strains vs Time at Point (0,0)', fontsize=14)
    axes_strain[0].grid(True)
    axes_strain[0].legend(fontsize=10, loc='upper right')
    axes_strain[0].tick_params(axis='both', which='major', labelsize=10)

    # Plot principal strains at key points
    # Point with maximum principal strain
    row, col = max_principal_loc
    axes_strain[1].plot(time_points, major_principal_strain[:, row, col], 'r-', 
                    label=f'Major ({row},{col}), Max={np.nanmax(major_principal_strain[:, row, col]):.2e}')
    # Point with minimum principal strain (most negative)
    row, col = min_principal_loc
    axes_strain[1].plot(time_points, minor_principal_strain[:, row, col], 'g-', 
                    label=f'Minor ({row},{col}), Min={np.nanmin(minor_principal_strain[:, row, col]):.2e}')
    axes_strain[1].set_xlabel('Time (seconds)', fontsize=12)
    axes_strain[1].set_ylabel('Strain (ε)', fontsize=12, labelpad=10)  # Added strain symbol
    axes_strain[1].set_title('Principal Strains vs Time at Key Points', fontsize=14)
    axes_strain[1].grid(True)
    axes_strain[1].legend(fontsize=9, loc='upper right', framealpha=0.7)
    axes_strain[1].tick_params(axis='both', which='major', labelsize=10)

    # Adjust spacing between plots
    fig_strain.subplots_adjust(wspace=0.5)  # Increase horizontal spacing between plots
    fig_strain.suptitle('Principal Strain Analysis for Tungsten Component', fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(os.getcwd(), 'strain_analysis.svg'), bbox_inches='tight', dpi=300)
    plt.show()

def plot_fatigue_analysis_signals(time_points, 
                                major_principal_strain, max_shear_strain, 
                                max_principal_loc, max_shear_loc):
    """Create and display fatigue analysis plots for strain signals
    
    Args:
        time_points: Array of time points
        major_principal_strain, max_shear_strain: Strain data arrays
        max_principal_loc, max_shear_loc: Locations of extreme strain values
        
    Returns:
        tuple: Figure and axes for further customization
    """
    # Setup the plot - changed to single plot
    fig_fatigue, ax_fatigue = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot max principal strain location
    row, col = max_principal_loc
    signal_clean = np.copy(major_principal_strain[:, row, col])
    mask = ~np.isnan(signal_clean)
    if np.any(mask):
        indices = np.arange(len(signal_clean))
        signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
        ax_fatigue.plot(time_points, signal_clean, '-', 
                        label=f"Max Principal Strain Location ({row},{col})")
        
    # Configure the principal strain plot
    ax_fatigue.set_xlabel("Time (seconds)", fontsize=13)
    ax_fatigue.set_ylabel("Strain (ε)", fontsize=13)
    ax_fatigue.set_title(f"Strain History - Max Principal Strain Location ({row},{col})", 
                        fontsize=15, pad=10)
    ax_fatigue.tick_params(axis='both', which='major', labelsize=12)
    ax_fatigue.grid(True, alpha=0.7)
    
    # Removed max shear strain plot section
    
    return fig_fatigue, ax_fatigue

def configure_rul_plot_axes(ax_rul, rul_values):
    """Configure the axes for RUL plot with appropriate scales and formatting
    
    Args:
        ax_rul: Matplotlib axis for the RUL plot
        rul_values: RUL data
    """
    # Format function for y-axis labels (percentage)
    def format_func(value, pos):
        return f'{value:.0f}%'
    
    # Set y-axis limits for percentage life (0-100%)
    ax_rul.set_ylim([0, 100])
    ax_rul.yaxis.set_major_locator(plt.MultipleLocator(10))  # 10% intervals
    ax_rul.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Add grid with light gray dotted lines
    ax_rul.grid(True, linestyle='--', color='lightgray', alpha=0.7)

def plot_rul_estimation(rul_max_strain, cycles_max_strain_plot, 
                       rul_max_shear, cycles_max_shear_plot,
                       max_principal_loc, max_shear_loc, time_per_cycle=70.8):  # Updated to 70.8 seconds
    """Create and display RUL estimation plots
    
    Args:
        rul_max_strain, cycles_max_strain_plot: RUL data for principal strain
        rul_max_shear, cycles_max_shear_plot: RUL data for shear strain (not used)
        max_principal_loc, max_shear_loc: Locations of strain values
        time_per_cycle: Time duration for each cycle in seconds (default: 70.8)
    """
    # Create the RUL figure - changed to single plot
    fig_rul, ax_rul = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot RUL for maximum principal strain location
    if rul_max_strain is not None and cycles_max_strain_plot is not None:
        row, col = max_principal_loc
        clean_location = f"Max Principal Strain Location ({row},{col})"
        
        # Convert cycles to years
        seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
        years = cycles_max_strain_plot * time_per_cycle / seconds_per_year
        
        # Convert RUL to percentage of initial life
        initial_rul = rul_max_strain[0]
        rul_percentage = (rul_max_strain / initial_rul) * 100
        
        # Plot RUL % vs Years with markers
        ax_rul.plot(years, rul_percentage, '-', color='#1f77b4', linewidth=2.5)
        marker_indices = np.linspace(0, len(years)-1, min(10, len(years))).astype(int)
        ax_rul.plot(years[marker_indices], rul_percentage[marker_indices], 'o', color='#1f77b4', markersize=7)
        
        # Add text information boxes
        final_rul_percentage = rul_percentage[-1]
        final_percentage_used = 100 - final_rul_percentage
        
        def format_large_number(num):
            if num > 1_000_000: return f"{num/1_000_000:.1f}M"
            elif num > 1_000: return f"{num/1_000:.1f}k"
            else: return f"{num:.1f}"
        
        # Format time duration
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
            
        ax_rul.text(0.97, 0.97, 
                   f"Initial RUL: {format_large_number(initial_rul)} cycles",
                   transform=ax_rul.transAxes, fontsize=12,
                   horizontalalignment='right', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul.text(0.97, 0.89, 
                   f"Final RUL: {format_large_number(rul_max_strain[-1])} cycles",
                   transform=ax_rul.transAxes, fontsize=12,
                   horizontalalignment='right', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul.text(0.97, 0.81, 
                   f"Life used: {final_percentage_used:.1f}%",
                   transform=ax_rul.transAxes, fontsize=12,
                   horizontalalignment='right', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Add time-based information
        if time_per_cycle > 0:
            max_years = years[-1]
            ax_rul.text(0.97, 0.73, 
                       f"Elapsed time: {max_years:.2f} years",
                       transform=ax_rul.transAxes, fontsize=12,
                       horizontalalignment='right', verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Set plot formatting
        ax_rul.set_xlabel("Time (years)", fontsize=14)
        ax_rul.set_ylabel("Remaining Useful Life (%)", fontsize=14, labelpad=10)
        ax_rul.set_title(f"RUL Estimation - {clean_location}", fontsize=16, pad=10)
        ax_rul.grid(True, linestyle='--', alpha=0.7)
        ax_rul.tick_params(axis='both', which='major', labelsize=12)
    
    # Removed max shear strain plot section
    
    # Configure y-axis for plot
    configure_rul_plot_axes(ax_rul, rul_max_strain)
    
    # Add overall title with time information
    if time_per_cycle > 0 and cycles_max_strain_plot is not None:
        max_cycles = np.max(cycles_max_strain_plot) if cycles_max_strain_plot is not None else 0
        max_years = max_cycles * time_per_cycle / seconds_per_year
        fig_rul.suptitle(f'Tungsten Component RUL Estimation (Max time: {max_years:.2f} years)', 
                         fontsize=18, fontweight='bold', y=0.98)
    else:
        fig_rul.suptitle('Tungsten Component Remaining Useful Life Estimation', 
                         fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(os.getcwd(), 'rul_estimation.svg'), bbox_inches='tight', dpi=300)
    plt.show() 

def format_large_number(num):
    """Format large numbers with k/M/B suffixes for readability
    
    Args:
        num: Number to format
    
    Returns:
        str: Formatted number string
    """
    if num == float('inf'):
        return "∞"
    elif num > 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num > 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num > 1_000:
        return f"{num/1_000:.1f}k"
    else:
        return f"{num:.1f}"

def plot_feedback_rul_estimation(rul_values, cycle_points, damage_values, 
                               point_location, time_per_cycle=70.8,
                               feedback_factor=3.0, base_damage_rate=None):
    """Create and display comprehensive feedback RUL model plots
    
    Args:
        rul_values: Array of RUL values
        cycle_points: Array of cycle points
        damage_values: Array of damage values
        point_location: Location of the analyzed point (tuple)
        time_per_cycle: Time duration for each cycle in seconds (default: 70.8)
        feedback_factor: Feedback factor used in the model
        base_damage_rate: Initial damage rate before feedback (optional)
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert cycles to years for x-axis
    seconds_per_year = 365.25 * 24 * 60 * 60  # seconds in a year
    years = cycle_points * time_per_cycle / seconds_per_year
    
    # Unpack point location
    row, col = point_location
    clean_location = f"Point ({row},{col})"
    
    # Define common colors
    rul_color = "#1f77b4"  # Blue
    damage_color = "#d62728"  # Red
    rate_color = "#2ca02c"  # Green
    
    # 1. Top Left: RUL vs Time (absolute cycles)
    ax1 = axes[0, 0]
    initial_rul = rul_values[0]
    
    # Plot RUL curve
    ax1.plot(cycle_points, rul_values, '-', color=rul_color, linewidth=2.5)
    
    # Add markers at reasonable intervals
    marker_indices = np.linspace(0, len(cycle_points)-1, min(10, len(cycle_points))).astype(int)
    ax1.plot(cycle_points[marker_indices], rul_values[marker_indices], 'o', 
            color=rul_color, markersize=6)
    
    # Format plot
    ax1.set_xlabel("Cycles", fontsize=12)
    ax1.set_ylabel("Remaining Useful Life (cycles)", fontsize=12)
    ax1.set_title("Absolute RUL vs Cycles", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, cycle_points[-1] * 1.02)
    ax1.set_ylim(0, initial_rul * 1.05)
    
    # 2. Top Right: RUL Percentage vs Time (years)
    ax2 = axes[0, 1]
    
    # Convert RUL to percentage of initial life
    rul_percentage = (rul_values / initial_rul) * 100
    
    # Plot RUL percentage curve
    ax2.plot(years, rul_percentage, '-', color=rul_color, linewidth=2.5)
    
    # Add markers
    ax2.plot(years[marker_indices], rul_percentage[marker_indices], 'o', 
             color=rul_color, markersize=6)
    
    # Format plot
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Remaining Useful Life (%)", fontsize=12)
    ax2.set_title("RUL Percentage vs Time", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(0, years[-1] * 1.02)
    ax2.set_ylim(0, 105)
    
    # Add percentage formatter
    def percentage_formatter(x, pos):
        return f"{x:.0f}%"
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(percentage_formatter))
    
    # 3. Bottom Left: Damage Accumulation vs Cycles
    ax3 = axes[1, 0]
    
    # Plot damage curve
    ax3.plot(cycle_points, damage_values, '-', color=damage_color, linewidth=2.5)
    
    # Add markers
    ax3.plot(cycle_points[marker_indices], damage_values[marker_indices], 'o', 
             color=damage_color, markersize=6)
    
    # Add failure threshold line
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label="Failure Threshold")
    
    # Format plot
    ax3.set_xlabel("Cycles", fontsize=12)
    ax3.set_ylabel("Cumulative Damage", fontsize=12)
    ax3.set_title("Damage Accumulation", fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xlim(0, cycle_points[-1] * 1.02)
    ax3.set_ylim(0, min(1.1, max(1.05, damage_values[-1] * 1.1)))
    ax3.legend(loc='upper left', fontsize=10)
    
    # 4. Bottom Right: Damage Rate vs Cycles
    ax4 = axes[1, 1]
    
    # Calculate damage rates if not provided
    if base_damage_rate is None:
        # Estimate base damage rate from initial RUL
        base_damage_rate = 1 / initial_rul if initial_rul > 0 else 0
    
    # Calculate damage rates at each point
    damage_rates = base_damage_rate * (1 + feedback_factor * damage_values)
    
    # Plot damage rate curve
    ax4.plot(cycle_points, damage_rates, '-', color=rate_color, linewidth=2.5)
    
    # Add markers
    ax4.plot(cycle_points[marker_indices], damage_rates[marker_indices], 'o', 
             color=rate_color, markersize=6)
    
    # Add initial damage rate reference line
    ax4.axhline(y=base_damage_rate, color='k', linestyle='--', alpha=0.7, 
                label="Initial Damage Rate")
    
    # Format plot
    ax4.set_xlabel("Cycles", fontsize=12)
    ax4.set_ylabel("Damage Rate (per cycle)", fontsize=12)
    ax4.set_title("Damage Rate Acceleration", fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_xlim(0, cycle_points[-1] * 1.02)
    ax4.set_ylim(0, damage_rates[-1] * 1.1)
    ax4.legend(loc='upper left', fontsize=10)
    
    # Use scientific notation for damage rate
    ax4.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Add overall figure information
    final_rul = rul_values[-1]
    final_percentage_used = (1 - (final_rul / initial_rul)) * 100 if initial_rul > 0 else 0
    failure_index = np.argmin(rul_values) if 0 in rul_values else -1
    failure_cycles = cycle_points[failure_index] if failure_index >= 0 else None
    
    # Format labels for overall information
    initial_rul_str = format_large_number(initial_rul)
    final_rul_str = format_large_number(final_rul)
    
    if failure_cycles is not None:
        failure_years = failure_cycles * time_per_cycle / seconds_per_year
        failure_info = f"Failure at: {format_large_number(failure_cycles)} cycles ({failure_years:.2f} years)"
    else:
        projected_years = years[-1]
        failure_info = f"No failure projected within {format_large_number(cycle_points[-1])} cycles ({projected_years:.2f} years)"
    
    # Create overall title
    title = f"Tungsten Component Feedback RUL Model (Factor: {feedback_factor})"
    subtitle = f"{clean_location} | Initial RUL: {initial_rul_str} | Final RUL: {final_rul_str} | Life used: {final_percentage_used:.1f}%"
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.91, subtitle, ha='center', fontsize=14)
    plt.figtext(0.5, 0.885, failure_info, ha='center', fontsize=12, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Save figure
    plt.savefig(os.path.join(os.getcwd(), 'feedback_rul_model.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), 'feedback_rul_model.png'), bbox_inches='tight', dpi=300)
    
    plt.show() 