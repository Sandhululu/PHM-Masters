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
    axes_strain[0].plot(time_points, max_shear_strain[:, 0, 0], 'b-', label='Max Shear')
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
    # Point with maximum shear strain
    row, col = max_shear_loc
    axes_strain[1].plot(time_points, max_shear_strain[:, row, col], 'b-', 
                    label=f'Max Shear ({row},{col}), Max={np.nanmax(max_shear_strain[:, row, col]):.2e}')
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
    # Setup the plots
    fig_fatigue, axes_fatigue = plt.subplots(1, 2, figsize=(20, 10))
    fig_fatigue.subplots_adjust(wspace=0.7, right=0.95)  # Increased spacing between plots
    
    # Plot max principal strain location
    row, col = max_principal_loc
    signal_clean = np.copy(major_principal_strain[:, row, col])
    mask = ~np.isnan(signal_clean)
    if np.any(mask):
        indices = np.arange(len(signal_clean))
        signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
        axes_fatigue[0].plot(time_points, signal_clean, '-', 
                           label=f"Max Principal Strain Location ({row},{col})")
        
    # Configure the principal strain plot
    axes_fatigue[0].set_xlabel("Time (seconds)", fontsize=13)
    axes_fatigue[0].set_ylabel("Strain (ε)", fontsize=13)
    axes_fatigue[0].set_title(f"Strain History - Max Principal Strain Location ({row},{col})", 
                            fontsize=15, pad=10)
    axes_fatigue[0].tick_params(axis='both', which='major', labelsize=12)
    axes_fatigue[0].grid(True, alpha=0.7)
    
    # Plot max shear strain location
    row, col = max_shear_loc
    signal_clean = np.copy(max_shear_strain[:, row, col])
    mask = ~np.isnan(signal_clean)
    if np.any(mask):
        indices = np.arange(len(signal_clean))
        signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
        axes_fatigue[1].plot(time_points, signal_clean, '-', 
                           label=f"Max Shear Strain Location ({row},{col})")
        
    # Configure the shear strain plot
    axes_fatigue[1].set_xlabel("Time (seconds)", fontsize=13)
    axes_fatigue[1].set_ylabel("Strain (ε)", fontsize=13)
    axes_fatigue[1].set_title(f"Strain History - Max Shear Strain Location ({row},{col})", 
                            fontsize=15, pad=10)
    axes_fatigue[1].tick_params(axis='both', which='major', labelsize=12)
    axes_fatigue[1].grid(True, alpha=0.7)
    
    return fig_fatigue, axes_fatigue

def configure_rul_plot_axes(ax_rul1, ax_rul2, rul_max_strain, rul_max_shear):
    """Configure the axes for RUL plots with appropriate scales and formatting
    
    Args:
        ax_rul1, ax_rul2: Matplotlib axes for the two RUL plots
        rul_max_strain, rul_max_shear: RUL data for both locations
    """
    # Format function for y-axis labels
    def format_func(value, pos):
        if value >= 1000:
            return f'{value/1000:.1f}k'.rstrip('0').rstrip('.')
        return f'{value:.0f}'
    
    # Custom y-axis limits for the first plot (Max Principal Strain)
    # Set exactly according to the example plot
    ax_rul1.set_ylim([0, 2000])
    ax_rul1.yaxis.set_major_locator(plt.MultipleLocator(250))
    ax_rul1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Custom y-axis limits for the second plot (Max Shear Strain)
    # Set exactly according to the example plot (31.5k to 32k+)
    ax_rul2.set_ylim([30000, 31800])
    ax_rul2.yaxis.set_major_locator(plt.MultipleLocator(500))
    ax_rul2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Add grid with light gray dotted lines
    ax_rul1.grid(True, linestyle='--', color='lightgray', alpha=0.7)
    ax_rul2.grid(True, linestyle='--', color='lightgray', alpha=0.7)

def plot_rul_estimation(rul_max_strain, cycles_max_strain_plot, 
                       rul_max_shear, cycles_max_shear_plot,
                       max_principal_loc, max_shear_loc, time_per_cycle=70.6):
    """Create and display RUL estimation plots
    
    Args:
        rul_max_strain, cycles_max_strain_plot: RUL data for principal strain
        rul_max_shear, cycles_max_shear_plot: RUL data for shear strain
        max_principal_loc, max_shear_loc: Locations of strain values
        time_per_cycle: Time duration for each cycle in seconds (default: 70.6)
    """
    # Create the RUL figure
    fig_rul, (ax_rul1, ax_rul2) = plt.subplots(1, 2, figsize=(22, 10))
    fig_rul.subplots_adjust(wspace=0.6, right=0.85)  # Increased horizontal spacing
    
    # Plot RUL for maximum principal strain location
    if rul_max_strain is not None and cycles_max_strain_plot is not None:
        row, col = max_principal_loc
        clean_location = f"Max Principal Strain Location ({row},{col})"
        
        # Plot RUL vs Cycles with markers
        ax_rul1.plot(cycles_max_strain_plot, rul_max_strain, '-', color='#1f77b4', linewidth=2.5)
        marker_indices = np.linspace(0, len(cycles_max_strain_plot)-1, min(10, len(cycles_max_strain_plot))).astype(int)
        ax_rul1.plot(cycles_max_strain_plot[marker_indices], rul_max_strain[marker_indices], 'o', color='#1f77b4', markersize=7)
        
        # Add text information boxes
        initial_rul = rul_max_strain[0]
        final_rul = rul_max_strain[-1]
        final_percentage_used = (1 - (final_rul / initial_rul)) * 100
        
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
            
        ax_rul1.text(0.97, 0.97, 
                    f"Initial RUL: {format_large_number(initial_rul)} cycles",
                    transform=ax_rul1.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul1.text(0.97, 0.89, 
                    f"Final RUL: {format_large_number(final_rul)} cycles",
                    transform=ax_rul1.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul1.text(0.97, 0.81, 
                    f"Life used: {final_percentage_used:.1f}%",
                    transform=ax_rul1.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Add time-based information
        if time_per_cycle > 0:
            ax_rul1.text(0.97, 0.73, 
                        f"Elapsed time: {format_time(cycles_max_strain_plot[-1])}",
                        transform=ax_rul1.transAxes, fontsize=12,
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Set plot formatting
        ax_rul1.set_xlabel("Cycles Experienced", fontsize=14)
        ax_rul1.set_ylabel("Remaining Useful Life (cycles)", fontsize=14, labelpad=10)
        ax_rul1.set_title(f"RUL Estimation - {clean_location}", fontsize=16, pad=10)
        ax_rul1.grid(True, linestyle='--', alpha=0.7)
        ax_rul1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot RUL for maximum shear strain location
    if rul_max_shear is not None and cycles_max_shear_plot is not None:
        row, col = max_shear_loc
        clean_location = f"Max Shear Strain Location ({row},{col})"
        
        # Plot RUL vs Cycles with markers
        ax_rul2.plot(cycles_max_shear_plot, rul_max_shear, '-', color='#d62728', linewidth=2.5)
        marker_indices = np.linspace(0, len(cycles_max_shear_plot)-1, min(10, len(cycles_max_shear_plot))).astype(int)
        ax_rul2.plot(cycles_max_shear_plot[marker_indices], rul_max_shear[marker_indices], 'o', color='#d62728', markersize=7)
        
        # Add text information boxes
        initial_rul = rul_max_shear[0]
        final_rul = rul_max_shear[-1]
        final_percentage_used = (1 - (final_rul / initial_rul)) * 100
        
        ax_rul2.text(0.97, 0.97, 
                    f"Initial RUL: {format_large_number(initial_rul)} cycles",
                    transform=ax_rul2.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul2.text(0.97, 0.89, 
                    f"Final RUL: {format_large_number(final_rul)} cycles",
                    transform=ax_rul2.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        ax_rul2.text(0.97, 0.81, 
                    f"Life used: {final_percentage_used:.1f}%",
                    transform=ax_rul2.transAxes, fontsize=12,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Add time-based information
        if time_per_cycle > 0:
            ax_rul2.text(0.97, 0.73, 
                        f"Elapsed time: {format_time(cycles_max_shear_plot[-1])}",
                        transform=ax_rul2.transAxes, fontsize=12,
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Set plot formatting
        ax_rul2.set_xlabel("Cycles Experienced", fontsize=14)
        ax_rul2.set_ylabel("Remaining Useful Life (cycles)", fontsize=14, labelpad=10)
        ax_rul2.set_title(f"RUL Estimation - {clean_location}", fontsize=16, pad=10)
        ax_rul2.grid(True, linestyle='--', alpha=0.7)
        ax_rul2.tick_params(axis='both', which='major', labelsize=12)
    
    # Configure y-axis for plots
    configure_rul_plot_axes(ax_rul1, ax_rul2, rul_max_strain, rul_max_shear)
    
    # Add overall title with time information
    if time_per_cycle > 0 and (cycles_max_strain_plot is not None or cycles_max_shear_plot is not None):
        max_cycles = max(
            np.max(cycles_max_strain_plot) if cycles_max_strain_plot is not None else 0,
            np.max(cycles_max_shear_plot) if cycles_max_shear_plot is not None else 0
        )
        time_str = format_time(max_cycles)
        fig_rul.suptitle(f'Tungsten Component RUL Estimation (Max time: {time_str})', 
                         fontsize=18, fontweight='bold', y=0.98)
    else:
        fig_rul.suptitle('Tungsten Component Remaining Useful Life Estimation', 
                         fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(os.getcwd(), 'rul_estimation.svg'), bbox_inches='tight', dpi=300)
    plt.show() 