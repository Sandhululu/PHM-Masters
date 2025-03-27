# -*- coding: utf-8 -*-
"""
RUL estimation from strain DIC measurement

@author: Jayron Sandhu
"""

import glob
import os
import pandas as pd
import tkinter
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
import rainflow
import sys

def read_strain_data(strain_type='exx'):
    """Generic function to read strain data from CSV files. Consolidates the duplicate code."""
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-alpha', 0.0)
    root.attributes('-topmost', True)
    root.update()
    
    dirDIC = f'/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/{strain_type}'
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    arrays = [pd.read_csv(f, header=None).values for f in csv_files]
    
    root.destroy()
    
    # Determine max dimensions and pad arrays
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)
    
    padded_arrays = [np.pad(arr, 
                           ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), 
                           'constant', constant_values=np.nan) 
                    for arr in arrays]
    
    DICData = np.stack(padded_arrays)
    Average = np.nanmean(DICData, axis=0)
    return Average, DICData

# Replace the two separate functions with calls to the generic function
def read_DIC_data_from_csv():
    return read_strain_data('exx')

def read_eyy_data():
    return read_strain_data('eyy')

AverageExx, DICExx = read_DIC_data_from_csv()
AverageEyy, DICEyy = read_eyy_data()
ThermalStrain = (DICExx+DICEyy)/2

# Create time array and calculate statistical measures in fewer lines
time_points = np.arange(0, len(ThermalStrain) * 0.2, 0.2)
mean_strain, std_strain, max_strain = np.nanmean(ThermalStrain, axis=0), np.nanstd(ThermalStrain, axis=0), np.nanmax(ThermalStrain, axis=0)
high_strain_threshold = mean_strain + 0.25 * std_strain
high_strain_points = np.where(max_strain > high_strain_threshold)

# Create plots with fewer lines
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Original strain at (0,0) - consolidated plotting commands
ax1.plot(time_points, ThermalStrain[:,0,0], 'b-', label='Thermal Strain (0,0)')
ax1.plot(time_points, DICExx[:,0,0], 'r--', label='Exx Strain (0,0)')
ax1.plot(time_points, DICEyy[:,0,0], 'g--', label='Eyy Strain (0,0)')
ax1.set_xlabel('Time (seconds)'), ax1.set_ylabel('Strain'), ax1.set_title('Strain vs Time at Point (0,0)')
ax1.grid(True), ax1.legend()

# Plot 2: Points with highest strain
if len(high_strain_points[0]) > 0:
    max_strain_values = max_strain[high_strain_points]
    top_5_indices = np.argsort(max_strain_values)[-5:]
    for idx in top_5_indices:
        row, col = high_strain_points[0][idx], high_strain_points[1][idx]
        ax2.plot(time_points, ThermalStrain[:,row,col], 
                label=f'Point ({row},{col}), Max={max_strain[row,col]:.2e}')

ax2.set_xlabel('Time (seconds)'), ax2.set_ylabel('Strain'), ax2.set_title('Strain vs Time at High Strain Points')
ax2.grid(True), ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'strain_analysis.svg'), bbox_inches='tight')
plt.show()

# Print statistical summary more compactly
print(f"\nStatistical Summary:\n"
      f"Number of high strain points found: {len(high_strain_points[0])}\n"
      f"Global mean strain: {np.nanmean(ThermalStrain):.2e}\n"
      f"Global max strain: {np.nanmax(ThermalStrain):.2e}\n"
      f"Global strain standard deviation: {np.nanstd(ThermalStrain):.2e}")

# Material properties for Tungsten - grouped together
E, poisson, alpha = 400e9, 0.28, 4.5e-6

# Initialize arrays for principal strains - combined initialization
principal_strain1 = np.zeros_like(ThermalStrain)  # Major principal strain
principal_strain2 = np.zeros_like(ThermalStrain)  # Minor principal strain
max_shear_strain = np.zeros_like(ThermalStrain)   # Maximum shear strain

# Calculate principal strains more efficiently
for t in range(ThermalStrain.shape[0]):
    for i in range(ThermalStrain.shape[1]):
        for j in range(ThermalStrain.shape[2]):
            # Calculate principal strains in fewer lines
            exx, eyy, exy = DICExx[t, i, j], DICEyy[t, i, j], 0
            avg, diff = (exx + eyy) / 2, (exx - eyy) / 2
            radius = np.sqrt(diff**2 + exy**2)
            
            # Store principal strains and shear strain
            principal_strain1[t, i, j], principal_strain2[t, i, j], max_shear_strain[t, i, j] = avg + radius, avg - radius, radius

# Initialize and calculate stress components more concisely
stress_xx, stress_yy, stress_von_mises = [np.zeros_like(ThermalStrain) for _ in range(3)]

# Calculate stresses using plane stress equations with vectorized operations
for t in range(ThermalStrain.shape[0]):
    factor = E / (1 - poisson**2)
    stress_xx[t] = factor * (DICExx[t] + poisson * DICEyy[t])
    stress_yy[t] = factor * (DICEyy[t] + poisson * DICExx[t])
    stress_von_mises[t] = np.sqrt(stress_xx[t]**2 + stress_yy[t]**2 - stress_xx[t] * stress_yy[t])

# Calculate mean and maximum stresses and strains in a more compact way
mean_stress_xx, max_stress_xx = np.nanmean(stress_xx, axis=0), np.nanmax(stress_xx, axis=0)
mean_stress_yy, max_stress_yy = np.nanmean(stress_yy, axis=0), np.nanmax(stress_yy, axis=0)
mean_stress_von_mises, max_stress_von_mises = np.nanmean(stress_von_mises, axis=0), np.nanmax(stress_von_mises, axis=0)

# Find a safe point for plotting (first time step, position 0,0)
# In case there are issues with finding maximum points
safe_point = (0, 0)

# Find locations of maximum values with safety checks in a more compact way
try:
    # Find locations more concisely
    max_principal = np.nan_to_num(principal_strain1[0], nan=-np.inf)
    min_principal = np.nan_to_num(principal_strain2[0], nan=np.inf)
    max_shear = np.nan_to_num(max_shear_strain[0], nan=-np.inf)
    
    max_principal_loc = np.unravel_index(np.argmax(max_principal), principal_strain1[0].shape)
    min_principal_loc = np.unravel_index(np.argmin(min_principal), principal_strain2[0].shape)
    max_shear_loc = np.unravel_index(np.argmax(max_shear), max_shear_strain[0].shape)
except Exception as e:
    print(f"Error finding max locations: {e}")
    # Use safe defaults
    max_principal_loc = min_principal_loc = max_shear_loc = safe_point

# Verify locations more efficiently
max_rows, max_cols = principal_strain1[0].shape
def is_valid_location(loc, data_array):
    return (loc[0] < max_rows and loc[1] < max_cols and 
            not np.isnan(data_array[0, loc[0], loc[1]]))

# Check and fix invalid locations
if not is_valid_location(max_principal_loc, principal_strain1): max_principal_loc = safe_point
if not is_valid_location(min_principal_loc, principal_strain2): min_principal_loc = safe_point
if not is_valid_location(max_shear_loc, max_shear_strain): max_shear_loc = safe_point

print(f"Maximum principal strain location: {max_principal_loc}\n"
      f"Minimum principal strain location: {min_principal_loc}\n"
      f"Maximum shear strain location: {max_shear_loc}")

# Plot stress and strain fields
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Plot stress at point (0,0) over time
axes[0, 0].plot(time_points, stress_xx[:, 0, 0]/1e6, 'r-', label='σxx')
axes[0, 0].plot(time_points, stress_yy[:, 0, 0]/1e6, 'g-', label='σyy')
axes[0, 0].plot(time_points, stress_von_mises[:, 0, 0]/1e6, 'b-', label='von Mises')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Stress (MPa)')
axes[0, 0].set_title('Stress vs Time at Point (0,0)')
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot principal strains at point (0,0) over time
axes[0, 1].plot(time_points, principal_strain1[:, 0, 0], 'r-', label='Major Principal')
axes[0, 1].plot(time_points, principal_strain2[:, 0, 0], 'g-', label='Minor Principal')
axes[0, 1].plot(time_points, max_shear_strain[:, 0, 0], 'b-', label='Max Shear')
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Strain')
axes[0, 1].set_title('Principal Strains vs Time at Point (0,0)')
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot von Mises stress for highest strain points
if len(high_strain_points[0]) > 0:
    for idx in top_5_indices:
        row, col = high_strain_points[0][idx], high_strain_points[1][idx]
        axes[1, 0].plot(time_points, stress_von_mises[:, row, col]/1e6, 
                      label=f'Point ({row},{col}), Max={np.nanmax(stress_von_mises[:, row, col])/1e6:.1f} MPa')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('von Mises Stress (MPa)')
    axes[1, 0].set_title('von Mises Stress vs Time at High Strain Points')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

# Plot principal strains for key points
# Point with maximum principal strain
row, col = max_principal_loc
axes[1, 1].plot(time_points, principal_strain1[:, row, col], 'r-', 
                label=f'Major Principal at ({row},{col}), Max={np.nanmax(principal_strain1[:, row, col]):.2e}')
# Point with minimum principal strain (most negative)
row, col = min_principal_loc
axes[1, 1].plot(time_points, principal_strain2[:, row, col], 'g-', 
                label=f'Minor Principal at ({row},{col}), Min={np.nanmin(principal_strain2[:, row, col]):.2e}')
# Point with maximum shear strain
row, col = max_shear_loc
axes[1, 1].plot(time_points, max_shear_strain[:, row, col], 'b-', 
                label=f'Max Shear at ({row},{col}), Max={np.nanmax(max_shear_strain[:, row, col]):.2e}')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Strain')
axes[1, 1].set_title('Principal Strains vs Time at Key Points')
axes[1, 1].grid(True)
axes[1, 1].legend()

# Create stress and strain field heat maps for final time step
t_final = -1  # Use the last time step
im1 = axes[2, 0].imshow(stress_von_mises[t_final]/1e6, cmap='hot')
axes[2, 0].set_title('von Mises Stress Field (MPa)')
plt.colorbar(im1, ax=axes[2, 0])

im2 = axes[2, 1].imshow(principal_strain1[t_final], cmap='hot')
axes[2, 1].set_title('Major Principal Strain Field')
plt.colorbar(im2, ax=axes[2, 1])

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'stress_strain_analysis.svg'), bbox_inches='tight')
plt.show()

# Print stress and strain summary
print("\nStress Analysis Summary:")
print(f"Maximum σxx: {np.nanmax(stress_xx)/1e6:.2f} MPa")
print(f"Maximum σyy: {np.nanmax(stress_yy)/1e6:.2f} MPa")
print(f"Maximum von Mises stress: {np.nanmax(stress_von_mises)/1e6:.2f} MPa")
print(f"Yield strength of tungsten (typical): ~1000 MPa")
print(f"Safety factor: {1000/np.nanmax(stress_von_mises)*1e6:.2f}")

print("\nPrincipal Strain Analysis:")
max_e1 = np.nanmax(principal_strain1)
max_e2 = np.nanmax(principal_strain2)
min_e2 = np.nanmin(principal_strain2)
max_shear = np.nanmax(max_shear_strain)
print(f"Maximum major principal strain: {max_e1:.2e}")
print(f"Maximum minor principal strain: {max_e2:.2e}")
print(f"Minimum minor principal strain: {min_e2:.2e}")
print(f"Maximum shear strain: {max_shear:.2e}")

# Print locations of maximum values
row, col = max_principal_loc
print(f"Location of maximum principal strain: Point ({row},{col})")
row, col = min_principal_loc
print(f"Location of maximum compressive strain: Point ({row},{col})")
row, col = max_shear_loc
print(f"Location of maximum shear strain: Point ({row},{col})")

# ================= FATIGUE ANALYSIS USING RAINFLOW COUNTING =================
print("\n\n=========== FATIGUE ANALYSIS ===========")

# Create a new figure for fatigue analysis with improved size and spacing
fig_fatigue, axes_fatigue = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Increase spacing between subplots

# Replace the analyze_fatigue function with a more concise version
def analyze_fatigue(signal, location_name, ax_time, ax_hist):
    """More concise function to extract cycles and calculate damage"""
    # Clean signal by removing NaNs
    signal_clean = np.copy(signal)
    mask = ~np.isnan(signal_clean)
    if not np.any(mask):
        print(f"Warning: All NaN values in signal for {location_name}")
        return None
    
    # Interpolate and plot in fewer lines
    indices = np.arange(len(signal_clean))
    signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
    
    # Plot the signal in fewer lines
    ax_time.plot(time_points, signal_clean, '-', label=f"{location_name}")
    ax_time.set_xlabel("Time (seconds)", fontsize=12)
    ax_time.set_ylabel("Strain", fontsize=12)
    ax_time.set_title(f"Strain History - {location_name}", fontsize=14, pad=10)
    ax_time.tick_params(axis='both', which='major', labelsize=11)
    ax_time.grid(True)
    ax_time.legend(fontsize=12)
    
    # Extract and process cycles
    cycles = rainflow.count_cycles(signal_clean)
    if not cycles:
        print(f"No cycles found for {location_name}")
        return None
    
    # Handle different cycle formats more efficiently
    first_cycle = next(iter(cycles))
    if len(first_cycle) == 2:
        cycles_array = np.array([(rng, count) for rng, count in cycles])
        ranges, counts = cycles_array[:, 0], cycles_array[:, 1]
    elif len(first_cycle) == 5:
        cycles_array = np.array([(rng, mean, count, i_start, i_end) for rng, mean, count, i_start, i_end in cycles])
        ranges, counts = cycles_array[:, 0], cycles_array[:, 2]
    else:
        print(f"Unexpected cycle format: {first_cycle}")
        return None
    
    # Plot histogram
    ax_hist.hist(ranges, weights=counts, bins=10, alpha=0.7)
    ax_hist.set_xlabel("Strain Range", fontsize=12)
    ax_hist.set_ylabel("Number of Cycles", fontsize=12)
    ax_hist.set_title(f"Cycle Histogram - {location_name}", fontsize=14, pad=10)
    ax_hist.tick_params(axis='both', which='major', labelsize=11)
    ax_hist.grid(True)
    
    # Material properties
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    # Calculate fatigue life with vectorized operations where possible
    N_f_cycles = []
    for strain_range in ranges:
        strain_amp = strain_range / 2
        
        # Numerical solution
        N_values = np.logspace(1, 10, 1000)  
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find closest N and apply safety factor and cap
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))] / safety_factor
        N_f_cycles.append(min(N_f, 1e6))  # Cap at 1 million cycles
    
    # Calculate damage metrics more concisely
    damage_per_cycle = [count/N_f for count, N_f in zip(counts, N_f_cycles)]
    total_damage = sum(damage_per_cycle)
    
    # Calculate life statistics
    if N_f_cycles:
        avg_life = min(np.mean(N_f_cycles), 1e6)  # Cap average life
        min_life = np.min(N_f_cycles)
    else:
        avg_life = min_life = float('inf')
    
    # Calculate estimated life with cap
    estimated_life = min(1/total_damage if total_damage > 0 else float('inf'), 100000)
    
    # Print analysis summary in a more compact format
    print(f"\nFatigue Analysis for {location_name} (Manson-Coffin with safety factor {safety_factor}x):")
    print(f"  Total cycles detected: {len(cycles)}")
    print(f"  Maximum strain range: {np.max(ranges):.2e}, Minimum: {np.min(ranges):.2e}")
    print(f"  Average cycles to failure: {avg_life:.1f}, Minimum: {min_life:.1f}")
    print(f"  Cumulative damage (Miner's rule): {total_damage:.6f}")
    
    if total_damage > 0:
        print(f"  Estimated fatigue life: {estimated_life:.1f} repetitions of this loading")
        
        # Add annotation
        ax_hist.text(0.05, 0.95, 
                    f"Estimated Life: {estimated_life:.1f} cycles\n(with {safety_factor}x safety factor)",
                    transform=ax_hist.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    return cycles_array

# Analyze fatigue at key locations
print("\nPerforming rainflow analysis at key locations...")

# Point with maximum stress
row, col = max_principal_loc
cycles_max_strain = analyze_fatigue(
    principal_strain1[:, row, col], 
    f"Max Principal Strain Location ({row},{col})",
    axes_fatigue[0, 0], axes_fatigue[0, 1]
)

# Point with maximum shear (often critical for fatigue)
row, col = max_shear_loc
cycles_max_shear = analyze_fatigue(
    max_shear_strain[:, row, col],
    f"Max Shear Strain Location ({row},{col})",
    axes_fatigue[1, 0], axes_fatigue[1, 1]
)

# Apply tight layout with more padding to prevent text clipping
plt.tight_layout(pad=4.0)
plt.savefig(os.path.join(os.getcwd(), 'fatigue_analysis.svg'), bbox_inches='tight', dpi=300)
plt.show()

# ================= REMAINING USEFUL LIFE (RUL) ESTIMATION =================
print("\n\n=========== RUL ESTIMATION ===========")

# Create RUL figure with improved formatting
fig_rul, (ax_rul1, ax_rul2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.3)  # Increase spacing between subplots

def estimate_rul(cycles_array, location_name, location_coords, ax, color='#1f77b4', cycle_multiplier=10):
    """More concise function to estimate and plot Remaining Useful Life (RUL)"""
    if cycles_array is None or len(cycles_array) == 0:
        print(f"No cycle data available for {location_name}")
        return None, None
    
    # Extract data with format handling
    if cycles_array.shape[1] >= 3:  # Format from rainflow: range, mean, count, ...
        ranges, original_counts = cycles_array[:, 0], cycles_array[:, 2]
    else:  # Simple format: range, count
        ranges, original_counts = cycles_array[:, 0], cycles_array[:, 1]
    
    counts = original_counts * cycle_multiplier  # Apply multiplier
    
    # Material properties in a more compact format
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    # Calculate cycles to failure more efficiently
    N_f_cycles = []
    for strain_range in ranges:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 10, 1000)
        
        # Calculate strain components
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life with safety factor and cap
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))] / safety_factor
        N_f_cycles.append(min(N_f, 1e6))
    
    # Process cycles by sorting and calculating damage
    sorted_indices = np.argsort(N_f_cycles)
    sorted_ranges, sorted_counts, sorted_N_f = ranges[sorted_indices], counts[sorted_indices], np.array(N_f_cycles)[sorted_indices]
    
    # Calculate damage metrics
    damage_per_cycle = sorted_counts / sorted_N_f
    cumulative_damage = np.cumsum(damage_per_cycle)
    avg_damage_rate = np.sum(damage_per_cycle) / np.sum(sorted_counts)
    
    # Calculate cycle experience and RUL
    cycles_experienced = np.cumsum(sorted_counts)
    total_cycles = np.sum(sorted_counts)
    rul_cycles = np.maximum((1 - cumulative_damage) / avg_damage_rate, 0)
    
    # Initial RUL with reasonable capping
    initial_rul = 1 / avg_damage_rate if avg_damage_rate > 0 else float('inf')
    max_reasonable_life = 100000
    
    if initial_rul > max_reasonable_life:
        print(f"Note: Capping very high initial RUL ({initial_rul:.1f}) to {max_reasonable_life}")
        initial_rul = max_reasonable_life
        scale_factor = initial_rul / rul_cycles[0] if rul_cycles[0] > 0 else 1.0
        rul_cycles = rul_cycles * scale_factor
    
    # Create smoother curve with interpolation
    if len(cycles_experienced) > 1:
        interp_cycles = np.linspace(0, cycles_experienced[-1], 100)
        interp_rul = np.interp(interp_cycles, cycles_experienced, rul_cycles)
        cycles_plot = np.insert(interp_cycles, 0, 0)
        rul_plot = np.insert(interp_rul, 0, initial_rul)
    else:
        cycles_plot = np.insert(cycles_experienced, 0, 0)
        rul_plot = np.insert(rul_cycles, 0, initial_rul)
    
    # Helper function for formatting large numbers
    def format_large_number(num):
        if num > 1_000_000: return f"{num/1_000_000:.1f}M"
        elif num > 1_000: return f"{num/1_000:.1f}k"
        else: return f"{num:.1f}"
    
    # Display formatting
    clean_location = f"{location_name} ({location_coords[0]},{location_coords[1]})"
    
    # Plot RUL vs Cycles with markers
    ax.plot(cycles_plot, rul_plot, '-', color=color, linewidth=2)
    marker_indices = np.linspace(0, len(cycles_plot)-1, min(20, len(cycles_plot))).astype(int)
    ax.plot(cycles_plot[marker_indices], rul_plot[marker_indices], 'o', color=color, markersize=6)
    
    # Set up plot formatting
    ax.set_xlabel("Cycles Experienced", fontsize=14)
    ax.set_ylabel("Remaining Useful Life (cycles)", fontsize=14)
    ax.set_title(f"RUL Estimation - {clean_location}", fontsize=16, pad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Format y-axis for large numbers
    if initial_rul > 1000:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, pos: f'{y/1_000_000:.0f}M' if y >= 1_000_000 else 
                          (f'{y/1_000:.0f}k' if y >= 1_000 else f'{y:.0f}')
        ))
    
    # Add annotations more concisely
    if len(cycles_plot) > 1:
        # Initial RUL annotation
        ax.annotate(f"Initial RUL: {format_large_number(initial_rul)} cycles",
                   xy=(0, initial_rul), xytext=(40, -20), textcoords="offset points", fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        # Final RUL annotation
        ax.annotate(f"Final RUL: {format_large_number(rul_plot[-1])} cycles",
                   xy=(cycles_plot[-1], rul_plot[-1]), xytext=(-60, 30), textcoords="offset points", fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))
        
        # Percentage annotation
        mid_idx = len(cycles_plot) // 2
        percentage_used = (1 - (rul_plot[mid_idx] / initial_rul)) * 100
        ax.annotate(f"{percentage_used:.1f}% life used",
                  xy=(cycles_plot[mid_idx], rul_plot[mid_idx]), xytext=(0, -40), textcoords="offset points", fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Calculate life metrics
    estimated_life = initial_rul + total_cycles
    final_percentage_used = (1 - (rul_plot[-1] / initial_rul)) * 100
    
    # Print RUL summary
    print(f"\nRUL Analysis for {clean_location}:")
    print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
    print(f"  Initial RUL: {initial_rul:.1f} cycles")
    print(f"  Final RUL: {rul_plot[-1]:.1f} cycles")
    print(f"  Life used: {final_percentage_used:.2f}%")
    
    return rul_plot, cycles_plot

# Set cycle multiplier - increase this value to add more cycles
CYCLE_MULTIPLIER = 10  # Increase this number for more cycles (10x, 20x, etc.)

# Estimate RUL for both locations
print("\nEstimating RUL for maximum principal strain location...")
rul_max_strain, cycles_max_strain_plot = estimate_rul(
    cycles_max_strain, 
    "Max Principal Strain Location",
    max_principal_loc,
    ax_rul1,
    color='#1f77b4',  # Blue
    cycle_multiplier=CYCLE_MULTIPLIER
)

print("\nEstimating RUL for maximum shear strain location...")
rul_max_shear, cycles_max_shear_plot = estimate_rul(
    cycles_max_shear,
    "Max Shear Strain Location",
    max_shear_loc,
    ax_rul2,
    color='#d62728',  # Red
    cycle_multiplier=CYCLE_MULTIPLIER
)

# Synchronize y-axis limits if both plots have data
if rul_max_strain is not None and rul_max_shear is not None:
    y_max = max(np.max(rul_max_strain), np.max(rul_max_shear))
    y_min = min(np.min(rul_max_strain), np.min(rul_max_shear))
    y_range = y_max - y_min
    # Add 10% padding
    ax_rul1.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])
    ax_rul2.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])

# Set better y-axis scaling specifically for the maximum principal strain plot
if rul_max_strain is not None:
    # Get the range of values in the principal strain RUL
    max_val = np.max(rul_max_strain)
    min_val = np.min(rul_max_strain)
    
    # Set axis limits with better scaling for principal strain (left plot)
    # Focus more on the curve by reducing the upper limit
    upper_limit = min(max_val * 1.2, 50000)  # Cap at 50k or 20% above max, whichever is smaller
    ax_rul1.set_ylim([min_val * 0.9, upper_limit])
    
    # Add text annotations on the plot instead of in a box
    ax_rul1.text(0.05, 0.95, f'Cycle multiplier: {CYCLE_MULTIPLIER}x', 
                transform=ax_rul1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add overall title without the box
fig_rul.suptitle('Tungsten Component Remaining Useful Life Estimation', 
                fontsize=18, fontweight='bold', y=0.98)

# Remove the explanatory text box and use tight layout
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'rul_estimation.svg'), bbox_inches='tight', dpi=300)
plt.show()




