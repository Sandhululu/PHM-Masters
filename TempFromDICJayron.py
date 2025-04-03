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
import matplotlib.ticker as ticker

def read_strain_data(strain_type='exx'):
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-alpha', 0.0)
    root.attributes('-topmost', True)
    root.update()
    
    dirDIC = f'/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/{strain_type}'
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    arrays = [pd.read_csv(f, header=None).values for f in csv_files]

    root.destroy()

    # Determine max dimensions for pad arrays
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)

    # Pad arrays to ensure consistent dimensions
    padded_arrays = [np.pad(arr, 
                           ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), 
                           'constant', constant_values=np.nan) 
                    for arr in arrays]
    
    DICData = np.stack(padded_arrays)
    Average = np.nanmean(DICData, axis=0)
    return Average, DICData

def load_all_data():
   # Load strain data
    AverageExx, DICExx = read_strain_data('exx')
    AverageEyy, DICEyy = read_strain_data('eyy')
    ThermalStrain = (DICExx + DICEyy) / 2
    
    # Generate time points
    time_points = np.arange(0, len(ThermalStrain) * 0.2, 0.2)
    
    # Calculate statistical values
    mean_strain = np.nanmean(ThermalStrain, axis=0)
    std_strain = np.nanstd(ThermalStrain, axis=0)
    max_strain = np.nanmax(ThermalStrain, axis=0)
    
    # Find high strain points
    high_strain_threshold = mean_strain + 0.25 * std_strain
    high_strain_points = np.where(max_strain > high_strain_threshold)
    
    return (AverageExx, DICExx, AverageEyy, DICEyy, ThermalStrain, 
            time_points, mean_strain, std_strain, max_strain, high_strain_points)

def plot_initial_strain_analysis(ThermalStrain, DICExx, DICEyy, time_points, high_strain_points, max_strain):
 
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

def print_statistical_summary(ThermalStrain, high_strain_points):

    print(f"\nStatistical Summary:\n"
          f"Number of high strain points found: {len(high_strain_points[0])}\n"
          f"Global mean strain: {np.nanmean(ThermalStrain):.2e}\n"
          f"Global max strain: {np.nanmax(ThermalStrain):.2e}\n"
          f"Global strain standard deviation: {np.nanstd(ThermalStrain):.2e}")

# Load all data
(AverageExx, DICExx, AverageEyy, DICEyy, ThermalStrain, 
 time_points, mean_strain, std_strain, max_strain, high_strain_points) = load_all_data()

# Plot initial strain analysis
top_5_indices = plot_initial_strain_analysis(ThermalStrain, DICExx, DICEyy, time_points, 
high_strain_points, max_strain)
    
    # Print statistical summary
print_statistical_summary(ThermalStrain, high_strain_points)

# Material properties of Tungsten
E, poisson, alpha = 400e9, 0.28, 4.5e-6

def calculate_principal_strains(ThermalStrain, DICExx, DICEyy):
  
    # Initialize arrays for principal strains
    major_principal_strain = np.zeros_like(ThermalStrain)  
    minor_principal_strain = np.zeros_like(ThermalStrain)  
    max_shear_strain = np.zeros_like(ThermalStrain)   

    # Calculate principal strains 
    for t in range(ThermalStrain.shape[0]):
        for i in range(ThermalStrain.shape[1]):
            for j in range(ThermalStrain.shape[2]):
                exx, eyy, exy = DICExx[t, i, j], DICEyy[t, i, j], 0
                avg, diff = (exx + eyy) / 2, (exx - eyy) / 2
                radius = np.sqrt(diff**2 + exy**2)
                
                # Store principal strains and shear strain
                major_principal_strain[t, i, j] = avg + radius
                minor_principal_strain[t, i, j] = avg - radius
                max_shear_strain[t, i, j] = radius
    
    return major_principal_strain, minor_principal_strain, max_shear_strain

def calculate_stresses(DICExx, DICEyy, E, poisson, ThermalStrain):

    # Initialize and calculate stress components
    stress_xx = np.zeros_like(ThermalStrain)
    stress_yy = np.zeros_like(ThermalStrain)
    stress_von_mises = np.zeros_like(ThermalStrain)

    # Calculate stresses using plane stress equations
    for t in range(ThermalStrain.shape[0]):
        factor = E / (1 - poisson**2)
        stress_xx[t] = factor * (DICExx[t] + poisson * DICEyy[t])
        stress_yy[t] = factor * (DICEyy[t] + poisson * DICExx[t])
        stress_von_mises[t] = np.sqrt(stress_xx[t]**2 + stress_yy[t]**2 - stress_xx[t] * stress_yy[t])
        
    return stress_xx, stress_yy, stress_von_mises

def find_extreme_locations(major_principal_strain, minor_principal_strain, max_shear_strain):
  
    # Safe default point
    safe_point = (0, 0)
    
    try:
        # Find locations efficiently
        max_principal = np.nan_to_num(major_principal_strain[0], nan=-np.inf)
        min_principal = np.nan_to_num(minor_principal_strain[0], nan=np.inf)
        max_shear = np.nan_to_num(max_shear_strain[0], nan=-np.inf)
        
        max_principal_loc = np.unravel_index(np.argmax(max_principal), major_principal_strain[0].shape)
        min_principal_loc = np.unravel_index(np.argmin(min_principal), minor_principal_strain[0].shape)
        max_shear_loc = np.unravel_index(np.argmax(max_shear), max_shear_strain[0].shape)
    except Exception as e:
        print(f"Error finding max locations: {e}")
        # Use safe defaults
        max_principal_loc = min_principal_loc = max_shear_loc = safe_point

    # Verify locations
    max_rows, max_cols = major_principal_strain[0].shape
    
    def is_valid_location(loc, data_array):
        return (loc[0] < max_rows and loc[1] < max_cols and 
                not np.isnan(data_array[0, loc[0], loc[1]]))

    # Check and fix invalid locations
    if not is_valid_location(max_principal_loc, major_principal_strain): 
        max_principal_loc = safe_point
    if not is_valid_location(min_principal_loc, minor_principal_strain): 
        min_principal_loc = safe_point
    if not is_valid_location(max_shear_loc, max_shear_strain): 
        max_shear_loc = safe_point
        
    print(f"Maximum principal strain location: {max_principal_loc}\n"
          f"Minimum principal strain location: {min_principal_loc}\n"
          f"Maximum shear strain location: {max_shear_loc}")
          
    return max_principal_loc, min_principal_loc, max_shear_loc

# Calculate principal strains
major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
    ThermalStrain, DICExx, DICEyy)

# Calculate stresses
stress_xx, stress_yy, stress_von_mises = calculate_stresses(
    DICExx, DICEyy, E, poisson, ThermalStrain)

# Calculate mean and maximum stresses and strains for analysis
mean_stress_xx = np.nanmean(stress_xx, axis=0)
max_stress_xx = np.nanmax(stress_xx, axis=0)
mean_stress_yy = np.nanmean(stress_yy, axis=0)
max_stress_yy = np.nanmax(stress_yy, axis=0)
mean_stress_von_mises = np.nanmean(stress_von_mises, axis=0)
max_stress_von_mises = np.nanmax(stress_von_mises, axis=0)

# Find locations of extreme strain values
max_principal_loc, min_principal_loc, max_shear_loc = find_extreme_locations(
    major_principal_strain, minor_principal_strain, max_shear_strain)

def plot_stress_analysis(time_points, stress_xx, stress_yy, stress_von_mises, 
high_strain_points, top_5_indices):
  
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
    
    fig_strain, axes_strain = plt.subplots(1, 2, figsize=(20, 10))  

    # Plot principal strains at point (0,0) over time
    axes_strain[0].plot(time_points, major_principal_strain[:, 0, 0], 'r-', label='Major Principal')
    axes_strain[0].plot(time_points, minor_principal_strain[:, 0, 0], 'g-', label='Minor Principal')
    axes_strain[0].plot(time_points, max_shear_strain[:, 0, 0], 'b-', label='Max Shear')
    axes_strain[0].set_xlabel('Time (seconds)', fontsize=12)
    axes_strain[0].set_ylabel('Strain (ε)', fontsize=12, labelpad=10)  
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
    fig_strain.subplots_adjust(wspace=0.5) 
    fig_strain.suptitle('Principal Strain Analysis for Tungsten Component', fontsize=16, fontweight='bold')
    

    plt.savefig(os.path.join(os.getcwd(), 'strain_analysis.svg'), bbox_inches='tight', dpi=300)
    plt.show()

def print_analysis_summary(stress_xx, stress_yy, stress_von_mises, 
    major_principal_strain, minor_principal_strain, max_shear_strain,
    max_principal_loc, min_principal_loc, max_shear_loc):
   
    # Print stress summary
    print("\nStress Analysis Summary:")
    print(f"Maximum σxx: {np.nanmax(stress_xx)/1e6:.2f} MPa")
    print(f"Maximum σyy: {np.nanmax(stress_yy)/1e6:.2f} MPa")
    print(f"Maximum von Mises stress: {np.nanmax(stress_von_mises)/1e6:.2f} MPa")
    print(f"Yield strength of tungsten (typical): ~1000 MPa")
    print(f"Safety factor: {1000/np.nanmax(stress_von_mises)*1e6:.2f}")

    # Print strain summary
    print("\nPrincipal Strain Analysis:")
    max_e1 = np.nanmax(major_principal_strain)
    max_e2 = np.nanmax(minor_principal_strain)
    min_e2 = np.nanmin(minor_principal_strain)
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

# Create stress and strain plots
plot_stress_analysis(time_points, stress_xx, stress_yy, stress_von_mises, 
                     high_strain_points, top_5_indices)

plot_strain_analysis(time_points, major_principal_strain, minor_principal_strain, max_shear_strain,
                      max_principal_loc, min_principal_loc, max_shear_loc)

# Print stress and strain analysis summary
print_analysis_summary(stress_xx, stress_yy, stress_von_mises, 
                       major_principal_strain, minor_principal_strain, max_shear_strain,
                       max_principal_loc, min_principal_loc, max_shear_loc)

print("\n\n=========== FATIGUE ANALYSIS ===========")

def analyze_fatigue(signal, location_name, ax_time):
    # Clean signal by removing NaNs
    signal_clean = np.copy(signal)
    mask = ~np.isnan(signal_clean)
    if not np.any(mask):
        print(f"Warning: All NaN values in signal for {location_name}")
        return None
    
    # Interpolate and plot signal
    indices = np.arange(len(signal_clean))
    signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
    
    # Plot the signal
    ax_time.plot(time_points, signal_clean, '-', label=location_name)
    ax_time.set_xlabel("Time (seconds)", fontsize=13)
    ax_time.set_ylabel("Strain (ε)", fontsize=13)  # Added strain symbol
    ax_time.set_title(f"Strain History - {location_name}", fontsize=15, pad=10)
    ax_time.tick_params(axis='both', which='major', labelsize=12)
    ax_time.grid(True, alpha=0.7)
    
    # Extract and process cycles
    cycles = rainflow.count_cycles(signal_clean)
    if not cycles:
        print(f"No cycles found for {location_name}")
        return None
    
    # Handle different cycle formats
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
    
    # Material properties
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    # Calculate fatigue life
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
    
    # Calculate damage metrics
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
    
    # Print analysis summary
    print(f"\nFatigue Analysis for {location_name} (Manson-Coffin with safety factor {safety_factor}x):")
    print(f"  Total cycles detected: {len(cycles)}")
    print(f"  Maximum strain range: {np.max(ranges):.2e}, Minimum: {np.min(ranges):.2e}")
    print(f"  Average cycles to failure: {avg_life:.1f}, Minimum: {min_life:.1f}")
    print(f"  Cumulative damage (Miner's rule): {total_damage:.6f}")
    
    if total_damage > 0:
        print(f"  Estimated fatigue life: {estimated_life:.1f} repetitions of this loading")
    
    return cycles_array

def run_fatigue_analysis(major_principal_strain, max_shear_strain, max_principal_loc, max_shear_loc):

    # Setup the plots
    fig_fatigue, axes_fatigue = plt.subplots(1, 2, figsize=(20, 10))
    fig_fatigue.subplots_adjust(wspace=0.7, right=0.95)  # Increased spacing between plots
    
    print("\nPerforming rainflow analysis at key locations...")
    
    # Analyze max principal strain location
    row, col = max_principal_loc
    cycles_max_strain = analyze_fatigue(
        major_principal_strain[:, row, col], 
        f"Max Principal Strain Location ({row},{col})",
        axes_fatigue[0]
    )
    
    # Analyze max shear strain location
    row, col = max_shear_loc
    cycles_max_shear = analyze_fatigue(
        max_shear_strain[:, row, col],
        f"Max Shear Strain Location ({row},{col})",
        axes_fatigue[1]
    )
    
    plt.savefig(os.path.join(os.getcwd(), 'fatigue_analysis.svg'), bbox_inches='tight', dpi=300)
    plt.show()
    
    return cycles_max_strain, cycles_max_shear

# Run fatigue analysis
cycles_max_strain, cycles_max_shear = run_fatigue_analysis(
    major_principal_strain, max_shear_strain, max_principal_loc, max_shear_loc)

# ================= RUL ESTIMATION FUNCTIONS =================
print("\n\n=========== RUL ESTIMATION ===========")

def estimate_rul(cycles_array, location_name, location_coords, ax, color='#1f77b4', cycle_multiplier=10):
    
    if cycles_array is None or len(cycles_array) == 0:
        print(f"No cycle data available for {location_name}")
        return None, None
    
    # Extract data with format handling
    if cycles_array.shape[1] >= 3:  # Format from rainflow: range, mean, count, ...
        ranges, original_counts = cycles_array[:, 0], cycles_array[:, 2]
    else:  # Simple format: range, count
        ranges, original_counts = cycles_array[:, 0], cycles_array[:, 1]
    
    counts = original_counts * cycle_multiplier  # Apply multiplier
    
    # Material properties and fatigue parameters
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 10.0
    
    # Calculate cycles to failure
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
    sorted_ranges = ranges[sorted_indices]
    sorted_counts = counts[sorted_indices]
    sorted_N_f = np.array(N_f_cycles)[sorted_indices]
    
    # Calculate damage metrics
    damage_per_cycle = sorted_counts / sorted_N_f
    cumulative_damage = np.cumsum(damage_per_cycle)
    avg_damage_rate = np.sum(damage_per_cycle) / np.sum(sorted_counts)
    
    # Calculate cycle experience and RUL
    cycles_experienced = np.cumsum(sorted_counts)
    total_cycles = np.sum(sorted_counts)
    rul_cycles = np.maximum((1 - cumulative_damage) / avg_damage_rate, 0)
    
    # Initial RUL and capping for very high values
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
    ax.plot(cycles_plot, rul_plot, '-', color=color, linewidth=2.5)
    marker_indices = np.linspace(0, len(cycles_plot)-1, min(10, len(cycles_plot))).astype(int)
    ax.plot(cycles_plot[marker_indices], rul_plot[marker_indices], 'o', color=color, markersize=7)
    
    # Set up plot formatting
    ax.set_xlabel("Cycles Experienced", fontsize=14)
    ax.set_ylabel("Remaining Useful Life (cycles)", fontsize=14, labelpad=10)
    ax.set_title(f"RUL Estimation - {clean_location}", fontsize=16, pad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Format y-axis for large numbers
    if initial_rul > 1000:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: f'{y/1_000_000:.0f}M' if y >= 1_000_000 else 
                          (f'{y/1_000:.0f}k' if y >= 1_000 else f'{y:.0f}')
        ))
    
    # Add key information text boxes
    if len(cycles_plot) > 1:
        # Initial RUL
        ax.text(0.97, 0.97, 
                f"Initial RUL: {format_large_number(initial_rul)} cycles",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Final RUL
        ax.text(0.97, 0.89, 
                f"Final RUL: {format_large_number(rul_plot[-1])} cycles",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
        
        # Life used percentage
        final_percentage_used = (1 - (rul_plot[-1] / initial_rul)) * 100
        ax.text(0.97, 0.81, 
                f"Life used: {final_percentage_used:.1f}%",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))
    
    # Calculate and print RUL metrics
    final_percentage_used = (1 - (rul_plot[-1] / initial_rul)) * 100
    
    print(f"\nRUL Analysis for {clean_location}:")
    print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
    print(f"  Initial RUL: {initial_rul:.1f} cycles")
    print(f"  Final RUL: {rul_plot[-1]:.1f} cycles")
    print(f"  Life used: {final_percentage_used:.2f}%")
    
    return rul_plot, cycles_plot

def run_rul_estimation(cycles_max_strain, cycles_max_shear, max_principal_loc, max_shear_loc, cycle_multiplier=10):
    # Create the RUL figure
    fig_rul, (ax_rul1, ax_rul2) = plt.subplots(1, 2, figsize=(22, 10))
    fig_rul.subplots_adjust(wspace=0.6, right=0.85)  # Increased horizontal spacing
    
    # Estimate RUL for maximum principal strain location
    print("\nEstimating RUL for maximum principal strain location...")
    rul_max_strain, cycles_max_strain_plot = estimate_rul(
        cycles_max_strain, 
        "Max Principal Strain Location",
        max_principal_loc,
        ax_rul1,
        color='#1f77b4',  # Blue
        cycle_multiplier=cycle_multiplier
    )
    
    # Estimate RUL for maximum shear strain location
    print("\nEstimating RUL for maximum shear strain location...")
    rul_max_shear, cycles_max_shear_plot = estimate_rul(
        cycles_max_shear,
        "Max Shear Strain Location",
        max_shear_loc,
        ax_rul2,
        color='#d62728',  # Red
        cycle_multiplier=cycle_multiplier
    )
    
    # Configure y-axis for plots
    configure_rul_plot_axes(ax_rul1, ax_rul2, rul_max_strain, rul_max_shear)
    
    # Add overall title
    fig_rul.suptitle('Tungsten Component Remaining Useful Life Estimation', 
    fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(os.getcwd(), 'rul_estimation.svg'), bbox_inches='tight', dpi=300)
    plt.show()
    
    return rul_max_strain, rul_max_shear

def configure_rul_plot_axes(ax_rul1, ax_rul2, rul_max_strain, rul_max_shear):

    # Format function for y-axis labels
    def format_func(value, pos):
        if value >= 1000:
            return f'{value/1000:.2f}k'.rstrip('0').rstrip('.')
        return f'{value:.0f}'
    
    # Custom y-axis limits for the first plot (Max Principal Strain)
    if rul_max_strain is not None:
        # Set limits and ticks
        ax_rul1.set_ylim([0, 2000])  # Focus on 0 to 2000 cycles
        ax_rul1.yaxis.set_major_locator(ticker.MultipleLocator(250))  # 250 unit intervals
        ax_rul1.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    # Scaling for the second plot (Max Shear Strain)
    if rul_max_shear is not None:
        y_max = np.max(rul_max_shear)
        y_min = np.min(rul_max_shear)
        y_range = y_max - y_min
        
        # Add padding to the plot
        ax_rul2.set_ylim([y_min - 0.05*y_range, y_max + 0.2*y_range])
        
        # Apply the same formatting if the range is reasonable
        if y_max <= 5000:
            ax_rul2.yaxis.set_major_locator(ticker.MultipleLocator(250))
            ax_rul2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

# Run RUL estimation with cycle multiplier
CYCLE_MULTIPLIER = 10
rul_max_strain, rul_max_shear = run_rul_estimation(
    cycles_max_strain, cycles_max_shear, 
    max_principal_loc, max_shear_loc, 
    cycle_multiplier=CYCLE_MULTIPLIER)




