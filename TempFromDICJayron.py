# -*- coding: utf-8 -*-
"""
Temperature from strain DIC measurement

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

def read_DIC_data_from_csv():
    # the path to your csv file directory
    root = tkinter.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    cwd = os.getcwd()
    # dirDIC = filedialog.askdirectory(parent=root, initialdir=cwd, title='Please select DIC csv directory')

    dirDIC = '/Users/jayron/Downloads/PHM-Masters/Paper_Data_Set/DIC data/withoutCoil/exx'

    print(dirDIC)

    # get csv files of the from static images results 
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    # List to store 2D arrays
    arrays = []

    # Read each CSV file into a 2D NumPy array and add to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)  # Read the CSV file without headers
        arrays.append(df.values)  # Convert DataFrame to NumPy array

    # Determine the maximum dimensions
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)

    # Pad each array to match the maximum dimensions with NaN
    padded_arrays = []
    for arr in arrays:
        pad_height = max_rows - arr.shape[0]
        pad_width = max_cols - arr.shape[1]
        padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width)), 'constant', constant_values=np.nan)
        padded_arrays.append(padded_arr)

    # Stack the padded arrays into a 3D array
    DICData = np.stack(padded_arrays)
    print(DICData)
    
    Average = np.nanmean(DICData, axis=0)
    return Average, DICData

def read_eyy_data():
    root = tkinter.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    dirDIC = '/Users/jayron/Downloads/PHM-Masters/Paper_Data_Set/DIC data/withoutCoil/eyy'
    print(dirDIC)

    # get csv files of the from static images results 
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    arrays = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        arrays.append(df.values)

    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)

    padded_arrays = []
    for arr in arrays:
        pad_height = max_rows - arr.shape[0]
        pad_width = max_cols - arr.shape[1]
        padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width)), 'constant', constant_values=np.nan)
        padded_arrays.append(padded_arr)

    DICData = np.stack(padded_arrays)    
    Average = np.nanmean(DICData, axis=0)
    return Average, DICData

# AverageX, DICX = read_DIC_data_from_csv()
# AverageY, DICY = read_DIC_data_from_csv()
AverageExx, DICExx = read_DIC_data_from_csv()
AverageEyy, DICEyy = read_eyy_data()
# print(AverageExx)
# print(AverageEyy)

ThermalStrain = (DICExx+DICEyy)/2

print(ThermalStrain.shape)



# Create time array based on 0.2s intervals
time_points = np.arange(0, len(ThermalStrain) * 0.2, 0.2)  # time in seconds

# Calculate statistical measures for each spatial point over time
mean_strain = np.nanmean(ThermalStrain, axis=0)  # Mean strain at each point
std_strain = np.nanstd(ThermalStrain, axis=0)    # Standard deviation
max_strain = np.nanmax(ThermalStrain, axis=0)    # Maximum strain

# Find points with consistently high strain (mean + 2*std)
high_strain_threshold = mean_strain + 0.25 * std_strain
high_strain_points = np.where(max_strain > high_strain_threshold)

# Create multiple plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Original strain at (0,0)
ax1.plot(time_points, ThermalStrain[:,0,0], 'b-', label='Thermal Strain (0,0)')
ax1.plot(time_points, DICExx[:,0,0], 'r--', label='Exx Strain (0,0)')
ax1.plot(time_points, DICEyy[:,0,0], 'g--', label='Eyy Strain (0,0)')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Strain')
ax1.set_title('Strain vs Time at Point (0,0)')
ax1.grid(True)
ax1.legend()

# Plot 2: Points with highest strain
if len(high_strain_points[0]) > 0:
    max_strain_values = max_strain[high_strain_points]
    top_5_indices = np.argsort(max_strain_values)[-5:]
    for idx in top_5_indices:
        row, col = high_strain_points[0][idx], high_strain_points[1][idx]
        ax2.plot(time_points, ThermalStrain[:,row,col], 
                label=f'Point ({row},{col}), Max={max_strain[row,col]:.2e}')

ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Strain')
ax2.set_title('Strain vs Time at High Strain Points')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname('/Users/jayron/Downloads/PHM-Masters/Paper_Data_Set'), 'strain_analysis.svg'), bbox_inches='tight')
plt.show()

# Print statistical summary
print("\nStatistical Summary:")
print(f"Number of high strain points found: {len(high_strain_points[0])}")
print(f"Global mean strain: {np.nanmean(ThermalStrain):.2e}")
print(f"Global max strain: {np.nanmax(ThermalStrain):.2e}")
print(f"Global strain standard deviation: {np.nanstd(ThermalStrain):.2e}")

# Calculate stress fields from strain using properties of Tungsten
# Material properties for Tungsten
E = 400e9  # Young's modulus in Pa (400 GPa)
poisson = 0.28  # Poisson's ratio
alpha = 4.5e-6  # Thermal expansion coefficient in K^-1

# Calculate stress components for each time step
# Initialize stress arrays
stress_xx = np.zeros_like(ThermalStrain)
stress_yy = np.zeros_like(ThermalStrain)
stress_von_mises = np.zeros_like(ThermalStrain)

# Calculate stresses using plane stress equations
for t in range(ThermalStrain.shape[0]):
    # Calculate stresses using Hooke's law for plane stress
    stress_xx[t] = (E / (1 - poisson**2)) * (DICExx[t] + poisson * DICEyy[t])
    stress_yy[t] = (E / (1 - poisson**2)) * (DICEyy[t] + poisson * DICExx[t])
    
    # Calculate von Mises stress (equivalent stress)
    stress_von_mises[t] = np.sqrt(stress_xx[t]**2 + stress_yy[t]**2 - stress_xx[t] * stress_yy[t])

# Calculate mean and maximum stresses
mean_stress_xx = np.nanmean(stress_xx, axis=0)
max_stress_xx = np.nanmax(stress_xx, axis=0)
mean_stress_yy = np.nanmean(stress_yy, axis=0)
max_stress_yy = np.nanmax(stress_yy, axis=0)
mean_stress_von_mises = np.nanmean(stress_von_mises, axis=0)
max_stress_von_mises = np.nanmax(stress_von_mises, axis=0)

# Plot stress fields
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot stress at point (0,0) over time
axes[0, 0].plot(time_points, stress_xx[:, 0, 0]/1e6, 'r-', label='σxx')
axes[0, 0].plot(time_points, stress_yy[:, 0, 0]/1e6, 'g-', label='σyy')
axes[0, 0].plot(time_points, stress_von_mises[:, 0, 0]/1e6, 'b-', label='von Mises')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Stress (MPa)')
axes[0, 0].set_title('Stress vs Time at Point (0,0)')
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot von Mises stress for highest strain points
if len(high_strain_points[0]) > 0:
    for idx in top_5_indices:
        row, col = high_strain_points[0][idx], high_strain_points[1][idx]
        axes[0, 1].plot(time_points, stress_von_mises[:, row, col]/1e6, 
                      label=f'Point ({row},{col}), Max={np.nanmax(stress_von_mises[:, row, col])/1e6:.1f} MPa')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('von Mises Stress (MPa)')
    axes[0, 1].set_title('von Mises Stress vs Time at High Strain Points')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

# Create stress field heat maps for final time step
t_final = -1  # Use the last time step
im1 = axes[1, 0].imshow(stress_xx[t_final]/1e6, cmap='hot')
axes[1, 0].set_title('σxx Stress Field (MPa)')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(stress_von_mises[t_final]/1e6, cmap='hot')
axes[1, 1].set_title('von Mises Stress Field (MPa)')
plt.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname('/Users/jayron/Downloads/PHM-Masters/Paper_Data_Set'), 'stress_analysis.svg'), bbox_inches='tight')
plt.show()

# Print stress summary
print("\nStress Analysis Summary:")
print(f"Maximum σxx: {np.nanmax(stress_xx)/1e6:.2f} MPa")
print(f"Maximum σyy: {np.nanmax(stress_yy)/1e6:.2f} MPa")
print(f"Maximum von Mises stress: {np.nanmax(stress_von_mises)/1e6:.2f} MPa")
print(f"Yield strength of tungsten (typical): ~1000 MPa")
print(f"Safety factor: {1000/np.nanmax(stress_von_mises)*1e6:.2f}")




