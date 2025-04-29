#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature from strain DIC measurement - Fixed version with direct file paths

@author: Adel Tayeb (modified)
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol

def read_DIC_data_from_csv(strain_type):
    """Load strain data from CSV files
    
    Args:
        strain_type (str): Type of strain data to load ('exx', 'eyy', 'x', or 'y')
        
    Returns:
        tuple: (Average data array, Complete DIC data array)
    """
    # Direct path to DIC data
    base_dir = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil'
    dirDIC = os.path.join(base_dir, strain_type)
    
    print(f"Loading data from: {dirDIC}")
    
    # Check if directory exists
    if not os.path.exists(dirDIC):
        print(f"ERROR: Directory does not exist: {dirDIC}")
        return np.array([[]]), np.array([[[]]])
    
    # Get csv files
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    
    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {dirDIC}")
        return np.array([[]]), np.array([[[]]])
    
    # List to store 2D arrays
    arrays = []
    
    # Read each CSV file into a 2D NumPy array and add to the list
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, header=None)  # Read the CSV file without headers
            arrays.append(df.values)  # Convert DataFrame to NumPy array
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if len(arrays) == 0:
        print("ERROR: No valid data loaded")
        return np.array([[]]), np.array([[[]]])
    
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
    Average = np.nanmean(DICData, axis=0)
    
    print(f"Successfully loaded {len(arrays)} files from {strain_type}")
    return Average, DICData

def main():
    """Main function with direct file paths"""
    print("\n=== Temperature Map Generation ===\n")
    
    # Load DIC data for different strain types
    print("\nLoading strain data...")
    AverageX, DICX = read_DIC_data_from_csv('x')
    AverageY, DICY = read_DIC_data_from_csv('y')
    AverageExx, DICExx = read_DIC_data_from_csv('exx')
    AverageEyy, DICEyy = read_DIC_data_from_csv('eyy')
    
    if AverageExx.size <= 1 or AverageEyy.size <= 1:
        print("ERROR: Failed to load required strain data. Exiting.")
        return 1
    
    print("\nCalculating thermal strain...")
    ThermalStrain = (DICExx + DICEyy) / 2
    
    # Calculate average thermal strain in the steady state region
    AverageThermalStrain = np.zeros(shape=(AverageX.shape[0], AverageX.shape[1]))
    start_frame, end_frame = 25, 310  # Steady state region
    
    for k in range(start_frame, end_frame):
        AverageThermalStrain = AverageThermalStrain + ThermalStrain[k, :, :] / (end_frame - start_frame)
    
    print("\nCalculating temperature map...")
    TempMap = np.zeros(shape=(AverageX.shape[0], AverageX.shape[1]))
    
    # STE = STE_coef0 + STE_coef1 * Temp
    T = Symbol('T')
    STE_coef0 = 4.496e-6  # Strain-temperature coefficient
    STE_coef1 = 4.6e-10   # Temperature dependent coefficient
    
    # Convert strain to temperature using the coefficients
    for i in range(0, AverageX.shape[0]):
        for j in range(0, AverageX.shape[1]):
            if not np.isnan(AverageThermalStrain[i, j]):
                try:
                    a = solve(T - 150 - AverageThermalStrain[i, j] / (STE_coef0 + STE_coef1 * T), T)
                    if len(a) == 1:
                        TempMap[i, j] = a[0]
                    else:
                        TempMap[i, j] = a[1]  # Take the physically meaningful solution
                except Exception as e:
                    print(f"Error solving temperature at point ({i},{j}): {e}")
                    TempMap[i, j] = np.nan
    
    # For graphics purposes, replace zeros with NaN
    TempMap[TempMap == 0] = np.nan
    
    # Direct path to thermocouple data - updated with correct path from our search
    tc_data_path = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/Temperature_19kW.csv'
    output_dir = '/Users/jayron/Downloads/Paper_Data_Set'
    
    try:
        # Load thermocouple data
        TC_Data = np.genfromtxt(tc_data_path, delimiter=',')
        print(f"\nLoaded thermocouple data from {tc_data_path}")
    except Exception as e:
        print(f"\nWARNING: Could not load thermocouple data: {e}")
        TC_Data = np.array([])  # Empty array as fallback
    
    # Calculate X, Y coordinates for plotting
    X = (AverageX - AverageX[0, 0])
    Y = -(AverageY - AverageY[-1, 0])
    
    # Extract strain at thermocouple locations
    StrainTC02 = np.nanmean(ThermalStrain[:, 12:14, 13:15], axis=(1, 2))
    StrainTC04 = np.nanmean(ThermalStrain[:, 38:40, 25:27], axis=(1, 2))
    StrainTC06 = np.nanmean(ThermalStrain[:, 78:80, 23:26], axis=(1, 2))
    time = np.linspace(0, 70.8, 354)  # Time points
    
    print("\nCreating temperature map visualization...")
    # Temperature map
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(TempMap[:, 0:-1], extent=(X[-2, 0], X[0, -2], Y[-2, 0], Y[0, -2]), cmap='inferno')
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label(r'Temperature $~[^{\circ} C]$', fontsize=12)
    plt.title(r'Temperature Map from Thermal Strain', fontsize=14)
    plt.xlabel('X (mm)', fontsize=12)
    plt.ylabel('Y (mm)', fontsize=12)
    
    # Save the temperature map
    temp_map_path = os.path.join(output_dir, 'TempMap.png')
    fig.savefig(temp_map_path, bbox_inches='tight', dpi=300)
    print(f"Temperature map saved to: {temp_map_path}")
    plt.show()
    
    # If thermocouple data is available, create comparison plots
    if TC_Data.size > 0:
        print("\nCreating thermocouple comparison plots...")
        # TC02 strain trace
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(time - 0.8, 1e6 * StrainTC02, color=(1, 0, 0, 1), label='DIC')
        plt.fill_between(time - 0.8, 1e6 * StrainTC02 + 83.16, 1e6 * StrainTC02 - 83.16, color=(1, 0, 0, 0.2))
        plt.plot(TC_Data[:, 0], 1e6 * TC_Data[:, 4], color=(0, 0, 1, 1), label='TC02')
        plt.xlabel('Time [s]', labelpad=15, fontsize=12)
        plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad=15, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.savefig(os.path.join(output_dir, 'TC02.png'), bbox_inches='tight', dpi=300)
        plt.show()
        
        # TC04 strain trace
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(time - 0.8, 1e6 * StrainTC04, color=(1, 0, 0, 1), label='DIC')
        plt.fill_between(time - 0.8, 1e6 * StrainTC04 + 83.16, 1e6 * StrainTC04 - 83.16, color=(1, 0, 0, 0.2))
        plt.plot(TC_Data[:, 0], 1e6 * TC_Data[:, 5], color=(0, 0, 1, 1), label='TC04')
        plt.xlabel('Time [s]', labelpad=15, fontsize=12)
        plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad=15, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.savefig(os.path.join(output_dir, 'TC04.png'), bbox_inches='tight', dpi=300)
        plt.show()
        
        # TC06 strain trace
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(time - 0.8, 1e6 * StrainTC06, color=(1, 0, 0, 1), label='DIC')
        plt.fill_between(time - 0.8, 1e6 * StrainTC06 + 83.16, 1e6 * StrainTC06 - 83.16, color=(1, 0, 0, 0.2))
        plt.plot(TC_Data[:, 0], 1e6 * TC_Data[:, 6], color=(0, 0, 1, 1), label='TC06')
        plt.xlabel('Time [s]', labelpad=15, fontsize=12)
        plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad=15, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.savefig(os.path.join(output_dir, 'TC06.png'), bbox_inches='tight', dpi=300)
        plt.show()
        
        # Temperature trace comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(TC_Data[:, 0], TC_Data[:, 1], color=(0, 0, 1, 1), label='TC 02')
        plt.plot(TC_Data[:, 0], TC_Data[:, 2], color=(1, 0, 0, 1), label='TC 04')
        plt.plot(TC_Data[:, 0], TC_Data[:, 3], 'g', label='TC 06')
        plt.xlabel('Time [s]', labelpad=15, fontsize=12)
        plt.ylabel(r'Temperature $~[^{\circ} C]$', labelpad=15, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, 'TCTraceExp.png'), bbox_inches='tight', dpi=300)
        plt.show()
    
    print("\nTemperature map analysis complete!")
    return 0

if __name__ == "__main__":
    main() 