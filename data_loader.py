# -*- coding: utf-8 -*-
"""
Data loading module for RUL estimation from strain DIC measurement

@author: Jayron Sandhu
"""

import glob
import os
import pandas as pd
import tkinter
from tkinter import filedialog
import numpy as np

def read_strain_data(strain_type='exx'):
    """Load strain data from CSV files
    
    Args:
        strain_type (str): Type of strain data to load ('exx' or 'eyy')
        
    Returns:
        tuple: (Average data array, Complete DIC data array)
    """
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
    """Load and prepare all strain data
    
    Returns:
        tuple: All the processed strain data arrays and time points
    """
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

def print_statistical_summary(ThermalStrain, high_strain_points):
    """Print statistical summary of strain data
    
    Args:
        ThermalStrain: Thermal strain data array
        high_strain_points: Locations of high strain
    """
    print(f"\nStatistical Summary:\n"
          f"Number of high strain points found: {len(high_strain_points[0])}\n"
          f"Global mean strain: {np.nanmean(ThermalStrain):.2e}\n"
          f"Global max strain: {np.nanmax(ThermalStrain):.2e}\n"
          f"Global strain standard deviation: {np.nanstd(ThermalStrain):.2e}") 