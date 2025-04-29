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
    dirDIC = f'DIC data/withoutCoil/{strain_type}'
    csv_files = sorted(glob.glob(os.path.join(dirDIC, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dirDIC}")
        
    arrays = [pd.read_csv(f, header=None).values for f in csv_files]
    
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
    """Load all strain data from the DIC data directory"""
    # Load exx and eyy strain data
    AverageExx, DICExx = read_strain_data('exx')
    AverageEyy, DICEyy = read_strain_data('eyy')
    
    # Calculate thermal strain
    ThermalStrain = (DICExx + DICEyy) / 2
    
    # Generate time points (0.2s intervals)
    time_points = np.arange(0, DICExx.shape[0] * 0.2, 0.2)
    
    # Calculate principal strains
    exx = DICExx
    eyy = DICEyy
    exy = np.zeros_like(exx)  # Assuming zero shear strain
    
    # Calculate principal strains
    avg = (exx + eyy) / 2
    diff = (exx - eyy) / 2
    radius = np.sqrt(diff**2 + exy**2)
    
    major_principal_strain = avg + radius
    minor_principal_strain = avg - radius
    max_shear_strain = 2 * radius
    
    return {
        'exx': exx,
        'eyy': eyy,
        'thermal_strain': ThermalStrain,
        'time_points': time_points,
        'major_principal_strain': major_principal_strain,
        'minor_principal_strain': minor_principal_strain,
        'max_shear_strain': max_shear_strain
    }

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