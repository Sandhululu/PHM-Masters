# -*- coding: utf-8 -*-
"""
Strain and stress calculation module for RUL estimation

@author: Jayron Sandhu
"""

import numpy as np

def calculate_principal_strains(ThermalStrain, DICExx, DICEyy):
    """Calculate principal strains from raw strain data
    
    Args:
        ThermalStrain, DICExx, DICEyy: Strain data arrays
        
    Returns:
        tuple: Major principal strain, minor principal strain, and max shear strain arrays
    """
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
    """Calculate stress components from strain data
    
    Args:
        DICExx, DICEyy: Strain data arrays
        E: Young's modulus
        poisson: Poisson's ratio
        ThermalStrain: Thermal strain array for sizing
        
    Returns:
        tuple: Stress arrays (xx, yy, von Mises)
    """
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
    """Find locations of extreme strain values with safety checks
    
    Args:
        major_principal_strain, minor_principal_strain, max_shear_strain: Strain arrays
        
    Returns:
        tuple: Locations of maximum principal, minimum principal, and maximum shear strain
    """
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

def print_analysis_summary(stress_xx, stress_yy, stress_von_mises, 
                          major_principal_strain, minor_principal_strain, max_shear_strain,
                          max_principal_loc, min_principal_loc, max_shear_loc):
    """Print summary of stress and strain analysis
    
    Args:
        stress_xx, stress_yy, stress_von_mises: Stress data arrays
        major_principal_strain, minor_principal_strain, max_shear_strain: Strain data arrays
        max_principal_loc, min_principal_loc, max_shear_loc: Locations of extreme strain values
    """
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