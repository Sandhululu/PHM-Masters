import numpy as np

def calculate_principal_strains(exx, eyy):
    """Calculate principal strains from strain components
    
    Args:
        exx (ndarray): Array of xx strain components
        eyy (ndarray): Array of yy strain components
        
    Returns:
        tuple: (major_principal_strain, minor_principal_strain)
    """
    # Calculate average strain
    avg_strain = (exx + eyy) / 2
    
    # Calculate radius (shear strain)
    radius = np.sqrt(((exx - eyy) / 2)**2)
    
    # Calculate principal strains
    major_principal_strain = avg_strain + radius
    minor_principal_strain = avg_strain - radius
    
    return major_principal_strain, minor_principal_strain

def find_max_principal_strain_point(major_principal_strain):
    """Find the point with maximum principal strain
    
    Args:
        major_principal_strain (ndarray): Array of major principal strains
        
    Returns:
        tuple: (max_strain, point_index)
    """
    max_strain = np.nanmax(major_principal_strain)
    point_index = np.nanargmax(major_principal_strain)
    
    return max_strain, point_index

def get_strain_time_history(major_principal_strain, point_index):
    """Get the time history of strain at a specific point
    
    Args:
        major_principal_strain (ndarray): Array of major principal strains
        point_index (int): Index of the point to track
        
    Returns:
        ndarray: Time history of strain at the specified point
    """
    return major_principal_strain[:, point_index] 