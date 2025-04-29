#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an average strain map without critical points
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the project
from data_loader import load_all_data
from strain_calculator import calculate_principal_strains

def create_average_strain_map(major_principal_strain, minor_principal_strain, DICExx, DICEyy):
    """Create average strain maps without marking critical points
    
    Args:
        major_principal_strain: 3D array of major principal strain data
        minor_principal_strain: 3D array of minor principal strain data 
        DICExx: 3D array of original exx strain data
        DICEyy: 3D array of original eyy strain data
        
    Returns:
        None (saves maps as files)
    """
    # Calculate average strain values across all time points
    avg_major = np.nanmean(major_principal_strain, axis=0)
    avg_minor = np.nanmean(minor_principal_strain, axis=0)
    avg_exx = np.nanmean(DICExx, axis=0)
    avg_eyy = np.nanmean(DICEyy, axis=0)
    
    # Create figure with 2x2 grid with increased height to prevent clipping
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Function to add colorbar
    def add_colorbar(im, ax, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label)
        return cbar
    
    # Plot 1: Major Principal Strain Map
    im1 = axes[0, 0].imshow(avg_major, cmap='viridis')
    axes[0, 0].set_title('Average Major Principal Strain', fontsize=12)
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    add_colorbar(im1, axes[0, 0], 'Strain')
    
    # Plot 2: Minor Principal Strain Map
    im2 = axes[0, 1].imshow(avg_minor, cmap='plasma')
    axes[0, 1].set_title('Average Minor Principal Strain', fontsize=12)
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    add_colorbar(im2, axes[0, 1], 'Strain')
    
    # Plot 3: Average Exx Strain Map
    im3 = axes[1, 0].imshow(avg_exx, cmap='cool')
    axes[1, 0].set_title('Average Exx Strain', fontsize=12)
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    add_colorbar(im3, axes[1, 0], 'Strain')
    
    # Plot 4: Average Eyy Strain Map
    im4 = axes[1, 1].imshow(avg_eyy, cmap='hot')
    axes[1, 1].set_title('Average Eyy Strain', fontsize=12)
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    add_colorbar(im4, axes[1, 1], 'Strain')
    
    # Add overall title
    plt.suptitle('Average Strain Maps for Tungsten Component', fontsize=16, fontweight='bold', y=0.98)
    
    # Increase spacing between subplots to prevent clipping
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Adjust layout with more space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figures
    plt.savefig(os.path.join(os.getcwd(), 'average_strain_maps.svg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), 'average_strain_maps.png'), bbox_inches='tight', dpi=300)
    
    print("Average strain maps saved as 'average_strain_maps.svg/png'")
    plt.show()

def main():
    """Main function to generate average strain maps"""
    print("\nLoading strain data...")
    data = load_all_data()
    
    # Check if data is dictionary or tuple (for backward compatibility)
    if isinstance(data, dict):
        # Use dictionary format
        ThermalStrain = data['thermal_strain']
        DICExx = data['exx']
        DICEyy = data['eyy']
        time_points = data['time_points']
    else:
        # Use tuple format for backward compatibility
        _, DICExx, _, DICEyy, ThermalStrain, time_points, _, _, _, _ = data
    
    print("\nCalculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    print("\nCreating average strain maps...")
    create_average_strain_map(major_principal_strain, minor_principal_strain, DICExx, DICEyy)
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 