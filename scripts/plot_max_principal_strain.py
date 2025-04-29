import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the Python scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Python scripts'))

from data_loader import load_all_data
from strain_calculator import calculate_principal_strains

def plot_max_principal_strain():
    """Plot the time history of the maximum principal strain point"""
    # Load data
    print("Loading strain data...")
    (AverageExx, DICExx, AverageEyy, DICEyy, ThermalStrain, 
     time_points, mean_strain, std_strain, max_strain, high_strain_points) = load_all_data()
    
    # Calculate principal strains
    print("Calculating principal strains...")
    major_principal_strain, minor_principal_strain, max_shear_strain = calculate_principal_strains(
        ThermalStrain, DICExx, DICEyy)
    
    # Find the point with maximum principal strain
    print("Finding maximum principal strain point...")
    max_strain_idx = np.nanargmax(np.nanmax(major_principal_strain, axis=0))
    
    # Get the strain time history at this point
    print("Extracting strain time history...")
    strain_signal = major_principal_strain[:, max_strain_idx]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, strain_signal, 'b-', linewidth=1.5)
    plt.title(f'Major Principal Strain Time History at Point {max_strain_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Major Principal Strain (ε)')
    plt.grid(True, alpha=0.7)
    
    # Add statistics to the plot
    stats_text = f'Max Strain: {np.nanmax(strain_signal):.2e}\n' \
                 f'Min Strain: {np.nanmin(strain_signal):.2e}\n' \
                 f'Mean Strain: {np.nanmean(strain_signal):.2e}\n' \
                 f'Strain Range: {np.nanmax(strain_signal) - np.nanmin(strain_signal):.2e}'
    
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('max_principal_strain_time_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_max_principal_strain() 