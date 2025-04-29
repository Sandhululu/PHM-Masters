import numpy as np
from scipy import interpolate
import glob
import os
import pandas as pd
from sympy import Symbol, solve
import matplotlib.pyplot as plt

def fill_nan_with_interpolation(data):
    """Fill NaN values using linear interpolation"""
    if data.ndim == 2:
        x, y = np.where(~np.isnan(data))
        xy = np.column_stack((x, y))
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 3:  # Need at least 4 points for interpolation
            grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
            return interpolate.griddata(xy, valid_data, (grid_x, grid_y), method='linear')
    return data

def read_DIC_data_from_csv():
    # the path to your csv file directory
    dirDIC = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/exx'

    # get csv files of the from static images results 
    csv_files = glob.glob(os.path.join(dirDIC, '*.csv'))
    # List to store 2D arrays
    arrays = []

    # Read each CSV file into a 2D NumPy array and add to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)  # Read the CSV file without headers
        arr = df.values
        # Interpolate NaN values in each 2D slice
        arr_filled = fill_nan_with_interpolation(arr)
        arrays.append(arr_filled)  # Convert DataFrame to NumPy array

    # Determine the maximum dimensions
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)

    # Print array shapes and NaN counts before padding to help diagnose issues
    print("\nDiagnostic information:")
    print(f"Number of arrays loaded: {len(arrays)}")
    for i, arr in enumerate(arrays):
        nan_count = np.isnan(arr).sum()
        total_elements = arr.size
        nan_percentage = (nan_count / total_elements) * 100
        print(f"Array {i}: Shape {arr.shape}, NaN count: {nan_count} ({nan_percentage:.2f}% NaN)")

    # Filter out arrays that are completely NaN
    valid_arrays = []
    for i, arr in enumerate(arrays):
        if not np.isnan(arr).all():
            valid_arrays.append(arr)
        else:
            print(f"Warning: Array {i} contains only NaN values - skipping")
    
    if not valid_arrays:
        raise ValueError("All input arrays contain only NaN values")
        
    # Recalculate max dimensions after filtering
    arrays = valid_arrays
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)
    
    print(f"\nAfter filtering: {len(arrays)} valid arrays remaining")

    # Pad each array to match the maximum dimensions with NaN
    padded_arrays = []
    for arr in arrays:
        pad_height = max_rows - arr.shape[0]
        pad_width = max_cols - arr.shape[1]
        padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width)), 'constant', constant_values=np.nan)
        # Interpolate any remaining NaN values
        padded_arr = fill_nan_with_interpolation(padded_arr)
        padded_arrays.append(padded_arr)

    # Stack the padded arrays into a 3D array
    DICData = np.stack(padded_arrays)
    DICData = DICData[:,0,0]
    print("DICData shape:", DICData.shape)

    # Create time array based on 0.2s intervals
    time_points = np.arange(0, len(DICData) * 0.2, 0.2)  # time in seconds
    
    # Create strain vs time plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, DICData, 'b-', label='Strain')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Strain')
    plt.title('Strain vs Time at Point (0,0)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(dirDIC), 'strain_vs_time.svg'), bbox_inches='tight')
    plt.show()

    #SpatialStd = np.array([np.nanstd(arr) for arr in DICData])
    Average = np.nanmean(DICData, axis=0)
    #TempStd = np.nanstd(DICData, axis=0)
    return Average, DICData # change accordingly!!

AverageX, DICX = read_DIC_data_from_csv()
AverageY, DICY = read_DIC_data_from_csv()
AverageExx, DICExx = read_DIC_data_from_csv()
AverageEyy, DICEyy = read_DIC_data_from_csv()

print("\nCalculating thermal strain...")
ThermalStrain = (DICExx + DICEyy) / 2

# Initialize AverageThermalStrain with NaN
AverageThermalStrain = np.full((AverageX.shape[0], AverageX.shape[1]), np.nan)

# Calculate mean only for valid data points
valid_data = ThermalStrain[25:310]
valid_mask = ~np.isnan(valid_data).all(axis=0)
if valid_mask.any():
    AverageThermalStrain[valid_mask] = np.nanmean(valid_data[:, valid_mask], axis=0)

print(f"AverageThermalStrain shape: {AverageThermalStrain.shape}")
print(f"AverageThermalStrain NaN percentage: {(np.isnan(AverageThermalStrain).sum() / AverageThermalStrain.size) * 100:.2f}%")

TempMap = np.full((AverageX.shape[0], AverageX.shape[1]), np.nan)

## STE = STE_coef0 + STE_coef1 * Temp
T = Symbol('T')
STE_coef0 = 4.496e-6
STE_coef1 = 4.6e-10

# Only process non-NaN values
valid_indices = ~np.isnan(AverageThermalStrain)
for i, j in zip(*np.where(valid_indices)):
    a = solve(T-150-AverageThermalStrain[i,j]/(STE_coef0 + STE_coef1 * T), T)
    if len(a) == 1:
        TempMap[i,j] = a[0]
    else:
        TempMap[i,j] = a[1]

print(f"\nTemperature map statistics:")
print(f"TempMap NaN percentage: {(np.isnan(TempMap).sum() / TempMap.size) * 100:.2f}%")

# NOTE: we did not remove half of a strain window for these data to get the whole map