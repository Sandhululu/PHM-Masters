
# -*- coding: utf-8 -*-
"""
Temperature from strain DIC measurement

@author: Adel Tayeb
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

    dirDIC = '/Users/jayron/Downloads/Paper_Data_Set/DIC data/withoutCoil/exx'

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

    #SpatialStd = np.array([np.nanstd(arr) for arr in DICData])
    Average = np.nanmean(DICData, axis=0)
    #TempStd = np.nanstd(DICData, axis=0)
    return Average, DICData # change accordingly!!
AverageX, DICX = read_DIC_data_from_csv()
AverageY, DICY = read_DIC_data_from_csv()
AverageExx, DICExx = read_DIC_data_from_csv()
AverageEyy, DICEyy = read_DIC_data_from_csv()


ThermalStrain = (DICExx+DICEyy)/2

AverageThermalStrain = np.zeros(shape=(AverageX.shape[0],AverageX.shape[1]))
for k in range(25, 310):
    AverageThermalStrain = AverageThermalStrain + ThermalStrain[k,:,:]/285

TempMap = np.zeros(shape=(AverageX.shape[0],AverageX.shape[1]))

## STE = STE_coef0 + STE_coef1 * Temp
T = Symbol('T')
STE_coef0 = 4.496e-6
STE_coef1 = 4.6e-10

for i in range(0, AverageX.shape[0]):
    for j in range(0, AverageX.shape[1]):
        if not np.isnan(AverageThermalStrain[i,j]):
            a = solve(T-150-AverageThermalStrain[i,j]/(STE_coef0 + STE_coef1 * T), T)
            if len(a) == 1:
                TempMap[i,j] = a[0]
            else:
                TempMap[i,j] = a[1]
#for graphics purposes          
TempMap[TempMap ==0] =  np.nan
        

# the path to your csv file directory
root = tkinter.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()

# cwd = os.getcwd()
# tempdir = filedialog.askdirectory(parent=root, initialdir=cwd, title='Please select a directory')
# # Get csv containing TC data (file Temperature_19kW.csv)
# TCfull_path = filedialog.askopenfilename(parent=root,
#                                        initialdir=cwd,
#                                        title="Select csv file containing TC data ")
# TC_path, TC_file = os.path.split(TCfull_path)
# TC_Data = np.genfromtxt(TCfull_path, delimiter=',')

# NOTE: we did not remove half of a strain window for these data to get the whole map
X= (AverageX - AverageX[0,0])
Y= -(AverageY - AverageY[-1,0])

StrainTC02 = np.nanmean(ThermalStrain[:,12:14,13:15], axis =(1,2))
StrainTC04 = np.nanmean(ThermalStrain[:,38:40,25:27], axis =(1,2))
StrainTC06 = np.nanmean(ThermalStrain[:,78:80,23:26], axis =(1,2))
time = np.linspace(0, 70.8, 354)

# Temperature map

fig, ax = plt.subplots()
im = ax.imshow(TempMap[:,0:-1],extent=(X[-2,0], X[0,-2], Y[-2,0], Y[0,-2]), cmap='hot')#, vmin = 0, vmax =5,vmin = minU, vmax =maxU
plt.colorbar(im, orientation='vertical')
plt.title(r' Temperature $~[^{\circ} C]$' )
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
fig.savefig(os.path.join(TC_path,'TempMap.svg'), bbox_inches='tight')
plt.show()

# TC strain traces 
# TC02 
fig, ax = plt.subplots()
plt.plot(time-0.8,1e6*StrainTC02, color = (1, 0, 0, 1), label='DIC')
plt.fill_between(time-0.8,1e6*StrainTC02+83.16,1e6*StrainTC02-83.16, color = (1, 0, 0, 0.2))
plt.plot(TC_Data[:,0],1e6*TC_Data[:,4], color = (0, 0, 1, 1), label='TC02')
plt.xlabel('Time [s]', labelpad = 15)
plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad = 15)
plt.grid()
plt.legend()
fig.savefig(os.path.join(TC_path,'TC02.svg'), bbox_inches='tight')
plt.show()

# TC04 
fig, ax = plt.subplots()
plt.plot(time-0.8,1e6*StrainTC04, color = (1, 0, 0, 1), label='DIC')
plt.fill_between(time-0.8,1e6*StrainTC04+83.16,1e6*StrainTC04-83.16, color = (1, 0, 0, 0.2))
plt.plot(TC_Data[:,0],1e6*TC_Data[:,5], color = (0, 0, 1, 1), label='TC04')
plt.xlabel('Time [s]', labelpad = 15)
plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad = 15)
plt.grid()
plt.legend()
fig.savefig(os.path.join(TC_path,'TC04.svg'), bbox_inches='tight')
plt.show()

# TC06 
fig, ax = plt.subplots()
plt.plot(time-0.8,1e6*StrainTC06, color = (1, 0, 0, 1), label='DIC')
plt.fill_between(time-0.8,1e6*StrainTC06+83.16,1e6*StrainTC06-83.16, color = (1, 0, 0, 0.2))
plt.plot(TC_Data[:,0],1e6*TC_Data[:,6], color = (0, 0, 1, 1), label='TC06')
plt.xlabel('Time [s]', labelpad = 15)
plt.ylabel(r'$\overline{(E_{xx}+E_{yy})/2}~[\mu \varepsilon]$', labelpad = 15)
plt.grid()
plt.legend()
fig.savefig(os.path.join(TC_path,'TC06.svg'), bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
plt.plot(TC_Data[:,0],TC_Data[:,1], color = (0, 0, 1, 1), label='TC 02')
plt.plot(TC_Data[:,0],TC_Data[:,2], color = (1, 0, 0, 1), label='TC 04')
plt.plot(TC_Data[:,0],TC_Data[:,3],'g', label='TC 06')
plt.xlabel('Time [s]', labelpad = 15)
plt.ylabel(r' Temperature $~[^{\circ} C]$', labelpad = 15)
plt.legend()
plt.grid()
fig.savefig(os.path.join(TC_path,'TCTraceExp.svg'), bbox_inches='tight')
plt.show()