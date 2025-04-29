# Remaining Useful Life (RUL) Estimation Workflow

This repository contains a comprehensive Jupyter notebook workflow for estimating the Remaining Useful Life (RUL) of tungsten components based on Digital Image Correlation (DIC) strain measurements.

## Overview

The workflow demonstrates a complete process for estimating component fatigue life and remaining useful life, including:

1. Loading and processing strain data from DIC measurements
2. Visualizing strain patterns to identify critical regions
3. Converting strain to stress using material properties
4. Calculating principal strains for fatigue analysis
5. Implementing rainflow cycle counting for fatigue assessment
6. Estimating RUL using the Manson-Coffin relationship and Miner's rule
7. Visualizing RUL curves and providing recommendations

## Notebook Contents

The Jupyter notebook (`RUL_Estimation_Workflow.ipynb`) in the `Python scripts` directory is structured as follows:

- **Introduction**: Overview of the RUL estimation process and its significance
- **Setup and Imports**: Required libraries and modules for the analysis
- **Data Loading**: Processing DIC strain measurements for analysis
- **Initial Strain Visualization**: Understanding strain patterns and identifying critical regions
- **Stress Analysis**: Converting strain to stress and calculating safety factors
- **Principal Strain Analysis**: Determining principal and shear strains for fatigue analysis
- **Rainflow Cycle Counting**: Identifying and counting strain cycles for fatigue assessment
- **RUL Estimation**: Calculating remaining useful life and visualizing decay curves
- **Conclusions and Recommendations**: Interpreting results for maintenance decisions

## How to Use

1. Open the notebook using Jupyter:
   ```
   cd "/Users/jayron/Downloads/Paper_Data_Set/Python scripts"
   jupyter notebook RUL_Estimation_Workflow.ipynb
   ```

2. Run the cells sequentially to see the complete workflow from data loading to RUL estimation.

3. The notebook includes detailed explanations between code cells to help understand each step of the process.

4. The final visualization shows RUL curves for both principal strain and shear strain, providing a comprehensive view of component health.

## Key Features

- **Rainflow Cycle Counting**: Accurately identifies strain cycles in irregular loading patterns
- **Dual Strain Analysis**: Compares RUL based on both principal and shear strain
- **Adaptive RUL Calculation**: Adjusts to different strain types with appropriate parameter settings
- **Visual Insights**: Multiple visualizations of strain, stress, and RUL decay
- **Recommendations**: Practical guidance for maintenance decision-making

## Requirements

The notebook requires the following Python packages:
- numpy
- matplotlib
- rainflow

Additionally, it relies on the custom modules included in the Python scripts directory:
- data_loader.py
- fatigue_analysis.py
- plotter.py

## Example Results

The analysis typically produces:
- Initial RUL values that represent the expected lifecycle without damage
- Final RUL values that account for accumulated fatigue damage
- Life used percentage that indicates how much of the component's fatigue life has been consumed
- RUL decay curves that show how remaining life decreases over time

For tungsten components analyzed in this workflow, the principal strain typically shows a higher rate of life consumption compared to shear strain, making it the critical factor for maintenance planning.

## Author

Jayron Sandhu 