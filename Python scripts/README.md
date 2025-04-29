# Remaining Useful Life (RUL) Estimation Tool

A Python-based tool for estimating the Remaining Useful Life (RUL) of components using strain Digital Image Correlation (DIC) measurements.

## Overview

This project provides a comprehensive toolset for analyzing strain data from DIC measurements, calculating principal strains and stresses, performing fatigue analysis, and estimating the remaining useful life of components. The code is particularly focused on tungsten components but can be adapted for other materials.

## Project Structure

The codebase has been modularized for better organization, maintainability, and reusability:

- `main.py`: The entry point that orchestrates the full analysis pipeline
- `data_loader.py`: Handles loading and preprocessing strain data from CSV files
- `strain_calculator.py`: Performs strain and stress calculations
- `fatigue_analysis.py`: Implements cycle counting and fatigue life estimation
- `plotter.py`: Contains visualization functions for all analysis results

## Installation

### Prerequisites

- Python 3.7+
- Required Python packages:
  - numpy
  - matplotlib
  - pandas
  - scipy
  - tkinter (for file dialogs)

### Setup

1. Clone this repository or download the source code
2. Install the required packages:

```bash
pip install numpy matplotlib pandas scipy
```

## Usage

To run the full analysis pipeline:

```bash
python main.py
```

This will:
1. Load strain data from CSV files
2. Calculate principal strains and stresses
3. Perform fatigue analysis
4. Estimate the RUL of the component
5. Generate visualizations at each step

### Customization

You can modify the following parameters in `main.py`:

- `cycle_multiplier`: Factor to multiply the detected cycles for long-term analysis (default: 10)
- `material`: Material type for fatigue properties (default: 'tungsten')

## Module Descriptions

### data_loader.py

Handles all data-related operations including:
- Loading strain data from CSV files (Exx, Eyy, and thermal strain)
- Padding arrays to ensure consistent dimensions
- Generating time points for analysis
- Identifying high strain points for focused analysis

### strain_calculator.py

Performs strain and stress calculations:
- Calculates major and minor principal strains and maximum shear strain
- Calculates stress components (σxx, σyy, and von Mises stress)
- Identifies locations of extreme strain values
- Provides analysis summaries of stress and strain results

### fatigue_analysis.py

Implements fatigue analysis functionality:
- Identifies peaks and valleys in strain signals
- Analyzes fatigue cycles in strain histories
- Estimates fatigue life based on strain ranges
- Calculates damage accumulation
- Estimates remaining useful life

### plotter.py

Contains all visualization functions:
- Initial strain analysis plots
- Stress analysis plots
- Principal strain analysis plots
- Fatigue analysis plots with cycle information
- RUL estimation plots with text annotations

## Output

The analysis produces the following visualizations:
- `strain_analysis.svg`: Initial strain analysis at reference and high-strain points
- `stress_analysis.svg`: Stress plots showing stress components at key points
- `principal_strain_analysis.svg`: Principal strain plots at key locations
- `fatigue_analysis.svg`: Fatigue analysis showing strain histories with cycle information
- `rul_estimation.svg`: RUL estimation plots showing remaining life vs. cycles experienced

Additionally, detailed analysis results are printed to the console.

## Author

Jayron Sandhu

## License

This project is provided for educational and research purposes.

## References

For more information on the techniques used in this project:
- ASTM E606 for strain-controlled fatigue testing
- ASTM E1049 for cycle counting methods
- Miner's rule for cumulative damage calculation 