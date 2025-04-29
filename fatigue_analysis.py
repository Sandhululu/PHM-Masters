# -*- coding: utf-8 -*-
"""
Fatigue analysis module for RUL estimation from strain DIC measurement
"""

import numpy as np
import rainflow

def identify_cycles(strain_signal, is_shear_strain=False):
    """Identify cycles in a strain signal using rainflow analysis
    
    Args:
        strain_signal: 1D array of strain values
        is_shear_strain: Flag to indicate if this is shear strain data (default: False)
    
    Returns:
        numpy.ndarray: Array of cycle data with format [(range, mean, count, i_start, i_end), ...]
    """
    # Clean signal by removing NaNs
    signal_clean = np.copy(strain_signal)
    mask = ~np.isnan(signal_clean)
    if not np.any(mask):
        print(f"Warning: All NaN values in signal")
        return np.array([])
    
    # Interpolate missing data
    indices = np.arange(len(signal_clean))
    signal_clean = np.interp(indices, indices[mask], signal_clean[mask])
    
    # Extract and process cycles using rainflow algorithm
    try:
        cycles = rainflow.count_cycles(signal_clean)
    except Exception as e:
        print(f"Error with rainflow counting: {e}")
        return np.array([])
    
    if not cycles:
        print(f"No cycles found")
        return np.array([])
    
    # Convert the cycle data to array with consistent format
    cycles_array = []
    
    # Process each cycle
    for c in cycles:
        if len(c) == 5:  # Full format (range, mean, count, i_start, i_end)
            cycles_array.append(c)
        elif len(c) == 4:  # Alternative format (range, mean, count, i_start)
            rng, mean, count, i_start = c
            cycles_array.append((rng, mean, count, i_start, i_start + 1))
        elif len(c) == 3:  # Basic format (range, mean, count)
            rng, mean, count = c
            cycles_array.append((rng, mean, count, 0, 0))
        elif len(c) == 2:  # Simple format (range, count)
            rng, count = c
            cycles_array.append((rng, 0, count, 0, 0))
    
    return np.array(cycles_array) 