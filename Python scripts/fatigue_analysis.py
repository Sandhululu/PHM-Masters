# -*- coding: utf-8 -*-
"""
Fatigue analysis module for RUL estimation from strain DIC measurement

@author: Jayron Sandhu
"""

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
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
    
    # Examine the first cycle to determine format
    first_cycle = next(iter(cycles))
    
    # Determines the format of the cycle data
    if hasattr(first_cycle, '__len__'):
        for c in cycles:
            if len(c) == 2:  # Simple format (range, count)
                rng, count = c
                mean, i_start, i_end = 0, 0, 0
                cycles_array.append((rng, mean, count, i_start, i_end))
            elif len(c) == 5:  # Full format (range, mean, count, i_start, i_end)
                rng, mean, count, i_start, i_end = c
                cycles_array.append((rng, mean, count, i_start, i_end))
            elif len(c) == 4:  # Alternative format sometimes used
                rng, mean, count, _ = c
                cycles_array.append((rng, mean, count, 0, 0))
    else:
        # If cycles are returned as single values, assume they are ranges with count=1
        for c in cycles:
            cycles_array.append((c, 0, 1, 0, 0))
    
    if not cycles_array:
        print("No valid cycle data could be extracted")
        return np.array([])

    print(f"Cycles array shape: {np.array(cycles_array).shape}")
    print(f"Cycles array: {cycles_array}")
    
    return np.array(cycles_array)

def analyze_fatigue(strain_cycles):
    """Analyze fatigue cycles to estimate damage
    
    Args:
        strain_cycles: Array of cycle data from identify_cycles
    
    Returns:
        dict: Fatigue analysis results
    """
    # Check if we have valid cycle data
    if len(strain_cycles) == 0:
        print("No cycle data available for fatigue analysis")
        return {
            'cycles': np.array([]),
            'damages': np.array([]),
            'cumulative_damage': np.array([]),
            'cycles_array': np.array([])
        }
    
    # Material properties for fatigue analysis
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c, safety_factor = -0.12, -0.7, 1.0
    
    # Extract ranges and counts
    if strain_cycles.shape[1] >= 3:  # Make sure we have at least 3 columns
        ranges = strain_cycles[:, 0]  # First column is range
        counts = strain_cycles[:, 2]  # Third column is count
    else:
        # Fallback in case we have simple format
        ranges = strain_cycles[:, 0]
        counts = np.ones_like(ranges)
    
    # Sort cycles by range (amplitude) in descending order
    sorted_indices = np.argsort(ranges)[::-1]
    sorted_ranges = ranges[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Calculate cycles to failure for each strain range
    N_f_cycles = []
    for strain_range in sorted_ranges:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 10, 1000)
        
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life (no safety factor applied)
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))] / safety_factor
        N_f_cycles.append(min(N_f, 1e6))  # Cap at 1 million cycles
    
    # Calculate damage per cycle using Miner's rule
    damage_per_cycle = [count/N_f for count, N_f in zip(sorted_counts, N_f_cycles)]
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Return comprehensive fatigue analysis results
    return {
        'cycles': sorted_ranges,
        'counts': sorted_counts,
        'N_f_cycles': np.array(N_f_cycles),
        'damages': np.array(damage_per_cycle),
        'cumulative_damage': cumulative_damage,
        'cycles_array': np.arange(len(sorted_ranges))
    }

def estimate_fatigue_life(fatigue_results, cycle_multiplier=1, force_shear=False):
    """Estimate remaining useful life (RUL) based on fatigue analysis
    
    This implementation uses proper rainflow cycle counting and Miner's rule
    to calculate cumulative damage and remaining useful life based on
    material properties and fatigue characteristics.
    
    Args:
        fatigue_results: Fatigue analysis results from analyze_fatigue
        cycle_multiplier: Multiplier for number of cycles (default: 1)
        
    
    Returns:
        tuple: (rul_values, cycles_experienced)
    """
    # Extract cycle data from fatigue_results
    cycles = fatigue_results.get('cycles', np.array([]))
    counts = fatigue_results.get('counts', np.array([]))
    N_f_cycles = fatigue_results.get('N_f_cycles', np.array([]))
    damages = fatigue_results.get('damages', np.array([]))
    
    # Check if we have valid cycle data
    if len(cycles) == 0 or len(N_f_cycles) == 0:
        print("No valid cycle data available for RUL estimation")
        return np.array([0]), np.array([0])
    
    # Apply cycle multiplier to counts
    scaled_counts = counts * cycle_multiplier
    
    # Calculate damage metrics
    damage_per_cycle = scaled_counts / N_f_cycles
    cumulative_damage = np.cumsum(damage_per_cycle)
    
    # Calculate average damage rate (used for RUL calculation)
    total_cycles = np.sum(scaled_counts)
    avg_damage_rate = np.sum(damage_per_cycle) / total_cycles if total_cycles > 0 else 0
    
    # Calculate cycles experienced
    cycles_experienced = np.cumsum(scaled_counts)
    
    # Initial RUL (before any cycles)
    initial_rul = 1 / avg_damage_rate if avg_damage_rate > 0 else float('inf')
    
    # Cap unreasonably high RUL values
    max_reasonable_life = 100000
    if initial_rul > max_reasonable_life:
        print(f"Note: Capping very high initial RUL ({initial_rul:.1f}) to {max_reasonable_life}")
        initial_rul = max_reasonable_life
    
    # Calculate RUL for each point using remaining damage capacity
    # The theoretical RUL is (1-D)/Ḋ, where D is cumulative damage and Ḋ is avg damage rate
    rul_values = np.maximum((1 - cumulative_damage) / avg_damage_rate if avg_damage_rate > 0 else np.ones_like(cumulative_damage) * initial_rul, 0)
    
    # Create a smooth RUL curve with interpolation
    if len(cycles_experienced) > 1:
        # Generate 100 points for a smooth curve
        interp_cycles = np.linspace(0, cycles_experienced[-1], 100)
        interp_rul = np.interp(interp_cycles, cycles_experienced, rul_values)
        
        # Add point at cycle 0 for initial RUL
        cycles_plot = np.insert(interp_cycles, 0, 0)
        rul_plot = np.insert(interp_rul, 0, initial_rul)
    else:
        # If only one cycle point, create a simple two-point curve
        cycles_plot = np.array([0, cycle_multiplier])
        rul_plot = np.array([initial_rul, initial_rul * 0.95])  # 5% decrease for visualization
    
    # Calculate and print RUL metrics
    final_percentage_used = (1 - (rul_plot[-1] / rul_plot[0])) * 100
    print(f"\nRUL Analysis:")
    print(f"  Total cycles: {total_cycles:.1f} (with multiplier {cycle_multiplier})")
    print(f"  Initial RUL: {rul_plot[0]:.1f} cycles")
    print(f"  Final RUL: {rul_plot[-1]:.1f} cycles")
    print(f"  Life used: {final_percentage_used:.2f}%")
    
    return np.array(rul_plot), np.array(cycles_plot) 