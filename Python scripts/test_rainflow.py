#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the rainflow module to understand cycle mean values and position tracking
"""

import numpy as np
import rainflow
import matplotlib.pyplot as plt

def main():
    """Test the rainflow module with a simple signal"""
    print("\n=== Rainflow Module Testing ===\n")
    
    # Create a simple test signal with a non-zero mean and clear cycles
    time = np.linspace(0, 10, 1000)
    # Signal with a mean of 5.0 and amplitude varying between +/- 3
    signal = 5.0 + 3.0 * np.sin(2 * np.pi * time)
    # Add a smaller higher frequency component
    signal += 1.0 * np.sin(10 * np.pi * time)
    # Add a ramp to ensure non-zero mean
    signal += time * 0.2  
    
    # Plot the signal
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title('Test Signal with Non-Zero Mean')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('test_signal.png')
    
    print(f"Signal properties:")
    print(f"  Mean: {np.mean(signal):.2f}")
    print(f"  Min: {np.min(signal):.2f}")
    print(f"  Max: {np.max(signal):.2f}")
    
    # Test different rainflow methods
    
    # Method 1: Basic count_cycles
    print("\nMethod 1: Basic count_cycles")
    cycles_basic = list(rainflow.count_cycles(signal))
    print(f"Number of cycles: {len(cycles_basic)}")
    print(f"Format of first cycle: {cycles_basic[0]}")
    if len(cycles_basic) >= 5:
        print("First 5 cycles:")
        for i, cycle in enumerate(cycles_basic[:5]):
            print(f"  {i+1}: {cycle}")
    
    # Check if there's a parameter for tracking positions
    try:
        print("\nTrying with rainflow.count_cycles with different parameters")
        print("\nParameter exploration:")
        
        # Try different parameter options
        for param in ['nbins', 'with_residuals', 'use_hysteresis']:
            try:
                kwargs = {param: True}
                cycles_param = list(rainflow.count_cycles(signal, **kwargs))
                print(f"  Using {param}=True: Success, returned {len(cycles_param)} cycles")
                print(f"  First cycle format: {cycles_param[0]}")
            except Exception as e:
                print(f"  Using {param}=True: Error - {e}")
    except Exception as e:
        print(f"Error with parameter exploration: {e}")
    
    # Examine the rainflow features more generally
    print("\nExploring available rainflow module:")
    rainflow_attrs = dir(rainflow)
    print(f"Available attributes: {[attr for attr in rainflow_attrs if not attr.startswith('_')]}")
    
    # Try custom method to calculate means
    print("\nTesting our own implementation that calculates means and tracks positions")
    cycles_with_means = []
    
    # Process each basic cycle and calculate the mean
    for i, (rng, count) in enumerate(cycles_basic):
        # For real data, we would calculate the mean from the original signal
        # But since count_cycles doesn't return position info, we approximate with global mean
        mean_value = np.mean(signal)
        
        # Add cycle with calculated mean
        cycles_with_means.append((rng, mean_value, count, 0, 0))
    
    print(f"Cycles with calculated means: {len(cycles_with_means)}")
    if len(cycles_with_means) >= 5:
        print("First 5 cycles with means:")
        for i, cycle in enumerate(cycles_with_means[:5]):
            print(f"  {i+1}: Range={cycle[0]:.6f}, Mean={cycle[1]:.6f}, Count={cycle[2]}")
    
    # Try to implement our own rainflow algorithm that keeps track of positions
    print("\nImplementing our own rainflow algorithm to track positions")
    # Simplified rainflow implementation that tracks turning points
    turning_points, turning_indices = find_turning_points(signal)
    
    print(f"Found {len(turning_points)} turning points")
    if len(turning_points) >= 10:
        print("First 10 turning points and their indices:")
        for i in range(10):
            print(f"  {i+1}: Value={turning_points[i]:.4f}, Index={turning_indices[i]}")
    
    print("\nTesting complete!")
    return 0

def find_turning_points(signal):
    """
    Find turning points (peaks and valleys) in a signal
    
    Args:
        signal: 1D array of signal values
        
    Returns:
        tuple: (turning_points, turning_indices)
    """
    turning_points = []
    turning_indices = []
    
    if len(signal) < 3:
        return np.array(turning_points), np.array(turning_indices)
    
    # Add first point as a turning point
    turning_points.append(signal[0])
    turning_indices.append(0)
    
    # Find turning points by checking slope changes
    for i in range(1, len(signal)-1):
        # Check if this is a turning point (peak or valley)
        if (signal[i-1] < signal[i] and signal[i] > signal[i+1]) or \
           (signal[i-1] > signal[i] and signal[i] < signal[i+1]):
            turning_points.append(signal[i])
            turning_indices.append(i)
    
    # Add last point as a turning point
    turning_points.append(signal[-1])
    turning_indices.append(len(signal)-1)
    
    return np.array(turning_points), np.array(turning_indices)

if __name__ == "__main__":
    main() 