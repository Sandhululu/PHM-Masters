#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the enhanced cycle details CSV file and calculate damage for each cycle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    """Analyze cycle details and calculate damage"""
    print("\n=== Cycle Details Analysis ===\n")
    
    # Load the enhanced cycle details CSV file
    csv_path = os.path.join(os.getcwd(), "enhanced_cycle_details.csv")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} cycles from {csv_path}")
    
    # Material properties for fatigue analysis
    E_mod, sigma_f_prime, epsilon_f_prime = 400e9, 1000e6, 0.1
    b, c = -0.12, -0.7  # Basquin and Coffin-Manson exponents
    
    # Calculate cycles to failure for each strain range
    ranges = df['Range'].values
    N_f_cycles = []
    
    for strain_range in ranges:
        strain_amp = strain_range / 2
        N_values = np.logspace(1, 10, 1000)
        
        # Calculate strain components using Manson-Coffin relationship
        elastic_strain = (sigma_f_prime/E_mod) * (2*N_values)**b
        plastic_strain = epsilon_f_prime * (2*N_values)**c
        total_strain = elastic_strain + plastic_strain
        
        # Find cycle life
        N_f = N_values[np.argmin(np.abs(total_strain - strain_amp))]
        N_f_cycles.append(min(N_f, 1e6))  # Cap at 1 million cycles
    
    # Add cycles to failure and damage to the dataframe
    df['N_f'] = N_f_cycles
    df['Damage'] = df['Count'] / df['N_f']
    
    # Print statistics
    print("\nCycles to failure statistics:")
    print(df['N_f'].describe())
    
    print("\nDamage statistics:")
    print(df['Damage'].describe())
    
    print(f"\nTotal cumulative damage: {df['Damage'].sum():.6e}")
    
    # Save enhanced DataFrame to CSV
    output_csv = os.path.join(os.getcwd(), "enhanced_cycle_details_with_damage.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved enhanced data to: {output_csv}")
    
    # Create S-N curve plot (log-log)
    plt.figure(figsize=(10, 6))
    plt.loglog(ranges, N_f_cycles, 'o')
    plt.title('Strain Range vs Cycles to Failure')
    plt.xlabel('Strain Range')
    plt.ylabel('Cycles to Failure (N_f)')
    plt.grid(True, which="both", ls="-")
    
    # Add trend line
    try:
        # Log transform for linear regression
        log_ranges = np.log10(ranges)
        log_cycles = np.log10(N_f_cycles)
        
        # Remove any potential infinities or NaNs
        valid_indices = np.isfinite(log_ranges) & np.isfinite(log_cycles)
        if np.sum(valid_indices) > 1:  # Need at least 2 points for regression
            z = np.polyfit(log_ranges[valid_indices], log_cycles[valid_indices], 1)
            p = np.poly1d(z)
            
            # Generate points for trend line
            x_trend = np.logspace(np.log10(min(ranges)), np.log10(max(ranges)), 100)
            y_trend = 10**p(np.log10(x_trend))
            
            # Plot trend line
            plt.loglog(x_trend, y_trend, 'r-', linewidth=2)
            
            # Add equation to plot
            slope = z[0]
            intercept = z[1]
            plt.text(0.05, 0.05, f'log(N_f) = {slope:.2f}·log(Δε) + {intercept:.2f}',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error creating trend line: {e}")
    
    plt.savefig('strain_to_failure_cycles.png')
    print("\nCreated strain-life plot: strain_to_failure_cycles.png")
    
    # Create histogram of damage values
    plt.figure(figsize=(10, 6))
    plt.hist(df['Damage'], bins=20, alpha=0.7, color='green')
    plt.title('Histogram of Damage per Cycle')
    plt.xlabel('Damage')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('damage_histogram.png')
    print("\nCreated damage histogram: damage_histogram.png")
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    main() 