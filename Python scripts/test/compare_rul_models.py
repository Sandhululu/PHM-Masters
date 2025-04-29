#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RUL Model Comparison

Compare linear and feedback damage models to understand how damage acceleration 
affects the RUL curve shape.
"""

import numpy as np
import matplotlib.pyplot as plt

def compare_rul_models(initial_rul=10000, feedback_factors=[1.0, 1.5, 3.0, 5.0]):
    """
    Compare linear and feedback damage models for RUL estimation
    
    Args:
        initial_rul: Initial RUL value to use for both models
        feedback_factors: List of feedback factors to compare
    """
    # Setup figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplot grid
    ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)  # Damage vs Cycles
    ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)  # RUL vs Cycles
    ax3 = plt.subplot2grid((2, 2), (1, 0), fig=fig)  # Damage Rate vs Cycles
    ax4 = plt.subplot2grid((2, 2), (1, 1), fig=fig)  # RUL % Remaining vs Cycles
    
    # Calculate base damage rate
    base_damage_rate = 1 / initial_rul
    
    # Create consistent cycle points for both models
    max_cycles = initial_rul * 1.5  # Go beyond failure for visualization
    num_points = 1000
    cycle_points = np.linspace(0, max_cycles, num_points)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot linear model first (feedback_factor = 0)
    # Linear damage calculation
    linear_damage = cycle_points * base_damage_rate
    linear_rul = np.maximum(initial_rul - cycle_points, 0)
    linear_damage_rate = np.ones_like(cycle_points) * base_damage_rate
    linear_rul_percent = np.maximum(linear_rul / initial_rul * 100, 0)
    
    # Add linear model to plots
    ax1.plot(cycle_points, linear_damage, '-', color='black', linewidth=2.5, 
             label='Linear Model (No Feedback)')
    ax2.plot(cycle_points, linear_rul, '-', color='black', linewidth=2.5,
             label='Linear Model (No Feedback)')
    ax3.plot(cycle_points, linear_damage_rate, '-', color='black', linewidth=2.5,
             label='Linear Model (No Feedback)')
    ax4.plot(cycle_points, linear_rul_percent, '-', color='black', linewidth=2.5,
             label='Linear Model (No Feedback)')
    
    # Add failure threshold line to damage plot
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Failure Threshold (D=1)")
    
    # For each feedback factor, calculate and plot feedback model
    for i, feedback_factor in enumerate(feedback_factors):
        # Initialize arrays
        damage = np.zeros_like(cycle_points)
        rul = np.zeros_like(cycle_points)
        damage_rate = np.zeros_like(cycle_points)
        
        # Set initial values
        rul[0] = initial_rul
        current_damage_rate = base_damage_rate
        
        # Calculate damage and RUL for each cycle point with feedback
        for j in range(1, len(cycle_points)):
            # Calculate cycle increment
            cycle_increment = cycle_points[j] - cycle_points[j-1]
            
            # Current damage rate increases with cumulative damage
            damage_rate[j-1] = current_damage_rate  # Store for plotting
            
            # Calculate damage increment for this step
            damage_increment = current_damage_rate * cycle_increment
            
            # Update cumulative damage
            damage[j] = damage[j-1] + damage_increment
            
            # Update damage rate for next step
            current_damage_rate = base_damage_rate * (1 + feedback_factor * damage[j])
            
            # Calculate RUL based on damage rate and remaining capacity
            remaining_capacity = 1 - damage[j]
            
            # If we've reached failure, set RUL to zero
            if remaining_capacity <= 0:
                rul[j:] = 0
                damage_rate[j:] = current_damage_rate
                damage[j:] = 1.0  # Cap at 1.0
                break
                
            # Calculate RUL with current damage rate
            rul[j] = remaining_capacity / current_damage_rate
        
        # Store the final damage rate for the last point
        damage_rate[-1] = current_damage_rate
        
        # Calculate RUL percentage
        rul_percent = rul / initial_rul * 100
        
        # Determine projected failure point (where damage reaches 1.0)
        failure_idx = np.argmax(damage >= 1.0) if np.any(damage >= 1.0) else -1
        if failure_idx > 0:
            failure_point = cycle_points[failure_idx]
            # Mark failure point on plots
            label = f"Feedback {feedback_factor} (Failure: {failure_point:.0f} cycles)"
        else:
            label = f"Feedback {feedback_factor}"
        
        # Add to plots
        color = colors[i % len(colors)]
        ax1.plot(cycle_points, damage, '-', color=color, linewidth=2, label=label)
        ax2.plot(cycle_points, rul, '-', color=color, linewidth=2, label=label)
        ax3.plot(cycle_points, damage_rate, '-', color=color, linewidth=2, label=label)
        ax4.plot(cycle_points, rul_percent, '-', color=color, linewidth=2, label=label)
        
        # Mark failure points if available
        if failure_idx > 0:
            ax1.scatter([failure_point], [1.0], color=color, s=100, zorder=10)
            ax2.scatter([failure_point], [0], color=color, s=100, zorder=10)
            ax4.scatter([failure_point], [0], color=color, s=100, zorder=10)
    
    # Configure plot 1: Damage vs Cycles
    ax1.set_xlabel('Cycles', fontsize=12)
    ax1.set_ylabel('Cumulative Damage', fontsize=12)
    ax1.set_title('Damage Accumulation', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.2)
    ax1.legend(fontsize=9, loc='upper left')
    
    # Configure plot 2: RUL vs Cycles
    ax2.set_xlabel('Cycles', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax2.set_title('Remaining Useful Life', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, initial_rul * 1.05)
    ax2.legend(fontsize=9, loc='upper right')
    
    # Configure plot 3: Damage Rate vs Cycles
    ax3.set_xlabel('Cycles', fontsize=12)
    ax3.set_ylabel('Damage Rate (per cycle)', fontsize=12)
    ax3.set_title('Damage Rate Acceleration', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    # Set y-axis limit to show up to 5x the base damage rate
    max_rate = base_damage_rate * (1 + max(feedback_factors) * 1.2)
    ax3.set_ylim(0, max_rate)
    ax3.legend(fontsize=9, loc='upper left')
    
    # Format damage rate values on y-axis
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Configure plot 4: RUL % Remaining vs Cycles
    ax4.set_xlabel('Cycles', fontsize=12)
    ax4.set_ylabel('RUL Remaining (%)', fontsize=12)
    ax4.set_title('Percentage of Remaining Useful Life', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=9, loc='upper right')
    
    # Overall title
    plt.suptitle('Comparison of Linear vs. Feedback Damage Models', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig('rul_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('rul_model_comparison.svg', bbox_inches='tight', dpi=300)
    
    print("RUL model comparison visualization created and saved.")

if __name__ == "__main__":
    print("\n=== RUL Model Comparison ===\n")
    compare_rul_models(initial_rul=10000, feedback_factors=[1.0, 1.5, 3.0, 5.0])
    print("\nComparison complete.") 