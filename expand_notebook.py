import json

# Load the existing notebook
with open('RUL_Estimation_Workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Add additional cells to the notebook
additional_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load and Process Strain Data\n",
            "\n",
            "Digital Image Correlation (DIC) provides full-field displacement and strain measurements by comparing images of a specimen at different loading stages. Here, we're loading strain data from the DIC measurements of a tungsten component under thermal cycling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load strain data\n",
            "print(\"Loading strain data...\")\n",
            "data = load_all_data()\n",
            "\n",
            "# Extract data components\n",
            "if isinstance(data, dict):\n",
            "    # Use dictionary format\n",
            "    ThermalStrain = data['thermal_strain']\n",
            "    DICExx = data['strain_exx']\n",
            "    DICEyy = data['strain_eyy']\n",
            "    time_points = data['time_points']\n",
            "    high_strain_points = data['high_strain_points']\n",
            "    mean_strain = data['mean_strain']\n",
            "    max_strain = data['max_strain'] \n",
            "    std_strain = data['std_strain']\n",
            "else:\n",
            "    # Use tuple format for backward compatibility\n",
            "    _, DICExx, _, DICEyy, ThermalStrain, time_points, mean_strain, std_strain, max_strain, high_strain_points = data\n",
            "\n",
            "# Print statistical summary\n",
            "print_statistical_summary(ThermalStrain, high_strain_points)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Visualize Initial Strain Data\n",
            "\n",
            "Let's visualize the strain data to understand the distribution and patterns. This will help identify high-strain regions and potential failure points."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot initial strain patterns\n",
            "plot_initial_strain_analysis(DICExx, DICEyy, ThermalStrain, time_points, high_strain_points)\n",
            "\n",
            "print(\"\\nInitial strain analysis complete.\\n\")\n",
            "print(f\"Maximum strain magnitude: {max_strain:.2e}\")\n",
            "print(f\"Mean strain magnitude: {mean_strain:.2e}\")\n",
            "print(f\"Strain standard deviation: {std_strain:.2e}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Calculate and Visualize Stress Components\n",
            "\n",
            "Using Hooke's law, we can convert strain measurements to stress. For tungsten, which has isotropic mechanical properties, we use its Young's modulus and Poisson's ratio for this conversion."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define material properties for tungsten\n",
            "E = 400e3  # Young's modulus in MPa (400 GPa)\n",
            "nu = 0.28  # Poisson's ratio\n",
            "yield_strength = 1000.0  # Approximate yield strength in MPa\n",
            "\n",
            "# Calculate stress components\n",
            "sigma_xx, sigma_yy, von_mises_stress, safety_factor = calculate_stress(DICExx, DICEyy, E, nu, yield_strength)\n",
            "\n",
            "# Plot stress components and safety factor\n",
            "plot_stress_analysis(sigma_xx, sigma_yy, von_mises_stress, safety_factor, high_strain_points)\n",
            "\n",
            "# Print stress analysis summary\n",
            "print(\"\\nStress analysis complete.\\n\")\n",
            "print(f\"Maximum σxx: {np.nanmax(sigma_xx):.2f} MPa\")\n",
            "print(f\"Maximum σyy: {np.nanmax(sigma_yy):.2f} MPa\")\n",
            "print(f\"Maximum von Mises stress: {np.nanmax(von_mises_stress):.2f} MPa\")\n",
            "print(f\"Yield strength of tungsten: ~{yield_strength:.1f} MPa\")\n",
            "print(f\"Safety factor: {np.nanmin(safety_factor):.2f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Calculate and Visualize Principal Strains\n",
            "\n",
            "Principal strains represent the maximum and minimum strains at each point, regardless of direction. They are critical for fatigue analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate principal strains\n",
            "e1, e2, max_shear, theta = calculate_principal_strains(DICExx, DICEyy)\n",
            "\n",
            "# Plot principal strain components\n",
            "plot_strain_analysis(e1, e2, max_shear, theta, high_strain_points)\n",
            "\n",
            "# Print principal strain analysis summary\n",
            "print(\"\\nPrincipal strain analysis complete.\\n\")\n",
            "print(f\"Maximum principal strain (e1): {np.nanmax(e1):.2e}\")\n",
            "print(f\"Minimum principal strain (e2): {np.nanmin(e2):.2e}\")\n",
            "print(f\"Maximum shear strain: {np.nanmax(max_shear):.2e}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Perform Fatigue Analysis with Rainflow Cycle Counting\n",
            "\n",
            "Rainflow cycle counting is a technique used in fatigue analysis to count strain cycles in irregular strain-time histories. This is crucial for accurate fatigue life prediction."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Identify cycles in principal strain signals\n",
            "print(\"Performing rainflow cycle counting...\")\n",
            "cycles_e1 = identify_cycles(e1, high_strain_points)\n",
            "cycles_max_shear = identify_cycles(max_shear, high_strain_points, is_shear_strain=True)\n",
            "\n",
            "# Extract and analyze fatigue results\n",
            "fatigue_results_e1 = analyze_fatigue(e1, high_strain_points, cycles_e1, 'Principal Strain')\n",
            "fatigue_results_max_shear = analyze_fatigue(max_shear, high_strain_points, cycles_max_shear, 'Max Shear Strain')\n",
            "\n",
            "# Plot fatigue analysis signals\n",
            "plot_fatigue_analysis_signals(fatigue_results_e1, fatigue_results_max_shear)\n",
            "\n",
            "# Print fatigue analysis summary\n",
            "print(\"\\nFatigue analysis complete.\\n\")\n",
            "print(f\"Principal strain cycles identified: {len(cycles_e1)}\")\n",
            "print(f\"Shear strain cycles identified: {len(cycles_max_shear)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Estimate and Visualize Remaining Useful Life (RUL)\n",
            "\n",
            "Using the identified strain cycles, we can estimate the remaining useful life (RUL) of the component, based on the Manson-Coffin relationship and Miner's rule of cumulative damage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Estimate RUL for principal strain\n",
            "print(\"Estimating remaining useful life...\")\n",
            "x_rul_e1, y_rul_e1, initial_rul_e1, final_rul_e1, life_used_percentage_e1 = estimate_fatigue_life(\n",
            "    fatigue_results_e1, cycle_multiplier=10)\n",
            "\n",
            "# Estimate RUL for max shear strain\n",
            "x_rul_max_shear, y_rul_max_shear, initial_rul_max_shear, final_rul_max_shear, life_used_percentage_max_shear = estimate_fatigue_life(\n",
            "    fatigue_results_max_shear, cycle_multiplier=10, force_shear=True)\n",
            "\n",
            "# Plot RUL estimations\n",
            "plot_rul_estimation(x_rul_e1, y_rul_e1, x_rul_max_shear, y_rul_max_shear)\n",
            "\n",
            "# Print RUL estimation summary\n",
            "print(\"\\nRUL estimation complete.\\n\")\n",
            "print(f\"Principal Strain - Initial RUL: {initial_rul_e1:.1f} cycles, Final RUL: {final_rul_e1:.1f} cycles\")\n",
            "print(f\"Principal Strain - Life used: {life_used_percentage_e1:.2f}%\")\n",
            "print(f\"Max Shear Strain - Initial RUL: {initial_rul_max_shear:.1f} cycles, Final RUL: {final_rul_max_shear:.1f} cycles\")\n",
            "print(f\"Max Shear Strain - Life used: {life_used_percentage_max_shear:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Conclusion and Recommendations\n",
            "\n",
            "Based on the fatigue analysis and RUL estimation, we can draw the following conclusions:\n",
            "\n",
            "1. The tungsten component is undergoing significant strain cycling, particularly in regions of high strain concentration.\n",
            "2. The principal strain analysis indicates a higher rate of life consumption compared to the shear strain analysis.\n",
            "3. Based on our RUL calculations, the component has used approximately:\n",
            "   - For principal strain: The displayed percentage of its fatigue life\n",
            "   - For shear strain: The displayed percentage of its fatigue life\n",
            "\n",
            "**Recommendations:**\n",
            "\n",
            "1. If the life used percentage exceeds 50% for either strain type, consider scheduling maintenance or replacement.\n",
            "2. Continue monitoring the component for changes in strain patterns that might accelerate fatigue damage.\n",
            "3. Consider design modifications to reduce strain concentrations in critical areas.\n",
            "4. For more accurate RUL estimation, collect additional cycles of data and refine material parameters.\n",
            "\n",
            "This analysis demonstrates the power of combining DIC strain measurements with fatigue analysis techniques to predict component life and inform maintenance decisions."
        ]
    }
]

# Append the additional cells to the notebook
notebook['cells'].extend(additional_cells)

# Update metadata to include language_info
notebook['metadata']['language_info'] = {
    "codemirror_mode": {
        "name": "ipython",
        "version": 3
    },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.8.5"
}

# Write the updated notebook to a file
with open('RUL_Estimation_Workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Jupyter notebook expanded successfully with all sections!") 