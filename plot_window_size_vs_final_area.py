import numpy as np
import matplotlib.pyplot as plt
import sys


# Configuration
simulation_number = int(sys.argv[1])                                     # Simulation number
y_variable = 'final_window_area'

# 1. Load your result file
filename = f"results/simulation_{simulation_number}_window_size_vs_final_area.txt"
param_values = []
final_window_areas = []

with open(filename, 'r') as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        param_values.append(float(parts[0]))
        final_window_areas.append(float(parts[1]))

param_values = np.array(param_values)
final_window_areas = np.array(final_window_areas)

# 2. Choose the Y values and degree of polynomial
fit_degree = 2

# 3. Fit polynomial
coeffs = np.polyfit(param_values, final_window_areas, deg=fit_degree)
poly = np.poly1d(coeffs)
formula = " + ".join([f"{coeff:.2f}x^{len(coeffs)-i-1}" for i, coeff in enumerate(coeffs)])

# 4. Generate smooth curve for plotting
x_fit = np.linspace(min(param_values), max(param_values), 200)
y_fit = poly(x_fit)


# 5. Plotting
plt.figure(figsize=(6, 8))
plt.scatter(param_values, final_window_areas, label='Data Points', color='blue', s=10)
plt.plot(x_fit, y_fit, '-', label=f'{fit_degree}° Fit: {formula}', color='orange')
plt.xlabel('Window Size [# cells]')
plt.ylabel('Final Window Area [%]')
plt.title(f'Simulation {simulation_number}: Window Size vs Final Area')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/simulation_{simulation_number}_window_size_vs_final_area.png")
plt.show()
