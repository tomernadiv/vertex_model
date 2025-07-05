import numpy as np
import matplotlib.pyplot as plt
import sys


# Configuration
simulation_number = int(sys.argv[1])                                     # Simulation number
y_variable = sys.argv[2] if len(sys.argv) > 2 else "max_velocity_at_10"  # Choose variable to plot
target_y = 1.5 if y_variable == "max_velocity_at_10" else 150                          

# 1. Load your result file
filename = f"results/simulation_{simulation_number}_final_results.txt"
param_values = []
final_window_areas = []
max_velocities = []
final_total_areas = []

with open(filename, 'r') as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        param_values.append(float(parts[0]))
        final_window_areas.append(float(parts[1]))
        max_velocities.append(float(parts[2]))
        final_total_areas.append(float(parts[3]))

param_values = np.array(param_values)
final_window_areas = np.array(final_window_areas)
max_velocities = np.array(max_velocities)
final_total_areas = np.array(final_total_areas)
vertex_overlap_error = np.abs(final_total_areas - 100) > 5

# 2. Choose the Y values and degree of polynomial
if y_variable == "final_window_area":
    y_values = final_window_areas
    fit_degree = 2
elif y_variable == "max_velocity_at_10":
    y_values = max_velocities
    fit_degree = 1
else:
    raise ValueError(f"Unknown y_variable: {y_variable}")

# 3. Fit polynomial
coeffs = np.polyfit(param_values, y_values, deg=fit_degree)
poly = np.poly1d(coeffs)

# 4. Generate smooth curve for plotting
x_fit = np.linspace(min(param_values), max(param_values), 200)
y_fit = poly(x_fit)

# 6. Solve for x where y = target_y
adjusted_coeffs = coeffs.copy()
adjusted_coeffs[-1] -= target_y
roots = np.roots(adjusted_coeffs)
real_roots = [r.real for r in roots if np.isreal(r) and min(param_values) <= r.real <= max(param_values)]

# 5. Plotting
plt.figure(figsize=(6, 8))
plt.scatter(param_values, y_values, label='Data Points', color='blue', s=10)
plt.plot(x_fit, y_fit, '-', label=f'{fit_degree}° Fit', color='orange')
plt.axhline(y=target_y, color='r', linestyle='--', label=f'y = {target_y}')
plt.xlabel('Parameter Value')
plt.ylabel(y_variable.replace("_", " ").title())
formatted_roots = ", ".join(f"{root:.2f}" for root in real_roots)
plt.title(f'Simulation {simulation_number}: {y_variable.replace("_", " ").title()} vs Parameter [found {formatted_roots}]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/simulation_{simulation_number}_parameter_vs_{y_variable}.png")
plt.show()

print(f"x value(s) where {y_variable} = {target_y}:", real_roots)
print("final window area:", final_window_areas[-1])
